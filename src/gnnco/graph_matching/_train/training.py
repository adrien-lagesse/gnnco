import pathlib
import statistics
from typing import Literal, NamedTuple
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import torch
import torch.utils.data
from safetensors.torch import save_model
from scipy.optimize import linear_sum_assignment

from ..._core import BatchedSignals
from .utils import (
    GMDatasetBatch,
    get_kwargs,
    model_factory,
    optimizer_factory,
    setup_data,
)


def forward_pass(model, batch):
    pass


def siamese_similarity(
    model: torch.nn.Module, batch: GMDatasetBatch
) -> torch.FloatTensor:
    embeddings_base: BatchedSignals = model.forward(
        batch.base_signals, batch.base_graphs
    )

    embeddings_corrupted: BatchedSignals = model.forward(
        batch.corrupted_signals, batch.base_graphs
    )

    padded_base = torch.zeros(
        (len(batch.padded_batch), embeddings_base.dim()),
        device=embeddings_base.device(),
        requires_grad=True
    )

    padded_base = padded_base[batch.padded_batch >= 0].copy_(embeddings_base.x())

    padded_corrupted = torch.zeros(
        (len(batch.padded_batch), embeddings_corrupted.dim()),
        device=embeddings_corrupted.device(),
        requires_grad=True
    )
    padded_corrupted = padded_corrupted[batch.padded_batch >= 0].copy_(embeddings_corrupted.x())

    alignement_similarities = torch.bmm(
        padded_base.reshape((len(batch), -1, embeddings_base.dim())),
        padded_corrupted.reshape(
            (len(batch), -1, embeddings_corrupted.dim())
        ).transpose(1, 2),
    )

    return alignement_similarities


@torch.vmap
def batched_loss(
    similarity_matrix: torch.FloatTensor, mask: torch.BoolTensor
) -> torch.FloatTensor:
    similarity_matrix.masked_fill_(torch.logical_not(mask), -float('inf'))
    diag_logits = torch.diag(torch.softmax(similarity_matrix, dim=1))
    diag_logits.masked_fill_(torch.logical_not(mask), 1)
    loss = -torch.log(diag_logits + 1e-7).mean()
    return loss


def compute_losses(
    similarity_matrices: torch.FloatTensor, masks: torch.BoolTensor
) -> torch.FloatTensor:
    return batched_loss(
        similarity_matrices,
        masks,
    )


class AccuraciesResults(NamedTuple):
    top1: torch.FloatTensor
    top3: torch.FloatTensor
    top5: torch.FloatTensor


def __top_k_accuracy(
    alignement_similarity: torch.FloatTensor, mask: torch.BoolTensor, top_n: int
) -> torch.FloatTensor:
    _, indices = torch.sort(
        torch.masked_fill(
            alignement_similarity, torch.logical_not(mask), -float("inf")
        ),
        descending=True,
    )
    mask = mask.float()
    m = (
        torch.isin(torch.arange(len(alignement_similarity), device=alignement_similarity.device), indices[:, :top_n])
        .float()
        .squeeze()
    )
    acc = (m * mask).sum() / (mask.sum())
    return acc


@torch.no_grad
def compute_accuracies(
    alignement_similarities: torch.FloatTensor, masks: torch.BoolTensor
) -> AccuraciesResults:
    batched_top_k_accuracy = torch.vmap(__top_k_accuracy, in_dims=(0, 0, None))
    return AccuraciesResults(
        top1=batched_top_k_accuracy(alignement_similarities, masks, 1),
        top3=batched_top_k_accuracy(alignement_similarities, masks, 3),
        top5=batched_top_k_accuracy(alignement_similarities, masks, 5),
    )


class LAPResults(NamedTuple):
    permutations: list[torch.LongTensor]
    lap: list[float]


@torch.no_grad
def compute_lap(
    alignement_similarities: torch.FloatTensor, masks: torch.BoolTensor
) -> LAPResults:
    permuations = []
    lap = []
    for similarity_matrix, mask in zip(alignement_similarities, masks):
        similarity_matrix = similarity_matrix[mask].cpu().numpy()
        idx, permutation_pred = linear_sum_assignment(similarity_matrix, maximize=True)
        permuations.append(
            torch.tensor(
                permutation_pred,
                dtype=torch.long,
                device=alignement_similarities.device,
            )
        )
        lap.append(float((idx == permutation_pred).astype(float).mean()))

    return LAPResults(permutations=permuations, lap=lap)


@torch.no_grad
def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_l: dict[str, list[float]] = {
        "loss": [],
        "lap": [],
        "top_1": [],
        "top_3": [],
        "top_5": [],
    }

    batch: GMDatasetBatch
    for batch in loader:
        batch = batch.to(device)

        similarity_matrices = siamese_similarity(model, batch)
        masks = (batch.padded_batch >= 0).reshape((len(batch), -1))

        losses = compute_losses(similarity_matrices, masks)
        metrics_l["loss"].append(float(losses.mean()))

        (top_1, top_3, top_5) = compute_accuracies(similarity_matrices, masks)
        metrics_l["top_1"].append(float(top_1.mean()))
        metrics_l["top_3"].append(float(top_3.mean()))
        metrics_l["top_5"].append(float(top_5.mean()))

        (_permutations, lap) = compute_lap(similarity_matrices, masks)

        metrics_l["lap"].append(statistics.mean(lap))

    return {k: statistics.mean(v) for (k, v) in metrics_l.items()}


def train(
    *,
    custom_model: torch.nn.Module | None = None,
    dataset: pathlib.Path,
    experiment: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    cuda: bool = True,
    log_frequency: int = 25,
    profile: bool = False,
    model: Literal["GCN", "GIN", "GAT", "GatedGCN", "GATv2"] | None,
    layers: int | None,
    heads: int | None,
    features: int | None,
    out_features: int | None,
    optimizer: Literal["adam", "adam-one-cycle"] = "adam-one-cycle",
    lr: float | None = 5e-4,
    max_lr: float | None = 1e-3,
    start_factor: int | None = 5,
    end_factor: int | None = 500,
    grad_clip: float = 1e-1,
):
    device = torch.device("cuda") if cuda else torch.device("cpu")

    mlflow.set_experiment(experiment_name=experiment)

    with mlflow.start_run(run_name=run_name, log_system_metrics=profile) as run:
        mlflow.log_params(get_kwargs())

        # Load the training and validation datasets and build suitable loaders to batch the graphs together.
        (train_dataset, val_dataset, train_loader, val_loader) = setup_data(
            dataset_path=dataset,
            batch_size=batch_size,
        )

        # Setting up the GNN model and loading it onto the gpu if needed
        gnn_model: torch.nn.Module
        if custom_model is not None:
            gnn_model = custom_model
        else:
            gnn_model = model_factory(
                model=model,
                layers=layers,
                heads=heads,
                features=features,
                out_features=out_features,
            )
        gnn_model = gnn_model.to(device)

        # Computing the number of parameters in the GNN
        mlflow.log_param(
            "nb_params", sum([np.prod(p.size()) for p in gnn_model.parameters()])
        )

        # Build the optimizer and scheduler
        gnn_optimizer, gnn_scheduler = optimizer_factory(
            gnn_model,
            optimizer=optimizer,
            epochs=epochs,
            lr=lr,
            max_lr=max_lr,
            start_factor=start_factor,
            end_factor=end_factor,
        )

        for epoch in range(epochs):
            mlflow.log_metric("learning_rate", gnn_scheduler.get_last_lr()[0], epoch)

            # Training loop
            gnn_model.train()
            batch: GMDatasetBatch
            for batch in train_loader:
                batch = batch.to(device)

                gnn_model.zero_grad()

                similarity_matrices = siamese_similarity(gnn_model, batch)
                masks = (batch.padded_batch >= 0).reshape((len(batch), -1))

                losses = compute_losses(similarity_matrices, masks)
                loss = losses.mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(gnn_model.parameters(), grad_clip)
                gnn_optimizer.step()
            gnn_scheduler.step()

            # Metrics Logging
            if epoch % log_frequency == 0:
                gnn_model.eval()
                train_metrics = {
                    f"{k}/train": v
                    for (k, v) in compute_metrics(
                        gnn_model, train_loader, device
                    ).items()
                }
                val_metrics = {
                    f"{k}/val": v
                    for (k, v) in compute_metrics(gnn_model, val_loader, device).items()
                }
                mlflow.log_metrics(train_metrics, epoch)
                mlflow.log_metrics(val_metrics, epoch)

        checkpoint_path = unquote(
            urlparse(run.info.artifact_uri + "/checkpoint.safetensors").path
        )
        save_model(gnn_model, checkpoint_path)
