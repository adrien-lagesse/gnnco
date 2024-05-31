import inspect
import pathlib
import statistics
from functools import partial
from typing import Literal
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import torch
import torch.utils.data
from safetensors.torch import save_model
from scipy.optimize import linear_sum_assignment

from gnnco._core import BatchedSignals, BatchedSparseGraphs, SparseGraph
from gnnco.dataset import GMDataset
from gnnco.models import GAT, GCN, GIN, GatedGCN, GATv2


def __get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


def setup_data(
    *,
    dataset_path: pathlib.Path,
    batch_size: int,
    pin_memory: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    single_base_graph: bool = False
) -> tuple[
    GMDataset, GMDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    train_dataset = GMDataset(root=dataset_path, single_base_graph=single_base_graph)
    val_dataset = GMDataset(root=dataset_path, validation=True, single_base_graph=single_base_graph)

    def collate_fn(
        batch_l: list[
            tuple[
                SparseGraph,
                SparseGraph,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
            ]
        ],
    ) -> tuple[
        BatchedSparseGraphs,
        BatchedSparseGraphs,
        BatchedSignals,
        BatchedSignals,
        torch.FloatTensor,
    ]:
        base_batch = BatchedSparseGraphs.from_graphs(list(map(lambda t: t[0], batch_l)))
        corrupted_batch = BatchedSparseGraphs.from_graphs(
            list(map(lambda t: t[1], batch_l))
        )
        base_signal_batch = BatchedSignals.from_signals(
            list(map(lambda t: t[2], batch_l))
        )
        corrupted_signal_batch = BatchedSignals.from_signals(
            list(map(lambda t: t[3], batch_l))
        )
        qap_values_batch = torch.stack(list(map(lambda t: t[4], batch_l)))

        return (
            base_batch,
            corrupted_batch,
            base_signal_batch,
            corrupted_signal_batch,
            qap_values_batch,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def _model_factory(
    *,
    model: Literal["GCN", "GIN", "GAT", "GatedGCN", "GATv2"],
    layers: int | None,
    heads: int | None,
    features: int | None,
    out_features: int | None,
):
    if model == "GCN":
        return GCN(layers, features, out_features)
    elif model == "GIN":
        return GIN(layers, features, out_features)
    elif model == "GAT":
        return GAT(layers, heads, features, out_features)
    elif model == "GatedGCN":
        return GatedGCN(layers, features, out_features)
    elif model == "GATv2":
        return GATv2(layers, heads, features, out_features)
    else:
        raise RuntimeError(f"Model name '{model}' does not exists")


def _onecycle(
    epoch: int,
    epochs: int,
    max_lr: float,
    start_factor: int,
    end_factor: int,
) -> float:
    """
    One-Cycle Schedule
    """
    if epoch <= epochs * 0.2:
        return max_lr / start_factor + epoch / (0.2 * epochs) * (
            max_lr - max_lr / start_factor
        )
    else:
        return max_lr + (epoch - 0.2 * epochs) / (0.8 * epochs) * (
            max_lr / end_factor - max_lr
        )


def _optimizer_factory(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    optimizer: Literal["adam", "adam-one-cycle"],
    epochs: int | None,
    lr: float | None,
    max_lr: float | None,
    start_factor: int | None,
    end_factor: int | None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Create the optimizer and scheduler
    """
    mod_list = torch.nn.ModuleList([model, mlp])
    if optimizer == "adam":
        optimizer = torch.optim.Adam(mod_list.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr)
        return optimizer, scheduler
    elif optimizer == "adam-one-cycle":
        optimizer = torch.optim.AdamW(mod_list.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(
                _onecycle,
                epochs=epochs,
                max_lr=max_lr,
                start_factor=start_factor,
                end_factor=end_factor,
            ),
        )
        return optimizer, scheduler
    else:
        raise RuntimeError(f"Optimizer not found: {optimizer}")


def siamese_similarity(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    *,
    graph_base: BatchedSparseGraphs,
    signal_base: BatchedSignals,
    graph_corrupted: BatchedSparseGraphs,
    signal_corrupted: BatchedSignals,
) -> torch.FloatTensor:
    embeddings_base: BatchedSignals = model.forward(signal_base, graph_base)
    embeddings_corrupted: BatchedSignals = model.forward(
        signal_corrupted, graph_corrupted
    )

    stacked_base = embeddings_base.force_stacking()
    stacked_corrupted = embeddings_corrupted.force_stacking()

    alignement_similarities = torch.bmm(stacked_base, stacked_corrupted.transpose(1, 2))

    embeddings_base: BatchedSignals = mlp.forward(embeddings_base)
    embeddings_corrupted: BatchedSignals = mlp.forward(embeddings_corrupted)

    stacked_base = embeddings_base.force_stacking()
    stacked_corrupted = embeddings_corrupted.force_stacking()

    frobenius_similarities = torch.bmm(stacked_base, stacked_corrupted.transpose(1, 2))

    return alignement_similarities, frobenius_similarities


def compute_losses(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    *,
    graph_base: BatchedSparseGraphs,
    signal_base: BatchedSignals,
    graph_corrupted: BatchedSparseGraphs,
    signal_corrupted: BatchedSignals,
    qap_values: torch.FloatTensor,
    beta: float = 1,
) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
    alignement_similarities, frobenius_similarities = siamese_similarity(
        model,
        mlp,
        graph_base=graph_base,
        graph_corrupted=graph_corrupted,
        signal_base=signal_base,
        signal_corrupted=signal_corrupted,
    )

    logits = torch.softmax(alignement_similarities.flatten(end_dim=-2), dim=1).reshape(
        alignement_similarities.shape
    )

    alignement_loss = -torch.log(
        torch.diagonal(logits, dim1=1, dim2=2).mean(dim=1) + 1e-7
    )

    # ALPHA = 1

    # alignement_loss = torch.relu(
    #     alignement_similarities
    #     - torch.diagonal(alignement_similarities - ALPHA, dim1=-2, dim2=-1).unsqueeze(
    #         -1
    #     )
    #     + ALPHA
    # ).mean(dim=-1)
    # alignement_loss = alignement_loss.mean(dim=1)

    frobenius_loss = (
        (torch.diagonal(frobenius_similarities, dim1=1, dim2=2) - qap_values)
        / (qap_values.mean() + 1)
    ) ** 2
    frobenius_loss = frobenius_loss.mean(dim=1)

    loss = (1 - beta) * alignement_loss + beta * frobenius_loss

    return loss, (alignement_loss, frobenius_loss)


def compute_accuracies(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    *,
    graph_base: BatchedSparseGraphs,
    signal_base: BatchedSignals,
    graph_corrupted: BatchedSparseGraphs,
    signal_corrupted: BatchedSignals,
) -> tuple[list[torch.LongTensor], list[float], list[float], list[float]]:
    alignement_similarities, _ = siamese_similarity(
        model,
        mlp,
        graph_base=graph_base,
        graph_corrupted=graph_corrupted,
        signal_base=signal_base,
        signal_corrupted=signal_corrupted,
    )

    accuracies = []
    permuations = []
    top3_accuracies = []
    top5_accuracies = []
    for i in range(len(alignement_similarities)):
        similarity = alignement_similarities[i].detach().cpu().numpy()
        idx, permutation_pred = linear_sum_assignment(similarity, maximize=True)
        accuracies.append(float((idx == permutation_pred).astype(float).mean()))
        permuations.append(
            torch.tensor(permutation_pred, dtype=torch.long, device=graph_base.device())
        )
        _, indices = torch.sort(alignement_similarities[i], descending=True)
        top3_indices = indices[:, :3].detach().cpu()
        top3_accuracies.append(
            float(
                torch.isin(torch.arange(len(alignement_similarities[i])), top3_indices)
                .float()
                .mean()
            )
        )
        top5_indices = indices[:, :5].detach().cpu()
        top5_accuracies.append(
            float(
                torch.isin(torch.arange(len(alignement_similarities[i])), top5_indices)
                .float()
                .mean()
            )
        )

    return permuations, accuracies, top3_accuracies, top5_accuracies


def compute_distance_losses(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    *,
    graph_base: BatchedSparseGraphs,
    signal_base: BatchedSignals,
    graph_corrupted: BatchedSparseGraphs,
    signal_corrupted: BatchedSignals,
    permutations: list[torch.LongTensor],
    qap_values: torch.FloatTensor,
) -> torch.FloatTensor:
    embeddings_base: BatchedSignals = model.forward(signal_base, graph_base)
    embeddings_corrupted: BatchedSignals = model.forward(
        signal_corrupted, graph_corrupted
    )

    embeddings_base = mlp.forward(embeddings_base)
    embeddings_corrupted = mlp.forward(embeddings_corrupted)

    embeddings_base = embeddings_base.force_stacking()
    embeddings_corrupted = embeddings_corrupted.force_stacking()

    for i, p in enumerate(permutations):
        embeddings_corrupted[i] = embeddings_corrupted[i, p, :]

    frobenius_similarities = torch.bmm(
        embeddings_base, embeddings_corrupted.transpose(1, 2)
    )

    distance_loss = (
        (torch.diagonal(frobenius_similarities, dim1=1, dim2=2) - qap_values)
        / (qap_values.mean() + 1)
    ) ** 2
    distance_loss = distance_loss.mean(dim=1)

    return distance_loss


def compute_metrics(
    model: torch.nn.Module,
    mlp: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    beta: float,
    qap_mean_predictor: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics_l: dict[str, list[float]] = {
        "loss": [],
        "alignement_loss": [],
        "frobenius_loss": [],
        "frobenius_rscore": [],
        "distance_loss": [],
        "distance_rscore": [],
        "accuracy": [],
        "top3_accuracy": [],
        "top5_accuracy": [],
    }
    base: BatchedSparseGraphs
    corrupted: BatchedSparseGraphs
    base_signal: BatchedSignals
    corrupted_signal: BatchedSignals
    qap_values: torch.FloatTensor
    for (
        base,
        corrupted,
        base_signal,
        corrupted_signal,
        qap_values,
    ) in loader:
        base = base.to(device)
        corrupted = corrupted.to(device)
        base_signal = base_signal.to(device)
        corrupted_signal = corrupted_signal.to(device)
        qap_values = qap_values.to(device)

        permutations, accuracies, top3_accuracies, top5_accuracies = compute_accuracies(
            model,
            mlp,
            graph_base=base,
            graph_corrupted=corrupted,
            signal_base=base_signal,
            signal_corrupted=corrupted_signal,
        )

        metrics_l["accuracy"].append(statistics.mean(accuracies))
        metrics_l["top3_accuracy"].append(statistics.mean(top3_accuracies))
        metrics_l["top5_accuracy"].append(statistics.mean(top5_accuracies))

        losses, (alignement_losses, frobenius_losses) = compute_losses(
            model,
            mlp,
            graph_base=base,
            graph_corrupted=corrupted,
            signal_base=base_signal,
            signal_corrupted=corrupted_signal,
            qap_values=qap_values,
            beta=beta,
        )

        metrics_l["loss"].append(float(losses.mean()))
        metrics_l["alignement_loss"].append(float(alignement_losses.mean()))
        metrics_l["frobenius_loss"].append(float(frobenius_losses.mean()))
        frobenius_loss_of_mean_predictor = (
            (qap_mean_predictor - qap_values) / (qap_values.mean() + 1)
        ) ** 2
        frobenius_loss_of_mean_predictor = float(
            frobenius_loss_of_mean_predictor.mean()
        )

        metrics_l["frobenius_rscore"].append(
            float(frobenius_losses.mean()) / frobenius_loss_of_mean_predictor
        )

        distance_losses = compute_distance_losses(
            model,
            mlp,
            graph_base=base,
            graph_corrupted=corrupted,
            signal_base=base_signal,
            signal_corrupted=corrupted_signal,
            permutations=permutations,
            qap_values=qap_values,
        )
        metrics_l["distance_loss"].append(float(distance_losses.mean()))
        frobenius_loss_of_mean_predictor = (
            (qap_mean_predictor - qap_values) / (qap_values.mean() + 1)
        ) ** 2
        frobenius_loss_of_mean_predictor = float(
            frobenius_loss_of_mean_predictor.mean()
        )

        metrics_l["distance_rscore"].append(
            float(distance_losses.mean()) / frobenius_loss_of_mean_predictor
        )

    return {k: statistics.mean(v) for (k, v) in metrics_l.items()}


class FrobeniusMLP(torch.nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(features, features),
            torch.nn.ReLU(),
            torch.nn.Linear(features, features),
            torch.nn.ReLU(),
            torch.nn.Linear(features, features),
        )

    def forward(self, batched_signals: BatchedSignals) -> BatchedSignals:
        return BatchedSignals(
            self.layers.forward(batched_signals.x()), batched_signals._batch
        )


def train(
    *,
    custom_model: torch.nn.Module | None = None,
    dataset: pathlib.Path,
    experiment: str,
    run_name: str,
    beta: float,
    epochs: int,
    batch_size: int,
    cuda: bool = True,
    single_base_graph: bool = False,
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
        mlflow.log_params(__get_kwargs())

        # Load the training and validation datasets and build suitable loaders to batch the graphs together.
        (train_dataset, val_dataset, train_loader, val_loader) = setup_data(
            dataset_path=dataset, batch_size=batch_size, single_base_graph=single_base_graph
        )

        # Setting up the GNN model and loading it onto the gpu if needed
        gnn_model: torch.nn.Module
        if custom_model is not None:
            gnn_model = custom_model
        else:
            gnn_model = _model_factory(
                model=model,
                layers=layers,
                heads=heads,
                features=features,
                out_features=out_features,
            )
        gnn_model = gnn_model.to(device)

        # Setting up the MLP model and loading it onto the gpu if needed
        mlp = FrobeniusMLP(out_features)
        mlp = mlp.to(device)

        # Computing the number of parameters in the GNN
        mlflow.log_param(
            "nb_params", sum([np.prod(p.size()) for p in gnn_model.parameters()])
        )

        # Build the optimizer and scheduler
        gnn_optimizer, gnn_scheduler = _optimizer_factory(
            gnn_model,
            mlp,
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

            base: BatchedSparseGraphs
            corrupted: BatchedSparseGraphs
            base_signal: BatchedSignals
            corrupted_signal: BatchedSignals
            qap_values: torch.FloatTensor
            for (
                base,
                corrupted,
                base_signal,
                corrupted_signal,
                qap_values,
            ) in train_loader:
                base = base.to(device)
                corrupted = corrupted.to(device)
                base_signal = base_signal.to(device)
                corrupted_signal = corrupted_signal.to(device)
                qap_values = qap_values.to(device)

                gnn_model.zero_grad()
                mlp.zero_grad()

                losses, (_alignement_losses, _frobenius_losses) = compute_losses(
                    gnn_model,
                    mlp,
                    graph_base=base,
                    graph_corrupted=corrupted,
                    signal_base=base_signal,
                    signal_corrupted=corrupted_signal,
                    qap_values=qap_values,
                    beta=beta,
                )

                loss = losses.mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(gnn_model.parameters(), grad_clip)
                torch.nn.utils.clip_grad_value_(mlp.parameters(), grad_clip)
                gnn_optimizer.step()
            gnn_scheduler.step()

            # Metrics Logging
            if epoch % log_frequency == 0:
                qap_mean_l = []
                qap_values: torch.FloatTensor
                for _, _, _, _, qap_values in train_loader:
                    qap_mean_l.append(float(qap_values.mean()))

                qap_mean_predictor = statistics.mean(qap_mean_l)

                gnn_model.eval()
                train_metrics = {
                    f"{k}/train": v
                    for (k, v) in compute_metrics(
                        gnn_model, mlp, train_loader, beta, qap_mean_predictor, device
                    ).items()
                }
                val_metrics = {
                    f"{k}/val": v
                    for (k, v) in compute_metrics(
                        gnn_model, mlp, val_loader, beta, qap_mean_predictor, device
                    ).items()
                }
                mlflow.log_metrics(train_metrics, epoch)
                mlflow.log_metrics(val_metrics, epoch)

        checkpoint_path = unquote(
            urlparse(run.info.artifact_uri + "/checkpoint.safetensors").path
        )
        save_model(gnn_model, checkpoint_path)
