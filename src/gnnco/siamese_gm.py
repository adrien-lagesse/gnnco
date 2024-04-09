import inspect
import pathlib
from typing import Literal

import mlflow
import numpy as np
import torch
import torch.utils.data

from gnnco import BatchedSparseGraphs, SparseGraph
from gnnco.dataset import GMDataset
from gnnco.models import GAT, GCN, GIN


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
) -> tuple[
    GMDataset, GMDataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    train_dataset = GMDataset(root=dataset_path)
    val_dataset = GMDataset(root=dataset_path, validation=True)

    def collate_fn(
        batch_l: list[tuple[SparseGraph, SparseGraph, torch.FloatTensor, float]],
    ) -> tuple[
        BatchedSparseGraphs, BatchedSparseGraphs, torch.FloatTensor, torch.FloatTensor
    ]:
        base_batch = BatchedSparseGraphs.from_graphs(list(map(lambda t: t[0], batch_l)))
        corrupted_batch = BatchedSparseGraphs.from_graphs(
            list(map(lambda t: t[1], batch_l))
        )
        signal_batch = torch.vstack(list(map(lambda t: t[2], batch_l)))
        qap_values_batch = torch.FloatTensor(list(map(lambda t: t[3], batch_l)))

        return base_batch, corrupted_batch, signal_batch, qap_values_batch

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
    model: Literal["GCN", "GIN", "GAT"],
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
    else:
        raise RuntimeError(f"Model name '{model}' does not exists")


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
    model: Literal["GCN", "GIN", "GAT"] | None,
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

    with mlflow.start_run(run_name=run_name, log_system_metrics=profile):
        mlflow.log_params(__get_kwargs())

        # Load the training and validation datasets and build suitable loaders to batch the graphs together.
        (train_dataset, val_dataset, train_loader, val_loader) = setup_data(
            dataset_path=dataset, batch_size=batch_size
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

        # Computing the number of parameters in the GNN
        mlflow.log_param("nb_params", sum([np.prod(p.size()) for p in gnn_model.parameters()]))
