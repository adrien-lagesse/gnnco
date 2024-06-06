import inspect
import pathlib
from functools import partial
from typing import Literal, NamedTuple, Self

import torch
import torch.utils.data

from ..._core import BatchedSignals, BatchedSparseGraphs
from ...models import GAT, GCN, GIN, GatedGCN, GATv2
from .._dataset import GMDataset, GMDatasetItem


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


class GMDatasetBatch(NamedTuple):
    base_graphs: BatchedSparseGraphs
    base_signals: BatchedSignals
    corrupted_graphs: BatchedSparseGraphs
    corrupted_signals: BatchedSignals
    padded_batch: torch.LongTensor

    def to(self, device: torch.device) -> Self:
        return GMDatasetBatch(
            self.base_graphs.to(device),
            self.base_signals.to(device),
            self.corrupted_graphs.to(device),
            self.corrupted_signals.to(device),
            self.padded_batch.to(device),
        )

    def device(self) -> torch.device:
        return self.base_graphs.device()

    def __len__(self) -> int:
        return len(self.base_graphs)


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
        batch_l: list[GMDatasetItem],
    ) -> GMDatasetBatch:
        base_batch = BatchedSparseGraphs.from_graphs(
            [item.base_graph for item in batch_l]
        )
        corrupted_batch = BatchedSparseGraphs.from_graphs(
            [item.corrupted_graph for item in batch_l]
        )
        base_signal_batch = BatchedSignals.from_signals(
            [torch.ones((item.base_graph.order(), 1)) for item in batch_l]
        )
        corrupted_signal_batch = BatchedSignals.from_signals(
            [torch.ones((item.corrupted_graph.order(), 1)) for item in batch_l]
        )

        max_order = max([item.base_graph.order() for item in batch_l])

        padded_batch = torch.cat(
            [
                torch.tensor(
                    [i] * batch_l[i].base_graph.order()
                    + [-i - 1] * (max_order - batch_l[i].base_graph.order()),
                    dtype=torch.long,
                )
                for i in range(len(batch_l))
            ]
        )

        return GMDatasetBatch(
            base_graphs=base_batch,
            base_signals=base_signal_batch,
            corrupted_graphs=corrupted_batch,
            corrupted_signals=corrupted_signal_batch,
            padded_batch=padded_batch,
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


def model_factory(
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


def onecycle(
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


def optimizer_factory(
    model: torch.nn.Module,
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
    if optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr)
        return optimizer, scheduler
    elif optimizer == "adam-one-cycle":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(
                onecycle,
                epochs=epochs,
                max_lr=max_lr,
                start_factor=start_factor,
                end_factor=end_factor,
            ),
        )
        return optimizer, scheduler
    else:
        raise RuntimeError(f"Optimizer not found: {optimizer}")
