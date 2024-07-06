"""
Module providing random operations linked to graphs
"""

from typing import Literal

import torch

from gnnco._core import BatchedDenseGraphs


def erdos_renyi(
    nb_graphs: int,
    order: int,
    p: float,
    *,
    directed: bool = False,
    self_loops: bool | None = False,
) -> BatchedDenseGraphs:
    """
    Generate a batch of random Erdos-Renyi graphs
    """

    assert 0.0 <= p <= 1, "'p' must be between 0 and 1"

    batch = torch.empty(size=(nb_graphs, order, order), dtype=torch.bool).bernoulli_(p)

    if not directed:
        tri_up = batch.triu(0)
        batch = tri_up | tri_up.transpose(1, 2)

    if self_loops is not None:
        idx = torch.arange(order)
        if self_loops:
            batch[:, idx, idx] = True
        else:
            batch[:, idx, idx] = False

    return BatchedDenseGraphs(batch, torch.full((nb_graphs,), order))


@torch.vmap
def __graph_normalization(adj_matrix: torch.Tensor, mask: torch.BoolTensor):
    order = mask.sum()
    avg_degree = adj_matrix.masked_fill(mask.logical_not(), 0).float().sum() / (
        order - 1
    )
    degrees_matrix = torch.empty_like(adj_matrix, dtype=torch.float)
    degrees_matrix.fill_(avg_degree)
    return degrees_matrix / (order - 1 - degrees_matrix)


@torch.vmap
def __node_normalization(adj_matrix: torch.Tensor, mask: torch.BoolTensor):
    order = mask.sum()
    degrees = (
        adj_matrix.masked_fill(mask.logical_not(), 0).sum(dim=1).reshape(-1, 1).float()
    )
    degrees_matrix = torch.sqrt(degrees @ degrees.T)
    return degrees_matrix / (order - 1 - degrees_matrix)


def bernoulli_corruption(
    batch: BatchedDenseGraphs,
    noise: float,
    *,
    directed: bool = False,
    self_loops: bool = False,
    type: Literal["full", "graph_normalized", "node_normalized"] = "node_normalized",
) -> BatchedDenseGraphs:
    """
    Apply a Bernoulli corruption to each graph in the batch
    """

    assert 0.0 <= noise <= 1, "'noise' must be between 0 and 1"

    masks = batch.get_masks()
    stacked_adjacency_matrices = batch.get_stacked_adj()

    edge_noise = torch.empty_like(
        stacked_adjacency_matrices,
        dtype=torch.bool,
    ).bernoulli_(noise)

    if type == "full":
        normalization_tensor = torch.ones_like(stacked_adjacency_matrices)
    elif type == "graph_normalized":
        normalization_tensor = __graph_normalization(stacked_adjacency_matrices, masks)
    elif type == "node_normalized":
        normalization_tensor = __node_normalization(stacked_adjacency_matrices, masks)
    else:
        raise RuntimeError("Unimplemented")

    nonedge_noise = torch.bernoulli(
        torch.clip(noise * normalization_tensor, 0, 1)
    ).bool()

    if not directed:
        tri_up = edge_noise.triu()
        edge_noise = tri_up | tri_up.transpose(1, 2)

        tri_up = nonedge_noise.triu()
        nonedge_noise = tri_up | tri_up.transpose(1, 2)

    if not self_loops:
        idx = torch.arange(int(batch._orders.max()))
        edge_noise[:, idx, idx] = False
        nonedge_noise[:, idx, idx] = False

    corrupted_batch = stacked_adjacency_matrices.clone()
    corrupted_batch[stacked_adjacency_matrices & edge_noise] = False
    corrupted_batch[torch.logical_not(stacked_adjacency_matrices) & nonedge_noise] = (
        True
    )

    return BatchedDenseGraphs(
        corrupted_batch,
        batch.orders().clone(),
    )
