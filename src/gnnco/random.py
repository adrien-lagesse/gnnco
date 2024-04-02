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
    batch = torch.empty(size=(nb_graphs, order, order), dtype=torch.bool).bernoulli_(p)

    if not directed:
        tri_up = batch.triu(-1)
        batch = tri_up | tri_up.transpose(1, 2)

    if self_loops is not None:
        idx = torch.arange(order)
        if self_loops:
            batch[:, idx, idx] = True
        else:
            batch[:, idx, idx] = False

    return BatchedDenseGraphs(batch, torch.full((nb_graphs,), order))


def bernoulli_corruption(
    batch: BatchedDenseGraphs,
    p_edge: float,
    p_nonedge: float,
    *,
    directed: bool = False,
    self_loops: bool = False,
) -> BatchedDenseGraphs:
    edge_noise = torch.empty(
        size=(len(batch), int(batch._orders.max()), int(batch._orders.max())),
        dtype=torch.bool,
    ).bernoulli_(p_edge)

    nonedge_noise = torch.empty(
        size=(len(batch), int(batch._orders.max()), int(batch._orders.max())),
        dtype=torch.bool,
    ).bernoulli_(p_nonedge)

    if not directed:
        tri_up = edge_noise.triu()
        edge_noise = tri_up | tri_up.transpose(1, 2)

        tri_up = nonedge_noise.triu()
        nonedge_noise = tri_up | tri_up.transpose(1, 2)

    if not self_loops:
        idx = torch.arange(int(batch._orders.max()))
        edge_noise[:, idx, idx] = False
        nonedge_noise[:, idx, idx] = False

    corrupted_batch = batch._stacked_adj_matrices.clone()
    corrupted_batch[batch._stacked_adj_matrices & edge_noise] = False
    corrupted_batch[torch.logical_not(batch._stacked_adj_matrices) & nonedge_noise] = (
        True
    )
    return BatchedDenseGraphs(
        corrupted_batch,
        batch._orders.clone(),
    )
