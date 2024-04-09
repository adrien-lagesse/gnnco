from typing import Self

import torch


class SparseGraph:
    _senders: torch.LongTensor
    _receivers: torch.LongTensor
    _order: int

    def __init__(
        self, *, senders: torch.LongTensor, receivers: torch.LongTensor, order: int
    ) -> None:
        self._senders = senders
        self._receivers = receivers
        self._order = order

    def to(self, device: torch.device) -> Self:
        return SparseGraph(
            senders=self._senders.to(device),
            receivers=self._receivers.to(device),
            order=self._order,
        )

    def device(self) -> torch.device:
        return self._senders.device

    def order(self) -> int:
        return self._order

    def size(self) -> float:
        return int(0.5 * len(self._senders))

    def to_dense(self) -> "DenseGraph":
        adj = torch.full(
            (self._order, self._order), False, dtype=torch.bool, device=self.device()
        )
        adj[self._senders, self._receivers] = True
        return DenseGraph(adjacency_matrix=adj)
    
    def adj(self) -> torch.BoolTensor:
        return self.to_dense().adj()
    
    def edge_index(self) ->torch.LongTensor:
        return torch.vstack([self._senders, self._receivers])


class BatchedSparseGraphs:
    _senders: torch.LongTensor
    _receivers: torch.LongTensor
    _batch: torch.LongTensor
    _orders: torch.LongTensor

    def __init__(
        self,
        *,
        senders: torch.LongTensor,
        receivers: torch.LongTensor,
        batch: torch.LongTensor,
        orders: torch.LongTensor,
    ) -> None:
        self._senders = senders
        self._receivers = receivers
        self._batch = batch
        self._orders = orders

    def from_graphs(graphs: list[SparseGraph]) -> Self:
        device = graphs[0].device()

        senders_l: list[torch.LongTensor] = []
        receivers_l: list[torch.LongTensor] = []
        batch_l: list[torch.LongTensor] = []
        orders_l: list[int] = []

        node_shift = 0
        for i, graph in enumerate(graphs):
            num_edge = len(graph._senders)

            senders_l.append(node_shift + graph._senders)
            receivers_l.append(node_shift + graph._receivers)
            batch_l.append(
                torch.tensor([i] * num_edge, dtype=torch.long, device=device)
            )
            orders_l.append(graph.order())

            node_shift += graph.order()

        return BatchedSparseGraphs(
            senders=torch.cat(senders_l),
            receivers=torch.cat(receivers_l),
            batch=torch.cat(batch_l),
            orders=torch.tensor(orders_l, dtype=torch.long, device=device),
        )

    def to(self, device: torch.device) -> Self:
        return BatchedSparseGraphs(
            senders=self._senders.to(device),
            receivers=self._receivers.to(device),
            order=self._batch.to(device),
        )

    def device(self) -> torch.device:
        return self._senders.device

    def __len__(self) -> int:
        return len(self._orders)

    def __getitem__(self, idx) -> SparseGraph:
        nodes_mask: torch.BoolTensor = self._batch == idx
        node_shift = int(torch.sum(self._orders[:idx]))
        return SparseGraph(
            senders=self._senders[nodes_mask] - node_shift,
            receivers=self._receivers[nodes_mask] - node_shift,
            order=int(self._orders[idx]),
        )

    def unbatch(self) -> list[SparseGraph]:
        unbatched_graphs: list[SparseGraph] = []
        for i in range(len(self)):
            unbatched_graphs.append(self[i])
        return unbatched_graphs

    def to_dense(self) -> "BatchedDenseGraphs":
        return BatchedDenseGraphs.from_graphs(
            list(map(lambda sparse_graph: sparse_graph.to_dense(), self.unbatch()))
        )
    def edge_index(self) ->torch.LongTensor:
        return torch.vstack([self._senders, self._receivers])


class DenseGraph:
    _adj_matrix: torch.BoolTensor

    def __init__(self, *, adjacency_matrix: torch.BoolTensor) -> None:
        self._adj_matrix = adjacency_matrix

    def to(self, device: torch.device) -> Self:
        return DenseGraph(adjacency_matrix=self._adj_matrix.to(device))

    def device(self) -> torch.device:
        return self._adj_matrix.device

    def order(self) -> int:
        return int(self._adj_matrix.shape[0])

    def size(self) -> float:
        return float(0.5 * torch.count_nonzero(self._adj_matrix).float())

    def adj(self) -> torch.BoolTensor:
        return self._adj_matrix
    
    def edge_index(self) -> torch.LongTensor:
        return self.to_sparse().edge_index()

    def to_sparse(self) -> SparseGraph:
        senders, receivers = self._adj_matrix.nonzero(as_tuple=True)
        return SparseGraph(senders=senders, receivers=receivers, order=self.order())


class BatchedDenseGraphs:
    _stacked_adj_matrices: torch.BoolTensor
    _orders: torch.LongTensor

    def __init__(
        self, stacked_adjacency_matrices: torch.BoolTensor, orders: torch.LongTensor
    ) -> None:
        self._stacked_adj_matrices = stacked_adjacency_matrices
        self._orders = orders

    def from_graphs(graphs: list[DenseGraph]) -> Self:
        device = graphs[0].device()
        orders = torch.tensor(
            list(map(lambda graph: graph.order(), graphs)),
            dtype=torch.long,
            device=device,
        )
        max_order = int(torch.max(orders))
        stacked_adj_matrices = torch.empty(
            (len(graphs), max_order, max_order), dtype=torch.bool, device=device
        )
        for i, graph in enumerate(graphs):
            order = graph.order()
            stacked_adj_matrices[i, :order, :order].copy_(graph.adj())

        return BatchedDenseGraphs(stacked_adj_matrices, orders)

    def to(self, device: torch.device) -> Self:
        return BatchedDenseGraphs(
            stacked_adjacency_matrices=self._stacked_adj_matrices.to(device),
            orders=self._orders.to(device),
        )

    def device(self) -> torch.device:
        return self._stacked_adj_matrices.device

    def __len__(self) -> int:
        return len(self._orders)

    def __getitem__(self, idx) -> DenseGraph:
        order = self._orders[idx]
        return DenseGraph(
            adjacency_matrix=self._stacked_adj_matrices[idx, :order, :order]
        )

    def unbatch(self) -> list[DenseGraph]:
        unbatched_graphs: list[DenseGraph] = []
        for i in range(len(self)):
            unbatched_graphs.append(self[i])
        return unbatched_graphs

    def to_sparse(self) -> BatchedSparseGraphs:
        return BatchedSparseGraphs.from_graphs(
            list(map(lambda dense_graph: dense_graph.to_sparse(), self.unbatch()))
        )
