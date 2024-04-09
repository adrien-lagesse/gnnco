import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm

from gnnco import BatchedSparseGraphs

GN_FREQ = 3


class GIN(torch.nn.Module):
    def __init__(self, layers: int, features: int, out_features: int):
        super().__init__()
        assert layers > 0

        self.layer0 = GINConv(nn=torch.nn.Linear(1, features))
        self.layers = torch.nn.ModuleList(
            [
                GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(features, features),
                        torch.nn.ReLU(),
                        torch.nn.Linear(features, features, bias=(i % GN_FREQ == 0))
                    )
                )
                for i in range(layers - 1)
            ]
        )
        self.gns = torch.nn.ModuleList(
            [GraphNorm(features) for _ in range(layers // GN_FREQ + 1)]
        )
        self.linear = torch.nn.Linear(features, out_features)

    def forward(self, signals: torch.FloatTensor, batch: BatchedSparseGraphs):
        edge_index = batch.edge_index()
        x = F.relu(self.layer0(signals, edge_index))
        for i in range(len(self.layers)):
            if i % GN_FREQ == 0:
                x = self.gns[i // GN_FREQ](x)
            x = x + F.relu(self.layers[i](x, edge_index))

        x = self.linear(x)

        return x
