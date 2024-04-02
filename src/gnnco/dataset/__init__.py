import os
import os.path
from typing import override

import torch.utils.data
from safetensors.torch import load_file

from gnnco._core import SparseGraph


class GMDataset(torch.utils.data.Dataset):
    base_graphs: dict[int, SparseGraph]
    corrupted_graphs: dict[int, SparseGraph]
    qap_values: dict[int, float]

    def __init__(self, root: str | os.PathLike) -> None:
        super().__init__()
        try:
            orders = {
                int(k): v
                for k, v in load_file(os.path.join(root, "orders.safetensors")).items()
            }
            self.base_graphs = {
                int(k): SparseGraph(senders=v[0], receivers=v[1], order=int(orders[int(k)][0]))
                for k, v in load_file(os.path.join(root, "base-graphs.safetensors")).items()
            }
            self.corrupted_graphs = {
                int(k): SparseGraph(senders=v[0], receivers=v[1], order=int(orders[int(k)][1]))
                for k, v in load_file(
                    os.path.join(root, "corrupted-graphs.safetensors")
                ).items()
            }
            self.qap_values = {
                int(k): float(v)
                for k, v in load_file(os.path.join(root, "qap-values.safetensors")).items()
            }

        except:  # noqa: E722
            raise RuntimeError("Unable to load database")

    @override
    def __len__(self) -> int:
        return len(self.base_graphs)

    @override
    def __getitem__(self, index) -> tuple[SparseGraph, SparseGraph, float]:
        return (
            self.base_graphs[index],
            self.corrupted_graphs[index],
            self.qap_values[index],
        )
