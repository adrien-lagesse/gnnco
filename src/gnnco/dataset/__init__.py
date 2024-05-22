import os
import os.path
from typing import Self, override

import torch.utils.data
from safetensors.torch import load_file

from gnnco._core import SparseGraph


class GMDataset(torch.utils.data.Dataset):
    base_graphs: dict[int, SparseGraph]
    corrupted_graphs: dict[int, SparseGraph]
    base_signals: dict[int, torch.FloatTensor]
    corrupted_signals: dict[int, torch.FloatTensor]
    qap_values: dict[int, float]

    def __init__(self, root: str | os.PathLike, *, validation: bool = False) -> None:
        super().__init__()
        prefix = "val" if validation else "train"
        try:
            orders = {
                int(k): v
                for k, v in load_file(os.path.join(root, f"{prefix}-orders.safetensors")).items()
            }
            self.base_graphs = {
                int(k): SparseGraph(
                    senders=v[0], receivers=v[1], order=int(orders[int(k)][0])
                )
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-base-graphs.safetensors")
                ).items()
            }
            self.corrupted_graphs = {
                int(k): SparseGraph(
                    senders=v[0], receivers=v[1], order=int(orders[int(k)][1])
                )
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-corrupted-graphs.safetensors")
                ).items()
            }
            self.base_signals = {
                int(k): v
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-base-signals.safetensors")
                ).items()
            }
            self.corrupted_signals = {
                int(k): v
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-corrupted-signals.safetensors")
                ).items()
            }
            self.qap_values = {
                int(k): v
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-qap-values.safetensors")
                ).items()
            }

        except:  # noqa: E722
            raise RuntimeError("Unable to load database")

    @override
    def __len__(self) -> int:
        return len(self.base_graphs)

    @override
    def __getitem__(self, index) -> tuple[SparseGraph, SparseGraph, torch.FloatTensor, torch.FloatTensor, float]:
        return (
            self.base_graphs[index],
            self.corrupted_graphs[index],
            self.base_signals[index],
            self.corrupted_signals[index],
            self.qap_values[index],
        )
    
    @override
    def __iter__(self) -> Self:
        self.iter_index = 0
        return self
    
    @override
    def __next__(self) -> tuple[SparseGraph, SparseGraph, torch.FloatTensor, torch.FloatTensor, float]:
        if self.iter_index < len(self):
            res = self[self.iter_index]
            self.iter_index += 1
            return res
        else:
            raise StopIteration
    
