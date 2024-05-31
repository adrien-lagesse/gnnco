"""
The `gnnco` package simplifies benchmarking and training GNNs on tasks issued from Combinatorial Optimization (CO).
It is based on PyTorch and the default models are written using the Pytorch Geometric package.

Several functionalities are provided:
- Generating CO Datasets
- Using existing CO Dataset
- A framework to benchmark different GNNs architectures
- Using pretrained GNNs for generating Graph Positional Encodings
"""

__all__ = [
    "BatchedDenseGraphs",
    "BatchedSignals",
    "BatchedSparseGraphs",
    "DenseGraph",
    "SparseGraph",
    "dataset",
    "models",
    "random",
    "siamese_gm"
]

from . import dataset, models, random, siamese_gm
from ._core import (
    BatchedDenseGraphs,
    BatchedSignals,
    BatchedSparseGraphs,
    DenseGraph,
    SparseGraph,
)
