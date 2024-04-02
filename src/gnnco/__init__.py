"""
The `gnnco` package simplifies benchmarking and training GNNs on task issued from Combinatorial Optimization (CO).
It is based on PyTorch and the default models are written using the Pytorch Geometric package.

Several functionalities are provided:
- Generating CO Datasets
- Using existing CO Dataset
- A framework to benchmark different GNNs architecture
- Using pretrained GNNs for generating Graph Positional Encodings
"""

__all__ = [
    "BatchedDenseGraphs",
    "BatchedSparseGraphs",
    "DenseGraph",
    "SparseGraph",
    "random",
    "dataset",
]

from . import dataset, random
from ._core import BatchedDenseGraphs, BatchedSparseGraphs, DenseGraph, SparseGraph
