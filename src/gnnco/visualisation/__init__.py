import graphviz
import matplotlib.colors as clr
from matplotlib import colormaps

import torch

from gnnco._core import DenseGraph, SparseGraph


def plot_graph(graph: SparseGraph | DenseGraph, *, size: float = 8) -> graphviz.Graph:
    dot = graphviz.Graph(strict=True)
    dot.graph_attr = {"size": str(size)}
    dot.node_attr = {
        "shape": "circle",
        "label": "",
        "style": "filled",
        "fillcolor": "black"
        # "fillcolor": "#0D41E1",
    }
    [dot.node(str(i)) for i in range(graph.order())]
    [
        dot.edge(str(i), str(j))
        for i, j in zip(graph.edge_index()[0].tolist(), graph.edge_index()[1].tolist())
    ]
    return dot


def plot_similarities(
    graph: SparseGraph | DenseGraph,
    node: int,
    similarity_matix: torch.Tensor,
    *,
    size: float = 8,
) -> graphviz.Graph:
    softmax_matrix = (similarity_matix - similarity_matix[node].min())/(similarity_matix[node].max() - similarity_matix[node].min())
    # colormap = clr.LinearSegmentedColormap.from_list(
    #     "similarity_cm", ["#a9d6e5", "#012a4a"]
    # )
    colormap = colormaps.get("viridis")
    dot = graphviz.Graph(strict=True)
    dot.graph_attr = {"size": str(size)}
    dot.node_attr = {
        "shape": "circle",
        "label": "",
        "style": "filled",
    }
    [
        dot.node(str(i), fillcolor=clr.to_hex(colormap(float(softmax_matrix[node, i]))))
        for i in range(graph.order())
    ]
    dot
    [
        dot.edge(str(i), str(j))
        for i, j in zip(graph.edge_index()[0].tolist(), graph.edge_index()[1].tolist())
    ]
    return dot
