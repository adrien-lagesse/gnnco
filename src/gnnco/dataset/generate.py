import os
import os.path
import pathlib

import click
import torch
from safetensors.torch import save_file

from gnnco.random import bernoulli_corruption, erdos_renyi


@click.group()
def cli():
    pass


@cli.command(name="gm")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
@click.option("--n-graphs", required=True, type=int, help="Number of graphs")
@click.option(
    "--order", required=True, type=int, help="Order of the graphs to generate"
)
@click.option(
    "--density", required=True, type=int, help="Density of the Erdos-Renyi graphs"
)
@click.option("--noise", required=True, type=float, help="Bernouilli noise corruption")
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_erdos_renyi(
    *,
    output_dir: str | os.PathLike,
    n_graphs: int,
    order: int,
    density: int,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating Erdos-Renyi graphs"""
    os.makedirs(output_dir)

    save_file(
        {
            str(i): torch.tensor([order, order], dtype=torch.long)
            for i in range(n_graphs)
        },
        filename=os.path.join(output_dir, "orders.safetensors"),
    )

    with torch.device("cuda" if cuda else "cpu"):
        base_graphs = erdos_renyi(n_graphs, order, density / (order - 1))
        corrupted_graphs = bernoulli_corruption(
            base_graphs, p_edge=noise, p_nonedge=noise * density / (order - density)
        )
        qap_values = {
            str(i): (
                base_graphs[i].adj().float() * corrupted_graphs[i].adj().float()
            ).sum()
            for i in range(n_graphs)
        }

    save_file(
        {str(i): base_graphs[i].edge_index() for i in range(n_graphs)},
        filename=os.path.join(output_dir, "base-graphs.safetensors"),
    )

    save_file(
        {str(i): corrupted_graphs[i].edge_index() for i in range(n_graphs)},
        filename=os.path.join(output_dir, "corrupted-graphs.safetensors"),
    )

    save_file(
        qap_values,
        filename=os.path.join(output_dir, "qap-values.safetensors"),
    )


def main():
    cli()