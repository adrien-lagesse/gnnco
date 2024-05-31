import os
import pathlib

import click
import gnnco
import torch
from gnnco.random import bernoulli_corruption
from safetensors.torch import save_file
from torch_geometric.datasets import KarateClub
from tqdm.auto import tqdm

KARATECLUB_ROOT = "/scratch/jlagesse/KarateClub"  # To be changed to .tmp/CoraFull
dataset = KarateClub()
edge_index = dataset.edge_index
num_nodes = int(torch.max(dataset.edge_index) + 1)


@click.command()
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
@click.option("--n-graphs", required=True, type=int, help="Number of graphs")
@click.option(
    "--n-val-graphs", required=True, type=int, help="Number of validation graphs"
)
@click.option(
    "--noise",
    required=True,
    type=float,
    help="Bernouilli noise corruption",
)
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_cora_full(
    *,
    output_dir: str | os.PathLike,
    n_graphs: int,
    n_val_graphs: int,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating Erdos-Renyi graphs"""
    os.makedirs(output_dir)

    def generate_and_save(N, prefix):
        orders_dict: dict[str, torch.LongTensor] = {}
        base_graphs_dict: dict[str, torch.LongTensor] = {}
        corrupted_graphs_dict: dict[str, torch.LongTensor] = {}
        base_signals_dict: dict[str, torch.FloatTensor] = {}
        corrupted_signals_dict: dict[str, torch.FloatTensor] = {}
        qap_values_dict: dict[str, torch.FloatTensor] = {}

        device = torch.device("cuda" if cuda else "cpu")
        with device:
            base_graph_sparse = gnnco.SparseGraph(
                senders=edge_index[0].long(),
                receivers=edge_index[1].long(),
                order=num_nodes,
            ).to(device)

            base_graphs_dict["0"] = base_graph_sparse.edge_index()

            base_graph_dense = base_graph_sparse.to_dense()

            edge_probability = base_graph_dense.size() / (
                num_nodes * (num_nodes - 1)
            )

            base_signals_dict["0"] = torch.ones(
                [base_graph_sparse.order(), 1], dtype=torch.float
            )

            for i in tqdm(range(N), total=N):
                orders_dict[str(i)] = torch.tensor(
                    [base_graph_sparse.order(), base_graph_sparse.order()],
                    dtype=torch.long,
                )
                corrupted_graph_dense = bernoulli_corruption(
                    gnnco.BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    noise * edge_probability / (1 - edge_probability),
                )[0]
                corrupted_graphs_dict[str(i)] = corrupted_graph_dense.edge_index()

                corrupted_signals_dict[str(i)] = torch.ones(
                    [base_graph_sparse.order(), 1], dtype=torch.float
                )
                qap_values_dict[str(i)] = (
                    base_graph_dense.adj().float() * corrupted_graph_dense.adj().float()
                ).sum(dim=1)

        save_file(
            orders_dict,
            filename=os.path.join(output_dir, f"{prefix}-orders.safetensors"),
        )

        save_file(
            base_graphs_dict,
            filename=os.path.join(output_dir, f"{prefix}-base-graphs.safetensors"),
        )

        save_file(
            corrupted_graphs_dict,
            filename=os.path.join(output_dir, f"{prefix}-corrupted-graphs.safetensors"),
        )

        save_file(
            base_signals_dict,
            filename=os.path.join(output_dir, f"{prefix}-base-signals.safetensors"),
        )

        save_file(
            corrupted_signals_dict,
            filename=os.path.join(
                output_dir, f"{prefix}-corrupted-signals.safetensors"
            ),
        )

        save_file(
            qap_values_dict,
            filename=os.path.join(output_dir, f"{prefix}-qap-values.safetensors"),
        )

    print("------ Generating the training dataset   ------")
    print()
    generate_and_save(n_graphs, prefix="train")
    print()
    print()
    print("------ Generating the validation dataset -----")
    print()
    generate_and_save(n_val_graphs, prefix="val")


if __name__ == "__main__":
    graph_matching_cora_full()
