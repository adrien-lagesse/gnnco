import os
import pathlib

import click
import gnnco
import torch
from gnnco.random import bernoulli_corruption
from safetensors.torch import save_file
from torch_geometric.datasets import QM7b
from tqdm.auto import tqdm

SPLIT = 6200  # train set 86%, validation set 14%
QM7B_ROOT = "/scratch/jlagesse/QM7b"  # To be changed to .tmp/QM7b


dataset = QM7b(root=QM7B_ROOT)
train_dataset = dataset[:SPLIT]
validation_dataset = dataset[SPLIT:]

@click.command()
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
@click.option(
    "--noise",
    required=True,
    type=float,
    help="Bernouilli noise corruption",
)
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_qm7b(
    *,
    output_dir: str | os.PathLike,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating Erdos-Renyi graphs"""
    os.makedirs(output_dir)

    def generate_and_save(sparse_dataset, prefix):
        orders_dict: dict[str, torch.LongTensor] = {}
        base_graphs_dict: dict[str, torch.LongTensor] = {}
        corrupted_graphs_dict: dict[str, torch.LongTensor] = {}
        base_signals_dict: dict[str, torch.FloatTensor] = {}
        corrupted_signals_dict: dict[str, torch.FloatTensor] = {}
        qap_values_dict: dict[str, torch.FloatTensor] = {}

        device = torch.device("cuda" if cuda else "cpu")
        with device:
            sparse_graphs = [
                gnnco.SparseGraph(
                    senders=data.edge_index[0].long(),
                    receivers=data.edge_index[1].long(),
                    order=data.num_nodes,
                ).to(device)
                for data in sparse_dataset
            ]

            for i, base_graph_sparse in tqdm(enumerate(sparse_graphs), total=len(sparse_graphs)):
                orders_dict[str(i)] = torch.tensor(
                    [base_graph_sparse.order(), base_graph_sparse.order()],
                    dtype=torch.long,
                )
                base_graphs_dict[str(i)] = base_graph_sparse.edge_index()
                base_graph_dense = base_graph_sparse.to_dense()
                edge_probability = base_graph_dense.size()/(base_graph_dense.order()*(base_graph_dense.order()-1))
                corrupted_graph_dense = bernoulli_corruption(
                    gnnco.BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    noise*edge_probability/(1-edge_probability),
                )[0]
                corrupted_graphs_dict[str(i)] = corrupted_graph_dense.edge_index()
                base_signals_dict[str(i)] = torch.ones([base_graph_sparse.order(), 1], dtype=torch.float)
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
    generate_and_save(train_dataset, prefix="train")
    print()
    print()
    print("------ Generating the validation dataset -----")
    print()
    generate_and_save(validation_dataset, prefix="val")

if __name__ == '__main__':
    graph_matching_qm7b()