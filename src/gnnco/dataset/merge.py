import os
import os.path
import pathlib

import click
from safetensors.torch import load_file, save_file


@click.command()
@click.argument("datasets", required=True, nargs=-1)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
def merge_datasets(
    *,
    datasets: list[str],
    output_dir: str | os.PathLike,
):
    """Merge several GM datasets"""
    os.makedirs(output_dir)

    file_names = [
        "train-base-graphs.safetensors",
        "train-corrupted-graphs.safetensors",   
        "train-orders.safetensors",
        "val-base-graphs.safetensors",
        "val-corrupted-graphs.safetensors",
        "val-orders.safetensors",
        "train-base-signals.safetensors",
        "train-corrupted-signals.safetensors",
        "train-qap-values.safetensors",
        "val-base-signals.safetensors",
        "val-corrupted-signals.safetensors",
        "val-qap-values.safetensors",
    ]
    for file_name in file_names:
        new_list = []
        for dataset_path in datasets:
            d = load_file(f"{dataset_path}/{file_name}")
            for k,v in d.items():
                new_list.append(v)
        save_file({str(i): v for i,v in enumerate(new_list)}, f"{output_dir}/{file_name}")
        print(f"Merged {file_name}: {len(new_list)}")


def main():
    merge_datasets()
