# Benchmarking and Training GNNs with Combinatorial Optimization

The `gnnco` package simplifies benchmarking and training GNNs on task issued from Combinatorial Optimization (CO).
It is based on PyTorch and the default models are written using the Pytorch Geometric package.

Several functionalities are provided:
- Generating CO Datasets
- Using existing CO Dataset
- A framework to benchmark different GNNs architecture
- Using pretrained GNNs for generating Graph Positional Encodings


# Running the Repo

We use [Rye](https://rye.astral.sh/) to manage the python project. See the documentation for a complete guide.

### Quick installation (Linux and MacOS)

`curl -sSf https://rye.astral.sh/get | bash`

`echo 'source "$HOME/.rye/env"' >> ~/.profile    # For Bash`

`echo 'source "$HOME/.rye/env"' >> ~/.zprofile   # For ZSH`

You may have to restart you shell.

### Cloning the repo

`git clone https://github.com/adrien-lagesse/gnnco.git`

`cd gnnco`

`rye sync`

`rye list`

You sould have a list of all the dependencies of the project.


# Graph Matching problem for benchmarking

## Dataset generation

We provide several command line application to generate graph matching datasets:

- **gm-generate-er** : Generate Erdos-Renyi GM datasets.
- **gm-generate-karateclub** : Generate GM datasets based on the KarateClub Benchmark dataset.
- **gm-generate-corafull** : Generate GM datasets based on the CoraFull Benchmark dataset.
- **gm-generate-aqsol** : Generate GM datasets based on the AQSOL dataset.

To know more about them run:

`gm-generate-er --help`

`gm-generate-karateclub --help`

`gm-generate-corafull --help`

`gm-generate-qm7b --help`

Once you have a dataset, you can print key statistics with `gm-data-stats`

## Training

Use the `gm-train` command line tool to train a Siamese Graph Matching model. (run `gm-train --help` for more information and see scripts/train-siamese-gm.sh for an example)






