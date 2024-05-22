import torch
from rdkit import Chem

from gnnco._core import DenseGraph


def smiles_to_graph(smiles: str) -> DenseGraph:
    mol = Chem.MolFromSmiles(smiles)
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO = True)
    return DenseGraph(adjacency_matrix=torch.LongTensor(adjacency_matrix))