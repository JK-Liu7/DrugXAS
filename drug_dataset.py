import dgl
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from drug_preprocess import smiles_to_graph


if torch.cuda.is_available():
    device = torch.device('cuda')


class DrugDataset(Dataset):
    def __init__(self, drug_smiles):

        self.drug_smiles = drug_smiles

    def __len__(self):
        return len(self.drug_smiles)

    def __getitem__(self, idx):
        drug_smiles = self.drug_smiles[idx]
        drug_graph = smiles_to_graph(drug_smiles)
        drug_node = drug_graph.num_nodes()
        drug_edge = drug_graph.num_edges()
        return drug_graph, drug_node, drug_edge

    def collate(self, sample):
        drug_graph, drug_node, drug_edge = map(list, zip(*sample))
        drug_graph = dgl.batch(drug_graph).to(device)
        drug_node = torch.tensor(drug_node).to(device)
        drug_edge = torch.tensor(drug_edge).to(device)
        return drug_graph, drug_node, drug_edge
