"""Data reading utils."""

import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data

def hierarchical_graph_reader(path):
    edges = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph

class GraphDatasetGenerator(object):
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.number_of_features = 7 
        self._create_target()

    def _create_target(self):
        self.target = torch.LongTensor(self.df['label']) if 'label' in self.df.columns else None

    def get_pyg_data_list(self):
        data_list = []
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            node_features = torch.tensor([row['Num_nodes'], row['Num_edges'], row['Density'],
                                          row['Assortativity'], row['Reciprocity'], 
                                          row['Effective_Diameter'], row['Clustering_Coefficient']], 
                                          dtype=torch.float)
            

            data = Data(x=node_features.unsqueeze(0))  
            if self.target is not None:
                data.y = self.target[idx]
            data_list.append(data)
        return data_list