import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data


def graph_level_reader(path):
    """
    Reading a single graph from disk.
    :param path: Path to the JSON file.
    :return data: Dictionary of data.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

class GraphDatasetGenerator(object):
    """
    Creating an in-memory version of the graphs without node features.
    :param path: Folder with JSON files.
    """
    def __init__(self, path):
        self.path = path
        self.graphs = []
        self._enumerate_graphs()
        self._create_target()
        self._create_dataset()

    def _enumerate_graphs(self):
        """
        Listing the graph files and loading data.
        """
        graph_files = glob.glob(self.path + "*.json")
        for graph_file in tqdm(graph_files):
            data = graph_level_reader(graph_file)
            self.graphs.append(data)

    def _transform_edges(self, raw_data):
        """
        Transforming an edge list from the data dictionary to a tensor.
        :param raw_data: Dictionary with edge list.
        :return : Edge list matrix.
        """
        edges = [[edge[0], edge[1]] for edge in raw_data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in raw_data["edges"]]
        return torch.LongTensor(edges).t().contiguous()

    def _create_target(self):
        """
        Creating a target vector based on labels.
        """
        label_set = {graph.get("label", -1) for graph in self.graphs}
        self.label_map = {label: i for i, label in enumerate(label_set)}
        self.target = torch.LongTensor([self.label_map.get(graph.get("label", -1), -1) for graph in self.graphs])

    def _data_transform(self, raw_data):
        """
        Creating a dictionary with only the edge list matrix.
        """
        clean_data = dict()
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["label"] = raw_data.get("label", -1) 
        return clean_data

    def _create_dataset(self):
        """
        Creating a list of dictionaries with edge list matrices.
        """
        self.graphs = [self._data_transform(graph) for graph in self.graphs]

    def get_pyg_data_list(self):
        """
        Creating a list of PyG Data objects from the graphs.
        :return data_list: List of PyG Data objects.
        """
        data_list = []
        for graph in self.graphs:
            data_obj = Data(edge_index=graph['edges'])
            label = graph.get("label", -1)
            if label != -1:
                data_obj.y = torch.tensor([self.label_map[label]])
            data_list.append(data_obj)
        return data_list

