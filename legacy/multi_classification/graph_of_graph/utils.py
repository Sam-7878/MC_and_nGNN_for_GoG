"""Data reading utils."""

import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from texttable import Texttable
import os
from collections import defaultdict

def hierarchical_graph_reader(path):
    """
    Reading the macro-level graph from disk.
    :param path: Path to the edge list.
    :return graph: Hierarchical graph as a NetworkX object.
    """
    edges = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph

def graph_level_reader(path, number_of_features=3):
    """
    Reading a single graph from disk.
    :param path: Path to the JSON file.
    :return data: Dictionary of data.
    """
    data = json.load(open(path))
    data['features'] = {k:v[:number_of_features] for (k,v) in data['features'].items()}
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

class GraphDatasetGenerator(object):
    """
    Creating an in memory version of the graphs.
    :param path: Folder with json files.
    """
    def __init__(self, path, device):
        self.device = device
        self.path = path
        self._enumerate_graphs()
        self._count_features_and_labels()
        self._create_target()
        self._create_dataset()

    def _enumerate_graphs(self):
        graph_count = len(glob.glob(self.path + "*.json"))
        labels = set()
        features = set()
        self.graphs = []
        for index in tqdm(range(graph_count)):
            graph_file = self._concatenate_name(index)
            data = graph_level_reader(graph_file)
            self.graphs.append(data)
            labels = labels.union(set([data["label"]]))
            feature_lengths = [len(v) for k, v in data["features"].items()]
            features = features.union(set(feature_lengths))
        self.label_map = {v: i for i, v in enumerate(labels)}
        self.feature_map = {v: i for i, v in enumerate(features)}
        

    def _count_features_and_labels(self):
        """
        Counting the number of unique features and labels.
        """
        self.number_of_features = list(self.feature_map.keys())[0]
        self.number_of_labels = len(self.label_map)

    def _transform_edges(self, raw_data):
        """
        Transforming an edge list from the data dictionary to a tensor.
        :param raw_data: Dictionary with edge list.
        :return : Edge list matrix.
        """
        edges = [[edge[0], edge[1]] for edge in raw_data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in raw_data["edges"]]
        return torch.t(torch.LongTensor(edges)).to(self.device)

    def _concatenate_name(self, index):
        """
        Creating a file name from an index.
        :param index: Graph index.
        :return : File name.
        """
        return self.path + str(index) + ".json"

    def _transform_features(self, raw_data):
        """
        Creating a feature matrix from the raw data.
        :param raw_data: Dictionary with features.
        :return feature_matrix: FloatTensor of features.
        """
        number_of_nodes = len(raw_data["features"])
        feature_matrix = np.zeros((number_of_nodes, self.number_of_features))
        feature_matrix = [feats for n, feats in raw_data["features"].items()]
        feature_matrix = torch.FloatTensor(feature_matrix)
        return feature_matrix.to(self.device)

    def _data_transform(self, raw_data):
        """
        Creating a dictionary with the edge list matrix and the features matrix.
        """
        clean_data = dict()
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["features"] = self._transform_features(raw_data)
        return clean_data

    def _create_target(self):
        """
        Creating a target vector.
        """
        self.target = [graph["label"] for graph in self.graphs]
        self.target = torch.LongTensor(self.target).to(self.device)

    def _create_dataset(self):
        """
        Creating a list of dictionaries with edge list matrices and feature matrices.
        """
        self.graphs = [self._data_transform(graph) for graph in self.graphs]

