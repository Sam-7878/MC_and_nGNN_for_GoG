"""Data reading utils."""

import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data   
import ast
import torch
from sklearn.neighbors import NearestNeighbors


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


    def make_pyg_data_from_row(row, feature_cols, label_col="label"):
        x_list = parse_serialized_cell(row["node_features"])
        x = torch.tensor(x_list, dtype=torch.float32)

        y = torch.tensor(int(row[label_col]), dtype=torch.long)

        edge_index = None

        # 1) edge_index 컬럼이 있으면 최우선 사용
        if "edge_index" in row.index:
            parsed = parse_serialized_cell(row["edge_index"])
            if parsed is not None:
                edge_index = ensure_edge_index(parsed)

        # 2) src/dst 컬럼이 따로 있으면 사용
        elif "src" in row.index and "dst" in row.index:
            src = parse_serialized_cell(row["src"])
            dst = parse_serialized_cell(row["dst"])
            if src is not None and dst is not None and len(src) > 0:
                edge_index = ensure_edge_index([src, dst])

        # 3) edges 컬럼이 있으면 사용
        elif "edges" in row.index:
            parsed = parse_serialized_cell(row["edges"])
            if parsed is not None:
                edge_index = ensure_edge_index(parsed)

        # 4) edge 정보가 전혀 없으면 kNN fallback
        if edge_index is None:
            edge_index = build_knn_edge_index(x, k=3, undirected=True)

        return Data(x=x, edge_index=edge_index, y=y)

    def build_edge_index_for_graph(x, row):
        # 실제 edge가 있으면 그걸 사용
        for pair in [("src", "dst")]:
            if pair[0] in row.index and pair[1] in row.index:
                src = parse_serialized_cell(row[pair[0]])
                dst = parse_serialized_cell(row[pair[1]])
                if src is not None and dst is not None and len(src) > 0:
                    return ensure_edge_index([src, dst])

        for col in ["edge_index", "edges"]:
            if col in row.index:
                parsed = parse_serialized_cell(row[col])
                if parsed is not None:
                    return ensure_edge_index(parsed)

        # edge가 없으면 fallback
        return build_knn_edge_index(x, k=3, undirected=True)



def parse_serialized_cell(cell):
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return cell
    if isinstance(cell, str):
        cell = cell.strip()
        if cell in {"", "[]", "None", "nan"}:
            return None
        return ast.literal_eval(cell)
    return cell


def ensure_edge_index(edge_like):
    if edge_like is None:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(edge_like, dtype=torch.long)

    if edge_index.ndim != 2:
        raise ValueError(f"Invalid edge_index ndim: {edge_index.ndim}")

    # [E, 2] -> [2, E]
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()

    if edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape: {tuple(edge_index.shape)}")

    return edge_index.contiguous()


def build_knn_edge_index(x, k=3, undirected=True):
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k, num_nodes - 1)
    x_np = x.detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x_np)
    indices = nbrs.kneighbors(x_np, return_distance=False)

    src, dst = [], []
    for i in range(num_nodes):
        for j in indices[i, 1:]:  # 자기 자신 제외
            src.append(i)
            dst.append(int(j))
            if undirected:
                src.append(int(j))
                dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index
