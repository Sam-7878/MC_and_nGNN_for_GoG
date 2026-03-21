from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_level2_graph(
    graph: Data,
    require_label: bool = True,
    require_edge_index: bool = True,
    require_node_feature: bool = True,
) -> None:
    if require_node_feature:
        if not hasattr(graph, "x") or graph.x is None:
            raise ValueError("Level 2 graph must contain node features in `.x`")

    if require_edge_index:
        if not hasattr(graph, "edge_index") or graph.edge_index is None:
            raise ValueError("Level 2 graph must contain `edge_index`")

    if require_label:
        if not hasattr(graph, "y") or graph.y is None:
            raise ValueError("Level 2 graph must contain graph-level label `y`")


def normalize_level2_graph(graph: Data) -> Data:
    if hasattr(graph, "y") and graph.y is not None:
        graph.y = graph.y.view(-1).float()
    return graph


# ──────────────────────────────────────────────
# Load / Save
# ──────────────────────────────────────────────

def load_level2_graph(
    path: str,
    require_label: bool = True,
    validate: bool = True,
) -> Data:
    path = Path(path)
    graph = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(graph, Data):
        raise TypeError(
            f"Expected a single torch_geometric.data.Data object, got: {type(graph)}"
        )

    graph = normalize_level2_graph(graph)
    if validate:
        validate_level2_graph(graph, require_label=require_label)
    return graph


def load_level2_graph_list(
    paths: Sequence[str],
    require_label: bool = True,
    validate: bool = True,
) -> List[Data]:
    return [
        load_level2_graph(path, require_label=require_label, validate=validate)
        for path in paths
    ]


def save_level2_graph_list(
    graphs: Sequence[Data],
    path: str,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(graphs), path)
    return str(path)


def load_level2_graph_list_from_pt(
    path: str,
    require_label: bool = True,
    validate: bool = True,
) -> List[Data]:
    path = Path(path)
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list of PyG Data objects, got: {type(graphs)}")

    normalized = []
    for g in graphs:
        g = normalize_level2_graph(g)
        if validate:
            validate_level2_graph(g, require_label=require_label)
        normalized.append(g)
    return normalized


# ──────────────────────────────────────────────
# Dimension helpers
# ──────────────────────────────────────────────

def infer_level2_node_dim(graphs: Sequence[Data]) -> int:
    if len(graphs) == 0:
        raise ValueError("Cannot infer node dim from empty graph list")
    if graphs[0].x is None:
        raise ValueError("Graph `.x` is None")
    return int(graphs[0].x.size(-1))


def infer_level2_edge_dim(graphs: Sequence[Data]) -> int:
    if len(graphs) == 0:
        return 0
    g = graphs[0]
    if not hasattr(g, "edge_attr") or g.edge_attr is None:
        return 0
    if g.edge_attr.dim() == 1:
        return 1
    return int(g.edge_attr.size(-1))


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class Level2GraphDataset(Dataset):
    def __init__(
        self,
        graphs: Sequence[Data],
        require_label: bool = True,
        validate: bool = True,
    ):
        self.graphs = []
        for g in graphs:
            g = normalize_level2_graph(g)
            if validate:
                validate_level2_graph(g, require_label=require_label)
            self.graphs.append(g)

        self.require_label = require_label

    @classmethod
    def from_pt(
        cls,
        path: str,
        require_label: bool = True,
        validate: bool = True,
    ):
        graphs = load_level2_graph_list_from_pt(
            path=path,
            require_label=require_label,
            validate=validate,
        )
        return cls(graphs=graphs, require_label=require_label, validate=False)

    def to_pt(self, path: str) -> str:
        return save_level2_graph_list(self.graphs, path)

    def node_dim(self) -> int:
        return infer_level2_node_dim(self.graphs)

    def edge_dim(self) -> int:
        return infer_level2_edge_dim(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
