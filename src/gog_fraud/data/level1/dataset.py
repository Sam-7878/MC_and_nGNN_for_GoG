from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


STRUCT_ATTR_CANDIDATES = ("struct_feat", "graph_attr", "struct_x")


def _to_1d_long_tensor(value, default: Optional[int] = None) -> torch.Tensor:
    if value is None:
        if default is None:
            raise ValueError("Value is None and no default was provided")
        return torch.tensor([default], dtype=torch.long)

    if torch.is_tensor(value):
        return value.view(-1).long()

    return torch.tensor([int(value)], dtype=torch.long)


def _to_1d_float_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.view(-1).float()
    return torch.tensor([float(value)], dtype=torch.float32)


def ensure_graph_id(graph, fallback_graph_id: int):
    graph_id = getattr(graph, "graph_id", None)
    graph.graph_id = _to_1d_long_tensor(graph_id, default=fallback_graph_id)
    return graph


def ensure_graph_label(graph):
    y = getattr(graph, "y", None)
    if y is None:
        return graph
    graph.y = _to_1d_float_tensor(y)
    return graph


def validate_level1_graph(
    graph,
    require_label: bool = True,
    require_graph_id: bool = True,
    require_node_feature: bool = True,
    require_edge_index: bool = True,
) -> None:
    if require_node_feature and not hasattr(graph, "x"):
        raise ValueError("Each graph must contain node features in `.x`")

    if require_node_feature and graph.x is None:
        raise ValueError("Graph `.x` is None")

    if require_edge_index and not hasattr(graph, "edge_index"):
        raise ValueError("Each graph must contain `edge_index`")

    if require_edge_index and graph.edge_index is None:
        raise ValueError("Graph `.edge_index` is None")

    if require_graph_id and not hasattr(graph, "graph_id"):
        raise ValueError("Each graph must contain `graph_id`")

    if require_label and not hasattr(graph, "y"):
        raise ValueError("Each graph must contain graph-level label `y`")


def normalize_graphs_inplace(
    graphs: Sequence,
    require_label: bool = True,
    require_graph_id: bool = True,
) -> List:
    normalized = []
    for idx, graph in enumerate(graphs):
        graph = ensure_graph_id(graph, fallback_graph_id=idx)
        graph = ensure_graph_label(graph)
        validate_level1_graph(
            graph,
            require_label=require_label,
            require_graph_id=require_graph_id,
        )
        normalized.append(graph)
    return normalized


def load_graph_list(
    path: str,
    require_label: bool = True,
    require_graph_id: bool = True,
    validate: bool = True,
):
    path = Path(path)
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list of PyG Data objects, got: {type(graphs)}")

    graphs = normalize_graphs_inplace(
        graphs,
        require_label=require_label,
        require_graph_id=require_graph_id,
    )

    if validate:
        for graph in graphs:
            validate_level1_graph(
                graph,
                require_label=require_label,
                require_graph_id=require_graph_id,
            )
    return graphs


def save_graph_list(path: str, graphs: Sequence) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(graphs), path)
    return str(path)


def infer_input_dim(graphs: Sequence) -> int:
    if len(graphs) == 0:
        raise ValueError("Cannot infer input dimension from empty graph list")
    if not hasattr(graphs[0], "x") or graphs[0].x is None:
        raise ValueError("Each graph must contain node features in `.x`")
    return int(graphs[0].x.size(-1))


def infer_struct_dim(graphs: Sequence, attr_candidates=STRUCT_ATTR_CANDIDATES) -> int:
    if len(graphs) == 0:
        return 0

    graph = graphs[0]
    for attr_name in attr_candidates:
        if hasattr(graph, attr_name):
            feat = getattr(graph, attr_name)
            if feat is None:
                continue
            if feat.dim() == 1:
                return int(feat.numel())
            if feat.dim() >= 2:
                return int(feat.size(-1))
    return 0


def get_graph_ids(graphs: Sequence) -> torch.Tensor:
    ids = []
    for idx, graph in enumerate(graphs):
        graph = ensure_graph_id(graph, fallback_graph_id=idx)
        ids.append(graph.graph_id.view(-1)[0].long())
    return torch.stack(ids, dim=0)


class Level1GraphDataset(Dataset):
    def __init__(
        self,
        graphs: Sequence,
        require_label: bool = True,
        require_graph_id: bool = True,
        validate: bool = True,
    ):
        self.graphs = list(graphs)
        self.graphs = normalize_graphs_inplace(
            self.graphs,
            require_label=require_label,
            require_graph_id=require_graph_id,
        )

        if validate:
            for graph in self.graphs:
                validate_level1_graph(
                    graph,
                    require_label=require_label,
                    require_graph_id=require_graph_id,
                )

        self.require_label = require_label
        self.require_graph_id = require_graph_id

    @classmethod
    def from_pt(
        cls,
        path: str,
        require_label: bool = True,
        require_graph_id: bool = True,
        validate: bool = True,
    ):
        graphs = load_graph_list(
            path=path,
            require_label=require_label,
            require_graph_id=require_graph_id,
            validate=validate,
        )
        return cls(
            graphs=graphs,
            require_label=require_label,
            require_graph_id=require_graph_id,
            validate=validate,
        )

    def to_pt(self, path: str) -> str:
        return save_graph_list(path, self.graphs)

    def input_dim(self) -> int:
        return infer_input_dim(self.graphs)

    def struct_dim(self) -> int:
        return infer_struct_dim(self.graphs)

    def graph_ids(self) -> torch.Tensor:
        return get_graph_ids(self.graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx]
