from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class RelationBuilderConfig:
    relation_modes: List[str] = field(
        default_factory=lambda: ["embedding_knn"]
    )
    # embedding knn
    knn_k: int = 5
    knn_similarity: str = "cosine"           # "cosine" | "dot"
    knn_self_loops: bool = False

    # temporal window
    temporal_window_size: int = 3
    timestamp_attr: str = "timestamp"        # key in bundle metadata

    # shared entity
    entity_adj_path: Optional[str] = None   # path to prebuilt adjacency tensor

    # edge feature
    include_edge_weight: bool = True

    # graph-level label for Level 2 (how to derive from Level 1 scores)
    level2_label_strategy: str = "any"       # "any" | "majority" | "mean" | "max"


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _load_bundle(path: str) -> Dict:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict bundle, got: {type(bundle)}")
    return bundle


def _require_key(bundle: Dict, key: str):
    if key not in bundle or bundle[key] is None:
        raise KeyError(f"Level 1 bundle must contain '{key}'")


def _validate_bundle(bundle: Dict):
    for key in ("graph_id", "embedding", "score", "logits"):
        _require_key(bundle, key)


def _normalize_embeddings(emb: torch.Tensor) -> torch.Tensor:
    # Use a slightly larger epsilon for numerical stability in cosine similarity
    norm = emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    return emb / norm


def _cosine_similarity_matrix(emb: torch.Tensor) -> torch.Tensor:
    emb_n = _normalize_embeddings(emb)
    return emb_n @ emb_n.t()


def _dot_similarity_matrix(emb: torch.Tensor) -> torch.Tensor:
    return emb @ emb.t()


# ──────────────────────────────────────────────
# Relation edge builders
# ──────────────────────────────────────────────

def build_knn_edges(
    embedding: torch.Tensor,
    k: int,
    similarity: str = "cosine",
    self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        edge_index: [2, E]
        edge_weight: [E]
    """
    n = embedding.size(0)
    k = min(k, n - 1)

    if similarity == "cosine":
        sim_matrix = _cosine_similarity_matrix(embedding)
    elif similarity == "dot":
        sim_matrix = _dot_similarity_matrix(embedding)
    else:
        raise ValueError(f"Unsupported knn_similarity: {similarity}")

    if not self_loops:
        sim_matrix.fill_diagonal_(float("-inf"))

    topk_vals, topk_idx = torch.topk(sim_matrix, k=k, dim=-1)

    src = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = topk_idx.reshape(-1)
    weights = topk_vals.reshape(-1)

    # undirected
    edge_index = torch.stack([
        torch.cat([src, dst], dim=0),
        torch.cat([dst, src], dim=0),
    ], dim=0)
    edge_weight = torch.cat([weights, weights], dim=0)

    return edge_index, edge_weight


def build_temporal_window_edges(
    timestamps: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort nodes by timestamp, connect each node to its
    `window_size` nearest predecessors.

    Returns:
        edge_index: [2, E]
        edge_weight: [E]  (constant 1.0)
    """
    n = timestamps.size(0)
    order = torch.argsort(timestamps.view(-1))

    src_list, dst_list = [], []
    for rank, node_idx in enumerate(order.tolist()):
        start = max(0, rank - window_size)
        predecessors = order[start:rank].tolist()
        for pred in predecessors:
            src_list.append(node_idx)
            dst_list.append(pred)
            src_list.append(pred)
            dst_list.append(node_idx)

    if len(src_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        return edge_index, edge_weight

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long
    )
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    return edge_index, edge_weight


def build_shared_entity_edges(
    entity_adj: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        entity_adj: [N, N] adjacency or [E, 2] edge list
    Returns:
        edge_index: [2, E]
        edge_weight: [E]
    """
    if entity_adj.dim() == 2 and entity_adj.size(0) == entity_adj.size(1):
        # Square matrix case: treat as adjacency
        src, dst = entity_adj.nonzero(as_tuple=True)
        weights = entity_adj[src, dst].float()
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, weights

    if entity_adj.dim() == 2 and entity_adj.size(1) == 2:
        # Edge list case: [E, 2]
        edge_index = entity_adj.t().long()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
        return edge_index, edge_weight

    raise ValueError(
        f"entity_adj must be [N, N] adjacency or [E, 2] edge list, "
        f"got shape={tuple(entity_adj.shape)}"
    )


# ──────────────────────────────────────────────
# Level 2 label derivation from Level 1 scores
# ──────────────────────────────────────────────

def derive_level2_label(
    level1_labels: torch.Tensor,
    strategy: str = "any",
) -> torch.Tensor:
    """
    Derive a single graph-level label for the Level 2 graph
    from Level 1 per-node labels.

    Args:
        level1_labels: [N] float tensor of 0/1 labels
        strategy: "any" | "majority" | "mean" | "max"
    Returns:
        scalar float tensor
    """
    y = level1_labels.view(-1).float()

    if strategy == "any":
        return (y.sum() > 0).float().view(1)
    elif strategy == "majority":
        return (y.mean() >= 0.5).float().view(1)
    elif strategy == "mean":
        return y.mean().view(1)
    elif strategy == "max":
        return y.max().view(1)
    else:
        raise ValueError(f"Unsupported label strategy: {strategy}")


# ──────────────────────────────────────────────
# Edge deduplication and merge
# ──────────────────────────────────────────────

def _dedup_and_merge_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deduplicate edges by (src, dst), averaging duplicate weights.
    """
    if edge_index.size(1) == 0:
        return edge_index, edge_weight

    n = int(edge_index.max().item()) + 1
    keys = edge_index[0] * n + edge_index[1]

    unique_keys, inverse = torch.unique(keys, return_inverse=True)

    new_edge_index = torch.stack([
        unique_keys // n,
        unique_keys % n,
    ], dim=0)

    # Scatter mean for edge weights
    new_weight = torch.zeros(unique_keys.size(0), dtype=torch.float32)
    count = torch.zeros(unique_keys.size(0), dtype=torch.float32)
    new_weight.scatter_add_(0, inverse, edge_weight.float())
    count.scatter_add_(0, inverse, torch.ones_like(edge_weight.float()))
    new_weight = new_weight / count.clamp(min=1.0)

    return new_edge_index, new_weight


def _merge_multi_relation_edges(
    edge_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(edge_pairs) == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
        )

    all_edge_index = torch.cat([ep[0] for ep in edge_pairs], dim=1)
    all_edge_weight = torch.cat([ep[1] for ep in edge_pairs], dim=0)
    return _dedup_and_merge_edges(all_edge_index, all_edge_weight)


# ──────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────

def build_level2_graph(
    bundle: Dict,
    cfg: RelationBuilderConfig,
    entity_adj: Optional[torch.Tensor] = None,
) -> Data:
    """
    Build a single Level 2 PyG Data graph from a Level 1 bundle.

    Each node  = one Level 1 graph (transaction cluster)
    Node feat  = Level 1 embedding
    Edge       = relation between two Level 1 graphs
    Graph label = derived from Level 1 labels
    """
    _validate_bundle(bundle)

    embedding = bundle["embedding"].float()         # [N, D]
    score = bundle["score"].float().view(-1, 1)     # [N, 1]
    logits = bundle["logits"].float().view(-1, 1)   # [N, 1]
    graph_id = bundle["graph_id"].view(-1)          # [N]
    label = bundle.get("label", None)               # [N, 1] or None

    n = embedding.size(0)

    # ── Node features: concat embedding + level1 score
    x = torch.cat([embedding, score], dim=-1)       # [N, D+1]

    # ── Build edges
    edge_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for mode in cfg.relation_modes:
        if mode == "embedding_knn":
            ei, ew = build_knn_edges(
                embedding=embedding,
                k=cfg.knn_k,
                similarity=cfg.knn_similarity,
                self_loops=cfg.knn_self_loops,
            )
            edge_pairs.append((ei, ew))

        elif mode == "temporal_window":
            meta = bundle.get("metadata", {})
            timestamps = meta.get(cfg.timestamp_attr, None)
            if timestamps is None:
                # fallback: use graph_id as pseudo-timestamp
                timestamps = graph_id.float()
            if not torch.is_tensor(timestamps):
                timestamps = torch.tensor(timestamps, dtype=torch.float32)
            timestamps = timestamps.view(-1).float()

            ei, ew = build_temporal_window_edges(
                timestamps=timestamps,
                window_size=cfg.temporal_window_size,
            )
            edge_pairs.append((ei, ew))

        elif mode == "shared_entity":
            if entity_adj is None and cfg.entity_adj_path is not None:
                entity_adj = torch.load(
                    cfg.entity_adj_path,
                    map_location="cpu",
                    weights_only=False,
                )
            if entity_adj is None:
                raise ValueError(
                    "relation_mode 'shared_entity' requires entity_adj tensor "
                    "or entity_adj_path in config."
                )
            ei, ew = build_shared_entity_edges(entity_adj, num_nodes=n)
            edge_pairs.append((ei, ew))

        else:
            raise ValueError(f"Unsupported relation_mode: {mode}")

    edge_index, edge_weight = _merge_multi_relation_edges(edge_pairs)

    # ── Level 2 graph label
    y = None
    if label is not None:
        y = derive_level2_label(
            level1_labels=label.view(-1),
            strategy=cfg.level2_label_strategy,
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.view(-1, 1) if cfg.include_edge_weight else None,
        y=y,
        graph_id=graph_id,
        level1_embedding=embedding,
        level1_score=score,
        level1_logits=logits,
        level1_label=label,
        num_nodes=n,
    )


def build_level2_graph_from_bundle_pt(
    bundle_path: str,
    cfg: RelationBuilderConfig,
    entity_adj: Optional[torch.Tensor] = None,
) -> Data:
    bundle = _load_bundle(bundle_path)
    return build_level2_graph(bundle=bundle, cfg=cfg, entity_adj=entity_adj)


def save_level2_graph(graph: Data, path: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)
    return str(path)
