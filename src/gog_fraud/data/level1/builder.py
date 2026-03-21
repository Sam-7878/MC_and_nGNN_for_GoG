import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gog_fraud.data.level1.dataset import (
    STRUCT_ATTR_CANDIDATES,
    Level1GraphDataset,
    ensure_graph_id,
    normalize_graphs_inplace,
    save_graph_list,
    load_graph_list,
)


@dataclass
class Level1BuildConfig:
    split_mode: str = "random"  # "random" or "temporal"
    train_ratio: float = 0.7
    valid_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    timestamp_attr: str = "timestamp"
    normalize_struct_features: bool = False
    struct_attr_candidates: Tuple[str, ...] = STRUCT_ATTR_CANDIDATES
    output_dir: Optional[str] = None


@dataclass
class Level1SplitBundle:
    train_graphs: List
    valid_graphs: List
    test_graphs: List
    metadata: Dict = field(default_factory=dict)


def _validate_ratios(cfg: Level1BuildConfig) -> None:
    total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio must be 1.0, got {total}"
        )


def _to_scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar-like tensor, got shape={tuple(value.shape)}")
        return value.view(-1)[0].item()
    return value


def _extract_timestamp(graph, timestamp_attr: str):
    if not hasattr(graph, timestamp_attr):
        raise ValueError(f"Temporal split requested, but graph has no `{timestamp_attr}`")
    value = getattr(graph, timestamp_attr)
    if value is None:
        raise ValueError(f"Temporal split requested, but graph `{timestamp_attr}` is None")
    return _to_scalar(value)


def _split_counts(n_total: int, cfg: Level1BuildConfig):
    n_train = int(n_total * cfg.train_ratio)
    n_valid = int(n_total * cfg.valid_ratio)
    n_test = n_total - n_train - n_valid
    return n_train, n_valid, n_test


def _clone_graphs(graphs: Sequence):
    return [copy.deepcopy(g) for g in graphs]


def _random_split(graphs: Sequence, cfg: Level1BuildConfig):
    graphs = _clone_graphs(graphs)
    rng = random.Random(cfg.seed)
    rng.shuffle(graphs)

    n_train, n_valid, _ = _split_counts(len(graphs), cfg)
    train_graphs = graphs[:n_train]
    valid_graphs = graphs[n_train:n_train + n_valid]
    test_graphs = graphs[n_train + n_valid:]
    return train_graphs, valid_graphs, test_graphs


def _temporal_split(graphs: Sequence, cfg: Level1BuildConfig):
    graphs = _clone_graphs(graphs)
    graphs = sorted(graphs, key=lambda g: _extract_timestamp(g, cfg.timestamp_attr))

    n_train, n_valid, _ = _split_counts(len(graphs), cfg)
    train_graphs = graphs[:n_train]
    valid_graphs = graphs[n_train:n_train + n_valid]
    test_graphs = graphs[n_train + n_valid:]
    return train_graphs, valid_graphs, test_graphs


def _get_struct_tensor(graph, attr_candidates: Tuple[str, ...]):
    for attr_name in attr_candidates:
        if hasattr(graph, attr_name):
            value = getattr(graph, attr_name)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=torch.float32)
            return attr_name, value.float()
    return None, None


def _set_struct_tensor(graph, attr_name: str, value: torch.Tensor):
    setattr(graph, attr_name, value)


def _collect_train_struct_matrix(graphs: Sequence, attr_candidates: Tuple[str, ...]):
    rows = []
    used_attr_name = None

    for graph in graphs:
        attr_name, value = _get_struct_tensor(graph, attr_candidates)
        if value is None:
            continue

        used_attr_name = attr_name
        if value.dim() == 1:
            rows.append(value.view(1, -1))
        elif value.dim() == 2 and value.size(0) == 1:
            rows.append(value)
        else:
            raise ValueError(
                f"Structural feature must be graph-level [D] or [1, D], got {tuple(value.shape)}"
            )

    if len(rows) == 0:
        return None, None

    return used_attr_name, torch.cat(rows, dim=0)


def compute_struct_normalization_stats(
    train_graphs: Sequence,
    attr_candidates: Tuple[str, ...] = STRUCT_ATTR_CANDIDATES,
):
    attr_name, matrix = _collect_train_struct_matrix(train_graphs, attr_candidates)
    if matrix is None:
        return None, None, None

    mean = matrix.mean(dim=0, keepdim=True)
    std = matrix.std(dim=0, unbiased=False, keepdim=True)
    std = torch.clamp(std, min=1e-8)
    return attr_name, mean, std


def apply_struct_normalization(
    graphs: Sequence,
    attr_name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    for graph in graphs:
        if not hasattr(graph, attr_name):
            continue

        value = getattr(graph, attr_name)
        if value is None:
            continue

        original_dim = value.dim() if torch.is_tensor(value) else None
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32)
        value = value.float()

        if value.dim() == 1:
            value_2d = value.view(1, -1)
            normalized = (value_2d - mean) / std
            normalized = normalized.view(-1)
        elif value.dim() == 2 and value.size(0) == 1:
            normalized = (value - mean) / std
        else:
            raise ValueError(
                f"Structural feature must be graph-level [D] or [1, D], got {tuple(value.shape)}"
            )

        if original_dim == 1:
            _set_struct_tensor(graph, attr_name, normalized.view(-1))
        else:
            _set_struct_tensor(graph, attr_name, normalized)


def _graph_ids_as_list(graphs: Sequence) -> List[int]:
    ids = []
    for idx, graph in enumerate(graphs):
        graph = ensure_graph_id(graph, fallback_graph_id=idx)
        ids.append(int(graph.graph_id.view(-1)[0].item()))
    return ids


def _label_stats(graphs: Sequence) -> Dict[str, int]:
    num_positive = 0
    num_negative = 0

    for graph in graphs:
        y = getattr(graph, "y", None)
        if y is None:
            continue
        y = y.view(-1)[0].item() if torch.is_tensor(y) else float(y)
        if int(y) == 1:
            num_positive += 1
        else:
            num_negative += 1

    return {
        "num_positive": num_positive,
        "num_negative": num_negative,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def build_level1_split_bundle(
    graphs: Sequence,
    cfg: Level1BuildConfig,
    require_label: bool = True,
) -> Level1SplitBundle:
    _validate_ratios(cfg)

    graphs = _clone_graphs(graphs)
    graphs = normalize_graphs_inplace(
        graphs,
        require_label=require_label,
        require_graph_id=True,
    )

    if cfg.split_mode == "random":
        train_graphs, valid_graphs, test_graphs = _random_split(graphs, cfg)
    elif cfg.split_mode == "temporal":
        train_graphs, valid_graphs, test_graphs = _temporal_split(graphs, cfg)
    else:
        raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")

    struct_norm_meta = {
        "enabled": cfg.normalize_struct_features,
        "applied": False,
        "attr_name": None,
        "mean": None,
        "std": None,
    }

    if cfg.normalize_struct_features:
        attr_name, mean, std = compute_struct_normalization_stats(
            train_graphs=train_graphs,
            attr_candidates=cfg.struct_attr_candidates,
        )
        if attr_name is not None:
            apply_struct_normalization(train_graphs, attr_name, mean, std)
            apply_struct_normalization(valid_graphs, attr_name, mean, std)
            apply_struct_normalization(test_graphs, attr_name, mean, std)

            struct_norm_meta = {
                "enabled": True,
                "applied": True,
                "attr_name": attr_name,
                "mean": mean.view(-1).detach().cpu().tolist(),
                "std": std.view(-1).detach().cpu().tolist(),
            }

    metadata = {
        "build_config": asdict(cfg),
        "split_mode": cfg.split_mode,
        "num_total_graphs": len(graphs),
        "num_train_graphs": len(train_graphs),
        "num_valid_graphs": len(valid_graphs),
        "num_test_graphs": len(test_graphs),
        "train_graph_ids": _graph_ids_as_list(train_graphs),
        "valid_graph_ids": _graph_ids_as_list(valid_graphs),
        "test_graph_ids": _graph_ids_as_list(test_graphs),
        "train_label_stats": _label_stats(train_graphs),
        "valid_label_stats": _label_stats(valid_graphs),
        "test_label_stats": _label_stats(test_graphs),
        "struct_normalization": struct_norm_meta,
    }

    return Level1SplitBundle(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        test_graphs=test_graphs,
        metadata=metadata,
    )


def save_split_bundle(
    bundle: Level1SplitBundle,
    output_dir: str,
    metadata_filename: str = "metadata.json",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.pt"
    valid_path = output_dir / "valid.pt"
    test_path = output_dir / "test.pt"
    metadata_path = output_dir / metadata_filename

    save_graph_list(str(train_path), bundle.train_graphs)
    save_graph_list(str(valid_path), bundle.valid_graphs)
    save_graph_list(str(test_path), bundle.test_graphs)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(bundle.metadata), f, ensure_ascii=False, indent=2)

    return {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "metadata_path": str(metadata_path),
    }


def build_and_save_level1_splits(
    graphs: Sequence,
    cfg: Level1BuildConfig,
    require_label: bool = True,
):
    bundle = build_level1_split_bundle(
        graphs=graphs,
        cfg=cfg,
        require_label=require_label,
    )

    save_result = None
    if cfg.output_dir is not None:
        save_result = save_split_bundle(bundle, cfg.output_dir)

    return {
        "bundle": bundle,
        "saved": save_result,
    }


def build_level1_splits_from_pt(
    input_path: str,
    cfg: Level1BuildConfig,
    require_label: bool = True,
):
    graphs = load_graph_list(
        path=input_path,
        require_label=require_label,
        require_graph_id=True,
        validate=True,
    )
    return build_and_save_level1_splits(
        graphs=graphs,
        cfg=cfg,
        require_label=require_label,
    )


def as_datasets(bundle: Level1SplitBundle):
    return {
        "train": Level1GraphDataset(bundle.train_graphs),
        "valid": Level1GraphDataset(bundle.valid_graphs),
        "test": Level1GraphDataset(bundle.test_graphs),
    }
