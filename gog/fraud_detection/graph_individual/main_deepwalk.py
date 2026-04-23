#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_individual/main_deepwalk.py

Level-1 graph classification using DeepWalk-based intra-graph embeddings.

Expected input format
---------------------
1) node_csv
   graph_id,node_id,<node_feature_1>,<node_feature_2>,...

2) edge_csv
   graph_id,src,dst,<optional_edge_feature_1>,<optional_edge_feature_2>,...

3) label_csv
   Contract,label
   or any equivalent key column among:
     Contract, contract, contract_address, address, graph_id

Example
-------
python ./graph_individual/main_deepwalk.py \
  --chain bsc \
  --node_csv ./_data/level1/bsc_level1_nodes.csv \
  --edge_csv ./_data/level1/bsc_level1_edges.csv \
  --label_csv ./_data/dataset/features/bsc_basic_metrics_processed.csv \
  --graph_id_col graph_id \
  --node_id_col node_id \
  --label_col label \
  --undirected \
  --deepwalk_dim 64 \
  --walk_length 20 \
  --num_workers 4 \
  --window_size 5 \
  --dw_epochs 5 \
  --mlp_hidden 128 \
  --mlp_out 64 \
  --epochs 50 \
  --batch_size 64 \
  --save_graph_artifacts
"""

import os
import json
import math
import random
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


LABEL_KEY_CANDIDATES = ["Contract", "contract", "contract_address", "address", "graph_id"]
EDGE_GRAPH_ID_CANDIDATES = ["graph_id", "GraphID", "graph", "contract", "Contract"]
EDGE_SRC_CANDIDATES = ["src", "source", "from", "from_address", "sender"]
EDGE_DST_CANDIDATES = ["dst", "target", "to", "to_address", "receiver"]


# ============================================================
# Utils
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_id(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in {"", "nan", "none", "null"}:
        return None
    return x


def infer_col(columns, candidates, default=None):
    lowered = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if str(cand).lower() in lowered:
            return lowered[str(cand).lower()]
    return default


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def metric_or_nan(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def compute_best_threshold(y_true, y_prob):
    if len(y_true) == 0:
        return 0.5, float("nan")

    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def signed_log1p_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# ============================================================
# Feature stabilization
# ============================================================

def stabilize_node_features(
    node_df: pd.DataFrame,
    graph_id_col: str,
    node_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    feat = safe_numeric_df(node_df[node_feature_cols]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cols_lower = {c: str(c).lower() for c in feat.columns}

    value_cols = [c for c in feat.columns if "value" in cols_lower[c]]
    count_cols = [c for c in feat.columns if ("count" in cols_lower[c] or "degree" in cols_lower[c])]
    duration_cols = [c for c in feat.columns if "duration" in cols_lower[c]]
    ratio_cols = [c for c in feat.columns if "ratio" in cols_lower[c]]
    ts_cols = [c for c in feat.columns if cols_lower[c].endswith("_ts") or "timestamp" in cols_lower[c]]

    # 1) value columns: wei -> ether -> signed log1p
    for c in value_cols:
        x = feat[c].to_numpy(dtype=np.float64)
        x = x / 1e18
        x = signed_log1p_np(x)
        feat[c] = x

    # 2) counts / degree / duration: log1p on non-negative part
    for c in count_cols + duration_cols:
        if c in feat.columns:
            x = feat[c].to_numpy(dtype=np.float64)
            x = np.clip(x, a_min=0.0, a_max=None)
            feat[c] = np.log1p(x)

    # 3) timestamps -> relative time inside each graph
    if len(ts_cols) > 0:
        tmp = pd.concat(
            [
                node_df[[graph_id_col]].reset_index(drop=True),
                feat[ts_cols].reset_index(drop=True),
            ],
            axis=1,
        )

        for gid, idx in tmp.groupby(graph_id_col).groups.items():
            sub = tmp.loc[idx, ts_cols].replace(0, np.nan)
            if sub.notna().any().any():
                base_ts = np.nanmin(sub.to_numpy(dtype=np.float64))
                tmp.loc[idx, ts_cols] = tmp.loc[idx, ts_cols] - base_ts

        feat[ts_cols] = tmp[ts_cols].fillna(0.0)

    # 4) ratio clip
    for c in ratio_cols:
        feat[c] = feat[c].clip(-10.0, 10.0)

    # 5) final global clip for numerical safety
    feat = feat.clip(lower=-50.0, upper=50.0)

    # 6) final finite guarantee
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat, feat.columns.tolist()


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Level-1 graph classification using DeepWalk on precomputed node/edge CSVs"
    )

    parser.add_argument("--chain", type=str, required=True)

    parser.add_argument("--node_csv", type=str, required=True,
                        help="Level1 node csv from common_node.py")
    parser.add_argument("--edge_csv", type=str, required=True,
                        help="Level1 edge csv from common_node.py")
    parser.add_argument("--label_csv", type=str, required=True,
                        help="Label csv, e.g. basic_metrics_processed.csv")

    parser.add_argument("--graph_id_col", type=str, default="graph_id")
    parser.add_argument("--node_id_col", type=str, default="node_id")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--label_key_col", type=str, default=None)

    parser.add_argument("--edge_graph_id_col", type=str, default="graph_id")
    parser.add_argument("--src_col", type=str, default="src")
    parser.add_argument("--dst_col", type=str, default="dst")
    parser.add_argument("--undirected", action="store_true")

    # DeepWalk
    parser.add_argument("--deepwalk_dim", type=int, default=64)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--dw_epochs", type=int, default=5)
    parser.add_argument("--dw_workers", type=int, default=4)
    parser.add_argument("--max_walk_start_nodes", type=int, default=512,
                        help="Cap number of start nodes per graph for DeepWalk speed control")
    parser.add_argument("--use_node_feature_stats", action="store_true",
                        help="Concatenate stabilized node feature mean/max into graph features")
    parser.add_argument("--use_graph_size_features", action="store_true",
                        help="Concatenate basic graph size features into graph features")

    # Classifier
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_out", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--save_graph_artifacts", action="store_true")
    parser.add_argument("--artifacts_dir", type=str, default="graph_individual_artifacts")
    parser.add_argument("--fail_if_all_singleton", action="store_true")

    return parser.parse_args()


# ============================================================
# Label loading
# ============================================================

def load_label_map(label_csv: str, label_col: str, label_key_col: Optional[str] = None):
    df = pd.read_csv(label_csv, low_memory=False)

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in label_csv")

    if label_key_col is None:
        label_key_col = infer_col(df.columns, LABEL_KEY_CANDIDATES)

    if label_key_col is None:
        raise ValueError(
            f"Could not infer label key column from label_csv. Tried: {LABEL_KEY_CANDIDATES}"
        )

    work = df[[label_key_col, label_col]].copy()
    work[label_key_col] = work[label_key_col].map(normalize_id)
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[label_key_col, label_col]).copy()
    work[label_col] = work[label_col].astype(int)

    label_map = dict(zip(work[label_key_col], work[label_col]))
    return label_map, label_key_col, len(work)


# ============================================================
# DeepWalk
# ============================================================

def import_word2vec():
    try:
        from gensim.models import Word2Vec
        return Word2Vec
    except Exception as e:
        raise ImportError(
            "DeepWalk requires gensim. Please install it first: pip install gensim"
        ) from e


def random_walk(G, start_node, walk_length: int, rng: np.random.Generator):
    walk = [str(start_node)]
    cur = start_node

    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(cur))
        if len(neighbors) == 0:
            break
        cur = neighbors[int(rng.integers(0, len(neighbors)))]
        walk.append(str(cur))

    return walk


def generate_walks(
    G,
    walk_length: int,
    num_workers: int,
    seed: int,
    max_walk_start_nodes: int,
):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return []

    if max_walk_start_nodes is not None and max_walk_start_nodes > 0 and len(nodes) > max_walk_start_nodes:
        start_nodes = rng.choice(nodes, size=max_walk_start_nodes, replace=False).tolist()
    else:
        start_nodes = nodes[:]

    walks = []
    for _ in range(num_workers):
        rng.shuffle(start_nodes)
        for node in start_nodes:
            walks.append(random_walk(G, node, walk_length, rng))
    return walks


def fit_deepwalk_node_embeddings(
    G,
    local_nodes: List[str],
    dim: int,
    walk_length: int,
    num_workers: int,
    window_size: int,
    epochs: int,
    workers: int,
    seed: int,
    max_walk_start_nodes: int,
):
    if len(local_nodes) == 0:
        return np.zeros((0, dim), dtype=np.float32), 0.0

    if G.number_of_edges() == 0:
        return np.zeros((len(local_nodes), dim), dtype=np.float32), 0.0

    walks = generate_walks(
        G=G,
        walk_length=walk_length,
        num_workers=num_workers,
        seed=seed,
        max_walk_start_nodes=max_walk_start_nodes,
    )

    if len(walks) == 0:
        return np.zeros((len(local_nodes), dim), dtype=np.float32), 0.0

    Word2Vec = import_word2vec()

    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window_size,
        min_count=1,
        sg=1,
        hs=1,
        negative=0,
        workers=max(1, workers),
        epochs=epochs,
        seed=seed,
    )

    emb = np.zeros((len(local_nodes), dim), dtype=np.float32)
    found = 0
    for i, nid in enumerate(local_nodes):
        key = str(nid)
        if key in model.wv:
            emb[i] = model.wv[key]
            found += 1

    coverage = float(found / max(len(local_nodes), 1))
    return emb, coverage


# ============================================================
# Graph feature building
# ============================================================

def summarize_graphs(graph_infos: list) -> pd.DataFrame:
    df = pd.DataFrame(graph_infos)
    if df.empty:
        return df
    df["singleton"] = (df["num_nodes"] <= 1).astype(int)
    df["zero_edge"] = (df["num_edges"] <= 0).astype(int)
    return df


def build_graph_feature_vector(
    node_feat_np: np.ndarray,
    dw_emb_np: np.ndarray,
    num_nodes: int,
    num_edges: int,
    dw_coverage: float,
    use_node_feature_stats: bool,
    use_graph_size_features: bool,
):
    pieces = []

    # DeepWalk pooled features
    if dw_emb_np.shape[0] > 0:
        dw_mean = dw_emb_np.mean(axis=0)
        dw_max = dw_emb_np.max(axis=0)
    else:
        dw_mean = np.zeros((0,), dtype=np.float32)
        dw_max = np.zeros((0,), dtype=np.float32)

    pieces.append(dw_mean.astype(np.float32))
    pieces.append(dw_max.astype(np.float32))

    # Stabilized node feature pooled features
    if use_node_feature_stats:
        if node_feat_np.shape[0] > 0:
            nf_mean = node_feat_np.mean(axis=0)
            nf_max = node_feat_np.max(axis=0)
        else:
            nf_mean = np.zeros((node_feat_np.shape[1],), dtype=np.float32)
            nf_max = np.zeros((node_feat_np.shape[1],), dtype=np.float32)

        pieces.append(nf_mean.astype(np.float32))
        pieces.append(nf_max.astype(np.float32))

    # Graph size features
    if use_graph_size_features:
        density = 0.0
        if num_nodes > 1:
            density = float(num_edges / (num_nodes * (num_nodes - 1)))

        size_vec = np.array(
            [
                math.log1p(max(num_nodes, 0)),
                math.log1p(max(num_edges, 0)),
                min(max(density, 0.0), 1.0),
                min(max(dw_coverage, 0.0), 1.0),
            ],
            dtype=np.float32,
        )
        pieces.append(size_vec)

    out = np.concatenate(pieces, axis=0).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_graph_dataset_deepwalk(
    node_csv: str,
    edge_csv: str,
    label_map: Dict[str, int],
    graph_id_col: str,
    node_id_col: str,
    edge_graph_id_col: str,
    src_col: str,
    dst_col: str,
    undirected: bool,
    deepwalk_dim: int,
    walk_length: int,
    num_workers: int,
    window_size: int,
    dw_epochs: int,
    dw_workers: int,
    max_walk_start_nodes: int,
    use_node_feature_stats: bool,
    use_graph_size_features: bool,
    seed: int,
):
    import networkx as nx

    node_df = pd.read_csv(node_csv, low_memory=False)
    edge_df = pd.read_csv(edge_csv, low_memory=False)

    if graph_id_col not in node_df.columns:
        raise ValueError(f"graph_id_col '{graph_id_col}' not found in node_csv")
    if node_id_col not in node_df.columns:
        raise ValueError(f"node_id_col '{node_id_col}' not found in node_csv")

    if edge_graph_id_col not in edge_df.columns:
        inferred = infer_col(edge_df.columns, EDGE_GRAPH_ID_CANDIDATES)
        if inferred is None:
            raise ValueError(f"edge_graph_id_col '{edge_graph_id_col}' not found in edge_csv")
        edge_graph_id_col = inferred

    if src_col not in edge_df.columns:
        inferred = infer_col(edge_df.columns, EDGE_SRC_CANDIDATES)
        if inferred is None:
            raise ValueError(f"src_col '{src_col}' not found in edge_csv")
        src_col = inferred

    if dst_col not in edge_df.columns:
        inferred = infer_col(edge_df.columns, EDGE_DST_CANDIDATES)
        if inferred is None:
            raise ValueError(f"dst_col '{dst_col}' not found in edge_csv")
        dst_col = inferred

    node_df[graph_id_col] = node_df[graph_id_col].map(normalize_id)
    node_df[node_id_col] = node_df[node_id_col].map(normalize_id)
    node_df = node_df.dropna(subset=[graph_id_col, node_id_col]).copy()

    edge_df[edge_graph_id_col] = edge_df[edge_graph_id_col].map(normalize_id)
    edge_df[src_col] = edge_df[src_col].map(normalize_id)
    edge_df[dst_col] = edge_df[dst_col].map(normalize_id)
    edge_df = edge_df.dropna(subset=[edge_graph_id_col, src_col, dst_col]).copy()

    # node features
    excluded_node_cols = {graph_id_col, node_id_col, "label", "Label"}
    raw_node_feature_cols = [c for c in node_df.columns if c not in excluded_node_cols]

    raw_node_feat_df = safe_numeric_df(node_df[raw_node_feature_cols]).replace([np.inf, -np.inf], np.nan)
    valid_numeric_cols = [c for c in raw_node_feat_df.columns if not raw_node_feat_df[c].isna().all()]

    if len(valid_numeric_cols) == 0:
        node_df["__const__"] = 1.0
        node_feature_cols = ["__const__"]
        node_feat_df = node_df[node_feature_cols].copy()
    else:
        node_feature_cols = valid_numeric_cols
        node_feat_df, node_feature_cols = stabilize_node_features(
            node_df=node_df,
            graph_id_col=graph_id_col,
            node_feature_cols=node_feature_cols,
        )

    node_df = node_df[[graph_id_col, node_id_col]].copy().join(node_feat_df)

    # edge features only for metadata
    excluded_edge_cols = {edge_graph_id_col, src_col, dst_col}
    edge_feature_cols = [c for c in edge_df.columns if c not in excluded_edge_cols]

    node_groups = dict(tuple(node_df.groupby(graph_id_col, sort=True)))
    edge_groups = dict(tuple(edge_df.groupby(edge_graph_id_col, sort=True)))

    common_graph_ids = sorted(set(node_groups.keys()) & set(label_map.keys()))

    graph_feature_list = []
    graph_labels = []
    graph_id_list = []
    graph_infos = []

    skipped_no_label = len(set(node_groups.keys()) - set(label_map.keys()))
    skipped_empty = 0
    dropped_edge_rows_total = 0

    for gidx, graph_id in enumerate(tqdm(common_graph_ids, desc="Building DeepWalk graphs")):
        g_nodes = node_groups[graph_id].copy()
        g_edges = edge_groups.get(graph_id, pd.DataFrame(columns=edge_df.columns)).copy()

        g_nodes = g_nodes.drop_duplicates(subset=[node_id_col]).reset_index(drop=True)
        if g_nodes.empty:
            skipped_empty += 1
            continue

        local_nodes = g_nodes[node_id_col].tolist()
        local_node_set = set(local_nodes)

        if undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        G.add_nodes_from(local_nodes)

        local_num_edges = 0
        if not g_edges.empty:
            for _, row in g_edges.iterrows():
                s = row[src_col]
                d = row[dst_col]

                if s not in local_node_set or d not in local_node_set:
                    dropped_edge_rows_total += 1
                    continue

                G.add_edge(s, d)
                local_num_edges += 1

        x_np = g_nodes[node_feature_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(x_np).all():
            bad = np.size(x_np) - np.isfinite(x_np).sum()
            raise ValueError(f"Non-finite node features detected in graph {graph_id}: {bad}")

        dw_emb_np, dw_coverage = fit_deepwalk_node_embeddings(
            G=G,
            local_nodes=local_nodes,
            dim=deepwalk_dim,
            walk_length=walk_length,
            num_workers=num_workers,
            window_size=window_size,
            epochs=dw_epochs,
            workers=dw_workers,
            seed=seed,
            max_walk_start_nodes=max_walk_start_nodes,
        )

        graph_feat = build_graph_feature_vector(
            node_feat_np=x_np,
            dw_emb_np=dw_emb_np,
            num_nodes=len(local_nodes),
            num_edges=int(G.number_of_edges()),
            dw_coverage=dw_coverage,
            use_node_feature_stats=use_node_feature_stats,
            use_graph_size_features=use_graph_size_features,
        )

        if not np.isfinite(graph_feat).all():
            bad = np.size(graph_feat) - np.isfinite(graph_feat).sum()
            raise ValueError(f"Non-finite graph feature vector detected in graph {graph_id}: {bad}")

        graph_feature_list.append(graph_feat)
        graph_labels.append(int(label_map[graph_id]))
        graph_id_list.append(graph_id)

        graph_infos.append({
            "graph_id": graph_id,
            "label": int(label_map[graph_id]),
            "num_nodes": int(len(local_nodes)),
            "num_edges": int(G.number_of_edges()),
            "dw_coverage": float(dw_coverage),
        })

    graph_summary_df = summarize_graphs(graph_infos)

    if len(graph_feature_list) == 0:
        raise RuntimeError("No graph features were built")

    X = np.vstack(graph_feature_list).astype(np.float32)
    y = np.asarray(graph_labels, dtype=np.float32)

    summary = {
        "loaded_graphs": len(graph_feature_list),
        "skipped_no_label": skipped_no_label,
        "skipped_empty": skipped_empty,
        "dropped_edge_rows": dropped_edge_rows_total,
        "feature_dim": int(X.shape[1]),
        "node_feature_dim": int(len(node_feature_cols)),
        "edge_feature_dim": int(len(edge_feature_cols)),
        "node_feature_cols": node_feature_cols,
        "edge_feature_cols": edge_feature_cols,
        "graph_summary_df": graph_summary_df,
        "graph_id_list": graph_id_list,
        "X": X,
        "y": y,
    }

    return summary


# ============================================================
# Split / scaling
# ============================================================

def split_dataset_indices(y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")

    idx_all = np.arange(len(y))
    labels = np.asarray(y, dtype=int)

    if len(idx_all) < 3:
        raise ValueError("Need at least 3 graphs for train/val/test split")

    use_stratify = len(np.unique(labels)) > 1 and np.min(np.bincount(labels)) >= 2

    try:
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels if use_stratify else None,
        )

        remain_size = 1.0 - test_ratio
        val_rel = val_ratio / remain_size

        labels_trainval = labels[idx_trainval]
        use_stratify_tv = len(np.unique(labels_trainval)) > 1 and np.min(np.bincount(labels_trainval)) >= 2

        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_rel,
            random_state=seed,
            stratify=labels_trainval if use_stratify_tv else None,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(idx_all)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        idx_train = shuffled[:n_train]
        idx_val = shuffled[n_train:n_train + n_val]
        idx_test = shuffled[n_train + n_val:]

    return idx_train, idx_val, idx_test


def fit_feature_scaler(X_train: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs.astype(np.float32)


# ============================================================
# Model
# ============================================================

class GraphMLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logit = self.classifier(h).view(-1)
        return logit, h


# ============================================================
# Train / Eval
# ============================================================

def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    y_true_all = []
    y_prob_all = []
    idx_all = []
    emb_all = []

    with torch.no_grad():
        for xb, yb, ib in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, emb = model(xb)
            probs = torch.sigmoid(logits)

            y_true_all.append(yb.detach().cpu().numpy())
            y_prob_all.append(probs.detach().cpu().numpy())
            idx_all.append(ib.detach().cpu().numpy())
            emb_all.append(emb.detach().cpu().numpy())

    if len(y_true_all) == 0:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": threshold,
            "y_true": np.array([]),
            "y_prob": np.array([]),
            "idx": np.array([]),
            "emb": np.empty((0, 0), dtype=np.float32),
        }

    y_true = np.concatenate(y_true_all).astype(int)
    y_prob = np.concatenate(y_prob_all)
    idx = np.concatenate(idx_all).astype(int)
    emb = np.concatenate(emb_all, axis=0)

    y_pred = (y_prob >= threshold).astype(int)

    auc = metric_or_nan(roc_auc_score, y_true, y_prob)
    ap = metric_or_nan(average_precision_score, y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "auc": auc,
        "ap": ap,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": threshold,
        "y_true": y_true,
        "y_prob": y_prob,
        "idx": idx,
        "emb": emb,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits, _ = model(xb)
        loss = criterion(logits, yb)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if len(losses) > 0 else float("nan")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = (
        torch.device("cuda")
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else torch.device("cpu")
    )

    artifact_dir = Path(args.artifacts_dir) / args.chain.lower()
    ensure_dir(artifact_dir)

    print(f"[Load] node_csv  -> {args.node_csv}")
    print(f"[Load] edge_csv  -> {args.edge_csv}")
    print(f"[Load] label_csv -> {args.label_csv}")

    label_map, inferred_label_key_col, _ = load_label_map(
        args.label_csv,
        label_col=args.label_col,
        label_key_col=args.label_key_col,
    )

    print("\n" + "=" * 60)
    print(f" graph_individual/main_deepwalk.py  |  Chain: {args.chain.upper()}")
    print("=" * 60)
    print(f"graph_id_col : {args.graph_id_col}")
    print(f"label_col    : {args.label_col}")
    print(f"label_key_col: {inferred_label_key_col}")
    print(f"node_id_col  : {args.node_id_col}")
    print(f"deepwalk_dim : {args.deepwalk_dim}")
    print("=" * 60)

    summary = build_graph_dataset_deepwalk(
        node_csv=args.node_csv,
        edge_csv=args.edge_csv,
        label_map=label_map,
        graph_id_col=args.graph_id_col,
        node_id_col=args.node_id_col,
        edge_graph_id_col=args.edge_graph_id_col,
        src_col=args.src_col,
        dst_col=args.dst_col,
        undirected=args.undirected,
        deepwalk_dim=args.deepwalk_dim,
        walk_length=args.walk_length,
        num_workers=args.num_workers,
        window_size=args.window_size,
        dw_epochs=args.dw_epochs,
        dw_workers=args.dw_workers,
        max_walk_start_nodes=args.max_walk_start_nodes,
        use_node_feature_stats=args.use_node_feature_stats,
        use_graph_size_features=args.use_graph_size_features,
        seed=args.seed,
    )

    X = summary["X"]
    y = summary["y"]
    graph_id_list = summary["graph_id_list"]
    graph_summary_df = summary["graph_summary_df"]

    loaded_graphs = summary["loaded_graphs"]
    feature_dim = summary["feature_dim"]

    if loaded_graphs == 0:
        raise RuntimeError("No graphs were built.")

    fraud_graphs = int(y.sum())
    node_count_mean = float(graph_summary_df["num_nodes"].mean()) if not graph_summary_df.empty else 0.0
    edge_count_mean = float(graph_summary_df["num_edges"].mean()) if not graph_summary_df.empty else 0.0
    singleton_graphs = int(graph_summary_df["singleton"].sum()) if not graph_summary_df.empty else 0
    zero_edge_graphs = int(graph_summary_df["zero_edge"].sum()) if not graph_summary_df.empty else 0
    mean_dw_coverage = float(graph_summary_df["dw_coverage"].mean()) if "dw_coverage" in graph_summary_df.columns else 0.0

    print("\n[Dataset Summary]")
    print(f"Loaded graphs              : {loaded_graphs}")
    print(f"Fraud graphs               : {fraud_graphs} / {loaded_graphs}")
    print(f"Node count mean            : {node_count_mean:.4f}")
    print(f"Edge count mean            : {edge_count_mean:.4f}")
    print(f"Singleton graphs           : {singleton_graphs} / {loaded_graphs}")
    print(f"Zero-edge after build      : {zero_edge_graphs} / {loaded_graphs}")
    print(f"Mean DeepWalk coverage     : {mean_dw_coverage:.4f}")
    print(f"Skipped no label           : {summary['skipped_no_label']}")
    print(f"Dropped edge rows          : {summary['dropped_edge_rows']}")
    print(f"Graph feature dim          : {feature_dim}")
    print(f"Node feature dim           : {summary['node_feature_dim']}")
    print(f"Edge feature dim           : {summary['edge_feature_dim']}")

    if args.fail_if_all_singleton and singleton_graphs == loaded_graphs:
        raise RuntimeError("All graphs are singleton. Aborting due to --fail_if_all_singleton")

    if args.save_graph_artifacts:
        graph_summary_path = artifact_dir / "graph_level1_deepwalk_summary.csv"
        graph_summary_df.to_csv(graph_summary_path, index=False)
        print(f"[Saved] graph summary -> {graph_summary_path}")

        meta_path = artifact_dir / f"{args.chain.lower()}_deepwalk_feature_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_feature_cols": summary["node_feature_cols"],
                "edge_feature_cols": summary["edge_feature_cols"],
                "graph_feature_dim": feature_dim,
                "node_feature_dim": summary["node_feature_dim"],
                "edge_feature_dim": summary["edge_feature_dim"],
                "deepwalk_dim": args.deepwalk_dim,
                "use_node_feature_stats": args.use_node_feature_stats,
                "use_graph_size_features": args.use_graph_size_features,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Saved] feature meta  -> {meta_path}")

    idx_train, idx_val, idx_test = split_dataset_indices(
        y=y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]

    mean, std = fit_feature_scaler(X_train)

    X_train = apply_feature_scaler(X_train, mean, std)
    X_val = apply_feature_scaler(X_val, mean, std)
    X_test = apply_feature_scaler(X_test, mean, std)
    X_all = apply_feature_scaler(X, mean, std)

    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    if not np.isfinite(X_all).all():
        bad = np.size(X_all) - np.isfinite(X_all).sum()
        raise ValueError(f"Non-finite scaled graph features detected: {bad}")

    train_tensor = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(idx_train, dtype=torch.long),
    )
    val_tensor = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(idx_val, dtype=torch.long),
    )
    test_tensor = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(idx_test, dtype=torch.long),
    )
    all_tensor = TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(np.arange(len(y)), dtype=torch.long),
    )

    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=args.batch_size, shuffle=False)
    all_loader = DataLoader(all_tensor, batch_size=args.batch_size, shuffle=False)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))

    print("\n[Train Config]")
    print(f"device      : {device}")
    print(f"train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    print(f"pos_weight  : {pos_weight:.4f}")
    print(f"feature_dim : {feature_dim}")

    model = GraphMLPClassifier(
        in_dim=feature_dim,
        hidden_dim=args.mlp_hidden,
        out_dim=args.mlp_out,
        dropout=args.mlp_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    best_state = None
    best_val_auc = -1.0
    best_val_thr = 0.5

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_raw = evaluate(model, val_loader, device, threshold=0.5)
        val_thr, _ = compute_best_threshold(val_raw["y_true"], val_raw["y_prob"])
        val_eval = evaluate(model, val_loader, device, threshold=val_thr)

        val_auc = val_eval["auc"]
        val_ap = val_eval["ap"]
        val_f1 = val_eval["f1"]

        improved = False
        if np.isnan(best_val_auc) and not np.isnan(val_auc):
            improved = True
        elif not np.isnan(val_auc) and val_auc > best_val_auc:
            improved = True

        if improved:
            best_val_auc = val_auc
            best_val_thr = val_thr
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_f1": val_f1,
                "val_threshold": val_thr,
            }

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_ap={val_ap:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"thr={val_thr:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_eval = evaluate(model, test_loader, device, threshold=best_val_thr)

    print("\n[Test Result]")
    print(f"ROC-AUC : {test_eval['auc']:.4f}")
    print(f"AP      : {test_eval['ap']:.4f}")
    print(f"F1      : {test_eval['f1']:.4f}")
    print(f"Threshold: {best_val_thr:.2f}")

    model_path = artifact_dir / f"{args.chain.lower()}_graph_individual_deepwalk_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "best_val_auc": best_val_auc,
        "best_val_threshold": best_val_thr,
        "feature_dim": feature_dim,
        "node_feature_cols": summary["node_feature_cols"],
        "edge_feature_cols": summary["edge_feature_cols"],
        "scaler_mean": mean,
        "scaler_std": std,
        "graph_id_list": graph_id_list,
    }, model_path)
    print(f"[Saved] model -> {model_path}")

    all_eval = evaluate(model, all_loader, device, threshold=best_val_thr)
    emb = all_eval["emb"]
    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])] if emb.size > 0 else []

    split_map = {}
    for i in idx_train:
        split_map[int(i)] = "train"
    for i in idx_val:
        split_map[int(i)] = "val"
    for i in idx_test:
        split_map[int(i)] = "test"

    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "gidx", all_eval["idx"])
    emb_df["graph_id"] = emb_df["gidx"].map(lambda x: graph_id_list[int(x)])
    emb_df["label"] = all_eval["y_true"]
    emb_df["prob"] = all_eval["y_prob"]
    emb_df["pred"] = (all_eval["y_prob"] >= best_val_thr).astype(int)
    emb_df["split"] = emb_df["gidx"].map(lambda x: split_map.get(int(x), "unknown"))

    emb_out = artifact_dir / f"{args.chain.lower()}_graph_embeddings_deepwalk.csv"
    emb_df.to_csv(emb_out, index=False)
    print(f"[Saved] graph embeddings -> {emb_out}")


if __name__ == "__main__":
    main()
