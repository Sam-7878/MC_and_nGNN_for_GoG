#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_individual/main.py

Level-1 graph classification for contract-level intra-graphs.

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
python ./graph_individual/main.py \
  --chain bsc \
  --node_csv ./_data/level1/bsc_level1_nodes.csv \
  --edge_csv ./_data/level1/bsc_level1_edges.csv \
  --label_csv ./_data/dataset/features/bsc_basic_metrics_processed.csv \
  --graph_id_col graph_id \
  --node_id_col node_id \
  --label_col label \
  --edge_mode precomputed \
  --gnn_hidden 64 \
  --gnn_out 32 \
  --gnn_epochs 50 \
  --undirected \
  --save_graph_artifacts
"""

import os
import sys
import json
import math
import random
import argparse
import warnings
from pathlib import Path

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

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm

warnings.filterwarnings("ignore")


# ============================================================
# Utils
# ============================================================

LABEL_KEY_CANDIDATES = ["Contract", "contract", "contract_address", "address", "graph_id"]
EDGE_GRAPH_ID_CANDIDATES = ["graph_id", "GraphID", "graph", "contract", "Contract"]
EDGE_SRC_CANDIDATES = ["src", "source", "from", "from_address", "sender"]
EDGE_DST_CANDIDATES = ["dst", "target", "to", "to_address", "receiver"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_id(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x == "" or x == "nan" or x == "none" or x == "null":
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


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def metric_or_nan(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def compute_best_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def summarize_graphs(graph_infos: list) -> pd.DataFrame:
    df = pd.DataFrame(graph_infos)
    if df.empty:
        return df
    df["singleton"] = (df["num_nodes"] <= 1).astype(int)
    df["zero_edge"] = (df["num_edges"] <= 0).astype(int)
    return df


## Feature Engineering Functions
def signed_log1p_np(x):
    return np.sign(x) * np.log1p(np.abs(x))


def stabilize_node_features(node_df: pd.DataFrame, graph_id_col: str, node_feature_cols: list) -> tuple[pd.DataFrame, list]:
    feat = safe_numeric_df(node_df[node_feature_cols]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cols_lower = {c: c.lower() for c in feat.columns}

    value_cols = [c for c in feat.columns if "value" in cols_lower[c]]
    count_cols = [c for c in feat.columns if ("count" in cols_lower[c] or "degree" in cols_lower[c])]
    duration_cols = [c for c in feat.columns if "duration" in cols_lower[c]]
    ratio_cols = [c for c in feat.columns if "ratio" in cols_lower[c]]
    ts_cols = [c for c in feat.columns if cols_lower[c].endswith("_ts") or "timestamp" in cols_lower[c]]

    # 1) wei -> ether + signed log1p
    for c in value_cols:
        feat[c] = feat[c] / 1e18
        feat[c] = signed_log1p_np(feat[c].to_numpy(dtype=np.float64))

    # 2) counts / degree / duration -> log1p
    for c in count_cols + duration_cols:
        if c in feat.columns:
            x = feat[c].to_numpy(dtype=np.float64)
            x = np.clip(x, a_min=0.0, a_max=None)
            feat[c] = np.log1p(x)

    # 3) timestamp는 graph 내 상대시간으로 변환
    if len(ts_cols) > 0:
        tmp = pd.concat([node_df[[graph_id_col]].reset_index(drop=True), feat[ts_cols].reset_index(drop=True)], axis=1)

        for gid, idx in tmp.groupby(graph_id_col).groups.items():
            sub = tmp.loc[idx, ts_cols].replace(0, np.nan)
            if sub.notna().any().any():
                base_ts = np.nanmin(sub.to_numpy(dtype=np.float64))
                tmp.loc[idx, ts_cols] = tmp.loc[idx, ts_cols] - base_ts

        feat[ts_cols] = tmp[ts_cols].fillna(0.0)

    # 4) ratio는 범위 clip
    for c in ratio_cols:
        feat[c] = feat[c].clip(-10.0, 10.0)

    # 5) 마지막 방어: extreme clip
    feat = feat.clip(lower=-50.0, upper=50.0)

    # 6) finite 보장
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feat, feat.columns.tolist()

# ============================================================
# Parsing args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Level-1 graph classification using precomputed node/edge CSVs"
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
    parser.add_argument("--label_key_col", type=str, default=None,
                        help="Optional explicit graph key column in label_csv")
    parser.add_argument("--edge_graph_id_col", type=str, default="graph_id")
    parser.add_argument("--src_col", type=str, default="src")
    parser.add_argument("--dst_col", type=str, default="dst")

    parser.add_argument("--edge_mode", type=str, default="precomputed",
                        choices=["precomputed"],
                        help="This updated script expects precomputed edges")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--fail_if_all_singleton", action="store_true")

    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_out", type=int, default=32)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--gnn_epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--save_graph_artifacts", action="store_true")
    parser.add_argument("--artifacts_dir", type=str, default="graph_individual_artifacts")

    return parser.parse_args()


# ============================================================
# Label loading
# ============================================================

def load_label_map(label_csv: str, label_col: str, label_key_col: str | None = None):
    df = pd.read_csv(label_csv, low_memory=False)

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in label_csv")

    if label_key_col is None:
        label_key_col = infer_col(df.columns, LABEL_KEY_CANDIDATES)

    if label_key_col is None:
        raise ValueError(
            f"Could not infer label key column from label_csv. "
            f"Tried: {LABEL_KEY_CANDIDATES}"
        )

    work = df[[label_key_col, label_col]].copy()
    work[label_key_col] = work[label_key_col].map(normalize_id)
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")

    work = work.dropna(subset=[label_key_col, label_col]).copy()
    work[label_col] = work[label_col].astype(int)

    label_map = dict(zip(work[label_key_col], work[label_col]))
    return label_map, label_key_col, len(work)


# ============================================================
# Graph building
# ============================================================

def build_graph_dataset(
    node_csv: str,
    edge_csv: str,
    label_map: dict,
    graph_id_col: str = "graph_id",
    node_id_col: str = "node_id",
    edge_graph_id_col: str = "graph_id",
    src_col: str = "src",
    dst_col: str = "dst",
    undirected: bool = False,
):
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

    # Node feature columns
    excluded_node_cols = {graph_id_col, node_id_col, "label", "Label"}
    node_feature_cols = [c for c in node_df.columns if c not in excluded_node_cols]

    # node_feat_df = safe_numeric_df(node_df[node_feature_cols]).fillna(0.0)
    # valid_numeric_cols = [c for c in node_feat_df.columns if not node_feat_df[c].isna().all()]
    ## 첫 번째 방어선: numeric으로 변환 후 infinite 제거. 이후 valid_numeric_cols 기준으로 안정화 또는 상수화 진행
    raw_node_feat_df = safe_numeric_df(node_df[node_feature_cols]).replace([np.inf, -np.inf], np.nan)
    valid_numeric_cols = [c for c in raw_node_feat_df.columns if not raw_node_feat_df[c].isna().all()]

    if len(valid_numeric_cols) == 0:
        node_df["__const__"] = 1.0
        node_feature_cols = ["__const__"]
        node_feat_df = node_df[node_feature_cols].copy()
    else:
        node_feature_cols = valid_numeric_cols
        stable_feat_df, node_feature_cols = stabilize_node_features(
            node_df=node_df,
            graph_id_col=graph_id_col,
            node_feature_cols=node_feature_cols,
        )
        node_feat_df = stable_feat_df[node_feature_cols]

    node_df = node_df[[graph_id_col, node_id_col]].copy().join(node_feat_df)




    if len(valid_numeric_cols) == 0:
        node_df["__const__"] = 1.0
        node_feature_cols = ["__const__"]
        node_feat_df = node_df[node_feature_cols].copy()
    else:
        node_feature_cols = valid_numeric_cols
        node_feat_df = node_feat_df[node_feature_cols]

    node_df = node_df[[graph_id_col, node_id_col]].copy().join(node_feat_df)

    # Edge feature columns
    excluded_edge_cols = {edge_graph_id_col, src_col, dst_col}
    edge_feature_cols = [c for c in edge_df.columns if c not in excluded_edge_cols]
    if len(edge_feature_cols) > 0:
        edge_attr_df = safe_numeric_df(edge_df[edge_feature_cols]).fillna(0.0)
        edge_feature_cols = [c for c in edge_attr_df.columns if not edge_attr_df[c].isna().all()]
        if len(edge_feature_cols) > 0:
            edge_df = edge_df[[edge_graph_id_col, src_col, dst_col]].copy().join(edge_attr_df[edge_feature_cols])
        else:
            edge_feature_cols = []
            edge_df = edge_df[[edge_graph_id_col, src_col, dst_col]].copy()
    else:
        edge_feature_cols = []
        edge_df = edge_df[[edge_graph_id_col, src_col, dst_col]].copy()

    # Group by graph_id
    node_groups = dict(tuple(node_df.groupby(graph_id_col, sort=True)))
    edge_groups = dict(tuple(edge_df.groupby(edge_graph_id_col, sort=True)))

    common_graph_ids = sorted(set(node_groups.keys()) & set(label_map.keys()))

    dataset = []
    graph_infos = []
    graph_id_list = []

    skipped_no_label = len(set(node_groups.keys()) - set(label_map.keys()))
    skipped_empty = 0
    dropped_edge_rows_total = 0

    for gidx, graph_id in enumerate(tqdm(common_graph_ids, desc="Building graphs")):
        g_nodes = node_groups[graph_id].copy()
        g_edges = edge_groups.get(graph_id, pd.DataFrame(columns=edge_df.columns)).copy()

        g_nodes = g_nodes.drop_duplicates(subset=[node_id_col]).reset_index(drop=True)
        if g_nodes.empty:
            skipped_empty += 1
            continue

        local_nodes = g_nodes[node_id_col].tolist()
        nid2idx = {nid: i for i, nid in enumerate(local_nodes)}

        # x = g_nodes[node_feature_cols].to_numpy(dtype=np.float32)
        # x = torch.tensor(x, dtype=torch.float)
        ## 두 번째 방어선: node_feature_cols 기준으로 numeric 변환 후 finite 보장. 이후 tensor 변환
        x_np = g_nodes[node_feature_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(x_np).all():
            bad = np.size(x_np) - np.isfinite(x_np).sum()
            raise ValueError(f"Non-finite node features detected in graph {graph_id}: {bad}")

        x = torch.tensor(x_np, dtype=torch.float)

        edge_pairs = []
        edge_attrs = []

        if not g_edges.empty:
            attr_cols = [c for c in g_edges.columns if c not in [edge_graph_id_col, src_col, dst_col]]

            for _, row in g_edges.iterrows():
                s = row[src_col]
                d = row[dst_col]

                if s not in nid2idx or d not in nid2idx:
                    dropped_edge_rows_total += 1
                    continue

                si = nid2idx[s]
                di = nid2idx[d]

                edge_pairs.append([si, di])

                if len(attr_cols) > 0:
                    edge_attrs.append([float(row[c]) if pd.notna(row[c]) else 0.0 for c in attr_cols])

                if undirected and si != di:
                    edge_pairs.append([di, si])
                    if len(attr_cols) > 0:
                        edge_attrs.append([float(row[c]) if pd.notna(row[c]) else 0.0 for c in attr_cols])

        if len(edge_pairs) > 0:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if len(edge_attrs) > 0:
            edge_attr = torch.tensor(np.asarray(edge_attrs, dtype=np.float32), dtype=torch.float)
        else:
            edge_attr = None

        y = torch.tensor([float(label_map[graph_id])], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.gidx = torch.tensor([gidx], dtype=torch.long)
        if edge_attr is not None:
            data.edge_attr = edge_attr

        dataset.append(data)
        graph_id_list.append(graph_id)

        graph_infos.append({
            "graph_id": graph_id,
            "label": int(label_map[graph_id]),
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
        })

    graph_summary_df = summarize_graphs(graph_infos)

    summary = {
        "loaded_graphs": len(dataset),
        "skipped_no_label": skipped_no_label,
        "skipped_empty": skipped_empty,
        "dropped_edge_rows": dropped_edge_rows_total,
        "feature_dim": int(dataset[0].x.size(1)) if len(dataset) > 0 else 0,
        "edge_feature_dim": int(dataset[0].edge_attr.size(1)) if (len(dataset) > 0 and hasattr(dataset[0], "edge_attr")) else 0,
        "node_feature_cols": node_feature_cols,
        "edge_feature_cols": edge_feature_cols,
        "graph_summary_df": graph_summary_df,
        "graph_id_list": graph_id_list,
    }

    return dataset, summary


# ============================================================
# Data split & scaling
# ============================================================

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")

    labels = np.array([int(d.y.item()) for d in dataset])
    idx_all = np.arange(len(dataset))

    if len(dataset) < 3:
        raise ValueError("Need at least 3 graphs for train/val/test split")

    test_size = test_ratio
    remain_size = 1.0 - test_size

    use_stratify = len(np.unique(labels)) > 1 and np.min(np.bincount(labels)) >= 2

    try:
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=seed,
            stratify=labels if use_stratify else None,
        )

        labels_trainval = labels[idx_trainval]
        val_rel = val_ratio / remain_size

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

    train_set = [dataset[i] for i in idx_train]
    val_set = [dataset[i] for i in idx_val]
    test_set = [dataset[i] for i in idx_test]

    return train_set, val_set, test_set, idx_train, idx_val, idx_test


def fit_node_scaler(graphs):
    xs = [g.x.cpu().numpy() for g in graphs if g.x is not None and g.x.numel() > 0]
    if len(xs) == 0:
        return None, None
    X = np.vstack(xs)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# def apply_node_scaler(graphs, mean, std):
#     if mean is None or std is None:
#         return
#     mean_t = torch.tensor(mean, dtype=torch.float)
#     std_t = torch.tensor(std, dtype=torch.float)
#     for g in graphs:
#         g.x = (g.x - mean_t) / std_t
def apply_node_scaler(graphs, mean, std):
    if mean is None or std is None:
        return

    mean_t = torch.tensor(mean, dtype=torch.float)
    std_t = torch.tensor(std, dtype=torch.float)

    for g in graphs:
        g.x = (g.x - mean_t) / std_t
        g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Model
# ============================================================

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, num_layers=3, dropout=0.2):
        super().__init__()

        assert num_layers >= 2, "gnn_layers must be >= 2"

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.norms.append(BatchNorm(out_dim))

        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def encode(self, x, edge_index, batch):
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        x = F.relu(x)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        g = torch.cat([mean_pool, max_pool], dim=1)
        return g

    def forward(self, data):
        g = self.encode(data.x, data.edge_index, data.batch)
        logit = self.classifier(g).view(-1)
        return logit, g


# ============================================================
# Train / Eval
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    y_true_all = []
    y_prob_all = []
    gidx_all = []
    emb_all = []

    for batch in loader:
        batch = batch.to(device)
        logits, emb = model(batch)
        probs = torch.sigmoid(logits)

        y_true_all.append(batch.y.view(-1).detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())
        gidx_all.append(batch.gidx.view(-1).detach().cpu().numpy())
        emb_all.append(emb.detach().cpu().numpy())

    if len(y_true_all) == 0:
        return {
            "loss": float("nan"),
            "auc": float("nan"),
            "ap": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": threshold,
            "y_true": np.array([]),
            "y_prob": np.array([]),
            "gidx": np.array([]),
            "emb": np.empty((0, 0), dtype=np.float32),
        }

    y_true = np.concatenate(y_true_all).astype(int)
    y_prob = np.concatenate(y_prob_all)
    gidx = np.concatenate(gidx_all).astype(int)
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
        "gidx": gidx,
        "emb": emb,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, _ = model(batch)
        loss = criterion(logits, batch.y.view(-1))

        ## 세 번째 방어선: loss가 non-finite한 경우 해당 배치 스킵
        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue

        loss.backward()
        ## 네 번째 방어선: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")



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

    label_map, inferred_label_key_col, n_labels = load_label_map(
        args.label_csv,
        label_col=args.label_col,
        label_key_col=args.label_key_col
    )

    print("\n" + "=" * 60)
    print(f" graph_individual/main.py  |  Chain: {args.chain.upper()}")
    print("=" * 60)
    print(f"graph_id_col : {args.graph_id_col}")
    print(f"label_col    : {args.label_col}")
    print(f"label_key_col: {inferred_label_key_col}")
    print(f"node_id_col  : {args.node_id_col}")
    print(f"edge_mode    : {args.edge_mode}")
    print("=" * 60)

    dataset, summary = build_graph_dataset(
        node_csv=args.node_csv,
        edge_csv=args.edge_csv,
        label_map=label_map,
        graph_id_col=args.graph_id_col,
        node_id_col=args.node_id_col,
        edge_graph_id_col=args.edge_graph_id_col,
        src_col=args.src_col,
        dst_col=args.dst_col,
        undirected=args.undirected,
    )

    if len(dataset) == 0:
        raise RuntimeError("No graphs were built. Check node_csv / edge_csv / label_csv key alignment.")

    graph_summary_df = summary["graph_summary_df"]
    loaded_graphs = summary["loaded_graphs"]
    feature_dim = summary["feature_dim"]
    edge_feature_dim = summary["edge_feature_dim"]
    graph_id_list = summary["graph_id_list"]

    fraud_graphs = int(sum(int(d.y.item()) for d in dataset))
    node_count_mean = float(graph_summary_df["num_nodes"].mean()) if not graph_summary_df.empty else 0.0
    edge_count_mean = float(graph_summary_df["num_edges"].mean()) if not graph_summary_df.empty else 0.0
    singleton_graphs = int(graph_summary_df["singleton"].sum()) if not graph_summary_df.empty else 0
    zero_edge_graphs = int(graph_summary_df["zero_edge"].sum()) if not graph_summary_df.empty else 0

    print("\n[Dataset Summary]")
    print(f"Loaded graphs              : {loaded_graphs}")
    print(f"Fraud graphs               : {fraud_graphs} / {loaded_graphs}")
    print(f"Node count mean            : {node_count_mean:.4f}")
    print(f"Edge count mean            : {edge_count_mean:.4f}")
    print(f"Singleton graphs           : {singleton_graphs} / {loaded_graphs}")
    print(f"Zero-edge after build      : {zero_edge_graphs} / {loaded_graphs}")
    print(f"Skipped no label           : {summary['skipped_no_label']}")
    print(f"Dropped edge rows          : {summary['dropped_edge_rows']}")
    print(f"Node feature dim           : {feature_dim}")
    print(f"Edge feature dim           : {edge_feature_dim}")

    if args.fail_if_all_singleton and singleton_graphs == loaded_graphs:
        raise RuntimeError("All graphs are singleton. Aborting due to --fail_if_all_singleton")

    if args.save_graph_artifacts:
        graph_summary_path = artifact_dir / "graph_level1_summary.csv"
        graph_summary_df.to_csv(graph_summary_path, index=False)
        print(f"[Saved] graph summary -> {graph_summary_path}")

        meta_path = artifact_dir / f"{args.chain.lower()}_feature_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_feature_cols": summary["node_feature_cols"],
                "edge_feature_cols": summary["edge_feature_cols"],
                "feature_dim": feature_dim,
                "edge_feature_dim": edge_feature_dim,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Saved] feature meta  -> {meta_path}")

    train_set, val_set, test_set, idx_train, idx_val, idx_test = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    mean, std = fit_node_scaler(train_set)
    apply_node_scaler(train_set, mean, std)
    apply_node_scaler(val_set, mean, std)
    apply_node_scaler(test_set, mean, std)

    y_train = np.array([int(d.y.item()) for d in train_set])
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))

    print("\n[Train Config]")
    print(f"device      : {device}")
    print(f"train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"pos_weight  : {pos_weight:.4f}")
    print(f"feature_dim : {feature_dim}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    all_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GraphClassifier(
        in_dim=feature_dim,
        hidden_dim=args.gnn_hidden,
        out_dim=args.gnn_out,
        num_layers=args.gnn_layers,
        dropout=args.gnn_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device)
    )

    best_state = None
    best_val_auc = -1.0
    best_val_thr = 0.5

    for epoch in range(1, args.gnn_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_eval_raw = evaluate(model, val_loader, device, threshold=0.5)
        val_thr, _ = compute_best_threshold(val_eval_raw["y_true"], val_eval_raw["y_prob"])
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

    model_path = artifact_dir / f"{args.chain.lower()}_graph_individual_model.pt"
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

    # Save graph embeddings for all graphs
    all_eval = evaluate(model, all_loader, device, threshold=best_val_thr)

    split_map = {}
    for i in idx_train:
        split_map[int(dataset[i].gidx.item())] = "train"
    for i in idx_val:
        split_map[int(dataset[i].gidx.item())] = "val"
    for i in idx_test:
        split_map[int(dataset[i].gidx.item())] = "test"

    emb = all_eval["emb"]
    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])] if emb.size > 0 else []

    emb_df = pd.DataFrame(emb, columns=emb_cols)
    emb_df.insert(0, "gidx", all_eval["gidx"])
    emb_df["graph_id"] = emb_df["gidx"].map(lambda x: graph_id_list[int(x)])
    emb_df["label"] = all_eval["y_true"]
    emb_df["prob"] = all_eval["y_prob"]
    emb_df["pred"] = (all_eval["y_prob"] >= best_val_thr).astype(int)
    emb_df["split"] = emb_df["gidx"].map(lambda x: split_map.get(int(x), "unknown"))

    emb_out = artifact_dir / f"{args.chain.lower()}_graph_embeddings.csv"
    emb_df.to_csv(emb_out, index=False)
    print(f"[Saved] graph embeddings -> {emb_out}")


if __name__ == "__main__":
    main()
