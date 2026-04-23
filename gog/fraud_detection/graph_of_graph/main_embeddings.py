#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_of_graph/main_embeddings.py

Level-2 Graph-of-Graphs (GoG) node classification.

Input
-----
1) embedding_csv
   Output from graph_individual/main.py
   Expected columns:
     - graph_id
     - label (optional if label_csv is given)
     - split (optional)
     - emb_0, emb_1, ...

2) edge_csv
   Output from analysis/common_node.py
   Expected columns:
     - Contract1
     - Contract2
     - Common_Nodes
     - Unique_Addresses
   Optional:
     - Label1
     - Label2

3) label_csv (optional)
   Used only to fill or override labels when needed.

Example
-------
python ./graph_of_graph/main.py \
  --chain bsc \
  --embedding_csv ./graph_individual_artifacts/bsc/bsc_graph_embeddings.csv \
  --edge_csv ./_data/GoG/edges/bsc_common_nodes_except_null_labels.csv \
  --label_csv ./_data/dataset/features/bsc_basic_metrics_processed.csv \
  --gnn_hidden 64 \
  --gnn_out 32 \
  --gnn_layers 3 \
  --gnn_epochs 100 \
  --save_artifacts
"""

import os
import sys
import json
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
from torch_geometric.nn import GCNConv, BatchNorm

warnings.filterwarnings("ignore")


# ============================================================
# Utils
# ============================================================

LABEL_KEY_CANDIDATES = ["Contract", "contract", "contract_address", "address", "graph_id"]
EMBED_KEY_CANDIDATES = ["graph_id", "Contract", "contract", "contract_address", "address"]
EDGE_U_CANDIDATES = ["Contract1", "contract1", "src", "source", "u"]
EDGE_V_CANDIDATES = ["Contract2", "contract2", "dst", "target", "v"]


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


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Level-2 GoG node classification using Level-1 graph embeddings"
    )

    parser.add_argument("--chain", type=str, required=True)

    parser.add_argument("--embedding_csv", type=str, required=True,
                        help="graph_individual output embeddings csv")
    parser.add_argument("--edge_csv", type=str, required=True,
                        help="common_node.py Level2 edge csv")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="optional label csv for fill/override")

    parser.add_argument("--embedding_key_col", type=str, default=None,
                        help="graph key column in embedding_csv. auto if omitted")
    parser.add_argument("--label_key_col", type=str, default=None,
                        help="graph key column in label_csv. auto if omitted")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--edge_u_col", type=str, default=None,
                        help="edge source column in edge_csv. auto if omitted")
    parser.add_argument("--edge_v_col", type=str, default=None,
                        help="edge target column in edge_csv. auto if omitted")

    parser.add_argument("--edge_weight_mode", type=str, default="log_common",
                        choices=["none", "common", "log_common", "jaccard"],
                        help="how to build GoG edge weights")
    parser.add_argument("--undirected", action="store_true",
                        help="duplicate reverse edges")
    parser.add_argument("--ignore_level1_split", action="store_true",
                        help="ignore split column from embedding_csv and re-split")
    parser.add_argument("--fail_if_no_edges", action="store_true")

    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--gnn_out", type=int, default=32)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--gnn_epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])

    parser.add_argument("--save_artifacts", action="store_true")
    parser.add_argument("--artifacts_dir", type=str, default="graph_of_graph_artifacts")

    return parser.parse_args()


# ============================================================
# Load label map
# ============================================================

def load_label_map(label_csv: str, label_col: str = "label", label_key_col: str | None = None):
    df = pd.read_csv(label_csv, low_memory=False)

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in label_csv")

    if label_key_col is None:
        label_key_col = infer_col(df.columns, LABEL_KEY_CANDIDATES)

    if label_key_col is None:
        raise ValueError(f"Could not infer label key column from label_csv: {LABEL_KEY_CANDIDATES}")

    work = df[[label_key_col, label_col]].copy()
    work[label_key_col] = work[label_key_col].map(normalize_id)
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[label_key_col, label_col]).copy()
    work[label_col] = work[label_col].astype(int)

    label_map = dict(zip(work[label_key_col], work[label_col]))
    return label_map, label_key_col


# ============================================================
# Load embeddings
# ============================================================

def load_embedding_nodes(embedding_csv: str, embedding_key_col: str | None, label_csv: str | None, label_col: str):
    emb_df = pd.read_csv(embedding_csv, low_memory=False)

    if embedding_key_col is None:
        embedding_key_col = infer_col(emb_df.columns, EMBED_KEY_CANDIDATES)

    if embedding_key_col is None:
        raise ValueError(f"Could not infer embedding key column from embedding_csv: {EMBED_KEY_CANDIDATES}")

    emb_df[embedding_key_col] = emb_df[embedding_key_col].map(normalize_id)
    emb_df = emb_df.dropna(subset=[embedding_key_col]).copy()

    # embedding feature columns
    emb_cols = [c for c in emb_df.columns if str(c).startswith("emb_")]
    if len(emb_cols) == 0:
        excluded = {embedding_key_col, "label", "prob", "pred", "split", "gidx"}
        numeric_df = safe_numeric_df(emb_df[[c for c in emb_df.columns if c not in excluded]])
        emb_cols = [c for c in numeric_df.columns if not numeric_df[c].isna().all()]
        emb_df = emb_df[[embedding_key_col] + [c for c in emb_df.columns if c not in excluded]].copy()
        emb_df = emb_df[[embedding_key_col]].join(numeric_df[emb_cols])
    else:
        emb_df = emb_df[[embedding_key_col] + [c for c in emb_df.columns if c in emb_cols or c in {"label", "split", "prob", "pred", "gidx"}]].copy()

    if len(emb_cols) == 0:
        raise ValueError("No embedding feature columns found in embedding_csv")

    # label from embedding csv if present
    if "label" in emb_df.columns:
        emb_df["label"] = pd.to_numeric(emb_df["label"], errors="coerce")
    else:
        emb_df["label"] = np.nan

    # fill/override labels from label_csv if needed
    inferred_label_key_col = None
    if label_csv is not None and Path(label_csv).exists():
        label_map, inferred_label_key_col = load_label_map(label_csv, label_col=label_col, label_key_col=None)
        missing_mask = emb_df["label"].isna()
        emb_df.loc[missing_mask, "label"] = emb_df.loc[missing_mask, embedding_key_col].map(label_map)

    emb_df = emb_df.dropna(subset=["label"]).copy()
    emb_df["label"] = emb_df["label"].astype(int)

    # split
    if "split" in emb_df.columns:
        emb_df["split"] = emb_df["split"].astype(str).str.strip().str.lower()
    else:
        emb_df["split"] = None

    # drop duplicate graph_id
    emb_df = emb_df.drop_duplicates(subset=[embedding_key_col], keep="first").reset_index(drop=True)

    return emb_df, embedding_key_col, emb_cols, inferred_label_key_col


# ============================================================
# Build GoG graph
# ============================================================

def build_edge_weight(row, mode: str):
    common = pd.to_numeric(row.get("Common_Nodes", 1), errors="coerce")
    unique = pd.to_numeric(row.get("Unique_Addresses", 1), errors="coerce")

    common = 1.0 if pd.isna(common) else float(common)
    unique = 1.0 if pd.isna(unique) or float(unique) <= 0 else float(unique)

    if mode == "none":
        return 1.0
    if mode == "common":
        return common
    if mode == "log_common":
        return float(np.log1p(max(common, 0.0)))
    if mode == "jaccard":
        return float(common / unique)
    return 1.0


def build_masks(labels: np.ndarray, split_series: pd.Series | None, ignore_level1_split: bool,
                train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    n = len(labels)

    if split_series is not None and not ignore_level1_split:
        s = split_series.fillna("").astype(str).str.lower()
        train_mask = (s == "train").to_numpy()
        val_mask = (s == "val").to_numpy()
        test_mask = (s == "test").to_numpy()

        if train_mask.sum() > 0 and val_mask.sum() > 0 and test_mask.sum() > 0:
            return train_mask, val_mask, test_mask, "level1_split"

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")

    idx_all = np.arange(n)
    use_stratify = len(np.unique(labels)) > 1 and np.min(np.bincount(labels)) >= 2

    try:
        idx_trainval, idx_test = train_test_split(
            idx_all,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels if use_stratify else None,
        )

        remain = 1.0 - test_ratio
        val_rel = val_ratio / remain
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
        perm = rng.permutation(idx_all)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return train_mask, val_mask, test_mask, "resplit"


def build_gog_data(
    embedding_csv: str,
    edge_csv: str,
    label_csv: str | None,
    label_col: str,
    embedding_key_col: str | None,
    edge_u_col: str | None,
    edge_v_col: str | None,
    edge_weight_mode: str,
    undirected: bool,
    ignore_level1_split: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    node_df, inferred_embed_key_col, emb_cols, inferred_label_key_col = load_embedding_nodes(
        embedding_csv=embedding_csv,
        embedding_key_col=embedding_key_col,
        label_csv=label_csv,
        label_col=label_col,
    )

    edge_df = pd.read_csv(edge_csv, low_memory=False)

    if edge_u_col is None:
        edge_u_col = infer_col(edge_df.columns, EDGE_U_CANDIDATES)
    if edge_v_col is None:
        edge_v_col = infer_col(edge_df.columns, EDGE_V_CANDIDATES)

    if edge_u_col is None or edge_v_col is None:
        raise ValueError("Could not infer edge endpoint columns from edge_csv")

    edge_df[edge_u_col] = edge_df[edge_u_col].map(normalize_id)
    edge_df[edge_v_col] = edge_df[edge_v_col].map(normalize_id)
    edge_df = edge_df.dropna(subset=[edge_u_col, edge_v_col]).copy()

    # node index
    graph_ids = node_df[inferred_embed_key_col].tolist()
    gid2idx = {gid: i for i, gid in enumerate(graph_ids)}

    x = torch.tensor(node_df[emb_cols].to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor(node_df["label"].to_numpy(dtype=np.float32), dtype=torch.float)

    # masks
    split_series = node_df["split"] if "split" in node_df.columns else None
    train_mask, val_mask, test_mask, split_source = build_masks(
        labels=node_df["label"].to_numpy(dtype=int),
        split_series=split_series,
        ignore_level1_split=ignore_level1_split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # edges
    edge_pairs = []
    edge_weights = []
    dropped_edges = 0

    for _, row in edge_df.iterrows():
        u = row[edge_u_col]
        v = row[edge_v_col]

        if u not in gid2idx or v not in gid2idx:
            dropped_edges += 1
            continue

        ui = gid2idx[u]
        vi = gid2idx[v]
        w = build_edge_weight(row, edge_weight_mode)

        edge_pairs.append([ui, vi])
        edge_weights.append(w)

        if undirected and ui != vi:
            edge_pairs.append([vi, ui])
            edge_weights.append(w)

    if len(edge_pairs) > 0:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(np.asarray(edge_weights, dtype=np.float32), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = None

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
    )

    if edge_weight is not None:
        data.edge_weight = edge_weight

    summary = {
        "num_nodes": int(x.size(0)),
        "num_edges": int(edge_index.size(1)),
        "feature_dim": int(x.size(1)),
        "embedding_key_col": inferred_embed_key_col,
        "label_key_col": inferred_label_key_col,
        "edge_u_col": edge_u_col,
        "edge_v_col": edge_v_col,
        "edge_weight_mode": edge_weight_mode,
        "dropped_edges": int(dropped_edges),
        "split_source": split_source,
        "graph_ids": graph_ids,
        "node_df": node_df,
        "embedding_cols": emb_cols,
    }

    return data, summary


# ============================================================
# Scaling
# ============================================================

def fit_scaler_train_only(x: torch.Tensor, train_mask: torch.Tensor):
    x_train = x[train_mask]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (x - mean) / std


# ============================================================
# Model
# ============================================================

class GoGNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32, num_layers: int = 3, dropout: float = 0.2):
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

        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def encode(self, x, edge_index, edge_weight=None):
        h = x
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_weight=edge_weight)
        h = self.norms[-1](h)
        h = F.relu(h)
        return h

    def forward(self, data):
        edge_weight = getattr(data, "edge_weight", None)
        z = self.encode(data.x, data.edge_index, edge_weight=edge_weight)
        logits = self.head(z).view(-1)
        return logits, z


# ============================================================
# Train / Eval
# ============================================================

def evaluate_mask(model, data, mask, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits, z = model(data)
        probs = torch.sigmoid(logits)

        y_true = data.y[mask].detach().cpu().numpy().astype(int)
        y_prob = probs[mask].detach().cpu().numpy()
        y_pred = (y_prob >= threshold).astype(int)

        auc = metric_or_nan(roc_auc_score, y_true, y_prob)
        ap = metric_or_nan(average_precision_score, y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else float("nan")
        precision = precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else float("nan")
        recall = recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else float("nan")

        return {
            "auc": auc,
            "ap": ap,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "threshold": threshold,
            "y_true": y_true,
            "y_prob": y_prob,
            "node_emb": z.detach().cpu().numpy(),
            "logits": logits.detach().cpu().numpy(),
            "probs_all": probs.detach().cpu().numpy(),
        }


def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    logits, _ = model(data)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return float(loss.item())


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

    print(f"[Load] embedding_csv -> {args.embedding_csv}")
    print(f"[Load] edge_csv      -> {args.edge_csv}")
    print(f"[Load] label_csv     -> {args.label_csv}")

    data, summary = build_gog_data(
        embedding_csv=args.embedding_csv,
        edge_csv=args.edge_csv,
        label_csv=args.label_csv,
        label_col=args.label_col,
        embedding_key_col=args.embedding_key_col,
        edge_u_col=args.edge_u_col,
        edge_v_col=args.edge_v_col,
        edge_weight_mode=args.edge_weight_mode,
        undirected=args.undirected,
        ignore_level1_split=args.ignore_level1_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.fail_if_no_edges and summary["num_edges"] == 0:
        raise RuntimeError("GoG has zero edges. Aborting due to --fail_if_no_edges")

    print("\n" + "=" * 60)
    print(f" graph_of_graph/main.py  |  Chain: {args.chain.upper()}")
    print("=" * 60)
    print(f"num_nodes        : {summary['num_nodes']}")
    print(f"num_edges        : {summary['num_edges']}")
    print(f"feature_dim      : {summary['feature_dim']}")
    print(f"edge_weight_mode : {summary['edge_weight_mode']}")
    print(f"split_source     : {summary['split_source']}")
    print(f"dropped_edges    : {summary['dropped_edges']}")
    print("=" * 60)

    # scale node features using train nodes only
    mean, std = fit_scaler_train_only(data.x, data.train_mask)
    data.x = apply_scaler(data.x, mean, std)

    data = data.to(device)

    train_labels = data.y[data.train_mask].detach().cpu().numpy().astype(int)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = float(n_neg / max(n_pos, 1))

    print("\n[Train Config]")
    print(f"device           : {device}")
    print(f"train nodes      : {int(data.train_mask.sum())}")
    print(f"val nodes        : {int(data.val_mask.sum())}")
    print(f"test nodes       : {int(data.test_mask.sum())}")
    print(f"train pos_weight : {pos_weight:.4f}")

    model = GoGNodeClassifier(
        in_dim=summary["feature_dim"],
        hidden_dim=args.gnn_hidden,
        out_dim=args.gnn_out,
        num_layers=args.gnn_layers,
        dropout=args.gnn_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device)
    )

    best_state = None
    best_val_auc = -1.0
    best_val_thr = 0.5

    for epoch in range(1, args.gnn_epochs + 1):
        train_loss = train_one_epoch(model, data, optimizer, criterion)

        val_raw = evaluate_mask(model, data, data.val_mask, threshold=0.5)
        val_thr, _ = compute_best_threshold(val_raw["y_true"], val_raw["y_prob"])
        val_eval = evaluate_mask(model, data, data.val_mask, threshold=val_thr)

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

    test_eval = evaluate_mask(model, data, data.test_mask, threshold=best_val_thr)

    print("\n[Test Result]")
    print(f"ROC-AUC  : {test_eval['auc']:.4f}")
    print(f"AP       : {test_eval['ap']:.4f}")
    print(f"F1       : {test_eval['f1']:.4f}")
    print(f"Threshold: {best_val_thr:.2f}")

    # save model
    model_path = artifact_dir / f"{args.chain.lower()}_graph_of_graph_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "best_val_auc": best_val_auc,
        "best_val_threshold": best_val_thr,
        "feature_dim": summary["feature_dim"],
        "embedding_cols": summary["embedding_cols"],
        "graph_ids": summary["graph_ids"],
        "scaler_mean": mean.detach().cpu().numpy(),
        "scaler_std": std.detach().cpu().numpy(),
        "summary": {k: v for k, v in summary.items() if k not in {"node_df", "graph_ids", "embedding_cols"}},
    }, model_path)
    print(f"[Saved] model -> {model_path}")

    # save predictions + GoG node embeddings
    full_eval = evaluate_mask(model, data, torch.ones(data.num_nodes, dtype=torch.bool, device=device), threshold=best_val_thr)
    node_emb = full_eval["node_emb"]
    prob_all = full_eval["probs_all"]

    node_df = summary["node_df"].copy().reset_index(drop=True)

    out_df = pd.DataFrame({
        "graph_id": summary["graph_ids"],
        "label": node_df["label"].astype(int).tolist(),
        "prob": prob_all.astype(float),
        "pred": (prob_all >= best_val_thr).astype(int),
        "split": node_df["split"].tolist() if "split" in node_df.columns else ["unknown"] * len(node_df),
        "is_train": data.train_mask.detach().cpu().numpy().astype(int),
        "is_val": data.val_mask.detach().cpu().numpy().astype(int),
        "is_test": data.test_mask.detach().cpu().numpy().astype(int),
    })

    emb_cols = [f"gog_emb_{i}" for i in range(node_emb.shape[1])]
    emb_df = pd.DataFrame(node_emb, columns=emb_cols)
    out_df = pd.concat([out_df, emb_df], axis=1)

    pred_path = artifact_dir / f"{args.chain.lower()}_gog_node_predictions.csv"
    out_df.to_csv(pred_path, index=False)
    print(f"[Saved] predictions + node embeddings -> {pred_path}")

    # save GoG summary
    if args.save_artifacts:
        summary_path = artifact_dir / "gog_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "chain": args.chain.lower(),
                "num_nodes": summary["num_nodes"],
                "num_edges": summary["num_edges"],
                "feature_dim": summary["feature_dim"],
                "edge_weight_mode": summary["edge_weight_mode"],
                "embedding_key_col": summary["embedding_key_col"],
                "label_key_col": summary["label_key_col"],
                "edge_u_col": summary["edge_u_col"],
                "edge_v_col": summary["edge_v_col"],
                "split_source": summary["split_source"],
                "dropped_edges": summary["dropped_edges"],
                "best_val_auc": best_val_auc,
                "best_val_threshold": best_val_thr,
                "test_auc": test_eval["auc"],
                "test_ap": test_eval["ap"],
                "test_f1": test_eval["f1"],
            }, f, ensure_ascii=False, indent=2)
        print(f"[Saved] summary -> {summary_path}")


if __name__ == "__main__":
    main()
