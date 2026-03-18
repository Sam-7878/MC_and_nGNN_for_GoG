#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_of_graph/main.py

Graph-of-Graphs (Level-2) node classification using:
- Level-1 embeddings from graph_individual/main.py or main_deepwalk.py
- Level-2 GoG edges between contracts

This version supports:
1) Reading Level-1 embedding CSV:
   - e.g. bsc_graph_embeddings.csv
   - e.g. bsc_graph_embeddings_deepwalk.csv

2) Reading GoG edge CSV:
   - each row describes an inter-graph relation between graph_id nodes

3) Fusion of Level-1 and Level-2:
   - Level-1 branch: MLP projection of graph embeddings
   - Level-2 branch: GNN over GoG edges
   - Fusion head: concatenate both and classify

4) Resource logging:
   - peak process RAM
   - peak process CPU
   - peak system CPU / RAM
   - peak GPU utilization
   - peak GPU memory
   - torch peak allocated/reserved GPU memory

Example
-------
python ./graph_of_graph/main.py \
  --chain bsc \
  --embedding_csv ../../_data/dataset/graph_individual_artifacts/bsc/bsc_graph_embeddings_deepwalk.csv \
  --edge_csv ../../_data/level2/bsc_gog_edges.csv \
  --label_csv ../../_data/dataset/features/bsc_basic_metrics_processed.csv \
  --graph_id_col graph_id \
  --label_col label \
  --undirected \
  --use_precomputed_split \
  --use_level1_prob \
  --summary_csv ../../_data/dataset/graph_individual_artifacts/bsc/graph_level1_deepwalk_summary.csv \
  --use_summary_features \
  --gnn_hidden 128 \
  --gnn_out 64 \
  --num_layers 2 \
  --dropout 0.2 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --patience 30 \
  --artifacts_dir ../../_data/dataset/graph_of_graph_artifacts
"""

import os
import json
import time
import math
import random
import logging
import argparse
import threading
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

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
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import add_self_loops


LABEL_KEY_CANDIDATES = ["Contract", "contract", "contract_address", "address", "graph_id"]
GRAPH_ID_CANDIDATES = ["graph_id", "GraphID", "graph", "Contract", "contract", "address"]
EDGE_SRC_CANDIDATES = [
    "src", "source", "from", "u", "src_graph_id", "source_graph_id",
    "contract_a", "graph_id_src", "graph_src"
]
EDGE_DST_CANDIDATES = [
    "dst", "target", "to", "v", "dst_graph_id", "target_graph_id",
    "contract_b", "graph_id_dst", "graph_dst"
]


# ============================================================
# Basic utilities
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


def setup_logger(log_file: Path):
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Resource Monitor
# ============================================================

class ResourceMonitor:
    def __init__(
        self,
        log_dir: Path,
        sample_interval_sec: float = 1.0,
        enable_timeline: bool = True,
        device: Optional[torch.device] = None,
        gpu_index: int = 0,
    ):
        self.log_dir = Path(log_dir)
        self.sample_interval_sec = sample_interval_sec
        self.enable_timeline = enable_timeline
        self.device = device
        self.gpu_index = gpu_index

        self._stop_event = threading.Event()
        self._thread = None
        self._timeline = []

        self._ps_process = None
        self._nvml_handle = None
        self._nvml_ok = False

        self.summary = {
            "process_cpu_percent_peak_raw": 0.0,
            "process_cpu_percent_peak_normalized": 0.0,
            "system_cpu_percent_peak": 0.0,
            "process_rss_gb_peak": 0.0,
            "system_ram_used_gb_peak": 0.0,
            "gpu_util_percent_peak": None,
            "gpu_mem_used_gb_peak": None,
            "gpu_mem_total_gb": None,
            "torch_cuda_max_allocated_gb": None,
            "torch_cuda_max_reserved_gb": None,
            "sampling_interval_sec": sample_interval_sec,
        }

    def start(self):
        if psutil is not None:
            self._ps_process = psutil.Process(os.getpid())
            try:
                self._ps_process.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        if self.device is not None and self.device.type == "cuda":
            try:
                if pynvml is not None:
                    pynvml.nvmlInit()
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

        if self.device is not None and self.device.type == "cuda":
            try:
                self.summary["torch_cuda_max_allocated_gb"] = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                )
                self.summary["torch_cuda_max_reserved_gb"] = float(
                    torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
                )
            except Exception:
                pass

        try:
            if self._nvml_ok:
                pynvml.nvmlShutdown()
        except Exception:
            pass

        self._dump()

    def _run(self):
        logical_cores = 1
        if psutil is not None:
            try:
                logical_cores = max(psutil.cpu_count(logical=True), 1)
            except Exception:
                logical_cores = 1

        while not self._stop_event.is_set():
            ts = time.time()

            row = {
                "timestamp": ts,
                "process_cpu_percent_raw": None,
                "process_cpu_percent_normalized": None,
                "system_cpu_percent": None,
                "process_rss_gb": None,
                "system_ram_used_gb": None,
                "gpu_util_percent": None,
                "gpu_mem_used_gb": None,
                "gpu_mem_total_gb": None,
            }

            if psutil is not None and self._ps_process is not None:
                try:
                    cpu_raw = float(self._ps_process.cpu_percent(interval=None))
                    cpu_norm = float(cpu_raw / logical_cores)
                    rss_gb = float(self._ps_process.memory_info().rss / (1024 ** 3))
                    sys_cpu = float(psutil.cpu_percent(interval=None))
                    sys_ram_used_gb = float(psutil.virtual_memory().used / (1024 ** 3))

                    row["process_cpu_percent_raw"] = cpu_raw
                    row["process_cpu_percent_normalized"] = cpu_norm
                    row["system_cpu_percent"] = sys_cpu
                    row["process_rss_gb"] = rss_gb
                    row["system_ram_used_gb"] = sys_ram_used_gb

                    self.summary["process_cpu_percent_peak_raw"] = max(
                        self.summary["process_cpu_percent_peak_raw"], cpu_raw
                    )
                    self.summary["process_cpu_percent_peak_normalized"] = max(
                        self.summary["process_cpu_percent_peak_normalized"], cpu_norm
                    )
                    self.summary["system_cpu_percent_peak"] = max(
                        self.summary["system_cpu_percent_peak"], sys_cpu
                    )
                    self.summary["process_rss_gb_peak"] = max(
                        self.summary["process_rss_gb_peak"], rss_gb
                    )
                    self.summary["system_ram_used_gb_peak"] = max(
                        self.summary["system_ram_used_gb_peak"], sys_ram_used_gb
                    )
                except Exception:
                    pass

            if self._nvml_ok:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)

                    gpu_util = float(util.gpu)
                    gpu_mem_used_gb = float(mem.used / (1024 ** 3))
                    gpu_mem_total_gb = float(mem.total / (1024 ** 3))

                    row["gpu_util_percent"] = gpu_util
                    row["gpu_mem_used_gb"] = gpu_mem_used_gb
                    row["gpu_mem_total_gb"] = gpu_mem_total_gb

                    self.summary["gpu_util_percent_peak"] = max(
                        self.summary["gpu_util_percent_peak"] or 0.0, gpu_util
                    )
                    self.summary["gpu_mem_used_gb_peak"] = max(
                        self.summary["gpu_mem_used_gb_peak"] or 0.0, gpu_mem_used_gb
                    )
                    self.summary["gpu_mem_total_gb"] = gpu_mem_total_gb
                except Exception:
                    pass

            if self.enable_timeline:
                self._timeline.append(row)

            time.sleep(self.sample_interval_sec)

    def _dump(self):
        ensure_dir(self.log_dir)

        summary_path = self.log_dir / "resource_usage_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2)

        if self.enable_timeline and len(self._timeline) > 0:
            timeline_path = self.log_dir / "resource_usage_timeline.csv"
            pd.DataFrame(self._timeline).to_csv(timeline_path, index=False)


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph-of-Graphs training with Level-1 embeddings + Level-2 GoG edges"
    )

    parser.add_argument("--chain", type=str, required=True)

    parser.add_argument("--embedding_csv", type=str, required=True,
                        help="Level-1 embedding csv from main.py or main_deepwalk.py")
    parser.add_argument("--edge_csv", type=str, required=True,
                        help="Level-2 GoG edge csv")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="Optional external label csv. If omitted, label can be read from embedding_csv")

    parser.add_argument("--graph_id_col", type=str, default="graph_id")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--label_key_col", type=str, default=None)

    parser.add_argument("--edge_src_col", type=str, default="src")
    parser.add_argument("--edge_dst_col", type=str, default="dst")
    parser.add_argument("--undirected", action="store_true")

    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Optional graph summary csv from main_deepwalk or main.py")
    parser.add_argument("--use_summary_features", action="store_true")
    parser.add_argument("--use_level1_prob", action="store_true")
    parser.add_argument("--use_precomputed_split", action="store_true",
                        help="Use split column from embedding_csv if present")

    parser.add_argument("--conv_type", type=str, default="sage", choices=["sage", "gcn", "gat"])
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_out", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu_index", type=int, default=0)

    parser.add_argument("--resource_log_interval_sec", type=float, default=1.0)
    parser.add_argument("--disable_resource_timeline", action="store_true")

    parser.add_argument("--artifacts_dir", type=str, default="graph_of_graph_artifacts")

    return parser.parse_args()


# ============================================================
# Loading labels / embeddings / summary / edges
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
    return label_map, label_key_col


def load_embedding_df(
    embedding_csv: str,
    graph_id_col: str,
    label_col: str,
    use_level1_prob: bool,
    use_precomputed_split: bool,
):
    df = pd.read_csv(embedding_csv, low_memory=False)

    if graph_id_col not in df.columns:
        inferred = infer_col(df.columns, GRAPH_ID_CANDIDATES)
        if inferred is None:
            raise ValueError(f"graph_id_col '{graph_id_col}' not found in embedding_csv")
        graph_id_col = inferred

    df[graph_id_col] = df[graph_id_col].map(normalize_id)
    df = df.dropna(subset=[graph_id_col]).copy()
    df = df.drop_duplicates(subset=[graph_id_col]).reset_index(drop=True)

    emb_cols = [c for c in df.columns if str(c).startswith("emb_")]
    if len(emb_cols) == 0:
        raise ValueError(
            "No embedding columns found in embedding_csv. Expected columns like emb_0, emb_1, ..."
        )

    feature_cols = emb_cols[:]

    if use_level1_prob and "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
        feature_cols.append("prob")

    split_col = "split" if (use_precomputed_split and "split" in df.columns) else None

    return df, graph_id_col, feature_cols, split_col


def load_summary_df(summary_csv: Optional[str], graph_id_col: str):
    if summary_csv is None:
        return None, []

    df = pd.read_csv(summary_csv, low_memory=False)

    if graph_id_col not in df.columns:
        inferred = infer_col(df.columns, GRAPH_ID_CANDIDATES)
        if inferred is None:
            return None, []
        graph_id_col = inferred

    df[graph_id_col] = df[graph_id_col].map(normalize_id)
    df = df.dropna(subset=[graph_id_col]).copy()
    df = df.drop_duplicates(subset=[graph_id_col]).reset_index(drop=True)

    excluded = {graph_id_col, "label"}
    numeric_cols = []
    for c in df.columns:
        if c in excluded:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if not s.isna().all():
            df[c] = s
            numeric_cols.append(c)

    return df, numeric_cols


def load_edge_df(edge_csv: str, edge_src_col: str, edge_dst_col: str):
    df = pd.read_csv(edge_csv, low_memory=False)

    if edge_src_col not in df.columns:
        inferred = infer_col(df.columns, EDGE_SRC_CANDIDATES)
        if inferred is None:
            raise ValueError(f"edge_src_col '{edge_src_col}' not found in edge_csv")
        edge_src_col = inferred

    if edge_dst_col not in df.columns:
        inferred = infer_col(df.columns, EDGE_DST_CANDIDATES)
        if inferred is None:
            raise ValueError(f"edge_dst_col '{edge_dst_col}' not found in edge_csv")
        edge_dst_col = inferred

    df[edge_src_col] = df[edge_src_col].map(normalize_id)
    df[edge_dst_col] = df[edge_dst_col].map(normalize_id)
    df = df.dropna(subset=[edge_src_col, edge_dst_col]).copy()

    return df, edge_src_col, edge_dst_col


# ============================================================
# Dataset building
# ============================================================

def split_indices_by_ratio(y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")

    idx_all = np.arange(len(y))
    labels = np.asarray(y, dtype=int)

    if len(idx_all) < 3:
        raise ValueError("Need at least 3 samples for train/val/test split")

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

    return np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)


def build_masks_from_split(df: pd.DataFrame, split_col: str):
    split = df[split_col].fillna("").astype(str).str.lower()

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    return train_idx, val_idx, test_idx


def fit_feature_scaler(X_train: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs.astype(np.float32)


def build_gog_data(
    embedding_csv: str,
    edge_csv: str,
    label_csv: Optional[str],
    graph_id_col: str,
    label_col: str,
    label_key_col: Optional[str],
    edge_src_col: str,
    edge_dst_col: str,
    undirected: bool,
    summary_csv: Optional[str],
    use_summary_features: bool,
    use_level1_prob: bool,
    use_precomputed_split: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    emb_df, graph_id_col, emb_feature_cols, split_col = load_embedding_df(
        embedding_csv=embedding_csv,
        graph_id_col=graph_id_col,
        label_col=label_col,
        use_level1_prob=use_level1_prob,
        use_precomputed_split=use_precomputed_split,
    )

    # labels
    if label_csv is not None:
        label_map, inferred_label_key_col = load_label_map(
            label_csv=label_csv,
            label_col=label_col,
            label_key_col=label_key_col,
        )
        emb_df[label_col] = emb_df[graph_id_col].map(label_map)
    else:
        inferred_label_key_col = None
        if label_col not in emb_df.columns:
            raise ValueError("label_csv is not given and label_col is not present in embedding_csv")
        emb_df[label_col] = pd.to_numeric(emb_df[label_col], errors="coerce")

    emb_df = emb_df.dropna(subset=[label_col]).copy()
    emb_df[label_col] = emb_df[label_col].astype(int)

    # summary features
    summary_feature_cols = []
    if use_summary_features and summary_csv is not None:
        summary_df, summary_feature_cols = load_summary_df(summary_csv, graph_id_col=graph_id_col)
        if summary_df is not None and len(summary_feature_cols) > 0:
            emb_df = emb_df.merge(
                summary_df[[graph_id_col] + summary_feature_cols],
                on=graph_id_col,
                how="left",
            )
        else:
            summary_feature_cols = []

    feature_cols = emb_feature_cols + summary_feature_cols

    feat_df = safe_numeric_df(emb_df[feature_cols]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_df = feat_df.clip(lower=-50.0, upper=50.0)

    x_raw = feat_df.to_numpy(dtype=np.float32)
    y = emb_df[label_col].to_numpy(dtype=np.int64)
    graph_ids = emb_df[graph_id_col].tolist()

    if not np.isfinite(x_raw).all():
        bad = np.size(x_raw) - np.isfinite(x_raw).sum()
        raise ValueError(f"Non-finite features detected in GoG node features: {bad}")

    # split
    if split_col is not None:
        idx_train, idx_val, idx_test = build_masks_from_split(emb_df, split_col)
        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            idx_train, idx_val, idx_test = split_indices_by_ratio(
                y=y,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
    else:
        idx_train, idx_val, idx_test = split_indices_by_ratio(
            y=y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    mean, std = fit_feature_scaler(x_raw[idx_train])
    x_scaled = apply_feature_scaler(x_raw, mean, std)

    # edges
    edge_df, edge_src_col, edge_dst_col = load_edge_df(
        edge_csv=edge_csv,
        edge_src_col=edge_src_col,
        edge_dst_col=edge_dst_col,
    )

    node_set = set(graph_ids)
    edge_df = edge_df[
        edge_df[edge_src_col].isin(node_set) & edge_df[edge_dst_col].isin(node_set)
    ].copy()

    gid2idx = {gid: i for i, gid in enumerate(graph_ids)}

    if len(edge_df) > 0:
        src_idx = edge_df[edge_src_col].map(gid2idx).to_numpy(dtype=np.int64)
        dst_idx = edge_df[edge_dst_col].map(gid2idx).to_numpy(dtype=np.int64)
        edge_pairs = np.stack([src_idx, dst_idx], axis=1)

        if undirected:
            rev = edge_pairs[:, [1, 0]]
            edge_pairs = np.vstack([edge_pairs, rev])

        edge_pairs = np.unique(edge_pairs, axis=0)
        edge_index = torch.tensor(edge_pairs.T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index, _ = add_self_loops(edge_index, num_nodes=len(graph_ids))

    data = Data(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32),
    )

    train_mask = torch.zeros(len(graph_ids), dtype=torch.bool)
    val_mask = torch.zeros(len(graph_ids), dtype=torch.bool)
    test_mask = torch.zeros(len(graph_ids), dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    summary = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_pos": int(y.sum()),
        "num_neg": int((y == 0).sum()),
        "feature_dim": int(data.x.shape[1]),
        "train_size": int(train_mask.sum().item()),
        "val_size": int(val_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
        "graph_id_list": graph_ids,
        "feature_cols": feature_cols,
        "embedding_feature_cols": emb_feature_cols,
        "summary_feature_cols": summary_feature_cols,
        "label_key_col": inferred_label_key_col,
        "scaler_mean": mean,
        "scaler_std": std,
    }

    return data, summary


# ============================================================
# Model
# ============================================================

def build_conv(conv_type: str, in_dim: int, out_dim: int, heads: int = 4):
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim)
    if conv_type == "sage":
        return SAGEConv(in_dim, out_dim)
    if conv_type == "gat":
        return GATConv(in_dim, out_dim, heads=1, concat=False)
    raise ValueError(f"Unknown conv_type: {conv_type}")


class GoGFusionNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = "sage",
    ):
        super().__init__()

        self.dropout = dropout

        self.level1_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(build_conv(conv_type, hidden_dim, hidden_dim))

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        h_local = self.level1_proj(x)

        h_graph = h_local
        for conv in self.convs:
            h_graph = conv(h_graph, edge_index)
            h_graph = F.relu(h_graph)
            h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)

        h = torch.cat([h_local, h_graph], dim=1)
        z = self.fusion(h)
        logits = self.classifier(z).view(-1)
        return logits, z


# ============================================================
# Train / Eval
# ============================================================

def evaluate_masked(model, data, mask, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits, emb = model(data)
        probs = torch.sigmoid(logits)

    mask_np = mask.detach().cpu().numpy().astype(bool)
    if mask_np.sum() == 0:
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

    y_true = data.y.detach().cpu().numpy()[mask_np].astype(int)
    y_prob = probs.detach().cpu().numpy()[mask_np]
    idx = np.where(mask_np)[0]
    emb_np = emb.detach().cpu().numpy()[mask_np]

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
        "emb": emb_np,
    }


def train_one_epoch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    logits, _ = model(data)
    train_logits = logits[data.train_mask]
    train_targets = data.y[data.train_mask]

    loss = criterion(train_logits, train_targets)

    if not torch.isfinite(loss):
        return float("nan")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    run_dir = Path(args.artifacts_dir) / args.chain.lower()
    ensure_dir(run_dir)

    log_file = run_dir / "graph_of_graph_train.log"
    logger = setup_logger(log_file)

    monitor = ResourceMonitor(
        log_dir=run_dir,
        sample_interval_sec=args.resource_log_interval_sec,
        enable_timeline=not args.disable_resource_timeline,
        device=device,
        gpu_index=args.gpu_index,
    )
    monitor.start()

    t0_all = time.perf_counter()

    try:
        logger.info(f"[Load] embedding_csv -> {args.embedding_csv}")
        logger.info(f"[Load] edge_csv      -> {args.edge_csv}")
        logger.info(f"[Load] label_csv     -> {args.label_csv}")
        logger.info(f"[Load] summary_csv   -> {args.summary_csv}")

        logger.info("=" * 70)
        logger.info(f" graph_of_graph/main.py  |  Chain: {args.chain.upper()}")
        logger.info("=" * 70)
        logger.info(f"graph_id_col         : {args.graph_id_col}")
        logger.info(f"label_col            : {args.label_col}")
        logger.info(f"edge_src_col         : {args.edge_src_col}")
        logger.info(f"edge_dst_col         : {args.edge_dst_col}")
        logger.info(f"conv_type            : {args.conv_type}")
        logger.info(f"use_summary_features : {args.use_summary_features}")
        logger.info(f"use_level1_prob      : {args.use_level1_prob}")
        logger.info(f"use_precomputed_split: {args.use_precomputed_split}")
        logger.info("=" * 70)

        t0_build = time.perf_counter()
        data, summary = build_gog_data(
            embedding_csv=args.embedding_csv,
            edge_csv=args.edge_csv,
            label_csv=args.label_csv,
            graph_id_col=args.graph_id_col,
            label_col=args.label_col,
            label_key_col=args.label_key_col,
            edge_src_col=args.edge_src_col,
            edge_dst_col=args.edge_dst_col,
            undirected=args.undirected,
            summary_csv=args.summary_csv,
            use_summary_features=args.use_summary_features,
            use_level1_prob=args.use_level1_prob,
            use_precomputed_split=args.use_precomputed_split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        build_sec = time.perf_counter() - t0_build

        data = data.to(device)

        logger.info("[Dataset Summary]")
        logger.info(f"Loaded GoG nodes          : {summary['num_nodes']}")
        logger.info(f"Loaded GoG edges          : {summary['num_edges']}")
        logger.info(f"Fraud nodes               : {summary['num_pos']} / {summary['num_nodes']}")
        logger.info(f"Feature dim               : {summary['feature_dim']}")
        logger.info(f"Train/Val/Test            : {summary['train_size']}/{summary['val_size']}/{summary['test_size']}")
        logger.info(f"Embedding feature dim     : {len(summary['embedding_feature_cols'])}")
        logger.info(f"Summary feature dim       : {len(summary['summary_feature_cols'])}")
        logger.info(f"Label key col             : {summary['label_key_col']}")
        logger.info(f"Build time sec            : {build_sec:.2f}")

        pos_count = int(summary["num_pos"])
        neg_count = int(summary["num_neg"])
        train_mask_np = data.train_mask.detach().cpu().numpy()
        y_np = data.y.detach().cpu().numpy().astype(int)
        train_pos = int(y_np[train_mask_np].sum())
        train_neg = int((y_np[train_mask_np] == 0).sum())
        pos_weight = float(train_neg / max(train_pos, 1))

        logger.info("[Train Config]")
        logger.info(f"device      : {device}")
        logger.info(f"pos_weight  : {pos_weight:.4f}")
        logger.info(f"hidden/out  : {args.gnn_hidden}/{args.gnn_out}")
        logger.info(f"num_layers  : {args.num_layers}")
        logger.info(f"dropout     : {args.dropout}")
        logger.info(f"epochs      : {args.epochs}")
        logger.info(f"lr          : {args.lr}")
        logger.info(f"weight_decay: {args.weight_decay}")
        logger.info(f"patience    : {args.patience}")

        model = GoGFusionNet(
            in_dim=summary["feature_dim"],
            hidden_dim=args.gnn_hidden,
            out_dim=args.gnn_out,
            num_layers=args.num_layers,
            dropout=args.dropout,
            conv_type=args.conv_type,
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
        best_epoch = 0
        patience_counter = 0

        t0_train = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, data, optimizer, criterion, device)

            val_raw = evaluate_masked(model, data, data.val_mask, device, threshold=0.5)
            val_thr, _ = compute_best_threshold(val_raw["y_true"], val_raw["y_prob"])
            val_eval = evaluate_masked(model, data, data.val_mask, device, threshold=val_thr)

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
                best_epoch = epoch
                patience_counter = 0
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "val_f1": val_f1,
                    "val_threshold": val_thr,
                }
            else:
                patience_counter += 1

            logger.info(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"val_ap={val_ap:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"thr={val_thr:.2f} | "
                f"patience={patience_counter}/{args.patience}"
            )

            if patience_counter >= args.patience:
                logger.info(f"[Early Stop] Stop at epoch {epoch}, best_epoch={best_epoch}")
                break

        train_sec = time.perf_counter() - t0_train

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        test_eval = evaluate_masked(model, data, data.test_mask, device, threshold=best_val_thr)

        logger.info("[Test Result]")
        logger.info(f"ROC-AUC  : {test_eval['auc']:.4f}")
        logger.info(f"AP       : {test_eval['ap']:.4f}")
        logger.info(f"F1       : {test_eval['f1']:.4f}")
        logger.info(f"Precision: {test_eval['precision']:.4f}")
        logger.info(f"Recall   : {test_eval['recall']:.4f}")
        logger.info(f"Threshold: {best_val_thr:.2f}")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Train time sec: {train_sec:.2f}")

        # Save model
        model_path = run_dir / f"{args.chain.lower()}_graph_of_graph_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "best_val_threshold": best_val_thr,
            "feature_dim": summary["feature_dim"],
            "feature_cols": summary["feature_cols"],
            "embedding_feature_cols": summary["embedding_feature_cols"],
            "summary_feature_cols": summary["summary_feature_cols"],
            "graph_id_list": summary["graph_id_list"],
            "scaler_mean": summary["scaler_mean"],
            "scaler_std": summary["scaler_std"],
        }, model_path)
        logger.info(f"[Saved] model -> {model_path}")

        # Save all node embeddings / predictions
        model.eval()
        with torch.no_grad():
            logits_all, z_all = model(data)
            prob_all = torch.sigmoid(logits_all).detach().cpu().numpy()
            pred_all = (prob_all >= best_val_thr).astype(int)
            z_all = z_all.detach().cpu().numpy()

        emb_cols = [f"gog_emb_{i}" for i in range(z_all.shape[1])]
        out_df = pd.DataFrame(z_all, columns=emb_cols)
        out_df.insert(0, "graph_id", summary["graph_id_list"])
        out_df["label"] = data.y.detach().cpu().numpy().astype(int)
        out_df["prob"] = prob_all
        out_df["pred"] = pred_all

        split_arr = np.array(["unknown"] * len(summary["graph_id_list"]), dtype=object)
        split_arr[data.train_mask.detach().cpu().numpy()] = "train"
        split_arr[data.val_mask.detach().cpu().numpy()] = "val"
        split_arr[data.test_mask.detach().cpu().numpy()] = "test"
        out_df["split"] = split_arr

        emb_out = run_dir / f"{args.chain.lower()}_gog_node_embeddings.csv"
        out_df.to_csv(emb_out, index=False)
        logger.info(f"[Saved] node embeddings -> {emb_out}")

        # Save prediction table
        pred_out = run_dir / f"{args.chain.lower()}_gog_predictions.csv"
        out_df[["graph_id", "label", "prob", "pred", "split"]].to_csv(pred_out, index=False)
        logger.info(f"[Saved] predictions -> {pred_out}")

        # Save metadata
        meta_path = run_dir / f"{args.chain.lower()}_gog_meta.json"
        total_sec = time.perf_counter() - t0_all
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "chain": args.chain,
                "feature_dim": summary["feature_dim"],
                "num_nodes": summary["num_nodes"],
                "num_edges": summary["num_edges"],
                "num_pos": summary["num_pos"],
                "num_neg": summary["num_neg"],
                "train_size": summary["train_size"],
                "val_size": summary["val_size"],
                "test_size": summary["test_size"],
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "best_val_threshold": best_val_thr,
                "test_auc": test_eval["auc"],
                "test_ap": test_eval["ap"],
                "test_f1": test_eval["f1"],
                "test_precision": test_eval["precision"],
                "test_recall": test_eval["recall"],
                "build_time_sec": build_sec,
                "train_time_sec": train_sec,
                "total_time_sec": total_sec,
                "feature_cols": summary["feature_cols"],
                "embedding_feature_cols": summary["embedding_feature_cols"],
                "summary_feature_cols": summary["summary_feature_cols"],
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"[Saved] meta -> {meta_path}")

    finally:
        monitor.stop()

        try:
            resource_summary_path = run_dir / "resource_usage_summary.json"
            if resource_summary_path.exists():
                with open(resource_summary_path, "r", encoding="utf-8") as f:
                    rs = json.load(f)

                logger.info("[Resource Summary]")
                logger.info(
                    f"Process CPU peak raw (%)         : {rs.get('process_cpu_percent_peak_raw')}"
                )
                logger.info(
                    f"Process CPU peak normalized (%)  : {rs.get('process_cpu_percent_peak_normalized')}"
                )
                logger.info(
                    f"System CPU peak (%)              : {rs.get('system_cpu_percent_peak')}"
                )
                logger.info(
                    f"Process RAM peak (GB)            : {rs.get('process_rss_gb_peak')}"
                )
                logger.info(
                    f"System RAM used peak (GB)        : {rs.get('system_ram_used_gb_peak')}"
                )
                logger.info(
                    f"GPU util peak (%)                : {rs.get('gpu_util_percent_peak')}"
                )
                logger.info(
                    f"GPU memory used peak (GB)        : {rs.get('gpu_mem_used_gb_peak')}"
                )
                logger.info(
                    f"torch max allocated (GB)         : {rs.get('torch_cuda_max_allocated_gb')}"
                )
                logger.info(
                    f"torch max reserved (GB)          : {rs.get('torch_cuda_max_reserved_gb')}"
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()
