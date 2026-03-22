# src/gog_fraud/data/preprocessing/graph_builder.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data

from gog_fraud.data.preprocessing.normalizer import normalize_edge_features, _canon

log = logging.getLogger(__name__)

_SRC_COLS = ("from", "from_address", "fromaddress", "src", "sender")
_DST_COLS = ("to", "to_address", "toaddress", "dst", "receiver")


def _find_col(df: pd.DataFrame, candidates: tuple) -> Optional[str]:
    canon_map = {_canon(c): c for c in df.columns}
    for cand in candidates:
        if _canon(cand) in canon_map:
            return canon_map[_canon(cand)]
    return None


def build_graph_from_csv(
    path: Path,
    contract_id: str,
    chain: str,
) -> tuple[Data, dict]:
    """
    Raw transaction CSV → normalized PyG Data.

    Returns:
        graph : torch_geometric.data.Data (tensor-only, NaN-free)
        meta  : 메타 정보 dict
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if df.empty:
        raise ValueError("empty csv")

    src_col = _find_col(df, _SRC_COLS)
    dst_col = _find_col(df, _DST_COLS)

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Cannot find src/dst columns. columns={list(df.columns)}"
        )

    # 주소 정제
    df[src_col] = df[src_col].astype(str).str.strip().str.lower()
    df[dst_col] = df[dst_col].astype(str).str.strip().str.lower()

    invalid_tokens = {"", "nan", "none", "null"}
    df = df[
        ~df[src_col].isin(invalid_tokens) &
        ~df[dst_col].isin(invalid_tokens)
    ].copy()

    if df.empty:
        raise ValueError("no valid rows after address cleanup")

    # 노드 인덱싱
    nodes = sorted(
        set(df[src_col].tolist()) |
        set(df[dst_col].tolist()) |
        {contract_id}
    )
    node2idx = {addr: i for i, addr in enumerate(nodes)}
    num_nodes = len(nodes)

    src_idx = [node2idx[a] for a in df[src_col].tolist()]
    dst_idx = [node2idx[a] for a in df[dst_col].tolist()]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # edge_attr: log1p 정규화 포함
    edge_feat_df, selected_map = normalize_edge_features(df)
    edge_attr = torch.tensor(
        edge_feat_df.values.astype("float64"),
        dtype=torch.float32,
    )

    # NaN/Inf 최종 방어
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=88.0, neginf=0.0)

    # node feature: log1p(in_deg), log1p(out_deg), log1p(total_deg)
    in_deg = torch.zeros(num_nodes, dtype=torch.float64)
    out_deg = torch.zeros(num_nodes, dtype=torch.float64)

    for s in src_idx:
        out_deg[s] += 1.0
    for d in dst_idx:
        in_deg[d] += 1.0

    x = torch.stack([
        torch.log1p(in_deg),
        torch.log1p(out_deg),
        torch.log1p(in_deg + out_deg),
    ], dim=1).to(torch.float32)

    x = torch.nan_to_num(x, nan=0.0, posinf=88.0, neginf=0.0)

    graph = Data(
        x=x.contiguous(),
        edge_index=edge_index.contiguous(),
        edge_attr=edge_attr.contiguous(),
        num_nodes=num_nodes,
    )

    meta = {
        "contract_id": contract_id,
        "chain": chain,
        "num_transactions": int(len(df)),
        "num_nodes": num_nodes,
        "node_addresses": nodes,
        "selected_numeric_cols": selected_map,
        "source_csv": str(path),
    }

    return graph, meta
