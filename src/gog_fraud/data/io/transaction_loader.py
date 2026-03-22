# src/gog_fraud/data/io/transaction_loader.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


@dataclass
class TransactionGraph:
    contract_id: str
    graph: Data
    chain: str
    source_path: Optional[str] = None
    meta: Dict = field(default_factory=dict)


def _canon(s: str) -> str:
    return str(s).replace("\ufeff", "").strip().lower().replace(" ", "").replace("_", "")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    canon_map = {_canon(c): c for c in df.columns}
    for cand in candidates:
        key = _canon(cand)
        if key in canon_map:
            return canon_map[key]
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _validate_graph(graph: Data, contract_id: str) -> None:
    if getattr(graph, "x", None) is None or not torch.is_tensor(graph.x):
        raise ValueError(f"{contract_id}: graph.x is missing or invalid")

    if getattr(graph, "edge_index", None) is None or not torch.is_tensor(graph.edge_index):
        raise ValueError(f"{contract_id}: graph.edge_index is missing or invalid")

    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(f"{contract_id}: edge_index must have shape [2, E]")

    if graph.x.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"{contract_id}: graph.x must be float tensor, got {graph.x.dtype}")

    if graph.edge_index.dtype != torch.long:
        raise ValueError(f"{contract_id}: edge_index must be torch.long, got {graph.edge_index.dtype}")

    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        raise ValueError(f"{contract_id}: graph.x contains NaN/Inf")

    if getattr(graph, "edge_attr", None) is not None:
        if not torch.is_tensor(graph.edge_attr):
            raise ValueError(f"{contract_id}: edge_attr must be tensor if present")
        if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
            raise ValueError(f"{contract_id}: edge_attr contains NaN/Inf")

    if int(graph.num_nodes) != int(graph.x.size(0)):
        raise ValueError(
            f"{contract_id}: num_nodes ({graph.num_nodes}) != x.size(0) ({graph.x.size(0)})"
        )

    if graph.edge_index.numel() > 0:
        max_idx = int(graph.edge_index.max().item())
        min_idx = int(graph.edge_index.min().item())
        if min_idx < 0:
            raise ValueError(f"{contract_id}: edge_index contains negative node index")
        if max_idx >= int(graph.num_nodes):
            raise ValueError(
                f"{contract_id}: edge_index out of bounds: max={max_idx}, num_nodes={graph.num_nodes}"
            )


class TransactionLoader:
    _SRC_COLS = ("from", "from_address", "fromaddress", "src", "sender")
    _DST_COLS = ("to", "to_address", "toaddress", "dst", "receiver")

    # raw tx csv에서 자주 보이는 숫자 컬럼 후보들
    _NUMERIC_GROUPS = {
        "value": ("value", "amount"),
        "gas": ("gas", "gasused", "gas_used"),
        "gas_price": ("gasprice", "gas_price"),
        "timestamp": ("timestamp", "timeStamp", "block_timestamp"),
        "block": ("blocknumber", "block_number"),
        "nonce": ("nonce",),
    }

    def __init__(
        self,
        root: str,
        chain: str,
        supported_extensions: Tuple[str, ...] = (".csv",),
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.chain = chain
        self.chain_dir = self.root / chain
        self.exts = supported_extensions
        self.verbose = verbose

        if not self.chain_dir.exists():
            raise FileNotFoundError(f"Transaction directory not found: {self.chain_dir}")

    def load_all(self) -> List[TransactionGraph]:
        results: List[TransactionGraph] = []
        skipped = 0

        files = sorted(p for p in self.chain_dir.iterdir() if p.is_file() and p.suffix.lower() in self.exts)

        for path in files:
            try:
                results.append(self._load_single_file(path))
            except Exception as exc:
                skipped += 1
                log.warning(f"[TransactionLoader] Skip {path.name}: {exc}")

        log.info(
            f"[TransactionLoader] Loaded {len(results)} graphs from {self.chain_dir} "
            f"(skipped={skipped})"
        )
        return results

    def _load_single_file(self, path: Path) -> TransactionGraph:
        contract_id = path.stem.lower()
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported extension: {path.suffix}")
        return self._load_csv_file(path, contract_id, self.chain)

    def _load_csv_file(self, path: Path, contract_id: str, chain: str) -> TransactionGraph:
        # 모든 컬럼을 우선 문자열로 읽은 뒤, 필요한 컬럼만 명시적으로 숫자 변환
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df = _normalize_columns(df)

        if df.empty:
            raise ValueError("empty csv")

        src_col = _find_col(df, self._SRC_COLS)
        dst_col = _find_col(df, self._DST_COLS)

        if src_col is None or dst_col is None:
            raise ValueError(
                f"Could not detect src/dst columns. columns={list(df.columns)}"
            )

        work = df.copy()
        work[src_col] = work[src_col].astype(str).str.strip().str.lower()
        work[dst_col] = work[dst_col].astype(str).str.strip().str.lower()

        # 명백히 invalid한 주소 행 제거
        invalid_tokens = {"", "nan", "none", "null"}
        work = work[
            (~work[src_col].isin(invalid_tokens)) &
            (~work[dst_col].isin(invalid_tokens))
        ].copy()

        if work.empty:
            raise ValueError("no valid rows after src/dst cleanup")

        # 노드 집합 구성
        nodes = sorted(set(work[src_col].tolist()) | set(work[dst_col].tolist()) | {contract_id})
        node2idx = {addr: i for i, addr in enumerate(nodes)}

        src = [node2idx[a] for a in work[src_col].tolist()]
        dst = [node2idx[a] for a in work[dst_col].tolist()]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # edge feature: 대표 numeric 컬럼만 정규화해서 사용
        numeric_df = pd.DataFrame(index=work.index)
        selected_numeric_cols: Dict[str, str] = {}

        for out_name, aliases in self._NUMERIC_GROUPS.items():
            col = _find_col(work, aliases)
            if col is not None:
                numeric_df[out_name] = _safe_numeric(work[col])
                selected_numeric_cols[out_name] = col

        # edge_attr는 항상 tensor가 되도록 구성
        # 숫자 컬럼이 하나도 없으면 상수 1 feature 사용
        if numeric_df.shape[1] == 0:
            edge_attr = torch.ones((len(work), 1), dtype=torch.float32)
        else:
            edge_attr = torch.tensor(numeric_df.values, dtype=torch.float32)

        # 간단하고 안정적인 node feature
        num_nodes = len(nodes)
        in_deg = torch.zeros(num_nodes, dtype=torch.float32)
        out_deg = torch.zeros(num_nodes, dtype=torch.float32)

        for s in src:
            out_deg[s] += 1.0
        for d in dst:
            in_deg[d] += 1.0

        total_deg = in_deg + out_deg
        x = torch.stack(
            [
                torch.log1p(in_deg),
                torch.log1p(out_deg),
                torch.log1p(total_deg),
            ],
            dim=1,
        ).to(torch.float32)

        graph = Data(
            x=x.contiguous(),
            edge_index=edge_index.contiguous(),
            edge_attr=edge_attr.contiguous(),
            num_nodes=num_nodes,
        )

        _validate_graph(graph, contract_id)

        return TransactionGraph(
            contract_id=contract_id,
            graph=graph,
            chain=chain,
            source_path=str(path),
            meta={
                "num_transactions": int(len(work)),
                "num_nodes": int(num_nodes),
                "selected_numeric_cols": selected_numeric_cols,
                "node_addresses": nodes,  # Data 내부가 아니라 meta에 보관
            },
        )
