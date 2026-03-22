# src/gog_fraud/data/io/transaction_loader.py
"""
transactions/{chain}/ 디렉토리에서 개별 transaction graph를 읽는 loader.

디렉토리 레이아웃(두 형식 모두 지원):
  형식 A – 파일 1개 = 그래프 1개
      transactions/{chain}/{contract_address}.pt   (torch_geometric Data)
      transactions/{chain}/{contract_address}.json (edge list + feature dict)

  형식 B – 단일 집계 파일
      transactions/{chain}/graphs.pt   (list[Data] + meta list)
"""

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


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _load_csv_file(path: Path, contract_id: str, chain: str) -> TransactionGraph:
    df = pd.read_csv(path)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    src_col = _find_col(df, ("from", "from_address", "src", "sender"))
    dst_col = _find_col(df, ("to", "to_address", "dst", "receiver"))

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Could not detect src/dst columns in {path.name}. "
            f"Columns={list(df.columns)}"
        )

    df = df[[c for c in df.columns if c in df.columns]].copy()
    df[src_col] = df[src_col].astype(str).str.lower().str.strip()
    df[dst_col] = df[dst_col].astype(str).str.lower().str.strip()

    # 주소 인덱싱
    nodes = sorted(set(df[src_col].tolist()) | set(df[dst_col].tolist()) | {contract_id})
    node2idx = {addr: i for i, addr in enumerate(nodes)}

    src = [node2idx[a] for a in df[src_col]]
    dst = [node2idx[a] for a in df[dst_col]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # numeric edge features
    numeric_cols = []
    for cand in ["value", "amount", "gas", "gas_price", "timestamp", "blocknumber", "nonce"]:
        col = _find_col(df, (cand,))
        if col is not None:
            numeric_cols.append(col)

    if numeric_cols:
        edge_attr = torch.tensor(
            df[numeric_cols].fillna(0).astype(float).values,
            dtype=torch.float,
        )
    else:
        edge_attr = None

    # simple node features
    in_deg = torch.zeros(len(nodes), dtype=torch.float)
    out_deg = torch.zeros(len(nodes), dtype=torch.float)

    for s in src:
        out_deg[s] += 1
    for d in dst:
        in_deg[d] += 1

    x = torch.stack([in_deg, out_deg, in_deg + out_deg], dim=1)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes),
    )
    graph.contract_id = contract_id
    graph.node_addresses = nodes

    return TransactionGraph(
        contract_id=contract_id,
        graph=graph,
        chain=chain,
        source_path=str(path),
        meta={"num_transactions": len(df)},
    )







# ---------------------------------------------------------------------------
# 개별 파일 파서
# ---------------------------------------------------------------------------

def _load_pt_file(path: Path, contract_id: str, chain: str) -> TransactionGraph:
    """단일 .pt 파일(torch_geometric.data.Data)을 로드."""
    obj = torch.load(str(path), map_location="cpu")

    if isinstance(obj, Data):
        graph = obj
    elif isinstance(obj, dict):
        # {x, edge_index, edge_attr, ...} dict 형태 지원
        graph = Data(**obj)
    else:
        raise ValueError(
            f"Unsupported .pt content type {type(obj)} in {path}"
        )

    return TransactionGraph(
        contract_id=contract_id,
        graph=graph,
        chain=chain,
        source_path=str(path),
    )


def _load_json_file(path: Path, contract_id: str, chain: str) -> TransactionGraph:
    """
    edge-list JSON 포맷 지원.

    예시 schema:
    {
      "nodes": [{"id": 0, "features": [...]}, ...],
      "edges": [{"src": 0, "dst": 1, "attr": [...]}, ...],
      "meta":  {...}
    }
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])

    # node feature matrix
    if nodes and "features" in nodes[0]:
        x = torch.tensor([n["features"] for n in nodes], dtype=torch.float)
    else:
        x = torch.zeros((len(nodes), 1), dtype=torch.float)

    # edge index
    if edges:
        src = [e["src"] for e in edges]
        dst = [e["dst"] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # edge attribute (optional)
    if edges and "attr" in edges[0]:
        edge_attr = torch.tensor([e["attr"] for e in edges], dtype=torch.float)
    else:
        edge_attr = None

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return TransactionGraph(
        contract_id=contract_id,
        graph=graph,
        chain=chain,
        source_path=str(path),
        meta=raw.get("meta", {}),
    )


# ---------------------------------------------------------------------------
# 집계 파일 파서
# ---------------------------------------------------------------------------

def _load_aggregate_pt(path: Path, chain: str) -> List[TransactionGraph]:
    """
    단일 graphs.pt 파일에 다수 그래프가 묶인 경우.

    예상 포맷:
      {"graphs": [Data, ...], "contract_ids": ["0xabc", ...]}
    또는
      [{"contract_id": "...", "data": Data}, ...]
    """
    obj = torch.load(str(path), map_location="cpu")

    results: List[TransactionGraph] = []

    if isinstance(obj, dict) and "graphs" in obj:
        graphs = obj["graphs"]
        ids = obj.get("contract_ids", [str(i) for i in range(len(graphs))])
        for cid, g in zip(ids, graphs):
            results.append(
                TransactionGraph(
                    contract_id=cid,
                    graph=g,
                    chain=chain,
                    source_path=str(path),
                )
            )
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                cid = item.get("contract_id", str(i))
                g = item.get("data") or Data(**{k: v for k, v in item.items()
                                                if k != "contract_id"})
            else:
                cid, g = str(i), item
            results.append(
                TransactionGraph(
                    contract_id=cid,
                    graph=g,
                    chain=chain,
                    source_path=str(path),
                )
            )
    else:
        raise ValueError(f"Unsupported aggregate .pt format in {path}")

    return results


# ---------------------------------------------------------------------------
# 메인 Loader
# ---------------------------------------------------------------------------

class TransactionLoader:
    """
    Usage
    -----
    loader = TransactionLoader(root="../_data/dataset/transactions", chain="plygon")
    graphs = loader.load_all()          # List[TransactionGraph]
    g      = loader.load_one("0xabc")   # TransactionGraph
    """
    def __init__(
        self,
        root: str,
        chain: str,
        supported_extensions: Tuple[str, ...] = (".pt", ".json", ".csv"),
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.chain = chain
        self.chain_dir = self.root / chain
        self.exts = supported_extensions
        self.verbose = verbose
 
        if not self.chain_dir.exists():
            raise FileNotFoundError(f"Transaction directory not found: {self.chain_dir}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            log.info(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> List[TransactionGraph]:
        results: List[TransactionGraph] = []
        for path in sorted(self.chain_dir.iterdir()):
            if path.suffix not in self.exts:
                continue
            try:
                results.append(self._load_single_file(path))
            except Exception as exc:
                log.warning(f"[TransactionLoader] Skip {path.name}: {exc}")
 
        log.info(f"[TransactionLoader] Loaded {len(results)} graphs from {self.chain_dir}")
        return results


    

    def load_one(self, contract_id: str) -> TransactionGraph:
        """단일 contract ID에 해당하는 그래프를 로드."""
        for ext in self.exts:
            path = self.chain_dir / f"{contract_id}{ext}"
            if path.exists():
                return self._load_single_file(path)
        raise FileNotFoundError(
            f"No file found for contract_id={contract_id} "
            f"in {self.chain_dir}"
        )

    def contract_ids(self) -> List[str]:
        """chain 디렉토리에 존재하는 모든 contract ID 목록."""
        agg = self.chain_dir / "graphs.pt"
        if agg.exists():
            graphs = self.load_all()
            return [g.contract_id for g in graphs]

        ids = []
        for path in sorted(self.chain_dir.iterdir()):
            if path.suffix in self.exts:
                ids.append(path.stem)
        return ids

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_single_file(self, path: Path) -> TransactionGraph:
        contract_id = path.stem.lower()
        if path.suffix == ".csv":
            return _load_csv_file(path, contract_id, self.chain)
        elif path.suffix == ".pt":
            return _load_pt_file(path, contract_id, self.chain)
        elif path.suffix == ".json":
            return _load_json_file(path, contract_id, self.chain)
        else:
            raise ValueError(f"Unsupported extension: {path.suffix}")
