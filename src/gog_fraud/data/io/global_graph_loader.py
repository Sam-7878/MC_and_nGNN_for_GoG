# src/gog_fraud/data/io/global_graph_loader.py
"""
../_data/dataset/global_graph/ 에서 contract mapping data를 읽는 loader.

지원 포맷:
  global_graph/
    edges.csv         - src_contract, dst_contract, edge_type, weight(opt)
    edges.pt          - torch_geometric Data (전체 global graph)
    node_features.pt  - contract_id -> feature tensor 매핑
    meta.json         - 전체 메타 정보
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 반환 타입
# ---------------------------------------------------------------------------

@dataclass
class GlobalGraphData:
    """로드된 global graph 전체를 담는 컨테이너."""

    # 핵심 그래프 구조
    graph: Optional[Data] = None

    # contract_id 순서 목록 (graph.x의 i번째 = contract_ids[i])
    contract_ids: List[str] = field(default_factory=list)

    # contract_id -> node index 매핑 (빠른 탐색)
    id_to_idx: Dict[str, int] = field(default_factory=dict)

    # contract_id -> raw feature tensor 매핑 (graph 없이도 접근 가능)
    node_features: Dict[str, torch.Tensor] = field(default_factory=dict)

    # meta 정보
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def contract_subgraph(self, contract_ids: List[str]) -> Data:
        """
        주어진 contract_id 목록에 해당하는 subgraph를 반환.
        Level 2 모델 입력으로 사용.
        """
        if self.graph is None:
            raise RuntimeError("Global graph was not loaded (no edges.pt found).")

        from torch_geometric.utils import subgraph as tg_subgraph

        idx = [self.id_to_idx[c] for c in contract_ids if c in self.id_to_idx]
        if not idx:
            raise ValueError("None of the contract_ids are in the global graph.")

        subset = torch.tensor(idx, dtype=torch.long)
        edge_index, edge_attr = tg_subgraph(
            subset=subset,
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
            relabel_nodes=True,
            num_nodes=self.graph.num_nodes,
        )
        x = self.graph.x[subset] if self.graph.x is not None else None
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def num_nodes(self) -> int:
        return len(self.contract_ids)

    def num_edges(self) -> int:
        if self.graph is None:
            return 0
        return self.graph.edge_index.shape[1]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class GlobalGraphLoader:
    """
    Usage
    -----
    gl = GlobalGraphLoader("../_data/dataset/global_graph")
    gd = gl.load()                          # GlobalGraphData
    sub = gd.contract_subgraph(["0xabc"])   # Level2 입력용 subgraph
    """

    def __init__(self, root: str, chain: str, normalize_address: bool = True):
        self.root = Path(root)
        self.chain = chain
        self.normalize_address = normalize_address
 
    def load(self):
        mapping_path = self.root / f"{self.chain}_contract_to_number_mapping.json"
        graph_path = self.root / f"{self.chain}_graph_more_than_1_ratio.csv"
 
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph CSV not found: {graph_path}")
 
        with open(mapping_path, "r", encoding="utf-8") as f:
            contract_to_number = json.load(f)
 
        if self.normalize_address:
            contract_to_number = {
                str(k).lower(): int(v)
                for k, v in contract_to_number.items()
            }
        else:
            contract_to_number = {
                str(k): int(v)
                for k, v in contract_to_number.items()
            }
 
        number_to_contract = {v: k for k, v in contract_to_number.items()}
 
        df = pd.read_csv(graph_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
 
        src = df["contract1"].astype(int).tolist()
        dst = df["contract2"].astype(int).tolist()
        weight = torch.tensor(
            df["weight"].astype(float).values,
            dtype=torch.float
        ).unsqueeze(-1)
 
        edge_index = torch.tensor([src, dst], dtype=torch.long)
 
        max_idx = max(max(src), max(dst))
        contract_ids = [number_to_contract.get(i, str(i)) for i in range(max_idx + 1)]
 
        graph = Data(
            edge_index=edge_index,
            edge_attr=weight,
            num_nodes=len(contract_ids),
        )
 
        return {
            "graph": graph,
            "contract_ids": contract_ids,
            "id_to_idx": {cid: i for i, cid in enumerate(contract_ids)},
            "contract_to_number": contract_to_number,
            "number_to_contract": number_to_contract,
        }
 

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> GlobalGraphData:
        result = GlobalGraphData()

        # 1) meta.json
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                result.meta = json.load(f)
            log.info(f"[GlobalGraphLoader] Loaded meta from {meta_path}")

        # 2) node_features.pt → {contract_id: tensor}
        feat_path = self.root / "node_features.pt"
        if feat_path.exists():
            feat_obj = torch.load(str(feat_path), map_location="cpu")
            if isinstance(feat_obj, dict):
                result.node_features = {
                    (k.lower() if self.normalize else k): v
                    for k, v in feat_obj.items()
                }
            log.info(
                f"[GlobalGraphLoader] Loaded node_features "
                f"for {len(result.node_features)} contracts"
            )

        # 3) edges.pt → torch_geometric Data (preferred)
        edges_pt = self.root / "edges.pt"
        if edges_pt.exists():
            obj = torch.load(str(edges_pt), map_location="cpu")
            if isinstance(obj, Data):
                result.graph = obj
            elif isinstance(obj, dict):
                result.graph = Data(**obj)
            log.info(f"[GlobalGraphLoader] Loaded graph from {edges_pt}")

            # contract_ids 복원 (graph에 ids 속성이 있으면 사용)
            if hasattr(result.graph, "contract_ids") and result.graph.contract_ids:
                ids = result.graph.contract_ids
                result.contract_ids = list(ids)
            elif result.node_features:
                result.contract_ids = list(result.node_features.keys())
            elif result.graph.num_nodes is not None:
                result.contract_ids = [str(i) for i in range(result.graph.num_nodes)]

        # 4) edges.csv → 없으면 csv에서 Data 생성
        elif (self.root / "edges.csv").exists():
            result.graph, result.contract_ids = self._build_from_csv()
            log.info(
                f"[GlobalGraphLoader] Built graph from edges.csv "
                f"({len(result.contract_ids)} nodes)"
            )

        # contract_id -> idx 매핑
        result.id_to_idx = {
            cid: idx for idx, cid in enumerate(result.contract_ids)
        }

        # node feature matrix를 graph.x에 통합
        if result.graph is not None and result.graph.x is None and result.node_features:
            result.graph.x = self._build_feature_matrix(
                result.contract_ids, result.node_features
            )

        log.info(
            f"[GlobalGraphLoader] GlobalGraph: "
            f"nodes={result.num_nodes()}, edges={result.num_edges()}"
        )
        return result

    def _build_from_csv(self) -> Tuple[Data, List[str]]:
        csv_path = self.root / "edges.csv"
        df = pd.read_csv(str(csv_path))
        df.columns = [c.strip().lower() for c in df.columns]

        src_col = self._find_col(df, ("src_contract", "src", "source", "from"))
        dst_col = self._find_col(df, ("dst_contract", "dst", "target", "to"))

        all_nodes = sorted(
            set(df[src_col].tolist()) | set(df[dst_col].tolist())
        )
        if self.normalize:
            all_nodes = [str(n).lower() for n in all_nodes]
        node2idx = {n: i for i, n in enumerate(all_nodes)}

        src_idx = [node2idx[str(n).lower() if self.normalize else str(n)]
                   for n in df[src_col]]
        dst_idx = [node2idx[str(n).lower() if self.normalize else str(n)]
                   for n in df[dst_col]]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        edge_attr = None
        if "weight" in df.columns:
            edge_attr = torch.tensor(df["weight"].values, dtype=torch.float).unsqueeze(-1)
        elif "edge_type" in df.columns:
            # edge_type을 정수로 인코딩
            types = df["edge_type"].astype("category").cat.codes.values
            edge_attr = torch.tensor(types, dtype=torch.float).unsqueeze(-1)

        graph = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(all_nodes),
        )
        return graph, all_nodes

    @staticmethod
    def _build_feature_matrix(
        ids: List[str],
        features: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        vecs = [features[cid] for cid in ids if cid in features]
        if not vecs:
            return None
        return torch.stack(vecs, dim=0)

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"None of {candidates} found in CSV columns: {list(df.columns)}"
        )
