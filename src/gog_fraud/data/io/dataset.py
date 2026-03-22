# src/gog_fraud/data/io/dataset.py
"""
세 loader를 묶는 통합 Façade.

Usage
-----
ds = FraudDataset.from_config(cfg)   # config dict 또는 DictConfig
ds.load()                            # 전체 로드

# contract_id 기준 정렬된 데이터 접근
graphs  = ds.transaction_graphs      # List[TransactionGraph]
labels  = ds.labels                  # {contract_id: int}
splits  = ds.splits                  # {contract_id: "train"/"valid"/"test"}
gg      = ds.global_graph            # GlobalGraphData

# split-aware 반환
train_ids = ds.split_ids("train")
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .global_graph_loader import GlobalGraphData, GlobalGraphLoader
from .label_loader import LabelLoader, LabelRecord
from .transaction_loader import TransactionGraph, TransactionLoader

from typing import Optional, List
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    transactions_root: str = "../_data/dataset/transactions"
    labels_path: str = "../_data/dataset/labels.csv"
    global_graph_root: str = "../_data/dataset/global_graph"
 
    # None이면 TransactionLoader가 자동으로 transactions_root 옆에 결정
    cache_root: Optional[str] = None
 
    chain: str = "polygon"
    auto_split: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    split_seed: int = 42
    normalize_address: bool = True
 
    label_chain_col: Optional[str] = "Chain"
    label_address_col: Optional[str] = "Contract"
    label_label_col: Optional[str] = "Category"
    label_split_col: Optional[str] = None
 
    normal_categories: List[int] = field(default_factory=lambda: [0])
    fraud_categories: Optional[List[int]] = None
    load_global_graph: bool = False




# ---------------------------------------------------------------------------
# 통합 Façade
# ---------------------------------------------------------------------------

class FraudDataset:
    """
    transactions / labels / global_graph 세 소스를 한 번에 로드하고
    contract_id 기준으로 정렬·join해 사용 가능한 형태로 제공한다.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

        # ✅ 수정: cache_root 명시 전달
        self._tx_loader = TransactionLoader(
            root=self.cfg.transactions_root,
            chain=self.cfg.chain,
            cache_root=self.cfg.cache_root,   # ← 이 줄이 핵심
            chunk_size=100,
        )
        
        self._lbl_loader = LabelLoader(
            path=cfg.labels_path,
            chain=cfg.chain,
            chain_col=cfg.label_chain_col,
            address_col=cfg.label_address_col,
            label_col=cfg.label_label_col,
            split_col=cfg.label_split_col,
            normalize_address=cfg.normalize_address,
            normal_categories=cfg.normal_categories,
            fraud_categories=cfg.fraud_categories,
        )

        self._gg_loader = GlobalGraphLoader(
            root=cfg.global_graph_root,
            normalize_address=cfg.normalize_address,
            chain=cfg.chain,
        )

        # 로드 후 채워지는 필드
        self.transaction_graphs: List[TransactionGraph] = []
        self.labels: Dict[str, int] = {}
        self.splits: Dict[str, str] = {}
        self.global_graph: Optional[GlobalGraphData] = None

        self._loaded = False

    @classmethod
    def from_config(cls, cfg) -> "FraudDataset":
        """dict / DictConfig / DatasetConfig 모두 지원."""
        if isinstance(cfg, DatasetConfig):
            return cls(cfg)
        if isinstance(cfg, dict):
            return cls(DatasetConfig(**cfg))
        # OmegaConf DictConfig 등
        return cls(DatasetConfig(**dict(cfg)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "FraudDataset":
        if self._loaded:
            return self

        log.info("[FraudDataset] Loading transaction graphs …")
        self.transaction_graphs = self._tx_loader.load_all()

        log.info("[FraudDataset] Loading labels …")
        records = self._lbl_loader.load()
        self.labels = {r.contract_id: r.label for r in records}

        has_split = any(r.split is not None for r in records)
        if has_split:
            self.splits = {r.contract_id: r.split for r in records if r.split is not None}
        elif self.cfg.auto_split:
            self.splits = self._auto_split(records)

        if self.cfg.load_global_graph:
            log.info("[FraudDataset] Loading global graph …")
            self.global_graph = self._gg_loader.load()
        else:
            log.info("[FraudDataset] Skipping global graph load.")
            self.global_graph = None

        self._loaded = True
        return self



    def split_ids(self, split: str) -> List[str]:
        """특정 split에 속한 contract_id 목록."""
        return [cid for cid, s in self.splits.items() if s == split]

    def split_graphs(self, split: str) -> List[TransactionGraph]:
        """특정 split의 TransactionGraph 목록."""
        ids = set(self.split_ids(split))
        return [g for g in self.transaction_graphs if g.contract_id in ids]

    def get_labels_tensor(
        self,
        contract_ids: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        주어진 contract_ids 순서대로 label tensor와
        실제로 label이 있는 id 목록을 반환.
        """
        valid_ids = [cid for cid in contract_ids if cid in self.labels]
        labels = torch.tensor(
            [self.labels[cid] for cid in valid_ids],
            dtype=torch.float,
        )
        return labels, valid_ids

    def fraud_rate(self, split: Optional[str] = None) -> float:
        if split:
            ids = self.split_ids(split)
            lbls = [self.labels[i] for i in ids if i in self.labels]
        else:
            lbls = list(self.labels.values())
        return sum(lbls) / max(len(lbls), 1)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "FraudDataset Summary",
            f"  Chain         : {self.cfg.chain}",
            f"  TX Graphs     : {len(self.transaction_graphs)}",
            f"  Labels        : {len(self.labels)} "
            f"(fraud_rate={self.fraud_rate():.2%})",
            f"  Global Graph  : nodes={self.global_graph.num_nodes() if self.global_graph else 0}, "
            f"edges={self.global_graph.num_edges() if self.global_graph else 0}",
        ]
        for s in ("train", "valid", "test"):
            n = len(self.split_ids(s))
            fr = self.fraud_rate(s)
            lines.append(f"  {s:>5} split : {n} samples, fraud_rate={fr:.2%}")
        lines.append("=" * 50)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_split(self, records: List[LabelRecord]) -> Dict[str, str]:
        """train/val/test 자동 split (stratified)."""
        rng = random.Random(self.cfg.split_seed)

        fraud_ids  = [r.contract_id for r in records if r.label == 1]
        normal_ids = [r.contract_id for r in records if r.label == 0]

        splits: Dict[str, str] = {}
        for ids in (fraud_ids, normal_ids):
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(n * self.cfg.train_ratio)
            n_val   = int(n * self.cfg.val_ratio)
            for cid in ids[:n_train]:
                splits[cid] = "train"
            for cid in ids[n_train: n_train + n_val]:
                splits[cid] = "valid"
            for cid in ids[n_train + n_val:]:
                splits[cid] = "test"
        return splits

    def _split_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.splits.values():
            counts[s] = counts.get(s, 0) + 1
        return counts
