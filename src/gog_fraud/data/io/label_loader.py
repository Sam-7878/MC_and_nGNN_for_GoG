# src/gog_fraud/data/io/label_loader.py
"""
../_data/dataset/labels.csv 를 읽어 contract_id -> label 매핑을 반환.

CSV 예시:
  contract_address,label,split
  0xabc...,1,train
  0xdef...,0,test
  ...

- label: 1=fraud / 0=normal
- split: train / valid / test (없으면 자동 생성)
"""

# src/gog_fraud/data/io/label_loader.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class LabelRecord:
    contract_id: str
    label: int
    split: Optional[str] = None
    meta: Dict = field(default_factory=dict)


class LabelLoader:
    def __init__(
        self,
        path: str,
        chain: Optional[str] = None,
        chain_col: Optional[str] = "Chain",
        address_col: Optional[str] = "Contract",
        label_col: Optional[str] = "Category",
        split_col: Optional[str] = None,
        normalize_address: bool = True,
        normal_categories: Optional[List[int]] = None,
        fraud_categories: Optional[List[int]] = None,
    ) -> None:
        self.path = Path(path)
        self.chain = chain.lower() if chain else None
        self.chain_col = chain_col.lower() if chain_col else None
        self.address_col = address_col.lower() if address_col else None
        self.label_col = label_col.lower() if label_col else None
        self.split_col = split_col.lower() if split_col else None
        self.normalize_address = normalize_address
        self.normal_categories = set(normal_categories or [0])
        self.fraud_categories = set(fraud_categories) if fraud_categories else None

        if not self.path.exists():
            raise FileNotFoundError(f"labels.csv not found: {self.path}")

        self._records: Optional[List[LabelRecord]] = None

    def load(self) -> List[LabelRecord]:
        if self._records is None:
            self._records = self._parse()
        return self._records

    def _to_binary_label(self, category: int) -> int:
        if self.fraud_categories is not None:
            return int(category in self.fraud_categories)
        return int(category not in self.normal_categories)

    def _parse(self) -> List[LabelRecord]:
        df = pd.read_csv(str(self.path))
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

        required = [self.address_col, self.label_col]
        for col in required:
            if col is None or col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in labels.csv. "
                    f"Available columns: {list(df.columns)}"
                )

        if self.chain and self.chain_col:
            if self.chain_col not in df.columns:
                raise ValueError(
                    f"Chain column '{self.chain_col}' not found in labels.csv. "
                    f"Available columns: {list(df.columns)}"
                )
            df = df[df[self.chain_col].astype(str).str.lower() == self.chain].copy()

        records: List[LabelRecord] = []
        for _, row in df.iterrows():
            cid = str(row[self.address_col]).strip()
            if self.normalize_address:
                cid = cid.lower()

            category = int(row[self.label_col])
            label = self._to_binary_label(category)

            split = None
            if self.split_col and self.split_col in df.columns:
                split = str(row[self.split_col]).strip().lower()

            records.append(
                LabelRecord(
                    contract_id=cid,
                    label=label,
                    split=split,
                    meta={"category": category},
                )
            )

        log.info(
            f"[LabelLoader] Loaded {len(records)} records for chain={self.chain}"
        )
        return records



    @staticmethod
    def _detect_col(
        df: pd.DataFrame,
        override: Optional[str],
        candidates: tuple,
        default: Optional[str],
    ) -> Optional[str]:
        if override:
            if override.lower() in df.columns:
                return override.lower()
            raise ValueError(
                f"Specified column '{override}' not found in CSV. "
                f"Available: {list(df.columns)}"
            )
        for c in candidates:
            if c in df.columns:
                return c
        if default and default in df.columns:
            return default
        return None
