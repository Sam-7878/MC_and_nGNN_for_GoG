# src/gog_fraud/adapters/legacy_adapter.py
"""
DOMINANT / DONE / GAE / AnomalyDAE / CoLA의 출력 score를
contract 단위로 정규화하는 래퍼.

모든 pygod detector는 node-level score를 반환하므로,
graph/contract 단위로 집계(aggregate)하는 과정이 필요하다.

지원 집계 방식
  - "max"   : 그래프 내 노드 score의 최대값
  - "mean"  : 평균
  - "sum"   : 합산
  - "topk"  : 상위 K개 평균
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)

AggMethod = Literal["max", "mean", "sum", "topk"]
SUPPORTED_MODELS = ("DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class LegacyAdapterConfig:
    model_name: str         = "DOMINANT"
    agg_method: AggMethod   = "max"   # contract 단위 집계 방식
    topk: int               = 3       # agg_method="topk" 일 때만 사용
    normalize_score: bool   = True    # [0, 1]로 min-max 정규화
    gpu: int                = -1      # -1=CPU, 0=GPU:0, ...
    # pygod detector 공통 하이퍼파라미터
    hid_dim: int            = 64
    num_layers: int         = 2
    epoch: int              = 100
    lr: float               = 0.003
    dropout: float          = 0.0
    weight_decay: float     = 0.0


# ---------------------------------------------------------------------------
# score 집계 함수
# ---------------------------------------------------------------------------

def _aggregate_scores(
    node_scores: np.ndarray,
    method: AggMethod,
    topk: int = 3,
) -> float:
    """단일 그래프의 node score → contract score."""
    if len(node_scores) == 0:
        return 0.0
    if method == "max":
        return float(node_scores.max())
    elif method == "mean":
        return float(node_scores.mean())
    elif method == "sum":
        return float(node_scores.sum())
    elif method == "topk":
        k = min(topk, len(node_scores))
        top = np.sort(node_scores)[::-1][:k]
        return float(top.mean())
    else:
        raise ValueError(f"Unknown agg_method: {method}")


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Legacy Model Runner
# ---------------------------------------------------------------------------

class LegacyModelRunner:
    """
    단일 pygod detector 모델을 주어진 graph 리스트에 대해 실행하고
    contract 단위 score를 반환한다.

    Usage
    -----
    runner = LegacyModelRunner(cfg=LegacyAdapterConfig(model_name="DOMINANT"))
    scores = runner.run(graphs)   # {contract_id: float}
    """

    def __init__(self, cfg: LegacyAdapterConfig) -> None:
        self.cfg = cfg
        self._model = None

    def _build_model(self):
        """pygod detector 인스턴스 생성."""
        try:
            from pygod.detector import (
                AnomalyDAE,
                CoLA,
                DOMINANT,
                DONE,
                GAE,
            )
        except ImportError as e:
            raise ImportError(
                "pygod is not installed. "
                "Run: pip install pygod"
            ) from e

        common = dict(
            hid_dim    = self.cfg.hid_dim,
            num_layers = self.cfg.num_layers,
            epoch      = self.cfg.epoch,
            lr         = self.cfg.lr,
            dropout    = self.cfg.dropout,
            weight_decay = self.cfg.weight_decay,
            gpu        = self.cfg.gpu,
            verbose    = 0,
        )

        model_map = {
            "DOMINANT":   DOMINANT,
            "DONE":       DONE,
            "GAE":        GAE,
            "AnomalyDAE": AnomalyDAE,
            "CoLA":       CoLA,
        }

        name = self.cfg.model_name.upper()
        if name not in model_map:
            raise ValueError(
                f"Unknown model: {name}. "
                f"Supported: {list(model_map.keys())}"
            )

        # 모델별 지원 파라미터가 다를 수 있으므로 방어적으로 생성
        try:
            return model_map[name](**common)
        except TypeError:
            # 일부 모델은 일부 파라미터 미지원
            safe = {k: v for k, v in common.items()
                    if k in ("hid_dim", "num_layers", "epoch", "lr", "gpu", "verbose")}
            return model_map[name](**safe)

    def run(
        self,
        transaction_graphs,   # List[TransactionGraph]
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        각 TransactionGraph를 독립적으로 detector에 입력하고
        contract_id -> fraud_score 매핑을 반환.
        """
        from gog_fraud.data.io.transaction_loader import TransactionGraph

        contract_scores: Dict[str, float] = {}
        n = len(transaction_graphs)

        for i, tg in enumerate(transaction_graphs):
            if verbose and (i % max(1, n // 10) == 0):
                log.info(
                    f"[LegacyRunner:{self.cfg.model_name}] "
                    f"{i}/{n} ({i/max(n,1):.0%})"
                )

            try:
                score = self._run_single(tg.graph)
                contract_scores[tg.contract_id] = score
            except Exception as exc:
                log.warning(
                    f"[LegacyRunner] Skip contract={tg.contract_id}: {exc}"
                )
                contract_scores[tg.contract_id] = 0.0

        # 전체 정규화
        if self.cfg.normalize_score and contract_scores:
            arr = np.array(list(contract_scores.values()))
            arr = _minmax_normalize(arr)
            for cid, v in zip(contract_scores.keys(), arr):
                contract_scores[cid] = float(v)

        log.info(
            f"[LegacyRunner:{self.cfg.model_name}] Done. "
            f"Scored {len(contract_scores)} contracts."
        )
        return contract_scores

    def _run_single(self, graph: Data) -> float:
        """단일 그래프에 대해 node score 추출 → 집계."""
        model = self._build_model()   # 매번 새 모델 (그래프마다 독립 학습)

        # pygod는 torch_geometric Data를 직접 입력받음
        model.fit(graph)
        node_scores = model.decision_score_   # np.ndarray shape (N,)

        return _aggregate_scores(
            node_scores,
            method=self.cfg.agg_method,
            topk=self.cfg.topk,
        )


# ---------------------------------------------------------------------------
# 다중 모델 동시 실행 (batch runner)
# ---------------------------------------------------------------------------

class LegacyBatchRunner:
    """
    DOMINANT, DONE, GAE, AnomalyDAE, CoLA 를 한 번에 실행하는 편의 클래스.

    Usage
    -----
    batch = LegacyBatchRunner(
        model_names=["DOMINANT", "DONE"],
        base_cfg=LegacyAdapterConfig(),
    )
    all_scores = batch.run_all(graphs)
    # all_scores["DOMINANT"] = {contract_id: float}
    """

    DEFAULT_MODELS = list(SUPPORTED_MODELS)

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        base_cfg: Optional[LegacyAdapterConfig] = None,
    ) -> None:
        self.model_names = model_names or self.DEFAULT_MODELS
        self.base_cfg = base_cfg or LegacyAdapterConfig()

    def run_all(
        self,
        transaction_graphs,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns
        -------
        {model_name: {contract_id: score}}
        """
        results: Dict[str, Dict[str, float]] = {}
        for name in self.model_names:
            log.info(f"\n[LegacyBatchRunner] === Running {name} ===")
            cfg = LegacyAdapterConfig(
                model_name=name,
                agg_method=self.base_cfg.agg_method,
                topk=self.base_cfg.topk,
                normalize_score=self.base_cfg.normalize_score,
                gpu=self.base_cfg.gpu,
                hid_dim=self.base_cfg.hid_dim,
                num_layers=self.base_cfg.num_layers,
                epoch=self.base_cfg.epoch,
                lr=self.base_cfg.lr,
            )
            runner = LegacyModelRunner(cfg)
            results[name] = runner.run(transaction_graphs, verbose=verbose)
        return results
