# src/gog_fraud/evaluation/benchmark.py
"""
Benchmark metric 계산 및 결과 정리.

지원 Metric:
  - ROC-AUC
  - PR-AUC
  - F1 / Precision / Recall
  - Confusion Matrix
  - Ranking Metrics (NDCG@K, Precision@K, Recall@K)
  - 신뢰구간 (bootstrap)
"""

from __future__ import annotations

import logging
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model_name: str
    setting: str                         # "strict" / "full_system" 등

    # 주요 metric
    roc_auc: float = 0.0
    pr_auc: float  = 0.0
    f1: float      = 0.0
    precision: float = 0.0
    recall: float    = 0.0
    specificity: float = 0.0
    accuracy: float    = 0.0
    bce_loss: float    = 0.0

    # confusion matrix 원소
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # ranking metric
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)   # {K: value}
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)

    # 신뢰구간 (bootstrap)
    roc_auc_ci: Optional[Tuple[float, float]] = None
    pr_auc_ci:  Optional[Tuple[float, float]] = None

    # 메타
    n_samples: int = 0
    fraud_rate: float = 0.0
    threshold: float  = 0.5
    extra: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        d = {
            "model":       self.model_name,
            "setting":     self.setting,
            "roc_auc":     round(self.roc_auc,   4),
            "pr_auc":      round(self.pr_auc,    4),
            "f1":          round(self.f1,         4),
            "precision":   round(self.precision,  4),
            "recall":      round(self.recall,     4),
            "specificity": round(self.specificity,4),
            "accuracy":    round(self.accuracy,   4),
            "bce_loss":    round(self.bce_loss,   4),
            "tp": self.tp, "tn": self.tn,
            "fp": self.fp, "fn": self.fn,
            "n_samples":   self.n_samples,
            "fraud_rate":  round(self.fraud_rate, 4),
            "threshold":   self.threshold,
        }
        for k, v in self.ndcg_at_k.items():
            d[f"ndcg@{k}"] = round(v, 4)
        for k, v in self.precision_at_k.items():
            d[f"precision@{k}"] = round(v, 4)
        for k, v in self.recall_at_k.items():
            d[f"recall@{k}"] = round(v, 4)
        if self.roc_auc_ci:
            d["roc_auc_ci_lo"] = round(self.roc_auc_ci[0], 4)
            d["roc_auc_ci_hi"] = round(self.roc_auc_ci[1], 4)
        if self.pr_auc_ci:
            d["pr_auc_ci_lo"] = round(self.pr_auc_ci[0], 4)
            d["pr_auc_ci_hi"] = round(self.pr_auc_ci[1], 4)
        return d

    def __str__(self) -> str:
        lines = [
            f"[{self.model_name}] ({self.setting})",
            f"  ROC-AUC={self.roc_auc:.4f}  PR-AUC={self.pr_auc:.4f}",
            f"  F1={self.f1:.4f}  Prec={self.precision:.4f}  Rec={self.recall:.4f}",
            f"  Acc={self.accuracy:.4f}  Spec={self.specificity:.4f}",
            f"  TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}",
            f"  n={self.n_samples}  fraud_rate={self.fraud_rate:.2%}",
        ]
        if self.ndcg_at_k:
            ndcg_str = "  NDCG@K: " + ", ".join(
                f"@{k}={v:.4f}" for k, v in sorted(self.ndcg_at_k.items())
            )
            lines.append(ndcg_str)
        if self.roc_auc_ci:
            lines.append(
                f"  ROC-AUC 95% CI=({self.roc_auc_ci[0]:.4f}, {self.roc_auc_ci[1]:.4f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def _ensure_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# 핵심 metric 함수
# ---------------------------------------------------------------------------

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.0
    from sklearn.metrics import roc_auc_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return 0.0
    from sklearn.metrics import average_precision_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_true, y_score))


def compute_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Returns (TP, TN, FP, FN)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def compute_ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """NDCG@K: anomaly/fraud 탐지 ranking 성능."""
    n = len(y_score)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)

    sorted_idx = np.argsort(y_score)[::-1]
    top_k_labels = y_true[sorted_idx[:k]]

    # DCG
    dcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(top_k_labels)
    )

    # Ideal DCG (perfect ranking: all frauds first)
    n_fraud = int(y_true.sum())
    ideal_k = min(k, n_fraud)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_k))

    return _safe_div(dcg, idcg)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    n = len(y_score)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    sorted_idx = np.argsort(y_score)[::-1]
    top_k = y_true[sorted_idx[:k]]
    return float(top_k.sum()) / k


def compute_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    n = len(y_score)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    n_fraud = int(y_true.sum())
    if n_fraud == 0:
        return 0.0
    sorted_idx = np.argsort(y_score)[::-1]
    top_k = y_true[sorted_idx[:k]]
    return float(top_k.sum()) / n_fraud


# ---------------------------------------------------------------------------
# Bootstrap 신뢰구간
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        try:
            values.append(metric_fn(yt, ys))
        except Exception:
            pass

    if not values:
        return 0.0, 0.0

    alpha = (1 - ci) / 2
    lo = float(np.quantile(values, alpha))
    hi = float(np.quantile(values, 1 - alpha))
    return lo, hi


# ---------------------------------------------------------------------------
# 통합 benchmark 평가
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    y_true,
    y_score,
    model_name: str,
    setting: str,
    threshold: float = 0.5,
    k_list: List[int] = (10, 20, 50),
    bootstrap: bool = True,
    n_bootstrap: int = 500,
    bootstrap_seed: int = 42,
) -> BenchmarkResult:
    """
    단일 모델의 예측 결과를 받아 BenchmarkResult를 반환.

    Parameters
    ----------
    y_true  : shape (N,), {0, 1}
    y_score : shape (N,), [0, 1] 확률 (또는 anomaly score)
    """
    yt = _ensure_numpy(y_true).astype(np.float32)
    ys = _ensure_numpy(y_score).astype(np.float32)
    yp = (ys >= threshold).astype(np.int32)

    n = len(yt)
    fraud_rate = float(yt.mean()) if n > 0 else 0.0

    # Threshold-based
    tp, tn, fp, fn = compute_confusion(yt.astype(int), yp)
    precision   = _safe_div(tp, tp + fp)
    recall      = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy    = _safe_div(tp + tn, n)
    f1          = _safe_div(2 * precision * recall, precision + recall)

    # Threshold-free
    roc_auc = compute_roc_auc(yt, ys)
    pr_auc  = compute_pr_auc(yt, ys)

    # BCE
    try:
        ys_c = np.clip(ys, 1e-6, 1 - 1e-6).astype(np.float64)
        yt_c = yt.astype(np.float64)
        bce = float(-np.mean(
            yt_c * np.log(ys_c) + (1 - yt_c) * np.log(1 - ys_c)
        ))
    except Exception:
        bce = 0.0

    # Ranking metrics
    ndcg_at_k = {k: compute_ndcg_at_k(yt, ys, k)      for k in k_list}
    prec_at_k  = {k: compute_precision_at_k(yt, ys, k) for k in k_list}
    rec_at_k   = {k: compute_recall_at_k(yt, ys, k)    for k in k_list}

    # Bootstrap CI
    roc_ci, pr_ci = None, None
    if bootstrap and n >= 20:
        roc_ci = bootstrap_ci(yt, ys, compute_roc_auc, n_bootstrap, seed=bootstrap_seed)
        pr_ci  = bootstrap_ci(yt, ys, compute_pr_auc,  n_bootstrap, seed=bootstrap_seed)

    return BenchmarkResult(
        model_name=model_name,
        setting=setting,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        f1=f1,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=accuracy,
        bce_loss=bce,
        tp=tp, tn=tn, fp=fp, fn=fn,
        ndcg_at_k=ndcg_at_k,
        precision_at_k=prec_at_k,
        recall_at_k=rec_at_k,
        roc_auc_ci=roc_ci,
        pr_auc_ci=pr_ci,
        n_samples=n,
        fraud_rate=fraud_rate,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# 결과 집계 및 출력
# ---------------------------------------------------------------------------

class BenchmarkTable:
    """
    여러 BenchmarkResult를 한 테이블로 정리.

    Usage
    -----
    table = BenchmarkTable()
    table.add(result_dominant)
    table.add(result_level1)
    print(table.to_markdown())
    table.to_csv("results/benchmark.csv")
    table.to_json("results/benchmark.json")
    """

    def __init__(self) -> None:
        self.results: List[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> "BenchmarkTable":
        self.results.append(result)
        return self

    def to_markdown(self, k_list: List[int] = (10, 50)) -> str:
        if not self.results:
            return "(empty)"

        header = (
            "| Model | Setting | ROC-AUC | PR-AUC | F1 | Prec | Rec "
            "| Acc | N | FraudRate |"
        )
        sep = "|" + "|".join(["---"] * 9) + "|"
        rows = [header, sep]

        for r in self.results:
            row = (
                f"| {r.model_name} "
                f"| {r.setting} "
                f"| {r.roc_auc:.4f} "
                f"| {r.pr_auc:.4f} "
                f"| {r.f1:.4f} "
                f"| {r.precision:.4f} "
                f"| {r.recall:.4f} "
                f"| {r.accuracy:.4f} "
                f"| {r.n_samples} "
                f"| {r.fraud_rate:.2%} |"
            )
            rows.append(row)

        return "\n".join(rows)

    def to_csv(self, path: str) -> None:
        import csv
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self.results]
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"[BenchmarkTable] Saved CSV → {path}")

    def to_json(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        log.info(f"[BenchmarkTable] Saved JSON → {path}")

    def best(self, metric: str = "pr_auc") -> Optional[BenchmarkResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: getattr(r, metric, 0.0))

    def print_all(self) -> None:
        print("\n" + "=" * 60)
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        for r in self.results:
            print(r)
            print("-" * 60)
        best = self.best("pr_auc")
        if best:
            print(f"\n★ Best PR-AUC: [{best.model_name}] = {best.pr_auc:.4f}")
        print("=" * 60 + "\n")
