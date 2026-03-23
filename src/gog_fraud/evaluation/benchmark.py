# src/gog_fraud/evaluation/benchmark.py
"""
NaN-safe benchmark evaluation module.

Public API (expected by run_fraud_benchmark.py):
    BenchmarkResult   - single model × single setting 결과 dataclass
    BenchmarkTable    - 여러 BenchmarkResult 누적 / 출력 / CSV 저장
    evaluate_benchmark(...) -> BenchmarkResult
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# sklearn imports – all wrapped to avoid cascade failures
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)


# ============================================================
# 1. BenchmarkResult
# ============================================================

@dataclass
class BenchmarkResult:
    """
    한 모델 × 한 setting의 평가 결과.

    run_fraud_benchmark.py 는 이 필드들을 직접 접근합니다:
        result.roc_auc
        result.pr_auc
        result.best_f1
        result.best_threshold
        result.f1_at_05
        result.precision_at_05
        result.recall_at_05
        result.accuracy_at_05
        result.p_at_k   (dict: {k: float})
        result.r_at_k   (dict: {k: float})
        result.num_samples
        result.num_pos
        result.num_neg
        result.num_dropped
        result.model_name
        result.setting
        result.ci_roc_auc  (Optional bootstrap 구간)
        result.ci_pr_auc   (Optional bootstrap 구간)
    """

    # ─── Identity ────────────────────────────────────────
    model_name: str = ""
    setting: str = ""

    # ─── Core metrics ────────────────────────────────────
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")

    # Best-F1 (from PR curve sweep)
    best_f1: float = float("nan")
    best_threshold: float = float("nan")

    # Fixed-threshold metrics (default threshold=0.5)
    f1_at_05: float = float("nan")
    precision_at_05: float = float("nan")
    recall_at_05: float = float("nan")
    accuracy_at_05: float = float("nan")

    # Precision/Recall @ K (dict keyed by k)
    p_at_k: Dict[int, float] = field(default_factory=dict)
    r_at_k: Dict[int, float] = field(default_factory=dict)

    # ─── Data stats ──────────────────────────────────────
    num_samples: int = 0
    num_pos: int = 0
    num_neg: int = 0
    num_dropped: int = 0

    # ─── Bootstrap confidence intervals ──────────────────
    ci_roc_auc: Optional[Tuple[float, float]] = None
    ci_pr_auc: Optional[Tuple[float, float]] = None

    # ─── Serialization ───────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten p_at_k / r_at_k for CSV export
        for k, v in self.p_at_k.items():
            d[f"p@{k}"] = v
        for k, v in self.r_at_k.items():
            d[f"r@{k}"] = v
        # Flatten CI tuples
        if self.ci_roc_auc is not None:
            d["ci_roc_auc_lo"] = self.ci_roc_auc[0]
            d["ci_roc_auc_hi"] = self.ci_roc_auc[1]
        if self.ci_pr_auc is not None:
            d["ci_pr_auc_lo"] = self.ci_pr_auc[0]
            d["ci_pr_auc_hi"] = self.ci_pr_auc[1]
        return d

    def pretty(self) -> str:
        sep = "─" * 52
        lines = [
            sep,
            f"  Model   : {self.model_name}",
            f"  Setting : {self.setting}",
            sep,
            f"  ROC-AUC : {self.roc_auc:.4f}"
            + (
                f"  [{self.ci_roc_auc[0]:.4f}, {self.ci_roc_auc[1]:.4f}]"
                if self.ci_roc_auc else ""
            ),
            f"  PR-AUC  : {self.pr_auc:.4f}"
            + (
                f"  [{self.ci_pr_auc[0]:.4f}, {self.ci_pr_auc[1]:.4f}]"
                if self.ci_pr_auc else ""
            ),
            f"  Best-F1 : {self.best_f1:.4f}  (thr={self.best_threshold:.4f})",
            f"  F1@0.5  : {self.f1_at_05:.4f}  "
            f"P={self.precision_at_05:.4f}  R={self.recall_at_05:.4f}",
            f"  ACC@0.5 : {self.accuracy_at_05:.4f}",
        ]
        if self.p_at_k:
            pk_str = "  ".join(
                f"P@{k}={v:.4f}" for k, v in sorted(self.p_at_k.items())
            )
            lines.append(f"  {pk_str}")
        if self.r_at_k:
            rk_str = "  ".join(
                f"R@{k}={v:.4f}" for k, v in sorted(self.r_at_k.items())
            )
            lines.append(f"  {rk_str}")
        lines += [
            sep,
            f"  Samples : {self.num_samples}  "
            f"(+{self.num_pos}  -{self.num_neg}  dropped={self.num_dropped})",
            sep,
        ]
        return "\n".join(lines)


# ============================================================
# 2. BenchmarkTable
# ============================================================

class BenchmarkTable:
    """
    여러 BenchmarkResult를 누적하고
    콘솔 출력 / CSV 저장을 제공.

    run_fraud_benchmark.py 사용 패턴:
        table = BenchmarkTable()
        table.add(result)
        table.print_summary()
        table.save_csv(output_dir / "benchmark_results.csv")
    """

    def __init__(self) -> None:
        self._results: List[BenchmarkResult] = []

    # ─── Mutation ────────────────────────────────────────

    def add(self, result: BenchmarkResult) -> None:
        self._results.append(result)

    def add_all(self, results: Iterable[BenchmarkResult]) -> None:
        for r in results:
            self._results.append(r)

    # ─── Access ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def results(self) -> List[BenchmarkResult]:
        return list(self._results)

    def get_by_model(self, model_name: str) -> List[BenchmarkResult]:
        return [r for r in self._results if r.model_name == model_name]

    def get_by_setting(self, setting: str) -> List[BenchmarkResult]:
        return [r for r in self._results if r.setting == setting]

    # ─── Output ──────────────────────────────────────────

    def print_summary(self) -> None:
        if not self._results:
            print("[BenchmarkTable] No results to display.")
            return

        header = (
            f"{'Model':<25} {'Setting':<18} "
            f"{'ROC-AUC':>8} {'PR-AUC':>8} "
            f"{'Best-F1':>8} {'F1@0.5':>8} "
            f"{'Pos':>5} {'Neg':>5} {'Drop':>5}"
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]

        for r in self._results:
            lines.append(
                f"{r.model_name:<25} {r.setting:<18} "
                f"{_fmt(r.roc_auc):>8} {_fmt(r.pr_auc):>8} "
                f"{_fmt(r.best_f1):>8} {_fmt(r.f1_at_05):>8} "
                f"{r.num_pos:>5} {r.num_neg:>5} {r.num_dropped:>5}"
            )

        lines.append(sep)
        print("\n".join(lines))

    def save_csv(self, path: str | Path) -> None:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._results:
            log.warning("[BenchmarkTable] save_csv called but no results exist.")
            return

        rows = [r.to_dict() for r in self._results]

        # Collect all keys in order
        all_keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})

        log.info(f"[BenchmarkTable] Saved {len(rows)} rows → {path}")


def _fmt(v: float, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  N/A"
    return f"{v:.{decimals}f}"


# ============================================================
# 3. Internal calculation helpers
# ============================================================

def _to_numpy_1d_int(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).reshape(-1)


def _to_numpy_1d_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _sanitize(
    y_true: Any,
    y_score: Any,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    핵심 NaN-safe 전처리.

    1. 배열로 변환
    2. 길이 검증
    3. y_score에서 non-finite 제거 (NaN/Inf)
    4. 제거 수(dropped) 반환
    """
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)

    if yt.shape[0] != ys.shape[0]:
        raise ValueError(
            f"[Benchmark] len(y_true)={yt.shape[0]} ≠ len(y_score)={ys.shape[0]}"
        )

    mask = np.isfinite(ys)
    dropped = int((~mask).sum())

    if dropped > 0:
        log.warning(
            f"[Benchmark] Dropping {dropped} non-finite y_score entries (NaN/Inf)."
        )

    return yt[mask], ys[mask], dropped


def _has_two_classes(yt: np.ndarray, metric_name: str = "") -> bool:
    if np.unique(yt).size < 2:
        log.warning(
            f"[Benchmark] {metric_name} undefined – y_true has only one class "
            f"(pos={int((yt==1).sum())}, neg={int((yt==0).sum())})."
        )
        return False
    return True


def _binarize(ys: np.ndarray, threshold: float) -> np.ndarray:
    return (ys >= threshold).astype(np.int64)


def _calc_roc_auc(yt: np.ndarray, ys: np.ndarray) -> float:
    if not _has_two_classes(yt, "ROC-AUC"):
        return float("nan")
    try:
        return float(roc_auc_score(yt, ys))
    except Exception as e:
        log.warning(f"[Benchmark] ROC-AUC failed: {e}")
        return float("nan")


def _calc_pr_auc(yt: np.ndarray, ys: np.ndarray) -> float:
    if not _has_two_classes(yt, "PR-AUC"):
        return float("nan")
    try:
        return float(average_precision_score(yt, ys))
    except Exception as e:
        log.warning(f"[Benchmark] PR-AUC failed: {e}")
        return float("nan")


def _calc_best_f1(
    yt: np.ndarray,
    ys: np.ndarray,
) -> Tuple[float, float]:
    """
    PR curve 전 구간에서 best F1과 그 threshold.
    """
    if not _has_two_classes(yt, "best-F1"):
        return float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, thresholds = precision_recall_curve(yt, ys)

        if len(thresholds) == 0:
            yp = _binarize(ys, threshold=0.5)
            return float(f1_score(yt, yp, zero_division=0)), 0.5

        # sklearn: precision/recall 배열은 threshold 배열보다 1 길다
        precision = precision[:-1]
        recall = recall[:-1]

        denom = precision + recall
        f1 = np.where(
            denom > 0,
            2 * precision * recall / denom,
            0.0,
        )

        best_idx = int(np.argmax(f1))
        return float(f1[best_idx]), float(thresholds[best_idx])

    except Exception as e:
        log.warning(f"[Benchmark] best-F1 failed: {e}")
        return float("nan"), float("nan")


def _calc_p_at_k(yt: np.ndarray, ys: np.ndarray, k: int) -> float:
    n = len(yt)
    if n == 0:
        return float("nan")
    k = max(1, min(k, n))
    order = np.argsort(-ys)
    topk = order[:k]
    tp = int((yt[topk] == 1).sum())
    return float(tp / k)


def _calc_r_at_k(yt: np.ndarray, ys: np.ndarray, k: int) -> float:
    n = len(yt)
    num_pos = int((yt == 1).sum())
    if n == 0 or num_pos == 0:
        return float("nan")
    k = max(1, min(k, n))
    order = np.argsort(-ys)
    topk = order[:k]
    tp = int((yt[topk] == 1).sum())
    return float(tp / num_pos)


# ============================================================
# 4. Bootstrap helpers
# ============================================================

def _bootstrap_ci(
    yt: np.ndarray,
    ys: np.ndarray,
    metric_fn,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """
    metric_fn(yt, ys) → float のbootstrap 신뢰구간.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(yt)
    boot_vals: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]
        ys_b = ys[idx]
        if np.unique(yt_b).size < 2:
            continue
        try:
            v = metric_fn(yt_b, ys_b)
            if np.isfinite(v):
                boot_vals.append(v)
        except Exception:
            pass

    if len(boot_vals) < 10:
        return float("nan"), float("nan")

    arr = np.array(boot_vals)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


# ============================================================
# 5. Main public API
# ============================================================

def evaluate_benchmark(
    y_true: Any,
    y_score: Any,
    *,
    model_name: str = "",
    setting: str = "",
    threshold: float = 0.5,
    k_list: List[int] | None = None,
    bootstrap: bool = False,
    n_boot: int = 200,
    bootstrap_alpha: float = 0.05,
    missing_score: float = 0.0,
    drop_missing: bool = False,
) -> BenchmarkResult:
    """
    평가 메인 함수.

    지원 입력 형태:
        A) y_true=list/array, y_score=list/array
        B) y_true=dict{contract_id->label}, y_score=dict{contract_id->score}

    Args:
        y_true         : 정답 레이블 (0/1)
        y_score        : 이상 탐지 점수 (높을수록 이상)
        model_name     : 결과 식별용 이름
        setting        : 실험 setting 이름 (예: "strict", "smoke")
        threshold      : 고정 threshold 기반 metric 계산용
        k_list         : P@K / R@K 계산할 K 목록 (None이면 [10, 20, 50])
        bootstrap      : Bootstrap CI 계산 여부
        n_boot         : Bootstrap 반복 횟수
        bootstrap_alpha: 신뢰구간 유의수준
        missing_score  : dict 입력 시 score 누락 계약에 채울 값
        drop_missing   : True면 score 누락 계약 제거

    Returns:
        BenchmarkResult
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError(
            "[Benchmark] scikit-learn is required but not installed."
        )

    if k_list is None:
        k_list = [10, 20, 50]

    # ─── Align dict inputs ───────────────────────────────
    if isinstance(y_true, Mapping) and isinstance(y_score, Mapping):
        yt_list, ys_list = _align_dict_inputs(
            y_true, y_score,
            missing_score=missing_score,
            drop_missing=drop_missing,
        )
        yt_raw = np.asarray(yt_list, dtype=np.int64)
        ys_raw = np.asarray(ys_list, dtype=np.float64)
    else:
        yt_raw = _to_numpy_1d_int(y_true)
        ys_raw = _to_numpy_1d_float(y_score)

    # ─── NaN-safe sanitize ───────────────────────────────
    yt, ys, dropped = _sanitize(yt_raw, ys_raw)

    n = len(yt)
    num_pos = int((yt == 1).sum())
    num_neg = int((yt == 0).sum())

    if n == 0:
        log.error(
            "[Benchmark] No valid samples after sanitization. "
            "Returning empty BenchmarkResult."
        )
        return BenchmarkResult(
            model_name=model_name,
            setting=setting,
            num_dropped=dropped,
        )

    log.info(
        f"[Benchmark] Evaluating '{model_name}' / '{setting}': "
        f"n={n}  pos={num_pos}  neg={num_neg}  dropped={dropped}"
    )

    # ─── Core metrics ────────────────────────────────────
    roc_auc = _calc_roc_auc(yt, ys)
    pr_auc = _calc_pr_auc(yt, ys)
    best_f1, best_threshold = _calc_best_f1(yt, ys)

    # Fixed-threshold metrics
    yp = _binarize(ys, threshold=threshold)
    f1_at_05 = float(f1_score(yt, yp, zero_division=0))
    precision_at_05 = float(precision_score(yt, yp, zero_division=0))
    recall_at_05 = float(recall_score(yt, yp, zero_division=0))
    accuracy_at_05 = float(accuracy_score(yt, yp))

    # P@K / R@K
    p_at_k: Dict[int, float] = {}
    r_at_k: Dict[int, float] = {}
    for k in k_list:
        p_at_k[k] = _calc_p_at_k(yt, ys, k)
        r_at_k[k] = _calc_r_at_k(yt, ys, k)

    # ─── Bootstrap CI ────────────────────────────────────
    ci_roc_auc: Optional[Tuple[float, float]] = None
    ci_pr_auc: Optional[Tuple[float, float]] = None

    if bootstrap and num_pos > 1 and num_neg > 1:
        log.info(f"[Benchmark] Running bootstrap (n_boot={n_boot})...")
        ci_roc_auc = _bootstrap_ci(
            yt, ys,
            metric_fn=roc_auc_score,
            n_boot=n_boot,
            alpha=bootstrap_alpha,
        )
        ci_pr_auc = _bootstrap_ci(
            yt, ys,
            metric_fn=average_precision_score,
            n_boot=n_boot,
            alpha=bootstrap_alpha,
        )
    elif bootstrap:
        log.warning(
            "[Benchmark] Bootstrap skipped: insufficient class samples "
            f"(pos={num_pos}, neg={num_neg})."
        )

    result = BenchmarkResult(
        model_name=model_name,
        setting=setting,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        best_f1=best_f1,
        best_threshold=best_threshold,
        f1_at_05=f1_at_05,
        precision_at_05=precision_at_05,
        recall_at_05=recall_at_05,
        accuracy_at_05=accuracy_at_05,
        p_at_k=p_at_k,
        r_at_k=r_at_k,
        num_samples=n,
        num_pos=num_pos,
        num_neg=num_neg,
        num_dropped=dropped,
        ci_roc_auc=ci_roc_auc,
        ci_pr_auc=ci_pr_auc,
    )

    log.info("\n" + result.pretty())
    return result


# ============================================================
# 6. Dict alignment helper
# ============================================================

def _align_dict_inputs(
    labels: Mapping[str, int],
    scores: Mapping[str, float],
    *,
    missing_score: float = 0.0,
    drop_missing: bool = False,
) -> Tuple[List[int], List[float]]:
    """
    contract_id 기준으로 label/score 정렬.
    """
    yt_list: List[int] = []
    ys_list: List[float] = []
    missing = 0

    for cid, label in labels.items():
        if cid in scores:
            ys_list.append(float(scores[cid]))
        else:
            if drop_missing:
                missing += 1
                continue
            ys_list.append(float(missing_score))
            missing += 1
        yt_list.append(int(label))

    if missing > 0:
        mode = "dropped" if drop_missing else f"filled={missing_score}"
        log.warning(
            f"[Benchmark] {missing} contracts missing score: {mode}"
        )

    return yt_list, ys_list


# ============================================================
# 7. Public exports
# ============================================================

__all__ = [
    "BenchmarkResult",
    "BenchmarkTable",
    "evaluate_benchmark",
]
