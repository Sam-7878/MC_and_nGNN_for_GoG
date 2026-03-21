from typing import Dict, Iterable, Optional, Sequence, Union

import torch


def _to_1d_cpu_float_tensor(x) -> torch.Tensor:
    if x is None:
        raise ValueError("Input cannot be None")
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.detach().view(-1).float().cpu()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true.numpy(), y_score.numpy()))
    except Exception:
        return 0.0


def _safe_pr_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true.numpy(), y_score.numpy()))
    except Exception:
        return 0.0


def confusion_at_threshold(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> Dict[str, int]:
    y_true = _to_1d_cpu_float_tensor(y_true)
    y_score = _to_1d_cpu_float_tensor(y_score)

    y_pred = (y_score >= threshold).long()
    y_true_long = y_true.long()

    tp = int(((y_pred == 1) & (y_true_long == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true_long == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true_long == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true_long == 1)).sum().item())

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def binary_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = _to_1d_cpu_float_tensor(y_true)
    y_score = _to_1d_cpu_float_tensor(y_score)

    conf = confusion_at_threshold(y_true, y_score, threshold=threshold)
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(y_true))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    positive_rate = float((y_true == 1).float().mean().item()) if len(y_true) > 0 else 0.0

    result = {
        "threshold": float(threshold),
        "num_examples": int(len(y_true)),
        "positive_rate": positive_rate,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    return result


def _default_threshold_grid(y_score: torch.Tensor) -> torch.Tensor:
    unique_scores = torch.unique(y_score)
    unique_scores = torch.sort(unique_scores).values

    if unique_scores.numel() <= 100:
        return unique_scores

    return torch.linspace(0.01, 0.99, steps=99)


def find_best_f1_threshold(
    y_true,
    y_score,
    thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
) -> Dict[str, float]:
    y_true = _to_1d_cpu_float_tensor(y_true)
    y_score = _to_1d_cpu_float_tensor(y_score)

    if thresholds is None:
        thresholds = _default_threshold_grid(y_score)
    elif not torch.is_tensor(thresholds):
        thresholds = torch.tensor(list(thresholds), dtype=torch.float32)

    thresholds = thresholds.view(-1).float().cpu()

    best = {
        "best_threshold": 0.5,
        "best_f1": -1.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
    }

    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_score, threshold=float(threshold))
        if metrics["f1"] > best["best_f1"]:
            best = {
                "best_threshold": float(threshold.item()),
                "best_f1": float(metrics["f1"]),
                "best_precision": float(metrics["precision"]),
                "best_recall": float(metrics["recall"]),
            }

    return best


def topk_metrics(
    y_true,
    y_score,
    k: int,
) -> Dict[str, float]:
    if k <= 0:
        raise ValueError("k must be >= 1")

    y_true = _to_1d_cpu_float_tensor(y_true)
    y_score = _to_1d_cpu_float_tensor(y_score)

    n = len(y_true)
    if n == 0:
        return {
            "k": int(k),
            "effective_k": 0,
            "topk_positive_count": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
        }

    effective_k = min(k, n)
    top_indices = torch.topk(y_score, k=effective_k, largest=True).indices
    top_true = y_true[top_indices]

    topk_positive_count = float(top_true.sum().item())
    total_positive = float(y_true.sum().item())

    precision_at_k = _safe_div(topk_positive_count, effective_k)
    recall_at_k = _safe_div(topk_positive_count, total_positive)

    return {
        "k": int(k),
        "effective_k": int(effective_k),
        "topk_positive_count": float(topk_positive_count),
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
    }


def multi_topk_metrics(
    y_true,
    y_score,
    ks: Iterable[int],
) -> Dict[str, float]:
    result = {}
    for k in ks:
        metrics = topk_metrics(y_true, y_score, k=k)
        result[f"top{k}_precision"] = metrics["precision_at_k"]
        result[f"top{k}_recall"] = metrics["recall_at_k"]
        result[f"top{k}_positive_count"] = metrics["topk_positive_count"]
    return result


def bce_loss_from_logits(logits, y_true) -> float:
    logits = _to_1d_cpu_float_tensor(logits)
    y_true = _to_1d_cpu_float_tensor(y_true)

    if len(logits) != len(y_true):
        raise ValueError("logits and y_true must have the same length")

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.view(-1, 1),
        y_true.view(-1, 1),
    )
    return float(loss.item())
