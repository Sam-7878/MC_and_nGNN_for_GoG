import numpy as np
import warnings

def _to_numpy_1d_float(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)

def _to_numpy_1d_int(x):
    return np.asarray(x, dtype=np.int64).reshape(-1)

def calc_calibration_ece(y_true, y_score, num_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for binary classification.
    y_score should be probabilities [0, 1].
    """
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    
    if len(yt) == 0:
        return float('nan')
        
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    binids = np.digitize(ys, bins) - 1
    
    bin_total = np.bincount(binids, minlength=num_bins)
    bin_true = np.bincount(binids, weights=yt, minlength=num_bins)
    bin_pred = np.bincount(binids, weights=ys, minlength=num_bins)
    
    nonzero = bin_total > 0
    if not np.any(nonzero):
        return float('nan')
        
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_pred[nonzero] / bin_total[nonzero]
    
    ece = np.sum(bin_total[nonzero] * np.abs(prob_true - prob_pred)) / len(yt)
    return float(ece)

def calc_uncertainty_correlation(y_true, y_score, y_unc):
    """
    Calculate Pearson correlation between absolute prediction error and uncertainty.
    Indicates if higher uncertainty corresponds to larger errors (a good thing).
    """
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    unc = _to_numpy_1d_float(y_unc)
    
    if len(yt) < 2:
        return float('nan')
        
    errors = np.abs(yt - ys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(errors, unc)[0, 1]
    
    return float(corr) if not np.isnan(corr) else float('nan')

def run_selective_prediction(y_true, y_score, y_unc, coverage_ratio=0.8):
    """
    Evaluate metrics only on the top `coverage_ratio` subset of samples
    that have the lowest uncertainty.
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    unc = _to_numpy_1d_float(y_unc)
    
    n = len(yt)
    if n == 0:
        return {}
        
    threshold_idx = int(n * coverage_ratio)
    if threshold_idx == 0:
        return {}
        
    order = np.argsort(unc)
    selected_idx = order[:threshold_idx]
    
    yt_sel = yt[selected_idx]
    ys_sel = ys[selected_idx]
    yp_sel = (ys_sel >= 0.5).astype(np.int64)
    
    res = {}
    if len(np.unique(yt_sel)) >= 2:
        res["roc_auc"] = float(roc_auc_score(yt_sel, ys_sel))
    
    res["f1"] = float(f1_score(yt_sel, yp_sel, zero_division=0))
    res["precision"] = float(precision_score(yt_sel, yp_sel, zero_division=0))
    res["recall"] = float((yt_sel[yp_sel == 1] == 1).sum() / max(yt_sel.sum(), 1))
    
    return res

def calc_bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=100, ci=0.95):
    """
    Calculate confidence interval for a metric using bootstrapping.
    """
    from sklearn.utils import resample
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    
    scores = []
    for _ in range(n_bootstrap):
        yt_sample, ys_sample = resample(yt, ys)
        try:
            val = metric_fn(yt_sample, ys_sample)
            if not np.isnan(val):
                scores.append(val)
        except Exception:
            continue
            
    if not scores:
        return float('nan'), float('nan'), float('nan')
        
    scores = np.sort(scores)
    alpha = (1.0 - ci) / 2.0
    lower = scores[int(alpha * len(scores))]
    median = np.median(scores)
    upper = scores[int((1.0 - alpha) * len(scores))]
    
    return float(median), float(lower), float(upper)

def calc_fixed_budget_utility(y_true, y_score, y_unc, budget=100):
    """
    Compare Top-N (budget) alerting strategies:
    1. Standard: Top-N by raw score.
    2. Filtered: Top-N by score after filtering out top 20% most uncertain cases.
    """
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    unc = _to_numpy_1d_float(y_unc)
    
    # Strategy 1: Standard
    std_order = np.argsort(ys)[::-1]
    std_top = std_order[:budget]
    std_hits = int(yt[std_top].sum())
    std_precision = std_hits / budget
    
    # Strategy 2: Filtered (Excluding top 20% uncertainty)
    unc_threshold = np.percentile(unc, 80)
    mask = unc <= unc_threshold
    
    ys_filtered = ys[mask]
    yt_filtered = yt[mask]
    
    if len(ys_filtered) < budget:
        filtered_top = np.arange(len(ys_filtered))
        actual_budget = len(ys_filtered)
    else:
        filtered_order = np.argsort(ys_filtered)[::-1]
        filtered_top = filtered_order[:budget]
        actual_budget = budget
        
    filtered_hits = int(yt_filtered[filtered_top].sum())
    filtered_precision = filtered_hits / max(actual_budget, 1)
    
    return {
        "budget": budget,
        "std_hits": std_hits,
        "std_precision": std_precision,
        "filtered_hits": filtered_hits,
        "filtered_precision": filtered_precision,
        "precision_gain": filtered_precision - std_precision
    }

def calc_abstention_cost(y_true, y_score, y_unc, uncertainty_threshold):
    """
    Calculate the trade-off: Precision Gain vs. Recall Loss when abstaining.
    """
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    unc = _to_numpy_1d_float(y_unc)
    
    # Original (No abstention)
    yp = (ys >= 0.5).astype(np.int64)
    orig_tp = int(((yp == 1) & (yt == 1)).sum())
    orig_fp = int(((yp == 1) & (yt == 0)).sum())
    orig_fn = int(((yp == 0) & (yt == 1)).sum())
    
    orig_precision = orig_tp / max(orig_tp + orig_fp, 1)
    orig_recall = orig_tp / max(orig_tp + orig_fn, 1)
    
    # With Abstention
    mask = unc <= uncertainty_threshold
    yt_abs = yt[mask]
    ys_abs = ys[mask]
    yp_abs = (ys_abs >= 0.5).astype(np.int64)
    
    abs_tp = int(((yp_abs == 1) & (yt_abs == 1)).sum())
    abs_fp = int(((yp_abs == 1) & (yt_abs == 0)).sum())
    
    # Note: abstained TPs are now considered FNs in the global context
    total_pos = yt.sum()
    abs_precision = abs_tp / max(abs_tp + abs_fp, 1)
    abs_recall = abs_tp / max(total_pos, 1)
    
    return {
        "orig_precision": float(orig_precision),
        "orig_recall": float(orig_recall),
        "abs_precision": float(abs_precision),
        "abs_recall": float(abs_recall),
        "precision_gain": float(abs_precision - orig_precision),
        "recall_loss": float(orig_recall - abs_recall)
    }
