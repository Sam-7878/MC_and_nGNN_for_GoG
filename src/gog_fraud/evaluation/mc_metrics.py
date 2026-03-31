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
    Evaluate ROC-AUC and F1 only on the top `coverage_ratio` subset of samples
    that have the lowest uncertainty. Returning ROC-AUC.
    """
    from sklearn.metrics import roc_auc_score
    yt = _to_numpy_1d_int(y_true)
    ys = _to_numpy_1d_float(y_score)
    unc = _to_numpy_1d_float(y_unc)
    
    n = len(yt)
    if n == 0:
        return float('nan')
        
    threshold_idx = int(n * coverage_ratio)
    if threshold_idx == 0:
        return float('nan')
        
    order = np.argsort(unc)
    selected_idx = order[:threshold_idx]
    
    yt_sel = yt[selected_idx]
    ys_sel = ys[selected_idx]
    
    if len(np.unique(yt_sel)) < 2:
        return float('nan')
        
    try:
        auc = roc_auc_score(yt_sel, ys_sel)
        return float(auc)
    except Exception:
        return float('nan')
