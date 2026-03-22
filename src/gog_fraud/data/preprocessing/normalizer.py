# src/gog_fraud/data/preprocessing/normalizer.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


# ----------------------------------------------------
# 컬럼 그룹별 변환 전략 정의
# ----------------------------------------------------
#
# 전략:
#   "log1p_wei"   : raw wei (매우 큰 정수) → log1p 후 float32
#   "log1p"       : 일반 큰 양수 (gas, block 등) → log1p 후 float32
#   "linear_norm" : 작은 범위 숫자 → float32 그대로 (0으로 clip)
#   "identity"    : 변환 없이 float32
# ----------------------------------------------------

NUMERIC_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "value":     (("value", "amount"),              "log1p_wei"),
    "gas":       (("gas", "gasused", "gas_used"),   "log1p"),
    "gas_price": (("gasprice", "gas_price"),        "log1p_wei"),
    "timestamp": (("timestamp", "timeStamp", "block_timestamp"), "log1p"),
    "block":     (("blocknumber", "block_number"),  "log1p"),
    "nonce":     (("nonce",),                       "log1p"),
}

# float32 안전 범위
_FLOAT32_MAX = 3.4e+38
_FLOAT32_MIN = -3.4e+38


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    문자열 포함된 컬럼을 숫자로 변환.
    - 16진수 wei 표현 ("0x...") 지원
    - 변환 실패 시 0으로 채움
    """
    def _convert(v):
        if pd.isna(v):
            return 0.0
        s = str(v).strip()
        if s.startswith("0x") or s.startswith("0X"):
            try:
                return float(int(s, 16))
            except Exception:
                return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0

    return series.apply(_convert)


def apply_log1p_wei(series: pd.Series) -> pd.Series:
    """
    wei 단위 초대형 숫자를 log1p 변환.
    음수 불가능 → clip(0) 후 log1p.
    float32 overflow 방지를 위해 log 공간에서 clip.
    """
    numeric = _safe_to_numeric(series).clip(lower=0.0)
    # float64로 log1p 계산 후 float32 범위로 clip
    logged = np.log1p(numeric.values.astype(np.float64))
    logged = np.clip(logged, 0.0, 88.0)  # log1p(e^88) ≈ float32 안전 최대
    return pd.Series(logged, index=series.index)


def apply_log1p(series: pd.Series) -> pd.Series:
    """
    일반 큰 양수 숫자에 log1p 변환.
    """
    numeric = _safe_to_numeric(series).clip(lower=0.0)
    logged = np.log1p(numeric.values.astype(np.float64))
    logged = np.clip(logged, 0.0, 88.0)
    return pd.Series(logged, index=series.index)


def apply_strategy(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "log1p_wei":
        return apply_log1p_wei(series)
    elif strategy == "log1p":
        return apply_log1p(series)
    elif strategy == "linear_norm":
        numeric = _safe_to_numeric(series)
        return numeric.clip(lower=0.0)
    elif strategy == "identity":
        return _safe_to_numeric(series)
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def normalize_edge_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    NUMERIC_GROUPS 정의에 따라 컬럼을 탐지하고 정규화한다.

    Returns:
        result_df    : 정규화된 컬럼들만 담긴 DataFrame
        selected_map : {출력 컬럼명: 원본 컬럼명}
    """
    canon_map = {_canon(c): c for c in df.columns}
    result_df = pd.DataFrame(index=df.index)
    selected_map: dict[str, str] = {}

    for out_name, (aliases, strategy) in NUMERIC_GROUPS.items():
        original_col = None
        for alias in aliases:
            if _canon(alias) in canon_map:
                original_col = canon_map[_canon(alias)]
                break

        if original_col is None:
            continue

        result_df[out_name] = apply_strategy(df[original_col], strategy)
        selected_map[out_name] = original_col

    if result_df.empty:
        result_df["const"] = 1.0
        selected_map["const"] = "__constant__"

    return result_df, selected_map


def _canon(s: str) -> str:
    return str(s).replace("\ufeff", "").strip().lower().replace(" ", "").replace("_", "")
