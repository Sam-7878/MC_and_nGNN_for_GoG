# gog.py  — NaN-safe, overflow-proof version
import json
import os
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import multiprocessing
from pathlib import Path


# =========================================================================
# JSON 직렬화 헬퍼
# =========================================================================
class JSONEncoderWithNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =========================================================================
# 데이터 검증 & 정제 유틸
# =========================================================================
FLOAT32_MAX = 3.4e38
CONTRACT_FEATURE_CLIP = 10.0  # z-score 이상치 클리핑 범위

def safe_log1p(x: float) -> float:
    """
    log1p 변환: overflow/NaN 방어.
    - 음수 → 0
    - inf  → log1p(float32 max) ≈ 87.3
    - nan  → 0
    """
    if not math.isfinite(x) or x < 0:
        return 0.0
    return math.log1p(x)


def validate_and_transform_node_features(features: list[list[float]]) -> list[list[float]] | None:
    """
    노드 피처를 검증하고 log1p 변환합니다.
    
    원본 피처 구조: [in_degree, out_degree, in_value, out_value]
    변환 후 구조:   [log1p(in_degree), log1p(out_degree), log1p(in_value), log1p(out_value)]

    Returns:
        변환된 피처 리스트. 빈 피처이면 None 반환.
    """
    if not features:
        return None

    transformed = []
    for feat in features:
        if len(feat) != 4:
            # 피처 길이 불일치 → 0으로 패딩
            feat = (feat + [0.0] * 4)[:4]

        in_deg, out_deg, in_val, out_val = feat
        t = [
            safe_log1p(in_deg),
            safe_log1p(out_deg),
            safe_log1p(in_val),
            safe_log1p(out_val),
        ]

        # 변환 후에도 nan/inf가 남으면 0으로 대체
        t = [0.0 if (not math.isfinite(v)) else v for v in t]
        transformed.append(t)

    return transformed


def validate_contract_feature(contract_feature: list[float]) -> list[float]:
    """
    contract_feature(z-score 정규화된 값)를 검증합니다.
    - NaN/Inf → 0.0
    - 극단 이상치 → clip to [-CONTRACT_FEATURE_CLIP, +CONTRACT_FEATURE_CLIP]
    """
    cleaned = []
    for v in contract_feature:
        if not math.isfinite(v):
            cleaned.append(0.0)
        else:
            cleaned.append(max(-CONTRACT_FEATURE_CLIP, min(CONTRACT_FEATURE_CLIP, v)))
    return cleaned


def validate_edges(edges: list[list[int]], n_nodes: int) -> list[list[int]]:
    """
    엣지 인덱스 검증:
    - 노드 인덱스 범위 초과 엣지 제거
    - self-loop 제거 (optional, 주석 처리로 선택 가능)
    - 중복 엣지 제거 (optional)
    """
    valid_edges = []
    seen = set()
    for u, v in edges:
        if not (isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer))):
            continue
        u, v = int(u), int(v)
        if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes:
            continue
        # self-loop 제거 (GoG 거래 그래프에서 self-loop는 대부분 노이즈)
        if u == v:
            continue
        key = (u, v)
        if key not in seen:
            seen.add(key)
            valid_edges.append([u, v])
    return valid_edges


def is_valid_graph(edges, features, min_edges: int = 1) -> tuple[bool, str]:
    """
    최종 저장 전 그래프 유효성 체크.
    Returns: (is_valid, reason)
    """
    if not features:
        return False, "empty features"
    if len(edges) < min_edges:
        return False, f"too few edges ({len(edges)} < {min_edges})"
    # 피처에 nan/inf가 남아있으면 거부
    flat = [v for feat in features for v in feat]
    if any(not math.isfinite(v) for v in flat):
        return False, "nan/inf in features after transform"
    return True, "ok"


# =========================================================================
# 전역 변수 (Copy-on-Write 메모리 공유)
# =========================================================================
global_feature_dict = {}
global_label_dict = {}
global_address_index = {}


# =========================================================================
# 그래프 피처 계산 (벡터화)
# =========================================================================
def compute_graph_features(df: pd.DataFrame):
    """Vectorized graph feature computation"""
    unique_addresses = pd.concat([df['from'], df['to']]).unique()
    address_to_index = {addr: i for i, addr in enumerate(unique_addresses)}
    n_nodes = len(unique_addresses)

    from_indices = df['from'].map(address_to_index).values
    to_indices   = df['to'].map(address_to_index).values
    values = pd.to_numeric(df['value'], errors='coerce').fillna(0).values

    in_degree  = np.bincount(to_indices,   minlength=n_nodes).astype(float)
    out_degree = np.bincount(from_indices, minlength=n_nodes).astype(float)
    in_value   = np.bincount(to_indices,   weights=values, minlength=n_nodes)
    out_value  = np.bincount(from_indices, weights=values, minlength=n_nodes)

    # numpy inf/nan 제거 (value가 극단적으로 크면 bincount 합산 중 overflow 가능)
    in_value  = np.nan_to_num(in_value,  nan=0.0, posinf=0.0, neginf=0.0)
    out_value = np.nan_to_num(out_value, nan=0.0, posinf=0.0, neginf=0.0)

    features = [
        [float(in_degree[i]), float(out_degree[i]),
         float(in_value[i]),  float(out_value[i])]
        for i in range(n_nodes)
    ]

    return features, address_to_index, from_indices, to_indices


# =========================================================================
# 워커 프로세스
# =========================================================================
def process_single_tx_worker(args):
    contract, chain_dir, directory = args

    global global_feature_dict, global_label_dict, global_address_index

    node_features    = global_feature_dict.get(contract, [])
    label            = global_label_dict.get(contract, 0)
    idx              = global_address_index.get(contract, -1)

    if idx == -1:
        return  # 매핑 실패 → 건너뜀

    try:
        df = pd.read_csv(f'{chain_dir}/{contract}.csv', low_memory=False)
        df['from'] = df['from'].str.lower()
        df['to']   = df['to'].str.lower()

        # -----------------------------------------------------------------
        # 1. 그래프 피처 계산
        # -----------------------------------------------------------------
        raw_features, address_to_index, from_indices, to_indices = \
            compute_graph_features(df)
        n_nodes = len(raw_features)

        # -----------------------------------------------------------------
        # 2. 노드 피처 변환 (log1p → overflow 방지)
        # -----------------------------------------------------------------
        transformed_features = validate_and_transform_node_features(raw_features)
        if transformed_features is None:
            print(f"[SKIP] {contract}: node features empty after transform")
            return

        # -----------------------------------------------------------------
        # 3. 엣지 검증
        # -----------------------------------------------------------------
        raw_edges = [[int(u), int(v)] for u, v in zip(from_indices, to_indices)]
        clean_edges = validate_edges(raw_edges, n_nodes)

        # -----------------------------------------------------------------
        # 4. contract_feature 검증 (z-score 이상치 클리핑)
        # -----------------------------------------------------------------
        contract_feature_clean = validate_contract_feature(node_features)

        # -----------------------------------------------------------------
        # 5. 최종 유효성 검사
        # -----------------------------------------------------------------
        is_valid, reason = is_valid_graph(clean_edges, transformed_features, min_edges=1)
        if not is_valid:
            print(f"[SKIP] {contract} (idx={idx}): {reason}")
            return

        # -----------------------------------------------------------------
        # 6. JSON 저장
        # -----------------------------------------------------------------
        graph_data = {
            'edges':            clean_edges,
            'features':         transformed_features,   # log1p 변환 완료
            'contract_feature': contract_feature_clean, # 클리핑 완료
            'label':            int(label),
            # 메타데이터: 나중에 디버깅/재현성 확보를 위해 기록
            '_meta': {
                'n_nodes':    n_nodes,
                'n_edges':    len(clean_edges),
                'n_raw_edges': len(raw_edges),
                'contract':   contract,
            }
        }

        out_path = f'{directory}/{idx}.json'
        with open(out_path, 'w') as f:
            json.dump(graph_data, f, cls=JSONEncoderWithNumpy)

    except FileNotFoundError:
        # CSV 없음 → 정상적으로 건너뜀 (로그 최소화)
        pass
    except Exception as e:
        print(f"[ERROR] {contract}: {e}")


# =========================================================================
# main
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',            type=str, default='polygon')
    parser.add_argument('--parallel_workers', type=int, default=-1)
    parser.add_argument('--min_edges',        type=int, default=1,
                        help='그래프 최소 유효 엣지 수 (이하 skip)')
    args = parser.parse_args()

    chain = args.chain

    labels_final = pd.read_csv(
        f'../../_data/dataset/features/{chain}_basic_metrics_processed.csv'
    )
    select_address = labels_final.Contract.tolist()

    contract_mapping_file = (
        f'../../_data/graphs/{chain}/'
        f'{chain}_common_nodes_except_null_labels.csv'
    )
    global_graph = pd.read_csv(contract_mapping_file)
    global_graph_select = global_graph.query(
        'Contract1 in @select_address & Contract2 in @select_address'
    ).copy()

    all_address_index = dict(zip(labels_final.Contract, labels_final.index))
    global_graph_select['graph_1'] = global_graph_select['Contract1'].map(all_address_index)
    global_graph_select['graph_2'] = global_graph_select['Contract2'].map(all_address_index)

    chain_dir = f'../../_data/dataset/transactions/{chain}'
    directory = f'../../_data/GoG/{chain}'
    os.makedirs(f'{directory}/edges',  exist_ok=True)
    os.makedirs(f'{directory}/graphs', exist_ok=True)

    global_graph_select[['graph_1', 'graph_2']].to_csv(
        f'{directory}/edges/global_edges.csv', index=False
    )

    # -------------------------------------------------------------------------
    # 전역 딕셔너리 준비
    # -------------------------------------------------------------------------
    global global_feature_dict, global_label_dict, global_address_index
    print("Preparing global dictionaries for fast O(1) lookup...")

    feature_cols = labels_final.columns[1:-1]
    global_feature_dict  = labels_final.set_index('Contract')[feature_cols].T.to_dict('list')
    global_label_dict    = labels_final.set_index('Contract')['label'].to_dict()
    global_address_index = all_address_index

    # -------------------------------------------------------------------------
    # 멀티프로세싱
    # -------------------------------------------------------------------------
    n_workers = (
        args.parallel_workers if args.parallel_workers > 0
        else max(2, os.cpu_count() // 2)
    )
    contracts = labels_final['Contract'].tolist()
    tasks = [(c, chain_dir, f'{directory}/graphs') for c in contracts]

    print(f"Starting parallel processing with {n_workers} workers...")
    print(f"  • Node features   : log1p transformed (overflow-safe)")
    print(f"  • contract_feature: clipped to ±{CONTRACT_FEATURE_CLIP}")
    print(f"  • Edges           : validated (range + self-loop + dedup)")
    print(f"  • Min edges filter: {args.min_edges}")

    with multiprocessing.Pool(processes=n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_single_tx_worker, tasks, chunksize=10),
            total=len(tasks),
            desc="Processing TXs"
        ):
            pass

    print(f"\n✅ GoG data generation for [{chain}] completed successfully!")
    print(f"   Output: {directory}/graphs/")


if __name__ == '__main__':
    main()
