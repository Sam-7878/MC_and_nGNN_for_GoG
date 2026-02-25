import json
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import multiprocessing
from pathlib import Path

class JSONEncoderWithNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_contract_mapping(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compute_graph_features(df):
    """Vectorized graph feature computation"""
    unique_addresses = pd.concat([df['from'], df['to']]).unique()
    address_to_index = {addr: i for i, addr in enumerate(unique_addresses)}
    n_nodes = len(unique_addresses)

    from_indices = df['from'].map(address_to_index).values
    to_indices = df['to'].map(address_to_index).values
    values = pd.to_numeric(df['value'], errors='coerce').fillna(0).values

    in_degree = np.bincount(to_indices, minlength=n_nodes)
    out_degree = np.bincount(from_indices, minlength=n_nodes)
    in_value = np.bincount(to_indices, weights=values, minlength=n_nodes)
    out_value = np.bincount(from_indices, weights=values, minlength=n_nodes)

    features = []
    for i in range(n_nodes):
        features.append([
            float(in_degree[i]),
            float(out_degree[i]),
            float(in_value[i]),
            float(out_value[i])
        ])
    
    return features, address_to_index, from_indices, to_indices

# =========================================================================
# ✅ 전역 변수 선언 (Copy-on-Write를 통해 메모리 공유 및 IPC 병목 제거)
# =========================================================================
global_feature_dict = {}
global_label_dict = {}
global_address_index = {}


def process_single_tx_worker(args):
    """
    워커 프로세스용 함수. 인자를 최소화하고 전역 변수를 직접 참조합니다.
    """
    contract, chain_dir, directory = args
    
    # 전역 딕셔너리에서 O(1) 속도로 즉시 데이터 가져오기 (Pandas 탐색 제거)
    global global_feature_dict, global_label_dict, global_address_index
    
    node_features = global_feature_dict.get(contract, [])
    label = global_label_dict.get(contract, 0)
    idx = global_address_index.get(contract, -1)

    if idx == -1:
        return  # 매핑 실패 시 건너뜀

    try:
        df = pd.read_csv(f'{chain_dir}/{contract}.csv', low_memory=False)
        df['from'] = df['from'].str.lower()
        df['to'] = df['to'].str.lower()
        
        features, address_to_index, from_indices, to_indices = compute_graph_features(df)
        edges = [[int(u), int(v)] for u, v in zip(from_indices, to_indices)]

        graph_data = {
            'edges': edges,
            'features': features,
            'contract_feature': node_features,
            'label': int(label)
        }
        
        with open(f'{directory}/{idx}.json', 'w') as f:
            json.dump(graph_data, f, cls=JSONEncoderWithNumpy)
            
    except Exception as e:
        print(f"Error processing {contract}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='bsc')
    parser.add_argument('--parallel_workers', type=int, default=-1)
    args = parser.parse_args()
    
    chain = args.chain

    labels_final = pd.read_csv(f'../../_data/dataset/features/{chain}_basic_metrics_processed.csv')
    select_address = labels_final.Contract.tolist()

    contract_mapping_file = f'../../_data/graphs/{chain}/{chain}_common_nodes_except_null_labels.csv'
    global_graph = pd.read_csv(contract_mapping_file)
    
    global_graph_select = global_graph.query('Contract1 in @select_address & Contract2 in @select_address').copy()

    all_address_index = dict(zip(labels_final.Contract, labels_final.index))
    global_graph_select['graph_1'] = global_graph_select['Contract1'].map(all_address_index)
    global_graph_select['graph_2'] = global_graph_select['Contract2'].map(all_address_index)

    chain_dir = f'../../_data/dataset/transactions/{chain}'
    directory = f'../../_data/GoG/{chain}'
    os.makedirs(f'{directory}/edges', exist_ok=True)
    os.makedirs(f'{directory}/graphs', exist_ok=True)
    
    global_graph_select[['graph_1', 'graph_2']].to_csv(f'{directory}/edges/global_edges.csv', index=False)

    # =========================================================================
    # ✅ 1. Pandas DataFrame을 고속 탐색용 Dictionary로 변환하여 전역 변수에 저장
    # =========================================================================
    global global_feature_dict, global_label_dict, global_address_index
    print("Preparing global dictionaries for fast O(1) lookup...")
    
    feature_cols = labels_final.columns[1:-1] # Contract와 label 제외한 피처들
    # Contract -> [feature list] 형태로 변환
    global_feature_dict = labels_final.set_index('Contract')[feature_cols].T.to_dict('list')
    # Contract -> label
    global_label_dict = labels_final.set_index('Contract')['label'].to_dict()
    # Contract -> index
    global_address_index = all_address_index

    # =========================================================================
    # ✅ 2. 멀티프로세싱 최적화 (Joblib -> Multiprocessing Pool.imap_unordered)
    # =========================================================================
    n_workers = args.parallel_workers if args.parallel_workers > 0 else max(2, os.cpu_count()//2)
    contracts = labels_final['Contract'].tolist()
    
    # 워커에 넘길 인자에서 무거운 객체 제거
    tasks = [(contract, chain_dir, f'{directory}/graphs') for contract in contracts]
    
    print(f"Starting parallel processing with {n_workers} workers...")
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_tx_worker, tasks, chunksize=10), total=len(tasks), desc="Processing TXs"):
            pass

    print(f"GoG data generation for {chain} completed successfully!")

if __name__ == '__main__':
    main()