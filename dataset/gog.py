import json
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from joblib import Parallel, delayed
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

    # Vectorized degrees & values
    from_indices = df['from'].map(address_to_index).values
    to_indices = df['to'].map(address_to_index).values
    values = pd.to_numeric(df['value'], errors='coerce').fillna(0).values

    in_degree = np.bincount(to_indices, minlength=n_nodes)
    out_degree = np.bincount(from_indices, minlength=n_nodes)
    in_value = np.bincount(to_indices, weights=values, minlength=n_nodes)
    out_value = np.bincount(from_indices, weights=values, minlength=n_nodes)

    # 추가 특징
    tx_dates = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    time_range = (tx_dates.max() - tx_dates.min()).total_seconds() if len(tx_dates) > 1 else 1.0
    tx_freq = np.bincount(from_indices, minlength=n_nodes) / time_range
    total_degree = in_degree + out_degree
    avg_tx_interval = np.where(total_degree > 0, np.sum(values) / total_degree, 0)

    features = {}
    for i, addr in enumerate(unique_addresses):
        norm_in_deg = in_degree[i] / max(total_degree.max(), 1)
        norm_out_deg = out_degree[i] / max(total_degree.max(), 1)
        features[str(i)] = [
            int(total_degree[i]), int(in_degree[i]), int(out_degree[i]),
            float(in_value[i]), float(out_value[i]),
            float(tx_freq[i]), float(avg_tx_interval[i]),
            float(norm_in_deg), float(norm_out_deg)
        ]

    edges = list(zip(from_indices.tolist(), to_indices.tolist()))
    return features, edges, address_to_index

def save_transaction_graph(df, label, idx, directory, mc_sampling_ratio=1.0):
    """Optimized save with MC sampling prep"""
    os.makedirs(directory, exist_ok=True)

    if mc_sampling_ratio < 1.0:
        sample_idx = np.random.choice(len(df), int(len(df) * mc_sampling_ratio), replace=False)
        df = df.iloc[sample_idx].reset_index(drop=True)

    features, edges, _ = compute_graph_features(df)

    graph_dict = {
        "label": int(label),
        "features": features,
        "edges": edges,
        "mc_sampling_ratio": float(mc_sampling_ratio),
        "n_nodes": len(features),
        "n_edges": len(edges)
    }

    file_name = os.path.join(directory, f'{idx}.json')
    with open(file_name, 'w') as file:
        json.dump(graph_dict, file, cls=JSONEncoderWithNumpy, indent=None)
    print(f"Graph {idx} (label={label}, ratio={mc_sampling_ratio}) saved")

def process_single_tx(contract, chain_dir, labels_df, all_address_index, directory):
    """Single tx processing for parallelization"""
    try:
        tx_path = os.path.join(chain_dir, f'{contract}.csv')
        if not os.path.exists(tx_path):
            print(f"Skipping missing {tx_path}")
            return None


        dtype = {'from': str, 'to': str, 'value': float, 'timestamp': float}  # Type consistency
        tx = pd.read_csv(tx_path, dtype=dtype)
        label = labels_df.loc[all_address_index[contract], 'Category']
        idx = labels_df.index[labels_df['Contract'] == contract].tolist()[0]
        save_transaction_graph(tx, label, idx, directory)
    except Exception as e:
        print(f"Error processing {contract}: {e}")




def main():
    parser = argparse.ArgumentParser(description="Optimized GoG Dataset Builder")
    parser.add_argument('--chain', type=str, default='polygon')
    parser.add_argument('--n_classes', type=int, default=3, help="Number of classes to use")
    parser.add_argument('--parallel_workers', type=int, default=-1, help="Parallel workers (-1: auto)")
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],  # 0.2 -> 0.1 (합 1.0)
                        help="train/val/test split ratios")
    parser.add_argument('--mc_sampling', type=float, default=1.0, help="MC subsampling ratio")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    chain = args.chain
    labels = pd.read_csv('../../_data/dataset/labels.csv').query('Chain == @chain').reset_index(drop=True)

    # Balanced class selection (over-fitting 방지)
    category_counts = labels['Category'].value_counts()
    select_class = list(category_counts.head(args.n_classes).index)
    category_to_label = {cat: i for i, cat in enumerate(select_class)}
    
    # ✅ 수정: 필터링을 먼저 하고 나서 map
    labels_filtered = labels[labels['Category'].isin(select_class)].copy()
    labels_filtered['Category'] = labels_filtered['Category'].map(category_to_label)
    
    # ✅ 수정: NaN 체크 (혹시 모를 경우 대비)
    if labels_filtered['Category'].isna().any():
        print("Warning: NaN found in Category after mapping. Dropping rows...")
        labels_filtered = labels_filtered.dropna(subset=['Category']).reset_index(drop=True)
    
    # Train/val/test split (stratify는 이제 NaN 없음)
    from sklearn.model_selection import train_test_split
    
    train_labels, temp = train_test_split(
        labels_filtered, 
        test_size=1-args.split_ratio[0], 
        stratify=labels_filtered['Category'],  # ✅ 필터링된 데이터의 Category
        random_state=42  # 재현성
    )
    
    val_labels, test_labels = train_test_split(
        temp, 
        test_size=args.split_ratio[2]/(args.split_ratio[1]+args.split_ratio[2]), 
        stratify=temp['Category'],  # ✅ temp의 Category
        random_state=42
    )
    
    # Split 컬럼 추가 (train은 기본값 없음, val/test 표시)
    train_labels = train_labels.assign(Split='train')
    val_labels = val_labels.assign(Split='val')
    test_labels = test_labels.assign(Split='test')
    
    labels_final = pd.concat([train_labels, val_labels, test_labels]).reset_index(drop=True)
    
    # 저장 (디렉토리 생성 보장)
    os.makedirs(f'../../_data/GoG/{chain}/graphs', exist_ok=True)
    labels_final.to_csv(f'../../_data/GoG/{chain}/labels_split.csv', index=False)
    print(f"Labels saved with splits: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")

    select_address = labels_final['Contract'].values
    print(f"Selected classes: {select_class}, Total samples: {len(select_address)}")

    # 나머지 코드는 동일...
    contract_mapping_file = f'../../_data/dataset/global_graph/{chain}_contract_to_number_mapping.json'
    contract_to_number = load_contract_mapping(contract_mapping_file)
    number_to_contract = {v: k for k, v in contract_to_number.items()}
    global_graph = pd.read_csv(f'../../_data/dataset/global_graph/{chain}_graph_more_than_1_ratio.csv')
    global_graph['Contract1'] = global_graph['Contract1'].apply(lambda x: number_to_contract[x])
    global_graph['Contract2'] = global_graph['Contract2'].apply(lambda x: number_to_contract[x])
    global_graph_select = global_graph.query('Contract1 in @select_address & Contract2 in @select_address')

    all_address_index = dict(zip(labels_final.Contract, labels_final.index))
    global_graph_select['graph_1'] = global_graph_select['Contract1'].map(all_address_index)
    global_graph_select['graph_2'] = global_graph_select['Contract2'].map(all_address_index)

    # 병렬 tx 처리
    chain_dir = f'../../_data/dataset/transactions/{chain}'
    directory = f'../../_data/GoG/{chain}'
    os.makedirs(f'{directory}/edges', exist_ok=True)
    global_graph_select[['graph_1', 'graph_2']].to_csv(f'{directory}/edges/global_edges.csv', index=False)

    n_workers = args.parallel_workers if args.parallel_workers > 0 else os.cpu_count()
    contracts = labels_final['Contract'].values
    Parallel(n_jobs=n_workers)(
        delayed(process_single_tx)(contract, chain_dir, labels_final, all_address_index, directory)
        for contract in tqdm(contracts, desc="Processing tx graphs")
    )
    print("All graphs processed!")




if __name__ == "__main__":
    main()
