import multiprocessing

import pandas as pd
import random
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_data(chain, timestamps, index_mapping, edges):
    # Process timestamps
    timestamps = timestamps.query('address in @index_mapping')
    timestamps['addr_index'] = timestamps['address'].apply(lambda x: index_mapping[x])
    timestamps = timestamps.sort_values(by='first_timestamp')

    # Merge edges with timestamps
    edges_with_timestamps = edges.merge(timestamps, left_on='graph_1', right_on='addr_index', how='left')
    edges_with_timestamps.rename(columns={'first_timestamp': 'timestamp_1'}, inplace=True)

    edges_with_timestamps = edges_with_timestamps.merge(timestamps, left_on='graph_2', right_on='addr_index', how='left', suffixes=('', '_2'))
    edges_with_timestamps.rename(columns={'first_timestamp': 'timestamp_2'}, inplace=True)

    edges_with_timestamps['max_timestamp'] = edges_with_timestamps[['timestamp_1', 'timestamp_2']].max(axis=1)
    edges_with_timestamps_sorted = edges_with_timestamps.sort_values(by='max_timestamp', ascending=True)

    return edges_with_timestamps_sorted

def generate_negative_samples(nodes_list, existing_edges_set, num_samples):
    """
    모든 조합을 생성하지 않고 필요한 개수만큼만 랜덤하게 추출하여 메모리 사용량을 O(1)로 유지합니다.
    """
    negative_edges = set()
    num_nodes = len(nodes_list)
    
    while len(negative_edges) < num_samples:
        i = random.choice(nodes_list)
        j = random.choice(nodes_list)
        
        # 자기 자신이 아니며, 실제 존재하는 엣지가 아니고, 이미 뽑힌 엣지가 아닌 경우만 추가
        if i != j and (i, j) not in existing_edges_set and (i, j) not in negative_edges:
            negative_edges.add((i, j))
            
    return list(negative_edges)

def generate_train_test_data(edges_with_timestamps_sorted, chain):
    # Splitting indices for train and test
    train_num = int(len(edges_with_timestamps_sorted) * 0.8)
    train_data = edges_with_timestamps_sorted.iloc[:train_num]
    test_data = edges_with_timestamps_sorted.iloc[train_num:]

    # Nodes
    train_nodes_list = list(set(train_data['graph_1']).union(set(train_data['graph_2'])))
    all_nodes_list = list(set(edges_with_timestamps_sorted['graph_1']).union(set(edges_with_timestamps_sorted['graph_2'])))

    # Existing edges for fast lookup
    train_existing_edges = set(zip(train_data['graph_1'], train_data['graph_2']))
    test_existing_edges = set(zip(test_data['graph_1'], test_data['graph_2']))

    # Generate negative edges dynamically (메모리 폭발 방지)
    print("Generating train negative samples...")
    train_negative_edges = generate_negative_samples(train_nodes_list, train_existing_edges, len(train_data))
    
    print("Generating test negative samples...")
    test_negative_edges = generate_negative_samples(all_nodes_list, test_existing_edges, len(test_data))

    # Save train and test edges with labels
    train_path = f'../../_data/GoG/edges/{chain}/{chain}_train_edges.txt'
    test_path = f'../../_data/GoG/edges/{chain}/{chain}_test_edges.txt'
    
    with open(train_path, 'w') as f:
        for edge in train_data[['graph_1', 'graph_2']].itertuples(index=False):
            f.write(f"{edge.graph_1} {edge.graph_2} 1\n")
        for edge in train_negative_edges:
            f.write(f"{edge[0]} {edge[1]} 0\n")

    with open(test_path, 'w') as f:
        for edge in test_data[['graph_1', 'graph_2']].itertuples(index=False):
            f.write(f"{edge.graph_1} {edge.graph_2} 1\n")
        for edge in test_negative_edges:
            f.write(f"{edge[0]} {edge[1]} 0\n")

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    return parser.parse_args()

def fetch_min_timestamp(args):
    """병렬 처리를 위한 헬퍼 함수: 필요한 timestamp 컬럼만 읽어 메모리를 최소화합니다."""
    chain, addr = args
    file_path = f'../../_data/dataset/transactions/{chain}/{addr}.csv'
    try:
        # 파일 전체가 아닌 'timestamp' 컬럼 하나만 읽어서 즉시 메모리 해제
        tx = pd.read_csv(file_path, usecols=['timestamp'])
        return {'address': addr, 'first_timestamp': tx['timestamp'].min()}
    except Exception as e:
        return None

def main():
    args = get_args()
    chain = str(args.chain)

    os.makedirs(os.path.dirname(f'../../_data/GoG/edges/{chain}/'), exist_ok=True)

    chain_labels = pd.read_csv(f'../../_data/dataset/labels.csv').query('Chain == @chain')
    chain_class = list(chain_labels.Contract.values)

    # 1. 병렬 처리 및 메모리 최소화 로딩
    print(f"Loading {len(chain_class)} files in parallel...")
    stats = []
    tasks = [(chain, addr) for addr in chain_class]
    
    # ProcessPoolExecutor를 이용해 다중 CPU 코어 활용
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()//2) as executor:
        for result in executor.map(fetch_min_timestamp, tasks):
            if result is not None:
                stats.append(result)
                
    timestamps = pd.DataFrame(stats)

    # create index mapping
    all_address = list(chain_labels.Contract.values)
    index_mapping = {addr: idx for idx, addr in enumerate(all_address)}

    # 명시적 메모리 해제
    del chain_labels
    gc.collect()

    edges = pd.read_csv(f'../../_data/GoG/{chain}/edges/global_edges.csv')
    edges_with_timestamps_sorted = process_data(chain, timestamps, index_mapping, edges)

    # 더 이상 필요 없는 데이터프레임 해제
    del edges
    del timestamps
    gc.collect()

    generate_train_test_data(edges_with_timestamps_sorted, chain)

if __name__ == "__main__":
    main()