import pandas as pd
import json
import random
import torch
import os

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

def generate_train_test_data(edges_with_timestamps_sorted, chain):
    # Splitting indices for train and test
    train_num = int(len(edges_with_timestamps_sorted) * 0.8)
    train_data = edges_with_timestamps_sorted.iloc[:train_num]
    test_data = edges_with_timestamps_sorted.iloc[train_num:]

    # Nodes from the entire dataset to ensure test set can also have negative samples
    all_nodes = set(edges_with_timestamps_sorted['graph_1']).union(set(edges_with_timestamps_sorted['graph_2']))

    # Generate negative edges for training using only train nodes
    train_nodes = set(train_data['graph_1']).union(set(train_data['graph_2']))
    train_possible_edges = set((i, j) for i in train_nodes for j in train_nodes if i != j)
    train_existing_edges = set(zip(train_data['graph_1'], train_data['graph_2']))
    train_non_edges = list(train_possible_edges - train_existing_edges)
    random.shuffle(train_non_edges)
    train_negative_edges = train_non_edges[:len(train_data)]

    # Generate negative edges for testing using all nodes
    test_possible_edges = set((i, j) for i in all_nodes for j in all_nodes if i != j)
    test_existing_edges = set(zip(test_data['graph_1'], test_data['graph_2']))
    test_non_edges = list(test_possible_edges - test_existing_edges)
    random.shuffle(test_non_edges)
    test_negative_edges = test_non_edges[:len(test_data)]

    # Save train and test edges with labels
    with open(f'../../_data/GoG/edges/{chain}/{chain}_train_edges.txt', 'w') as f:
        for edge in train_data[['graph_1', 'graph_2']].itertuples(index=False):
            f.write(f"{edge.graph_1} {edge.graph_2} 1\n")
        for edge in train_negative_edges:
            f.write(f"{edge[0]} {edge[1]} 0\n")

    with open(f'../../_data/GoG/edges/{chain}/{chain}_test_edges.txt', 'w') as f:
        for edge in test_data[['graph_1', 'graph_2']].itertuples(index=False):
            f.write(f"{edge.graph_1} {edge.graph_2} 1\n")
        for edge in test_negative_edges:
            f.write(f"{edge[0]} {edge[1]} 0\n")

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    return parser.parse_args()


def main():
    args = get_args()
    chain = str(args.chain)

    os.makedirs(os.path.dirname(f'../../_data/GoG/edges/{chain}/'), exist_ok=True)

    chain_labels = pd.read_csv(f'../../_data/dataset/labels.csv').query('Chain == @chain')
    chain_class = list(chain_labels.Contract.values)

    # create timestamps
    stats = []
    for addr in chain_class:
        tx = pd.read_csv(f'../../_data/dataset/transactions/{chain}/{addr}.csv')
        first_timestamp = tx['timestamp'].min()
        stats.append({'address': addr, 'first_timestamp': first_timestamp})
        
    timestamps = pd.DataFrame(stats)

    # create index mapping
    all_address = list(chain_labels.Contract.values)
    index_mapping = {addr: idx for idx, addr in enumerate(all_address)}

    edges = pd.read_csv(f'../../_data/GoG/{chain}/edges/global_edges.csv')
    edges_with_timestamps_sorted = process_data(chain, timestamps, index_mapping, edges)

    generate_train_test_data(edges_with_timestamps_sorted, chain)

if __name__ == "__main__":
    main()