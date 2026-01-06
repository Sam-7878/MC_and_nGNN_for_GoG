import snap
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import json


def build_snap_graph(tx):
    G = snap.TNGraph.New()
    node_dict = {}  # Dictionary to map Ethereum addresses to node IDs
    node_id = 0     # Initial node ID
    
    for index, row in tx.iterrows():
        from_addr = row['from']
        to_addr = row['to']
        
        if from_addr not in node_dict:
            node_dict[from_addr] = node_id
            G.AddNode(node_id)
            node_id += 1
        if to_addr not in node_dict:
            node_dict[to_addr] = node_id
            G.AddNode(node_id)
            node_id += 1
        
        G.AddEdge(node_dict[from_addr], node_dict[to_addr])
    
    return G

def compute_metrics(G):
    effective_diameter = snap.GetBfsEffDiam(G, 100, False)
    clustering_coefficient = snap.GetClustCf(G, -1)
    
    return effective_diameter, clustering_coefficient

def main():
    chain = 'polygon'

    chain_labels = pd.read_csv(f'../../../_data/data/labels.csv').query('Chain == @chain')
    chain_class = list(chain_labels.Contract.values)

    output_file = '../../../_data/result/'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    stats = []
    for addr in tqdm(chain_class):
        try:
            tx = pd.read_csv(f'../../../_data/data/transactions/{chain}/{addr}.csv')
            tx['timestamp'] = pd.to_datetime(tx['timestamp'], unit='s')
            end_date = pd.Timestamp('2024-03-01')
            tx = tx[tx['timestamp'] < end_date]

            G = build_snap_graph(tx)
            effective_diameter, clustering_coefficient = compute_metrics(G)
            
            stats.append({
                'Contract': addr,
                'Effective_Diameter': effective_diameter,
                'Clustering_Coefficient': clustering_coefficient
            })
        except Exception as e:
            print(f'Error for address {addr}: {e}')

    df = pd.DataFrame(stats)
    df.to_csv(f'../../../_data/result/{chain}_advanced_metrics_labels.csv', index=False)

if __name__ == "__main__":
    main()
