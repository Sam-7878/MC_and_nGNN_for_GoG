import pandas as pd
import networkx as nx
from tqdm import tqdm
import random
import os
import numpy as np
import json

random.seed(1)

def calculate_stats(tx, end_date):
    G = nx.DiGraph()
    edges = [(row['from'], row['to'], {'weight': row['value']}) for index, row in tx[tx['timestamp'] < end_date].iterrows()]
    G.add_edges_from(edges)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G) if num_nodes > 0 else None
    assortativity = nx.degree_assortativity_coefficient(G) if num_nodes > 0 else None
    reciprocity = nx.overall_reciprocity(G) if num_nodes > 0 else None
    
    return num_nodes, num_edges, density, assortativity, reciprocity

def main():
    chain = 'polygon'

    chain_labels = pd.read_csv(f'../../_data/data/labels.csv').query('Chain == @chain')
    chain_class = list(chain_labels.Contract.values)

    output_file = '../../_data/results/'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    stats = []
    for addr in tqdm(chain_class):
        try:
            tx = pd.read_csv(f'../../_data/data/transactions/{chain}/{addr}.csv')
            tx['timestamp'] = pd.to_datetime(tx['timestamp'], unit='s')
            end_date = pd.Timestamp('2024-03-01')
            num_nodes, num_edges, density, assortativity, reciprocity = calculate_stats(tx, end_date)
            
            stats.append({
                'Contract': addr,
                'Num_nodes': num_nodes,
                'Num_edges': num_edges, 
                'Density': density,
                'Assortativity': assortativity,
                'Reciprocity': reciprocity,
            })
        except Exception as e:
            print(f'Error for address {addr}: {e}')
    
    pd.DataFrame(stats).to_csv(f'../../_data/results/{chain}_basic_metrics.csv', index=False)

if __name__ == "__main__":
    main()
