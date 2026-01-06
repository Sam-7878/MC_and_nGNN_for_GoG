import json
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm
import torch
import pandas as pd


class JSONEncoderWithNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def load_contract_mapping(file_path):
    with open(file_path, 'r') as file:
        contract_mapping = json.load(file)
    return contract_mapping

def save_transaction_graph(df, label, idx, directory):
    os.makedirs(directory, exist_ok=True)
    unique_addresses = pd.concat([df['from'], df['to']]).unique()
    address_to_index = {address: i for i, address in enumerate(unique_addresses)}
    
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    in_value = defaultdict(float)
    out_value = defaultdict(float)
    
    edges = []
    for index, row in df.iterrows():
        from_idx = address_to_index[row['from']]
        to_idx = address_to_index[row['to']]
        value = float(row['value'].replace(',', '')) if isinstance(row['value'], str) else float(row['value'])
        
        edges.append([from_idx, to_idx])
        
        # Update degrees
        out_degree[from_idx] += 1
        in_degree[to_idx] += 1
        
        # Update transaction values
        out_value[from_idx] += value
        in_value[to_idx] += value

   
    features = {}
    for address, i in address_to_index.items():
        total_degree = in_degree[i] + out_degree[i]
        features[str(i)] = [total_degree, in_degree[i], out_degree[i], in_value[i], out_value[i]]

    # Construct the graph dictionary
    graph_dict = {
        "label": label,
        "features": features,
        "edges": edges
    }

    file_name = os.path.join(directory, f'{idx}.json')
    with open(file_name, 'w') as file:
        json.dump(graph_dict, file, cls=JSONEncoderWithNumpy, indent=None)  # No indentation for compactness

    print(f"Graph {idx} saved in {directory}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    chain = 'polygon'
    labels = pd.read_csv('../../_data/data/labels.csv').query('Chain == @chain').reset_index(drop=True)

    ### Use three-class as an example.
    n = 3
    category_counts = labels['Category'].value_counts()
    select_class = list(category_counts.head(n).index)
    category_to_label = {category: i for i, category in enumerate(select_class)}
    labels['Category'] = labels['Category'].map(category_to_label)
    select_address = list(labels.query('Category in @select_class').Contract.values)

    # read in full global_graph
    contract_mapping_file = f'../../_data/data/global_graph/{chain}_contract_to_number_mapping.json'
    contract_to_number = load_contract_mapping(contract_mapping_file)
    number_to_contract = {v: k for k, v in contract_to_number.items()}
    global_graph = pd.read_csv(f'../../_data/data/global_graph/{chain}_graph_more_than_1_ratio.csv') 

    global_graph['Contract1'] = global_graph['Contract1'].apply(lambda x: number_to_contract[x])
    global_graph['Contract2'] = global_graph['Contract2'].apply(lambda x: number_to_contract[x])
    labels_select = list(labels.query('Category < @n').Category.values)
    labels_select_df = labels.query('Category < @n').reset_index(drop = True)
    global_graph_select = global_graph.query('Contract1 in @select_address and Contract2 in @select_address')
    print(labels_select_df.Category.max())
    
    # read in transaction data
    transaction_dfs_select = []
    for i in tqdm(labels_select_df.Contract.values):
        tx = pd.read_csv(f'../../_data/data/transactions/{chain}/{i}.csv')
        tx['date'] = pd.to_datetime(tx['timestamp'], unit='s')
        transaction_dfs_select.append(tx)

    directory = f'../../_data/GoG/{chain}'
    for idx, (df, label) in enumerate(zip(transaction_dfs_select, labels_select)):
        save_transaction_graph(df, label, idx, directory)

    all_address_index = dict(zip(labels_select_df.Contract, labels_select_df.index))
    
    global_graph_select['graph_1'] = global_graph_select['Contract1'].apply(lambda x: int(all_address_index[x]))
    global_graph_select['graph_2'] = global_graph_select['Contract2'].apply(lambda x: int(all_address_index[x]))

    os.makedirs(f'../../_data/GoG/{chain}/edges/', exist_ok=True)
    global_graph_select[['graph_1', 'graph_2']].to_csv(f'../../_data/GoG/{chain}/edges/global_edges.csv', index = 0)
    
    

