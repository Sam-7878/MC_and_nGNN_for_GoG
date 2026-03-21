import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import warnings
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import random
from torch_geometric.utils import dropout_edge

warnings.filterwarnings("ignore")

class TransactionDataset(InMemoryDataset):
    def __init__(self, root, transaction_dfs, labels, contract_addresses, chain, 
                 split='train', sample_size=10000, mc_samples=1, transform=None, pre_transform=None):
        self.transaction_dfs = transaction_dfs
        self.labels = labels
        self.contract_addresses = contract_addresses
        self.chain = chain
        self.split = split  # ✅ 수정: 값 할당 추가
        self.sample_size = sample_size
        self.mc_samples = mc_samples
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        return f'{self.split}_data.pt'

    def process(self):
        data_list = []
        print(f"Processing {len(self.transaction_dfs)} contracts...")
        for df, label, contract in tqdm(zip(self.transaction_dfs, self.labels, self.contract_addresses), 
                                        total=len(self.transaction_dfs)):
            for mc_idx in range(self.mc_samples):
                graph_data = self.graph_to_data_object(self.create_graph(df), label, contract, mc_idx)
                if graph_data is not None:
                    data_list.append(graph_data)
        
        print(f"Collating {len(data_list)} graphs...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"✅ Saved to {self.processed_paths[0]}")

    def create_graph(self, transaction_df):
        """Optimized graph creation with sampling"""
        if transaction_df.empty:
            return nx.DiGraph()
        
        # Sample recent transactions (anti-overfitting)
        if len(transaction_df) > self.sample_size:
            transaction_df = transaction_df.nlargest(self.sample_size, 'timestamp')
        
        # Normalize value
        transaction_df['value'] = pd.to_numeric(transaction_df['value'].astype(str).str.replace(',', ''), errors='coerce')
        transaction_df['value'].fillna(0, inplace=True)
        
        min_val, max_val = transaction_df['value'].min(), transaction_df['value'].max()
        if max_val > min_val:
            transaction_df['scaled_value'] = ((transaction_df['value'] - min_val) / (max_val - min_val)) * 100
        else:
            transaction_df['scaled_value'] = 1.0
        
        graph = nx.DiGraph()
        
        # Build graph from transactions
        for _, row in transaction_df.iterrows():
            from_addr = row['from']
            to_addr = row['to']
            graph.add_edge(from_addr, to_addr, 
                         weight=float(row.get('scaled_value', 1.0)), 
                         timestamp=int(row['timestamp']))
        
        # Node sampling (anti-overfitting) - 그래프 빌드 후 샘플링
        if graph.number_of_nodes() > 1000:
            nodes_to_keep = random.sample(list(graph.nodes()), 1000)
            graph = graph.subgraph(nodes_to_keep).copy()
        
        return graph

    def graph_to_data_object(self, graph, label, contract_address, mc_idx):
        """Convert NetworkX graph to PyG Data object"""
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            return None

        # ✅ Step 1: 그래프를 0부터 연속된 인덱스로 완전히 relabel
        nodelist = sorted(graph.nodes())
        if len(nodelist) == 0:
            return None
        
        mapping = {old_id: new_id for new_id, old_id in enumerate(nodelist)}
        graph_relabeled = nx.relabel_nodes(graph, mapping, copy=True)

        # ✅ Step 2: Relabeled 그래프로 sparse array 생성
        num_nodes = graph_relabeled.number_of_nodes()
        adj = nx.to_scipy_sparse_array(graph_relabeled, nodelist=list(range(num_nodes)), format='coo')
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        
        # ✅ Step 3: Edge attributes (이제 인덱스 일치 보장)
        edge_weights = []
        edge_timestamps = []
        for u, v in zip(adj.row, adj.col):
            edge_data = graph_relabeled.edges[int(u), int(v)]
            edge_weights.append(edge_data.get('weight', 1.0))
            edge_timestamps.append(edge_data.get('timestamp', 0))
        
        edge_attr = torch.tensor(list(zip(edge_weights, edge_timestamps)), dtype=torch.float)

        # ✅ Step 4: Node features (0 ~ num_nodes-1)
        degrees = np.array([graph_relabeled.degree(i) for i in range(num_nodes)])
        in_degrees = np.array([graph_relabeled.in_degree(i) for i in range(num_nodes)])
        out_degrees = np.array([graph_relabeled.out_degree(i) for i in range(num_nodes)])
        in_values = np.array([sum(d.get('weight', 0) for _, _, d in graph_relabeled.in_edges(i, data=True)) 
                              for i in range(num_nodes)])
        out_values = np.array([sum(d.get('weight', 0) for _, _, d in graph_relabeled.out_edges(i, data=True)) 
                               for i in range(num_nodes)])

        x = torch.tensor(np.column_stack([degrees, in_degrees, out_degrees, in_values, out_values]), 
                        dtype=torch.float)

        # ✅ Data augmentation (MC sampling 시)
        if self.mc_samples > 1 or mc_idx > 0:
            try:
                edge_index_dropout, _ = dropout_edge(edge_index, p=0.1)
                edge_index = edge_index_dropout
            except:
                pass  # dropout 실패 시 원본 유지

        # ✅ Graph statistics (Relabeled 그래프 사용)
        timestamps = [d.get('timestamp', 0) for _, _, d in graph_relabeled.edges(data=True)]
        if timestamps:
            min_ts, max_ts, avg_ts = min(timestamps), max(timestamps), np.mean(timestamps)
        else:
            min_ts, max_ts, avg_ts = 0, 0, 0
        
        # Local network stats (GoG integration)
        density = len(graph_relabeled.edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        try:
            avg_clust = np.mean(list(nx.clustering(graph_relabeled.to_undirected()).values())) if num_nodes > 2 else 0
        except:
            avg_clust = 0
        
        try:
            if nx.is_strongly_connected(graph_relabeled):
                ecc = np.mean(list(nx.eccentricity(graph_relabeled).values()))
            else:
                ecc = 0
        except:
            ecc = 0
        
        graph_stats = torch.tensor([min_ts, max_ts, avg_ts, density, avg_clust, ecc], dtype=torch.float)

        chain_index = chain_indexes.get(self.chain, 0)
        contract_index = all_address_index.get(contract_address, 0)

        graph_attr = torch.tensor([chain_index, contract_index, mc_idx], dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    num_nodes=num_nodes, graph_attr=graph_attr, graph_stats=graph_stats)
        return data


def load_single_tx_data(args_tuple):
    """병렬 처리용: 단일 컨트랙트 CSV 로드 + 전처리"""
    idx, contract, chain_dir, sample_size = args_tuple
    try:
        tx_path = f'{chain_dir}/{contract}.csv'
        if os.path.exists(tx_path):
            tx = pd.read_csv(tx_path, dtype={'timestamp': 'int64'})
            tx['date'] = pd.to_datetime(tx['timestamp'], unit='s')
            tx = tx.sort_values('timestamp').tail(sample_size)  # Latest tx
            return idx, tx
    except Exception as e:
        print(f"Error loading {contract}: {e}")
    return idx, pd.DataFrame()


def get_args():
    parser = argparse.ArgumentParser(description='Individual Transaction Dataset Builder')
    parser.add_argument('--chain', type=str, default='bsc', help='Chain: bsc, ethereum, polygon')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of classes to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'], 
                        help='Dataset split')
    parser.add_argument('--sample_size', type=int, default=10000, help='Max tx per contract')
    parser.add_argument('--mc_samples', type=int, default=1, help='Monte Carlo samples per graph')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    args = get_args()
    chain = args.chain.lower()
    
    # Labels load & split
    print(f"Loading labels for {chain}...")
    labels = pd.read_csv('../../_data/dataset/labels.csv').query('Chain == @chain')
    category_counts = labels['Category'].value_counts()
    select_class = list(category_counts.head(args.n_classes).index)
    category_to_label = {cat: i for i, cat in enumerate(select_class)}
    labels['label'] = labels['Category'].map(category_to_label)
    
    labels_select = labels.dropna(subset=['label']).reset_index(drop=True)
    print(f"Selected {len(labels_select)} contracts from {args.n_classes} classes: {select_class}")
    
    # Stratified Train/Val/Test Split (anti-overfitting)
    train_labels, temp_labels = train_test_split(labels_select, test_size=0.2, stratify=labels_select['label'], random_state=42)
    val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, stratify=temp_labels['label'], random_state=42)
    
    split_dfs = {'train': train_labels, 'val': val_labels, 'test': test_labels}
    
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    chain_indexes = {'bsc': 0, 'ethereum': 1, 'polygon': 2}
    chain_dir = f'../../_data/dataset/transactions/{chain}'
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split for {chain}...")
        print(f"{'='*60}")
        labels_split = split_dfs[split]
        
        # 병렬 CSV 로드 (속도 10x↑)
        cache_file = f'../../_data/dataset/.cache/{chain}_{split}_tx_cache.pkl'
        if os.path.exists(cache_file):
            print("✅ Loading cached transaction data...")
            with open(cache_file, 'rb') as f:
                transaction_dfs = pickle.load(f)
        else:
            print(f"📂 Parallel loading transaction CSVs ({args.workers} workers)...")
            tasks = [(i, contract, chain_dir, args.sample_size) 
                     for i, contract in enumerate(labels_split.Contract.values)]
            transaction_dfs = [None] * len(tasks)
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(load_single_tx_data, task): task[0] for task in tasks}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    idx, tx_df = future.result()
                    transaction_dfs[idx] = tx_df
            
            print("💾 Caching transaction data...")
            with open(cache_file, 'wb') as f:
                pickle.dump(transaction_dfs, f)
        
        # Filter valid
        valid_mask = [df is not None and not df.empty for df in transaction_dfs]
        transaction_dfs_valid = [df for df, valid in zip(transaction_dfs, valid_mask) if valid]
        labels_split = labels_split.iloc[[i for i, valid in enumerate(valid_mask) if valid]].reset_index(drop=True)
        
        print(f"Valid contracts: {len(transaction_dfs_valid)} / {len(transaction_dfs)}")
        
        all_address_index = dict(zip(labels_split.Contract, labels_split.index))
        
        # Dataset 생성
        print(f"\n🔨 Building PyG dataset...")
        dataset = TransactionDataset(
            root=f'../../_data/dataset/GCN/{chain}/{split}',
            transaction_dfs=transaction_dfs_valid,
            labels=labels_split.label.values.tolist(),
            contract_addresses=labels_split.Contract.values.tolist(),
            chain=chain,
            split=split,
            sample_size=args.sample_size,
            mc_samples=args.mc_samples
        )
        
        # Dataset 생성 후 실제 개수 확인
        data_pt = torch.load(dataset.processed_paths[0])
        num_saved_graphs = len(data_pt[1]['x']) - 1  # slices에서 계산
        print(f"\n✅ {split} dataset complete: {num_saved_graphs} graphs")
        print(f"   Saved to: ../../_data/dataset/GCN/{chain}/{split}/")
    
    print("\n" + "="*60)
    print("🎉 All datasets ready for Nested GNN + MC training!")
    print("="*60)

