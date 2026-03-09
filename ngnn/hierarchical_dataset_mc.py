#!/usr/bin/env python3
"""
Hierarchical Dataset for Graph-of-Graphs
Combines local transaction graphs with global contract graph
(MC Data-level Edge Dropout & CSV Logging Enabled)
"""

import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge
import argparse
from datetime import datetime

def hierarchical_collate_fn(batch_list):
    local_graphs = [sample['local_graph'] for sample in batch_list]
    local_batch = Batch.from_data_list(local_graphs)
    
    contract_ids = torch.tensor([sample['contract_id'] for sample in batch_list], dtype=torch.long)
    labels = torch.tensor([sample['label'] for sample in batch_list], dtype=torch.long)
    global_graph = batch_list[0]['global_graph']
    
    return {
        'local_batch': local_batch,
        'contract_ids': contract_ids,
        'labels': labels,
        'global_edge_index': global_graph['edge_index'],
        'global_features': global_graph['features'],
        'global_labels': global_graph['labels']
    }

class HierarchicalDatasetMC(Dataset):
    def __init__(self, data_dir, contract_graph_path, split='train', mc_edge_dropout=0.1):
        self.data_dir = Path(data_dir)
        self.split = split
        self.mc_edge_dropout = mc_edge_dropout
        self.contract_graph = torch.load(contract_graph_path)
        self.labels = self.contract_graph['labels']
        self._load_local_graphs()

    def _load_local_graphs(self):
        self.valid_contracts = []
        labels_list = []
        for pt_file in self.data_dir.glob('*.pt'):
            contract_idx = int(pt_file.stem)
            if contract_idx < len(self.labels):
                self.valid_contracts.append(contract_idx)
                labels_list.append(self.labels[contract_idx].item())
        
        np.random.seed(42)
        indices = np.random.permutation(len(self.valid_contracts))
        train_end = int(0.8 * len(indices))
        val_end = int(0.9 * len(indices))
        
        if self.split == 'train':
            split_indices = indices[:train_end]
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:
            split_indices = indices[val_end:]
            
        self.split_contracts = [self.valid_contracts[i] for i in split_indices]
        
    def __len__(self):
        return len(self.split_contracts)

    def __getitem__(self, idx):
        contract_idx = self.split_contracts[idx]
        local_graph_path = self.data_dir / f"{contract_idx}.pt"
        local_graph = torch.load(local_graph_path)
        
        # 💡 Data-level Monte Carlo: 무작위 Edge Dropout 적용 (Train 시에만)
        if self.split == 'train' and self.mc_edge_dropout > 0.0:
            edge_index, _ = dropout_edge(local_graph.edge_index, p=self.mc_edge_dropout, training=True)
            local_graph.edge_index = edge_index
            
        label = self.labels[contract_idx].item()
        
        return {
            'local_graph': local_graph,
            'contract_id': contract_idx,
            'label': label,
            'global_graph': self.contract_graph
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='bsc')
    parser.add_argument('--edge_dropout', type=float, default=0.15) # MC 적용 확률
    args = parser.parse_args()
    
    data_dir = f"../../_data/GoG/{args.chain}/local_graphs"
    contract_graph_path = f"../../_data/GoG/{args.chain}/{args.chain}_hybrid_graph.pt"
    
    dataset = HierarchicalDatasetMC(
        data_dir=data_dir,
        contract_graph_path=contract_graph_path,
        split='train',
        mc_edge_dropout=args.edge_dropout
    )
    
    label_0 = sum(1 for c in dataset.valid_contracts if dataset.labels[c].item() == 0)
    label_1 = sum(1 for c in dataset.valid_contracts if dataset.labels[c].item() == 1)
    
    sample = dataset[0] # Edge Dropout이 적용된 첫 번째 샘플 로드
    
    results = [{
        "Timestamp": datetime.now().strftime("%Y:%m:%d_%H:%M:%S"),
        "Dataset Version": f"MC Edge Dropout (p={args.edge_dropout})",
        "Chain": args.chain,
        "Split": dataset.split,
        "Total Contracts loaded": len(dataset.valid_contracts),
        "Train Samples": len(dataset),
        "Label 0 Count": label_0,
        "Label 1 Count": label_1,
        "Sample Local Nodes": sample['local_graph'].num_nodes,
        "Sample Local Edges": sample['local_graph'].edge_index.shape[1] # MC로 인해 엣지가 줄어듦
    }]
    
    RESULT_PATH = f"../../_data/results/ngnn"
    os.makedirs(RESULT_PATH, exist_ok=True)
    csv_file = f'{RESULT_PATH}/{args.chain}_mc_dataset_stats_log.csv'
    
    df = pd.DataFrame(results)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)
        
    print(f"✅ Saved MC applied dataset statistics to {csv_file}")