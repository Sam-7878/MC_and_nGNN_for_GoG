# hierarchical_dataset.py (수정판)

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import json

class HierarchicalDataset(Dataset):
    """
    Hierarchical GNN용 데이터셋
    - Local graph: 각 contract의 transaction graph
    - Global graph: Contract-level graph
    """
    def __init__(self, data_dir, contract_graph_path, split='train'):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load contract graph
        print(f"📥 Loading contract graph from {contract_graph_path}")
        contract_data = torch.load(contract_graph_path, weights_only=False)
        
        # 🔧 FIX: 실제 저장된 키에 맞춰 수정
        print(f"📦 Available keys: {contract_data.keys()}")
        
        self.contract_edge_index = contract_data['edge_index']
        
        # 'embeddings' 또는 'node_features' 등 다른 이름일 수 있음
        if 'embeddings' in contract_data:
            self.contract_features = contract_data['embeddings']
        elif 'node_features' in contract_data:
            self.contract_features = contract_data['node_features']
        elif 'features' in contract_data:
            self.contract_features = contract_data['features']
        else:
            print("⚠️  No embeddings found, will initialize randomly")
            num_contracts = contract_data['num_nodes']
            self.contract_features = None  # Will be learned
        
        # Labels
        if 'labels' in contract_data:
            self.labels = contract_data['labels']
        elif 'y' in contract_data:
            self.labels = contract_data['y']
        else:
            print("⚠️  No labels found in contract graph")
            self.labels = None
        
        print(f"✅ Contract graph loaded:")
        print(f"   Nodes: {contract_data['num_nodes']}")
        print(f"   Edges: {self.contract_edge_index.shape[1]}")
        if self.contract_features is not None:
            print(f"   Features: {self.contract_features.shape}")
        
        # Load local graphs
        print(f"\n📥 Loading local transaction graphs...")
        self.local_graphs = []
        
        graph_files = sorted(self.data_dir.glob("*.json"))
        
        for idx, file in enumerate(graph_files):
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Convert to PyG Data
            edges = data.get('edges', [])
            if len(edges) == 0:
                print(f"⚠️  Skipping {file.name}: no edges")
                continue
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            # Features
            num_nodes = data.get('num_nodes', edge_index.max().item() + 1)
            
            if 'features' in data and len(data['features']) > 0:
                # 🔧 FIX: features가 list of lists인지 확인
                features = data['features']
                if isinstance(features[0], list):
                    x = torch.tensor(features, dtype=torch.float32)
                else:
                    # features가 scalar면 degree 사용
                    degree = torch.zeros(num_nodes, 1)
                    for src, dst in edge_index.t().tolist():
                        degree[src] += 1
                        degree[dst] += 1
                    x = degree
            else:
                # Degree as feature
                degree = torch.zeros(num_nodes, 1)
                for src, dst in edge_index.t().tolist():
                    degree[src] += 1
                    degree[dst] += 1
                x = degree
            
            # Edge features
            num_edges = edge_index.size(1)
            edge_attr = torch.ones(num_edges, 1)
            
            # Label (contract-level)
            if self.labels is not None and idx < len(self.labels):
                label = self.labels[idx].item()
            else:
                label = 0  # Default
            
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                contract_id=idx,
                y=label
            )
            
            self.local_graphs.append(graph_data)
        
        print(f"✅ Loaded {len(self.local_graphs)} local graphs")
        
        # Split indices
        self.indices = self._get_split_indices()
        print(f"✅ Split '{split}': {len(self.indices)} samples")
    
    def _get_split_indices(self):
        """Split dataset (70/15/15)"""
        n = len(self.local_graphs)
        indices = torch.randperm(n)
        
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        if self.split == 'train':
            return indices[:train_size].tolist()
        elif self.split == 'val':
            return indices[train_size:train_size+val_size].tolist()
        else:  # test
            return indices[train_size+val_size:].tolist()
    
    def len(self):
        return len(self.indices)
    
    def get(self, idx):
        """Get single local graph"""
        real_idx = self.indices[idx]
        return self.local_graphs[real_idx]
    
    def get_global_graph(self):
        """Get contract-level graph"""
        return {
            'edge_index': self.contract_edge_index,
            'features': self.contract_features,
            'labels': self.labels
        }
    
    def get_num_contracts(self):
        """Total number of contracts"""
        return len(self.local_graphs)


class HierarchicalBatchSampler:
    """커스텀 배치 샘플러"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield batch_indices
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def hierarchical_collate_fn(batch_list):
    """
    Custom collate function
    """
    from torch_geometric.data import Batch
    
    # Batch local graphs
    local_batch = Batch.from_data_list(batch_list)
    
    # Extract info
    contract_ids = torch.tensor([data.contract_id for data in batch_list])
    labels = torch.tensor([data.y for data in batch_list])
    
    return {
        'local_batch': local_batch,
        'contract_ids': contract_ids,
        'labels': labels
    }


# Test
if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing HierarchicalDataset")
    print("="*60)
    
    # Load dataset
    dataset = HierarchicalDataset(
        data_dir="../../_data/GoG/polygon",
        contract_graph_path="../../_data/GoG/polygon/polygon_hybrid.pt",
        split='train'
    )
    
    print(f"\n📊 Dataset Info:")
    print(f"  Total samples: {len(dataset)}")
    
    # Sample data
    sample = dataset[0]
    print(f"\n📌 Sample data:")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Contract ID: {sample.contract_id}")
    print(f"  Label: {sample.y}")
    
    # Global graph
    global_graph = dataset.get_global_graph()
    print(f"\n🌐 Global graph:")
    print(f"  Edges: {global_graph['edge_index'].shape}")
    if global_graph['features'] is not None:
        print(f"  Features: {global_graph['features'].shape}")
    if global_graph['labels'] is not None:
        print(f"  Labels: {global_graph['labels'].shape}")
    
    # Test DataLoader
    print(f"\n📦 Testing DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=hierarchical_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"  Batch size: {len(batch['contract_ids'])}")
    print(f"  Local batch nodes: {batch['local_batch'].num_nodes}")
    print(f"  Local batch edges: {batch['local_batch'].num_edges}")
    print(f"  Contract IDs: {batch['contract_ids']}")
    print(f"  Labels: {batch['labels']}")
    
    print(f"\n✅ All tests passed!")

