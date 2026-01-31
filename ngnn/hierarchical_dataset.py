# ngnn/hierarchical_dataset.py (수정판)

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import json


class HierarchicalDataset(Dataset):
    """Hierarchical Dataset for Contract-level and Transaction-level Graphs"""

    def __init__(self, data_dir, contract_graph_path, split='train'):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
            
        # Load contract graph
        print(f"📥 Loading contract graph from {contract_graph_path}")
        contract_data = torch.load(contract_graph_path, weights_only=False)
        
        print(f"📦 Available keys: {list(contract_data.keys())}")
        
        self.contract_edge_index = contract_data['edge_index']
        self.contract_to_idx = contract_data['contract_to_idx']
        self.idx_to_contract = contract_data['idx_to_contract']
        
        # 🔧 FIX: embeddings와 labels가 없으면 나중에 계산
        self.contract_features = contract_data.get('embeddings', None)
        self.labels = contract_data.get('labels', None)
        
        num_contracts = contract_data.get('num_nodes', len(self.idx_to_contract))
        
        print(f"✅ Contract graph loaded:")
        print(f"   Nodes: {num_contracts}")
        print(f"   Edges: {self.contract_edge_index.shape[1]}")
        
        # Load local graphs
        print(f"\n📥 Loading local transaction graphs...")
        self.local_graphs = []
        
        graph_files = sorted(self.data_dir.glob("*.json"))
        
        # 🔧 labels를 여기서 수집
        collected_labels = []
        
        for idx, file in enumerate(graph_files):
            if idx >= num_contracts:
                break
                
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Convert to PyG Data
            edges = data.get('edges', [])
            if len(edges) == 0:
                continue
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            # Node features
            num_nodes = data.get('num_nodes', edge_index.max().item() + 1)
            
            # Degree as features
            degree = torch.zeros(num_nodes, 1)
            for src, dst in edge_index.t().tolist():
                degree[src] += 1
                degree[dst] += 1
            x = degree
            
            # Edge features
            num_edges = edge_index.size(1)
            edge_attr = torch.ones(num_edges, 1)
            
            # 🔧 Label 수집 (JSON에서)
            label = data.get('label', 0)
            collected_labels.append(label)
            
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                contract_id=idx,
                y=label
            )
            
            self.local_graphs.append(graph_data)
        
        # 🔧 labels가 저장되지 않았으면 수집한 것 사용
        if self.labels is None:
            self.labels = torch.tensor(collected_labels, dtype=torch.long)
            print(f"✅ Collected labels from JSON files")
        
        # 🔧 embeddings가 없으면 나중에 학습 가능하도록 None으로 유지
        if self.contract_features is None:
            print(f"⚠️  No pre-computed embeddings, will be learned during training")
            # 초기 임베딩 생성 (랜덤 또는 degree 기반)
            self.contract_features = torch.randn(len(self.local_graphs), 8)
        
        print(f"✅ Loaded {len(self.local_graphs)} local graphs")
        
        # Label distribution
        label_counts = torch.bincount(self.labels)
        print(f"\n📊 Label distribution:")
        for i, count in enumerate(label_counts):
            print(f"   Label {i}: {count.item()} contracts")
        
        # Split indices
        # [수정 전] 에러 발생 원인
        # self.indices = [i for i, label in enumerate(labels) if ...] 
        
        # [수정 후] 변수명 변경 (indices -> split_indices)
        all_indices = range(label_counts.size(0))
        
        # Split 로직 (예시)
        # 실제 구현하신 split 로직에 맞춰 변수명만 바꾸면 됩니다.
        if split == 'train':
            # 예: 70% 학습
            self.split_indices = [i for i in all_indices if (i % 10) < 7]
        elif split == 'val':
             self.split_indices = [i for i in all_indices if 7 <= (i % 10) < 8]
        elif split == 'test':
             self.split_indices = [i for i in all_indices if (i % 10) >= 8]
        else:
            self.split_indices = list(all_indices)

        print(f"✅ Split '{split}': {len(self.split_indices)} samples")

    
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
        # [수정 전] return len(self.indices)
        # [수정 후]
        return len(self.split_indices)
    
    def get(self, idx):
        # [수정 전] real_idx = self.indices[idx]
        # [수정 후] 매핑된 실제 인덱스 가져오기
        real_idx = self.split_indices[idx]
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
    print(f"\n📋 Batch Info:")
    print(f"  Batch size: {len(batch['contract_ids'])}")
    print(f"  Local batch nodes: {batch['local_batch'].num_nodes}")
    print(f"  Local batch edges: {batch['local_batch'].num_edges}")
    print(f"  Contract IDs: {batch['contract_ids']}")
    print(f"  Labels: {batch['labels']}")
    
    print(f"\n✅ All tests passed!")

