#!/usr/bin/env python3
"""
Hierarchical Dataset for Graph-of-Graphs
Combines local transaction graphs with global contract graph
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import argparse


def hierarchical_collate_fn(batch_list):
    """
    Custom collate function for hierarchical GNN
    
    Args:
        batch_list: List of samples from HierarchicalDataset
        
    Returns:
        Dictionary containing batched data
    """
    # Local graphs 배치화
    local_graphs = [sample['local_graph'] for sample in batch_list]
    local_batch = Batch.from_data_list(local_graphs)
    
    # Contract IDs
    contract_ids = torch.tensor([sample['contract_id'] for sample in batch_list], 
                                 dtype=torch.long)
    
    # Labels
    labels = torch.tensor([sample['label'] for sample in batch_list], 
                          dtype=torch.long)
    
    # Global graph (모든 샘플이 동일한 global graph 공유)
    global_graph = batch_list[0]['global_graph']
    
    return {
        'local_batch': local_batch,
        'contract_ids': contract_ids,
        'labels': labels,
        'global_edge_index': global_graph['edge_index'],
        'global_features': global_graph['features'],
        'global_labels': global_graph['labels']
    }


class HierarchicalDataset(Dataset):
    """
    Dataset for hierarchical GNN on cryptocurrency graphs
    
    Each sample contains:
    - local_graph: Transaction-level graph for one contract
    - global_graph: Contract-level graph (shared across all samples)
    - contract_id: Index of the contract
    - label: Contract label
    """
    
    def __init__(self, data_dir, contract_graph_path, split='train', 
                 train_ratio=0.6, val_ratio=0.2):
        """
        Args:
            data_dir: Directory containing local graph .pt files
            contract_graph_path: Path to contract-level graph
            split: 'train', 'val', or 'test'
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        
        print(f"📂 data_dir: {self.data_dir}")
        print(f"📂 data_dir exists: {self.data_dir.exists()}")
        print(f"📂 data_dir is directory: {self.data_dir.is_dir()}")
        
        # Load contract graph
        self._load_contract_graph(contract_graph_path)
        
        # Load local graphs
        self._load_local_graphs()
        
        # Create train/val/test splits
        self._create_splits(train_ratio, val_ratio)
        
        print(f"✅ Split '{split}': {len(self.contract_ids)} samples")
    
    def _load_contract_graph(self, contract_graph_path):
        """Load global contract-level graph"""
        print(f"📥 Loading contract graph from {contract_graph_path}")
        
        contract_data = torch.load(contract_graph_path, weights_only=False)
        
        print(f"📦 Available keys: {list(contract_data.keys())}")
        
        # Extract data
        self.global_edge_index = contract_data['edge_index']
        self.global_features = contract_data.get('embeddings', 
                                                  torch.randn(contract_data['num_nodes'], 128))
        self.global_labels = contract_data['labels']
        self.num_contracts = contract_data['num_nodes']
        
        # Mappings
        self.contract_to_idx = contract_data['contract_to_idx']
        self.idx_to_contract = contract_data['idx_to_contract']
        
        print(f"✅ Contract graph loaded:")
        print(f"   Nodes: {self.num_contracts}")
        print(f"   Edges: {self.global_edge_index.shape[1]}")
    
    def _load_local_graphs(self):
        """Load local transaction graphs"""
        print(f"\n📥 Loading local transaction graphs from {self.data_dir}...")
        
        self.local_graphs = {}
        
        if not self.data_dir.exists():
            print(f"❌ Directory does not exist: {self.data_dir}")
            return
        
        # Find all .pt files
        all_files = list(self.data_dir.glob("*.pt"))
        print(f"📁 Found {len(all_files)} .pt files in directory")
        
        # Show sample files
        for i, f in enumerate(all_files[:5]):
            print(f"   - {f.name}")
        if len(all_files) > 5:
            print(f"   ... and {len(all_files) - 5} more")
        
        # Load graphs
        loaded_count = 0
        for contract_id in range(self.num_contracts):
            local_graph_path = self.data_dir / f"{contract_id}.pt"
            
            if local_graph_path.exists():
                try:
                    self.local_graphs[contract_id] = torch.load(
                        local_graph_path, 
                        weights_only=False
                    )
                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️ Error loading {local_graph_path}: {e}")
        
        print(f"✅ Loaded {loaded_count} local graphs out of {self.num_contracts} contracts")
    
    def _create_splits(self, train_ratio, val_ratio):
        """Create train/val/test splits"""
        
        # Get all contract IDs
        all_contract_ids = list(range(self.num_contracts))
        
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(all_contract_ids)
        
        # Split
        n_train = int(len(all_contract_ids) * train_ratio)
        n_val = int(len(all_contract_ids) * val_ratio)
        
        train_ids = all_contract_ids[:n_train]
        val_ids = all_contract_ids[n_train:n_train + n_val]
        test_ids = all_contract_ids[n_train + n_val:]
        
        # Print label distribution
        print(f"\n📊 Label distribution:")
        unique_labels, counts = torch.unique(self.global_labels, return_counts=True)
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"   Label {label}: {count} contracts")
        
        # Select IDs based on split
        if self.split == 'train':
            self.contract_ids = train_ids
        elif self.split == 'val':
            self.contract_ids = val_ids
        else:  # test
            self.contract_ids = test_ids
    
    def __len__(self):
        """Return number of samples in this split"""
        return len(self.contract_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index within the split (0 to len(self)-1)
            
        Returns:
            Dictionary containing local graph, global graph info, and labels
        """
        # Get actual contract ID
        contract_id = self.contract_ids[idx]
        
        # Get local graph (transaction-level)
        local_graph = self.local_graphs.get(contract_id)
        
        # If no local graph, create empty one
        if local_graph is None:
            local_graph = Data(
                x=torch.zeros((1, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1)),
                num_nodes=1
            )
        
        # Get label
        label = self.global_labels[contract_id].item()
        
        # Prepare global graph info
        global_graph = {
            'edge_index': self.global_edge_index,
            'features': self.global_features,
            'labels': self.global_labels
        }
        
        return {
            'local_graph': local_graph,
            'global_graph': global_graph,
            'contract_id': contract_id,
            'label': label
        }


# ============================================================
# Test Code
# ============================================================

def test_dataset(chain='polygon'):
    """Test the HierarchicalDataset"""
    
    print("="*60)
    print("🧪 Testing HierarchicalDataset with Custom Collate")
    print("="*60)
    
    # Paths
    data_dir = f"../../_data/GoG/{chain}/local_graphs"
    contract_graph_path = f"../../_data/GoG/{chain}/{chain}_hybrid.pt"
    
    # Create dataset
    dataset = HierarchicalDataset(
        data_dir=data_dir,
        contract_graph_path=contract_graph_path,
        split='train',
        train_ratio=0.6,
        val_ratio=0.2
    )
    
    print(f"\n📊 Dataset Info:")
    print(f"  Total samples: {len(dataset)}")
    
    # Test single sample
    print(f"\n🔍 Testing single sample...")
    sample = dataset[0]
    
    print(f"  Local graph nodes: {sample['local_graph'].num_nodes}")
    print(f"  Local graph edges: {sample['local_graph'].edge_index.shape[1]}")
    print(f"  Contract ID: {sample['contract_id']}")
    print(f"  Label: {sample['label']}")
    
    # Test DataLoader
    print(f"\n📦 Testing DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=hierarchical_collate_fn
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n  Batch {batch_idx}:")
        print(f"    Local batch nodes: {batch['local_batch'].num_nodes}")
        print(f"    Local batch edges: {batch['local_batch'].edge_index.shape[1]}")
        print(f"    Contract IDs: {batch['contract_ids']}")
        print(f"    Labels: {batch['labels']}")
        print(f"    Global graph nodes: {batch['global_features'].shape[0]}")
        print(f"    Global graph edges: {batch['global_edge_index'].shape[1]}")
        
        if batch_idx >= 2:  # Only test first 3 batches
            break
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon', 
                        help='Blockchain name (polygon, ethereum, etc.)')
    args = parser.parse_args()
    
    test_dataset(chain=args.chain)
