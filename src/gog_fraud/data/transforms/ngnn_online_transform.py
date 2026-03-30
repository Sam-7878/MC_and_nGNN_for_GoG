# src/gog_fraud/data/transforms/ngnn_online_transform.py

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from src.gog_fraud.data.preprocessing.ngnn.subgraph_extraction import extract_rooted_subgraphs

class nGNNOnlineTransform(BaseTransform):
    """
    A PyG Data Transform that dynamically extracts k-hop rooted subgraphs
    during the DataLoader batching process.
    
    This is useful for hybrid approaches, debugging, or applying MC dropout 
    augmentations on the fly before nested extraction.
    """
    def __init__(
        self, 
        num_hops: int = 1, 
        max_nodes_per_subgraph: int = 50, 
        root_policy: str = "all"
    ):
        self.num_hops = num_hops
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.root_policy = root_policy

    def __call__(self, data: Data) -> Data:
        """
        Takes a single Level 1 graph PyG Data object and returns a Batch-like 
        Data object containing multiple disconnected rooted subgraphs.
        
        Note: The nested encoder expects either a list of subgraphs or a 
        disjoint union of subgraphs (like a PyG Batch) for a single sample.
        For simplicity, we store the subgraphs as a list in a custom attribute 
        or return a PyG Batch natively.
        To maintain compatibility with standard PyG data loaders, we store 
        the disconnected subgraphs using `torch_geometric.data.Batch.from_data_list`.
        """
        from torch_geometric.data import Batch

        subgraphs = extract_rooted_subgraphs(
            data=data,
            num_hops=self.num_hops,
            max_nodes_per_subgraph=self.max_nodes_per_subgraph,
            root_policy=self.root_policy
        )

        if not subgraphs:
            # Fallback for empty subgraphs
            sub_data = Data(x=data.x if data.x is not None else torch.ones((1, 1)),
                            edge_index=torch.empty((2, 0), dtype=torch.long))
            sub_data.root_node_idx = torch.tensor([0], dtype=torch.long)
            sub_data.root_indicator = torch.tensor([1.0], dtype=torch.float)
            sub_data.original_node_ids = torch.tensor([0], dtype=torch.long)
            subgraphs = [sub_data]

        # Combine all rooted subgraphs for this Level 1 sample into a Batch
        nested_batch = Batch.from_data_list(subgraphs)
        
        # PyG sets nested_batch.batch as mapping from node -> subgraph 
        # We rename it to subgraph_idx so the outer DataLoader doesn't overwrite it
        nested_batch.subgraph_idx = nested_batch.batch
        # We can delete .batch or leave it; PyG data collate will overwrite it 
        # with the parent graph mapping [0...batch_size-1].
        del nested_batch.batch
        
        # We also want to preserve the original sample ID
        for key in ['sample_id', 'contract_id', 'label', 'split', 'y']:
            if hasattr(data, key):
                setattr(nested_batch, key, getattr(data, key))

        # Attach original data reference if needed (optional, takes memory)
        # nested_batch.original_data = data
        
        return nested_batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_hops={self.num_hops}, max_nodes={self.max_nodes_per_subgraph})"
