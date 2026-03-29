import torch
from typing import List, Optional
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def extract_rooted_subgraphs(
    data: Data,
    num_hops: int = 1,
    max_nodes_per_subgraph: int = 100,
    root_policy: str = "all"
) -> List[Data]:
    """
    Extracts k-hop rooted subgraphs from a given PyG Data object.
    
    Args:
        data: A PyG Data object representing the Level 1 graph.
        num_hops: Number of hops for the nested subgraph.
        max_nodes_per_subgraph: Budget for maximum nodes to prevent OOM.
        root_policy: "all" (every node is a root) or potentially specific indices.
        
    Returns:
        A list of PyG Data objects, each representing a rooted subgraph.
        Each graph contains extra attributes:
            - root_node_idx: The index of the root in the SUBGRAPH's node list.
            - original_node_ids: The mapping back to the original graph's node indices.
            - root_indicator: A boolean/float mask indicating which node is the root.
            - hop_ids: (Optional) The distance of each node from the root.
    """
    subgraphs = []
    
    num_nodes = data.num_nodes
    if num_nodes is None:
        if data.x is not None:
            num_nodes = data.x.size(0)
        else:
            num_nodes = data.edge_index.max().item() + 1 if data.edge_index.numel() > 0 else 0

    # Determine root nodes
    if root_policy == "all":
        roots = range(num_nodes)
    else:
        # Extend here for specific root selection, e.g. center contract 
        roots = range(num_nodes)

    for root in roots:
        # k_hop_subgraph returns: (subset, edge_index, mapping, edge_mask)
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            root, num_hops, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        # Apply max nodes budget
        if subset.size(0) > max_nodes_per_subgraph:
            # Simple truncation policy: Keep the root, and the first N-1 nodes 
            # (In a more complex setup, we'd sort by hop distance)
            # Find the index of root in the subset
            root_idx_in_subset = (subset == root).nonzero(as_tuple=True)[0].item()
            
            keep_indices = [root_idx_in_subset]
            for i in range(subset.size(0)):
                if len(keep_indices) >= max_nodes_per_subgraph:
                    break
                if i != root_idx_in_subset:
                    keep_indices.append(i)
                    
            keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=subset.device)
            # Re-filter subset and edge_index
            subset = subset[keep_tensor]
            
            # Filter edges where both endpoints are in keep_tensor
            # This requires some non-trivial mapping, for simplicity in MVP we 
            # just use PyG's subgraph utility again on the new subset
            from torch_geometric.utils import subgraph
            sub_edge_index, _ = subgraph(keep_tensor, sub_edge_index, relabel_nodes=True, num_nodes=len(keep_indices))
            
            # The root is always at index 0 now
            new_root_idx = 0
            mapping = torch.tensor([0], dtype=torch.long) # Root maps to 0
        else:
            new_root_idx = mapping.item() if mapping.numel() == 1 else (subset == root).nonzero(as_tuple=True)[0].item()

        # Create the new Data object
        sub_data = Data(edge_index=sub_edge_index)
        
        if data.x is not None:
            sub_data.x = data.x[subset]
        else:
            # If no features, create dummy ones
            sub_data.x = torch.ones((subset.size(0), 1), dtype=torch.float)
            
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # We need to filter edge_attr using edge_mask
            sub_data.edge_attr = data.edge_attr[edge_mask]
            
        # Add required custom attributes
        sub_data.root_node_idx = torch.tensor([new_root_idx], dtype=torch.long)
        
        root_indicator = torch.zeros(subset.size(0), dtype=torch.float)
        root_indicator[new_root_idx] = 1.0
        sub_data.root_indicator = root_indicator
        
        sub_data.original_node_ids = subset
        
        # Copy metadata 
        for key in ['sample_id', 'contract_id', 'label', 'split']:
            if hasattr(data, key):
                setattr(sub_data, key, getattr(data, key))
                
        subgraphs.append(sub_data)

    return subgraphs
