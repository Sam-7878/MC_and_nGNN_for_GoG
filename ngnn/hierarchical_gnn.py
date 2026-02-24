# ngnn/hierarchical_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool
from torch_geometric.utils import softmax

class LocalGNN(nn.Module):
    """Transaction-level GNN"""
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim, hidden_dim,
                    heads=4, concat=False,
                    dropout=dropout,
                    edge_dim=hidden_dim
                )
            )
            self.norms.append(BatchNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res  # Residual
        
        return x


class AttentionPooling(nn.Module):
    """Transaction → Contract aggregation with attention"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        attn_scores = self.attention(x)
        attn_weights = softmax(attn_scores, batch)
        
        x_weighted = x * attn_weights
        return global_mean_pool(x_weighted, batch)


class GlobalGNN(nn.Module):
    """Contract-level GNN"""
    def __init__(self, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            )
            self.norms.append(BatchNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res
        
        return x


class HierarchicalGNN(nn.Module):
    """
    Full Hierarchical GNN for Graph-of-Graphs
    """
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_classes,
        num_local_layers=2,
        num_global_layers=2,
        dropout=0.3,
        use_global_gnn=True
    ):
        super().__init__()
        
        self.use_global_gnn = use_global_gnn
        
        # Local GNN (Transaction-level)
        self.local_gnn = LocalGNN(
            node_dim, edge_dim, hidden_dim,
            num_layers=num_local_layers,
            dropout=dropout
        )
        
        # Pooling (Transaction → Contract)
        self.pooling = AttentionPooling(hidden_dim)
        
        # Global GNN (Contract-level)
        if use_global_gnn:
            self.global_gnn = GlobalGNN(
                hidden_dim,
                num_layers=num_global_layers,
                dropout=dropout
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    


    def forward(self, batch):
        """
        Args:
            batch: Dictionary from hierarchical_collate_fn
        """
        # Local GNN
        local_batch = batch['local_batch']
        local_embeddings = self.local_gnn(
            local_batch.x, 
            local_batch.edge_index, 
            local_batch.edge_attr
        )
        
        # Pooling to get contract-level embeddings
        contract_embeddings = global_mean_pool(
            local_embeddings, 
            local_batch.batch
        )
        
        # Global GNN
        global_edge_index = batch['global_edge_index']
        global_features = batch['global_features']
        contract_ids = batch['contract_ids']
        
        # Combine local and global features
        enhanced_features = torch.cat([
            contract_embeddings,
            global_features[contract_ids]
        ], dim=1)
        
        # Final prediction
        output = self.global_gnn(
            enhanced_features,
            global_edge_index
        )
        
        return output[contract_ids]


    # def forward(self, local_batch, contract_edge_index, contract_ids):
    #     """
    #     Args:
    #         local_batch: PyG Batch of transaction graphs
    #         contract_edge_index: [2, E] global edges
    #         contract_ids: [B] contract IDs in this batch
        
    #     Returns:
    #         logits: [B, num_classes]
    #     """
    #     # Step 1: Local GNN
    #     tx_embeddings = self.local_gnn(
    #         x=local_batch.x,
    #         edge_index=local_batch.edge_index,
    #         edge_attr=local_batch.edge_attr,
    #         batch=local_batch.batch
    #     )
        
    #     # Step 2: Pooling
    #     contract_embeddings = self.pooling(tx_embeddings, local_batch.batch)
        
    #     # Step 3: Global GNN (optional)
    #     if self.use_global_gnn:
    #         # Build subgraph for this batch
    #         # (In practice, use full contract graph but extract batch contracts)
    #         global_embeddings = self.global_gnn(
    #             contract_embeddings,
    #             contract_edge_index
    #         )
    #     else:
    #         global_embeddings = contract_embeddings
        
    #     # Step 4: Classification
    #     logits = self.classifier(global_embeddings)
        
    #     return logits
    
    def forward_with_mcdropout(self, *args, n_samples=10, **kwargs):
        """MC-Dropout for uncertainty"""
        self.train()  # Enable dropout
        
        logits_list = []
        for _ in range(n_samples):
            logits = self.forward(*args, **kwargs)
            logits_list.append(logits)
        
        logits_stack = torch.stack(logits_list)  # [n_samples, B, C]
        
        mean_logits = logits_stack.mean(dim=0)
        std_logits = logits_stack.std(dim=0)
        
        return mean_logits, std_logits
