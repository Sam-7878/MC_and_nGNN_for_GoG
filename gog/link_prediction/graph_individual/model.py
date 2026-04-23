import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool

class GCN2(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = nn.Linear(2 * 16, 16)  # Process edge features from node pairs
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_features = F.relu(self.fc1(edge_features))
        out = self.fc2(edge_features)
        return F.log_softmax(out, dim=1)

class GIN2(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIN2, self).__init__()
        self.conv1 = GINConv(nn.Linear(num_features, 16), train_eps=False)
        self.conv2 = GINConv(nn.Linear(16, 16), train_eps=False)
        self.fc1 = nn.Linear(2 * 16, 16)  # Process edge features from node pairs
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_features = F.relu(self.fc1(edge_features))
        out = self.fc2(edge_features)
        return F.log_softmax(out, dim=1)

class GraphSage2(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSage2, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, 16)
        self.fc1 = nn.Linear(2 * 16, 16)  # Process edge features from node pairs
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_features = F.relu(self.fc1(edge_features))
        out = self.fc2(edge_features)
        return F.log_softmax(out, dim=1)

class GAT2(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT2, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=1)
        self.conv2 = GATConv(16, 16, heads=1)
        self.fc1 = nn.Linear(2 * 16, 16)  # Process edge features from node pairs
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_features = F.relu(self.fc1(edge_features))
        out = self.fc2(edge_features)
        return F.log_softmax(out, dim=1)

class ResidualGCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0.5):
        super(ResidualGCN, self).__init__()
        self.dropout = dropout

        # Define GCN layers
        self.initial_conv = GCNConv(num_features, hidden_dim)
        self.residual_conv = GCNConv(hidden_dim, hidden_dim)  # Middle residual layer
        self.final_conv = GCNConv(hidden_dim, num_classes)  # Final layer outputs class scores

        # Linear layers for processing edge features
        self.fc1 = nn.Linear(2 * num_classes, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initial convolution
        x = F.relu(self.initial_conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Residual connection
        initial = x
        x = F.relu(self.residual_conv(x, edge_index)) + initial
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Final convolution
        x = self.final_conv(x, edge_index)
        
        # Concatenating edge features and processing
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_features = F.relu(self.fc1(edge_features))
        out = self.fc2(edge_features)

        return F.log_softmax(out, dim=1)