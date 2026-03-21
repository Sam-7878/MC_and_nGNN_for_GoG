import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        self.convs = ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)  
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '\n'.join([str(conv) for conv in self.convs] + [str(self.fc)])

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.convs = ModuleList()
        
        self.convs.append(GINConv(Linear(num_features, hidden_dim), train_eps=True))
        self.convs.append(GINConv(Linear(hidden_dim, hidden_dim), train_eps=True))
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x) 
        
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '\n'.join([str(conv) for conv in self.convs] + [str(self.fc)])

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=2, hidden_dim=16, dropout=0, aggregation='mean'):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim, aggr=aggregation))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))
        
        self.dropout = dropout
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch) 
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
    def __repr__(self):
        return '\n'.join([str(conv) for conv in self.convs])


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=2, hidden_dim=16, heads=8, dropout=0):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList([GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)])
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        x = global_mean_pool(x, data.batch) 

        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '\n'.join([str(conv) for conv in self.convs])


class ResidualGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2, dropout=0):
        super(ResidualGCN, self).__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.initial_conv = GCNConv(num_features, hidden_dim)

        self.convs = ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.final_conv = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.initial_conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        initial = x
        for conv in self.convs:
            x = F.relu(conv(x, edge_index)) + initial  
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.final_conv(x, edge_index)

        x = global_mean_pool(x, data.batch)  
        return F.log_softmax(x, dim=1)
