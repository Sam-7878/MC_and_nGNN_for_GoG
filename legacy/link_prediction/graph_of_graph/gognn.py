import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class NetModular(torch.nn.Module):
    def __init__(self, args, num_features, num_labels):
        super(NetModular, self).__init__()
        self.args = args
        self.num_features = 1 #num_features
        self.num_edge_features = 1 #args.num_edge_features
        self.nhid = args.nhid
        self.ddi_nhid = args.ddi_nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.number_labels = num_labels

        self.conv1 = GCNConv(self.num_features, self.nhid).to(args.device)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv2 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv3 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)

        self.nn = torch.nn.Linear(self.num_edge_features, 2 * self.nhid * self.ddi_nhid)
        self.conv4 = NNConv(2 * self.nhid, self.ddi_nhid, self.nn).to(args.device)
        self.conv_noattn = GCNConv(2 * self.nhid, self.ddi_nhid).to(args.device)

        self.lin1 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin2 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin3 = torch.nn.Linear(self.num_edge_features, self.ddi_nhid)
        
        self.pred_layer = torch.nn.Linear(4 * self.nhid, 1).to(self.args.device)


    def forward(self, modular_data, ddi_edge_index):
        ddi_edge_attr = torch.tensor([1.0]*ddi_edge_index.shape[1], dtype=torch.float).to(self.args.device)
        ddi_edge_index = ddi_edge_index.to(self.args.device)
        modular_output = []
        modular_data = {i: graph for i, graph in enumerate(modular_data)}
        ids = list(modular_data.keys())
        for modular_id in ids:

            data = modular_data[modular_id]
            num_nodes = data['features'].shape[0]
            x = torch.ones((num_nodes, 1)) 
            edge_index = data['edges']
            num_edges = edge_index.shape[1]
            edge_weight = torch.tensor([1.0]*num_edges, dtype=torch.float) 
            batch_list = [0] * num_nodes
            batch = torch.IntTensor(batch_list)  # int tensor for batch

            x = x.to(self.args.device)
            edge_index = edge_index.to(self.args.device)
            edge_weight = edge_weight.to(self.args.device)
            batch = batch.to(self.args.device)

            x = F.relu(self.conv1(x, edge_index, edge_weight))
            batch = batch.long()
            x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            out_x = x1

            modular_output.append(out_x)

        modular_feature = torch.cat(tuple(modular_output))
        
        modular_feature = nn.Dropout(self.args.dropout_ratio)(modular_feature)
        modular_feature = modular_feature.to(self.args.device)

        src, dst = ddi_edge_index
        edge_features = torch.cat([modular_feature[src], modular_feature[dst]], dim=1)

        predictions = torch.sigmoid(self.pred_layer(edge_features)).squeeze()

        return predictions