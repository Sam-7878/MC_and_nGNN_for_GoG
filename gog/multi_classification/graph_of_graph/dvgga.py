import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv
from torch_geometric.utils import (
    to_dense_adj,
    get_laplacian,
    contains_isolated_nodes,
    remove_isolated_nodes,
    dense_to_sparse,
)
from torch_geometric.utils import negative_sampling

class SAGE(torch.nn.Module):
    def __init__(self, args, num_features):
        super(SAGE, self).__init__()
        self.args = args
        
        self.num_features = num_features 
        self._setup()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()

    def _setup(self):
        self.graph_convolution_1 = GCNConv(self.num_features, self.args.first_gcn_dimensions).to(self.args.device)

        self.fully_connected_1 = torch.nn.Linear(self.args.first_gcn_dimensions, self.args.first_dense_neurons)
        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons, self.args.second_dense_neurons)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data['features'], data['edges'], torch.tensor([1.0]*data['edges'].shape[1], dtype=torch.float) 

        edges = edge_index.to(self.args.device)
        features = x.to(self.args.device)
        edge_weight = edge_weight.to(self.args.device)
        
        max_nodes = features.shape[0]
        is_single = False
        if contains_isolated_nodes(edge_index=edges, num_nodes=max_nodes):
            edges, edge_weight, mask = remove_isolated_nodes(edge_index=edges, 
                                                             edge_attr=edge_weight,
                                                             num_nodes=max_nodes)
            features = features[mask]
            if features.nelement() == 0:
                is_single = True
                features = x

        node_features_2 = self.graph_convolution_1(features.to(self.args.device), 
                                                    edges.to(self.args.device), 
                                                    edge_weight.to(self.args.device))
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        assignment = torch.nn.functional.softmax(
            self.fully_connected_2(abstract_features_1), dim=1
        ).to(self.args.device)

        node_features_2 = node_features_2.to(self.args.device)
        graph_embedding = torch.mm(torch.t(assignment), node_features_2)
        graph_embedding = torch.mean(graph_embedding,dim = 0,keepdim= True)

        return graph_embedding


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(p=0.3)
        self.emb = torch.nn.Embedding(num_nodes, in_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class DVGGA(torch.nn.Module):
    def __init__(self, args, num_features, num_nodes, num_labels):
        super(DVGGA, self).__init__()
        # self.pooling_ratio = args.pooling_ratio
        self.dropout = args.dropout
        self.second_gcn_dimensions = args.second_gcn_dimensions
        self.vgae_hidden_dimensions = args.vgae_hidden_dimensions
        self.args = args
        self.num_nodes = num_nodes
        self.number_features = num_features
        self.graph_level_model = SAGE(self.args, self.number_features).to(args.device)
        self.vgae = VGAE(
            VariationalLinearEncoder(
                self.second_gcn_dimensions, self.second_gcn_dimensions
            )
        )
        self.vgae_nl = VGAE(
            VariationalGCNEncoder(
                args.first_dense_neurons, self.second_gcn_dimensions, self.num_nodes
            )
        )
        self.vgae_nl = self.vgae_nl.to(self.args.device)

        self.lin1 = torch.nn.Linear(
            2 * self.second_gcn_dimensions, 2 * self.second_gcn_dimensions
        )
        self.lin2 = torch.nn.Linear(
            2 * self.second_gcn_dimensions, 2 * self.second_gcn_dimensions
        )
        self.lin3 = torch.nn.Linear(
            self.vgae_hidden_dimensions, self.vgae_hidden_dimensions
        )

        self.emb = torch.nn.Embedding(self.num_nodes, self.second_gcn_dimensions).to(
            self.args.device
        )
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.node_clf = nn.Linear(self.second_gcn_dimensions, num_labels).to(
            self.args.device
        )

    def forward(self, data, pos_edges):
        graphs = data
        embeddings = []
        for id in range(len(graphs)):
            graph = graphs[id]
            embedding = self.graph_level_model(graph) 
            embeddings.append(embedding)

        embeddings = torch.cat(tuple(embeddings), dim=0)
        embeddings = nn.Dropout(self.args.dropout)(embeddings)
        
        pos_edges = pos_edges.to(self.args.device)
        embeddings = self.vgae_nl.encode(embeddings, pos_edges)
        
        predictions = torch.nn.functional.log_softmax(self.node_clf(embeddings), dim=1)
        
        return predictions, None
    
    def unsupervise_predict_loss(
        self, embeddings, ddi_edge_index, neg_edge_index, pos_attr=None, neg_attr=None
    ):
        pos_source, pos_target, neg_source, neg_target = self.feature_split(
            embeddings, ddi_edge_index, neg_edge_index
        )
        pos_feat_x = self.lin1(pos_source)
        pos_feat_y = self.lin2(pos_target)
        neg_feat_x = self.lin1(neg_source)
        neg_feat_y = self.lin2(neg_target)
        EPS = 1e-15

        norm_pos = torch.sum(torch.mul(pos_feat_x, pos_feat_y), 1)
        norm_neg = torch.sum(torch.mul(neg_feat_x, neg_feat_y), 1)
        norm_pos = torch.sigmoid(norm_pos)
        norm_neg = torch.sigmoid(norm_neg)

        rec_loss = (
            -torch.log(norm_pos + EPS).mean() - torch.log(1 - norm_neg + EPS).mean()
        )
        return rec_loss, norm_pos, norm_neg

    def feature_split(self, features, edge_index, neg_index):
        source, target = edge_index
        pos_source = features[source]
        pos_target = features[target]
        source, target = neg_index
        neg_source = features[source]
        neg_target = features[target]

        return pos_source, pos_target, neg_source, neg_target


