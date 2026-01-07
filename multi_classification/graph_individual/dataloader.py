import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import json

class TransactionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TransactionDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def get_label(self, idx):
        return self.y[idx].item()

    def process(self):
        data_list = []
        for df, label in zip(self.transaction_dfs, self.labels):
            graph = self.create_graph(df)
            data = self.graph_to_data_object(graph, label)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_graph(self, transaction_df):
        graph = nx.DiGraph()
        address_to_node = {address: node_id for node_id, address in enumerate(set(transaction_df['from']) | set(transaction_df['to']))}
        for _, row in transaction_df.iterrows():
            value = float(row['value'].replace(',', '')) if isinstance(row['value'], str) else float(row['value'])
            from_node, to_node = address_to_node[row['from']], address_to_node[row['to']]
            graph.add_edge(from_node, to_node, weight=value)
        return graph

    def graph_to_data_object(self, graph, label, contract_address):
            
        adj = nx.to_scipy_sparse_array(graph, nodelist=sorted(graph.nodes()), format='coo')
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_attr = torch.tensor([[graph.edges[u, v]['weight'], graph.edges[u, v]['timestamp']] 
                                  for u, v in zip(adj.row, adj.col)], dtype=torch.float)

        num_nodes = graph.number_of_nodes()
        total_degree = [graph.degree(i) for i in range(num_nodes)]
        in_degree = [graph.in_degree(i) for i in range(num_nodes)]
        out_degree = [graph.out_degree(i) for i in range(num_nodes)]

        in_value = [sum(data['weight'] for _, _, data in graph.in_edges(i, data=True)) for i in range(num_nodes)]
        out_value = [sum(data['weight'] for _, _, data in graph.out_edges(i, data=True)) for i in range(num_nodes)]

        x = torch.tensor([[td, ind, outd, inv, outv] for td, ind, outd, inv, outv in zip(total_degree, in_degree, out_degree, in_value, out_value)], dtype=torch.float)

        timestamps = [data['timestamp'] for _, _, data in graph.edges(data=True)]
        min_timestamp, max_timestamp = min(timestamps), max(timestamps)
        average_timestamp = sum(timestamps) / len(timestamps)

        chain_index = chain_indexes.get(chain, None)
        contract_index = all_address_index.get(contract_address, None)

        graph_attr = torch.tensor([min_timestamp, max_timestamp, average_timestamp, 
                                   chain_index, contract_index], dtype=torch.float)

        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    num_nodes=num_nodes, graph_attr=graph_attr)
        return data
