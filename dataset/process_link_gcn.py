import pandas as pd
import torch
from torch_scatter import scatter_add
from torch_geometric.data import InMemoryDataset, Data
import os

class TransactionEdgeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, chain='polygon', use_train=True):
        self.chain = chain
        self.use_train = use_train  # Flag to toggle between train and test data
        super().__init__(root, transform, pre_transform)
        self.data = None  # Placeholder for loaded data
        self.load_data()

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    def process(self):
        train_path =  f'{self.root}/{self.chain}_train_edges.txt' # read in train data
        test_path = f'{self.root}/{self.chain}_test_edges.txt' # read in test data
    
        train_df = pd.read_csv(train_path, sep=' ', header=None, names=['node1', 'node2', 'label'])
        test_df = pd.read_csv(test_path, sep=' ', header=None, names=['node1', 'node2', 'label'])
    
        all_nodes = pd.concat([train_df[col] for col in ['node1', 'node2']] + [test_df[col] for col in ['node1', 'node2']]).unique()
        node_mapping = {node_id: idx for idx, node_id in enumerate(all_nodes)}
        
        train_df['node1'] = train_df['node1'].map(node_mapping)
        train_df['node2'] = train_df['node2'].map(node_mapping)
        test_df['node1'] = test_df['node1'].map(node_mapping)
        test_df['node2'] = test_df['node2'].map(node_mapping)

        node_features = self.prepare_node_features(pd.concat([train_df, test_df]))
        train_data = self.prepare_graph_data(train_df, node_features)
        test_data = self.prepare_graph_data(test_df, node_features)
    
        torch.save(train_data, self.processed_paths[0])
        torch.save(test_data, self.processed_paths[1])

    def load_data(self):
        data_path = self.processed_paths[0] if self.use_train else self.processed_paths[1]
        self.data = torch.load(data_path) if os.path.exists(data_path) else None

    def prepare_node_features(self, df):
        num_nodes = df[['node1', 'node2']].max().max() + 1
        ones = torch.ones(df.shape[0], dtype=torch.long)
        node2_indices = torch.tensor(df['node2'].values, dtype=torch.long)
        node1_indices = torch.tensor(df['node1'].values, dtype=torch.long)
    
        in_degree = torch.zeros(num_nodes, dtype=torch.long).scatter_add_(0, node2_indices, ones)
        out_degree = torch.zeros(num_nodes, dtype=torch.long).scatter_add_(0, node1_indices, ones)
        
        return torch.stack([in_degree, out_degree, in_degree + out_degree], dim=1).float()

    def prepare_graph_data(self, df, node_features):
        edge_index = torch.tensor([df['node1'].values, df['node2'].values], dtype=torch.long)
        # edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        labels = torch.tensor(df['label'].values, dtype=torch.float)
        # return Data(x=node_features, edge_index=edge_index.t().contiguous(), y=labels)
        return Data(x=node_features, edge_index=edge_index.contiguous(), y=labels)

    def __getitem__(self, idx):
        # Directly returns the data object, which should be a Data object
        return self.data

    def __len__(self):
        # This could count nodes or edges based on your specific implementation needs
        return 1  # Since only one Data object is stored per file


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    chain = 'polygon'
    
    root_path = f'../../_data/GoG/edges/{chain}'

    train_data = TransactionEdgeDataset(root=root_path, chain=chain, use_train=True)
    test_data = TransactionEdgeDataset(root=root_path, chain=chain, use_train=False)