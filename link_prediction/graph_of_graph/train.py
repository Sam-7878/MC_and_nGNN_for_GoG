import torch
import random
import glob
from tqdm import trange
from seal import SEAL
from gognn import NetModular
from dvgga import DVGGA
import pandas as pd
from utils import hierarchical_graph_reader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, accuracy_score

import json
from pathlib import Path
from torch_geometric.data import Data

# =========================================================================
# ✅ 메모리 최적화: GraphDatasetGenerator를 대체하는 초경량 지연 로딩 클래스
# - 모델이 특정 그래프를 요청할 때만 JSON을 열어 데이터를 읽고 반환 (OOM 원천 차단)
# =========================================================================
class LazyGraphDataset:
    def __init__(self, graphs_dir, device):
        self.graphs_dir = Path(graphs_dir)
        self.device = device
        
        # 디렉토리 내 JSON 파일 탐색
        self.json_files = sorted(list(self.graphs_dir.glob("*.json")), key=lambda x: int(x.stem))
        self.num_graphs = len(self.json_files)
        
        # 첫 번째 파일만 읽어서 Feature 차원 확인
        self.number_of_features = 0
        self.number_of_labels = 2  # 이진 분류 기본값
        
        if self.num_graphs > 0:
            with open(self.json_files[0], 'r') as f:
                data = json.load(f)
                features = data.get('features', [])
                if features and len(features) > 0:
                    self.number_of_features = len(features[0])
                else:
                    self.number_of_features = len(data.get('contract_feature', []))
                    
        # train.py 코드 구조 수정을 최소화하기 위해 self를 반환 (모델은 리스트처럼 이 객체에 접근)
        self.graphs = self
        
    def __len__(self):
        return self.num_graphs

    def _load_graph(self, idx):
        # 런타임에 JSON에서 데이터를 On-the-fly로 로딩
        with open(self.graphs_dir / f"{idx}.json", 'r') as f:
            data = json.load(f)
            
        contract_feature = data.get('contract_feature', [])
        label = data.get('label', 0)
        edges = data.get('edges', [])
        features = data.get('features', [])
        
        y = torch.tensor([label], dtype=torch.long).to(self.device)
        
        if features:
            x = torch.tensor(features, dtype=torch.float).to(self.device)
        else:
            x = torch.tensor([contract_feature], dtype=torch.float).to(self.device)
            
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            
        pyg_data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data.contract_feature = torch.tensor([contract_feature], dtype=torch.float).to(self.device)
        
        return pyg_data

    def __getitem__(self, idx):
        # 모델 내부에서 slicing, list, tensor 등 다양한 형태로 접근할 경우를 완벽 지원
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.num_graphs
            step = idx.step or 1
            return [self._load_graph(i) for i in range(start, stop, step)]
        elif torch.is_tensor(idx) and idx.dtype == torch.bool:
            indices = idx.nonzero(as_tuple=False).view(-1).tolist()
            return [self._load_graph(i) for i in indices]
        elif hasattr(idx, '__iter__') and not isinstance(idx, str):
            return [self._load_graph(int(i)) for i in idx]
        else:
            return self._load_graph(int(idx))
# =========================================================================

class Trainer(object):
    def __init__(self, args, seed):
        self.args = args
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        
        # ✅ 무거운 GraphDatasetGenerator 대신 LazyGraphDataset 적용
        self.dataset_generator = LazyGraphDataset(self.args.graphs, self.args.device)
        
        self.seed = seed
        self.chain = self.args.chain
        self._setup_macro_graph()
        self._load_macro_graph()

    def _load_macro_graph(self):
        train_path =  f'../../_data/GoG/edges/{self.args.chain}/{self.args.chain}_train_edges.txt' 
        test_path = f'../../_data/GoG/edges/{self.args.chain}/{self.args.chain}_test_edges.txt' 
        
        train_edges = pd.read_csv(train_path, sep=' ', header=None, names=['node1', 'node2', 'label'])
        test_edges = pd.read_csv(test_path, sep=' ', header=None, names=['node1', 'node2', 'label'])

        self.positive_edges = train_edges[train_edges['label'] == 1][['node1', 'node2']].values.tolist()
        self.negative_edges = train_edges[train_edges['label'] == 0][['node1', 'node2']].values.tolist()
        self.test_positive_edges = test_edges[test_edges['label'] == 1][['node1', 'node2']].values.tolist()
        self.test_negative_edges = test_edges[test_edges['label'] == 0][['node1', 'node2']].values.tolist()

        self.train_edges = torch.tensor(self.positive_edges + self.negative_edges, dtype=torch.long).t().to(self.args.device)
        self.test_edges = torch.tensor(self.test_positive_edges + self.test_negative_edges, dtype=torch.long).t().to(self.args.device)
        self.train_labels = torch.cat([torch.ones(len(self.positive_edges)), torch.zeros(len(self.negative_edges))]).to(self.args.device)
        self.test_labels = torch.cat([torch.ones(len(self.test_positive_edges)), torch.zeros(len(self.test_negative_edges))]).to(self.args.device)

        self.macro_graph_edges = torch.tensor(self.positive_edges + self.negative_edges, dtype=torch.long).t().to(self.args.device)
        self.all_labels = torch.cat([torch.ones(len(self.positive_edges)), torch.zeros(len(self.negative_edges))]).to(self.args.device)

    def _setup_macro_graph(self):
        self.positive_edges = [[edge[0], edge[1]] for edge in self.macro_graph.edges()]
        self.negative_edges = self._generate_negative_edges()
        self.macro_graph_edges = torch.tensor(self.positive_edges + self.negative_edges, dtype=torch.long).t()
        self.all_labels = torch.cat([torch.ones(len(self.positive_edges)), torch.zeros(len(self.negative_edges))])
        self.macro_graph_edges = self.macro_graph_edges.to(self.args.device)
        self.all_labels = self.all_labels.to(self.args.device)

    def _create_split(self):
        total_edges = len(self.positive_edges) + len(self.negative_edges)
        all_indices = torch.randperm(total_edges)
        split_index = int(total_edges * self.args.train_ratio)
        self.train_indices = all_indices[:split_index]
        self.test_indices = all_indices[split_index:]

    def _create_masks(self):
        self.train_mask = torch.zeros(len(self.macro_graph_edges[0]), dtype=torch.bool)
        self.test_mask = torch.zeros(len(self.macro_graph_edges[0]), dtype=torch.bool)
        self.train_mask[self.train_indices] = True
        self.test_mask[self.test_indices] = True

    def _generate_negative_edges(self):
        all_possible_edges = set((i, j) for i in range(len(self.macro_graph.nodes())) for j in range(i + 1, len(self.macro_graph.nodes())))
        existing_edges = set((i, j) for i, j in self.macro_graph.edges())
        non_edges = list(all_possible_edges - existing_edges)
        random.shuffle(non_edges)
        return non_edges[:len(existing_edges)]

    def _setup_model(self):
        if self.args.model == "SEAL":
            self.model = SEAL(self.args, self.dataset_generator.number_of_features, 
                             self.dataset_generator.number_of_labels).to(self.args.device)
        elif self.args.model == "GOGNN":
            self.model = NetModular(self.args, self.dataset_generator.number_of_features,
                                self.dataset_generator.number_of_labels).to(self.args.device)
        elif self.args.model == "DVGGA":
            self.model = DVGGA(self.args, self.dataset_generator.number_of_features, 
                                len(glob.glob(self.args.graphs + "*.json")),
                                self.dataset_generator.number_of_labels)

    def fit(self):
        self._setup_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        for epoch in range(self.args.epochs):
            self.model.train()
            optimizer.zero_grad()
            if self.args.model == 'SEAL':
                predictions, penalties = self.model(self.dataset_generator.graphs, self.train_edges)
                loss = torch.nn.functional.binary_cross_entropy(predictions, self.train_labels)
                loss = loss + self.args.gamma * penalties
            
            if self.args.model == 'GOGNN':
                predictions = self.model(self.dataset_generator.graphs, self.train_edges)
                loss = torch.nn.functional.binary_cross_entropy(predictions, self.train_labels)
                loss = torch.mean(loss)

            if self.args.model == 'DVGGA':           
                pre_loss, positive_penalty, pos_pred, neg_pred = self.model(self.dataset_generator.graphs, self.positive_edges, self.negative_edges)
                loss = pre_loss + self.args.beta2 * positive_penalty
            
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")

    def score(self):
        self.model.eval()
        with torch.no_grad():
            if self.args.model == 'SEAL':
                test_predictions, _ = self.model(self.dataset_generator.graphs, self.test_edges)
            if self.args.model == 'GOGNN':
                test_predictions = self.model(self.dataset_generator.graphs, self.test_edges)
            if self.args.model == 'DVGGA':
                _, _, pos_pred, neg_pred = self.model(self.dataset_generator.graphs, self.test_positive_edges, self.test_negative_edges)
                test_predictions = torch.cat([pos_pred, neg_pred], dim=0).squeeze()

            test_pred_labels = (test_predictions > 0.5).long()
            test_true_labels = self.test_labels.cpu().numpy()
            test_pred_labels = test_pred_labels.cpu().numpy()

            # Compute metrics safely
            metrics = {
                'test': (test_true_labels, test_pred_labels)
            }
            results = {}

            for set_name, (true_labels, pred_labels) in metrics.items():
                accuracy = accuracy_score(test_true_labels, test_pred_labels)
                precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

                if len(set(test_true_labels)) > 1:
                    auc = roc_auc_score(test_true_labels, test_predictions.cpu().numpy())
                    ap = average_precision_score(test_true_labels, test_predictions.cpu().numpy())
                else:
                    auc = float('nan')
                    ap = float('nan')

                results['test'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro,
                    'auc': auc,
                    'average_precision': ap
                }

                print(f"{set_name.capitalize()} Metrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Macro: {f1_macro:.4f}")
                print(f"F1-Micro: {f1_micro:.4f}")
                print(f"Average Precision: {ap:.4f}")

                if set_name == 'test':
                    print('Test report')
                    print(classification_report(test_true_labels, test_pred_labels, zero_division=0))

            return results