import torch
import random
from tqdm import trange
from seal import SEAL
from dvgga import DVGGA
from gognn import NetModular
from utils import hierarchical_graph_reader, GraphDatasetGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
import random
import numpy as np


class SEALCITrainer(object):
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        self.dataset_generator = GraphDatasetGenerator(self.args.graphs, self.args.device)
        self._setup_macro_graph()
        if self.args.split_type == 'random':
            self._create_split()  
        elif self.args.split_type == 'temporal':
            self._load_split()
        self._create_masks()  

    def _setup_model(self):
        """
        Conditionally creating a model based on the specified model type.
        """
        if self.args.model == "SEAL":
            self.model = SEAL(self.args, self.dataset_generator.number_of_features,
                            self.dataset_generator.number_of_labels).to(self.args.device)
        elif self.args.model == "GOGNN":
            self.model = NetModular(self.args, self.dataset_generator.number_of_features,
                                    self.dataset_generator.number_of_labels).to(self.args.device)
        elif self.args.model == "DVGGA":
            self.model = DVGGA(self.args, self.dataset_generator.number_of_features, 
                            self.macro_graph.number_of_nodes(),
                            self.dataset_generator.number_of_labels).to(self.args.device)
        else:
            raise ValueError("Unsupported model type provided.")

    def read_in_file(self, path): 
        with open(path, 'r') as file:
            lines = file.readlines()
        return [int(line.strip()) for line in lines]


    def _setup_macro_graph(self):
        """
        Creating an edge list for the hierarchical graph.
        """
        self.macro_graph_edges = [[edge[0], edge[1]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = self.macro_graph_edges + [[edge[1], edge[0]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = torch.t(torch.LongTensor(self.macro_graph_edges))
        self.macro_graph_edges = self.macro_graph_edges.to(self.args.device)

    def _load_split(self):
        self.train_indices = self.read_in_file(f'../../_data/GoG/node/{self.args.chain}_train_index_{self.args.num_classes}.txt')
        self.test_indices = self.read_in_file(f'../../_data/GoG/node/{self.args.chain}_test_index_{self.args.num_classes}.txt')

        # Calculate and print the number of labels per class in each dataset
        train_labels = [self.dataset_generator.target[idx].item() for idx in self.train_indices]
        test_labels = [self.dataset_generator.target[idx].item() for idx in self.test_indices]

    def _create_split(self):
        """
        Creating a train-test split with stratification.
        """
        graph_indices = list(range(len(self.dataset_generator.graphs)))
        labels = [self.dataset_generator.target[idx].item() for idx in graph_indices]

        train_indices, test_indices = train_test_split(
            graph_indices, 
            test_size=0.2, 
            random_state=self.seed, 
            stratify=labels 
        )

        self.train_indices = train_indices
        self.test_indices = test_indices

        train_labels = [self.dataset_generator.target[idx].item() for idx in self.train_indices]
        test_labels = [self.dataset_generator.target[idx].item() for idx in self.test_indices]
        
        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)
        print("Training set class distribution:", dict(train_counts))


    def calculate_average_graph_size(self, indices):
        total_nodes = sum(len(self.dataset_generator.graphs[index]["features"]) for index in indices)
        return total_nodes / len(indices)

    def print_average_sizes(self, train_indices, test_indices):
        avg_train_size = self.calculate_average_graph_size(train_indices)
        avg_test_size = self.calculate_average_graph_size(test_indices)
        print(f"Average size of training graphs: {avg_train_size:.2f} nodes")
        print(f"Average size of testing graphs: {avg_test_size:.2f} nodes")


    def _create_labeled_target(self):
        """
        Creating a mask for labeled instances and a target for them.
        """
        self.labeled_mask = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        self.labeled_target = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        indices = torch.LongTensor(self.labeled_indices)
        self.labeled_mask[indices] = 1
        self.labeled_target[indices] = self.dataset_generator.target[indices]


    def _create_masks(self):
        """
        Creating masks for training, validation, and testing.
        """
        self.train_mask = torch.zeros(len(self.dataset_generator.graphs), dtype=torch.bool)
        self.test_mask = torch.zeros(len(self.dataset_generator.graphs), dtype=torch.bool)

        self.train_mask[self.train_indices] = True
        self.test_mask[self.test_indices] = True

     
    def fit(self):
        """
        Training the model on the training set.
        """
        self._setup_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        for epoch in range(self.args.epochs):
            optimizer.zero_grad()
            predictions, penalty = self.model(self.dataset_generator.graphs, self.macro_graph_edges)

            loss = torch.nn.functional.nll_loss(predictions[self.train_mask], 
                                    self.dataset_generator.target[self.train_mask])
            if penalty:
                loss = loss + self.args.gamma * penalty
            
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")

    def score(self):
        """
        Scoring the model on the training, validation, and test sets with additional metrics, including F1-macro and F1-micro.
        """
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(self.dataset_generator.graphs, self.macro_graph_edges)
            probs = predictions.softmax(dim=1)  # Get probabilities for each class

            train_pred_labels = probs[self.train_mask].max(dim=1)[1]
            test_pred_labels = probs[self.test_mask].max(dim=1)[1]

            # Move tensors to CPU for scikit-learn compatibility
            train_true_labels = self.dataset_generator.target[self.train_mask].cpu().numpy()
            test_true_labels = self.dataset_generator.target[self.test_mask].cpu().numpy()
            train_pred_labels = train_pred_labels.cpu().numpy()
            test_pred_labels = test_pred_labels.cpu().numpy()

            # Compute metrics safely
            metrics = {
                'train': (train_true_labels, train_pred_labels),
                'test': (test_true_labels, test_pred_labels)
            }
            results = {}

            for set_name, (true_labels, pred_labels) in metrics.items():
                precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

                results[set_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro
                }

                print(f"{set_name.capitalize()} Metrics:")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Macro: {f1_macro:.4f}")
                print(f"F1-Micro: {f1_micro:.4f}")

                if set_name == 'test':
                    print('Test report')
                    print(classification_report(test_true_labels, test_pred_labels, zero_division=0))
                elif set_name == 'train':
                    print('Train report')
                    print(classification_report(train_true_labels, train_pred_labels, zero_division=0))

            return results