# src/gog_fraud/models/extensions/ngnn/interfaces.py

from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Batch

class RootedSubgraphExtractor(ABC):
    @abstractmethod
    def extract(self, data, root_index=None):
        pass

class NestedEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, nested_batch: Batch) -> torch.Tensor:
        """
        Takes a Batch of rooted subgraphs and returns an embedding 
        for each subgraph. 
        Output shape: [num_subgraphs, hidden_dim]
        """
        pass

class NestedReadout(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, subgraph_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Takes subgraph embeddings and a batch index mapping each subgraph
        to its parent Level 1 graph.
        Returns a single set of embeddings for each parent Level 1 graph.
        Output shape: [num_parent_graphs, hidden_dim]
        """
        pass
