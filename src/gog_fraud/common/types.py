from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class Level1Output:
    graph_id: torch.Tensor
    embedding: torch.Tensor
    logits: torch.Tensor
    score: torch.Tensor
    label: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Level1EmbeddingBundle:
    graph_id: torch.Tensor
    embedding: torch.Tensor
    logits: torch.Tensor
    score: torch.Tensor
    label: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
