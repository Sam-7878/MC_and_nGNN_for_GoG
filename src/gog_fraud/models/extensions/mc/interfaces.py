from typing import Protocol, Any, Dict, Optional, Generic, TypeVar
from dataclasses import dataclass, field
import torch

T = TypeVar('T')

@dataclass
class MCOutput(Generic[T]):
    """Wrapper output containing the original level output along with MC statistics."""
    base_output: T
    mean_score: torch.Tensor
    uncertainty: torch.Tensor
    raw_scores: Optional[torch.Tensor] = None
    aux: Dict[str, Any] = field(default_factory=dict)

class UncertaintyEstimator(Protocol):
    def estimate(self, model: Any, batch: Any) -> MCOutput[Any]:
        ...
