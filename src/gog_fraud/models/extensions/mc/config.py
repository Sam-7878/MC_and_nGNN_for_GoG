from dataclasses import dataclass
from typing import Literal

@dataclass
class MCDropoutConfig:
    mc_samples: int = 8
    dropout_p: float = 0.10
    execution_mode: Literal["sequential", "batched", "auto"] = "sequential"
    parallel_chunk_size: int = 2
    keep_raw_scores: bool = False
    inject_into_aux: bool = False
