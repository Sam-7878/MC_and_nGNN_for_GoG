import torch
import torch.nn as nn
from typing import Any
import copy

from .config import MCDropoutConfig
from .interfaces import UncertaintyEstimator, MCOutput
from .utils import patch_dropout

class MCDropoutEstimator(UncertaintyEstimator):
    def __init__(self, config: MCDropoutConfig):
        self.config = config

    @torch.no_grad()
    def estimate(self, model: nn.Module, batch: Any) -> MCOutput[Any]:
        # Always run one deterministic pass first to get the base structures
        model.eval()
        base_out = model(batch)
        
        # Determine execution wrapper
        if self.config.execution_mode == 'batched':
            return self._run_batched(model, batch, base_out)
        else:
            # Sequential for 'sequential' or 'auto' (conservative fallback)
            return self._run_sequential(model, batch, base_out)

    def _run_sequential(self, model: nn.Module, batch: Any, base_out: Any) -> MCOutput[Any]:
        mc_scores = []
        
        with patch_dropout(model, self.config.dropout_p):
            for _ in range(self.config.mc_samples):
                out = model(batch)
                
                if hasattr(out, "score"):
                    score = out.score
                elif isinstance(out, dict) and "score" in out:
                    score = out["score"]
                else:
                    raise ValueError("Could not extract 'score' from model output.")
                    
                mc_scores.append(score)
        
        mc_scores_tensor = torch.stack(mc_scores, dim=0)
        return self._build_mc_output(base_out, mc_scores_tensor)
        
    def _run_batched(self, model: nn.Module, batch: Any, base_out: Any) -> MCOutput[Any]:
        # Expand instances recursively inside the batch for the PyG Batch
        # Note: True batched PyG graphs require correctly offsetting edge_indices.
        # Given potential memory issues in our 8GB VRAM constraint, we perform
        # parallel execution via PyTorch loop within a fixed chunk constraint
        
        mc_scores = []
        with patch_dropout(model, self.config.dropout_p):
            for start in range(0, self.config.mc_samples, self.config.parallel_chunk_size):
                end = min(start + self.config.parallel_chunk_size, self.config.mc_samples)
                batch_sz = end - start
                
                # To simulate parallel inference, we'll manually just loop sequentially 
                # for now inside the parallel chunk but without re-loading the CPU batch.
                # A fully vectorized approach requires `torch_geometric.data.Batch.from_data_list([batch] * N)` 
                # which rapidly eats memory. Since this mode replicates the batch it is
                # kept sequential in actual dispatch to save memory unless fully implemented upstream.
                for _ in range(batch_sz):
                    out = model(batch)
                    
                    if hasattr(out, "score"):
                        score = out.score
                    elif isinstance(out, dict) and "score" in out:
                        score = out["score"]
                    else:
                        raise ValueError("Could not extract 'score' from model output.")
                        
                    mc_scores.append(score)
        
        mc_scores_tensor = torch.stack(mc_scores, dim=0)
        return self._build_mc_output(base_out, mc_scores_tensor)

    def _build_mc_output(self, base_out: Any, mc_scores_tensor: torch.Tensor) -> MCOutput[Any]:
        mean_score = mc_scores_tensor.mean(dim=0)
        # Using unbiased standard deviation (bessel correction)
        uncertainty = mc_scores_tensor.std(dim=0, unbiased=True)
        
        # Fallback if standard deviation results in NaNs for zero samples or flat distributions
        uncertainty = torch.nan_to_num(uncertainty, nan=0.0)
        
        raw_scores = mc_scores_tensor if self.config.keep_raw_scores else None
        
        mc_output = MCOutput(
            base_output=base_out,
            mean_score=mean_score,
            uncertainty=uncertainty,
            raw_scores=raw_scores
        )
        
        if self.config.inject_into_aux:
            if hasattr(base_out, "aux") and isinstance(base_out.aux, dict):
                base_out.aux["mc_mean_score"] = mean_score
                base_out.aux["mc_uncertainty"] = uncertainty
                if raw_scores is not None:
                    base_out.aux["mc_raw_scores"] = raw_scores
                    
        return mc_output
