import torch.nn as nn
from contextlib import contextmanager
import copy

@contextmanager
def patch_dropout(model: nn.Module, target_p: float):
    """
    Context manager that temporarily patches all nn.Dropout layers in the model to
    have probability `target_p` and sets them to training mode (so they drop out in eval).
    Non-dropout layers remain untouched (e.g. BatchNorm remains in eval mode).
    """
    original_states = {}
    
    # Save states and patch
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_states[name] = {
                'p': module.p,
                'training': module.training
            }
            if module.p == 0.0:
                module.p = target_p
            module.train()

    try:
        yield
    finally:
        # Restore states
        for name, module in model.named_modules():
            if name in original_states and isinstance(module, nn.Dropout):
                module.p = original_states[name]['p']
                module.training = original_states[name]['training']
