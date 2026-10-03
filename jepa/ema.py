"""
Exponential Moving Average (EMA) updates for the target encoder.

The target encoder is moved toward the context endoer at after each optimiser step via an ema.
"""

import torch
from torch import nn


def momentum_at(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Return the EMA momentum for a given training step.
    
    Args:
        step: current optimiser step
        total_steps: total number of optimiser steps in training
        start: momentum at first step
        end: momentum at the last step.

    Returns:
        the momentum at step a given step, to be passed to ema_update.
    """

    progress = min(step / max(total_steps, 1), 1.0) #begin at 0, finish at 1
    return start + progress * (end - start)

@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, momentum: float) -> None:
    """Move parameter of target a small increment toward the source.
    
    Args:
        target: the target encoder (ema copy of source). Parameters get changed in place. 
        source: the model trained by optimiser (context encoder)
        momentum: value between 0 and 1. 1 indicates a slower change.
    """

    for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.mul_(momentum).add_(source_param, alpha=1.0 - momentum)
