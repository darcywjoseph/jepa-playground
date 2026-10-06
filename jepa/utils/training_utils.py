"""Utility functions related to training."""


import math
from collections.abc import Sequence
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def pick_device() -> torch.device:
    """Returns the fastest available device."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")

def load_config(path: str) -> dict[str, Any]:
    """Reads YAML config to nested dictionaries.

    Args:
        path: Path to a .yaml file.

    Returns:
        The config as dictionaries, e.g. config["optimization"]["lr"].
    """

    with open(path) as file:
        return yaml.safe_load(file)

def load_cifar(data_dir: str, train: bool, augmentations: Sequence[nn.Module] = (), fake: bool = False) -> Dataset:
    """Load CIFAR-10 dataset.

    Args:
        data_dir: path to dataset
        train: Load 50k training images if True; Load 10k test images if False.
        augmentations: Random transforms applied to each image before it is normalised.
        fake: Use 512 random images instead. useful for test

    Returns:
        A dataset of (image [3, 32, 32], label) pairs.
    """

    transform = transforms.Compose([*augmentations, transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    
    if fake:
        return datasets.FakeData(512, (3, 32, 32), 10, transform=transform)
    
    return datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)

def sort_parameter_groups(*modules: nn.Module) -> list[dict]:
    """Split parameters into two groups; 1. with weight decay 2. without weight decay.

    exlcude biases and LayerNorm params from weight decay.

    Args:
        *modules: The modules to train.

    Returns:
        Two parameter groups for the optimizer. first has "decay" set to True.
    """

    with_decay: list[nn.Parameter] = []
    without_decay: list[nn.Parameter] = []

    for module in modules:
        for param in module.parameters():
            (without_decay if param.ndim == 1 else with_decay).append(param)
    
    return [
        {"params": with_decay, "decay": True},
        {"params": without_decay, "decay": False, "weight_decay": 0.0},
    ]

def warmup_cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start: float,
    peak: float,
    end: float,
) -> float:
    """Returns cosine schedule value.

    Args:
        step: Current optimizer step. begins at 0.
        total_steps: Total optimizer steps in training.
        warmup_steps: Steps spent rising on warmup.
        start: Value at step 0.
        peak: Value at the end of warmup.
        end: Value at the last step.

    Returns:
        The value for this step.
    """

    if step < warmup_steps:
        return start + step / warmup_steps * (peak - start)
    
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)  # 0 -> 1 after warmup
    
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Returns linear schedule.

    Args:
        step: Current optimizer step. begins at 0.
        total_steps: Total optimizer steps in training.
        start: Value at step 0.
        end: Value at the last step.
    
    Returns:
        The value for this step.
    """
    
    progress = min(step / max(total_steps, 1), 1.0)

    return start + progress * (end - start)