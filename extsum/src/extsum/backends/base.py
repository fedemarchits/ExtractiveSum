"""Generation settings and seeding shared by every backend."""
from __future__ import annotations

import random
from dataclasses import dataclass


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local generation.

    This affects the local Hugging Face backend. vLLM and endpoint backends
    receive their seeds through their own generation APIs.
    """
    import numpy as np
    import torch

    normalized_seed = int(seed)

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(normalized_seed)
        torch.cuda.manual_seed_all(normalized_seed)

    # These settings improve repeatability for supported CUDA operations.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class GenConfig:
    """Shared generation settings."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42
