"""Helpers for moving latents between torch tensors and numpy storage."""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

LatentState = Union[np.ndarray, torch.Tensor]


def latent_to_numpy(latent: LatentState) -> np.ndarray:
    """Copy a latent to a float32 numpy vector (for replay buffer / memory)."""
    if isinstance(latent, torch.Tensor):
        return latent.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(latent, dtype=np.float32)
