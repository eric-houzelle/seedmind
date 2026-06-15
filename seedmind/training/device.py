"""Device selection helpers for optional torch acceleration."""
from __future__ import annotations

import torch


def resolve_device(name: str = "cpu") -> torch.device:
    """Resolve a user device string without changing CPU-default behavior."""
    requested = (name or "cpu").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    return torch.device(requested)
