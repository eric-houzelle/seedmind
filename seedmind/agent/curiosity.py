"""Curiosity Module (SPEC section 13).

Gives an intrinsic reward when the agent meets something it predicts poorly.
A cap prevents the agent from chasing pure chaos.
"""
from __future__ import annotations

import numpy as np


def compute_prediction_error(predicted_next: np.ndarray, actual_next: np.ndarray) -> float:
    """Mean-squared error between predicted and actual next latent states."""
    pred = np.asarray(predicted_next, dtype=np.float32)
    actual = np.asarray(actual_next, dtype=np.float32)
    return float(np.mean((pred - actual) ** 2))


class CuriosityModule:
    def __init__(self, weight: float = 0.1, max_reward: float = 1.0, enabled: bool = True) -> None:
        self.weight = weight
        self.max_reward = max_reward
        self.enabled = enabled

    def compute(self, prediction_error: float) -> float:
        """intrinsic_reward = weight * min(prediction_error, max_reward)."""
        if not self.enabled:
            return 0.0
        capped = min(float(prediction_error), self.max_reward)
        return self.weight * capped

    def compute_array(self, prediction_error: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`compute`."""
        if not self.enabled:
            return np.zeros_like(np.asarray(prediction_error, dtype=np.float32))
        capped = np.minimum(np.asarray(prediction_error, dtype=np.float32), self.max_reward)
        return self.weight * capped
