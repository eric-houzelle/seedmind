"""Curiosity Module (SPEC section 13).

Gives an intrinsic reward when the agent meets something it predicts poorly.
A cap prevents the agent from chasing pure chaos.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

LatentLike = Union[np.ndarray, torch.Tensor]


def compute_prediction_error_tensor(
    predicted_next: torch.Tensor, actual_next: torch.Tensor
) -> torch.Tensor:
    """Mean-squared error on device (scalar tensor)."""
    return torch.mean((predicted_next - actual_next) ** 2)


def compute_prediction_error(predicted_next: LatentLike, actual_next: LatentLike) -> float:
    """Mean-squared error between predicted and actual next latent states."""
    if isinstance(predicted_next, torch.Tensor) and isinstance(actual_next, torch.Tensor):
        return float(compute_prediction_error_tensor(predicted_next, actual_next).item())
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


class CausalCuriosityModule:
    """Intrinsic reward for discovering action consequences.

    This is not task reward. It rewards rare/novel causal events so the agent
    investigates state-changing actions long enough to learn their downstream
    utility.
    """

    def __init__(
        self,
        weight: float = 0.0,
        max_reward: float = 0.2,
        novelty_bonus: float = 1.0,
        repeat_bonus: float = 0.2,
        enabled: bool = False,
    ) -> None:
        self.weight = float(weight)
        self.max_reward = float(max_reward)
        self.novelty_bonus = float(novelty_bonus)
        self.repeat_bonus = float(repeat_bonus)
        self.enabled = bool(enabled)
        self._seen: set[str] = set()

    def compute(self, event: str | None, event_amount: int = 0) -> float:
        if not self.enabled or self.weight <= 0.0 or not event:
            return 0.0
        if event.endswith("_noop"):
            return 0.0

        event = str(event)
        meaningful = {
            "harvest_food",
            "harvest_food_tool",
            "harvest_wood",
            "harvest_stone",
            "craft_tool",
            "eat_ok",
        }
        if event not in meaningful:
            return 0.0
        base = self.novelty_bonus if event not in self._seen else self.repeat_bonus
        amount_bonus = min(max(int(event_amount), 0), 4) * 0.1
        self._seen.add(event)
        return self.weight * min(base + amount_bonus, self.max_reward)

    def state_dict(self) -> dict:
        return {"seen": sorted(self._seen)}

    def load_state_dict(self, state: dict) -> None:
        self._seen = set(state.get("seen", []))
