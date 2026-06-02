"""Policy (SPEC section 15).

V1 is an epsilon-greedy policy with a decaying exploration rate. The greedy
branch relies on an external action scorer (the Planner, which uses the World
Model). Without a scorer it falls back to uniform random exploration. No
solution is hardcoded here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class EpsilonGreedyPolicy:
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.total_steps = 0
        self.rng = np.random.default_rng(seed)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def choose(
        self,
        latent_state: np.ndarray,
        goal: str,
        memories: Optional[List[Dict[str, Any]]],
        available_actions: List[str],
        action_scorer: Optional[Callable[[str], float]] = None,
    ) -> str:
        if not available_actions:
            raise ValueError("No available actions to choose from.")

        eps = self.epsilon
        self.total_steps += 1

        explore = action_scorer is None or self.rng.random() < eps
        if explore:
            return str(self.rng.choice(available_actions))

        scores = np.array([action_scorer(a) for a in available_actions], dtype=np.float64)
        best = np.flatnonzero(scores == scores.max())
        return available_actions[int(self.rng.choice(best))]
