"""Planner (SPEC section 16).

Uses the World Model to imagine the futures of candidate actions and score them
with ``score = predicted_reward + intrinsic_reward + goal_progress``.

V1 implements a lightweight, vectorised **random-shooting** planner: for each
first action it rolls out ``num_samples`` random continuations of length
``horizon`` through the World Model and aggregates the discounted predicted
(reward + intrinsic) return. A short horizon already lets the agent steer
toward a reward that is a few steps away, while averaging over random
continuations avoids the degenerate looping of a 1-step greedy planner.
Setting ``horizon=1`` recovers the simplest possible planner.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from seedmind.agent.curiosity import CuriosityModule
from seedmind.agent.world_model import WorldModel


class Planner:
    def __init__(
        self,
        world_model: WorldModel,
        actions: List[str],
        curiosity: CuriosityModule,
        horizon: int = 4,
        num_samples: int = 16,
        gamma: float = 0.95,
        goal_progress_weight: float = 0.0,
        causal_feature_weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.world_model = world_model
        self.actions = actions
        self.action_index = {a: i for i, a in enumerate(actions)}
        self.curiosity = curiosity
        self.horizon = max(1, horizon)
        self.num_samples = max(1, num_samples)
        self.gamma = gamma
        self.goal_progress_weight = goal_progress_weight
        self.causal_feature_weights = (
            np.asarray(causal_feature_weights, dtype=np.float32)
            if causal_feature_weights is not None else None
        )
        self.rng = np.random.default_rng(seed)

    def _causal_value(self, latents: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.causal_feature_weights is None or self.causal_feature_weights.size == 0:
            return np.zeros(len(actions), dtype=np.float32)
        delta, _ = self.world_model.predict_causal_batch(latents, actions)
        if delta.shape[1] != self.causal_feature_weights.shape[0]:
            return np.zeros(len(actions), dtype=np.float32)
        return delta @ self.causal_feature_weights

    def action_values(
        self, latent_state: np.ndarray, available_actions: List[str]
    ) -> Dict[str, float]:
        """Vectorised random-shooting value estimate for each first action."""
        num_actions = self.world_model.num_actions
        first_idx = np.array([self.action_index[a] for a in available_actions])
        a = len(available_actions)
        n = self.num_samples

        latent = np.asarray(latent_state, dtype=np.float32)
        particles = np.repeat(latent[None, :], a * n, axis=0)
        first_actions = np.repeat(first_idx, n)

        next_l, reward, uncertainty = self.world_model.predict_batch(particles, first_actions)
        value = (
            reward
            + self.curiosity.compute_array(uncertainty)
            + self._causal_value(particles, first_actions)
        )

        cur = next_l
        disc = self.gamma
        for _ in range(self.horizon - 1):
            rand_actions = self.rng.integers(0, num_actions, size=a * n)
            prev = cur
            cur, reward, uncertainty = self.world_model.predict_batch(prev, rand_actions)
            value = value + disc * (
                reward
                + self.curiosity.compute_array(uncertainty)
                + self._causal_value(prev, rand_actions)
            )
            disc *= self.gamma

        agg = value.reshape(a, n).mean(axis=1)
        return {action: float(agg[i]) for i, action in enumerate(available_actions)}

    def score_action(self, latent_state: np.ndarray, action: str) -> float:
        """Single-action score (1-step), kept for convenience/tests."""
        idx = self.action_index[action]
        _, predicted_reward, uncertainty = self.world_model.predict(latent_state, idx)
        intrinsic = self.curiosity.compute(uncertainty)
        return predicted_reward + intrinsic

    def make_scorer(
        self, latent_state: np.ndarray, available_actions: Optional[List[str]] = None
    ) -> Callable[[str], float]:
        """Return a closure scoring an action from the current latent state.

        Values for all available actions are precomputed once (vectorised) so
        the policy can call the scorer per action cheaply.
        """
        actions = available_actions if available_actions is not None else self.actions
        values = self.action_values(latent_state, actions)
        return lambda action: values.get(action, float("-inf"))
