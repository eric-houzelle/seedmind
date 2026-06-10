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
from seedmind.agent.value_model import ValueModel
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
        causal_feature_targets: Optional[np.ndarray] = None,
        value_model: Optional[ValueModel] = None,
        terminal_value_weight: float = 0.0,
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
        self.causal_feature_targets = (
            np.asarray(causal_feature_targets, dtype=np.float32)
            if causal_feature_targets is not None else None
        )
        self.value_model = value_model
        self.terminal_value_weight = float(terminal_value_weight)
        self.rng = np.random.default_rng(seed)

    def _causal_value(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        current_features: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.causal_feature_weights is None or self.causal_feature_weights.size == 0:
            return np.zeros(len(actions), dtype=np.float32), current_features
        delta, _ = self.world_model.predict_causal_batch(latents, actions)
        if delta.shape[1] != self.causal_feature_weights.shape[0]:
            return np.zeros(len(actions), dtype=np.float32), current_features
        if (
            current_features is not None
            and self.causal_feature_targets is not None
            and self.causal_feature_targets.shape[0] == delta.shape[1]
        ):
            cur = np.asarray(current_features, dtype=np.float32)
            if cur.ndim == 1:
                cur = np.repeat(cur[None, :], len(actions), axis=0)
            nxt = np.clip(cur + delta, 0.0, 1.0)
            before = np.abs(self.causal_feature_targets[None, :] - cur)
            after = np.abs(self.causal_feature_targets[None, :] - nxt)
            value = ((before - after) * self.causal_feature_weights[None, :]).sum(axis=1)
            return value.astype(np.float32), nxt
        return (delta @ self.causal_feature_weights).astype(np.float32), current_features

    def action_values(
        self,
        latent_state: np.ndarray,
        available_actions: List[str],
        current_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        values, _ = self.action_values_with_stats(
            latent_state, available_actions, current_features=current_features,
        )
        return values

    def action_values_with_stats(
        self,
        latent_state: np.ndarray,
        available_actions: List[str],
        current_features: Optional[np.ndarray] = None,
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Vectorised random-shooting value estimate for each first action."""
        num_actions = self.world_model.num_actions
        first_idx = np.array([self.action_index[a] for a in available_actions])
        a = len(available_actions)
        n = self.num_samples

        latent = np.asarray(latent_state, dtype=np.float32)
        particles = np.repeat(latent[None, :], a * n, axis=0)
        first_actions = np.repeat(first_idx, n)
        feature_particles = None
        if current_features is not None:
            feature_particles = np.repeat(
                np.asarray(current_features, dtype=np.float32)[None, :],
                a * n,
                axis=0,
            )

        next_l, reward, uncertainty = self.world_model.predict_batch(particles, first_actions)
        uncertainty_sum = uncertainty.astype(np.float32).copy()
        uncertainty_count = np.ones_like(uncertainty_sum, dtype=np.float32)
        causal_value, feature_particles = self._causal_value(
            particles, first_actions, feature_particles,
        )
        value = (
            reward
            + self.curiosity.compute_array(uncertainty)
            + causal_value
        )

        cur = next_l
        disc = self.gamma
        for _ in range(self.horizon - 1):
            rand_actions = self.rng.integers(0, num_actions, size=a * n)
            prev = cur
            cur, reward, uncertainty = self.world_model.predict_batch(prev, rand_actions)
            uncertainty_sum = uncertainty_sum + uncertainty.astype(np.float32)
            uncertainty_count = uncertainty_count + 1.0
            causal_value, feature_particles = self._causal_value(
                prev, rand_actions, feature_particles,
            )
            value = value + disc * (
                reward
                + self.curiosity.compute_array(uncertainty)
                + causal_value
            )
            disc *= self.gamma

        if self.value_model is not None and self.terminal_value_weight != 0.0:
            terminal = self.value_model.predict_batch(cur)
            value = value + disc * self.terminal_value_weight * terminal

        agg = value.reshape(a, n).mean(axis=1)
        unc = (uncertainty_sum / np.maximum(uncertainty_count, 1.0)).reshape(a, n).mean(axis=1)
        values = {action: float(agg[i]) for i, action in enumerate(available_actions)}
        stats = {
            action: {"uncertainty": float(unc[i])}
            for i, action in enumerate(available_actions)
        }
        return values, stats

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
