"""Experience buffer and the universal experience format (SPEC sections 8-9).

The buffer stores every transition the agent collects. It feeds the World
Model, the Policy and the curiosity module. Experiences follow a common,
JSON-compatible schema so they remain reusable across future worlds.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def make_experience(
    *,
    episode_id: str,
    world_id: str,
    step: int,
    observation: Any,
    action: str,
    next_observation: Any,
    reward_external: float,
    reward_intrinsic: float,
    goal: str,
    prediction_error: float,
    done: bool,
    memory_used: Optional[Sequence[str]] = None,
    latent_state: Optional[np.ndarray] = None,
    next_latent_state: Optional[np.ndarray] = None,
    action_index: Optional[int] = None,
    obs_state: Optional[Dict[str, Any]] = None,
    next_obs_state: Optional[Dict[str, Any]] = None,
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a transition dict in the universal experience format.

    Extra training-oriented fields (``latent_state``, ``next_latent_state``,
    ``action_index``) are stored alongside the canonical SPEC section 8 schema.
    ``obs_state``/``next_obs_state`` hold the compact observation (grid +
    inventory) the learned Q-network needs (V2).
    """
    return {
        "episode_id": episode_id,
        "world_id": world_id,
        "step": step,
        "observation": observation,
        "action": action,
        "next_observation": next_observation,
        "reward_external": float(reward_external),
        "reward_intrinsic": float(reward_intrinsic),
        "goal": goal,
        "prediction_error": float(prediction_error),
        "memory_used": list(memory_used) if memory_used else [],
        "done": bool(done),
        "timestamp": int(timestamp if timestamp is not None else time.time()),
        # --- training extras ---
        "latent_state": latent_state,
        "next_latent_state": next_latent_state,
        "action_index": action_index,
        # --- V2 policy-learning extras ---
        "obs_state": obs_state,
        "next_obs_state": next_obs_state,
    }


class ExperienceBuffer:
    """A capped ring buffer of experiences with several sampling strategies."""

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._data: List[Dict[str, Any]] = []
        self._cursor = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, experience: Dict[str, Any]) -> None:
        if len(self._data) < self.capacity:
            self._data.append(experience)
        else:
            self._data[self._cursor] = experience
            self._cursor = (self._cursor + 1) % self.capacity

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_indices(self, indices: Sequence[int], batch_size: int) -> List[Dict[str, Any]]:
        if not indices:
            return []
        idx = np.asarray(indices)
        take = min(batch_size, len(idx))
        chosen = self.rng.choice(idx, size=take, replace=False)
        return [self._data[i] for i in chosen]

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return self._sample_indices(range(len(self._data)), batch_size)

    def sample_recent(self, batch_size: int) -> List[Dict[str, Any]]:
        return self._data[-batch_size:] if self._data else []

    def sample_high_error(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self._data:
            return []
        order = sorted(
            range(len(self._data)),
            key=lambda i: self._data[i].get("prediction_error", 0.0),
            reverse=True,
        )
        return [self._data[i] for i in order[:batch_size]]

    def sample_high_reward(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self._data:
            return []

        def total_reward(i: int) -> float:
            e = self._data[i]
            return e.get("reward_external", 0.0) + e.get("reward_intrinsic", 0.0)

        order = sorted(range(len(self._data)), key=total_reward, reverse=True)
        return [self._data[i] for i in order[:batch_size]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {"capacity": self.capacity, "data": self._data, "cursor": self._cursor},
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.capacity = state["capacity"]
        self._data = state["data"]
        self._cursor = state.get("cursor", 0)
