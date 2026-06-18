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
    event: Optional[str] = None,
    event_amount: int = 0,
    event_index: Optional[int] = None,
    causal_features: Optional[np.ndarray] = None,
    next_causal_features: Optional[np.ndarray] = None,
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
        # --- causal-event extras for sparse composed behaviors ---
        "event": event,
        "event_amount": int(event_amount),
        "event_index": event_index,
        "causal_features": causal_features,
        "next_causal_features": next_causal_features,
    }


class ExperienceBuffer:
    """A capped ring buffer of experiences with several sampling strategies."""

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._data: List[Dict[str, Any]] = []
        self._cursor = 0
        self._episode_index: Dict[str, Dict[int, int]] = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, experience: Dict[str, Any]) -> None:
        if len(self._data) < self.capacity:
            idx = len(self._data)
            self._data.append(experience)
        else:
            idx = self._cursor
            self._remove_index(self._data[idx], idx)
            self._data[self._cursor] = experience
        self._add_index(experience, idx)
        self._cursor = (self._cursor + 1) % self.capacity

    def _add_index(self, experience: Dict[str, Any], idx: int) -> None:
        episode_id = experience.get("episode_id")
        step = experience.get("step")
        if episode_id is None or step is None:
            return
        self._episode_index.setdefault(str(episode_id), {})[int(step)] = idx

    def _remove_index(self, experience: Dict[str, Any], idx: int) -> None:
        episode_id = experience.get("episode_id")
        step = experience.get("step")
        if episode_id is None or step is None:
            return
        steps = self._episode_index.get(str(episode_id))
        if not steps:
            return
        if steps.get(int(step)) == idx:
            del steps[int(step)]
        if not steps:
            del self._episode_index[str(episode_id)]

    def _rebuild_index(self) -> None:
        self._episode_index = {}
        for idx, experience in enumerate(self._data):
            self._add_index(experience, idx)

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

    def sample_causal(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample informative causal transitions without world-specific rules.

        Transitions with explicit event ids/labels are weighted by inverse event
        frequency so rare consequences are replayed more often. This keeps the
        mechanism generic: the buffer does not know what an event means.
        """
        indices = []
        keys = []
        for i, e in enumerate(self._data):
            key = e.get("event_index")
            if key is None:
                key = e.get("event")
            if key is None:
                continue
            indices.append(i)
            keys.append(str(key))
        if not indices:
            return []
        counts: Dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = np.asarray([
            1.0 / np.sqrt(float(counts[key]))
            for key in keys
        ], dtype=np.float64)
        weights = weights / weights.sum()
        take = min(batch_size, len(indices))
        chosen = self.rng.choice(np.asarray(indices), size=take, replace=False, p=weights)
        return [self._data[i] for i in chosen]

    def n_step_sequence(self, experience: Dict[str, Any], n_step: int) -> List[Dict[str, Any]]:
        """Return up to n contiguous experiences from the same episode."""
        if n_step <= 1:
            return [experience]
        episode_id = str(experience.get("episode_id"))
        start_step = int(experience.get("step", 0))
        steps = self._episode_index.get(episode_id, {})
        sequence = []
        for offset in range(n_step):
            if offset == 0:
                current = experience
            else:
                idx = steps.get(start_step + offset)
                if idx is None:
                    break
                current = self._data[idx]
            sequence.append(current)
            if current.get("done", False):
                break
        return sequence

    def n_step_target(self, experience: Dict[str, Any], n_step: int, gamma: float) -> Dict[str, Any]:
        """Return n-step reward/bootstrap info for an experience.

        Falls back gracefully when the episode segment is missing or terminal.
        """
        if n_step <= 1:
            return {
                "reward": float(experience.get("reward_external", 0.0)),
                "next_obs_state": experience.get("next_obs_state"),
                "done": bool(experience.get("done", False)),
                "steps": 1,
            }

        reward = 0.0
        discount = 1.0
        last = experience
        sequence = self.n_step_sequence(experience, n_step)
        for current in sequence:
            reward += discount * float(current.get("reward_external", 0.0))
            discount *= gamma
            last = current
        return {
            "reward": reward,
            "next_obs_state": last.get("next_obs_state"),
            "done": bool(last.get("done", False)),
            "steps": len(sequence),
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> List[List[Dict[str, Any]]]:
        """Sample contiguous transition sequences for recurrent (BPTT) training.

        Each returned sequence is ``seq_len`` consecutive transitions from a
        single life, in step order, never crossing an episode boundary nor a
        ``done`` (a sequence may *end* on a terminal transition). Sequences
        shorter than ``seq_len`` (cut by death / missing steps) are rejected.
        Uses rejection sampling over random start transitions; may return fewer
        than ``batch_size`` sequences when few full runs exist.
        """
        if not self._data or seq_len < 1:
            return []
        if seq_len == 1:
            return [[e] for e in self.sample(batch_size)]
        sequences: List[List[Dict[str, Any]]] = []
        max_attempts = max(batch_size * 8, 32)
        for _ in range(max_attempts):
            if len(sequences) >= batch_size:
                break
            start = self._data[int(self.rng.integers(0, len(self._data)))]
            seq = self.n_step_sequence(start, seq_len)
            if len(seq) == seq_len:
                sequences.append(seq)
        return sequences

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
        self._rebuild_index()
