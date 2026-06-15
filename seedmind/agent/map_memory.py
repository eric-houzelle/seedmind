"""Egocentric map memory: what the agent has *seen*, and how long ago.

The capacity to remember is architectural (like a hippocampus); the content
comes exclusively from lived observations — no world truth ever leaks in.
Remembered information decays with age (resources regrow, the world moves on)
and the whole map dies with the individual (reset on death: new layout).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from seedmind.envs.micro_fouloide_world import AGENT, UNKNOWN

NEVER_SEEN = -1


class MapMemory:
    def __init__(self, size: int, horizon: int = 300) -> None:
        self.size = int(size)
        self.horizon = max(1, int(horizon))
        self.known = np.full((self.size, self.size), NEVER_SEEN, dtype=np.int16)
        self.age = np.zeros((self.size, self.size), dtype=np.int32)

    def reset(self) -> None:
        self.known.fill(NEVER_SEEN)
        self.age.fill(0)

    def observe(self, observation: Dict[str, Any]) -> None:
        """Integrate one observation: visible cells refresh, the rest ages."""
        grid = np.asarray(observation["grid"], dtype=np.int64)
        visible = grid != UNKNOWN
        self.age += 1
        self.known[visible] = grid[visible].astype(np.int16)
        self.age[visible] = 0
        # The AGENT overlay hides the ground truth under the agent's feet.
        r, c = (int(v) for v in observation.get("agent_pos", (-1, -1)))
        if 0 <= r < self.size and 0 <= c < self.size and grid[r, c] == AGENT:
            self.known[r, c] = np.int16(int(observation.get("standing_entity", 0)))

    def freshness(self) -> np.ndarray:
        """1.0 just seen → 0.0 forgotten (or never seen)."""
        fresh = np.clip(1.0 - self.age / float(self.horizon), 0.0, 1.0)
        fresh[self.known == NEVER_SEEN] = 0.0
        return fresh.astype(np.float32)

    def augment(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return the observation with memory keys attached."""
        observation = dict(observation)
        observation["memory_grid"] = self.known.copy()
        observation["memory_fresh"] = self.freshness()
        return observation
