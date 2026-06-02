"""Procedural GridWorld generator (SPEC section 6).

Each episode can produce a fresh map: new object positions and, optionally,
a new unlock rule linking an item to a target. The agent must therefore learn
to *discover the rules of a world* rather than memorise a single map.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.gridworld import (
    DANGER,
    DOOR_CLOSED,
    EMPTY,
    KEY,
    REWARD,
    WALL,
    GridWorld,
)


class ProceduralGridWorld(GridWorld):
    """GridWorld that regenerates a random solvable layout on every reset."""

    world_id = "procedural_gridworld_v1"

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 100,
        num_dangers: int = 2,
        num_walls: int = 0,
        regenerate_each_reset: bool = True,
        visibility_radius: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.num_dangers = num_dangers
        # Default to a fraction of the interior as random wall blocks.
        interior = max(1, (size - 2) * (size - 2))
        self.num_walls = num_walls if num_walls > 0 else max(0, interior // 8)
        self.regenerate_each_reset = regenerate_each_reset
        self._generated = False
        super().__init__(size=size, max_steps=max_steps, layout=None,
                         visibility_radius=visibility_radius, seed=seed)

    def reset(self) -> Dict[str, Any]:
        if self.regenerate_each_reset or not self._generated:
            self._initial_layout = self._generate_layout()
            self._generated = True
        return super().reset()

    def _generate_layout(self) -> np.ndarray:
        size = self.size
        grid = np.full((size, size), EMPTY, dtype=np.int64)
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        free = self._interior_cells(grid)
        self.rng.shuffle(free)

        def pop() -> Tuple[int, int]:
            return tuple(free.pop())  # type: ignore[return-value]

        # Place a random unlock rule: a key opens a door, reward behind nothing
        # in particular (fully observable, so the agent can reach it directly,
        # but a door optionally blocks part of the map).
        agent_start = pop()
        key_pos = pop()
        door_pos = pop()
        reward_pos = pop()

        grid[key_pos] = KEY
        grid[door_pos] = DOOR_CLOSED
        grid[reward_pos] = REWARD

        for _ in range(min(self.num_dangers, len(free))):
            grid[pop()] = DANGER

        for _ in range(min(self.num_walls, len(free))):
            grid[pop()] = WALL

        # Remember where the agent should start by carving a guaranteed empty
        # cell; the base class re-derives the start from the first empty cell,
        # so make sure agent_start stays empty and is the first empty interior.
        grid[agent_start] = EMPTY
        self._agent_start = agent_start
        self._door_pos = door_pos
        return grid

    def _find_agent_start(self) -> Tuple[int, int]:
        start = getattr(self, "_agent_start", None)
        if start is not None and self.grid[start] == EMPTY:
            return start
        return super()._find_agent_start()

    @staticmethod
    def _interior_cells(grid: np.ndarray) -> List[List[int]]:
        size = grid.shape[0]
        cells: List[List[int]] = []
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                cells.append([r, c])
        return cells
