"""Deployment-time spatial memory for visible Micro-Fouloide resources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seedmind.envs.micro_fouloide_world import (
    FOOD,
    INTERACT,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    OBSTACLE,
    UNKNOWN,
    WATER,
)

MOVE_ACTIONS = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
}


def _agent_pos(obs: dict[str, Any]) -> tuple[int, int]:
    return tuple(int(x) for x in obs.get("agent_pos", (-1, -1)))


@dataclass
class SpatialResourceMemory:
    """Remember visible food/water cells and route back when drives are low.

    Entity ids are injectable so the memory works with any entity registry
    (defaults reproduce the historical hardcoded ids).
    """

    water: set[tuple[int, int]] = field(default_factory=set)
    food: set[tuple[int, int]] = field(default_factory=set)
    water_ids: set[int] = field(default_factory=lambda: {WATER})
    food_ids: set[int] = field(default_factory=lambda: {FOOD})
    solid_ids: set[int] = field(default_factory=lambda: {OBSTACLE})
    unknown_id: int = UNKNOWN

    def reset(self) -> None:
        self.water.clear()
        self.food.clear()

    def refresh(self, obs: dict[str, Any]) -> None:
        grid = np.asarray(obs.get("grid", []), dtype=np.int64)
        if grid.size == 0:
            return

        standing_entity = int(obs.get("standing_entity", 0))
        agent_pos = _agent_pos(obs)
        visible: set[tuple[int, int]] = set()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                entity = int(grid[row, col])
                if entity == self.unknown_id:
                    continue
                pos = (row, col)
                visible.add(pos)
                if entity in self.water_ids or (pos == agent_pos and standing_entity in self.water_ids):
                    self.water.add(pos)
                if entity in self.food_ids or (pos == agent_pos and standing_entity in self.food_ids):
                    self.food.add(pos)

        self._forget_stale("water", self.water_ids, visible, grid, agent_pos, standing_entity)
        self._forget_stale("food", self.food_ids, visible, grid, agent_pos, standing_entity)

    def choose_action(
        self,
        obs: dict[str, Any],
        available_actions: list[str],
        hydration_threshold: float = 0.55,
        energy_threshold: float = 0.25,
    ) -> str | None:
        agent_pos = _agent_pos(obs)
        standing_entity = int(obs.get("standing_entity", 0))
        hydration = float(obs.get("hydration", 1.0))
        energy = float(obs.get("energy", 1.0))

        targets: set[tuple[int, int]]
        target_ids: set[int]
        if hydration <= hydration_threshold:
            targets = self.water
            target_ids = self.water_ids
        elif energy <= energy_threshold:
            targets = self.food
            target_ids = self.food_ids
        else:
            return None

        if standing_entity in target_ids and INTERACT in available_actions:
            return INTERACT

        target = self._nearest(agent_pos, targets)
        if target is None:
            return None

        grid = np.asarray(obs.get("grid", []), dtype=np.int64)
        candidates: list[tuple[int, str]] = []
        for action, (dr, dc) in MOVE_ACTIONS.items():
            if action not in available_actions:
                continue
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
            if grid.size and 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if int(grid[nr, nc]) in self.solid_ids:
                    continue
            distance = abs(target[0] - nr) + abs(target[1] - nc)
            candidates.append((distance, action))
        if not candidates:
            return None
        return min(candidates)[1]

    def _forget_stale(
        self,
        name: str,
        entity_ids: set[int],
        visible: set[tuple[int, int]],
        grid: np.ndarray,
        agent_pos: tuple[int, int],
        standing_entity: int,
    ) -> None:
        resources = self.water if name == "water" else self.food
        stale = {
            pos
            for pos in resources
            if pos in visible
            and int(grid[pos]) not in entity_ids
            and not (pos == agent_pos and standing_entity in entity_ids)
        }
        resources.difference_update(stale)

    @staticmethod
    def _nearest(
        agent_pos: tuple[int, int],
        targets: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not targets:
            return None
        return min(
            targets,
            key=lambda pos: (
                abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1]),
                pos,
            ),
        )
