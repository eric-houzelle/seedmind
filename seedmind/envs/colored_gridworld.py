"""Colored GridWorld with variable rules (SPEC sections 6 & 24).

The task is: *open the door whose color matches the key you carry*. The active
color changes between episodes (rule change), and distractor doors/keys of
other colors force the agent to discriminate colors rather than memorise a
single door. Holding the wrong-color key (or no key) does nothing on a door.

This is the substrate for the transfer experiment (SPEC 24.6): train on some
colors, test on a held-out color, and check the abstract rule "matching color
opens" generalises.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.gridworld import (
    AGENT,
    COLOR_DOOR_CLOSED,
    COLOR_DOOR_OPEN,
    COLOR_KEY,
    COLORS,
    DANGER,
    DOOR_OPEN,
    EMPTY,
    REWARD_DANGER,
    REWARD_GOAL,
    REWARD_OPEN_DOOR,
    WALL,
)
from seedmind.envs.procedural_gridworld import ProceduralGridWorld

# Reverse lookups: entity code -> color.
_CLOSED_DOOR_COLOR = {code: color for color, code in COLOR_DOOR_CLOSED.items()}
_OPEN_DOOR_CODES = set(COLOR_DOOR_OPEN.values()) | {DOOR_OPEN}
_KEY_COLOR = {code: color for color, code in COLOR_KEY.items()}
_MOVE_DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
}


class ColoredGridWorld(ProceduralGridWorld):
    """Procedural world whose goal is to open the matching-color door."""

    world_id = "colored_gridworld_v2"

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 100,
        allowed_colors: Optional[List[str]] = None,
        num_distractor_doors: int = 1,
        num_distractor_keys: int = 1,
        num_dangers: int = 2,
        num_walls: int = 0,
        shaping: bool = True,
        shaping_coef: float = 0.1,
        shaping_gamma: float = 0.99,
        bump_penalty: float = 0.05,
        visibility_radius: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.allowed_colors = list(allowed_colors) if allowed_colors else list(COLORS)
        self.num_distractor_doors = num_distractor_doors
        self.num_distractor_keys = num_distractor_keys
        # Extra penalty for bumping into a wall/closed door: without it the
        # greedy policy can dead-lock against a wall (same state -> same argmax
        # forever). It also gives a crisp "don't do that" learning signal.
        self.bump_penalty = bump_penalty
        # Potential-based reward shaping toward the current subgoal (matching key,
        # then matching door). This dense, color-agnostic gradient is what lets
        # the policy generalise across procedurally-generated maps and transfer
        # to unseen colors; being potential-based it preserves the optimal policy.
        self.shaping = shaping
        self.shaping_coef = shaping_coef
        self.shaping_gamma = shaping_gamma
        self.key_color: Optional[str] = None
        self.active_color: Optional[str] = None
        self._door_pos: Optional[Tuple[int, int]] = None
        super().__init__(
            size=size,
            max_steps=max_steps,
            num_dangers=num_dangers,
            num_walls=num_walls,
            regenerate_each_reset=True,
            visibility_radius=visibility_radius,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.key_color = None
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

        active = str(self.rng.choice(self.allowed_colors))
        self.active_color = active
        other_colors = [c for c in COLORS if c != active]

        agent_start = pop()
        grid[pop()] = COLOR_KEY[active]
        door_pos = pop()
        grid[door_pos] = COLOR_DOOR_CLOSED[active]

        for _ in range(min(self.num_distractor_doors, len(free))):
            if not other_colors:
                break
            color = str(self.rng.choice(other_colors))
            grid[pop()] = COLOR_DOOR_CLOSED[color]

        for _ in range(min(self.num_distractor_keys, len(free))):
            if not other_colors:
                break
            color = str(self.rng.choice(other_colors))
            grid[pop()] = COLOR_KEY[color]

        for _ in range(min(self.num_dangers, len(free))):
            grid[pop()] = DANGER

        for _ in range(min(self.num_walls, len(free))):
            grid[pop()] = WALL

        grid[agent_start] = EMPTY
        self._agent_start = agent_start
        self._door_pos = door_pos
        return grid

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self) -> Dict[str, Any]:
        view = self.grid.copy()
        r, c = self.agent_pos
        if int(view[r, c]) == EMPTY or int(view[r, c]) in _OPEN_DOOR_CODES:
            view[r, c] = AGENT
        self._apply_visibility_mask(view)
        return {
            "grid": view,
            "agent_pos": self.agent_pos,
            "has_key": int(self.has_key),
            "key_color": self.key_color,
            "door_open": int(self.door_open),
            "active_color": self.active_color,
        }

    # ------------------------------------------------------------------
    # Rule logic (colored): lives entirely in the world
    # ------------------------------------------------------------------
    def _try_move(self, action: str) -> float:
        dr, dc = _MOVE_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size):
            self._last_event = "blocked_bounds"
            return -self.bump_penalty

        target = int(self.grid[nr, nc])

        if target == WALL:
            self._last_event = "blocked_wall"
            return -self.bump_penalty
        if target in _CLOSED_DOOR_COLOR:
            self._last_event = "blocked_door"
            return -self.bump_penalty

        bonus = 0.0
        if target in _KEY_COLOR:
            self.has_key = True
            self.key_color = _KEY_COLOR[target]
            self.grid[nr, nc] = EMPTY
            self._last_event = f"picked_key_{self.key_color}"
            # Shaping: picking the matching-color key is progress toward the
            # goal, which densifies an otherwise very sparse reward.
            if self.key_color == self.active_color:
                bonus += REWARD_OPEN_DOOR
        elif target == DANGER:
            bonus += REWARD_DANGER
            self._last_event = "hit_danger"
        elif target in _OPEN_DOOR_CODES:
            self._last_event = "through_door"

        self.agent_pos = (nr, nc)
        return bonus

    def _try_interact(self) -> float:
        r, c = self.agent_pos
        for dr, dc in _MOVE_DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            target = int(self.grid[nr, nc])
            if target in _CLOSED_DOOR_COLOR:
                door_color = _CLOSED_DOOR_COLOR[target]
                if self.has_key and self.key_color == door_color:
                    self.grid[nr, nc] = COLOR_DOOR_OPEN[door_color]
                    self.door_open = True
                    self.success = True
                    self._last_event = f"opened_door_{door_color}"
                    return REWARD_GOAL
        # Interacting with nothing (or the wrong-color door) is a no-op; penalise
        # it like a wall bump so the greedy policy can't dead-lock on INTERACT
        # and learns to only interact at its matching door.
        self._last_event = "interact_noop"
        return -self.bump_penalty

    # ------------------------------------------------------------------
    # Potential-based reward shaping
    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _matching_key_pos(self) -> Optional[Tuple[int, int]]:
        matches = np.argwhere(self.grid == COLOR_KEY[self.active_color])
        return tuple(matches[0]) if len(matches) else None

    def _potential(self) -> float:
        """Stationary potential = negative *total remaining path length*.

        Using the whole remaining path (agent -> matching key -> matching door
        when the key is not yet held, else agent -> door) keeps the potential
        monotonic along the optimal trajectory, so picking up the key is no
        longer punished by a sudden subgoal switch.
        """
        if self._door_pos is None:
            return 0.0
        if self.has_key and self.key_color == self.active_color:
            remaining = self._manhattan(self.agent_pos, self._door_pos)
        else:
            key_pos = self._matching_key_pos()
            if key_pos is None:
                remaining = self._manhattan(self.agent_pos, self._door_pos)
            else:
                remaining = (
                    self._manhattan(self.agent_pos, key_pos)
                    + self._manhattan(key_pos, self._door_pos)
                )
        return -self.shaping_coef * float(remaining)

    def step(self, action: str):
        phi_before = self._potential() if self.shaping else 0.0
        obs, reward, done, info = super().step(action)
        if self.shaping:
            phi_after = 0.0 if self.success else self._potential()
            reward += self.shaping_gamma * phi_after - phi_before
        return obs, reward, done, info

    def describe_transition(self) -> str:
        return (
            f"event={self._last_event} pos={self.agent_pos} "
            f"active={self.active_color} key={self.key_color} "
            f"door_open={int(self.door_open)} success={int(self.success)}"
        )
