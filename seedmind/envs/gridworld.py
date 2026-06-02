"""GridWorld: the V1 2D environment for SeedMind.

The world owns *all* the rules (SPEC section 27). The agent must discover
them through experience; nothing here hints the agent about the solution.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.base import EnvironmentAdapter

# --- Entity codes (SPEC section 5) ---
EMPTY = 0
WALL = 1
AGENT = 2
KEY = 3
DOOR_CLOSED = 4
DOOR_OPEN = 5
REWARD = 6
DANGER = 7
UNKNOWN_OBJECT = 8

# --- Colored entities (V2, SPEC section 6) ---
# Colored keys/doors live in entity codes after the base V1 set so the V1
# GridWorld is unaffected (its grids simply never use these codes). Only a key
# of the matching color opens its door, which is the rule the agent must learn.
COLORS: List[str] = ["red", "blue", "green"]

COLOR_KEY = {color: 9 + i for i, color in enumerate(COLORS)}
COLOR_DOOR_CLOSED = {color: 9 + len(COLORS) + i for i, color in enumerate(COLORS)}
COLOR_DOOR_OPEN = {color: 9 + 2 * len(COLORS) + i for i, color in enumerate(COLORS)}

NUM_ENTITIES = 9 + 3 * len(COLORS)

ENTITY_NAMES = {
    EMPTY: "EMPTY",
    WALL: "WALL",
    AGENT: "AGENT",
    KEY: "KEY",
    DOOR_CLOSED: "DOOR_CLOSED",
    DOOR_OPEN: "DOOR_OPEN",
    REWARD: "REWARD",
    DANGER: "DANGER",
    UNKNOWN_OBJECT: "UNKNOWN_OBJECT",
}
for _color in COLORS:
    ENTITY_NAMES[COLOR_KEY[_color]] = f"KEY_{_color.upper()}"
    ENTITY_NAMES[COLOR_DOOR_CLOSED[_color]] = f"DOOR_{_color.upper()}_CLOSED"
    ENTITY_NAMES[COLOR_DOOR_OPEN[_color]] = f"DOOR_{_color.upper()}_OPEN"

# --- Actions (SPEC section 5) ---
MOVE_UP = "MOVE_UP"
MOVE_DOWN = "MOVE_DOWN"
MOVE_LEFT = "MOVE_LEFT"
MOVE_RIGHT = "MOVE_RIGHT"
INTERACT = "INTERACT"
WAIT = "WAIT"

ACTIONS: List[str] = [
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    INTERACT,
    WAIT,
]

_MOVE_DELTAS = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
}

# Reward constants (kept in the world, never in the agent).
REWARD_GOAL = 1.0
REWARD_DANGER = -1.0
REWARD_OPEN_DOOR = 0.25
REWARD_STEP = -0.01


class GridWorld(EnvironmentAdapter):
    """A fixed 2D grid with a key -> door -> reward scenario.

    Parameters
    ----------
    size:
        Side length of the (square) grid.
    max_steps:
        Episode horizon enforced by the world.
    layout:
        Optional pre-built ``size x size`` integer grid. When ``None`` a
        default solvable layout is built.
    rule:
        Optional dict describing the unlock rule, e.g.
        ``{"item": "key", "target": "door", "action": "interact",
        "effect": "unlock"}``. V1 GridWorld only uses it for description.
    seed:
        Seed for the (rarely used) internal RNG.
    """

    world_id = "gridworld_v1"

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 100,
        layout: Optional[np.ndarray] = None,
        rule: Optional[Dict[str, str]] = None,
        visibility_radius: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self._initial_layout = layout
        self.visibility_radius = visibility_radius
        self.rule = rule or {
            "item": "key",
            "target": "door",
            "action": "interact",
            "effect": "unlock",
        }
        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((size, size), dtype=np.int64)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.has_key: bool = False
        self.door_open: bool = False
        self.success: bool = False
        self.steps: int = 0
        self._last_event: str = "reset"

        self.reset()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_default_layout(self) -> np.ndarray:
        size = self.size
        grid = np.full((size, size), EMPTY, dtype=np.int64)

        # Outer walls.
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # Vertical wall splitting the room, with a door in the middle.
        wall_col = size // 2
        grid[1:-1, wall_col] = WALL
        door_row = size // 2
        grid[door_row, wall_col] = DOOR_CLOSED
        self._door_pos = (door_row, wall_col)

        # Key on the agent's side (left of the dividing wall).
        grid[1, 1] = KEY

        # Reward behind the door (right side).
        grid[size - 2, size - 2] = REWARD

        # A danger tile to learn to avoid (left side, out of the direct path).
        if size >= 6:
            grid[size - 2, 1] = DANGER

        return grid

    # ------------------------------------------------------------------
    # EnvironmentAdapter interface
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        if self._initial_layout is not None:
            self.grid = self._initial_layout.astype(np.int64).copy()
        else:
            self.grid = self._build_default_layout()

        # Place the agent on the first empty interior cell (top-left area).
        self.agent_pos = self._find_agent_start()
        self.has_key = False
        self.door_open = False
        self.success = False
        self.steps = 0
        self._last_event = "reset"
        return self.observe()

    def _find_agent_start(self) -> Tuple[int, int]:
        # Prefer cell (1, 1) area; fall back to any empty cell.
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                if self.grid[r, c] == EMPTY:
                    return (r, c)
        return (1, 1)

    def _apply_visibility_mask(self, view: np.ndarray) -> np.ndarray:
        """Replace cells outside the visibility radius with UNKNOWN_OBJECT."""
        if self.visibility_radius is None:
            return view
        ar, ac = self.agent_pos
        for r in range(self.size):
            for c in range(self.size):
                if abs(r - ar) + abs(c - ac) > self.visibility_radius:
                    view[r, c] = UNKNOWN_OBJECT
        return view

    def observe(self) -> Dict[str, Any]:
        """Return the agent's view of the grid plus inventory.

        When ``visibility_radius`` is set, cells beyond that Manhattan distance
        from the agent are replaced with ``UNKNOWN_OBJECT``. The agent always
        knows its own position.
        """
        view = self.grid.copy()
        r, c = self.agent_pos
        if view[r, c] in (EMPTY, DOOR_OPEN):
            view[r, c] = AGENT
        self._apply_visibility_mask(view)
        return {
            "grid": view,
            "agent_pos": self.agent_pos,
            "has_key": int(self.has_key),
            "door_open": int(self.door_open),
        }

    def available_actions(self) -> List[str]:
        return list(ACTIONS)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")

        reward = REWARD_STEP
        self._last_event = action
        done = False

        if action in _MOVE_DELTAS:
            reward += self._try_move(action)
        elif action == INTERACT:
            reward += self._try_interact()
        elif action == WAIT:
            self._last_event = "wait"

        self.steps += 1

        if self.success:
            done = True
        if self.steps >= self.max_steps:
            done = True

        info = {
            "success": self.success,
            "has_key": int(self.has_key),
            "door_open": int(self.door_open),
            "event": self._last_event,
            "steps": self.steps,
        }
        return self.observe(), float(reward), done, info

    # ------------------------------------------------------------------
    # Rule logic (lives entirely in the world)
    # ------------------------------------------------------------------
    def _try_move(self, action: str) -> float:
        dr, dc = _MOVE_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size):
            self._last_event = "blocked_bounds"
            return 0.0

        target = self.grid[nr, nc]

        if target == WALL:
            self._last_event = "blocked_wall"
            return 0.0
        if target == DOOR_CLOSED:
            self._last_event = "blocked_door"
            return 0.0

        # The move succeeds; resolve what is on the destination cell.
        bonus = 0.0
        if target == KEY:
            self.has_key = True
            self.grid[nr, nc] = EMPTY
            self._last_event = "picked_key"
        elif target == DANGER:
            bonus += REWARD_DANGER
            self._last_event = "hit_danger"
        elif target == REWARD:
            bonus += REWARD_GOAL
            self.success = True
            self.grid[nr, nc] = EMPTY
            self._last_event = "reached_reward"
        elif target == DOOR_OPEN:
            self._last_event = "through_door"

        self.agent_pos = (nr, nc)
        return bonus

    def _try_interact(self) -> float:
        """Open an adjacent closed door if the agent holds a key."""
        r, c = self.agent_pos
        for dr, dc in _MOVE_DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            if self.grid[nr, nc] == DOOR_CLOSED and self.has_key:
                self.grid[nr, nc] = DOOR_OPEN
                self.door_open = True
                self.has_key = False  # key is consumed
                self._last_event = "opened_door"
                return REWARD_OPEN_DOOR
        self._last_event = "interact_noop"
        return 0.0

    def describe_transition(self) -> str:
        return (
            f"event={self._last_event} pos={self.agent_pos} "
            f"has_key={int(self.has_key)} door_open={int(self.door_open)} "
            f"success={int(self.success)}"
        )
