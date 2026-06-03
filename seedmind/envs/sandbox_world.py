"""SandboxWorld — open-ended survival environment.

The agent starts with a finite energy reserve that drains every step. The grid
contains food sources that can be harvested for food items, then eaten to
restore energy. Optional craft mechanics add a longer causal chain:
wood + stone -> tool -> more efficient food harvesting. The agent knows
*nothing* about these rules — it must discover them on its own.

Entities and actions are defined in **registries** (module-level dicts) so that
adding wood, stone, crafting recipes, or construction later requires only new
registry entries, not structural changes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.base import EnvironmentAdapter

# ---------------------------------------------------------------------------
# Entity registry — extensible via new entries
# ---------------------------------------------------------------------------
EMPTY = 0
WALL = 1
AGENT = 2
FOOD_SOURCE = 3
FOOD_SOURCE_DEPLETED = 4
UNKNOWN = 5
WOOD_SOURCE = 6
STONE_SOURCE = 7
WORKBENCH = 8

NUM_ENTITIES = 6
CRAFT_NUM_ENTITIES = 9

ENTITY_NAMES: Dict[int, str] = {
    EMPTY: "EMPTY",
    WALL: "WALL",
    AGENT: "AGENT",
    FOOD_SOURCE: "FOOD_SOURCE",
    FOOD_SOURCE_DEPLETED: "FOOD_SOURCE_DEPLETED",
    UNKNOWN: "UNKNOWN",
    WOOD_SOURCE: "WOOD_SOURCE",
    STONE_SOURCE: "STONE_SOURCE",
    WORKBENCH: "WORKBENCH",
}

# ---------------------------------------------------------------------------
# Action registry — extensible via new entries
# ---------------------------------------------------------------------------
MOVE_UP = "MOVE_UP"
MOVE_DOWN = "MOVE_DOWN"
MOVE_LEFT = "MOVE_LEFT"
MOVE_RIGHT = "MOVE_RIGHT"
HARVEST = "HARVEST"
EAT = "EAT"
WAIT = "WAIT"
CRAFT = "CRAFT"

BASE_ACTIONS: List[str] = [
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    HARVEST, EAT, WAIT,
]
CRAFT_ACTIONS: List[str] = BASE_ACTIONS + [CRAFT]

# Backward-compatible alias used by the v0/v1 sandbox configs and checkpoints.
ACTIONS: List[str] = list(BASE_ACTIONS)

_MOVE_DELTAS = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
}

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
ALIVE_BONUS = 0.01
DEATH_PENALTY = -1.0


class SandboxWorld(EnvironmentAdapter):
    """Grid world where survival is the only objective.

    Parameters
    ----------
    size : int
        Side length of the square grid.
    max_steps : int
        Hard episode horizon (prevents infinite episodes when the agent is
        very good at surviving).
    energy_max : float
        Energy cap.
    energy_start : float
        Energy at the beginning of each life.
    energy_decay : float
        Energy lost every step.
    food_energy : float
        Energy restored by eating one food item.
    num_food_sources : int
        Number of food sources placed in each generated map.
    num_walls : int
        Number of random interior wall blocks.
    regrow_delay : int
        Steps until a depleted food source regenerates.
    visibility_radius : int or None
        When set, cells beyond this Manhattan distance are hidden.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 200,
        energy_max: float = 100.0,
        energy_start: float = 50.0,
        energy_decay: float = 1.0,
        food_energy: float = 15.0,
        num_food_sources: int = 5,
        num_wood_sources: int = 0,
        num_stone_sources: int = 0,
        num_workbenches: int = 0,
        num_walls: int = 0,
        regrow_delay: int = 15,
        craft_enabled: bool = False,
        base_food_yield: int = 1,
        tool_food_bonus: int = 1,
        visibility_radius: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.energy_max = energy_max
        self.energy_start = energy_start
        self.energy_decay = energy_decay
        self.food_energy = food_energy
        self.num_food_sources = num_food_sources
        self.num_wood_sources = num_wood_sources
        self.num_stone_sources = num_stone_sources
        self.num_workbenches = num_workbenches
        self.num_walls = num_walls if num_walls > 0 else max(0, (size - 2) ** 2 // 10)
        self.regrow_delay = regrow_delay
        self.craft_enabled = craft_enabled or num_wood_sources > 0 or num_stone_sources > 0
        self.base_food_yield = max(0, int(base_food_yield))
        self.tool_food_bonus = max(0, int(tool_food_bonus))
        self.visibility_radius = visibility_radius
        self.rng = np.random.default_rng(seed)

        # Mutable state — initialised properly in reset()
        self.grid = np.zeros((size, size), dtype=np.int64)
        self.agent_pos: Tuple[int, int] = (1, 1)
        self.energy: float = energy_start
        self.inventory: Dict[str, int] = self._empty_inventory()
        self.steps: int = 0
        self._regrow_timers: Dict[Tuple[int, int], int] = {}
        self._last_event: str = "reset"
        self._last_event_amount: int = 0

        self.reset()

    @property
    def world_id(self) -> str:
        return "sandbox_craft" if self.craft_enabled else "sandbox_v0"

    def _empty_inventory(self) -> Dict[str, int]:
        inv = {"food": 0}
        if self.craft_enabled:
            inv.update({"wood": 0, "stone": 0, "tool": 0})
        return inv

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------
    def _generate_layout(self) -> np.ndarray:
        grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)
        # Border walls
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        interior = [
            (r, c)
            for r in range(1, self.size - 1)
            for c in range(1, self.size - 1)
        ]
        self.rng.shuffle(interior)
        idx = 0

        # Random walls
        for _ in range(min(self.num_walls, len(interior) - self.num_food_sources - 1)):
            grid[interior[idx]] = WALL
            idx += 1

        placements = (
            [(FOOD_SOURCE, self.num_food_sources)]
            + ([
                (WOOD_SOURCE, self.num_wood_sources),
                (STONE_SOURCE, self.num_stone_sources),
                (WORKBENCH, self.num_workbenches),
            ] if self.craft_enabled else [])
        )
        for entity, count in placements:
            for _ in range(count):
                if idx >= len(interior):
                    break
                grid[interior[idx]] = entity
                idx += 1

        return grid

    def _find_empty(self) -> Tuple[int, int]:
        empties = list(zip(*np.where(self.grid == EMPTY)))
        if empties:
            return empties[int(self.rng.integers(len(empties)))]
        return (1, 1)

    # ------------------------------------------------------------------
    # EnvironmentAdapter interface
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.grid = self._generate_layout()
        self.agent_pos = self._find_empty()
        self.energy = self.energy_start
        self.inventory = self._empty_inventory()
        self.steps = 0
        self._regrow_timers = {}
        self._last_event = "reset"
        self._last_event_amount = 0
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        view = self.grid.copy()
        r, c = self.agent_pos
        if view[r, c] == EMPTY:
            view[r, c] = AGENT
        if self.visibility_radius is not None:
            ar, ac = self.agent_pos
            for row in range(self.size):
                for col in range(self.size):
                    if abs(row - ar) + abs(col - ac) > self.visibility_radius:
                        view[row, col] = UNKNOWN
        return {
            "grid": view,
            "agent_pos": self.agent_pos,
            "energy": self.energy,
            "energy_max": self.energy_max,
            "inventory_food": self.inventory["food"],
            "inventory_wood": self.inventory.get("wood", 0),
            "inventory_stone": self.inventory.get("stone", 0),
            "inventory_tool": self.inventory.get("tool", 0),
        }

    def available_actions(self) -> List[str]:
        return list(CRAFT_ACTIONS if self.craft_enabled else BASE_ACTIONS)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.steps += 1
        self._last_event = action
        self._last_event_amount = 0

        # --- Execute action ---
        if action in _MOVE_DELTAS:
            self._try_move(action)
        elif action == HARVEST:
            self._try_harvest()
        elif action == EAT:
            self._try_eat()
        elif action == CRAFT and self.craft_enabled:
            self._try_craft()
        # WAIT — do nothing

        # --- Tick energy ---
        self.energy -= self.energy_decay
        self.energy = max(0.0, self.energy)

        # --- Tick regrow ---
        self._tick_regrow()

        # --- Done / reward ---
        dead = self.energy <= 0
        timeout = self.steps >= self.max_steps
        done = dead or timeout

        if dead:
            reward = DEATH_PENALTY
        else:
            reward = ALIVE_BONUS

        info: Dict[str, Any] = {
            "dead": dead,
            "timeout": timeout,
            "lifespan": self.steps,
            "event": self._last_event,
            "event_amount": self._last_event_amount,
            "energy": self.energy,
            "inventory": dict(self.inventory),
        }

        return self.observe(), reward, done, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _try_move(self, action: str) -> None:
        dr, dc = _MOVE_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if self.grid[nr, nc] != WALL:
                self.agent_pos = (nr, nc)

    def _try_harvest(self) -> None:
        r, c = self.agent_pos
        entity = int(self.grid[r, c])
        if entity == FOOD_SOURCE:
            tool_used = self.inventory.get("tool", 0) > 0
            amount = self.base_food_yield + (self.tool_food_bonus if tool_used else 0)
            self.inventory["food"] += amount
            self.grid[r, c] = FOOD_SOURCE_DEPLETED
            self._regrow_timers[(r, c)] = self.regrow_delay
            self._last_event = "harvest_food_tool" if tool_used else "harvest_food"
            self._last_event_amount = amount
        elif self.craft_enabled and entity == WOOD_SOURCE:
            self.inventory["wood"] += 1
            self._last_event = "harvest_wood"
            self._last_event_amount = 1
        elif self.craft_enabled and entity == STONE_SOURCE:
            self.inventory["stone"] += 1
            self._last_event = "harvest_stone"
            self._last_event_amount = 1
        else:
            self._last_event = "harvest_noop"

    def _try_eat(self) -> None:
        if self.inventory["food"] > 0:
            self.inventory["food"] -= 1
            self.energy = min(self.energy_max, self.energy + self.food_energy)
            self._last_event = "eat_ok"
            self._last_event_amount = 1
        else:
            self._last_event = "eat_noop"

    def _try_craft(self) -> None:
        if self.inventory.get("wood", 0) >= 1 and self.inventory.get("stone", 0) >= 1:
            self.inventory["wood"] -= 1
            self.inventory["stone"] -= 1
            self.inventory["tool"] += 1
            self._last_event = "craft_tool"
            self._last_event_amount = 1
        else:
            self._last_event = "craft_noop"

    def _tick_regrow(self) -> None:
        regrown = []
        for pos in list(self._regrow_timers):
            self._regrow_timers[pos] -= 1
            if self._regrow_timers[pos] <= 0:
                self.grid[pos] = FOOD_SOURCE
                regrown.append(pos)
        for pos in regrown:
            del self._regrow_timers[pos]

    def describe_transition(self) -> str:
        return self._last_event
