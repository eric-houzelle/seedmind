"""MicroFouloideWorld — minimal multi-drive organism world.

This world is a bridge between the sandbox and a future fouloide-like
ecosystem. It models one organism with internal drives. The agent does not know
the rules; it only observes state, acts, and receives consequences through the
EnvironmentAdapter interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.base import EnvironmentAdapter
from seedmind.envs.entities import EntityRegistry, default_registry

# ---------------------------------------------------------------------------
# Historical entity ids (the default registry reproduces them exactly)
# ---------------------------------------------------------------------------
EMPTY = 0
OBSTACLE = 1
AGENT = 2
FOOD = 3
WATER = 4
WARM_ZONE = 5
COLD_ZONE = 6
DANGER = 7
UNKNOWN = 8

NUM_ENTITIES = 9

ENTITY_NAMES: Dict[int, str] = {
    EMPTY: "EMPTY",
    OBSTACLE: "OBSTACLE",
    AGENT: "AGENT",
    FOOD: "FOOD",
    WATER: "WATER",
    WARM_ZONE: "WARM_ZONE",
    COLD_ZONE: "COLD_ZONE",
    DANGER: "DANGER",
    UNKNOWN: "UNKNOWN",
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
MOVE_UP = "MOVE_UP"
MOVE_DOWN = "MOVE_DOWN"
MOVE_LEFT = "MOVE_LEFT"
MOVE_RIGHT = "MOVE_RIGHT"
INTERACT = "INTERACT"
REST = "REST"
WAIT = "WAIT"

ACTIONS: List[str] = [
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    INTERACT, REST, WAIT,
]

_MOVE_DELTAS = {
    MOVE_UP: (-1, 0),
    MOVE_DOWN: (1, 0),
    MOVE_LEFT: (0, -1),
    MOVE_RIGHT: (0, 1),
}

ALIVE_BONUS = 0.01
DEATH_PENALTY = -1.0


class MicroFouloideWorld(EnvironmentAdapter):
    """Single-organism world with multiple internal drives."""

    def __init__(
        self,
        size: int = 16,
        max_steps: int = 500,
        energy_start: float = 0.75,
        hydration_start: float = 0.75,
        temperature_start: float = 0.5,
        health_start: float = 1.0,
        energy_decay: float = 0.006,
        hydration_decay: float = 0.008,
        rest_energy_decay_scale: float = 0.35,
        food_energy_gain: float = 0.35,
        water_hydration_gain: float = 0.45,
        temperature_drift: float = 0.015,
        temperature_recovery: float = 0.004,
        critical_threshold: float = 0.12,
        health_decay: float = 0.025,
        danger_damage: float = 0.08,
        soft_death: bool = False,
        health_floor: float = 0.05,
        health_regen: float = 0.01,
        resource_regrow_steps: int = 0,
        num_food: int = 10,
        num_water: int = 8,
        num_warm_zones: int = 6,
        num_cold_zones: int = 6,
        num_dangers: int = 8,
        num_obstacles: int = 20,
        visibility_radius: Optional[int] = 4,
        filter_blocked_moves: bool = False,
        filter_noop_interact: bool = False,
        registry: Optional[EntityRegistry] = None,
        entity_counts: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.energy_start = float(energy_start)
        self.hydration_start = float(hydration_start)
        self.temperature_start = float(temperature_start)
        self.health_start = float(health_start)
        self.energy_decay = float(energy_decay)
        self.hydration_decay = float(hydration_decay)
        self.rest_energy_decay_scale = float(rest_energy_decay_scale)
        self.food_energy_gain = float(food_energy_gain)
        self.water_hydration_gain = float(water_hydration_gain)
        self.temperature_drift = float(temperature_drift)
        self.temperature_recovery = float(temperature_recovery)
        self.critical_threshold = float(critical_threshold)
        self.health_decay = float(health_decay)
        self.danger_damage = float(danger_damage)
        self.soft_death = bool(soft_death)
        self.health_floor = float(health_floor)
        self.health_regen = float(health_regen)
        self.resource_regrow_steps = int(resource_regrow_steps)
        self.num_food = int(num_food)
        self.num_water = int(num_water)
        self.num_warm_zones = int(num_warm_zones)
        self.num_cold_zones = int(num_cold_zones)
        self.num_dangers = int(num_dangers)
        self.num_obstacles = int(num_obstacles)
        self.visibility_radius = visibility_radius
        self.filter_blocked_moves = bool(filter_blocked_moves)
        self.filter_noop_interact = bool(filter_noop_interact)
        if registry is None:
            registry = default_registry({
                "food_energy_gain": self.food_energy_gain,
                "water_hydration_gain": self.water_hydration_gain,
                "danger_damage": self.danger_damage,
                "temperature_drift": self.temperature_drift,
            })
        self.registry = registry
        self._entity_counts = dict(entity_counts or {})
        self._solid_ids = registry.solid_ids
        self._consumable_ids = {e.id for e in registry.consumables}
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.size, self.size), dtype=np.int64)
        self.agent_pos: Tuple[int, int] = (1, 1)
        self.energy = self.energy_start
        self.hydration = self.hydration_start
        self.temperature = self.temperature_start
        self.health = self.health_start
        self.steps = 0
        self._last_event = "reset"
        self._last_event_amount = 0
        self._last_health_delta = 0.0
        self._regrow_queue: List[Tuple[int, int, int, int]] = []

        self.reset()

    @property
    def world_id(self) -> str:
        return "micro_fouloide_v0"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _generate_layout(self) -> np.ndarray:
        grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)
        grid[0, :] = OBSTACLE
        grid[-1, :] = OBSTACLE
        grid[:, 0] = OBSTACLE
        grid[:, -1] = OBSTACLE

        interior = [
            (r, c)
            for r in range(1, self.size - 1)
            for c in range(1, self.size - 1)
        ]
        self.rng.shuffle(interior)
        idx = 0
        legacy_counts = {
            "num_obstacles": self.num_obstacles,
            "num_food": self.num_food,
            "num_water": self.num_water,
            "num_warm_zones": self.num_warm_zones,
            "num_cold_zones": self.num_cold_zones,
            "num_dangers": self.num_dangers,
        }
        placements = []
        for entity_type in self.registry.placeable():
            count = legacy_counts.get(entity_type.count_key)
            if count is None:
                count = self._entity_counts.get(entity_type.count_key, entity_type.default_count)
            placements.append((entity_type.id, int(count)))
        for entity, count in placements:
            for _ in range(count):
                if idx >= len(interior):
                    break
                grid[interior[idx]] = entity
                idx += 1
        return grid

    def _find_empty(self) -> Tuple[int, int]:
        empties = list(zip(*np.where(self.grid == EMPTY)))
        if not empties:
            return (1, 1)
        return empties[int(self.rng.integers(len(empties)))]

    # ------------------------------------------------------------------
    # EnvironmentAdapter
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.grid = self._generate_layout()
        self.agent_pos = self._find_empty()
        self.energy = self.energy_start
        self.hydration = self.hydration_start
        self.temperature = self.temperature_start
        self.health = self.health_start
        self.steps = 0
        self._last_event = "reset"
        self._last_event_amount = 0
        self._last_health_delta = 0.0
        self._regrow_queue = []
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        view = self.grid.copy()
        r, c = self.agent_pos
        standing_entity = int(self.grid[r, c])
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
            "standing_entity": standing_entity,
            "energy": self.energy,
            "hydration": self.hydration,
            "temperature": self.temperature,
            "health": self.health,
        }

    def available_actions(self) -> List[str]:
        return [
            action
            for action in ACTIONS
            if self._action_is_available(action)
        ]

    def _action_is_available(self, action: str) -> bool:
        if self.filter_blocked_moves and action in _MOVE_DELTAS and not self._can_move(action):
            return False
        if self.filter_noop_interact and action == INTERACT and not self._can_interact():
            return False
        return True

    def causal_feature_names(self) -> List[str]:
        return [
            "energy",
            "hydration",
            "temperature",
            "health",
            "standing_entity",
            "local_danger",
            "local_food_signal",
            "local_water_signal",
            "local_heat_signal",
        ]

    def causal_features(self, observation: Dict[str, Any]) -> np.ndarray:
        grid = np.asarray(observation["grid"], dtype=np.int64)
        registry = self.registry
        local_danger = float(np.any(np.isin(grid, list(registry.danger_ids))))
        local_food = float(np.any(np.isin(grid, list(registry.drive_signal_ids("energy")))))
        local_water = float(np.any(np.isin(grid, list(registry.drive_signal_ids("hydration")))))
        local_heat = float(np.any(np.isin(grid, list(registry.heat_ids))))
        return np.asarray([
            float(observation.get("energy", 0.0)),
            float(observation.get("hydration", 0.0)),
            float(observation.get("temperature", 0.5)),
            float(observation.get("health", 0.0)),
            float(observation.get("standing_entity", EMPTY)) / max(registry.size - 1, 1),
            local_danger,
            local_food,
            local_water,
            local_heat,
        ], dtype=np.float32)

    def causal_event_names(self) -> List[str]:
        return self.registry.causal_event_names()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.steps += 1
        self._last_event = action
        self._last_event_amount = 0
        self._last_health_delta = 0.0
        self._tick_regrowth()

        if action in _MOVE_DELTAS:
            self._try_move(action)
        elif action == INTERACT:
            self._try_interact()
        elif action == REST:
            self._last_event = "rest"
        elif action == WAIT:
            self._last_event = "wait"

        self._tick_drives(action)
        timeout = self.max_steps > 0 and self.steps >= self.max_steps
        dead = self.health <= 0.0
        done = dead or timeout
        if dead:
            self._last_event = "death"
        reward = DEATH_PENALTY if dead else ALIVE_BONUS
        info = {
            "dead": dead,
            "timeout": timeout and not dead,
            "lifespan": self.steps,
            "event": self._last_event,
            "event_amount": self._last_event_amount,
            "energy": self.energy,
            "hydration": self.hydration,
            "temperature": self.temperature,
            "health": self.health,
            "health_delta": self._last_health_delta,
            "drives": {
                "energy": self.energy,
                "hydration": self.hydration,
                "temperature": self.temperature,
                "health": self.health,
            },
        }
        return self.observe(), reward, done, info

    # ------------------------------------------------------------------
    # Action and dynamics
    # ------------------------------------------------------------------
    def _can_move(self, action: str) -> bool:
        dr, dc = _MOVE_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        return (
            0 <= nr < self.size
            and 0 <= nc < self.size
            and int(self.grid[nr, nc]) not in self._solid_ids
        )

    def _can_interact(self) -> bool:
        r, c = self.agent_pos
        return int(self.grid[r, c]) in self._consumable_ids

    def _try_move(self, action: str) -> None:
        if not self._can_move(action):
            self._last_event = "move_blocked"
            return
        dr, dc = _MOVE_DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        self.agent_pos = (nr, nc)
        self._last_event = "move_ok"

    def _try_interact(self) -> None:
        r, c = self.agent_pos
        entity = int(self.grid[r, c])
        entity_type = self.registry.get(entity)
        if entity_type is not None and entity_type.consumable is not None:
            drive = str(entity_type.consumable.get("drive", "energy"))
            gain = float(entity_type.consumable.get("gain", 0.0))
            if drive in {"energy", "hydration", "health"}:
                setattr(self, drive, min(1.0, getattr(self, drive) + gain))
            self.grid[r, c] = EMPTY
            self._queue_regrowth(r, c, entity)
            self._last_event = entity_type.interact_event
            self._last_event_amount = 1
        else:
            self._last_event = "interact_noop"

    def _queue_regrowth(self, r: int, c: int, entity: int) -> None:
        if self.resource_regrow_steps > 0:
            self._regrow_queue.append((r, c, entity, self.steps + self.resource_regrow_steps))

    def _tick_regrowth(self) -> None:
        if not self._regrow_queue:
            return
        remaining = []
        for r, c, entity, due_step in self._regrow_queue:
            if (
                self.steps >= due_step
                and self.grid[r, c] == EMPTY
                and (r, c) != self.agent_pos
            ):
                self.grid[r, c] = entity
            else:
                remaining.append((r, c, entity, due_step))
        self._regrow_queue = remaining

    def _tick_drives(self, action: str) -> None:
        energy_decay = self.energy_decay
        if action == REST:
            energy_decay *= self.rest_energy_decay_scale
        self.energy = max(0.0, self.energy - energy_decay)
        self.hydration = max(0.0, self.hydration - self.hydration_decay)

        r, c = self.agent_pos
        entity = int(self.grid[r, c])
        entity_type = self.registry.get(entity)
        temperature_delta = entity_type.temperature_delta if entity_type else 0.0
        if temperature_delta > 0.0:
            self.temperature = min(1.0, self.temperature + temperature_delta)
            if self._last_event in {"move_ok", WAIT, REST}:
                self._last_event = "temperature_up"
        elif temperature_delta < 0.0:
            self.temperature = max(0.0, self.temperature + temperature_delta)
            if self._last_event in {"move_ok", WAIT, REST}:
                self._last_event = "temperature_down"
        elif self.temperature < 0.5:
            self.temperature = min(0.5, self.temperature + self.temperature_recovery)
        elif self.temperature > 0.5:
            self.temperature = max(0.5, self.temperature - self.temperature_recovery)

        health_before = self.health
        dangerous = entity_type.dangerous if entity_type else 0.0
        if dangerous > 0.0:
            self.health = max(0.0, self.health - dangerous)
            self._last_event = "damage"
        critical = (
            self.energy <= self.critical_threshold
            or self.hydration <= self.critical_threshold
            or self.temperature <= self.critical_threshold
            or self.temperature >= 1.0 - self.critical_threshold
        )
        if critical:
            if self.soft_death:
                # Degraded state: critical drives erode health down to a floor,
                # only danger tiles can take health below it.
                if self.health > self.health_floor:
                    self.health = max(self.health_floor, self.health - self.health_decay)
                    if self._last_event not in {"damage", "death"}:
                        self._last_event = "health_loss"
            else:
                self.health = max(0.0, self.health - self.health_decay)
                if self._last_event not in {"damage", "death"}:
                    self._last_event = "health_loss"
        elif self.soft_death and dangerous == 0.0 and self.health < self.health_start:
            self.health = min(self.health_start, self.health + self.health_regen)
        self._last_health_delta = self.health - health_before

    def describe_transition(self) -> str:
        return self._last_event
