"""Data-driven entity registry (EPIC A — artifact engine, A1).

Entities are defined by *properties* (solid, consumable, dangerous,
temperature effect…), not by code. The world engine applies properties
generically; adding a new artifact is a YAML block, not a Python class.

The default registry reproduces exactly the nine historical entities of
``MicroFouloideWorld`` (same ids, same causal event names in the same
order) so every existing config, test and checkpoint keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Per-cell perception channels in "properties" observation mode: the agent
# perceives WHAT things are like, not WHICH entity they are — adding a new
# entity never changes these dimensions.
PROPERTY_NAMES: List[str] = [
    "occupied", "solid", "energy_gain", "hydration_gain",
    "portable", "plantable", "dangerous", "warming", "cooling",
    "unknown", "agent",
]
PROPERTY_DIM = len(PROPERTY_NAMES)
DANGEROUS_SCALE = 0.1
TEMP_SCALE = 0.03


@dataclass(frozen=True)
class EntityType:
    id: int
    name: str
    solid: bool = False              # blocks movement
    consumable: Optional[Dict[str, Any]] = None  # {"drive": "energy"|"hydration", "gain": float}
    portable: bool = False           # can be picked up into the inventory
    plantable: Optional[Dict[str, Any]] = None   # {"becomes": "<entity name>"} when planted
    dangerous: float = 0.0           # health damage per step standing on it
    temperature_delta: float = 0.0   # per-step temperature drift (warm > 0, cold < 0)
    structural: bool = False         # EMPTY/AGENT/UNKNOWN: never placed nor interacted
    count_key: Optional[str] = None  # env config key for placement count
    default_count: int = 0
    render: str = "none"             # viewer hint: rock|danger|bath|apple|terrain_warm|terrain_cold|none

    @property
    def interact_event(self) -> str:
        return f"interact_{self.name}"


@dataclass(frozen=True)
class Recipe:
    """Engine-side combination rule, never exposed to the agent."""

    inputs: Tuple[int, int]  # entity ids, sorted
    output: int


class EntityRegistry:
    """Dense id-indexed registry with property lookups."""

    def __init__(
        self,
        entities: List[EntityType],
        recipes: Optional[List[Recipe]] = None,
    ) -> None:
        by_id = sorted(entities, key=lambda e: e.id)
        for expected, entity in enumerate(by_id):
            if entity.id != expected:
                raise ValueError(
                    f"Entity ids must be dense 0..N-1; got id {entity.id} "
                    f"({entity.name!r}) at position {expected}."
                )
        names = [e.name for e in by_id]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate entity names in registry: {names}")
        self._entities: List[EntityType] = by_id
        self._by_name: Dict[str, EntityType] = {e.name: e for e in by_id}
        for entity in by_id:
            if entity.plantable is not None:
                becomes = entity.plantable.get("becomes")
                if becomes not in self._by_name:
                    raise ValueError(
                        f"Entity {entity.name!r}: plantable.becomes references "
                        f"unknown entity {becomes!r}."
                    )
        self.recipes: List[Recipe] = list(recipes or [])
        self._recipe_by_inputs: Dict[Tuple[int, int], int] = {}
        for recipe in self.recipes:
            for entity_id in (*recipe.inputs, recipe.output):
                if not 0 <= entity_id < len(by_id):
                    raise ValueError(f"Recipe references unknown entity id {entity_id}.")
            self._recipe_by_inputs[tuple(sorted(recipe.inputs))] = recipe.output

    def __len__(self) -> int:
        return len(self._entities)

    def __getitem__(self, entity_id: int) -> EntityType:
        return self._entities[int(entity_id)]

    def __iter__(self):
        return iter(self._entities)

    @property
    def size(self) -> int:
        return len(self._entities)

    def by_name(self, name: str) -> EntityType:
        return self._by_name[name]

    def get(self, entity_id: int) -> Optional[EntityType]:
        if 0 <= int(entity_id) < len(self._entities):
            return self._entities[int(entity_id)]
        return None

    # ------------------------------------------------------------------
    # Property lookups
    # ------------------------------------------------------------------
    @property
    def consumables(self) -> List[EntityType]:
        return [e for e in self._entities if e.consumable is not None]

    @property
    def solid_ids(self) -> set:
        return {e.id for e in self._entities if e.solid}

    @property
    def danger_ids(self) -> set:
        return {e.id for e in self._entities if e.dangerous > 0.0}

    @property
    def portable_ids(self) -> set:
        return {e.id for e in self._entities if e.portable}

    @property
    def heat_ids(self) -> set:
        return {e.id for e in self._entities if e.temperature_delta != 0.0}

    def drive_signal_ids(self, drive: str) -> set:
        return {
            e.id for e in self._entities
            if e.consumable is not None and e.consumable.get("drive") == drive
        }

    def placeable(self) -> List[EntityType]:
        return [
            e for e in self._entities
            if not e.structural and (e.count_key is not None or e.default_count > 0)
        ]

    def render_ids(self, render: str) -> set:
        return {e.id for e in self._entities if e.render == render}

    def find_recipe(self, item_a: int, item_b: int) -> Optional[int]:
        """Output entity id for a pair of inputs, order-insensitive."""
        return self._recipe_by_inputs.get(tuple(sorted((int(item_a), int(item_b)))))

    # ------------------------------------------------------------------
    # Property projection ("properties" observation mode)
    # ------------------------------------------------------------------
    def property_vector(self, entity_id: int) -> np.ndarray:
        return self.property_matrix()[int(entity_id)]

    def property_matrix(self) -> np.ndarray:
        """``(size, PROPERTY_DIM)`` matrix; ``matrix[grid]`` projects a grid."""
        if getattr(self, "_property_matrix", None) is None:
            matrix = np.zeros((self.size, PROPERTY_DIM), dtype=np.float32)
            for e in self._entities:
                row = matrix[e.id]
                if e.name == "unknown":
                    row[PROPERTY_NAMES.index("unknown")] = 1.0
                    continue
                if e.name == "agent":
                    row[PROPERTY_NAMES.index("agent")] = 1.0
                    continue
                if e.structural:  # empty: null vector
                    continue
                row[PROPERTY_NAMES.index("occupied")] = 1.0
                row[PROPERTY_NAMES.index("solid")] = float(e.solid)
                if e.consumable is not None:
                    drive = e.consumable.get("drive")
                    gain = float(np.clip(e.consumable.get("gain", 0.0), 0.0, 1.0))
                    if drive == "energy":
                        row[PROPERTY_NAMES.index("energy_gain")] = gain
                    elif drive == "hydration":
                        row[PROPERTY_NAMES.index("hydration_gain")] = gain
                row[PROPERTY_NAMES.index("portable")] = float(e.portable)
                row[PROPERTY_NAMES.index("plantable")] = float(e.plantable is not None)
                row[PROPERTY_NAMES.index("dangerous")] = float(
                    np.clip(e.dangerous / DANGEROUS_SCALE, 0.0, 1.0)
                )
                row[PROPERTY_NAMES.index("warming")] = float(
                    np.clip(max(e.temperature_delta, 0.0) / TEMP_SCALE, 0.0, 1.0)
                )
                row[PROPERTY_NAMES.index("cooling")] = float(
                    np.clip(max(-e.temperature_delta, 0.0) / TEMP_SCALE, 0.0, 1.0)
                )
            self._property_matrix = matrix
        return self._property_matrix

    # ------------------------------------------------------------------
    # Causal event vocabulary (order matters: WM event_index depends on it)
    # ------------------------------------------------------------------
    def causal_event_names(
        self,
        include_inventory: bool = False,
        property_events: bool = False,
    ) -> List[str]:
        """Causal event vocabulary (order is contractual: WM event_index).

        ``property_events=True`` replaces per-entity interact events with a
        fixed property-level vocabulary (``interact_energy`` …) so adding a
        consumable entity never resizes the world-model event head.
        """
        if property_events:
            interact_events = ["interact_energy", "interact_hydration", "interact_health"]
        else:
            interact_events = [e.interact_event for e in self.consumables]
        inventory_events = (
            [
                "pick_ok", "pick_noop", "drop_ok", "drop_noop",
                "plant_ok", "plant_noop", "combine_ok", "combine_noop",
            ]
            if include_inventory else []
        )
        return (
            ["move_ok", "move_blocked"]
            + interact_events
            + ["interact_noop"]
            + inventory_events
            + [
                "rest", "wait",
                "temperature_up", "temperature_down",
                "damage", "health_loss", "death",
            ]
        )


# ---------------------------------------------------------------------------
# Default registry — the nine historical entities, behaviour-identical
# ---------------------------------------------------------------------------

def default_entities(env_cfg: Optional[Dict[str, Any]] = None) -> List[EntityType]:
    ec = env_cfg or {}
    temperature_drift = float(ec.get("temperature_drift", 0.015))
    return [
        EntityType(id=0, name="empty", structural=True),
        EntityType(
            id=1, name="obstacle", solid=True, render="rock",
            count_key="num_obstacles", default_count=20,
        ),
        EntityType(id=2, name="agent", structural=True),
        EntityType(
            id=3, name="food", render="apple",
            consumable={"drive": "energy", "gain": float(ec.get("food_energy_gain", 0.35))},
            count_key="num_food", default_count=10,
        ),
        EntityType(
            id=4, name="water", render="bath",
            consumable={"drive": "hydration", "gain": float(ec.get("water_hydration_gain", 0.45))},
            count_key="num_water", default_count=8,
        ),
        EntityType(
            id=5, name="warm_zone", temperature_delta=temperature_drift,
            render="terrain_warm", count_key="num_warm_zones", default_count=6,
        ),
        EntityType(
            id=6, name="cold_zone", temperature_delta=-temperature_drift,
            render="terrain_cold", count_key="num_cold_zones", default_count=6,
        ),
        EntityType(
            id=7, name="danger", dangerous=float(ec.get("danger_damage", 0.08)),
            render="danger", count_key="num_dangers", default_count=8,
        ),
        EntityType(id=8, name="unknown", structural=True),
    ]


def default_registry(env_cfg: Optional[Dict[str, Any]] = None) -> EntityRegistry:
    return EntityRegistry(default_entities(env_cfg))


# ---------------------------------------------------------------------------
# YAML loading: default registry extended/overridden by env.entities
# ---------------------------------------------------------------------------

_ENTITY_FIELDS = {
    "id", "name", "solid", "consumable", "portable", "plantable",
    "dangerous", "temperature_delta",
    "structural", "count_key", "default_count", "render",
}


def _entity_from_spec(spec: Dict[str, Any]) -> EntityType:
    unknown = set(spec) - _ENTITY_FIELDS
    if unknown:
        raise ValueError(f"Unknown entity properties {sorted(unknown)} in {spec.get('name', spec)}")
    return EntityType(**spec)


def load_registry(config: Dict[str, Any]) -> EntityRegistry:
    """Build the registry for a run config.

    ``config["env"]["entities"]`` may be a path to a YAML file or an inline
    list of entity specs. New entities take ids >= 9; specs whose name matches
    a default entity override it (merge by replacing given fields).
    """
    env_cfg = config.get("env", {})
    entities = {e.name: e for e in default_entities(env_cfg)}
    recipe_specs: List[Dict[str, Any]] = list(env_cfg.get("recipes") or [])

    spec_source = env_cfg.get("entities")
    specs: List[Dict[str, Any]] = []
    if isinstance(spec_source, (str, Path)):
        with open(spec_source, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if isinstance(payload, dict):
            specs = payload.get("entities", [])
            recipe_specs.extend(payload.get("recipes") or [])
        else:
            specs = payload
    elif spec_source is not None:
        specs = spec_source
    if not isinstance(specs, list):
        raise ValueError("env.entities must be a list of entity specs or a YAML path containing one.")

    for spec in specs:
        name = spec.get("name")
        if name in entities:
            base = entities[name]
            overrides = {k: v for k, v in spec.items() if k != "id"}
            entities[name] = replace(base, **overrides)
        else:
            spec = dict(spec)
            spec.setdefault("id", max(e.id for e in entities.values()) + 1)
            entities[name] = _entity_from_spec(spec)

    recipes = [_recipe_from_spec(spec, entities) for spec in recipe_specs]
    return EntityRegistry(list(entities.values()), recipes=recipes)


def _recipe_from_spec(spec: Dict[str, Any], entities: Dict[str, EntityType]) -> Recipe:
    inputs = spec.get("inputs")
    output = spec.get("output")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError(f"Recipe {spec!r}: 'inputs' must list exactly two entity names.")
    resolved = []
    for name in [*inputs, output]:
        if name not in entities:
            raise ValueError(f"Recipe {spec!r}: unknown entity {name!r}.")
        resolved.append(entities[name].id)
    return Recipe(inputs=(resolved[0], resolved[1]), output=resolved[2])
