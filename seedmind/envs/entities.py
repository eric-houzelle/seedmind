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
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class EntityType:
    id: int
    name: str
    solid: bool = False              # blocks movement
    consumable: Optional[Dict[str, Any]] = None  # {"drive": "energy"|"hydration", "gain": float}
    dangerous: float = 0.0           # health damage per step standing on it
    temperature_delta: float = 0.0   # per-step temperature drift (warm > 0, cold < 0)
    structural: bool = False         # EMPTY/AGENT/UNKNOWN: never placed nor interacted
    count_key: Optional[str] = None  # env config key for placement count
    default_count: int = 0
    render: str = "none"             # viewer hint: rock|danger|bath|apple|terrain_warm|terrain_cold|none

    @property
    def interact_event(self) -> str:
        return f"interact_{self.name}"


class EntityRegistry:
    """Dense id-indexed registry with property lookups."""

    def __init__(self, entities: List[EntityType]) -> None:
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

    # ------------------------------------------------------------------
    # Causal event vocabulary (order matters: WM event_index depends on it)
    # ------------------------------------------------------------------
    def causal_event_names(self) -> List[str]:
        return (
            ["move_ok", "move_blocked"]
            + [e.interact_event for e in self.consumables]
            + [
                "interact_noop", "rest", "wait",
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
    "id", "name", "solid", "consumable", "dangerous", "temperature_delta",
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

    spec_source = env_cfg.get("entities")
    if spec_source is None:
        return EntityRegistry(list(entities.values()))

    if isinstance(spec_source, (str, Path)):
        with open(spec_source, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        specs = payload.get("entities", payload) if isinstance(payload, dict) else payload
    else:
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

    return EntityRegistry(list(entities.values()))
