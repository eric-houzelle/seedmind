"""Tests for the data-driven entity registry (A1)."""
from __future__ import annotations

import numpy as np
import pytest

from seedmind.envs.entities import EntityRegistry, EntityType, default_registry, load_registry
from seedmind.envs.micro_fouloide_world import (
    EMPTY,
    INTERACT,
    WAIT,
    MicroFouloideWorld,
)


BERRY_SPEC = [{
    "name": "berry_bush",
    "consumable": {"drive": "energy", "gain": 0.15},
    "count_key": "num_berry_bush",
    "default_count": 4,
    "render": "apple",
}]


def _config_with_berries(num: int = 4) -> dict:
    return {"env": {"entities": list(BERRY_SPEC), "num_berry_bush": num}}


# ---------------------------------------------------------------------------
# Default registry == historical behaviour
# ---------------------------------------------------------------------------

def test_default_registry_matches_historical_entities():
    registry = default_registry()
    assert registry.size == 9
    names = [e.name for e in registry]
    assert names == [
        "empty", "obstacle", "agent", "food", "water",
        "warm_zone", "cold_zone", "danger", "unknown",
    ]
    assert registry.by_name("obstacle").solid
    assert registry.by_name("food").consumable["drive"] == "energy"
    assert registry.by_name("water").consumable["drive"] == "hydration"
    assert registry.by_name("danger").dangerous > 0.0
    assert registry.by_name("warm_zone").temperature_delta > 0.0
    assert registry.by_name("cold_zone").temperature_delta < 0.0


def test_default_event_names_are_stable_in_order():
    # L'ordre est contractuel : event_index du WM causal en dépend.
    assert default_registry().causal_event_names() == [
        "move_ok", "move_blocked",
        "interact_food", "interact_water",
        "interact_noop", "rest", "wait",
        "temperature_up", "temperature_down",
        "damage", "health_loss", "death",
    ]


def test_world_without_registry_behaves_as_before():
    env = MicroFouloideWorld(size=8, seed=0)
    assert env.registry.size == 9
    assert env.causal_event_names()[2:4] == ["interact_food", "interact_water"]


def test_event_names_with_inventory_extension():
    events = default_registry().causal_event_names(include_inventory=True)
    idx = events.index("interact_noop")
    assert events[idx + 1: idx + 9] == [
        "pick_ok", "pick_noop", "drop_ok", "drop_noop",
        "plant_ok", "plant_noop", "combine_ok", "combine_noop",
    ]
    assert events[-1] == "death"


def test_recipes_load_and_resolve_names():
    registry = load_registry({"env": {
        "entities": [
            {"name": "branch", "portable": True},
            {"name": "flint", "portable": True},
            {"name": "campfire", "temperature_delta": 0.03},
        ],
        "recipes": [{"inputs": ["branch", "flint"], "output": "campfire"}],
    }})
    branch = registry.by_name("branch").id
    flint = registry.by_name("flint").id
    campfire = registry.by_name("campfire").id
    assert registry.find_recipe(branch, flint) == campfire
    assert registry.find_recipe(flint, branch) == campfire  # insensible à l'ordre
    assert registry.find_recipe(branch, branch) is None


def test_recipe_with_unknown_entity_raises():
    with pytest.raises(ValueError):
        load_registry({"env": {
            "entities": [{"name": "branch", "portable": True}],
            "recipes": [{"inputs": ["branch", "ghost"], "output": "branch"}],
        }})


def test_recipe_requires_exactly_two_inputs():
    with pytest.raises(ValueError):
        load_registry({"env": {
            "entities": [{"name": "branch", "portable": True}],
            "recipes": [{"inputs": ["branch"], "output": "branch"}],
        }})


def test_registry_rejects_unknown_plantable_target():
    with pytest.raises(ValueError):
        load_registry({"env": {"entities": [{
            "name": "ghost_seed",
            "portable": True,
            "plantable": {"becomes": "nonexistent"},
        }]}})


def test_registry_rejects_non_dense_ids():
    with pytest.raises(ValueError):
        EntityRegistry([
            EntityType(id=0, name="empty", structural=True),
            EntityType(id=2, name="hole"),
        ])


# ---------------------------------------------------------------------------
# Extensibility: a new entity is just a YAML block
# ---------------------------------------------------------------------------

def test_load_registry_extends_defaults():
    registry = load_registry(_config_with_berries())
    assert registry.size == 10
    berry = registry.by_name("berry_bush")
    assert berry.id == 9
    assert berry.consumable == {"drive": "energy", "gain": 0.15}
    events = registry.causal_event_names()
    assert "interact_berry_bush" in events
    assert events.index("interact_berry_bush") == events.index("interact_water") + 1


def test_load_registry_can_override_default_entity():
    registry = load_registry({"env": {"entities": [{"name": "danger", "dangerous": 0.5}]}})
    assert registry.size == 9
    assert registry.by_name("danger").dangerous == 0.5
    assert registry.by_name("danger").id == 7


def test_world_places_and_consumes_new_entity():
    registry = load_registry(_config_with_berries())
    env = MicroFouloideWorld(
        size=10, energy_decay=0.0, hydration_decay=0.0,
        energy_start=0.3, resource_regrow_steps=2,
        registry=registry, entity_counts={"num_berry_bush": 6},
        seed=0,
    )
    env.reset()
    berry_id = registry.by_name("berry_bush").id
    assert int(np.sum(env.grid == berry_id)) == 6

    # Place l'agent sur un buisson et mange.
    pos = tuple(int(v) for v in np.argwhere(env.grid == berry_id)[0])
    env.agent_pos = pos
    _, _, _, info = env.step(INTERACT)
    assert info["event"] == "interact_berry_bush"
    assert env.energy == pytest.approx(0.3 + 0.15)
    assert env.grid[pos] == EMPTY

    # Et ça repousse comme les autres ressources.
    env.agent_pos = (1, 1) if pos != (1, 1) else (2, 2)
    env.step(WAIT)
    env.step(WAIT)
    assert env.grid[pos] == berry_id


def test_build_agent_sizes_networks_from_registry():
    from scripts.run_micro_fouloide import build_agent, build_env

    config = {
        "env": {
            "size": 8, "entities": list(BERRY_SPEC), "num_berry_bush": 3,
        },
        "agent": {"latent_dim": 16},
        "world_model": {"hidden_dim": 32, "num_layers": 1},
        "dqn": {"conv_channels": 8, "hidden_dim": 32},
    }
    env = build_env(config, seed=0)
    assert env.registry.size == 10
    agent = build_agent(config, seed=0)
    obs = env.reset()
    latent = agent.encode(obs)
    action = agent.choose_action(latent, "explore_unknown_area", [], env.available_actions(), observation=obs)
    assert action in env.available_actions()
    # Le QNetwork accepte l'observation à 10 entités.
    q_values = agent.q_network.q_values(obs)
    assert q_values.shape[-1] == 7
