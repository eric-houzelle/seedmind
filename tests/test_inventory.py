"""Tests for the inventory and generic actions PICK/DROP/COMBINE/PLANT (A2)."""
from __future__ import annotations

import numpy as np
import pytest

from seedmind.envs.entities import load_registry
from seedmind.envs.micro_fouloide_world import (
    ACTIONS,
    COMBINE,
    DROP,
    EMPTY,
    INTERACT,
    PICK,
    PLANT,
    WAIT,
    MicroFouloideWorld,
)


SEED_AND_BUSH = [
    {
        "name": "berry_bush",
        "consumable": {"drive": "energy", "gain": 0.15},
        "count_key": "num_berry_bush",
        "default_count": 0,
        "render": "apple",
    },
    {
        "name": "berry_seed",
        "portable": True,
        "plantable": {"becomes": "berry_bush"},
        "count_key": "num_berry_seed",
        "default_count": 0,
        "render": "apple",
    },
    {"name": "branch", "portable": True, "render": "rock"},
    {"name": "flint", "portable": True, "render": "rock"},
    {"name": "twine", "portable": True, "render": "rock"},
    {"name": "campfire", "temperature_delta": 0.03, "render": "terrain_warm"},
]

RECIPES = [
    {"inputs": ["branch", "flint"], "output": "campfire"},
    {"inputs": ["branch", "twine"], "output": "berry_seed"},  # produit portable
]


def _make_env(**kwargs) -> MicroFouloideWorld:
    registry = load_registry({"env": {"entities": list(SEED_AND_BUSH), "recipes": list(RECIPES)}})
    defaults = dict(
        size=8, energy_decay=0.0, hydration_decay=0.0,
        inventory_enabled=True, registry=registry, seed=0,
    )
    defaults.update(kwargs)
    return MicroFouloideWorld(**defaults)


def _clear_around_agent(env: MicroFouloideWorld) -> None:
    env.grid[1:-1, 1:-1] = EMPTY
    env.agent_pos = (3, 3)


def test_world_without_inventory_keeps_seven_actions_and_obs():
    env = MicroFouloideWorld(size=8, seed=0)
    assert env.actions == list(ACTIONS)
    assert len(env.available_actions()) == 7
    assert "inventory" not in env.observe()
    assert len(env.causal_event_names()) == 12


def test_inventory_world_exposes_eleven_actions_and_events():
    env = _make_env()
    assert len(env.actions) == 11
    assert {PICK, DROP, COMBINE, PLANT} <= set(env.actions)
    events = env.causal_event_names()
    for name in ("pick_ok", "drop_noop", "plant_ok", "combine_noop"):
        assert name in events
    obs = env.observe()
    assert obs["inventory"].shape == (env.registry.size,)


def test_pick_and_drop_round_trip():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    seed_id = env.registry.by_name("berry_seed").id
    env.grid[3, 3] = seed_id

    _, _, _, info = env.step(PICK)
    assert info["event"] == "pick_ok"
    assert env.inventory == [seed_id]
    assert env.grid[3, 3] == EMPTY
    assert env.observe()["inventory"][seed_id] == pytest.approx(1 / 3)

    _, _, _, info = env.step(DROP)
    assert info["event"] == "drop_ok"
    assert env.inventory == []
    assert env.grid[3, 3] == seed_id


def test_pick_noop_when_nothing_portable_or_full():
    env = _make_env(inventory_capacity=1)
    env.reset()
    _clear_around_agent(env)
    _, _, _, info = env.step(PICK)
    assert info["event"] == "pick_noop"

    seed_id = env.registry.by_name("berry_seed").id
    env.grid[3, 3] = seed_id
    env.step(PICK)
    env.grid[3, 3] = seed_id  # another seed appears underfoot
    _, _, _, info = env.step(PICK)
    assert info["event"] == "pick_noop"  # capacity 1, already full


def test_plant_transforms_seed_into_target_entity():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    seed_id = env.registry.by_name("berry_seed").id
    bush_id = env.registry.by_name("berry_bush").id
    env.grid[3, 3] = seed_id
    env.step(PICK)

    _, _, _, info = env.step(PLANT)
    assert info["event"] == "plant_ok"
    assert env.inventory == []
    assert env.grid[3, 3] == bush_id

    # Chaîne complète : le buisson planté se mange.
    _, _, _, info = env.step(INTERACT)
    assert info["event"] == "interact_berry_bush"


def test_plant_noop_without_plantable_item_or_on_occupied_cell():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    _, _, _, info = env.step(PLANT)
    assert info["event"] == "plant_noop"  # rien en main

    seed_id = env.registry.by_name("berry_seed").id
    env.grid[3, 3] = seed_id
    env.step(PICK)
    env.grid[3, 3] = env.registry.by_name("food").id  # case occupée
    _, _, _, info = env.step(PLANT)
    assert info["event"] == "plant_noop"


def test_combine_without_matching_pair_is_noop():
    env = _make_env()
    env.reset()
    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_noop"  # inventaire vide

    _clear_around_agent(env)
    env.inventory = [env.registry.by_name("branch").id, env.registry.by_name("branch").id]
    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_noop"  # pas de recette branch+branch
    assert len(env.inventory) == 2  # rien n'est consommé


def test_combine_places_non_portable_output_on_empty_cell():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    branch = env.registry.by_name("branch").id
    flint = env.registry.by_name("flint").id
    campfire = env.registry.by_name("campfire").id
    env.inventory = [branch, flint]

    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_ok"
    assert env.inventory == []
    assert env.grid[3, 3] == campfire

    # Retombée drive : rester sur le feu fait monter la température.
    temp_before = env.temperature
    env.step(WAIT)
    assert env.temperature > temp_before


def test_combine_noop_on_occupied_cell_consumes_nothing():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    env.grid[3, 3] = env.registry.by_name("food").id  # case occupée
    env.inventory = [
        env.registry.by_name("branch").id,
        env.registry.by_name("flint").id,
    ]
    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_noop"
    assert len(env.inventory) == 2


def test_combine_portable_output_goes_to_inventory():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    env.grid[3, 3] = env.registry.by_name("food").id  # case occupée : OK si portable
    branch = env.registry.by_name("branch").id
    twine = env.registry.by_name("twine").id
    seed = env.registry.by_name("berry_seed").id
    env.inventory = [branch, twine]

    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_ok"
    assert env.inventory == [seed]


def test_combine_is_order_insensitive():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    env.inventory = [
        env.registry.by_name("flint").id,
        env.registry.by_name("branch").id,  # ordre inversé
    ]
    _, _, _, info = env.step(COMBINE)
    assert info["event"] == "combine_ok"


def test_filter_noop_inventory_gates_combine():
    env = _make_env(filter_noop_inventory=True)
    env.reset()
    _clear_around_agent(env)
    assert COMBINE not in env.available_actions()
    env.inventory = [
        env.registry.by_name("branch").id,
        env.registry.by_name("flint").id,
    ]
    assert COMBINE in env.available_actions()


def test_filter_noop_inventory_prunes_impossible_actions():
    env = _make_env(filter_noop_inventory=True)
    env.reset()
    _clear_around_agent(env)
    available = env.available_actions()
    assert PICK not in available and DROP not in available and PLANT not in available

    seed_id = env.registry.by_name("berry_seed").id
    env.grid[3, 3] = seed_id
    assert PICK in env.available_actions()
    env.step(PICK)
    available = env.available_actions()
    assert DROP in available and PLANT in available


def test_inventory_cleared_on_reset():
    env = _make_env()
    env.reset()
    _clear_around_agent(env)
    env.grid[3, 3] = env.registry.by_name("berry_seed").id
    env.step(PICK)
    assert env.inventory
    env.reset()
    assert env.inventory == []


def test_agent_loop_with_inventory_world():
    from scripts.run_micro_fouloide import build_agent, build_env

    config = {
        "env": {
            "size": 8,
            "entities": list(SEED_AND_BUSH),
            "num_berry_seed": 3, "num_berry_bush": 2,
            "inventory": {"enabled": True, "capacity": 3},
        },
        "agent": {"latent_dim": 16},
        "world_model": {"hidden_dim": 32, "num_layers": 1},
        "dqn": {"conv_channels": 8, "hidden_dim": 32},
    }
    env = build_env(config, seed=0)
    assert len(env.actions) == 11
    agent = build_agent(config, seed=0)
    assert len(agent.actions) == 11
    obs = env.reset()
    latent = agent.encode(obs)
    for _ in range(5):
        action = agent.choose_action(
            latent, "explore_unknown_area", [], env.available_actions(), observation=obs,
        )
        obs, _, _, _ = env.step(action)
        latent = agent.encode(obs)
    q_values = agent.q_network.q_values(obs)
    assert q_values.shape[-1] == 11


def test_make_scorer_alignment_with_filtered_actions():
    """Régression : Q-values alignées même quand available_actions est filtrée."""
    from scripts.run_micro_fouloide import build_agent, build_env

    config = {
        "env": {"size": 8},
        "agent": {"latent_dim": 16},
        "world_model": {"hidden_dim": 32, "num_layers": 1},
        "dqn": {"conv_channels": 8, "hidden_dim": 32},
    }
    env = build_env(config, seed=0)
    agent = build_agent(config, seed=0)
    obs = env.reset()
    full_values = agent.q_network.q_values(obs)
    # Sous-ensemble filtré : MOVE_DOWN retiré → les indices ne doivent pas glisser.
    filtered = [a for a in ACTIONS if a != "MOVE_DOWN"]
    scorer = agent.q_network.make_scorer(obs, filtered, action_index=agent.action_index)
    for action in filtered:
        assert scorer(action) == pytest.approx(float(full_values[agent.action_index[action]]))
