"""Tests for the property-vector observation mode and the egocentric map memory (A3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from seedmind.agent.map_memory import NEVER_SEEN, MapMemory
from seedmind.agent.micro_fouloide_encoder import (
    make_micro_fouloide_obs_fns,
    make_micro_fouloide_property_obs_fns,
)
from seedmind.envs.entities import PROPERTY_DIM, PROPERTY_NAMES, default_registry, load_registry
from seedmind.envs.micro_fouloide_world import AGENT, FOOD, UNKNOWN, WAIT, MicroFouloideWorld


EXTRA_ENTITIES = [
    {"name": "berry_bush", "consumable": {"drive": "energy", "gain": 0.15}, "render": "apple"},
    {"name": "branch", "portable": True, "render": "rock"},
    {"name": "flint", "portable": True, "render": "rock"},
    {"name": "campfire", "temperature_delta": 0.03, "render": "terrain_warm"},
]


def _prop(name: str) -> int:
    return PROPERTY_NAMES.index(name)


# ---------------------------------------------------------------------------
# Property matrix
# ---------------------------------------------------------------------------

def test_property_matrix_for_historical_entities():
    registry = default_registry()
    m = registry.property_matrix()
    assert m.shape == (9, PROPERTY_DIM)
    assert np.all(m[registry.by_name("empty").id] == 0.0)
    assert m[registry.by_name("unknown").id][_prop("unknown")] == 1.0
    assert m[registry.by_name("agent").id][_prop("agent")] == 1.0
    obstacle = m[registry.by_name("obstacle").id]
    assert obstacle[_prop("solid")] == 1.0 and obstacle[_prop("occupied")] == 1.0
    food = m[registry.by_name("food").id]
    assert food[_prop("energy_gain")] == pytest.approx(0.35)
    water = m[registry.by_name("water").id]
    assert water[_prop("hydration_gain")] == pytest.approx(0.45)
    danger = m[registry.by_name("danger").id]
    assert danger[_prop("dangerous")] == pytest.approx(0.8)  # 0.08 / 0.1
    assert m[registry.by_name("warm_zone").id][_prop("warming")] == pytest.approx(0.5)
    assert m[registry.by_name("cold_zone").id][_prop("cooling")] == pytest.approx(0.5)


def test_dimensions_independent_of_entity_count():
    small = default_registry()
    big = load_registry({"env": {"entities": list(EXTRA_ENTITIES)}})
    assert big.size > small.size
    for inventory in (False, True):
        for memory in (False, True):
            _, _, ch_small, sc_small = make_micro_fouloide_property_obs_fns(
                small, inventory=inventory, memory=memory)
            _, _, ch_big, sc_big = make_micro_fouloide_property_obs_fns(
                big, inventory=inventory, memory=memory)
            assert ch_small == ch_big
            assert sc_small == sc_big
    # Contraste : le mode one-hot, lui, dépend de N.
    _, _, ch_onehot_small, _ = make_micro_fouloide_obs_fns(small.size)
    _, _, ch_onehot_big, _ = make_micro_fouloide_obs_fns(big.size)
    assert ch_onehot_small != ch_onehot_big


def test_property_obs_shapes():
    registry = default_registry()
    env = MicroFouloideWorld(size=8, seed=0, registry=registry)
    obs = env.reset()
    obs_to_vec, obs_batch, num_channels, num_scalars = make_micro_fouloide_property_obs_fns(
        registry, inventory=False, memory=False)
    assert num_channels == PROPERTY_DIM
    channels, scalars = obs_batch([obs])
    assert channels.shape == (1, num_channels, 8, 8)
    assert scalars.shape == (1, num_scalars)
    assert obs_to_vec(obs).shape == (8 * 8 * num_channels + num_scalars,)


# ---------------------------------------------------------------------------
# MapMemory
# ---------------------------------------------------------------------------

def test_map_memory_remembers_and_ages():
    env = MicroFouloideWorld(size=10, visibility_radius=2, seed=0)
    obs = env.reset()
    memory = MapMemory(env.size, horizon=10)
    memory.observe(obs)

    grid = np.asarray(obs["grid"])
    visible = grid != UNKNOWN
    not_agent = grid != AGENT  # la case de l'agent stocke la vérité, pas l'overlay
    assert np.all(memory.known[visible & not_agent] == grid[visible & not_agent])
    assert np.all(memory.known[~visible] == NEVER_SEEN)
    fresh = memory.freshness()
    assert np.all(fresh[visible] == 1.0)
    assert np.all(fresh[~visible] == 0.0)

    # Une case vue puis sortie du champ vieillit.
    seen_pos = tuple(np.argwhere(visible)[0])
    for _ in range(5):
        obs, _, _, _ = env.step(WAIT)
        memory.observe(obs)
    if np.asarray(obs["grid"])[seen_pos] == UNKNOWN:  # sortie du champ (agent immobile: non) —
        assert memory.freshness()[seen_pos] < 1.0


def test_map_memory_truth_under_agent():
    env = MicroFouloideWorld(size=8, seed=0)
    env.reset()
    env.grid[env.agent_pos] = FOOD
    obs = env.observe()
    ar, ac = obs["agent_pos"]
    assert np.asarray(obs["grid"])[ar, ac] == AGENT  # overlay
    memory = MapMemory(env.size)
    memory.observe(obs)
    assert memory.known[ar, ac] == FOOD  # la vérité, pas l'overlay


def test_map_memory_reset():
    env = MicroFouloideWorld(size=8, seed=0)
    memory = MapMemory(env.size)
    memory.observe(env.reset())
    assert np.any(memory.known != NEVER_SEEN)
    memory.reset()
    assert np.all(memory.known == NEVER_SEEN)


def test_augment_adds_memory_keys():
    env = MicroFouloideWorld(size=8, seed=0)
    obs = env.reset()
    memory = MapMemory(env.size, horizon=50)
    memory.observe(obs)
    augmented = memory.augment(obs)
    assert augmented["memory_grid"].shape == (8, 8)
    assert augmented["memory_fresh"].dtype == np.float32
    assert "grid" in augmented  # l'observation d'origine est préservée


# ---------------------------------------------------------------------------
# Acceptation A3 : ajout d'entité À CHAUD sans casser le cerveau
# ---------------------------------------------------------------------------

def _properties_config(entities: list) -> dict:
    return {
        "env": {
            "size": 8, "max_steps": 0, "soft_death": True,
            "entities": list(entities),
            "inventory": {"enabled": True, "capacity": 3},
            "property_events": True,  # vocabulaire d'événements fixe
        },
        "agent": {
            "latent_dim": 16,
            "observation": {
                "mode": "properties",
                "spatial_memory": {"enabled": True, "horizon": 50},
            },
        },
        "world_model": {"hidden_dim": 32, "num_layers": 1, "batch_size": 8},
        "causal_world_model": {"enabled": True, "predict_events": True},
        "dqn": {"conv_channels": 8, "hidden_dim": 32, "batch_size": 8},
        "policy": {"epsilon_start": 1.0, "epsilon_end": 0.5, "epsilon_decay_steps": 100},
        "online": {"update_every": 10, "updates_per_cycle": 1, "warmup_steps": 1000},
    }


def test_entity_added_mid_life_without_resizing_brain(tmp_path):
    from scripts.run_fouloide_online import OnlineFouloideSession

    torch.manual_seed(0)
    session = OnlineFouloideSession(
        _properties_config(EXTRA_ENTITIES), seed=0, device=torch.device("cpu"))
    for _ in range(60):
        session.step()
    path = tmp_path / "brain.pt"
    session.save(str(path))

    # Nouvelle entité ajoutée au monde — le cerveau existant doit la percevoir
    # par ses propriétés, sans aucun redimensionnement.
    extended = EXTRA_ENTITIES + [{
        "name": "golden_apple",
        "consumable": {"drive": "energy", "gain": 0.9},
        "count_key": "num_golden_apple", "default_count": 6,
        "render": "apple",
    }]
    config = _properties_config(extended)
    config["env"]["num_golden_apple"] = 6
    torch.manual_seed(1)
    reborn = OnlineFouloideSession(config, seed=0, device=torch.device("cpu"))
    assert reborn.env.registry.size == session.env.registry.size + 1

    resumed = reborn.resume(str(path))  # ne doit PAS lever d'erreur de dimensions
    assert resumed["env_steps"] == 60
    for _ in range(30):
        reborn.step()
    assert reborn.learner.env_steps == 90
