"""Tests for MicroFouloideWorld."""
from __future__ import annotations

import numpy as np
import pytest

from seedmind.envs.micro_fouloide_world import (
    AGENT,
    COLD_ZONE,
    DANGER,
    EMPTY,
    FOOD,
    INTERACT,
    MOVE_RIGHT,
    OBSTACLE,
    REST,
    UNKNOWN,
    WAIT,
    WATER,
    WARM_ZONE,
    MicroFouloideWorld,
)


def _place_agent_on(env: MicroFouloideWorld, entity: int) -> None:
    env.grid[:, :] = 0
    env.grid[0, :] = OBSTACLE
    env.grid[-1, :] = OBSTACLE
    env.grid[:, 0] = OBSTACLE
    env.grid[:, -1] = OBSTACLE
    env.agent_pos = (2, 2)
    env.grid[2, 2] = entity


def test_drives_decay_each_step():
    env = MicroFouloideWorld(
        size=8, energy_start=0.8, hydration_start=0.7,
        energy_decay=0.01, hydration_decay=0.02, seed=0,
    )
    env.reset()
    energy = env.energy
    hydration = env.hydration
    env.step(WAIT)
    assert env.energy == pytest.approx(energy - 0.01)
    assert env.hydration == pytest.approx(hydration - 0.02)


def test_rest_reduces_energy_decay_only():
    env = MicroFouloideWorld(
        size=8, energy_start=0.8, hydration_start=0.8,
        energy_decay=0.02, hydration_decay=0.02,
        rest_energy_decay_scale=0.25, seed=0,
    )
    env.reset()
    env.step(REST)
    assert env.energy == pytest.approx(0.8 - 0.005)
    assert env.hydration == pytest.approx(0.8 - 0.02)


def test_interact_food_increases_energy_and_consumes_food():
    env = MicroFouloideWorld(
        size=8, energy_start=0.3, energy_decay=0.0,
        hydration_decay=0.0, food_energy_gain=0.4, seed=0,
    )
    env.reset()
    _place_agent_on(env, FOOD)
    _, _, _, info = env.step(INTERACT)
    assert env.energy == pytest.approx(0.7)
    assert env.grid[env.agent_pos] == 0
    assert info["event"] == "interact_food"


def test_interact_water_increases_hydration_and_consumes_water():
    env = MicroFouloideWorld(
        size=8, hydration_start=0.2, energy_decay=0.0,
        hydration_decay=0.0, water_hydration_gain=0.5, seed=0,
    )
    env.reset()
    _place_agent_on(env, WATER)
    _, _, _, info = env.step(INTERACT)
    assert env.hydration == pytest.approx(0.7)
    assert env.grid[env.agent_pos] == 0
    assert info["event"] == "interact_water"


def test_temperature_changes_on_warm_and_cold_zones():
    env = MicroFouloideWorld(
        size=8, temperature_start=0.5, temperature_drift=0.1,
        energy_decay=0.0, hydration_decay=0.0, seed=0,
    )
    env.reset()
    _place_agent_on(env, WARM_ZONE)
    env.step(WAIT)
    assert env.temperature == pytest.approx(0.6)
    _place_agent_on(env, COLD_ZONE)
    env.step(WAIT)
    assert env.temperature == pytest.approx(0.5)


def test_danger_reduces_health_and_reports_damage():
    env = MicroFouloideWorld(
        size=8, health_start=1.0, danger_damage=0.2,
        energy_decay=0.0, hydration_decay=0.0, seed=0,
    )
    env.reset()
    _place_agent_on(env, DANGER)
    _, _, _, info = env.step(WAIT)
    assert env.health == pytest.approx(0.8)
    assert info["event"] == "damage"
    assert info["health_delta"] == pytest.approx(-0.2)


def test_health_loss_can_end_episode():
    env = MicroFouloideWorld(
        size=8, health_start=0.05, energy_start=0.0,
        hydration_start=0.8, health_decay=0.1,
        energy_decay=0.0, hydration_decay=0.0, seed=0,
    )
    env.reset()
    _, reward, done, info = env.step(WAIT)
    assert done is True
    assert info["dead"] is True
    assert info["event"] == "death"
    assert reward < 0.0


def test_observation_masks_cells_outside_visibility_radius():
    env = MicroFouloideWorld(size=8, visibility_radius=1, seed=0)
    obs = env.reset()
    ar, ac = obs["agent_pos"]
    far = (ar + 3) % env.size
    if abs(far - ar) <= 1:
        far = 0
    assert obs["grid"][far, ac] == UNKNOWN
    assert np.any(obs["grid"] == AGENT)


def test_observation_exposes_entity_under_agent():
    env = MicroFouloideWorld(size=8, visibility_radius=1, seed=0)
    env.reset()
    _place_agent_on(env, FOOD)
    obs = env.observe()
    ar, ac = obs["agent_pos"]
    assert obs["grid"][ar, ac] == AGENT
    assert obs["standing_entity"] == FOOD


def test_move_blocked_by_obstacle():
    env = MicroFouloideWorld(size=8, energy_decay=0.0, hydration_decay=0.0, seed=0)
    env.reset()
    env.agent_pos = (2, 2)
    env.grid[2, 3] = OBSTACLE
    _, _, _, info = env.step(MOVE_RIGHT)
    assert env.agent_pos == (2, 2)
    assert info["event"] == "move_blocked"


def test_available_actions_can_filter_blocked_moves():
    legacy = MicroFouloideWorld(size=8, filter_blocked_moves=False, seed=0)
    legacy.reset()
    legacy.agent_pos = (2, 2)
    legacy.grid[2, 3] = OBSTACLE
    assert MOVE_RIGHT in legacy.available_actions()

    guarded = MicroFouloideWorld(size=8, filter_blocked_moves=True, seed=0)
    guarded.reset()
    guarded.agent_pos = (2, 2)
    guarded.grid[2, 3] = OBSTACLE
    assert MOVE_RIGHT not in guarded.available_actions()
    assert INTERACT in guarded.available_actions()


def test_available_actions_can_filter_noop_interact():
    legacy = MicroFouloideWorld(size=8, filter_noop_interact=False, seed=0)
    legacy.reset()
    _place_agent_on(legacy, OBSTACLE)
    legacy.grid[2, 2] = EMPTY
    assert INTERACT in legacy.available_actions()

    guarded = MicroFouloideWorld(size=8, filter_noop_interact=True, seed=0)
    guarded.reset()
    _place_agent_on(guarded, OBSTACLE)
    guarded.grid[2, 2] = EMPTY
    assert INTERACT not in guarded.available_actions()

    _place_agent_on(guarded, WATER)
    assert INTERACT in guarded.available_actions()


def test_causal_features_and_events_are_exposed():
    env = MicroFouloideWorld(size=8, seed=0)
    obs = env.reset()
    names = env.causal_feature_names()
    features = env.causal_features(obs)
    assert names == [
        "energy", "hydration", "temperature", "health",
        "standing_entity",
        "local_danger", "local_food_signal", "local_water_signal", "local_heat_signal",
    ]
    assert features.shape == (len(names),)
    assert "interact_food" in env.causal_event_names()
    assert "death" in env.causal_event_names()
