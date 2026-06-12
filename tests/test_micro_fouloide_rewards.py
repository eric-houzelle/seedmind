"""Tests for Micro-Fouloide learning reward shaping."""
from __future__ import annotations

from scripts.run_micro_fouloide import _learning_reward


def _config() -> dict:
    return {
        "drive_reward": {"enabled": False},
        "resource_reward": {
            "enabled": True,
            "weight": 1.0,
            "low_threshold": 0.35,
            "critical_threshold": 0.18,
            "event_rewards": {
                "interact_water": 0.10,
                "interact_food": 0.06,
                "interact_noop": -0.15,
            },
            "low_hydration_water_bonus": 0.45,
            "critical_hydration_water_bonus": 0.70,
            "low_energy_food_bonus": 0.25,
            "critical_energy_food_bonus": 0.40,
            "low_drive_passive_penalty": 0.08,
            "low_hydration_water_signal_bonus": 0.05,
            "critical_hydration_water_signal_bonus": 0.08,
            "low_energy_food_signal_bonus": 0.04,
            "critical_energy_food_signal_bonus": 0.06,
        },
    }


def test_resource_reward_prioritizes_water_when_hydration_is_low():
    cfg = _config()
    high_hydration = {
        "energy": 0.8,
        "hydration": 0.8,
        "temperature": 0.5,
        "health": 1.0,
    }
    low_hydration = {**high_hydration, "hydration": 0.15}
    info = {"event": "interact_water", "hydration": 0.57, "energy": 0.8}

    normal = _learning_reward(0.01, high_hydration, info, cfg)
    urgent = _learning_reward(0.01, low_hydration, info, cfg)

    assert urgent > normal
    assert urgent == 0.01 + 0.10 + 0.45 + 0.70


def test_resource_reward_penalizes_passive_low_drive_steps():
    cfg = _config()
    observation = {
        "energy": 0.7,
        "hydration": 0.2,
        "temperature": 0.5,
        "health": 1.0,
    }

    wait = _learning_reward(
        0.01,
        observation,
        {"event": "wait", "hydration": 0.19, "energy": 0.7},
        cfg,
    )

    assert wait == 0.01 - 0.08


def test_resource_reward_uses_separate_passive_penalty_threshold():
    cfg = _config()
    cfg["resource_reward"]["passive_penalty_threshold"] = 0.55
    observation = {
        "energy": 0.7,
        "hydration": 0.58,
        "temperature": 0.5,
        "health": 1.0,
    }

    wait = _learning_reward(
        0.01,
        observation,
        {"event": "wait", "hydration": 0.54, "energy": 0.7},
        cfg,
    )

    assert wait == 0.01 - 0.08


def test_resource_reward_penalizes_noop_interaction():
    cfg = _config()
    observation = {
        "energy": 0.7,
        "hydration": 0.7,
        "temperature": 0.5,
        "health": 1.0,
    }

    reward = _learning_reward(
        0.01,
        observation,
        {"event": "interact_noop", "hydration": 0.69, "energy": 0.69},
        cfg,
    )

    assert reward == 0.01 - 0.15


def test_resource_reward_rewards_active_water_search_when_hydration_is_low():
    cfg = _config()
    observation = {
        "grid": [[0, 4, 0]],
        "energy": 0.7,
        "hydration": 0.15,
        "temperature": 0.5,
        "health": 1.0,
    }

    reward = _learning_reward(
        0.01,
        observation,
        {"event": "move_ok", "hydration": 0.14, "energy": 0.69},
        cfg,
    )

    assert reward == 0.01 + 0.05 + 0.08


def test_resource_reward_does_not_reward_passive_water_signal():
    cfg = _config()
    observation = {
        "grid": [[0, 4, 0]],
        "energy": 0.7,
        "hydration": 0.15,
        "temperature": 0.5,
        "health": 1.0,
    }

    reward = _learning_reward(
        0.01,
        observation,
        {"event": "rest", "hydration": 0.14, "energy": 0.69},
        cfg,
    )

    assert reward == 0.01 - 0.08
