"""Tests for soft-death grace behavior in MicroFouloideWorld."""
from __future__ import annotations

import pytest

from seedmind.envs.micro_fouloide_world import WAIT, MicroFouloideWorld


def test_soft_death_grace_can_expire_and_allow_starvation_death():
    env = MicroFouloideWorld(
        size=8,
        soft_death=True,
        health_floor=0.2,
        soft_death_grace_steps=2,
        health_start=0.3,
        health_decay=0.1,
        critical_kill_health_decay=0.1,
        energy_start=0.0,
        hydration_start=0.8,
        energy_decay=0.0,
        hydration_decay=0.0,
        seed=0,
    )
    env.reset()

    _, _, done, _ = env.step(WAIT)
    assert done is False
    assert env.health == pytest.approx(0.2)

    _, _, done, _ = env.step(WAIT)
    assert done is False
    assert env.health == pytest.approx(0.2)

    _, _, done, _ = env.step(WAIT)
    assert done is False
    assert env.health == pytest.approx(0.1)

    _, _, done, info = env.step(WAIT)
    assert done is True
    assert info["dead"] is True


def test_soft_death_grace_resets_when_drives_recover():
    env = MicroFouloideWorld(
        size=8,
        soft_death=True,
        health_floor=0.2,
        soft_death_grace_steps=2,
        health_start=0.5,
        health_decay=0.1,
        critical_kill_health_decay=0.1,
        energy_start=0.0,
        hydration_start=0.8,
        energy_decay=0.0,
        hydration_decay=0.0,
        seed=0,
    )
    env.reset()

    env.step(WAIT)
    env.energy = 0.8
    env.step(WAIT)
    env.energy = 0.0
    _, _, done, _ = env.step(WAIT)

    assert done is False
    assert env.health == pytest.approx(0.31)
