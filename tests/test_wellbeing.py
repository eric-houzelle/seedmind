"""Tests for the homeostatic wellbeing signal."""
from __future__ import annotations

import pytest

from seedmind.training.wellbeing import drive_regulation, wellbeing


COMFORT = {
    "enabled": True,
    "energy": {"low": 0.5, "high": 1.0},
    "hydration": {"low": 0.5, "high": 1.0},
    "health": {"low": 0.6, "high": 1.0},
    "temperature": {"target": 0.5, "tolerance": 0.15},
}


def test_drive_regulation_matches_legacy_formula():
    drives = {"energy": 0.8, "hydration": 0.6, "temperature": 0.4, "health": 1.0}
    expected = (0.8 + 0.6 + (1.0 - 0.2) + 1.0) / 4.0
    assert drive_regulation(drives) == pytest.approx(expected)


def test_wellbeing_without_comfort_falls_back_to_drive_regulation():
    drives = {"energy": 0.7, "hydration": 0.3, "temperature": 0.55, "health": 0.9}
    assert wellbeing(drives, None) == pytest.approx(drive_regulation(drives))
    assert wellbeing(drives, {}) == pytest.approx(drive_regulation(drives))


def test_wellbeing_is_one_inside_comfort_zones():
    drives = {"energy": 0.75, "hydration": 0.9, "temperature": 0.5, "health": 1.0}
    assert wellbeing(drives, COMFORT) == pytest.approx(1.0)


def test_wellbeing_decreases_outside_comfort_zones():
    comfortable = {"energy": 0.75, "hydration": 0.9, "temperature": 0.5, "health": 1.0}
    thirsty = dict(comfortable, hydration=0.25)
    very_thirsty = dict(comfortable, hydration=0.05)
    assert wellbeing(thirsty, COMFORT) < wellbeing(comfortable, COMFORT)
    assert wellbeing(very_thirsty, COMFORT) < wellbeing(thirsty, COMFORT)


def test_wellbeing_penalizes_temperature_distance_from_target():
    base = {"energy": 0.75, "hydration": 0.9, "health": 1.0}
    ok = wellbeing(dict(base, temperature=0.6), COMFORT)
    hot = wellbeing(dict(base, temperature=0.9), COMFORT)
    assert ok == pytest.approx(1.0)
    assert hot < ok


def test_wellbeing_is_zero_at_extreme_deprivation():
    drives = {"energy": 0.0, "hydration": 0.0, "temperature": 1.0, "health": 0.0}
    assert wellbeing(drives, COMFORT) == pytest.approx(0.0)
