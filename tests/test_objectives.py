"""Tests for configurable planner objectives."""
from __future__ import annotations

import numpy as np

from seedmind.objectives import build_objective_scorer


def test_survival_objective_prefers_healthy_drives():
    scorer = build_objective_scorer(
        {
            "objective": {
                "enabled": True,
                "type": "survival_v0",
                "critical_threshold": 0.18,
                "risk_weight": 3.0,
            }
        },
        ["energy", "hydration", "temperature", "health"],
    )
    assert scorer is not None

    features = np.asarray([
        [0.8, 0.8, 0.5, 1.0],
        [0.8, 0.05, 0.5, 1.0],
        [0.8, 0.8, 0.5, 0.1],
    ], dtype=np.float32)

    scores = scorer.score_batch(features)

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_disabled_objective_returns_none():
    assert build_objective_scorer({"objective": {"enabled": False}}, ["energy"]) is None


def test_survival_delta_objective_penalizes_drive_decline():
    scorer = build_objective_scorer(
        {
            "objective": {
                "enabled": True,
                "type": "survival_v0",
                "mode": "delta",
                "decline_weight": 4.0,
                "critical_decline_weight": 8.0,
                "weights": {"hydration": 3.0, "health": 3.0, "energy": 1.0},
            }
        },
        ["energy", "hydration", "temperature", "health"],
    )
    assert scorer is not None

    before = np.asarray([
        [0.7, 0.25, 0.5, 1.0],
        [0.7, 0.25, 0.5, 1.0],
    ], dtype=np.float32)
    after = np.asarray([
        [0.7, 0.30, 0.5, 1.0],
        [0.7, 0.15, 0.5, 1.0],
    ], dtype=np.float32)

    scores = scorer.score_transition_batch(before, after)

    assert scores[0] > scores[1]
    assert scores[1] < 0.0
