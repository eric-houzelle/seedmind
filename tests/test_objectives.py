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
