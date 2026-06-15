"""Tests for the live demo action curriculum."""
from __future__ import annotations

from collections import deque

from scripts.run_fouloide_online import curriculum_available_actions
from seedmind.envs.micro_fouloide_world import COMBINE, DROP, INTERACT, MOVE_UP, PICK, PLANT, REST


def _cfg() -> dict:
    return {
        "action_curriculum": {
            "enabled": True,
            "artifact_actions": [PICK, DROP, PLANT, COMBINE],
            "unlock_after_steps": 100,
            "min_recent_samples": 3,
            "min_recent_wellbeing": 0.6,
            "min_recent_hydration_events": 2,
            "min_recent_energy_events": 1,
        }
    }


def test_curriculum_is_noop_when_disabled():
    actions = [MOVE_UP, INTERACT, PICK, PLANT, REST]
    filtered, unlocked = curriculum_available_actions(
        actions, {}, 0, deque(), deque()
    )
    assert unlocked is True
    assert filtered == actions


def test_curriculum_blocks_artifacts_before_survival_competence():
    actions = [MOVE_UP, INTERACT, PICK, DROP, PLANT, COMBINE, REST]
    filtered, unlocked = curriculum_available_actions(
        actions,
        _cfg(),
        200,
        deque([0.8, 0.8, 0.8], maxlen=10),
        deque(["interact_hydration"], maxlen=10),
    )
    assert unlocked is False
    assert filtered == [MOVE_UP, INTERACT, REST]


def test_curriculum_unlocks_artifacts_after_survival_competence():
    actions = [MOVE_UP, INTERACT, PICK, DROP, PLANT, COMBINE, REST]
    filtered, unlocked = curriculum_available_actions(
        actions,
        _cfg(),
        200,
        deque([0.7, 0.8, 0.75], maxlen=10),
        deque(["interact_hydration", "interact_energy", "interact_water"], maxlen=10),
    )
    assert unlocked is True
    assert filtered == actions
