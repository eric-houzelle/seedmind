"""Tests for Micro-Fouloide spatial resource memory."""
from __future__ import annotations

from seedmind.agent.spatial_resource_memory import SpatialResourceMemory
from seedmind.envs.micro_fouloide_world import (
    AGENT,
    EMPTY,
    FOOD,
    INTERACT,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    OBSTACLE,
    UNKNOWN,
    WATER,
)


def test_spatial_resource_memory_remembers_visible_resources():
    memory = SpatialResourceMemory()
    obs = {
        "grid": [
            [EMPTY, WATER, UNKNOWN],
            [FOOD, AGENT, EMPTY],
        ],
        "agent_pos": (1, 1),
        "standing_entity": EMPTY,
    }

    memory.refresh(obs)

    assert memory.water == {(0, 1)}
    assert memory.food == {(1, 0)}


def test_spatial_resource_memory_interacts_when_standing_on_needed_resource():
    memory = SpatialResourceMemory(water={(1, 1)})
    obs = {
        "grid": [
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, AGENT, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        "agent_pos": (1, 1),
        "standing_entity": WATER,
        "hydration": 0.1,
        "energy": 0.9,
    }

    action = memory.choose_action(
        obs,
        [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT],
    )

    assert action == INTERACT


def test_spatial_resource_memory_moves_toward_remembered_water_when_thirsty():
    memory = SpatialResourceMemory(water={(0, 1)})
    obs = {
        "grid": [
            [EMPTY, WATER, EMPTY],
            [EMPTY, AGENT, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        "agent_pos": (1, 1),
        "standing_entity": EMPTY,
        "hydration": 0.2,
        "energy": 0.9,
    }

    action = memory.choose_action(obs, [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])

    assert action == MOVE_UP


def test_spatial_resource_memory_forgets_visible_stale_resources():
    memory = SpatialResourceMemory(water={(0, 1)})
    obs = {
        "grid": [
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, AGENT, EMPTY],
        ],
        "agent_pos": (1, 1),
        "standing_entity": EMPTY,
    }

    memory.refresh(obs)

    assert memory.water == set()


def test_spatial_resource_memory_avoids_visible_obstacles_and_unavailable_actions():
    memory = SpatialResourceMemory(water={(0, 1)})
    obs = {
        "grid": [
            [EMPTY, WATER, EMPTY],
            [EMPTY, AGENT, EMPTY],
            [EMPTY, OBSTACLE, EMPTY],
        ],
        "agent_pos": (1, 1),
        "standing_entity": EMPTY,
        "hydration": 0.2,
        "energy": 0.9,
    }

    action = memory.choose_action(obs, [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT])

    assert action in {MOVE_LEFT, MOVE_RIGHT}
