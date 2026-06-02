import numpy as np

from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import (
    COLOR_DOOR_CLOSED,
    COLOR_DOOR_OPEN,
    COLOR_KEY,
    COLORS,
    EMPTY,
    INTERACT,
    MOVE_RIGHT,
    WALL,
)


def _world(allowed_colors=None):
    return ColoredGridWorld(size=5, max_steps=30, allowed_colors=allowed_colors, seed=0)


def _setup_door_scenario(env, key_color, door_color):
    """Place the agent at (1,1) with a closed `door_color` door to its right."""
    grid = np.full((5, 5), EMPTY, dtype=np.int64)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    grid[1, 2] = COLOR_DOOR_CLOSED[door_color]
    env.grid = grid
    env.agent_pos = (1, 1)
    env.has_key = key_color is not None
    env.key_color = key_color
    env.door_open = False
    env.success = False
    env.steps = 0


def test_matching_color_opens_door():
    env = _world(["red"])
    env.reset()
    _setup_door_scenario(env, key_color="red", door_color="red")
    _, reward, done, info = env.step(INTERACT)
    assert info["door_open"] == 1
    assert info["success"] is True
    assert reward > 0
    assert env.grid[1, 2] == COLOR_DOOR_OPEN["red"]


def test_wrong_color_does_not_open():
    env = _world(["red"])
    env.reset()
    _setup_door_scenario(env, key_color="blue", door_color="red")
    _, reward, done, info = env.step(INTERACT)
    assert info["door_open"] == 0
    assert info["success"] is False
    assert env.grid[1, 2] == COLOR_DOOR_CLOSED["red"]


def test_no_key_does_not_open():
    env = _world(["red"])
    env.reset()
    _setup_door_scenario(env, key_color=None, door_color="red")
    _, _, _, info = env.step(INTERACT)
    assert info["door_open"] == 0


def test_closed_colored_door_blocks_movement():
    env = _world(["red"])
    env.reset()
    _setup_door_scenario(env, key_color=None, door_color="red")
    env.step(MOVE_RIGHT)  # into the closed door
    assert env.agent_pos == (1, 1)


def test_picking_colored_key_sets_key_color():
    env = _world(["blue"])
    env.reset()
    grid = np.full((5, 5), EMPTY, dtype=np.int64)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    grid[1, 2] = COLOR_KEY["blue"]
    env.grid = grid
    env.agent_pos = (1, 1)
    env.has_key = False
    env.key_color = None
    env.step(MOVE_RIGHT)
    assert env.has_key is True
    assert env.key_color == "blue"


def test_active_color_respects_allowed_colors():
    env = _world(["green"])
    for _ in range(10):
        obs = env.reset()
        assert obs["active_color"] == "green"


def test_color_varies_between_episodes():
    env = _world(list(COLORS))
    seen = set()
    for _ in range(40):
        obs = env.reset()
        seen.add(obs["active_color"])
    assert len(seen) >= 2
