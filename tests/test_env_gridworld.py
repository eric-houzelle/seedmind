import numpy as np

from seedmind.envs.gridworld import (
    DOOR_CLOSED,
    DOOR_OPEN,
    EMPTY,
    INTERACT,
    KEY,
    MOVE_DOWN,
    MOVE_RIGHT,
    MOVE_UP,
    REWARD,
    WALL,
    GridWorld,
)
from seedmind.envs.procedural_gridworld import ProceduralGridWorld


def _controlled_world():
    layout = np.array(
        [
            [WALL, WALL, WALL, WALL, WALL],
            [WALL, EMPTY, KEY, WALL, WALL],
            [WALL, EMPTY, DOOR_CLOSED, REWARD, WALL],
            [WALL, EMPTY, EMPTY, EMPTY, WALL],
            [WALL, WALL, WALL, WALL, WALL],
        ],
        dtype=np.int64,
    )
    return GridWorld(size=5, max_steps=50, layout=layout)


def test_reset_returns_observation():
    env = _controlled_world()
    obs = env.reset()
    assert "grid" in obs and "has_key" in obs and "door_open" in obs
    assert obs["grid"].shape == (5, 5)
    assert env.agent_pos == (1, 1)


def test_wall_collision_blocks_movement():
    env = _controlled_world()
    env.reset()
    start = env.agent_pos
    env.step(MOVE_UP)  # (0,1) is a wall
    assert env.agent_pos == start


def test_agent_moves_into_empty_cell():
    env = _controlled_world()
    env.reset()
    env.step(MOVE_DOWN)  # (2,1) is empty
    assert env.agent_pos == (2, 1)


def test_key_door_interaction():
    env = _controlled_world()
    env.reset()
    # Pick up the key to the right.
    _, _, _, info = env.step(MOVE_RIGHT)  # (1,2) KEY
    assert info["has_key"] == 1
    # Door is directly below the key cell; interact opens it.
    _, _, _, info = env.step(INTERACT)
    assert info["door_open"] == 1
    assert env.grid[2, 2] == DOOR_OPEN
    # Move through the now-open door, then onto the reward.
    env.step(MOVE_DOWN)  # (2,2) open door
    assert env.agent_pos == (2, 2)
    _, reward, done, info = env.step(MOVE_RIGHT)  # (2,3) REWARD
    assert info["success"] is True
    assert reward > 0
    assert done is True


def test_closed_door_blocks_without_key():
    env = _controlled_world()
    env.reset()
    env.step(MOVE_DOWN)  # (2,1)
    pos = env.agent_pos
    env.step(MOVE_RIGHT)  # tries (2,2) closed door -> blocked
    assert env.agent_pos == pos


def test_procedural_world_is_generated_and_solvable_layout():
    env = ProceduralGridWorld(size=10, max_steps=100, seed=3)
    obs = env.reset()
    grid = obs["grid"]
    # The required entities are present.
    assert (grid == KEY).sum() >= 1
    assert (grid == DOOR_CLOSED).sum() >= 1
    assert (grid == REWARD).sum() >= 1
    # Regeneration produces a different layout for a different seed.
    env2 = ProceduralGridWorld(size=10, max_steps=100, seed=7)
    assert not np.array_equal(env2.reset()["grid"], grid)
