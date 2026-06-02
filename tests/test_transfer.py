import numpy as np

from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import COLOR_DOOR_CLOSED, COLOR_KEY


def _active_color(obs):
    return obs["active_color"]


def test_training_colors_held_out_at_eval():
    """An agent's training and eval color sets can be disjoint (transfer setup)."""
    train_env = ColoredGridWorld(size=8, allowed_colors=["red", "green"], seed=0)
    eval_env = ColoredGridWorld(size=8, allowed_colors=["blue"], seed=0)

    train_colors = {_active_color(train_env.reset()) for _ in range(30)}
    eval_colors = {_active_color(eval_env.reset()) for _ in range(30)}

    assert train_colors <= {"red", "green"}
    assert eval_colors == {"blue"}
    assert train_colors.isdisjoint(eval_colors)


def test_held_out_color_entities_present_in_eval():
    eval_env = ColoredGridWorld(size=8, allowed_colors=["blue"],
                                num_distractor_doors=0, num_distractor_keys=0, seed=2)
    obs = eval_env.reset()
    grid = obs["grid"]
    assert (grid == COLOR_KEY["blue"]).sum() >= 1
    assert (grid == COLOR_DOOR_CLOSED["blue"]).sum() >= 1


def test_rule_is_color_agnostic_mechanic():
    """The open mechanic is identical across colors (what makes transfer possible)."""
    for color in ["red", "blue", "green"]:
        env = ColoredGridWorld(size=5, allowed_colors=[color], seed=1)
        env.reset()
        grid = np.full((5, 5), 0, dtype=np.int64)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1  # walls
        grid[1, 2] = COLOR_DOOR_CLOSED[color]
        env.grid = grid
        env.agent_pos = (1, 1)
        env.has_key = True
        env.key_color = color
        env.success = False
        _, reward, _, info = env.step("INTERACT")
        assert info["success"] is True and reward > 0
