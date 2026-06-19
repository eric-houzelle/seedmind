"""Tests for resource regrowth location (anti-camping).

Default: resources regrow in place. With resource_regrow_elsewhere=True they
reappear at a new random cell — defeating the camp-and-oscillate strategy.
"""
from __future__ import annotations

from seedmind.envs.micro_fouloide_world import EMPTY, WATER, MicroFouloideWorld


def _env(elsewhere: bool) -> MicroFouloideWorld:
    env = MicroFouloideWorld(
        size=12, max_steps=0, resource_regrow_steps=5,
        resource_regrow_elsewhere=elsewhere, num_water=4, num_food=4,
        num_obstacles=4, num_dangers=0, seed=0,
    )
    env.reset()
    return env


def _an_empty_cell(env):
    for r in range(1, env.size - 1):
        for c in range(1, env.size - 1):
            if env.grid[r, c] == EMPTY and (r, c) != env.agent_pos:
                return r, c
    raise AssertionError("no empty cell")


def test_regrow_in_place_by_default():
    env = _env(elsewhere=False)
    r, c = _an_empty_cell(env)
    env._queue_regrowth(r, c, WATER)
    env.steps += env.resource_regrow_steps + 1
    env._tick_regrowth()
    assert env.grid[r, c] == WATER  # reappears in place


def test_regrow_elsewhere_when_enabled():
    env = _env(elsewhere=True)
    r, c = _an_empty_cell(env)
    before = int((env.grid == WATER).sum())
    env._queue_regrowth(r, c, WATER)
    env.steps += env.resource_regrow_steps + 1
    env._tick_regrowth()
    assert env.grid[r, c] != WATER                    # NOT in place
    assert int((env.grid == WATER).sum()) == before + 1  # reappeared somewhere


def test_regrow_elsewhere_not_lost_when_no_space():
    # If the world is full, the regrowth is retried later (not dropped).
    env = _env(elsewhere=True)
    env.grid[1:-1, 1:-1] = WATER  # fill interior — no empty cell
    env._queue_regrowth(2, 2, WATER)
    env.steps += env.resource_regrow_steps + 1
    env._tick_regrowth()
    assert len(env._regrow_queue) == 1  # kept for later
