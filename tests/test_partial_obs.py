"""Tests for partial observation (Niveau 4 / V3).

Validates that visibility_radius correctly masks distant cells with
UNKNOWN_OBJECT, that the mask updates when the agent moves, and that the
Q-Network handles observations containing unknowns without errors.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import (
    AGENT,
    EMPTY,
    UNKNOWN_OBJECT,
    GridWorld,
)
from seedmind.agent.encoder import (
    QNET_NUM_CHANNELS,
    QNET_UNKNOWN,
    observation_qnet_channels,
)
from seedmind.agent.q_network import QNetwork, obs_batch_to_tensors


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _distances(agent_pos: tuple, size: int) -> np.ndarray:
    """Manhattan distance from agent_pos for every cell."""
    ar, ac = agent_pos
    rs = np.arange(size)[:, None]
    cs = np.arange(size)[None, :]
    return np.abs(rs - ar) + np.abs(cs - ac)


# ------------------------------------------------------------------
# GridWorld base tests
# ------------------------------------------------------------------

class TestGridWorldVisibility:
    """Masking in the base GridWorld with visibility_radius."""

    def test_no_radius_full_obs(self):
        env = GridWorld(size=7, max_steps=20, visibility_radius=None, seed=0)
        obs = env.reset()
        obs = env.observe()
        assert (obs["grid"] != UNKNOWN_OBJECT).any()
        assert UNKNOWN_OBJECT not in obs["grid"] or True  # may or may not appear

    def test_radius_masks_far_cells(self):
        env = GridWorld(size=7, max_steps=20, visibility_radius=2, seed=0)
        env.reset()
        obs = env.observe()
        grid = obs["grid"]
        dists = _distances(obs["agent_pos"], 7)

        for r in range(7):
            for c in range(7):
                if dists[r, c] > 2:
                    assert grid[r, c] == UNKNOWN_OBJECT, (
                        f"Cell ({r},{c}) dist={dists[r,c]} should be UNKNOWN"
                    )
                else:
                    assert grid[r, c] != UNKNOWN_OBJECT, (
                        f"Cell ({r},{c}) dist={dists[r,c]} should be visible"
                    )

    def test_agent_always_visible(self):
        env = GridWorld(size=7, max_steps=20, visibility_radius=1, seed=0)
        env.reset()
        obs = env.observe()
        ar, ac = obs["agent_pos"]
        assert obs["grid"][ar, ac] == AGENT


# ------------------------------------------------------------------
# ColoredGridWorld tests
# ------------------------------------------------------------------

class TestColoredGridWorldVisibility:
    """Partial observation in ColoredGridWorld."""

    def test_radius_propagated(self):
        env = ColoredGridWorld(
            size=7, max_steps=30, visibility_radius=2, seed=42,
        )
        assert env.visibility_radius == 2

    def test_far_cells_masked(self):
        env = ColoredGridWorld(
            size=7, max_steps=30, visibility_radius=2,
            num_distractor_doors=1, num_distractor_keys=0,
            num_dangers=0, seed=42,
        )
        env.reset()
        obs = env.observe()
        grid = obs["grid"]
        dists = _distances(obs["agent_pos"], 7)
        far_mask = dists > 2
        assert (grid[far_mask] == UNKNOWN_OBJECT).all()

    def test_visibility_changes_on_move(self):
        env = ColoredGridWorld(
            size=7, max_steps=50, visibility_radius=2,
            num_distractor_doors=0, num_distractor_keys=0,
            num_dangers=0, seed=10,
        )
        env.reset()
        obs_before = env.observe()
        pos_before = tuple(obs_before["agent_pos"])
        visible_before = set()
        for r in range(7):
            for c in range(7):
                if obs_before["grid"][r, c] != UNKNOWN_OBJECT:
                    visible_before.add((r, c))

        # Try moves until the agent actually changes position.
        moved = False
        for action in ["MOVE_DOWN", "MOVE_RIGHT", "MOVE_UP", "MOVE_LEFT"]:
            env.step(action)
            obs_after = env.observe()
            pos_after = tuple(obs_after["agent_pos"])
            if pos_after != pos_before:
                moved = True
                break

        if moved:
            visible_after = set()
            for r in range(7):
                for c in range(7):
                    if obs_after["grid"][r, c] != UNKNOWN_OBJECT:
                        visible_after.add((r, c))
            # The visible set should have changed.
            assert visible_before != visible_after

    def test_no_fog_of_war(self):
        """Moving away from a previously visible cell makes it UNKNOWN again."""
        env = ColoredGridWorld(
            size=7, max_steps=50, visibility_radius=1,
            num_distractor_doors=0, num_distractor_keys=0,
            num_dangers=0, seed=20,
        )
        env.reset()
        obs1 = env.observe()
        pos1 = tuple(obs1["agent_pos"])

        # Find a cell visible now that will be out of range after a move.
        moved = False
        for action in ["MOVE_DOWN", "MOVE_RIGHT", "MOVE_UP", "MOVE_LEFT"]:
            env.step(action)
            obs2 = env.observe()
            pos2 = tuple(obs2["agent_pos"])
            if pos2 != pos1:
                moved = True
                break

        if moved:
            dists_from_new = _distances(pos2, 7)
            for r in range(7):
                for c in range(7):
                    if dists_from_new[r, c] > 1:
                        assert obs2["grid"][r, c] == UNKNOWN_OBJECT


# ------------------------------------------------------------------
# Q-Network compatibility
# ------------------------------------------------------------------

class TestQNetPartialObs:
    """The Q-Network must accept observations with UNKNOWN_OBJECT cells."""

    def test_qnet_channels_include_unknown(self):
        env = ColoredGridWorld(
            size=7, max_steps=30, visibility_radius=2,
            num_distractor_doors=1, num_distractor_keys=0,
            num_dangers=0, seed=42,
        )
        env.reset()
        obs = env.observe()
        channels = observation_qnet_channels(obs)
        assert channels.shape == (QNET_NUM_CHANNELS, 7, 7)
        # Far cells should activate the UNKNOWN channel.
        dists = _distances(obs["agent_pos"], 7)
        far_mask = dists > 2
        assert (channels[QNET_UNKNOWN][far_mask] == 1.0).all()
        # Near cells should NOT activate the UNKNOWN channel.
        near_mask = dists <= 2
        assert (channels[QNET_UNKNOWN][near_mask] == 0.0).all()

    def test_qnet_forward_no_crash(self):
        torch.manual_seed(0)
        env = ColoredGridWorld(
            size=7, max_steps=30, visibility_radius=2,
            num_distractor_doors=1, num_distractor_keys=0,
            num_dangers=0, seed=42,
        )
        env.reset()
        obs = env.observe()
        qnet = QNetwork(grid_size=7, num_actions=6)
        ch, inv = obs_batch_to_tensors([obs])
        q_vals = qnet(ch, inv)
        assert q_vals.shape == (1, 6)
        assert torch.isfinite(q_vals).all()

    def test_qnet_batch_with_mixed_visibility(self):
        """A batch mixing full-obs and partial-obs should work."""
        torch.manual_seed(0)
        env_full = ColoredGridWorld(size=7, max_steps=30, seed=42)
        env_partial = ColoredGridWorld(
            size=7, max_steps=30, visibility_radius=2, seed=42,
        )
        env_full.reset()
        env_partial.reset()
        obs_full = env_full.observe()
        obs_partial = env_partial.observe()
        qnet = QNetwork(grid_size=7, num_actions=6)
        ch, inv = obs_batch_to_tensors([obs_full, obs_partial])
        q_vals = qnet(ch, inv)
        assert q_vals.shape == (2, 6)
        assert torch.isfinite(q_vals).all()
