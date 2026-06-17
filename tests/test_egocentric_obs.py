"""Tests for egocentric perception (RSSM trajectory, stage 1).

The egocentric crop must: produce a fixed window independent of world size,
centre the agent, fill out-of-world cells with the sentinel, preserve local
content exactly, and compose with the existing onehot/property obs encoders
and the (already convolutional) Q-network.
"""
from __future__ import annotations

import numpy as np
import torch

from seedmind.agent.encoder import ConvEncoder
from seedmind.agent.micro_fouloide_encoder import (
    egocentric_grid,
    egocentric_observation,
    make_micro_fouloide_obs_fns,
    wrap_egocentric,
)
from seedmind.agent.q_network import QNetwork
from seedmind.envs.micro_fouloide_world import (
    AGENT,
    NUM_ENTITIES,
    OBSTACLE,
    WATER,
)


def _grid(rows):
    return np.asarray(rows, dtype=np.int64)


class TestEgocentricGrid:
    def test_shape_is_window(self):
        grid = np.zeros((10, 10), dtype=np.int64)
        out = egocentric_grid(grid, (5, 5), radius=3)
        assert out.shape == (7, 7)

    def test_agent_at_centre(self):
        grid = np.zeros((9, 9), dtype=np.int64)
        grid[4, 6] = AGENT
        out = egocentric_grid(grid, (4, 6), radius=2)
        assert out[2, 2] == AGENT

    def test_content_preserved_around_agent(self):
        grid = np.zeros((9, 9), dtype=np.int64)
        grid[4, 4] = AGENT
        grid[4, 5] = WATER  # one cell to the agent's right
        out = egocentric_grid(grid, (4, 4), radius=2)
        # Centre is the agent; the cell to its right holds the water.
        assert out[2, 2] == AGENT
        assert out[2, 3] == WATER

    def test_out_of_world_filled_with_sentinel(self):
        grid = np.full((6, 6), WATER, dtype=np.int64)  # all water, no edges
        out = egocentric_grid(grid, (0, 0), radius=2, oob_fill=OBSTACLE)
        # Agent in the corner: the top and left bands fall outside the world.
        assert (out[0, :] == OBSTACLE).all()
        assert (out[:, 0] == OBSTACLE).all()
        # Cells inside the world keep their content.
        assert out[2, 2] == WATER  # centre = world (0,0)
        assert out[3, 3] == WATER

    def test_custom_oob_fill(self):
        grid = np.zeros((4, 4), dtype=np.int64)
        out = egocentric_grid(grid, (0, 0), radius=1, oob_fill=WATER)
        assert out[0, 0] == WATER  # out of world
        assert out[1, 1] == 0      # world (0,0)


class TestSizeInvariance:
    """The same local neighbourhood yields the same window in any world size."""

    def test_identical_window_across_world_sizes(self):
        radius = 3
        # Build the same 7x7 local pattern around the agent in a 16- and a
        # 32-world; both should produce a byte-identical egocentric window.
        def world(size, ar, ac):
            g = np.zeros((size, size), dtype=np.int64)
            g[ar, ac] = AGENT
            g[ar - 1, ac] = WATER
            g[ar + 2, ac - 1] = OBSTACLE
            return g

        small = egocentric_grid(world(16, 8, 8), (8, 8), radius)
        big = egocentric_grid(world(32, 20, 11), (20, 11), radius)
        assert np.array_equal(small, big)


class TestWrapEgocentric:
    def test_channels_shape_is_window(self):
        obs_to_vec, obs_batch, n_ch, n_sc = make_micro_fouloide_obs_fns(NUM_ENTITIES)
        radius = 5
        ego_to_vec, ego_batch = wrap_egocentric(obs_to_vec, obs_batch, radius)
        win = 2 * radius + 1

        grid = np.zeros((32, 32), dtype=np.int64)
        grid[10, 10] = AGENT
        obs = {"grid": grid, "agent_pos": (10, 10), "standing_entity": 0,
               "energy": 0.5, "hydration": 0.5, "temperature": 0.5, "health": 1.0}

        channels, scalars = ego_batch([obs])
        assert channels.shape == (1, n_ch, win, win)
        assert scalars.shape == (1, n_sc)

    def test_vector_length_matches_window(self):
        obs_to_vec, obs_batch, n_ch, n_sc = make_micro_fouloide_obs_fns(NUM_ENTITIES)
        radius = 4
        ego_to_vec, _ = wrap_egocentric(obs_to_vec, obs_batch, radius)
        win = 2 * radius + 1

        grid = np.zeros((20, 20), dtype=np.int64)
        grid[3, 3] = AGENT
        obs = {"grid": grid, "agent_pos": (3, 3), "standing_entity": 0,
               "energy": 0.5, "hydration": 0.5, "temperature": 0.5, "health": 1.0}

        vec = ego_to_vec(obs)
        assert vec.shape == (win * win * n_ch + n_sc,)

    def test_does_not_mutate_original_observation(self):
        obs_to_vec, obs_batch, _, _ = make_micro_fouloide_obs_fns(NUM_ENTITIES)
        ego_to_vec, _ = wrap_egocentric(obs_to_vec, obs_batch, radius=3)
        grid = np.zeros((12, 12), dtype=np.int64)
        grid[5, 5] = AGENT
        obs = {"grid": grid, "agent_pos": (5, 5), "standing_entity": 0,
               "energy": 0.5, "hydration": 0.5, "temperature": 0.5, "health": 1.0}
        ego_to_vec(obs)
        assert obs["grid"].shape == (12, 12)
        assert tuple(obs["agent_pos"]) == (5, 5)


class TestQNetworkOnWindow:
    """The existing CNN Q-net runs on the fixed window with grid_size=window."""

    def test_qnet_forward_on_egocentric_window(self):
        torch.manual_seed(0)
        obs_to_vec, obs_batch, n_ch, n_sc = make_micro_fouloide_obs_fns(NUM_ENTITIES)
        radius = 5
        win = 2 * radius + 1
        _, ego_batch = wrap_egocentric(obs_to_vec, obs_batch, radius)

        grid = np.zeros((32, 32), dtype=np.int64)
        grid[0, 0] = AGENT  # corner: forces out-of-world padding
        obs = {"grid": grid, "agent_pos": (0, 0), "standing_entity": 0,
               "energy": 0.5, "hydration": 0.5, "temperature": 0.5, "health": 1.0}

        qnet = QNetwork(
            grid_size=win, num_actions=7,
            num_grid_channels=n_ch, num_scalars=n_sc, obs_batch_fn=ego_batch,
        )
        ch, sc = ego_batch([obs])
        q = qnet(ch, sc)
        assert q.shape == (1, 7)
        assert torch.isfinite(q).all()


def _obs(grid, agent_pos, **drives):
    base = {"grid": grid, "agent_pos": agent_pos, "standing_entity": 0,
            "energy": 0.5, "hydration": 0.5, "temperature": 0.5, "health": 1.0}
    base.update(drives)
    return base


def _conv_encoder(radius=5, latent_dim=64, seed=0):
    obs_to_vec, obs_batch, n_ch, n_sc = make_micro_fouloide_obs_fns(NUM_ENTITIES)
    _, ego_batch = wrap_egocentric(obs_to_vec, obs_batch, radius)
    enc = ConvEncoder(
        num_channels=n_ch, num_scalars=n_sc, window_size=2 * radius + 1,
        latent_dim=latent_dim, seed=seed, obs_batch_fn=ego_batch,
    )
    return enc, latent_dim


class TestConvEncoder:
    def test_encode_shape(self):
        enc, latent_dim = _conv_encoder()
        grid = np.zeros((32, 32), dtype=np.int64)
        grid[10, 10] = AGENT
        latent = enc.encode(_obs(grid, (10, 10)))
        assert latent.shape == (latent_dim,)
        assert np.isfinite(latent).all()

    def test_encode_batch_shape(self):
        enc, latent_dim = _conv_encoder()
        grid = np.zeros((32, 32), dtype=np.int64)
        grid[10, 10] = AGENT
        out = enc.encode_batch([_obs(grid, (10, 10)), _obs(grid, (10, 10))])
        assert out.shape == (2, latent_dim)

    def test_frozen(self):
        enc, _ = _conv_encoder()
        assert all(not p.requires_grad for p in enc.parameters())

    def test_latent_size_invariant_to_world_size(self):
        """Same local neighbourhood -> same latent, regardless of world size."""
        enc, _ = _conv_encoder(seed=7)

        def world(size, ar, ac):
            g = np.zeros((size, size), dtype=np.int64)
            g[ar, ac] = AGENT
            g[ar - 2, ac + 1] = WATER
            g[ar + 3, ac] = OBSTACLE
            return g

        small = enc.encode(_obs(world(16, 8, 8), (8, 8)))
        big = enc.encode(_obs(world(48, 30, 20), (30, 20)))
        assert np.allclose(small, big, atol=1e-6)
