"""Tests for the recurrent-aware QNetwork (RSSM trajectory, stage 2 brick 2).

The Q-network must optionally consume a recurrent memory state h_t and let it
influence the action values, while staying backward compatible (recurrent_dim=0).
"""
from __future__ import annotations

import torch

from seedmind.agent.q_network import QNetwork

GRID, NCH, NSC, NACT = 5, 3, 4, 7


def _inputs(batch=1, seed=0):
    torch.manual_seed(seed)
    channels = torch.rand(batch, NCH, GRID, GRID)
    scalars = torch.rand(batch, NSC)
    return channels, scalars


def _qnet(recurrent_dim=0):
    torch.manual_seed(0)
    return QNetwork(
        grid_size=GRID, num_actions=NACT,
        num_grid_channels=NCH, num_scalars=NSC, recurrent_dim=recurrent_dim,
    )


class TestBackwardCompatible:
    def test_memoryless_forward_unchanged(self):
        qnet = _qnet(recurrent_dim=0)
        ch, sc = _inputs()
        out = qnet(ch, sc)
        assert out.shape == (1, NACT)
        assert torch.isfinite(out).all()

    def test_recurrent_dim_zero_ignores_recurrent_arg(self):
        qnet = _qnet(recurrent_dim=0)
        ch, sc = _inputs()
        # Passing a recurrent vector when dim=0 is simply ignored.
        out = qnet(ch, sc, recurrent=torch.rand(1, 16))
        assert out.shape == (1, NACT)


class TestRecurrentInput:
    def test_forward_with_recurrent(self):
        rdim = 8
        qnet = _qnet(recurrent_dim=rdim)
        ch, sc = _inputs(batch=2)
        h = torch.rand(2, rdim)
        out = qnet(ch, sc, h)
        assert out.shape == (2, NACT)
        assert torch.isfinite(out).all()

    def test_missing_recurrent_falls_back_to_zeros(self):
        qnet = _qnet(recurrent_dim=8)
        ch, sc = _inputs()
        out = qnet(ch, sc)  # no recurrent provided
        assert out.shape == (1, NACT)
        assert torch.isfinite(out).all()

    def test_memory_influences_q_values(self):
        """Different h_t must change the action values (else memory is inert)."""
        rdim = 8
        qnet = _qnet(recurrent_dim=rdim)
        ch, sc = _inputs()
        out_a = qnet(ch, sc, torch.zeros(1, rdim))
        out_b = qnet(ch, sc, torch.ones(1, rdim) * 3.0)
        assert not torch.allclose(out_a, out_b, atol=1e-4)


class TestScorerThreadsRecurrent:
    def test_q_values_and_scorer_accept_recurrent(self):
        import numpy as np
        rdim = 8
        qnet = _qnet(recurrent_dim=rdim)
        obs = {"grid": np.zeros((GRID, GRID), dtype=np.int64)}

        # A custom obs_batch_fn for this dummy obs.
        def batch_fn(observations):
            n = len(observations)
            return torch.rand(n, NCH, GRID, GRID), torch.rand(n, NSC)

        qnet._obs_batch_fn = batch_fn
        actions = [f"a{i}" for i in range(NACT)]
        h0 = np.zeros(rdim, dtype=np.float32)
        h1 = np.ones(rdim, dtype=np.float32)
        v0 = qnet.q_values(obs, recurrent=h0)
        v1 = qnet.q_values(obs, recurrent=h1)
        assert v0.shape == (NACT,)
        # make_scorer must accept and use recurrent without error.
        scorer = qnet.make_scorer(obs, actions, recurrent=h1)
        assert isinstance(scorer(actions[0]), float)
