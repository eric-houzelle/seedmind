"""Tests for recurrent world-model BPTT training (RSSM stage 2, brick 4b)."""
from __future__ import annotations

import numpy as np
import torch

from seedmind.agent.world_model import RecurrentWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.recurrent import train_recurrent_world_model

LATENT, NACT, DETER, NFEAT, NEV = 8, 4, 16, 3, 3


def _wm(seed=0):
    torch.manual_seed(seed)
    return RecurrentWorldModel(
        latent_dim=LATENT, num_actions=NACT, deter_dim=DETER,
        causal_feature_dim=NFEAT, num_events=NEV,
    )


def _fill_learnable(buf, n=80, episode="life_0001", seed=0):
    """Contiguous episode with an identity next-latent target (learnable)."""
    rng = np.random.default_rng(seed)
    for s in range(1, n + 1):
        z = rng.standard_normal(LATENT).astype(np.float32)
        cf = rng.random(NFEAT).astype(np.float32)
        e = make_experience(
            episode_id=episode, world_id="w", step=s, observation=None,
            action="MOVE_UP", next_observation=None,
            reward_external=float(z.sum()), reward_intrinsic=0.0, goal="g",
            prediction_error=0.0, done=(s == n),
            latent_state=z, next_latent_state=z.copy(),  # identity target
            action_index=int(rng.integers(0, NACT)),
            causal_features=cf, next_causal_features=cf.copy(),
            event_index=int(rng.integers(0, NEV)),
        )
        buf.add(e)


class TestTrainRecurrentWorldModel:
    def test_empty_buffer_returns_zero(self):
        wm = _wm()
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        out = train_recurrent_world_model(wm, ExperienceBuffer(seed=0), opt)
        assert out["updates"] == 0.0 and out["total"] == 0.0

    def test_runs_and_returns_finite_losses(self):
        wm = _wm()
        buf = ExperienceBuffer(seed=0)
        _fill_learnable(buf)
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        out = train_recurrent_world_model(
            wm, buf, opt, batch_size=4, seq_len=8, num_updates=3,
            causal_feature_weight=1.0, causal_event_weight=0.25,
            event_class_balance=True,
        )
        assert out["updates"] == 3.0
        for k in ("total", "state", "reward", "feature", "event"):
            assert np.isfinite(out[k])

    def test_gradients_flow_through_gru(self):
        wm = _wm()
        buf = ExperienceBuffer(seed=1)
        _fill_learnable(buf)
        before = wm.gru.weight_ih.detach().clone()
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        train_recurrent_world_model(wm, buf, opt, batch_size=4, seq_len=8, num_updates=5)
        assert not torch.allclose(before, wm.gru.weight_ih)  # GRU was updated

    def test_state_loss_decreases_on_learnable_target(self):
        wm = _wm(seed=3)
        buf = ExperienceBuffer(seed=3)
        _fill_learnable(buf, n=80, seed=3)
        opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
        first = train_recurrent_world_model(wm, buf, opt, batch_size=8, seq_len=8, num_updates=1)
        for _ in range(60):
            train_recurrent_world_model(wm, buf, opt, batch_size=8, seq_len=8, num_updates=1)
        last = train_recurrent_world_model(wm, buf, opt, batch_size=8, seq_len=8, num_updates=1)
        assert last["state"] < first["state"] * 0.6  # the WM learned the dynamics
