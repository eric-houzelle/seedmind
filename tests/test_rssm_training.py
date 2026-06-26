"""Tests de l'entraînement RSSM world-model (recon + KL + reward), phase 1 brique 3."""
from __future__ import annotations

import numpy as np
import torch

from seedmind.agent.world_model import RSSMWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.recurrent import train_rssm_world_model

EMBED, NACT, NFEAT, NEV = 8, 4, 3, 3


def _wm(seed=0):
    torch.manual_seed(seed)
    return RSSMWorldModel(embed_dim=EMBED, num_actions=NACT, stoch=8, discrete=8,
                          deter=16, hidden=32, causal_feature_dim=NFEAT, num_events=NEV)


def _fill(buf, n=80, seed=0):
    rng = np.random.default_rng(seed)
    # Structured embeds: a few prototypes (+ tiny noise) so the categorical latent
    # can actually compress & reconstruct them (random noise is unlearnable by design).
    protos = rng.standard_normal((4, EMBED)).astype(np.float32)
    for s in range(1, n + 1):
        z = (protos[rng.integers(0, len(protos))] + 0.05 * rng.standard_normal(EMBED)).astype(np.float32)
        cf = rng.random(NFEAT).astype(np.float32)
        buf.add(make_experience(
            episode_id="life_0001", world_id="w", step=s, observation=None,
            action="MOVE_UP", next_observation=None,
            reward_external=float(z.sum()), reward_intrinsic=0.0, goal="g",
            prediction_error=0.0, done=(s == n),
            latent_state=z, next_latent_state=z.copy(),
            action_index=int(rng.integers(0, NACT)),
            causal_features=cf, next_causal_features=cf.copy(),
            event_index=int(rng.integers(0, NEV)),
        ))


def test_empty_buffer_returns_zero():
    wm = _wm()
    opt = torch.optim.Adam(wm.parameters(), 1e-3)
    out = train_rssm_world_model(wm, ExperienceBuffer(seed=0), opt)
    assert out["updates"] == 0.0 and out["total"] == 0.0


def test_runs_finite_losses():
    wm = _wm()
    buf = ExperienceBuffer(seed=0); _fill(buf)
    opt = torch.optim.Adam(wm.parameters(), 1e-3)
    out = train_rssm_world_model(wm, buf, opt, batch_size=4, seq_len=8, num_updates=2,
                                 causal_feature_weight=1.0, causal_event_weight=0.25)
    for k in ("total", "recon", "reward", "kl", "feature", "event"):
        assert np.isfinite(out[k]), k
    # weighted free-bits floor: dyn_scale*free + rep_scale*free = 0.5 + 0.1 = 0.6
    assert out["kl"] >= 0.6 - 1e-2


def test_recon_and_reward_decrease():
    wm = _wm()
    buf = ExperienceBuffer(seed=0); _fill(buf)
    opt = torch.optim.Adam(wm.parameters(), 3e-3)
    first = train_rssm_world_model(wm, buf, opt, batch_size=8, seq_len=10, num_updates=1)
    for _ in range(60):
        last = train_rssm_world_model(wm, buf, opt, batch_size=8, seq_len=10, num_updates=1)
    assert last["recon"] < first["recon"] * 0.8    # the WM learns to reconstruct the embed
    assert last["reward"] < first["reward"] * 0.8  # and to predict the reward
