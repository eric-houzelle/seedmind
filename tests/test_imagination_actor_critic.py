"""Tests for actor-critic training in imagination (RSSM stage 2, bricks 5b/5c)."""
from __future__ import annotations

import numpy as np
import torch

from seedmind.agent.actor_critic import Actor
from seedmind.agent.value_model import ValueModel
from seedmind.agent.world_model import RecurrentWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.imagination_actor_critic import (
    _lambda_returns,
    train_imagination_actor_critic,
)

LATENT, NACT, DETER = 8, 4, 16


def _fill(buf, n=60, episode="life_0001", seed=0):
    rng = np.random.default_rng(seed)
    for s in range(1, n + 1):
        buf.add(make_experience(
            episode_id=episode, world_id="w", step=s, observation=None,
            action="MOVE_UP", next_observation=None, reward_external=0.0,
            reward_intrinsic=0.0, goal="g", prediction_error=0.0, done=(s == n),
            latent_state=rng.standard_normal(LATENT).astype(np.float32),
            next_latent_state=rng.standard_normal(LATENT).astype(np.float32),
            action_index=int(rng.integers(0, NACT)),
        ))


class _MockWM:
    """Static recurrent state; reward = 1 iff the rewarded action is taken."""

    def __init__(self, deter=DETER, num_actions=NACT, rewarded=0):
        self.deter_dim = deter
        self.num_actions = num_actions
        self.rewarded = rewarded

    def initial_state(self, B, device=None):
        return torch.zeros(B, self.deter_dim, device=device)

    def observe_step(self, z, prev_a, h):
        return h

    @torch.no_grad()
    def imagine_batch(self, h, a):
        r = (a == self.rewarded).float()
        return h, h, r, torch.zeros_like(r)


class TestLambdaReturns:
    def test_undiscounted_full_trace(self):
        # gamma=1, lam=1, V=0, bootstrap=0 -> R_t = sum of remaining rewards.
        rewards = torch.ones(3, 2)
        values = torch.zeros(3, 2)
        boot = torch.zeros(2)
        R = _lambda_returns(rewards, values, boot, gamma=1.0, lam=1.0)
        assert torch.allclose(R[:, 0], torch.tensor([3.0, 2.0, 1.0]))


class TestTrainImaginationActorCritic:
    def test_empty_buffer(self):
        actor = Actor(DETER, NACT)
        critic = ValueModel(DETER)
        ao = torch.optim.Adam(actor.parameters(), lr=1e-3)
        co = torch.optim.Adam(critic.parameters(), lr=1e-3)
        out = train_imagination_actor_critic(actor, critic, _MockWM(), ExperienceBuffer(seed=0), ao, co)
        assert out["updates"] == 0.0

    def test_runs_on_real_wm(self):
        torch.manual_seed(0)
        wm = RecurrentWorldModel(latent_dim=LATENT, num_actions=NACT, deter_dim=DETER)
        actor = Actor(DETER, NACT)
        critic = ValueModel(DETER)
        buf = ExperienceBuffer(seed=0)
        _fill(buf)
        ao = torch.optim.Adam(actor.parameters(), lr=1e-3)
        co = torch.optim.Adam(critic.parameters(), lr=1e-3)
        out = train_imagination_actor_critic(
            actor, critic, wm, buf, ao, co,
            batch_size=8, context_len=4, horizon=6, num_updates=3,
        )
        assert out["updates"] == 3.0
        assert np.isfinite(out["actor_loss"]) and np.isfinite(out["critic_loss"])

    def test_params_change(self):
        wm = RecurrentWorldModel(latent_dim=LATENT, num_actions=NACT, deter_dim=DETER)
        actor = Actor(DETER, NACT)
        critic = ValueModel(DETER)
        buf = ExperienceBuffer(seed=1)
        _fill(buf)
        a_before = actor.net[0].weight.detach().clone()
        c_before = critic.net[0].weight.detach().clone()
        ao = torch.optim.Adam(actor.parameters(), lr=1e-3)
        co = torch.optim.Adam(critic.parameters(), lr=1e-3)
        train_imagination_actor_critic(actor, critic, wm, buf, ao, co,
                                       batch_size=8, context_len=4, horizon=6, num_updates=5)
        assert not torch.allclose(a_before, actor.net[0].weight)
        assert not torch.allclose(c_before, critic.net[0].weight)

    def test_learns_rewarded_action_in_imagination(self):
        """The decisive test: the actor learns to prefer the rewarded action."""
        torch.manual_seed(0)
        wm = _MockWM(rewarded=0)
        actor = Actor(DETER, NACT)
        critic = ValueModel(DETER)
        buf = ExperienceBuffer(seed=0)
        _fill(buf)
        ao = torch.optim.Adam(actor.parameters(), lr=3e-3)
        co = torch.optim.Adam(critic.parameters(), lr=3e-3)

        zeros = torch.zeros(1, DETER)
        with torch.no_grad():
            p0_before = float(actor.distribution(zeros).probs[0, 0])
        for _ in range(150):
            train_imagination_actor_critic(actor, critic, wm, buf, ao, co,
                                           batch_size=16, context_len=4, horizon=8,
                                           num_updates=1, entropy_coef=0.0)
        with torch.no_grad():
            p0_after = float(actor.distribution(zeros).probs[0, 0])
        assert p0_after > p0_before + 0.2  # policy shifted toward the rewarded action
        assert p0_after > 0.5
