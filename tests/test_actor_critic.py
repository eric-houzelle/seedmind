"""Tests for the imagination-policy Actor (RSSM stage 2, brick 5a)."""
from __future__ import annotations

import numpy as np
import torch

from seedmind.agent.actor_critic import Actor
from seedmind.agent.value_model import ValueModel

DETER, NACT = 16, 7


class TestActor:
    def test_logits_shape(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        logits = actor(torch.randn(4, DETER))
        assert logits.shape == (4, NACT)

    def test_distribution_is_valid(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        dist = actor.distribution(torch.randn(3, DETER))
        probs = dist.probs
        assert probs.shape == (3, NACT)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_act_returns_valid_indices(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        a = actor.act(torch.randn(5, DETER))
        assert a.shape == (5,)
        assert int(a.min()) >= 0 and int(a.max()) < NACT

    def test_greedy_is_argmax(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        s = torch.randn(5, DETER)
        assert torch.equal(actor.act(s, greedy=True), actor(s).argmax(dim=-1))

    def test_act_one_numpy(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        idx = actor.act_one(np.zeros(DETER, dtype=np.float32), greedy=True)
        assert isinstance(idx, int) and 0 <= idx < NACT

    def test_evaluate_grad_flows(self):
        actor = Actor(input_dim=DETER, num_actions=NACT)
        s = torch.randn(8, DETER)
        actions = torch.randint(0, NACT, (8,))
        log_prob, entropy = actor.evaluate(s, actions)
        assert log_prob.shape == (8,) and entropy.shape == (8,)
        # A REINFORCE-style surrogate must backprop into the actor.
        (-(log_prob * 1.0).mean() - 0.01 * entropy.mean()).backward()
        assert actor.net[0].weight.grad is not None
        assert actor.net[0].weight.grad.abs().sum() > 0


class TestCriticReuse:
    def test_value_model_works_on_recurrent_state(self):
        critic = ValueModel(latent_dim=DETER)
        v = critic(torch.randn(6, DETER))
        assert v.shape == (6,)
        assert torch.isfinite(v).all()
