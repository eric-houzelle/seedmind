"""Tests for the deterministic RecurrentWorldModel (RSSM trajectory, stage 2).

Validates the recurrent state shapes, the *memory* property (h depends on the
history, not just the current latent), imagination rollouts, and that gradients
flow through the GRU (needed for sequence/BPTT training).
"""
from __future__ import annotations

import torch

from seedmind.agent.world_model import RecurrentWorldModel


def _wm(latent_dim=16, num_actions=7, deter_dim=32, **kw):
    torch.manual_seed(0)
    return RecurrentWorldModel(
        latent_dim=latent_dim, num_actions=num_actions, deter_dim=deter_dim, **kw
    )


class TestShapes:
    def test_initial_state(self):
        wm = _wm()
        h = wm.initial_state(batch_size=4)
        assert h.shape == (4, wm.deter_dim)
        assert torch.count_nonzero(h) == 0

    def test_observe_step_shape(self):
        wm = _wm()
        h0 = wm.initial_state(2)
        z = torch.randn(2, wm.latent_dim)
        a = torch.tensor([0, 3])
        h1 = wm.observe_step(z, a, h0)
        assert h1.shape == (2, wm.deter_dim)

    def test_forward_heads(self):
        wm = _wm(causal_feature_dim=5, num_events=4)
        h = wm.initial_state(3)
        a = torch.tensor([1, 2, 0])
        next_z, reward, unc = wm.forward(h, a)
        assert next_z.shape == (3, wm.latent_dim)
        assert reward.shape == (3,)
        assert unc.shape == (3,)
        assert (unc >= 0).all()  # softplus
        aux = wm.forward_aux(h, a)
        assert aux["causal_feature_delta"].shape == (3, 5)
        assert aux["event_logits"].shape == (3, 4)


class TestMemory:
    """h must encode the history, not only the current observation."""

    def test_same_current_latent_different_history(self):
        wm = _wm()
        z_common = torch.randn(1, wm.latent_dim)

        # History A
        hA = wm.initial_state(1)
        hA = wm.observe_step(torch.randn(1, wm.latent_dim), torch.tensor([0]), hA)
        hA = wm.observe_step(z_common, torch.tensor([1]), hA)

        # History B: different first observation, same final latent + action.
        hB = wm.initial_state(1)
        hB = wm.observe_step(torch.randn(1, wm.latent_dim), torch.tensor([2]), hB)
        hB = wm.observe_step(z_common, torch.tensor([1]), hB)

        # Memory: the recurrent state differs because the past differs.
        assert not torch.allclose(hA, hB, atol=1e-4)

    def test_predictions_depend_on_history(self):
        wm = _wm()
        z_common = torch.randn(1, wm.latent_dim)
        a = torch.tensor([1])
        hA = wm.observe_step(torch.randn(1, wm.latent_dim), torch.tensor([0]), wm.initial_state(1))
        hA = wm.observe_step(z_common, a, hA)
        hB = wm.observe_step(torch.randn(1, wm.latent_dim), torch.tensor([4]), wm.initial_state(1))
        hB = wm.observe_step(z_common, a, hB)
        pa, _, _ = wm.forward(hA, a)
        pb, _, _ = wm.forward(hB, a)
        assert not torch.allclose(pa, pb, atol=1e-4)


class TestImagination:
    def test_rollout_shapes_and_finite(self):
        wm = _wm()
        particles = 8
        h = wm.initial_state(particles)
        for _ in range(5):
            a = torch.randint(0, wm.num_actions, (particles,))
            h, z, reward, unc = wm.imagine_batch(h, a)
            assert h.shape == (particles, wm.deter_dim)
            assert z.shape == (particles, wm.latent_dim)
            assert reward.shape == (particles,)
            assert torch.isfinite(h).all() and torch.isfinite(z).all()


class TestGradients:
    """Gradients must flow through the GRU for BPTT sequence training."""

    def test_grad_flows_through_recurrence(self):
        wm = _wm()
        h = wm.initial_state(2)
        loss = 0.0
        for t in range(4):
            z = torch.randn(2, wm.latent_dim)
            a = torch.randint(0, wm.num_actions, (2,))
            h = wm.observe_step(z, a, h)
            next_z, reward, _ = wm.forward(h, a)
            target = torch.randn_like(next_z)
            loss = loss + torch.mean((next_z - target) ** 2)
        loss.backward()
        # GRU and head params received gradient.
        assert wm.gru.weight_ih.grad is not None
        assert wm.gru.weight_ih.grad.abs().sum() > 0
        assert wm.next_state_head.weight.grad is not None
        assert wm.next_state_head.weight.grad.abs().sum() > 0
