"""Tests du RSSM stochastique (DreamerV3, phase 1 du port)."""
import torch

from seedmind.agent.rssm import RSSM

EMBED, NACT, STOCH, DISC, DETER = 48, 7, 8, 8, 32


def _rssm():
    torch.manual_seed(0)
    return RSSM(embed_dim=EMBED, num_actions=NACT, stoch=STOCH, discrete=DISC,
                deter=DETER, hidden=32)


def test_shapes_and_feat_dim():
    rssm = _rssm()
    B = 5
    st = rssm.initial_state(B)
    assert st["deter"].shape == (B, DETER)
    assert st["stoch"].shape == (B, STOCH, DISC)
    a = torch.randint(0, NACT, (B,))
    prior = rssm.img_step(st, a)
    assert prior["deter"].shape == (B, DETER)
    assert prior["stoch"].shape == (B, STOCH, DISC)
    embed = torch.randn(B, EMBED)
    post, prior2 = rssm.obs_step(st, a, embed)
    assert post["stoch"].shape == (B, STOCH, DISC)
    # posterior shares the prior's deterministic state
    assert torch.allclose(post["deter"], prior2["deter"])
    feat = rssm.get_feat(post)
    assert feat.shape == (B, STOCH * DISC + DETER) == (B, rssm.feat_dim)


def test_stoch_is_onehot_per_group():
    rssm = _rssm()
    st = rssm.initial_state(4)
    a = torch.zeros(4, dtype=torch.long)
    z = rssm.img_step(st, a)["stoch"]
    # each of the STOCH groups is a one-hot over DISC classes -> sums to 1
    assert torch.allclose(z.sum(-1), torch.ones(4, STOCH), atol=1e-4)


def test_kl_free_bits_and_balance():
    rssm = _rssm()
    B = 16
    st = rssm.initial_state(B)
    a = torch.randint(0, NACT, (B,))
    embed = torch.randn(B, EMBED)
    post, prior = rssm.obs_step(st, a, embed)
    loss, dyn, rep = rssm.kl_loss(post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1)
    assert torch.isfinite(loss)
    assert dyn >= 1.0 - 1e-4 and rep >= 1.0 - 1e-4   # clamped at free bits
    # balance: loss == dyn_scale*dyn + rep_scale*rep
    assert torch.allclose(loss, 0.5 * dyn + 0.1 * rep, atol=1e-4)


def test_straight_through_gradients_flow():
    rssm = _rssm()
    st = rssm.initial_state(8)
    a = torch.randint(0, NACT, (8,))
    prior = rssm.img_step(st, a)
    # a loss on the sampled (discrete) z must backprop into the RSSM params
    loss = rssm.get_feat(prior).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in rssm.img_logits.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_mode_is_deterministic():
    rssm = _rssm()
    st = rssm.initial_state(3)
    a = torch.randint(0, NACT, (3,))
    z1 = rssm.img_step(st, a, sample=False)["stoch"]
    z2 = rssm.img_step(st, a, sample=False)["stoch"]
    assert torch.allclose(z1, z2)  # mode is deterministic across calls
