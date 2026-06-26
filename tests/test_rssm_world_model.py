"""Tests du RSSMWorldModel (DreamerV3, phase 1 brique 2) : heads sur feat=[z,h]."""
import torch

from seedmind.agent.world_model import RSSMWorldModel

EMBED, NACT, STOCH, DISC, DETER = 48, 7, 8, 8, 32
CF, NEV = 5, 4


def _wm():
    torch.manual_seed(0)
    return RSSMWorldModel(embed_dim=EMBED, num_actions=NACT, stoch=STOCH, discrete=DISC,
                          deter=DETER, hidden=32, causal_feature_dim=CF, num_events=NEV)


def test_feat_and_heads_shapes():
    wm = _wm()
    B = 6
    st = wm.initial_state(B)
    a = torch.randint(0, NACT, (B,))
    embed = torch.randn(B, EMBED)
    post, prior = wm.observe_step(embed, a, st)
    feat = wm.get_feat(post)
    assert feat.shape == (B, wm.feat_dim) == (B, STOCH * DISC + DETER)
    out = wm.heads(feat)
    assert out["recon"].shape == (B, EMBED)        # decoder reconstructs the embedding
    assert out["reward"].shape == (B,)
    assert out["continue"].shape == (B,)
    assert out["uncertainty"].shape == (B,) and (out["uncertainty"] >= 0).all()
    assert out["causal_feature_delta"].shape == (B, CF)
    assert out["event_logits"].shape == (B, NEV)


def test_decoder_can_fit_an_embedding():
    """The decoder(feat) must be able to reconstruct a target embedding (grounding signal)."""
    wm = _wm()
    B = 32
    st = wm.initial_state(B)
    a = torch.randint(0, NACT, (B,))
    embed = torch.randn(B, EMBED)
    opt = torch.optim.Adam(wm.parameters(), 3e-3)
    post, _ = wm.observe_step(embed, a, st)
    before = (wm.heads(wm.get_feat(post))["recon"] - embed).pow(2).mean().item()
    for _ in range(100):
        post, _ = wm.observe_step(embed, a, st)
        loss = (wm.heads(wm.get_feat(post))["recon"] - embed).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = (wm.heads(wm.get_feat(post))["recon"] - embed).pow(2).mean().item()
    assert after < before * 0.5  # recon loss clearly drops → gradients reach RSSM + decoder


def test_imagine_batch_shapes():
    wm = _wm()
    B = 4
    st = wm.initial_state(B)
    a = torch.randint(0, NACT, (B,))
    nxt, feat, reward, unc = wm.imagine_batch(st, a)
    assert nxt["deter"].shape == (B, DETER) and nxt["stoch"].shape == (B, STOCH, DISC)
    assert feat.shape == (B, wm.feat_dim)
    assert reward.shape == (B,) and unc.shape == (B,)
