"""POURQUOI le WM sur-prédit INTERACT-à-vide (+0.03 au lieu de -0.02) ?

Hypothèse : le reward head twohot HÉDGE — il n'arrive pas à dire avec certitude,
depuis le latent post-INTERACT, si l'INTERACT a touché une cible, donc il met un
peu de masse sur les bins '+1' → l'espérance remonte de -0.02 à +0.03.

Test : sur les états INTERACT-hors-cible vs INTERACT-sur-cible, on inspecte la
DISTRIBUTION complète du reward head (softmax sur bins), pas juste l'espérance.
Si les états hors-cible ont une masse non négligeable sur les bins 'reward > 0.3'
(cible-like) → hédging confirmé = le latent ne sépare pas nettement on/off cible.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config
from seedmind.envs.simple_grid_world import GOAL

CONFIG = os.environ.get("PROBE_CONFIG", "configs/simple_grid_dense_rssm_v3.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/w1_dense_fixed_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "8000"))
MAXSTATES = 160

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
ag = sess.agent; wm = ag.world_model
rh = wm.reward_head
assert hasattr(rh, "bins"), "reward head n'est pas un TwoHotCritic ?"
INT = ag.action_index["INTERACT"]; nact = len(ag.actions)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


bins_reward = symexp(rh.bins).cpu().numpy()   # valeur en reward-space de chaque bin
print(f"bins reward-space: min={bins_reward.min():.2f} max={bins_reward.max():.2f} | "
      f"bin le + proche de +1 = {bins_reward[np.argmin(np.abs(bins_reward-1.0))]:.3f}")

rng = np.random.default_rng(0)
on, off = [], []
ag.reset_state(); obs = sess.env.reset()
for _ in range(N):
    e = ag.encoder.encode_tensor(obs)
    ag.advance((e.squeeze(0) if e.dim() == 2 else e).detach().cpu().numpy().astype(np.float32))
    if ag.rssm_state is not None:
        st = {k: v.detach().clone() for k, v in ag.rssm_state.items()}
        (on if int(obs.get("standing_entity", -1)) == GOAL else off).append(st)
    ai = int(rng.integers(nact)); ag._prev_action_idx = ai
    obs, _, d, _ = sess.env.step(ag.actions[ai])
    if d: ag.reset_state(); obs = sess.env.reset()
    if len(on) >= MAXSTATES and len(off) >= MAXSTATES: break


@torch.no_grad()
def analyse(name, sl):
    if not sl:
        print(f"\n[{name}] vide"); return
    S = {k: torch.cat([s[k][:1] for s in sl[:MAXSTATES]], 0) for k in sl[0]}
    post = wm.img_step(S, torch.full((S['deter'].shape[0],), INT, dtype=torch.long, device=device), sample=False)
    feat = wm.get_feat(post)
    probs = torch.softmax(rh.forward(feat), dim=-1).cpu().numpy()   # (B, nbins)
    exp_val = float(wm.reward_value(feat).mean().item())
    mass_goal = probs[:, bins_reward > 0.3].sum(1).mean()           # masse sur bins cible-like
    mass_neg = probs[:, bins_reward < -0.005].sum(1).mean()         # masse sur bins négatifs (noop réel)
    modal = bins_reward[probs.mean(0).argmax()]
    print(f"\n=== INTERACT depuis {name} ({S['deter'].shape[0]} états) ===")
    print(f"  reward espéré (value)       : {exp_val:+.4f}")
    print(f"  masse sur bins reward>0.3   : {100*mass_goal:.1f}%   (cible-like)")
    print(f"  masse sur bins reward<0     : {100*mass_neg:.1f}%   (noop réel)")
    print(f"  bin modal (reward-space)    : {modal:+.3f}")


analyse("HORS-CIBLE (devrait être -0.02)", off)
analyse("SUR-CIBLE (devrait être +1.0)", on)
print("\n-> si HORS-CIBLE a une masse notable sur bins>0.3 = HÉDGING confirmé (latent ne sépare pas on/off cible).")
