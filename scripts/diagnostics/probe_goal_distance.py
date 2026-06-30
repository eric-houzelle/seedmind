"""Probe — la représentation encode-t-elle la POSITION de la cible ? (SimpleGridWorld)

Outil de vérification du chantier "encodeur entraînable + reconstruction d'obs"
(seedmind-10e.6). Le critic ne peut créditer la navigation que si son entrée (le
``feat`` RSSM) encode où est la cible. On labellise des états réels par la distance
de Manhattan à la cible la plus proche et on mesure, par un probe linéaire FRAIS
(ridge + PCA pour éviter le sur-apprentissage en haute dimension) :

  - encodeur (embed) -> distance
  - feat RSSM [z,h]  -> distance   <- CE QUE CONSOMMENT actor/critic (le seul qui compte)
  - h déterministe / z stochastique séparément
  - corrélation V(critic) vs distance

Diagnostic établi 2026-06-30 (frozen encoder, latent 64): conv 0.65 -> embed 0.09
-> feat ~0 ; latent 256: embed 0.38 mais feat ~0 (h=0, z inexploitable). CIBLE du
chantier: feat R^2 >> 0 ET corr(V,distance) nettement négative ET collecte qui monte.

Usage:
  EVAL_CONFIG=configs/simple_grid_sparse_reveal.yaml \
  EVAL_CKPT=runs/<run>/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_goal_distance.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config  # noqa: E402
from seedmind.envs.simple_grid_world import GOAL  # noqa: E402

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_sparse_reveal.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_sparse_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "4000"))
device = torch.device("cpu")  # MPS fuit sur le RSSM; CPU pour les probes
print(f"config={CONFIG}\nckpt={CKPT}\nN={N}\n")

cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent
wm = agent.world_model
critic = agent.critic
nact = len(agent.actions)


def nearest_goal_dist(env):
    gs = np.argwhere(env.grid == GOAL)
    if len(gs) == 0:
        return None
    ar, ac = env.agent_pos
    return int(min(abs(int(r) - ar) + abs(int(c) - ac) for r, c in gs))


feats, lats, deters, stochs, dists, vs = [], [], [], [], [], []
rng = np.random.default_rng(0)
agent.reset_state()
obs = sess.env.reset()
for _ in range(N):
    latent = agent.encoder.encode_tensor(obs)
    lnp = latent.squeeze(0) if latent.dim() == 2 else latent
    agent.advance(lnp.detach().cpu().numpy().astype(np.float32))
    d = nearest_goal_dist(sess.env)
    if agent.rssm_state is not None and d is not None:
        st = agent.rssm_state
        f = wm.get_feat(st).detach()
        feats.append(f)
        lats.append(lnp.detach().reshape(1, -1))
        deters.append(st["deter"].detach().reshape(1, -1))
        stochs.append(st["stoch"].detach().reshape(1, -1))
        dists.append(d)
        with torch.no_grad():
            vs.append(float(critic.value(f).item()) if hasattr(critic, "value") else float(critic(f).item()))
    a = int(rng.integers(nact))
    agent._prev_action_idx = a
    obs, _, done, _ = sess.env.step(agent.actions[a])
    if done:
        agent.reset_state()
        obs = sess.env.reset()

d = np.array(dists, dtype=np.float32)
v = np.array(vs, dtype=np.float32)
n = len(d)
idx = rng.permutation(n)
tr, te = idx[: n * 4 // 5], idx[n * 4 // 5:]


def pca_ridge_r2(X, k=48, lam=1.0):
    """Reliable probe: PCA to k dims (kills high-dim overfit) then ridge."""
    X = np.asarray(X, dtype=np.float64)
    k = min(k, X.shape[1])
    mu = X[tr].mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    P = Vt[:k].T
    Z = Xc @ P
    Ztr = np.c_[Z[tr], np.ones(len(tr))]
    W = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]), Ztr.T @ d[tr])
    p = np.c_[Z[te], np.ones(len(te))] @ W
    return 1 - ((d[te] - p) ** 2).sum() / ((d[te] - d[te].mean()) ** 2).sum()


print(f"states={n}  distance range {d.min():.0f}..{d.max():.0f} mean {d.mean():.2f}")
print(f"\n[critic V vs distance]  corr={np.corrcoef(v, d)[0,1]:+.3f}  (cible: nettement < 0)")
for dd in sorted(set(d.astype(int)))[:8]:
    m = d.astype(int) == dd
    print(f"    dist={dd:2d}: V={v[m].mean():+.4f} (n={m.sum()})")

E = np.concatenate(lats, 0)
F = torch.cat(feats, 0).numpy()
H = np.concatenate(deters, 0)
Z = np.concatenate(stochs, 0)
print("\n[probe FRAIS PCA-48 + ridge -> distance]  (>0.5 = position utilisable)")
print(f"    encodeur embed ({E.shape[1]}d) : R^2={pca_ridge_r2(E):+.3f}")
print(f"    feat RSSM [z,h] ({F.shape[1]}d) : R^2={pca_ridge_r2(F):+.3f}   <- CE QUI COMPTE (actor/critic)")
print(f"      .deter h ({H.shape[1]}d) : R^2={pca_ridge_r2(H):+.3f}")
print(f"      .stoch z ({Z.shape[1]}d) : R^2={pca_ridge_r2(Z):+.3f}")
print("\n-> Chantier réussi si feat R^2 passe de ~0 à >0.5 ET corr(V,distance) << 0.")
