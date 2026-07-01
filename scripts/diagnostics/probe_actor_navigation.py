"""Probe — l'actor est-il state-dependent (navigue) ou collé à la marginale ?

Suite de seedmind-10e.6. La représentation encode désormais la position (probe:
feat R²→dist=0.34, h=0.37) ET le critic est discriminant (corr(V,dist)=-0.24),
mais l'eval montre une policy dégénérée (greedy quasi-constant, sampled ≈ aléatoire).
Ce probe DÉPARTAGE le verrou résiduel :

  - si l'actor IGNORE le critic discriminant (biais directionnel indépendant de
    l'état) → verrou crédit/policy (marginale, pas conditionnelle) → seedmind-10e.5 ;
  - s'il TRACKE la cible → le verrou est ailleurs (proba d'action, masquage, ...).

On labellise des états réels par la direction (dr, dc) vers la cible la plus proche
et on mesure, sur la distribution d'action de l'actor lue sur le feat RSSM :

  - corr( P(RIGHT)-P(LEFT), dc )  et  corr( P(DOWN)-P(UP), dr )
      ~0  => le push directionnel NE dépend PAS de l'état (marginale)
      >>0 => l'actor oriente ses moves vers la cible (conditionnelle)
  - P(vers la cible) vs P(à l'opposé) : masse de proba sur les moves qui réduisent
    la distance vs ceux qui l'augmentent (baseline uniforme fournie).
  - sensibilité d'état : H(policy moyennée sur les états) − moyenne_états H(policy).
    ~0 => la policy ne change quasi pas d'un état à l'autre.
  - std inter-états des logits par action : ~0 => logits figés.

Usage:
  EVAL_CONFIG=configs/simple_grid_sparse_obsrecon.yaml \
  EVAL_CKPT=runs/w1_obsrecon_12k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_actor_navigation.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config  # noqa: E402
from seedmind.envs.simple_grid_world import GOAL  # noqa: E402

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_sparse_obsrecon.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_obsrecon_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "4000"))
device = torch.device("cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N}\n")

cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent
wm = agent.world_model
actor = agent.actor
nact = len(agent.actions)
idx = agent.action_index
i_up, i_down = idx.get("MOVE_UP"), idx.get("MOVE_DOWN")
i_left, i_right = idx.get("MOVE_LEFT"), idx.get("MOVE_RIGHT")
assert None not in (i_up, i_down, i_left, i_right), f"actions={agent.actions}"


def nearest_goal_delta(env):
    gs = np.argwhere(env.grid == GOAL)
    if len(gs) == 0:
        return None
    ar, ac = env.agent_pos
    d = [abs(int(r) - ar) + abs(int(c) - ac) for r, c in gs]
    r, c = gs[int(np.argmin(d))]
    return int(r) - ar, int(c) - ac  # (dr, dc): dr<0 cible au-dessus, dc>0 cible à droite


drs, dcs, P = [], [], []
logits_all = []
rng = np.random.default_rng(0)
agent.reset_state()
obs = sess.env.reset()
for _ in range(N):
    latent = agent.encoder.encode_tensor(obs)
    lnp = latent.squeeze(0) if latent.dim() == 2 else latent
    agent.advance(lnp.detach().cpu().numpy().astype(np.float32))
    delta = nearest_goal_delta(sess.env)
    if agent.rssm_state is not None and delta is not None:
        f = wm.get_feat(agent.rssm_state)
        with torch.no_grad():
            lg = actor.forward(f).squeeze(0)
            p = torch.softmax(lg, dim=-1).cpu().numpy()
        drs.append(delta[0]); dcs.append(delta[1]); P.append(p)
        logits_all.append(lg.cpu().numpy())
    a = int(rng.integers(nact))
    agent._prev_action_idx = a
    obs, _, done, _ = sess.env.step(agent.actions[a])
    if done:
        agent.reset_state()
        obs = sess.env.reset()

dr = np.array(drs, dtype=np.float64)
dc = np.array(dcs, dtype=np.float64)
P = np.stack(P).astype(np.float64)              # (n, nact)
L = np.stack(logits_all).astype(np.float64)     # (n, nact)
n = len(dr)
p_up, p_down = P[:, i_up], P[:, i_down]
p_left, p_right = P[:, i_left], P[:, i_right]


def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# Push directionnel vs offset de la cible (le test décisif).
c_h = corr(p_right - p_left, dc)   # push horizontal vs cible à droite/gauche
c_v = corr(p_down - p_up, dr)      # push vertical  vs cible en bas/haut

# Masse de proba vers / à l'opposé de la cible.
toward = ((dc > 0) * p_right + (dc < 0) * p_left + (dr > 0) * p_down + (dr < 0) * p_up)
away = ((dc > 0) * p_left + (dc < 0) * p_right + (dr > 0) * p_up + (dr < 0) * p_down)
# baseline uniforme (1/nact par move) : #directions réductrices / nact
ndir = (dc != 0).astype(float) + (dr != 0).astype(float)
toward_unif = float((ndir / nact).mean())

# Sensibilité d'état (mutual-information-like en nats).
eps = 1e-12
H_state = float((-(P * np.log(P + eps)).sum(1)).mean())
Pbar = P.mean(0)
H_marg = float(-(Pbar * np.log(Pbar + eps)).sum())

print(f"états={n}  dr∈[{dr.min():.0f},{dr.max():.0f}]  dc∈[{dc.min():.0f},{dc.max():.0f}]")
print("\n[proba d'action moyenne — révèle la marginale/biais]")
for a in range(nact):
    print(f"    {agent.actions[a]:12s}: {Pbar[a]:.3f}   (std inter-états logit={L[:,a].std():.3f})")

print("\n[TEST DÉCISIF — push directionnel vs direction de la cible]")
print(f"    corr( P(RIGHT)-P(LEFT) , dc )  = {c_h:+.3f}   (>>0 = tracke horizontalement)")
print(f"    corr( P(DOWN)-P(UP)    , dr )  = {c_v:+.3f}   (>>0 = tracke verticalement)")

print("\n[masse de proba vers la cible]")
print(f"    P(vers cible) = {toward.mean():.3f}   P(à l'opposé) = {away.mean():.3f}"
      f"   (baseline uniforme vers = {toward_unif:.3f})")

print("\n[sensibilité d'état de la policy]")
print(f"    H(policy moyenne) = {H_marg:.3f} nats   moyenne_états H = {H_state:.3f} nats")
print(f"    dépendance à l'état (H_marg - H_state) = {H_marg - H_state:+.4f}   (~0 = policy figée)")

# INTERACT sur la cible : collecter = INTERACT en étant SUR la cible (dr=dc=0).
i_int = idx.get("INTERACT")
if i_int is not None:
    on_goal = (dr == 0) & (dc == 0)
    p_int = P[:, i_int]
    adj = (np.abs(dr) + np.abs(dc)) == 1
    print("\n[INTERACT — clé de la collecte : il faut INTERACT EN ÉTANT SUR la cible]")
    print(f"    P(INTERACT | SUR cible)  = {p_int[on_goal].mean():.3f}  (n={int(on_goal.sum())})")
    print(f"    P(INTERACT | adjacent)   = {p_int[adj].mean():.3f}  (n={int(adj.sum())})")
    print(f"    P(INTERACT | loin)       = {p_int[~on_goal & ~adj].mean():.3f}  (n={int((~on_goal & ~adj).sum())})")
    if on_goal.sum() > 0:
        best = int(np.argmax(P[on_goal].mean(0)))
        print(f"    action préférée SUR la cible = {agent.actions[best]}  (devrait être INTERACT)")

print("\n-> Lecture:")
print("   - corr≈0 ET dépendance≈0 => l'actor IGNORE le critic (marginale) => crédit/policy (10e.5).")
print("   - corr>>0 ET P(INTERACT|sur cible) élevé MAIS argmax≠INTERACT / nav faible")
print("     => structure conditionnelle APPRISE mais SOUS-AFFÛTÉE (entropie, durée, advantage)")
print("        => tuning CPU (finir l'anneal d'entropie, sharpen), PAS le régime GPU.")
