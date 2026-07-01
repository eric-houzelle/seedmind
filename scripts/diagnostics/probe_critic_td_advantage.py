"""Probe couche 5 (seedmind-10e.7) — le critic donne-t-il un gradient de NAVIGATION
   en TD (donc exploitable par un lambda plus BAS) ?

Contexte (run H=6, runs/w1_obsrecon_h6_12k) : imagination HONNÊTE (gap imag↔réel
éliminé), représentation intacte, ET critic discriminant (corr(V,dist)=-0.41,
V=1.12 sur cible → 0.84 loin) — mais l'actor collapse (0 collecte). Hypothèse :
avec lambda=0.95 la lambda-return est ~Monte-Carlo (rewards honnêtes mais nuls en
H pas de monde éparse) et n'injecte que (1-lambda)=5 % du V discriminant → l'actor
ne « voit » pas le gradient du critic. Un lambda plus bas (plus de TD-bootstrap)
propagerait V(near)>V(far) dans les returns.

Ce probe teste, SANS ré-entraînement, si le signal TD one-step pointe vers la cible.
Pour chaque état réel labellisé par (dr,dc) vers la cible la plus proche, et chaque
action de déplacement a, on calcule la valeur-action TD one-step (le terme qui
DISCRIMINE les actions dans une return à lambda bas) :

    q(s,a) = r_pred(s,a) + gamma * V( prior(s,a) )

puis :
  - corr( q(RIGHT)-q(LEFT), dc )  et  corr( q(DOWN)-q(UP), dr )
      >>0 => le TD one-step pointe vers la cible => un lambda bas donnerait un
             gradient de navigation (le critic est exploitable) ;
      ~0  => même le critic+dynamique une-étape n'orientent pas => le verrou est
             plus profond (fidélité dynamique / crédit), pas juste lambda.
  - fraction d'états où argmax_a q(s,a) est un move QUI RÉDUIT la distance
    (vs baseline actor greedy et vs 1/ndir).

On compare au push directionnel de l'ACTOR courant (déjà mesuré) : si le TD pointe
vers la cible bien PLUS que l'actor, c'est que le signal existe mais n'atteint pas
la policy (→ lambda). Si le TD ne pointe pas non plus, lambda ne suffira pas.

Usage (CPU) :
  EVAL_CONFIG=configs/simple_grid_sparse_obsrecon.yaml \
  EVAL_CKPT=runs/w1_obsrecon_h6_12k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_critic_td_advantage.py
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
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_obsrecon_h6_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "4000"))
GAMMA = float(os.environ.get("PROBE_GAMMA", "0.97"))
device = torch.device("cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N}  gamma={GAMMA}\n")

cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent
wm = agent.world_model
actor = agent.actor
critic = agent.critic
assert critic is not None, "pas de critic sur l'agent"
twohot = hasattr(critic, "value")
idx = agent.action_index
i_up, i_down = idx.get("MOVE_UP"), idx.get("MOVE_DOWN")
i_left, i_right = idx.get("MOVE_LEFT"), idx.get("MOVE_RIGHT")
i_int = idx.get("INTERACT")
assert None not in (i_up, i_down, i_left, i_right), f"actions={agent.actions}"


def V(feat):
    if twohot:
        return critic.value(feat)
    from seedmind.training.imagination_actor_critic import symexp
    return symexp(critic(feat).clamp(-8, 8))


def nearest_goal_delta(env):
    gs = np.argwhere(env.grid == GOAL)
    if len(gs) == 0:
        return None
    ar, ac = env.agent_pos
    d = [abs(int(r) - ar) + abs(int(c) - ac) for r, c in gs]
    r, c = gs[int(np.argmin(d))]
    return int(r) - ar, int(c) - ac


# collecte d'états réels + leur (dr,dc)
states, drs, dcs = [], [], []
rng = np.random.default_rng(0)
agent.reset_state()
obs = sess.env.reset()
with torch.no_grad():
    for _ in range(N):
        latent = agent.encoder.encode_tensor(obs)
        lnp = (latent.squeeze(0) if latent.dim() == 2 else latent)
        agent.advance(lnp.detach().cpu().numpy().astype(np.float32))
        delta = nearest_goal_delta(sess.env)
        if agent.rssm_state is not None and delta is not None:
            states.append({k: v.detach().clone() for k, v in agent.rssm_state.items()})
            drs.append(delta[0]); dcs.append(delta[1])
        a = int(rng.integers(len(agent.actions)))
        agent._prev_action_idx = a
        obs, _, done, _ = sess.env.step(agent.actions[a])
        if done:
            agent.reset_state(); obs = sess.env.reset()

M = len(states)
dr = np.array(drs, dtype=np.float64)
dc = np.array(dcs, dtype=np.float64)
S = {k: torch.cat([s[k] for s in states], 0) for k in states[0]}


@torch.no_grad()
def q_action(a_idx, sample=True):
    """q(s,a) = r_pred(s,a) + gamma*V(prior(s,a)). sample=False => prior déterministe
    (pas de bruit d'échantillonnage de z) pour isoler l'effet directionnel de l'action."""
    B = S["deter"].shape[0]
    a = torch.full((B,), a_idx, dtype=torch.long, device=device)
    prior = wm.img_step(S, a, sample=sample)
    feat = wm.get_feat(prior)
    r = wm.reward_value(feat)
    q = r + GAMMA * V(feat)
    return q.cpu().numpy(), r.cpu().numpy()


qs = {}
rs = {}
qs_det = {}
for name, a in [("UP", i_up), ("DOWN", i_down), ("LEFT", i_left), ("RIGHT", i_right), ("INT", i_int)]:
    if a is not None:
        qs[name], rs[name] = q_action(a, sample=True)
        qs_det[name], _ = q_action(a, sample=False)
with torch.no_grad():
    Vs = V(wm.get_feat(S)).cpu().numpy()


def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


print(f"états={M}  dr∈[{dr.min():.0f},{dr.max():.0f}]  dc∈[{dc.min():.0f},{dc.max():.0f}]")
print(f"V(s) moyen={Vs.mean():+.3f}  (référence)\n")

print("[q-valeurs TD one-step moyennes par action]  q = r_pred + gamma*V(s')")
for name in ["UP", "DOWN", "LEFT", "RIGHT", "INT"]:
    if name in qs:
        print(f"    {name:5s}: q={qs[name].mean():+.4f}   r_pred={rs[name].mean():+.4f}")

c_h = corr(qs["RIGHT"] - qs["LEFT"], dc)
c_v = corr(qs["DOWN"] - qs["UP"], dr)
c_h_det = corr(qs_det["RIGHT"] - qs_det["LEFT"], dc)
c_v_det = corr(qs_det["DOWN"] - qs_det["UP"], dr)
print("\n[TEST DÉCISIF — le TD one-step pointe-t-il vers la cible ?]")
print(f"    prior ÉCHANTILLONNÉ : corr(q(R)-q(L),dc)={c_h:+.3f}   corr(q(D)-q(U),dr)={c_v:+.3f}")
print(f"    prior DÉTERMINISTE  : corr(q(R)-q(L),dc)={c_h_det:+.3f}   corr(q(D)-q(U),dr)={c_v_det:+.3f}")
print("    (déterministe >> échantillonné => bruit de z tue le signal => levier KL/unimix ;")
print("     les deux ~0 => la dynamique n'encode pas la direction => plus profond)")

# argmax des q sur les moves : réduit-il la distance ?
move_names = ["UP", "DOWN", "LEFT", "RIGHT"]
Q = np.stack([qs[n] for n in move_names], axis=1)          # (M,4)
argm = Q.argmax(1)
# le move réduit la distance si: UP&dr<0, DOWN&dr>0, LEFT&dc<0, RIGHT&dc>0
reduces = np.zeros(M, dtype=bool)
reduces |= (argm == 0) & (dr < 0)
reduces |= (argm == 1) & (dr > 0)
reduces |= (argm == 2) & (dc < 0)
reduces |= (argm == 3) & (dc > 0)
ndir = (dc != 0).astype(float) + (dr != 0).astype(float)
baseline = float((ndir / 4.0).mean())   # move réducteur au hasard parmi 4 moves
print("\n[greedy sur les q-valeurs TD — fraction de moves qui RÉDUISENT la distance]")
print(f"    frac(argmax q réduit la distance) = {reduces.mean():.3f}   (baseline hasard = {baseline:.3f})")

# comparaison au push de l'ACTOR courant (ce qu'il fait déjà)
with torch.no_grad():
    lg = actor.forward(wm.get_feat(S))
    P = torch.softmax(lg, dim=-1).cpu().numpy()
c_h_actor = corr(P[:, i_right] - P[:, i_left], dc)
c_v_actor = corr(P[:, i_down] - P[:, i_up], dr)
print("\n[comparaison : push directionnel de l'ACTOR courant]")
print(f"    corr( P(RIGHT)-P(LEFT), dc ) = {c_h_actor:+.3f}   corr( P(DOWN)-P(UP), dr ) = {c_v_actor:+.3f}")

print("\n-> Lecture:")
print("   - q-TD pointe vers la cible (corr>>0) BIEN plus que l'actor => le signal de")
print("     navigation EXISTE dans critic+dynamique mais n'atteint pas la policy via la")
print("     lambda-return Monte-Carlo => baisser lambda (TD-bootstrap) devrait débloquer.")
print("   - q-TD ne pointe pas non plus (corr~0) => lambda ne suffira pas ; verrou = fidélité")
print("     dynamique une-étape / crédit => KL / plus de données / régime GPU (10e.5).")
