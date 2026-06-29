"""Planner sur le WM (h,z) — couche 3 : court-circuiter l'actor-critic cassé.

Idée : on a prouvé que le WM est bon (reward fidèle). Donc à chaque pas réel, on
PLANIFIE directement dessus — recherche par force brute des séquences d'actions de
longueur H, déroulées dans l'imagination (img_step), en maximisant la SOMME des
rewards imaginés (que le WM prédit correctement). On prend la 1re action de la
meilleure séquence. AUCUN critic, AUCUN actor : que le WM.

Si collectes(planner) >> aléatoire → le WM suffit, l'actor-critic n'était qu'une
méthode d'extraction de policy remplaçable. C'est la validation de la bascule.
"""
import sys, os, itertools
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_dense_rssm_v3.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_dense_fixed_12k/checkpoint_online.pt")
N = int(os.environ.get("EVAL_N", "3000"))
H = int(os.environ.get("PLAN_H", "4"))     # horizon de planification (force brute nact^H)
GAMMA = 0.97

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N} H={H} device={device}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
ag = sess.agent; wm = ag.world_model
nact = len(ag.actions)

# toutes les séquences d'actions de longueur H : (nseq, H)
SEQS = torch.tensor(list(itertools.product(range(nact), repeat=H)), dtype=torch.long, device=device)
nseq = SEQS.shape[0]
DISC = torch.tensor([GAMMA ** t for t in range(H)], device=device)
print(f"{nseq} séquences de longueur {H}")


@torch.no_grad()
def plan_action(state):
    """Déroule toutes les séquences depuis `state` (batch 1), renvoie la 1re action
    de celle qui maximise la somme de rewards imaginés (déterministe: prior mode)."""
    st = {k: v.expand(nseq, *v.shape[1:]).contiguous() for k, v in state.items()}
    total = torch.zeros(nseq, device=device)
    for t in range(H):
        a = SEQS[:, t]
        st = wm.img_step(st, a, sample=False)          # prior déterministe (mode)
        r = wm.reward_value(wm.get_feat(st))
        total += DISC[t] * r
    best = int(total.argmax().item())
    return int(SEQS[best, 0].item())


def run(policy_fn, seed):
    sess2 = OnlineFouloideSession(load_config(CONFIG), seed=seed, device=device)
    sess2.resume(CKPT); sess2.learner.observe = lambda *a, **k: None
    a2 = sess2.agent; ev = Counter()
    a2.reset_state(); obs = sess2.env.reset(); rng = np.random.default_rng(seed)
    for _ in range(N):
        lat = a2.encoder.encode_tensor(obs)
        a2.advance(lat.squeeze(0).detach().cpu().numpy().astype(np.float32))
        if policy_fn == "random":
            ai = int(rng.integers(nact))
        else:
            ai = plan_action(a2.rssm_state) if a2.rssm_state is not None else int(rng.integers(nact))
        a2._prev_action_idx = ai
        obs, _, done, info = sess2.env.step(a2.actions[ai])
        ev[info.get("event", "?")] += 1
        if done:
            a2.reset_state(); obs = sess2.env.reset()
    return ev


print("\n=== ALÉATOIRE ===")
r = run("random", 123)
rg = r.get("interact_goal", 0)
print(f"  collectes: {rg}  ({1000*rg/N:.1f}/1000)  events={dict(r)}")

print(f"\n=== PLANNER sur le WM (H={H}) ===")
p = run("planner", 0)
pg = p.get("interact_goal", 0)
print(f"  collectes: {pg}  ({1000*pg/N:.1f}/1000)  events={dict(p)}")

print(f"\n=== VERDICT ===\n  ratio planner/aléatoire = {pg/max(rg,1):.1f}×")
print("  -> WM SUFFISANT (l'actor-critic était le seul fautif)" if pg >= 2*rg else "  -> le planner ne décolle pas non plus (creuser le WM/horizon)")
