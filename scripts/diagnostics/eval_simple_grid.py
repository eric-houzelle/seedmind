"""Éval W1 — l'agent a-t-il APPRIS à collecter la cible dans SimpleGridWorld ?

Le logger online est fouloïde-spécifique (wellbeing/eau/bouffe/morts) et ne compte
pas les collectes. Ici : rollout GREEDY (learner gelé, actor en mode greedy), on
compte les évènements `interact_goal` (collectes) et la distribution d'actions,
COMPARÉS à une baseline ALÉATOIRE sur le même monde.

Verdict : si collectes(greedy) >> collectes(aléatoire) → le stack DreamerV3 a appris
à naviguer vers la cible et à INTERACT → le port est bon, la boucle online apprend.
"""
import sys, os
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_online_rssm_v3.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_simplegrid_15k/checkpoint_online.pt")
N = int(os.environ.get("EVAL_N", "5000"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N} device={device}")


def random_baseline(seed=123):
    """Politique aléatoire uniforme sur le même monde (référence)."""
    cfg = load_config(CONFIG)
    sess = OnlineFouloideSession(cfg, seed=seed, device=device)
    rng = np.random.default_rng(seed)
    actions = sess.env.available_actions()
    events = Counter()
    for _ in range(N):
        a = actions[int(rng.integers(len(actions)))]
        _, _, done, info = sess.env.step(a)
        events[info.get("event", "?")] += 1
        if done:
            sess.env.reset()
    return events


def greedy_eval():
    cfg = load_config(CONFIG)
    sess = OnlineFouloideSession(cfg, seed=0, device=device)
    sess.resume(CKPT)
    sess.learner.observe = lambda *a, **k: None          # gel : on évalue un ckpt figé
    actor = sess.agent.actor
    _orig = actor.act_masked
    actor.act_masked = lambda rec, avail, greedy=False: _orig(rec, avail, greedy=True)  # GREEDY
    events = Counter(); acts = Counter()
    for _ in range(N):
        sess.step()
        events[sess.env.describe_transition()] += 1
        acts[sess.last_action] += 1
    return events, acts


print("\n=== BASELINE ALÉATOIRE ===")
rnd = random_baseline()
rnd_goals = rnd.get("interact_goal", 0)
print(f"  collectes (interact_goal) : {rnd_goals}  ({1000*rnd_goals/N:.1f} / 1000 pas)")
print(f"  events: {dict(rnd)}")

print("\n=== POLICY APPRISE (greedy) ===")
ev, acts = greedy_eval()
goals = ev.get("interact_goal", 0)
print(f"  collectes (interact_goal) : {goals}  ({1000*goals/N:.1f} / 1000 pas)")
print(f"  events: {dict(ev)}")
print(f"  actions: {dict(acts)}")
interacts = acts.get("INTERACT", 0)
if interacts:
    print(f"  précision INTERACT : {100*goals/interacts:.0f}% des INTERACT tombent sur la cible")

print("\n=== VERDICT ===")
if rnd_goals == 0 and goals == 0:
    print("  ni l'un ni l'autre ne collecte — investiguer (perception/horizon/run trop court)")
else:
    ratio = goals / max(rnd_goals, 1)
    print(f"  ratio appris/aléatoire = {ratio:.1f}×")
    print("  -> APPRIS" if ratio >= 2 else "  -> PAS (ENCORE) APPRIS (run plus long ?)")
