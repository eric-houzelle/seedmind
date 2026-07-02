"""Probe fouloïde (seedmind-10e.7 universel) — la policy commitée fourrage-t-elle,
et le WM hallucine-t-il le retour (model-exploitation en reward DENSE) ?

Contexte : run runs/fouloide_dreamerfix_50k (fixes DreamerV3 universels : encodeur
trainable + obs-recon + prior déterministe). Post-anneal l'actor se committe
(imag_entropy 1.94→0.24) : les morts chutent (12→5/10k) MAIS le fourrage baisse et
imag_return explose (+13/15) alors que le wellbeing réel ≈ 0. Ce probe départage :

  [1] COMPORTEMENT réel de la policy commitée : distribution d'actions, événements
      (interact_water/food/noop), morts, durée de vie — attracteur « conservation
      REST » (survivre sans vivre) ou vrai fourrage ?
  [2] GAP imaginé↔réel apparié : depuis des états réels visités par l'actor,
      G_imag (rollout H pas, prior déterministe = celui de l'entraînement, reward
      head) vs G_real (reward_learning réellement encaissé sur les H pas suivants).

Usage (CPU) :
  EVAL_CONFIG=configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml \
  EVAL_CKPT=runs/fouloide_dreamerfix_50k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_fouloide_imag_gap.py
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config  # noqa: E402

CONFIG = os.environ.get("EVAL_CONFIG", "configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/fouloide_dreamerfix_50k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "4000"))
H = int(os.environ.get("PROBE_H", "15"))
GAMMA = float(os.environ.get("PROBE_GAMMA", "0.97"))
MAXSTARTS = int(os.environ.get("PROBE_STARTS", "192"))
DET_PRIOR = os.environ.get("PROBE_DET_PRIOR", "1") == "1"   # = imagination d'entraînement
device = torch.device("cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N} H={H} det_prior={DET_PRIOR}\n")

cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
agent = sess.agent
wm = agent.world_model
actor = agent.actor

# Capture reward_learning/event/done via learner.observe (sans entraînement).
captured = []
sess.learner.observe = lambda exp: captured.append(
    (float(exp.get("reward_learning", 0.0)), str(exp.get("event", "?")), bool(exp.get("done", False)))
)

# [1] La session agit avec la policy réelle de l'agent (actor échantillonné).
posts, boundaries = [], []
acts = Counter()
for _ in range(N):
    sess.step()
    if agent.rssm_state is not None:
        posts.append({k: v.detach().clone() for k, v in agent.rssm_state.items()})
    else:
        posts.append(None)
    boundaries.append(captured[-1][2] if captured else False)
    acts[sess.last_action] += 1

r_real = np.array([c[0] for c in captured], dtype=np.float64)
events = Counter(c[1] for c in captured)
dones = np.array([c[2] for c in captured], dtype=bool)
n_deaths = int(dones.sum())
life_spans = np.diff(np.flatnonzero(np.concatenate([[True], dones]))) if n_deaths else np.array([N])

print("=" * 70)
print("[1] COMPORTEMENT réel de la policy commitée")
print("=" * 70)
tot = sum(acts.values())
for a, c in acts.most_common():
    print(f"    {a:12s}: {c:5d}  ({100*c/tot:.1f}%)")
print(f"\n  événements: { {k: v for k, v in events.most_common(8)} }")
print(f"  morts: {n_deaths} sur {N} pas ({1000*n_deaths/N:.1f}/1000) | vie moyenne {life_spans.mean():.0f} pas")
print(f"  eau {events.get('interact_water',0)} | bouffe {events.get('interact_food',0)} | "
      f"noop {events.get('interact_noop',0)}")
print(f"  reward_learning réel moyen/pas: {r_real.mean():+.4f}")

# [2] Gap imaginé↔réel apparié.
cand = [t for t in range(len(posts) - H)
        if posts[t] is not None and not dones[t:t + H].any()]
if len(cand) > MAXSTARTS:
    step = len(cand) / MAXSTARTS
    cand = [cand[int(k * step)] for k in range(MAXSTARTS)]
S = {k: torch.cat([posts[t][k] for t in cand], 0) for k in posts[cand[0]]}
ns = len(cand)

imag_r = np.zeros((H, ns))
with torch.no_grad():
    state = {k: v.clone() for k, v in S.items()}
    for h in range(H):
        feat = wm.get_feat(state)
        a = actor.act(feat)
        if DET_PRIOR:
            state = wm.img_step(state, a, sample=False)
            r = wm.reward_value(wm.get_feat(state))
        else:
            state, _f, r, _u = wm.imagine_batch(state, a)
        imag_r[h] = r.cpu().numpy()

real_r = np.stack([r_real[t:t + H] for t in cand], axis=1)
disc = GAMMA ** np.arange(H)
G_imag = (disc[:, None] * imag_r).sum(0)
G_real = (disc[:, None] * real_r).sum(0)

print("\n" + "=" * 70)
print(f"[2] GAP imaginé↔réel — {ns} départs réels, H={H}, prior {'déterministe' if DET_PRIOR else 'échantillonné'}")
print("=" * 70)
print(f"  G_imag : {G_imag.mean():+.3f} (±{G_imag.std():.3f})")
print(f"  G_real : {G_real.mean():+.3f} (±{G_real.std():.3f})")
print(f"  => ÉCART imag−réel : {G_imag.mean()-G_real.mean():+.3f}")
print(f"\n  reward par pas imaginé : {np.array2string(imag_r.mean(1), precision=3, floatmode='fixed', max_line_width=200)}")
print(f"  reward par pas réel    : {np.array2string(real_r.mean(1), precision=3, floatmode='fixed', max_line_width=200)}")
print("\n-> ÉCART >> 0 et croissant avec la profondeur = model-exploitation en reward dense")
print("   (le WM imagine des drives confortables inexistants; l'actor optimise ce fantasme).")
