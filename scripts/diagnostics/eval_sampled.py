"""Éval SimpleGridWorld — collecte en GREEDY *et* ÉCHANTILLONNÉ (+ températures).

LEÇON (2026-06-30): DreamerV3 se déploie en ÉCHANTILLONNANT l'actor, pas en argmax.
Évaluer uniquement en greedy a longtemps masqué que la policy apprenait (l'argmax
d'une policy quasi-uniforme dégénère en action constante -> 0 collecte, alors que
l'échantillonné collectait ~3× l'aléatoire). TOUJOURS regarder les deux.

Compare la policy apprise (plusieurs modes/températures) à une baseline ALÉATOIRE
sur le même monde.

Usage:
  EVAL_CONFIG=configs/simple_grid_sparse_reveal.yaml \
  EVAL_CKPT=runs/<run>/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/eval_sampled.py
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config  # noqa: E402

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_sparse_reveal.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_sparse_12k/checkpoint_online.pt")
N = int(os.environ.get("EVAL_N", "5000"))
device = torch.device("cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N}\n")


def random_baseline(seed=123):
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
    return events.get("interact_goal", 0)


def eval_mode(label, temperature=None, greedy=False):
    cfg = load_config(CONFIG)
    sess = OnlineFouloideSession(cfg, seed=0, device=device)
    sess.resume(CKPT)
    sess.learner.observe = lambda *a, **k: None
    actor = sess.agent.actor

    def patched(rec, avail, greedy_flag=False):
        s = torch.as_tensor(rec, dtype=torch.float32, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        logits = actor.forward(s).squeeze(0)
        mask = torch.full_like(logits, float("-inf"))
        mask[torch.as_tensor(list(avail), dtype=torch.long, device=device)] = 0.0
        masked = logits + mask
        if greedy:
            return int(masked.argmax().item())
        t = temperature if temperature else 1.0
        return int(torch.distributions.Categorical(logits=masked / t).sample().item())

    actor.act_masked = lambda rec, avail, greedy=False: patched(rec, avail, greedy)
    events = Counter()
    acts = Counter()
    for _ in range(N):
        sess.step()
        events[sess.env.describe_transition()] += 1
        acts[sess.last_action] += 1
    g = events.get("interact_goal", 0)
    print(f"  {label:16s}: {g:5d} collectes ({1000*g/N:6.2f}/1000)  actions={dict(acts)}")
    return g


rnd = random_baseline()
print(f"=== ALÉATOIRE: {rnd} ({1000*rnd/N:.2f}/1000) ===\n")
print("=== POLICY APPRISE, par mode de sélection (DreamerV3 = sampling, pas argmax) ===")
eval_mode("greedy(argmax)", greedy=True)
eval_mode("sample T=1.0")
eval_mode("sample T=0.5", temperature=0.5)
eval_mode("sample T=0.3", temperature=0.3)
print(f"\n-> succès chantier: collecte (greedy ET sampled) >> aléatoire ({1000*rnd/N:.2f}/1000), argmax non dégénéré.")
