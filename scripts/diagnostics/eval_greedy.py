"""Test discriminant : l'agent agit-il mieux en GLOUTON (argmax) qu'en échantillonnant ?

Si oui -> la policy a appris une préférence utile mais l'exécute trop au hasard
(fix = agir greedy au déploiement). Si non -> la policy n'a rien appris d'utile.
Compare % critique sampled vs greedy sur le même checkpoint.
"""
import sys, os, collections
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = "configs/micro_fouloide_online_homeostatic_rssm.yaml"
CKPT = "runs/rssm_mapmem_long_150k/checkpoint_online.pt"
N = 6000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def run(greedy: bool):
    cfg = load_config(CONFIG)
    sess = OnlineFouloideSession(cfg, seed=0, device=device)
    sess.resume(CKPT)
    if greedy:
        actor = sess.agent.actor
        orig = actor.act_masked
        actor.act_masked = lambda sv, ai, greedy=False: orig(sv, ai, greedy=True)
    events = collections.Counter(); deaths = 0; crit = 0; blocked = 0
    hs = []
    for i in range(N):
        info = sess.step()
        ev = str(info.get("event", "?")); events[ev] += 1
        d = info.get("drives", {})
        if min(d.get("energy",1), d.get("hydration",1)) <= 0.14: crit += 1
        if ev == "move_blocked": blocked += 1
        hs.append(float(info.get("health",1)))
        if info.get("dead", False): deaths += 1
    hs = np.array(hs)
    label = "GREEDY (argmax)" if greedy else "SAMPLED (actuel)"
    print(f"\n=== {label} · {N} steps ===")
    print(f"  % critique     : {100*crit/N:.1f}%")
    print(f"  morts          : {deaths} (1/{N//max(deaths,1)})")
    print(f"  move_blocked   : {100*blocked/N:.1f}%")
    print(f"  santé moy {hs.mean():.3f} | %>0.6 {100*np.mean(hs>0.6):.1f}%")
    print(f"  eau {events.get('interact_water',0)} bouf {events.get('interact_food',0)} | move_ok {events.get('move_ok',0)}")

run(greedy=False)
run(greedy=True)
print("\n(rappel sampled @130k précédent: 82.9% critique)")
