"""Probe : le WM imagine-t-il que INTERACT (sur une ressource) rapporte, et
l'actor met-il de la proba dessus ?

Sur les états où l'agent est SUR une case eau/bouffe, on compare par action :
  - reward 1-pas imaginé par le WM   (le WM "sait"-il que fourrager paie ?)
  - proba de l'actor                  (la policy agit-elle sur ce signal ?)
Verdict : si WM_reward[INTERACT] n'est PAS le plus haut -> le WM est le verrou
(rien à apprendre). S'il l'est mais actor_prob[INTERACT] est bas -> l'actor/le
signal d'avantage est le verrou (le signal existe mais ne sharpen pas la policy).
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = "configs/micro_fouloide_online_homeostatic_rssm.yaml"
CKPT = "runs/rssm_mapmem_long_150k/checkpoint_online.pt"
N = 8000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)

reg = sess.env.registry
water_ids = set(reg.drive_signal_ids("hydration"))
food_ids = set(reg.drive_signal_ids("energy"))
actions = sess.agent.actions
nact = len(actions)
INT = sess.agent.action_index["INTERACT"]
print(f"actions={actions} | INTERACT idx={INT} | water_ids={water_ids} food_ids={food_ids}")

on_res = {"water": [], "food": []}   # listes de (probs, wm_rewards) par action

for i in range(N):
    info = sess.step()
    obs = sess.observation
    se = int(obs.get("standing_entity", -1)) if isinstance(obs, dict) else -1
    kind = "water" if se in water_ids else ("food" if se in food_ids else None)
    if kind is None:
        continue
    h = sess.agent.h
    if h is None:
        continue
    hb = h if h.dim() == 2 else h.unsqueeze(0)
    with torch.no_grad():
        probs = sess.agent.actor.distribution(hb).probs[0].detach().cpu().numpy()
        wm_r = np.array([
            float(sess.agent.world_model.imagine_batch(
                hb, torch.tensor([a], device=hb.device))[2].item())
            for a in range(nact)
        ])
    on_res[kind].append((probs, wm_r))

def report(kind):
    data = on_res[kind]
    if not data:
        print(f"\n[{kind}] aucun état sur ressource rencontré"); return
    P = np.stack([d[0] for d in data]); R = np.stack([d[1] for d in data])
    n = len(data)
    others = [a for a in range(nact) if a != INT]
    print(f"\n=== SUR {kind.upper()} ({n} états) ===")
    print(f"  WM reward imaginé  : INTERACT={R[:,INT].mean():+.4f} | max(autres)={R[:,others].max(1).mean():+.4f} | moy(autres)={R[:,others].mean():+.4f}")
    rank = (R.argmax(1) == INT).mean()
    print(f"  -> INTERACT est l'action de reward imaginé MAX dans {100*rank:.0f}% des cas")
    print(f"  actor proba        : INTERACT={P[:,INT].mean():.3f} | max(autres)={P[:,others].max(1).mean():.3f} | uniforme={1/nact:.3f}")
    arank = (P.argmax(1) == INT).mean()
    print(f"  -> INTERACT est l'action la PLUS probable dans {100*arank:.0f}% des cas")

report("water"); report("food")
