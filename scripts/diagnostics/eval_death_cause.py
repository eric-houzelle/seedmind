"""Diagnostic % critique sur le checkpoint du run long (post-budget)."""
import sys, os, collections
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
res = sess.resume(CKPT)
print(f"repris: {res.get('env_steps','?')} steps vécus, device={device}")

events = collections.Counter(); deaths = 0
health_samples = []; critical_steps = 0; prev = None
hbd, ebd = [], []
for i in range(N):
    info = sess.step()
    ev = str(info.get("event", "?")); events[ev] += 1
    h = float(info.get("health", 1.0)); health_samples.append(h)
    d = info.get("drives", {})
    if min(d.get("energy", 1), d.get("hydration", 1)) <= 0.14: critical_steps += 1
    if info.get("dead", False):
        deaths += 1
        if prev: hbd.append(float(prev.get("health",1))); ebd.append(str(prev.get("event","?")))
    prev = info

hs = np.array(health_samples)
print(f"\n=== {N} steps · morts={deaths} (1/{N//max(deaths,1)}) ===")
print("Top events:", events.most_common(6))
print(f"steps en drive CRITIQUE : {critical_steps} ({100*critical_steps/N:.1f}%)")
print(f"santé moy {hs.mean():.3f} | %<0.30 {100*np.mean(hs<0.30):.1f}% | %plancher {100*np.mean(np.abs(hs-0.20)<0.01):.1f}% | %>0.6 {100*np.mean(hs>0.6):.1f}%")
if ebd: print(f"event avant mort: {collections.Counter(ebd).most_common()} | santé avant mort {np.mean(hbd):.3f}")
print("\nRAPPELS % critique: original 85.7% | comfort_wide 92.6% | mapmem-300/60k 93.3%")
