"""Probe couche 3 — l'imagination HALLUCINE-t-elle ? (dérive sur l'horizon)

L'actor optimise un imag_return qui grimpe à 4.5 alors que le réel collecte 0.
Hypothèse : sur l'horizon 15, l'état latent imaginé quitte la variété réelle et le
WM y sur-prédit le reward (model exploitation). Ici, depuis des états RÉELS, on
déroule le WM H pas sous des politiques FIXES et on regarde la courbe de reward
imaginé par pas :

  - si 'toujours INTERACT' depuis un état HORS-cible donne un reward imaginé qui
    MONTE avec la profondeur → hallucination : le WM imagine qu'on finit sur des
    cibles / que INTERACT paie hors-cible → l'actor commit dessus → échoue en vrai.
  - si le reward imaginé reste plat/négatif hors-cible → pas de dérive, chercher
    ailleurs (critic / scale d'avantage).
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config
from seedmind.envs.simple_grid_world import GOAL

CONFIG = os.environ.get("PROBE_CONFIG", "configs/simple_grid_dense_rssm_v3.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/w1_dense_fixed_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "8000"))
H = int(os.environ.get("PROBE_H", "15"))
GAMMA = 0.97
MAXSTATES = 96

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nH={H} device={device}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent; wm = agent.world_model; actor = agent.actor
nact = len(agent.actions); INT = agent.action_index["INTERACT"]

# collecte d'états réels (on-goal / off-goal) sous actions aléatoires
rng = np.random.default_rng(0)
on, off = [], []
agent.reset_state(); obs = sess.env.reset()
for _ in range(N):
    latent = agent.encoder.encode_tensor(obs)
    lnp = latent.squeeze(0).detach().cpu().numpy().astype(np.float32) if latent.dim() == 2 else latent.detach().cpu().numpy().astype(np.float32)
    agent.advance(lnp)
    st = agent.rssm_state
    if st is not None:
        st = {k: v.detach().clone() for k, v in st.items()}
        (on if int(obs.get("standing_entity", -1)) == GOAL else off).append(st)
    a_idx = int(rng.integers(nact)); agent._prev_action_idx = a_idx
    obs, _, done, _ = sess.env.step(agent.actions[a_idx])
    if done:
        agent.reset_state(); obs = sess.env.reset()
    if len(on) >= MAXSTATES and len(off) >= MAXSTATES:
        break


def cat(sl):
    return {k: torch.cat([s[k][:1] for s in sl[:MAXSTATES]], dim=0) for k in sl[0]}


@torch.no_grad()
def rollout(start, policy):
    """Déroule H pas. policy: 'interact' | 'uniform' | 'actor'. Retourne reward/pas (H,)."""
    state = {k: v.clone() for k, v in start.items()}
    B = state["deter"].shape[0]
    per_step = []
    for t in range(H):
        if policy == "interact":
            a = torch.full((B,), INT, dtype=torch.long, device=device)
        elif policy == "uniform":
            a = torch.randint(0, nact, (B,), device=device)
        else:
            a = actor.act(wm.get_feat(state))
        state, _, r, _ = wm.imagine_batch(state, a)
        per_step.append(r.mean().item())
    return np.array(per_step)


for name, sl in [("HORS-CIBLE", off), ("SUR-CIBLE", on)]:
    if not sl:
        print(f"\n[{name}] aucun état"); continue
    S = cat(sl)
    print(f"\n{'='*64}\nDepuis états {name} ({S['deter'].shape[0]}) — reward imaginé par pas\n{'='*64}")
    for pol in ["interact", "uniform", "actor"]:
        rs = rollout(S, pol)
        disc = float(sum(GAMMA**t * rs[t] for t in range(H)))
        print(f"  [{pol:8s}] pas 0,1,2..: {np.array2string(rs[:6], precision=3, floatmode='fixed')} ... cumul γ={disc:+.3f}")

print("\n-> si [interact] HORS-CIBLE monte/cumule gros = HALLUCINATION confirmée (le WM imagine qu'INTERACT paie hors-cible).")
