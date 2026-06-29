"""Probe W1 — WM ou ACTOR ? (le verrou de la policy, isolé sur un monde trivial)

L'agent spamme INTERACT immobile au lieu de naviguer vers la cible. Deux causes
possibles, qu'on tranche ici sur le checkpoint W1 :

  Q1/Q2 (WM)   : SUR la cible, le WM imagine-t-il INTERACT = gros reward, et au
                 max ? HORS cible, INTERACT donne-t-il ~0 ? → le WM conditionne-t-il
                 correctement le reward sur (état=sur-cible × action=INTERACT) ?
  Q3 (ACTOR)   : la proba de l'actor sur INTERACT est-elle ~la même SUR et HORS
                 cible (= commit aveugle) ?

Verdict :
  • WM correct (on-goal INTERACT ≫ off-goal) MAIS actor spamme partout → le bug est
    dans l'ACTOR / l'assignation de crédit en imagination.
  • WM donne INTERACT élevé PARTOUT → le WM ne conditionne pas le reward (bug WM).

Collecte par actions ALÉATOIRES (pour couvrir on-goal ET off-goal ; la policy
apprise, dégénérée, ne bougerait pas).
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = os.environ.get("PROBE_CONFIG", "configs/simple_grid_online_rssm_v3.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/w1_simplegrid_15k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "8000"))
MAXSTATES = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N} device={device}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent; wm = agent.world_model; actor = agent.actor
from seedmind.envs.simple_grid_world import GOAL
nact = len(agent.actions); INT = agent.action_index["INTERACT"]
print(f"actions={agent.actions} | INTERACT={INT} | GOAL id={GOAL}")

rng = np.random.default_rng(0)
on_goal, off_goal = [], []
agent.reset_state()
obs = sess.env.reset()
for _ in range(N):
    latent = agent.encoder.encode_tensor(obs)
    latent_np = latent.squeeze(0).detach().cpu().numpy().astype(np.float32) if latent.dim() == 2 else latent.detach().cpu().numpy().astype(np.float32)
    agent.advance(latent_np)
    st = agent.rssm_state
    if st is not None:
        st = {k: v.detach().clone() for k, v in st.items()}
        if int(obs.get("standing_entity", -1)) == GOAL:
            if len(on_goal) < MAXSTATES:
                on_goal.append(st)
        elif len(off_goal) < MAXSTATES and rng.random() < 0.1:
            off_goal.append(st)
    a_idx = int(rng.integers(nact))
    agent._prev_action_idx = a_idx
    obs, _, done, _ = sess.env.step(agent.actions[a_idx])
    if done:
        agent.reset_state(); obs = sess.env.reset()
    if len(on_goal) >= MAXSTATES and len(off_goal) >= MAXSTATES:
        break


def cat(sl):
    return {k: torch.cat([s[k] for s in sl], dim=0) for k in sl[0]}


@torch.no_grad()
def report(name, sl):
    if not sl:
        print(f"\n[{name}] aucun état collecté"); return None
    S = cat(sl); B = S["deter"].shape[0]
    feat = wm.get_feat(S)
    probs = actor.distribution(feat).probs.cpu().numpy()
    r = np.zeros((B, nact))
    for a in range(nact):
        at = torch.full((B,), a, dtype=torch.long, device=device)
        r[:, a] = wm.imagine_batch(S, at)[2].cpu().numpy()
    others = [a for a in range(nact) if a != INT]
    print(f"\n=== {name} ({B} états) ===")
    print(f"  WM reward imaginé : INTERACT={r[:,INT].mean():+.4f} | max(autres)={r[:,others].max(1).mean():+.4f} | moy(autres)={r[:,others].mean():+.4f}")
    print(f"  -> INTERACT = reward imaginé MAX dans {100*(r.argmax(1)==INT).mean():.0f}% des états")
    print(f"  actor proba INTERACT = {probs[:,INT].mean():.3f}  (uniforme={1/nact:.3f})")
    return r[:, INT].mean(), probs[:, INT].mean()


on = report("SUR LA CIBLE (on-goal)", on_goal)
off = report("HORS CIBLE (off-goal)", off_goal)

print(f"\n{'='*60}\nVERDICT\n{'='*60}")
if on and off:
    r_on, p_on = on; r_off, p_off = off
    print(f"  WM: reward INTERACT  on-goal={r_on:+.3f}  vs off-goal={r_off:+.3f}  (écart={r_on-r_off:+.3f})")
    print(f"  ACTOR: proba INTERACT on-goal={p_on:.3f}  vs off-goal={p_off:.3f}  (écart={p_on-p_off:+.3f})")
    wm_ok = (r_on - r_off) > 0.15
    actor_blind = abs(p_on - p_off) < 0.05
    print(f"  -> WM conditionne le reward sur l'état : {'OUI' if wm_ok else 'NON'}")
    print(f"  -> ACTOR commit aveugle (même proba on/off) : {'OUI' if actor_blind else 'NON'}")
    if wm_ok and actor_blind:
        print("  => BUG dans l'ACTOR / l'assignation de crédit (le WM sait, l'actor ignore).")
    elif not wm_ok:
        print("  => BUG dans le WM (ne conditionne pas le reward sur état×action).")
    else:
        print("  => mixte / à creuser.")
