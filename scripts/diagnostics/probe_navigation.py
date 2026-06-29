"""Le WM supporte-t-il la NAVIGATION ? (test contrôlé, sans confound horizon/sparsité)

On place l'agent à 1 (ou 2) case(s) d'UNE cible, et on demande au WM la récompense
imaginée de séquences d'actions. Si « bouger-vers-la-cible puis INTERACT » domine
nettement → le WM propage move→cible→+1, donc la navigation EST apprenable et le
verrou est l'actor-critic (commit sur l'optimum local). Sinon → le WM ne supporte
pas la planification de navigation (verrou dynamique).
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config
from seedmind.envs.simple_grid_world import GOAL
from seedmind.envs.micro_fouloide_world import EMPTY

CONFIG = os.environ.get("PROBE_CONFIG", "configs/simple_grid_dense_binfix.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/w1_binfix_12k/checkpoint_online.pt")
GAMMA = 0.97

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
ag = sess.agent; wm = sess.agent.world_model; env = sess.env
A = {n: i for i, n in enumerate(ag.actions)}
INT = A["INTERACT"]


def setup(agent_rc, goal_rc):
    """Place une grille propre : agent en agent_rc, UNE cible en goal_rc."""
    env.grid[:] = EMPTY
    env.agent_pos = (int(agent_rc[0]), int(agent_rc[1]))
    env.grid[goal_rc[0], goal_rc[1]] = GOAL
    env.goal_pos = (int(goal_rc[0]), int(goal_rc[1]))
    ag.reset_state(); ag._prev_action_idx = A["WAIT"]
    obs = env.observe()
    lat = ag.encoder.encode_tensor(obs)
    ag.advance((lat.squeeze(0) if lat.dim() == 2 else lat).detach().cpu().numpy().astype(np.float32))
    return {k: v.detach().clone() for k, v in ag.rssm_state.items()}


@torch.no_grad()
def seq_reward(state, actions):
    """Récompense imaginée discountée d'une séquence d'actions (déterministe)."""
    s = {k: v.clone() for k, v in state.items()}
    total = 0.0
    for t, a in enumerate(actions):
        s = wm.img_step(s, torch.tensor([a], device=device), sample=False)
        r = float(wm.reward_value(wm.get_feat(s)).item())
        total += (GAMMA ** t) * r
    return total


# agent au centre, cible 1 case dans chaque direction
center = (3, 2)
cases = {
    "cible 1 case À DROITE":  ((3, 2), (3, 3), "MOVE_RIGHT"),
    "cible 1 case À GAUCHE":  ((3, 3), (3, 2), "MOVE_LEFT"),
    "cible 1 case EN BAS":    ((2, 3), (3, 3), "MOVE_DOWN"),
    "cible 2 cases À DROITE":  ((3, 1), (3, 3), "MOVE_RIGHT"),
}

for name, (arc, grc, toward) in cases.items():
    st = setup(arc, grc)
    dist = abs(arc[0] - grc[0]) + abs(arc[1] - grc[1])
    print(f"\n=== {name} (dist {dist}) ===")
    # séquences candidates de longueur = dist+1 (naviguer puis INTERACT)
    seqs = {
        f"{toward}×{dist} puis INTERACT": [A[toward]] * dist + [INT],
        "INTERACT sur place (×%d)" % (dist + 1): [INT] * (dist + 1),
        "rester (WAIT) puis INTERACT": [A["WAIT"]] * dist + [INT],
    }
    # une direction "à l'opposé"
    opp = {"MOVE_RIGHT": "MOVE_LEFT", "MOVE_LEFT": "MOVE_RIGHT", "MOVE_DOWN": "MOVE_UP"}[toward]
    seqs[f"{opp}×{dist} (à l'opposé) puis INTERACT"] = [A[opp]] * dist + [INT]
    scored = {k: seq_reward(st, v) for k, v in seqs.items()}
    best = max(scored, key=scored.get)
    for k, v in sorted(scored.items(), key=lambda kv: -kv[1]):
        mark = " <-- MEILLEURE" if k == best else ""
        print(f"   {v:+.3f}  {k}{mark}")
    ok = best.startswith(toward)
    print(f"   -> le WM préfère naviguer vers la cible : {'OUI ✅' if ok else 'NON ❌'}")

print("\n-> si 'naviguer puis INTERACT' gagne partout : le WM SAIT naviguer, verrou = actor-critic.")
print("-> sinon : le WM ne propage pas move->cible->reward (verrou dynamique multi-pas).")
