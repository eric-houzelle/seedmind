"""Probe couche 5 (seedmind-10e.7) — OÙ le world model hallucine-t-il du retour ?

Diagnostic établi : la représentation est levée (embed 0.65, h 0.37), l'actor a
appris la structure conditionnelle, MAIS `imag_return ~2.6` alors que la collecte
RÉELLE est nulle → MODEL-EXPLOITATION : l'actor optimise un retour *imaginé* que le
WM hallucine et qui ne transfère pas au réel (cf. `couche3-model-exploitation`).

Ce probe LOCALISE la source de l'hallucination en pilotant l'actor UNE fois dans
l'env réel, puis en décomposant l'écart imaginé↔réel en deux niveaux :

  [A] Calibration one-step de la reward-head, SUR la trajectoire réellement visitée
      (teacher-forced). Pour chaque transition réelle (post_t --a_t--> post_{t+1}, r) :
        - r_real  : reward de l'env ;
        - r_post  : reward_value(feat(post_{t+1}))  — la reward-head lue sur le VRAI
                    latent suivant (« la head est-elle juste LÀ OÙ l'agent va ? ») ;
        - r_prior : reward_value(feat(img_step(post_t, a_t))) — la reward-head lue
                    sur le prior 1-pas (ce que l'imagination produit à son 1er pas).
      Si r_post ≈ r_real mais r_prior s'écarte → la dérive naît DÈS le 1er pas de prior.
      Si r_post lui-même s'écarte de r_real → c'est la reward-head qui est mal calibrée
      (elle « fire » hors-cible / rate les vraies collectes). On mesure aussi la dérive
      dynamique 1-pas ||feat(prior) − feat(post_{t+1})||.

  [B] Rollout imaginé LIBRE apparié (H pas) sous l'actor — le vrai model-exploitation.
      Depuis les mêmes états de départ réels : retour actualisé imaginé G_imag (prior,
      reward-head, exactement comme l'entraînement) vs retour actualisé réel G_real
      (l'actor exécuté dans l'env). Courbes reward par pas (le reward imaginé MONTE-t-il
      avec la profondeur = signature de dérive composée hors-variété ?), fraction de
      pas « goal-like » (reward>0.3) imaginée vs collectes réelles, et courbe
      d'incertitude du WM (corrèle-t-elle avec le reward imaginé ? → levier pénalité).

reward_learning == reward extrinsèque pur ici (drive/resource off), et la reward-head
RSSM est entraînée SANS curiosité → l'imaginé se compare directement au reward de l'env.

Usage (CPU obligatoire — MPS fuit sur le RSSM) :
  EVAL_CONFIG=configs/simple_grid_sparse_obsrecon.yaml \
  EVAL_CKPT=runs/w1_obsrecon_12k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_imag_real_gap.py
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config  # noqa: E402
from seedmind.envs.simple_grid_world import GOAL  # noqa: E402

CONFIG = os.environ.get("EVAL_CONFIG", "configs/simple_grid_sparse_obsrecon.yaml")
CKPT = os.environ.get("EVAL_CKPT", "runs/w1_obsrecon_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "6000"))          # pas réels pilotés par l'actor
H = int(os.environ.get("PROBE_H", "15"))            # horizon d'imagination (= config)
GAMMA = float(os.environ.get("PROBE_GAMMA", "0.97"))
MAXSTARTS = int(os.environ.get("PROBE_STARTS", "256"))
GOAL_LIKE = float(os.environ.get("PROBE_GOAL_LIKE", "0.3"))  # seuil reward « cible-like »
device = torch.device("cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N}  H={H}  gamma={GAMMA}  device={device}\n")

cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
agent = sess.agent
wm = agent.world_model
actor = agent.actor
assert hasattr(wm, "get_feat"), "ce probe cible le RSSM stochastique (get_feat manquant)"
nact = len(agent.actions)
INT = agent.action_index.get("INTERACT")


def clone_state(st):
    return {k: v.detach().clone() for k, v in st.items()}


def batch_states(states):
    """Liste de State (batch=1) -> un State batché (batch=len)."""
    return {k: torch.cat([s[k] for s in states], dim=0) for k in states[0]}


@torch.no_grad()
def unc_of(feat):
    return torch.nn.functional.softplus(wm.uncertainty_head(feat)).squeeze(-1)


# ---------------------------------------------------------------------------
# 1) UNE trajectoire réelle pilotée par l'actor (échantillonné, comme le déploiement
#    DreamerV3 et comme l'imagination). On enregistre post_t, a_t, r_real_t, event.
# ---------------------------------------------------------------------------
posts, acts, r_real, is_goal, boundary = [], [], [], [], []
events = Counter()
agent.reset_state()
obs = sess.env.reset()
with torch.no_grad():
    for _ in range(N):
        latent = agent.encoder.encode_tensor(obs)
        lnp = (latent.squeeze(0) if latent.dim() == 2 else latent)
        agent.advance(lnp.detach().cpu().numpy().astype(np.float32))
        st = agent.rssm_state
        if st is None:
            continue
        feat = wm.get_feat(st)
        a = int(actor.act(feat)[0].item())
        posts.append(clone_state(st))
        acts.append(a)
        agent._prev_action_idx = a
        obs, rew, done, info = sess.env.step(agent.actions[a])
        ev = info.get("event", "?")
        events[ev] += 1
        r_real.append(float(rew))
        is_goal.append(1 if ev == "interact_goal" else 0)
        boundary.append(1 if done else 0)
        if done:
            agent.reset_state()
            obs = sess.env.reset()

M = len(posts)
a_t = torch.as_tensor(acts, dtype=torch.long, device=device)
r_real = np.asarray(r_real, dtype=np.float64)
is_goal = np.asarray(is_goal, dtype=np.int64)
boundary = np.asarray(boundary, dtype=np.int64)
print(f"trajectoire réelle: {M} transitions | events={dict(events)}")
print(f"collecte RÉELLE (actor échantillonné): {is_goal.sum()} "
      f"({1000*is_goal.sum()/max(1,M):.2f}/1000)  reward réel moyen/pas={r_real.mean():+.4f}\n")

# ---------------------------------------------------------------------------
# [A] Calibration one-step de la reward-head sur la trajectoire RÉELLE (teacher-forced).
#     r_prior_t = reward_value(feat(img_step(post_t, a_t)))   (ce qu'imagine le 1er pas)
#     r_post_t  = reward_value(feat(post_{t+1}))              (head sur le VRAI latent suivant)
# ---------------------------------------------------------------------------
r_prior = np.zeros(M, dtype=np.float64)
r_post = np.zeros(M, dtype=np.float64)
drift = np.zeros(M, dtype=np.float64)          # ||feat(prior_1pas) - feat(post_{t+1})||
CH = 512
with torch.no_grad():
    for i in range(0, M, CH):
        j = min(M, i + CH)
        sl = batch_states(posts[i:j])
        prior, feat_pr, r_pr, _ = wm.imagine_batch(sl, a_t[i:j])
        r_prior[i:j] = r_pr.cpu().numpy()
        # post_{t+1} = posts[t+1] (déjà filtré avec a_t) ; dernier / avant reset: pas de suivant
        nxt_idx = list(range(i + 1, j + 1))
        valid = [(k, ni) for k, ni in zip(range(i, j), nxt_idx)
                 if ni < M and boundary[k] == 0]
        if valid:
            ks = [k for k, _ in valid]
            nis = [ni for _, ni in valid]
            post_next = batch_states([posts[ni] for ni in nis])
            feat_next = wm.get_feat(post_next)
            r_post_v = wm.reward_value(feat_next).cpu().numpy()
            for m, k in enumerate(ks):
                r_post[k] = r_post_v[m]
            # dérive dynamique 1-pas : le prior atterrit-il là où le posterior est ?
            off = i
            fpr = feat_pr[[k - off for k in ks]]
            drift_v = (fpr - feat_next).norm(dim=-1).cpu().numpy()
            for m, k in enumerate(ks):
                drift[k] = drift_v[m]

has_next = (boundary == 0)
has_next[-1] = False  # pas de post_{t+1} pour la dernière transition

print("=" * 72)
print("[A] Calibration ONE-STEP de la reward-head (teacher-forced sur le RÉEL)")
print("=" * 72)
print(f"  reward réel moyen/pas              : {r_real.mean():+.4f}")
print(f"  reward imaginé 1-pas (prior)       : {r_prior.mean():+.4f}   "
      f"[écart prior−réel = {r_prior.mean()-r_real.mean():+.4f}]")
print(f"  reward teacher-forced (posterior)  : {r_post[has_next].mean():+.4f}   "
      f"[écart post−réel  = {r_post[has_next].mean()-r_real[has_next].mean():+.4f}]")
print(f"  dérive dynamique 1-pas |feat_prior−feat_post| : {drift[has_next].mean():.3f}")

print("\n  -- décomposition par type de transition réelle --")
for name, mask in [("COLLECTE réelle (r=+1)", is_goal == 1),
                   ("pénalité réelle (r<0)", (is_goal == 0) & (r_real < 0)),
                   ("neutre (r≈0)", (is_goal == 0) & (np.abs(r_real) < 1e-6))]:
    n = int(mask.sum())
    if n == 0:
        print(f"    {name:24s}: n=0")
        continue
    mp = mask & has_next
    print(f"    {name:24s}: n={n:5d} | r_real={r_real[mask].mean():+.3f}"
          f"  r_prior(1-pas)={r_prior[mask].mean():+.3f}"
          f"  r_post={r_post[mp].mean() if mp.sum() else float('nan'):+.3f}")
# fuite = masse de reward positif prédite là où le réel est ≤ 0
off_goal = (is_goal == 0)
print(f"\n  FUITE hors-cible : sur les transitions r_real≤0 (n={int(off_goal.sum())}), "
      f"reward imaginé 1-pas moyen = {r_prior[off_goal].mean():+.4f}  "
      f"(>0 = la head sur-prédit hors-cible)")

# ---------------------------------------------------------------------------
# [B] Rollout imaginé LIBRE apparié (H pas) — le model-exploitation composé.
#     Départs = états réels espacés, avec H pas réels contigus disponibles (pas de reset).
# ---------------------------------------------------------------------------
cand = [t for t in range(M - H) if boundary[t:t + H].sum() == 0]
if len(cand) > MAXSTARTS:
    step = len(cand) / MAXSTARTS
    cand = [cand[int(k * step)] for k in range(MAXSTARTS)]
starts = batch_states([posts[t] for t in cand])
S = len(cand)

imag_r = np.zeros((H, S), dtype=np.float64)     # reward imaginé par pas
imag_u = np.zeros((H, S), dtype=np.float64)     # incertitude WM imaginée par pas
with torch.no_grad():
    state = {k: v.clone() for k, v in starts.items()}
    for h in range(H):
        feat = wm.get_feat(state)
        a = actor.act(feat)
        state, feat_next, r, unc = wm.imagine_batch(state, a)
        imag_r[h] = r.cpu().numpy()
        imag_u[h] = unc.cpu().numpy()

# reward réel apparié : continuation contiguë r_real[t : t+H] pour chaque départ t
real_r = np.stack([r_real[t:t + H] for t in cand], axis=1)          # (H, S)
real_goal = np.stack([is_goal[t:t + H] for t in cand], axis=1)      # (H, S)

disc = GAMMA ** np.arange(H)
G_imag = (disc[:, None] * imag_r).sum(0)        # (S,) retour actualisé imaginé
G_real = (disc[:, None] * real_r).sum(0)        # (S,) retour actualisé réel
imag_goal_like = (imag_r > GOAL_LIKE).mean()    # fraction de pas imaginés « cible-like »
real_goal_frac = real_goal.mean()

print("\n" + "=" * 72)
print(f"[B] Rollout imaginé LIBRE apparié — {S} départs réels, H={H} pas sous l'actor")
print("=" * 72)
print(f"  G_imag (retour actualisé IMAGINÉ) : {G_imag.mean():+.3f}  (±{G_imag.std():.3f})")
print(f"  G_real (retour actualisé RÉEL)    : {G_real.mean():+.3f}  (±{G_real.std():.3f})")
print(f"  => ÉCART imag−réel                 : {G_imag.mean()-G_real.mean():+.3f}")
print(f"  fraction de pas « cible-like » (reward>{GOAL_LIKE}) : "
      f"imaginé={100*imag_goal_like:.1f}%   réel(collectes)={100*real_goal_frac:.1f}%")

print("\n  reward par pas (0..H-1) — la courbe imaginée monte-t-elle avec la profondeur ?")
print(f"    imaginé : {np.array2string(imag_r.mean(1), precision=3, floatmode='fixed', max_line_width=200)}")
print(f"    réel    : {np.array2string(real_r.mean(1), precision=3, floatmode='fixed', max_line_width=200)}")
print(f"    incert. : {np.array2string(imag_u.mean(1), precision=3, floatmode='fixed', max_line_width=200)}")

# corrélation incertitude ↔ reward imaginé (levier « pénaliser l'imagination incertaine »)
fu, fr = imag_u.reshape(-1), imag_r.reshape(-1)
if fu.std() > 1e-9 and fr.std() > 1e-9:
    c_ur = float(np.corrcoef(fu, fr)[0, 1])
else:
    c_ur = 0.0
print(f"\n  corr(incertitude imaginée, reward imaginé) = {c_ur:+.3f}   "
      f"(>0 = le WM est INCERTAIN là où il promet du reward → pénalité utile)")

# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("LECTURE / localisation :")
print("  • r_post ≈ r_real MAIS G_imag ≫ G_real et courbe imaginée QUI MONTE avec la")
print("    profondeur ⇒ hallucination = DÉRIVE PRIOR composée (l'imagination sort de la")
print("    variété réelle) ⇒ leviers CPU : horizon court, imaginer depuis le posterior,")
print("    pénaliser l'incertitude, resserrer la KL.")
print("  • r_prior/r_post ≫ r_real DÈS 1 pas ⇒ reward-head mal calibrée (fire hors-cible)")
print("    ⇒ leviers : reward_vmax / twohot bins / plus de données de reward.")
print("  • fraction « cible-like » imaginée ≫ collectes réelles ⇒ le WM hallucine des")
print("    contacts-cible fréquents que l'actor exploite mais qui n'existent pas.")
