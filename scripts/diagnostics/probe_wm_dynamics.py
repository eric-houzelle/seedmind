"""Confirmation directe couche 3 — le WM se trompe-t-il LÀ où le contrôleur abuse ?

Hypothèse model-exploitation : le WM est bon sur sa distribution (on-goal INTERACT)
mais imprécis ailleurs (déplacements, bords), et un contrôleur qui maximise fonce
dans ces zones d'erreur (50% de move_blocked observés).

Test : de vraies transitions (état_t, action, état_réel_{t+1}), on compare la
PRÉDICTION du WM (img_step → décodeur) à la réalité, et le REWARD prédit au reward
réel — SÉPARÉ PAR ÉVÈNEMENT. Lecture décisive :
  - si move_blocked a une erreur de prédiction BIEN plus haute que move_ok / interact
    (le WM 'croit' avoir bougé alors qu'on s'est cogné) ET/OU prédit un reward ≥ 0
    pour des coups bloqués → MODEL EXPLOITATION CONFIRMÉE.
  - si le WM prédit bien partout → c'est autre chose, on ne construit pas la pénalité
    d'incertitude pour rien.
"""
import sys, os
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config

CONFIG = os.environ.get("PROBE_CONFIG", "configs/simple_grid_dense_rssm_v3.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/w1_dense_fixed_12k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "6000"))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\nN={N} device={device}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
sess.learner.observe = lambda *a, **k: None
ag = sess.agent; wm = ag.world_model
nact = len(ag.actions)

# accumulateurs par évènement : (erreur de prédiction d'état, reward prédit, reward réel)
by_event = defaultdict(lambda: {"err": [], "pred_r": [], "real_r": []})
rng = np.random.default_rng(0)
ag.reset_state(); obs = sess.env.reset()


def embed_np(o):
    e = ag.encoder.encode_tensor(o)
    return e.squeeze(0) if e.dim() == 2 else e


with torch.no_grad():
    for _ in range(N):
        emb = embed_np(obs)                     # embedding réel de obs_t
        ag.advance(emb.detach().cpu().numpy().astype(np.float32))
        st = ag.rssm_state
        if st is None:
            ai = int(rng.integers(nact)); ag._prev_action_idx = ai
            obs, _, d, _ = sess.env.step(ag.actions[ai])
            if d: ag.reset_state(); obs = sess.env.reset()
            continue
        ai = int(rng.integers(nact))
        at = torch.tensor([ai], device=device)
        prior = wm.img_step(st, at, sample=False)          # prédiction d'état après l'action
        feat = wm.get_feat(prior)
        pred_next_emb = wm.decoder(feat).squeeze(0)        # embedding prédit du prochain état
        pred_r = float(wm.reward_value(feat).item())
        ag._prev_action_idx = ai
        next_obs, real_r, done, info = sess.env.step(ag.actions[ai])
        real_next_emb = embed_np(next_obs)
        err = float(((pred_next_emb - real_next_emb) ** 2).mean().item())
        ev = info.get("event", "?")
        by_event[ev]["err"].append(err)
        by_event[ev]["pred_r"].append(pred_r)
        by_event[ev]["real_r"].append(real_r)
        obs = next_obs
        if done: ag.reset_state(); obs = sess.env.reset()

print(f"\n{'évènement':16s} | {'n':>5s} | {'err prédiction état':>20s} | {'reward prédit':>13s} | {'reward réel':>11s}")
print("-" * 78)
# référence: erreur sur move_ok (déplacement valide, sur-distribution)
order = sorted(by_event, key=lambda e: -len(by_event[e]["err"]))
for ev in order:
    d = by_event[ev]
    n = len(d["err"])
    if n < 5:
        continue
    print(f"{ev:16s} | {n:5d} | {np.mean(d['err']):20.4f} | {np.mean(d['pred_r']):+13.4f} | {np.mean(d['real_r']):+11.4f}")

print("\n=== VERDICT ===")
ok = np.mean(by_event['move_ok']['err']) if by_event['move_ok']['err'] else None
bl = np.mean(by_event['move_blocked']['err']) if by_event['move_blocked']['err'] else None
if ok and bl:
    print(f"  erreur prédiction : move_ok={ok:.4f}  vs  move_blocked={bl:.4f}  (ratio {bl/ok:.1f}×)")
    blr_pred = np.mean(by_event['move_blocked']['pred_r']); blr_real = np.mean(by_event['move_blocked']['real_r'])
    print(f"  move_blocked : reward prédit={blr_pred:+.4f} vs réel={blr_real:+.4f}")
    if bl > 1.5 * ok:
        print("  => Le WM se trompe BIEN PLUS sur les coups bloqués (il 'croit' avoir bougé).")
    if blr_pred > blr_real + 0.02:
        print("  => Le WM SUR-PRÉDIT le reward des coups bloqués → un maximiseur les choisit.")
    print("  => si l'un des deux est vrai : MODEL EXPLOITATION CONFIRMÉE.")
