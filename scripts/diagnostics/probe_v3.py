"""Probe v3 — re-mesure le 'mur' sur le STACK DREAMERV3 FINAL (pas l'ancien RSSM).

Deux questions tranchées sur le checkpoint v3 réel :

  D1  Sur les états-ressource, le PRIOR d'imagination différencie-t-il le reward
      par action ? (INTERACT a-t-il le reward imaginé MAX ? l'actor met-il de la
      proba dessus ?) + AVANTAGE λ-return par action.

  D2  Le WM croit-il qu'on ne meurt JAMAIS ? -> distribution de continue_prob
      (P(épisode continue)). Si ~1 partout, alors le death-discount ne tronque
      jamais les returns -> 'ne rien faire' ≈ 'fourrager' en return long -> la
      policy n'a aucune raison de s'engager. (= l'hypothèse 'survie gratuite'.)

Adapté à l'état stochastique v3 : l'agent porte rssm_state={deter,stoch,logits},
l'actor/critic consomment feat=get_feat(state), imagine_batch(state, action) prend
le dict. (L'ancienne probe passait un tenseur h et casse sur v3.)
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config
from seedmind.training.imagination_actor_critic import symexp, _lambda_returns, _SYMLOG_CLAMP

CONFIG = os.environ.get("PROBE_CONFIG", "configs/micro_fouloide_online_homeostatic_rssm_v3.yaml")
CKPT = os.environ.get("PROBE_CKPT", "runs/rssm_v3_full_80k/checkpoint_online.pt")
N = int(os.environ.get("PROBE_N", "8000"))
MAXSTATES = 128
H = 15; GAMMA = 0.97; LAM = 0.95

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"config={CONFIG}\nckpt={CKPT}\ndevice={device}")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
# PROBE = on sonde un checkpoint FIGÉ : pas d'entraînement pendant le stepping
# (sinon le learner ré-entraîne le WM à chaque pas → fuite mémoire/OOM, et on ne
#  mesurerait plus le checkpoint mais un modèle qui dérive).
sess.learner.observe = lambda *a, **k: None
wm = sess.agent.world_model; actor = sess.agent.actor; critic = sess.agent.critic
reg = sess.env.registry
water_ids = set(reg.drive_signal_ids("hydration"))
food_ids = set(reg.drive_signal_ids("energy"))
actions = sess.agent.actions
nact = len(actions); INT = sess.agent.action_index["INTERACT"]
print(f"actions={actions} | INTERACT idx={INT} | water_ids={water_ids} food_ids={food_ids}")


def value_of(feat):
    if hasattr(critic, "value"):
        return critic.value(feat)
    return symexp(critic(feat).clamp(-_SYMLOG_CLAMP, _SYMLOG_CLAMP))


def cat_states(state_list):
    return {k: torch.cat([s[k] for s in state_list], dim=0) for k in state_list[0]}


@torch.no_grad()
def policy_lambda_return(state):
    """λ-return en continuant la POLICY depuis `state` (batch)."""
    feats = []; rewards = []; conts = []
    s = state
    for t in range(H):
        feat = wm.get_feat(s)
        a = actor.act(feat)
        s, _, r, _ = wm.imagine_batch(s, a)
        feats.append(feat); rewards.append(r)
        conts.append(wm.continue_prob(wm.get_feat(s)))
    boot = value_of(wm.get_feat(s))
    F = torch.stack(feats, 0); R = torch.stack(rewards, 0); C = torch.stack(conts, 0)
    B = R.shape[1]
    V = value_of(F.reshape(-1, F.shape[-1])).reshape(H, B)
    ret = _lambda_returns(R, V, boot, GAMMA, LAM, discount=GAMMA * C)
    return ret[0]


# --- collecte : états-ressource + échantillon général ---
res_states = {"water": [], "food": []}
all_cont = []   # continue_prob sur un échantillon général d'états
for i in range(N):
    sess.step()
    obs = sess.observation
    st = sess.agent.rssm_state
    if st is None:
        continue
    st = {k: v.detach().clone() for k, v in st.items()}
    # D2 général : continue_prob sur tout état visité (échantillonné)
    if i % 5 == 0:
        all_cont.append(float(wm.continue_prob(wm.get_feat(st)).item()))
    se = int(obs.get("standing_entity", -1)) if isinstance(obs, dict) else -1
    kind = "water" if se in water_ids else ("food" if se in food_ids else None)
    if kind and len(res_states[kind]) < MAXSTATES:
        res_states[kind].append(st)

# --- D2 : le WM croit-il qu'on meurt ? ---
print(f"\n{'='*64}\nD2 — CONTINUE PREDICTOR (P(épisode continue)) — 'survie gratuite ?'\n{'='*64}")
if all_cont:
    c = np.array(all_cont)
    print(f"  n={len(c)} états | mean continue_prob = {c.mean():.4f}")
    print(f"  min={c.min():.4f}  p5={np.percentile(c,5):.4f}  median={np.median(c):.4f}")
    print(f"  -> P(continue) < 0.99 dans {100*(c<0.99).mean():.1f}% des états ; < 0.90 dans {100*(c<0.90).mean():.1f}%")
    print(f"  INTERPRÉTATION : si ~1 partout, le WM n'imagine JAMAIS la mort →")
    print(f"  death-discount inerte → fourrager n'augmente pas le return long → MUR EXPLIQUÉ.")


# --- D1 : le prior différencie-t-il le reward par action ? ---
def report_d1(kind):
    data = res_states[kind]
    if not data:
        print(f"\n[{kind}] aucun état-ressource rencontré"); return
    S = cat_states(data); B = S["deter"].shape[0]
    feat0 = wm.get_feat(S)
    with torch.no_grad():
        Vh = value_of(feat0)
        probs = actor.distribution(feat0).probs.cpu().numpy()       # (B,nact)
        cont0 = wm.continue_prob(feat0).cpu().numpy()
        r1 = np.zeros((B, nact)); adv = np.zeros((B, nact))
        for a in range(nact):
            at = torch.full((B,), a, dtype=torch.long, device=device)
            s2, _, r_a, _ = wm.imagine_batch(S, at)
            cont = policy_lambda_return(s2)
            Q = r_a + GAMMA * cont
            r1[:, a] = r_a.cpu().numpy()
            adv[:, a] = (Q - Vh).cpu().numpy()
    others = [a for a in range(nact) if a != INT]
    print(f"\n{'='*64}\nD1 — SUR {kind.upper()} ({B} états)\n{'='*64}")
    print(f"  WM continue_prob ici : {cont0.mean():.4f}")
    print(f"  reward 1-pas imaginé : INTERACT={r1[:,INT].mean():+.4f} | max(autres)={r1[:,others].max(1).mean():+.4f} | moy(autres)={r1[:,others].mean():+.4f}")
    print(f"  -> INTERACT = reward imaginé MAX dans {100*(r1.argmax(1)==INT).mean():.0f}% des cas")
    print(f"  AVANTAGE λ-ret      : INTERACT={adv[:,INT].mean():+.4f} | max(autres)={adv[:,others].max(1).mean():+.4f}")
    print(f"  -> INTERACT a l'avantage MAX dans {100*(adv.argmax(1)==INT).mean():.0f}% | avantage>0 dans {100*(adv[:,INT]>0).mean():.0f}%")
    print(f"  actor proba          : INTERACT={probs[:,INT].mean():.3f} | max(autres)={probs[:,others].max(1).mean():.3f} | uniforme={1/nact:.3f}")
    print(f"  -> INTERACT = action la PLUS probable dans {100*(probs.argmax(1)==INT).mean():.0f}% des cas")


report_d1("water"); report_d1("food")
print("\n[done]")
