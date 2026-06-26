"""Probe avantage λ-return : ce que l'actor optimise VRAIMENT par action.

Sur des états-eau, pour chaque action a :
  Q(h,a) = r(h,a) + γ · λ-return( en continuant la POLICY depuis h' )
  advantage(a) = Q(h,a) − V(h)
Si le reward 1-pas favorise INTERACT mais l'AVANTAGE λ-return ne le favorise pas
(voire le défavorise) → le crédit multi-pas est le bug : la continuation avec la
policy (mauvaise) noie l'avantage immédiat. Réutilise symexp/clamp/_lambda_returns
de l'entraînement pour la fidélité.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.argv = ["eval"]
from scripts.run_fouloide_online import OnlineFouloideSession, load_config
from seedmind.training.imagination_actor_critic import symexp, _lambda_returns, _SYMLOG_CLAMP

CONFIG = "configs/micro_fouloide_online_homeostatic_rssm.yaml"
CKPT = "runs/rssm_twohot_60k/checkpoint_online.pt"
N = 6000
H = 15; GAMMA = 0.97; LAM = 0.95; MAXSTATES = 96

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cfg = load_config(CONFIG)
sess = OnlineFouloideSession(cfg, seed=0, device=device)
sess.resume(CKPT)
wm = sess.agent.world_model; actor = sess.agent.actor; critic = sess.agent.critic
reg = sess.env.registry
water_ids = set(reg.drive_signal_ids("hydration"))
nact = len(sess.agent.actions); INT = sess.agent.action_index["INTERACT"]

def value_of(h):
    if hasattr(critic, 'value'):
        return critic.value(h)
    return symexp(critic(h).clamp(-_SYMLOG_CLAMP, _SYMLOG_CLAMP))

@torch.no_grad()
def policy_lambda_return(h0):
    h = h0; B = h.shape[0]; D = h.shape[1]
    states=[]; rewards=[]
    for t in range(H):
        a = actor.act(h)
        h2,_,r,_ = wm.imagine_batch(h, a)
        states.append(h); rewards.append(r); h = h2
    boot = value_of(h)
    S = torch.stack(states,0); R = torch.stack(rewards,0)
    V = value_of(S.reshape(-1,D)).reshape(H,B)
    ret = _lambda_returns(R, V, boot, GAMMA, LAM)
    return ret[0]  # (B,)

# 1) collecter des états-eau (h)
hs = []
for i in range(N):
    sess.step()
    obs = sess.observation
    se = int(obs.get("standing_entity",-1)) if isinstance(obs,dict) else -1
    if se in water_ids and sess.agent.h is not None:
        h = sess.agent.h
        hs.append((h if h.dim()==2 else h.unsqueeze(0)).squeeze(0).detach())
        if len(hs) >= MAXSTATES: break

if not hs:
    print("aucun état-eau"); sys.exit(0)
H0 = torch.stack(hs,0).to(device)  # (B,D)
B,D = H0.shape
print(f"{B} états-eau collectés")

# 2) avantage par action
with torch.no_grad():
    Vh = value_of(H0)                                  # (B,)
    adv = np.zeros((B,nact)); q1 = np.zeros((B,nact))
    for a in range(nact):
        at = torch.full((B,), a, dtype=torch.long, device=device)
        h2,_,r_a,_ = wm.imagine_batch(H0, at)          # reward 1-pas
        cont = policy_lambda_return(h2)                # continuation policy
        Q = r_a + GAMMA*cont
        adv[:,a] = (Q - Vh).cpu().numpy()
        q1[:,a]  = r_a.cpu().numpy()

others=[a for a in range(nact) if a!=INT]
print(f"\n=== AVANTAGE λ-return (ce que l'actor optimise) sur {B} états-eau ===")
print(f"  reward 1-pas    : INTERACT={q1[:,INT].mean():+.3f} | moy(autres)={q1[:,others].mean():+.3f}")
print(f"  AVANTAGE(a)     : INTERACT={adv[:,INT].mean():+.4f} | max(autres)={adv[:,others].max(1).mean():+.4f} | moy(autres)={adv[:,others].mean():+.4f}")
print(f"  -> INTERACT a l'avantage MAX dans {100*(adv.argmax(1)==INT).mean():.0f}% des états")
print(f"  -> avantage INTERACT POSITIF (devrait être renforcé) dans {100*(adv[:,INT]>0).mean():.0f}% des états")
