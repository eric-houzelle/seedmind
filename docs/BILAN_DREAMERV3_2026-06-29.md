# Bilan — Port DreamerV3 fidèle + le mur de la policy de fourrage

**Date : 2026-06-29 · Branche : `rssm-egocentric` (mergée dans `main` pour le déploiement) · Statut : port complet, WM validé, policy de fourrage = mur de recherche ouvert**

Document de reprise. Fait suite à `BILAN_RSSM_2026-06-18.md` (perception égocentrée +
WM récurrent déterministe, piste parquée). Cette phase = **port DreamerV3 fidèle**
pour débloquer la survie, puis déploiement live « cerveau vierge qui apprend ».

---

## 1. Pourquoi ce port

Le bilan précédent avait localisé le verrou survie par des probes : le WM savait que
fourrager paie, mais l'**actor-critic n'engageait pas la policy vers le fourrage** —
l'**assignation de crédit** (critic mal calibré + avantage non informatif) classait les
actions au hasard. Réparer pièce par pièce était lent/fragile. **Décision : porter un
DreamerV3 fidèle d'un coup** (les ~10 stabilisateurs co-dépendants), guidé par
`NM512/dreamerv3-torch` / `r2dreamer`.

## 2. Ce qui a été construit (toutes les briques, intégrées, testées)

| Brique | Fichier | Commit |
|---|---|---|
| RSSM stochastique (z catégoriel 32×32 + prior/posterior + KL balancé + free-bits + straight-through) | `seedmind/agent/rssm.py` | 744aec8 |
| RSSMWorldModel (décodeur + heads sur `feat=[z,h]`) | `seedmind/agent/world_model.py` | 91cb3f0 |
| Training BPTT recon+KL+reward | `seedmind/training/recurrent.py` (`train_rssm_world_model`) | e8e9b4f |
| Agent porte l'état `(h,z)` + build_agent câble le RSSM | `agent.py`, `run_micro_fouloide.py` | 7fd620c |
| Imagination sur `(h,z)` + routage online | `imagination_actor_critic.py`, `online.py` | e7195f7 |
| **Twohot critic** (catégoriel bins symlog + cross-entropy) | `seedmind/agent/value_model.py` (`TwoHotCritic`) | 0758fd5 |
| **Twohot reward predictor** | `world_model.py` (`reward_value`/`reward_loss`) | b398e65 |
| **Normalisation des returns par percentile EMA** (5-95) | `imagination_actor_critic.py` (`advantage_norm=percentile`) | 9a3c266 |
| **LayerNorm + SiLU** partout (RSSM, décodeur, twohot) | rssm/world_model/value_model | ed45d67 |
| **Continue predictor** + death-discount en imagination | recurrent/world_model/imagination_actor_critic | 779a24c |
| **Schedule d'entropie** explore→commit | `online.py` (`entropy_coef_start/end/decay_steps`) | 9d4c384 |
| symlog inputs | **SKIPPÉ** (justifié : couvert par twohot, inputs bornés) | — |

**Config phare : `configs/micro_fouloide_online_homeostatic_rssm_v3.yaml`** (flag `world_model.rssm_stochastic: true`). Tout est opt-in : défaut = ancien comportement, tests legacy verts (~50 tests).

## 3. Ce qui MARCHE (acquis durables, validés)

- ✅ **Le world-model apprend remarquablement bien** : recon ~1e-4, continue ~1e-4
  (prédit la mort), reward-loss **1.9 → 0.70** sur 124k steps, KL au plancher free-bits.
- ✅ **Calibration réparée** (probe) : les avantages passent de « 83% positifs/non
  calibrés » à **centrés sur zéro** grâce au twohot critic + percentile.
- ✅ **Size-invariance** : le même cerveau tourne sur 96×96 (le full-grid en est incapable).
- ✅ **Déployé live** (cerveau vierge, prod `www.releaskills.com:8443`), front responsive
  + dashboard + **API `/fouloides-stats`** (`{latest, history}`) pour le monitoring.

## 4. LE MUR : la policy ne s'engage pas vers le fourrage

Malgré le stack complet, **l'agent n'apprend pas à fourrager de façon décidée**. Testé :

| Stratégie d'entropie | Résultat |
|---|---|
| **fixe haute (0.03)** | policy reste **uniforme** (entropie collée au max ~1.94), ne commit jamais. wellbeing ~0.1, plat. |
| **fixe basse (0.001)** | l'entropie descend (~1.4) → **commit sur le NON-fourrage** (eau→0, wellbeing 0). |
| **schedule 0.03→3e-4 sur 40k** | explore haut (fourrage aléatoire) puis l'entropie s'effondre (0.55) → **commit ENCORE sur le non-fourrage** (eau→0, wellbeing 0). |

**Cause racine (probes d'avantage, sur `scripts/diagnostics/`)** : sur les états-eau, le
WM imagine **~le même reward pour TOUTES les actions** (INTERACT ≈ autres) — le **prior
ne différencie pas la conséquence-reward de chaque action en imagination**. Donc l'avantage
de INTERACT n'est jamais décisivement le max → dès que la policy s'engage, elle s'engage
sur n'importe quoi. **Aucune stratégie d'entropie ne peut corriger ça** (c'est en amont).

**Confirmation prod à 124k** : WM excellent, record de survie 18 221 (variance + soft-death),
mais entropie ~1.81 (quasi-uniforme), wellbeing ~0.1. La policy ne s'est pas engagée vers
le fourrage même à grande échelle.

→ **Ce n'est ni une brique manquante, ni un hyperparamètre : c'est un vrai mur RL** (le
prior d'imagination ne crédite pas assez l'action de fourrage).

## 5. Pistes de reprise (profondes, sans garantie)

1. **Entraînement BEAUCOUP plus long** : le WM/prior continue de s'affiner (reward-loss
   1.9→0.7 à 124k) ; il est *possible* que le prior finisse par différencier les actions
   et que l'avantage favorise enfin INTERACT. Coût : ~1M steps = jours machine (GPU
   recommandé — voir §6). Incertain (l'entropie n'a amorcé qu'une lente descente à 124k).
2. **Rendre le reward action-conditionné** dans l'imagination : prédire le reward depuis
   `(feat, action)` plutôt que `feat` seul, OU améliorer la qualité du prior (plus de
   poids KL, rollouts de contexte plus longs) pour qu'il différencie les next-states par action.
3. **Repenser l'apprentissage de la policy** (au-delà de Dreamer-imagination) : un signal
   de fourrage plus direct, ou un planner sur `(h,z)` qui exploite le WM (bon) sans dépendre
   du gradient de policy.
4. **Twohot/discrétisation plus fine, KL-balancing exact DreamerV3, init/échelles précises** —
   on a l'essentiel mais pas chaque détail au pixel près.

## 6. Note ops / déploiement

- **CPU = goulot** : le RSSM complet entraîné en ligne sature un CPU. Pour le live regardable
  → **GPU** (`DEVICE=cuda`, NVIDIA ≥6 Go VRAM suffit — T4/L4/RTX 3060) ou Apple Silicon (`mps`).
  (Décision 2026-06-29 : pas de GPU pour le moment, on reste CPU.)
- **Déploiement** : `git checkout main && git pull` sur le serveur, puis
  `SOURCE=live LIVE_CONFIG=configs/micro_fouloide_online_homeostatic_rssm_v3.yaml LIVE_FRESH=1 DEVICE=cpu docker compose -f docker-compose.fouloides.tls.yml up -d --build`.
  Front = Vercel (rebuild auto depuis `main`). Monitoring = `https://www.releaskills.com:8443/fouloides-stats`.
- **Lancer les runs longs avec `caffeinate -i`** (la veille machine tue les runs détachés).

## 7. Pointeurs

- **Configs** : `..._rssm_v3.yaml` (DreamerV3 32×32), `..._rssm_bigmap.yaml` (96×96, ancien RSSM).
- **Runs clés** : `rssm_v3_full_80k` (entropy fixe 0.03 → policy figée), `rssm_v3_lowent_60k`
  (0.001 → commit non-fourrage), `rssm_v3_sched_60k` (schedule → commit non-fourrage).
- **Probes/diagnostic** : `scripts/diagnostics/` (`probe_advantage*`, `eval_death_cause`).
- **Tests** : `tests/test_rssm*.py`, `test_agent_rssm.py`, `test_imagination_actor_critic.py`.
- **Mémoires bd** : `rssm-idle-basin-2026-06-22`, `rssm-survie-famine-chronique-2026-06-22`,
  `rssm-fourrage-transitoire-2026-06-24`, `rssm-survie-verdict-budget-2026-06-25`.
- **Epic bd** : `seedmind-10e` (port DreamerV3).
