# Bilan — Décomposition du « mur » de la policy de fourrage (via un 2e monde)

**Date : 2026-06-29 · Branche : `rssm-egocentric` · Statut : 3 bugs réels corrigés ; le fourrage reste bloqué par le cœur dur (actor-critic en optimum local).**

Fait suite à `BILAN_DREAMERV3_2026-06-29.md` (« mur RL fondamental, ni brique ni
hyperparamètre »). Cette session **réfute ce verdict** : le « mur » était un
empilement de problèmes concrets, révélé en testant l'architecture sur un **2e monde**.

---

## 1. Méthode : le 2e monde comme instrument de diagnostic

Le livrable visé est un world-model **universel** ; on ne le testait que sur le
fouloïde. On a ajouté `SimpleGridWorld` (navigation dense vers une cible visible,
sans homéostasie ; `seedmind/envs/simple_grid_world.py`, 9 tests) branché sur le
**même cerveau v3**. Il **reproduit la pathologie « spam INTERACT sur place »** sans
aucun confound (ni perception partielle, ni sparsité, ni survie) — et permet un
**repro de 5 min** au lieu de runs fouloïde de 60k. C'est ce qui a tout débloqué.

Discipline : **diagnostiquer avant de tuner** (réfuter des hypothèses par des probes
discriminantes avant tout fix). Probes dans `scripts/diagnostics/` :
`probe_v3`, `probe_simple_grid`, `probe_imagination_drift`, `probe_wm_dynamics`,
`probe_reward_hedge`, `probe_navigation`, `planner_eval_simple_grid`, `eval_simple_grid`.

## 2. Le « mur » décomposé en couches (≠ un problème unique)

| Couche | Statut | Preuve |
|---|---|---|
| **1. Survie gratuite** | réelle, testée (F1 mortalité), **insuffisante seule** | `continue_prob`≈1 partout ; F1 = 363 morts mais wellbeing plat 0.55 |
| **2. Off-by-one du reward** | 🟢 **CORRIGÉ + validé** | WM apprend on-goal INTERACT +0.78 (écart 0.000 → +0.726) |
| **3. Bins twohot mal calibrés** | 🟢 **CORRIGÉ + validé** | `reward_vmax=20` → bins ±485M, malus <0.085 invisibles ; fix vmax=2/5 |
| **4. Perception cache la case sous l'agent** | 🟢 **CORRIGÉ (opt-in)** | latent ~90% sur on-goal ; `reveal_standing` → 2/4 cas de navigation réparés |
| **5. Actor-critic en optimum local** | 🔴 **FRONTIÈRE — non résolu** | commit sur INTERACT-sur-place ; `imag_return` grimpe (7.4) mais réel = 0 collecte |

## 3. Les fixes (corrigés, testés, poussés)

- **Off-by-one reward** (`recurrent.py`) : `feat_k` encode l'action `a[k-1]` mais
  était régressé contre `reward[k]` (récompense de `a[k]`) → reward head moyenné sur
  les actions → plat. Fix : régresser contre `reward[k-1]` (convention d'arrivée
  DreamerV3) + continue idem.
- **Bins twohot** (`reward_vmax 20→2`, `critic_vmax 20→5`) : à vmax=20 (symlog) les
  bins valent ±485M et l'espacement near-0 est 0.17 → tout petit reward (-0.02 noop,
  -0.01 pas) s'écrase sur le bin 0 = « gratuit ». Recalibrés à l'échelle réelle.
- **`reveal_standing`** (`micro_fouloide_encoder.py`, opt-in) : en égocentré l'agent
  est toujours au centre → le marqueur AGENT est redondant et CACHE la ressource sous
  ses pieds. On affiche `standing_entity` au centre → « sur cible » devient net.

**Les 3 bénéficient aussi au fouloïde** (mêmes mécanismes). Modes legacy préservés
(opt-in / défauts inchangés ; suites de tests vertes).

## 4. La frontière non résolue (couche 5)

L'actor-critic en imagination **commit sur un optimum local** : « INTERAGIR sur
place ». Il apprend « INTERACT paie en moyenne » (vrai sous échantillonnage qui tombe
parfois sur des cibles) au lieu de la politique **conditionnelle** (bouger si hors
cible, INTERAGIR si dessus). `imag_return` grimpe (la policy échantillonnée collecte
dans l'imaginaire) mais la policy apprise (argmax) reste bloquée → 0 collecte réelle.

## 5. Hypothèses RÉFUTÉES (ne pas y revenir)

1. Hallucination de l'imagination — WM fidèle (probe_imagination_drift, même saturé).
2. Sur-estimation du critic — valeurs petites (0.05-0.13).
3. Divergence du critic — valeurs petites/négatives, pas d'emballement.
4. Horizon trop long — h3 échoue comme h15.
5. Entropie trop molle — entropie basse → commit dur sur du garbage.
6. « Le planner sur le WM suffit » — échoue aussi (model exploitation / horizon).
7. Model exploitation via déplacements hors-grille — WM prédit les moves parfaitement.
8. **Plan2Explore** — l'`uncertainty_head` est plat (mean 0.775, std 0.05) car le
   monde trivial est entièrement modélisé → aucune incertitude à exploiter. Plan2Explore
   est l'outil des mondes **sous-explorés** ; ici le problème est la **policy**, pas le monde.

## 6. Recommandations (structurelles, pas des correctifs)

Continuer à patcher en session ne crackera probablement pas la couche 5 (c'est *le*
problème central du model-based RL). Options réelles :
1. **Setup type DreamerV3 réel** : gros replay + envs parallèles + GPU. C'est
   probablement ce qui sépare « converge » de « coincé » (notre online à flux unique
   en est très loin).
2. **Actor-critic fidèle à une référence** (NM512/dreamerv3-torch) : notre
   REINFORCE+baseline diffère du gradient à travers la dynamique de DreamerV3.
3. **Finir la précision on-cible du WM** (seul levier non-structurel restant) : les
   2/4 cas de navigation qui sur-prédisent encore INTERACT-près. Sans garantie.

## 7. Repro (5 min)

```bash
# entraîner (monde trivial dense + tous les fixes)
.venv/bin/python scripts/run_fouloide_online.py \
  --config configs/simple_grid_dense_reveal.yaml --steps 12000 --seed 0 --device cpu \
  --out-dir runs/w1_reveal_12k
# le WM apprend-il à naviguer ? l'agent collecte-t-il ?
EVAL_CKPT=runs/w1_reveal_12k/checkpoint_online.pt .venv/bin/python scripts/diagnostics/probe_navigation.py
EVAL_CKPT=runs/w1_reveal_12k/checkpoint_online.pt .venv/bin/python scripts/diagnostics/eval_simple_grid.py
```

Mémoires bd : `mur-policy-v3-deux-mecanismes`, `reward-off-by-one-cause-racine`,
`trois-couches-du-mur`, `couche3-critic-proximale`, `couche3-model-exploitation` (2026-06-29).
NB : MPS fuit la mémoire sur les longs entraînements RSSM → **CPU obligatoire** pour les runs.
