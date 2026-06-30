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
9. **Couverture des états de départ d'imagination** (2026-06-30) — imaginer depuis
   **tous** les `B×L` états posterior au lieu du seul état final (`start_states=all`,
   fidèle à DreamerV3) : **testé, négatif** (voir §8).

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
`trois-couches-du-mur`, `couche3-critic-proximale`, `couche3-model-exploitation` (2026-06-29),
`couche5-startstates-refute-2026-06-30`.
NB : MPS fuit la mémoire sur les longs entraînements RSSM → **CPU obligatoire** pour les runs.

---

## 8. Addendum 2026-06-30 — la couverture des départs réfutée (couche 5 reste structurelle)

**Hypothèse testée** : l'actor commit sur la marginale (« INTERAGIR sur place ») parce
qu'il imagine depuis trop peu d'états — notre `_sample_start_states` ne gardait que
l'état **final** de chaque séquence (`B` départs), là où DreamerV3 imagine depuis **tous**
les `B×T` états posterior aplatis (couverture large de la distribution visitée). C'est la
seule **divergence de fidélité** claire trouvée en comparant notre actor-critic à la
référence (le REINFORCE pour actions discrètes, lui, est *correct* en DreamerV3 — le
gradient-à-travers-la-dynamique n'est que pour le continu, donc §6.2 était une fausse piste).

**Fix** (opt-in, fidèle) : `imagination.start_states=all` → `_sample_start_states(mode="all")`
aplatit les `B×L` états posterior (`_stack_states`). Défaut `final` inchangé (fouloïde
déployé + ~50 tests bit-à-bit intacts).

**Résultat : NÉGATIF.** Run `w1_reveal_startall_12k` (config `simple_grid_dense_reveal` +
`start_states=all`), éval greedy :

- **0 collecte** (vs aléatoire 24.6/1000) ; policy = `INTERACT`(4232) + `REST`(768), **jamais MOVE**.
- `imag_return` **0 → 10** (la policy *échantillonnée* collecte dans l'imaginaire),
  `imag_entropy` 1.95 → 0.94, mais `critic_loss` **monte** 0.21 → 2.29 → ~1.7 : le critic
  **ne converge pas** sur un monde *trivial* — il chasse une cible qui grandit.

**Pourquoi ça réfute la couverture** : quand `imagination_policy=true`, l'epsilon-greedy est
**court-circuité** (`agent.py:186` échantillonne l'actor, `greedy=False`) — l'agent bougeait
donc déjà via sampling (`entropy` 0.94), le buffer avait **déjà** de la couverture off-cible,
et `start_states=all` l'a encore élargie. La couverture **n'était pas** le verrou.
(Artefact repéré : l'`epsilon` loggé reste 1.0, jamais utilisé sur ce chemin.)

**Verdict** : la couche 5 est la **scission sampled-collecte / greedy-collapse** + un critic
non convergent — le cœur dur du model-based RL (crédit/policy). **8 hypothèses réfutées**
en session (les 8 ci-dessus + la couverture). Le patching online ne la craquera pas. Restent :
le **régime DreamerV3 complet** (§6.1, replay + envs parallèles + GPU — projet délibéré,
issue `seedmind-10e.5`) ou l'admettre comme **limite d'approche**. Le world-model — la partie
dure et générique — marche, est size-invariant, déployé : c'est le livrable. Code `start_states`
conservé (infra fidèle, opt-in).

**⚠️ CET ADDENDUM EST DÉPASSÉ — voir §9.** Le verdict ci-dessus reposait encore sur des évals
greedy biaisées par un bug de checkpoint (l'actor n'était jamais sauvegardé). Cf. §9.

---

## 9. Addendum 2026-06-30 (suite) — DEUX causes racines, dont un bug

Après le résultat négatif de `start_states`, j'ai voulu comprendre *pourquoi* l'actor était
uniforme à l'éval. Deux découvertes successives **invalident la majeure partie du tableau
de réfutations** (qui testait via une éval cassée) :

### 9.1 — CAUSE RACINE #1 : bug de checkpoint (artefact d'éval) 🟢 CORRIGÉ
`save_checkpoint`/`load_checkpoint` (`checkpointing.py`) ne persistaient **ni `agent.actor`
ni `agent.critic`** (l'actor-critic d'imagination). L'actor s'entraînait puis était **jeté
au save** → toute éval via `sess.resume()` chargeait un actor **vierge** (uniforme, entropie
ln7 exacte) → 0 collecte greedy. **Toute la saga couche 5 évaluait un réseau non entraîné**
— comme l'off-by-one, un bug mécanique. Non détecté car `test_session_save_and_resume_roundtrip`
est cassé en amont par le bug `reward_key`. **Fix (commit 0ac3381)** : actor/critic + optimizers
+ target_critic EMA persistés ; round-trip vérifié. Issue `seedmind-cto` (P0).

Avec le vrai actor enfin évalué (DreamerV3 se déploie en **échantillonnant**, pas en argmax) :
la policy **échantillonnée collecte ~3× l'aléatoire** (dense ET éparse) → **elle apprend**.

### 9.2 — CAUSE RACINE #2 : la représentation n'encode pas la position de la cible
Le critic reste constant (V(on)=V(off)) même biaisé sur des départs on-cible (`start_states=
highreward`). Probe décisif (états réels labellisés par distance-à-cible, monde éparse) :

| Représentation | R² (probe linéaire frais → distance) |
|---|---|
| observation → **encodeur (64d)** | **+0.26** (faible) |
| → **feat RSSM (1152d)** | **−0.42** (≈ bruit) |
| · `h` déterministe (128d) | **+0.00** |
| · `z` stochastique (1024d) | **−0.39** |

`corr(V, distance) = −0.05`, V identique de dist 0 à 7. **La position relative de la cible est
faiblement captée par l'encodeur puis ENTIÈREMENT perdue dans le goulot RSSM.** L'actor et le
critic consomment le feat → ils naviguent **à l'aveugle**. Le critic constant n'est pas un bug
de crédit : son entrée ne contient pas l'information de navigation. Cohérent avec
`rssm-survie-verdict-budget` (« perception égocentrée = désavantage structurel »).

**Conséquence** : le « mur couche 5 » = bug de checkpoint (artefact) **+** limite de
représentation/perception (le WM n'encode pas la position de la cible). Le levier réel n'est
ni l'actor-critic, ni le critic, ni la couverture : c'est **faire encoder la position de la
cible par la représentation** (encodeur spatial / loss auxiliaire / goulot RSSM). Travail
côté world-model, identifié et borné. Mémoires : `couche5-cause-racine-checkpoint-2026-06-30`,
`couche5-representation-aveugle-2026-06-30`.

### 9.3 — Localisation fine + fix cheap RÉFUTÉ → le vrai fix est un redesign WM
L'encodeur est **gelé** (projection aléatoire fixe, `encoder.py:337`). Étage par étage :
features **conv** (3872d, random) → distance **R²=0.65** (la position y est !), mais la
**projection aléatoire figée → 64d** l'écrase à **R²=0.09**. Fix cheap tenté : élargir
`latent_dim` 64→256 (`simple_grid_sparse_wide.yaml`, run `w1_sparse_wide256_12k`).

**Résultat : embed R² 0.09→0.38 (le fix prend au niveau embed) MAIS navigation INCHANGÉE**
(critic constant V=−0.346 ∀dist, collecte ~4/1000 ≈ 3× aléatoire, greedy 0). Mesure fiable
(PCA-48+ridge) : embed R²=0.37 **mais feat RSSM R²=−0.02**, `h` déterministe **R²=0.00**.
`recon(feat→embed)` parfait (0.00008) : le décodeur *entraîné* récupère l'embed, mais la
position vit dans `z` (catégoriel) sous une forme **inexploitable** par un probe propre comme
par les MLP actor/critic, et **n'atteint jamais `h`**.

**Verdict** : améliorer la capacité de la représentation ne suffit pas — la position, même
présente dans l'embed, ne se propage pas en `h` et reste inutilisable dans `z`. Les leviers
cheap sont épuisés (checkpoint #1 corrigé ; start_states all+highreward, critic/crédit,
couverture, largeur encodeur : tous réfutés). **Le vrai fix = redesign façon DreamerV3 :
encodeur ENTRAÎNABLE + reconstruction de l'OBSERVATION** (pas de l'embed) → le latent
s'organise pour exposer la position de façon utilisable par la policy. Projet délibéré.
Mémoire : `couche5-latent-width-refute-2026-06-30`.
