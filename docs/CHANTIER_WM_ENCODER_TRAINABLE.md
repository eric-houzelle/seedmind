# Chantier — Encodeur entraînable + reconstruction d'observation (couche 5, cause racine #2)

> **Issue** : `seedmind-10e.6` · **Branche** : `rssm-egocentric` · **Prêt à démarrer.**
> Préparé 2026-06-30. Démarrage clé-en-main : lire ce doc + `bd show seedmind-10e.6`, puis suivre §4.

## 1. TL;DR

La policy d'imagination n'apprend pas à **naviguer** parce que le `feat` RSSM `[z,h]`
qu'elle consomme **n'encode pas la position de la cible**. Cause : l'encodeur est **gelé**
(projection aléatoire) et le WM reconstruit l'**embedding**, pas l'**observation** → rien ne
force le latent à encoder *où* est la cible. **Fix (fidèle DreamerV3)** : reconstruire
l'**observation** depuis le `feat` + rendre l'encodeur **entraînable**, en **opt-in** (le
fouloïde déployé garde le chemin gelé bit-à-bit).

## 2. Pourquoi (diagnostic établi le 2026-06-30, chiffré)

Contexte : le « mur couche 5 » avait DEUX causes. La #1 (bug de checkpoint : l'actor-critic
n'était jamais sauvé → évals sur réseau vierge) est **corrigée** (`seedmind-cto`, commit
`0ac3381`). Une fois le vrai actor évalué : **la policy apprend** (échantillonnée = 3×
l'aléatoire). Reste la #2, ci-dessous.

Localisation étage par étage (probe `scripts/diagnostics/probe_goal_distance.py`, R² d'un
probe linéaire frais → distance-Manhattan à la cible, monde `simple_grid_sparse_reveal`) :

| Représentation | R² → distance | Lecture |
|---|---|---|
| features **conv** (3872d, random) | **0.65** | la position EST captée par le conv |
| → **embed** projeté (gelé, 64d) | **0.09** | la **projection aléatoire figée la jette** |
| → embed élargi (latent_dim 256) | 0.38 | élargir aide *l'embed*… |
| → **feat RSSM [z,h]** (ce que voit la policy) | **≈ 0** | …mais ça **n'atteint pas le feat** |
| · `h` déterministe | **0.00** | h n'encode rien de la position |
| · `z` stochastique | ≈ 0 (probe propre) | présent mais sous forme **inexploitable** |

`recon(feat→embed)` est parfait (MSE 0.00008) : le décodeur *entraîné* récupère l'embed, mais
la position est dans `z` (catégoriel) sous une forme qu'aucun probe propre ni les MLP
actor/critic n'exploitent. **Conséquence** : critic constant (V identique ∀ distance) → aucun
gradient de navigation → l'argmax dégénère en action constante.

**Leviers cheap ÉPUISÉS** (tous réfutés, mesurés) : `start_states` all + highreward,
critic/crédit, couverture, **largeur d'encodeur** (256 → embed 0.38 mais feat toujours ≈0).
Le `decoder→embed` est le verrou structurel.

## 3. Le fix

Faire en sorte que le `feat` `[z,h]` encode la position **de façon utilisable par la policy**,
en **reconstruisant l'observation** (pas l'embed) et en **entraînant l'encodeur** (joint, façon
DreamerV3). Étapes :

1. **Stocker l'observation égocentrique** dans le buffer (actuellement `observation=None`
   dans `make_experience`, cf. `scripts/run_fouloide_online.py` step). La fenêtre est
   **11×11×C bornée** quel que soit le monde → coût mémoire borné. (Alternative plus légère :
   un auto-encodeur d'encodeur entraîné **en ligne** sur l'obs vivante, sans stockage — mais
   il faut que la cible de recon soit le `feat`, pas seulement l'embed, pour viser la policy.)
2. **Décodeur spatial** `feat → fenêtre 11×11×C` (déconv ou MLP→reshape) dans le WorldModel
   (`seedmind/agent/world_model.py`), à côté du `decoder` actuel (qui reconstruit l'embed).
3. **Perte de reconstruction d'OBS** dans `train_rssm_world_model`
   (`seedmind/training/recurrent.py`, actuellement `recon = (decoder(feat) - embed)²`,
   ligne ~398) : régresser `decoder_obs(feat)` vs la fenêtre d'obs stockée. **Opt-in** via une
   clé config `world_model.obs_reconstruction: true`.
4. **Dégeler l'encodeur** : `ConvEncoder` (`seedmind/agent/encoder.py:337`) gèle tout
   (`requires_grad_(False)`). Le rendre entraînable en **opt-in** (`agent.encoder.trainable:
   true`) + l'ajouter à l'optimiseur du WM. ⚠️ casse l'hypothèse « cible latente stable » →
   co-entraînement encodeur+RSSM+décodeur (standard DreamerV3, surveiller la stabilité).
5. **Config de test** : nouvelle variante de `configs/simple_grid_sparse_reveal.yaml` avec les
   deux flags. Tester sur ce monde éparse (l'instrument de navigation propre).
6. **Garder legacy intact** : flags off par défaut → fouloïde déployé + ~50 tests bit-à-bit.

## 4. Comment démarrer (commandes)

```bash
bd update seedmind-10e.6 --claim
# 1) implémenter (étapes §3) en opt-in
# 2) entraîner sur le monde éparse (CPU obligatoire — MPS fuit sur le RSSM)
.venv/bin/python scripts/run_fouloide_online.py \
  --config configs/<nouvelle_config_obs_recon>.yaml --steps 12000 --seed 0 --device cpu \
  --out-dir runs/w1_obsrecon_12k
# 3) VÉRIFIER LA REPRÉSENTATION (le test décisif)
EVAL_CONFIG=configs/<...>.yaml EVAL_CKPT=runs/w1_obsrecon_12k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/probe_goal_distance.py
# 4) VÉRIFIER LA POLICY (greedy ET échantillonné — jamais greedy seul)
EVAL_CONFIG=configs/<...>.yaml EVAL_CKPT=runs/w1_obsrecon_12k/checkpoint_online.pt \
  .venv/bin/python scripts/diagnostics/eval_sampled.py
```

## 5. Critères de succès (mesurables, binaires)

- **probe_goal_distance** : `feat R²→distance` passe de **~0 à > 0.5**, `h` R² > 0, et
  **corr(V, distance) nettement < 0** (le critic discrimine enfin).
- **eval_sampled** : collecte (greedy **ET** échantillonné) **≫ aléatoire** (baseline éparse
  ≈ 1.4/1000), et l'argmax greedy **n'est plus une action constante** (il navigue).

Si oui → la représentation était bien le verrou, couche 5 craquée sur W1, enchaîner sur le
fouloïde. Si non (feat encode enfin la position mais la policy ne navigue toujours pas) → le
verrou résiduel est le **crédit/policy** (régime DreamerV3 complet, `seedmind-10e.5`), mais
proprement isolé.

## 6. Risques / garde-fous

- **Stabilité** : dégeler l'encodeur rend la cible latente non-stationnaire (le WM prédit une
  cible qui bouge). DreamerV3 le gère, mais surveiller recon/KL ; au besoin, geler le conv et
  ne dégeler que la tête de projection.
- **Déployé** : le fouloïde live utilise l'encodeur gelé → **tout en opt-in**, défaut inchangé,
  garder les tests verts (dont le run d'or legacy).
- **Mémoire buffer** : stocker l'obs (11×11×C) augmente le buffer ; borné, OK pour W1 ; vérifier
  pour le bigmap fouloïde si on étend.

## 8. Résultat (exécuté le 2026-07-01)

Implémenté en opt-in (`agent.encoder.trainable`, `world_model.obs_reconstruction`),
config `configs/simple_grid_sparse_obsrecon.yaml`, run `runs/w1_obsrecon_12k` (12k CPU).
**0 régression** (290 tests verts ; les 4 échecs sont un bug préexistant `train_world_model`/
`reward_key` du chemin feed-forward, filé à part).

**Verrou représentation CASSÉ (thèse validée).** Probe (frozen → 4k → 8k → 12k) :

| Mesure | frozen | 4k | 8k | 12k |
|---|---|---|---|---|
| encodeur embed R²→dist | 0.09 | 0.24 | 0.61 | **0.65** |
| feat RSSM [z,h] R² | ~0 | −0.01 | 0.18 | **0.34** (en montée) |
| · h déterministe R² | 0.00 | 0.00 | 0.28 | **0.37** |
| · z stochastique R² | ~0 | −0.02 | 0.03 | 0.03 |
| corr(V, distance) | ~0 | +0.01 | −0.10 | **−0.24** |

Le `h` qui n'encodait **rien** encode désormais la position ; le critic devient discriminant
(V=2.95 sur la cible → 2.0 à distance 5).

**MAIS navigation toujours bloquée.** `eval_sampled` @12k : greedy dégénéré (0 collecte,
argmax quasi-constant MOVE_RIGHT/LEFT), sampled **1.6/1000 ≈ aléatoire** (1.4). Le critic sait
où est la valeur, l'actor ne grimpe pas le gradient.

**→ Branche §5 confirmée** : le verrou résiduel est le **crédit/policy**, désormais proprement
isolé (représentation levée) → `seedmind-10e.5`. Mémoire : `couche5-obsrecon-encodeur-2026-07-01`.

## 9. Suivi — isolation du verrou résiduel (2026-07-01)

Outil ajouté : `scripts/diagnostics/probe_actor_navigation.py` (l'actor tracke-t-il la
cible, ou colle-t-il à la marginale ?). Sur le checkpoint 12k il **réfute** « l'actor
ignore le critic » : il a **appris la structure conditionnelle** — corr(P(R)−P(L), dc)=
+0.49 (nav horizontale), +0.22 (verticale), P(INTERACT | **sur** la cible)=0.35 vs 0.001
loin — mais trop mollement pour chaîner (argmax sur la cible = WAIT, pas INTERACT).

Hypothèse testée : **sous-affûtage** (`entropy_decay_steps=15000` > run 12000 → entropie
jamais au plancher). Test : reprise 12k→24k (`runs/w1_obsrecon_24k`), entropie au plancher
0.001 dès 15k. **RÉFUTÉE** — le sharpening **empire** la collecte (1.6 → ~0/1000). La
policy se committe (entropie/état 0.88→0.29, dépendance +0.40→+1.02) mais vers un **mauvais
attracteur** : `MOVE_DOWN=0.000` (n'atteint jamais une cible en-dessous), argmax sur la
cible = MOVE_LEFT (elle **quitte** la cible). Représentation stable (embed 0.70, feat 0.31).

**Verrou résiduel isolé = model-exploitation / imagination-reality gap.** `imag_return`
reste ~2.6 alors que la collecte **réelle** est nulle : l'actor optimise un retour *imaginé*
que le world model hallucine mais qui ne transfère pas au réel (cf. mémoire
`couche3-model-exploitation`). C'est la classe `seedmind-10e.5` (fidélité du WM en
imagination), **pas** la représentation ni un simple réglage d'entropie. Diagnostic
CPU-friendly ; mitigations CPU à tenter avant le régime GPU (horizon d'imagination,
posterior-in-imagination, calibration de la reward-head, équilibrage KL).
Mémoire : `couche5-actor-sous-affute-2026-07-01`.

## 10. Localisation du model-exploitation (2026-07-01, seedmind-10e.7)

Nouvelle sonde `scripts/diagnostics/probe_imag_real_gap.py` (CPU) : pilote l'actor
**une fois** dans l'env réel puis décompose l'écart imaginé↔réel. Ici
`reward_learning == reward` extrinsèque pur (drive/resource off) et la reward-head
RSSM est entraînée **sans** curiosité → l'imaginé se compare directement au reward
de l'env.

**[A] La reward-head est BIEN calibrée one-step** (teacher-forced sur la trajectoire
réellement visitée) — donc **pas** la cause primaire :

| Mesure (12k / 24k) | 12k | 24k affûté |
|---|---|---|
| `r_post` (head sur le vrai latent suivant) vs `r_real` | −0.005 / −0.010 | −0.002 / −0.012 |
| reward prédit sur les VRAIES collectes (`r_real=+1`) | +0.83 | +0.99 |
| fuite hors-cible (`r_prior` sur `r_real≤0`) | −0.006 (négatif) | −0.003 (négatif) |

**[B] L'hallucination = dérive prior COMPOSÉE.** Sur un rollout imaginé libre de H=15
pas sous l'actor, le reward imaginé **monte monotone avec la profondeur** alors que le
réel reste plat à ~−0.011 :

| Mesure | 12k | 24k affûté |
|---|---|---|
| `G_imag` (retour actualisé, reward-only) | +0.34 | **+1.26** |
| `G_real` (idem, réel) | −0.11 | −0.15 |
| fraction de pas « cible-like » (reward>0.3) imaginée vs réelle | 5.3 % vs 0.1 % | **13.4 %** vs 0.1 % |
| courbe reward imaginé (pas 0→14) | −0.003 → 0.067 | −0.003 → 0.18 |
| **corr(incertitude imaginée, reward imaginé)** | **+0.24** | **+0.31** |

**Mécanisme confirmé par 12k→24k** : affûter fait **commettre l'actor plus fort dans
l'hallucination** (l'imagination plonge plus profond dans le fantasme : goal-like
5.3→13.4 %, `G_imag` ×3.7) pendant que la collecte réelle **baisse** (1.0→0.5/1000).
C'est *exactement* pourquoi le sharpening empire (§9). L'`imag_return≈2.6` du training
est encore plus haut que `G_imag` reward-only car la λ-return injecte le bootstrap d'un
critic inflé (V≈2.6) — **levier distinct** (runaway-bootstrap, `couche5-cause-racine-checkpoint`).

**⚠️ Le levier « pénaliser par l'incertitude » n'est PAS viable tel quel.** Lecture code :
pour le WM RSSM, l'`uncertainty_head` n'est **jamais entraînée** (`_update_models` délègue
à `_update_models_recurrent` et `return` avant l'appel `train_world_model_uncertainty_head`,
online.py:295/320 ; et `train_rssm_world_model` ne l'inclut pas dans sa loss). Le réglage
`uncertainty_head_updates_per_train: 2` est **mort** pour le chemin RSSM. La tête est donc
une projection linéaire **aléatoire** → la corr(unc, reward imaginé)=+0.24/+0.31 est un
**artefact** (la magnitude du feat croît off-manifold), pas un signal épistémique. Utiliser
ce levier exigerait d'abord d'entraîner une vraie tête d'erreur/désaccord (Plan2Explore).
Mémoire : `couche5-uncertainty-head-morte-rssm-2026-07-01`.

**Leviers CPU classés (opt-in, à tester un à un en mesurant) :**
1. **Horizon d'imagination plus court** — config-only (`--horizon` / `imagination.horizon`),
   zéro risque représentation, attaque directement le compounding monotone (le gros de
   l'hallucination s'accumule aux pas profonds 8–15). **Premier test** : run `w1_obsrecon_h6_12k`.
2. **Resserrer la KL dynamique** (`kl_dyn_scale`) — rapproche le prior du posterior → moins de
   dérive par pas (attaque la racine). Config-only mais ré-entraîne le WM ; risque = collapse
   de la représentation → re-vérifier `probe_goal_distance` après.
3. **Vraie tête d'incertitude/désaccord** (Plan2Explore) puis pénalité — plus de travail (code).
4. Reward-head (`reward_vmax`/bins) — **déprioritisé** : la head est calibrée on-manifold.

⚠️ La « dérive `|feat_prior−feat_post|≈6.7` » est quasi constante 12k↔24k alors que
`G_imag` triple → dominée par la variance d'échantillonnage de `z` catégoriel, **pas** le
signal discriminant. Se fier au reward-space (rampe + fraction cible-like).

Mémoire : `couche5-model-exploit-localise-2026-07-01`.

## 7. Références

- Bilan : `docs/BILAN_FORAGING_DECOMPOSITION_2026-06-29.md` §8–§9 (chaîne causale complète).
- Mémoires bd (2026-06-30) : `couche5-cause-racine-checkpoint`, `couche5-representation-aveugle`,
  `couche5-latent-width-refute`. Recall : `bd memories couche5`.
- Issues : `seedmind-cto` (checkpoint, fermée), `seedmind-10e.6` (ce chantier),
  `seedmind-10e.5` (régime DreamerV3 complet, si le crédit reste le verrou après).
- Outils durables : `scripts/diagnostics/probe_goal_distance.py`, `scripts/diagnostics/eval_sampled.py`.
- Configs : `simple_grid_sparse_reveal.yaml` (éparse, instrument navigation),
  `simple_grid_sparse_wide.yaml` (latent 256, réfuté), `simple_grid_dense_reveal.yaml` (dense).
