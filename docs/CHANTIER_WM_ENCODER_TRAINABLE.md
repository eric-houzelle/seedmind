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

## 7. Références

- Bilan : `docs/BILAN_FORAGING_DECOMPOSITION_2026-06-29.md` §8–§9 (chaîne causale complète).
- Mémoires bd (2026-06-30) : `couche5-cause-racine-checkpoint`, `couche5-representation-aveugle`,
  `couche5-latent-width-refute`. Recall : `bd memories couche5`.
- Issues : `seedmind-cto` (checkpoint, fermée), `seedmind-10e.6` (ce chantier),
  `seedmind-10e.5` (régime DreamerV3 complet, si le crédit reste le verrou après).
- Outils durables : `scripts/diagnostics/probe_goal_distance.py`, `scripts/diagnostics/eval_sampled.py`.
- Configs : `simple_grid_sparse_reveal.yaml` (éparse, instrument navigation),
  `simple_grid_sparse_wide.yaml` (latent 256, réfuté), `simple_grid_dense_reveal.yaml` (dense).
