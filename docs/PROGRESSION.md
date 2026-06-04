# SeedMind — Progression du projet

Document de synthèse : ce qui a été fait, pourquoi, les résultats mesurés, et la feuille de route pour la suite.

> Hypothèse centrale (voir [SPEC.md](../SPEC.md)) : un agent doté d'un modèle du monde, d'une mémoire persistante, de curiosité et d'apprentissage continu peut développer des comportements adaptatifs **sans connaître les règles à l'avance**.

---

## 1. Vision et principes

### Cycle universel

```text
observation → action → conséquence → mémoire → amélioration
```

L'agent ne dépend jamais directement d'un environnement : tout passe par l'interface `EnvironmentAdapter` (`seedmind/envs/base.py`).

### Critère de validation récurrent

À chaque niveau, on cherche une preuve mesurable du type **trained bat naive** :

- GridWorld : taux de succès (clé → porte → récompense)
- Sandbox : **lifespan moyen** (nombre de pas survécus avant famine)

### Architecture agent (inchangée depuis V1)

| Module | Rôle |
|--------|------|
| **Encoder** | Projection observation → vecteur latent |
| **World Model** | Prédit `(next_latent, reward, uncertainty)` |
| **Policy (DQN)** | Réseau Q CNN, décision epsilon-greedy |
| **Planner** | Rollouts via le World Model (random-shooting) |
| **Goal Generator** | Micro-objectifs internes |
| **Curiosity** | Récompense intrinsèque sur erreur de prédiction |
| **Memory** | Mémoire persistante (traverse les épisodes / vies) |

---

## 2. Historique des niveaux

### V1 — GridWorld procédural

**Objectif :** prouver la boucle agent complète sur un monde simple (clé → porte).

- Environnements : `GridWorld`, `ProceduralGridWorld`
- Policy : heuristique / exploration aléatoire
- Encoder : MLP gelé (projection aléatoire)
- World Model + curiosité + mémoire en place

**Commandes :**

```bash
python scripts/run_v1.py --episodes 200
python scripts/evaluate_agent.py --checkpoint runs/<run>/checkpoint_final.pt
```

**Config :** `configs/v1_gridworld.yaml`

---

### V2 — Policy apprise (DQN) + règles colorées

**Objectif :** l'agent apprend **seul** une policy par réseau Q, avec règles de couleur (clé et porte doivent matcher).

- Environnement : `ColoredGridWorld`
- **QNetwork** CNN avec encodage agnostique aux couleurs (`QNET_COLOR_MATCH`)
- DQN + target network + replay buffer
- Expérience de **transfert** : entraînement sur certaines couleurs, test sur des couleurs jamais vues

**Résultat :** après corrections (canal `COLOR_MATCH`), l'agent entraîné surpasse le naïf sur de nouvelles cartes et généralise à des couleurs inédites.

**Commandes :**

```bash
python scripts/run_v2.py --config configs/v2_gridworld.yaml --episodes 10000
python scripts/transfer_experiment.py
```

**Config :** `configs/v2_gridworld.yaml`

---

### V3 / Niveau 4 — Observation partielle

**Objectif :** l'agent ne voit qu'un **rayon** autour de lui ; le reste est masqué (`UNKNOWN_OBJECT`).

- `visibility_radius` sur tous les GridWorld
- Canal dédié `QNET_UNKNOWN` dans le Q-Network (10 canaux)
- Pas de fog of war : modèle pur par rayon

**Résultat :** agent entraîné significativement meilleur que le naïf en observation partielle.

**Config :** `configs/v3_gridworld.yaml`

**Tests :** `tests/test_partial_obs.py`

---

### Live Agent Viewer

**Objectif :** observer l'agent apprendre en continu dans le navigateur.

- `scripts/live_agent.py` — boucle d'entraînement + WebSocket
- `seedmind/visualization/web_viewer.html` — grille, métriques, contrôle vitesse/pause
- Checkpoints automatiques (`runs/live_<seed>/`)

```bash
python scripts/live_agent.py --config configs/v3_gridworld.yaml
# Ouvrir http://localhost:8765
```

---

## 3. Monde Sandbox — Survie nue

### Pourquoi un sandbox ?

Avant un monde physique ou un écosystème type « fouloïdes », on a voulu un monde ouvert où :

- la **survie** est le seul drive (pas de tâche imposée),
- l'agent ne connaît **aucune règle** (HARVEST, EAT, énergie…),
- l'architecture est **extensible** (registres d'entités et d'actions).

### Niveau 0 — `SandboxWorld` (`sandbox_v0`)

**Fichier :** `seedmind/envs/sandbox_world.py`

**Mécaniques :**

| Élément | Comportement |
|---------|--------------|
| Énergie | −1 par pas ; mort à 0 |
| Sources de nourriture | HARVEST sur place → +1 food, source vidée |
| Inventaire | EAT consomme 1 food → +15 énergie (cap 100) |
| Repousse | Source régénère après `regrow_delay` pas |
| Récompense | +0.01 vivant, −1.0 à la mort |

**Actions :** `MOVE_*`, `HARVEST`, `EAT`, `WAIT`

**Entités (registre extensible) :** `EMPTY`, `WALL`, `AGENT`, `FOOD_SOURCE`, `FOOD_SOURCE_DEPLETED`

**Chaîne causale à découvrir :**

```text
se déplacer → HARVEST → EAT → énergie restaurée → survivre plus longtemps
```

**Fichiers clés :**

| Fichier | Rôle |
|---------|------|
| `seedmind/envs/sandbox_world.py` | Monde + registres |
| `seedmind/agent/sandbox_encoder.py` | Encodage Q-Net + Encoder MLP |
| `configs/sandbox_v0.yaml` | Config 8×8, vision complète |
| `configs/sandbox_v2_craft.yaml` | Config ressources/craft |
| `scripts/run_sandbox.py` | Boucle de vies DQN + WM |
| `scripts/evaluate_sandbox.py` | Comparaison trained vs naïf |
| `scripts/live_sandbox.py` | Viewer live sandbox |
| `seedmind/visualization/sandbox_viewer.html` | UI (énergie, food, lifespan) |
| `tests/test_sandbox_world.py` | Tests unitaires |

**Modifications transverses (backward compatible) :**

- `QNetwork` paramétrique : `num_grid_channels`, `num_scalars`, `obs_batch_fn`
- `Encoder` paramétrique : `input_dim`, `obs_to_vec_fn`
- `dqn.py` utilise `_obs_batch_fn` du Q-Network

#### Résultats v0 (8×8, vision complète, ~3000 vies)

| Agent | Lifespan moyen (200 épisodes eval) |
|-------|-------------------------------------|
| Naïf (aléatoire) | **64.8** |
| Entraîné | **186.9** (max 200) |
| **Ratio** | **2.89×** |

Checkpoint : `runs/sandbox_0/checkpoint_final.pt`

---

### Niveau 1 — Monde plus grand + observation partielle (`sandbox_v1`)

**Étape A de la roadmap** — socle spatial avant craft/planification.

**Changements :**

- Grille **16×16**, 16 sources, `max_steps=500`
- `visibility_radius=4` — cellules lointaines masquées
- Entité **`UNKNOWN`** (code 5) + canal dédié dans l'encodeur (6 canaux)
- Réseau Q plus large (64 conv / 256 hidden)

**Config :** `configs/sandbox_v1.yaml`

#### Résultats v1 (16×16, radius=4, 5000 vies)

| Métrique | Valeur |
|----------|--------|
| Lifespan fin d'entraînement (100 dernières vies) | **166.3** |
| Naïf (eval, 200 épisodes) | **68.0** |
| Entraîné (eval, 200 épisodes) | **288.3** (max 500) |
| **Ratio eval** | **4.24×** |

Checkpoint : `runs/sandbox_v1/checkpoint_final.pt`

**Interprétation :** l'agent apprend à naviguer, trouver de la nourritue et survivre **sans voir toute la carte**. Il atteint régulièrement la limite d'épisode (500 pas).

---

### Niveau 1b — Planification via World Model (expérimental)

**Étape B de la roadmap** — faire « réfléchir » l'agent, pas seulement réagir.

**Implémenté :**

| Composant | Fichier | Description |
|-----------|---------|-------------|
| Scoring combiné Q + WM | `seedmind/agent/agent.py` | `score = (1−α)·Q_norm + α·WM_norm` avec `planning_weight=α` |
| Planner | `seedmind/agent/planner.py` | Random-shooting, horizon configurable |
| Dyna (imagination) | `seedmind/training/imagination.py` | Expériences synthétiques via WM → replay buffer |
| Config | `configs/sandbox_v1_planning.yaml` | planning + dyna (dyna désactivé en prod) |

**Tests :** `tests/test_planning.py` (6 tests)

#### Résultats planification

| Run | Config | Lifespan (100 dernières vies) | Notes |
|-----|--------|-------------------------------|-------|
| `sandbox_v1` | sans planning | **166.3** | Baseline |
| `sandbox_v1_planning` | α=0.3, Dyna ON | **85.6** | WM imprécis + Dyna incompatible avec Q en espace observation → dégradation |
| `sandbox_v1_planning2` | α=0.15, Dyna OFF | **126.0** | Mieux, mais **sous** le baseline |

**Leçons :**

1. **Dyna désactivé** : le Q-Network apprend en espace `(grid, scalars)` ; les rêves du WM sont en espace **latent** — les mélanger pollue le replay.
2. **Planning trop tôt** : un WM encore imprécis ajoute du bruit aux Q-values quand ε est bas.
3. **Piste future** : activer le planning seulement après un seuil de qualité WM, ou planifier en espace latent avec un Q-network latent.

Checkpoints : `runs/sandbox_v1_planning/`, `runs/sandbox_v1_planning2/`

---

## 4. Suite de tests

```bash
pytest -q   # 79 tests (dernière exécution documentée)
```

Couverture principale :

- GridWorld, ColoredGridWorld, observation partielle
- DQN, mémoire, buffer, transfert
- Sandbox (énergie, harvest, eat, repousse, UNKNOWN, Q-Net)
- Planning + imagination

---

## 5. Commandes utiles

```bash
# Sandbox v0 (8×8, vision complète)
python scripts/run_sandbox.py --config configs/sandbox_v0.yaml --episodes 5000

# Sandbox v1 (16×16, observation partielle)
python scripts/run_sandbox.py --config configs/sandbox_v1.yaml --episodes 5000 --out-dir runs/sandbox_v1

# Évaluation trained vs naïf
python scripts/evaluate_sandbox.py \
  --checkpoint runs/sandbox_v1/checkpoint_final.pt \
  --config configs/sandbox_v1.yaml \
  --num-episodes 200

# Viewer live
python scripts/live_sandbox.py --config configs/sandbox_v1.yaml
# http://localhost:8765
```

---

## 6. Feuille de route — étapes à venir

Ordre retenu : chaque étape s'appuie sur la précédente, avec validation **trained bat naive** avant de continuer.

```text
[A] Monde plus grand + obs. partielle  ✅  (sandbox_v1)
[B] Planification via World Model       ⚠️  (code OK, pas encore gagnant)
[C] Ressources & Craft                ⚠️  (socle code OK, validation à lancer)
[D] Multi-pulsions homéostatiques       ⬜  (soif, froid, arbitrage)
```

### C — Ressources & Craft

**Pourquoi :** tester la découverte de **combinaisons** (bois + pierre → outil), pas seulement une chaîne à 2 maillons.

**Implémenté (sans réécriture grâce aux registres) :**

- Nouvelles entités : `WOOD_SOURCE`, `STONE_SOURCE`, `WORKBENCH`
- Action `CRAFT` activée uniquement par les configs craft
- Inventaire multi-ressources : `food`, `wood`, `stone`, `tool`
- Encodage étendu en mode craft : 9 canaux + 5 scalaires
- Effet causal de l'outil : `HARVEST` sur nourriture produit plus de food
- Config isolée : `configs/sandbox_v2_craft.yaml`
- Config pression craft : `configs/sandbox_v2_craft_pressure.yaml`
- Config intermédiaire : `configs/sandbox_v2_craft_balanced.yaml`
- Config intermédiaire + replay causal : `configs/sandbox_v2_craft_balanced_causal.yaml`
- Config intermédiaire + n-step DQN : `configs/sandbox_v2_craft_balanced_nstep.yaml`
- Config intermédiaire + curiosité causale : `configs/sandbox_v2_craft_balanced_intrinsic.yaml`

**Chaîne causale à tester :**

```text
se déplacer → récolter bois → récolter pierre → CRAFT outil
→ récolter nourriture plus efficacement → survivre plus longtemps
```

**Commandes :**

```bash
python scripts/run_sandbox.py \
  --config configs/sandbox_v2_craft.yaml \
  --episodes 10000 \
  --out-dir runs/sandbox_v2_craft

python scripts/evaluate_sandbox.py \
  --checkpoint runs/sandbox_v2_craft/checkpoint_final.pt \
  --config configs/sandbox_v2_craft.yaml \
  --num-episodes 200
```

**Risque :** explosion combinatoire — l'exploration aléatoire ne suffira plus ; le WM / planification deviendront nécessaires.

**Statut actuel :** tests unitaires et smoke test OK. Il reste à lancer un entraînement long et mesurer `trained bat naive`.

**Validation attendue :**

Le lifespan seul ne suffit pas à prouver que la chaîne craft est apprise. Les
runs `sandbox_v2_craft` loggent maintenant `metrics.json` avec :

- `harvest_wood`, `harvest_stone`
- `craft_tool`
- `harvest_food_tool`
- `bonus_food_from_tool`
- `eat_ok`

L'hypothèse craft est validée seulement si l'agent entraîné bat le naïf **et**
utilise davantage la chaîne :

```text
wood + stone -> craft_tool -> harvest_food_tool -> eat_ok -> lifespan plus long
```

La commande `scripts/evaluate_sandbox.py` compare aussi ces métriques causales
entre agent naïf et agent entraîné.

`sandbox_v2_craft_pressure.yaml` rend la stratégie sans outil moins suffisante :
nourriture plus rare, repousse plus lente, énergie alimentaire plus faible et
outil plus rentable (`base_food_yield=1`, `tool_food_bonus=3`).

`sandbox_v2_craft_balanced.yaml` est le point intermédiaire après deux bornes :
`sandbox_v2_craft.yaml` était trop facile sans outil, tandis que
`sandbox_v2_craft_pressure.yaml` était trop sparse. La config balanced garde
assez de nourriture pour découvrir `eat`, mais maintient l'outil très rentable
et ralentit l'epsilon decay pour laisser plus de temps à l'exploration.

`sandbox_v2_craft_balanced_causal.yaml` garde le même monde que balanced mais
active `dqn.sampler=causal`. Ce sampler ne change pas les rewards ; il
sur-échantillonne les transitions rares `craft_tool`, `harvest_food_tool`,
`eat_ok`, `harvest_wood`, `harvest_stone` pour éviter que les chaînes causales
découvertes brièvement soient noyées dans le replay uniforme.

`sandbox_v2_craft_balanced_nstep.yaml` garde le même monde que balanced et
active `dqn.n_step=8`. Le but est de corriger le credit assignment long-horizon
de manière générale : la valeur d'une action comme `CRAFT` peut être propagée
depuis les conséquences plusieurs pas plus tard, sans reward artificiel.

`sandbox_v2_craft_balanced_intrinsic.yaml` teste une autre hypothèse : l'agent
ne doit pas seulement optimiser la survie, il doit aussi s'intéresser aux
actions qui produisent de nouvelles conséquences causales. La curiosité causale
ajoute un reward intrinsèque borné pour les vrais événements d'état
(`harvest_*`, `craft_tool`, `eat_ok`), sans modifier le reward externe.

L'évaluation sandbox peut maintenant comparer `Q only` vs `Q + World Model
planner` sur le même checkpoint avec `--compare-planner`. C'est le test le plus
direct de la thèse centrale : le World Model doit améliorer la décision ou
l'exploitation d'une chaîne causale différée, à poids entraînés identiques.

### D — Multi-pulsions homéostatiques

**Pourquoi :** forcer l'**arbitrage** entre besoins concurrents (énergie, soif, chaleur).

**Prévu :**

- Jauges multiples avec decay indépendants
- Récompense = survie globale (pas de shaping par action)
- Comportements riches émergents (prioriser nourriture vs eau vs abri)

### B — Consolidation planification (à reprendre)

Pistes identifiées :

1. **Planning différé** : `planning_weight` monte seulement quand la loss WM est sous un seuil
2. **Dyna latent** : Q-network en espace latent, ou WM qui prédit aussi l'observation
3. **Planification seule en eval** : entraîner au DQN pur, planifier seulement à l'inférence

---

## 7. Arborescence des configs et runs

```text
configs/
  v1_gridworld.yaml
  v2_gridworld.yaml
  v3_gridworld.yaml          # partial obs GridWorld
  sandbox_v0.yaml            # 8×8 survie, vision complète
  sandbox_v1.yaml            # 16×16 survie, radius=4
  sandbox_v1_planning.yaml   # v1 + planning (expérimental)
  sandbox_v2_craft.yaml      # v1 + ressources/craft
  sandbox_v2_craft_balanced.yaml # intermédiaire craft
  sandbox_v2_craft_balanced_causal.yaml # balanced + replay causal
  sandbox_v2_craft_balanced_nstep.yaml # balanced + n-step DQN
  sandbox_v2_craft_balanced_intrinsic.yaml # balanced + curiosité causale
  sandbox_v2_craft_pressure.yaml # craft sous pression causale

runs/                        # gitignored
  sandbox_0/                   # v0 entraîné
  sandbox_v1/                  # v1 entraîné (baseline)
  sandbox_v1_planning/         # 1er essai planning
  sandbox_v1_planning2/        # planning conservateur
  sandbox_v2_craft/            # prochain run craft à produire
  live_sandbox_0/              # viewer live
```

---

## 8. Synthèse des résultats

| Niveau | Monde | Métrique clé | Naïf | Entraîné | Ratio |
|--------|-------|--------------|------|----------|-------|
| V2/V3 | GridWorld coloré + partial obs | Success rate | ~ | > naïf | ✓ |
| Sandbox v0 | 8×8 survie | Lifespan eval | 64.8 | 186.9 | **2.89×** |
| Sandbox v1 | 16×16 partial obs | Lifespan eval | 68.0 | 288.3 | **4.24×** |
| Sandbox v1 + planning | 16×16 + WM mix | Lifespan train | — | 126 vs 166 baseline | ✗ (pour l'instant) |

**Conclusion actuelle :** la preuve de concept « apprendre seul à survivre par causalité » tient sur v0 et v1. Le monde plus grand avec vision partielle est **plus difficile mais mieux résolu** (ratio 4.24× vs 2.89×). La planification via WM est en place mais **pas encore utile** — à consolider en parallèle ou après le craft.

---

## 9. Hors scope immédiat (vision long terme)

- Reproduction, multi-agents, construction libre (BUILD)
- Monde physique, ComputerWorld
- Interface type « fouloïdes » avec écosystème autonome

**Objectif final détaillé** (preuve d'efficacité du World Model, protocole A/B, chemin B→F) : voir [GOAL_WORLD_MODEL_FOULOIDES.md](./GOAL_WORLD_MODEL_FOULOIDES.md).

L'architecture (registres, adapters, modules découplés) est conçue pour accueillir ces extensions sans réécriture du cœur agent.

---

*Dernière mise à jour : juin 2026*
