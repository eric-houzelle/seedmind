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
pytest -q   # 109 tests (dernière exécution documentée)
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

### Bilan architectural — juin 2026

Objectif acté :

```text
Construire un agent générique capable d'apprendre seul un monde,
via interaction, mémoire, drives internes et World Model,
sans règles spécifiques codées dans l'agent.
```

Le sandbox craft est un banc de test. Il n'est pas la finalité du projet.

#### Frontière agent / monde

Un monde peut exposer :

- observations brutes ;
- actions disponibles ;
- features perceptibles structurées ;
- événements observés ;
- états internes perceptibles de l'organisme simulé.

Un monde ne doit pas exposer :

- la solution ;
- les règles internes ;
- la valeur stratégique d'une action ;
- la prochaine meilleure action.

Principe retenu :

```text
Le monde expose ce qui est observable.
L'agent apprend ce qui est utile.
```

Exemples de features structurées acceptables :

```text
energy, health, hydration, temperature
position, velocity, fatigue
inventory / body state
visible_entities: type_id, distance, direction
terrain / light / smell / sound / heat
```

Ces variables sont des capteurs ou perceptions organisées. Elles ne disent pas
à l'agent quoi faire.

#### Rôle cible du World Model

Le World Model ne doit pas rester un simple module auxiliaire du DQN. Il est le
mécanisme central d'autonomie :

- apprendre la dynamique du monde ;
- prédire les conséquences des actions ;
- anticiper plusieurs futurs possibles ;
- guider l'exploration ;
- servir de base à une décision plus intelligente.

La preuve forte recherchée reste :

```text
agent avec World Model exploité > agent sans World Model exploité
```

sur le même monde, le même budget, et idéalement plusieurs seeds.

#### Drives

Deux drives sont retenus :

```text
drive homéostatique = rester viable / vivant
drive épistémique = comprendre le monde / réduire l'incertitude utile
```

Le système ne doit pas recevoir une mission du type "utilise le craft". Il doit
survivre, explorer utilement, puis apprendre quelles interactions améliorent sa
viabilité.

#### Statut des poids manuels actuels

Les `planner_feature_weights` et `planner_feature_targets` de la config
`causalwm` sont considérés comme un **échafaudage expérimental**.

Ils ont permis d'obtenir un premier signal positif, mais ils ne doivent pas
être conservés comme principe architectural final.

Objectif cible :

```text
Le planner imagine avec le World Model.
La valeur des conséquences est apprise par l'agent.
Elle n'est pas codée à la main dans la config.
```

#### Critère de validation sérieux

Pour valider une évolution importante :

```text
Q + WM planner > Q-only
sur moyenne multi-seed
avec amélioration d'au moins une métrique homéostatique ou causale.
```

Le simple fait que `trained > naive` ne suffit plus pour valider le rôle du
World Model. Cela valide seulement que la policy apprend.

#### Direction suivante : micro-fouloïde

Avant de viser un écosystème fouloïde complet, l'étape raisonnable est un
**micro-fouloïde** :

- un seul organisme ;
- plusieurs besoins homéostatiques ;
- monde virtuel plus riche que le sandbox ;
- interactions simples mais nombreuses ;
- aucune règle métier dans l'agent.

Exemple de monde :

```text
besoins:
  énergie, hydratation, température / sécurité

monde:
  nourriture, eau, chaleur/froid, dangers, obstacles, zones

actions:
  bouger, manger, boire, se reposer, interagir/manipuler
```

Objectif :

```text
Voir si un organisme minimal apprend des routines adaptatives
par interaction + World Model, sans règles codées.
```

```text
[A] Monde plus grand + obs. partielle  ✅  (sandbox_v1)
[B] Planification via World Model       ⚠️  (premier signal positif, encore fragile)
[C] Ressources & Craft                  ✅/⚠️ (craft validé en eval, planner causal à consolider)
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
- Config causal-WM structurée : `configs/sandbox_v2_craft_balanced_causalwm.yaml`

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

**Statut actuel :** entraînements longs effectués. Le craft est validé en évaluation
pour la branche causal-WM rebalanced, avec un premier signal positif du planner
World Model. Le résultat reste fragile et doit être consolidé avant de
généraliser.

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
sur-échantillonne les transitions dotées d'un événement par rareté générique
(`event_index` / `event`), sans connaître le sens de ces événements. Dans le
sandbox, cela aide à éviter que les chaînes causales découvertes brièvement
soient noyées dans le replay uniforme.

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

#### Résultats expérimentaux craft / causal-WM — juin 2026

Objectif global rappelé : prouver qu'un **World Model causal** peut aider un
agent à apprendre seul un monde, pas simplement construire une simulation de
singe ou optimiser un scénario craft à la main. Le sandbox reste un banc de
test, pas la finalité.

**Premier run craft simple :** `sandbox_v2_craft_mps_hybrid`

```text
Naïf:      lifespan 64.7
Trained:   lifespan 222.3  (3.44×)
craft:     ~0.01 / épisode
tool_food: 0.00
```

Conclusion : l'agent apprend fortement la survie, mais contourne le craft. Le
lifespan seul ne prouve donc pas l'apprentissage de chaînes causales composées.

**Run balanced + intrinsic causal :**
`sandbox_v2_craft_balanced_intrinsic_mps_hybrid`

En entraînement, le signal craft apparaît :

```text
life 7999 | lifespan(100)=97.6
craft100=1.00 toolfood100=0.49 eat100=3.26
```

Évaluation :

```text
Naïf:              lifespan 62.2
Q-only:            lifespan 77.8
Q + WM planner:    lifespan 76.2
planner/Q:         0.98×
```

Conclusion : Q apprend quelque chose, mais le planner WM ne fournit pas encore
de gain. Les diagnostics ont montré que le WM prédit assez bien les latents,
mais produit un signal de décision non actionnable : il préfère trop souvent
`HARVEST`, même dans des états où `CRAFT` serait pertinent.

**Évolution causal-WM structurée :**
`sandbox_v2_craft_balanced_causalwm.yaml`

Ajout générique, sans adhérence au sandbox :

- `EnvironmentAdapter.causal_features()` : vecteur observable optionnel ;
- `EnvironmentAdapter.causal_event_names()` : vocabulaire d'événements optionnel ;
- World Model avec têtes optionnelles `feature_delta` et `event_logits` ;
- planner capable d'utiliser des deltas de features via des vecteurs
  `planner_feature_weights` / `planner_feature_targets` fournis par la config ;
- l'agent ne connaît toujours pas `food`, `tool`, `craft`, etc. Il manipule
  uniquement des vecteurs et des ids.

Premier entraînement `causalwm` jusqu'à 8000 vies :

```text
Final mean lifespan(100): 106.2
craft100=1.92 toolfood100=0.61 eat100=4.03
```

Évaluation :

```text
Naïf:              lifespan 62.2
Q-only:            lifespan 102.5
Q + WM planner:    lifespan 91.6
planner/Q:         0.89×
```

Conclusion : la représentation causale aide clairement l'apprentissage Q et
fait émerger du craft utile en évaluation, mais le planner gêne encore la
décision. Diagnostic : le WM choisit `EAT` trop souvent, même dans `craft_ready`.

**Rebalancing générique du World Model causal :**

Changements universels, non liés aux noms sandbox :

- `ExperienceBuffer.sample_causal()` pondère les événements par rareté
  générique (`event_index` / `event`), sans connaître leur sens ;
- `world_model.sampler: causal` ;
- `causal_world_model.event_class_balance: true` ;
- `feature_loss_weight: 2.0` ;
- `event_loss_weight: 0.2`.

Continuation du checkpoint 8000 vers 12000 vies :
`sandbox_v2_craft_balanced_causalwm_rebalanced_cuda`

```text
life 11999 | lifespan(100)=149.2
craft100=3.32 toolfood100=1.39 eat100=7.82 causal100=3.262
```

Évaluation finale documentée :

```text
Naïf:              lifespan 62.2 +/- 6.2
Q-only:            lifespan 115.7 +/- 109.9
Q + WM planner:    lifespan 124.0 +/- 123.5
planner/Q:         1.07×
```

Métriques causales :

```text
Q-only:         food=4.84 tool_food=0.33 craft=0.67 eat=4.71
Q + WM planner: food=5.62 tool_food=0.37 craft=0.35 eat=5.51
```

Conclusion expérimentale provisoire :

1. L'agent entraîné bat très nettement le naïf : **1.86×**.
2. Le craft est réellement présent en évaluation (`craft`, `tool_food`,
   `bonus_food_from_tool` > naïf).
3. Pour la première fois, `Q + World Model planner` bat `Q-only` :
   **124.0 vs 115.7**, soit **1.07×**.
4. Le gain planner porte surtout sur survie / food / eat ; il ne prouve pas
   encore une exploitation parfaite de la chaîne craft, car `craft` moyen baisse
   avec le planner.

Diagnostic causal après rebalancing :

```text
actual_craft_tool + action CRAFT:
  craft_tool: 0.71

craft_ready + action CRAFT:
  craft_tool: 0.66

actual_tool_food + action HARVEST:
  inventory_food delta: +0.032
```

Le World Model causal est donc devenu plus informatif, mais il reste imparfait :
il continue à survaloriser `EAT` dans certains états `craft_ready`.

**Statut de preuve :**

On a une preuve minimale que :

```text
perception causale générique + World Model auxiliaire + planner
peut améliorer la décision par rapport à Q-only.
```

On n'a pas encore une preuve robuste que le planner exploite proprement des
chaînes causales longues et composées. Cette distinction est importante pour la
suite du projet.

### D — Micro-fouloïde V0 / multi-pulsions homéostatiques

**Pourquoi :** sortir du sandbox "survie nue" et tester une régulation
homéostatique plus générale : énergie, hydratation, température, santé et
dangers locaux.

**Implémenté :**

- `MicroFouloideWorld` : organisme unique sur grille partiellement observable ;
- drives internes : `energy`, `hydration`, `temperature`, `health` ;
- entités : `FOOD`, `WATER`, `WARM_ZONE`, `COLD_ZONE`, `DANGER`,
  `OBSTACLE`, `UNKNOWN` ;
- actions : `MOVE_*`, `INTERACT`, `REST`, `WAIT` ;
- observation structurée `standing_entity` pour que l'agent puisse apprendre
  les conséquences de `INTERACT` sans règle codée ;
- encodeur dédié micro-fouloïde : grille one-hot + drives + entité sous agent ;
- runner/evaluator : `scripts/run_micro_fouloide.py`,
  `scripts/evaluate_micro_fouloide.py`.

**Changement architectural important :**

Le DQN peut maintenant optimiser une clé de reward configurable
(`dqn.reward_key`). Pour micro-fouloïde, le replay conserve la reward externe du
monde, mais la policy apprend sur `reward_learning`, qui ajoute un signal
générique de variation des drives :

```text
reward_learning = reward_external + weight * Δdrive_regulation
```

Ce n'est pas une règle "manger/boire" codée à la main. C'est un signal de
régulation interne basé sur des variables exposées par l'adapter, compatible
avec d'autres mondes qui exposeraient d'autres drives.

**Bug corrigé pendant V0 :**

Le premier essai exposait `standing_entity` en observation live, mais pas dans
`obs_state` compact stocké en replay. Le Q-network décidait donc avec une
feature que son entraînement ne voyait pas. Après correction, l'agent apprend
effectivement à interagir avec nourriture/eau.

#### Résultats micro-fouloïde V0 — juin 2026

Run initial 1000 épisodes :
`runs/micro_fouloide_v0_drive_reward_1000`

```text
Final mean lifespan(100): 142.9
food100=1.04 water100=0.92 dmg100=3.05 drive100=0.680
```

Évaluation 100 épisodes :

```text
Naïf:              lifespan 120.5 +/- 31.6
Q-only:            lifespan 123.5 +/- 24.0
Q + WM planner:    lifespan 125.3 +/- 25.7
```

Conclusion : direction correcte mais preuve faible. L'agent interagit à nouveau
avec nourriture/eau, mais le gain greedy reste marginal après 1000 épisodes.

Continuation jusqu'à 3000 épisodes :
`runs/micro_fouloide_v0_drive_reward_3000`

```text
ep 2999 | lifespan(100)=204.4
drive100=0.715 food100=2.25 water100=2.25 dmg100=2.31 eps=0.10
```

Évaluation 100 épisodes :

```text
Naïf:
  lifespan=120.5 +/- 31.6
  drive=0.661
  food=0.46 water=0.51 damage=4.23

Q-only:
  lifespan=192.9 +/- 72.7
  drive=0.691
  food=0.96 water=1.67 damage=0.80
  actions: move=43.2% interact=8.4% rest=46.2% wait=2.3%

Q + WM planner:
  lifespan=185.1 +/- 74.4
  drive=0.691
  food=1.13 water=1.60 damage=0.87
  actions: move=48.5% interact=10.9% rest=36.5% wait=4.2%
```

**Conclusion V0 :**

1. Le trained Q-only bat clairement le naïf : **192.9 vs 120.5**, soit
   **1.60×**.
2. L'amélioration ne vient pas seulement d'un artefact de lifespan :
   nourriture/eau augmentent et les dégâts baissent fortement.
3. Le drive moyen trained dépasse le naïf : **0.691 vs 0.661**.
4. Le planner manuel ne valide pas encore le rôle du World Model dans ce monde :
   `Q + WM planner` est sous `Q-only` (**0.96×**).

#### Robustesse multi-seed micro-fouloïde V0

Trois runs indépendants de 3000 épisodes confirment que le signal Q-only >
naïf n'est pas limité à une seule seed.

```text
seed | Q lifespan | Q/naive | Q drive | food | water | damage | planner/Q
1    | 153.8      | 1.28x   | 0.651   | 0.78 | 1.00  | 1.10   | 0.95x
2    | 145.6      | 1.21x   | 0.653   | 0.77 | 0.82  | 0.97   | 0.98x
3    | 173.4      | 1.44x   | 0.677   | 1.24 | 1.63  | 1.50   | 1.01x
```

Moyenne seeds 1-3 :

```text
Naive lifespan: 120.5
Q-only lifespan: 157.6
Q/naive: 1.31x

Q food:   0.93 vs naive 0.46
Q water:  1.15 vs naive 0.51
Q damage: 1.19 vs naive 4.23

Planner/Q moyen: ~0.98x
```

Conclusion multi-seed :

1. `Q-only > naive` sur les 3 seeds.
2. Le comportement appris est cohérent : plus de nourriture, plus d'eau,
   beaucoup moins de dégâts.
3. Le drive moyen reste plus fragile que le lifespan : seed 1/2 restent
   légèrement sous le naïf en `drive`, malgré un lifespan supérieur.
4. Le planner World Model manuel reste non validé sur V0 : neutre à négatif en
   moyenne.

**Statut de preuve :**

```text
Agent générique + observation structurée + reward de régulation interne
=> apprend une routine adaptative dans un monde multi-drives.
```

Cela valide la transition sandbox -> micro-fouloïde V0 côté policy, y compris
en première robustesse multi-seed. Cela ne valide pas encore l'hypothèse forte
"World Model exploité > Q-only" sur micro-fouloïde.

**Prochaines validations :**

1. Généralisation sur variantes de monde sans changer l'agent.
2. Diagnostic World Model sur les drives/events micro-fouloïde.
3. Mise de côté du planner manuel pour micro-fouloïde tant qu'une valeur
   apprise des conséquences n'est pas disponible.

#### Généralisation rough V0.1

Variante : `configs/micro_fouloide_v0_rough.yaml`.

Le monde reste compatible avec le même agent et le même adapter, mais devient
plus difficile :

- nourriture/eau plus rares ;
- hydratation/énergie décroissent plus vite ;
- température plus instable ;
- plus de dangers et d'obstacles ;
- santé un peu plus fragile.

Validation multi-seed 3000 épisodes :

```text
seed | Q lifespan | Q/naive | Q drive | food | water | damage | planner/Q
1    | 108.0      | 1.23x   | 0.636   | 0.42 | 0.57  | 1.22   | 0.97x
2    | 101.6      | 1.15x   | 0.643   | 0.23 | 0.32  | 1.12   | 0.95x
3    | 115.2      | 1.31x   | 0.663   | 0.32 | 0.65  | 0.77   | 1.00x
```

Moyenne rough seeds 1-3 :

```text
Naive lifespan: 88.1
Q-only lifespan: 108.3
Q/naive: 1.23x

Q food:   0.32 vs naive 0.30
Q water:  0.51 vs naive 0.20
Q damage: 1.04 vs naive 3.75

Planner/Q moyen: ~0.97x
```

Conclusion rough :

1. `Q-only > naive` sur les 3 seeds.
2. La généralisation du cadre tient sur une variante plus dure sans changer
   l'agent.
3. Le signal est plus faible que V0 : la politique apprend surtout à éviter les
   dangers et à trouver de l'eau ; la nourriture reste proche du naïf.
4. Le drive moyen reste fragile, mais le lifespan et les métriques
   comportementales sont meilleurs.
5. Le planner World Model reste non validé : neutre à négatif.

#### Diagnostic World Model micro-fouloïde

`scripts/evaluate_micro_fouloide.py` expose maintenant
`--diagnose-world-model`. Le diagnostic lit le replay du checkpoint et compare,
par type d'événement :

- erreur latent next-state ;
- erreur reward externe ;
- erreur delta de features causales ;
- accuracy de prédiction d'événement ;
- incertitude moyenne.

Commande type :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough.yaml \
  --num-episodes 1 \
  --device mps \
  --diagnose-world-model \
  --diagnostic-samples 20000
```

Premier diagnostic court sur rough seed1 (`1000` samples) :

```text
interact_food   event_acc=0.667 feature_mae=0.05370
interact_water  event_acc=0.833 feature_mae=0.06078
damage          event_acc=0.118 feature_mae=0.10716
death           event_acc=0.000 reward_mae=0.98121
temperature_*   event_acc=0.000 feature_mae≈0.104
rest/wait       event_acc≈0.99 feature_mae≈0.011
```

Interprétation provisoire :

Le World Model apprend bien les transitions fréquentes et simples
(`rest`, `wait`, `interact_noop`), mais reste faible sur les événements rares ou
critiques (`damage`, `death`, changements de température). Cela explique
probablement pourquoi le planner n'aide pas encore : il ne sait pas assez bien
anticiper les conséquences décisives.

Suite expérimentale créée : `configs/micro_fouloide_v0_rough_wmfocus.yaml`.

Cette config ne change pas le monde ni l'agent. Elle modifie uniquement
l'entraînement du World Model :

- `reward_abs_weight: 2.0` ;
- `reward_done_weight: 8.0` ;
- `event_loss_weight: 1.0` ;
- `event_class_balance_power: 1.0` ;
- `feature_loss_weight: 3.0`.

Objectif :

```text
améliorer la prédiction des événements rares/critiques
sans introduire de règle spécifique au micro-fouloïde.
```

Protocole A/B :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_wmfocus.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_wmfocus_seed1

python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_wmfocus_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_wmfocus.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-planner \
  --diagnose-world-model \
  --diagnostic-samples 20000
```

Validation recherchée :

1. `death` reward MAE baisse nettement ;
2. `damage` / `temperature_*` event accuracy augmente ;
3. `planner/Q` devient neutre positif, idéalement > 1.0 ;
4. `Q-only` ne régresse pas fortement.

Résultat `wmfocus` seed1 :

```text
Rough baseline seed1:
  Q-only lifespan=108.0
  planner/Q=0.97
  death reward_mae=0.96182 event_acc=0.032
  damage event_acc=0.087

Rough wmfocus seed1:
  Q-only lifespan=103.4
  planner/Q=0.99
  death reward_mae=0.71078 event_acc=0.231
  damage event_acc=0.253
```

Conclusion :

`wmfocus` améliore bien les événements critiques (`death`, `damage`,
`interact_food`), mais il dégrade trop la calibration reward globale :
`reward_mae` augmente fortement sur les transitions fréquentes
(`health_loss`, `move_ok`, `rest`, `wait`). La policy Q-only baisse légèrement
et le planner reste seulement neutre.

Décision : conserver l'idée, mais réduire l'intensité. Nouvelle config à tester :
`configs/micro_fouloide_v0_rough_wmfocus_light.yaml`.

Résultat `wmfocus_light` seed1 :

```text
Rough baseline seed1:
  Q-only lifespan=108.0
  planner/Q=0.97
  death reward_mae=0.96182 event_acc=0.032
  damage event_acc=0.087

Rough wmfocus_light seed1:
  Q-only lifespan=105.5
  planner/Q=1.01
  death reward_mae=0.79020 event_acc=0.061
  damage event_acc=0.140
  temperature_up event_acc=0.067
```

Conclusion :

`wmfocus_light` est un meilleur compromis que `wmfocus` : il garde une partie du
gain sur les événements critiques et le planner devient légèrement positif sur
seed1. Mais la reward reste mal calibrée sur beaucoup de transitions
fréquentes, et le gain planner est trop faible pour conclure.

Observation architecturale importante :

Le planner micro-fouloïde score encore principalement :

```text
predicted_reward_external + curiosity
```

Alors que la policy DQN apprend sur :

```text
reward_learning = reward_external + Δdrive_regulation
```

Il y a donc un décalage d'objectif. Avant de chercher un planner robuste, il
faut donner au planner une valeur apprise des conséquences, idéalement apprise
sur le même signal que la policy, plutôt que renforcer indéfiniment la reward
externe du World Model.

#### Planner avec valeur apprise

Implémentation ajoutée :

- `seedmind/agent/value_model.py` : `ValueModel(latent) -> value` ;
- `seedmind/training/value.py` : entraînement TD générique sur une clé de reward
  configurable ;
- checkpoints étendus avec `value_model_state`, `value_optimizer_state`,
  `target_value_model_state` ;
- `Planner` peut ajouter une valeur terminale après rollout imaginé ;
- `run_micro_fouloide.py` entraîne le ValueModel en parallèle si
  `value_model.enabled: true`.

Config A/B :

```text
configs/micro_fouloide_v0_rough_valueplanner.yaml
```

Principe :

```text
World Model -> imagine les conséquences
ValueModel  -> évalue le latent final selon reward_learning
Planner     -> predicted rewards + terminal learned value
```

Cette étape est plus alignée avec l'objectif global que les poids manuels
`wmfocus`, car la valeur des conséquences est apprise depuis le replay au lieu
d'être codée dans la config.

Commande :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_seed1

python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-planner \
  --diagnose-world-model \
  --diagnostic-samples 20000
```

Validation recherchée :

```text
Q + WM rollout + learned terminal value > Q-only
```

sur le même checkpoint. Si le signal apparaît sur une seed, refaire en
multi-seed.

Résultat `valueplanner` rough seed1 :

```text
Training:
  Final mean lifespan(100): 122.7
  drive100=0.664 food100=0.89 water100=1.14 dmg100=1.55

Evaluation:
  Naive:            lifespan=88.1  drive=0.654
  Q-only:          lifespan=114.7 drive=0.658
  Q + WM + Value:  lifespan=119.1 drive=0.661

Ratios:
  Q/naive:    1.30x
  planner/Q:  1.04x
```

Métriques comportementales :

```text
Q-only:         food=0.67 water=0.83 damage=1.32
Q + WM + Value: food=0.77 water=0.92 damage=1.30
Naive:          food=0.30 water=0.20 damage=3.75
```

Diagnostic WM :

```text
interact_food  event_acc=0.768
interact_water event_acc=0.776
damage         event_acc=0.091
death          event_acc=0.013
temperature_*  event_acc=0.000
```

Conclusion provisoire :

Le planner aligné par `ValueModel` donne le premier signal positif sur
micro-fouloïde rough : **planner/Q = 1.04×**. Le gain s'accompagne d'un peu plus
de nourriture/eau et d'un drive moyen légèrement meilleur. Le World Model reste
toutefois faible sur `death`, `damage` et température ; le signal doit donc être
validé en multi-seed avant conclusion.

Validation multi-seed `valueplanner` rough :

```text
seed | Q lifespan | Planner lifespan | planner/Q | Q drive | Planner drive | Q food/water/damage | Planner food/water/damage
1    | 114.7      | 119.1            | 1.04x     | 0.658   | 0.661         | 0.67 / 0.83 / 1.32  | 0.77 / 0.92 / 1.30
2    | 102.8      | 104.8            | 1.02x     | 0.657   | 0.662         | 0.24 / 0.30 / 0.61  | 0.28 / 0.35 / 0.55
3    | 107.4      | 105.3            | 0.98x     | 0.641   | 0.639         | 0.55 / 0.67 / 1.50  | 0.52 / 0.63 / 1.44
```

Moyenne seeds 1-3 :

```text
Naive lifespan: 88.1
Q-only lifespan: 108.3
Q + WM + Value lifespan: 109.7

Q/naive:       1.23x
Planner/naive: 1.25x
Planner/Q:     ~1.01x

Q-only:         drive=0.652 food=0.49 water=0.60 damage=1.14
Q + WM + Value: drive=0.654 food=0.52 water=0.63 damage=1.10
```

Conclusion multi-seed :

Le planner aligné par `ValueModel` donne un **signal positif moyen**, mais
encore faible : +1.3% de lifespan environ, drive légèrement meilleur, un peu
plus de nourriture/eau et un peu moins de dégâts. Seed3 reste négative, donc ce
n'est pas encore une preuve forte. C'est néanmoins le premier résultat
micro-fouloïde où un planner basé sur World Model et valeur apprise améliore
Q-only en moyenne sur plusieurs seeds.

Sweep d'évaluation ajouté pour consolider sans réentraîner :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 100 \
  --device mps \
  --planner-sweep 0,0.025,0.05,0.1,0.15,0.25 \
  --terminal-value-sweep 0,0.5,1,1.5,2 \
  --planner-horizon 3 \
  --planner-samples 8
```

Même commande à répéter sur `seed2` et `seed3`. Le sweep teste la sensibilité du
planner au poids de mélange Q/WM et au poids de valeur terminale apprise, sur le
même checkpoint.

Résultats sweep seeds 1-3 :

```text
seed | Q-only | meilleur planner | meilleur ratio | meilleur réglage
1    | 114.7  | 116.6            | 1.02x          | p=0.10 tv=1.5
2    | 102.8  | 106.5            | 1.04x          | p=0.10 tv=2.0
3    | 107.4  | 108.0            | 1.01x          | p=0.25 tv=1.5
```

Lecture :

```text
Le planner peut battre Q-only sur chaque seed si on choisit le meilleur réglage
par seed, mais le gain reste faible et les réglages optimaux ne sont pas
stables. Les poids trop élevés de planning, surtout p=0.25, dégradent souvent
les seeds 1-2.
```

Réglages communs intéressants :

```text
p=0.10 tv=1.5 -> seed1 116.6, seed2 105.7, seed3 105.3
p=0.15 tv=1.5 -> seed1 116.5, seed2 105.2, seed3 106.4
```

Conclusion sweep :

Le sweep confirme que le signal planner n'est pas nul, mais il reste trop faible
pour en faire une preuve robuste. La prochaine amélioration doit probablement
porter sur la qualité/stabilité du World Model et du ValueModel, plus que sur un
réglage manuel du poids planner.

Note méthodologique : le planner random-shooting est maintenant seedé via le
seed de l'agent afin de rendre ces comparaisons reproductibles.

Diagnostic suivant ajouté : `--diagnose-value-model`.

Commande type :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --device mps \
  --diagnose-value-model \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Le diagnostic compare `ValueModel(latent)` au retour Monte Carlo réel du replay,
avec le même `reward_key` et le même `gamma` que l'entraînement du ValueModel.
Il rapporte `mae`, `bias` et corrélation par buckets génériques dérivés des
features causales exposées par l'environnement.

Premier smoke diagnostic seed1 (`5000` samples, CPU) :

```text
all:           corr=0.427 mae=0.2726 bias=-0.0183
low_energy:    corr=0.382 mae=0.2982 bias=+0.1861
low_hydration: corr=0.364 mae=0.3049 bias=+0.1643
low_health:    corr=0.285 mae=0.3517 bias=+0.2410
terminal:      corr=-0.298 mae=0.4679 bias=+0.4679
```

Lecture provisoire :

Le ValueModel apprend un signal global modérément corrélé, mais il sous-estime
la gravité des états critiques et terminaux. Cela peut expliquer pourquoi le
planner améliore parfois la décision mais reste fragile.

Évolution implémentée : entraînement ValueModel pondéré, générique.

Nouveaux paramètres optionnels de `value_model` :

```text
target_abs_weight  # augmente le poids des cibles TD fortes
terminal_weight    # augmente le poids des transitions terminales
td_error_weight    # augmente le poids des exemples encore mal prédits
max_weight         # borne la pondération
```

Ces paramètres ne dépendent d'aucun événement micro-fouloïde. Ils utilisent
uniquement les signaux RL génériques du replay : cible TD, terminalité et erreur
de valeur.

Nouvelle config :

```text
configs/micro_fouloide_v0_rough_valueplanner_focus.yaml
```

Commande de test seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_focus.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_focus_seed1
```

Évaluation après entraînement :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_focus_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_focus.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-planner \
  --diagnose-value-model \
  --diagnostic-samples 20000 \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8
```

Critère de validation :

```text
low_health/terminal MAE baisse
corr globale monte
planner/Q devient plus stable ou dépasse clairement la config valueplanner
```

Résultat multi-seed `valueplanner_focus` :

```text
seed | Q-only | planner | planner/Q | value corr | low_health mae | terminal mae
1    | 101.8  | 99.7    | 0.98x     | 0.483      | 0.1997         | 0.1809
2    | 101.8  | 98.3    | 0.97x     | 0.530      | 0.1728         | 0.1394
3    | 119.6  | 112.2   | 0.94x     | 0.549      | 0.1971         | 0.1675
```

Comparaison au diagnostic `valueplanner` seed1 :

```text
valueplanner seed1:
  corr all=0.427 low_health_mae=0.3517 terminal_mae=0.4679

valueplanner_focus seed1:
  corr all=0.483 low_health_mae=0.1997 terminal_mae=0.1809
```

Conclusion :

Le focus ValueModel réussit son objectif local : la valeur devient nettement
meilleure sur `low_health` et `terminal`. Mais il dégrade la performance agent
et le planner devient négatif sur les 3 seeds. Cela indique que mieux apprendre
les fins critiques ne suffit pas : le ValueModel est devenu trop pessimiste en
moyenne (`value_mean` plus négatif que les retours réels), et ce pessimisme
perturbe la planification.

Décision :

```text
Ne pas garder valueplanner_focus comme config gagnante.
Garder le diagnostic ValueModel et les poids optionnels.
Prochaine piste : calibration de valeur, pas pondération plus forte.
```

Pivot Dyna latent minimal :

Le focus de valeur a montré que corriger un diagnostic local ne suffit pas. Pour
tester directement l'hypothèse "World Model utile à l'apprentissage", une mise à
jour Dyna latente a été ajoutée au `ValueModel` :

```text
Replay réel latent s,a
WorldModel imagine s', r
ValueModel apprend V(s) ≈ r + gamma * V_target(s')
```

Cette version est volontairement prudente :

- elle ne crée pas encore d'observations synthétiques ;
- elle ne modifie pas directement le DQN observationnel ;
- elle entraîne seulement la valeur latente auxiliaire utilisée par le planner ;
- elle est désactivée par défaut.

Nouvelle config :

```text
configs/micro_fouloide_v0_rough_valueplanner_dyna.yaml
```

Commande de test seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_dyna.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_dyna_seed1
```

Évaluation :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_dyna_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_dyna.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-planner \
  --diagnose-value-model \
  --diagnostic-samples 20000 \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8
```

Critère :

```text
Si Dyna latent améliore le ValueModel sans pessimisme global et augmente
planner/Q, on tient une meilleure piste que le planner-only.
```

Résultat multi-seed `valueplanner_dyna` :

```text
seed | Q-only | planner | planner/Q | value corr | value bias | terminal mae
1    | 106.2  | 107.8   | 1.02x     | 0.440      | -0.0910    | 0.3922
2    | 102.8  | 100.7   | 0.98x     | 0.514      | -0.0790    | 0.3229
3    | 120.8  | 119.9   | 0.99x     | 0.529      | -0.0073    | 0.3814
```

Comparaison synthétique :

```text
valueplanner baseline:
  planner/Q moyen ≈ 1.01x
  Q-only moyen ≈ 108.3

valueplanner_focus:
  planner/Q moyen ≈ 0.96x
  Q-only moyen ≈ 107.7
  meilleur diagnostic terminal, mais pessimisme global

valueplanner_dyna:
  planner/Q moyen ≈ 1.00x
  Q-only moyen ≈ 109.9
  moins pessimiste que focus, mais terminal encore mal appris
```

Conclusion :

Le Dyna latent minimal est une meilleure direction que le focus pondéré : il ne
rend pas la valeur globalement aussi pessimiste et il préserve mieux la policy.
Mais il ne donne pas encore une preuve robuste que `WM improves learning`.
L'effet planner reste essentiellement neutre, avec seed1 positif et seeds 2-3
légèrement négatives.

Décision :

```text
Arrêter les variantes ValueModel seules.
Le prochain saut doit être architectural : représentation/value en latent
directement exploitable par la policy, ou Dyna qui entraîne une policy latente,
pas seulement une valeur terminale branchée au planner.
```

Pivot implémenté : policy Q latente.

Nouveaux modules :

```text
seedmind/agent/latent_q_network.py
seedmind/training/latent_dqn.py
```

Principe :

```text
Encoder(obs) -> latent
LatentQ(latent, action) -> Q
```

Le `LatentQNetwork` est entraîné en parallèle du Q observationnel, sur les mêmes
transitions réelles du replay. Il est sauvegardé dans les checkpoints et peut
être évalué avec `--compare-latent-q`. C'est le premier pas vers une policy qui
utilise directement la représentation apprise par le World Model.

Config de test :

```text
configs/micro_fouloide_v0_rough_latentq.yaml
```

Commande seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_latentq.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_latentq_seed1
```

Évaluation :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-latent-q \
  --compare-planner \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8
```

Critère :

```text
LatentQ >= Q-only sur lifespan ou drive, sans collapse comportemental.
Si oui, activer ensuite Dyna sur LatentQ plutôt que sur ValueModel seul.
```

Résultat multi-seed `latentq` :

```text
seed | Q-only | LatentQ | LatentQ/Q | Planner | Planner/Q | observation
1    | 112.2  | 92.9    | 0.83x     | 110.2   | 0.98x     | LatentQ sur-interagit
2    | 103.5  | 96.2    | 0.93x     | 104.5   | 1.01x     | LatentQ sous-performe
3    | 111.9  | 92.8    | 0.83x     | 108.7   | 0.97x     | LatentQ sous-performe
```

Comportement LatentQ :

```text
seed1: interact=82.2%, food=0.04, water=0.04
seed2: interact=60.2%, food=0.03, water=0.03
seed3: interact=46.5%, food=0.03, water=0.08
```

Conclusion :

Le latent actuel n'est pas encore un bon support direct de policy. LatentQ
apprend une préférence d'action dégénérée, surtout `INTERACT`, sans capter assez
la condition contextuelle "interagir seulement quand l'entité utile est sous
l'agent". Cela indique que le latent est suffisant pour la prédiction WM, mais
pas assez structuré ou stable pour remplacer le Q observationnel.

Décision :

```text
Ne pas activer Dyna sur LatentQ maintenant.
Diagnostiquer d'abord le contenu informationnel du latent pour la décision :
peut-on prédire les features causales et l'entité sous l'agent depuis le latent ?
```

Diagnostic ajouté :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq.yaml \
  --device mps \
  --diagnose-latent \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Ce diagnostic entraîne des probes linéaires hors-policy depuis `latent_state` :

```text
latent_state -> causal_features
latent_state -> standing_entity
```

Lecture attendue :

- si les features causales et `standing_entity` sont mal décodables, il faut
  renforcer la représentation latente avant de travailler LatentQ/Dyna ;
- si elles sont bien décodables, le problème est plutôt côté apprentissage de
  policy latente : équilibre d'actions, masques, objectifs ou échelle de valeur.

Smoke test local sur seed1, 1000 échantillons :

```text
standing_entity probe:
  acc=0.860 baseline_majority=0.870 balanced_acc=0.473 gain=-0.010
  top_pred=0:0.85,6:0.07,5:0.06
```

Résultat multi-seed complet sur 20000 échantillons :

```text
seed | entity acc | majority | gain  | standing_entity corr | food_signal corr | water_signal corr
1    | 0.911      | 0.889    | +0.022| 0.506                | 0.190            | 0.222
2    | 0.917      | 0.898    | +0.019| 0.475                | 0.208            | 0.219
3    | 0.906      | 0.882    | +0.024| 0.448                | 0.214            | 0.238
```

Lecture :

Le latent encode bien une partie des drives internes (`energy`, `hydration`,
`health`, corr ≈ 0.37-0.50) et un faible signal sur `standing_entity`. Mais le
gain d'accuracy sur la majorité est seulement **+2 points** environ, dans un
problème très déséquilibré où la classe dominante vaut déjà ~0.89. Les signaux
locaux utiles à l'interaction (`local_food_signal`, `local_water_signal`,
`local_danger`) restent faibles.

Conclusion :

Le latent n'est pas vide, mais il n'est pas assez discriminant pour une policy
latente directe. Cela explique le collapse LatentQ : le réseau apprend qu'il
faut souvent `INTERACT`, sans disposer d'une représentation suffisamment nette
pour savoir **quand** interagir.

Décision suivante :

```text
Ne pas ajouter Dyna sur LatentQ.
Renforcer d'abord la représentation latente avec des objectifs auxiliaires
génériques de reconstruction/prédiction des features perceptibles courantes.
```

Implémentation pragmatique ajoutée : `structured_latent_features`.

Comme l'encodeur actuel est gelé, entraîner une tête auxiliaire seule ne
modifierait pas le latent. La première version utile est donc un skip perceptif
générique :

```text
latent = [projection_observation, adapter.causal_features(observation)]
```

L'agent ne connaît toujours pas le sens de ces features. Il reçoit seulement un
vecteur de capteurs structurés exposés par l'`EnvironmentAdapter`. Cela reste
agnostique au monde, mais donne à LatentQ un support de décision plus net.

Nouvelle config :

```text
configs/micro_fouloide_v0_rough_latentq_structured.yaml
```

Commande de test seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_latentq_structured.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_latentq_structured_seed1
```

Évaluation :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_structured_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq_structured.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-latent-q \
  --compare-planner \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8
```

Diagnostic latent :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_structured_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq_structured.yaml \
  --device mps \
  --diagnose-latent \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Smoke test local sur 5 épisodes :

```text
causal feature probe: mae=0.0000, corr=1.000 sur toutes les features
standing_entity probe:
  acc=0.966 baseline_majority=0.798 balanced_acc=0.962 gain=+0.169
```

Critère de validation :

```text
LatentQ structured > LatentQ précédent
et idéalement LatentQ structured >= Q-only
sans collapse massif vers INTERACT.
```

Résultat multi-seed `latentq_structured` :

```text
seed | Q-only | LatentQ structured | LatentQ/Q | Planner | Planner/Q | observation
1    | 107.8  | 100.6             | 0.93x     | 106.8   | 0.99x     | LatentQ préfère REST/WAIT
2    | 101.1  | 96.3              | 0.95x     | 99.9    | 0.99x     | LatentQ sous Q-only
3    | 105.4  | 91.3              | 0.87x     | 106.5   | 1.01x     | LatentQ sous Q-only
```

Conclusion :

Le skip perceptif corrige l'information du latent, mais ne suffit pas. LatentQ
reste inférieur à Q-only et ne devient pas une policy fiable. Le problème se
déplace donc de la représentation vers l'alignement de l'objectif/policy
latente.

Diagnostic ajouté : `--diagnose-latent-q`.

Commande :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_structured_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq_structured.yaml \
  --device mps \
  --diagnose-latent-q \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Smoke diagnostic seed1, 5000 échantillons :

```text
best_action_agreement=0.123
q_matches_replay=0.236
latent_matches_replay=0.149
q_best:      MOVE_UP:0.35 MOVE_DOWN:0.20 MOVE_LEFT:0.17 REST:0.11
latent_best: REST:0.35 WAIT:0.19 MOVE_DOWN:0.13 INTERACT:0.12
```

Lecture :

LatentQ n'imite pas la policy observationnelle apprise. Il apprend une surface
de valeur différente, plus orientée `REST/WAIT`, et son accord avec Q-only est
très faible. La prochaine expérience doit donc ancrer LatentQ sur Q-only.

Nouvelle config : LatentQ structuré + distillation Qobs.

```text
configs/micro_fouloide_v0_rough_latentq_structured_distill.yaml
```

Principe :

```text
LatentQ TD loss + distill_weight * MSE(centered LatentQ, centered Qobs)
```

Cette distillation est générique : elle n'encode aucune règle du monde. Elle
force seulement la policy latente à apprendre les préférences d'action d'une
autre policy apprise qui fonctionne mieux.

Commande seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_latentq_structured_distill.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_latentq_structured_distill_seed1
```

Évaluation :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_structured_distill_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq_structured_distill.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-latent-q \
  --compare-planner \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8
```

Diagnostic d'alignement :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_structured_distill_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq_structured_distill.yaml \
  --device mps \
  --diagnose-latent-q \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Critère :

```text
best_action_agreement monte nettement
LatentQ/Q se rapproche de 1.0 ou dépasse Q-only
et LatentQ ne collapse ni vers INTERACT, ni vers REST/WAIT.
```

Résultat multi-seed `latentq_structured_distill` avec distillation MSE centrée :

```text
seed | Q-only | LatentQ distill | LatentQ/Q | Planner | Planner/Q | latent_best dominant
1    | 107.8  | 97.1            | 0.90x     | 106.8   | 0.99x     | INTERACT / WAIT
2    | 101.1  | 96.9            | 0.96x     | 99.9    | 0.99x     | INTERACT / MOVE_UP / REST
3    | 105.4  | 94.8            | 0.90x     | 106.5   | 1.01x     | MOVE_LEFT / INTERACT
```

Diagnostics d'alignement :

```text
seed | agreement | q_matches_replay | latent_matches_replay | q_margin | latent_margin
1    | 0.121     | 0.245            | 0.137                 | 0.0130   | 0.0048
2    | 0.219     | 0.236            | 0.153                 | 0.0104   | 0.0037
3    | 0.236     | 0.239            | 0.162                 | 0.0118   | 0.0061
```

Conclusion :

La distillation MSE centrée échoue. Les valeurs Qobs ont des marges très faibles
(`q_margin≈0.01`), donc imiter les valeurs centrées produit un signal trop mou.
LatentQ reste mal aligné avec Q-only et choisit des actions dominantes
différentes.

Nouvelle variante : distillation de politique.

```text
configs/micro_fouloide_v0_rough_latentq_structured_policy_distill.yaml
```

Principe :

```text
LatentQ TD loss + distill_weight * CE(LatentQ logits, argmax(Qobs))
```

Ce test est plus strict : on ne demande plus à LatentQ d'imiter une échelle de
valeurs ambiguë, mais directement l'action préférée par le teacher Qobs.

Commande seed1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_latentq_structured_policy_distill.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_latentq_structured_policy_distill_seed1
```

Évaluation et diagnostic : mêmes commandes que `structured_distill`, avec
`structured_policy_distill` dans les chemins/config.

Critère :

```text
agreement Qobs/LatentQ doit monter fortement.
Si ce n'est pas le cas, arrêter la branche LatentQ séparée.
```

Résultat multi-seed `latentq_structured_policy_distill` :

```text
seed | Q-only | LatentQ policy-distill | LatentQ/Q | Planner | Planner/Q | comportement LatentQ
1    | 107.8  | 96.5                  | 0.90x     | 106.8   | 0.99x     | move=96.8%, interact=1.8%
2    | 101.1  | 94.8                  | 0.94x     | 99.9    | 0.99x     | move=98.0%, interact=2.0%
3    | 105.4  | 93.1                  | 0.88x     | 106.5   | 1.01x     | move=91.6%, interact=7.2%
```

Diagnostics d'alignement :

```text
seed | agreement | q_matches_replay | latent_matches_replay | q_margin | latent_margin | latent_best
1    | 0.308     | 0.245            | 0.173                 | 0.0130   | 0.1530        | MOVE_UP / MOVE_LEFT
2    | 0.286     | 0.236            | 0.168                 | 0.0104   | 0.1980        | MOVE_LEFT / MOVE_RIGHT
3    | 0.282     | 0.239            | 0.183                 | 0.0118   | 0.1275        | MOVE_DOWN / MOVE_UP
```

Conclusion :

La distillation de politique augmente l'alignement brut par rapport à MSE, mais
elle transforme LatentQ en politique quasi exclusivement de déplacement. Elle
reste sous Q-only sur les 3 seeds et n'apprend pas l'arbitrage
déplacement/interactions/repos. La branche `LatentQ` séparée est donc arrêtée.

Décision :

```text
STOP LatentQ séparé.
Ne pas ajouter Dyna sur LatentQ.
Ne pas chercher à faire du latent une policy concurrente au Q observationnel
dans cette architecture.
```

Interprétation :

Le latent structuré est utile comme **support de modèle du monde** et de
diagnostic, mais pas encore comme substrat direct de policy. Pour prouver
l'apport du World Model, la suite doit agir là où le signal est déjà stable :

```text
1. garder Q observationnel comme policy principale ;
2. utiliser le World Model pour améliorer l'apprentissage ou l'évaluation de Q ;
3. éviter les policies auxiliaires concurrentes tant qu'elles ne battent pas Q.
```

### B — Consolidation planification (à reprendre)

Pistes identifiées :

1. **Planning différé** : `planning_weight` monte seulement quand la loss WM est sous un seuil
2. **Dyna/augmentation pour Q observationnel** : le WM doit aider la policy
   principale, pas entraîner une policy latente séparée ;
3. **Planification seule en eval** : entraîner au DQN pur, planifier seulement à l'inférence ;
4. **Gating du planner** : activer le planner seulement dans les états où il
   améliore une métrique prédite avec confiance.

Implémenté : planner gated par confiance.

Le planner expose maintenant, pour chaque action candidate :

```text
WM rollout value
mean rollout uncertainty
```

L'agent peut utiliser deux seuils génériques :

```text
planning.uncertainty_threshold
planning.margin_threshold
```

Principe :

```text
si WM incertain ou marge WM trop faible:
  utiliser Q-only
sinon:
  mélanger Q + WM selon planning.weight
```

Ce mécanisme ne dépend d'aucun événement micro-fouloïde. Il teste directement
l'hypothèse : un World Model doit aider seulement quand il prédit des
conséquences avec assez de confiance.

Évaluation CLI :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 100 \
  --device mps \
  --compare-planner \
  --planning-weight 0.1 \
  --terminal-value-weight 1.5 \
  --planner-horizon 3 \
  --planner-samples 8 \
  --planner-uncertainty-threshold 0.65 \
  --planner-margin-threshold 0.0
```

Smoke tests locaux :

```text
threshold=-1:
  planner_used=0%, résultat identique à Q-only

threshold=999:
  planner_used=100%, résultat différent de Q-only
```

Critère de validation :

```text
planner_used doit être partiel, pas 0% ni 100%.
planner/Q doit être >= 1.0 sur moyenne multi-seed.
Si planner_used est partiel mais planner/Q < 1.0, le problème reste la qualité
des conséquences imaginées, pas seulement leur gating.
```

Résultat multi-seed `valueplanner` avec gate confiance :

```text
config: configs/micro_fouloide_v0_rough_valueplanner.yaml
planning_weight=0.1
terminal_value_weight=1.5
planner_horizon=3
planner_samples=8
uncertainty_threshold=0.65
margin_threshold=0.0
```

seed | Q-only | Q + WM gated | planner/Q | planner_used | drive Q -> WM | food/water/damage Q -> WM
-----|--------|--------------|-----------|--------------|---------------|---------------------------
1    | 114.7  | 115.5        | 1.01x     | 100.0%       | 0.658 -> 0.659 | 0.67/0.83/1.32 -> 0.71/0.82/1.23
2    | 102.8  | 103.8        | 1.01x     | 37.3%        | 0.657 -> 0.659 | 0.24/0.30/0.61 -> 0.24/0.33/0.55
3    | 107.4  | 107.4        | 1.00x     | 12.0%        | 0.641 -> 0.641 | 0.55/0.67/1.50 -> 0.55/0.68/1.47

Synthèse :

```text
Q-only moyen:       108.3
Q + WM gated moyen: 108.9
gain moyen:         +0.6 lifespan, ≈1.006x
```

Interprétation :

1. Le gate de confiance n'est pas une preuve forte du planner, mais il retire
   l'effet destructeur observé sur plusieurs variantes.
2. Le signal est petit mais orienté dans le bon sens : le WM devient un
   conseiller contrôlé plutôt qu'une policy concurrente.
3. Cette direction est plus cohérente avec l'objectif global : un World Model
   générique qui apprend cause/conséquence et aide la décision principale
   seulement quand ses prédictions sont assez fiables.
4. La suite immédiate est un sweep automatisé des seuils `uncertainty` et
   `margin`, pour trouver une zone robuste sans réglage à la main seed par seed.

Commande de sweep confiance :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 100 \
  --device mps \
  --planner-sweep 0.05,0.1,0.15 \
  --terminal-value-sweep 1.0,1.5,2.0 \
  --planner-uncertainty-sweep 0.55,0.6,0.65,0.7 \
  --planner-margin-sweep 0,0.01,0.02 \
  --planner-horizon 3 \
  --planner-samples 8
```

Résultat du sweep confiance sur les 3 seeds :

```text
Q-only moyen: 108.3
```

Deux réglages communs ressortent :

```text
Réglage safe:
  planning_weight=0.15
  terminal_value_weight=1.0
  uncertainty_threshold=0.60
  margin_threshold=0.0

  seed1: 117.1 vs Q 114.7, planner_used=98.8%
  seed2: 103.5 vs Q 102.8, planner_used=7.7%
  seed3: 107.4 vs Q 107.4, planner_used=0.5%
  moyenne: 109.3, soit +1.0 lifespan environ
```

Ce réglage est conservateur : il bat ou égale Q-only, mais sur deux seeds il
revient presque entièrement à Q-only. Il valide surtout que le gate sait
empêcher le planner de nuire.

```text
Réglage actif:
  planning_weight=0.15
  terminal_value_weight=2.0
  uncertainty_threshold=0.70
  margin_threshold=0.01

  seed1: 114.6 vs Q 114.7, planner_used=87.2%
  seed2: 107.2 vs Q 102.8, planner_used=70.6%
  seed3: 107.6 vs Q 107.4, planner_used=41.9%
  moyenne: 109.8, soit +1.5 lifespan environ
```

Ce réglage est le plus intéressant pour la thèse WM : le planner est vraiment
utilisé sur les 3 seeds et la moyenne passe au-dessus de Q-only, sans collapse.
Le gain reste faible, mais il n'est plus seulement dû à une désactivation du
planner.

Conclusion du sweep :

```text
Le signal planner existe, mais il est petit.
Le gate de confiance est utile et générique.
Le meilleur réglage commun actif donne environ 109.8 vs 108.3 Q-only,
soit ~1.014x.
```

Interprétation :

1. On ne peut pas encore dire que le World Model est "au top".
2. On peut dire qu'il commence à fournir une aide mesurable quand il est
   utilisé comme conseiller contrôlé.
3. Le blocage principal n'est plus seulement le planner ; c'est la qualité et
   la calibration des conséquences imaginées par le World Model.
4. La prochaine étape utile est donc d'améliorer le World Model lui-même :
   meilleure calibration d'incertitude, meilleur apprentissage des transitions
   rares/critiques, et métriques de corrélation entre incertitude et erreur.

Diagnostic de calibration ajouté :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --device mps \
  --diagnose-wm-calibration \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Smoke diagnostic sur seed1, 5000 transitions :

```text
correlation uncertainty -> state_mse:      -0.017
correlation uncertainty -> reward_err:     +0.037
correlation uncertainty -> feature_mae:    -0.011
correlation uncertainty -> event_miss:     -0.007
correlation uncertainty -> composite_rank: +0.072
top20 uncertainty captures top20 errors:    18.6%
```

Lecture : l'incertitude actuelle n'est pas calibrée. Elle ne prédit presque pas
les erreurs réelles du World Model, et son top 20% capture moins d'erreurs que
le hasard attendu autour de 20%. Le gate fonctionne donc surtout comme un filtre
empirique, pas encore comme une vraie confiance apprise.

Correction générique ajoutée :

```text
world_model.uncertainty_loss_weight
```

Quand ce poids est > 0, la tête `uncertainty` apprend à prédire l'erreur
composite détachée du World Model :

```text
latent next-state error
+ reward prediction error
+ causal feature delta error
+ event prediction error
```

Cette cible ne dépend d'aucun événement micro-fouloïde. Elle entraîne seulement
le modèle à savoir quand ses propres prédictions sont mauvaises.

Nouvelle config de test :

```text
configs/micro_fouloide_v0_rough_valueplanner_calibrated.yaml
```

Commande proposée :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_calibrated.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_calibrated_seed1
```

Critère de validation :

```text
1. diagnose-wm-calibration doit montrer une corrélation positive claire.
2. top20 uncertainty capture top20 errors doit dépasser nettement 20%.
3. Le planner gated doit rester >= Q-only en moyenne multi-seed.
```

Résultat seed1 calibré :

```text
train final lifespan100: 106.1

diagnose-wm-calibration:
  state_mse corr:       +0.340
  reward_abs_err corr:  +0.120
  feature_mae corr:     +0.405
  event_miss corr:      +0.292
  composite_rank corr:  +0.610
  top20 capture:         48.1%
```

Conclusion calibration :

```text
La tête uncertainty est maintenant réellement informative.
```

Évaluation planner avec l'ancien seuil actif :

```text
threshold=0.70
margin=0.01
planning_weight=0.15
terminal_value_weight=2.0

Q-only:       lifespan 105.6
Q + planner: lifespan 105.3
planner/Q:   1.00x
planner_used=86.9%
```

Lecture :

Le résultat n'invalide pas la calibration. Il montre surtout que le seuil
`0.70`, utile avec l'ancienne incertitude non calibrée, est trop haut pour la
nouvelle échelle calibrée. Après calibration, les bins d'incertitude observés
sont environ :

```text
q1=0.06 q2=0.14 q3=0.23 q4=0.29 q5=0.36
```

Le prochain sweep doit donc tester des seuils autour de cette nouvelle échelle,
pas réutiliser `0.70`.

Sweep planner sur checkpoint calibré seed1 :

```text
Q-only calibré: 105.6
meilleur planner calibré: 107.5
gain planner interne: +1.9 lifespan, ≈1.02x
```

Bons réglages observés :

```text
planning_weight=0.05
terminal_value_weight=1.0
uncertainty_threshold=0.36
margin_threshold=0.0
planner_used=99.1%
lifespan=107.5

planning_weight=0.15
terminal_value_weight=2.0
uncertainty_threshold=0.30
margin_threshold=0.01
planner_used=77.5%
lifespan=107.5
```

Lecture :

1. La calibration de l'incertitude est validée.
2. Le planner peut de nouveau apporter un petit gain sur le checkpoint calibré.
3. Mais le Q-only calibré descend par rapport au `valueplanner` non calibré
   seed1 (`105.6` vs `114.7` en évaluation Q-only).
4. `uncertainty_loss_weight=0.2` est probablement trop fort : il améliore la
   confiance du WM mais coûte trop à l'apprentissage comportemental.

Décision suivante :

```text
Ne pas lancer seeds 2/3 sur uncertainty_loss_weight=0.2.
Tester une calibration plus légère sur seed1.
```

Résultat `calibrated_light` (`uncertainty_loss_weight=0.05`) seed1 :

```text
Q-only:          106.3
meilleur planner: 109.0
gain planner:    +2.7, ≈1.03x

meilleur réglage planner:
  planning_weight=0.10
  terminal_value_weight=1.0
  uncertainty_threshold=0.36
  margin_threshold=0.02
  planner_used=55.7%
```

Diagnostic calibration :

```text
state_mse corr:       +0.379
reward_abs_err corr:  +0.096
feature_mae corr:     +0.378
event_miss corr:      +0.265
composite_rank corr:  +0.574
top20 capture:         45.4%
```

Lecture :

```text
0.05 conserve presque toute la calibration de 0.2
mais coûte encore du Q-only par rapport au valueplanner baseline seed1.
```

Décision :

Tester une calibration encore plus légère (`uncertainty_loss_weight=0.01`) pour
voir si on récupère le Q-only tout en gardant une incertitude suffisamment
corrélée aux erreurs réelles.

Résultat `calibrated_tiny` (`uncertainty_loss_weight=0.01`) seed1 :

```text
Q-only:           108.3
meilleur planner: 109.1
gain planner:     +0.8, ≈1.01x

diagnose-wm-calibration:
  state_mse corr:       +0.310
  reward_abs_err corr:  +0.051
  feature_mae corr:     +0.365
  event_miss corr:      +0.261
  composite_rank corr:  +0.484
  top20 capture:         37.8%
```

Lecture :

```text
La supervision d'incertitude est générique et fonctionne, même à faible poids.
Mais même 0.01 ne récupère pas le Q-only du valueplanner non calibré seed1
(`108.3` vs `114.7`). Le problème n'est donc pas seulement le poids : le loss
d'incertitude partage encore le tronc du World Model et peut perturber les
représentations utiles à la dynamique/récompense.
```

Décision suivante :

```text
Découpler la calibration d'incertitude.
La tête uncertainty doit apprendre l'erreur composite du WM sans envoyer de
gradient dans le tronc partagé. Objectif : conserver la calibration utile
pour le planner sans dégrader l'apprentissage dynamique/Q-only.
```

Nouvelle config de test :

```text
configs/micro_fouloide_v0_rough_valueplanner_calibrated_head.yaml
world_model.uncertainty_loss_weight = 0.05
world_model.uncertainty_detach = true
```

Résultat `calibrated_head` seed1 :

```text
train final lifespan100: 105.3

diagnose-wm-calibration:
  state_mse corr:       +0.396
  reward_abs_err corr:  +0.097
  feature_mae corr:     +0.409
  event_miss corr:      +0.316
  composite_rank corr:  +0.619
  top20 capture:         45.1%

eval Q-only:       103.2
best planner sweep: 105.9
best planner/Q:     1.03x
```

Lecture :

```text
Le découplage head-only donne la meilleure calibration d'incertitude observée,
mais ne restaure pas le comportement Q-only. Le gain planner est réel en
interne au checkpoint, mais le checkpoint lui-même est trop faible par rapport
au valueplanner baseline.
```

Décision :

```text
Ne pas lancer seeds 2/3 sur calibrated_head.
La prochaine expérience doit calibrer l'incertitude après entraînement, sur un
checkpoint Q/WM déjà bon, en gelant toute la dynamique et la policy. Cela teste
si une incertitude fiable peut améliorer le planner sans perturber
l'apprentissage comportemental.
```

Outil ajouté :

```text
scripts/calibrate_micro_fouloide_uncertainty.py
```

Il charge un checkpoint existant, réutilise son replay buffer et entraîne
uniquement `world_model.uncertainty_head`. Le `world_model_state` calibré est
sauvegardé dans un nouveau checkpoint ; Q-network, value model, mémoire et
dynamique WM restent inchangés.

Résultat posthoc sur `valueplanner_seed1` :

```text
posthoc uncertainty calibration:
  updates=2000
  loss=0.017324

diagnose-wm-calibration:
  state_mse corr:       +0.220
  reward_abs_err corr:  -0.013
  feature_mae corr:     +0.324
  event_miss corr:      +0.248
  composite_rank corr:  +0.305
  top20 capture:         32.6%

Q-only inchangé:
  lifespan=114.7

meilleurs planners:
  p=0.15, terminal_value=1.0, uncertainty=0.24, margin=0.01
  lifespan=117.0, planner/Q=1.02x, planner_used=59.6%, max=290

  p=0.15, terminal_value=1.0, uncertainty=0.30/0.36, margin=0.0
  lifespan=117.0, planner/Q=1.02x, planner_used≈100%, max=290

  p=0.15, terminal_value=2.0, uncertainty=0.30/0.36, margin=0.01
  lifespan=116.8, planner/Q=1.02x, planner_used≈81.8%, max=214
```

Lecture :

```text
C'est le meilleur compromis observé jusqu'ici sur micro-fouloïde rough :
la calibration d'incertitude devient utile, Q-only n'est pas dégradé, et le
planner repasse au-dessus de Q-only sur le checkpoint seed1.
```

Décision :

```text
Valider posthoc sur seeds 2 et 3 avant toute nouvelle architecture.
Le réglage principal à tester en comparaison multi-seed est :
p=0.15, terminal_value=1.0, uncertainty_threshold=0.24, margin=0.01
car il combine gain lifespan, usage planner modéré et meilleur max observé.
```

Validation posthoc seeds 1-3 avec réglage fixe :

```text
p=0.15
terminal_value=1.0
uncertainty_threshold=0.24
margin=0.01
horizon=3
samples=8
```

Résultats :

```text
seed | Q-only | Q+planner | delta | planner/Q | planner_used
1    | 114.7  | 117.0     | +2.3  | 1.02x     | 59.6%
2    | 102.8  | 104.0     | +1.2  | 1.01x     | 53.9%
3    | 107.4  | 106.3     | -1.1  | 0.99x     | 67.5%

mean Q-only:    108.3
mean Q+planner: 109.1
mean delta:     +0.8
mean ratio:     ~1.01x
```

Lecture :

```text
La calibration posthoc préserve le Q-only et donne un signal planner positif en
moyenne, mais encore trop faible pour parler de preuve robuste. Seed3 reste
négatif au réglage fixe. Le résultat est donc un signal encourageant, pas une
validation finale.
```

Décision :

```text
Ne pas changer encore l'architecture.
Augmenter le budget d'évaluation sur le même réglage fixe pour réduire le bruit
avant de conclure. Si le delta moyen reste positif sur 300-500 épisodes, on
peut ensuite tester un réglage adaptatif ou calibré par quantile d'incertitude.
```

Validation 300 épisodes, même réglage fixe :

```text
p=0.15
terminal_value=1.0
uncertainty_threshold=0.24
margin=0.01
horizon=3
samples=8
```

Résultats :

```text
seed | Q-only | Q+planner | delta | planner/Q | planner_used
1    | 116.6  | 117.8     | +1.2  | 1.01x     | 58.2%
2    | 102.9  | 104.4     | +1.5  | 1.01x     | 51.8%
3    | 105.5  | 105.6     | +0.1  | 1.00x     | 65.3%

mean Q-only:    108.3
mean Q+planner: 109.3
mean delta:     +0.9
mean ratio:     ~1.01x
```

Lecture :

```text
Le signal planner posthoc devient positif sur les 3 seeds quand le bruit
d'évaluation est réduit. C'est une preuve minimale que le World Model calibré
peut améliorer Q-only à checkpoint identique. L'effet reste faible : il ne
suffit pas encore comme preuve forte d'un planner causal robuste.
```

Effets secondaires observés :

```text
food/water montent avec le planner sur les 3 seeds.
drive moyen monte légèrement sur les 3 seeds.
damage monte aussi légèrement, donc le planner semble pousser vers plus
d'activité/ressources plutôt que vers une stratégie globalement plus sûre.
```

Décision :

```text
Acter la calibration posthoc comme meilleur chemin actuel.
Prochaine étape : comparer le même réglage avec et sans checkpoint calibré pour
isoler l'effet propre de l'incertitude calibrée, puis tester un gate adaptatif
par quantile au lieu d'un seuil absolu fixe.
```

Implémentation du gate adaptatif :

```text
scripts/evaluate_micro_fouloide.py
  --planner-uncertainty-quantile <q>
  --planner-uncertainty-quantile-sweep <q1,q2,...>
```

Le seuil absolu est résolu depuis la distribution d'incertitude du replay du
checkpoint :

```text
threshold = quantile(WM_uncertainty(replay_latent, replay_action), q)
```

Cela évite de choisir manuellement `0.24` quand l'échelle d'incertitude change
entre checkpoints ou mondes. Le cœur agent reste inchangé : l'évaluateur
convertit seulement le quantile en seuil numérique avant d'appeler le planner.

Résultat quantile seed1, 300 épisodes :

```text
q0.60 -> threshold=0.23632
Q-only: 116.6
Q+planner q0.60 margin=0.01: 117.5
planner_used=55.3%
```

Sweep quantile seed1 :

```text
q0.50 margin=0.02: 117.3
q0.60 margin=0.01: 117.5
q0.70 margin=0.02: 117.7
q0.80 margin=0.02: 117.7
```

Lecture :

```text
Le quantile reproduit le réglage seuil fixe sans dépendre de l'échelle absolue
d'incertitude. Sur seed1, q0.70/q0.80 avec margin=0.02 donne un léger mieux,
mais l'écart reste petit. La validation doit donc se faire sur seeds 2/3 avant
de changer le réglage principal.
```

Validation quantile seeds 1-3, 300 épisodes :

```text
Réglage conservateur:
  quantile=0.60
  margin=0.01

seed | threshold | Q-only | Q+planner | delta | planner_used
1    | 0.23632   | 116.6  | 117.5     | +0.9  | 55.3%
2    | 0.24326   | 102.9  | 104.7     | +1.8  | 55.6%
3    | 0.23433   | 105.5  | 106.0     | +0.5  | 60.5%

mean Q-only:    108.3
mean Q+planner: 109.4
mean delta:     +1.1
```

```text
Candidat seed1:
  quantile=0.70
  margin=0.02

seed | threshold | Q-only | Q+planner | delta | planner_used
1    | 0.248     | 116.6  | 117.7     | +1.1  | 44.6%
2    | 0.25976   | 102.9  | 104.1     | +1.2  | 56.0%
3    | 0.24455   | 105.5  | 105.3     | -0.2  | 47.9%

mean delta: +0.7
```

Lecture :

```text
q0.60/margin=0.01 généralise mieux que le candidat q0.70/margin=0.02.
Le gate adaptatif conserve le signal positif sur les 3 seeds et évite un seuil
absolu réglé à la main. Le gain reste modeste, mais c'est la version la plus
propre du protocole WM-calibré à ce stade.
```

Décision :

```text
Figer q0.60/margin=0.01 comme baseline planner calibré micro-fouloïde rough.
La prochaine amélioration ne doit plus chercher un meilleur seuil ; elle doit
améliorer la qualité des rollouts/valeurs du WM, car le gate fonctionne.
```

Prochaine expérience ajoutée dans l'évaluateur :

```text
Sweep rollout/sampling du planner, à protocole constant :
  planning_weight=0.15
  terminal_value_weight=1.0
  uncertainty_quantile=0.60
  margin=0.01

Nouveaux flags :
  --planner-horizon-sweep 2,3,4,5
  --planner-samples-sweep 4,8,16

But : mesurer si un planner plus profond ou plus échantillonné transforme le
signal positif faible en gain net, sans retuner le gate d'incertitude.
```

Résultat seed1, 100 épisodes :

```text
Q-only: 114.7

horizon | samples | Q+planner | delta | planner_used
2       | 4       | 113.1     | -1.6  | 41.7%
2       | 8       | 111.7     | -3.0  | 45.7%
2       | 16      | 113.5     | -1.2  | 47.8%
3       | 4       | 112.0     | -2.7  | 50.9%
3       | 8       | 116.1     | +1.4  | 55.8%
3       | 16      | 114.8     | +0.1  | 57.5%
4       | 4       | 115.8     | +1.1  | 59.9%
4       | 8       | 112.5     | -2.2  | 63.4%
4       | 16      | 116.2     | +1.5  | 63.0%
5       | 4       | 113.5     | -1.2  | 64.6%
5       | 8       | 117.7     | +3.0  | 68.0%
5       | 16      | 111.5     | -3.2  | 66.6%
```

Lecture :

```text
Le meilleur candidat court est horizon=5/samples=8. Le gain n'est pas monotone :
samples=16 peut dégrader fortement, donc la piste n'est pas "plus gros partout"
mais une profondeur un peu plus longue avec échantillonnage modéré.

À valider en 300 épisodes sur les 3 seeds avant de l'adopter.
```

Validation `horizon=5/samples=8`, 300 épisodes :

```text
planning_weight=0.15
terminal_value_weight=1.0
uncertainty_quantile=0.60
margin=0.01
horizon=5
samples=8

seed | threshold | Q-only | Q+planner | delta | planner_used
1    | 0.23632   | 116.6  | 117.6     | +1.0  | 67.4%
2    | 0.24326   | 102.9  | 104.4     | +1.5  | 67.8%
3    | 0.23433   | 105.5  | 107.0     | +1.5  | 71.9%

mean Q-only:    108.3
mean Q+planner: 109.7
mean delta:     +1.3
```

Lecture :

```text
horizon=5/samples=8 confirme le signal positif sur les 3 seeds. Le gain moyen
est légèrement supérieur au baseline horizon=3/samples=8 (+1.3 vs +1.1), mais
le coût planner et le taux d'utilisation montent nettement (~69%).

Le résultat valide que regarder un peu plus loin aide, mais ne transforme pas
encore le planner en avantage fort. Le prochain levier doit donc améliorer la
qualité de la valeur imaginée ou des rollouts, pas simplement augmenter le
budget de recherche.
```

Outil suivant ajouté :

```text
scripts/calibrate_micro_fouloide_value.py
```

But :

```text
Calibrer posthoc uniquement le ValueModel sur les retours discountés observés
dans le replay, sans toucher Q-network, World Model, policy ni replay.

Cela teste si le faible gain planner vient d'une valeur terminale trop bruitée
ou biaisée. C'est le parallèle propre de la calibration posthoc d'incertitude.
```

Protocole à tester :

```text
1. Partir du checkpoint déjà calibré en incertitude.
2. Produire checkpoint_value_calibrated.pt.
3. Diagnostiquer ValueModel.
4. Réévaluer planner avec q0.60/margin=0.01/horizon=5/samples=8.
```

Résultat seed1 :

```text
Posthoc value calibration:
  samples=100000
  updates=5000
  loss=0.043099

ValueModel avant:
  mae=0.2703
  bias=-0.0212
  corr=0.438

ValueModel après:
  mae=0.2107
  bias=-0.0082
  corr=0.675
```

Diagnostic replay 20k après calibration :

```text
all:           mae=0.2117 bias=-0.0074 corr=0.674
low_energy:    mae=0.2378 bias=+0.1390 corr=0.590
low_hydration: mae=0.2387 bias=+0.1179 corr=0.594
low_health:    mae=0.2757 bias=+0.1731 corr=0.471
terminal:      mae=0.3786 bias=+0.3624 corr=-0.033
```

Évaluation seed1, 300 épisodes, q0.60/margin=0.01/horizon=5/samples=8 :

```text
Q-only:                         116.6
Q+planner incertitude seule:    117.6
Q+planner incertitude + valeur: 118.8
delta vs Q-only:                +2.2
planner_used:                   71.6%
```

Lecture :

```text
La calibration posthoc du ValueModel améliore nettement la corrélation aux
retours observés et augmente le gain planner sur seed1. Le biais terminal reste
mauvais : les états de mort sont encore sous-pénalisés, ce qui peut expliquer
la hausse de damage malgré un meilleur lifespan.

À valider sur seeds 2/3 avant adoption.
```

Validation seeds 1-3, posthoc incertitude + valeur, 300 épisodes :

```text
seed | value corr before -> after | value mae before -> after | Q-only | Q+planner | delta | planner_used
1    | 0.438 -> 0.675           | 0.270 -> 0.211            | 116.6  | 118.8     | +2.2  | 71.6%
2    | 0.504 -> 0.733           | 0.232 -> 0.174            | 102.9  | 104.6     | +1.7  | 70.1%
3    | 0.506 -> 0.718           | 0.246 -> 0.187            | 105.5  | 106.3     | +0.8  | 74.8%

mean Q-only:    108.3
mean Q+planner: 109.9
mean delta:     +1.6
```

Comparaison à `horizon=5/samples=8` sans calibration valeur :

```text
seed | incertitude seule | incertitude + valeur | diff
1    | 117.6             | 118.8                | +1.2
2    | 104.4             | 104.6                | +0.2
3    | 107.0             | 106.3                | -0.7

mean diff: +0.2
```

Lecture :

```text
La calibration valeur améliore fortement le diagnostic sur les 3 seeds et
augmente légèrement le gain moyen planner (+1.6 vs +1.3). En revanche, l'effet
lifespan n'est pas homogène : seed3 régresse par rapport à l'incertitude seule.

Conclusion : utile comme direction technique, mais pas encore assez décisif pour
remplacer automatiquement le baseline h5/s8 incertitude seule. Le problème
restant semble être la valeur des états terminaux/dangereux, pas la corrélation
globale aux retours.
```

Extension ajoutée au calibrateur valeur :

```text
scripts/calibrate_micro_fouloide_value.py
  --terminal-weight <w>
  --low-feature-weight feature:threshold:weight
```

But :

```text
Pondérer posthoc les états rares/terminaux sans changer la policy ni le WM.
Premier test raisonnable :
  terminal_weight=3.0
  low-feature health:0.5:2.0

Hypothèse : améliorer la pénalisation des états terminal/low-health, où le
ValueModel reste biaisé, peut réduire les dégâts du planner tout en gardant le
gain food/water.
```

Résultat seed1 avec `terminal_weight=3.0` et `health:0.5:2.0` :

```text
Weights:
  mean=1.413
  max=6.000

ValueModel avant:
  mae=0.2703 bias=-0.0212 corr=0.438

ValueModel après:
  mae=0.2281 bias=-0.0682 corr=0.634

Buckets dangereux:
  low_health mae: 0.2757 -> 0.2269
  low_health bias: +0.1731 -> +0.0915
  terminal mae: 0.3786 -> 0.2874
  terminal bias: +0.3624 -> +0.2638
```

Évaluation seed1, 300 épisodes :

```text
Q-only:                                116.6
Q+planner incertitude seule h5/s8:     117.6
Q+planner valeur simple:               118.8
Q+planner valeur terminal/health:      117.0
planner_used:                          73.2%
```

Lecture :

```text
La pondération terminal/low-health améliore les buckets dangereux, mais rend la
valeur trop pessimiste globalement (`bias=-0.0682`) et dégrade le planner par
rapport à la calibration valeur simple. C'est une ablation utile : la direction
est bonne localement, mais le poids testé est trop fort.

Ne pas valider ce réglage. Si on continue cette piste, tester plus doux :
terminal_weight=1.0 et health:0.5:1.0, ou corriger le sampling plutôt que la
loss globale.
```

Résultat seed1 avec pondération douce `terminal_weight=1.0` et `health:0.5:1.0` :

```text
Weights:
  mean=1.202
  max=3.000

ValueModel après:
  mae=0.2193
  bias=-0.0435
  corr=0.654

Évaluation 300 épisodes:
  Q-only:                           116.6
  Q+planner valeur simple:          118.8
  Q+planner terminal/health soft:   116.9
  planner_used:                     72.8%
```

Lecture :

```text
Même une pondération douce reste moins bonne que la calibration valeur simple.
Elle rend la valeur plus pessimiste et ne réduit pas suffisamment les dégâts
pour compenser. Arrêter cette branche posthoc pondérée pour l'instant.

Baseline à conserver :
  checkpoint_uncertainty_value_calibrated.pt
  q0.60 / margin=0.01 / horizon=5 / samples=8
```

Nouvelle piste ajoutée : calibration posthoc du World Model sur transitions rares.

Motivation :

```text
La calibration valeur simple donne un petit gain planner, mais le modèle reste
fragile sur les conséquences rares : dégâts, perte de santé, mort.

Pondérer seulement le ValueModel ne suffit pas : cela rend la valeur trop
pessimiste sans améliorer assez l'imagination du World Model. La prochaine
hypothèse testable est donc de réentraîner le World Model lui-même sur ces
transitions rares, tout en gardant le planner générique.
```

Changements techniques :

```text
- train_world_model accepte maintenant des poids génériques par transition :
  event_sample_names
  event_sample_name_weight
  event_sample_done_weight
  event_sample_reward_abs_weight

- Ces poids affectent la prédiction next_state/reward/features/events.
  Ce point est important : le planner n'utilise pas directement event_logits,
  donc pondérer uniquement la tête d'événement ne suffirait pas.

- scripts/calibrate_micro_fouloide_wm_events.py permet de tester cette
  calibration sur un checkpoint existant sans relancer 3000 épisodes.
```

Premier test recommandé :

```bash
python scripts/calibrate_micro_fouloide_wm_events.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_calibrated.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --device mps \
  --updates 3000 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --event-loss-weight 0.5 \
  --event-sample-name damage \
  --event-sample-name health_loss \
  --event-sample-name death \
  --event-sample-name-weight 3.0 \
  --event-sample-done-weight 3.0 \
  --reward-done-weight 2.0 \
  --out runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_wm_events.pt
```

Critère :

```text
Comparer au meilleur seed1 actuel :
  Q+planner valeur simple h5/s8 = 118.8

On valide seulement si le planner gagne ou si les dégâts/health_loss baissent
sans perte de lifespan. Sinon, il faudra corriger la calibration WM plus
finement plutôt que forcer plus fort les poids.
```

Résultat seed1 :

```text
Posthoc WM event calibration:
  updates=3000
  total=1.179395
  reward=0.283546
  feature=0.042365
  event=1.618385
  events pondérés: damage, health_loss, death

Évaluation 300 épisodes:
  Q-only:                         116.6
  Q+planner valeur simple:        118.8
  Q+planner WM rare-events:       119.2
  planner_used:                   80.6%
  food/water:                     0.82 / 0.88
  damage / health_loss:           1.16 / 31.11
```

Lecture :

```text
Signal positif mais encore faible. Le gain seed1 est supérieur à la meilleure
calibration valeur simple (+0.4 lifespan) et le planner est plus utilisé
(80.6%). Les dégâts ne baissent pas clairement, donc l'amélioration semble
venir surtout d'une meilleure collecte food/water et d'une sélection d'actions
un peu plus robuste.

Décision : valider sur seeds 2 et 3 avant de modifier les configs longues.
```

Validation seeds 2 et 3 avec le même protocole :

```text
seed | Q-only | value simple h5/s8 | WM rare-events h5/s8 | delta WM/Q
1    | 116.6  | 118.8             | 119.2                | +2.6
2    | 102.9  | 104.6             | 103.4                | +0.5
3    | 105.5  | 106.3             | 106.3                | +0.8

moy. | 108.3  | 109.9             | 109.6                | +1.3
```

Lecture :

```text
La calibration WM rare-events est positive vs Q-only sur les 3 seeds, mais elle
ne bat pas la calibration valeur simple en moyenne. Elle augmente fortement
planner_used (seed2 78.7%, seed3 86.5%), mais cela ne se transforme pas encore
en gain robuste de lifespan.

Décision : ne pas remplacer la baseline par WM rare-events. Conserver le script
comme outil expérimental. La prochaine piste utile est de rendre le planner plus
sélectif ou d'améliorer la cible d'incertitude/valeur sur les conséquences
imaginées, pas d'augmenter brutalement les poids rare-events.
```

Sélectivité Q-vs-WM ajoutée :

```text
Nouveau seuil générique :
  planning.q_advantage_threshold
  --planner-q-advantage-threshold
  --planner-q-advantage-sweep

Principe :
  le WM ne peut influencer Q que si son action préférée a un avantage normalisé
  suffisant par rapport à l'action préférée par Q.

Ce n'est pas une règle du monde : c'est une règle de confiance entre policy
apprise et imagination. Elle doit réduire planner_used et éviter les overrides
faiblement justifiés.
```

Premier sweep recommandé :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_calibrated.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 300 \
  --device mps \
  --planner-sweep 0.15 \
  --terminal-value-sweep 1.0 \
  --planner-uncertainty-quantile-sweep 0.60 \
  --planner-margin-sweep 0.01 \
  --planner-q-advantage-sweep 0,0.02,0.05,0.10,0.15 \
  --planner-horizon 5 \
  --planner-samples 8
```

Critère :

```text
On cherche un seuil qui conserve ou améliore le lifespan tout en réduisant
planner_used. Si planner_used baisse mais lifespan baisse aussi, le seuil est
trop strict. Si planner_used reste haut et lifespan ne bouge pas, le seuil ne
résout pas le problème.
```

Résultat seed1 :

```text
q_adv | lifespan | planner_used | food | water | damage | max
0.00  | 118.8    | 71.6%        | 0.85 | 0.89  | 1.44   | 290
0.02  | 118.8    | 61.2%        | 0.85 | 0.89  | 1.44   | 290
0.05  | 118.3    | 60.3%        | 0.81 | 0.88  | 1.39   | 253
0.10  | 118.3    | 57.6%        | 0.81 | 0.88  | 1.39   | 253
0.15  | 117.6    | 54.2%        | 0.84 | 0.85  | 1.36   | 231
```

Lecture :

```text
q_adv=0.02 est meilleur comme règle de confiance : même lifespan que la
baseline planner simple (118.8), mais planner_used réduit de 71.6% à 61.2%.
Les seuils plus hauts réduisent encore l'usage du planner mais commencent à
perdre du lifespan.

Décision provisoire : tester q_adv=0.02 sur seeds 2 et 3.
```

Validation 3 seeds avec `q_adv=0.02` :

```text
seed | Q-only | planner q_adv=0.00 | used  | planner q_adv=0.02 | used
1    | 116.6  | 118.8              | 71.6% | 118.8              | 61.2%
2    | 102.9  | 104.6              | 70.1% | 104.6              | 61.2%
3    | 105.5  | 106.3              | 74.8% | 106.3              | 61.1%

moy. | 108.3  | 109.9              | 72.2% | 109.9              | 61.2%
```

Lecture :

```text
q_adv=0.02 conserve le gain planner moyen (+1.6 lifespan vs Q-only) tout en
réduisant l'usage du World Model d'environ 11 points absolus. C'est une
amélioration de sélectivité, pas encore une amélioration de performance brute.

Décision : adopter q_adv=0.02 comme nouveau réglage d'évaluation par défaut
pour la baseline posthoc uncertainty+value calibrée :
  planning_weight=0.15
  terminal_value_weight=1.0
  uncertainty_quantile=0.60
  margin=0.01
  q_advantage=0.02
  horizon=5
  samples=8
```

Sweep planning_weight avec `q_adv=0.02`, seed1 :

```text
p_weight | lifespan | planner/Q | food | water | damage | used
0.10     | 116.0    | 0.99      | 0.76 | 0.82  | 1.28   | 61.9%
0.15     | 118.8    | 1.02      | 0.85 | 0.89  | 1.44   | 61.2%
0.20     | 119.2    | 1.02      | 0.86 | 0.90  | 1.29   | 60.9%
0.25     | 121.7    | 1.04      | 0.98 | 0.99  | 1.46   | 60.3%
0.30     | 118.0    | 1.01      | 0.92 | 0.92  | 1.63   | 60.0%
```

Lecture :

```text
Avec le filtre q_adv=0.02, augmenter le poids du WM à 0.25 devient utile sur
seed1. Le gain semble venir d'une meilleure collecte food/water, sans hausse
de planner_used. 0.30 est déjà trop fort.

Décision provisoire : valider p_weight=0.25 sur seeds 2 et 3.
```

Validation 3 seeds de `p_weight=0.25`, `q_adv=0.02` :

```text
seed | Q-only | p=0.15 q_adv=0.02 | p=0.25 q_adv=0.02 | delta p0.25/Q
1    | 116.6  | 118.8             | 121.7             | +5.1
2    | 102.9  | 104.6             | 104.5             | +1.6
3    | 105.5  | 106.3             | 107.5             | +2.0

moy. | 108.3  | 109.9             | 111.2             | +2.9
```

Lecture :

```text
p=0.25 améliore la moyenne 3 seeds par rapport à p=0.15 (+1.3 lifespan) et
augmente le gain planner/Q moyen (+2.9). Le gain n'est pas uniforme : seed2 est
quasi neutre, seed1 très positif, seed3 positif.

Décision : p=0.25 devient le meilleur réglage posthoc actuel, avec prudence.
La prochaine validation doit être un run 500 ou 1000 épisodes pour réduire le
bruit d'évaluation avant de l'inscrire dans une config de référence.
```

Validation longue 1000 épisodes avec `p=0.25`, `q_adv=0.02` :

```text
seed | Q-only | Q+WM planner | delta | ratio | planner_used
1    | 116.6  | 119.6        | +3.0  | 1.03x | 59.5%
2    | 101.5  | 102.6        | +1.1  | 1.01x | 59.2%
3    | 105.5  | 108.0        | +2.5  | 1.02x | 61.7%

moy. | 107.9  | 110.1        | +2.2  | 1.02x | 60.1%
```

Lecture :

```text
Le signal survit à l'évaluation longue : les 3 seeds sont positifs vs Q-only.
Le gain moyen est modeste (+2.2 lifespan), mais robuste, et le planner reste
sélectif (~60% d'usage au lieu de ~72% avec q_adv=0).

Effets comportementaux :
  seed1 : food/water augmente fortement (0.68/0.82 -> 0.91/0.95)
  seed2 : food/water augmente légèrement (0.22/0.26 -> 0.28/0.34)
  seed3 : food/water augmente (0.49/0.57 -> 0.59/0.70)

Décision : meilleur réglage actuel validé sur 1000 épisodes :
  planning_weight=0.25
  terminal_value_weight=1.0
  uncertainty_quantile=0.60
  margin=0.01
  q_advantage=0.02
  horizon=5
  samples=8
```

Preset CLI ajouté :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_calibrated.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Ce preset applique :

```text
planning_weight=0.25
terminal_value_weight=1.0
uncertainty_quantile=0.60
margin=0.01
q_advantage=0.02
horizon=5
samples=8
```

Rapport multi-seeds ajouté :

```bash
python scripts/report_micro_fouloide_planner.py \
  --num-episodes 1000 \
  --device mps \
  --output reports/micro_fouloide_wm_calibrated_1000.md
```

Par défaut, le rapport utilise :

```text
config=configs/micro_fouloide_v0_rough_valueplanner.yaml
checkpoint_template=runs/micro_fouloide_v0_rough_valueplanner_seed{seed}/checkpoint_uncertainty_value_calibrated.pt
seeds=1,2,3
planner_preset=wm-calibrated
```

Rapport généré :

```text
reports/micro_fouloide_wm_calibrated_1000.md

seed | Q-only | Q+WM | delta | ratio | used | food | water | damage | max
1    | 116.6  | 119.6 | +3.0  | 1.03  | 59.5% | 0.91 | 0.95  | 1.47   | 286
2    | 101.5  | 102.6 | +1.0  | 1.01  | 59.2% | 0.28 | 0.34  | 0.95   | 205
3    | 105.5  | 108.0 | +2.5  | 1.02  | 61.7% | 0.59 | 0.70  | 1.56   | 234
mean | 107.9  | 110.0 | +2.2  | 1.02  | 60.1% | 0.59 | 0.66  | 1.33   | 242
```

Chaîne de calibration posthoc automatisée :

```bash
python scripts/calibrate_micro_fouloide_posthoc_chain.py \
  --seeds 1,2,3 \
  --device mps
```

Par défaut, la chaîne produit pour chaque run :

```text
checkpoint_final.pt
  -> checkpoint_uncertainty_calibrated.pt
  -> checkpoint_uncertainty_value_calibrated.pt
```

Paramètres standards :

```text
uncertainty calibration:
  updates=2000
  batch_size=64
  learning_rate=0.0003

value calibration:
  updates=5000
  batch_size=64
  learning_rate=0.0003
```

Le script saute les sorties déjà existantes, sauf avec `--force`.

Ablation checkpoint non calibré, même réglage fixe et 300 épisodes :

```text
p=0.15
terminal_value=1.0
uncertainty_threshold=0.24
margin=0.01
```

Résultats :

```text
seed | Q-only | Q+planner non calibré | delta
1    | 116.6  | 116.6                 | 0.0
2    | 102.9  | 102.9                 | 0.0
3    | 105.5  | 105.5                 | 0.0
```

Lecture :

```text
Au même seuil, l'incertitude non calibrée désactive de fait le planner : Q+WM
retombe exactement sur Q-only. Le gain observé avec le checkpoint posthoc vient
donc bien de la calibration d'incertitude, pas du simple réglage planner.
```

Conclusion expérimentale micro-fouloïde rough posthoc :

```text
Q-only > naive est robuste.
WM planner non calibré ≈ Q-only.
WM planner avec incertitude calibrée posthoc > Q-only en moyenne 3 seeds
sur 1000 épisodes, avec un gain faible mais cohérent.

Statut : preuve minimale positive du rôle du World Model calibré.
Non encore preuve forte : l'effet reste autour de +2.2 lifespan moyen.
```

Étape 4 — passage du posthoc vers l'apprentissage continu :

```text
Objectif : réduire la dépendance à la calibration posthoc en entraînant la tête
d'incertitude pendant l'apprentissage, à partir des expériences de l'agent.

Principe :
  - continuer à entraîner le World Model complet comme avant ;
  - ajouter ensuite quelques updates head-only sur uncertainty_head ;
  - geler implicitement le tronc pendant ces updates via
    train_world_model_uncertainty_head ;
  - ne pas injecter de règle du monde : la cible reste l'erreur de prédiction
    observée dans le replay.
```

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_online_uncertainty.yaml
```

Différence principale avec la baseline valueplanner :

```yaml
world_model:
  uncertainty_head_updates_per_train: 1
  uncertainty_head_learning_rate: 0.0003
  uncertainty_head_batch_size: 64
  uncertainty_head_sampler: causal
```

Premier protocole recommandé :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_online_uncertainty.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_online_uncertainty_seed1
```

Puis diagnostic :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_online_uncertainty_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_online_uncertainty.yaml \
  --device mps \
  --diagnose-wm-calibration \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Puis évaluation avec le preset validé :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_online_uncertainty_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_online_uncertainty.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Critères de décision :

```text
1. Q-only ne doit pas régresser fortement vs valueplanner baseline.
2. diagnose-wm-calibration doit montrer une uncertainty utile
   (composite_rank corr et top20 capture au moins proches du posthoc).
3. Le planner preset doit rester positif vs Q-only sans posthoc.
4. Si seed1 est neutre/négatif, ne pas lancer seeds 2/3 ; revenir à la cible
   d'incertitude ou au rythme d'updates.
```

Résultat seed1 online uncertainty, 1000 épisodes :

```text
Resolved uncertainty q0.60 threshold: 0.25645

Q-only:
  lifespan=102.0
  food=0.12
  water=0.39

Q+WM planner preset:
  lifespan=102.1
  planner_used=49.2%
  food=0.21
  water=0.46
```

Comparaison au meilleur seed1 posthoc :

```text
posthoc uncertainty+value:
  Q-only=116.6
  Q+WM=119.6

online uncertainty:
  Q-only=102.0
  Q+WM=102.1
```

Lecture :

```text
La calibration d'incertitude online à chaque train perturbe l'apprentissage
comportemental. Le problème n'est pas le planner seul : la policy Q finale est
nettement plus faible. Ne pas lancer seeds 2/3 sur cette config.
```

Décision :

```text
Passer à une consolidation de fin de run :
  - apprendre normalement comme valueplanner ;
  - sauvegarder checkpoint_final.pt brut ;
  - sauvegarder checkpoint_best.pt sur la meilleure moyenne glissante ;
  - calibrer uncertainty_head puis ValueModel sur le replay final ;
  - sauvegarder checkpoint_final_calibrated.pt ;
  - calibrer aussi checkpoint_best.pt et sauvegarder checkpoint_best_calibrated.pt.

Cette étape garde l'objectif "apprendre de ses expériences", mais évite de
modifier le World Model pendant que la policy se forme.
```

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml
```

Elle active :

```yaml
training:
  best_checkpoint_enabled: true
  best_checkpoint_window: 100
  best_checkpoint_min_episode: 500
  best_checkpoint_min_delta: 0.1

final_calibration:
  enabled: true
  output_name: checkpoint_final_calibrated.pt
  uncertainty:
    updates: 2000
    learning_rate: 0.0003
  value:
    updates: 5000
    learning_rate: 0.0003
```

Principe opérationnel :

```text
checkpoint_final.pt              = dernier état brut
checkpoint_final_calibrated.pt   = dernier état consolidé
checkpoint_best.pt               = meilleur lifespan moyen glissant brut
checkpoint_best_calibrated.pt    = meilleur état consolidé
```

Pour les validations longues, évaluer d'abord `checkpoint_best_calibrated.pt`.
Si le best et le final divergent fortement, cela indique une instabilité de
formation de la policy ; dans ce cas le run est exploitable mais pas encore
suffisamment stable pour être qualifié de version opérationnelle.

Résultat seed1 avec best-checkpoint train :

```text
checkpoint_best_calibrated.pt:
  Q-only=99.7
  Q+WM=99.7
  planner_used=56.7%

checkpoint_final_calibrated.pt précédent:
  Q-only=104.5
  Q+WM=106.1
```

Lecture :

```text
Le meilleur lifespan train n'est pas un bon proxy de sélection eval. Ici le
checkpoint best-train est moins bon que le final. Il ne faut donc pas promouvoir
un agent sur métrique d'entraînement seule.
```

Outil ajouté :

```text
scripts/select_micro_fouloide_checkpoint.py
```

Principe :

```text
Évaluer plusieurs checkpoints candidats avec le même protocole Q-only/Q+WM,
puis classer par lifespan planner et delta planner/Q. C'est plus coûteux que le
proxy train, mais c'est le mécanisme correct pour choisir un artefact
opérationnel.
```

Commande de sélection locale :

```bash
python scripts/select_micro_fouloide_checkpoint.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_best_seed1/checkpoint_final_calibrated.pt \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_best_seed1/checkpoint_best_calibrated.pt \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_calibrated.pt \
  --num-episodes 300 \
  --device mps \
  --planner-preset wm-calibrated \
  --output reports/micro_fouloide_checkpoint_selection_seed1.md
```

Promotion d'un artefact stable :

```bash
python scripts/select_micro_fouloide_checkpoint.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_best_seed1/checkpoint_final_calibrated.pt \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_best_seed1/checkpoint_best_calibrated.pt \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_uncertainty_value_calibrated.pt \
  --num-episodes 300 \
  --device mps \
  --planner-preset wm-calibrated \
  --min-delta 1.0 \
  --promote-to runs/micro_fouloide_promoted/wm_calibrated_seed1.pt \
  --manifest runs/micro_fouloide_promoted/wm_calibrated_seed1.manifest.json
```

Le manifest garde le checkpoint source, le protocole de sélection et les
métriques Q-only/Q+WM. Cela donne un artefact consommable par la suite sans
perdre la justification expérimentale.

Décision :

```text
La version concrète minimale doit combiner :
  1. une chaîne qui produit des checkpoints calibrés ;
  2. une sélection par évaluation contrôlée ;
  3. un rapport/manifest de promotion.

Tant que relancer l'entraînement ne retrouve pas régulièrement le niveau des
checkpoints historiques, ceux-ci restent la baseline opérationnelle ; les runs
nouveaux servent à stabiliser la production automatique.
```

Promotion multi-seed de la baseline validée :

```bash
python scripts/promote_micro_fouloide_wm_calibrated.py --force
```

Artefacts produits :

```text
runs/micro_fouloide_promoted/wm_calibrated_v0/
  seed1.pt
  seed2.pt
  seed3.pt
  manifest.json
```

Le manifest référence le rapport 1000 épisodes :

```text
reports/micro_fouloide_wm_calibrated_1000.md

mean Q-only: 107.9
mean Q+WM:   110.0
mean delta:  +2.2
```

Statut :

```text
micro_fouloide_wm_calibrated_v0 est le premier artefact promu exploitable.
Il ne prouve pas encore une génération stable de nouveaux agents bons, mais il
donne une version de référence concrète pour démos, comparaisons et prochaines
itérations.
```

Démo / évaluation directe du manifest promu :

```bash
python scripts/demo_micro_fouloide_promoted.py \
  --manifest runs/micro_fouloide_promoted/wm_calibrated_v0/manifest.json \
  --seed 1 \
  --num-episodes 100 \
  --device mps \
  --find-rollout \
  --rollout-search-count 64 \
  --rollout-select median \
  --survival-objective-weight 0.5 \
  --rollout-max-steps 120
```

Cette commande :

```text
1. lit le manifest promu ;
2. charge le checkpoint seed choisi ;
3. résout le seuil d'incertitude du planner ;
4. réévalue Q-only vs Q+WM ;
5. cherche un rollout représentatif parmi plusieurs seeds ;
6. imprime un rollout court avec actions, événements, drives et usage planner.
```

`--rollout-select best` peut être utilisé pour une démo plus attractive ; le
mode `median` est plus honnête pour montrer un comportement typique.
La démo active aussi des garde-fous runtime `filter_blocked_moves` et
`filter_noop_interact` pour ne pas proposer au policy/planner les mouvements
impossibles contre les obstacles/bords ni les interactions sans ressource.
L'ancien comportement reste disponible avec `--allow-blocked-moves` et
`--allow-noop-interact`.

Commande :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_seed1
```

Évaluation du checkpoint calibré :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_late_calibrated_seed1/checkpoint_best_calibrated.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

---

## 6.14 Ligne d'arrivée : démo visuelle fouloïde

But concret : lancer une démo visuelle avec un fouloïde sur une grande carte,
observant ses drives et utilisant un modèle du monde pour essayer de survivre.
Cette démo ne doit pas attendre une intelligence parfaite ; elle doit montrer
un organisme minimal qui évite les actions absurdes, cherche des ressources et
reste vivant de façon lisible.

Critères de passage avant démo visuelle :

```text
Artefact:
  - un manifest promu existe (`micro_fouloide_wm_calibrated_v0` ou suivant)
  - les commandes de démo et d'évaluation sont documentées

Métriques:
  - Q+WM >= Q-only sur 3 seeds en moyenne
  - évaluation confirmée sur au moins 1000 épisodes
  - ratio Q+WM/Q-only >= 1.01

Comportement:
  - `move_blocked` absent ou marginal dans les rollouts avec guards
  - `interact_noop` absent ou marginal dans les rollouts avec guards
  - un rollout `best` sur 64 seeds atteint le cap de démo
  - le rollout `median` ne doit pas être dominé par une seule action passive
    (`REST` ou `WAIT`) jusqu'à la mort

Décision:
  - si les trois premiers blocs sont validés, on lance la démo visuelle v0
  - si le median reste passif, on ajoute d'abord un objectif survival_v0
    configurable au planner, sans encoder cet objectif dans le World Model
```

Commande go/no-go actuelle :

```bash
python scripts/demo_micro_fouloide_promoted.py \
  --manifest runs/micro_fouloide_promoted/wm_calibrated_v0/manifest.json \
  --seed 1 \
  --num-episodes 100 \
  --device mps \
  --find-rollout \
  --rollout-search-count 64 \
  --rollout-select median \
  --rollout-max-steps 120
```

Statut au 2026-06-11 :

```text
Métriques agrégées: validées pour une première RC.
Actions absurdes: `move_blocked` et `interact_noop` corrigées par guards runtime.
Blocage restant: le rollout median peut encore être dominé par REST/WAIT et
mourir malgré l'objectif de survie.

Prochaine étape avant démo visuelle:
  tester l'objectif `survival_v0` paramétrable dans le planner.
  Le World Model reste neutre ; l'objectif est une couche de décision.
```

#### Diagnostic et fix survival_v0 — 2026-06-11

Diagnostic : l'objectif `survival_v0` était **structurellement no-op** dans le
pipeline micro-fouloïde. Ce n'était pas un problème de calibration des poids.

```text
1. `causal_features_fn` jamais câblé dans `build_agent`
   (`scripts/run_micro_fouloide.py`) → `current_features=None` →
   `Planner._objective_value()` retournait 0.0 quel que soit le poids.
2. Même câblées, les features restaient gelées dans l'imagination : la
   propagation `features + delta` exigeait `causal_feature_weights` ET
   `causal_feature_targets`, jamais passés → l'objectif aurait ajouté la même
   constante à toutes les actions, sans effet sur le classement.
```

Fix (couche décision uniquement, WM inchangé, aucun réentraînement) :

```text
- planner.py : propagation des features découplée du shaping causal — les
  feature_particles évoluent via la tête causale du WM dès qu'un objectif est
  actif, indépendamment des poids/targets causaux.
- run_micro_fouloide.py : `causal_features_fn` câblé quand
  `causal_world_model.enabled` (bénéficie à demo/evaluate/select).
- demo : stats `passive_steps` + `planner_used_passive` (usage planner segmenté
  global vs phases REST/WAIT) et flag `--debug-objective` (valeurs WM
  avec/sans objectif par action).
- tests : sensibilité du planner à l'objectif via WM stub à tête causale
  (tests/test_planning.py::TestObjectiveSensitivity).
```

Sanity (median, 8 seeds, cap 120, seed checkpoint 1) :

```text
                       weight 0.0          weight 0.5
lifespan median        115 (mort)          112 (mort)
steps passifs          80 (dont 51 WAIT)   28
comportement           WAIT jusqu'à mort   67 move_ok, interact food+water
planner_used           48.7%               58.0%
planner_used_passive   45.0%               50.0%
```

Le seed 9999 passe de 115/mort (weight 0.0) à 120/vivant drive 0.599
(weight 0.5). `planner_used_passive` ~45-50% dans les deux cas : le gating
n'était pas le blocage principal, c'était le câblage.

Prochaine étape : relancer la commande go/no-go complète (64 seeds) pour
décider du passage au viewer visuel.

Ligne d'arrivée pratique :

```text
Après `survival_v0`, si le rollout median sur 64 seeds atteint >= 120 steps
ou meurt après >= 120 steps sur une carte de démo plus grande, on arrête la
phase micro-recherche et on construit le viewer visuel.
```

### 6.15 Verdict go/no-go survival_v0 runtime

Résultat après correction du câblage objectif :

```text
- L'objectif survival_v0 est maintenant bien pris en compte par le planner :
  les features causales évoluent pendant l'imagination et le debug objectif
  montre un signal différentiel au lieu d'un bonus commun à toutes les actions.
- Les guards runtime (`filter_blocked_moves`, `filter_noop_interact`) suppriment
  les boucles les plus triviales.
- Une pénalité configurable `objective.action_penalties` permet de décourager
  REST/WAIT côté objectif, sans modifier le World Model.
- Un mécanisme optionnel `objective.force_planner_below` permet de forcer
  l'arbitrage planner sous seuil critique, mais il n'est pas activé dans la
  démo promue actuelle : sur ce checkpoint, forcer le planner sous hydratation
  critique amplifie parfois la préférence apprise pour REST au lieu de corriger
  la recherche d'eau.
```

Smoke CPU sur le checkpoint promu seed 1 :

```text
Réglage non destructeur (delta objective + action_penalties REST/WAIT=0.35)
  Evaluation 100 épisodes : Q-only 98.9, Q+WM 103.0, delta +4.2
  Rollout médian 64 seeds : encore mort à 96 steps, mais passivité réduite
  selon les seeds ; le blocage principal devient la compétence de recherche
  de ressource, surtout l'eau.

Réglage force critique testé puis rejeté
  Q+WM monte à 105.9 sur l'évaluation courte, mais le rollout médian reste
  mort à 96 steps et peut redevenir très passif (`REST` sous hydratation
  critique). Ce n'est donc pas un bon réglage de démo.
```

Décision : la prochaine étape concrète n'est plus d'ajouter des rustines runtime
au checkpoint promu. Il faut entraîner/évaluer une variante où l'objectif de
survie apprend explicitement la recherche de ressources avant la démo visuelle :

```text
Go demo visuelle si :
  rollout médian 64 seeds >= 120 steps vivant/cappé
  et pas de boucle passive dominante en phase critique
  et au moins une interaction utile ressource sur les rollouts représentatifs.

No-go actuel :
  le checkpoint promu sait améliorer Q-only en moyenne, mais ne sait pas encore
  garantir une stratégie robuste de recherche d'eau sur rollout médian.
```

### 6.16 Étape suivante : resource_seek

Ajout d'une config dédiée :

```text
configs/micro_fouloide_v0_rough_valueplanner_resource_seek.yaml
```

Objectif : ne pas modifier le World Model universel, mais densifier le signal
d'apprentissage de la politique/ValueModel pour apprendre explicitement :

```text
- boire quand hydratation basse/critique ;
- manger quand énergie basse/critique ;
- éviter les pas passifs quand une ressource vitale est basse ;
- mieux sampler les événements ressource/terminal dans le WM causal.
```

Commande seed 1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_seek.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_resource_seek_seed1
```

Puis évaluation go/no-go :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_seek_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_seek.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Critère pour passer à la démo visuelle :

```text
- Q-only ne régresse pas sous le baseline seed 1 de manière majeure ;
- Q+WM garde un delta positif ;
- `interact_water` augmente nettement vs baseline ;
- rollout médian 64 seeds atteint le cap 120 ou ne montre plus de boucle
  passive dominante sous hydratation critique.
```

Résultat seed 1 après 3000 épisodes :

```text
Train final last100 = 110.2
Eval 1000 :
  Q-only = 109.6, water = 0.57, actions interact = 32.7%
  Q+WM   = 110.6, water = 0.62, planner_used = 14.9%
```

Interprétation :

```text
- La récompense resource_seek fonctionne localement : l'agent interagit beaucoup
  plus avec les ressources, surtout l'eau.
- Le lifespan reste inférieur au meilleur checkpoint promu seed 1.
- Le rollout médian direct reste mort à ~96 steps, avec hydratation à zéro.
- Avec ou sans survival_v0 runtime, le comportement médian n'est pas encore
  robuste : l'agent peut traverser la carte mais ne garantit pas la recherche
  d'eau utile avant le seuil critique.
```

Décision : `resource_seek_seed1` est un no-go pour la démo visuelle. Il valide
la direction “récompense ressource”, mais il faut une variante suivante plus
forte ou différente :

```text
Option A : curriculum plus facile puis rough
  plus d'eau / moins d'obstacles / decay hydratation légèrement réduit,
  puis transfert ou même config rough.

Option B : objectif ressource explicite dans policy/value
  pénalité plus forte pour hydratation basse et bonus de signal local_water,
  pas seulement bonus d'interaction.

Option C : séparer exploration et exploitation
  entraîner un comportement de recherche ressource avec epsilon/curiosity plus
  long, puis évaluer avec planner.
```

#### Diagnostic discriminant farming vs navigation — 2026-06-11

Avant de choisir entre A/B/C, un diagnostic peu coûteux : distribution du
niveau de drive **au moment de l'interaction** (hydratation@`interact_water`,
énergie@`interact_food`), ajoutée à `evaluate_micro_fouloide.py`
(`hydration_at_water_*`, `energy_at_food_*` dans le résumé). Si l'agent boit
surtout à hydratation haute, la récompense est farmée → option B ; s'il boit
bas mais rarement, c'est un problème de navigation/couverture → option A ou C.

Résultats (300 épisodes Q-only, seed eval 9999) :

```text
                          resource_seek_seed1    baseline promue seed1
lifespan                  110.7                  116.6
water / épisode           0.58                   0.82
hydration@water median    0.200 (91% <= 0.35)    0.310 (54% <= 0.35)
energy@food median        0.542                  0.682
actions interact          31.7%                  15.1%
```

Verdict :

```text
1. Pas de farming d'eau : resource_seek boit exclusivement bas
   (100% <= 0.50, 91% <= 0.35). Le signal conditionnel fonctionne — la partie
   "timing" de l'option B est déjà acquise.
2. Le vrai blocage est la fréquence d'accès : resource_seek boit MOINS souvent
   que la baseline (0.58 vs 0.82/ép.) et meurt déshydraté. C'est un problème
   de navigation/couverture → l'option A (curriculum) est la bonne direction.
3. Pathologie révélée au passage : interact = 31.7% des actions mais ~0.95
   interaction utile par épisode → ~33 INTERACT no-op par épisode. L'agent
   spamme INTERACT à vide (éval sans guards), ce qui gaspille les pas
   d'exploration et explique probablement le lifespan inférieur à la baseline.
4. Farming léger sur la nourriture (median 0.54) mais l'énergie n'est pas la
   cause de mort.
```

Implications pour le curriculum (option A) :

```text
- préférer réduire hydration_decay plutôt qu'augmenter num_water (limiter le
  risque de farming facilité) ;
- traiter le spam INTERACT no-op pendant l'entraînement (pénalité event
  interact_noop ou guard), sinon il plombera aussi le curriculum ;
- critère de validation : rollout médian sur la config ROUGH, pas sur la
  config curriculum — valider le transfert, pas la facilité.
```

#### Curriculum resource navigation — 2026-06-11

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_resource_curriculum.yaml
```

Principes :

```text
- num_water reste à 6 : on évite de rendre le problème trivial par densité.
- hydration_decay passe de 0.010 à 0.008 : plus de temps pour découvrir l'eau.
- obstacles/dangers légèrement réduits (24→20, 11→9) : curriculum de navigation,
  pas changement de but.
- filter_blocked_moves/filter_noop_interact activés pendant l'entraînement :
  pas de gaspillage massif d'actions sur INTERACT à vide.
- resource_reward garde une pénalité interact_noop/move_blocked pour les runs
  diagnostiques sans guards.
```

Run seed 1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_curriculum.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_resource_curriculum_seed1
```

Évaluation locale curriculum :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_curriculum_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_curriculum.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Validation de transfert rough (la vraie décision pour la démo) :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_curriculum_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_seek.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Résultat curriculum seed 1 :

```text
Train final last100 = 115.1
Eval curriculum :
  Q-only = 112.3, water = 0.10, rest+wait = 76.3%
  Q+WM   = 112.7, water = 0.12
Eval rough/transfert :
  Q-only = 95.2, water = 0.17, interact = 53.1%
  Q+WM   = 95.3, water = 0.18
```

Verdict : no-go. Le curriculum a appris une stratégie plus passive et ne
transfère pas au rough. Prochaine variante : rough inchangé + guards actifs +
reward ressource/no-op, sans rendre le monde plus facile.

#### Resource seek guarded — 2026-06-11

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_resource_seek_guarded.yaml
```

Principes :

```text
- environnement rough inchangé : hydration_decay=0.010, num_water=6,
  num_dangers=11, num_obstacles=24 ;
- guards actifs pendant l'entraînement : pas de `INTERACT` à vide ni de
  mouvement bloqué proposé au policy ;
- reward ressource conservé, avec pénalité explicite `interact_noop` pour les
  diagnostics sans guards ;
- validation directe sur rough, pas de transfert depuis un monde plus facile.
```

Run seed 1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_seek_guarded.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_resource_seek_guarded_seed1
```

Évaluation :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_seek_guarded_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_seek_guarded.yaml \
  --num-episodes 1000 \
  --device mps \
  --compare-planner \
  --planner-preset wm-calibrated
```

Résultat seed 1, training 3000 épisodes :

```text
Final mean lifespan last100: 92.6
food100=0.14 water100=0.11 dmg100=1.80
```

Décision : **NO-GO**.

Interprétation :

```text
Le filtrage/pénalité corrige les actions invalides, mais retire trop de signal
d'exploration. L'agent boit/mange moins que `resource_seek` et le lifespan
retombe près du naïf rough. Le problème n'est donc pas seulement le spam
INTERACT/no-op ; il manque un gradient de navigation vers ressource.
```

### Étape D — Ressource + navigation locale

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml
```

Principe :

```text
- monde rough inchangé ;
- pas de guard pendant l'entraînement ;
- pénalité légère pour `interact_noop` / `move_blocked` ;
- bonus dense faible quand une ressource utile est visible localement et que
  l'agent fait une action active (`move_ok`) avec le drive correspondant bas.
```

Cette variante teste précisément l'hypothèse issue des diagnostics :

```text
L'agent sait consommer l'eau quand il la trouve, mais il n'a pas assez de signal
pour apprendre à se rapprocher d'une ressource visible avant la crise.
```

Run seed 1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed1
```

Critère go/no-go :

```text
GO si last100 >= 110 et eau >= 0.50/episode, puis eval 1000 + demo directe.
NO-GO si eau reste < 0.35/episode ou lifespan < resource_seek seed1.
```

Résultat seed 1 :

```text
Training 3000 épisodes :
- final mean lifespan last100 = 120.1
- food100 = 0.70
- water100 = 0.94
- damage100 = 2.21

Eval 1000 épisodes, rough/resource_navigation :
- naïf = 89.0
- Q-only = 117.1 ± 34.2, max 289
- Q+WM = 117.4 ± 33.8, max 272
- Q/naïf = 1.31×
- WM/Q = 1.00×
- Q+WM water = 0.82/episode
- hydration@interact_water median = 0.220, 93.5% <= 0.35
```

Demo directe alignée avec la config d'entraînement (`--allow-blocked-moves`,
`--allow-noop-interact`, `--disable-survival-objective`) :

```text
rollout median search 64 seeds:
- selected_seed = 10047
- lifespan = 120, dead = false, capped = true
- planner_used = 65.8%
- events: interact_water=1, move_ok=32, health_loss=32
```

Décision : **GO seed1 / candidat demo**, mais **pas encore promotion stable**.

Interprétation :

```text
Le signal de navigation locale corrige le point bloquant principal : l'agent
trouve et utilise l'eau plus souvent, et le rollout médian atteint le cap 120
sans mourir. Il reste deux limites : l'énergie finit basse dans certains
rollouts et le planner WM n'apporte quasiment pas de gain au-dessus de Q-only.
La prochaine validation doit être multi-seed avant promotion.
```

Prochaine étape :

```text
1. entraîner resource_navigation seeds 2 et 3 ;
2. évaluer les 3 seeds sur 1000 épisodes ;
3. si moyenne Q-only >= 115 et demos médianes non mortes à 120,
   promouvoir une version `resource_navigation_v0` pour la première démo visuelle.
```

Résultat training seeds 2 et 3 :

```text
Seed 2, training 3000 épisodes :
- final mean lifespan last100 = 113.6
- food100 = 0.47
- water100 = 0.80
- damage100 = 1.96

Seed 3, training 3000 épisodes :
- final mean lifespan last100 = 116.0
- food100 = 0.53
- water100 = 0.80
- damage100 = 1.92
```

Décision training multi-seed : **GO**.

```text
Les trois seeds passent le seuil `water100 >= 0.50` et terminent au-dessus de
113 de lifespan last100. La variante est reproductible côté apprentissage.
Il reste à faire la validation officielle 1000 épisodes sur les 3 seeds avant
promotion.
```

Validation MPS à lancer :

```bash
python scripts/report_micro_fouloide_planner.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml \
  --checkpoint-template 'runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed{seed}/checkpoint_final.pt' \
  --seeds 1,2,3 \
  --num-episodes 1000 \
  --device mps \
  --planner-preset wm-calibrated \
  --output reports/micro_fouloide_resource_navigation_1000.md
```

Validation officielle 1000 épisodes, seeds 1/2/3 :

```text
seed | Q-only | Q+WM | delta | ratio | used | food | water | damage | max
-----|--------|------|-------|-------|------|------|-------|--------|----
   1 |  117.1 | 117.4 |  +0.4 |  1.00 | 23.8% | 0.49 |  0.82 |   1.78 | 272
   2 |  107.8 | 109.3 |  +1.5 |  1.01 | 62.7% | 0.48 |  0.67 |   2.06 | 259
   3 |  112.7 | 114.0 |  +1.3 |  1.01 | 54.9% | 0.49 |  0.72 |   1.90 | 270
mean |  112.5 | 113.6 |  +1.0 |  1.01 | 47.1% | 0.49 |  0.74 |   1.91 | 267
```

Rapport : `reports/micro_fouloide_resource_navigation_1000.md`

Décision validation : **GO partiel / candidat demo**, pas encore version finale.

```text
Le score moyen Q+WM monte à 113.6, supérieur au rough calibré précédent
(110.0), avec un vrai gain d'hydratation (0.74 eau/episode). Le World Model
reste marginal (+1.0 lifespan moyen), et le seuil ambitieux `mean Q-only >= 115`
n'est pas atteint. Cette variante est donc un candidat concret pour démo courte,
mais pas encore un modèle final robuste.
```

Prochaine décision avant promotion :

```text
Faire une demo directe médiane sur seeds 1/2/3. Si les 3 rollouts médians
atteignent le cap 120 sans mort, promouvoir `resource_navigation_v0_demo`.
Sinon, corriger le point restant : énergie basse / repos excessif / planner
WM encore trop peu utile.
```

Résultat demos médianes alignées training, seeds 2 et 3 :

```text
Seed 2 :
- selected_seed = 10042
- lifespan = 96, dead = true
- planner_used = 74.0%
- events: move_blocked=26, interact_food=1, interact_water=0, health_loss=35

Seed 3 :
- selected_seed = 10049
- lifespan = 96, dead = true
- planner_used = 61.5%
- events: move_blocked=17, interact_food=1, interact_water=0, health_loss=35
```

Décision demo : **NO-GO promotion visuelle stable**.

```text
Le modèle apprend bien en moyenne et boit plus souvent en évaluation, mais la
demo médiane n'est pas robuste. Les seeds 2/3 se bloquent contre les obstacles,
ne trouvent pas l'eau, puis meurent à hydratation zéro. Le prochain correctif
n'est plus le reward ressource global : il faut durcir la politique de
déploiement contre les actions invalides sans réintroduire le biais `REST`.
```

Test de déploiement suivant :

```text
Réessayer les demos médianes avec guards runtime activés mais objectif runtime
désactivé. Cela teste uniquement l'anti-boucle `move_blocked` / `interact_noop`,
sans le shaping de survie qui avait encouragé trop de repos.
```

Commande type :

```bash
python scripts/demo_micro_fouloide_promoted.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed2/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml \
  --seed 2 \
  --device mps \
  --skip-eval \
  --find-rollout \
  --rollout-search-count 64 \
  --rollout-select median \
  --rollout-max-steps 120 \
  --disable-survival-objective
```

Résultat de ce test :

```text
Seed 2, guards ON + objectif runtime OFF :
- selected_seed = 10013
- lifespan = 96, dead = true
- events: move_ok=34, rest=21, wait=5, interact_water=0, health_loss=35

Seed 3, guards ON + objectif runtime OFF :
- selected_seed = 10012
- lifespan = 96, dead = true
- events: move_ok=33, rest=19, wait=8, interact_water=0, health_loss=35
```

Décision : **NO-GO**.

```text
Les guards suppriment bien les boucles `move_blocked`, mais la demo médiane
meurt encore par hydratation zéro. Le pattern restant est un excès de passif
avant la crise (`REST/WAIT` alors que l'eau descend de 0.55 vers 0.30), puis
une recherche trop tardive.
```

### Étape E — Navigation active anti-passif

Config ajoutée :

```text
configs/micro_fouloide_v0_rough_valueplanner_resource_navigation_active.yaml
```

Principe :

```text
- monde rough inchangé ;
- `passive_penalty_threshold=0.55` pour pénaliser REST/WAIT avant la crise ;
- pénalité passive renforcée ;
- léger renforcement du signal local d'eau visible ;
- mêmes critères : training multi-seed, eval 1000, puis demos médianes.
```

Run seed 1 :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation_active.yaml \
  --episodes 3000 \
  --seed 1 \
  --device mps \
  --inference-device cpu \
  --out-dir runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_active_seed1
```

Résultat seed 1 :

```text
Training 3000 épisodes :
- final mean lifespan last100 = 109.4
- food100 = 0.69
- water100 = 0.67
- damage100 = 2.42
```

Décision : **NO-GO**.

```text
La pénalité passive plus précoce n'améliore pas le modèle. L'eau reste correcte,
mais le lifespan descend sous `resource_navigation_seed1` (120.1) et sous le
seuil GO. Il ne faut pas poursuivre cette branche sur seeds 2/3.
```

### Étape F — Mémoire de ressources en déploiement

Implémentation ajoutée dans `scripts/demo_micro_fouloide_promoted.py` :

```text
--resource-memory
```

Principe :

```text
Pendant le rollout uniquement, mémoriser les positions globales des ressources
visibles (`WATER`, `FOOD`). Si l'hydratation ou l'énergie tombe sous seuil,
diriger l'action vers la ressource mémorisée la plus proche, ou interagir si
l'agent est dessus.
```

Cette mémoire est volontairement hors training :

```text
Elle teste l'hypothèse que le modèle sait survivre en moyenne mais manque d'une
navigation persistante en demo médiane. Si cela marche, la vraie étape suivante
sera d'intégrer une mémoire spatiale propre dans l'agent plutôt qu'un patch demo.
```

Commande seed 2 :

```bash
python scripts/demo_micro_fouloide_promoted.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed2/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml \
  --seed 2 \
  --device mps \
  --skip-eval \
  --find-rollout \
  --rollout-search-count 64 \
  --rollout-select median \
  --rollout-max-steps 120 \
  --disable-survival-objective \
  --resource-memory
```

Résultat :

```text
Seed 2 + resource memory :
- selected_seed = 10039
- lifespan = 120, dead = false, capped = true
- drive = 0.668
- planner_used = 92.5%
- resource_memory_used = 15.0%
- events: interact_water=3, interact_food=1, damage=1, health_loss=25

Seed 3 + resource memory :
- selected_seed = 10027
- lifespan = 120, dead = false, capped = true
- drive = 0.612
- planner_used = 86.7%
- resource_memory_used = 13.3%
- events: interact_water=3, interact_food=1, damage=8, health_loss=0
```

Avec le résultat seed 1 déjà capé à 120, les trois seeds ont maintenant une
demo médiane courte non morte à 120.

Rapport : `reports/micro_fouloide_resource_navigation_demo.md`

Décision : **GO pour démo visuelle courte**, avec réserve importante.

```text
La démo courte peut être lancée avec le checkpoint `resource_navigation`, guards
runtime, objectif runtime désactivé et `--resource-memory`. Cela prouve que le
blocage des rollouts médians venait bien d'un manque de mémoire spatiale
persistante plus que d'un manque global de reward ressource.

Ce n'est pas encore un modèle final : `--resource-memory` est un patch de
déploiement. La prochaine étape sérieuse est d'intégrer cette mémoire spatiale
dans l'agent/planner, puis de valider que la démo fonctionne sans logique
spéciale dans le script.
```

Commande de démo candidate :

```bash
python scripts/demo_micro_fouloide_promoted.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed2/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml \
  --seed 2 \
  --device mps \
  --skip-eval \
  --find-rollout \
  --rollout-search-count 64 \
  --rollout-select median \
  --rollout-max-steps 120 \
  --disable-survival-objective \
  --resource-memory
```

---

### 6.17 Pivot homéostatique : apprentissage online sans pré-entraînement (2026-06-12)

**Changement de paradigme.** Abandon de « survie difficile + entraînement
offline + checkpoint » au profit d'un agent **homéostatique** qui apprend en
continu, en direct, sans aucun pré-entraînement. Motivation : un vrai world
model apprend de ses expériences courantes ; et avec une mort à ~96 steps,
l'agent n'avait jamais le temps d'apprendre en ligne. La preuve causale
offline `Q + WM > Q-only` (sections 6.x précédentes) n'est plus la thèse
centrale ; l'objectif devient la démo d'apprentissage **visible en live**.

**Implémenté (commits `0bdc979` → `c6059ad`, issues seedmind-rdb/dpy/bxh/bvt/pvg) :**

- *Monde doux persistant* (`micro_fouloide_world.py`, opt-in rétro-compatible) :
  `soft_death` (drives critiques ⇒ santé plancher 0.20 pendant une grâce
  bornée, puis la faim/soif peuvent tuer ; régénération quand les drives vont
  bien),
  `resource_regrow_steps` (repousse sur place ⇒ géographie stable, routines
  possibles), `max_steps: 0` (monde infini, plus de resets).
- *Bien-être* (`seedmind/training/wellbeing.py`) : zones de confort par drive
  (bloc `drive_reward.comfort`), consolide les trois copies dupliquées de
  `drive_regulation`.
- *Apprentissage continu* (`seedmind/training/online.py`, `OnlineLearner`) :
  WM + DQN + Value mis à jour tous les 8 steps depuis le buffer live ; le
  seuil du gate planner devient un **quantile glissant** (q0.60 sur ≤2000
  transitions récentes, rafraîchi tous les 500 steps) — remplace la
  calibration posthoc ; gate fermé pendant le warmup (cold start : Q-only,
  le planner s'ouvre quand le WM devient fiable).
- *Démo live* (`demo_fouloides_front.py --source live`) : agent **vierge**
  branché au viewer, 1.6 ms/tick CPU apprentissage inclus. Cerveau
  **persistant** : checkpoint périodique + au Ctrl+C, auto-repris au
  lancement (`--live-fresh` pour repartir de zéro). HUD enrichi : bien-être,
  wm_loss, planner_used, `soif→eau` (steps entre soif et gorgée), vie
  courante + record. Viewer : brouillard de perception (losange Manhattan
  r=4), barres de drives type Sims, DANGER en lave distinct des rochers,
  zones chaude (ocre) / froide (givre), légende.
- Config : `configs/micro_fouloide_online_homeostatic.yaml`. Runner headless :
  `scripts/run_fouloide_online.py` (expose `OnlineFouloideSession`,
  `--checkpoint-every/--resume`).

**Épisode de recherche : reward hacking découvert et corrigé (seedmind-pvg).**
Première validation 3 seeds × 90k steps : le bien-être *baissait* avec
l'apprentissage (0.45 → 0.31) et les gorgées s'effondraient (4-7/1k → 0).
Diagnostic par analyse du buffer du checkpoint (10k transitions récentes) :
l'agent avait appris à **tourner autour de l'eau, éternellement assoiffé,
sans jamais boire** — 87 % MOVE, hydratation 0.00, 0 gorgée/10k, en farmant
les `local_signal_bonus` (+0.216/step en moyenne sur les steps « move_ok +
soif + eau visible », 27 % des steps). Sans échéance de mort, la rente était
infinie ; boire aurait éteint la soif qui la conditionnait. Deux leçons
générales : (1) tout bonus d'état conditionné à un besoin non satisfait est
farmable dès qu'aucune échéance ne force la consommation ; (2) le shaping en
mode `delta` télescope à ~0 sur tout cycle et rend l'inaction au plancher
gratuite. Fix : suppression des bonus signal (les bonus one-shot de
consommation suffisent) + `drive_reward.mode: absolute` (le confort paie à
chaque step, la privation coûte à chaque step — formulation homéostatique
propre).

**Validation du fix (3 seeds × 80-84k steps) — go sans ambiguïté :**

| phase            | bien-être   | gorgées/1k |
|------------------|-------------|------------|
| 0-16k (eps haut) | 0.44-0.51   | 4-6        |
| 16-40k           | 0.59-0.76   | 9-12       |
| 40-84k (eps 0.05)| **0.96-0.98** | **~15, stable** |

Trajectoire inverse de la baseline : plus l'exploration diminue, plus il boit
et mieux il vit. À eps 0.05 l'agent passe ~95 % du temps dans ses zones de
confort. Confirmé visuellement en démo live (observation user : « il a appris
à survivre »). C'est la **première preuve du pivot** : un agent vierge
apprend l'homéostasie en ligne, sans pré-entraînement.

**Limite identifiée / prochaine marche :** pas de mémoire spatiale apprise —
l'encodeur est sans état, ce qui sort du losange de vision n'existe que via
les habitudes du Q-network. Quand l'eau est hors de vue, l'agent erre au lieu
d'y *retourner* (l'indicateur `soif→eau` du HUD le montre). Chantier ouvert :
carte égocentrique apprise, remplie par le vécu, en canaux d'entrée de
l'encodeur (seedmind-cfg). Ensuite : reproduction/population (phase 2 de la
vision).

**Ajustement live suivant (2026-06-13).** La démo `online_properties` ne doit
pas valider une survie artificielle à `HP=0.20` avec `E=0/H2O=0`. Le plancher
devient une fenêtre d'apprentissage (`soft_death_grace_steps: 300`) : assez
longue pour apprendre en ligne, mais finie. Après expiration, la privation
continue tue réellement. En parallèle, la curiosité est relevée et les actions
d'objets (`pick/plant/combine`) reçoivent un signal positif/négatif explicite ;
le HUD affiche aussi les taux roulants mouvement/passivité/boire/manger/objet
pour vérifier si le fouloïde explore ou s'immobilise.

**Ajustement du 2026-06-14.** Le passage au bien-être strict a rendu la démo
plus honnête mais trop dure : à ~61k steps, vie 131, record 1271, l'agent
était à `bien-être=0`, `E=0/H2O=0`, buvait/mangeait ~0.4 % du temps, avec
~29 % de blocages et un planner utilisé ~94 %. Diagnostic : le planner online
prenait trop tôt le contrôle avec un WM encore mauvais. Fix : planner plus
conservateur (`warmup_steps: 12000`, quantile 0.35, poids 0.08, marges plus
strictes) et signal de survie plus net pour boire/manger/perte de HP/mort.

```bash
# Démo live (agent vierge, cerveau persistant auto-repris)
python scripts/demo_fouloides_front.py --source live --tick-ms 60
# http://localhost:8787   (--live-fresh pour repartir de zéro)

# Validation headless
python scripts/run_fouloide_online.py \
  --config configs/micro_fouloide_online_properties.yaml \
  --steps 100000 --seed 1
```

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
  sandbox_v2_craft_balanced_causalwm.yaml # balanced + causal features/event heads
  sandbox_v2_craft_pressure.yaml # craft sous pression causale
  micro_fouloide_v0.yaml       # organisme minimal multi-drives
  micro_fouloide_v0_rough.yaml # variante V0 plus dure
  micro_fouloide_v0_rough_wmfocus.yaml # rough + WM focalisé rare/terminal
  micro_fouloide_v0_rough_wmfocus_light.yaml # rough + WM focus modéré
  micro_fouloide_v0_rough_valueplanner.yaml # rough + ValueModel pour planner
  micro_fouloide_v0_rough_valueplanner_online_uncertainty.yaml # étape 4: calibration incertitude online
  micro_fouloide_v0_rough_valueplanner_late_calibrated.yaml # étape 4: consolidation finale intégrée
  micro_fouloide_v0_rough_valueplanner_resource_seek.yaml # rough + reward ressource
  micro_fouloide_v0_rough_valueplanner_resource_curriculum.yaml # monde facilité, no-go transfert
  micro_fouloide_v0_rough_valueplanner_resource_seek_guarded.yaml # guards training, no-go
  micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml # rough + signal local de navigation ressource

runs/                        # gitignored
  sandbox_0/                   # v0 entraîné
  sandbox_v1/                  # v1 entraîné (baseline)
  sandbox_v1_planning/         # 1er essai planning
  sandbox_v1_planning2/        # planning conservateur
  sandbox_v2_craft/            # prochain run craft à produire
  sandbox_v2_craft_balanced_causalwm_rebalanced_cuda/ # premier signal WM planner > Q-only
  micro_fouloide_v0_drive_reward_3000/ # V0 multi-drives validé Q-only > naive
  micro_fouloide_v0_rough_seed*/ # rough validé Q-only > naive sur 3 seeds
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
| Sandbox craft simple | 16×16 craft optionnel | Lifespan eval | 64.7 | 222.3 | **3.44×**, craft faible |
| Sandbox causal-WM rebalanced | 16×16 craft + causal features | Lifespan eval | 62.2 | 115.7 Q-only / 124.0 Q+WM | **WM/Q 1.07×** |
| Micro-fouloïde V0 | 16×16 multi-drives | Lifespan eval | 120.5 | 157.6 Q-only moy. 3 seeds | **Q/naïf 1.31×**, WM/Q ~0.98× |
| Micro-fouloïde rough | 16×16 multi-drives plus dur | Lifespan eval | 89.0 | 107.9 Q-only / 110.0 Q+WM calibré moy. 3 seeds | **Q/naïf 1.21×**, **WM/Q 1.02×** |
| Micro-fouloïde rough + navigation ressource | 16×16 rough + signal local ressource | Lifespan eval seed1 | 89.0 | 117.1 Q-only / 117.4 Q+WM | **Q/naïf 1.31×**, demo médiane cap 120 |

**Conclusion actuelle :** la preuve de concept « apprendre seul à survivre par causalité » tient sur v0 et v1. Le monde plus grand avec vision partielle est **plus difficile mais mieux résolu** (ratio 4.24× vs 2.89×). Le craft simple prouve que l'agent sait survivre, mais pas qu'il exploite spontanément les outils. Le causal-WM rebalanced a fourni le premier signal positif direct pour la thèse centrale. Micro-fouloïde rough donne maintenant une preuve minimale plus proche de l'objectif : **à checkpoint identique, Q + World Model planner calibré bat Q-only sur 3 seeds et 1000 épisodes**. Le signal reste modeste ; l'étape suivante est de rendre cette calibration moins posthoc et plus apprise en continu.

**Jalon démo courte :** la variante `resource_navigation` a maintenant un
chemin de démo praticable avec mémoire spatiale de ressources. La mémoire est
sortie du script de démo vers `seedmind/agent/spatial_resource_memory.py` et
reste opt-in via `--resource-memory`. Elle ne remplace pas l'objectif final
d'un agent qui apprend sa mémoire en interne, mais elle transforme le patch de
déploiement en composant testable et réutilisable pour brancher une première
démo visuelle.

**Démo live riche (14 juin 2026) :** le viewer doit refléter le chantier réel,
pas revenir à une vitrine plus simple. Le diagnostic actuel est que les actions
d'artefacts (`PICK`, `DROP`, `PLANT`, `COMBINE`) arrivent trop tôt : elles
diluent l'exploration alors que boire/manger n'est pas encore stabilisé. La
solution retenue est un curriculum d'actions : les objets restent visibles et
observables, mais leur manipulation ne se débloque qu'après une compétence
récente de survie (bien-être moyen, eau et nourriture consommées). Ce n'est pas
une règle métier codée en dur ; c'est une progression de complexité pour que
l'apprentissage online commence par l'homéostasie avant le craft. La démo live
active aussi les filtres d'actions physiquement inutiles (`move_blocked`,
`interact_noop`, inventaire impossible) afin que l'exploration aléatoire
produise plus souvent des expériences informatives sans écrire la stratégie.

**Gel de branche `sandbox-world` (14 juin 2026).** Verdict de la démo riche
online : **échec partiel à ne pas masquer**. Malgré les garde-fous, à ~39k
steps la vie 60 affiche `record=6584`, `HP=0.20`, `E=0.00`, `H2O=0.00`,
`bien-être=0.00`, `bien-être_avg=0.158`, `boire=0.0%` sur la fenêtre récente
et `manger=0.2%`. C'est très inférieur au baseline homéostatique précédent
(`configs/micro_fouloide_online_homeostatic.yaml`) qui atteignait 40k+ steps et
un bien-être stable `0.96-0.98` à exploration faible. Conclusion : la branche
actuelle est conservée comme témoin de régression riche. La suite se fait sur
une nouvelle branche dédiée : repartir du monde `online_homeostatic` validé,
puis réintroduire **une nouveauté à la fois** avec critères de passage
explicites :

1. baseline homéostatique 16×16 inchangée ;
2. carte plus grande seule ;
3. mémoire spatiale seule ;
4. regrowth non exploitable (respawn non fixe si besoin) ;
5. entités/propriétés visibles sans actions d'artefacts ;
6. actions `PICK/DROP/PLANT/COMBINE` après stabilité.

Critères minimum avant chaque ajout : `bien-être_avg` remonte et reste haut,
`boire` et `manger` restent non nuls, `soif→eau` baisse, et le record progresse
sans camping au plancher `HP=0.20`.

**Branche de reprise :** `codex-homeostasis-rebuild` repart de ce gel mais
repointe le mode live par défaut vers
`configs/micro_fouloide_online_homeostatic.yaml` avec un checkpoint séparé
`runs/fouloide_live_homeostatic/checkpoint_live.pt`. La config riche reste
exécutable explicitement via `--live-config
configs/micro_fouloide_online_properties.yaml`, mais n'est plus le chemin par
défaut de reconstruction.

**Validation après run long (15 juin 2026).** La baseline de reprise est
confirmée fonctionnelle en démo live après temps d'apprentissage suffisant :
l'agent vierge repart bien de zéro, puis atteint de nouveaux records autour de
25k+ steps. La phase initiale peut paraître mauvaise visuellement, mais elle se
redresse quand l'expérience accumulée devient suffisante. Verdict : `master`
peut représenter cet état fonctionnel ; la suite doit continuer par
l'introduction isolée d'une **carte plus grande seule**, sans artefacts ni autre
nouveauté simultanée.

**Étape incrémentale en cours — carte 32×32 seule (15 juin 2026).** La première
nouveauté après le retour au baseline est volontairement minimale :
`configs/micro_fouloide_online_homeostatic_bigmap.yaml` garde le même agent, les
mêmes récompenses, les mêmes actions et le même monde homéostatique, mais passe
la carte à 32×32 avec des densités de ressources/obstacles approximativement
conservées. Le bien-être y utilise l'agrégation stricte
`mean_min_product` : si eau, énergie, santé ou température sortent vraiment de
la zone viable, les autres besoins ne peuvent pas compenser artificiellement le
score. Le viewer live expose cette étape via `--live-bigmap`, avec un checkpoint
séparé pour ne pas polluer le cerveau validé 16×16 :

```bash
python scripts/demo_fouloides_front.py \
  --source live \
  --tick-ms 60 \
  --live-bigmap \
  --live-fresh
```

Critère de passage : laisser apprendre depuis zéro et vérifier visuellement que
les records repartent à la hausse, que `boire` et `manger` deviennent non nuls,
et que le bien-être remonte durablement avant d'ajouter la mémoire spatiale ou
les artefacts.

---

## 9. Demo front fouloïdes (préparation)

**Objectif :** préparer l'interface visible navigateur de la future démo
écosystème fouloïdes, avant que le moteur world model soit prêt.

**Implémenté (juin 2026) :**

- `seedmind/visualization/fouloides_viewer.html` — viewer canvas pixel-art
  fidèle au style fouloïdes de référence (herbe avec patchs usés, arbres,
  pommes, fouloïdes jaunes, baignoires, rochers) ; sprites générés
  procéduralement, caméra pan/zoom (drag, molette, flèches), HUD type jeu
  (badge population, bannière OBJECTIF), interpolation des déplacements.
- `scripts/demo_fouloides_front.py` — serveur HTTP + WebSocket (même pattern
  que `live_sandbox.py`). Le mode par défaut reste un **stub**
  (`StubFouloideWorld`, marche aléatoire attirée par les pommes) qui alimente
  le front avec des données plausibles sur un grand monde (96×96 par défaut).
- `scripts/demo_fouloides_front.py --source micro` — premier branchement réel :
  charge un checkpoint Micro-Fouloïde `resource_navigation`, active le planner
  calibré + la mémoire de ressources opt-in, et projette l'état réel dans le
  viewer (agent, pommes, baignoires/eau, obstacles/dangers, HUD HP/H2O/E).

**Point de branchement moteur :** l'interface `WorldSource`
(`world_message()` statique + `step_message()` par tick). Le premier adaptateur
réel est `MicroFouloideWorldSource`; le front reste agnostique et peut encore
servir le stub ou le moteur entraîné.

```bash
python scripts/demo_fouloides_front.py --size 96 --fouloides 14
# Ouvrir http://localhost:8787

python scripts/demo_fouloides_front.py \
  --source micro \
  --seed 3 \
  --device cpu \
  --micro-uncertainty-threshold 0.69918
# Ouvrir http://127.0.0.1:8787
```

---

## 10. Hors scope immédiat (vision long terme)

- Reproduction, multi-agents, construction libre (BUILD)
- Monde physique, ComputerWorld
- Interface type « fouloïdes » avec écosystème autonome

**Objectif final détaillé** (preuve d'efficacité du World Model, protocole A/B, chemin B→F) : voir [GOAL_WORLD_MODEL_FOULOIDES.md](./GOAL_WORLD_MODEL_FOULOIDES.md).

**Étape intermédiaire définie** : voir [MICRO_FOULOIDE_SPEC.md](./MICRO_FOULOIDE_SPEC.md).

Le premier environnement V0 est en place : `MicroFouloideWorld`,
`configs/micro_fouloide_v0.yaml`, l'encodeur dédié, le runner/evaluator et les
tests unitaires associés.

L'architecture (registres, adapters, modules découplés) est conçue pour accueillir ces extensions sans réécriture du cœur agent.

---

## Piste recherche — World-model récurrent (RSSM) + perception égocentrée (18 juin 2026)

Exploration sur la branche **`rssm-egocentric`** (isolée ; `main`/prod intacts).
Résultat clé prouvé : la **perception égocentrée sans mémoire s'effondre** alors
que le full-grid fourrage (contrôle discriminant) → **la mémoire est nécessaire**.
Machinerie RSSM construite et testée (encodeur conv gelé, WM récurrent GRU, DRQN
récurrent, actor-critic en imagination façon Dreamer). Verrou restant : stabilité
↔ exploration de l'actor-critic (le critic-cible EMA règle la divergence, mais
l'exploration s'effondre — piste : entropie / DreamerV3 complet).

**Décision** : la vitrine reste le **fouloïde full-grid en prod (qui vit déjà)** ;
le générique est un pari de recherche parké. Bilan complet, verdicts chiffrés et
pistes de reprise : **[BILAN_RSSM_2026-06-18.md](./BILAN_RSSM_2026-06-18.md)**
(issues bd `seedmind-oc4` / `seedmind-oc4.1`).

---

*Dernière mise à jour : 18 juin 2026*
