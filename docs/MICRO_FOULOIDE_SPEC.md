# Micro-Fouloïde — Spécification expérimentale

## 1. Objectif

Construire un monde virtuel plus riche que le sandbox craft, mais encore assez
minimal pour tester proprement l'hypothèse centrale :

```text
Un agent générique peut apprendre seul à réguler ses variables internes
dans un monde donné, via interaction, mémoire, drives internes et World Model,
sans règles spécifiques codées dans l'agent.
```

Le micro-fouloïde n'est pas encore un écosystème complet. C'est une étape
intermédiaire entre le sandbox et un monde fouloïde-like. Dans ce monde
particulier, la régulation des drives ressemble à de la survie biologique, mais
la survie n'est pas un principe architectural universel.

---

## 2. Principes non négociables

### Agent générique

L'agent ne connaît pas :

- les règles du monde ;
- la sémantique des entités ;
- les recettes ;
- la prochaine meilleure action ;
- la valeur stratégique d'un objet ou d'une zone.

L'agent apprend par :

```text
observation -> action -> conséquence -> mémoire -> amélioration
```

### Monde via `EnvironmentAdapter`

Le monde expose :

- observations ;
- actions disponibles ;
- features perceptibles structurées ;
- variables internes / drives perceptibles ;
- événements observés ;
- conséquences externes liées à la régulation des drives.

Le monde n'expose pas :

- solution ;
- heuristique de décision ;
- stratégie optimale ;
- shaping spécifique du type "va boire" ou "mange maintenant".

### Features structurées

Les features sont des capteurs ou états perceptibles :

```text
energy, hydration, temperature, health
position, local terrain, nearby entities
inventory/body state
```

Elles ne doivent pas encoder une règle ou une valeur stratégique.

### Drives internes

Le principe général n'est pas "survivre", mais :

```text
réguler des variables internes propres au monde ou à l'organisme simulé.
```

Dans le micro-fouloïde, ces variables prennent une forme biologique
(`energy`, `hydration`, `temperature`, `health`). Dans un autre monde, elles
pourraient représenter autre chose : stabilité, cohérence, précision, ressource
interne, intégrité mémoire, ou objectifs adaptatifs temporaires.

---

## 3. Organisme minimal

Le micro-fouloïde est un seul organisme contrôlé par l'agent.

### États internes

| Feature | Sens | Dynamique initiale |
|---------|------|--------------------|
| `energy` | réserve énergétique | baisse à chaque pas |
| `hydration` | réserve d'eau | baisse à chaque pas |
| `temperature` | confort thermique interne | dérive selon zone |
| `health` | intégrité globale | baisse si besoins critiques |

Toutes les features sont normalisées dans `[0, 1]` pour l'agent.

### Mort

L'épisode se termine si :

```text
health <= 0
```

`health` baisse quand un besoin reste trop bas ou trop haut trop longtemps :

```text
energy trop basse     -> health diminue
hydration trop basse  -> health diminue
temperature extrême   -> health diminue
danger / collision    -> health diminue
```

---

## 4. Monde minimal

Grille 2D partiellement observable.

### Entités initiales

| Entité | Rôle observable | Interaction attendue |
|--------|-----------------|----------------------|
| `FOOD` | ressource énergétique | `INTERACT` peut augmenter `energy` |
| `WATER` | ressource hydratation | `INTERACT` peut augmenter `hydration` |
| `WARM_ZONE` | zone chaude | influence `temperature` |
| `COLD_ZONE` | zone froide | influence `temperature` |
| `DANGER` | zone ou objet nocif | peut réduire `health` |
| `OBSTACLE` | bloque mouvement | collision / contournement |
| `EMPTY` | espace libre | aucun effet direct |

Ces noms existent dans le monde pour générer les observations et les events.
L'agent ne doit manipuler que des ids / vecteurs.

### Observation partielle

L'agent voit un rayon local :

```text
visibility_radius = 4
```

Les cellules hors rayon sont `UNKNOWN`.

---

## 5. Actions

Actions de base :

```text
MOVE_UP
MOVE_DOWN
MOVE_LEFT
MOVE_RIGHT
INTERACT
REST
WAIT
```

### Sémantique monde

Le monde applique les conséquences, mais l'agent ne les connaît pas.

Exemples :

- `INTERACT` sur `FOOD` peut augmenter `energy`.
- `INTERACT` sur `WATER` peut augmenter `hydration`.
- `REST` peut ralentir la perte d'énergie mais ne résout pas l'hydratation.
- `WAIT` laisse les dynamiques agir.
- Se déplacer dans une zone chaude/froide influence `temperature`.

---

## 6. Drives

### Drive de régulation interne

Objectif local du micro-fouloïde :

```text
maintenir ses variables internes dans des zones viables.
```

Dans ce monde biologique minimal, cela ressemble à rester vivant. Mais dans
l'architecture générale, ce drive doit être compris comme une **régulation de
variables internes**, pas comme une survie obligatoire.

Reward externe minimal proposé pour ce monde :

```text
+alive_bonus à chaque pas vivant
-death_penalty à la mort
```

Option expérimentale à discuter : ajouter une petite pénalité générique lorsque
`health` baisse, car c'est une conséquence observable de dégradation interne.
Cette option reste spécifique au monde biologique et ne doit pas devenir une
abstraction universelle.

### Drive épistémique

Drive intrinsèque :

```text
réduire l'incertitude utile sur les conséquences des actions
```

Il ne doit pas récompenser le chaos pur. Il doit favoriser les transitions
informatives :

- forte erreur de prédiction ;
- changement de features ;
- événement peu compris ;
- conséquence contrôlable par action.

### Généralisation hors monde biologique

Dans un monde sans vie/mort, il n'y aurait pas de `death_penalty` ni de
`health`. Le même agent devrait apprendre à réguler d'autres variables internes
ou objectifs adaptatifs. Le micro-fouloïde sert donc de cas d'étude, pas de
définition générale du drive.

---

## 7. Perception causale générique

Le monde expose via `EnvironmentAdapter` :

```python
causal_feature_names() -> list[str]
causal_features(observation) -> np.ndarray
causal_event_names() -> list[str]
```

Pour le micro-fouloïde :

```text
causal_features:
  energy
  hydration
  temperature
  health
  local_danger
  local_food_signal
  local_water_signal
  local_heat_signal
```

Les signaux locaux sont des capteurs perceptibles, pas des instructions.

Exemples d'events :

```text
move_ok
move_blocked
interact_food
interact_water
interact_noop
rest
temperature_up
temperature_down
damage
health_loss
death
```

Ces events servent au diagnostic et au World Model auxiliaire. L'agent ne doit
pas contenir de logique spécifique à ces labels.

---

## 8. World Model cible

Le World Model doit apprendre :

```text
(latent_state, action) ->
  next_latent_state
  reward
  uncertainty
  delta_causal_features
  event_logits
  done / terminal risk
```

Point important : la valeur des conséquences doit être apprise par l'agent, pas
codée dans la config.

Les poids/targets manuels utilisés dans `sandbox_v2_craft_balanced_causalwm`
sont considérés comme un échafaudage expérimental, pas comme l'architecture
finale.

---

## 9. Planner cible

Le planner doit :

1. imaginer plusieurs futurs via le World Model ;
2. évaluer ces futurs avec une valeur apprise ;
3. choisir l'action selon l'utilité future attendue.

Architecture cible :

```text
state / latent
  -> World Model rollouts
  -> Value Model appris
  -> action
```

À éviter à terme :

```text
planner_feature_weights codés à la main
planner_feature_targets codés à la main
```

---

## 10. Protocole de validation

Comparer au minimum :

1. `naive` aléatoire ;
2. `Q-only` ;
3. `Q + World Model planner` ;
4. idéalement `Q + causal-WM aux heads` sans planner ;
5. idéalement plusieurs seeds.

### Critère sérieux

Validation d'une évolution importante :

```text
Q + WM planner > Q-only
sur moyenne multi-seed
avec amélioration d'au moins une métrique de régulation interne ou causale.
```

### Métriques principales

| Métrique | But |
|----------|-----|
| `lifespan` | durée d'épisode dans ce monde biologique |
| `drive_regulation_score` | stabilité globale des variables internes |
| `mean_health` | viabilité biologique locale |
| `mean_energy` | régulation énergétique |
| `mean_hydration` | régulation hydrique |
| `temperature_stability` | régulation thermique |
| `interact_food` | exploitation nourriture |
| `interact_water` | exploitation eau |
| `damage_events` | évitement danger |
| `death_cause` | diagnostic |

### Métriques World Model

| Métrique | But |
|----------|-----|
| `feature_delta_mse` | qualité prédiction des conséquences |
| `event_accuracy` | qualité prédiction événement |
| `terminal_risk_error` | qualité prédiction mort / danger |
| `planner/Q ratio` | gain décisionnel du WM |

---

## 11. Roadmap d'implémentation

### Étape 1 — Monde minimal

- Créer `MicroFouloideWorld`.
- Ajouter config `configs/micro_fouloide_v0.yaml`.
- Ajouter tests unitaires dynamiques :
  - decay energy/hydration ;
  - interact food/water ;
  - temperature zone ;
  - danger/health/death ;
  - observation partielle.

### Étape 2 — Runner / eval

- Réutiliser autant que possible `run_sandbox.py` ou créer un runner générique.
- Évaluation naive vs Q-only vs Q+WM.
- Metrics JSON avec besoins et events.

### Étape 3 — Baseline Q-only

- Prouver `trained > naive`.
- Vérifier routines simples :
  - manger quand énergie basse ;
  - boire quand hydratation basse ;
  - éviter danger ;
  - chercher zone thermique viable.

### Étape 4 — World Model

- Activer têtes auxiliaires génériques.
- Diagnostiquer :
  - delta features par action ;
  - event logits ;
  - terminal risk.

### Étape 5 — Planner appris

- Remplacer progressivement les poids/targets manuels par une valeur apprise.
- Objectif : `Q + WM planner > Q-only`.

---

## 12. Questions ouvertes

1. Faut-il inclure une petite pénalité externe sur baisse de `health`, ou garder
   uniquement `alive/death` pour ce monde biologique ?
2. Le `Value Model` du planner doit-il être le Q-network existant, un critic
   latent séparé, ou une tête supplémentaire du World Model ?
3. Les features locales (`local_food_signal`, etc.) doivent-elles être scalaires
   globaux ou uniquement présentes dans la grille ?
4. Combien de seeds pour valider sérieusement : 3, 5, 10 ?
5. À quel moment introduire plusieurs organismes ?
6. Comment nommer et représenter les drives de manière assez générale pour
   fonctionner dans des mondes sans vie/mort ?

---

## 13. Résultat V0 — juin 2026

Après correction de la perception (`standing_entity` présent aussi dans le
replay compact) et ajout d'une reward d'apprentissage générique de régulation
des drives (`reward_learning`), le micro-fouloïde V0 valide la baseline
`trained > naive`.

Run principal :

```text
runs/micro_fouloide_v0_drive_reward_3000
```

Entraînement :

```text
ep 2999 | lifespan(100)=204.4
drive100=0.715 food100=2.25 water100=2.25 dmg100=2.31 eps=0.10
```

Évaluation 100 épisodes :

```text
Naive:
  lifespan=120.5 +/- 31.6
  drive=0.661
  food=0.46 water=0.51 damage=4.23

Trained Q-only:
  lifespan=192.9 +/- 72.7
  drive=0.691
  food=0.96 water=1.67 damage=0.80

Trained Q + WM planner:
  lifespan=185.1 +/- 74.4
  drive=0.691
  food=1.13 water=1.60 damage=0.87
```

Conclusion :

```text
Q-only / naive = 1.60x
WM planner / Q-only = 0.96x
```

Le résultat valide l'apprentissage d'une routine adaptative multi-drives. Il ne
valide pas encore l'utilité du planner World Model dans ce monde.

Validation multi-seed 3000 épisodes :

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

Interprétation :

- `Q-only > naive` sur les 3 seeds ;
- la routine adaptative est cohérente : plus food/water, moins damage ;
- le drive moyen reste une métrique plus fragile que le lifespan ;
- le planner World Model manuel reste non validé sur V0.

Variante rough :

```text
configs/micro_fouloide_v0_rough.yaml
```

Cette variante garde la même interface agent, mais rend le monde plus dur :
moins de ressources, besoins plus rapides, plus de dangers, santé plus fragile.

Résultat multi-seed 3000 épisodes :

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

- la policy généralise à une variante plus dure sans changer l'agent ;
- le signal est plus faible que V0 ;
- l'agent apprend surtout évitement danger + eau ;
- le planner World Model manuel reste non validé.

Diagnostic World Model disponible :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough.yaml \
  --num-episodes 1 \
  --device mps \
  --diagnose-world-model \
  --diagnostic-samples 20000
```

Premier diagnostic court rough seed1 :

```text
interact_food   event_acc=0.667 feature_mae=0.05370
interact_water  event_acc=0.833 feature_mae=0.06078
damage          event_acc=0.118 feature_mae=0.10716
death           event_acc=0.000 reward_mae=0.98121
temperature_*   event_acc=0.000 feature_mae≈0.104
rest/wait       event_acc≈0.99 feature_mae≈0.011
```

Hypothèse actuelle sur l'échec du planner :

```text
Le World Model prédit correctement les transitions simples,
mais pas encore les conséquences rares/critiques.
```

Tant que `damage`, `death` et les changements thermiques sont mal prédits, le
planner ne peut pas améliorer robustement la décision.

Config expérimentale suivante :

```text
configs/micro_fouloide_v0_rough_wmfocus.yaml
```

Elle garde le même monde rough et le même agent, mais focalise l'entraînement
du World Model sur les transitions rares/critiques de manière générique :

```text
reward_abs_weight: 2.0
reward_done_weight: 8.0
event_loss_weight: 1.0
event_class_balance_power: 1.0
feature_loss_weight: 3.0
```

Critère de succès :

- meilleure prédiction de `death`, `damage`, `temperature_up/down` ;
- pas de régression forte Q-only ;
- `planner/Q` au moins neutre, idéalement positif.

Résultat seed1 :

```text
Rough baseline:
  Q-only lifespan=108.0
  planner/Q=0.97
  death reward_mae=0.96182 event_acc=0.032
  damage event_acc=0.087

Rough wmfocus:
  Q-only lifespan=103.4
  planner/Q=0.99
  death reward_mae=0.71078 event_acc=0.231
  damage event_acc=0.253
```

Le diagnostic est positif côté événements critiques, mais négatif côté
calibration reward globale. La pondération `wmfocus` est donc trop agressive.

Suite : tester une variante plus légère :

```text
configs/micro_fouloide_v0_rough_wmfocus_light.yaml
```

Résultat `wmfocus_light` seed1 :

```text
Rough baseline:
  Q-only lifespan=108.0
  planner/Q=0.97
  death reward_mae=0.96182 event_acc=0.032
  damage event_acc=0.087

Rough wmfocus_light:
  Q-only lifespan=105.5
  planner/Q=1.01
  death reward_mae=0.79020 event_acc=0.061
  damage event_acc=0.140
  temperature_up event_acc=0.067
```

Le planner devient légèrement positif, mais le gain est trop faible pour être
une validation. Le diagnostic révèle surtout un décalage d'objectif :

```text
planner actuel -> predicted_reward_external + curiosity
DQN actuel     -> reward_learning, incluant Δdrive_regulation
```

La prochaine évolution doit donc apprendre une valeur des conséquences utilisable
par le planner, au lieu d'ajouter davantage de poids manuels au World Model.

Évolution implémentée :

```text
ValueModel(latent_state) -> expected long-term reward_learning
```

Fichiers :

- `seedmind/agent/value_model.py`
- `seedmind/training/value.py`
- checkpoints étendus pour sauvegarder/restaurer le ValueModel
- `Planner` étendu avec valeur terminale optionnelle

Config de test :

```text
configs/micro_fouloide_v0_rough_valueplanner.yaml
```

Objectif expérimental :

```text
Q + WM rollout + ValueModel terminal > Q-only
```

Le ValueModel est entraîné sur `reward_learning`, donc sur le même signal que la
policy DQN. C'est le premier test propre du planner aligné avec la régulation
interne.

Résultat rough seed1 :

```text
Naive:            lifespan=88.1  drive=0.654
Q-only:          lifespan=114.7 drive=0.658
Q + WM + Value:  lifespan=119.1 drive=0.661

Q/naive:   1.30x
planner/Q: 1.04x
```

Comportement :

```text
Q-only:         food=0.67 water=0.83 damage=1.32
Q + WM + Value: food=0.77 water=0.92 damage=1.30
Naive:          food=0.30 water=0.20 damage=3.75
```

Statut :

```text
Premier signal positif du planner aligné ValueModel sur micro-fouloïde.
```

Validation multi-seed rough `valueplanner` :

```text
seed | Q lifespan | Planner lifespan | planner/Q | Q drive | Planner drive | Q food/water/damage | Planner food/water/damage
1    | 114.7      | 119.1            | 1.04x     | 0.658   | 0.661         | 0.67 / 0.83 / 1.32  | 0.77 / 0.92 / 1.30
2    | 102.8      | 104.8            | 1.02x     | 0.657   | 0.662         | 0.24 / 0.30 / 0.61  | 0.28 / 0.35 / 0.55
3    | 107.4      | 105.3            | 0.98x     | 0.641   | 0.639         | 0.55 / 0.67 / 1.50  | 0.52 / 0.63 / 1.44
```

Moyenne seeds 1-3 :

```text
Naive lifespan:          88.1
Q-only lifespan:        108.3
Q + WM + Value lifespan: 109.7

Q/naive:       1.23x
Planner/naive: 1.25x
Planner/Q:     ~1.01x

Q-only:         drive=0.652 food=0.49 water=0.60 damage=1.14
Q + WM + Value: drive=0.654 food=0.52 water=0.63 damage=1.10
```

Conclusion :

```text
Signal planner positif en moyenne, mais faible.
Le planner améliore seeds 1 et 2, régresse seed 3.
Ce n'est pas encore une preuve forte, mais c'est le premier résultat
micro-fouloïde où WM + valeur apprise améliore Q-only en moyenne multi-seed.
```

Protocole de consolidation sans réentraînement :

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

Ce sweep teste si le faible signal planner vient d'un mauvais réglage de
`planning.weight` ou de `terminal_value_weight`, sans modifier l'entraînement ni
le checkpoint.

Résultat du sweep multi-seed :

```text
seed | Q-only | meilleur planner | meilleur ratio | meilleur réglage
1    | 114.7  | 116.6            | 1.02x          | p=0.10 tv=1.5
2    | 102.8  | 106.5            | 1.04x          | p=0.10 tv=2.0
3    | 107.4  | 108.0            | 1.01x          | p=0.25 tv=1.5
```

Interprétation :

```text
Le planner peut améliorer chaque seed individuellement, mais le gain reste
faible et les meilleurs poids ne sont pas encore stables entre seeds.
```

Le planner random-shooting est seedé via le seed de l'agent pour rendre les
comparaisons reproductibles.

Diagnostic ValueModel ajouté :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_valueplanner_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_valueplanner.yaml \
  --device mps \
  --diagnose-value-model \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Ce diagnostic compare la valeur prédite au retour Monte Carlo réel du replay,
globalement et par buckets de features causales (`low_energy`,
`low_hydration`, `low_health`, `danger_near`, `food_signal`, `water_signal`,
`terminal`).

Évolution ValueModel focus :

```text
configs/micro_fouloide_v0_rough_valueplanner_focus.yaml
```

Cette variante ne change pas le monde. Elle pondère seulement l'entraînement de
la valeur avec des signaux RL génériques : amplitude de cible TD, terminalité et
erreur de valeur. Objectif : mieux apprendre les états critiques que le planner
doit éviter dans ses rollouts.

Résultat :

```text
seed | Q-only | planner | planner/Q | value corr | low_health mae | terminal mae
1    | 101.8  | 99.7    | 0.98x     | 0.483      | 0.1997         | 0.1809
2    | 101.8  | 98.3    | 0.97x     | 0.530      | 0.1728         | 0.1394
3    | 119.6  | 112.2   | 0.94x     | 0.549      | 0.1971         | 0.1675
```

La valeur critique est mieux apprise, mais le planner régresse. Interprétation :
le ValueModel devient trop pessimiste globalement. La prochaine consolidation
doit travailler la calibration de valeur plutôt que simplement augmenter les
poids des états critiques.

Évolution Dyna latente minimale :

```text
configs/micro_fouloide_v0_rough_valueplanner_dyna.yaml
```

Le World Model est utilisé comme moteur d'apprentissage de la valeur :

```text
s,a réel du replay -> WM imagine s',r -> entraînement V(s)
```

Cette étape teste l'axe `WM improves learning`, sans encore injecter de replay
synthétique dans le DQN observationnel.

Résultat :

```text
seed | Q-only | planner | planner/Q | value corr | value bias | terminal mae
1    | 106.2  | 107.8   | 1.02x     | 0.440      | -0.0910    | 0.3922
2    | 102.8  | 100.7   | 0.98x     | 0.514      | -0.0790    | 0.3229
3    | 120.8  | 119.9   | 0.99x     | 0.529      | -0.0073    | 0.3814
```

Dyna latent préserve mieux l'entraînement que `valueplanner_focus`, mais le gain
planner reste neutre. Cela indique que la prochaine étape doit rapprocher la
policy elle-même du latent/WM, plutôt que seulement ajouter une valeur terminale
au planner.

Évolution policy latente :

```text
configs/micro_fouloide_v0_rough_latentq.yaml
```

Le `LatentQNetwork` apprend directement depuis `latent_state` et
`next_latent_state` du replay :

```text
latent -> Q(action)
```

Il est entraîné en parallèle du Q observationnel. L'objectif est de tester si la
représentation apprise devient un meilleur support de décision que l'observation
brute seule, avant d'y brancher Dyna de façon plus ambitieuse.

Résultat :

```text
seed | Q-only | LatentQ | LatentQ/Q | Planner | Planner/Q
1    | 112.2  | 92.9    | 0.83x     | 110.2   | 0.98x
2    | 103.5  | 96.2    | 0.93x     | 104.5   | 1.01x
3    | 111.9  | 92.8    | 0.83x     | 108.7   | 0.97x
```

LatentQ collapse vers trop d'`INTERACT` et ne capte pas assez le contexte
nécessaire. Le latent actuel n'est donc pas encore un support de policy direct.
Avant de brancher Dyna sur LatentQ, il faut diagnostiquer ce que le latent encode
réellement pour la décision.

Diagnostic latent ajouté :

```bash
python scripts/evaluate_micro_fouloide.py \
  --checkpoint runs/micro_fouloide_v0_rough_latentq_seed1/checkpoint_final.pt \
  --config configs/micro_fouloide_v0_rough_latentq.yaml \
  --device mps \
  --diagnose-latent \
  --diagnostic-samples 20000 \
  --diagnostics-only
```

Le test entraîne des probes linéaires depuis `latent_state` vers les features
causales et `standing_entity`. Si `standing_entity` n'est pas mieux décodable
qu'une baseline majoritaire, le latent ne porte pas encore assez clairement le
contexte d'interaction pour servir de support direct à une policy.

Résultat actuel sur `latentq` seeds 1-3 :

```text
standing_entity accuracy: 0.906-0.917
majority baseline:        0.882-0.898
gain réel:                +0.019 à +0.024
```

Le latent contient donc un signal partiel, mais trop faible face au déséquilibre
de classes. Les features locales utiles (`food/water/danger signal`) restent
également peu corrélées. Cela explique que LatentQ apprenne une préférence
d'action dégénérée plutôt qu'un usage contextuel d'`INTERACT`.

Évolution suivante : latent structuré générique.

```text
configs/micro_fouloide_v0_rough_latentq_structured.yaml
```

Le mécanisme optionnel `agent.structured_latent_features` réserve la fin du
latent aux features perceptibles exposées par l'adapter :

```text
latent = [projection_observation, adapter.causal_features(observation)]
```

Ce n'est pas une règle métier : l'agent ne reçoit pas la signification
stratégique des features, seulement des capteurs structurés. L'objectif est de
tester si une policy latente devient viable quand le latent conserve clairement
les perceptions actionnables.

Résultat : l'information devient décodable, mais LatentQ reste sous Q-only. Le
diagnostic `--diagnose-latent-q` montre un mauvais alignement de préférences
d'action avec la policy observationnelle. Prochaine variante :

```text
configs/micro_fouloide_v0_rough_latentq_structured_distill.yaml
```

Elle ajoute une distillation générique :

```text
LatentQ apprend TD + préférences d'action Qobs
```

But : vérifier si le blocage vient de l'apprentissage LatentQ lui-même plutôt
que du contenu informationnel du latent.

Résultat : la distillation MSE des valeurs centrées reste insuffisante. Les
marges Qobs sont trop faibles et LatentQ ne s'aligne pas. Variante suivante :

```text
configs/micro_fouloide_v0_rough_latentq_structured_policy_distill.yaml
```

Cette fois, LatentQ apprend directement l'action préférée par Qobs via
cross-entropy. Si l'alignement reste faible, la branche `LatentQ` séparée doit
être arrêtée : elle ne sera pas le bon chemin pour prouver l'apport du World
Model dans cette architecture.

Résultat : l'alignement augmente un peu, mais LatentQ devient une policy
quasi-exclusivement orientée déplacement et reste sous Q-only sur les 3 seeds.
Décision : arrêter `LatentQ` séparé. Le latent structuré reste utile pour le
World Model et les diagnostics, mais pas comme policy concurrente.

Prochaine direction :

```text
Q observationnel reste la policy principale.
Le World Model doit aider Q par planification contrôlée, diagnostic, ou
augmentation d'apprentissage, pas via une policy latente séparée.
```

Planification contrôlée implémentée :

```text
planner_used = WM uncertainty <= threshold
               and WM best-action margin >= threshold
```

Quand le gate refuse, l'agent revient automatiquement à Q-only. Cela permet de
tester le WM comme conseiller causal sous contrôle de confiance, sans créer une
policy concurrente.

Décision :

- garder le planner manuel comme diagnostic, pas comme architecture cible ;
- poursuivre par variantes de monde et diagnostic World Model ;
- éviter d'ajouter un écosystème complet avant validation de généralisation.

Résultat multi-seed avec `valueplanner` + gate confiance :

```text
uncertainty_threshold=0.65
margin_threshold=0.0

seed1: Q 114.7 -> WM gated 115.5, planner_used=100.0%
seed2: Q 102.8 -> WM gated 103.8, planner_used=37.3%
seed3: Q 107.4 -> WM gated 107.4, planner_used=12.0%

moyenne Q-only:       108.3
moyenne Q + WM gated: 108.9
```

Lecture : le gain reste faible, mais la planification contrôlée devient
neutre/positive sur les 3 seeds. C'est le premier comportement micro-fouloïde
où le WM peut aider sans créer une policy concurrente. La prochaine étape est
un sweep des seuils de confiance, pas un retour vers `LatentQ` séparé.

Sweep confiance multi-seed :

```text
Q-only moyen: 108.3

Réglage safe:
  p=0.15, terminal_value=1.0, uncertainty=0.60, margin=0.0
  moyenne=109.3
  lecture: le gate protège Q-only, planner presque éteint sur seeds 2/3.

Réglage actif:
  p=0.15, terminal_value=2.0, uncertainty=0.70, margin=0.01
  moyenne=109.8
  planner_used: seed1=87.2%, seed2=70.6%, seed3=41.9%
  lecture: meilleur signal pour la thèse WM, car le planner reste actif.
```

Conclusion : le WM planner apporte un petit signal positif quand il est utilisé
comme conseiller contrôlé. Le prochain verrou est la qualité/calibration du
World Model, pas une nouvelle policy latente concurrente.

Diagnostic calibration :

```text
Sur le checkpoint valueplanner seed1, l'incertitude WM est presque décorrélée
des erreurs réelles :

composite_rank corr ≈ +0.07
top20 uncertainty capture top20 errors ≈ 18.6%
```

Décision : ajouter une loss générique optionnelle
`world_model.uncertainty_loss_weight` pour entraîner la tête d'incertitude à
prédire l'erreur composite détachée du WM. Nouvelle config de test :

```text
configs/micro_fouloide_v0_rough_valueplanner_calibrated.yaml
```

Résultat intermédiaire : la calibration devient bien informative, mais la
supervision d'incertitude appliquée à travers le tronc partagé réduit encore le
niveau Q-only. La suite doit donc séparer les rôles :

```text
world_model.uncertainty_detach = true
```

Dans ce mode, l'erreur composite du WM entraîne la tête `uncertainty`, mais le
gradient de ce loss ne modifie pas le tronc dynamique/récompense. C'est le test
le plus propre pour l'objectif global : un World Model qui apprend ses
conséquences et sait estimer sa propre incertitude sans sacrifier la policy.

---

## 14. Statut

Cette spec est volontairement minimale. Elle doit rester le garde-fou pour
éviter de transformer le micro-fouloïde en nouveau sandbox trop guidé.

Décision actuelle :

```text
Ne pas ajouter encore l'écosystème complet.
D'abord valider un organisme minimal multi-drives.
La survie est le cas d'étude local, pas la définition générale de l'agent.
```

Implémenté :

- `seedmind/envs/micro_fouloide_world.py`
- `seedmind/agent/micro_fouloide_encoder.py`
- `configs/micro_fouloide_v0.yaml`
- `configs/micro_fouloide_v0_rough.yaml`
- `tests/test_micro_fouloide_world.py`
- `scripts/run_micro_fouloide.py`
- `scripts/evaluate_micro_fouloide.py`

Commandes de smoke :

```bash
python scripts/run_micro_fouloide.py \
  --config configs/micro_fouloide_v0.yaml \
  --episodes 5 \
  --device cpu \
  --inference-device cpu \
  --out-dir /tmp/seedmind_micro_smoke

python scripts/evaluate_micro_fouloide.py \
  --checkpoint /tmp/seedmind_micro_smoke/checkpoint_final.pt \
  --config configs/micro_fouloide_v0.yaml \
  --num-episodes 3 \
  --device cpu \
  --compare-planner
```

À faire ensuite :

- diagnostic complet du World Model sur les drives/events ;
- remplacer le planner manuel par une valeur apprise avant de chercher un gain
  WM robuste.
