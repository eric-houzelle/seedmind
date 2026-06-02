# Spécification complète — Projet SeedMind

## 1. Objectif général

Créer un agent évolutif capable d’apprendre dans des mondes fictifs générés procéduralement, puis de pouvoir évoluer progressivement vers des environnements plus complexes et plus réalistes.

Le projet doit démarrer petit, mais être conçu dès le départ pour grandir.

L’objectif n’est pas de créer immédiatement une AGI, mais de construire une base expérimentale pour tester cette hypothèse :

> Un agent doté d’un modèle du monde, d’une mémoire persistante, de curiosité, de micro-objectifs et d’apprentissage continu peut-il développer progressivement des comportements adaptatifs dans des mondes nouveaux ?

---

## 2. Principe central

L’agent ne doit pas dépendre directement d’un environnement spécifique.

Il doit apprendre à partir du cycle universel :

```text
observation → action → conséquence → mémoire → amélioration
```

Tous les environnements doivent donc exposer la même interface.

---

## 3. Architecture globale

```text
SeedMind/
├── Agent
│   ├── Encoder
│   ├── World Model
│   ├── Policy
│   ├── Planner
│   ├── Goal Generator
│   ├── Curiosity Module
│   └── Memory System
│
├── Environment Adapters
│   ├── GridWorld
│   ├── ProceduralWorld
│   ├── PhysicsWorld
│   ├── ComputerWorld
│   └── FutureWorlds
│
├── Experience Buffer
├── Training System
├── Evaluation System
└── Visualization / Logs
```

---

## 4. Objectif V1

Créer une première version dans un monde 2D simple.

L’agent doit pouvoir :

* se déplacer ;
* explorer ;
* interagir avec des objets ;
* découvrir des règles simples ;
* mémoriser ses expériences ;
* apprendre à prédire les conséquences de ses actions ;
* réutiliser une connaissance dans une nouvelle carte.

Premier scénario cible :

```text
L’agent découvre qu’un objet permet d’en débloquer un autre.
Exemple : clé → porte.
```

---

## 5. Environnement V1 : GridWorld

### Grille

Taille initiale :

```text
10 x 10
```

Puis extensible vers :

```text
20 x 20
50 x 50
cartes procédurales
```

### Entités

```text
EMPTY = vide
WALL = mur
AGENT = agent
KEY = clé
DOOR_CLOSED = porte fermée
DOOR_OPEN = porte ouverte
REWARD = récompense
DANGER = danger
UNKNOWN_OBJECT = objet inconnu
```

### Actions

```text
MOVE_UP
MOVE_DOWN
MOVE_LEFT
MOVE_RIGHT
INTERACT
WAIT
```

### Règles V1

* L’agent ne traverse pas les murs.
* Une clé peut ouvrir une porte.
* Un danger donne une pénalité.
* Une récompense donne un score positif.
* Certains objets inconnus doivent être testés.

---

## 6. Générateur de mondes

Le projet doit inclure un générateur procédural.

Chaque épisode peut générer :

* une nouvelle carte ;
* de nouvelles positions d’objets ;
* de nouvelles règles ;
* de nouveaux liens entre objets.

Exemple de règle générée :

```yaml
rule:
  item: red_key
  target: red_door
  action: interact
  effect: unlock
```

Autre exemple :

```yaml
rule:
  item: crystal
  target: gate
  action: use
  effect: open
```

L’objectif est que l’agent n’apprenne pas seulement une carte, mais apprenne à découvrir les règles d’un monde.

---

## 7. Interface universelle des environnements

Tous les mondes doivent implémenter cette interface :

```python
class EnvironmentAdapter:
    def reset(self):
        """
        Réinitialise le monde.
        Retourne la première observation.
        """
        pass

    def observe(self):
        """
        Retourne l’observation actuelle de l’agent.
        """
        pass

    def available_actions(self):
        """
        Retourne les actions possibles dans l’état actuel.
        """
        pass

    def step(self, action):
        """
        Applique une action.
        Retourne :
        - next_observation
        - reward
        - done
        - info
        """
        pass

    def describe_transition(self):
        """
        Retourne une description optionnelle de la dernière transition.
        Utile pour debug, logs et analyse.
        """
        pass
```

L’agent ne doit jamais appeler directement des fonctions internes du monde.

---

## 8. Format universel d’expérience

Chaque transition doit être stockée dans un format commun :

```json
{
  "episode_id": "episode_000001",
  "world_id": "gridworld_v1",
  "step": 42,
  "observation": "...",
  "action": "MOVE_RIGHT",
  "next_observation": "...",
  "reward_external": 0.0,
  "reward_intrinsic": 0.31,
  "goal": "explore_unknown_object",
  "prediction_error": 0.24,
  "memory_used": [],
  "done": false,
  "timestamp": 123456789
}
```

Ce format doit être compatible avec plusieurs mondes futurs.

---

## 9. Experience Buffer

Créer un système qui stocke toutes les expériences de l’agent.

Fonctions nécessaires :

```python
add(experience)
sample(batch_size)
sample_recent(batch_size)
sample_high_error(batch_size)
sample_high_reward(batch_size)
save(path)
load(path)
```

Le buffer servira à entraîner :

* le World Model ;
* la Policy ;
* le module de curiosité ;
* éventuellement des modules futurs.

---

## 10. Mémoire persistante

L’agent doit avoir une mémoire à long terme séparée du buffer d’entraînement.

La mémoire stocke les expériences importantes.

Structure recommandée :

```json
{
  "memory_id": "mem_00001",
  "world_type": "gridworld",
  "state_embedding": "...",
  "summary": "interaction avec porte fermée en possession d’une clé",
  "action": "INTERACT",
  "result": "porte ouverte",
  "utility": 0.91,
  "novelty": 0.34,
  "confidence": 0.82,
  "uses": 5
}
```

Fonctions nécessaires :

```python
store(memory_item)
retrieve(query_embedding, top_k=5)
update_confidence(memory_id, delta)
decay_old_memories()
save(path)
load(path)
```

V1 peut utiliser une recherche vectorielle simple avec NumPy.

Plus tard, FAISS ou Chroma pourront être ajoutés.

---

## 11. Encoder

L’Encoder transforme une observation brute en représentation latente.

V1 :

```text
observation grille → embedding → vecteur latent
```

Configuration initiale :

```yaml
latent_dim: 128
```

L’Encoder doit pouvoir être remplacé plus tard pour traiter :

* image ;
* texte ;
* écran d’ordinateur ;
* données multimodales.

---

## 12. World Model

Le World Model apprend :

```text
état latent actuel + action → état latent suivant
```

Entrées :

```text
latent_state
action
optional_memory_context
```

Sorties :

```text
predicted_next_latent_state
predicted_reward
uncertainty
```

Loss V1 :

```text
loss = prediction_state_loss + reward_prediction_loss
```

Le World Model doit pouvoir être entraîné sur les expériences collectées par l’agent.

---

## 13. Curiosity Module

Le module de curiosité donne une récompense interne quand l’agent rencontre quelque chose qu’il comprend mal.

Formule V1 :

```text
intrinsic_reward = prediction_error
```

Mais prévoir une limite pour éviter que l’agent recherche uniquement le chaos :

```text
intrinsic_reward = min(prediction_error, max_curiosity_reward)
```

Plus tard :

```text
curiosité utile = surprise - bruit estimé
```

---

## 14. Goal Generator

L’agent doit pouvoir se fixer des micro-objectifs.

Objectifs V1 possibles :

```text
explore_unknown_area
interact_with_unknown_object
reach_visible_reward
avoid_known_danger
test_uncertain_rule
reuse_successful_memory
```

Le Goal Generator choisit un objectif selon :

```text
score = novelty + expected_utility + uncertainty
```

V1 peut être heuristique.

Plus tard, il pourra devenir un modèle entraîné.

---

## 15. Policy

La Policy choisit l’action à effectuer.

Entrées :

```text
latent_state
current_goal
retrieved_memories
available_actions
```

Sortie :

```text
action
```

V1 :

* exploration aléatoire contrôlée ;
* heuristiques simples ;
* epsilon-greedy.

V2 :

* petit réseau neuronal ;
* entraînement RL ;
* imitation des meilleures trajectoires.

---

## 16. Planner

Le Planner utilise le World Model pour simuler plusieurs actions possibles.

V1 optionnelle :

```text
tester N actions possibles
prédire les futurs
choisir l’action avec le meilleur score
```

Score :

```text
score = predicted_reward + intrinsic_reward + goal_progress
```

Au départ, le Planner peut être très simple.

---

## 17. Boucle principale

Implémenter cette boucle :

```python
for episode in range(num_episodes):
    observation = env.reset()

    for step in range(max_steps):
        latent_state = encoder.encode(observation)

        memories = memory.retrieve(latent_state, top_k=5)

        goal = goal_generator.choose(
            latent_state=latent_state,
            memories=memories
        )

        action = policy.choose(
            latent_state=latent_state,
            goal=goal,
            memories=memories,
            available_actions=env.available_actions()
        )

        next_observation, reward_external, done, info = env.step(action)

        next_latent_state = encoder.encode(next_observation)

        prediction = world_model.predict(latent_state, action)

        prediction_error = compute_prediction_error(
            prediction,
            next_latent_state
        )

        reward_intrinsic = curiosity.compute(prediction_error)

        experience = {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
            "reward_external": reward_external,
            "reward_intrinsic": reward_intrinsic,
            "goal": goal,
            "prediction_error": prediction_error,
            "done": done
        }

        experience_buffer.add(experience)

        memory.store_if_important(experience)

        observation = next_observation

        if done:
            break

    train_world_model()
    train_policy_if_enabled()
    save_logs()
```

---

## 18. Apprentissage continu

Le système doit être conçu pour ne pas repartir de zéro.

À sauvegarder régulièrement :

```text
world_model weights
policy weights
experience buffer
persistent memory
training metrics
environment configs
```

Prévoir :

```python
save_checkpoint(path)
load_checkpoint(path)
```

---

## 19. Scaling prévu

Le projet doit évoluer par étapes.

### Niveau 1

```text
GridWorld simple
```

### Niveau 2

```text
GridWorld procédural
```

### Niveau 3

```text
règles variables
```

### Niveau 4

```text
monde partiellement observable
```

### Niveau 5

```text
monde physique simplifié
```

### Niveau 6

```text
monde constructible
```

### Niveau 7

```text
interface ordinateur sandboxée
```

### Niveau 8

```text
web sandboxé
```

### Niveau 9

```text
capteurs / robotique
```

L’architecture ne doit pas être réécrite à chaque niveau.

Seuls les adaptateurs d’environnement doivent changer.

---

## 20. Structure du repository

Créer cette structure :

```text
seedmind/
  README.md
  requirements.txt
  configs/
    v1_gridworld.yaml
  seedmind/
    __init__.py

    envs/
      __init__.py
      base.py
      gridworld.py
      procedural_gridworld.py

    agent/
      __init__.py
      agent.py
      encoder.py
      world_model.py
      policy.py
      planner.py
      goal_generator.py
      curiosity.py

    memory/
      __init__.py
      experience_buffer.py
      persistent_memory.py

    training/
      __init__.py
      train.py
      losses.py
      checkpointing.py

    evaluation/
      __init__.py
      metrics.py
      scenarios.py

    visualization/
      __init__.py
      render_grid.py
      plot_metrics.py

  scripts/
    run_v1.py
    train_world_model.py
    evaluate_agent.py

  tests/
    test_env_gridworld.py
    test_experience_buffer.py
    test_memory.py
    test_agent_loop.py
```

---

## 21. Stack technique

Utiliser :

```text
Python 3.11+
PyTorch
NumPy
PyYAML
Matplotlib
pytest
```

Optionnel plus tard :

```text
FAISS
Chroma
Gymnasium
TensorBoard
Weights & Biases
```

---

## 22. Configuration V1

Créer un fichier :

```yaml
project:
  name: SeedMind

env:
  type: gridworld
  size: 10
  max_steps: 100
  procedural: true
  partial_observation: false

agent:
  latent_dim: 128
  memory_top_k: 5

world_model:
  enabled: true
  hidden_dim: 256
  num_layers: 2
  learning_rate: 0.0003
  batch_size: 64

curiosity:
  enabled: true
  weight: 0.1
  max_reward: 1.0

policy:
  type: epsilon_greedy
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay_steps: 10000

training:
  episodes: 10000
  train_every: 10
  checkpoint_every: 500
```

---

## 23. Critères d’acceptation V1

Le projet V1 est terminé si :

1. Un environnement GridWorld fonctionne.
2. L’agent peut effectuer des actions.
3. Les transitions sont stockées dans l’Experience Buffer.
4. Le World Model s’entraîne à prédire le prochain état.
5. La loss du World Model diminue avec le temps.
6. Le module de curiosité donne une récompense basée sur l’erreur de prédiction.
7. La mémoire persistante stocke les expériences importantes.
8. L’agent peut résoudre au moins un scénario simple clé → porte.
9. Les expériences, modèles et métriques sont sauvegardés.
10. Le code est modulaire et extensible.

---

## 24. Critères d’acceptation V2

La V2 est validée si :

1. Les cartes sont générées procéduralement.
2. Les règles peuvent changer entre épisodes.
3. L’agent améliore ses performances sur des mondes jamais vus.
4. Il réutilise des souvenirs utiles.
5. Il explore moins aléatoirement avec le temps.
6. Il transfère une règle simple vers une nouvelle configuration.

Exemple :

```text
Apprendre : clé rouge → porte rouge
Tester : clé bleue → porte bleue
Succès : l’agent comprend plus vite la seconde règle.
```

---

## 25. Métriques à logger

Logger à chaque épisode :

```text
episode_reward_external
episode_reward_intrinsic
steps_survived
success
prediction_error_mean
world_model_loss
memory_items_count
goal_distribution
exploration_rate
repeated_mistakes
```

Visualisations minimales :

```text
courbe reward externe
courbe reward intrinsèque
courbe prediction error
courbe success rate
nombre de souvenirs
```

---

## 26. Tests obligatoires

Créer des tests pour :

```text
reset environnement
déplacement agent
collision mur
interaction clé / porte
stockage expérience
sample experience buffer
stockage mémoire
récupération mémoire
forward world model
boucle agent complète sur 1 épisode
sauvegarde checkpoint
chargement checkpoint
```

---

## 27. Règles de conception importantes

Ne pas coder de solution dans l’agent.

Interdit dans l’agent :

```python
if object == KEY:
    go_to_door()
```

Autorisé dans l’environnement :

```python
if agent_has_key and action == INTERACT:
    door.open()
```

L’agent doit découvrir les règles par expérience.

---

## 28. Philosophie du projet

Ne pas chercher à faire gros au départ.

Ordre correct :

```text
mécanisme intéressant
→ comportement mesurable
→ généralisation
→ complexité environnementale
→ modèle plus grand
```

Ordre à éviter :

```text
gros modèle
→ environnement flou
→ comportement incompréhensible
```

---

## 29. Première milestone concrète

Créer une démo exécutable :

```bash
python scripts/run_v1.py
```

La démo doit montrer :

1. une grille ;
2. un agent qui bouge ;
3. des objets ;
4. des expériences collectées ;
5. le World Model qui s’entraîne ;
6. une courbe d’erreur de prédiction ;
7. une sauvegarde de checkpoint.

---

## 30. Deuxième milestone concrète

Créer une démo :

```bash
python scripts/evaluate_agent.py
```

Elle doit comparer :

```text
agent naïf
vs
agent entraîné
```

Sur plusieurs cartes générées.

Mesurer :

```text
temps moyen pour réussir
taux de réussite
erreur de prédiction
réutilisation mémoire
```

---

## 31. Résultat attendu

À la fin de la première version, on doit avoir une plateforme de recherche minimale, propre et extensible.

Elle doit permettre de tester progressivement :

* mémoire ;
* curiosité ;
* world model ;
* génération de règles ;
* adaptation ;
* transfert entre mondes ;
* montée vers environnements plus réalistes.

Nom du projet :

```text
SeedMind
```
