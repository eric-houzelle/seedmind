# Objectif long terme — Prouver l'efficacité du World Model

Document de référence : définition de l'objectif final, critères de preuve, chemin depuis l'état actuel du projet, et lien avec une simulation autonome type **fouloïdes**.

> Voir aussi : [SPEC.md](../SPEC.md) (hypothèse centrale), [PROGRESSION.md](./PROGRESSION.md) (jalons mesurés et historique).

---

## 1. Objectif en une phrase

Démontrer, de façon **causale et mesurable**, que le **World Model** (WM) améliore le comportement d'un agent dans une **simulation autonome riche** (cible : écosystème type fouloïdes), par rapport au même agent **sans** planification / imagination fondée sur le WM — et non seulement « avoir un WM dans l'architecture ».

---

## 2. Thèse à prouver (et ce qu'elle n'est pas)

### Ce qu'on veut établir

```text
Même agent + même environnement (adapter)
  → avec WM (planning / imagination)  >  sans WM
  sur des métriques définies à l'avance
```

L'agent ne doit **jamais** encoder les règles du monde (voir SPEC §27). Le WM apprend `(latent, action) → (next_latent, reward, uncertainty)` à partir de l'expérience ; le **Planner** utilise ce WM pour simuler des suites d'actions avant de décider.

### Ce que ce n'est pas

| Non-objectif | Pourquoi |
|--------------|----------|
| Avoir un WM qui s'entraîne sans gagner en perf | Déjà le cas en sandbox v1 + planning (régression documentée) |
| Remplacer le RL par un LLM | Paradigme différent : prédiction de tokens vs conséquences d'actions |
| Construire d'abord toute la sim fouloïdes | Risque : 80 % simulateur, 0 % preuve WM ; comportement incompréhensible |
| « L'agent survit » sans A/B | Ne prouve pas le WM, seulement la policy |

### Hypothèse du projet (inchangée)

> Un agent doté d'un modèle du monde, d'une mémoire persistante, de curiosité et d'apprentissage continu peut développer des comportements adaptatifs **sans connaître les règles à l'avance**.

L'objectif fouloïdes est le **théâtre final** de validation de cette hypothèse, avec le WM comme **levier mesurable** de performance.

---

## 3. Cible : simulation fouloïdes autonome

### Intention

Une simulation **réelle** (moteur dédié, pas seulement une grille pédagogique) où :

- le monde **évolue de façon autonome** (écosystème, ressources, autres entités) ;
- l'agent subit des **contraintes concurrentes** (énergie, soif, froid, reproduction, niche — à préciser dans la spec fouloïdes) ;
- la **planification** a du sens : anticiper une famine, une saison, un conflit de ressource, etc.

Le repo mentionne cette cible en vision long terme (`PROGRESSION.md` §9) : *interface type fouloïdes avec écosystème autonome*, après monde physique / ComputerWorld.

### Principes d'intégration (architecture SeedMind)

Le moteur fouloïdes expose le même contrat que tout environnement :

```text
EnvironmentAdapter : reset / step / observe
```

- Le WM et la policy ne lisent **pas** l'état interne du simulateur.
- Seuls les **adaptateurs** changent entre sandbox, monde physique, fouloïdes (SPEC §19, scaling niveaux 5–9).

### Fouloïdes vs sandbox actuel

| Dimension | Sandbox (v0–v2) | Fouloïdes (cible) |
|-----------|-----------------|-------------------|
| Dynamique | Grille discrète, steps | Souvent continu, multi-entités, temps long |
| Observabilité | Rayon fixe, canaux CNN | Partielle, possiblement multi-capteurs |
| Objectifs | Survie, craft (chaînes causales) | Homéostase, arbitrage entre besoins |
| Monde | Régénération contrôlée | Écosystème **non stationnaire** |
| Rôle du WM | En cours de validation | **Doit** apporter un gain A/B mesurable |

Le sandbox n'est pas un détour : c'est la **rampe contrôlée** où l'on corrige le couplage policy ↔ WM avant de monter la complexité du simulateur.

---

## 4. État actuel (rappel)

| Jalon | Statut | Implication pour l'objectif WM |
|-------|--------|------------------------------|
| Sandbox v0/v1 — trained bat naive | ✅ (lifespan 2,89× → 4,24×) | La policy DQN apprend la causalité sans règles hardcodées |
| Planning WM sur v1 | ❌ (régression vs baseline) | Le WM **n'est pas encore** « efficace » en prod |
| Craft v2 | ⚠️ code prêt, validation longue à faire | Tâche intermédiaire idéale pour re-tester le WM |
| Dyna (imagination) | Désactivé en prod | Mélange latent / observation pollue le replay |

**Leçons déjà documentées** (`PROGRESSION.md` §3, niveau 1b) :

1. Le Q-Network apprend en espace `(grid, scalars)` ; les rêves du WM sont en espace **latent** — ne pas mélanger sans alignement.
2. Activer le planning quand le WM est encore imprécis ajoute du **bruit** aux Q-values.
3. Pistes : planning différé, Dyna latent, planification seulement en évaluation.

---

## 5. Définition opérationnelle : « WM efficace »

Fixer ces critères **avant** d'investir massivement dans le moteur fouloïdes.

### Critère 1 — Prédiction

Le WM bat un baseline naïf (persistance, moyenne) sur :

- erreur `next_latent` (one-step et multi-step, horizon configurable) ;
- erreur sur `reward` ;
- éventuellement calibration de `uncertainty`.

**Métriques suggérées :** `world_model_loss`, `prediction_error_mean` (déjà dans SPEC §25).

### Critère 2 — Planification (preuve principale)

Avec la **même** policy entraînée (même checkpoint) :

```text
perf(trained + planning_WM) > perf(trained seul)
```

Sur le **même** adapter et protocole d'évaluation.

**Métriques selon le monde :**

- Sandbox / craft : lifespan, métriques causales (`craft_tool`, `harvest_food_tool`, etc.)
- Fouloïdes : survie / homéostase, fréquence de crises évitables, score de **regret** (action prise vs meilleure action selon rollouts WM)

**Signaux qualitatifs forts :** comportements de **lookahead** — ex. consommer / stocker avant une famine **prédite** par le WM alors que la ressource n'est pas encore visible.

### Critère 3 — Sample efficiency (optionnel, renforce la thèse)

Moins d'interactions réelles pour atteindre la même performance grâce à **Dyna / imagination** — **uniquement** après alignement des espaces (latent vs observation). Sinon reproduction du crash `sandbox_v1_planning`.

### Variante de contrôle (plafond théorique)

**Variante C :** WM remplacé par un **oracle** (accès à une copie du simulateur pour rollouts parfaits). Si même l'oracle n'aide pas, le goulot n'est pas le WM mais la policy ou le protocole de planning.

---

## 6. Protocole expérimental A/B (obligatoire sur fouloïdes)

Toute démo ou publication « WM efficace » doit inclure :

| Variante | Description |
|----------|-------------|
| **A** | Policy apprise seule (planning désactivé, `planning_weight=0`) |
| **B** | Même entraînement + planning WM activé selon règles ci-dessous |
| **C** (optionnel) | Oracle simulateur pour rollouts |

**Règles pour B :**

- `planning_weight` (ou équivalent) ne monte qu'après un **seuil de qualité WM** (loss / erreur multi-step sous τ).
- Pas de Dyna en espace latent tant que le Q-network reste en espace observation (sauf refonte explicite « Dyna latent »).

**Interdit comme preuve :**

- Comparer un agent entraîné 5000 vies avec planning vs un naïf sans planning.
- Changer à la fois le monde, l'entraînement et le planning entre deux runs.

---

## 7. Chemin recommandé (ordre des étapes)

Aligné avec la philosophie SPEC §28 : *mécanisme → comportement mesurable → généralisation → complexité*.

```text
[A] Sandbox v1 validé                         ✅  trained bat naive
[B] WM utile sur tâche intermédiaire          ⬜  craft (v2) ou mini-monde physique
[C] Critères WM + planning différé            ⬜  preuve locale du WM
[D] Sandbox multi-pulsions (homéostase)        ⬜  fouloïdes-lite : arbitrage
[E] Adapter fouloïdes + sim autonome          ⬜  moteur + EnvironmentAdapter
[F] A/B WM sur fouloïdes                      ⬜  objectif final documenté ici
```

### Détail des étapes

**B — Tâche intermédiaire (priorité scientifique)**  
Valider craft (`sandbox_v2_*`) avec métriques causales + tentative planning/Dyna **après** alignement latent. Si le WM ne gagne pas ici, il ne le prouvera pas magiquement sur fouloïdes.

**C — Consolidation WM**  
Implémenter / configurer :

1. Planning différé (`planning_weight` ↑ seulement si loss WM < τ).
2. Option : Q-network + planning en **espace latent** cohérent.
3. Option : planification **seulement en eval** (entraînement DQN pur).

**D — Multi-pulsions** (`PROGRESSION.md` étape D)  
Jauges concurrentes (énergie, soif, froid), reward = survie globale. Proche des contraintes fouloïdes sans construire tout l'écosystème.

**E — Moteur fouloïdes**  
Spec dédiée (entités, physique, autonomie du monde, fréquence des resets). Exposer uniquement via `EnvironmentAdapter`.

**F — Preuve finale**  
Runs A/B documentés, checkpoints, `metrics.json`, courbes WM loss vs perf planning.

---

## 8. Évolutions techniques probables

Pour que l'objectif soit atteignable, le code actuel devra probablement évoluer (sans réécriture du cœur agent) :

| Sujet | État actuel | Besoin pour la preuve WM |
|-------|-------------|---------------------------|
| WM | MLP `latent → latent` + reward + uncertainty | Qualité mesurée ; possiblement tête de reconstruction d'observation |
| Planner | Random-shooting, horizon court | Peut nécessiter CEM / MPC sur fouloïdes |
| Policy vs WM | Q en observation, planner en latent | **Alignement** des espaces ou policy latente |
| Dyna | Désactivé (pollution replay) | Dyna latent ou WM → observation |
| Mémoire | Présente | Centrale en monde non stationnaire / peu de resets |
| Évaluation | `evaluate_sandbox.py`, métriques causales | Même schéma pour fouloïdes + protocole A/B |

Fichiers concernés aujourd'hui : `seedmind/agent/world_model.py`, `planner.py`, `agent.py`, `training/imagination.py`, `training/dqn.py`.

---

## 9. Risques à surveiller

1. **Complexité du simulateur** — Prioriser une spec fouloïdes **minimale** (MVP écosystème) avant la version complète.
2. **Non-stationnarité** — Écosystème autonome = distribution qui dérive ; apprentissage **continu** obligatoire.
3. **Métrique trop faible** — Lifespan seul insuffisant ; exiger métriques de lookahead et chaînes causales.
4. **Confusion LLM** — Les LLM ne remplacent pas cette preuve ; ils peuvent servir d'interface ou de tooling autour de la sim, pas de substitut au WM embodied.

---

## 10. Relation avec les autres documents

| Document | Rôle |
|----------|------|
| [SPEC.md](../SPEC.md) | Architecture, interdictions agent, scaling niveaux 5–9 |
| [PROGRESSION.md](./PROGRESSION.md) | Historique chiffré, commandes, roadmap sandbox A–D |
| **Ce fichier** | Objectif final WM + fouloïdes, critères de preuve, protocole A/B |

### Critère de validation récurrent (rappel)

À chaque étape intermédiaire, conserver **trained bat naive** ; pour l'étape finale, ajouter explicitement **trained + WM planning > trained seul** sur le même checkpoint.

---

## 11. Prochaines actions concrètes (checklist)

- [ ] Clore validation craft v2 (run long + `evaluate_sandbox` + métriques causales)
- [ ] Implémenter seuil WM avant activation de `planning_weight`
- [ ] Mesurer erreur WM multi-step sur sandbox v1 / craft
- [ ] Run A/B : même checkpoint, planning ON vs OFF en eval
- [ ] Rédiger spec fouloïdes MVP (entités, obs, actions, autonomie du monde)
- [ ] Esquisser `FouloidesAdapter` (interface seulement, puis moteur)
- [ ] Reprendre Dyna seulement après alignement latent/documenté

---

## 12. LLM vs agent SeedMind (rappel utile pour le cadrage)

| | Agent SeedMind + WM | LLM |
|---|---------------------|-----|
| Apprentissage | Interaction, reward, conséquences | Prédiction de tokens sur corpus |
| Sortie | Actions dans le monde | Texte |
| Preuve visée | A/B planning WM dans la sim | Benchmarks langage (hors scope de cet objectif) |

Les deux peuvent se combiner plus tard (LLM = interface, agent = exécution), mais **l'objectif de ce document** reste la preuve du WM dans la boucle sensori-moteur fermée.

---

*Dernière mise à jour : juin 2026*
