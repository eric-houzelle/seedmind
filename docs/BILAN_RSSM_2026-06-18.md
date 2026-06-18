# Bilan recherche — World-model récurrent à mémoire (RSSM) + perception égocentrée

**Date : 2026-06-18 · Branche : `rssm-egocentric` (12 commits, poussée) · 270 tests verts**

Document de reprise. Tout le travail décrit ici est **isolé sur la branche
`rssm-egocentric`** ; `main` (et donc la prod) n'a jamais été touché. Issue de
suivi : **`seedmind-oc4`** (epic) / **`seedmind-oc4.1`** (point de reprise).

---

## 1. Objectif et cadrage

- **But long terme** : un world-model **générique et transférable** (robotique,
  agents logiciels, etc.). Le fouloïde est le **banc d'essai / la vitrine**, pas
  le produit.
- **Clarification stratégique importante (fin de session)** : le **fouloïde
  vivant ne dépend PAS** du world-model générique. Le **full-grid (en prod)
  vit déjà** (fourrage, survie, wellbeing positif). Le générique apporte le
  **passage à l'échelle** (mondes grands/infinis, transfert) — c'est un pari de
  recherche, pas le prérequis de la vitrine.
- **Règle d'archi** : le cœur appris (perception → WM → mémoire → policy) reste
  agnostique au domaine ; les spécificités (eau/nourriture/reward) vivent dans
  env + reward + config, jamais câblées dans le réseau.

## 2. Le diagnostic fondateur (résultat scientifique solide)

Point de départ : la perception actuelle est une **grille absolue aplatie**
(`encoder.py`, input_dim = size²×canaux) → ne scale pas, ne transfère pas. On
est passé à une **perception égocentrée convolutive** (fenêtre 11×11 centrée sur
l'agent, indépendante de la taille du monde).

**Contrôle discriminant (30k, même seed, tout identique sauf la perception) :**

| Agent | noop (spam INTERACT) | eau/bouffe | wellbeing | morts |
|---|---|---|---|---|
| Égocentré **sans mémoire** | ↑ 365→963 | → 0 | → 0 | 11 |
| Full-grid (voit tout) | ↓ 363→50 | stable, fourrage | positif | 5 |

**Conclusion prouvée** : l'effondrement de l'égocentré (spam INTERACT, arrêt du
fourrage) est **induit par la perception locale sans mémoire**, PAS par le reward
(le full-grid a le même reward/soft_death et fourrage). *« Voir l'eau parfois ≠
savoir y retourner. »* → **La mémoire est nécessaire** (pas optionnelle). C'est
ce qui motive tout le reste.

## 3. Ce qu'on a construit (machinerie RSSM, sur la branche)

Tout opt-in derrière des flags (`agent.observation.egocentric`,
`world_model.recurrent`, `agent.imagination_policy`) → défaut = comportement
full-grid d'origine, intact.

| Brique | Module | Rôle |
|---|---|---|
| Perception égocentrée | `agent/micro_fouloide_encoder.py` (`wrap_egocentric`, `egocentric_grid`) | recadrage fenêtre fixe, mode-agnostique |
| Encodeur conv gelé | `agent/encoder.py` (`ConvEncoder`) | latent stable + invariant à la taille |
| World-model récurrent | `agent/world_model.py` (`RecurrentWorldModel`) | GRU `h_t` = mémoire émergente |
| Q-net récurrent | `agent/q_network.py` (`recurrent_dim`) | reçoit `h_t` |
| Cycle de vie `h_t` | `agent/agent.py` (`reset_state`/`advance`) | porte `h_t`, reset à la mort |
| Buffer séquences | `memory/experience_buffer.py` (`sample_sequences`) | pour le BPTT |
| Entraînement WM BPTT | `training/recurrent.py` (`train_recurrent_world_model`) | BPTT sur séquences |
| DRQN récurrent | `training/recurrent.py` (`train_recurrent_dqn`) | TD avec `h_t` (+ burn-in R2D2) |
| Actor (imagination) | `agent/actor_critic.py` (`Actor`) | policy catégorielle sur `h` |
| Actor-critic en imagination | `training/imagination_actor_critic.py` | Dreamer-lite (λ-returns, REINFORCE, critic-cible EMA, norm. d'avantage) |
| Routage | `training/online.py`, `scripts/run_micro_fouloide.py`, `scripts/run_fouloide_online.py` | câblage build/train/live |

Configs : `configs/micro_fouloide_online_homeostatic_egocentric.yaml`,
`configs/micro_fouloide_online_homeostatic_rssm.yaml`.

## 4. Les expériences et leurs verdicts

Métriques = moyennes 2e moitié de run (≥30k), sauf mention. Barre à battre =
full-grid.

| Agent | wellbeing | eau | bouffe | noop | verdict |
|---|---|---|---|---|---|
| Égocentré seul | 0.000 | 0 | 0 | 963 | échec (spam) |
| Full-grid (**barre**) | **0.026** | 1.0 | 0 | 50 | fourrage, vit |
| RSSM + DRQN (60k) | 0.000 | 0 | 1.3 | 39 | échec : `td` **diverge** (0.01→0.34) |
| RSSM + DRQN + burn-in | 0.000 | 0 | ~0 | 1099 | burn-in **n'a pas corrigé** |
| Imagination (60k) | 0.008 | 3.4 | 2.9 | 2021 | **fourrage précoce** (well 0.103 @25k !) puis **critic explose** (1e9) |
| Imagination + critic-cible EMA + norm. | 0.000 | 0 | 0 | 0 | stable mais **exploration s'effondre** (ne fait rien) |

**Lecture de l'arc** :
1. égocentré sans mémoire → s'effondre → mémoire nécessaire ;
2. RSSM + DRQN model-free (h en feature annexe) → `td` diverge, ne fourrage pas →
   *ce n'est pas comme ça que Dreamer marche* ;
3. imagination-policy (actor-critic sur rollouts latents) → **fourrage précoce
   réel** (preuve que l'approche peut marcher) mais le **critic diverge** (j'avais
   omis les stabilisateurs Dreamer) ;
4. + **critic-cible EMA + normalisation d'avantage** → divergence réglée
   (critic_loss ~1e-4), mais l'**actor s'effondre sur « ne rien faire »** :
   `entropy_coef` trop bas → l'exploration meurt **avant** de découvrir le reward
   en imagination (chicken-and-egg).

## 5. Ce qu'on sait (acquis durables)

- **La mémoire est nécessaire** (contrôle discriminant). Solide.
- **Le WM récurrent apprend très bien** la dynamique (wm_loss → 0.02-0.04).
- **L'imagination-policy *peut* fourrager** (signal wellbeing 0.103, le meilleur
  de tous les récurrents) — l'approche est la bonne.
- **Le verrou restant** = l'équilibre **stabilité ↔ exploration** de l'actor-critic
  en imagination. Réglé la stabilité (critic-cible EMA), pas encore l'exploration.
- **Honnêteté méthodo** : on a réimplémenté Dreamer **à la pièce**, en heurtant
  chaque mode d'échec (spam → td → critic → exploration). Dreamer a une dizaine de
  stabilisateurs co-dépendants ; les redécouvrir un par un est lent et fragile.

## 6. Pistes de reprise (par ordre de priorité)

1. **Exploration** (le mode d'échec courant) : monter `entropy_coef` (~0.1-0.3),
   éventuellement schedule décroissant ; revoir la normalisation d'avantage (elle
   peut sur-écraser le signal précoce). Levier le plus direct.
2. **Implémenter DreamerV3 fidèlement et d'un coup** plutôt qu'en patches :
   normalisation des returns par percentile, KL-balancing du WM, free-bits,
   schedule d'entropie. C'est le chemin propre vers un actor-critic stable+explorant.
3. **Burn-in des états de départ d'imagination** (R2D2-style) si besoin.
4. **Régler horizon / lr / reward scale** une fois l'exploration saine.

## 7. Décision stratégique actée (2026-06-18)

- **Vitrine / court terme** : capitaliser sur le **fouloïde full-grid qui marche
  déjà** (prod intouchée, `main`). Pour du **spectaculaire rapide**, enrichir cet
  agent via les epics existants : **D (social + reproduction)** d'abord, puis
  **B (écosystème végétal)**, **C (matériaux/outils)**. Ces features accrochent
  les gens bien plus vite que de stabiliser Dreamer.
- **Recherche / long terme** : le world-model générique reste un vrai pari, **parké
  sur `rssm-egocentric`**, à reprendre **délibérément** (option 2 ci-dessus) quand
  le passage à l'échelle deviendra prioritaire. On ne merge dans `main` que si/quand
  l'agent récurrent **bat le baseline full-grid** (wellbeing + fourrage).

## 8. Pointeurs

- **Branche** : `rssm-egocentric` (poussée). `main` = prod, intact (`e253514`).
- **Issues bd** : `seedmind-oc4` (epic), `seedmind-oc4.1` (reprise = imagination-policy).
- **Runs** : `runs/ego_seed0`, `runs/bigmap_baseline_seed0`, `runs/rssm_long_seed0`
  (DRQN), `runs/rssm_burnin_long`, `runs/rssm_imag_long` (diverge),
  `runs/rssm_imag_fix_long` (stable mais inerte).
- **Tests** : `tests/test_egocentric_obs.py`, `test_recurrent_world_model.py`,
  `test_recurrent_training.py`, `test_agent_recurrent.py`, `test_qnet_recurrent.py`,
  `test_actor_critic.py`, `test_imagination_actor_critic.py`, `test_sequence_sampling.py`.
- **Reprendre** : `git checkout rssm-egocentric` ; `bd show seedmind-oc4.1`.
