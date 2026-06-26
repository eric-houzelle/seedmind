# Plan d'intégration — DreamerV3 fidèle dans SeedMind (port PyTorch)

**But :** remplacer notre actor-critic-en-imagination + WM bricolés par les composants
**fidèles** de DreamerV3, **dans notre stack PyTorch** (env `MicroFouloideWorld`, boucle
online, démo live, déploiement conservés). Objectif : un agent **générique** qui maximise
sa cible (wellbeing) sans tuning par-monde → la voie vers robotique/transfert.

Pourquoi ce port (résumé du diagnostic, branche `rssm-egocentric`) : on a prouvé que
l'architecture est la bonne (c'est Dreamer) et **localisé** les manques par des probes —
le WM sait que fourrager paie mais l'actor l'évite, parce que l'**assignation de crédit**
(critic + λ-return) classe mal les actions. On a réparé des bouts (return_range, twohot
value) mais une **réimplémentation pièce-par-pièce est lente et fragile** (≈10 stabilisateurs
co-dépendants). On porte donc l'algo **complet et fidèle**, guidé par une référence.

## Références
- **Primaire (à jour, ~5× plus rapide)** : `r2dreamer` (baseline DreamerV3 PyTorch maintenue, ICLR 2026).
- **Lecture/structure** : `NM512/dreamerv3-torch` (lisible mais antérieure aux dernières màj — bon pour la structure modulaire : `models.py` RSSM/heads, `networks.py` actor/critic, `tools.py` symlog/twohot/percentile, `dreamer.py` boucle).
- Officiel (JAX, NE PAS porter tel quel) : `danijar/dreamerv3` — référence algébrique.

## Analyse d'écart (ce qu'on a vs ce qu'il faut)

| Composant | Nous (`seedmind/`) | DreamerV3 | Écart |
|---|---|---|---|
| Encodeur | `ConvEncoder` égocentré (size-invariant) | CNN | **garder** (c'est notre atout démo) |
| Latent WM | `RecurrentWorldModel` : GRU `h` **déterministe seul** | **RSSM** : `h` (GRU déter) **+ `z` stochastique** (catégoriel 32×32) + prior/posterior + **KL balancé + free-bits** | **GROS — le cœur manquant** |
| Reward head | scalaire | **twohot** (bins symlog) | moyen |
| Continue/done | via env seulement | prédicteur continue (bernoulli) | moyen |
| Critic | `TwoHotCritic` | twohot | **FAIT ✓** |
| Actor | catégoriel sur `h` | sur `(h,z)` + **normalisation returns par percentile** (EMA 5–95) + échelle d'entropie | moyen |
| Imagination | rollout sur `h` (GRU) | rollout sur `(h,z)` (échantillonne `z` du prior) | lié au RSSM |
| Inputs | canaux bruts | **symlog** des inputs + reward | mineur |
| Replay / boucle online | `sample_sequences`, `OnlineLearner` | replay + train loop | **garder** |
| Démo live / déploiement | `demo_fouloides_front`, compose, bigmap | — | **garder, intouché** |

**Le cœur du port = le RSSM stochastique (z + KL).** Notre `RecurrentWorldModel` n'est que
la moitié déterministe ; le latent stochastique + KL est ce qui rend l'imagination robuste
(et ce qui manque pour que les avantages soient fiables).

## Plan par phases (tout derrière flags ; défaut = comportement actuel → démo intacte)

### Phase 0 — Réf & garde-fous (0 changement de comportement)
- Cloner `r2dreamer` (+ `NM512` en cross-check) en lecture seule, pin d'un commit, dans un dossier hors-build (`refs/` gitignored). Ne pas l'exécuter ; le lire.
- Geler une suite de tests « run d'or » du comportement actuel (forager reward_learning) pour non-régression.

### Phase 1 — RSSM stochastique (le gros morceau)
- Nouveau module `seedmind/agent/rssm.py` : `h` (GRU déter) + `z` catégoriel (32×32), `prior(h)`, `posterior(h, embed)`, KL-balancing (0.8/0.2) + free-bits (1 nat).
- `imagine_step` échantillonne `z` du prior ; `observe_step` calcule le posterior.
- Câbler dans `world_model.py` derrière `world_model.rssm_stochastic: true` (le déterministe reste en fallback + tests).
- Entraînement : `recurrent.py` ajoute la **KL loss** (prior‖posterior) au state-loss.
- `agent.py` : porter `(h,z)` dans le cycle de vie (reset/advance) au lieu de `h` seul.
- **Valider** : `wm_loss` (recon + KL) descend, imagination génère des `z` cohérents.

### Phase 2 — Reward + continue en twohot
- Reward head → twohot (réutiliser le head de `TwoHotCritic`). Prédicteur `continue` (bernoulli) pour `done`.
- `recurrent.py` régresse les deux (CE / BCE).

### Phase 3 — Actor sur (h,z) + normalisation des returns par percentile
- Actor input = `[h, z]`.
- Remplacer `return_range` par la **normalisation DreamerV3** : `returns / max(1, Per95−Per5)` avec EMA des percentiles (dans `imagination_actor_critic.py`).
- Échelle d'entropie + (option) schedule, façon DreamerV3.

### Phase 4 — symlog inputs, free-bits, finition + validation
- symlog sur les inputs d'obs/reward.
- **Juges** (les probes qui ont diagnostiqué) : re-run `probe_advantage` (INTERACT redevient-il le max d'avantage ? avantages centrés ?), `eval_death_cause` (% critique ↓), et un run 60k comparé aux baselines (forager 0.067 / full-grid morts ~5).

## Ordre & raison
RSSM d'abord (fondation : actor + imagination ont besoin de `(h,z)`), puis reward/continue
twohot, puis actor+normalisation returns, puis finition. Chaque phase derrière un flag,
testée, sans toucher la démo déployée tant que le nouvel agent ne bat pas les baselines.

## Effort
Phase 1 (RSSM) = l'essentiel (plusieurs jours). Le reste incrémental. On ne part PAS de
zéro : encodeur, env, reward, boucle online, démo, déploiement, twohot value, return_range
existent déjà et marchent.

## Critère de bascule (prod)
On ne flippe la config live (`rssm_bigmap`) sur le nouvel agent QUE s'il bat les baselines
sur **wellbeing ET survie** (le full-grid : wellbeing modeste mais ~5 morts ; viser % critique
nettement < 85% et morts ≪ 54). Sinon, le forager actuel reste en démo.
