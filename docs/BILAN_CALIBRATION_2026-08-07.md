# Bilan — Calibration externe : le mur 10e décomposé

**Date : 2026-08-07 · Branche : `rssm-egocentric` · Statut : port innocenté de « ne sait pas apprendre », pathologie renommée bistabilité, expérience 4 prête**

Document de reprise. Fait suite à `BILAN_DREAMERV3_2026-06-29.md` (port DreamerV3
complet, mur de la policy de fourrage) et à l'arrêt du 6 juillet
(`seedmind-10e.10` : « on s'arrête là pour l'instant »).

Campagne : 3 jours d'A10 (4-7 août 2026), 3 runs de 500k steps, 1 run de
référence externe. Données brutes : `runs_a10/`. Protocole et verdicts détaillés :
`CALIBRATION_DREAMERV3_REF.md` §7-9.

---

## 1. Pourquoi une calibration externe

Au 6 juillet, ~17 hypothèses avaient été testées et réfutées sans que la policy
fourrage. Le problème n'était pas le manque d'idées : **chaque échec avait quatre
explications compatibles** (bug du port, env mal réglé, tâche trop dure, régime de
compute insuffisant), donc aucune expérience ne pouvait trancher. C'est ce qui
épuise, pas la difficulté.

La Phase 0 du `DREAMERV3_PORT_PLAN` (calibrer contre une implémentation de
référence) avait été sautée : le port avait été fait de mémoire. **Cette campagne
la rattrape.**

## 2. Ce qui a été construit

| Brique | Fichier | Rôle |
|---|---|---|
| Wrapper Gymnasium information-fidèle | `scripts/calibration/gym_micro_fouloide.py` | Même monde, mêmes channels égocentrés 11×11, même `reward_learning`, mêmes actions |
| Adaptateur dreamerv3-torch | `scripts/calibration/dreamerv3_torch/fouloide.py` (+ bloc yaml) | Traduit vers leur protocole (vieux gym 4-tuple, `is_first/is_last/is_terminal`) |
| Baseline aléatoire | `scripts/calibration/random_baseline.py` | Le seuil que tout doit battre : **6,25 fourrages/1000 pas** (déterministe, reproductible sans GPU) |
| `online.buffer_capacity` | `seedmind/training/online.py` | Taille du replay configurable (défaut 100k inchangé) |
| `online.episode_limit` | `scripts/run_fouloide_online.py` | Reset périodique du monde (exp 4, cf. §6) |
| Fix `reward_key` | `seedmind/training/train.py` | Chemin WM feed-forward cassé depuis le 30 juin (4 tests rouges) |

Tests : 304 verts. 4 patchs d'intégration côté dreamerv3-torch documentés
(`CALIBRATION §3bis`) — leur repo n'a que des envs à pixels, un env vectoriel
fait sauter ses hypothèses `image` une par une.

## 3. Les quatre résultats

### 3.1 Le monde est apprenable (référence, 514k steps)

La référence DreamerV3 monte de 52 à ~135 de return (aléatoire = 76), pente
encore positive à 500k. **L'hypothèse « la tâche ne contient pas de signal » est
réfutée.**

### 3.2 Mais la référence ne résout pas la survie non plus

Survie plate à ~20 % du début à la fin, épisodes ~1000 pas. Elle apprend à
*mieux vivre*, pas à *ne pas mourir*. **Le mur de la survie n'était pas une
faiblesse de notre port** — l'env en fenêtre égocentrée 11×11 est objectivement
dur.

### 3.3 Tous les runs historiques étaient dans la zone morte

Fenêtre 0-50k de la référence : **au niveau aléatoire**. La séparation n'émerge
qu'après ~150-200k steps. Or **tous les runs de juin-juillet faisaient 50k**.
Les ~17 hypothèses réfutées l'ont été dans un régime où même une implémentation
correcte par construction n'apprend rien. Elles ne sont pas fausses : elles sont
**ininterprétables**. L'issue `seedmind-10e.5` avait raison.

### 3.4 La pathologie n'est pas la lenteur, c'est la bistabilité

Trois runs à nous (aléatoire = 6,25 fourrages/1000) :

| run | global | 2ᵉ moitié | tendance (r) | fenêtres idle |
|---|---|---|---|---|
| exp 2, buffer 100k, seed 0 | 6,7 | 8,2 | +0,68 | 4/49 |
| exp 3, buffer 500k, seed 0 | 6,2 | 6,7 | +0,19 | 4/34 (OOM à 350k) |
| exp 3, buffer 500k, seed 1 | **3,6** | 2,9 | **−0,30** | **21/49** |

- **Suspect « replay trop court » : RÉFUTÉ.** Le buffer 500k fait *moins* bien
  que le 100k, pour 22,6 Go de RSS (seed 0 tuée par l'OOM-killer).
- **Mode d'échec dominant : le bassin idle** — agent immortel, jauges à zéro,
  fourrage ~1/1000. Seed 1 y passe **43 % du temps**. Intermittent, non
  absorbant (elle en sort vers 350k puis y retombe). C'est le « bassin idle »
  de `rssm-idle-basin-2026-06-22` et le « critfix immortel » de `10e.10`,
  désormais **mesuré et quantifié**.

## 4. Erreur de lecture à ne pas refaire

Le run exp 2 (une graine) montrait une montée monotone 3,7 → 10,3/1000 et a été
lu comme « le port est validé, le mur était le budget ». **Deux graines de plus
ont réfuté cette lecture** : la trajectoire ne se reproduit pas, et aucun run ne
bat significativement l'aléatoire en moyenne globale.

**Règle adoptée : jamais de conclusion sur une seule graine.** Une trajectoire
unique raconte n'importe quelle histoire. Les critères de l'exp 4 exigent
désormais le succès **sur les deux graines**.

## 5. État du port au 7 août

| Question | Réponse | Preuve |
|---|---|---|
| La machinerie tourne-t-elle ? | **Oui** | WM apprend (loss 2,46→1,70), imagination et actor-critic actifs, 500k steps sans divergence |
| Le port sait-il apprendre ? | **Partiellement** | Progression réelle sur 2 runs sur 3, mais non reproductible |
| Est-il stable comme la référence ? | **Non** | La référence tient son niveau ; nous oscillons et tombons dans l'idle |
| Le mur était-il un bug ? | **Non prouvé** | Aucune divergence structurelle identifiée à ce jour, hors reset (§6) |
| Le mur était-il le budget ? | **Nécessaire mais pas suffisant** | Rien avant 150k, mais 500k ne suffit pas non plus |

## 6. Ce qu'on doit faire maintenant

### Expérience 4 — reset périodique (PRÊTE, code poussé)

**Hypothèse.** La référence remet le monde à zéro toutes les 2000 pas
(`wrappers.TimeLimit`) ; notre boucle online ne resette **que sur mort**. Un
agent idle-immortel n'est donc jamais interrompu chez nous, alors que chez la
référence cet état est structurellement intenable — monde neuf, ré-exploration
forcée, replay réalimenté en trajectoires variées. C'est **la seule différence
structurelle** qui explique à la fois la bistabilité, son intermittence, et
l'immunité de la référence.

**Implémentation.** `online.episode_limit` (opt-in, 0 = comportement
historique), avec deux garanties testées (`tests/test_episode_limit.py`) :
troncature ≠ mort (`done` reste False, sinon le `continue_head` apprend une mort
fantôme ; compteur `truncations` distinct) et troncature = nouvel `episode_id`
(sinon les séquences RSSM mélangent deux mondes).

```bash
cd /var/projects/seedmind && git pull
for s in 0 1; do
  nohup python -u -m scripts.run_fouloide_online \
    --config configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix_ep2000.yaml \
    --steps 500000 --device cuda --seed $s --checkpoint-every 50000 \
    > run_ep2000_s$s.log 2>&1 &
done
```

**Critères (fixés avant le run, succès requis sur les DEUX graines)** :
fenêtres idle ≈ 0/49, fourrage > 8/1000 en 2ᵉ moitié, tendance r > 0.

### Si l'exp 4 échoue — suspects suivants, dans l'ordre

1. **Régulation d'entropie** : plancher 0,012 schedulé chez nous vs coefficient
   fixe 3e-4 × entropie chez la référence. Un plancher force une exploration
   résiduelle mais peut aussi empêcher l'engagement.
2. **Gradient d'actor** : REINFORCE + baseline avec imagination sous `no_grad`
   (`imagination_actor_critic.py:332`) vs backprop-through-dynamics. Écart connu
   et jamais testé (recommandation §6.2 du bilan de juin).

### Chantiers indépendants (rouverts par cette campagne)

- **La survie est un problème d'environnement, pas de code** (§3.2) : même la
  référence plafonne à 20 %. Pistes : shaping, observabilité (fenêtre 11×11),
  ou accepter que la survie parfaite n'est pas l'objectif.
- **Régime standard** : tout run de validation fait désormais **≥ 300k steps et
  ≥ 2 graines**. En dessous, le résultat n'est pas publiable en interne.
- **Ne pas relancer** : buffer 500k (réfuté, 22 Go), runs de 50k (zone morte).

## 7. Hygiène de session A10

- `run_fouloide_online` a `--device` qui **défaute à `cpu`** : toujours passer
  `--device cuda` (un run entier a été perdu là-dessus).
- Sous `nohup`, stdout est bufferisé : lancer avec `python -u`.
- Deux runs en parallèle tiennent (GPU à 94 %, ~8,5 steps/s chacun) — même prix
  qu'un seul, puisque l'instance est facturée à l'heure.
- Buffer 500k × 2 runs = OOM sur 40 Go. Buffer 100k × 2 : large.
