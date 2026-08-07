# Calibration externe — référence DreamerV3 sur le micro-fouloïde

Date : 4 août 2026
Objet : Phase 0 du [DREAMERV3_PORT_PLAN](./DREAMERV3_PORT_PLAN.md), rattrapée.
Contexte : le mur `seedmind-10e` (la policy ne fourrage pas malgré un WM sain,
~17 hypothèses réfutées, leviers CPU épuisés — voir
[CHANTIER_WM_ENCODER_TRAINABLE](./CHANTIER_WM_ENCODER_TRAINABLE.md) §10-11).

## 1. Le principe

Tant qu'on ne compare qu'à nous-mêmes, chaque échec a quatre explications
compatibles : bug du port, env mal réglé, tâche dure, régime de compute
insuffisant. L'expérience de calibration les sépare :

| Résultat de la référence | Conclusion | Action |
|---|---|---|
| Elle fourrage (return ↑, survie ↑) | **notre port a un bug** | diff composant par composant contre la référence (borné) |
| Elle échoue pareil | **le code est innocenté** — env/régime en cause | travailler la reward / l'observabilité / le régime 10e.5, plus jamais le port |

Dans les deux cas, l'ambiguïté disparaît. Il n'y a pas d'issue perdante.

## 2. L'instrument

`scripts/calibration/gym_micro_fouloide.py` — wrapper Gymnasium
**information-fidèle** au pipeline seedmind (tests :
`tests/test_gym_micro_fouloide.py`) :

- même monde (`build_env` + la config de run), même graine ;
- même observation : les `(channels, scalars)` du ConvEncoder, crop égocentré
  11×11 inclus (`obs_mode="vector"` les aplatit pour un encodeur MLP) ;
- même récompense : `reward_learning` (la cible du WM, drive + fourrage),
  calculée comme dans `OnlineFouloideSession` ;
- mêmes actions (liste figée `env.actions`).

Dépendance : `pip install gymnasium` (fait dans le venv local ; à refaire sur A10).

## 3. Référence recommandée

**[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)** — même
framework que seedmind, donc diffable ligne à ligne si l'issue est « bug du
port ». (Alternative de contrôle : `danijar/dreamerv3`, JAX, l'original.)

Le branchement est **prêt à copier** (vérifié contre leur code du 2026-08-04 ;
leur API est du vieux gym 4-tuple avec clés `is_first`/`is_last`/`is_terminal`
dans l'obs, motif `envs/crafter.py`) :

- `scripts/calibration/dreamerv3_torch/fouloide.py` — l'adaptateur, à copier
  dans leur `envs/` (testé : `test_dreamerv3_torch_adapter_api`) ;
- `scripts/calibration/dreamerv3_torch/fouloide_configs.yaml` — le bloc de
  config à ajouter à leur `configs.yaml` (encodeur MLP sur la clé `vector`,
  actor `onehot`, `video_pred_log: false`, time_limit 2000 via LEUR wrapper —
  la mort reste le seul `is_terminal`, donc P(survie) apprenable).

Dans leur `dreamer.py`, fonction `make_env` (~ligne 190, après la branche
`crafter`), ajouter :

```python
    elif suite == "fouloide":
        import envs.fouloide as fouloide

        env = fouloide.Fouloide(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
```

### 3bis. Pièges d'intégration (rencontrés et résolus, session du 4 août 2026)

Le repo de référence n'a que des envs à pixels — un env vectoriel fait sauter
ses hypothèses « image » une par une. Les 4 patchs, tous appliqués côté
dreamerv3-torch après clone :

1. **`requirements.txt` inutilisable** (pins morts : torch 2.4.1, numpy 1.23).
   Ne PAS l'installer. À la place : `pip install "ruamel.yaml<0.18" tensorboard gym`
   (le pin ruamel est obligatoire : `dreamer.py:346` utilise `yaml.safe_load`,
   supprimé en 0.18+). Vérifier ensuite que torch/numpy n'ont pas bougé.
2. **`dreamer.py` `make_env`** : ajouter la branche `elif suite == "fouloide"`
   (§3) — patch python à ancre unique sur `else: raise NotImplementedError`.
3. **`tools.py:208`** (`simulate`) : `cache[...]["image"]` → `.get("image")` et
   garder `logger.video` sous `if video is not None:` (la vidéo d'éval suppose
   une image).
4. **`models.py:182`** (`WM.preprocess`) : garder `obs["image"] /= 255` sous
   `if "image" in obs:`. Les 3 autres accès image de `models.py` sont dans
   `video_pred`, appelé uniquement si `video_pred_log: true` — false chez nous.

(Le 5ᵉ piège était chez nous : gym 0.26 refuse Box(±inf) sur uint8 → bornes
0/1 pour les flags `is_*`, corrigé à la source dans `fouloide.py`.)

## 4. Protocole de la session A10

```bash
# une seule session, ~½ journée de GPU
cd ~/seedmind && git pull            # wrapper + adaptateur + ce doc
pip install gymnasium
cd ~ && git clone https://github.com/NM512/dreamerv3-torch && cd dreamerv3-torch
pip install -r requirements.txt

# branchement (3 gestes)
cp ~/seedmind/scripts/calibration/dreamerv3_torch/fouloide.py envs/
cat ~/seedmind/scripts/calibration/dreamerv3_torch/fouloide_configs.yaml >> configs.yaml
# + la branche elif "fouloide" dans make_env (dreamer.py, §3 ci-dessus)

SEEDMIND_ROOT=~/seedmind python -u dreamer.py --configs fouloide \
  --logdir ~/logdir/fouloide_ref 2>&1 | tee ref_fouloide.log
```

- Budget : **500k env steps** (≈ 10× nos runs 50k — c'est voulu : on teste
  aussi l'hypothèse régime de `seedmind-10e.5`).
- Suivi : `eval_return` et `train_return` dans leur logger (tensorboard ou
  jsonl). Le return EST `reward_learning` → directement comparable à nos runs
  (`runs/fouloide_entfloor_50k` : fourrage ~10× sous subsistance).
- Point de comparaison intermédiaire à 50k steps : si la référence fourrage
  déjà à budget égal au nôtre, le régime n'est même pas l'explication — c'est
  le port.

## 5. Critères de lecture (fixés avant le run, baseline MESURÉE)

Baseline aléatoire mesurée le 4 août 2026
(`python scripts/calibration/random_baseline.py`, 20 épisodes × 2000 pas max,
config dreamerfix, graines 0-19 — déterministe, reproductible) :

| Métrique | Policy aléatoire |
|---|---|
| return (reward_learning) / épisode | 76,5 ± 22,1 |
| survie (time_limit 2000 atteint) | 2/20 |
| fourrage (interact_food + interact_water) | **6,3 / 1000 pas** |
| longueur moyenne d'épisode | 856 pas |

Fait notable : le run seedmind `entfloor_50k` fourrageait à 0-4/1000 pas —
**sous la policy aléatoire**. Le return seul est donc un critère piégé (le
drive_reward absolu paie l'immobilité prudente) ; le fourrage/1000 pas est le
critère principal.

Verdict « la référence fourrage » (les trois, avec tendance croissante) :

- fourrage **> 20/1000 pas** (≥ 3× l'aléatoire) ;
- survie **> 10/20** épisodes d'éval ;
- return moyen **> 150** (≈ 2× l'aléatoire).

Tout le reste = échec de la référence (→ env/régime en cause).

## 6. Suites selon le verdict

- **Port en cause** → diff ordonné contre la référence : gradient de l'actor
  (REINFORCE vs backprop-through-dynamics — l'écart non tracé relevé dans
  `imagination_actor_critic.py:332`), normalisation des returns, KL/free-bits,
  train ratio, tailles de batch/séquence. Un composant à la fois, en gelant le
  reste.
- **Env/régime en cause** → arrêter définitivement de chercher un bug ;
  chantiers : shaping du fourrage (INTERACT trop rare dans l'exploration
  aléatoire ?), `time_limit` en entraînement seedmind aussi, régime 10e.5 sur
  A10 (gros replay, envs parallèles), et seulement ensuite l'observabilité
  (fenêtre 11×11).

## 7. Résultat (5 août 2026) — verdict rendu

Run complet : 514k steps sur A10, ~1 journée. Chiffres (fenêtres de 50k) :

| Fenêtre | return train moyen | survie | eval_return (fin de fenêtre) |
|---|---|---|---|
| 0-50k | 52,5 | 14/34 | ~50 |
| 100-150k | 81,3 | 13/49 | ~65 |
| 250-300k | 113,9 | 12/50 | ~90 |
| 400-450k | 136,0 | 11/52 | ~90 |
| 450-500k | 134,5 | 9/47 | ~100 (pic 131,7) |

Rappels : aléatoire = 76,5 ± 22,1, survie 10 %. Critères « la référence
fourrage » : return >150 ET survie >50 % ET fourrage >20/1000.

**Verdict pré-enregistré : la référence N'ATTEINT PAS les critères**
(return ~135 < 150, survie ~20 % plate < 50 %) → env/régime en cause, pas de
preuve de bug du port. Mais trois lectures fines changent la suite :

1. **Le monde est apprenable.** La référence monte de 52 à ~135 de return
   (+75 % au-dessus de l'aléatoire, pente toujours positive à 500k, moyenne de
   ~50 épisodes → hautement significatif). L'hypothèse « la tâche ne contient
   pas de signal » est réfutée.
2. **La survie ne s'améliore pas** (~20 % constant, épisodes ~1000 pas).
   La référence apprend à mieux vivre (wellbeing/step +44 %), pas à ne pas
   mourir — les dangers/famine en fenêtre 11×11 restent durs même pour elle.
3. **La lecture décisive : à 50k steps, la référence est AU NIVEAU ALÉATOIRE**
   (52,5 en fenêtre 0-50k, 75,6 en 50-100k). Or TOUS nos runs seedmind
   faisaient 50k. La séparation n'émerge qu'après ~150-200k steps. Le mur
   `seedmind-10e` est donc compatible avec un simple problème de BUDGET
   (l'hypothèse `seedmind-10e.5`, désormais étayée par des données) — les ~17
   leviers réfutés ont été testés dans un régime où même DreamerV3 de
   référence n'apprend rien.

**Décision — Expérience 2 (la contre-épreuve)** : lancer NOTRE agent
(config dreamerfix) à **500k steps sur l'A10**, mêmes conditions. Deux issues :

- sa courbe rejoint ~130+ vers 400k → le port est bon, le mur était le budget ;
  10e.5 devient le régime standard et on attaque la survie/le shaping ;
- il reste plat là où la référence montait → LÀ un défaut de port est prouvé,
  avec une cible de diff claire (gradient d'actor en premier).

Artefact : checkpoint de la référence dans `~/logdir/fouloide_ref` (A10) —
réutilisable pour mesurer son fourrage/1000 pas par rollout d'éval.

## 8. Expérience 2 (6 août 2026) — le port apprend, puis s'effondre

Notre agent (config dreamerfix, `--device cuda`), 500k steps, ~11 h A10.
Fourrage par fenêtres de 50k (aléatoire = 6,3/1000) :

```
3,7 → 4,3 → 4,8 → 5,4 → 7,9 → 8,3 → 10,3 → 9,0 → 7,2 → 6,3
                              pic 300-350k ↑    effondrement ↑
```

**Verdicts acquis :**

1. **Le mur 10e était un mur de budget** — fenêtre 0-50k à 3,7/1000 : tous les
   runs historiques (50k) se sont arrêtés dans la zone morte où la référence
   elle-même n'apprend rien. L'hypothèse `10e.5` est confirmée ; les ~17
   leviers de juin-juillet ont été testés dans un régime ininterprétable.
2. **La machinerie du port fonctionne** — montée monotone jusqu'à 1,6×
   l'aléatoire à 300-350k, même trajectoire que la référence.
3. **Divergence unique restante** : la référence TIENT son niveau en fin de
   run (fenêtres 400-500k stables à ~135 de return) ; notre agent REDESCEND
   au niveau aléatoire exact. Le diff n'est plus « pourquoi ça n'apprend
   pas » mais « qu'est-ce qui stabilise la fin de run ».

**Suspects de l'effondrement tardif, par ordre :**

1. **Replay 100k FIFO vs full-history** — la référence entraîne sur TOUT
   l'historique (`dataset_size` = 514k dans ses logs) ; nous sur une fenêtre
   glissante de 100k (`ExperienceBuffer` défaut, non surchargé par la config).
   Mécanisme : la distribution d'entraînement suit la policy qui se referme →
   le WM se dégrade sur les états rares (cohérent : `wm_loss` remonte de 1,70
   à 2,11 sur la fin) → l'imagination se trompe → la policy se dégrade →
   boucle. Testable en 1 run : `buffer_capacity: 500000`.
2. **Régulation d'entropie** — plancher 0,012 schedulé chez nous vs
   coefficient fixe 3e-4 × entropie chez la référence.
3. **Gradient d'actor** — REINFORCE+baseline avec imagination sous `no_grad`
   (`imagination_actor_critic.py:332`) ; a permis d'apprendre, moins suspect
   pour l'effondrement.

**Expérience 3** : `configs/..._dreamerfix_buf500k.yaml` (buffer 500k, seul
paramètre changé ; `online.buffer_capacity` branché dans `online.py`).
Si l'effondrement disparaît et que le fourrage continue au-delà de 10/1000 →
cause racine trouvée, port définitivement validé. Surveiller la RAM (≈ 4-5 Go
de buffer avec les fenêtres d'observation stockées).

```bash
cd /var/projects/seedmind && git pull
nohup python -u -m scripts.run_fouloide_online \
  --config configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix_buf500k.yaml \
  --steps 500000 --device cuda > run_500k_buf.log 2>&1 &
```

## 9. Expérience 3 (7 août 2026) — buffer réfuté, et la vraie pathologie

Deux graines, buffer 500k, en parallèle sur A10. Logs et métriques rapatriés
dans `runs_a10/` (le GPU a été rendu ; toute l'analyse suivante est à froid).

| run | steps | fourrage global | 2ᵉ moitié | tendance (r) | fenêtres idle |
|---|---|---|---|---|---|
| exp2 buffer 100k, seed 0 | 500k | 6,7 | 8,2 | **+0,68** | 4/49 |
| exp3 buffer 500k, seed 0 | 350k (OOM) | 6,2 | 6,7 | +0,19 | 4/34 |
| exp3 buffer 500k, seed 1 | 500k | **3,6** | 2,9 | **−0,30** | **21/49** |

(aléatoire = 6,25 ; « fenêtre idle » = 10k pas sans aucune mort et wellbeing < 0,01)

**Suspect n°1 (replay) : RÉFUTÉ.** Le buffer 500k ne corrige pas
l'effondrement — les deux runs buf500k font *moins* bien que le buf100k.
Coût annexe : 22,6 Go de RSS, seed 0 tué par l'OOM-killer à 350k (deux runs
buf500k ne tiennent pas ensemble dans 40 Go ; un seul, oui).

**Correction du verdict de l'exp 2.** Avec deux graines de plus, la belle
montée monotone 3,7→10,3 apparaît comme une **trajectoire particulière, non
reproductible**, pas comme une loi. Sur l'ensemble : aucun run ne bat
significativement l'aléatoire en moyenne globale (6,7 / 6,2 / 3,6 contre 6,25).
Ce qui reste acquis de l'exp 2 : **rien n'apprend avant ~150k steps** (donc les
runs 50k de juin-juillet étaient bien ininterprétables). Ce qui tombe : « le
budget suffit ».

**La pathologie est nommée : bistabilité / bassin idle.** Le mode d'échec
dominant n'est pas un apprentissage lent, c'est un agent qui bascule dans un
état où il ne meurt plus, ne fourrage plus et laisse ses jauges à zéro — 43 %
du temps pour seed 1. C'est le « bassin idle » déjà diagnostiqué en juin
(`rssm-idle-basin-2026-06-22`) et le profil « critfix immortel, jauges vides »
de `10e.10`. Il n'est pas absorbant (seed 1 en sort vers 350k puis y retombe) :
c'est un attracteur qui capture par intermittence.

**Nouveau suspect n°1 — l'absence de reset périodique.** La référence remet le
monde à zéro **toutes les 2000 pas** (`wrappers.TimeLimit`, config `time_limit:
2000`) ; notre boucle online ne reset **que sur mort**. Un agent idle-immortel
n'est donc jamais interrompu chez nous, alors que chez la référence l'état est
structurellement impossible à tenir : monde neuf toutes les 2000 pas,
ré-exploration forcée, et le replay reste alimenté en trajectoires variées.
C'est la seule différence structurelle qui explique à la fois la bistabilité,
son intermittence, et pourquoi la référence n'en souffre pas.

**Expérience 4 (à préparer à froid, une seule nuit de GPU)** : ajouter un
`online.episode_limit` (reset du monde tous les N pas, N=2000) et relancer
2 graines, buffer 100k (inutile de payer le 500k : réfuté). Critère : les
fenêtres idle doivent tomber à ~0 et le fourrage rester > 8/1000 en 2ᵉ moitié
sur les deux graines. Suspects suivants si échec : régulation d'entropie
(plancher 0,012 vs coef fixe 3e-4), puis gradient d'actor.
