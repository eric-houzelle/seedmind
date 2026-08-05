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
