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

Branchement (à ajuster à la version du repo) : dans leur fabrique d'envs
(`envs/__init__.py`, dispatch par `--task suite_name`), ajouter une branche :

```python
elif suite == "fouloide":
    import sys; sys.path.append("/chemin/vers/seedmind")
    from scripts.calibration.gym_micro_fouloide import MicroFouloideGymEnv
    env = MicroFouloideGymEnv(
        "/chemin/vers/seedmind/configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml",
        seed=config.seed,
        reward_mode="learning",
        obs_mode="vector",
        time_limit=2000,
    )
    # puis appliquer leurs wrappers habituels (TimeLimit est déjà géré
    # par time_limit ci-dessus ; garder leur action/obs bookkeeping)
```

Réglages de référence : encodeur/décodeur **MLP** (obs vectorielle), taille de
modèle la plus petite de leur grille, `time_limit=2000` (le monde dreamerfix
est infini — une référence a besoin d'épisodes bornés ; 2000 pas ≈ plusieurs
cycles de famine, la mort par négligence reste possible donc apprenable).

## 4. Protocole de la session A10

```bash
# une seule session, ~½ journée de GPU
git pull                       # récupère wrapper + tests + ce doc
pip install gymnasium
git clone https://github.com/NM512/dreamerv3-torch && cd dreamerv3-torch
pip install -r requirements.txt
# ... ajouter la branche "fouloide" (§3), puis :
python dreamer.py --configs defaults --task fouloide_default \
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
