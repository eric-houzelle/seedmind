"""Smoke tests du wrapper Gymnasium de calibration (scripts/calibration/).

Vérifie le contrat de fidélité : mêmes espaces que l'obs pipeline seedmind,
reward_learning identique à la sémantique de la session online, API gymnasium
correcte (reset/step/terminated/truncated), reproductibilité par graine.
"""
from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from scripts.calibration.gym_micro_fouloide import MicroFouloideGymEnv  # noqa: E402

CONFIG = "configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml"
# Monde réduit pour des tests rapides ; la structure (canaux, crop égocentré)
# vient de la config réelle, seule la taille change.
SMALL = {"size": 12, "max_steps": 0}


def _make(**kwargs) -> MicroFouloideGymEnv:
    return MicroFouloideGymEnv(CONFIG, seed=0, env_overrides=SMALL, **kwargs)


def test_spaces_match_seedmind_obs_pipeline():
    env = _make()
    # Config dreamerfix : égocentré radius 5 -> fenêtre 11x11.
    assert env._window == 11
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32
    assert obs.shape == (env._num_channels * 11 * 11 + env._num_scalars,)
    assert env.action_space.n == len(env.action_names)


def test_step_api_and_rewards_in_info():
    env = _make(time_limit=25)
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert "reward_external" in info and "reward_learning" in info
        assert info["action_name"] in env.action_names
        steps += 1
        assert steps <= 25
    assert truncated or terminated  # time_limit borne l'épisode infini


def test_reward_learning_is_the_wm_target():
    """reward_mode='learning' doit rendre exactement info['reward_learning']."""
    env = _make(reward_mode="learning", time_limit=50)
    env.reset(seed=3)
    for _ in range(20):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert reward == pytest.approx(info["reward_learning"])
        if terminated or truncated:
            break

    env_ext = _make(reward_mode="external", time_limit=50)
    env_ext.reset(seed=3)
    for _ in range(20):
        _, reward, terminated, truncated, info = env_ext.step(env_ext.action_space.sample())
        assert reward == pytest.approx(info["reward_external"])
        if terminated or truncated:
            break


def test_seed_reproducibility():
    a = _make()
    b = _make()
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    # Même graine + mêmes actions -> mêmes trajectoires d'observations.
    for act in [0, 1, 2, 3, 0, 1]:
        step_a = a.step(act)
        step_b = b.step(act)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1] == pytest.approx(step_b[1])


def test_grid_obs_mode():
    env = _make(obs_mode="grid")
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"image", "drives"}
    assert obs["image"].shape == (11, 11, env._num_channels)
    assert obs["drives"].shape == (env._num_scalars,)
    assert env.observation_space.contains(obs)
