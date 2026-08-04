"""Adaptateur micro-fouloïde pour NM512/dreamerv3-torch — À COPIER dans leur
répertoire ``envs/`` (voir docs/CALIBRATION_DREAMERV3_REF.md §3).

Traduit notre wrapper Gymnasium (API 5-tuple) vers le protocole attendu par
dreamerv3-torch (vieux gym : ``reset() -> obs``, ``step() -> (obs, reward,
done, info)``, obs dict avec ``is_first``/``is_last``/``is_terminal`` — même
motif que leur ``envs/crafter.py``).

Configuration par variables d'environnement (pour ne pas toucher à leur
plomberie de config) :

    SEEDMIND_ROOT    racine du repo seedmind (défaut : ~/seedmind)
    FOULOIDE_CONFIG  yaml de run seedmind (défaut : la config dreamerfix)

Sémantique de fin d'épisode : ``is_terminal`` = vraie mort (le continue
predictor apprend P(survie)) ; la troncature time-limit est gérée par LEUR
wrapper ``TimeLimit`` (discount 1.0), donc notre env tourne sans limite.
"""
import os
import sys
from pathlib import Path

import numpy as np

try:  # vieux gym dans dreamerv3-torch ; gymnasium en test local seedmind
    import gym
except ImportError:  # pragma: no cover
    import gymnasium as gym

_SEEDMIND_ROOT = Path(os.environ.get("SEEDMIND_ROOT", "~/seedmind")).expanduser()
sys.path.insert(0, str(_SEEDMIND_ROOT))

from scripts.calibration.gym_micro_fouloide import MicroFouloideGymEnv  # noqa: E402

_DEFAULT_CONFIG = (
    _SEEDMIND_ROOT / "configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml"
)


class Fouloide:
    """Micro-fouloïde au protocole dreamerv3-torch (motif envs/crafter.py)."""

    metadata = {}

    def __init__(self, task="default", seed=0):
        config_path = os.environ.get("FOULOIDE_CONFIG", str(_DEFAULT_CONFIG))
        # time_limit=0 : la troncature est gérée par wrappers.TimeLimit côté
        # dreamerv3-torch (config time_limit: 2000), qui pose discount=1.0.
        self._env = MicroFouloideGymEnv(
            config_path,
            seed=seed,
            reward_mode="learning",
            obs_mode="vector",
            time_limit=0,
        )
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        dim = int(np.prod(self._env.observation_space.shape))
        spaces = {
            "vector": gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = gym.spaces.Discrete(self._env.action_space.n)
        space.discrete = True
        return space

    def step(self, action):
        vector, reward, terminated, truncated, info = self._env.step(int(action))
        done = bool(terminated or truncated)
        obs = {
            "vector": vector,
            "is_first": False,
            "is_last": done,
            "is_terminal": bool(terminated),
        }
        info = dict(info)
        info["discount"] = np.array(0.0 if terminated else 1.0, dtype=np.float32)
        return obs, np.float32(reward), done, info

    def reset(self):
        vector, _ = self._env.reset()
        return {
            "vector": vector,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
