"""Gymnasium wrapper around ``MicroFouloideWorld`` for reference calibration.

Phase 0 du DREAMERV3_PORT_PLAN, rattrapée : faire tourner une implémentation
DreamerV3 de RÉFÉRENCE (ex. NM512/dreamerv3-torch) sur exactement le même
monde, la même information d'observation et le même signal de récompense que
l'agent seedmind, pour lever l'ambiguïté qui a bloqué seedmind-10e :

- la référence fourrage  -> notre port a un bug (diff composant par composant) ;
- la référence échoue aussi -> le problème est l'env/le régime, pas le code.

Contrat de fidélité (miroir de ``scripts/run_micro_fouloide.build_agent`` et de
``run_fouloide_online.OnlineFouloideSession``) :

- monde construit par le même ``build_env(config, seed)`` ;
- observation = les mêmes ``(channels, scalars)`` que consomme le ConvEncoder,
  crop égocentré inclus (mode onehot/properties + ``wrap_egocentric``) ;
- récompense = le même ``reward_learning`` que régresse le WM (drive_reward +
  ressources), calculé avec l'observation PRÉ-step comme la session online ;
- espace d'action = la même liste figée ``env.actions``.

Usage :

    from scripts.calibration.gym_micro_fouloide import MicroFouloideGymEnv
    env = MicroFouloideGymEnv(
        "configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml",
        seed=0,
    )
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Voir docs/CALIBRATION_DREAMERV3_REF.md pour le branchement dreamerv3-torch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from scripts.run_micro_fouloide import (  # noqa: E402
    _learning_reward,
    build_env,
    load_config,
)
from seedmind.agent.micro_fouloide_encoder import (  # noqa: E402
    make_micro_fouloide_obs_fns,
    make_micro_fouloide_property_obs_fns,
    wrap_egocentric,
)
from seedmind.envs.entities import load_registry  # noqa: E402
from seedmind.envs.micro_fouloide_world import OBSTACLE  # noqa: E402


def _build_obs_batch_fn(config: dict):
    """Rebuild the exact observation pipeline of ``build_agent`` (encoder-less).

    Returns ``(obs_batch_fn, num_channels, num_scalars, window_size)`` where
    ``obs_batch_fn([obs]) -> (channels [1,C,H,W], scalars [1,S])`` is the same
    callable the seedmind ConvEncoder consumes.
    """
    ec = config.get("env", {})
    ac = config.get("agent", {})
    grid_size = int(ec.get("size", 16))
    registry = load_registry(config)
    inventory_enabled = bool(ec.get("inventory", {}).get("enabled", False))
    obs_cfg = ac.get("observation", {})
    if str(obs_cfg.get("mode", "onehot")) == "properties":
        obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars = (
            make_micro_fouloide_property_obs_fns(
                registry,
                inventory=inventory_enabled,
                memory=bool(obs_cfg.get("spatial_memory", {}).get("enabled", False)),
            )
        )
    else:
        obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars = make_micro_fouloide_obs_fns(
            registry.size, inventory=inventory_enabled,
        )

    ego_cfg = obs_cfg.get("egocentric", {})
    if bool(ego_cfg.get("enabled", False)):
        radius = int(ego_cfg.get("radius", 5))
        oob_fill = int(ego_cfg.get("oob_fill", OBSTACLE))
        obs_to_vec_fn, obs_batch_fn = wrap_egocentric(
            obs_to_vec_fn, obs_batch_fn, radius, oob_fill,
            reveal_standing=bool(ego_cfg.get("reveal_standing", False)),
        )
        window_size = 2 * radius + 1
    else:
        window_size = grid_size

    return obs_batch_fn, num_channels, num_scalars, window_size


def _merge_env_overrides(config: dict, overrides: Optional[dict]) -> dict:
    if not overrides:
        return config
    merged = dict(config)
    merged["env"] = {**config.get("env", {}), **overrides}
    return merged


class MicroFouloideGymEnv(gym.Env):
    """Gymnasium API over the micro-fouloïde, information-faithful to seedmind.

    Parameters
    ----------
    config_path : yaml de run seedmind (ex. la config dreamerfix candidate prod).
    seed : graine de construction du monde (même sémantique que ``build_env``).
    reward_mode : ``"learning"`` (défaut, = la cible du WM seedmind) ou
        ``"external"`` (la récompense brute de l'env).
    obs_mode : ``"vector"`` (défaut) = Box plat ``channels+scalars`` — passe par
        l'encodeur MLP des implémentations de référence ; ``"grid"`` = Dict
        ``{"image": (H,W,C) float32, "drives": (S,)}`` pour un encodeur CNN.
    time_limit : tronque l'épisode après N steps (0 = jamais ; le monde
        dreamerfix est infini, une référence a besoin d'épisodes bornés).
    env_overrides : surcharges de la section ``env:`` de la config (tests,
        mondes réduits).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str,
        seed: int = 0,
        reward_mode: str = "learning",
        obs_mode: str = "vector",
        time_limit: int = 0,
        env_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if reward_mode not in ("learning", "external"):
            raise ValueError(f"unknown reward_mode: {reward_mode!r}")
        if obs_mode not in ("vector", "grid"):
            raise ValueError(f"unknown obs_mode: {obs_mode!r}")
        self._config = _merge_env_overrides(load_config(str(config_path)), env_overrides)
        self._reward_mode = reward_mode
        self._obs_mode = obs_mode
        self._time_limit = int(time_limit)
        self._seed = int(seed)

        (
            self._obs_batch_fn,
            self._num_channels,
            self._num_scalars,
            self._window,
        ) = _build_obs_batch_fn(self._config)

        self._env = build_env(self._config, self._seed)
        self._actions = list(self._env.actions)
        self.action_space = spaces.Discrete(len(self._actions))
        if obs_mode == "vector":
            dim = self._num_channels * self._window * self._window + self._num_scalars
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self._window, self._window, self._num_channels),
                    dtype=np.float32,
                ),
                "drives": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self._num_scalars,), dtype=np.float32,
                ),
            })

        self._obs: Optional[Dict[str, Any]] = None
        self._t = 0

    # ------------------------------------------------------------------
    @property
    def action_names(self) -> list[str]:
        return list(self._actions)

    def _encode(self, observation: Dict[str, Any]):
        channels, scalars = self._obs_batch_fn([observation])
        channels = channels[0].cpu().numpy().astype(np.float32)  # (C, H, W)
        scalars = scalars[0].cpu().numpy().astype(np.float32)    # (S,)
        if self._obs_mode == "vector":
            return np.concatenate([channels.reshape(-1), scalars])
        return {"image": np.transpose(channels, (1, 2, 0)), "drives": scalars}

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            # Same semantics as a fresh build_env(config, seed): new world layout.
            self._seed = int(seed)
            self._env = build_env(self._config, self._seed)
            self._actions = list(self._env.actions)
        self._obs = self._env.reset()
        self._t = 0
        return self._encode(self._obs), {}

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self._obs is None:
            raise RuntimeError("call reset() before step()")
        action_str = self._actions[int(action)]
        prev_obs = self._obs
        next_obs, reward_ext, done, info = self._env.step(action_str)
        # Same call as OnlineFouloideSession: learning reward is computed with
        # the PRE-step observation and the transition info.
        reward_learning = _learning_reward(float(reward_ext), prev_obs, info, self._config)
        reward = reward_learning if self._reward_mode == "learning" else float(reward_ext)
        self._obs = next_obs
        self._t += 1
        truncated = self._time_limit > 0 and self._t >= self._time_limit
        out_info = dict(info)
        out_info["reward_external"] = float(reward_ext)
        out_info["reward_learning"] = float(reward_learning)
        out_info["action_name"] = action_str
        return self._encode(next_obs), float(reward), bool(done), truncated, out_info


__all__ = ["MicroFouloideGymEnv"]
