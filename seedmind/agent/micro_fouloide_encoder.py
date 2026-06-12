"""Observation encoding for MicroFouloideWorld."""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from seedmind.envs.micro_fouloide_world import NUM_ENTITIES

MICRO_FOULOIDE_NUM_CHANNELS = NUM_ENTITIES
MICRO_FOULOIDE_NUM_SCALARS = 4 + NUM_ENTITIES
_SCALAR_KEYS = ("energy", "hydration", "temperature", "health")


def _qnet_channels(observation: Dict[str, Any], num_entities: int) -> np.ndarray:
    grid = np.asarray(observation["grid"], dtype=np.int64)
    h, w = grid.shape
    channels = np.zeros((num_entities, h, w), dtype=np.float32)
    flat = grid.ravel()
    rows, cols = np.divmod(np.arange(flat.size), w)
    valid = (flat >= 0) & (flat < num_entities)
    channels[flat[valid], rows[valid], cols[valid]] = 1.0
    return channels


def _qnet_scalars(observation: Dict[str, Any], num_entities: int) -> np.ndarray:
    drives = [float(observation.get(key, 0.0)) for key in _SCALAR_KEYS]
    standing = int(observation.get("standing_entity", 0))
    standing_onehot = np.zeros(num_entities, dtype=np.float32)
    if 0 <= standing < num_entities:
        standing_onehot[standing] = 1.0
    return np.concatenate([
        np.asarray(drives, dtype=np.float32),
        standing_onehot,
    ])


def _observation_to_vector(observation: Dict[str, Any], num_entities: int) -> np.ndarray:
    grid = np.asarray(observation["grid"], dtype=np.int64)
    onehot = np.zeros((grid.size, num_entities), dtype=np.float32)
    flat = grid.ravel()
    valid = (flat >= 0) & (flat < num_entities)
    onehot[np.arange(grid.size)[valid], flat[valid]] = 1.0
    return np.concatenate([
        onehot.reshape(-1),
        _qnet_scalars(observation, num_entities),
    ])


def make_micro_fouloide_obs_fns(num_entities: int) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray],
    Callable[[Sequence[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor]],
    int,
    int,
]:
    """Observation encoders sized for a registry of ``num_entities`` entities.

    Returns ``(obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars)``.
    """
    n = int(num_entities)

    def obs_to_vec(observation: Dict[str, Any]) -> np.ndarray:
        return _observation_to_vector(observation, n)

    def obs_batch(observations: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = np.stack([_qnet_channels(o, n) for o in observations])
        scalars = np.stack([_qnet_scalars(o, n) for o in observations])
        return torch.from_numpy(channels), torch.from_numpy(scalars)

    return obs_to_vec, obs_batch, n, len(_SCALAR_KEYS) + n


# ---------------------------------------------------------------------------
# Default (9-entity) versions, kept for backward compatibility
# ---------------------------------------------------------------------------

def micro_fouloide_qnet_channels(observation: Dict[str, Any]) -> np.ndarray:
    return _qnet_channels(observation, MICRO_FOULOIDE_NUM_CHANNELS)


def micro_fouloide_qnet_scalars(observation: Dict[str, Any]) -> np.ndarray:
    return _qnet_scalars(observation, NUM_ENTITIES)


def micro_fouloide_obs_batch_to_tensors(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    channels = np.stack([micro_fouloide_qnet_channels(o) for o in observations])
    scalars = np.stack([micro_fouloide_qnet_scalars(o) for o in observations])
    return torch.from_numpy(channels), torch.from_numpy(scalars)


def micro_fouloide_observation_to_vector(observation: Dict[str, Any]) -> np.ndarray:
    return _observation_to_vector(observation, MICRO_FOULOIDE_NUM_CHANNELS)


def make_micro_fouloide_observation_to_vector() -> Callable[[Dict[str, Any]], np.ndarray]:
    return micro_fouloide_observation_to_vector
