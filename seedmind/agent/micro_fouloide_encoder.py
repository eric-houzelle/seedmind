"""Observation encoding for MicroFouloideWorld."""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from seedmind.envs.micro_fouloide_world import NUM_ENTITIES

MICRO_FOULOIDE_NUM_CHANNELS = NUM_ENTITIES
MICRO_FOULOIDE_NUM_SCALARS = 4 + NUM_ENTITIES
_SCALAR_KEYS = ("energy", "hydration", "temperature", "health")


def micro_fouloide_qnet_channels(observation: Dict[str, Any]) -> np.ndarray:
    grid = np.asarray(observation["grid"], dtype=np.int64)
    h, w = grid.shape
    channels = np.zeros((MICRO_FOULOIDE_NUM_CHANNELS, h, w), dtype=np.float32)
    flat = grid.ravel()
    rows, cols = np.divmod(np.arange(flat.size), w)
    valid = (flat >= 0) & (flat < MICRO_FOULOIDE_NUM_CHANNELS)
    channels[flat[valid], rows[valid], cols[valid]] = 1.0
    return channels


def micro_fouloide_qnet_scalars(observation: Dict[str, Any]) -> np.ndarray:
    drives = [float(observation.get(key, 0.0)) for key in _SCALAR_KEYS]
    standing = int(observation.get("standing_entity", 0))
    standing_onehot = np.zeros(NUM_ENTITIES, dtype=np.float32)
    if 0 <= standing < NUM_ENTITIES:
        standing_onehot[standing] = 1.0
    return np.concatenate([
        np.asarray(drives, dtype=np.float32),
        standing_onehot,
    ])


def micro_fouloide_obs_batch_to_tensors(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    channels = np.stack([micro_fouloide_qnet_channels(o) for o in observations])
    scalars = np.stack([micro_fouloide_qnet_scalars(o) for o in observations])
    return torch.from_numpy(channels), torch.from_numpy(scalars)


def micro_fouloide_observation_to_vector(observation: Dict[str, Any]) -> np.ndarray:
    grid = np.asarray(observation["grid"], dtype=np.int64)
    onehot = np.zeros((grid.size, MICRO_FOULOIDE_NUM_CHANNELS), dtype=np.float32)
    flat = grid.ravel()
    valid = (flat >= 0) & (flat < MICRO_FOULOIDE_NUM_CHANNELS)
    onehot[np.arange(grid.size)[valid], flat[valid]] = 1.0
    return np.concatenate([
        onehot.reshape(-1),
        micro_fouloide_qnet_scalars(observation),
    ])


def make_micro_fouloide_observation_to_vector() -> Callable[[Dict[str, Any]], np.ndarray]:
    return micro_fouloide_observation_to_vector
