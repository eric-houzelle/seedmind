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


def _qnet_scalars(
    observation: Dict[str, Any], num_entities: int, inventory: bool = False,
) -> np.ndarray:
    drives = [float(observation.get(key, 0.0)) for key in _SCALAR_KEYS]
    standing = int(observation.get("standing_entity", 0))
    standing_onehot = np.zeros(num_entities, dtype=np.float32)
    if 0 <= standing < num_entities:
        standing_onehot[standing] = 1.0
    parts = [np.asarray(drives, dtype=np.float32), standing_onehot]
    if inventory:
        counts = np.zeros(num_entities, dtype=np.float32)
        raw = observation.get("inventory")
        if raw is not None:
            raw = np.asarray(raw, dtype=np.float32).ravel()
            counts[: min(len(raw), num_entities)] = raw[:num_entities]
        parts.append(counts)
    return np.concatenate(parts)


def _observation_to_vector(
    observation: Dict[str, Any], num_entities: int, inventory: bool = False,
) -> np.ndarray:
    grid = np.asarray(observation["grid"], dtype=np.int64)
    onehot = np.zeros((grid.size, num_entities), dtype=np.float32)
    flat = grid.ravel()
    valid = (flat >= 0) & (flat < num_entities)
    onehot[np.arange(grid.size)[valid], flat[valid]] = 1.0
    return np.concatenate([
        onehot.reshape(-1),
        _qnet_scalars(observation, num_entities, inventory=inventory),
    ])


def make_micro_fouloide_obs_fns(num_entities: int, inventory: bool = False) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray],
    Callable[[Sequence[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor]],
    int,
    int,
]:
    """Observation encoders sized for a registry of ``num_entities`` entities.

    ``inventory=True`` appends a per-entity inventory-count vector to the
    scalars. Returns ``(obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars)``.
    """
    n = int(num_entities)
    inv = bool(inventory)

    def obs_to_vec(observation: Dict[str, Any]) -> np.ndarray:
        return _observation_to_vector(observation, n, inventory=inv)

    def obs_batch(observations: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = np.stack([_qnet_channels(o, n) for o in observations])
        scalars = np.stack([_qnet_scalars(o, n, inventory=inv) for o in observations])
        return torch.from_numpy(channels), torch.from_numpy(scalars)

    num_scalars = len(_SCALAR_KEYS) + n + (n if inv else 0)
    return obs_to_vec, obs_batch, n, num_scalars


# ---------------------------------------------------------------------------
# "properties" observation mode: dimensions independent of the entity count
# ---------------------------------------------------------------------------

def make_micro_fouloide_property_obs_fns(
    registry, inventory: bool = False, memory: bool = False,
) -> Tuple[
    Callable[[Dict[str, Any]], np.ndarray],
    Callable[[Sequence[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor]],
    int,
    int,
]:
    """Property-vector encoders: cells are perceived as property vectors.

    Channel/scalar counts depend only on PROPERTY_DIM, never on the number of
    entity types — so artifacts can be added mid-life without resizing nets.
    Returns ``(obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars)``.
    """
    from seedmind.envs.entities import PROPERTY_DIM

    matrix = registry.property_matrix()
    p = PROPERTY_DIM
    inv = bool(inventory)
    mem = bool(memory)

    def channels_fn(observation: Dict[str, Any]) -> np.ndarray:
        grid = np.asarray(observation["grid"], dtype=np.int64)
        parts = [np.transpose(matrix[grid], (2, 0, 1))]
        if mem:
            known = np.asarray(
                observation.get("memory_grid", np.full_like(grid, -1)), dtype=np.int64,
            )
            fresh = np.asarray(
                observation.get("memory_fresh", np.zeros_like(grid, dtype=np.float32)),
                dtype=np.float32,
            )
            remembered = np.transpose(matrix[np.clip(known, 0, len(matrix) - 1)], (2, 0, 1))
            remembered = remembered * fresh[None, :, :]
            parts.append(remembered)
            parts.append(fresh[None, :, :])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def scalars_fn(observation: Dict[str, Any]) -> np.ndarray:
        drives = [float(observation.get(key, 0.0)) for key in _SCALAR_KEYS]
        standing = int(observation.get("standing_entity", 0))
        standing_props = (
            matrix[standing] if 0 <= standing < len(matrix)
            else np.zeros(p, dtype=np.float32)
        )
        parts = [np.asarray(drives, dtype=np.float32), standing_props]
        if inv:
            held = np.zeros(p, dtype=np.float32)
            load = 0.0
            raw = observation.get("inventory")
            if raw is not None:
                counts = np.asarray(raw, dtype=np.float32).ravel()
                n = min(len(counts), len(matrix))
                held = counts[:n] @ matrix[:n]
                load = float(counts.sum())
            parts.append(held.astype(np.float32))
            parts.append(np.asarray([load], dtype=np.float32))
        return np.concatenate(parts)

    def obs_to_vec(observation: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([
            channels_fn(observation).reshape(-1),
            scalars_fn(observation),
        ])

    def obs_batch(observations: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = np.stack([channels_fn(o) for o in observations])
        scalars = np.stack([scalars_fn(o) for o in observations])
        return torch.from_numpy(channels), torch.from_numpy(scalars)

    num_channels = p + (p + 1 if mem else 0)
    num_scalars = len(_SCALAR_KEYS) + p + (p + 1 if inv else 0)
    return obs_to_vec, obs_batch, num_channels, num_scalars


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
