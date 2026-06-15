"""Observation encoding for the SandboxWorld.

Converts a sandbox observation dict into the tensors expected by
:class:`~seedmind.agent.q_network.QNetwork`:
- A one-hot ``(C, H, W)`` grid of entity channels.
- A scalar vector for energy and inventory.

Separated from the gridworld encoder to keep the two worlds decoupled.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from seedmind.envs.sandbox_world import CRAFT_NUM_ENTITIES, NUM_ENTITIES

SANDBOX_NUM_CHANNELS = NUM_ENTITIES
SANDBOX_NUM_SCALARS = 2  # energy (normalised) + inventory food (normalised)
SANDBOX_CRAFT_NUM_CHANNELS = CRAFT_NUM_ENTITIES
SANDBOX_CRAFT_NUM_SCALARS = 5  # energy + food + wood + stone + tool

_MAX_INVENTORY = 10.0
_BASE_INVENTORY_KEYS = ("food",)
_CRAFT_INVENTORY_KEYS = ("food", "wood", "stone", "tool")


def sandbox_num_channels(include_craft: bool = False) -> int:
    return SANDBOX_CRAFT_NUM_CHANNELS if include_craft else SANDBOX_NUM_CHANNELS


def sandbox_num_scalars(include_craft: bool = False) -> int:
    return SANDBOX_CRAFT_NUM_SCALARS if include_craft else SANDBOX_NUM_SCALARS


def _inventory_keys(include_craft: bool = False) -> Tuple[str, ...]:
    return _CRAFT_INVENTORY_KEYS if include_craft else _BASE_INVENTORY_KEYS


def sandbox_qnet_channels(
    observation: Dict[str, Any],
    num_entities: int = NUM_ENTITIES,
) -> np.ndarray:
    """One-hot ``(num_entities, H, W)`` grid encoding."""
    grid = np.asarray(observation["grid"], dtype=np.int64)
    h, w = grid.shape
    channels = np.zeros((num_entities, h, w), dtype=np.float32)
    flat = grid.ravel()
    rows, cols = np.divmod(np.arange(flat.size), w)
    valid = (flat >= 0) & (flat < num_entities)
    channels[flat[valid], rows[valid], cols[valid]] = 1.0
    return channels


def sandbox_qnet_scalars(
    observation: Dict[str, Any],
    include_craft: bool = False,
) -> np.ndarray:
    """Scalar feature vector for energy and inventory."""
    energy_max = float(observation.get("energy_max", 100.0))
    energy = float(observation.get("energy", 0.0))
    values = [energy / max(energy_max, 1.0)]
    for key in _inventory_keys(include_craft):
        amount = float(observation.get(f"inventory_{key}", 0))
        values.append(min(amount, _MAX_INVENTORY) / _MAX_INVENTORY)
    return np.array(values, dtype=np.float32)


def sandbox_obs_batch_to_tensors(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch converter for :class:`QNetwork`."""
    channels = np.stack([sandbox_qnet_channels(o) for o in observations])
    scalars = np.stack([sandbox_qnet_scalars(o) for o in observations])
    return torch.from_numpy(channels), torch.from_numpy(scalars)


def make_sandbox_obs_batch_to_tensors(
    include_craft: bool = False,
) -> Callable[[Sequence[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor]]:
    num_entities = sandbox_num_channels(include_craft)

    def _convert(observations: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = np.stack([
            sandbox_qnet_channels(o, num_entities=num_entities)
            for o in observations
        ])
        scalars = np.stack([
            sandbox_qnet_scalars(o, include_craft=include_craft)
            for o in observations
        ])
        return torch.from_numpy(channels), torch.from_numpy(scalars)

    return _convert


def sandbox_observation_to_vector(
    observation: Dict[str, Any],
    include_craft: bool = False,
) -> np.ndarray:
    """Flatten a sandbox observation for the frozen MLP Encoder.

    Layout: one-hot grid (H*W*C) + scalar features.
    """
    num_entities = sandbox_num_channels(include_craft)
    grid = np.asarray(observation["grid"], dtype=np.int64)
    onehot = np.zeros((grid.size, num_entities), dtype=np.float32)
    valid = (grid.ravel() >= 0) & (grid.ravel() < num_entities)
    onehot[np.arange(grid.size)[valid], grid.ravel()[valid]] = 1.0
    flat = onehot.reshape(-1)
    scalars = sandbox_qnet_scalars(observation, include_craft=include_craft)
    return np.concatenate([flat, scalars])


def make_sandbox_observation_to_vector(
    include_craft: bool = False,
) -> Callable[[Dict[str, Any]], np.ndarray]:
    return lambda observation: sandbox_observation_to_vector(
        observation, include_craft=include_craft,
    )
