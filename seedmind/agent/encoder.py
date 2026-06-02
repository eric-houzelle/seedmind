"""Encoder: raw observation -> latent representation (SPEC section 11).

In V1 the encoder is a *frozen* randomly-initialised MLP. Freezing it gives the
World Model stable latent targets to predict (so its loss can decrease without
representation collapse) while keeping the module swappable later for image,
text or multimodal inputs.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from seedmind.envs.gridworld import (
    AGENT,
    COLOR_DOOR_CLOSED,
    COLOR_DOOR_OPEN,
    COLOR_KEY,
    COLORS,
    DANGER,
    DOOR_CLOSED,
    DOOR_OPEN,
    EMPTY,
    KEY,
    NUM_ENTITIES,
    UNKNOWN_OBJECT,
    WALL,
)

# Inventory vector: [has_key, door_open] + one-hot of the held key color.
INVENTORY_DIM = 2 + len(COLORS)
_COLOR_INDEX = {color: i for i, color in enumerate(COLORS)}


# --- Color-agnostic representation for the learned Q-network (V2) ---
# Instead of one separate channel per colored entity (which cannot transfer to
# an unseen color, since that color's channels stay zero during training), the
# Q-network sees *generic* structural channels plus a single scalar "color
# value" channel. Matching the held key color to a door then becomes a
# continuous comparison that generalises to colors never seen in training.
QNET_EMPTY, QNET_WALL, QNET_AGENT = 0, 1, 2
QNET_KEY, QNET_CLOSED_DOOR, QNET_OPEN_DOOR = 3, 4, 5
QNET_DANGER, QNET_COLOR_VALUE, QNET_COLOR_MATCH, QNET_UNKNOWN = 6, 7, 8, 9
QNET_NUM_CHANNELS = 10
# Inventory for the Q-net: [has_key, door_open].
# The held key color is no longer a separate scalar — the match channel
# already encodes "does this cell match my key?", which is the only
# information the agent needs. This makes the representation inherently
# color-agnostic: the match channel works identically for any color.
QNET_INVENTORY_DIM = 2


def _color_value(color: str) -> float:
    """Map a color name to a scalar in (0, 1]; shared across keys and doors."""
    return (_COLOR_INDEX[color] + 1) / len(COLORS)


# Entity code -> (qnet structural channel, color value or None).
_QNET_CODE_MAP: Dict[int, tuple] = {
    EMPTY: (QNET_EMPTY, None),
    WALL: (QNET_WALL, None),
    AGENT: (QNET_AGENT, None),
    KEY: (QNET_KEY, None),
    DOOR_CLOSED: (QNET_CLOSED_DOOR, None),
    DOOR_OPEN: (QNET_OPEN_DOOR, None),
    DANGER: (QNET_DANGER, None),
    UNKNOWN_OBJECT: (QNET_UNKNOWN, None),
}
for _c in COLORS:
    _QNET_CODE_MAP[COLOR_KEY[_c]] = (QNET_KEY, _color_value(_c))
    _QNET_CODE_MAP[COLOR_DOOR_CLOSED[_c]] = (QNET_CLOSED_DOOR, _color_value(_c))
    _QNET_CODE_MAP[COLOR_DOOR_OPEN[_c]] = (QNET_OPEN_DOOR, _color_value(_c))

# Vectorised lookups: entity code -> structural channel (-1 if none) / color value.
_QNET_CHANNEL_OF = np.full(NUM_ENTITIES, -1, dtype=np.int64)
_QNET_COLORVAL_OF = np.zeros(NUM_ENTITIES, dtype=np.float32)
for _code, (_ch, _cval) in _QNET_CODE_MAP.items():
    _QNET_CHANNEL_OF[_code] = _ch
    if _cval is not None:
        _QNET_COLORVAL_OF[_code] = _cval


def observation_qnet_channels(observation: Dict[str, Any]) -> np.ndarray:
    """Color-agnostic ``(QNET_NUM_CHANNELS, H, W)`` tensor for the Q-network.

    Includes a **match channel** (``QNET_COLOR_MATCH``): for each cell,
    ``+1`` if its color equals the held key color, ``-1`` if it has a
    different color, ``0`` if it has no color. This gives the agent an
    innate "same vs different" perception — it still has to *learn* that
    matching entities are the ones worth interacting with. Because the
    channel is computed relationally, it transfers to any color the agent
    has never seen during training.
    """
    grid = np.asarray(observation["grid"], dtype=np.int64)
    h, w = grid.shape
    flat = grid.reshape(-1)
    channels = np.zeros((QNET_NUM_CHANNELS, h, w), dtype=np.float32)
    chan = _QNET_CHANNEL_OF[flat]
    rows, cols = np.divmod(np.arange(flat.size), w)
    mask = chan >= 0
    channels[chan[mask], rows[mask], cols[mask]] = 1.0
    channels[QNET_COLOR_VALUE] = _QNET_COLORVAL_OF[flat].reshape(h, w)

    # Match channel: compare each cell's color to the held key color.
    key_color = observation.get("key_color")
    if key_color and key_color in _COLOR_INDEX:
        key_cv = _color_value(key_color)
        color_layer = channels[QNET_COLOR_VALUE]
        has_color = color_layer > 0
        match_layer = np.zeros((h, w), dtype=np.float32)
        match_layer[has_color & (np.abs(color_layer - key_cv) < 1e-4)] = 1.0
        match_layer[has_color & (np.abs(color_layer - key_cv) >= 1e-4)] = -1.0
        channels[QNET_COLOR_MATCH] = match_layer

    return channels


def observation_qnet_inventory(observation: Dict[str, Any]) -> np.ndarray:
    """Inventory for the Q-net: [has_key, door_open].

    The held key color is deliberately omitted — the match channel in the
    grid already encodes "does this cell match my key?" relationally, which
    is color-agnostic and transfers to unseen colors.
    """
    inv = np.zeros(QNET_INVENTORY_DIM, dtype=np.float32)
    inv[0] = float(observation.get("has_key", 0))
    inv[1] = float(observation.get("door_open", 0))
    return inv


def observation_inventory(observation: Dict[str, Any]) -> np.ndarray:
    """Encode inventory flags plus a one-hot of the held key color."""
    inv = np.zeros(INVENTORY_DIM, dtype=np.float32)
    inv[0] = float(observation.get("has_key", 0))
    inv[1] = float(observation.get("door_open", 0))
    key_color = observation.get("key_color")
    if key_color in _COLOR_INDEX:
        inv[2 + _COLOR_INDEX[key_color]] = 1.0
    return inv


def observation_grid_channels(
    observation: Dict[str, Any], num_entities: int = NUM_ENTITIES
) -> np.ndarray:
    """Return the grid as a one-hot ``(num_entities, H, W)`` tensor."""
    grid = np.asarray(observation["grid"], dtype=np.int64)
    h, w = grid.shape
    channels = np.zeros((num_entities, h, w), dtype=np.float32)
    rows, cols = np.indices((h, w))
    channels[grid.reshape(-1), rows.reshape(-1), cols.reshape(-1)] = 1.0
    return channels


def observation_to_vector(observation: Dict[str, Any], num_entities: int = NUM_ENTITIES) -> np.ndarray:
    """Flatten an observation into a one-hot grid plus inventory flags."""
    grid = np.asarray(observation["grid"], dtype=np.int64)
    onehot = np.zeros((grid.size, num_entities), dtype=np.float32)
    onehot[np.arange(grid.size), grid.reshape(-1)] = 1.0
    flat = onehot.reshape(-1)
    inventory = observation_inventory(observation)
    return np.concatenate([flat, inventory])


class Encoder(nn.Module):
    """Frozen MLP encoder producing a fixed-size latent vector."""

    def __init__(
        self,
        grid_size: int,
        latent_dim: int = 128,
        num_entities: int = NUM_ENTITIES,
        hidden_dim: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_entities = num_entities
        self.input_dim = grid_size * grid_size * num_entities + INVENTORY_DIM

        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

        # Freeze: the encoder is a fixed random projection in V1.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """Encode a single observation into a latent numpy vector."""
        vec = observation_to_vector(observation, self.num_entities)
        tensor = torch.from_numpy(vec).unsqueeze(0)
        latent = self.net(tensor).squeeze(0)
        return latent.numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, observations) -> np.ndarray:
        vecs = np.stack([observation_to_vector(o, self.num_entities) for o in observations])
        tensor = torch.from_numpy(vecs)
        return self.net(tensor).numpy().astype(np.float32)

    def forward(self, observation: Dict[str, Any]) -> np.ndarray:  # pragma: no cover
        return self.encode(observation)
