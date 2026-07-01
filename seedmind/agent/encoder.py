"""Encoder: raw observation -> latent representation (SPEC section 11).

In V1 the encoder is a *frozen* randomly-initialised MLP. Freezing it gives the
World Model stable latent targets to predict (so its loss can decrease without
representation collapse) while keeping the module swappable later for image,
text or multimodal inputs.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

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
    """Frozen MLP encoder producing a fixed-size latent vector.

    Parameters
    ----------
    grid_size, latent_dim, num_entities, hidden_dim, seed :
        Standard constructor args (see V1 SPEC).
    input_dim : int or None
        Override the input dimension. When ``None`` it is computed from
        *grid_size*, *num_entities* and ``INVENTORY_DIM``.
    obs_to_vec_fn : callable or None
        Custom vectorisation ``observation -> np.ndarray``. When ``None``
        the default gridworld ``observation_to_vector`` is used.
    structured_features_fn : callable or None
        Optional generic perception hook ``observation -> np.ndarray``. When
        present, these observable features are copied into the tail of the
        latent vector so downstream latent-space modules can use them directly.
    """

    def __init__(
        self,
        grid_size: int,
        latent_dim: int = 128,
        num_entities: int = NUM_ENTITIES,
        hidden_dim: int = 256,
        seed: int = 0,
        input_dim: Optional[int] = None,
        obs_to_vec_fn: Optional[Any] = None,
        structured_features_fn: Optional[Any] = None,
        structured_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_entities = num_entities
        self.input_dim = input_dim if input_dim is not None else (
            grid_size * grid_size * num_entities + INVENTORY_DIM
        )
        self._obs_to_vec_fn = obs_to_vec_fn
        self._structured_features_fn = structured_features_fn
        self.structured_feature_dim = int(structured_feature_dim)
        if self.structured_feature_dim < 0:
            raise ValueError("structured_feature_dim must be non-negative")
        if self.structured_feature_dim >= latent_dim:
            raise ValueError("structured_feature_dim must be smaller than latent_dim")
        self.projected_latent_dim = latent_dim - self.structured_feature_dim

        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.projected_latent_dim),
            nn.Tanh(),
        )

        # Freeze: the encoder is a fixed random projection in V1.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _vectorise(self, observation: Dict[str, Any]) -> np.ndarray:
        if self._obs_to_vec_fn is not None:
            return self._obs_to_vec_fn(observation)
        return observation_to_vector(observation, self.num_entities)

    def _structured_features(self, observation: Dict[str, Any]) -> np.ndarray:
        if self.structured_feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._structured_features_fn is None:
            return np.zeros((self.structured_feature_dim,), dtype=np.float32)
        features = np.asarray(self._structured_features_fn(observation), dtype=np.float32)
        if features.shape != (self.structured_feature_dim,):
            raise ValueError(
                "structured_features_fn returned shape "
                f"{features.shape}, expected {(self.structured_feature_dim,)}"
            )
        return features

    @torch.no_grad()
    def encode_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Encode a single observation into a latent vector on the module device."""
        vec = self._vectorise(observation)
        device = next(self.parameters()).device
        tensor = torch.from_numpy(vec).unsqueeze(0).to(device)
        projected = self.net(tensor).squeeze(0)
        if self.structured_feature_dim == 0:
            return projected
        features = torch.from_numpy(self._structured_features(observation)).to(device)
        return torch.cat([projected, features], dim=0)

    @torch.no_grad()
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """Encode a single observation into a latent numpy vector."""
        return self.encode_tensor(observation).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, observations) -> np.ndarray:
        vecs = np.stack([self._vectorise(o) for o in observations])
        device = next(self.parameters()).device
        tensor = torch.from_numpy(vecs).to(device)
        projected = self.net(tensor)
        if self.structured_feature_dim == 0:
            return projected.cpu().numpy().astype(np.float32)
        features = np.stack([self._structured_features(o) for o in observations])
        features_t = torch.from_numpy(features).to(device)
        return torch.cat([projected, features_t], dim=1).cpu().numpy().astype(np.float32)

    def forward(self, observation: Dict[str, Any]) -> np.ndarray:  # pragma: no cover
        return self.encode(observation)


class ConvEncoder(nn.Module):
    """Frozen *convolutional* encoder over a multi-channel grid + scalars.

    Same role and numpy interface as :class:`Encoder` (``encode`` /
    ``encode_tensor`` / ``encode_batch``), but instead of flattening the whole
    world into one big vector it runs a small CNN over a channel grid and
    concatenates the scalar features. This makes the latent **translation-aware**
    and, when fed an egocentric window of fixed side, **independent of the world
    size** — so the same encoder transfers across worlds of any size.

    It stays *frozen* (random fixed weights) for the same reason as
    :class:`Encoder`: the World Model needs a stable latent target to predict.
    Domain-agnostic: it consumes ``obs_batch_fn`` (observations -> ``(channels,
    scalars)`` tensors), so it works with any world or observation mode.
    """

    def __init__(
        self,
        num_channels: int,
        num_scalars: int,
        window_size: int,
        latent_dim: int = 128,
        conv_channels: int = 32,
        hidden_dim: int = 256,
        seed: int = 0,
        obs_batch_fn: Optional[Any] = None,
        structured_features_fn: Optional[Any] = None,
        structured_feature_dim: int = 0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if obs_batch_fn is None:
            raise ValueError("ConvEncoder requires an obs_batch_fn")
        self.latent_dim = int(latent_dim)
        self.num_channels = int(num_channels)
        self.num_scalars = int(num_scalars)
        self.window_size = int(window_size)
        self._obs_batch_fn = obs_batch_fn
        self._structured_features_fn = structured_features_fn
        self.structured_feature_dim = int(structured_feature_dim)
        if self.structured_feature_dim < 0:
            raise ValueError("structured_feature_dim must be non-negative")
        if self.structured_feature_dim >= latent_dim:
            raise ValueError("structured_feature_dim must be smaller than latent_dim")
        self.projected_latent_dim = latent_dim - self.structured_feature_dim

        torch.manual_seed(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = conv_channels * self.window_size * self.window_size
        self.head = nn.Sequential(
            nn.Linear(conv_out + self.num_scalars, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.projected_latent_dim),
            nn.Tanh(),
        )

        # By default the encoder is a fixed random projection (like Encoder): the
        # World Model needs a *stable* latent target to predict. The DreamerV3
        # ``trainable`` path (opt-in) instead co-trains the encoder with the WM
        # while the decoder reconstructs the OBSERVATION (not the embedding) — so
        # the latent is no longer a fixed target but the grounding stays external,
        # which is what forces the latent to encode the full scene (e.g. WHERE the
        # goal is). See ``encode_from_channels`` and ``RSSMWorldModel.decode_obs``.
        self.trainable = bool(trainable)
        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def _structured_features(self, observation: Dict[str, Any]) -> np.ndarray:
        if self.structured_feature_dim == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._structured_features_fn is None:
            return np.zeros((self.structured_feature_dim,), dtype=np.float32)
        features = np.asarray(self._structured_features_fn(observation), dtype=np.float32)
        if features.shape != (self.structured_feature_dim,):
            raise ValueError(
                "structured_features_fn returned shape "
                f"{features.shape}, expected {(self.structured_feature_dim,)}"
            )
        return features

    def _features_from_tensors(
        self, channels: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Projected latent from pre-extracted ``(channels, scalars)`` tensors.

        Grad-tracking follows the ambient context: ``_project`` wraps it in
        ``no_grad`` for inference, ``encode_from_channels`` calls it with grad
        enabled for joint World-Model training.
        """
        device = next(self.parameters()).device
        channels = channels.to(device).float()
        scalars = scalars.to(device).float()
        x = self.conv(channels).flatten(1)
        x = torch.cat([x, scalars], dim=1)
        return self.head(x)

    @torch.no_grad()
    def _project(self, observations) -> torch.Tensor:
        channels, scalars = self._obs_batch_fn(observations)
        return self._features_from_tensors(channels, scalars)

    def encode_from_channels(
        self, channels: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Grad-enabled embed from a *stored* egocentric window (DreamerV3).

        Used by the World-Model training loop when the encoder is ``trainable``:
        the embed fed to the RSSM posterior must be recomputed from the stored
        observation *with* gradient, so the encoder learns (jointly with the
        decoder reconstructing the observation) to encode the whole scene.

        Requires ``structured_feature_dim == 0``: the structured tail comes from
        a separate non-differentiable hook over the full observation and cannot
        be reconstructed from the window tensors alone.
        """
        if self.structured_feature_dim != 0:
            raise ValueError(
                "encode_from_channels requires structured_feature_dim == 0"
            )
        return self._features_from_tensors(channels, scalars)

    @torch.no_grad()
    def encode_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        projected = self._project([observation]).squeeze(0)
        if self.structured_feature_dim == 0:
            return projected
        features = torch.from_numpy(self._structured_features(observation)).to(projected.device)
        return torch.cat([projected, features], dim=0)

    @torch.no_grad()
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        return self.encode_tensor(observation).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, observations) -> np.ndarray:
        projected = self._project(observations)
        if self.structured_feature_dim == 0:
            return projected.cpu().numpy().astype(np.float32)
        features = np.stack([self._structured_features(o) for o in observations])
        features_t = torch.from_numpy(features).to(projected.device)
        return torch.cat([projected, features_t], dim=1).cpu().numpy().astype(np.float32)

    def forward(self, observation: Dict[str, Any]) -> np.ndarray:  # pragma: no cover
        return self.encode(observation)
