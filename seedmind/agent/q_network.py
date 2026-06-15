"""Learned action-value network (SPEC sections 15 & 24, V2+).

A small CNN over a multi-channel grid (plus scalar features like inventory)
that outputs a Q-value per action. The network is **parametric**: the number
of grid channels and scalar features are constructor arguments so the same
class serves both the colored-gridworld and the sandbox world.

It plugs into the epsilon-greedy policy via ``make_scorer`` and is trained by
TD-learning in ``seedmind.training.dqn``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from seedmind.agent.encoder import (
    QNET_INVENTORY_DIM,
    QNET_NUM_CHANNELS,
    observation_qnet_channels,
    observation_qnet_inventory,
)

# Type alias for the batch-conversion function used by each world.
ObsBatchFn = Callable[[Sequence[Dict[str, Any]]], Tuple[torch.Tensor, torch.Tensor]]


def obs_batch_to_tensors(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert gridworld observations into ``(channels, scalars)`` tensors."""
    channels = np.stack([observation_qnet_channels(o) for o in observations])
    inventory = np.stack([observation_qnet_inventory(o) for o in observations])
    return torch.from_numpy(channels), torch.from_numpy(inventory)


class QNetwork(nn.Module):
    """Parametric CNN Q-network.

    Parameters
    ----------
    grid_size : int
        Side length of the square grid.
    num_actions : int
        Size of the discrete action space.
    conv_channels : int
        Feature maps per convolutional layer.
    hidden_dim : int
        Width of the fully-connected hidden layer.
    num_grid_channels : int or None
        Grid input channels. ``None`` (default) uses the gridworld constant.
    num_scalars : int or None
        Scalar features (inventory, energy...). ``None`` uses the gridworld
        constant. These are broadcast as constant spatial channels into the
        conv input *and* concatenated after the flatten.
    obs_batch_fn : callable or None
        Function that converts a sequence of observation dicts into
        ``(channels_tensor, scalars_tensor)``. ``None`` uses the default
        gridworld converter.
    """

    def __init__(
        self,
        grid_size: int,
        num_actions: int,
        conv_channels: int = 32,
        hidden_dim: int = 128,
        num_grid_channels: Optional[int] = None,
        num_scalars: Optional[int] = None,
        obs_batch_fn: Optional[ObsBatchFn] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.num_channels = num_grid_channels if num_grid_channels is not None else QNET_NUM_CHANNELS
        self.num_scalars = num_scalars if num_scalars is not None else QNET_INVENTORY_DIM
        self._obs_batch_fn = obs_batch_fn or obs_batch_to_tensors

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels + self.num_scalars, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = conv_channels * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Linear(conv_out + self.num_scalars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, channels: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h, w = channels.shape[-2:]
        sc_map = scalars[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([channels, sc_map], dim=1)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.cat([x, scalars], dim=1)
        return self.head(x)

    @torch.no_grad()
    def q_values_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Q-values on the module device (one value per action index)."""
        self.eval()
        channels, scalars = self._obs_batch_fn([observation])
        device = next(self.parameters()).device
        channels = channels.to(device)
        scalars = scalars.to(device)
        return self.forward(channels, scalars).squeeze(0)

    @torch.no_grad()
    def q_values(self, observation: Dict[str, Any]) -> np.ndarray:
        """Q-values (one per action index) for a single observation."""
        return self.q_values_tensor(observation).cpu().numpy().astype(np.float32)

    def make_scorer(
        self,
        observation: Dict[str, Any],
        actions: List[str],
        action_index: Optional[Dict[str, int]] = None,
    ) -> Callable[[str], float]:
        """Return a scorer ``action -> Q-value`` for the epsilon-greedy policy.

        ``action_index`` must map action names to the network's output indices
        (the *full* action space). Without it, indices are inferred by
        enumerating ``actions`` — only correct when ``actions`` is the full,
        unfiltered list (a filtered subset would misalign Q-values).
        """
        values = self.q_values_tensor(observation).detach().cpu().numpy().astype(np.float32)
        if action_index is None:
            action_index = {a: i for i, a in enumerate(actions)}
        return lambda action: float(values[action_index[action]])
