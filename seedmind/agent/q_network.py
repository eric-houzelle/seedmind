"""Learned action-value network (SPEC sections 15 & 24, V2).

A small CNN over the one-hot grid (plus inventory flags and the held key
color) that outputs a Q-value per action. Unlike the frozen V1 encoder, this
network *learns its own representation*, which is what lets the policy finally
exploit what it perceives. It plugs into the existing epsilon-greedy policy via
``make_scorer`` and is trained by TD-learning in ``seedmind.training.dqn``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from seedmind.agent.encoder import (
    QNET_INVENTORY_DIM,
    QNET_NUM_CHANNELS,
    observation_qnet_channels,
    observation_qnet_inventory,
)


def obs_batch_to_tensors(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert observations into color-agnostic ``(channels, inventory)`` tensors."""
    channels = np.stack([observation_qnet_channels(o) for o in observations])
    inventory = np.stack([observation_qnet_inventory(o) for o in observations])
    return torch.from_numpy(channels), torch.from_numpy(inventory)


class QNetwork(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_actions: int,
        conv_channels: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.num_channels = QNET_NUM_CHANNELS

        # The inventory (notably the held key color value) is broadcast as
        # constant channels into the conv input, so a convolutional filter can
        # directly compare each door's color value to the key color value
        # (FiLM-style conditioning). Combined with the shared color-value
        # channel this makes the color-matching rule transfer to unseen colors.
        self.conv = nn.Sequential(
            nn.Conv2d(QNET_NUM_CHANNELS + QNET_INVENTORY_DIM, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = conv_channels * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Linear(conv_out + QNET_INVENTORY_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, channels: torch.Tensor, inventory: torch.Tensor) -> torch.Tensor:
        h, w = channels.shape[-2:]
        inv_map = inventory[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([channels, inv_map], dim=1)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.cat([x, inventory], dim=1)
        return self.head(x)

    @torch.no_grad()
    def q_values(self, observation: Dict[str, Any]) -> np.ndarray:
        """Q-values (one per action index) for a single observation."""
        self.eval()
        channels, inventory = obs_batch_to_tensors([observation])
        return self.forward(channels, inventory).squeeze(0).numpy().astype(np.float32)

    def make_scorer(
        self, observation: Dict[str, Any], actions: List[str]
    ) -> Callable[[str], float]:
        """Return a scorer ``action -> Q-value`` for the epsilon-greedy policy."""
        values = self.q_values(observation)
        action_index = {a: i for i, a in enumerate(actions)}
        return lambda action: float(values[action_index[action]])
