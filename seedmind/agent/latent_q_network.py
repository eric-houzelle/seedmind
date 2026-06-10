"""Latent action-value network.

This Q-network consumes encoder latents directly. It is the first step toward
using the World Model representation as a policy substrate rather than only as
an inference-time planner add-on.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LatentQNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)
        self.num_actions = num_actions

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)

    @torch.no_grad()
    def q_values(self, latent: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        latent_t = torch.as_tensor(latent[None, :], dtype=torch.float32, device=device)
        return self.forward(latent_t).squeeze(0).cpu().numpy().astype(np.float32)
