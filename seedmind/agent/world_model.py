"""World Model (SPEC section 12).

Learns ``(latent_state, action) -> (next_latent_state, reward, uncertainty)``.
Trained on experiences collected by the agent. The predicted uncertainty is a
positive scalar (softplus) the planner/curiosity can use later.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        input_dim = latent_dim + num_actions
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.next_state_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def _action_onehot(self, action_index: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action_index.long(), num_classes=self.num_actions).float()

    def forward(
        self, latent: torch.Tensor, action_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        onehot = self._action_onehot(action_index)
        x = torch.cat([latent, onehot], dim=-1)
        h = self.trunk(x)
        next_state = self.next_state_head(h)
        reward = self.reward_head(h).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(h)).squeeze(-1)
        return next_state, reward, uncertainty

    @torch.no_grad()
    def predict(
        self, latent: np.ndarray, action_index: int
    ) -> Tuple[np.ndarray, float, float]:
        """Single-step prediction from numpy inputs."""
        self.eval()
        device = next(self.parameters()).device
        latent_t = torch.as_tensor(latent, dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.as_tensor([action_index], dtype=torch.long, device=device)
        next_state, reward, uncertainty = self.forward(latent_t, action_t)
        return (
            next_state.squeeze(0).cpu().numpy().astype(np.float32),
            float(reward.item()),
            float(uncertainty.item()),
        )

    @torch.no_grad()
    def predict_batch(
        self, latents: np.ndarray, action_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised multi-particle prediction (used by the planner)."""
        self.eval()
        device = next(self.parameters()).device
        latents_t = torch.as_tensor(latents, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(action_indices, dtype=torch.long, device=device)
        next_state, reward, uncertainty = self.forward(latents_t, actions_t)
        return (
            next_state.cpu().numpy().astype(np.float32),
            reward.cpu().numpy().astype(np.float32),
            uncertainty.cpu().numpy().astype(np.float32),
        )
