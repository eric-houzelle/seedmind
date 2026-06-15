"""Latent value model for planner terminal evaluation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ValueModel(nn.Module):
    """Predict long-term scalar value from a latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)

    @torch.no_grad()
    def predict_batch(self, latents: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        latents_t = torch.as_tensor(latents, dtype=torch.float32, device=device)
        values = self.forward(latents_t)
        return values.cpu().numpy().astype(np.float32)
