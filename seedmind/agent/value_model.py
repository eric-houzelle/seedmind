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


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotCritic(nn.Module):
    """DreamerV3-style value critic: a categorical distribution over fixed bins
    spaced evenly in **symlog** space, trained by cross-entropy on a two-hot
    target.

    Why this over the scalar symlog-MSE critic: MSE regression on a moving,
    unbounded target is badly calibrated (we saw V(h) systematically
    underestimate → every advantage came out positive → the policy gradient
    ranked actions by noise, not by the world model's reward). A categorical
    head over symlog-spaced bins is scale-robust and self-normalising — the same
    bins/hyperparameters work across reward scales and environments, which is
    exactly what makes the value estimate (hence the advantage) trustworthy.
    """

    def __init__(
        self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2,
        num_bins: int = 255, vmax: float = 20.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, int(num_bins)))
        self.net = nn.Sequential(*layers)
        self.register_buffer("bins", torch.linspace(-float(vmax), float(vmax), int(num_bins)))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Logits over the bins, shape ``(..., num_bins)``."""
        return self.net(latent)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        """Expected value in **reward space** (symexp of the symlog-space mean)."""
        probs = torch.softmax(self.forward(latent), dim=-1)
        v_symlog = (probs * self.bins).sum(dim=-1)
        return _symexp(v_symlog)

    def _twohot_target(self, y_symlog: torch.Tensor) -> torch.Tensor:
        """Two-hot encode symlog-space targets ``(N,)`` over the bins → ``(N, num_bins)``."""
        bins = self.bins
        n = bins.numel()
        y = y_symlog.clamp(float(bins[0]), float(bins[-1]))
        hi = torch.searchsorted(bins, y, right=True).clamp(1, n - 1)
        lo = hi - 1
        b_lo, b_hi = bins[lo], bins[hi]
        w_hi = (y - b_lo) / (b_hi - b_lo + 1e-8)
        target = torch.zeros(y.shape[0], n, device=y.device, dtype=bins.dtype)
        target.scatter_(1, lo.unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
        target.scatter_(1, hi.unsqueeze(1), w_hi.unsqueeze(1))
        return target

    def twohot_loss(self, latent: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of the predicted bin distribution vs two-hot(symlog(returns))."""
        logits = self.forward(latent)
        with torch.no_grad():
            target = self._twohot_target(_symlog(target_returns))
        logp = torch.log_softmax(logits, dim=-1)
        return -(target * logp).sum(dim=-1).mean()
