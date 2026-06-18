"""Actor for Dreamer-style policy learning in imagination (RSSM stage 2).

The actor maps a (recurrent) latent state to a categorical action distribution.
It is trained on *imagined* rollouts of the recurrent world model rather than on
real transitions, so it can learn the value of distant goals the world model can
foresee. The critic reuses :class:`seedmind.agent.value_model.ValueModel`
(an MLP from the same state to a scalar value).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Categorical policy over discrete actions from a latent state vector."""

    def __init__(
        self, input_dim: int, num_actions: int, hidden_dim: int = 128, num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, self.num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Action logits for a batch of states ``(B, input_dim) -> (B, num_actions)``."""
        return self.net(state)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(state))

    def evaluate(self, state: torch.Tensor, actions: torch.Tensor):
        """Return ``(log_prob, entropy)`` of ``actions`` under the policy.

        Used by the actor loss (REINFORCE with a value baseline + entropy bonus).
        Differentiable w.r.t. the actor parameters.
        """
        dist = self.distribution(state)
        return dist.log_prob(actions), dist.entropy()

    @torch.no_grad()
    def act(self, state: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        """Sample (or, if ``greedy``, take the argmax) action(s) for state(s)."""
        logits = self.forward(state)
        if greedy:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()

    @torch.no_grad()
    def act_one(self, state_vec: np.ndarray, greedy: bool = False) -> int:
        """Convenience: single-state action index from a numpy vector."""
        device = next(self.parameters()).device
        s = torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        return int(self.act(s, greedy=greedy)[0].item())

    @torch.no_grad()
    def act_masked(self, state_vec: np.ndarray, available_indices, greedy: bool = False) -> int:
        """Single action index restricted to ``available_indices`` (others -inf)."""
        device = next(self.parameters()).device
        s = torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        logits = self.forward(s).squeeze(0)
        mask = torch.full_like(logits, float("-inf"))
        mask[torch.as_tensor(list(available_indices), dtype=torch.long, device=device)] = 0.0
        masked = logits + mask
        if greedy:
            return int(masked.argmax().item())
        return int(torch.distributions.Categorical(logits=masked).sample().item())
