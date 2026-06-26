"""Stochastic RSSM core (DreamerV3) — phase 1 of the faithful port.

The model state is a pair ``(h, z)``:
  - ``h`` (``deter``)  : deterministic GRU state, integrates history.
  - ``z`` (``stoch × discrete``) : a **stochastic categorical** latent, sampled
    each step. The stochasticity + a KL between the prior (predicted from ``h``
    alone) and the posterior (from ``h`` + the observation embedding) is what
    makes the learned dynamics robust enough to imagine from — exactly the piece
    our deterministic ``RecurrentWorldModel`` lacked (probes showed the value /
    advantage estimate was unreliable without it).

Faithful to DreamerV3:
  - ``img_step``  : prior   p(z_t | h_t),       h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
  - ``obs_step``  : posterior q(z_t | h_t, e_t),  e_t = encoder(obs_t)
  - categorical ``z`` with a uniform-mixture (``unimix``) and straight-through
    gradients (``rsample``), KL-balancing (dyn/rep) with free bits.

This module is intentionally standalone (no decoder / heads here) so it can be
unit-tested in isolation before it is wired into the world model.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

State = Dict[str, torch.Tensor]  # {"deter": (B,deter), "stoch": (B,stoch,disc), "logits": (B,stoch,disc)}


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_actions: int,
        stoch: int = 32,
        discrete: int = 32,
        deter: int = 256,
        hidden: int = 256,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.stoch = int(stoch)
        self.discrete = int(discrete)
        self.deter = int(deter)
        self.num_actions = int(num_actions)
        self.unimix = float(unimix)
        zdim = self.stoch * self.discrete

        # prior path: [z_{t-1}, a_{t-1}] -> img_in -> GRU -> img_out -> logits
        self.img_in = nn.Sequential(nn.Linear(zdim + self.num_actions, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, self.deter)
        self.img_out = nn.Sequential(nn.Linear(self.deter, hidden), nn.SiLU())
        self.img_logits = nn.Linear(hidden, zdim)
        # posterior path: [h_t, embed_t] -> obs_out -> logits
        self.obs_out = nn.Sequential(nn.Linear(self.deter + int(embed_dim), hidden), nn.SiLU())
        self.obs_logits = nn.Linear(hidden, zdim)

    # -- state -----------------------------------------------------------
    @property
    def feat_dim(self) -> int:
        return self.stoch * self.discrete + self.deter

    def initial_state(self, batch_size: int = 1, device=None) -> State:
        device = device or next(self.parameters()).device
        z = torch.zeros(batch_size, self.stoch, self.discrete, device=device)
        return {"deter": torch.zeros(batch_size, self.deter, device=device),
                "stoch": z, "logits": z.clone()}

    def get_feat(self, state: State) -> torch.Tensor:
        """Model feature [flatten(z), h] fed to the heads / actor / critic."""
        z = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([z, state["deter"]], dim=-1)

    # -- distribution helpers -------------------------------------------
    def _onehot_action(self, action_index: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action_index.long(), num_classes=self.num_actions).float()

    def _dist(self, logits: torch.Tensor) -> Independent:
        """Independent one-hot categorical over the ``stoch`` dim, with unimix."""
        logits = logits.reshape(logits.shape[0], self.stoch, self.discrete)
        probs = torch.softmax(logits, dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.discrete
        return Independent(OneHotCategoricalStraightThrough(probs=probs), 1)

    @staticmethod
    def _mode(dist: Independent) -> torch.Tensor:
        """Straight-through one-hot of the argmax (deterministic eval)."""
        probs = dist.base_dist.probs
        idx = probs.argmax(dim=-1, keepdim=True)
        onehot = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        return onehot + (probs - probs.detach())  # straight-through

    # -- prior (imagination) --------------------------------------------
    def img_step(self, prev: State, prev_action: torch.Tensor, sample: bool = True) -> State:
        z_flat = prev["stoch"].reshape(prev["stoch"].shape[0], -1)
        x = self.img_in(torch.cat([z_flat, self._onehot_action(prev_action)], dim=-1))
        h = self.gru(x, prev["deter"])
        logits = self.img_logits(self.img_out(h))
        dist = self._dist(logits)
        z = dist.rsample() if sample else self._mode(dist)
        return {"deter": h, "stoch": z, "logits": logits.reshape(-1, self.stoch, self.discrete)}

    # -- posterior (filtering) ------------------------------------------
    def obs_step(
        self, prev: State, prev_action: torch.Tensor, embed: torch.Tensor, sample: bool = True,
    ) -> Tuple[State, State]:
        prior = self.img_step(prev, prev_action, sample=sample)
        x = self.obs_out(torch.cat([prior["deter"], embed], dim=-1))
        logits = self.obs_logits(x)
        dist = self._dist(logits)
        z = dist.rsample() if sample else self._mode(dist)
        post = {"deter": prior["deter"], "stoch": z, "logits": logits.reshape(-1, self.stoch, self.discrete)}
        return post, prior

    # -- KL loss (balanced + free bits) ---------------------------------
    def kl_loss(
        self, post: State, prior: State, free: float = 1.0,
        dyn_scale: float = 0.5, rep_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        post_d = self._dist(post["logits"])
        prior_d = self._dist(prior["logits"])
        post_sg = self._dist(post["logits"].detach())
        prior_sg = self._dist(prior["logits"].detach())
        rep = kl_divergence(post_d, prior_sg).clamp(min=free)   # train the posterior toward the prior
        dyn = kl_divergence(post_sg, prior_d).clamp(min=free)   # train the prior toward the posterior
        loss = dyn_scale * dyn.mean() + rep_scale * rep.mean()
        return loss, dyn.mean().detach(), rep.mean().detach()
