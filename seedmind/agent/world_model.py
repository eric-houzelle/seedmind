"""World Model (SPEC section 12).

Learns ``(latent_state, action) -> (next_latent_state, reward, uncertainty)``.
Trained on experiences collected by the agent. The predicted uncertainty is a
positive scalar (softplus) the planner/curiosity can use later.

``RecurrentWorldModel`` (RSSM trajectory, deterministic variant) adds a GRU
recurrent state ``h_t`` that integrates encoded observations over time, giving
the agent memory beyond its (egocentric) field of view.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seedmind.agent.rssm import RSSM, State
from seedmind.agent.value_model import TwoHotCritic


class WorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        causal_feature_dim: int = 0,
        num_events: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.causal_feature_dim = int(causal_feature_dim)
        self.num_events = int(num_events)

        input_dim = latent_dim + num_actions
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.next_state_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.causal_feature_delta_head = (
            nn.Linear(hidden_dim, self.causal_feature_dim)
            if self.causal_feature_dim > 0 else None
        )
        self.event_head = (
            nn.Linear(hidden_dim, self.num_events)
            if self.num_events > 0 else None
        )

    def _action_onehot(self, action_index: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action_index.long(), num_classes=self.num_actions).float()

    def _trunk_features(self, latent: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
        onehot = self._action_onehot(action_index)
        x = torch.cat([latent, onehot], dim=-1)
        return self.trunk(x)

    def forward(
        self,
        latent: torch.Tensor,
        action_index: torch.Tensor,
        detach_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._trunk_features(latent, action_index)
        next_state = self.next_state_head(h)
        reward = self.reward_head(h).squeeze(-1)
        uncertainty_h = h.detach() if detach_uncertainty else h
        uncertainty = F.softplus(self.uncertainty_head(uncertainty_h)).squeeze(-1)
        return next_state, reward, uncertainty

    def forward_aux(
        self,
        latent: torch.Tensor,
        action_index: torch.Tensor,
        detach_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self._trunk_features(latent, action_index)
        uncertainty_h = h.detach() if detach_uncertainty else h
        out: dict[str, torch.Tensor] = {
            "next_state": self.next_state_head(h),
            "reward": self.reward_head(h).squeeze(-1),
            "uncertainty": F.softplus(self.uncertainty_head(uncertainty_h)).squeeze(-1),
        }
        if self.causal_feature_delta_head is not None:
            out["causal_feature_delta"] = self.causal_feature_delta_head(h)
        if self.event_head is not None:
            out["event_logits"] = self.event_head(h)
        return out

    @torch.no_grad()
    def predict_tensor(
        self, latent: torch.Tensor, action_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step prediction; latents stay on the module device."""
        self.eval()
        device = next(self.parameters()).device
        latent_t = latent.to(device)
        if latent_t.dim() == 1:
            latent_t = latent_t.unsqueeze(0)
        action_t = torch.as_tensor([action_index], dtype=torch.long, device=device)
        next_state, reward, uncertainty = self.forward(latent_t, action_t)
        return next_state.squeeze(0), reward.squeeze(0), uncertainty.squeeze(0)

    @torch.no_grad()
    def predict(
        self, latent: np.ndarray, action_index: int
    ) -> Tuple[np.ndarray, float, float]:
        """Single-step prediction from numpy inputs."""
        device = next(self.parameters()).device
        latent_t = torch.as_tensor(latent, dtype=torch.float32, device=device)
        next_state, reward, uncertainty = self.predict_tensor(latent_t, action_index)
        return (
            next_state.cpu().numpy().astype(np.float32),
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

    @torch.no_grad()
    def predict_causal_batch(
        self, latents: np.ndarray, action_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict optional structured causal outputs for planner diagnostics."""
        self.eval()
        device = next(self.parameters()).device
        latents_t = torch.as_tensor(latents, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(action_indices, dtype=torch.long, device=device)
        out = self.forward_aux(latents_t, actions_t)
        if "causal_feature_delta" in out:
            delta = out["causal_feature_delta"].cpu().numpy().astype(np.float32)
        else:
            delta = np.zeros((len(latents), 0), dtype=np.float32)
        if "event_logits" in out:
            events = out["event_logits"].cpu().numpy().astype(np.float32)
        else:
            events = np.zeros((len(latents), 0), dtype=np.float32)
        return delta, events


class RecurrentWorldModel(nn.Module):
    """Deterministic recurrent world model (the "M" of World Models).

    A GRU keeps a deterministic state ``h_t`` that integrates the encoded
    observations over time, so the agent remembers what is now outside its
    (egocentric) field of view::

        filtering:    h_t        = GRU([z_t, a_{t-1}], h_{t-1})   z_t = encoder(obs_t)
        transition:   ẑ_{t+1}, r̂ = heads(h_t, a_t)
        imagination:  h_{t+1}     = GRU([ẑ_{t+1}, a_t], h_t)      (feeds its own prediction)

    The encoder stays frozen, so the predicted next latent ``z`` remains a stable
    target — same rationale as the feed-forward :class:`WorldModel`. The model
    itself is trainable. The latent ``z`` is deterministic (no stochastic prior /
    KL); the stochastic RSSM variant is a later upgrade.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        deter_dim: int = 128,
        num_layers: int = 2,
        causal_feature_dim: int = 0,
        num_events: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_actions = int(num_actions)
        self.deter_dim = int(deter_dim)
        self.causal_feature_dim = int(causal_feature_dim)
        self.num_events = int(num_events)

        # Recurrence: integrate the current latent + last action into h.
        self.gru = nn.GRUCell(self.latent_dim + self.num_actions, self.deter_dim)

        # Transition trunk: (h, action) -> features for the prediction heads.
        input_dim = self.deter_dim + self.num_actions
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        self.next_state_head = nn.Linear(hidden_dim, self.latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.causal_feature_delta_head = (
            nn.Linear(hidden_dim, self.causal_feature_dim)
            if self.causal_feature_dim > 0 else None
        )
        self.event_head = (
            nn.Linear(hidden_dim, self.num_events) if self.num_events > 0 else None
        )

    # -- state -----------------------------------------------------------
    def initial_state(self, batch_size: int = 1, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.deter_dim, device=device)

    def _action_onehot(self, action_index: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action_index.long(), num_classes=self.num_actions).float()

    # -- recurrence (filtering) -----------------------------------------
    def observe_step(
        self, latent: torch.Tensor, prev_action_index: torch.Tensor, h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the recurrent state by integrating ``latent`` (``z_t``).

        ``latent`` is the *observed* encoded state and ``prev_action_index`` the
        action that led here. Returns the new ``h_t``.
        """
        x = torch.cat([latent, self._action_onehot(prev_action_index)], dim=-1)
        return self.gru(x, h_prev)

    # -- transition heads ------------------------------------------------
    def _trunk_features(self, h: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, self._action_onehot(action_index)], dim=-1)
        return self.trunk(x)

    def forward(
        self, h: torch.Tensor, action_index: torch.Tensor, detach_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._trunk_features(h, action_index)
        next_state = self.next_state_head(feat)
        reward = self.reward_head(feat).squeeze(-1)
        unc_feat = feat.detach() if detach_uncertainty else feat
        uncertainty = F.softplus(self.uncertainty_head(unc_feat)).squeeze(-1)
        return next_state, reward, uncertainty

    def forward_aux(
        self, h: torch.Tensor, action_index: torch.Tensor, detach_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feat = self._trunk_features(h, action_index)
        unc_feat = feat.detach() if detach_uncertainty else feat
        out: Dict[str, torch.Tensor] = {
            "next_state": self.next_state_head(feat),
            "reward": self.reward_head(feat).squeeze(-1),
            "uncertainty": F.softplus(self.uncertainty_head(unc_feat)).squeeze(-1),
        }
        if self.causal_feature_delta_head is not None:
            out["causal_feature_delta"] = self.causal_feature_delta_head(feat)
        if self.event_head is not None:
            out["event_logits"] = self.event_head(feat)
        return out

    # -- imagination (planner) ------------------------------------------
    @torch.no_grad()
    def imagine_batch(
        self, h: torch.Tensor, action_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagined step: predict next latent, then advance ``h`` with it.

        Returns ``(h_next, z_next, reward, uncertainty)``. Used to roll
        particles forward in the planner — each particle carries its own ``h``.
        """
        self.eval()
        next_state, reward, uncertainty = self.forward(h, action_index)
        h_next = self.observe_step(next_state, action_index, h)
        return h_next, next_state, reward, uncertainty

    @torch.no_grad()
    def recur_np(
        self, latent: np.ndarray, prev_action_index: int, h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Filtering step from a numpy latent; keeps ``h`` on the module device."""
        self.eval()
        device = next(self.parameters()).device
        z = torch.as_tensor(latent, dtype=torch.float32, device=device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if h_prev is None:
            h_prev = self.initial_state(z.shape[0], device=device)
        a = torch.as_tensor([int(prev_action_index)], dtype=torch.long, device=device)
        return self.observe_step(z, a, h_prev)


class RSSMWorldModel(nn.Module):
    """Stochastic RSSM world model (DreamerV3) — phase 1 brick 2.

    Wraps the :class:`RSSM` core with the prediction heads, all on the model
    feature ``feat = [flatten(z), h]``:
      - ``decoder``     : reconstruct the (frozen) encoder embedding from feat —
        this is the world-model's grounding signal (replaces "predict next latent").
      - ``reward_head`` : scalar reward (→ two-hot in phase 2).
      - ``uncertainty`` / ``causal_feature_delta`` / ``event`` heads: as before.

    Filtering uses the posterior (sees the embedding); imagination uses the prior
    (``z`` sampled from ``h`` alone). The KL between them — trained in phase 3 — is
    what lets the prior imagine plausible rollouts.
    """

    def __init__(
        self,
        embed_dim: int,
        num_actions: int,
        stoch: int = 32,
        discrete: int = 32,
        deter: int = 256,
        hidden: int = 256,
        unimix: float = 0.01,
        causal_feature_dim: int = 0,
        num_events: int = 0,
        reward_twohot: bool = True,
        reward_bins: int = 255,
        reward_vmax: float = 20.0,
        obs_recon: bool = False,
        obs_channels: int = 0,
        obs_window: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_actions = int(num_actions)
        self.causal_feature_dim = int(causal_feature_dim)
        self.num_events = int(num_events)
        self._reward_twohot = bool(reward_twohot)
        self.rssm = RSSM(self.embed_dim, self.num_actions, stoch, discrete, deter, hidden, unimix)
        feat = self.rssm.feat_dim

        self.decoder = nn.Sequential(
            nn.Linear(feat, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
        )
        # Observation decoder (DreamerV3 grounding, opt-in). The default decoder
        # above reconstructs the (frozen) *embedding* — which can never recover
        # information the frozen random projection threw away (e.g. the goal
        # position). Reconstructing the *observation* window instead forces feat
        # — hence z, hence the policy's input — to encode the whole scene. Paired
        # with a trainable encoder, this is the standard DreamerV3 world model.
        self.obs_recon = bool(obs_recon)
        self.obs_channels = int(obs_channels)
        self.obs_window = int(obs_window)
        if self.obs_recon:
            obs_flat = self.obs_channels * self.obs_window * self.obs_window
            self.decoder_obs = nn.Sequential(
                nn.Linear(feat, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, obs_flat),
            )
        # Reward predictor: a two-hot categorical head (DreamerV3) captures the rare
        # foraging-reward spikes that a scalar MSE head regresses to the mean (the
        # probe showed the scalar head predicting ~0 for INTERACT-on-water).
        if self._reward_twohot:
            self.reward_head = TwoHotCritic(feat, hidden_dim=hidden, num_layers=2,
                                            num_bins=int(reward_bins), vmax=float(reward_vmax))
        else:
            self.reward_head = nn.Sequential(
                nn.Linear(feat, hidden), nn.SiLU(), nn.Linear(hidden, 1),
            )
        self.uncertainty_head = nn.Linear(feat, 1)
        self.continue_head = nn.Linear(feat, 1)  # done predictor (trained in phase 2)
        self.causal_feature_delta_head = (
            nn.Linear(feat, self.causal_feature_dim) if self.causal_feature_dim > 0 else None
        )
        self.event_head = nn.Linear(feat, self.num_events) if self.num_events > 0 else None

    # -- state -----------------------------------------------------------
    @property
    def deter_dim(self) -> int:
        return self.rssm.deter

    @property
    def feat_dim(self) -> int:
        return self.rssm.feat_dim

    def initial_state(self, batch_size: int = 1, device=None) -> State:
        return self.rssm.initial_state(batch_size, device=device)

    def get_feat(self, state: State) -> torch.Tensor:
        return self.rssm.get_feat(state)

    # -- recurrence ------------------------------------------------------
    def observe_step(
        self, embed: torch.Tensor, prev_action: torch.Tensor, prev_state: State, sample: bool = True,
    ) -> Tuple[State, State]:
        """Posterior filtering step. Returns ``(post, prior)`` (prior kept for KL)."""
        return self.rssm.obs_step(prev_state, prev_action, embed, sample=sample)

    def img_step(self, prev_state: State, action: torch.Tensor, sample: bool = True) -> State:
        """Prior imagination step (``z`` sampled from ``h`` alone)."""
        return self.rssm.img_step(prev_state, action, sample=sample)

    # -- reward predictor (two-hot or scalar) ----------------------------
    def reward_value(self, feat: torch.Tensor) -> torch.Tensor:
        """Predicted reward in reward space."""
        if self._reward_twohot:
            return self.reward_head.value(feat)
        return self.reward_head(feat).squeeze(-1)

    def continue_prob(self, feat: torch.Tensor) -> torch.Tensor:
        """Predicted probability the episode continues (1 − P(done)) from this state."""
        return torch.sigmoid(self.continue_head(feat).squeeze(-1))

    def reward_loss(self, feat: torch.Tensor, target_reward: torch.Tensor) -> torch.Tensor:
        if self._reward_twohot:
            return self.reward_head.twohot_loss(feat, target_reward)
        return ((self.reward_head(feat).squeeze(-1) - target_reward) ** 2).mean()

    def decode_obs(self, feat: torch.Tensor) -> torch.Tensor:
        """Reconstruct the egocentric observation window ``(B, C, W, W)`` from feat.

        Only available when ``obs_recon`` is enabled. This is the DreamerV3
        grounding signal that forces the latent to encode the full scene.
        """
        out = self.decoder_obs(feat)
        return out.view(-1, self.obs_channels, self.obs_window, self.obs_window)

    # -- heads -----------------------------------------------------------
    def heads(self, feat: torch.Tensor, detach_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        unc_feat = feat.detach() if detach_uncertainty else feat
        out: Dict[str, torch.Tensor] = {
            "recon": self.decoder(feat),
            "reward": self.reward_value(feat),
            "continue": self.continue_head(feat).squeeze(-1),
            "uncertainty": F.softplus(self.uncertainty_head(unc_feat)).squeeze(-1),
        }
        if self.causal_feature_delta_head is not None:
            out["causal_feature_delta"] = self.causal_feature_delta_head(feat)
        if self.event_head is not None:
            out["event_logits"] = self.event_head(feat)
        return out

    # -- imagination -----------------------------------------------------
    @torch.no_grad()
    def imagine_batch(
        self, state: State, action: torch.Tensor,
    ) -> Tuple[State, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagined step under the prior. Returns ``(next_state, feat, reward, uncertainty)``."""
        self.eval()
        prior = self.img_step(state, action)
        feat = self.get_feat(prior)
        reward = self.reward_value(feat)
        uncertainty = F.softplus(self.uncertainty_head(feat)).squeeze(-1)
        return prior, feat, reward, uncertainty
