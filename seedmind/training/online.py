"""Continual (online) learning loop for a from-scratch fouloïde.

Wraps the per-episode training logic of ``scripts/run_micro_fouloide.py``
into a per-step learner: every transition feeds a replay buffer, and every
``update_every`` env steps the world model, DQN and value model get a few
minibatch updates. The planner's uncertainty-gate threshold is refreshed
online from a rolling quantile of recent WM uncertainties (replacing the
posthoc calibration), so the planner stays closed at cold start and opens
as the world model improves.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from seedmind.agent.world_model import RecurrentWorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_dqn,
)
from seedmind.training.recurrent import (
    train_recurrent_dqn,
    train_recurrent_world_model,
)
from seedmind.training.train import (
    make_optimizer,
    train_world_model,
    train_world_model_uncertainty_head,
)
from seedmind.training.value import (
    make_value_optimizer,
    sync_value_target,
    train_value_model,
)


class OnlineLearner:
    """Per-step learner: feed transitions, networks learn continuously."""

    def __init__(
        self,
        agent: Any,
        config: dict,
        device: torch.device,
        buffer: Optional[ExperienceBuffer] = None,
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.config = config
        self.device = device

        oc = config.get("online", {})
        wmc = config.get("world_model", {})
        self._cwm = config.get("causal_world_model", {})
        self._wmc = wmc
        dc = config.get("dqn", {})
        vc = config.get("value_model", {})

        # Recurrent world model → sequence/BPTT training instead of per-transition.
        self.recurrent = isinstance(agent.world_model, RecurrentWorldModel)
        self.seq_len = int(oc.get("seq_len", 16))

        self.update_every = int(oc.get("update_every", 8))
        self.updates_per_cycle = int(oc.get("updates_per_cycle", 4))
        self.warmup_steps = int(oc.get("warmup_steps", 2000))
        self.threshold_refresh_steps = int(oc.get("threshold_refresh_steps", 500))
        self.threshold_quantile = float(oc.get("threshold_quantile", 0.60))
        self.threshold_samples = int(oc.get("threshold_samples", 2000))

        self.wm_batch = int(wmc.get("batch_size", 64))
        self.wm_sampler = str(wmc.get("sampler", "uniform"))
        self.wm_uncertainty_head_updates = int(wmc.get("uncertainty_head_updates_per_train", 0))
        self.q_batch = int(dc.get("batch_size", 64))
        self.gamma = float(dc.get("gamma", 0.97))
        self.target_update = int(dc.get("target_update", 500))
        self.dqn_sampler = str(dc.get("sampler", "uniform"))
        self.curiosity_weight = float(dc.get("curiosity_weight", 0.0))
        self.double_dqn = bool(dc.get("double_dqn", True))
        drive_reward_enabled = bool(config.get("drive_reward", {}).get("enabled", False))
        self.dqn_reward_key = str(dc.get(
            "reward_key",
            "reward_learning" if drive_reward_enabled else "reward_external",
        ))
        self.value_batch = int(vc.get("batch_size", self.q_batch))
        self.value_gamma = float(vc.get("gamma", self.gamma))
        self.value_target_update = int(vc.get("target_update", self.target_update))
        self.value_sampler = str(vc.get("sampler", "uniform"))
        self.value_reward_key = str(vc.get("reward_key", self.dqn_reward_key))
        self._vc = vc

        self.buffer = buffer if buffer is not None else ExperienceBuffer(seed=seed)
        self.wm_optimizer = make_optimizer(
            agent.world_model, learning_rate=float(wmc.get("learning_rate", 3e-4)),
        )
        self.wm_uncertainty_optimizer = torch.optim.Adam(
            agent.world_model.uncertainty_head.parameters(),
            lr=float(wmc.get("uncertainty_head_learning_rate", wmc.get("learning_rate", 3e-4))),
        )
        self.q_optimizer = make_q_optimizer(
            agent.q_network, learning_rate=float(dc.get("learning_rate", 5e-4)),
        )
        self.target_network = make_target_network(agent.q_network)
        self.value_optimizer = None
        self.target_value_model = None
        if agent.value_model is not None:
            self.value_optimizer = make_value_optimizer(
                agent.value_model, learning_rate=float(vc.get("learning_rate", 3e-4)),
            )
            import copy
            self.target_value_model = copy.deepcopy(agent.value_model).to(device)
            self.target_value_model.eval()

        self.env_steps = 0
        self.total_q_updates = 0
        self.total_value_updates = 0
        self.next_target_sync = self.target_update
        self.next_value_target_sync = self.value_target_update
        self.last_wm_loss = 0.0
        self.last_wm_uncertainty_loss = 0.0
        self.last_td_loss = 0.0
        self.last_value_loss = 0.0
        self.uncertainty_threshold: Optional[float] = None

        # Cold start: WM uncertainty is softplus (> 0), so a zero threshold
        # keeps the planner gate closed until the first refresh after warmup.
        if getattr(agent, "use_planner", False):
            agent.planner_uncertainty_threshold = 0.0

    # ------------------------------------------------------------------
    def observe(self, experience: Dict[str, Any]) -> None:
        """Record one transition and learn if a cycle boundary is reached."""
        self.buffer.add(experience)
        self.env_steps += 1
        if self.env_steps % self.update_every == 0 and len(self.buffer) >= self.q_batch:
            self._update_models()
        if (
            self.env_steps >= self.warmup_steps
            and self.env_steps % self.threshold_refresh_steps == 0
        ):
            self._refresh_uncertainty_threshold()

    # ------------------------------------------------------------------
    def _update_models_recurrent(self) -> None:
        """BPTT world model + DRQN Q on sampled sequences (recurrent WM)."""
        cwm = self._cwm
        wmc = self._wmc
        wm_losses = train_recurrent_world_model(
            self.agent.world_model, self.buffer, self.wm_optimizer,
            batch_size=self.wm_batch, seq_len=self.seq_len,
            num_updates=self.updates_per_cycle,
            causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
            causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
            event_class_balance=bool(cwm.get("event_class_balance", False)),
            event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
            uncertainty_weight=float(wmc.get("uncertainty_loss_weight", 0.0)),
            uncertainty_detach=bool(wmc.get("uncertainty_detach", False)),
        )
        self.last_wm_loss = float(wm_losses["total"])
        q_losses = train_recurrent_dqn(
            self.agent.q_network, self.target_network, self.agent.world_model,
            self.buffer, self.q_optimizer,
            batch_size=self.q_batch, seq_len=self.seq_len, gamma=self.gamma,
            curiosity_weight=self.curiosity_weight, double_dqn=self.double_dqn,
            num_updates=self.updates_per_cycle, reward_key=self.dqn_reward_key,
        )
        self.last_td_loss = float(q_losses["td_loss"])
        self.total_q_updates += int(q_losses["updates"])
        if self.total_q_updates >= self.next_target_sync:
            sync_target(self.agent.q_network, self.target_network)
            self.next_target_sync += self.target_update

    # ------------------------------------------------------------------
    def _update_models(self) -> None:
        if self.recurrent:
            self._update_models_recurrent()
            return
        cwm = self._cwm
        wmc = self._wmc
        wm_losses = train_world_model(
            self.agent.world_model, self.buffer, self.wm_optimizer,
            batch_size=self.wm_batch, num_updates=self.updates_per_cycle,
            sampler=self.wm_sampler,
            causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
            causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
            event_class_balance=bool(cwm.get("event_class_balance", False)),
            event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
            reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
            reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
            event_sample_names=set(cwm.get("event_sample_names", [])),
            event_sample_name_weight=float(cwm.get("event_sample_name_weight", 0.0)),
            event_sample_done_weight=float(cwm.get("event_sample_done_weight", 0.0)),
            event_sample_reward_abs_weight=float(cwm.get("event_sample_reward_abs_weight", 0.0)),
            uncertainty_weight=float(wmc.get("uncertainty_loss_weight", 0.0)),
            uncertainty_detach=bool(wmc.get("uncertainty_detach", False)),
        )
        self.last_wm_loss = float(wm_losses["total"])
        if self.wm_uncertainty_head_updates > 0:
            uncertainty_losses = train_world_model_uncertainty_head(
                self.agent.world_model, self.buffer, self.wm_uncertainty_optimizer,
                batch_size=int(wmc.get("uncertainty_head_batch_size", self.wm_batch)),
                num_updates=self.wm_uncertainty_head_updates,
                sampler=str(wmc.get("uncertainty_head_sampler", self.wm_sampler)),
                causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
                causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
                event_class_balance=bool(cwm.get("event_class_balance", False)),
                event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
                reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
                reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
            )
            self.last_wm_uncertainty_loss = float(uncertainty_losses["uncertainty"])
        q_losses = train_dqn(
            self.agent.q_network, self.target_network, self.buffer, self.q_optimizer,
            batch_size=self.q_batch, gamma=self.gamma,
            curiosity_weight=self.curiosity_weight,
            double_dqn=self.double_dqn,
            num_updates=self.updates_per_cycle,
            sampler=self.dqn_sampler,
            reward_key=self.dqn_reward_key,
        )
        self.last_td_loss = float(q_losses["td_loss"])
        self.total_q_updates += int(q_losses["updates"])
        if self.total_q_updates >= self.next_target_sync:
            sync_target(self.agent.q_network, self.target_network)
            self.next_target_sync += self.target_update
        if self.agent.value_model is not None and self.value_optimizer is not None:
            value_losses = train_value_model(
                self.agent.value_model, self.target_value_model, self.buffer,
                self.value_optimizer,
                batch_size=self.value_batch,
                gamma=self.value_gamma,
                num_updates=self.updates_per_cycle,
                sampler=self.value_sampler,
                reward_key=self.value_reward_key,
                target_abs_weight=float(self._vc.get("target_abs_weight", 0.0)),
                terminal_weight=float(self._vc.get("terminal_weight", 0.0)),
                td_error_weight=float(self._vc.get("td_error_weight", 0.0)),
                max_weight=float(self._vc.get("max_weight", 10.0)),
            )
            self.last_value_loss = float(value_losses["value_loss"])
            self.total_value_updates += int(value_losses["updates"])
            if self.total_value_updates >= self.next_value_target_sync:
                sync_value_target(self.agent.value_model, self.target_value_model)
                self.next_value_target_sync += self.value_target_update

    # ------------------------------------------------------------------
    def _refresh_uncertainty_threshold(self) -> None:
        # The planner (sole consumer of this threshold) is disabled for the
        # recurrent WM, which also has no predict_batch — skip entirely.
        if self.recurrent:
            return
        rows = [
            row for row in self.buffer.sample_recent(self.threshold_samples)
            if row.get("latent_state") is not None and row.get("action_index") is not None
        ]
        if not rows:
            return
        latents = np.stack([
            np.asarray(row["latent_state"], dtype=np.float32) for row in rows
        ])
        actions = np.asarray([int(row["action_index"]) for row in rows], dtype=np.int64)
        uncertainties = []
        for start in range(0, len(rows), 4096):
            _, _, uncertainty = self.agent.world_model.predict_batch(
                latents[start:start + 4096],
                actions[start:start + 4096],
            )
            uncertainties.append(uncertainty)
        threshold = float(np.quantile(np.concatenate(uncertainties), self.threshold_quantile))
        self.uncertainty_threshold = threshold
        if getattr(self.agent, "use_planner", False):
            self.agent.planner_uncertainty_threshold = threshold

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "env_steps": self.env_steps,
            "buffer_size": len(self.buffer),
            "wm_loss": self.last_wm_loss,
            "wm_uncertainty_loss": self.last_wm_uncertainty_loss,
            "td_loss": self.last_td_loss,
            "value_loss": self.last_value_loss,
            "q_updates": self.total_q_updates,
            "uncertainty_threshold": self.uncertainty_threshold,
            "epsilon": float(self.agent.policy.epsilon),
        }
