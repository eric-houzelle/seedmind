"""World Model losses (SPEC section 12).

``loss = prediction_state_loss + reward_prediction_loss``
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def world_model_loss(
    predicted_next: torch.Tensor,
    target_next: torch.Tensor,
    predicted_reward: torch.Tensor,
    target_reward: torch.Tensor,
    reward_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return the total loss and its components."""
    state_loss = F.mse_loss(predicted_next, target_next)
    reward_loss = F.mse_loss(predicted_reward, target_reward)
    total = state_loss + reward_weight * reward_loss
    return {"total": total, "state": state_loss, "reward": reward_loss}


def world_model_aux_loss(
    outputs: Dict[str, torch.Tensor],
    target_next: torch.Tensor,
    target_reward: torch.Tensor,
    target_feature_delta: torch.Tensor | None = None,
    target_event: torch.Tensor | None = None,
    reward_weight: float = 1.0,
    feature_weight: float = 0.0,
    event_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """World Model loss with optional generic causal heads."""
    state_loss = F.mse_loss(outputs["next_state"], target_next)
    reward_loss = F.mse_loss(outputs["reward"], target_reward)
    total = state_loss + reward_weight * reward_loss
    losses = {"total": total, "state": state_loss, "reward": reward_loss}

    if (
        target_feature_delta is not None
        and "causal_feature_delta" in outputs
        and feature_weight > 0.0
    ):
        feature_loss = F.mse_loss(outputs["causal_feature_delta"], target_feature_delta)
        losses["feature"] = feature_loss
        losses["total"] = losses["total"] + feature_weight * feature_loss
    else:
        losses["feature"] = torch.zeros((), device=target_next.device)

    if target_event is not None and "event_logits" in outputs and event_weight > 0.0:
        event_loss = F.cross_entropy(outputs["event_logits"], target_event)
        losses["event"] = event_loss
        losses["total"] = losses["total"] + event_weight * event_loss
    else:
        losses["event"] = torch.zeros((), device=target_next.device)

    return losses
