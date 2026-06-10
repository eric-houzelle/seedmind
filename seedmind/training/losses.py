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
    predicted_uncertainty: torch.Tensor | None = None,
    reward_weight: float = 1.0,
    reward_sample_weight: torch.Tensor | None = None,
    uncertainty_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Return the total loss and its components."""
    state_error = torch.mean((predicted_next - target_next) ** 2, dim=1)
    state_loss = state_error.mean()
    reward_error = (predicted_reward - target_reward) ** 2
    if reward_sample_weight is not None:
        reward_error = reward_error * reward_sample_weight
    reward_loss = reward_error.mean()
    total = state_loss + reward_weight * reward_loss
    losses = {"total": total, "state": state_loss, "reward": reward_loss}

    if predicted_uncertainty is not None and uncertainty_weight > 0.0:
        target_error = (state_error + reward_weight * reward_error).detach()
        uncertainty_loss = F.smooth_l1_loss(
            torch.log1p(predicted_uncertainty),
            torch.log1p(target_error),
        )
        losses["uncertainty"] = uncertainty_loss
        losses["total"] = losses["total"] + uncertainty_weight * uncertainty_loss
    else:
        losses["uncertainty"] = torch.zeros((), device=target_next.device)

    return losses


def world_model_aux_loss(
    outputs: Dict[str, torch.Tensor],
    target_next: torch.Tensor,
    target_reward: torch.Tensor,
    target_feature_delta: torch.Tensor | None = None,
    target_event: torch.Tensor | None = None,
    event_class_weight: torch.Tensor | None = None,
    reward_sample_weight: torch.Tensor | None = None,
    reward_weight: float = 1.0,
    feature_weight: float = 0.0,
    event_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """World Model loss with optional generic causal heads."""
    state_error = torch.mean((outputs["next_state"] - target_next) ** 2, dim=1)
    state_loss = state_error.mean()
    reward_error = (outputs["reward"] - target_reward) ** 2
    if reward_sample_weight is not None:
        reward_error = reward_error * reward_sample_weight
    reward_loss = reward_error.mean()
    total = state_loss + reward_weight * reward_loss
    losses = {"total": total, "state": state_loss, "reward": reward_loss}

    if (
        target_feature_delta is not None
        and "causal_feature_delta" in outputs
        and feature_weight > 0.0
    ):
        feature_error = torch.mean((outputs["causal_feature_delta"] - target_feature_delta) ** 2, dim=1)
        feature_loss = feature_error.mean()
        losses["feature"] = feature_loss
        losses["total"] = losses["total"] + feature_weight * feature_loss
    else:
        feature_error = torch.zeros_like(state_error)
        losses["feature"] = torch.zeros((), device=target_next.device)

    if target_event is not None and "event_logits" in outputs and event_weight > 0.0:
        event_error = F.cross_entropy(
            outputs["event_logits"], target_event,
            weight=event_class_weight,
            reduction="none",
        )
        event_loss = event_error.mean()
        losses["event"] = event_loss
        losses["total"] = losses["total"] + event_weight * event_loss
    else:
        event_error = torch.zeros_like(state_error)
        losses["event"] = torch.zeros((), device=target_next.device)

    if "uncertainty" in outputs and uncertainty_weight > 0.0:
        target_error = (
            state_error
            + reward_weight * reward_error
            + feature_weight * feature_error
            + event_weight * event_error
        ).detach()
        uncertainty_loss = F.smooth_l1_loss(
            torch.log1p(outputs["uncertainty"]),
            torch.log1p(target_error),
        )
        losses["uncertainty"] = uncertainty_loss
        losses["total"] = losses["total"] + uncertainty_weight * uncertainty_loss
    else:
        losses["uncertainty"] = torch.zeros((), device=target_next.device)

    return losses
