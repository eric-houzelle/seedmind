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
