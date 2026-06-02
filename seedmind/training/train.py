"""World Model training loop (SPEC sections 12 & 17).

Trains the World Model on batches sampled from the Experience Buffer. The
frozen encoder provides stable latent targets, so the loss should decrease as
the model learns to predict next latent states and rewards.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.losses import world_model_loss


def _assemble_batch(batch: List[Dict[str, Any]]):
    """Build training tensors from a list of experiences.

    Skips experiences that lack cached latent vectors.
    """
    latents, actions, next_latents, rewards = [], [], [], []
    for e in batch:
        if e.get("latent_state") is None or e.get("next_latent_state") is None:
            continue
        if e.get("action_index") is None:
            continue
        latents.append(np.asarray(e["latent_state"], dtype=np.float32))
        next_latents.append(np.asarray(e["next_latent_state"], dtype=np.float32))
        actions.append(int(e["action_index"]))
        rewards.append(float(e["reward_external"]))

    if not latents:
        return None

    return (
        torch.from_numpy(np.stack(latents)),
        torch.tensor(actions, dtype=torch.long),
        torch.from_numpy(np.stack(next_latents)),
        torch.tensor(rewards, dtype=torch.float32),
    )


def train_world_model(
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    num_updates: int = 1,
    sampler: str = "mixed",
) -> Dict[str, float]:
    """Run ``num_updates`` gradient steps; return mean loss components."""
    if len(buffer) == 0:
        return {"total": 0.0, "state": 0.0, "reward": 0.0, "updates": 0.0}

    world_model.train()
    totals = {"total": 0.0, "state": 0.0, "reward": 0.0}
    done = 0

    for u in range(num_updates):
        if sampler == "recent":
            batch = buffer.sample_recent(batch_size)
        elif sampler == "high_error":
            batch = buffer.sample_high_error(batch_size)
        elif sampler == "mixed":
            # Blend uniform, high-error and high-reward transitions so the
            # model learns both the dynamics and the (sparse) reward structure.
            third = max(1, batch_size // 3)
            batch = (
                buffer.sample(third)
                + buffer.sample_high_error(third)
                + buffer.sample_high_reward(batch_size - 2 * third)
            )
        else:
            batch = buffer.sample(batch_size)

        assembled = _assemble_batch(batch)
        if assembled is None:
            continue
        latents, actions, next_latents, rewards = assembled

        predicted_next, predicted_reward, _ = world_model(latents, actions)
        losses = world_model_loss(predicted_next, next_latents, predicted_reward, rewards)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        for k in totals:
            totals[k] += float(losses[k].item())
        done += 1

    if done == 0:
        return {"total": 0.0, "state": 0.0, "reward": 0.0, "updates": 0.0}

    return {
        "total": totals["total"] / done,
        "state": totals["state"] / done,
        "reward": totals["reward"] / done,
        "updates": float(done),
    }


def make_optimizer(world_model: WorldModel, learning_rate: float = 3e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(world_model.parameters(), lr=learning_rate)
