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
from seedmind.training.losses import world_model_aux_loss, world_model_loss


def _assemble_batch(batch: List[Dict[str, Any]]):
    """Build training tensors from a list of experiences.

    Skips experiences that lack cached latent vectors.
    """
    latents, actions, next_latents, rewards, dones = [], [], [], [], []
    feature_deltas, events = [], []
    has_features = True
    has_events = True
    for e in batch:
        if e.get("latent_state") is None or e.get("next_latent_state") is None:
            continue
        if e.get("action_index") is None:
            continue
        latents.append(np.asarray(e["latent_state"], dtype=np.float32))
        next_latents.append(np.asarray(e["next_latent_state"], dtype=np.float32))
        actions.append(int(e["action_index"]))
        rewards.append(float(e["reward_external"]))
        dones.append(1.0 if e.get("done", False) else 0.0)
        if e.get("causal_features") is not None and e.get("next_causal_features") is not None:
            current = np.asarray(e["causal_features"], dtype=np.float32)
            next_current = np.asarray(e["next_causal_features"], dtype=np.float32)
            feature_deltas.append(next_current - current)
        else:
            has_features = False
        if e.get("event_index") is not None:
            events.append(int(e["event_index"]))
        else:
            has_events = False

    if not latents:
        return None

    target_features = None
    if has_features and len(feature_deltas) == len(latents):
        target_features = torch.from_numpy(np.stack(feature_deltas))

    target_events = None
    if has_events and len(events) == len(latents):
        target_events = torch.tensor(events, dtype=torch.long)

    return (
        torch.from_numpy(np.stack(latents)),
        torch.tensor(actions, dtype=torch.long),
        torch.from_numpy(np.stack(next_latents)),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.float32),
        target_features,
        target_events,
    )


def train_world_model(
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    num_updates: int = 1,
    sampler: str = "mixed",
    causal_feature_weight: float = 0.0,
    causal_event_weight: float = 0.0,
    event_class_balance: bool = False,
    event_class_balance_power: float = 0.5,
    reward_abs_weight: float = 0.0,
    reward_done_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
    uncertainty_detach: bool = False,
) -> Dict[str, float]:
    """Run ``num_updates`` gradient steps; return mean loss components."""
    if len(buffer) == 0:
        return {
            "total": 0.0, "state": 0.0, "reward": 0.0,
            "feature": 0.0, "event": 0.0, "uncertainty": 0.0, "updates": 0.0,
        }

    world_model.train()
    totals = {
        "total": 0.0, "state": 0.0, "reward": 0.0,
        "feature": 0.0, "event": 0.0, "uncertainty": 0.0,
    }
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
        elif sampler == "causal":
            half = max(1, batch_size // 2)
            causal = buffer.sample_causal(half)
            batch = buffer.sample(batch_size - len(causal)) + causal
        else:
            batch = buffer.sample(batch_size)

        assembled = _assemble_batch(batch)
        if assembled is None:
            continue
        latents, actions, next_latents, rewards, dones, feature_deltas, events = assembled
        device = next(world_model.parameters()).device
        latents = latents.to(device)
        actions = actions.to(device)
        next_latents = next_latents.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        if feature_deltas is not None:
            feature_deltas = feature_deltas.to(device)
        if events is not None:
            events = events.to(device)
        event_class_weight = None
        if (
            event_class_balance
            and events is not None
            and getattr(world_model, "num_events", 0) > 0
        ):
            counts = torch.bincount(events, minlength=world_model.num_events).float()
            weights = torch.zeros_like(counts)
            present = counts > 0
            weights[present] = 1.0 / torch.clamp(
                counts[present],
                min=1.0,
            ).pow(float(event_class_balance_power))
            if weights[present].numel() > 0:
                weights[present] = weights[present] / weights[present].mean()
            event_class_weight = weights.to(device)

        reward_sample_weight = None
        if reward_abs_weight > 0.0 or reward_done_weight > 0.0:
            reward_sample_weight = (
                torch.ones_like(rewards)
                + float(reward_abs_weight) * torch.abs(rewards)
                + float(reward_done_weight) * dones
            )
            reward_sample_weight = reward_sample_weight / reward_sample_weight.mean().clamp_min(1e-6)

        if causal_feature_weight > 0.0 or causal_event_weight > 0.0:
            outputs = world_model.forward_aux(
                latents,
                actions,
                detach_uncertainty=uncertainty_detach,
            )
            losses = world_model_aux_loss(
                outputs, next_latents, rewards,
                target_feature_delta=feature_deltas,
                target_event=events,
                event_class_weight=event_class_weight,
                reward_sample_weight=reward_sample_weight,
                feature_weight=causal_feature_weight,
                event_weight=causal_event_weight,
                uncertainty_weight=uncertainty_weight,
            )
        else:
            predicted_next, predicted_reward, predicted_uncertainty = world_model(
                latents,
                actions,
                detach_uncertainty=uncertainty_detach,
            )
            losses = world_model_loss(
                predicted_next, next_latents, predicted_reward, rewards,
                predicted_uncertainty=predicted_uncertainty,
                reward_sample_weight=reward_sample_weight,
                uncertainty_weight=uncertainty_weight,
            )
            losses["feature"] = torch.zeros((), device=device)
            losses["event"] = torch.zeros((), device=device)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        for k in totals:
            totals[k] += float(losses[k].item())
        done += 1

    if done == 0:
        return {
            "total": 0.0, "state": 0.0, "reward": 0.0,
            "feature": 0.0, "event": 0.0, "uncertainty": 0.0, "updates": 0.0,
        }

    return {
        "total": totals["total"] / done,
        "state": totals["state"] / done,
        "reward": totals["reward"] / done,
        "feature": totals["feature"] / done,
        "event": totals["event"] / done,
        "uncertainty": totals["uncertainty"] / done,
        "updates": float(done),
    }


def train_world_model_uncertainty_head(
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    num_updates: int = 1,
    sampler: str = "causal",
    causal_feature_weight: float = 0.0,
    causal_event_weight: float = 0.0,
    event_class_balance: bool = False,
    event_class_balance_power: float = 0.5,
    reward_abs_weight: float = 0.0,
    reward_done_weight: float = 0.0,
) -> Dict[str, float]:
    """Calibrate only the uncertainty head against detached WM errors."""
    if len(buffer) == 0:
        return {"uncertainty": 0.0, "updates": 0.0}

    for name, param in world_model.named_parameters():
        param.requires_grad_(name.startswith("uncertainty_head."))
    world_model.train()
    total_uncertainty = 0.0
    done = 0

    for _ in range(num_updates):
        if sampler == "recent":
            batch = buffer.sample_recent(batch_size)
        elif sampler == "high_error":
            batch = buffer.sample_high_error(batch_size)
        elif sampler == "mixed":
            third = max(1, batch_size // 3)
            batch = (
                buffer.sample(third)
                + buffer.sample_high_error(third)
                + buffer.sample_high_reward(batch_size - 2 * third)
            )
        elif sampler == "causal":
            half = max(1, batch_size // 2)
            causal = buffer.sample_causal(half)
            batch = buffer.sample(batch_size - len(causal)) + causal
        else:
            batch = buffer.sample(batch_size)

        assembled = _assemble_batch(batch)
        if assembled is None:
            continue
        latents, actions, next_latents, rewards, dones, feature_deltas, events = assembled
        device = next(world_model.parameters()).device
        latents = latents.to(device)
        actions = actions.to(device)
        next_latents = next_latents.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        if feature_deltas is not None:
            feature_deltas = feature_deltas.to(device)
        if events is not None:
            events = events.to(device)

        event_class_weight = None
        if (
            event_class_balance
            and events is not None
            and getattr(world_model, "num_events", 0) > 0
        ):
            counts = torch.bincount(events, minlength=world_model.num_events).float()
            weights = torch.zeros_like(counts)
            present = counts > 0
            weights[present] = 1.0 / torch.clamp(
                counts[present],
                min=1.0,
            ).pow(float(event_class_balance_power))
            if weights[present].numel() > 0:
                weights[present] = weights[present] / weights[present].mean()
            event_class_weight = weights.to(device)

        reward_sample_weight = None
        if reward_abs_weight > 0.0 or reward_done_weight > 0.0:
            reward_sample_weight = (
                torch.ones_like(rewards)
                + float(reward_abs_weight) * torch.abs(rewards)
                + float(reward_done_weight) * dones
            )
            reward_sample_weight = reward_sample_weight / reward_sample_weight.mean().clamp_min(1e-6)

        outputs = world_model.forward_aux(
            latents,
            actions,
            detach_uncertainty=True,
        )
        losses = world_model_aux_loss(
            outputs, next_latents, rewards,
            target_feature_delta=feature_deltas,
            target_event=events,
            event_class_weight=event_class_weight,
            reward_sample_weight=reward_sample_weight,
            feature_weight=causal_feature_weight,
            event_weight=causal_event_weight,
            uncertainty_weight=1.0,
        )

        optimizer.zero_grad()
        losses["uncertainty"].backward()
        optimizer.step()

        total_uncertainty += float(losses["uncertainty"].item())
        done += 1

    for param in world_model.parameters():
        param.requires_grad_(True)

    if done == 0:
        return {"uncertainty": 0.0, "updates": 0.0}
    return {
        "uncertainty": total_uncertainty / done,
        "updates": float(done),
    }


def make_optimizer(world_model: WorldModel, learning_rate: float = 3e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(world_model.parameters(), lr=learning_rate)
