"""Training utilities for latent value models."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from seedmind.agent.value_model import ValueModel
from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer


def make_value_optimizer(value_model: ValueModel, learning_rate: float = 3e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(value_model.parameters(), lr=learning_rate)


def _sample_batch(buffer: ExperienceBuffer, batch_size: int, sampler: str) -> List[Dict[str, Any]]:
    if sampler == "causal":
        half = max(1, batch_size // 2)
        causal = buffer.sample_causal(half)
        return buffer.sample(batch_size - len(causal)) + causal
    if sampler == "recent":
        return buffer.sample_recent(batch_size)
    return buffer.sample(batch_size)


def _reward(experience: Dict[str, Any], reward_key: str) -> float:
    return float(experience.get(reward_key, experience.get("reward_external", 0.0)))


def train_value_model(
    value_model: ValueModel,
    target_value_model: ValueModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    gamma: float = 0.97,
    num_updates: int = 1,
    sampler: str = "uniform",
    reward_key: str = "reward_external",
    grad_clip: float = 10.0,
    target_abs_weight: float = 0.0,
    terminal_weight: float = 0.0,
    td_error_weight: float = 0.0,
    max_weight: float = 10.0,
) -> Dict[str, float]:
    """Train V(s) with one-step TD targets from replay latents."""
    if len(buffer) == 0:
        return {"value_loss": 0.0, "updates": 0.0}

    value_model.train()
    total_loss = 0.0
    updates = 0

    for _ in range(num_updates):
        batch = _sample_batch(buffer, batch_size, sampler)
        rows = [
            e for e in batch
            if e.get("latent_state") is not None and e.get("next_latent_state") is not None
        ]
        if not rows:
            continue

        device = next(value_model.parameters()).device
        latents = torch.as_tensor(
            np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in rows]),
            dtype=torch.float32,
            device=device,
        )
        next_latents = torch.as_tensor(
            np.stack([np.asarray(e["next_latent_state"], dtype=np.float32) for e in rows]),
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.as_tensor(
            [_reward(e, reward_key) for e in rows],
            dtype=torch.float32,
            device=device,
        )
        dones = torch.as_tensor(
            [1.0 if e.get("done", False) else 0.0 for e in rows],
            dtype=torch.float32,
            device=device,
        )

        values = value_model(latents)
        with torch.no_grad():
            next_values = target_value_model(next_latents)
            target = rewards + gamma * next_values * (1.0 - dones)

        element_loss = F.smooth_l1_loss(values, target, reduction="none")
        weights = torch.ones_like(element_loss)
        if target_abs_weight > 0.0:
            weights = weights + float(target_abs_weight) * torch.abs(target)
        if terminal_weight > 0.0:
            weights = weights + float(terminal_weight) * dones
        if td_error_weight > 0.0:
            weights = weights + float(td_error_weight) * torch.abs(target - values.detach())
        if max_weight > 0.0:
            weights = torch.clamp(weights, max=float(max_weight))
        loss = torch.mean(element_loss * weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        updates += 1

    if updates == 0:
        return {"value_loss": 0.0, "updates": 0.0}
    return {"value_loss": total_loss / updates, "updates": float(updates)}


def train_value_model_dyna(
    value_model: ValueModel,
    target_value_model: ValueModel,
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    gamma: float = 0.97,
    num_updates: int = 1,
    sampler: str = "uniform",
    grad_clip: float = 10.0,
    loss_weight: float = 1.0,
) -> Dict[str, float]:
    """Train V(s) against one-step latent transitions imagined by the WM.

    This is a conservative Dyna-style auxiliary update: it does not add
    synthetic observations to replay and does not train the observation Q-model.
    It only asks the value model to become consistent with the dynamics learned
    by the world model on replay states/actions.
    """
    if len(buffer) == 0 or loss_weight <= 0.0:
        return {"value_dyna_loss": 0.0, "updates": 0.0}

    value_model.train()
    world_model.eval()
    target_value_model.eval()
    total_loss = 0.0
    updates = 0

    for _ in range(num_updates):
        batch = _sample_batch(buffer, batch_size, sampler)
        rows = [
            e for e in batch
            if e.get("latent_state") is not None and e.get("action_index") is not None
        ]
        if not rows:
            continue

        value_device = next(value_model.parameters()).device
        wm_device = next(world_model.parameters()).device
        latents = torch.as_tensor(
            np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in rows]),
            dtype=torch.float32,
            device=value_device,
        )
        actions = torch.as_tensor(
            [int(e["action_index"]) for e in rows],
            dtype=torch.long,
            device=wm_device,
        )

        values = value_model(latents)
        with torch.no_grad():
            wm_latents = latents.to(wm_device)
            out = world_model.forward_aux(wm_latents, actions)
            imagined_next = out["next_state"].to(value_device)
            imagined_reward = out["reward"].to(value_device)
            next_values = target_value_model(imagined_next)
            target = imagined_reward + gamma * next_values

        loss = F.smooth_l1_loss(values, target) * float(loss_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        updates += 1

    if updates == 0:
        return {"value_dyna_loss": 0.0, "updates": 0.0}
    return {"value_dyna_loss": total_loss / updates, "updates": float(updates)}


def train_value_model_on_returns(
    value_model: ValueModel,
    latents: np.ndarray,
    returns: np.ndarray,
    optimizer: torch.optim.Optimizer,
    sample_weights: np.ndarray | None = None,
    batch_size: int = 64,
    num_updates: int = 1000,
    grad_clip: float = 10.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Posthoc supervised V(s) calibration against observed discounted returns."""
    if len(latents) == 0 or len(returns) == 0 or num_updates <= 0:
        return {"value_return_loss": 0.0, "updates": 0.0}

    value_model.train()
    rng = np.random.default_rng(seed)
    n = len(latents)
    device = next(value_model.parameters()).device
    latents_np = np.asarray(latents, dtype=np.float32)
    returns_np = np.asarray(returns, dtype=np.float32)
    weights_np = (
        np.asarray(sample_weights, dtype=np.float32)
        if sample_weights is not None
        else np.ones(n, dtype=np.float32)
    )
    if len(weights_np) != n:
        raise ValueError("sample_weights must match latents length.")
    total_loss = 0.0
    updates = 0

    for _ in range(num_updates):
        replace = n < batch_size
        indices = rng.choice(n, size=min(batch_size, n) if not replace else batch_size, replace=replace)
        batch_latents = torch.as_tensor(latents_np[indices], dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns_np[indices], dtype=torch.float32, device=device)
        batch_weights = torch.as_tensor(weights_np[indices], dtype=torch.float32, device=device)

        values = value_model(batch_latents)
        element_loss = F.smooth_l1_loss(values, batch_returns, reduction="none")
        loss = torch.mean(element_loss * batch_weights / torch.clamp(torch.mean(batch_weights), min=1e-6))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        updates += 1

    return {"value_return_loss": total_loss / max(updates, 1), "updates": float(updates)}


@torch.no_grad()
def evaluate_value_model_on_returns(
    value_model: ValueModel,
    latents: np.ndarray,
    returns: np.ndarray,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Evaluate V(s) predictions against observed discounted returns."""
    if len(latents) == 0 or len(returns) == 0:
        return {"mae": float("nan"), "bias": float("nan"), "corr": float("nan")}

    value_model.eval()
    device = next(value_model.parameters()).device
    preds = []
    latents_np = np.asarray(latents, dtype=np.float32)
    returns_np = np.asarray(returns, dtype=np.float32)
    for start in range(0, len(latents_np), batch_size):
        batch = torch.as_tensor(
            latents_np[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        preds.append(value_model(batch).cpu().numpy())
    values = np.concatenate(preds).astype(np.float32)
    errors = values - returns_np
    if len(values) < 2 or float(np.std(values)) == 0.0 or float(np.std(returns_np)) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(values, returns_np)[0, 1])
    return {
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "corr": corr,
        "value_mean": float(np.mean(values)),
        "return_mean": float(np.mean(returns_np)),
    }


def sync_value_target(value_model: ValueModel, target_value_model: ValueModel) -> None:
    target_value_model.load_state_dict(value_model.state_dict())
