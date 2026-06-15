"""TD training for latent Q-networks."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from seedmind.agent.latent_q_network import LatentQNetwork
from seedmind.agent.q_network import QNetwork, obs_batch_to_tensors
from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.value import _sample_batch


def make_latent_q_optimizer(
    q_network: LatentQNetwork,
    learning_rate: float = 3e-4,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(q_network.parameters(), lr=learning_rate)


def make_latent_target_network(q_network: LatentQNetwork) -> LatentQNetwork:
    target = copy.deepcopy(q_network)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


def sync_latent_target(q_network: LatentQNetwork, target_network: LatentQNetwork) -> None:
    target_network.load_state_dict(q_network.state_dict())


def _reward(experience: Dict[str, Any], reward_key: str) -> float:
    return float(experience.get(reward_key, experience.get("reward_external", 0.0)))


def train_latent_dqn(
    q_network: LatentQNetwork,
    target_network: LatentQNetwork,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    gamma: float = 0.97,
    num_updates: int = 1,
    sampler: str = "uniform",
    reward_key: str = "reward_external",
    double_dqn: bool = True,
    grad_clip: float = 10.0,
    teacher_q_network: QNetwork | None = None,
    distill_weight: float = 0.0,
    distill_mode: str = "value",
) -> Dict[str, float]:
    if len(buffer) == 0:
        return {"latent_td_loss": 0.0, "latent_distill_loss": 0.0, "updates": 0.0}

    q_network.train()
    if teacher_q_network is not None:
        teacher_q_network.eval()
    total_loss = 0.0
    total_td_loss = 0.0
    total_distill_loss = 0.0
    updates = 0

    for _ in range(num_updates):
        batch = _sample_batch(buffer, batch_size, sampler)
        rows = [
            e for e in batch
            if e.get("latent_state") is not None
            and e.get("next_latent_state") is not None
            and e.get("action_index") is not None
        ]
        if not rows:
            continue

        device = next(q_network.parameters()).device
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
        actions = torch.as_tensor(
            [int(e["action_index"]) for e in rows],
            dtype=torch.long,
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

        q_taken = q_network(latents).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_target = target_network(next_latents)
            if double_dqn:
                next_actions = q_network(next_latents).argmax(dim=1, keepdim=True)
                next_q = next_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = next_target.max(dim=1).values
            target = rewards + gamma * next_q * (1.0 - dones)

        td_loss = F.smooth_l1_loss(q_taken, target)
        distill_loss = torch.zeros((), device=device)
        if teacher_q_network is not None and distill_weight > 0.0:
            obs_rows = [e.get("obs_state") for e in rows]
            if all(obs is not None for obs in obs_rows):
                batch_fn = getattr(teacher_q_network, "_obs_batch_fn", obs_batch_to_tensors)
                channels, scalars = batch_fn(obs_rows)
                teacher_device = next(teacher_q_network.parameters()).device
                with torch.no_grad():
                    raw_teacher_values = teacher_q_network(
                        channels.to(teacher_device),
                        scalars.to(teacher_device),
                    ).to(device)
                student_values = q_network(latents)
                if distill_mode == "policy":
                    teacher_actions = raw_teacher_values.argmax(dim=1)
                    distill_loss = F.cross_entropy(student_values, teacher_actions)
                else:
                    teacher_values = raw_teacher_values - raw_teacher_values.mean(dim=1, keepdim=True)
                    student_values = student_values - student_values.mean(dim=1, keepdim=True)
                    distill_loss = F.mse_loss(student_values, teacher_values)

        loss = td_loss + float(distill_weight) * distill_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_td_loss += float(td_loss.item())
        total_distill_loss += float(distill_loss.item())
        updates += 1

    if updates == 0:
        return {"latent_td_loss": 0.0, "latent_distill_loss": 0.0, "updates": 0.0}
    return {
        "latent_td_loss": total_td_loss / updates,
        "latent_distill_loss": total_distill_loss / updates,
        "latent_total_loss": total_loss / updates,
        "updates": float(updates),
    }


def train_latent_dqn_dyna(
    q_network: LatentQNetwork,
    target_network: LatentQNetwork,
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    num_actions: int,
    batch_size: int = 64,
    gamma: float = 0.97,
    num_updates: int = 1,
    sampler: str = "uniform",
    loss_weight: float = 1.0,
    grad_clip: float = 10.0,
) -> Dict[str, float]:
    if len(buffer) == 0 or loss_weight <= 0.0:
        return {"latent_dyna_loss": 0.0, "updates": 0.0}

    q_network.train()
    world_model.eval()
    target_network.eval()
    total_loss = 0.0
    updates = 0

    for _ in range(num_updates):
        batch = _sample_batch(buffer, batch_size, sampler)
        rows = [e for e in batch if e.get("latent_state") is not None]
        if not rows:
            continue

        q_device = next(q_network.parameters()).device
        wm_device = next(world_model.parameters()).device
        latents = torch.as_tensor(
            np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in rows]),
            dtype=torch.float32,
            device=q_device,
        )
        actions = torch.randint(0, num_actions, (len(rows),), dtype=torch.long, device=wm_device)
        q_taken = q_network(latents).gather(1, actions.to(q_device).unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            out = world_model.forward_aux(latents.to(wm_device), actions)
            imagined_next = out["next_state"].to(q_device)
            imagined_reward = out["reward"].to(q_device)
            target = imagined_reward + gamma * target_network(imagined_next).max(dim=1).values

        loss = F.smooth_l1_loss(q_taken, target) * float(loss_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        updates += 1

    if updates == 0:
        return {"latent_dyna_loss": 0.0, "updates": 0.0}
    return {"latent_dyna_loss": total_loss / updates, "updates": float(updates)}
