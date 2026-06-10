"""DQN training for the learned policy (SPEC sections 15 & 24, V2).

Trains the Q-network by temporal-difference learning on the replay buffer, with
a periodically-synced target network for stability. An optional behavioral
cloning warm-start imitates the actions taken in successful (high-reward)
transitions, which bootstraps learning under sparse rewards.

The default training reward is
``reward_external + curiosity_weight * reward_intrinsic`` so curiosity acts as
an exploration bonus. Callers can provide another reward field for value
learning while keeping the environment reward unchanged in the replay buffer.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seedmind.agent.q_network import QNetwork, obs_batch_to_tensors
from seedmind.memory.experience_buffer import ExperienceBuffer


def make_q_optimizer(q_network: QNetwork, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam(q_network.parameters(), lr=learning_rate)


def make_target_network(q_network: QNetwork) -> QNetwork:
    target = copy.deepcopy(q_network)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


def sync_target(q_network: QNetwork, target_network: QNetwork) -> None:
    target_network.load_state_dict(q_network.state_dict())


def _transition_reward(experience: Dict[str, Any], reward_key: str) -> float:
    return float(experience.get(reward_key, experience.get("reward_external", 0.0)))


def _assemble_dqn_batch(batch: List[Dict[str, Any]], curiosity_weight: float,
                        batch_fn=None, buffer: Optional[ExperienceBuffer] = None,
                        n_step: int = 1, gamma: float = 0.95,
                        reward_key: str = "reward_external"):
    """Build TD tensors from experiences; skip those lacking observations."""
    if batch_fn is None:
        batch_fn = obs_batch_to_tensors
    obs, next_obs, actions, rewards, dones, n_steps = [], [], [], [], [], []
    for e in batch:
        if e.get("obs_state") is None or e.get("next_obs_state") is None:
            continue
        if e.get("action_index") is None:
            continue
        if buffer is not None and n_step > 1:
            sequence = buffer.n_step_sequence(e, n_step)
            if not sequence:
                continue
            reward = 0.0
            discount = 1.0
            last = sequence[-1]
            for current in sequence:
                reward += discount * (
                    _transition_reward(current, reward_key)
                    + curiosity_weight * float(current.get("reward_intrinsic", 0.0))
                )
                discount *= gamma
        else:
            reward = (
                _transition_reward(e, reward_key)
                + curiosity_weight * float(e.get("reward_intrinsic", 0.0))
            )
            last = e
        if last.get("next_obs_state") is None:
            continue

        obs.append(e["obs_state"])
        next_obs.append(last["next_obs_state"])
        actions.append(int(e["action_index"]))
        rewards.append(reward)
        dones.append(1.0 if last.get("done", False) else 0.0)
        n_steps.append(len(sequence) if buffer is not None and n_step > 1 else 1)

    if not obs:
        return None

    channels, inventory = batch_fn(obs)
    next_channels, next_inventory = batch_fn(next_obs)
    return (
        channels,
        inventory,
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        next_channels,
        next_inventory,
        torch.tensor(dones, dtype=torch.float32),
        torch.tensor(n_steps, dtype=torch.float32),
    )


def _sample_batch(buffer: ExperienceBuffer, batch_size: int, sampler: str) -> List[Dict[str, Any]]:
    if sampler == "mixed":
        # Oversample successful transitions to cope with sparse rewards.
        half = max(1, batch_size // 2)
        return buffer.sample(batch_size - half) + buffer.sample_high_reward(half)
    if sampler == "causal":
        half = max(1, batch_size // 2)
        causal = buffer.sample_causal(half)
        return buffer.sample(batch_size - len(causal)) + causal
    return buffer.sample(batch_size)


def train_dqn(
    q_network: QNetwork,
    target_network: QNetwork,
    buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    gamma: float = 0.95,
    curiosity_weight: float = 0.0,
    double_dqn: bool = True,
    num_updates: int = 1,
    sampler: str = "uniform",
    grad_clip: float = 10.0,
    n_step: int = 1,
    reward_key: str = "reward_external",
) -> Dict[str, float]:
    """Run ``num_updates`` TD gradient steps; return the mean TD loss."""
    if len(buffer) == 0:
        return {"td_loss": 0.0, "updates": 0.0}

    q_network.train()
    total_loss = 0.0
    done_updates = 0

    batch_fn = getattr(q_network, '_obs_batch_fn', obs_batch_to_tensors)
    for _ in range(num_updates):
        batch = _sample_batch(buffer, batch_size, sampler)
        assembled = _assemble_dqn_batch(
            batch, curiosity_weight, batch_fn=batch_fn,
            buffer=buffer, n_step=n_step, gamma=gamma,
            reward_key=reward_key,
        )
        if assembled is None:
            continue
        (
            channels, inventory, actions, rewards,
            next_channels, next_inventory, dones, n_steps,
        ) = assembled
        device = next(q_network.parameters()).device
        channels = channels.to(device)
        inventory = inventory.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_channels = next_channels.to(device)
        next_inventory = next_inventory.to(device)
        dones = dones.to(device)
        n_steps = n_steps.to(device)

        q_values = q_network(channels, inventory)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_target = target_network(next_channels, next_inventory)
            if double_dqn:
                next_actions = q_network(next_channels, next_inventory).argmax(dim=1, keepdim=True)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q = next_q_target.max(dim=1).values
            td_target = rewards + (gamma ** n_steps) * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_taken, td_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        done_updates += 1

    if done_updates == 0:
        return {"td_loss": 0.0, "updates": 0.0}
    return {"td_loss": total_loss / done_updates, "updates": float(done_updates)}


def train_bc(
    q_network: QNetwork,
    demo_buffer: ExperienceBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    num_updates: int = 50,
) -> Dict[str, float]:
    """Behavioral-cloning warm-start from successful trajectories.

    ``demo_buffer`` must hold transitions from *whole successful episodes* (not
    just the winning step), so imitation teaches navigation + interaction
    rather than collapsing to the single terminal action. Treats the Q-network
    outputs as logits pushed toward the actions actually taken.
    """
    if len(demo_buffer) == 0:
        return {"bc_loss": 0.0, "updates": 0.0}

    q_network.train()
    total_loss = 0.0
    done_updates = 0

    for _ in range(num_updates):
        batch = demo_buffer.sample(batch_size)
        obs, actions = [], []
        for e in batch:
            if e.get("obs_state") is None or e.get("action_index") is None:
                continue
            obs.append(e["obs_state"])
            actions.append(int(e["action_index"]))

        if not obs:
            continue

        batch_fn = getattr(q_network, '_obs_batch_fn', obs_batch_to_tensors)
        channels, inventory = batch_fn(obs)
        device = next(q_network.parameters()).device
        channels = channels.to(device)
        inventory = inventory.to(device)
        logits = q_network(channels, inventory)
        loss = F.cross_entropy(logits, torch.tensor(actions, dtype=torch.long, device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        done_updates += 1

    if done_updates == 0:
        return {"bc_loss": 0.0, "updates": 0.0}
    return {"bc_loss": total_loss / done_updates, "updates": float(done_updates)}
