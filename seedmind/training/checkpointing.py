"""Continual-learning checkpoints (SPEC section 18).

Saves and restores everything needed to resume without starting from scratch:
World Model + encoder weights, optimizer state, policy schedule, experience
buffer, persistent memory, training metrics and the run config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from seedmind.agent.agent import Agent
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.memory.persistent_memory import PersistentMemory


def save_checkpoint(
    path: str,
    agent: Agent,
    optimizer: Optional[torch.optim.Optimizer] = None,
    buffer: Optional[ExperienceBuffer] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    target_network: Optional[Any] = None,
    value_optimizer: Optional[torch.optim.Optimizer] = None,
    target_value_model: Optional[Any] = None,
    latent_q_network: Optional[Any] = None,
    latent_q_optimizer: Optional[torch.optim.Optimizer] = None,
    target_latent_q_network: Optional[Any] = None,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
    target_critic: Optional[Any] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "world_model_state": agent.world_model.state_dict(),
        "encoder_state": agent.encoder.state_dict(),
        "policy": {
            "epsilon_start": agent.policy.epsilon_start,
            "epsilon_end": agent.policy.epsilon_end,
            "epsilon_decay_steps": agent.policy.epsilon_decay_steps,
            "total_steps": agent.policy.total_steps,
        },
        "memory_items": agent.memory._items,
        "memory_counter": agent.memory._counter,
        "actions": agent.actions,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if buffer is not None:
        payload["buffer"] = {
            "capacity": buffer.capacity,
            "data": buffer._data,
            "cursor": buffer._cursor,
        }
    # V2: learned policy.
    if agent.q_network is not None:
        payload["q_network_state"] = agent.q_network.state_dict()
    if target_network is not None:
        payload["target_network_state"] = target_network.state_dict()
    if q_optimizer is not None:
        payload["q_optimizer_state"] = q_optimizer.state_dict()
    if getattr(agent, "value_model", None) is not None:
        payload["value_model_state"] = agent.value_model.state_dict()
    if target_value_model is not None:
        payload["target_value_model_state"] = target_value_model.state_dict()
    if value_optimizer is not None:
        payload["value_optimizer_state"] = value_optimizer.state_dict()
    if latent_q_network is not None:
        payload["latent_q_network_state"] = latent_q_network.state_dict()
    if target_latent_q_network is not None:
        payload["target_latent_q_network_state"] = target_latent_q_network.state_dict()
    if latent_q_optimizer is not None:
        payload["latent_q_optimizer_state"] = latent_q_optimizer.state_dict()
    # Imagination actor-critic (DreamerV3 policy). Without these the trained policy
    # is silently dropped on save → eval/resume loads a fresh random actor (uniform).
    if getattr(agent, "actor", None) is not None:
        payload["actor_state"] = agent.actor.state_dict()
    if getattr(agent, "critic", None) is not None:
        payload["critic_state"] = agent.critic.state_dict()
    if actor_optimizer is not None:
        payload["actor_optimizer_state"] = actor_optimizer.state_dict()
    if critic_optimizer is not None:
        payload["critic_optimizer_state"] = critic_optimizer.state_dict()
    if target_critic is not None:
        payload["target_critic_state"] = target_critic.state_dict()

    torch.save(payload, p)


def load_checkpoint(
    path: str,
    agent: Agent,
    optimizer: Optional[torch.optim.Optimizer] = None,
    buffer: Optional[ExperienceBuffer] = None,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    target_network: Optional[Any] = None,
    value_optimizer: Optional[torch.optim.Optimizer] = None,
    target_value_model: Optional[Any] = None,
    latent_q_network: Optional[Any] = None,
    latent_q_optimizer: Optional[torch.optim.Optimizer] = None,
    target_latent_q_network: Optional[Any] = None,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
    target_critic: Optional[Any] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    agent.world_model.load_state_dict(payload["world_model_state"])
    agent.encoder.load_state_dict(payload["encoder_state"])

    pol = payload.get("policy", {})
    agent.policy.total_steps = pol.get("total_steps", agent.policy.total_steps)

    agent.memory._items = payload.get("memory_items", [])
    agent.memory._counter = payload.get("memory_counter", len(agent.memory._items))
    agent.memory._rebuild_cache()

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if buffer is not None and "buffer" in payload:
        b = payload["buffer"]
        buffer.capacity = b["capacity"]
        buffer._data = b["data"]
        buffer._cursor = b.get("cursor", 0)
        if hasattr(buffer, "_rebuild_index"):
            buffer._rebuild_index()

    # V2: learned policy.
    if agent.q_network is not None and "q_network_state" in payload:
        agent.q_network.load_state_dict(payload["q_network_state"])
    if target_network is not None and "target_network_state" in payload:
        target_network.load_state_dict(payload["target_network_state"])
    if q_optimizer is not None and "q_optimizer_state" in payload:
        q_optimizer.load_state_dict(payload["q_optimizer_state"])
    if getattr(agent, "value_model", None) is not None and "value_model_state" in payload:
        agent.value_model.load_state_dict(payload["value_model_state"])
    if target_value_model is not None and "target_value_model_state" in payload:
        target_value_model.load_state_dict(payload["target_value_model_state"])
    if value_optimizer is not None and "value_optimizer_state" in payload:
        value_optimizer.load_state_dict(payload["value_optimizer_state"])
    if latent_q_network is not None and "latent_q_network_state" in payload:
        latent_q_network.load_state_dict(payload["latent_q_network_state"])
    if target_latent_q_network is not None and "target_latent_q_network_state" in payload:
        target_latent_q_network.load_state_dict(payload["target_latent_q_network_state"])
    if latent_q_optimizer is not None and "latent_q_optimizer_state" in payload:
        latent_q_optimizer.load_state_dict(payload["latent_q_optimizer_state"])
    # Imagination actor-critic (DreamerV3 policy).
    if getattr(agent, "actor", None) is not None and "actor_state" in payload:
        agent.actor.load_state_dict(payload["actor_state"])
    if getattr(agent, "critic", None) is not None and "critic_state" in payload:
        agent.critic.load_state_dict(payload["critic_state"])
    if actor_optimizer is not None and "actor_optimizer_state" in payload:
        actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
    if critic_optimizer is not None and "critic_optimizer_state" in payload:
        critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
    if target_critic is not None and "target_critic_state" in payload:
        target_critic.load_state_dict(payload["target_critic_state"])

    return {
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
        "has_q_network": "q_network_state" in payload,
        "has_value_model": "value_model_state" in payload,
        "has_latent_q_network": "latent_q_network_state" in payload,
        "has_actor": "actor_state" in payload,
    }
