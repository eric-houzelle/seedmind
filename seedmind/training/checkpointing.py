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

    torch.save(payload, p)


def load_checkpoint(
    path: str,
    agent: Agent,
    optimizer: Optional[torch.optim.Optimizer] = None,
    buffer: Optional[ExperienceBuffer] = None,
    q_optimizer: Optional[torch.optim.Optimizer] = None,
    target_network: Optional[Any] = None,
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

    # V2: learned policy.
    if agent.q_network is not None and "q_network_state" in payload:
        agent.q_network.load_state_dict(payload["q_network_state"])
    if target_network is not None and "target_network_state" in payload:
        target_network.load_state_dict(payload["target_network_state"])
    if q_optimizer is not None and "q_optimizer_state" in payload:
        q_optimizer.load_state_dict(payload["q_optimizer_state"])

    return {
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
        "has_q_network": "q_network_state" in payload,
    }
