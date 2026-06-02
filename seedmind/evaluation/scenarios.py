"""Episode runner and agent comparison (SPEC sections 17, 23.8 & 30)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from seedmind.agent.agent import Agent
from seedmind.agent.curiosity import compute_prediction_error
from seedmind.envs.base import EnvironmentAdapter
from seedmind.evaluation.metrics import EpisodeMetrics
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience


def _compact_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Slim copy of an observation for replay (grid + inventory flags)."""
    return {
        "grid": np.asarray(observation["grid"], dtype=np.int16),
        "has_key": int(observation.get("has_key", 0)),
        "door_open": int(observation.get("door_open", 0)),
        "key_color": observation.get("key_color"),
    }


def run_episode(
    env: EnvironmentAdapter,
    agent: Agent,
    episode_index: int = 0,
    max_steps: int = 100,
    buffer: Optional[ExperienceBuffer] = None,
    store_memory: bool = True,
    episode_sink: Optional[List[Dict[str, Any]]] = None,
) -> EpisodeMetrics:
    """Run one full agent loop (SPEC section 17) and return its metrics."""
    observation = env.reset()
    latent_state = agent.encode(observation)
    episode_id = f"episode_{episode_index:06d}"
    world_id = getattr(env, "world_id", "unknown")

    total_external = 0.0
    total_intrinsic = 0.0
    pred_errors: List[float] = []
    goal_counter: Counter = Counter()
    memories_used = 0
    steps = 0
    success = False

    for step in range(max_steps):
        memories = agent.retrieve(latent_state)
        if memories:
            memories_used += 1
        goal = agent.choose_goal(latent_state, memories)
        goal_counter[goal] += 1

        action = agent.choose_action(
            latent_state, goal, memories, env.available_actions(),
            observation=observation,
        )
        action_index = agent.action_index[action]

        next_observation, reward_external, done, info = env.step(action)
        next_latent_state = agent.encode(next_observation)

        predicted_next, _, _ = agent.world_model.predict(latent_state, action_index)
        prediction_error = compute_prediction_error(predicted_next, next_latent_state)
        reward_intrinsic = agent.curiosity.compute(prediction_error)

        experience = make_experience(
            episode_id=episode_id,
            world_id=world_id,
            step=step,
            observation=observation["grid"].tolist(),
            action=action,
            next_observation=next_observation["grid"].tolist(),
            reward_external=reward_external,
            reward_intrinsic=reward_intrinsic,
            goal=goal,
            prediction_error=prediction_error,
            done=done,
            memory_used=[m["memory_id"] for m in memories],
            latent_state=latent_state,
            next_latent_state=next_latent_state,
            action_index=action_index,
            obs_state=_compact_obs(observation),
            next_obs_state=_compact_obs(next_observation),
        )

        if buffer is not None:
            buffer.add(experience)
        if episode_sink is not None:
            episode_sink.append(experience)
        if store_memory:
            agent.memory.store_if_important(experience)

        total_external += reward_external
        total_intrinsic += reward_intrinsic
        pred_errors.append(prediction_error)
        steps = step + 1

        observation = next_observation
        latent_state = next_latent_state

        if info.get("success"):
            success = True
        if done:
            break

    return EpisodeMetrics(
        episode=episode_index,
        episode_reward_external=total_external,
        episode_reward_intrinsic=total_intrinsic,
        steps_survived=steps,
        success=success,
        prediction_error_mean=float(np.mean(pred_errors)) if pred_errors else 0.0,
        memory_items_count=len(agent.memory),
        goal_distribution=dict(goal_counter),
        exploration_rate=agent.policy.epsilon,
        repeated_mistakes=0,
    )


def compare_agents(
    make_env: Callable[[int], EnvironmentAdapter],
    agents: Dict[str, Agent],
    num_maps: int = 20,
    max_steps: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Compare several agents on the same set of procedurally generated maps.

    ``make_env(seed)`` must build a fresh environment for the given seed so all
    agents face identical maps. Returns aggregated stats per agent name.
    """
    results: Dict[str, Dict[str, List[float]]] = {
        name: {"success": [], "steps_to_success": [], "prediction_error": [], "memory_reuse": []}
        for name in agents
    }

    for seed in range(num_maps):
        for name, agent in agents.items():
            env = make_env(seed)
            metrics = run_episode(
                env, agent, episode_index=seed, max_steps=max_steps,
                buffer=None, store_memory=False,
            )
            results[name]["success"].append(1.0 if metrics.success else 0.0)
            results[name]["steps_to_success"].append(
                metrics.steps_survived if metrics.success else float(max_steps)
            )
            results[name]["prediction_error"].append(metrics.prediction_error_mean)
            results[name]["memory_reuse"].append(float(len(agent.memory)))

    summary: Dict[str, Dict[str, float]] = {}
    for name, vals in results.items():
        summary[name] = {
            "success_rate": float(np.mean(vals["success"])),
            "mean_steps_to_success": float(np.mean(vals["steps_to_success"])),
            "mean_prediction_error": float(np.mean(vals["prediction_error"])),
            "mean_memory_items": float(np.mean(vals["memory_reuse"])),
        }
    return summary
