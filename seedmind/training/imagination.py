"""Dyna-style imagination for accelerating Q-learning.

Uses the World Model to generate synthetic experiences ("dreams") from real
latent states sampled from memory. These imagined transitions are added to
the replay buffer so the Q-network learns from both real and imagined data.

This is the "thinking" counterpart: the agent mentally rehearses scenarios
it has been in, exploring what *would have happened* under different actions.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from seedmind.agent.curiosity import compute_prediction_error, CuriosityModule
from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience


def imagine_experiences(
    world_model: WorldModel,
    buffer: ExperienceBuffer,
    curiosity: CuriosityModule,
    num_actions: int,
    num_imagined: int = 16,
    horizon: int = 1,
    rng: np.random.Generator | None = None,
) -> List[Dict[str, Any]]:
    """Generate imagined transitions by rolling the WM forward.

    For each imagined experience:
    1. Sample a real experience from the buffer to get a starting latent state.
    2. Pick a random action.
    3. Roll the WM forward for ``horizon`` steps (only the first step becomes
       an experience — multi-step rollouts warm up the latent).
    4. Package the result as a replay experience.

    Returns a list of experience dicts ready for ``buffer.add()``.
    """
    if len(buffer) < num_imagined or num_imagined <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    real_batch = buffer.sample(num_imagined)
    imagined: List[Dict[str, Any]] = []

    for real_exp in real_batch:
        latent = real_exp.get("latent_state")
        if latent is None:
            continue
        latent = np.asarray(latent, dtype=np.float32)

        action_idx = int(rng.integers(num_actions))
        next_latent, pred_reward, uncertainty = world_model.predict(latent, action_idx)
        pred_err = compute_prediction_error(next_latent, latent)
        reward_int = curiosity.compute(pred_err)

        exp = make_experience(
            episode_id="imagined",
            world_id="imagination",
            step=0,
            observation=None,
            action=f"action_{action_idx}",
            next_observation=None,
            reward_external=float(pred_reward),
            reward_intrinsic=float(reward_int),
            goal="imagine",
            prediction_error=float(pred_err),
            done=False,
            memory_used=[],
            latent_state=latent,
            next_latent_state=next_latent,
            action_index=action_idx,
            obs_state=real_exp.get("obs_state"),
            next_obs_state=real_exp.get("obs_state"),
        )
        imagined.append(exp)

    return imagined
