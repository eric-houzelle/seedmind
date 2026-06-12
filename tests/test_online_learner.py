"""Smoke tests for the continual OnlineLearner."""
from __future__ import annotations

import math

import torch

from scripts.run_micro_fouloide import _compact_obs, build_agent, build_env, causal_event_names
from seedmind.agent.curiosity import compute_prediction_error_tensor
from seedmind.memory.experience_buffer import make_experience
from seedmind.training.latent_utils import latent_to_numpy
from seedmind.training.online import OnlineLearner


def _config() -> dict:
    return {
        "env": {
            "type": "micro_fouloide",
            "size": 8,
            "max_steps": 0,
            "soft_death": True,
            "resource_regrow_steps": 10,
            "num_food": 2, "num_water": 2,
            "num_warm_zones": 1, "num_cold_zones": 1,
            "num_dangers": 1, "num_obstacles": 2,
        },
        "agent": {"latent_dim": 16},
        "world_model": {
            "hidden_dim": 32, "num_layers": 1, "batch_size": 8,
            "uncertainty_head_updates_per_train": 1,
        },
        "causal_world_model": {"enabled": True, "predict_events": True},
        "value_model": {"enabled": True, "hidden_dim": 32, "num_layers": 1, "batch_size": 8},
        "dqn": {"conv_channels": 8, "hidden_dim": 32, "batch_size": 8},
        "policy": {"epsilon_start": 1.0, "epsilon_end": 0.5, "epsilon_decay_steps": 100},
        "planning": {"enabled": True, "weight": 0.25, "horizon": 2, "num_samples": 2},
        "online": {
            "update_every": 10,
            "updates_per_cycle": 1,
            "warmup_steps": 50,
            "threshold_refresh_steps": 50,
            "threshold_samples": 100,
        },
    }


def _run_steps(num_steps: int):
    torch.manual_seed(0)
    config = _config()
    agent = build_agent(config, seed=0)
    learner = OnlineLearner(agent, config, torch.device("cpu"), seed=0)
    env = build_env(config, seed=0)
    observation = env.reset()
    latent_state = agent.encoder.encode_tensor(observation)
    event_to_index = {event: i for i, event in enumerate(causal_event_names(config))}

    for step in range(1, num_steps + 1):
        latent_np = latent_to_numpy(latent_state)
        memories = agent.retrieve(latent_np)
        goal = agent.choose_goal(latent_np, memories)
        action = agent.choose_action(
            latent_np, goal, memories, env.available_actions(), observation=observation,
        )
        action_index = agent.action_index[action]
        next_obs, reward_ext, done, info = env.step(action)
        next_latent = agent.encoder.encode_tensor(next_obs)
        predicted, _, _ = agent.world_model.predict_tensor(latent_state, action_index)
        pred_err = float(compute_prediction_error_tensor(predicted, next_latent).item())
        event = str(info.get("event", "unknown"))
        experience = make_experience(
            episode_id="online_test",
            world_id=env.world_id,
            step=step,
            observation=None,
            action=action,
            next_observation=None,
            reward_external=reward_ext,
            reward_intrinsic=0.0,
            goal=goal,
            prediction_error=pred_err,
            done=done,
            latent_state=latent_np,
            next_latent_state=latent_to_numpy(next_latent),
            action_index=action_index,
            obs_state=_compact_obs(observation),
            next_obs_state=_compact_obs(next_obs),
            event=event,
            event_index=event_to_index.get(event),
            causal_features=env.causal_features(observation),
            next_causal_features=env.causal_features(next_obs),
        )
        experience["reward_learning"] = reward_ext
        learner.observe(experience)
        observation = next_obs
        latent_state = next_latent
    return agent, learner


def test_planner_gate_closed_at_cold_start():
    config = _config()
    agent = build_agent(config, seed=0)
    OnlineLearner(agent, config, torch.device("cpu"), seed=0)
    # Softplus uncertainty is strictly positive: a zero threshold closes the gate.
    assert agent.planner_uncertainty_threshold == 0.0


def test_online_learner_updates_models_and_threshold():
    agent, learner = _run_steps(120)
    stats = learner.stats()
    assert stats["env_steps"] == 120
    assert stats["buffer_size"] == 120
    assert stats["q_updates"] > 0
    assert math.isfinite(stats["wm_loss"]) and stats["wm_loss"] > 0.0
    assert math.isfinite(stats["td_loss"])
    assert math.isfinite(stats["value_loss"])
    # Past warmup (50) with refresh every 50 steps: threshold resolved online.
    assert stats["uncertainty_threshold"] is not None
    assert stats["uncertainty_threshold"] > 0.0
    assert agent.planner_uncertainty_threshold == stats["uncertainty_threshold"]


def test_threshold_not_refreshed_before_warmup():
    _, learner = _run_steps(40)
    assert learner.stats()["uncertainty_threshold"] is None
