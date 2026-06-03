"""Tests for World-Model planning and Dyna imagination."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from seedmind.agent.agent import Agent
from seedmind.agent.curiosity import CuriosityModule
from seedmind.agent.encoder import Encoder
from seedmind.agent.goal_generator import GoalGenerator
from seedmind.agent.policy import EpsilonGreedyPolicy
from seedmind.agent.q_network import QNetwork
from seedmind.agent.world_model import WorldModel
from seedmind.agent.sandbox_encoder import (
    SANDBOX_NUM_CHANNELS,
    SANDBOX_NUM_SCALARS,
    sandbox_obs_batch_to_tensors,
    sandbox_observation_to_vector,
)
from seedmind.envs.sandbox_world import ACTIONS, SandboxWorld, NUM_ENTITIES
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.memory.persistent_memory import PersistentMemory
from seedmind.training.imagination import imagine_experiences


def _make_agent(planning_weight: float = 0.0) -> Agent:
    torch.manual_seed(0)
    grid_size = 6
    latent_dim = 32
    input_dim = grid_size * grid_size * NUM_ENTITIES + SANDBOX_NUM_SCALARS

    encoder = Encoder(
        grid_size=grid_size, latent_dim=latent_dim,
        num_entities=NUM_ENTITIES, seed=0,
        input_dim=input_dim,
        obs_to_vec_fn=sandbox_observation_to_vector,
    )
    wm = WorldModel(latent_dim=latent_dim, num_actions=len(ACTIONS), hidden_dim=32, num_layers=1)
    curiosity = CuriosityModule(weight=0.3, max_reward=1.0, enabled=True)
    policy = EpsilonGreedyPolicy(epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1, seed=0)
    q_net = QNetwork(
        grid_size=grid_size, num_actions=len(ACTIONS),
        num_grid_channels=SANDBOX_NUM_CHANNELS, num_scalars=SANDBOX_NUM_SCALARS,
        obs_batch_fn=sandbox_obs_batch_to_tensors, conv_channels=8, hidden_dim=16,
    )
    return Agent(
        encoder=encoder, world_model=wm, curiosity=curiosity,
        goal_generator=GoalGenerator(seed=0), policy=policy,
        memory=PersistentMemory(), actions=ACTIONS,
        use_planner=planning_weight > 0, q_network=q_net,
        planning_weight=planning_weight,
        planner_horizon=2, planner_samples=4,
    )


class TestCombinedScorer:

    def test_q_only_when_no_planning(self):
        agent = _make_agent(planning_weight=0.0)
        env = SandboxWorld(size=6, seed=0)
        obs = env.reset()
        latent = agent.encode(obs)
        action = agent.choose_action(latent, "explore", [], ACTIONS, observation=obs)
        assert action in ACTIONS

    def test_combined_when_planning_enabled(self):
        agent = _make_agent(planning_weight=0.5)
        env = SandboxWorld(size=6, seed=0)
        obs = env.reset()
        latent = agent.encode(obs)
        action = agent.choose_action(latent, "explore", [], ACTIONS, observation=obs)
        assert action in ACTIONS

    def test_different_weights_may_change_action(self):
        """With extreme weights, the chosen action can differ."""
        env = SandboxWorld(size=6, seed=42)
        obs = env.reset()

        agent_q = _make_agent(planning_weight=0.0)
        latent = agent_q.encode(obs)
        action_q = agent_q.choose_action(latent, "explore", [], ACTIONS, observation=obs)

        agent_wm = _make_agent(planning_weight=1.0)
        action_wm = agent_wm.choose_action(latent, "explore", [], ACTIONS, observation=obs)

        # Both should be valid actions regardless of weight
        assert action_q in ACTIONS
        assert action_wm in ACTIONS


class TestImagination:

    def _fill_buffer(self, buffer: ExperienceBuffer, agent: Agent, n: int = 50):
        env = SandboxWorld(size=6, seed=0)
        for i in range(n):
            obs = env.reset()
            latent = agent.encode(obs)
            exp = make_experience(
                episode_id=f"fill_{i}", world_id="test", step=0,
                observation=obs["grid"].tolist(), action="WAIT",
                next_observation=obs["grid"].tolist(),
                reward_external=0.01, reward_intrinsic=0.0,
                goal="survive", prediction_error=0.0, done=False,
                memory_used=[], latent_state=latent,
                next_latent_state=latent, action_index=6,
                obs_state=None, next_obs_state=None,
            )
            buffer.add(exp)

    def test_imagine_returns_experiences(self):
        agent = _make_agent()
        buffer = ExperienceBuffer(seed=0)
        self._fill_buffer(buffer, agent)

        dreams = imagine_experiences(
            agent.world_model, buffer, agent.curiosity,
            num_actions=len(ACTIONS), num_imagined=8,
        )
        assert len(dreams) == 8
        for d in dreams:
            assert d["latent_state"] is not None
            assert d["next_latent_state"] is not None
            assert d["action_index"] is not None

    def test_imagine_empty_buffer(self):
        agent = _make_agent()
        buffer = ExperienceBuffer(seed=0)
        dreams = imagine_experiences(
            agent.world_model, buffer, agent.curiosity,
            num_actions=len(ACTIONS), num_imagined=8,
        )
        assert len(dreams) == 0

    def test_imagined_experiences_addable_to_buffer(self):
        agent = _make_agent()
        buffer = ExperienceBuffer(seed=0)
        self._fill_buffer(buffer, agent)
        before = len(buffer)
        dreams = imagine_experiences(
            agent.world_model, buffer, agent.curiosity,
            num_actions=len(ACTIONS), num_imagined=10,
        )
        for d in dreams:
            buffer.add(d)
        assert len(buffer) == before + len(dreams)
