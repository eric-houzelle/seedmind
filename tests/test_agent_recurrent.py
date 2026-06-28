"""Tests for the Agent's recurrent memory lifecycle (RSSM stage 2, brick 3).

The agent must, when the world model is recurrent: carry h_t across steps,
feed it to the Q-network, track the previous action, reset on demand, and stay
a no-op when the world model is feed-forward.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_micro_fouloide import build_agent, build_env, load_config
from seedmind.agent.world_model import RecurrentWorldModel

RSSM_CFG = "configs/micro_fouloide_online_homeostatic_rssm.yaml"
EGO_CFG = "configs/micro_fouloide_online_homeostatic_egocentric.yaml"


@pytest.fixture(scope="module")
def rssm_agent_env():
    cfg = load_config(RSSM_CFG)
    return build_agent(cfg, seed=0), build_env(cfg, seed=0)


class TestRecurrentWiring:
    def test_recurrent_world_model_and_qnet(self, rssm_agent_env):
        agent, _ = rssm_agent_env
        assert agent.recurrent is True
        assert isinstance(agent.world_model, RecurrentWorldModel)
        assert agent.q_network.recurrent_dim == agent.world_model.deter_dim

    def test_planner_disabled_in_rssm_config(self, rssm_agent_env):
        agent, _ = rssm_agent_env
        assert agent.use_planner is False  # planner not yet adapted to (h, z)


class TestRecurrentLifecycle:
    def test_reset_clears_state(self, rssm_agent_env):
        agent, _ = rssm_agent_env
        agent.advance(np.zeros(agent.world_model.latent_dim, dtype=np.float32))
        agent.reset_state()
        assert agent.h is None
        assert agent._prev_action_idx == 0

    def test_advance_builds_and_evolves_h(self, rssm_agent_env):
        agent, env = rssm_agent_env
        obs = env.reset()
        agent.reset_state()
        h_first = None
        h_last = None
        for t in range(8):
            lat = agent.encode(obs)
            agent.advance(lat)
            vec = agent._h_vec()
            if t == 0:
                h_first = vec.copy()
            h_last = vec
            a = agent.choose_action(lat, "survive", [], env.available_actions(), observation=obs)
            step = env.step(a)
            obs = step.observation if hasattr(step, "observation") else step[0]
        assert h_first.shape == (agent.world_model.deter_dim,)
        assert not np.allclose(h_first, h_last, atol=1e-5)  # memory evolves

    def test_prev_action_tracked(self, rssm_agent_env):
        agent, env = rssm_agent_env
        obs = env.reset()
        agent.reset_state()
        lat = agent.encode(obs)
        agent.advance(lat)
        a = agent.choose_action(lat, "survive", [], env.available_actions(), observation=obs)
        assert agent._prev_action_idx == agent.action_index[a]

    def test_h_influences_policy(self, rssm_agent_env):
        agent, env = rssm_agent_env
        obs = env.reset()
        agent.h = agent.world_model.initial_state(1) * 0.0
        q_zero = agent.q_network.q_values(obs, recurrent=agent._h_vec())
        agent.h = agent.world_model.initial_state(1) + 5.0
        q_big = agent.q_network.q_values(obs, recurrent=agent._h_vec())
        assert not np.allclose(q_zero, q_big, atol=1e-4)


class TestNonRecurrentIsNoOp:
    def test_feedforward_agent_advance_is_noop(self):
        cfg = load_config(EGO_CFG)
        agent = build_agent(cfg, seed=0)
        env = build_env(cfg, seed=0)
        assert agent.recurrent is False
        obs = env.reset()
        lat = agent.encode(obs)
        assert agent.advance(lat) is None
        assert agent._h_vec() is None
        # The decision path still works with no recurrent state.
        a = agent.choose_action(lat, "survive", [], env.available_actions(), observation=obs)
        assert a in env.actions
