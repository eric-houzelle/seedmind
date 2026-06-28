"""Agent porte l'état stochastique (h, z) du RSSM (DreamerV3, phase 1 brique 4)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_micro_fouloide import build_agent, build_env, load_config
from seedmind.agent.world_model import RSSMWorldModel

RSSM_CFG = "configs/micro_fouloide_online_homeostatic_rssm.yaml"


@pytest.fixture(scope="module")
def rssm_agent_env():
    cfg = load_config(RSSM_CFG)
    wmc = cfg.setdefault("world_model", {})
    wmc["recurrent"] = True
    wmc["rssm_stochastic"] = True
    wmc["rssm_stoch"] = 8           # small for test speed
    wmc["rssm_discrete"] = 8
    return build_agent(cfg, seed=0), build_env(cfg, seed=0), cfg


def test_world_model_is_stochastic_rssm(rssm_agent_env):
    agent, _, _ = rssm_agent_env
    assert isinstance(agent.world_model, RSSMWorldModel)
    assert agent.recurrent is True and agent._rssm is True
    # actor/critic are sized for feat=[z,h]
    feat = agent.world_model.feat_dim
    assert feat == 8 * 8 + agent.world_model.deter_dim
    assert agent.actor.input_dim == feat


def test_reset_clears_rssm_state(rssm_agent_env):
    agent, env, _ = rssm_agent_env
    obs = env.reset()
    agent.advance(agent.encode(obs))
    assert agent.rssm_state is not None
    agent.reset_state()
    assert agent.rssm_state is None and agent.h is None and agent._prev_action_idx == 0


def test_advance_carries_and_evolves_hz(rssm_agent_env):
    agent, env, _ = rssm_agent_env
    obs = env.reset()
    agent.reset_state()
    feat_first = feat_last = None
    for t in range(8):
        lat = agent.encode(obs)
        agent.advance(lat)
        # state is the (h, z) dict
        st = agent.rssm_state
        assert set(st) >= {"deter", "stoch", "logits"}
        assert st["stoch"].shape[-2:] == (8, 8)
        vec = agent._h_vec()
        assert vec.shape == (agent.world_model.feat_dim,)
        if t == 0:
            feat_first = vec.copy()
        feat_last = vec
        a = agent.choose_action(lat, "survive", [], env.available_actions(), observation=obs)
        step = env.step(a)
        obs = step.observation if hasattr(step, "observation") else step[0]
        assert agent._prev_action_idx == agent.action_index[a]
    assert not np.allclose(feat_first, feat_last, atol=1e-5)  # (h,z) memory evolves
