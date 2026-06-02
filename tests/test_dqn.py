import numpy as np
import torch

from seedmind.agent.agent import Agent
from seedmind.agent.q_network import QNetwork, obs_batch_to_tensors
from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import ACTIONS, COLOR_DOOR_CLOSED, COLOR_KEY, EMPTY, WALL
from seedmind.evaluation.scenarios import run_episode
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_bc,
    train_dqn,
)

CONFIG = {
    "agent": {"latent_dim": 16, "memory_top_k": 3},
    "world_model": {"hidden_dim": 32, "num_layers": 2, "learning_rate": 1e-3, "batch_size": 16},
    "curiosity": {"enabled": True, "weight": 0.1, "max_reward": 1.0},
    "policy": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1500},
    "dqn": {"conv_channels": 16, "hidden_dim": 64, "learning_rate": 1e-3, "gamma": 0.99,
            "batch_size": 32, "double_dqn": True},
}


def _obs(grid):
    return {"grid": np.asarray(grid, dtype=np.int16), "has_key": 0, "door_open": 0, "key_color": None}


def test_qnetwork_forward_shapes():
    qnet = QNetwork(grid_size=5, num_actions=len(ACTIONS), conv_channels=8, hidden_dim=32)
    obs = [_obs(np.zeros((5, 5))) for _ in range(4)]
    channels, inventory = obs_batch_to_tensors(obs)
    out = qnet(channels, inventory)
    assert out.shape == (4, len(ACTIONS))


def test_qnetwork_q_values_single():
    qnet = QNetwork(grid_size=5, num_actions=len(ACTIONS))
    values = qnet.q_values(_obs(np.zeros((5, 5))))
    assert values.shape == (len(ACTIONS),)


def test_sync_target_copies_weights():
    qnet = QNetwork(grid_size=5, num_actions=len(ACTIONS))
    target = make_target_network(qnet)
    # Perturb qnet, then sync; weights must match afterwards.
    with torch.no_grad():
        for p in qnet.parameters():
            p.add_(1.0)
    sync_target(qnet, target)
    for p, t in zip(qnet.parameters(), target.parameters()):
        assert torch.allclose(p, t)


def test_dqn_update_runs_and_changes_weights():
    agent = Agent.from_config(CONFIG, actions=ACTIONS, grid_size=8,
                              use_planner=False, learned_policy=True, seed=0)
    env = ColoredGridWorld(size=8, max_steps=40, allowed_colors=["red"], seed=1)
    buffer = ExperienceBuffer(seed=0)
    for ep in range(8):
        run_episode(env, agent, episode_index=ep, max_steps=40, buffer=buffer)

    target = make_target_network(agent.q_network)
    optimizer = make_q_optimizer(agent.q_network, learning_rate=1e-3)
    before = [p.detach().clone() for p in agent.q_network.parameters()]
    result = train_dqn(agent.q_network, target, buffer, optimizer,
                       batch_size=32, num_updates=50)
    assert result["updates"] > 0
    assert np.isfinite(result["td_loss"])
    changed = any(
        not torch.allclose(b, p) for b, p in zip(before, agent.q_network.parameters())
    )
    assert changed


def test_bc_reduces_loss():
    qnet = QNetwork(grid_size=5, num_actions=len(ACTIONS), conv_channels=8, hidden_dim=32)
    optimizer = make_q_optimizer(qnet, learning_rate=1e-2)
    buffer = ExperienceBuffer(seed=0)
    # Same observation always maps to action index 2 (a clean supervised target).
    obs = _obs(np.zeros((5, 5)))
    for i in range(40):
        buffer.add(make_experience(
            episode_id="e", world_id="w", step=i, observation="o", action="INTERACT",
            next_observation="o2", reward_external=1.0, reward_intrinsic=0.0,
            goal="g", prediction_error=0.0, done=True,
            action_index=2, obs_state=obs, next_obs_state=obs,
        ))
    first = train_bc(qnet, buffer, optimizer, batch_size=16, num_updates=1)
    last = train_bc(qnet, buffer, optimizer, batch_size=16, num_updates=80)
    assert last["bc_loss"] < first["bc_loss"]


def test_learned_policy_beats_random_on_fixed_map():
    """Integration test: a DQN policy learns to solve a fixed easy colored map.

    Slightly heavier (a few hundred episodes) but deterministic for seed 0.
    """
    torch.manual_seed(0)
    config = dict(CONFIG)
    config["policy"] = {"epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 9000}

    # Easy fixed map: no distractors/dangers, small grid.
    env = ColoredGridWorld(size=5, max_steps=25, allowed_colors=["red"],
                           num_distractor_doors=0, num_distractor_keys=0,
                           num_dangers=0, seed=3)
    env.regenerate_each_reset = False
    env.reset()  # freeze the layout

    agent = Agent.from_config(config, actions=ACTIONS, grid_size=5,
                              use_planner=False, learned_policy=True, seed=0)
    buffer = ExperienceBuffer(seed=0)
    target = make_target_network(agent.q_network)
    optimizer = make_q_optimizer(agent.q_network, learning_rate=1e-3)

    # Baseline success of the untrained (random) policy on this map.
    base_hits = sum(
        int(run_episode(env, agent, episode_index=ep, max_steps=25, buffer=buffer).success)
        for ep in range(20)
    )

    for ep in range(400):
        run_episode(env, agent, episode_index=ep, max_steps=25, buffer=buffer)
        if len(buffer) >= 32:
            train_dqn(agent.q_network, target, buffer, optimizer,
                      batch_size=32, num_updates=8)
            if ep % 25 == 0:
                sync_target(agent.q_network, target)

    # Greedy evaluation: the learned policy should clearly beat random.
    agent.policy.total_steps = agent.policy.epsilon_decay_steps
    hits = sum(int(run_episode(env, agent, episode_index=i, max_steps=25,
                               buffer=None, store_memory=False).success) for i in range(20))
    assert hits > base_hits
    assert hits >= 12
