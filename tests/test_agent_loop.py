import numpy as np
import torch

from seedmind.agent.agent import Agent
from seedmind.agent.world_model import WorldModel
from seedmind.envs.gridworld import ACTIONS, GridWorld
from seedmind.evaluation.scenarios import run_episode
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.checkpointing import load_checkpoint, save_checkpoint
from seedmind.training.train import make_optimizer, train_world_model

CONFIG = {
    "agent": {"latent_dim": 16, "memory_top_k": 3},
    "world_model": {"hidden_dim": 32, "num_layers": 2, "learning_rate": 1e-3, "batch_size": 16},
    "curiosity": {"enabled": True, "weight": 0.1, "max_reward": 1.0},
    "policy": {"epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 100},
}


def _make_agent(use_planner=True):
    return Agent.from_config(CONFIG, actions=ACTIONS, grid_size=8, use_planner=use_planner, seed=0)


def test_world_model_forward_shapes():
    wm = WorldModel(latent_dim=16, num_actions=len(ACTIONS), hidden_dim=32, num_layers=2)
    latent = torch.randn(4, 16)
    actions = torch.tensor([0, 1, 2, 3])
    next_state, reward, uncertainty = wm(latent, actions)
    assert next_state.shape == (4, 16)
    assert reward.shape == (4,)
    assert uncertainty.shape == (4,)
    assert torch.all(uncertainty >= 0)


def test_world_model_predict_single():
    wm = WorldModel(latent_dim=16, num_actions=len(ACTIONS))
    next_state, reward, uncertainty = wm.predict(np.zeros(16, dtype=np.float32), action_index=2)
    assert next_state.shape == (16,)
    assert isinstance(reward, float)
    assert uncertainty >= 0.0


def test_full_episode_collects_experiences():
    agent = _make_agent()
    env = GridWorld(size=8, max_steps=30, seed=0)
    buffer = ExperienceBuffer(seed=0)
    metrics = run_episode(env, agent, episode_index=0, max_steps=30, buffer=buffer)
    assert metrics.steps_survived > 0
    assert len(buffer) == metrics.steps_survived
    assert metrics.prediction_error_mean >= 0.0


def test_training_reduces_loss():
    agent = _make_agent()
    env = GridWorld(size=8, max_steps=50, seed=1)
    buffer = ExperienceBuffer(seed=0)
    for ep in range(5):
        run_episode(env, agent, episode_index=ep, max_steps=50, buffer=buffer)

    optimizer = make_optimizer(agent.world_model, learning_rate=1e-3)
    first = train_world_model(agent.world_model, buffer, optimizer, batch_size=16, num_updates=5)
    last = train_world_model(agent.world_model, buffer, optimizer, batch_size=16, num_updates=300)
    assert last["updates"] > 0
    assert last["total"] < first["total"]


def test_checkpoint_save_and_load(tmp_path):
    agent = _make_agent()
    env = GridWorld(size=8, max_steps=20, seed=2)
    buffer = ExperienceBuffer(seed=0)
    run_episode(env, agent, episode_index=0, max_steps=20, buffer=buffer)
    optimizer = make_optimizer(agent.world_model)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), agent, optimizer, buffer, metrics={"x": 1}, config=CONFIG)

    # Reference prediction before reload.
    latent = np.zeros(16, dtype=np.float32)
    before, _, _ = agent.world_model.predict(latent, 0)

    fresh = _make_agent()
    new_buffer = ExperienceBuffer()
    payload = load_checkpoint(str(path), fresh, buffer=new_buffer)
    after, _, _ = fresh.world_model.predict(latent, 0)

    assert np.allclose(before, after, atol=1e-5)
    assert len(new_buffer) == len(buffer)
    assert payload["config"] == CONFIG
