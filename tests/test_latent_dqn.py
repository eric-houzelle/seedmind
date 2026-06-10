import numpy as np

from seedmind.agent.q_network import QNetwork
from seedmind.agent.latent_q_network import LatentQNetwork
from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.latent_dqn import (
    make_latent_q_optimizer,
    make_latent_target_network,
    train_latent_dqn,
    train_latent_dqn_dyna,
)


def _buffer() -> ExperienceBuffer:
    buffer = ExperienceBuffer(seed=0)
    for i in range(32):
        buffer.add(make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=1.0 if i % 3 == 0 else -0.1,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=i % 4 == 3,
            latent_state=np.full(5, i / 32, dtype=np.float32),
            next_latent_state=np.full(5, (i + 1) / 32, dtype=np.float32),
            action_index=i % 3,
        ))
    return buffer


def _obs(index: int) -> dict:
    return {
        "grid": np.zeros((2, 2), dtype=np.int64),
        "has_key": bool(index % 2),
        "door_open": bool(index % 3 == 0),
    }


def _buffer_with_observations() -> ExperienceBuffer:
    buffer = ExperienceBuffer(seed=0)
    for i in range(32):
        buffer.add(make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=1.0 if i % 3 == 0 else -0.1,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=i % 4 == 3,
            obs_state=_obs(i),
            next_obs_state=_obs(i + 1),
            latent_state=np.full(5, i / 32, dtype=np.float32),
            next_latent_state=np.full(5, (i + 1) / 32, dtype=np.float32),
            action_index=i % 3,
        ))
    return buffer


def test_latent_dqn_update_runs():
    qnet = LatentQNetwork(latent_dim=5, num_actions=3, hidden_dim=16, num_layers=1)
    target = make_latent_target_network(qnet)
    optimizer = make_latent_q_optimizer(qnet, learning_rate=1e-3)

    result = train_latent_dqn(
        qnet, target, _buffer(), optimizer,
        batch_size=16, num_updates=3,
    )

    assert result["updates"] == 3.0
    assert np.isfinite(result["latent_td_loss"])


def test_latent_dqn_distillation_update_runs():
    qnet = LatentQNetwork(latent_dim=5, num_actions=3, hidden_dim=16, num_layers=1)
    target = make_latent_target_network(qnet)
    teacher = QNetwork(grid_size=2, num_actions=3, conv_channels=4, hidden_dim=16)
    optimizer = make_latent_q_optimizer(qnet, learning_rate=1e-3)

    result = train_latent_dqn(
        qnet, target, _buffer_with_observations(), optimizer,
        batch_size=16, num_updates=3,
        teacher_q_network=teacher, distill_weight=0.5,
    )

    assert result["updates"] == 3.0
    assert np.isfinite(result["latent_td_loss"])
    assert np.isfinite(result["latent_distill_loss"])


def test_latent_dqn_policy_distillation_update_runs():
    qnet = LatentQNetwork(latent_dim=5, num_actions=3, hidden_dim=16, num_layers=1)
    target = make_latent_target_network(qnet)
    teacher = QNetwork(grid_size=2, num_actions=3, conv_channels=4, hidden_dim=16)
    optimizer = make_latent_q_optimizer(qnet, learning_rate=1e-3)

    result = train_latent_dqn(
        qnet, target, _buffer_with_observations(), optimizer,
        batch_size=16, num_updates=3,
        teacher_q_network=teacher, distill_weight=0.5,
        distill_mode="policy",
    )

    assert result["updates"] == 3.0
    assert np.isfinite(result["latent_td_loss"])
    assert np.isfinite(result["latent_distill_loss"])


def test_latent_dqn_dyna_update_runs():
    qnet = LatentQNetwork(latent_dim=5, num_actions=3, hidden_dim=16, num_layers=1)
    target = make_latent_target_network(qnet)
    world_model = WorldModel(latent_dim=5, num_actions=3, hidden_dim=16, num_layers=1)
    optimizer = make_latent_q_optimizer(qnet, learning_rate=1e-3)

    result = train_latent_dqn_dyna(
        qnet, target, world_model, _buffer(), optimizer,
        num_actions=3, batch_size=16, num_updates=2,
    )

    assert result["updates"] == 2.0
    assert np.isfinite(result["latent_dyna_loss"])
