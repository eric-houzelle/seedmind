import numpy as np

from seedmind.agent.value_model import ValueModel
from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.value import (
    evaluate_value_model_on_returns,
    make_value_optimizer,
    train_value_model,
    train_value_model_dyna,
    train_value_model_on_returns,
)


def test_weighted_value_model_update_runs():
    buffer = ExperienceBuffer(seed=0)
    obs = {"grid": np.zeros((2, 2), dtype=np.int16)}
    for i in range(32):
        exp = make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=-1.0 if i % 4 == 3 else 0.1,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=i % 4 == 3,
            latent_state=np.full(4, i / 32, dtype=np.float32),
            next_latent_state=np.full(4, (i + 1) / 32, dtype=np.float32),
            action_index=0,
            obs_state=obs,
            next_obs_state=obs,
        )
        exp["reward_learning"] = exp["reward_external"]
        buffer.add(exp)

    value_model = ValueModel(latent_dim=4, hidden_dim=16, num_layers=1)
    target_value_model = ValueModel(latent_dim=4, hidden_dim=16, num_layers=1)
    target_value_model.load_state_dict(value_model.state_dict())
    optimizer = make_value_optimizer(value_model, learning_rate=1e-3)

    result = train_value_model(
        value_model,
        target_value_model,
        buffer,
        optimizer,
        batch_size=16,
        num_updates=3,
        reward_key="reward_learning",
        target_abs_weight=1.0,
        terminal_weight=2.0,
        td_error_weight=0.5,
        max_weight=5.0,
    )

    assert result["updates"] == 3.0
    assert np.isfinite(result["value_loss"])


def test_dyna_value_model_update_runs():
    buffer = ExperienceBuffer(seed=0)
    obs = {"grid": np.zeros((2, 2), dtype=np.int16)}
    for i in range(24):
        buffer.add(make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=0.1,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=False,
            latent_state=np.full(4, i / 24, dtype=np.float32),
            next_latent_state=np.full(4, (i + 1) / 24, dtype=np.float32),
            action_index=i % 3,
            obs_state=obs,
            next_obs_state=obs,
        ))

    value_model = ValueModel(latent_dim=4, hidden_dim=16, num_layers=1)
    target_value_model = ValueModel(latent_dim=4, hidden_dim=16, num_layers=1)
    target_value_model.load_state_dict(value_model.state_dict())
    world_model = WorldModel(latent_dim=4, num_actions=3, hidden_dim=16, num_layers=1)
    optimizer = make_value_optimizer(value_model, learning_rate=1e-3)

    result = train_value_model_dyna(
        value_model,
        target_value_model,
        world_model,
        buffer,
        optimizer,
        batch_size=8,
        num_updates=2,
        loss_weight=0.25,
    )

    assert result["updates"] == 2.0
    assert np.isfinite(result["value_dyna_loss"])


def test_posthoc_value_model_return_calibration_runs():
    value_model = ValueModel(latent_dim=4, hidden_dim=16, num_layers=1)
    optimizer = make_value_optimizer(value_model, learning_rate=1e-3)
    latents = np.stack([
        np.full(4, i / 32, dtype=np.float32)
        for i in range(32)
    ])
    returns = np.asarray([
        -1.0 + i / 32
        for i in range(32)
    ], dtype=np.float32)

    result = train_value_model_on_returns(
        value_model,
        latents,
        returns,
        optimizer,
        sample_weights=np.linspace(1.0, 2.0, len(returns), dtype=np.float32),
        batch_size=8,
        num_updates=4,
    )
    metrics = evaluate_value_model_on_returns(value_model, latents, returns)

    assert result["updates"] == 4.0
    assert np.isfinite(result["value_return_loss"])
    assert np.isfinite(metrics["mae"])
