import numpy as np
import torch

from seedmind.agent.world_model import WorldModel
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.training.train import (
    make_optimizer,
    train_world_model,
    train_world_model_uncertainty_head,
)


def test_world_model_uncertainty_loss_update_runs():
    buffer = ExperienceBuffer(seed=0)
    for i in range(32):
        current_features = np.asarray([i / 32, (i % 3) / 3], dtype=np.float32)
        next_features = current_features + np.asarray([0.01, -0.02], dtype=np.float32)
        buffer.add(make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=1.0 if i % 5 == 0 else -0.1,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=i % 4 == 3,
            latent_state=np.full(5, i / 32, dtype=np.float32),
            next_latent_state=np.full(5, (i + 1) / 32, dtype=np.float32),
            action_index=i % 3,
            causal_features=current_features,
            next_causal_features=next_features,
            event_index=i % 4,
            event=f"event_{i % 4}",
        ))

    world_model = WorldModel(
        latent_dim=5,
        num_actions=3,
        hidden_dim=16,
        num_layers=1,
        causal_feature_dim=2,
        num_events=4,
    )
    optimizer = make_optimizer(world_model, learning_rate=1e-3)

    result = train_world_model(
        world_model,
        buffer,
        optimizer,
        batch_size=16,
        num_updates=3,
        sampler="causal",
        causal_feature_weight=1.0,
        causal_event_weight=0.2,
        event_sample_names={"event_3"},
        event_sample_name_weight=2.0,
        event_sample_done_weight=1.0,
        uncertainty_weight=0.5,
    )

    assert result["updates"] == 3.0
    assert np.isfinite(result["total"])
    assert np.isfinite(result["uncertainty"])


def test_world_model_uncertainty_detach_keeps_trunk_grad_free():
    world_model = WorldModel(
        latent_dim=5,
        num_actions=3,
        hidden_dim=16,
        num_layers=1,
        causal_feature_dim=2,
        num_events=4,
    )
    latents = torch.randn(8, 5)
    actions = torch.randint(0, 3, (8,))

    outputs = world_model.forward_aux(latents, actions, detach_uncertainty=True)
    loss = outputs["uncertainty"].sum()
    loss.backward()

    trunk_weight = world_model.trunk[0].weight
    assert world_model.uncertainty_head.weight.grad is not None
    assert trunk_weight.grad is None


def test_world_model_uncertainty_head_calibration_runs():
    buffer = ExperienceBuffer(seed=1)
    for i in range(24):
        current_features = np.asarray([i / 24, (i % 2) / 2], dtype=np.float32)
        next_features = current_features + np.asarray([0.02, -0.01], dtype=np.float32)
        buffer.add(make_experience(
            episode_id=f"e{i // 4}",
            world_id="w",
            step=i % 4,
            observation=None,
            action="A",
            next_observation=None,
            reward_external=0.5 if i % 6 == 0 else -0.05,
            reward_intrinsic=0.0,
            goal="g",
            prediction_error=0.0,
            done=i % 4 == 3,
            latent_state=np.full(5, i / 24, dtype=np.float32),
            next_latent_state=np.full(5, (i + 1) / 24, dtype=np.float32),
            action_index=i % 3,
            causal_features=current_features,
            next_causal_features=next_features,
            event_index=i % 4,
            event=f"event_{i % 4}",
        ))

    world_model = WorldModel(
        latent_dim=5,
        num_actions=3,
        hidden_dim=16,
        num_layers=1,
        causal_feature_dim=2,
        num_events=4,
    )
    optimizer = make_optimizer(world_model, learning_rate=1e-3)
    result = train_world_model_uncertainty_head(
        world_model,
        buffer,
        optimizer,
        batch_size=12,
        num_updates=2,
        causal_feature_weight=1.0,
        causal_event_weight=0.2,
    )

    assert result["updates"] == 2.0
    assert np.isfinite(result["uncertainty"])
