"""Tests for MicroFouloide observation encoding."""
from __future__ import annotations

import numpy as np

from seedmind.agent.micro_fouloide_encoder import (
    MICRO_FOULOIDE_NUM_SCALARS,
    micro_fouloide_observation_to_vector,
    micro_fouloide_qnet_scalars,
)
from seedmind.agent.encoder import Encoder
from seedmind.envs.micro_fouloide_world import FOOD, NUM_ENTITIES
from scripts.run_micro_fouloide import _compact_obs


def test_scalars_include_drives_and_standing_entity_onehot():
    obs = {
        "grid": np.zeros((3, 3), dtype=np.int64),
        "energy": 0.1,
        "hydration": 0.2,
        "temperature": 0.3,
        "health": 0.4,
        "standing_entity": FOOD,
    }

    scalars = micro_fouloide_qnet_scalars(obs)

    assert scalars.shape == (MICRO_FOULOIDE_NUM_SCALARS,)
    assert MICRO_FOULOIDE_NUM_SCALARS == 4 + NUM_ENTITIES
    np.testing.assert_allclose(scalars[:4], np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    standing = scalars[4:]
    assert standing.sum() == 1.0
    assert standing[FOOD] == 1.0


def test_observation_vector_includes_scalar_tail():
    obs = {
        "grid": np.zeros((2, 2), dtype=np.int64),
        "energy": 0.1,
        "hydration": 0.2,
        "temperature": 0.3,
        "health": 0.4,
        "standing_entity": FOOD,
    }

    vector = micro_fouloide_observation_to_vector(obs)

    assert vector.shape == (2 * 2 * NUM_ENTITIES + MICRO_FOULOIDE_NUM_SCALARS,)
    assert vector[-NUM_ENTITIES + FOOD] == 1.0


def test_compact_observation_preserves_standing_entity_for_replay():
    obs = {
        "grid": np.zeros((2, 2), dtype=np.int64),
        "energy": 0.1,
        "hydration": 0.2,
        "temperature": 0.3,
        "health": 0.4,
        "standing_entity": FOOD,
    }

    compact = _compact_obs(obs)

    assert compact["standing_entity"] == FOOD
    assert micro_fouloide_qnet_scalars(compact)[4 + FOOD] == 1.0


def test_structured_latent_tail_preserves_adapter_features():
    obs = {
        "grid": np.zeros((2, 2), dtype=np.int64),
        "energy": 0.1,
        "hydration": 0.2,
        "temperature": 0.3,
        "health": 0.4,
        "standing_entity": FOOD,
    }
    features = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
    encoder = Encoder(
        grid_size=2,
        latent_dim=8,
        num_entities=NUM_ENTITIES,
        seed=0,
        input_dim=2 * 2 * NUM_ENTITIES + MICRO_FOULOIDE_NUM_SCALARS,
        obs_to_vec_fn=micro_fouloide_observation_to_vector,
        structured_features_fn=lambda _: features,
        structured_feature_dim=len(features),
    )

    latent = encoder.encode(obs)

    assert latent.shape == (8,)
    np.testing.assert_allclose(latent[-len(features):], features)
