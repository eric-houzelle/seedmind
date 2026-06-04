import numpy as np
import torch

from seedmind.agent.curiosity import (
    compute_prediction_error,
    compute_prediction_error_tensor,
)
from seedmind.agent.encoder import Encoder
from seedmind.agent.world_model import WorldModel
from seedmind.training.latent_utils import latent_to_numpy


def test_encode_tensor_matches_encode():
    obs = {
        "grid": np.zeros((4, 4), dtype=np.int64),
        "energy": 10.0,
        "energy_max": 100.0,
        "inventory_food": 0,
    }
    enc = Encoder(grid_size=4, latent_dim=8, seed=0)
    np_latent = enc.encode(obs)
    t_latent = enc.encode_tensor(obs)
    assert np.allclose(np_latent, latent_to_numpy(t_latent), atol=1e-5)


def test_predict_tensor_matches_predict():
    wm = WorldModel(latent_dim=8, num_actions=4)
    latent = np.random.randn(8).astype(np.float32)
    pred_np, r_np, u_np = wm.predict(latent, 2)
    pred_t, r_t, u_t = wm.predict_tensor(torch.from_numpy(latent), 2)
    assert np.allclose(pred_np, latent_to_numpy(pred_t), atol=1e-5)
    assert abs(r_np - float(r_t.item())) < 1e-5
    assert abs(u_np - float(u_t.item())) < 1e-5


def test_prediction_error_tensor_matches_numpy():
    pred = torch.tensor([1.0, 0.0], dtype=torch.float32)
    actual = torch.tensor([0.0, 0.0], dtype=torch.float32)
    err_t = float(compute_prediction_error_tensor(pred, actual).item())
    err_np = compute_prediction_error(pred.numpy(), actual.numpy())
    assert abs(err_t - err_np) < 1e-6
