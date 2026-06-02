"""Standalone World Model training from a saved experience buffer.

    python scripts/train_world_model.py --buffer runs/v1_0/buffer.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.world_model import WorldModel
from seedmind.envs.gridworld import ACTIONS
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.train import make_optimizer, train_world_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the World Model from a buffer")
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--config", default="configs/v1_gridworld.yaml")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--out", default=None, help="path to save world model weights")
    args = parser.parse_args()

    config = load_config(args.config)
    wm_cfg = config.get("world_model", {})
    latent_dim = int(config.get("agent", {}).get("latent_dim", 128))

    buffer = ExperienceBuffer()
    buffer.load(args.buffer)
    print(f"Loaded buffer with {len(buffer)} experiences")

    world_model = WorldModel(
        latent_dim=latent_dim,
        num_actions=len(ACTIONS),
        hidden_dim=int(wm_cfg.get("hidden_dim", 256)),
        num_layers=int(wm_cfg.get("num_layers", 2)),
    )
    optimizer = make_optimizer(world_model, learning_rate=float(wm_cfg.get("learning_rate", 3e-4)))
    batch_size = int(wm_cfg.get("batch_size", 64))

    first_loss = None
    for u in range(0, args.updates, args.log_every):
        losses = train_world_model(
            world_model, buffer, optimizer, batch_size=batch_size, num_updates=args.log_every
        )
        if first_loss is None:
            first_loss = losses["total"]
        print(f"  updates {u + args.log_every:6d} | loss={losses['total']:.5f} "
              f"(state={losses['state']:.5f} reward={losses['reward']:.5f})")

    if args.out:
        import torch

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(world_model.state_dict(), args.out)
        print(f"Saved world model weights to {args.out}")

    if first_loss is not None:
        print(f"Loss went from {first_loss:.5f} to {losses['total']:.5f}")


if __name__ == "__main__":
    main()
