"""SeedMind V1 demo (SPEC sections 17 & 29).

Runs the main agent loop on a GridWorld: the agent moves, collects
experiences, the World Model trains, metrics are logged and plotted, and a
checkpoint is saved.

    python scripts/run_v1.py --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.agent import Agent
from seedmind.envs.gridworld import ACTIONS, GridWorld
from seedmind.envs.procedural_gridworld import ProceduralGridWorld
from seedmind.evaluation.metrics import MetricsLogger
from seedmind.evaluation.scenarios import run_episode
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.checkpointing import save_checkpoint
from seedmind.training.train import make_optimizer, train_world_model
from seedmind.visualization.plot_metrics import plot_metrics
from seedmind.visualization.render_grid import render_ascii


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(config: dict, seed: int):
    env_cfg = config.get("env", {})
    size = int(env_cfg.get("size", 10))
    max_steps = int(env_cfg.get("max_steps", 100))
    if env_cfg.get("procedural", True):
        return ProceduralGridWorld(size=size, max_steps=max_steps, seed=seed)
    return GridWorld(size=size, max_steps=max_steps, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="SeedMind V1 demo")
    parser.add_argument("--config", default="configs/v1_gridworld.yaml")
    parser.add_argument("--episodes", type=int, default=None, help="override config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})
    wm_cfg = config.get("world_model", {})

    episodes = args.episodes if args.episodes is not None else int(train_cfg.get("episodes", 1000))
    max_steps = args.max_steps if args.max_steps is not None else int(env_cfg.get("max_steps", 100))
    train_every = int(train_cfg.get("train_every", 10))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 500))
    batch_size = int(wm_cfg.get("batch_size", 64))
    lr = float(wm_cfg.get("learning_rate", 3e-4))

    out_dir = Path(args.out_dir or f"runs/v1_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    size = int(env_cfg.get("size", 10))
    agent = Agent.from_config(config, actions=ACTIONS, grid_size=size, use_planner=True, seed=args.seed)
    buffer = ExperienceBuffer(seed=args.seed)
    optimizer = make_optimizer(agent.world_model, learning_rate=lr)
    logger = MetricsLogger()

    last_loss = 0.0
    successes = 0
    print(f"Running SeedMind V1: {episodes} episodes, max_steps={max_steps}")

    for ep in range(episodes):
        env = build_env(config, seed=args.seed + ep)
        metrics = run_episode(env, agent, episode_index=ep, max_steps=max_steps, buffer=buffer)

        if ep % train_every == 0 and len(buffer) >= batch_size:
            losses = train_world_model(
                agent.world_model, buffer, optimizer,
                batch_size=batch_size, num_updates=max(1, train_every * 2),
            )
            last_loss = losses["total"]

        metrics.world_model_loss = last_loss
        logger.log(metrics)
        successes += int(metrics.success)

        if ep % checkpoint_every == 0 and ep > 0:
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, optimizer, buffer,
                metrics={"last": metrics.to_dict()}, config=config,
            )

        if ep % max(1, episodes // 10) == 0 or ep == episodes - 1:
            print(
                f"  ep {ep:5d} | ext={metrics.episode_reward_external:6.2f} "
                f"intr={metrics.episode_reward_intrinsic:5.2f} "
                f"pred_err={metrics.prediction_error_mean:.4f} "
                f"wm_loss={last_loss:.4f} eps={metrics.exploration_rate:.2f} "
                f"mem={metrics.memory_items_count} success={metrics.success}"
            )

    # Final artifacts.
    save_checkpoint(
        str(out_dir / "checkpoint_final.pt"), agent, optimizer, buffer,
        metrics={"success_rate": logger.success_rate()}, config=config,
    )
    buffer.save(str(out_dir / "buffer.pkl"))
    agent.memory.save(str(out_dir / "memory.pkl"))
    logger.save(str(out_dir / "metrics.json"))

    if not args.no_plot:
        plot_path = plot_metrics(logger, str(out_dir))
        print(f"Saved metric plots to {plot_path}")

    # Show a final grid and a quick learning summary.
    demo_env = build_env(config, seed=args.seed)
    print("\nExample world:")
    print(render_ascii(demo_env.observe()))

    losses = logger.series("world_model_loss")
    nonzero = [l for l in losses if l > 0]
    if len(nonzero) >= 2:
        print(f"\nWorld model loss: first={nonzero[0]:.4f} -> last={nonzero[-1]:.4f}")
    print(f"Episodes solved (reached reward): {successes}/{episodes}")
    print(f"Checkpoint + buffer + memory + metrics saved under {out_dir}/")


if __name__ == "__main__":
    main()
