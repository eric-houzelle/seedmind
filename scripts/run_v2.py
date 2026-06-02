"""SeedMind V2 demo (SPEC sections 15 & 24).

Runs the main agent loop on a ColoredGridWorld with a *learned* policy (DQN with
its own CNN encoder), while still training the World Model for curiosity. Logs a
rising success rate and a decaying exploration rate, and saves a checkpoint.

    python scripts/run_v2.py --episodes 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.agent import Agent
from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import ACTIONS
from seedmind.evaluation.metrics import MetricsLogger
from seedmind.evaluation.scenarios import run_episode
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.checkpointing import save_checkpoint
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_bc,
    train_dqn,
)
from seedmind.training.train import make_optimizer, train_world_model
from seedmind.visualization.plot_metrics import plot_metrics
from seedmind.visualization.render_grid import render_ascii


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(config: dict, seed: int, allowed_colors=None):
    env_cfg = config.get("env", {})
    vis_r = env_cfg.get("visibility_radius")
    if vis_r is not None:
        vis_r = int(vis_r)
    return ColoredGridWorld(
        size=int(env_cfg.get("size", 8)),
        max_steps=int(env_cfg.get("max_steps", 80)),
        allowed_colors=allowed_colors,
        num_distractor_doors=int(env_cfg.get("num_distractor_doors", 1)),
        num_distractor_keys=int(env_cfg.get("num_distractor_keys", 1)),
        num_dangers=int(env_cfg.get("num_dangers", 2)),
        visibility_radius=vis_r,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SeedMind V2 demo (learned policy)")
    parser.add_argument("--config", default="configs/v2_gridworld.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})
    wm_cfg = config.get("world_model", {})
    dqn_cfg = config.get("dqn", {})

    episodes = args.episodes if args.episodes is not None else int(train_cfg.get("episodes", 2000))
    max_steps = args.max_steps if args.max_steps is not None else int(env_cfg.get("max_steps", 80))
    train_every = int(train_cfg.get("train_every", 1))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 500))
    size = int(env_cfg.get("size", 8))

    wm_batch = int(wm_cfg.get("batch_size", 64))
    wm_lr = float(wm_cfg.get("learning_rate", 3e-4))

    q_batch = int(dqn_cfg.get("batch_size", 64))
    q_lr = float(dqn_cfg.get("learning_rate", 1e-3))
    gamma = float(dqn_cfg.get("gamma", 0.99))
    target_update = int(dqn_cfg.get("target_update", 500))
    double_dqn = bool(dqn_cfg.get("double_dqn", True))
    updates_per_train = int(dqn_cfg.get("updates_per_train", 8))
    bc_warmup_updates = int(dqn_cfg.get("bc_warmup_updates", 0))
    sampler = str(dqn_cfg.get("sampler", "uniform"))
    curiosity_weight = float(dqn_cfg.get("curiosity_weight", 0.0))

    out_dir = Path(args.out_dir or f"runs/v2_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent.from_config(
        config, actions=ACTIONS, grid_size=size,
        use_planner=False, learned_policy=True, seed=args.seed,
    )
    buffer = ExperienceBuffer(seed=args.seed)
    demo_buffer = ExperienceBuffer(capacity=20_000, seed=args.seed)
    wm_optimizer = make_optimizer(agent.world_model, learning_rate=wm_lr)
    q_optimizer = make_q_optimizer(agent.q_network, learning_rate=q_lr)
    target_network = make_target_network(agent.q_network)
    logger = MetricsLogger()

    last_wm_loss = 0.0
    last_td_loss = 0.0
    successes = 0
    total_q_updates = 0
    next_target_sync = target_update
    bc_done = False

    print(f"Running SeedMind V2: {episodes} episodes, max_steps={max_steps}")

    for ep in range(episodes):
        env = build_env(config, seed=args.seed + ep)
        sink: list = []
        metrics = run_episode(env, agent, episode_index=ep, max_steps=max_steps,
                              buffer=buffer, episode_sink=sink)
        if metrics.success:
            for e in sink:
                demo_buffer.add(e)

        # Behavioral-cloning warm-start from whole successful trajectories.
        if not bc_done and bc_warmup_updates > 0 and len(demo_buffer) >= q_batch:
            train_bc(agent.q_network, demo_buffer, q_optimizer,
                     batch_size=q_batch, num_updates=bc_warmup_updates)
            sync_target(agent.q_network, target_network)
            bc_done = True

        if ep % train_every == 0 and len(buffer) >= q_batch:
            wm_losses = train_world_model(
                agent.world_model, buffer, wm_optimizer,
                batch_size=wm_batch, num_updates=updates_per_train,
            )
            last_wm_loss = wm_losses["total"]
            q_losses = train_dqn(
                agent.q_network, target_network, buffer, q_optimizer,
                batch_size=q_batch, gamma=gamma, curiosity_weight=curiosity_weight,
                double_dqn=double_dqn, num_updates=updates_per_train, sampler=sampler,
            )
            last_td_loss = q_losses["td_loss"]
            total_q_updates += int(q_losses["updates"])
            if total_q_updates >= next_target_sync:
                sync_target(agent.q_network, target_network)
                next_target_sync += target_update

        metrics.world_model_loss = last_wm_loss
        logger.log(metrics)
        successes += int(metrics.success)

        if ep % checkpoint_every == 0 and ep > 0:
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, wm_optimizer, buffer,
                metrics={"last": metrics.to_dict()}, config=config,
                q_optimizer=q_optimizer, target_network=target_network,
            )

        if ep % max(1, episodes // 20) == 0 or ep == episodes - 1:
            print(
                f"  ep {ep:5d} | success_rate(100)={logger.success_rate(100):.2f} "
                f"ext={metrics.episode_reward_external:6.2f} "
                f"td_loss={last_td_loss:.4f} wm_loss={last_wm_loss:.4f} "
                f"eps={metrics.exploration_rate:.2f} mem={metrics.memory_items_count}"
            )

    save_checkpoint(
        str(out_dir / "checkpoint_final.pt"), agent, wm_optimizer, buffer,
        metrics={"success_rate": logger.success_rate()}, config=config,
        q_optimizer=q_optimizer, target_network=target_network,
    )
    buffer.save(str(out_dir / "buffer.pkl"))
    agent.memory.save(str(out_dir / "memory.pkl"))
    logger.save(str(out_dir / "metrics.json"))

    if not args.no_plot:
        plot_path = plot_metrics(logger, str(out_dir))
        print(f"Saved metric plots to {plot_path}")

    demo_env = build_env(config, seed=args.seed)
    print("\nExample colored world:")
    print(render_ascii(demo_env.observe()))

    print(f"\nFinal success rate (last 100 eps): {logger.success_rate(100):.2f}")
    print(f"Episodes solved (opened matching door): {successes}/{episodes}")
    print(f"Checkpoint + buffer + memory + metrics saved under {out_dir}/")


if __name__ == "__main__":
    main()
