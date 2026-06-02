"""Rule-transfer experiment (SPEC section 24.6).

Trains a learned-policy agent on a subset of colors (e.g. red + green), then
measures how well it does on a held-out color (e.g. blue) it never trained on,
compared to a naive agent. If the agent learned the abstract rule "the key
whose color matches the door opens it", the skill transfers.

    python scripts/transfer_experiment.py --episodes 1200
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
from seedmind.evaluation.scenarios import run_episode
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_bc,
    train_dqn,
)
from seedmind.training.train import make_optimizer, train_world_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config: dict, seed: int, allowed_colors):
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


def train_agent(config, allowed_colors, episodes, max_steps, seed):
    size = int(config.get("env", {}).get("size", 8))
    dqn_cfg = config.get("dqn", {})
    curiosity_weight = float(dqn_cfg.get("curiosity_weight", 0.0))
    sampler = str(dqn_cfg.get("sampler", "uniform"))

    agent = Agent.from_config(
        config, actions=ACTIONS, grid_size=size,
        use_planner=False, learned_policy=True, seed=seed,
    )
    buffer = ExperienceBuffer(seed=seed)
    demo_buffer = ExperienceBuffer(capacity=20_000, seed=seed)
    wm_opt = make_optimizer(agent.world_model, learning_rate=float(config["world_model"]["learning_rate"]))
    q_opt = make_q_optimizer(agent.q_network, learning_rate=float(dqn_cfg.get("learning_rate", 1e-3)))
    target = make_target_network(agent.q_network)

    q_batch = int(dqn_cfg.get("batch_size", 64))
    updates = int(dqn_cfg.get("updates_per_train", 8))
    target_update = int(dqn_cfg.get("target_update", 500))
    bc_warmup = int(dqn_cfg.get("bc_warmup_updates", 100))
    bc_done = False
    total_updates = 0
    next_sync = target_update

    for ep in range(episodes):
        env = make_env(config, seed=seed + ep, allowed_colors=allowed_colors)
        sink: list = []
        m = run_episode(env, agent, episode_index=ep, max_steps=max_steps,
                        buffer=buffer, episode_sink=sink)
        if m.success:
            for e in sink:
                demo_buffer.add(e)

        if not bc_done and bc_warmup > 0 and len(demo_buffer) >= q_batch:
            train_bc(agent.q_network, demo_buffer, q_opt, batch_size=q_batch, num_updates=bc_warmup)
            sync_target(agent.q_network, target)
            bc_done = True

        if len(buffer) >= q_batch:
            train_world_model(agent.world_model, buffer, wm_opt,
                              batch_size=int(config["world_model"]["batch_size"]), num_updates=updates)
            res = train_dqn(agent.q_network, target, buffer, q_opt,
                            batch_size=q_batch, gamma=float(dqn_cfg.get("gamma", 0.95)),
                            curiosity_weight=curiosity_weight, sampler=sampler,
                            double_dqn=bool(dqn_cfg.get("double_dqn", True)), num_updates=updates)
            total_updates += int(res["updates"])
            if total_updates >= next_sync:
                sync_target(agent.q_network, target)
                next_sync += target_update
    return agent


def eval_success(config, agent, allowed_colors, num_maps, max_steps, exploit=True):
    if exploit:
        agent.policy.total_steps = agent.policy.epsilon_decay_steps
    successes = 0
    for seed in range(num_maps):
        env = make_env(config, seed=5000 + seed, allowed_colors=allowed_colors)
        metrics = run_episode(env, agent, episode_index=seed, max_steps=max_steps,
                              buffer=None, store_memory=False)
        successes += int(metrics.success)
    return successes / num_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule transfer experiment (SPEC 24.6)")
    parser.add_argument("--config", default="configs/v2_gridworld.yaml")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--train-colors", nargs="*", default=["red", "green"])
    parser.add_argument("--held-out", default="blue")
    parser.add_argument("--num-maps", type=int, default=40)
    parser.add_argument("--adapt-episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    max_steps = int(config.get("env", {}).get("max_steps", 80))

    print(f"Training on colors {args.train_colors} for {args.episodes} episodes...")
    trained = train_agent(config, args.train_colors, args.episodes, max_steps, args.seed)

    naive = Agent.from_config(config, actions=ACTIONS,
                              grid_size=int(config["env"]["size"]),
                              use_planner=False, learned_policy=False, seed=args.seed)
    naive.policy.total_steps = 0

    train_success = eval_success(config, trained, args.train_colors, args.num_maps, max_steps)
    zeroshot = eval_success(config, trained, [args.held_out], args.num_maps, max_steps)
    naive_held = eval_success(config, naive, [args.held_out], args.num_maps, max_steps)

    # Few-shot adaptation on the held-out color.
    adapted = train_agent_resume(config, trained, [args.held_out], args.adapt_episodes, max_steps, args.seed)
    fewshot = eval_success(config, adapted, [args.held_out], args.num_maps, max_steps)

    print("\n=== Rule transfer results ===")
    print(f"Trained colors {args.train_colors}: success={train_success:.2f}")
    print(f"Held-out '{args.held_out}' zero-shot (trained):   {zeroshot:.2f}")
    print(f"Held-out '{args.held_out}' (naive baseline):       {naive_held:.2f}")
    print(f"Held-out '{args.held_out}' after {args.adapt_episodes} adapt eps: {fewshot:.2f}")
    verdict = "TRANSFER" if zeroshot > naive_held + 0.05 else "WEAK/NO transfer (zero-shot)"
    print(f"\nVerdict: {verdict}; few-shot adaptation reaches {fewshot:.2f}.")


def train_agent_resume(config, agent, allowed_colors, episodes, max_steps, seed):
    """Continue training an existing agent on new colors (few-shot adaptation)."""
    dqn_cfg = config.get("dqn", {})
    curiosity_weight = float(dqn_cfg.get("curiosity_weight", 0.0))
    sampler = str(dqn_cfg.get("sampler", "uniform"))
    buffer = ExperienceBuffer(seed=seed)
    wm_opt = make_optimizer(agent.world_model, learning_rate=float(config["world_model"]["learning_rate"]))
    q_opt = make_q_optimizer(agent.q_network, learning_rate=float(dqn_cfg.get("learning_rate", 1e-3)))
    target = make_target_network(agent.q_network)
    q_batch = int(dqn_cfg.get("batch_size", 64))
    updates = int(dqn_cfg.get("updates_per_train", 8))
    # Re-enable some exploration for adaptation.
    agent.policy.total_steps = int(agent.policy.epsilon_decay_steps * 0.5)

    for ep in range(episodes):
        env = make_env(config, seed=seed + 9000 + ep, allowed_colors=allowed_colors)
        run_episode(env, agent, episode_index=ep, max_steps=max_steps, buffer=buffer)
        if len(buffer) >= q_batch:
            train_world_model(agent.world_model, buffer, wm_opt,
                              batch_size=int(config["world_model"]["batch_size"]), num_updates=updates)
            train_dqn(agent.q_network, target, buffer, q_opt,
                      batch_size=q_batch, gamma=float(dqn_cfg.get("gamma", 0.95)),
                      curiosity_weight=curiosity_weight, sampler=sampler,
                      double_dqn=bool(dqn_cfg.get("double_dqn", True)), num_updates=updates)
            if ep % 50 == 0:
                sync_target(agent.q_network, target)
    return agent


if __name__ == "__main__":
    main()
