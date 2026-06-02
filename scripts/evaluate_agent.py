"""Compare a naive agent vs a trained agent (SPEC sections 30 & 24).

Works for both V1 (planner) and V2 (learned policy) checkpoints: if the
checkpoint contains a learned Q-network it is used for the trained agent,
otherwise the planner is used.

    python scripts/evaluate_agent.py --checkpoint runs/v2_0/checkpoint_final.pt --config configs/v2_gridworld.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.agent import Agent
from seedmind.envs.colored_gridworld import ColoredGridWorld
from seedmind.envs.gridworld import ACTIONS
from seedmind.envs.procedural_gridworld import ProceduralGridWorld
from seedmind.evaluation.scenarios import compare_agents
from seedmind.training.checkpointing import load_checkpoint


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive vs trained agent comparison")
    parser.add_argument("--checkpoint", default=None, help="trained agent checkpoint")
    parser.add_argument("--config", default="configs/v2_gridworld.yaml")
    parser.add_argument("--num-maps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--colors", nargs="*", default=None, help="restrict eval colors")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.get("env", {})
    size = int(env_cfg.get("size", 8))
    max_steps = args.max_steps if args.max_steps is not None else int(env_cfg.get("max_steps", 80))
    colored = bool(env_cfg.get("colored", False))

    vis_r = env_cfg.get("visibility_radius")
    if vis_r is not None:
        vis_r = int(vis_r)

    def make_env(seed: int):
        if colored:
            return ColoredGridWorld(
                size=size, max_steps=max_steps, allowed_colors=args.colors,
                num_distractor_doors=int(env_cfg.get("num_distractor_doors", 1)),
                num_distractor_keys=int(env_cfg.get("num_distractor_keys", 1)),
                num_dangers=int(env_cfg.get("num_dangers", 2)),
                visibility_radius=vis_r, seed=2000 + seed,
            )
        return ProceduralGridWorld(size=size, max_steps=max_steps,
                                   visibility_radius=vis_r, seed=2000 + seed)

    # Naive agent: random exploration, no learned policy.
    naive = Agent.from_config(
        config, actions=ACTIONS, grid_size=size,
        use_planner=False, learned_policy=False, seed=args.seed,
    )
    naive.policy.total_steps = 0

    # Trained agent: build with a learned policy slot, then load the checkpoint.
    trained = Agent.from_config(
        config, actions=ACTIONS, grid_size=size,
        use_planner=True, learned_policy=True, seed=args.seed,
    )
    if args.checkpoint:
        info = load_checkpoint(args.checkpoint, trained)
        print(f"Loaded trained agent from {args.checkpoint}")
        if not info.get("has_q_network"):
            # V1 checkpoint: fall back to the planner.
            trained.q_network = None
            trained.use_planner = True
            print("(checkpoint has no learned policy; using planner)")
    # Exploit: push exploration to its floor.
    trained.policy.total_steps = trained.policy.epsilon_decay_steps

    summary = compare_agents(
        make_env, {"naive": naive, "trained": trained},
        num_maps=args.num_maps, max_steps=max_steps,
    )

    label = "colored" if colored else "procedural"
    color_note = f" (colors={args.colors})" if args.colors else ""
    print(f"\nComparison over {args.num_maps} {label} maps{color_note}:")
    header = f"{'agent':>10} | {'success':>8} | {'avg_steps':>9} | {'pred_err':>9} | {'mem':>5}"
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        print(
            f"{name:>10} | {stats['success_rate']:>8.2f} | "
            f"{stats['mean_steps_to_success']:>9.1f} | "
            f"{stats['mean_prediction_error']:>9.4f} | "
            f"{stats['mean_memory_items']:>5.0f}"
        )


if __name__ == "__main__":
    main()
