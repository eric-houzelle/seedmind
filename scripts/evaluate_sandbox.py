"""Evaluate a trained sandbox agent vs a naive (random) baseline.

    python scripts/evaluate_sandbox.py --checkpoint runs/sandbox_0/checkpoint_final.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sandbox import build_env, sandbox_actions
from seedmind.training.device import resolve_device

CAUSAL_EVENTS = (
    "harvest_food",
    "harvest_food_tool",
    "harvest_wood",
    "harvest_stone",
    "craft_tool",
    "eat_ok",
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _empty_stats() -> Dict[str, float]:
    stats = {f"{event}_mean": 0.0 for event in CAUSAL_EVENTS}
    stats.update({
        "food_harvested_mean": 0.0,
        "bonus_food_from_tool_mean": 0.0,
        "episodes_with_tool_rate": 0.0,
        "episodes_with_tool_food_rate": 0.0,
    })
    return stats


def _summarise(lifespans, counters, num_episodes: int) -> Dict[str, float]:
    total = Counter()
    for c in counters:
        total.update(c)
    out = {
        "mean_lifespan": float(np.mean(lifespans)),
        "std": float(np.std(lifespans)),
        "max": float(np.max(lifespans)),
    }
    out.update(_empty_stats())
    denom = max(num_episodes, 1)
    for event in CAUSAL_EVENTS:
        out[f"{event}_mean"] = float(total.get(event, 0)) / denom
    out["food_harvested_mean"] = float(total.get("food_harvested", 0)) / denom
    out["bonus_food_from_tool_mean"] = float(total.get("bonus_food_from_tool", 0)) / denom
    out["episodes_with_tool_rate"] = float(sum(c.get("craft_tool", 0) > 0 for c in counters)) / denom
    out["episodes_with_tool_food_rate"] = float(
        sum(c.get("harvest_food_tool", 0) > 0 for c in counters)
    ) / denom
    return out


def _record_event(counter: Counter, info: Dict) -> None:
    event = str(info.get("event", "unknown"))
    amount = int(info.get("event_amount", 0))
    counter[event] += 1
    if event in {"harvest_food", "harvest_food_tool"}:
        counter["food_harvested"] += amount
    if event == "harvest_food_tool":
        counter["bonus_food_from_tool"] += max(0, amount - 1)


def run_naive(config: dict, num_episodes: int = 100, seed: int = 9999) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    actions = sandbox_actions(config)
    lifespans = []
    counters = []
    for ep in range(num_episodes):
        env = build_env(config, seed=seed + ep)
        env.reset()
        done = False
        steps = 0
        counter = Counter()
        while not done:
            action = actions[int(rng.integers(len(actions)))]
            _, _, done, info = env.step(action)
            _record_event(counter, info)
            steps += 1
        lifespans.append(steps)
        counters.append(counter)
    return _summarise(lifespans, counters, num_episodes)


def run_trained(config: dict, checkpoint_path: str,
                num_episodes: int = 100, seed: int = 9999,
                device: torch.device | None = None) -> Dict[str, float]:
    from scripts.run_sandbox import build_agent

    device = device or torch.device("cpu")
    agent = build_agent(config, seed=0)
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    actions = sandbox_actions(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    agent.policy.epsilon_start = 0.05
    agent.policy.epsilon_end = 0.05

    lifespans = []
    counters = []
    for ep in range(num_episodes):
        env = build_env(config, seed=seed + ep)
        obs = env.reset()
        done = False
        steps = 0
        counter = Counter()
        while not done:
            scorer = agent.q_network.make_scorer(obs, actions)
            action = max(actions, key=scorer)
            obs, _, done, info = env.step(action)
            _record_event(counter, info)
            steps += 1
        lifespans.append(steps)
        counters.append(counter)
    return _summarise(lifespans, counters, num_episodes)


def print_causal_metrics(label: str, stats: Dict[str, float]) -> None:
    print(f"\n  {label} causal metrics per episode:")
    print(
        "    food={food:.2f} tool_food={tool_food:.2f} "
        "wood={wood:.2f} stone={stone:.2f} craft={craft:.2f} eat={eat:.2f}".format(
            food=stats["food_harvested_mean"],
            tool_food=stats["harvest_food_tool_mean"],
            wood=stats["harvest_wood_mean"],
            stone=stats["harvest_stone_mean"],
            craft=stats["craft_tool_mean"],
            eat=stats["eat_ok_mean"],
        )
    )
    print(
        "    episodes_with_tool={tool_rate:.2%} "
        "episodes_with_tool_food={tool_food_rate:.2%} "
        "bonus_food_from_tool={bonus:.2f}".format(
            tool_rate=stats["episodes_with_tool_rate"],
            tool_food_rate=stats["episodes_with_tool_food_rate"],
            bonus=stats["bonus_food_from_tool_mean"],
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/sandbox_v0.yaml")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print("=== Naive (random) agent ===")
    naive = run_naive(config, num_episodes=args.num_episodes)
    print(f"  Mean lifespan: {naive['mean_lifespan']:.1f} +/- {naive['std']:.1f}  (max {naive['max']:.0f})")
    print_causal_metrics("Naive", naive)

    print("\n=== Trained agent ===")
    trained = run_trained(
        config, args.checkpoint, num_episodes=args.num_episodes, device=device,
    )
    print(f"  Mean lifespan: {trained['mean_lifespan']:.1f} +/- {trained['std']:.1f}  (max {trained['max']:.0f})")
    print_causal_metrics("Trained", trained)

    ratio = trained["mean_lifespan"] / max(naive["mean_lifespan"], 1)
    print(f"\n  Ratio trained/naive: {ratio:.2f}x")
    if ratio > 1.2:
        print("  => L'agent entraine survit significativement plus longtemps !")
    elif ratio > 1.0:
        print("  => Leger avantage pour l'agent entraine.")
    else:
        print("  => L'agent n'a pas encore appris a survivre mieux que le hasard.")

    craft_delta = trained["craft_tool_mean"] - naive["craft_tool_mean"]
    tool_food_delta = trained["harvest_food_tool_mean"] - naive["harvest_food_tool_mean"]
    if craft_delta > 0.05 and tool_food_delta > 0.05:
        print("  => Signal craft: l'agent utilise davantage outil -> recolte amelioree.")
    else:
        print("  => Signal craft faible: survie a interpreter sans preuve craft directe.")


if __name__ == "__main__":
    main()
