"""Evaluate a trained sandbox agent vs a naive (random) baseline.

    python scripts/evaluate_sandbox.py --checkpoint runs/sandbox_0/checkpoint_final.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_sandbox import build_env, observation_causal_features, sandbox_actions
from seedmind.envs.sandbox_world import CRAFT, EAT, HARVEST
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
                device: torch.device | None = None,
                decision_mode: str = "q",
                planning_weight: float | None = None,
                planner_horizon: int | None = None,
                planner_samples: int | None = None) -> Dict[str, float]:
    from scripts.run_sandbox import build_agent

    device = device or torch.device("cpu")
    if decision_mode == "planner":
        config = dict(config)
        planning = dict(config.get("planning", {}))
        planning["enabled"] = True
        if planning_weight is not None:
            planning["weight"] = planning_weight
        if planner_horizon is not None:
            planning["horizon"] = planner_horizon
        if planner_samples is not None:
            planning["num_samples"] = planner_samples
        config["planning"] = planning
    agent = build_agent(config, seed=0)
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    actions = sandbox_actions(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    if "world_model_state" in ckpt:
        agent.world_model.load_state_dict(ckpt["world_model_state"])
    if "encoder_state" in ckpt:
        agent.encoder.load_state_dict(ckpt["encoder_state"])
    agent.policy.epsilon_start = 0.0
    agent.policy.epsilon_end = 0.0

    lifespans = []
    counters = []
    for ep in range(num_episodes):
        env = build_env(config, seed=seed + ep)
        obs = env.reset()
        done = False
        steps = 0
        counter = Counter()
        latent = agent.encode(obs) if decision_mode == "planner" else None
        while not done:
            if decision_mode == "planner":
                memories = agent.retrieve(latent)
                goal = agent.choose_goal(latent, memories)
                action = agent.choose_action(
                    latent, goal, memories, actions, observation=obs,
                )
            else:
                scorer = agent.q_network.make_scorer(obs, actions)
                action = max(actions, key=scorer)
            obs, _, done, info = env.step(action)
            if decision_mode == "planner":
                latent = agent.encode(obs)
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


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def diagnose_world_model(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
    max_samples: int = 20000,
    batch_size: int = 512,
) -> None:
    from scripts.run_sandbox import build_agent

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    if not buffer:
        print("\n=== World Model diagnostic ===")
        print("  No replay buffer found in checkpoint.")
        return

    agent = build_agent(config, seed=0)
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.world_model.to(device)
    agent.world_model.eval()

    valid = [
        e for e in buffer
        if e.get("latent_state") is not None
        and e.get("next_latent_state") is not None
        and e.get("action_index") is not None
    ]
    if max_samples > 0 and len(valid) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(valid)), size=max_samples, replace=False)
        valid = [valid[int(i)] for i in indices]

    stats = defaultdict(lambda: {
        "n": 0,
        "state_mse": 0.0,
        "reward_mae": 0.0,
        "reward_signed": 0.0,
        "uncertainty": 0.0,
    })

    with torch.no_grad():
        for batch in _chunks(valid, batch_size):
            latents = torch.as_tensor(
                np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            next_latents = torch.as_tensor(
                np.stack([np.asarray(e["next_latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                [int(e["action_index"]) for e in batch],
                dtype=torch.long,
                device=device,
            )
            rewards = torch.as_tensor(
                [float(e.get("reward_external", 0.0)) for e in batch],
                dtype=torch.float32,
                device=device,
            )

            pred_next, pred_reward, uncertainty = agent.world_model(latents, actions)
            state_mse = torch.mean((pred_next - next_latents) ** 2, dim=1).cpu().numpy()
            reward_error = (pred_reward - rewards).cpu().numpy()
            uncertainty_np = uncertainty.cpu().numpy()

            for e, sm, re, un in zip(batch, state_mse, reward_error, uncertainty_np):
                event = str(e.get("event") or "unknown")
                row = stats[event]
                row["n"] += 1
                row["state_mse"] += float(sm)
                row["reward_mae"] += abs(float(re))
                row["reward_signed"] += float(re)
                row["uncertainty"] += float(un)

    print("\n=== World Model diagnostic on replay buffer ===")
    print(f"  samples={len(valid)} checkpoint={checkpoint_path}")
    print("  event                 n   state_mse  reward_mae  reward_bias  uncertainty")
    priority = list(CAUSAL_EVENTS) + ["move", "wait", "wall", "unknown"]
    ordered = [e for e in priority if e in stats]
    ordered += sorted(e for e in stats if e not in set(ordered))
    for event in ordered:
        row = stats[event]
        n = max(int(row["n"]), 1)
        print(
            f"  {event:<18} {n:6d} "
            f"{row['state_mse'] / n:9.5f} "
            f"{row['reward_mae'] / n:10.5f} "
            f"{row['reward_signed'] / n:11.5f} "
            f"{row['uncertainty'] / n:11.5f}"
        )


def _parse_sweep(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _obs_from_replay(experience: dict) -> dict | None:
    obs = experience.get("obs_state")
    if obs is None:
        return None
    return {
        "grid": np.asarray(obs["grid"], dtype=np.int16),
        "energy": float(obs.get("energy", 0.0)),
        "energy_max": float(obs.get("energy_max", 100.0)),
        "inventory_food": int(obs.get("inventory_food", 0)),
        "inventory_wood": int(obs.get("inventory_wood", 0)),
        "inventory_stone": int(obs.get("inventory_stone", 0)),
        "inventory_tool": int(obs.get("inventory_tool", 0)),
    }


def _normalise_scores(values: dict[str, float], actions: list[str]) -> dict[str, float]:
    arr = np.asarray([values[a] for a in actions], dtype=np.float32)
    rng = float(arr.max() - arr.min())
    if rng < 1e-8:
        return {a: 0.5 for a in actions}
    return {a: float((values[a] - float(arr.min())) / rng) for a in actions}


def _rank(scores: dict[str, float], action: str) -> int:
    ordered = sorted(scores, key=scores.get, reverse=True)
    return ordered.index(action) + 1 if action in ordered else len(ordered) + 1


def diagnose_decisions(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
    max_samples: int = 2000,
    planning_weight: float = 0.1,
    planner_horizon: int = 3,
    planner_samples: int = 8,
) -> None:
    from scripts.run_sandbox import build_agent

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    actions = sandbox_actions(config)

    planner_config = dict(config)
    planning = dict(planner_config.get("planning", {}))
    planning.update({
        "enabled": True,
        "weight": planning_weight,
        "horizon": planner_horizon,
        "num_samples": planner_samples,
    })
    planner_config["planning"] = planning

    agent = build_agent(planner_config, seed=0)
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    if "encoder_state" in ckpt:
        agent.encoder.load_state_dict(ckpt["encoder_state"])
    agent.policy.epsilon_start = 0.0
    agent.policy.epsilon_end = 0.0

    candidates = []
    for e in buffer:
        obs = _obs_from_replay(e)
        if obs is None or e.get("latent_state") is None:
            continue
        candidates.append((e, obs))

    if max_samples > 0 and len(candidates) > max_samples:
        rng = np.random.default_rng(67890)
        indices = rng.choice(np.arange(len(candidates)), size=max_samples, replace=False)
        candidates = [candidates[int(i)] for i in indices]

    buckets: dict[str, list[tuple[dict, dict]]] = {
        "all": candidates,
        "craft_ready": [
            x for x in candidates
            if x[1].get("inventory_wood", 0) >= 1 and x[1].get("inventory_stone", 0) >= 1
        ],
        "has_tool": [x for x in candidates if x[1].get("inventory_tool", 0) >= 1],
        "has_food": [x for x in candidates if x[1].get("inventory_food", 0) >= 1],
        "actual_craft_tool": [x for x in candidates if x[0].get("event") == "craft_tool"],
        "actual_tool_food": [x for x in candidates if x[0].get("event") == "harvest_food_tool"],
        "actual_eat_ok": [x for x in candidates if x[0].get("event") == "eat_ok"],
    }

    print("\n=== Decision diagnostic on replay states ===")
    print(
        f"  samples={len(candidates)} planning_weight={planning_weight} "
        f"horizon={planner_horizon} planner_samples={planner_samples}"
    )
    print(
        "  bucket              n | q_best        wm_best       combined_best | "
        "q_rank(C/H/E) wm_rank(C/H/E) combo_rank(C/H/E)"
    )

    tracked = [a for a in (CRAFT, HARVEST, EAT) if a in actions]
    for bucket, rows in buckets.items():
        if not rows:
            print(f"  {bucket:<18} 0")
            continue

        q_best = Counter()
        wm_best = Counter()
        combined_best = Counter()
        q_ranks = Counter()
        wm_ranks = Counter()
        combined_ranks = Counter()

        for e, obs in rows:
            latent = np.asarray(e["latent_state"], dtype=np.float32)
            q_raw = agent.q_network.q_values(obs)
            q_scores = {a: float(q_raw[agent.action_index[a]]) for a in actions}
            current_features = observation_causal_features(config, obs)
            wm_scores = agent.planner.action_values(
                latent, actions, current_features=current_features,
            )
            q_norm = _normalise_scores(q_scores, actions)
            wm_norm = _normalise_scores(wm_scores, actions)
            combined = {
                a: float((1 - planning_weight) * q_norm[a] + planning_weight * wm_norm[a])
                for a in actions
            }

            q_best[max(actions, key=q_scores.get)] += 1
            wm_best[max(actions, key=wm_scores.get)] += 1
            combined_best[max(actions, key=combined.get)] += 1
            for action in tracked:
                q_ranks[action] += _rank(q_scores, action)
                wm_ranks[action] += _rank(wm_scores, action)
                combined_ranks[action] += _rank(combined, action)

        n = len(rows)

        def top(counter: Counter) -> str:
            action, count = counter.most_common(1)[0]
            return f"{action}:{count / n:.0%}"

        def ranks(counter: Counter) -> str:
            return "/".join(f"{counter[a] / n:.1f}" for a in tracked)

        print(
            f"  {bucket:<18} {n:5d} | "
            f"{top(q_best):<13} {top(wm_best):<13} {top(combined_best):<13} | "
            f"{ranks(q_ranks):<13} {ranks(wm_ranks):<14} {ranks(combined_ranks):<15}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/sandbox_v0.yaml")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--compare-planner", action="store_true")
    parser.add_argument("--planning-weight", type=float, default=0.15)
    parser.add_argument("--planner-horizon", type=int, default=3)
    parser.add_argument("--planner-samples", type=int, default=8)
    parser.add_argument(
        "--planner-sweep",
        default=None,
        help="Comma-separated planning weights to evaluate without retraining, e.g. 0,0.05,0.15,0.3.",
    )
    parser.add_argument("--diagnose-world-model", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=20000)
    parser.add_argument("--diagnose-decisions", action="store_true")
    parser.add_argument("--decision-samples", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if args.diagnose_world_model:
        diagnose_world_model(
            config, args.checkpoint, device=device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnose_decisions:
        diagnose_decisions(
            config, args.checkpoint, device=device,
            max_samples=args.decision_samples,
            planning_weight=args.planning_weight,
            planner_horizon=args.planner_horizon,
            planner_samples=args.planner_samples,
        )

    print("=== Naive (random) agent ===")
    naive = run_naive(config, num_episodes=args.num_episodes)
    print(f"  Mean lifespan: {naive['mean_lifespan']:.1f} +/- {naive['std']:.1f}  (max {naive['max']:.0f})")
    print_causal_metrics("Naive", naive)

    print("\n=== Trained agent (Q only) ===")
    trained = run_trained(
        config, args.checkpoint, num_episodes=args.num_episodes,
        device=device, decision_mode="q",
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

    if args.compare_planner:
        print("\n=== Trained agent (Q + World Model planner) ===")
        planned = run_trained(
            config, args.checkpoint, num_episodes=args.num_episodes, device=device,
            decision_mode="planner", planning_weight=args.planning_weight,
            planner_horizon=args.planner_horizon, planner_samples=args.planner_samples,
        )
        print(
            f"  Mean lifespan: {planned['mean_lifespan']:.1f} +/- "
            f"{planned['std']:.1f}  (max {planned['max']:.0f})"
        )
        print_causal_metrics("Planner", planned)
        planner_ratio = planned["mean_lifespan"] / max(trained["mean_lifespan"], 1)
        print(f"\n  Ratio planner/Q-only: {planner_ratio:.2f}x")
        craft_gain = planned["craft_tool_mean"] - trained["craft_tool_mean"]
        tool_gain = planned["harvest_food_tool_mean"] - trained["harvest_food_tool_mean"]
        if planner_ratio > 1.05 or craft_gain > 0.05 or tool_gain > 0.05:
            print("  => Signal WM: planner ameliore la decision ou la chaine craft.")
        else:
            print("  => Pas de signal WM utile dans cette evaluation.")

    if args.planner_sweep:
        print("\n=== Planner weight sweep ===")
        weights = _parse_sweep(args.planner_sweep)
        print("  weight  lifespan  craft  tool_food  eat  max")
        for weight in weights:
            planned = run_trained(
                config, args.checkpoint, num_episodes=args.num_episodes,
                device=device, decision_mode="planner", planning_weight=weight,
                planner_horizon=args.planner_horizon,
                planner_samples=args.planner_samples,
            )
            print(
                f"  {weight:6.3f} "
                f"{planned['mean_lifespan']:8.1f} "
                f"{planned['craft_tool_mean']:6.2f} "
                f"{planned['harvest_food_tool_mean']:10.2f} "
                f"{planned['eat_ok_mean']:5.2f} "
                f"{planned['max']:4.0f}"
            )


if __name__ == "__main__":
    main()
