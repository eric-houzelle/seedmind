"""Evaluate and rollout a promoted Micro-Fouloide checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_micro_fouloide import (  # noqa: E402
    load_config,
    resolve_uncertainty_threshold_from_replay,
    run_trained,
)
from scripts.run_micro_fouloide import build_agent, build_env  # noqa: E402
from seedmind.training.device import resolve_device  # noqa: E402


def _preset_params(name: str | None) -> dict[str, Any]:
    if name is None or name == "wm-calibrated":
        return {
            "planning_weight": 0.25,
            "terminal_value_weight": 1.0,
            "planner_uncertainty_quantile": 0.60,
            "planner_margin_threshold": 0.01,
            "planner_q_advantage_threshold": 0.02,
            "planner_horizon": 5,
            "planner_samples": 8,
        }
    raise ValueError(f"Unknown planner preset: {name}")


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seed_entry(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    for row in manifest.get("seeds", []):
        if int(row.get("seed", -1)) == int(seed):
            return row
    raise ValueError(f"Seed {seed} is not present in manifest {manifest.get('name', '<unknown>')}.")


def _configure_planner(config: dict, params: dict[str, Any], threshold: float) -> dict:
    configured = dict(config)
    planning = dict(configured.get("planning", {}))
    planning.update({
        "enabled": True,
        "weight": float(params["planning_weight"]),
        "terminal_value_weight": float(params["terminal_value_weight"]),
        "uncertainty_threshold": float(threshold),
        "margin_threshold": float(params["planner_margin_threshold"]),
        "q_advantage_threshold": float(params["planner_q_advantage_threshold"]),
        "horizon": int(params["planner_horizon"]),
        "num_samples": int(params["planner_samples"]),
    })
    configured["planning"] = planning
    return configured


def _drive(info: dict[str, Any]) -> float:
    drives = info.get("drives", {})
    values = [
        float(drives.get("energy", 0.0)),
        float(drives.get("hydration", 0.0)),
        1.0 - abs(float(drives.get("temperature", 0.5)) - 0.5) * 2.0,
        float(drives.get("health", 0.0)),
    ]
    return float(np.mean([max(0.0, min(1.0, v)) for v in values]))


def _load_agent(config: dict, checkpoint: str, device: torch.device):
    agent = build_agent(config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.encoder.load_state_dict(ckpt["encoder_state"])
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    if agent.value_model is not None and "value_model_state" in ckpt:
        agent.value_model.load_state_dict(ckpt["value_model_state"])
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    if agent.value_model is not None:
        agent.value_model.to(device)
    agent.policy.epsilon_start = 0.0
    agent.policy.epsilon_end = 0.0
    return agent


def _run_rollout(
    config: dict,
    checkpoint: str,
    device: torch.device,
    rollout_seed: int,
    max_steps: int,
    trace_every: int,
) -> dict[str, Any]:
    agent = _load_agent(config, checkpoint, device)
    env = build_env(config, seed=rollout_seed)
    obs = env.reset()
    latent = agent.encode(obs)
    events: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    planner_used = 0
    trace: list[str] = []
    done = False
    info: dict[str, Any] = {}

    while not done and env.steps < max_steps:
        memories = agent.retrieve(latent)
        goal = agent.choose_goal(latent, memories)
        action = agent.choose_action(
            latent,
            goal,
            memories,
            env.available_actions(),
            observation=obs,
        )
        obs, _, done, info = env.step(action)
        latent = agent.encode(obs)
        event = str(info.get("event", "unknown"))
        actions[action] += 1
        events[event] += 1
        planner_used += int(getattr(agent, "last_planner_used", False))
        should_trace = (
            env.steps == 1
            or env.steps % max(trace_every, 1) == 0
            or event not in {"move_ok", "wait"}
            or done
        )
        if should_trace:
            trace.append(
                f"{env.steps:03d} action={action:<10} event={event:<14} "
                f"E={float(info.get('energy', 0.0)):.2f} "
                f"H2O={float(info.get('hydration', 0.0)):.2f} "
                f"T={float(info.get('temperature', 0.5)):.2f} "
                f"HP={float(info.get('health', 0.0)):.2f} "
                f"drive={_drive(info):.3f} "
                f"planner={int(getattr(agent, 'last_planner_used', False))}"
            )

    total_actions = max(sum(actions.values()), 1)
    return {
        "lifespan": int(info.get("lifespan", env.steps)),
        "dead": bool(info.get("dead", False)),
        "timeout": bool(info.get("timeout", False)),
        "drive": _drive(info) if info else 0.0,
        "planner_used": planner_used / total_actions,
        "events": dict(events),
        "actions": dict(actions),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="runs/micro_fouloide_promoted/wm_calibrated_v0/manifest.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=9999)
    parser.add_argument("--rollout-max-steps", type=int, default=80)
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    row = _seed_entry(manifest, args.seed)
    checkpoint = str(row["promoted_checkpoint"])
    config = load_config(str(manifest["config"]))
    device = resolve_device(args.device)
    params = _preset_params(str(manifest.get("planner_preset", "wm-calibrated")))
    threshold = resolve_uncertainty_threshold_from_replay(
        config,
        checkpoint,
        device,
        float(params["planner_uncertainty_quantile"]),
    )

    print(f"Manifest: {manifest.get('name', args.manifest)}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"Planner threshold q{params['planner_uncertainty_quantile']:.2f}: {threshold:.5f}")
    print(
        "Validated metrics: "
        f"Q-only={float(row['q_lifespan']):.1f} "
        f"Q+WM={float(row['planner_lifespan']):.1f} "
        f"delta={float(row['delta_lifespan']):+.1f}"
    )

    if not args.skip_eval:
        q_stats = run_trained(config, checkpoint, args.num_episodes, device, decision_mode="q")
        planner_stats = run_trained(
            config,
            checkpoint,
            args.num_episodes,
            device,
            decision_mode="planner",
            planning_weight=float(params["planning_weight"]),
            planner_horizon=int(params["planner_horizon"]),
            planner_samples=int(params["planner_samples"]),
            terminal_value_weight=float(params["terminal_value_weight"]),
            planner_uncertainty_threshold=threshold,
            planner_margin_threshold=float(params["planner_margin_threshold"]),
            planner_q_advantage_threshold=float(params["planner_q_advantage_threshold"]),
        )
        print("\nEvaluation")
        print(
            f"  Q-only lifespan={q_stats['mean_lifespan']:.1f} "
            f"max={q_stats['max_lifespan']:.0f}"
        )
        print(
            f"  Q+WM   lifespan={planner_stats['mean_lifespan']:.1f} "
            f"delta={planner_stats['mean_lifespan'] - q_stats['mean_lifespan']:+.1f} "
            f"used={planner_stats.get('planner_used', 0.0):.1%} "
            f"max={planner_stats['max_lifespan']:.0f}"
        )

    if not args.skip_rollout:
        rollout_config = _configure_planner(config, params, threshold)
        rollout = _run_rollout(
            rollout_config,
            checkpoint,
            device,
            rollout_seed=int(args.rollout_seed),
            max_steps=int(args.rollout_max_steps),
            trace_every=int(args.trace_every),
        )
        print("\nRollout")
        print(
            f"  lifespan={rollout['lifespan']} dead={rollout['dead']} "
            f"timeout={rollout['timeout']} drive={rollout['drive']:.3f} "
            f"planner_used={rollout['planner_used']:.1%}"
        )
        print(f"  events={rollout['events']}")
        print("  trace:")
        for line in rollout["trace"]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
