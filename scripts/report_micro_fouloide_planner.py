"""Aggregate Micro-Fouloide Q-only vs WM-planner evaluations across seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_micro_fouloide import (
    load_config,
    resolve_uncertainty_threshold_from_replay,
    run_naive,
    run_trained,
)
from seedmind.training.device import resolve_device


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _preset_params(name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    if name == "wm-calibrated":
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


def _format_table(rows: list[dict[str, Any]], summary: dict[str, float]) -> str:
    lines = [
        "seed | Q-only | Q+WM | delta | ratio | used | food | water | damage | max",
        "-----|--------|------|-------|-------|------|------|-------|--------|----",
    ]
    for row in rows:
        lines.append(
            f"{row['seed']:>4} | "
            f"{row['q_lifespan']:>6.1f} | "
            f"{row['planner_lifespan']:>4.1f} | "
            f"{row['delta_lifespan']:>+5.1f} | "
            f"{row['planner_ratio']:>5.2f} | "
            f"{row['planner_used']:>4.1%} | "
            f"{row['planner_food']:>4.2f} | "
            f"{row['planner_water']:>5.2f} | "
            f"{row['planner_damage']:>6.2f} | "
            f"{row['planner_max_lifespan']:>3.0f}"
        )
    lines.append(
        f"mean | "
        f"{summary['q_lifespan']:>6.1f} | "
        f"{summary['planner_lifespan']:>4.1f} | "
        f"{summary['delta_lifespan']:>+5.1f} | "
        f"{summary['planner_ratio']:>5.2f} | "
        f"{summary['planner_used']:>4.1%} | "
        f"{summary['planner_food']:>4.2f} | "
        f"{summary['planner_water']:>5.2f} | "
        f"{summary['planner_damage']:>6.2f} | "
        f"{summary['planner_max_lifespan']:>3.0f}"
    )
    return "\n".join(lines)


def _write_output(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Micro-Fouloide Planner Report\n\n")
        f.write("```text\n")
        f.write(payload["table"])
        f.write("\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/micro_fouloide_v0_rough_valueplanner.yaml")
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "runs/micro_fouloide_v0_rough_valueplanner_seed{seed}/"
            "checkpoint_uncertainty_value_calibrated.pt"
        ),
        help="Checkpoint path template. Use {seed} where the seed number should be inserted.",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--planner-preset", choices=["wm-calibrated"], default="wm-calibrated")
    parser.add_argument("--planning-weight", type=float, default=None)
    parser.add_argument("--terminal-value-weight", type=float, default=None)
    parser.add_argument("--planner-uncertainty-quantile", type=float, default=None)
    parser.add_argument("--planner-margin-threshold", type=float, default=None)
    parser.add_argument("--planner-q-advantage-threshold", type=float, default=None)
    parser.add_argument("--planner-horizon", type=int, default=None)
    parser.add_argument("--planner-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional .json or markdown/text output path.")
    args = parser.parse_args()

    params = _preset_params(args.planner_preset)
    for key in (
        "planning_weight",
        "terminal_value_weight",
        "planner_uncertainty_quantile",
        "planner_margin_threshold",
        "planner_q_advantage_threshold",
        "planner_horizon",
        "planner_samples",
    ):
        override = getattr(args, key)
        if override is not None:
            params[key] = override

    seeds = _parse_seeds(args.seeds)
    config = load_config(args.config)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Seeds: {','.join(str(seed) for seed in seeds)}")
    print(f"Planner preset: {args.planner_preset or 'manual'}")
    naive = run_naive(config, args.num_episodes)
    print(f"Naive mean lifespan: {naive['mean_lifespan']:.1f}")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = args.checkpoint_template.format(seed=seed)
        threshold = resolve_uncertainty_threshold_from_replay(
            config,
            checkpoint,
            device,
            float(params["planner_uncertainty_quantile"]),
        )
        print(
            f"Evaluating seed {seed}: checkpoint={checkpoint} "
            f"unc_thr={threshold:.5f}"
        )
        q_stats = run_trained(
            config,
            checkpoint,
            args.num_episodes,
            device,
            decision_mode="q",
        )
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
        q_lifespan = float(q_stats["mean_lifespan"])
        planner_lifespan = float(planner_stats["mean_lifespan"])
        rows.append({
            "seed": seed,
            "checkpoint": checkpoint,
            "uncertainty_threshold": threshold,
            "q_lifespan": q_lifespan,
            "planner_lifespan": planner_lifespan,
            "delta_lifespan": planner_lifespan - q_lifespan,
            "planner_ratio": planner_lifespan / max(q_lifespan, 1.0),
            "planner_used": float(planner_stats.get("planner_used", 0.0)),
            "planner_food": float(planner_stats.get("interact_food", 0.0)),
            "planner_water": float(planner_stats.get("interact_water", 0.0)),
            "planner_damage": float(planner_stats.get("damage", 0.0)),
            "planner_max_lifespan": float(planner_stats.get("max_lifespan", 0.0)),
        })

    summary = {
        "q_lifespan": _mean(rows, "q_lifespan"),
        "planner_lifespan": _mean(rows, "planner_lifespan"),
        "delta_lifespan": _mean(rows, "delta_lifespan"),
        "planner_ratio": _mean(rows, "planner_ratio"),
        "planner_used": _mean(rows, "planner_used"),
        "planner_food": _mean(rows, "planner_food"),
        "planner_water": _mean(rows, "planner_water"),
        "planner_damage": _mean(rows, "planner_damage"),
        "planner_max_lifespan": _mean(rows, "planner_max_lifespan"),
    }
    table = _format_table(rows, summary)
    payload = {
        "config": args.config,
        "num_episodes": args.num_episodes,
        "device": str(device),
        "planner_preset": args.planner_preset,
        "planner_params": params,
        "naive": naive,
        "rows": rows,
        "summary": summary,
        "table": table,
    }

    print("\n=== Planner Report ===")
    print(table)
    _write_output(args.output, payload)
    if args.output is not None:
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
