"""Select Micro-Fouloide checkpoints by controlled evaluation."""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
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
from seedmind.training.device import resolve_device  # noqa: E402


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


def _collect_checkpoints(explicit: list[str], patterns: list[str]) -> list[str]:
    paths = list(explicit)
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    deduped = sorted(dict.fromkeys(paths))
    if not deduped:
        raise ValueError("No checkpoints matched.")
    return deduped


def _checkpoint_metrics(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = dict(payload.get("metrics", {}))
    return {
        "episode": metrics.get("episode"),
        "mean_lifespan": metrics.get("mean_lifespan"),
        "best_mean_lifespan": metrics.get("best_mean_lifespan"),
        "has_final_calibration": "final_calibration" in metrics,
    }


def _write_output(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Micro-Fouloide Checkpoint Selection\n\n")
        f.write("```text\n")
        f.write(payload["table"])
        f.write("\n```\n")


def _promote_checkpoint(
    best: dict[str, Any],
    promote_to: str,
    manifest_path: str | None,
    payload: dict[str, Any],
) -> None:
    source = Path(str(best["checkpoint"]))
    target = Path(promote_to)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    manifest = {
        "promoted_checkpoint": str(target),
        "source_checkpoint": str(source),
        "selection": best,
        "selection_config": {
            "config": payload["config"],
            "num_episodes": payload["num_episodes"],
            "device": payload["device"],
            "planner_preset": payload["planner_preset"],
            "planner_params": payload["planner_params"],
        },
    }
    manifest_out = (
        Path(manifest_path)
        if manifest_path is not None
        else target.with_suffix(target.suffix + ".manifest.json")
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "rank | checkpoint | Q-only | Q+WM | delta | ratio | used | train_mean | max",
        "-----|------------|--------|------|-------|-------|------|------------|----",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank:>4} | "
            f"{Path(row['checkpoint']).name:<10} | "
            f"{row['q_lifespan']:>6.1f} | "
            f"{row['planner_lifespan']:>4.1f} | "
            f"{row['delta_lifespan']:>+5.1f} | "
            f"{row['planner_ratio']:>5.2f} | "
            f"{row['planner_used']:>4.1%} | "
            f"{row['checkpoint_mean_lifespan'] if row['checkpoint_mean_lifespan'] is not None else float('nan'):>10.1f} | "
            f"{row['planner_max_lifespan']:>3.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--planner-preset", choices=["wm-calibrated"], default="wm-calibrated")
    parser.add_argument("--planning-weight", type=float, default=None)
    parser.add_argument("--terminal-value-weight", type=float, default=None)
    parser.add_argument("--planner-uncertainty-quantile", type=float, default=None)
    parser.add_argument("--planner-margin-threshold", type=float, default=None)
    parser.add_argument("--planner-q-advantage-threshold", type=float, default=None)
    parser.add_argument("--planner-horizon", type=int, default=None)
    parser.add_argument("--planner-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--promote-to",
        default=None,
        help="Copy the best selected checkpoint to this stable output path.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path for promotion metadata. Defaults next to --promote-to.",
    )
    parser.add_argument(
        "--min-planner-lifespan",
        type=float,
        default=None,
        help="Refuse promotion unless the best planner lifespan reaches this threshold.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Refuse promotion unless best Q+WM minus Q-only reaches this threshold.",
    )
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

    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoints = _collect_checkpoints(args.checkpoint, args.checkpoint_glob)

    print(f"Device: {device}")
    print(f"Planner preset: {args.planner_preset or 'manual'}")
    print(f"Candidates: {len(checkpoints)}")

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        meta = _checkpoint_metrics(checkpoint)
        threshold = resolve_uncertainty_threshold_from_replay(
            config,
            checkpoint,
            device,
            float(params["planner_uncertainty_quantile"]),
        )
        print(f"Evaluating {checkpoint} unc_thr={threshold:.5f}")
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
            "checkpoint_episode": meta["episode"],
            "checkpoint_mean_lifespan": meta["best_mean_lifespan"] or meta["mean_lifespan"],
            "has_final_calibration": bool(meta["has_final_calibration"]),
        })

    rows.sort(
        key=lambda row: (
            float(row["planner_lifespan"]),
            float(row["delta_lifespan"]),
            float(row["q_lifespan"]),
        ),
        reverse=True,
    )
    table = _format_table(rows)
    payload = {
        "config": args.config,
        "num_episodes": args.num_episodes,
        "device": str(device),
        "planner_preset": args.planner_preset,
        "planner_params": params,
        "rows": rows,
        "best": rows[0],
        "table": table,
    }

    print("\n=== Checkpoint Selection ===")
    print(table)
    print(f"\nBest checkpoint: {rows[0]['checkpoint']}")
    _write_output(args.output, payload)
    if args.output is not None:
        print(f"Saved selection report to {args.output}")
    if args.promote_to is not None:
        best = rows[0]
        if (
            args.min_planner_lifespan is not None
            and float(best["planner_lifespan"]) < float(args.min_planner_lifespan)
        ):
            raise SystemExit(
                "Refusing promotion: best planner lifespan "
                f"{best['planner_lifespan']:.2f} < {args.min_planner_lifespan:.2f}."
            )
        if args.min_delta is not None and float(best["delta_lifespan"]) < float(args.min_delta):
            raise SystemExit(
                f"Refusing promotion: best delta {best['delta_lifespan']:.2f} < {args.min_delta:.2f}."
            )
        _promote_checkpoint(best, args.promote_to, args.manifest, payload)
        print(f"Promoted checkpoint to {args.promote_to}")


if __name__ == "__main__":
    main()
