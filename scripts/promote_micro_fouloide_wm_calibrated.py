"""Promote the validated Micro-Fouloide WM-calibrated checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


VALIDATED_ROWS: dict[int, dict[str, Any]] = {
    1: {
        "q_lifespan": 116.6,
        "planner_lifespan": 119.6,
        "delta_lifespan": 3.0,
        "planner_ratio": 1.03,
        "planner_used": 0.595,
        "food": 0.91,
        "water": 0.95,
        "damage": 1.47,
        "max_lifespan": 286,
    },
    2: {
        "q_lifespan": 101.5,
        "planner_lifespan": 102.6,
        "delta_lifespan": 1.0,
        "planner_ratio": 1.01,
        "planner_used": 0.592,
        "food": 0.28,
        "water": 0.34,
        "damage": 0.95,
        "max_lifespan": 205,
    },
    3: {
        "q_lifespan": 105.5,
        "planner_lifespan": 108.0,
        "delta_lifespan": 2.5,
        "planner_ratio": 1.02,
        "planner_used": 0.617,
        "food": 0.59,
        "water": 0.70,
        "damage": 1.56,
        "max_lifespan": 234,
    },
}


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    missing = [seed for seed in seeds if seed not in VALIDATED_ROWS]
    if missing:
        raise ValueError(f"No validated metrics embedded for seeds: {missing}")
    return seeds


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--source-template",
        default=(
            "runs/micro_fouloide_v0_rough_valueplanner_seed{seed}/"
            "checkpoint_uncertainty_value_calibrated.pt"
        ),
    )
    parser.add_argument("--out-dir", default="runs/micro_fouloide_promoted/wm_calibrated_v0")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    promoted = []
    for seed in seeds:
        source = Path(args.source_template.format(seed=seed))
        if not source.exists():
            raise FileNotFoundError(f"Missing source checkpoint for seed {seed}: {source}")
        target = out_dir / f"seed{seed}.pt"
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} already exists. Use --force to overwrite.")
        shutil.copy2(source, target)
        row = dict(VALIDATED_ROWS[seed])
        row.update({
            "seed": seed,
            "source_checkpoint": str(source),
            "promoted_checkpoint": str(target),
        })
        promoted.append(row)

    summary = {
        "q_lifespan": _mean(promoted, "q_lifespan"),
        "planner_lifespan": _mean(promoted, "planner_lifespan"),
        "delta_lifespan": _mean(promoted, "delta_lifespan"),
        "planner_ratio": _mean(promoted, "planner_ratio"),
        "planner_used": _mean(promoted, "planner_used"),
        "food": _mean(promoted, "food"),
        "water": _mean(promoted, "water"),
        "damage": _mean(promoted, "damage"),
        "max_lifespan": _mean(promoted, "max_lifespan"),
    }
    manifest = {
        "name": "micro_fouloide_wm_calibrated_v0",
        "description": "Validated Micro-Fouloide rough checkpoints with calibrated WM planner.",
        "config": "configs/micro_fouloide_v0_rough_valueplanner.yaml",
        "planner_preset": "wm-calibrated",
        "validation_report": "reports/micro_fouloide_wm_calibrated_1000.md",
        "validation_num_episodes": 1000,
        "seeds": promoted,
        "summary": summary,
    }
    manifest_path = out_dir / args.manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Promoted {len(promoted)} checkpoints to {out_dir}")
    print(f"Manifest saved to {manifest_path}")
    print(
        "mean: "
        f"Q-only={summary['q_lifespan']:.1f} "
        f"Q+WM={summary['planner_lifespan']:.1f} "
        f"delta={summary['delta_lifespan']:+.1f}"
    )


if __name__ == "__main__":
    main()
