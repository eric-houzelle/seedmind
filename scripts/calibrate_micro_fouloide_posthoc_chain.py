"""Run the standard posthoc Micro-Fouloide calibration chain across seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/micro_fouloide_v0_rough_valueplanner.yaml")
    parser.add_argument(
        "--run-template",
        default="runs/micro_fouloide_v0_rough_valueplanner_seed{seed}",
        help="Run directory template. Use {seed} where the seed number should be inserted.",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--uncertainty-updates", type=int, default=2000)
    parser.add_argument("--uncertainty-batch-size", type=int, default=64)
    parser.add_argument("--uncertainty-learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-updates", type=int, default=5000)
    parser.add_argument("--value-batch-size", type=int, default=64)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-max-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing calibrated outputs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    python = sys.executable
    for seed in seeds:
        run_dir = Path(args.run_template.format(seed=seed))
        final_checkpoint = run_dir / "checkpoint_final.pt"
        uncertainty_checkpoint = run_dir / "checkpoint_uncertainty_calibrated.pt"
        value_checkpoint = run_dir / "checkpoint_uncertainty_value_calibrated.pt"

        if not final_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {final_checkpoint}")

        print(f"\n=== Seed {seed} ===", flush=True)
        if args.force or not uncertainty_checkpoint.exists():
            _run(
                [
                    python,
                    "scripts/calibrate_micro_fouloide_uncertainty.py",
                    "--checkpoint",
                    str(final_checkpoint),
                    "--config",
                    args.config,
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--updates",
                    str(args.uncertainty_updates),
                    "--batch-size",
                    str(args.uncertainty_batch_size),
                    "--learning-rate",
                    str(args.uncertainty_learning_rate),
                    "--out",
                    str(uncertainty_checkpoint),
                ],
                dry_run=bool(args.dry_run),
            )
        else:
            print(f"Skipping uncertainty calibration, exists: {uncertainty_checkpoint}")

        if args.force or not value_checkpoint.exists():
            _run(
                [
                    python,
                    "scripts/calibrate_micro_fouloide_value.py",
                    "--checkpoint",
                    str(uncertainty_checkpoint),
                    "--config",
                    args.config,
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--updates",
                    str(args.value_updates),
                    "--batch-size",
                    str(args.value_batch_size),
                    "--learning-rate",
                    str(args.value_learning_rate),
                    "--max-samples",
                    str(args.value_max_samples),
                    "--out",
                    str(value_checkpoint),
                ],
                dry_run=bool(args.dry_run),
            )
        else:
            print(f"Skipping value calibration, exists: {value_checkpoint}")

    print("\nPosthoc calibration chain complete.")


if __name__ == "__main__":
    main()
