"""Posthoc value-model calibration for a Micro-Fouloide checkpoint."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_micro_fouloide import build_agent, load_config
from seedmind.training.device import resolve_device
from seedmind.training.value import (
    evaluate_value_model_on_returns,
    make_value_optimizer,
    train_value_model_on_returns,
)


def _discounted_returns_by_episode(rows: Iterable[dict], reward_key: str, gamma: float) -> dict[int, float]:
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    row_lookup: dict[int, dict] = {}
    for idx, row in enumerate(rows):
        episode_id = str(row.get("episode_id", "unknown"))
        step = int(row.get("step", len(grouped[episode_id])))
        reward = float(row.get(reward_key, row.get("reward_external", 0.0)))
        grouped[episode_id].append((step, idx, reward))
        row_lookup[idx] = row

    returns: dict[int, float] = {}
    for episode_rows in grouped.values():
        running = 0.0
        for _, idx, reward in sorted(episode_rows, reverse=True):
            row = row_lookup[idx]
            if row.get("done", False):
                running = reward
            else:
                running = reward + gamma * running
            returns[idx] = running
    return returns


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"mae={metrics['mae']:.4f} bias={metrics['bias']:.4f} "
        f"corr={metrics['corr']:.3f} value_mean={metrics['value_mean']:.4f} "
        f"return_mean={metrics['return_mean']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = [
        row for row in payload.get("buffer", {}).get("data", [])
        if row.get("latent_state") is not None
    ]
    if not rows:
        raise ValueError("Checkpoint replay buffer has no latent rows.")

    vc = config.get("value_model", {})
    reward_key = str(vc.get("reward_key", config.get("dqn", {}).get("reward_key", "reward_external")))
    gamma = float(vc.get("gamma", config.get("dqn", {}).get("gamma", 0.97)))
    returns_by_idx = _discounted_returns_by_episode(rows, reward_key, gamma)
    paired = [
        (row, returns_by_idx[idx])
        for idx, row in enumerate(rows)
        if idx in returns_by_idx
    ]
    if not paired:
        raise ValueError("Could not compute replay returns for value calibration.")

    if args.max_samples > 0 and len(paired) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(np.arange(len(paired)), size=args.max_samples, replace=False)
        paired = [paired[int(i)] for i in indices]

    latents = np.stack([
        np.asarray(row["latent_state"], dtype=np.float32)
        for row, _ in paired
    ])
    returns = np.asarray([ret for _, ret in paired], dtype=np.float32)

    agent = build_agent(config, seed=args.seed)
    if agent.value_model is None:
        raise ValueError("Config does not enable a value model.")
    if "value_model_state" in payload:
        agent.value_model.load_state_dict(payload["value_model_state"])
    agent.value_model.to(device)

    before = evaluate_value_model_on_returns(agent.value_model, latents, returns)
    optimizer = make_value_optimizer(agent.value_model, learning_rate=float(args.learning_rate))
    result = train_value_model_on_returns(
        agent.value_model,
        latents,
        returns,
        optimizer,
        batch_size=int(args.batch_size),
        num_updates=int(args.updates),
        seed=int(args.seed),
    )
    after = evaluate_value_model_on_returns(agent.value_model, latents, returns)

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(args.checkpoint).with_name("checkpoint_value_calibrated.pt")
    )
    payload["value_model_state"] = {
        k: v.detach().cpu()
        for k, v in agent.value_model.state_dict().items()
    }
    payload["target_value_model_state"] = payload["value_model_state"]
    metrics = dict(payload.get("metrics", {}))
    metrics["posthoc_value_calibration"] = {
        "updates": result["updates"],
        "value_return_loss": result["value_return_loss"],
        "source_checkpoint": str(args.checkpoint),
        "reward_key": reward_key,
        "gamma": gamma,
        "before": before,
        "after": after,
    }
    payload["metrics"] = metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(
        f"Posthoc value calibration: samples={len(paired)} updates={result['updates']:.0f} "
        f"loss={result['value_return_loss']:.6f} reward_key={reward_key} gamma={gamma:.3f}"
    )
    print(f"Before: {_format_metrics(before)}")
    print(f"After:  {_format_metrics(after)}")
    print(f"Saved calibrated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
