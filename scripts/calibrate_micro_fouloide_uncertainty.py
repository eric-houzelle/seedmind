"""Posthoc uncertainty calibration for a Micro-Fouloide checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_micro_fouloide import build_agent, load_config
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.device import resolve_device
from seedmind.training.train import train_world_model_uncertainty_head


def _load_buffer(payload: dict, seed: int) -> ExperienceBuffer:
    saved = payload.get("buffer")
    if saved is None:
        raise ValueError("Checkpoint does not contain a replay buffer.")
    buffer = ExperienceBuffer(capacity=int(saved["capacity"]), seed=seed)
    buffer._data = saved["data"]
    buffer._cursor = int(saved.get("cursor", 0))
    buffer._rebuild_index()
    return buffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sampler", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    agent = build_agent(config, seed=args.seed)
    agent.world_model.load_state_dict(payload["world_model_state"])
    agent.world_model.to(device)
    buffer = _load_buffer(payload, seed=args.seed)

    wmc = config.get("world_model", {})
    cwm = config.get("causal_world_model", {})
    optimizer = torch.optim.Adam(
        agent.world_model.uncertainty_head.parameters(),
        lr=float(args.learning_rate),
    )
    result = train_world_model_uncertainty_head(
        agent.world_model,
        buffer,
        optimizer,
        batch_size=int(args.batch_size),
        num_updates=int(args.updates),
        sampler=str(args.sampler or wmc.get("sampler", "causal")),
        causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
        causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
        event_class_balance=bool(cwm.get("event_class_balance", False)),
        event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
        reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
        reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
    )

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(args.checkpoint).with_name("checkpoint_uncertainty_calibrated.pt")
    )
    payload["world_model_state"] = {
        k: v.detach().cpu()
        for k, v in agent.world_model.state_dict().items()
    }
    metrics = dict(payload.get("metrics", {}))
    metrics["posthoc_uncertainty_calibration"] = {
        "updates": result["updates"],
        "uncertainty_loss": result["uncertainty"],
        "source_checkpoint": str(args.checkpoint),
    }
    payload["metrics"] = metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Posthoc uncertainty calibration: updates={result['updates']:.0f} loss={result['uncertainty']:.6f}")
    print(f"Saved calibrated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
