"""Posthoc rare-event World Model calibration for a Micro-Fouloide checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_micro_fouloide import build_agent, load_config
from seedmind.memory.experience_buffer import ExperienceBuffer
from seedmind.training.device import resolve_device
from seedmind.training.train import make_optimizer, train_world_model


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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--feature-loss-weight", type=float, default=None)
    parser.add_argument("--event-loss-weight", type=float, default=None)
    parser.add_argument("--event-sample-name", action="append", default=[])
    parser.add_argument("--event-sample-name-weight", type=float, default=0.0)
    parser.add_argument("--event-sample-done-weight", type=float, default=0.0)
    parser.add_argument("--event-sample-reward-abs-weight", type=float, default=0.0)
    parser.add_argument("--reward-abs-weight", type=float, default=None)
    parser.add_argument("--reward-done-weight", type=float, default=None)
    parser.add_argument("--uncertainty-weight", type=float, default=None)
    parser.add_argument("--uncertainty-detach", action="store_true")
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
    event_sample_names = set(cwm.get("event_sample_names", []))
    event_sample_names.update(str(name) for name in args.event_sample_name)

    optimizer = make_optimizer(agent.world_model, learning_rate=float(args.learning_rate))
    result = train_world_model(
        agent.world_model,
        buffer,
        optimizer,
        batch_size=int(args.batch_size),
        num_updates=int(args.updates),
        sampler=str(args.sampler or wmc.get("sampler", "causal")),
        causal_feature_weight=float(
            cwm.get("feature_loss_weight", 0.0)
            if args.feature_loss_weight is None else args.feature_loss_weight
        ),
        causal_event_weight=float(
            cwm.get("event_loss_weight", 0.0)
            if args.event_loss_weight is None else args.event_loss_weight
        ),
        event_class_balance=bool(cwm.get("event_class_balance", False)),
        event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
        reward_abs_weight=float(
            wmc.get("reward_abs_weight", 0.0)
            if args.reward_abs_weight is None else args.reward_abs_weight
        ),
        reward_done_weight=float(
            wmc.get("reward_done_weight", 0.0)
            if args.reward_done_weight is None else args.reward_done_weight
        ),
        event_sample_names=event_sample_names,
        event_sample_name_weight=float(args.event_sample_name_weight),
        event_sample_done_weight=float(args.event_sample_done_weight),
        event_sample_reward_abs_weight=float(args.event_sample_reward_abs_weight),
        uncertainty_weight=float(
            wmc.get("uncertainty_loss_weight", 0.0)
            if args.uncertainty_weight is None else args.uncertainty_weight
        ),
        uncertainty_detach=bool(args.uncertainty_detach or wmc.get("uncertainty_detach", False)),
    )

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(args.checkpoint).with_name("checkpoint_wm_events_calibrated.pt")
    )
    payload["world_model_state"] = {
        k: v.detach().cpu()
        for k, v in agent.world_model.state_dict().items()
    }
    metrics = dict(payload.get("metrics", {}))
    metrics["posthoc_wm_event_calibration"] = {
        "updates": result["updates"],
        "total_loss": result["total"],
        "state_loss": result["state"],
        "reward_loss": result["reward"],
        "feature_loss": result["feature"],
        "event_loss": result["event"],
        "uncertainty_loss": result["uncertainty"],
        "event_sample_names": sorted(event_sample_names),
        "event_sample_name_weight": float(args.event_sample_name_weight),
        "event_sample_done_weight": float(args.event_sample_done_weight),
        "event_sample_reward_abs_weight": float(args.event_sample_reward_abs_weight),
        "source_checkpoint": str(args.checkpoint),
    }
    payload["metrics"] = metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(
        "Posthoc WM event calibration: "
        f"updates={result['updates']:.0f} "
        f"total={result['total']:.6f} "
        f"reward={result['reward']:.6f} "
        f"feature={result['feature']:.6f} "
        f"event={result['event']:.6f}"
    )
    print(f"Event sample names: {', '.join(sorted(event_sample_names)) or '(none)'}")
    print(f"Saved calibrated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
