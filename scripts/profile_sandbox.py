"""Profile SeedMind sandbox training by coarse runtime sections.

This wrapper runs ``scripts/run_sandbox.py`` with the same behavior, but times
major sections with CUDA/MPS synchronisation so accelerator timings are usable.

Example:
    python scripts/profile_sandbox.py --config configs/sandbox_v2_craft.yaml --episodes 200 --device mps
    python scripts/profile_sandbox.py --config configs/sandbox_v2_craft.yaml --episodes 200 --device mps --inference-device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.agent import Agent
from seedmind.agent.encoder import Encoder
from seedmind.agent.world_model import WorldModel
from seedmind.envs.sandbox_world import SandboxWorld
from seedmind.training.device import resolve_device


class SectionTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def wrap(self, name: str, fn: Callable) -> Callable:
        def _wrapped(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.sync()
                self.times[name] += time.perf_counter() - start
                self.counts[name] += 1

        return _wrapped

    def report(self, wall_time: float) -> None:
        print("\n=== Section profile ===")
        print(f"wall_time: {wall_time:.3f}s")
        rows = sorted(self.times.items(), key=lambda item: item[1], reverse=True)
        measured = sum(value for _, value in rows)
        for name, value in rows:
            count = self.counts[name]
            avg = value / max(count, 1)
            pct = 100.0 * value / max(wall_time, 1e-9)
            print(f"{name:24s} {value:8.3f}s  {pct:6.1f}%  n={count:6d}  avg={avg:.6f}s")
        other = wall_time - measured
        print(f"{'unmeasured/overhead':24s} {other:8.3f}s  {100.0 * other / max(wall_time, 1e-9):6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SeedMind sandbox sections")
    parser.add_argument("--config", default="configs/sandbox_v2_craft.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--inference-device", default=None, choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    timer = SectionTimer(device)

    from scripts import run_sandbox

    # Patch globals used by run_sandbox.main().
    run_sandbox.train_dqn = timer.wrap("train_dqn", run_sandbox.train_dqn)
    run_sandbox.train_world_model = timer.wrap("train_world_model", run_sandbox.train_world_model)
    run_sandbox.save_checkpoint = timer.wrap("save_checkpoint", run_sandbox.save_checkpoint)

    # Patch class methods used inside the loop. These wrappers preserve return values.
    Agent.choose_action = timer.wrap("choose_action", Agent.choose_action)
    Encoder.encode = timer.wrap("encoder_encode", Encoder.encode)
    WorldModel.predict = timer.wrap("world_model_predict", WorldModel.predict)
    SandboxWorld.step = timer.wrap("env_step", SandboxWorld.step)
    SandboxWorld.observe = timer.wrap("env_observe", SandboxWorld.observe)

    out_dir = args.out_dir or f"/tmp/seedmind_profile_{device.type}"
    sys.argv = [
        "run_sandbox.py",
        "--config", args.config,
        "--episodes", str(args.episodes),
        "--device", str(device),
        "--seed", str(args.seed),
        "--out-dir", out_dir,
    ]
    if args.inference_device is not None:
        inference_device = resolve_device(args.inference_device)
        sys.argv.extend(["--inference-device", str(inference_device)])

    timer.sync()
    start = time.perf_counter()
    run_sandbox.main()
    timer.sync()
    timer.report(time.perf_counter() - start)


if __name__ == "__main__":
    main()
