"""Baseline aléatoire pour la calibration DreamerV3 (doc §5).

Mesure ce qu'une policy uniforme obtient sur le wrapper de calibration —
le seuil que la référence (et notre port) doivent battre avec marge franche.

Usage :
    python scripts/calibration/random_baseline.py \
        --config configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml \
        --episodes 20 --time-limit 2000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.calibration.gym_micro_fouloide import MicroFouloideGymEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/micro_fouloide_online_homeostatic_rssm_v3_dreamerfix.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--time-limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    returns, ext_returns, lengths, survived = [], [], [], 0
    events: Counter[str] = Counter()
    rng = np.random.default_rng(args.seed)

    env = MicroFouloideGymEnv(
        args.config, seed=args.seed, reward_mode="learning", time_limit=args.time_limit,
    )
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        ep_ret = ep_ext = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(rng.integers(env.action_space.n))
            _, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_ext += info["reward_external"]
            steps += 1
            event = str(info.get("event", "") or "")
            if event:
                events[event] += 1
        returns.append(ep_ret)
        ext_returns.append(ep_ext)
        lengths.append(steps)
        if truncated and not terminated:
            survived += 1
        print(f"  ep {ep:02d} | return={ep_ret:8.2f} | steps={steps:5d} | "
              f"{'survécu' if truncated and not terminated else 'mort'}")

    print("\n=== BASELINE ALÉATOIRE ===")
    print(f"config: {args.config}")
    print(f"épisodes: {args.episodes}, time_limit: {args.time_limit}")
    print(f"return (reward_learning): {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"return (reward_external): {np.mean(ext_returns):.2f} ± {np.std(ext_returns):.2f}")
    print(f"longueur moyenne: {np.mean(lengths):.0f} pas")
    print(f"survie (time_limit atteint): {survived}/{args.episodes}")
    if events:
        print("événements:", dict(events.most_common()))
    ambient = {"move_ok", "move_blocked", "wait", "rest", "reset", "death",
               "health_loss", "temperature_up", "temperature_down", "damage",
               "interact_noop", "pick_noop", "drop_noop", "plant_noop",
               "combine_noop"}
    forage = {k: v for k, v in events.items() if k not in ambient}
    total_steps = int(np.sum(lengths))
    n_forage = sum(forage.values())
    print(f"FOURRAGE (événements de consommation réussis): {n_forage} "
          f"sur {total_steps} pas ({1000 * n_forage / max(total_steps, 1):.2f}/1000 pas)")
    if forage:
        print("  détail:", dict(sorted(forage.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
