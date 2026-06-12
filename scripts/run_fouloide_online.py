"""SeedMind — Fouloïde homéostatique en apprentissage online (headless).

Un agent vierge (aucun checkpoint) vit dans un monde persistant doux et
apprend en continu via OnlineLearner : WM + DQN + Value mis à jour tous les
K steps, seuil du gate planner rafraîchi en ligne. Validation du pivot avant
le branchement viewer.

    .venv/bin/python scripts/run_fouloide_online.py \
        --config configs/micro_fouloide_online_homeostatic.yaml \
        --steps 20000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_micro_fouloide import (  # noqa: E402
    _comfort_config,
    _compact_obs,
    _learning_reward,
    build_agent,
    build_env,
    causal_event_names,
    load_config,
)
from seedmind.agent.curiosity import compute_prediction_error_tensor  # noqa: E402
from seedmind.memory.experience_buffer import make_experience  # noqa: E402
from seedmind.training.device import resolve_device  # noqa: E402
from seedmind.training.latent_utils import latent_to_numpy  # noqa: E402
from seedmind.training.online import OnlineLearner  # noqa: E402
from seedmind.training.wellbeing import wellbeing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micro_fouloide_online_homeostatic.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = load_config(args.config)
    comfort = _comfort_config(config)
    cwm = config.get("causal_world_model", {})
    causal_wm_enabled = bool(cwm.get("enabled", False))
    event_to_index = {event: i for i, event in enumerate(causal_event_names(config))}

    out_dir = Path(args.out_dir or f"runs/fouloide_online_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(config, seed=args.seed)
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    if agent.value_model is not None:
        agent.value_model.to(device)
    learner = OnlineLearner(agent, config, device, seed=args.seed)

    env = build_env(config, seed=args.seed)
    observation = env.reset()
    latent_state = agent.encoder.encode_tensor(observation)

    deaths = 0
    lives = 1
    window = max(1, int(args.log_every))
    wellbeing_window: deque = deque(maxlen=window)
    planner_window: deque = deque(maxlen=window)
    event_counts: Counter[str] = Counter()
    history: list[Dict[str, Any]] = []

    print(
        f"Online fouloïde: {args.steps} steps, config={args.config}, "
        f"seed={args.seed}, device={device}"
    )

    for step in range(1, args.steps + 1):
        latent_np = latent_to_numpy(latent_state)
        memories = agent.retrieve(latent_np)
        goal = agent.choose_goal(latent_np, memories)
        action = agent.choose_action(
            latent_np, goal, memories, env.available_actions(),
            observation=observation,
        )
        planner_window.append(int(getattr(agent, "last_planner_used", False)))
        action_index = agent.action_index[action]
        next_obs, reward_ext, done, info = env.step(action)
        next_latent = agent.encoder.encode_tensor(next_obs)
        event = str(info.get("event", "unknown"))
        event_counts[event] += 1

        predicted, _, _ = agent.world_model.predict_tensor(latent_state, action_index)
        pred_err = float(compute_prediction_error_tensor(predicted, next_latent).item())
        reward_int = agent.curiosity.compute(pred_err)
        reward_learning = _learning_reward(reward_ext, observation, info, config)

        experience = make_experience(
            episode_id=f"online_life_{lives:04d}",
            world_id=env.world_id,
            step=step,
            observation=None,
            action=action,
            next_observation=None,
            reward_external=reward_ext,
            reward_intrinsic=reward_int,
            goal=goal,
            prediction_error=pred_err,
            done=done,
            latent_state=latent_np,
            next_latent_state=latent_to_numpy(next_latent),
            action_index=action_index,
            obs_state=_compact_obs(observation),
            next_obs_state=_compact_obs(next_obs),
            event=event,
            event_amount=int(info.get("event_amount", 0)),
            event_index=event_to_index.get(event) if causal_wm_enabled else None,
            causal_features=env.causal_features(observation) if causal_wm_enabled else None,
            next_causal_features=env.causal_features(next_obs) if causal_wm_enabled else None,
        )
        experience["reward_learning"] = reward_learning
        learner.observe(experience)
        agent.memory.store_if_important(experience)

        wellbeing_window.append(wellbeing(info.get("drives", {}), comfort))
        observation = next_obs
        latent_state = next_latent

        if done:
            deaths += 1
            lives += 1
            observation = env.reset()
            latent_state = agent.encoder.encode_tensor(observation)

        if step % window == 0:
            stats = learner.stats()
            row = {
                "step": step,
                "wellbeing": float(np.mean(wellbeing_window)),
                "planner_used_rate": float(np.mean(planner_window)),
                "deaths": deaths,
                "interact_water": int(event_counts.get("interact_water", 0)),
                "interact_food": int(event_counts.get("interact_food", 0)),
                "interact_noop": int(event_counts.get("interact_noop", 0)),
                "health_loss": int(event_counts.get("health_loss", 0)),
                **{k: stats[k] for k in (
                    "wm_loss", "td_loss", "value_loss",
                    "uncertainty_threshold", "epsilon", "buffer_size",
                )},
            }
            history.append(row)
            event_counts.clear()
            threshold = row["uncertainty_threshold"]
            print(
                f"step {step:>7} | wellbeing {row['wellbeing']:.3f} | "
                f"wm_loss {row['wm_loss']:.4f} | td {row['td_loss']:.4f} | "
                f"planner {row['planner_used_rate']:.2f} | "
                f"seuil {threshold if threshold is None else round(threshold, 4)} | "
                f"eau {row['interact_water']} | bouffe {row['interact_food']} | "
                f"morts {deaths} | eps {row['epsilon']:.2f}"
            )
            metrics_path = out_dir / "metrics_online.json"
            tmp = metrics_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"config": args.config, "seed": args.seed, "windows": history}, f, indent=2)
            tmp.replace(metrics_path)

    print(f"Terminé : {args.steps} steps, {deaths} morts, métriques dans {out_dir}/metrics_online.json")


if __name__ == "__main__":
    main()
