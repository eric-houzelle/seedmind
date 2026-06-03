"""SeedMind Sandbox — survival training loop.

Trains the agent in SandboxWorld where the only objective is to stay alive by
discovering the harvest-eat-survive loop on its own. Logs mean lifespan
instead of success rate.

    python scripts/run_sandbox.py
    python scripts/run_sandbox.py --config configs/sandbox_v0.yaml --episodes 5000
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.curiosity import compute_prediction_error
from seedmind.agent.encoder import Encoder
from seedmind.agent.goal_generator import GoalGenerator
from seedmind.agent.policy import EpsilonGreedyPolicy
from seedmind.agent.q_network import QNetwork
from seedmind.agent.sandbox_encoder import (
    make_sandbox_obs_batch_to_tensors,
    make_sandbox_observation_to_vector,
    sandbox_obs_batch_to_tensors,
    sandbox_observation_to_vector,
    sandbox_num_channels,
    sandbox_num_scalars,
)
from seedmind.agent.world_model import WorldModel
from seedmind.agent.agent import Agent
from seedmind.envs.sandbox_world import ACTIONS, CRAFT_ACTIONS, SandboxWorld
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.memory.persistent_memory import PersistentMemory
from seedmind.training.checkpointing import save_checkpoint
from seedmind.training.device import resolve_device
from seedmind.training.imagination import imagine_experiences
from seedmind.training.dqn import (
    make_q_optimizer,
    make_target_network,
    sync_target,
    train_dqn,
)
from seedmind.training.train import make_optimizer, train_world_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def craft_enabled(config: dict) -> bool:
    cc = config.get("craft", {})
    ec = config.get("env", {})
    return bool(
        cc.get("enabled", False)
        or int(ec.get("num_wood_sources", 0)) > 0
        or int(ec.get("num_stone_sources", 0)) > 0
    )


def sandbox_actions(config: dict) -> list[str]:
    return list(CRAFT_ACTIONS if craft_enabled(config) else ACTIONS)


def build_env(config: dict, seed: int) -> SandboxWorld:
    ec = config.get("env", {})
    cc = config.get("craft", {})
    enabled = craft_enabled(config)
    return SandboxWorld(
        size=int(ec.get("size", 8)),
        max_steps=int(ec.get("max_steps", 200)),
        energy_max=float(ec.get("energy_max", 100.0)),
        energy_start=float(ec.get("energy_start", 50.0)),
        energy_decay=float(ec.get("energy_decay", 1.0)),
        food_energy=float(ec.get("food_energy", 15.0)),
        num_food_sources=int(ec.get("num_food_sources", 6)),
        num_wood_sources=int(ec.get("num_wood_sources", 0)),
        num_stone_sources=int(ec.get("num_stone_sources", 0)),
        num_workbenches=int(ec.get("num_workbenches", 0)),
        regrow_delay=int(ec.get("regrow_delay", 15)),
        craft_enabled=enabled,
        base_food_yield=int(cc.get("base_food_yield", 1)),
        tool_food_bonus=int(cc.get("tool_food_bonus", 1)),
        visibility_radius=ec.get("visibility_radius"),
        seed=seed,
    )


def build_agent(config: dict, seed: int) -> Agent:
    ac = config.get("agent", {})
    wmc = config.get("world_model", {})
    cc = config.get("curiosity", {})
    pc = config.get("policy", {})
    dc = config.get("dqn", {})
    ec = config.get("env", {})
    include_craft = craft_enabled(config)
    actions = sandbox_actions(config)

    grid_size = int(ec.get("size", 8))
    latent_dim = int(ac.get("latent_dim", 64))
    num_channels = sandbox_num_channels(include_craft)
    num_scalars = sandbox_num_scalars(include_craft)
    obs_to_vec = (
        make_sandbox_observation_to_vector(include_craft)
        if include_craft else sandbox_observation_to_vector
    )
    obs_batch = (
        make_sandbox_obs_batch_to_tensors(include_craft)
        if include_craft else sandbox_obs_batch_to_tensors
    )
    input_dim = grid_size * grid_size * num_channels + num_scalars

    encoder = Encoder(
        grid_size=grid_size, latent_dim=latent_dim,
        num_entities=num_channels, seed=seed or 0,
        input_dim=input_dim,
        obs_to_vec_fn=obs_to_vec,
    )
    world_model = WorldModel(
        latent_dim=latent_dim, num_actions=len(actions),
        hidden_dim=int(wmc.get("hidden_dim", 128)),
        num_layers=int(wmc.get("num_layers", 2)),
    )
    from seedmind.agent.curiosity import CuriosityModule
    curiosity = CuriosityModule(
        weight=float(cc.get("weight", 0.3)),
        max_reward=float(cc.get("max_reward", 1.0)),
        enabled=bool(cc.get("enabled", True)),
    )
    goal_gen = GoalGenerator(seed=seed)
    policy = EpsilonGreedyPolicy(
        epsilon_start=float(pc.get("epsilon_start", 1.0)),
        epsilon_end=float(pc.get("epsilon_end", 0.1)),
        epsilon_decay_steps=int(pc.get("epsilon_decay_steps", 40000)),
        seed=seed,
    )
    memory = PersistentMemory()

    q_network = QNetwork(
        grid_size=grid_size, num_actions=len(actions),
        conv_channels=int(dc.get("conv_channels", 32)),
        hidden_dim=int(dc.get("hidden_dim", 128)),
        num_grid_channels=num_channels,
        num_scalars=num_scalars,
        obs_batch_fn=obs_batch,
    )

    plc = config.get("planning", {})
    planning_enabled = bool(plc.get("enabled", False))
    planning_weight = float(plc.get("weight", 0.0)) if planning_enabled else 0.0

    return Agent(
        encoder=encoder, world_model=world_model, curiosity=curiosity,
        goal_generator=goal_gen, policy=policy, memory=memory,
        actions=actions, memory_top_k=int(ac.get("memory_top_k", 5)),
        use_planner=planning_enabled, q_network=q_network,
        planning_weight=planning_weight,
        planner_horizon=int(plc.get("horizon", 4)),
        planner_samples=int(plc.get("num_samples", 16)),
    )


def _compact_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "grid": np.asarray(obs["grid"], dtype=np.int16),
        "energy": float(obs.get("energy", 0)),
        "energy_max": float(obs.get("energy_max", 100)),
        "inventory_food": int(obs.get("inventory_food", 0)),
    }
    for key in ("wood", "stone", "tool"):
        field = f"inventory_{key}"
        if field in obs:
            compact[field] = int(obs.get(field, 0))
    return compact


def _save_metrics(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _count_recent(metrics: list[Dict[str, Any]], key: str, window: int = 100) -> float:
    recent = metrics[-window:]
    if not recent:
        return 0.0
    return float(sum(m.get(key, 0) for m in recent)) / len(recent)


def main() -> None:
    parser = argparse.ArgumentParser(description="SeedMind Sandbox training")
    parser.add_argument("--config", default="configs/sandbox_v0.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument(
        "--inference-device",
        default=None,
        choices=["cpu", "auto", "cuda", "mps"],
        help="Device for step-by-step inference. Defaults to --device.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    inference_device = resolve_device(args.inference_device or args.device)
    config = load_config(args.config)
    ec = config.get("env", {})
    tc = config.get("training", {})
    wmc = config.get("world_model", {})
    dc = config.get("dqn", {})

    episodes = args.episodes or int(tc.get("episodes", 5000))
    max_steps = int(ec.get("max_steps", 200))
    train_every = int(tc.get("train_every", 1))
    checkpoint_every = int(tc.get("checkpoint_every", 1000))

    wm_batch = int(wmc.get("batch_size", 64))
    wm_lr = float(wmc.get("learning_rate", 3e-4))
    q_batch = int(dc.get("batch_size", 64))
    q_lr = float(dc.get("learning_rate", 5e-4))
    gamma = float(dc.get("gamma", 0.95))
    target_update = int(dc.get("target_update", 300))
    double_dqn = bool(dc.get("double_dqn", True))
    updates_per_train = int(dc.get("updates_per_train", 8))
    sampler = str(dc.get("sampler", "uniform"))
    curiosity_weight = float(dc.get("curiosity_weight", 0.0))

    dyna_cfg = config.get("dyna", {})
    dyna_enabled = bool(dyna_cfg.get("enabled", False))
    dyna_imagined = int(dyna_cfg.get("imagined_per_step", 16))

    out_dir = Path(args.out_dir or f"runs/sandbox_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(config, args.seed)
    agent.encoder.to(inference_device)
    agent.world_model.to(inference_device)
    train_q_network = agent.q_network.to(device)
    if inference_device != device:
        inference_q_network = copy.deepcopy(train_q_network).to(inference_device)
        inference_q_network.eval()
        agent.q_network = inference_q_network
    else:
        agent.q_network = train_q_network
    actions = sandbox_actions(config)
    buffer = ExperienceBuffer(seed=args.seed)
    dyna_rng = np.random.default_rng(args.seed + 999)
    wm_optimizer = make_optimizer(agent.world_model, learning_rate=wm_lr)
    q_optimizer = make_q_optimizer(train_q_network, learning_rate=q_lr)
    target_network = make_target_network(train_q_network)

    last_wm_loss = 0.0
    last_td_loss = 0.0
    total_q_updates = 0
    next_target_sync = target_update
    recent_lifespan: deque = deque(maxlen=100)
    metrics_history: list[Dict[str, Any]] = []
    metrics_path = out_dir / "metrics.json"

    print(
        f"Running SeedMind Sandbox: {episodes} lives, max_steps={max_steps}, "
        f"device={device}, inference_device={inference_device}"
    )

    for ep in range(episodes):
        env = build_env(config, seed=args.seed + ep)
        observation = env.reset()
        latent_state = agent.encode(observation)

        ep_reward = 0.0
        ep_steps = 0
        event_counts: Counter[str] = Counter()
        food_harvested = 0
        bonus_food_from_tool = 0
        max_food = 0
        max_wood = 0
        max_stone = 0
        max_tool = 0

        for step in range(max_steps):
            memories = agent.retrieve(latent_state)
            goal = agent.choose_goal(latent_state, memories)
            action = agent.choose_action(
                latent_state, goal, memories, env.available_actions(),
                observation=observation,
            )
            action_index = agent.action_index[action]

            next_obs, reward_ext, done, info = env.step(action)
            next_latent = agent.encode(next_obs)
            event = str(info.get("event", "unknown"))
            amount = int(info.get("event_amount", 0))
            event_counts[event] += 1
            if event in {"harvest_food", "harvest_food_tool"}:
                food_harvested += amount
            if event == "harvest_food_tool":
                bonus_food_from_tool += max(0, amount - 1)
            inv = info.get("inventory", {})
            max_food = max(max_food, int(inv.get("food", 0)))
            max_wood = max(max_wood, int(inv.get("wood", 0)))
            max_stone = max(max_stone, int(inv.get("stone", 0)))
            max_tool = max(max_tool, int(inv.get("tool", 0)))

            predicted, _, _ = agent.world_model.predict(latent_state, action_index)
            pred_err = compute_prediction_error(predicted, next_latent)
            reward_int = agent.curiosity.compute(pred_err)

            experience = make_experience(
                episode_id=f"sandbox_{ep:06d}", world_id=env.world_id,
                step=step, observation=observation["grid"].tolist(),
                action=action, next_observation=next_obs["grid"].tolist(),
                reward_external=reward_ext, reward_intrinsic=reward_int,
                goal=goal, prediction_error=pred_err, done=done,
                memory_used=[], latent_state=latent_state,
                next_latent_state=next_latent, action_index=action_index,
                obs_state=_compact_obs(observation),
                next_obs_state=_compact_obs(next_obs),
                event=event,
                event_amount=amount,
            )
            buffer.add(experience)
            agent.memory.store_if_important(experience)

            ep_reward += reward_ext
            ep_steps = step + 1
            observation = next_obs
            latent_state = next_latent

            if done:
                break

        recent_lifespan.append(ep_steps)
        mean_life = float(np.mean(recent_lifespan))
        episode_metrics = {
            "episode": ep,
            "lifespan": ep_steps,
            "reward_external": float(ep_reward),
            "dead": bool(info.get("dead", False)),
            "timeout": bool(info.get("timeout", False)),
            "epsilon": float(agent.policy.epsilon),
            "td_loss": float(last_td_loss),
            "world_model_loss": float(last_wm_loss),
            "memory_items": len(agent.memory),
            "harvest_food": int(event_counts.get("harvest_food", 0)),
            "harvest_food_tool": int(event_counts.get("harvest_food_tool", 0)),
            "harvest_wood": int(event_counts.get("harvest_wood", 0)),
            "harvest_stone": int(event_counts.get("harvest_stone", 0)),
            "craft_tool": int(event_counts.get("craft_tool", 0)),
            "eat_ok": int(event_counts.get("eat_ok", 0)),
            "food_harvested": int(food_harvested),
            "bonus_food_from_tool": int(bonus_food_from_tool),
            "max_inventory_food": int(max_food),
            "max_inventory_wood": int(max_wood),
            "max_inventory_stone": int(max_stone),
            "max_inventory_tool": int(max_tool),
        }
        metrics_history.append(episode_metrics)

        # Training
        if ep % train_every == 0 and len(buffer) >= q_batch:
            wm_losses = train_world_model(
                agent.world_model, buffer, wm_optimizer,
                batch_size=wm_batch, num_updates=updates_per_train,
            )
            last_wm_loss = wm_losses["total"]

            # Dyna: inject imagined experiences from the World Model
            if dyna_enabled and len(buffer) >= dyna_imagined:
                dreams = imagine_experiences(
                    agent.world_model, buffer, agent.curiosity,
                    num_actions=len(actions), num_imagined=dyna_imagined,
                    rng=dyna_rng,
                )
                for d in dreams:
                    buffer.add(d)

            q_losses = train_dqn(
                train_q_network, target_network, buffer, q_optimizer,
                batch_size=q_batch, gamma=gamma, curiosity_weight=curiosity_weight,
                double_dqn=double_dqn, num_updates=updates_per_train, sampler=sampler,
            )
            last_td_loss = q_losses["td_loss"]
            total_q_updates += int(q_losses["updates"])
            if agent.q_network is not train_q_network:
                agent.q_network.load_state_dict(train_q_network.state_dict())
                agent.q_network.eval()
            if total_q_updates >= next_target_sync:
                sync_target(train_q_network, target_network)
                next_target_sync += target_update

        if ep % checkpoint_every == 0 and ep > 0:
            if agent.q_network is not train_q_network:
                agent.q_network.load_state_dict(train_q_network.state_dict())
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, wm_optimizer, buffer,
                metrics={
                    "mean_lifespan": mean_life,
                    "device": str(device),
                    "inference_device": str(inference_device),
                }, config=config,
                q_optimizer=q_optimizer, target_network=target_network,
            )
            _save_metrics(metrics_path, {
                "config_path": args.config,
                "device": str(device),
                "inference_device": str(inference_device),
                "episodes": metrics_history,
            })

        if ep % max(1, episodes // 20) == 0 or ep == episodes - 1:
            dead = info.get("dead", False)
            craft100 = _count_recent(metrics_history, "craft_tool")
            tool_food100 = _count_recent(metrics_history, "harvest_food_tool")
            eat100 = _count_recent(metrics_history, "eat_ok")
            print(
                f"  life {ep:5d} | lifespan(100)={mean_life:5.1f} "
                f"r={ep_reward:6.2f} dead={dead} "
                f"td={last_td_loss:.4f} wm={last_wm_loss:.4f} "
                f"eps={agent.policy.epsilon:.2f} mem={len(agent.memory)} "
                f"craft100={craft100:.2f} toolfood100={tool_food100:.2f} eat100={eat100:.2f}"
            )
            _save_metrics(metrics_path, {
                "config_path": args.config,
                "device": str(device),
                "inference_device": str(inference_device),
                "episodes": metrics_history,
            })

    if agent.q_network is not train_q_network:
        agent.q_network.load_state_dict(train_q_network.state_dict())
    save_checkpoint(
        str(out_dir / "checkpoint_final.pt"), agent, wm_optimizer, buffer,
        metrics={
            "mean_lifespan": float(np.mean(recent_lifespan)),
            "device": str(device),
            "inference_device": str(inference_device),
        },
        config=config, q_optimizer=q_optimizer, target_network=target_network,
    )
    _save_metrics(metrics_path, {
        "config_path": args.config,
        "device": str(device),
        "inference_device": str(inference_device),
        "episodes": metrics_history,
    })
    print(f"\nFinal mean lifespan (last 100): {float(np.mean(recent_lifespan)):.1f}")
    print(f"Checkpoint saved to {out_dir}/")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
