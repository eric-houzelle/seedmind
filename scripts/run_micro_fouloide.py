"""Train SeedMind in MicroFouloideWorld."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedmind.agent.agent import Agent
from seedmind.agent.curiosity import compute_prediction_error_tensor
from seedmind.agent.actor_critic import Actor
from seedmind.agent.encoder import ConvEncoder, Encoder
from seedmind.agent.goal_generator import GoalGenerator
from seedmind.agent.map_memory import MapMemory
from seedmind.agent.micro_fouloide_encoder import (
    make_micro_fouloide_obs_fns,
    make_micro_fouloide_property_obs_fns,
    wrap_egocentric,
)
from seedmind.agent.latent_q_network import LatentQNetwork
from seedmind.agent.policy import EpsilonGreedyPolicy
from seedmind.agent.q_network import QNetwork
from seedmind.agent.value_model import TwoHotCritic, ValueModel
from seedmind.agent.world_model import RecurrentWorldModel, RSSMWorldModel, WorldModel
from seedmind.envs.entities import load_registry
from seedmind.envs.micro_fouloide_world import MicroFouloideWorld, OBSTACLE
from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience
from seedmind.memory.persistent_memory import PersistentMemory
from seedmind.objectives import build_objective_scorer
from seedmind.training.checkpointing import load_checkpoint, save_checkpoint
from seedmind.training.device import resolve_device
from seedmind.training.dqn import make_q_optimizer, make_target_network, sync_target, train_dqn
from seedmind.training.latent_dqn import (
    make_latent_q_optimizer,
    make_latent_target_network,
    sync_latent_target,
    train_latent_dqn,
    train_latent_dqn_dyna,
)
from seedmind.training.latent_utils import latent_to_numpy
from seedmind.training.wellbeing import drive_regulation, wellbeing
from seedmind.training.train import (
    make_optimizer,
    train_world_model,
    train_world_model_uncertainty_head,
)
from seedmind.training.value import (
    evaluate_value_model_on_returns,
    make_value_optimizer,
    sync_value_target,
    train_value_model,
    train_value_model_dyna,
    train_value_model_on_returns,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(config: dict, seed: int) -> MicroFouloideWorld:
    ec = config.get("env", {})
    return MicroFouloideWorld(
        size=int(ec.get("size", 16)),
        max_steps=int(ec.get("max_steps", 500)),
        visibility_radius=ec.get("visibility_radius", 4),
        energy_start=float(ec.get("energy_start", 0.75)),
        hydration_start=float(ec.get("hydration_start", 0.75)),
        temperature_start=float(ec.get("temperature_start", 0.5)),
        health_start=float(ec.get("health_start", 1.0)),
        energy_decay=float(ec.get("energy_decay", 0.006)),
        hydration_decay=float(ec.get("hydration_decay", 0.008)),
        rest_energy_decay_scale=float(ec.get("rest_energy_decay_scale", 0.35)),
        food_energy_gain=float(ec.get("food_energy_gain", 0.35)),
        water_hydration_gain=float(ec.get("water_hydration_gain", 0.45)),
        temperature_drift=float(ec.get("temperature_drift", 0.015)),
        temperature_recovery=float(ec.get("temperature_recovery", 0.004)),
        critical_threshold=float(ec.get("critical_threshold", 0.12)),
        health_decay=float(ec.get("health_decay", 0.025)),
        danger_damage=float(ec.get("danger_damage", 0.08)),
        soft_death=bool(ec.get("soft_death", False)),
        health_floor=float(ec.get("health_floor", 0.05)),
        health_regen=float(ec.get("health_regen", 0.01)),
        soft_death_grace_steps=int(ec.get("soft_death_grace_steps", 0)),
        critical_kill_health_decay=(
            float(ec["critical_kill_health_decay"])
            if "critical_kill_health_decay" in ec
            else None
        ),
        resource_regrow_steps=int(ec.get("resource_regrow_steps", 0)),
        num_food=int(ec.get("num_food", 10)),
        num_water=int(ec.get("num_water", 8)),
        num_warm_zones=int(ec.get("num_warm_zones", 6)),
        num_cold_zones=int(ec.get("num_cold_zones", 6)),
        num_dangers=int(ec.get("num_dangers", 8)),
        num_obstacles=int(ec.get("num_obstacles", 20)),
        filter_blocked_moves=bool(ec.get("filter_blocked_moves", False)),
        filter_noop_interact=bool(ec.get("filter_noop_interact", False)),
        filter_noop_inventory=bool(ec.get("filter_noop_inventory", False)),
        inventory_enabled=bool(ec.get("inventory", {}).get("enabled", False)),
        inventory_capacity=int(ec.get("inventory", {}).get("capacity", 3)),
        property_events=bool(ec.get("property_events", False)),
        registry=load_registry(config),
        entity_counts=ec,
        seed=seed,
    )


def _probe_env(config: dict) -> MicroFouloideWorld:
    return build_env(config, seed=0)


def causal_feature_names(config: dict) -> list[str]:
    return _probe_env(config).causal_feature_names()


def causal_event_names(config: dict) -> list[str]:
    return _probe_env(config).causal_event_names()


def _planner_force_thresholds(config: dict) -> tuple[list[int], list[float]]:
    objective_cfg = config.get("objective", {})
    force_cfg = dict(objective_cfg.get("force_planner_below", {}))
    if not force_cfg:
        return [], []
    names = causal_feature_names(config)
    indices: list[int] = []
    thresholds: list[float] = []
    for name, threshold in force_cfg.items():
        if name not in names:
            continue
        indices.append(names.index(name))
        thresholds.append(float(threshold))
    return indices, thresholds


def build_agent(config: dict, seed: int) -> Agent:
    ac = config.get("agent", {})
    wmc = config.get("world_model", {})
    cwm = config.get("causal_world_model", {})
    cc = config.get("curiosity", {})
    pc = config.get("policy", {})
    dc = config.get("dqn", {})
    drc = config.get("drive_reward", {})
    vc = config.get("value_model", {})
    objective_cfg = config.get("objective", {})
    ec = config.get("env", {})
    grid_size = int(ec.get("size", 16))
    latent_dim = int(ac.get("latent_dim", 64))
    registry = load_registry(config)
    inventory_enabled = bool(ec.get("inventory", {}).get("enabled", False))
    obs_cfg = ac.get("observation", {})
    if str(obs_cfg.get("mode", "onehot")) == "properties":
        obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars = (
            make_micro_fouloide_property_obs_fns(
                registry,
                inventory=inventory_enabled,
                memory=bool(obs_cfg.get("spatial_memory", {}).get("enabled", False)),
            )
        )
    else:
        obs_to_vec_fn, obs_batch_fn, num_channels, num_scalars = make_micro_fouloide_obs_fns(
            registry.size, inventory=inventory_enabled,
        )
    actions = _probe_env(config).actions

    # Egocentric perception: a fixed window centred on the agent makes the
    # network independent of the world size (and lets the world grow). It crops
    # the grid before the channel encoders, so it composes with both obs modes.
    ego_cfg = obs_cfg.get("egocentric", {})
    egocentric = bool(ego_cfg.get("enabled", False))
    if egocentric:
        radius = int(ego_cfg.get("radius", 5))
        oob_fill = int(ego_cfg.get("oob_fill", OBSTACLE))
        obs_to_vec_fn, obs_batch_fn = wrap_egocentric(
            obs_to_vec_fn, obs_batch_fn, radius, oob_fill,
        )
        net_grid_size = 2 * radius + 1
    else:
        net_grid_size = grid_size

    input_dim = net_grid_size * net_grid_size * num_channels + num_scalars
    structured_latent_features = bool(ac.get("structured_latent_features", False))
    structured_feature_dim = len(causal_feature_names(config)) if structured_latent_features else 0
    structured_feature_env = _probe_env(config) if structured_latent_features else None
    structured_features_fn = (
        structured_feature_env.causal_features
        if structured_feature_env is not None else None
    )

    if egocentric:
        enc_cfg = ac.get("encoder", {})
        encoder = ConvEncoder(
            num_channels=num_channels,
            num_scalars=num_scalars,
            window_size=net_grid_size,
            latent_dim=latent_dim,
            conv_channels=int(enc_cfg.get("conv_channels", 32)),
            hidden_dim=int(enc_cfg.get("hidden_dim", 256)),
            seed=seed or 0,
            obs_batch_fn=obs_batch_fn,
            structured_features_fn=structured_features_fn,
            structured_feature_dim=structured_feature_dim,
        )
    else:
        encoder = Encoder(
            grid_size=grid_size,
            latent_dim=latent_dim,
            num_entities=num_channels,
            seed=seed or 0,
            input_dim=input_dim,
            obs_to_vec_fn=obs_to_vec_fn,
            structured_features_fn=structured_features_fn,
            structured_feature_dim=structured_feature_dim,
        )
    # Recurrent world model (RSSM trajectory): a GRU state h_t gives the agent
    # memory beyond its egocentric view. deter_dim sizes h_t; the Q-network
    # receives h_t (recurrent_dim) so the policy can act on memory.
    recurrent_wm = bool(wmc.get("recurrent", False))
    rssm_stochastic = recurrent_wm and bool(wmc.get("rssm_stochastic", False))
    deter_dim = int(wmc.get("deter_dim", 128))
    wm_causal_dim = (
        len(causal_feature_names(config)) if bool(cwm.get("enabled", False)) else 0
    )
    wm_num_events = (
        len(causal_event_names(config)) if bool(cwm.get("predict_events", False)) else 0
    )
    if rssm_stochastic:
        # DreamerV3 stochastic RSSM: state (h, z); embed = the (frozen) encoder latent.
        world_model = RSSMWorldModel(
            embed_dim=latent_dim,
            num_actions=len(actions),
            stoch=int(wmc.get("rssm_stoch", 32)),
            discrete=int(wmc.get("rssm_discrete", 32)),
            deter=deter_dim,
            hidden=int(wmc.get("hidden_dim", 128)),
            unimix=float(wmc.get("rssm_unimix", 0.01)),
            causal_feature_dim=wm_causal_dim,
            num_events=wm_num_events,
            reward_twohot=bool(wmc.get("reward_twohot", True)),
            reward_bins=int(wmc.get("reward_bins", 255)),
            reward_vmax=float(wmc.get("reward_vmax", 20.0)),
        )
    elif recurrent_wm:
        world_model = RecurrentWorldModel(
            latent_dim=latent_dim,
            num_actions=len(actions),
            hidden_dim=int(wmc.get("hidden_dim", 128)),
            deter_dim=deter_dim,
            num_layers=int(wmc.get("num_layers", 2)),
            causal_feature_dim=wm_causal_dim,
            num_events=wm_num_events,
        )
    else:
        world_model = WorldModel(
            latent_dim=latent_dim,
            num_actions=len(actions),
            hidden_dim=int(wmc.get("hidden_dim", 128)),
            num_layers=int(wmc.get("num_layers", 2)),
            causal_feature_dim=wm_causal_dim,
            num_events=wm_num_events,
        )
    from seedmind.agent.curiosity import CuriosityModule
    curiosity = CuriosityModule(
        weight=float(cc.get("weight", 0.5)),
        max_reward=float(cc.get("max_reward", 1.0)),
        enabled=bool(cc.get("enabled", True)),
    )
    # Feature the policy (actor/critic, and the Q-net's recurrent input) reads:
    # the model feature feat=[z,h] for the stochastic RSSM, else just h.
    policy_feat_dim = world_model.feat_dim if rssm_stochastic else deter_dim
    q_network = QNetwork(
        grid_size=net_grid_size,
        num_actions=len(actions),
        conv_channels=int(dc.get("conv_channels", 64)),
        hidden_dim=int(dc.get("hidden_dim", 256)),
        num_grid_channels=num_channels,
        num_scalars=num_scalars,
        obs_batch_fn=obs_batch_fn,
        recurrent_dim=policy_feat_dim if recurrent_wm else 0,
    )
    plc = config.get("planning", {})
    planning_enabled = bool(plc.get("enabled", False))
    value_model = None
    if bool(vc.get("enabled", False)):
        value_model = ValueModel(
            latent_dim=latent_dim,
            hidden_dim=int(vc.get("hidden_dim", 128)),
            num_layers=int(vc.get("num_layers", 2)),
        )
    # Imagination policy (Dreamer-style): actor + critic over the recurrent
    # state h_t, trained on imagined rollouts. Requires the recurrent world model.
    actor = None
    critic = None
    if bool(ac.get("imagination_policy", False)) and recurrent_wm:
        acc = ac.get("actor", {})
        actor = Actor(
            input_dim=policy_feat_dim,
            num_actions=len(actions),
            hidden_dim=int(acc.get("hidden_dim", 128)),
            num_layers=int(acc.get("num_layers", 2)),
        )
        imc = config.get("imagination", {})
        if bool(imc.get("critic_twohot", False)):
            critic = TwoHotCritic(
                latent_dim=policy_feat_dim,
                hidden_dim=int(acc.get("critic_hidden_dim", 128)),
                num_layers=int(acc.get("critic_num_layers", 2)),
                num_bins=int(imc.get("critic_num_bins", 255)),
                vmax=float(imc.get("critic_vmax", 20.0)),
            )
        else:
            critic = ValueModel(
                latent_dim=policy_feat_dim,
                hidden_dim=int(acc.get("critic_hidden_dim", 128)),
                num_layers=int(acc.get("critic_num_layers", 2)),
            )

    force_indices, force_thresholds = _planner_force_thresholds(config)
    return Agent(
        encoder=encoder,
        world_model=world_model,
        curiosity=curiosity,
        goal_generator=GoalGenerator(seed=seed),
        policy=EpsilonGreedyPolicy(
            epsilon_start=float(pc.get("epsilon_start", 1.0)),
            epsilon_end=float(pc.get("epsilon_end", 0.1)),
            epsilon_decay_steps=int(pc.get("epsilon_decay_steps", 200000)),
            seed=seed,
        ),
        memory=PersistentMemory(),
        actions=actions,
        memory_top_k=int(ac.get("memory_top_k", 5)),
        use_planner=planning_enabled,
        q_network=q_network,
        planning_weight=float(plc.get("weight", 0.0)) if planning_enabled else 0.0,
        planner_horizon=int(plc.get("horizon", 3)),
        planner_samples=int(plc.get("num_samples", 8)),
        value_model=value_model,
        planner_terminal_value_weight=float(plc.get("terminal_value_weight", 0.0)),
        planner_objective_scorer=build_objective_scorer(
            config,
            causal_feature_names(config),
        ),
        planner_objective_weight=float(objective_cfg.get("planner_weight", 0.0)),
        planner_action_penalties=dict(objective_cfg.get("action_penalties", {})),
        planner_force_feature_indices=force_indices,
        planner_force_feature_thresholds=force_thresholds,
        causal_features_fn=(
            _probe_env(config).causal_features
            if bool(cwm.get("enabled", False)) else None
        ),
        planner_seed=seed,
        planner_uncertainty_threshold=(
            float(plc["uncertainty_threshold"])
            if plc.get("uncertainty_threshold") is not None else None
        ),
        planner_margin_threshold=float(plc.get("margin_threshold", 0.0)),
        planner_q_advantage_threshold=float(plc.get("q_advantage_threshold", 0.0)),
        actor=actor,
        critic=critic,
    )


def _compact_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "grid": np.asarray(obs["grid"], dtype=np.int16),
        "standing_entity": int(obs.get("standing_entity", 0)),
        "energy": float(obs.get("energy", 0.0)),
        "hydration": float(obs.get("hydration", 0.0)),
        "temperature": float(obs.get("temperature", 0.5)),
        "health": float(obs.get("health", 0.0)),
    }
    if "inventory" in obs:
        compact["inventory"] = np.asarray(obs["inventory"], dtype=np.float32)
    if "memory_grid" in obs:
        compact["memory_grid"] = np.asarray(obs["memory_grid"], dtype=np.int16)
        compact["memory_fresh"] = np.asarray(obs["memory_fresh"], dtype=np.float16)
    return compact


def _spatial_memory_config(config: dict) -> Optional[dict]:
    sm = config.get("agent", {}).get("observation", {}).get("spatial_memory", {})
    return sm if bool(sm.get("enabled", False)) else None


def _save_metrics(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_metrics_history(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    return episodes if isinstance(episodes, list) else []


def _mean_recent(metrics: list[Dict[str, Any]], key: str, window: int = 100) -> float:
    recent = metrics[-window:]
    if not recent:
        return 0.0
    return float(sum(m.get(key, 0.0) for m in recent)) / len(recent)


def _drive_regulation(info: Dict[str, Any]) -> float:
    return drive_regulation(info.get("drives", {}))


def _comfort_config(config: dict) -> Optional[dict]:
    comfort = config.get("drive_reward", {}).get("comfort", {})
    return comfort if bool(comfort.get("enabled", False)) else None


def _event_resource_reward(
    observation: Dict[str, Any],
    info: Dict[str, Any],
    config: dict,
) -> float:
    cfg = config.get("resource_reward", {})
    if not bool(cfg.get("enabled", False)):
        return 0.0

    event = str(info.get("event", ""))
    reward = 0.0
    event_rewards = cfg.get("event_rewards", {})
    if event in event_rewards:
        reward += float(event_rewards[event])

    hydration = float(observation.get("hydration", info.get("hydration", 0.0)))
    energy = float(observation.get("energy", info.get("energy", 0.0)))
    low_threshold = float(cfg.get("low_threshold", 0.35))
    critical_threshold = float(cfg.get("critical_threshold", low_threshold))
    if event in {"interact_water", "interact_hydration"}:
        if hydration <= low_threshold:
            reward += float(cfg.get("low_hydration_water_bonus", 0.0))
        if hydration <= critical_threshold:
            reward += float(cfg.get("critical_hydration_water_bonus", 0.0))
    elif event in {"interact_food", "interact_energy"}:
        if energy <= low_threshold:
            reward += float(cfg.get("low_energy_food_bonus", 0.0))
        if energy <= critical_threshold:
            reward += float(cfg.get("critical_energy_food_bonus", 0.0))

    if event in {"wait", "rest"}:
        hydration_after = float(info.get("hydration", hydration))
        energy_after = float(info.get("energy", energy))
        passive_threshold = float(cfg.get("passive_penalty_threshold", low_threshold))
        if hydration_after <= passive_threshold or energy_after <= passive_threshold:
            reward -= float(cfg.get("low_drive_passive_penalty", 0.0))

    grid = np.asarray(observation.get("grid", []), dtype=np.int64)
    active_signal_events = set(cfg.get("local_signal_events", ["move_ok"]))
    if event in active_signal_events and grid.size > 0:
        water_ids, food_ids = _resource_signal_ids(config)
        water_visible = bool(np.any(np.isin(grid, water_ids)))
        food_visible = bool(np.any(np.isin(grid, food_ids)))
        if hydration <= low_threshold and water_visible:
            reward += float(cfg.get("low_hydration_water_signal_bonus", 0.0))
        if hydration <= critical_threshold and water_visible:
            reward += float(cfg.get("critical_hydration_water_signal_bonus", 0.0))
        if energy <= low_threshold and food_visible:
            reward += float(cfg.get("low_energy_food_signal_bonus", 0.0))
        if energy <= critical_threshold and food_visible:
            reward += float(cfg.get("critical_energy_food_signal_bonus", 0.0))

    return reward * float(cfg.get("weight", 1.0))


def _resource_signal_ids(config: dict) -> tuple[list[int], list[int]]:
    """Entity ids whose visibility counts as water/food signal (cached in config)."""
    cached = config.get("_resource_signal_ids")
    if cached is None:
        registry = load_registry(config)
        cached = (
            sorted(registry.drive_signal_ids("hydration")),
            sorted(registry.drive_signal_ids("energy")),
        )
        config["_resource_signal_ids"] = cached
    return cached


def _learning_reward(
    reward_ext: float,
    observation: Dict[str, Any],
    info: Dict[str, Any],
    config: dict,
) -> float:
    reward_learning = float(reward_ext)
    drc = config.get("drive_reward", {})
    if bool(drc.get("enabled", False)):
        comfort = _comfort_config(config)
        prev_drive = wellbeing(observation, comfort)
        next_drive = wellbeing(info.get("drives", {}), comfort)
        if str(drc.get("mode", "delta")) == "absolute":
            drive_bonus = next_drive
        else:
            drive_bonus = next_drive - prev_drive
        reward_learning += float(drc.get("weight", 0.0)) * float(drive_bonus)
    reward_learning += _event_resource_reward(observation, info, config)
    return float(reward_learning)


def _discounted_returns_by_episode(rows: list[dict], reward_key: str, gamma: float) -> dict[int, float]:
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


def _run_final_calibration(
    agent: Agent,
    buffer: ExperienceBuffer,
    config: dict,
    device: torch.device,
) -> dict[str, Any]:
    fcc = config.get("final_calibration", {})
    if not bool(fcc.get("enabled", False)):
        return {}

    metrics: dict[str, Any] = {}
    wmc = config.get("world_model", {})
    cwm = config.get("causal_world_model", {})
    uncertainty_cfg = fcc.get("uncertainty", {})
    if bool(uncertainty_cfg.get("enabled", True)):
        agent.world_model.to(device)
        optimizer = torch.optim.Adam(
            agent.world_model.uncertainty_head.parameters(),
            lr=float(uncertainty_cfg.get("learning_rate", 3e-4)),
        )
        result = train_world_model_uncertainty_head(
            agent.world_model,
            buffer,
            optimizer,
            batch_size=int(uncertainty_cfg.get("batch_size", wmc.get("batch_size", 64))),
            num_updates=int(uncertainty_cfg.get("updates", 2000)),
            sampler=str(uncertainty_cfg.get("sampler", wmc.get("sampler", "causal"))),
            causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
            causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
            event_class_balance=bool(cwm.get("event_class_balance", False)),
            event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
            reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
            reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
        )
        metrics["final_uncertainty_calibration"] = {
            "updates": result["updates"],
            "uncertainty_loss": result["uncertainty"],
        }

    value_cfg = fcc.get("value", {})
    if bool(value_cfg.get("enabled", True)):
        if agent.value_model is None:
            raise ValueError("final_calibration.value requires value_model.enabled=true.")
        rows = [
            row for row in buffer._data
            if row.get("latent_state") is not None
        ]
        if not rows:
            raise ValueError("Cannot run final value calibration without latent replay rows.")
        vc = config.get("value_model", {})
        reward_key = str(value_cfg.get(
            "reward_key",
            vc.get("reward_key", config.get("dqn", {}).get("reward_key", "reward_external")),
        ))
        gamma = float(value_cfg.get("gamma", vc.get("gamma", config.get("dqn", {}).get("gamma", 0.97))))
        returns_by_idx = _discounted_returns_by_episode(rows, reward_key, gamma)
        paired = [
            (row, returns_by_idx[idx])
            for idx, row in enumerate(rows)
            if idx in returns_by_idx
        ]
        if not paired:
            raise ValueError("Could not compute replay returns for final value calibration.")
        max_samples = int(value_cfg.get("max_samples", 0))
        if max_samples > 0 and len(paired) > max_samples:
            rng = np.random.default_rng(int(value_cfg.get("seed", 0)))
            indices = rng.choice(np.arange(len(paired)), size=max_samples, replace=False)
            paired = [paired[int(i)] for i in indices]

        latents = np.stack([
            np.asarray(row["latent_state"], dtype=np.float32)
            for row, _ in paired
        ])
        returns = np.asarray([ret for _, ret in paired], dtype=np.float32)
        agent.value_model.to(device)
        before = evaluate_value_model_on_returns(agent.value_model, latents, returns)
        optimizer = make_value_optimizer(
            agent.value_model,
            learning_rate=float(value_cfg.get("learning_rate", vc.get("learning_rate", 3e-4))),
        )
        result = train_value_model_on_returns(
            agent.value_model,
            latents,
            returns,
            optimizer,
            batch_size=int(value_cfg.get("batch_size", vc.get("batch_size", 64))),
            num_updates=int(value_cfg.get("updates", 5000)),
            seed=int(value_cfg.get("seed", 0)),
        )
        after = evaluate_value_model_on_returns(agent.value_model, latents, returns)
        metrics["final_value_calibration"] = {
            "samples": len(paired),
            "updates": result["updates"],
            "value_return_loss": result["value_return_loss"],
            "reward_key": reward_key,
            "gamma": gamma,
            "before": before,
            "after": after,
        }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SeedMind Micro-Fouloide training")
    parser.add_argument("--config", default="configs/micro_fouloide_v0.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--inference-device", default=None, choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    inference_device = resolve_device(args.inference_device or args.device)
    config = load_config(args.config)
    ec = config.get("env", {})
    tc = config.get("training", {})
    wmc = config.get("world_model", {})
    cwm = config.get("causal_world_model", {})
    dc = config.get("dqn", {})
    drc = config.get("drive_reward", {})
    vc = config.get("value_model", {})
    lpc = config.get("latent_policy", {})

    episodes = args.episodes or int(tc.get("episodes", 5000))
    max_steps = int(ec.get("max_steps", 500))
    train_every = int(tc.get("train_every", 1))
    checkpoint_every = int(tc.get("checkpoint_every", 1000))
    best_checkpoint_enabled = bool(tc.get("best_checkpoint_enabled", False))
    best_checkpoint_window = int(tc.get("best_checkpoint_window", 100))
    best_checkpoint_min_episode = int(tc.get("best_checkpoint_min_episode", best_checkpoint_window - 1))
    best_checkpoint_min_delta = float(tc.get("best_checkpoint_min_delta", 0.0))
    wm_batch = int(wmc.get("batch_size", 64))
    wm_sampler = str(wmc.get("sampler", "uniform"))
    wm_uncertainty_head_updates = int(wmc.get("uncertainty_head_updates_per_train", 0))
    wm_uncertainty_head_batch = int(wmc.get("uncertainty_head_batch_size", wm_batch))
    wm_uncertainty_head_sampler = str(wmc.get("uncertainty_head_sampler", wm_sampler))
    q_batch = int(dc.get("batch_size", 64))
    updates_per_train = int(dc.get("updates_per_train", 8))
    gamma = float(dc.get("gamma", 0.97))
    target_update = int(dc.get("target_update", 500))
    drive_reward_enabled = bool(drc.get("enabled", False))
    dqn_reward_key = str(dc.get(
        "reward_key",
        "reward_learning" if drive_reward_enabled else "reward_external",
    ))
    value_enabled = bool(vc.get("enabled", False))
    value_batch = int(vc.get("batch_size", q_batch))
    value_updates_per_train = int(vc.get("updates_per_train", updates_per_train))
    value_target_update = int(vc.get("target_update", target_update))
    value_reward_key = str(vc.get("reward_key", dqn_reward_key))
    value_dyna = vc.get("dyna", {})
    value_dyna_enabled = bool(value_dyna.get("enabled", False))
    latent_policy_enabled = bool(lpc.get("enabled", False))
    latent_policy_dyna = lpc.get("dyna", {})
    latent_policy_dyna_enabled = bool(latent_policy_dyna.get("enabled", False))
    event_to_index = {event: i for i, event in enumerate(causal_event_names(config))}
    causal_wm_enabled = bool(cwm.get("enabled", False))

    out_dir = Path(args.out_dir or f"runs/micro_fouloide_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    agent = build_agent(config, args.seed)
    agent.encoder.to(inference_device)
    agent.world_model.to(inference_device)
    if agent.value_model is not None:
        agent.value_model.to(device)
    train_q_network = agent.q_network.to(device)
    if inference_device != device:
        agent.q_network = copy.deepcopy(train_q_network).to(inference_device)
        agent.q_network.eval()
    else:
        agent.q_network = train_q_network
    buffer = ExperienceBuffer(seed=args.seed)
    wm_optimizer = make_optimizer(agent.world_model, learning_rate=float(wmc.get("learning_rate", 3e-4)))
    wm_uncertainty_optimizer = torch.optim.Adam(
        agent.world_model.uncertainty_head.parameters(),
        lr=float(wmc.get("uncertainty_head_learning_rate", wmc.get("learning_rate", 3e-4))),
    )
    q_optimizer = make_q_optimizer(train_q_network, learning_rate=float(dc.get("learning_rate", 5e-4)))
    target_network = make_target_network(train_q_network)
    value_optimizer = None
    target_value_model = None
    if agent.value_model is not None:
        value_optimizer = make_value_optimizer(
            agent.value_model,
            learning_rate=float(vc.get("learning_rate", 3e-4)),
        )
        import copy as _copy
        target_value_model = _copy.deepcopy(agent.value_model).to(device)
        target_value_model.eval()
    latent_q_network = None
    target_latent_q_network = None
    latent_q_optimizer = None
    if latent_policy_enabled:
        latent_q_network = LatentQNetwork(
            latent_dim=int(config.get("agent", {}).get("latent_dim", 64)),
            num_actions=len(agent.actions),
            hidden_dim=int(lpc.get("hidden_dim", 128)),
            num_layers=int(lpc.get("num_layers", 2)),
        ).to(device)
        target_latent_q_network = make_latent_target_network(latent_q_network).to(device)
        latent_q_optimizer = make_latent_q_optimizer(
            latent_q_network,
            learning_rate=float(lpc.get("learning_rate", 3e-4)),
        )

    last_wm_loss = 0.0
    last_wm_uncertainty_loss = 0.0
    last_td_loss = 0.0
    last_value_loss = 0.0
    last_value_dyna_loss = 0.0
    last_latent_q_loss = 0.0
    last_latent_dyna_loss = 0.0
    total_q_updates = 0
    total_value_updates = 0
    total_latent_q_updates = 0
    next_target_sync = target_update
    next_value_target_sync = value_target_update
    next_latent_target_sync = int(lpc.get("target_update", target_update))
    recent_lifespan: deque = deque(maxlen=best_checkpoint_window)
    metrics_history: list[Dict[str, Any]] = []
    start_episode = 0
    best_mean_life = float("-inf")
    best_checkpoint_path = out_dir / "checkpoint_best.pt"

    if args.resume:
        resume_path = Path(args.resume)
        resume_info = load_checkpoint(
            str(resume_path), agent, wm_optimizer, buffer,
            q_optimizer=q_optimizer, target_network=target_network,
            value_optimizer=value_optimizer, target_value_model=target_value_model,
            latent_q_network=latent_q_network,
            latent_q_optimizer=latent_q_optimizer,
            target_latent_q_network=target_latent_q_network,
        )
        agent.encoder.to(inference_device)
        agent.world_model.to(inference_device)
        target_network.to(device)
        _move_optimizer_state(wm_optimizer, inference_device)
        _move_optimizer_state(q_optimizer, device)
        if value_optimizer is not None:
            _move_optimizer_state(value_optimizer, device)
        if latent_q_optimizer is not None:
            _move_optimizer_state(latent_q_optimizer, device)
        if agent.q_network is not train_q_network:
            train_q_network.load_state_dict(agent.q_network.state_dict())
            agent.q_network.to(inference_device)
            agent.q_network.eval()
        resume_metrics = resume_info.get("metrics", {})
        total_q_updates = int(resume_metrics.get("total_q_updates", total_q_updates))
        total_value_updates = int(resume_metrics.get("total_value_updates", total_value_updates))
        total_latent_q_updates = int(resume_metrics.get("total_latent_q_updates", total_latent_q_updates))
        next_target_sync = int(resume_metrics.get("next_target_sync", next_target_sync))
        next_value_target_sync = int(resume_metrics.get("next_value_target_sync", next_value_target_sync))
        next_latent_target_sync = int(resume_metrics.get("next_latent_target_sync", next_latent_target_sync))
        metrics_history = _load_metrics_history(metrics_path)
        if not metrics_history and resume_path.parent != out_dir:
            metrics_history = _load_metrics_history(resume_path.parent / "metrics.json")
        resume_episode = resume_metrics.get("episode")
        if resume_episode is not None:
            metrics_history = [
                m for m in metrics_history
                if int(m.get("episode", -1)) <= int(resume_episode)
            ]
        if metrics_history:
            start_episode = int(metrics_history[-1].get("episode", len(metrics_history) - 1)) + 1
            recent_lifespan.extend(int(m.get("lifespan", 0)) for m in metrics_history[-best_checkpoint_window:])
            if best_checkpoint_enabled:
                for idx in range(len(metrics_history)):
                    if idx + 1 < best_checkpoint_window:
                        continue
                    episode = int(metrics_history[idx].get("episode", idx))
                    if episode < best_checkpoint_min_episode:
                        continue
                    window_rows = metrics_history[max(0, idx + 1 - best_checkpoint_window): idx + 1]
                    best_mean_life = max(
                        best_mean_life,
                        float(sum(float(m.get("lifespan", 0.0)) for m in window_rows) / len(window_rows)),
                    )
        elif resume_episode is not None:
            start_episode = int(resume_episode) + 1
        print(f"Resuming from {resume_path} at episode {start_episode}.")
        if start_episode >= episodes:
            print(f"Checkpoint already reached target: {episodes}.")
            return

    print(
        f"Running SeedMind Micro-Fouloide: {episodes} episodes, max_steps={max_steps}, "
        f"device={device}, inference_device={inference_device}, wm_sampler={wm_sampler}"
    )

    sm_cfg = _spatial_memory_config(config)
    for ep in range(start_episode, episodes):
        env = build_env(config, seed=args.seed + ep)
        observation = env.reset()
        map_memory = None
        if sm_cfg is not None:
            map_memory = MapMemory(env.size, horizon=int(sm_cfg.get("horizon", 300)))
            map_memory.observe(observation)
            observation = map_memory.augment(observation)
        latent_state = agent.encoder.encode_tensor(observation)
        ep_reward = 0.0
        ep_steps = 0
        event_counts: Counter[str] = Counter()
        drive_scores = []
        min_health = 1.0
        mean_energy = 0.0
        mean_hydration = 0.0
        mean_temp_error = 0.0

        for step in range(max_steps):
            latent_np = latent_to_numpy(latent_state)
            memories = agent.retrieve(latent_np)
            goal = agent.choose_goal(latent_np, memories)
            action = agent.choose_action(
                latent_np, goal, memories, env.available_actions(),
                observation=observation,
            )
            action_index = agent.action_index[action]
            next_obs, reward_ext, done, info = env.step(action)
            if map_memory is not None:
                map_memory.observe(next_obs)
                next_obs = map_memory.augment(next_obs)
            next_latent = agent.encoder.encode_tensor(next_obs)
            event = str(info.get("event", "unknown"))
            amount = int(info.get("event_amount", 0))
            event_counts[event] += 1
            causal_features = env.causal_features(observation) if causal_wm_enabled else None
            next_causal_features = env.causal_features(next_obs) if causal_wm_enabled else None
            event_index = event_to_index.get(event) if causal_wm_enabled else None

            predicted, _, _ = agent.world_model.predict_tensor(latent_state, action_index)
            pred_err = float(compute_prediction_error_tensor(predicted, next_latent).item())
            reward_int = agent.curiosity.compute(pred_err)
            reward_learning = _learning_reward(reward_ext, observation, info, config)

            experience = make_experience(
                episode_id=f"micro_fouloide_{ep:06d}",
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
                event_amount=amount,
                event_index=event_index,
                causal_features=causal_features,
                next_causal_features=next_causal_features,
            )
            experience["reward_learning"] = reward_learning
            buffer.add(experience)
            agent.memory.store_if_important(experience)

            ep_reward += reward_ext
            ep_steps = step + 1
            drive_scores.append(_drive_regulation(info))
            min_health = min(min_health, float(info.get("health", 0.0)))
            mean_energy += float(info.get("energy", 0.0))
            mean_hydration += float(info.get("hydration", 0.0))
            mean_temp_error += abs(float(info.get("temperature", 0.5)) - 0.5)
            observation = next_obs
            latent_state = next_latent
            if done:
                break

        denom = max(ep_steps, 1)
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
            "world_model_uncertainty_loss": float(last_wm_uncertainty_loss),
            "value_loss": float(last_value_loss),
            "memory_items": len(agent.memory),
            "drive_regulation": float(np.mean(drive_scores)) if drive_scores else 0.0,
            "min_health": float(min_health),
            "mean_energy": float(mean_energy / denom),
            "mean_hydration": float(mean_hydration / denom),
            "temperature_error": float(mean_temp_error / denom),
            "interact_food": int(event_counts.get("interact_food", 0)),
            "interact_water": int(event_counts.get("interact_water", 0)),
            "damage": int(event_counts.get("damage", 0)),
            "health_loss": int(event_counts.get("health_loss", 0)),
        }
        metrics_history.append(episode_metrics)

        if ep % train_every == 0 and len(buffer) >= q_batch:
            wm_losses = train_world_model(
                agent.world_model, buffer, wm_optimizer,
                batch_size=wm_batch, num_updates=updates_per_train,
                sampler=wm_sampler,
                causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
                causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
                event_class_balance=bool(cwm.get("event_class_balance", False)),
                event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
                reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
                reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
                event_sample_names=set(cwm.get("event_sample_names", [])),
                event_sample_name_weight=float(cwm.get("event_sample_name_weight", 0.0)),
                event_sample_done_weight=float(cwm.get("event_sample_done_weight", 0.0)),
                event_sample_reward_abs_weight=float(cwm.get("event_sample_reward_abs_weight", 0.0)),
                uncertainty_weight=float(wmc.get("uncertainty_loss_weight", 0.0)),
                uncertainty_detach=bool(wmc.get("uncertainty_detach", False)),
            )
            last_wm_loss = wm_losses["total"]
            if wm_uncertainty_head_updates > 0:
                uncertainty_losses = train_world_model_uncertainty_head(
                    agent.world_model,
                    buffer,
                    wm_uncertainty_optimizer,
                    batch_size=wm_uncertainty_head_batch,
                    num_updates=wm_uncertainty_head_updates,
                    sampler=wm_uncertainty_head_sampler,
                    causal_feature_weight=float(cwm.get("feature_loss_weight", 0.0)),
                    causal_event_weight=float(cwm.get("event_loss_weight", 0.0)),
                    event_class_balance=bool(cwm.get("event_class_balance", False)),
                    event_class_balance_power=float(cwm.get("event_class_balance_power", 0.5)),
                    reward_abs_weight=float(wmc.get("reward_abs_weight", 0.0)),
                    reward_done_weight=float(wmc.get("reward_done_weight", 0.0)),
                )
                last_wm_uncertainty_loss = uncertainty_losses["uncertainty"]
            q_losses = train_dqn(
                train_q_network, target_network, buffer, q_optimizer,
                batch_size=q_batch, gamma=gamma,
                curiosity_weight=float(dc.get("curiosity_weight", 0.0)),
                double_dqn=bool(dc.get("double_dqn", True)),
                num_updates=updates_per_train,
                sampler=str(dc.get("sampler", "uniform")),
                reward_key=dqn_reward_key,
            )
            last_td_loss = q_losses["td_loss"]
            total_q_updates += int(q_losses["updates"])
            if agent.value_model is not None and target_value_model is not None and value_optimizer is not None:
                value_losses = train_value_model(
                    agent.value_model, target_value_model, buffer, value_optimizer,
                    batch_size=value_batch,
                    gamma=float(vc.get("gamma", gamma)),
                    num_updates=value_updates_per_train,
                    sampler=str(vc.get("sampler", "uniform")),
                    reward_key=value_reward_key,
                    target_abs_weight=float(vc.get("target_abs_weight", 0.0)),
                    terminal_weight=float(vc.get("terminal_weight", 0.0)),
                    td_error_weight=float(vc.get("td_error_weight", 0.0)),
                    max_weight=float(vc.get("max_weight", 10.0)),
                )
                last_value_loss = value_losses["value_loss"]
                total_value_updates += int(value_losses["updates"])
                if value_dyna_enabled:
                    dyna_losses = train_value_model_dyna(
                        agent.value_model, target_value_model, agent.world_model,
                        buffer, value_optimizer,
                        batch_size=int(value_dyna.get("batch_size", value_batch)),
                        gamma=float(value_dyna.get("gamma", vc.get("gamma", gamma))),
                        num_updates=int(value_dyna.get("updates_per_train", 1)),
                        sampler=str(value_dyna.get("sampler", vc.get("sampler", "uniform"))),
                        loss_weight=float(value_dyna.get("loss_weight", 1.0)),
                    )
                    last_value_dyna_loss = dyna_losses["value_dyna_loss"]
                    total_value_updates += int(dyna_losses["updates"])
            if latent_q_network is not None and target_latent_q_network is not None and latent_q_optimizer is not None:
                latent_losses = train_latent_dqn(
                    latent_q_network, target_latent_q_network, buffer, latent_q_optimizer,
                    batch_size=int(lpc.get("batch_size", q_batch)),
                    gamma=float(lpc.get("gamma", gamma)),
                    num_updates=int(lpc.get("updates_per_train", updates_per_train)),
                    sampler=str(lpc.get("sampler", "uniform")),
                    reward_key=str(lpc.get("reward_key", dqn_reward_key)),
                    double_dqn=bool(lpc.get("double_dqn", True)),
                    teacher_q_network=train_q_network,
                    distill_weight=float(lpc.get("distill_from_q_weight", 0.0)),
                    distill_mode=str(lpc.get("distill_mode", "value")),
                )
                last_latent_q_loss = latent_losses["latent_td_loss"]
                total_latent_q_updates += int(latent_losses["updates"])
                if latent_policy_dyna_enabled:
                    latent_dyna_losses = train_latent_dqn_dyna(
                        latent_q_network, target_latent_q_network, agent.world_model,
                        buffer, latent_q_optimizer,
                        num_actions=len(agent.actions),
                        batch_size=int(latent_policy_dyna.get("batch_size", lpc.get("batch_size", q_batch))),
                        gamma=float(latent_policy_dyna.get("gamma", lpc.get("gamma", gamma))),
                        num_updates=int(latent_policy_dyna.get("updates_per_train", 1)),
                        sampler=str(latent_policy_dyna.get("sampler", lpc.get("sampler", "uniform"))),
                        loss_weight=float(latent_policy_dyna.get("loss_weight", 1.0)),
                    )
                    last_latent_dyna_loss = latent_dyna_losses["latent_dyna_loss"]
                    total_latent_q_updates += int(latent_dyna_losses["updates"])
            if agent.q_network is not train_q_network:
                agent.q_network.load_state_dict(train_q_network.state_dict())
                agent.q_network.eval()
            if total_q_updates >= next_target_sync:
                sync_target(train_q_network, target_network)
                next_target_sync += target_update
            if (
                agent.value_model is not None
                and target_value_model is not None
                and total_value_updates >= next_value_target_sync
            ):
                sync_value_target(agent.value_model, target_value_model)
                next_value_target_sync += value_target_update
            if (
                latent_q_network is not None
                and target_latent_q_network is not None
                and total_latent_q_updates >= next_latent_target_sync
            ):
                sync_latent_target(latent_q_network, target_latent_q_network)
                next_latent_target_sync += int(lpc.get("target_update", target_update))

        if (
            best_checkpoint_enabled
            and len(metrics_history) >= best_checkpoint_window
            and ep >= best_checkpoint_min_episode
            and mean_life > best_mean_life + best_checkpoint_min_delta
        ):
            best_mean_life = mean_life
            if agent.q_network is not train_q_network:
                agent.q_network.load_state_dict(train_q_network.state_dict())
            save_checkpoint(
                str(best_checkpoint_path), agent, wm_optimizer, buffer,
                metrics={
                    "episode": ep,
                    "mean_lifespan": mean_life,
                    "best_mean_lifespan": best_mean_life,
                    "best_checkpoint_window": best_checkpoint_window,
                    "device": str(device),
                    "inference_device": str(inference_device),
                    "total_q_updates": total_q_updates,
                    "next_target_sync": next_target_sync,
                    "total_value_updates": total_value_updates,
                    "next_value_target_sync": next_value_target_sync,
                    "total_latent_q_updates": total_latent_q_updates,
                    "next_latent_target_sync": next_latent_target_sync,
                },
                config=config, q_optimizer=q_optimizer, target_network=target_network,
                value_optimizer=value_optimizer, target_value_model=target_value_model,
                latent_q_network=latent_q_network,
                latent_q_optimizer=latent_q_optimizer,
                target_latent_q_network=target_latent_q_network,
            )

        if ep % checkpoint_every == 0 and ep > 0:
            if agent.q_network is not train_q_network:
                agent.q_network.load_state_dict(train_q_network.state_dict())
            save_checkpoint(
                str(out_dir / f"checkpoint_{ep}.pt"), agent, wm_optimizer, buffer,
                metrics={
                    "episode": ep,
                    "mean_lifespan": mean_life,
                    "device": str(device),
                    "inference_device": str(inference_device),
                    "total_q_updates": total_q_updates,
                    "next_target_sync": next_target_sync,
                    "total_value_updates": total_value_updates,
                    "next_value_target_sync": next_value_target_sync,
                    "total_latent_q_updates": total_latent_q_updates,
                    "next_latent_target_sync": next_latent_target_sync,
                },
                config=config, q_optimizer=q_optimizer, target_network=target_network,
                value_optimizer=value_optimizer, target_value_model=target_value_model,
                latent_q_network=latent_q_network,
                latent_q_optimizer=latent_q_optimizer,
                target_latent_q_network=target_latent_q_network,
            )

        if ep % max(1, episodes // 20) == 0 or ep == episodes - 1:
            print(
                f"  ep {ep:5d} | lifespan({best_checkpoint_window})={mean_life:5.1f} "
                f"drive100={_mean_recent(metrics_history, 'drive_regulation'):.3f} "
                f"food100={_mean_recent(metrics_history, 'interact_food'):.2f} "
                f"water100={_mean_recent(metrics_history, 'interact_water'):.2f} "
                f"dmg100={_mean_recent(metrics_history, 'damage'):.2f} "
                f"td={last_td_loss:.4f} wm={last_wm_loss:.4f} "
                f"{f'wu={last_wm_uncertainty_loss:.4f} ' if wm_uncertainty_head_updates > 0 else ''}"
                f"v={last_value_loss:.4f} "
                f"{f'vd={last_value_dyna_loss:.4f} ' if value_dyna_enabled else ''}"
                f"{f'lq={last_latent_q_loss:.4f} ' if latent_policy_enabled else ''}"
                f"{f'lqd={last_latent_dyna_loss:.4f} ' if latent_policy_dyna_enabled else ''}"
                f"eps={agent.policy.epsilon:.2f} mem={len(agent.memory)}"
            )
            _save_metrics(metrics_path, {
                "config_path": args.config,
                "device": str(device),
                "inference_device": str(inference_device),
                "episodes": metrics_history,
            })

    if agent.q_network is not train_q_network:
        agent.q_network.load_state_dict(train_q_network.state_dict())
    final_metrics = {
        "episode": episodes - 1,
        "mean_lifespan": float(np.mean(recent_lifespan)),
        "device": str(device),
        "inference_device": str(inference_device),
        "total_q_updates": total_q_updates,
        "next_target_sync": next_target_sync,
        "total_value_updates": total_value_updates,
        "next_value_target_sync": next_value_target_sync,
        "total_latent_q_updates": total_latent_q_updates,
        "next_latent_target_sync": next_latent_target_sync,
    }
    save_checkpoint(
        str(out_dir / "checkpoint_final.pt"), agent, wm_optimizer, buffer,
        metrics=final_metrics,
        config=config, q_optimizer=q_optimizer, target_network=target_network,
        value_optimizer=value_optimizer, target_value_model=target_value_model,
        latent_q_network=latent_q_network,
        latent_q_optimizer=latent_q_optimizer,
        target_latent_q_network=target_latent_q_network,
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

    final_calibration_metrics = _run_final_calibration(agent, buffer, config, device)
    if final_calibration_metrics:
        if target_value_model is not None and agent.value_model is not None:
            sync_value_target(agent.value_model, target_value_model)
        calibrated_metrics = dict(final_metrics)
        calibrated_metrics["final_calibration"] = final_calibration_metrics
        calibrated_name = str(
            config.get("final_calibration", {}).get(
                "output_name",
                "checkpoint_final_calibrated.pt",
            )
        )
        calibrated_path = out_dir / calibrated_name
        save_checkpoint(
            str(calibrated_path), agent, wm_optimizer, buffer,
            metrics=calibrated_metrics,
            config=config, q_optimizer=q_optimizer, target_network=target_network,
            value_optimizer=value_optimizer, target_value_model=target_value_model,
            latent_q_network=latent_q_network,
            latent_q_optimizer=latent_q_optimizer,
            target_latent_q_network=target_latent_q_network,
        )
        _save_metrics(metrics_path, {
            "config_path": args.config,
            "device": str(device),
            "inference_device": str(inference_device),
            "episodes": metrics_history,
            "final_calibration": final_calibration_metrics,
        })
        print(f"Final calibration saved to {calibrated_path}")

        if best_checkpoint_enabled and best_checkpoint_path.exists():
            best_info = load_checkpoint(
                str(best_checkpoint_path), agent, wm_optimizer, buffer,
                q_optimizer=q_optimizer, target_network=target_network,
                value_optimizer=value_optimizer, target_value_model=target_value_model,
                latent_q_network=latent_q_network,
                latent_q_optimizer=latent_q_optimizer,
                target_latent_q_network=target_latent_q_network,
            )
            agent.encoder.to(inference_device)
            agent.world_model.to(inference_device)
            if agent.value_model is not None:
                agent.value_model.to(device)
            if agent.q_network is not train_q_network:
                agent.q_network.to(inference_device)
                agent.q_network.eval()
            best_calibration_metrics = _run_final_calibration(agent, buffer, config, device)
            if target_value_model is not None and agent.value_model is not None:
                sync_value_target(agent.value_model, target_value_model)
            best_calibrated_metrics = dict(best_info.get("metrics", {}))
            best_calibrated_metrics["final_calibration"] = best_calibration_metrics
            best_calibrated_metrics["calibrated_from"] = str(best_checkpoint_path)
            best_calibrated_path = out_dir / "checkpoint_best_calibrated.pt"
            save_checkpoint(
                str(best_calibrated_path), agent, wm_optimizer, buffer,
                metrics=best_calibrated_metrics,
                config=config, q_optimizer=q_optimizer, target_network=target_network,
                value_optimizer=value_optimizer, target_value_model=target_value_model,
                latent_q_network=latent_q_network,
                latent_q_optimizer=latent_q_optimizer,
                target_latent_q_network=target_latent_q_network,
            )
            _save_metrics(metrics_path, {
                "config_path": args.config,
                "device": str(device),
                "inference_device": str(inference_device),
                "episodes": metrics_history,
                "final_calibration": final_calibration_metrics,
                "best_final_calibration": best_calibration_metrics,
                "best_checkpoint": str(best_checkpoint_path),
                "best_mean_lifespan": best_calibrated_metrics.get("best_mean_lifespan"),
            })
            print(f"Best final calibration saved to {best_calibrated_path}")


if __name__ == "__main__":
    main()
