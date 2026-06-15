"""Evaluate a trained MicroFouloide agent."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_micro_fouloide import build_agent, build_env
from seedmind.agent.latent_q_network import LatentQNetwork
from seedmind.envs.micro_fouloide_world import ACTIONS
from seedmind.training.device import resolve_device


def _chunks(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def _parse_float_sweep(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_sweep(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(item <= 0 for item in parsed):
        raise ValueError(f"Sweep values must be positive integers, got {value!r}.")
    return parsed


def _validate_quantile(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {value}.")
    return float(value)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank01(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    denom = max(len(values) - 1, 1)
    return ranks / float(denom)


def _top_capture_score(priority: np.ndarray, target: np.ndarray, fraction: float = 0.2) -> float:
    if len(priority) == 0 or len(target) == 0:
        return float("nan")
    k = max(1, int(round(len(priority) * fraction)))
    high_priority = set(np.argsort(priority)[-k:].tolist())
    high_target = set(np.argsort(target)[-k:].tolist())
    return len(high_priority & high_target) / float(k)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_uncertainty_threshold_from_replay(
    config: dict,
    checkpoint: str,
    device: torch.device,
    quantile: float,
    max_samples: int = 20000,
) -> float:
    """Resolve a planner gate threshold from replay-buffer WM uncertainty."""
    quantile = _validate_quantile(quantile)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    buffer_rows = ckpt.get("buffer", {}).get("data", [])
    rows = [
        row for row in buffer_rows
        if row.get("latent_state") is not None and row.get("action_index") is not None
    ]
    if not rows:
        raise ValueError("Checkpoint replay buffer has no latent/action rows for uncertainty quantile.")
    if max_samples > 0 and len(rows) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(rows)), size=max_samples, replace=False)
        rows = [rows[int(i)] for i in indices]

    agent = build_agent(config, seed=0)
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.world_model.to(device)
    latents = np.stack([
        np.asarray(row["latent_state"], dtype=np.float32)
        for row in rows
    ])
    actions = np.asarray([int(row["action_index"]) for row in rows], dtype=np.int64)
    uncertainties = []
    for start in range(0, len(rows), 4096):
        _, _, uncertainty = agent.world_model.predict_batch(
            latents[start:start + 4096],
            actions[start:start + 4096],
        )
        uncertainties.append(uncertainty)
    all_uncertainty = np.concatenate(uncertainties)
    return float(np.quantile(all_uncertainty, quantile))


def _drive_regulation(info: Dict) -> float:
    drives = info.get("drives", {})
    values = [
        float(drives.get("energy", 0.0)),
        float(drives.get("hydration", 0.0)),
        1.0 - abs(float(drives.get("temperature", 0.5)) - 0.5) * 2.0,
        float(drives.get("health", 0.0)),
    ]
    return float(np.mean([max(0.0, min(1.0, v)) for v in values]))


def _record(counter: Counter, info: Dict) -> None:
    event = str(info.get("event", "unknown"))
    counter[event] += 1


def _interaction_drive_stats(values: list[float], prefix: str) -> Dict[str, float]:
    """Distribution of a drive level at interaction time (farming diagnostic)."""
    if not values:
        return {f"{prefix}_count": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p_le_020": float(np.mean(arr <= 0.20)),
        f"{prefix}_p_le_035": float(np.mean(arr <= 0.35)),
        f"{prefix}_p_le_050": float(np.mean(arr <= 0.50)),
    }


def _summarise(rows: list[Dict], num_episodes: int) -> Dict[str, float]:
    total = Counter()
    actions = Counter()
    water_hydrations: list[float] = []
    food_energies: list[float] = []
    for row in rows:
        total.update(row["events"])
        actions.update(row.get("actions", Counter()))
        water_hydrations.extend(row.get("water_hydrations", []))
        food_energies.extend(row.get("food_energies", []))
    denom = max(num_episodes, 1)
    lifespans = [row["lifespan"] for row in rows]
    total_actions = max(sum(actions.values()), 1)
    return {
        "mean_lifespan": float(np.mean(lifespans)),
        "std_lifespan": float(np.std(lifespans)),
        "max_lifespan": float(np.max(lifespans)),
        "drive_regulation": float(np.mean([row["drive_regulation"] for row in rows])),
        "mean_energy": float(np.mean([row["mean_energy"] for row in rows])),
        "mean_hydration": float(np.mean([row["mean_hydration"] for row in rows])),
        "temperature_error": float(np.mean([row["temperature_error"] for row in rows])),
        "min_health": float(np.mean([row["min_health"] for row in rows])),
        "final_energy": float(np.mean([row["final_energy"] for row in rows])),
        "final_hydration": float(np.mean([row["final_hydration"] for row in rows])),
        "final_temperature": float(np.mean([row["final_temperature"] for row in rows])),
        "final_health": float(np.mean([row["final_health"] for row in rows])),
        "interact_food": float(total.get("interact_food", 0)) / denom,
        "interact_water": float(total.get("interact_water", 0)) / denom,
        "damage": float(total.get("damage", 0)) / denom,
        "health_loss": float(total.get("health_loss", 0)) / denom,
        "action_move": sum(actions.get(a, 0) for a in ACTIONS if a.startswith("MOVE_")) / total_actions,
        "action_interact": actions.get("INTERACT", 0) / total_actions,
        "action_rest": actions.get("REST", 0) / total_actions,
        "action_wait": actions.get("WAIT", 0) / total_actions,
        "planner_used": sum(row.get("planner_used", 0) for row in rows) / total_actions,
        **_interaction_drive_stats(water_hydrations, "hydration_at_water"),
        **_interaction_drive_stats(food_energies, "energy_at_food"),
    }


def run_naive(config: dict, num_episodes: int, seed: int = 9999) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(num_episodes):
        env = build_env(config, seed + ep)
        env.reset()
        done = False
        steps = 0
        events = Counter()
        actions = Counter()
        drives = []
        energy = hydration = temp_err = 0.0
        min_health = 1.0
        while not done:
            action = ACTIONS[int(rng.integers(len(ACTIONS)))]
            actions[action] += 1
            _, _, done, info = env.step(action)
            _record(events, info)
            drives.append(_drive_regulation(info))
            energy += float(info.get("energy", 0.0))
            hydration += float(info.get("hydration", 0.0))
            temp_err += abs(float(info.get("temperature", 0.5)) - 0.5)
            min_health = min(min_health, float(info.get("health", 0.0)))
            steps += 1
        denom = max(steps, 1)
        rows.append({
            "lifespan": steps,
            "drive_regulation": float(np.mean(drives)) if drives else 0.0,
            "mean_energy": energy / denom,
            "mean_hydration": hydration / denom,
            "temperature_error": temp_err / denom,
            "min_health": min_health,
            "final_energy": float(info.get("energy", 0.0)),
            "final_hydration": float(info.get("hydration", 0.0)),
            "final_temperature": float(info.get("temperature", 0.5)),
            "final_health": float(info.get("health", 0.0)),
            "events": events,
            "actions": actions,
        })
    return _summarise(rows, num_episodes)


def run_trained(
    config: dict,
    checkpoint: str,
    num_episodes: int,
    device: torch.device,
    decision_mode: str = "q",
    planning_weight: float = 0.1,
    planner_horizon: int = 3,
    planner_samples: int = 8,
    terminal_value_weight: float | None = None,
    planner_uncertainty_threshold: float | None = None,
    planner_margin_threshold: float = 0.0,
    planner_q_advantage_threshold: float = 0.0,
    seed: int = 9999,
) -> Dict[str, float]:
    if decision_mode == "planner":
        config = dict(config)
        planning = dict(config.get("planning", {}))
        planning.update({
            "enabled": True,
            "weight": planning_weight,
            "horizon": planner_horizon,
            "num_samples": planner_samples,
            "uncertainty_threshold": planner_uncertainty_threshold,
            "margin_threshold": planner_margin_threshold,
            "q_advantage_threshold": planner_q_advantage_threshold,
        })
        if terminal_value_weight is not None:
            planning["terminal_value_weight"] = terminal_value_weight
        config["planning"] = planning
    agent = build_agent(config, seed=0)
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    if agent.value_model is not None:
        agent.value_model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.encoder.load_state_dict(ckpt["encoder_state"])
    if agent.value_model is not None and "value_model_state" in ckpt:
        agent.value_model.load_state_dict(ckpt["value_model_state"])
    latent_q_network = None
    if decision_mode == "latent_q":
        lpc = config.get("latent_policy", {})
        latent_q_network = LatentQNetwork(
            latent_dim=int(config.get("agent", {}).get("latent_dim", 64)),
            num_actions=len(ACTIONS),
            hidden_dim=int(lpc.get("hidden_dim", 128)),
            num_layers=int(lpc.get("num_layers", 2)),
        ).to(device)
        latent_q_network.load_state_dict(ckpt["latent_q_network_state"])
        latent_q_network.eval()
    agent.policy.epsilon_start = 0.0
    agent.policy.epsilon_end = 0.0

    rows = []
    for ep in range(num_episodes):
        env = build_env(config, seed + ep)
        obs = env.reset()
        done = False
        steps = 0
        events = Counter()
        actions = Counter()
        planner_used = 0
        drives = []
        energy = hydration = temp_err = 0.0
        min_health = 1.0
        water_hydrations: list[float] = []
        food_energies: list[float] = []
        latent = agent.encode(obs) if decision_mode in {"planner", "latent_q"} else None
        while not done:
            pre_hydration = float(obs.get("hydration", 0.0))
            pre_energy = float(obs.get("energy", 0.0))
            if decision_mode == "planner":
                memories = agent.retrieve(latent)
                goal = agent.choose_goal(latent, memories)
                action = agent.choose_action(
                    latent, goal, memories, env.available_actions(), observation=obs,
                )
                planner_used += int(getattr(agent, "last_planner_used", False))
            elif decision_mode == "latent_q":
                values = latent_q_network.q_values(np.asarray(latent, dtype=np.float32))
                action = max(env.available_actions(), key=lambda a: float(values[ACTIONS.index(a)]))
            else:
                scorer = agent.q_network.make_scorer(obs, env.available_actions())
                action = max(env.available_actions(), key=scorer)
            actions[action] += 1
            obs, _, done, info = env.step(action)
            if decision_mode in {"planner", "latent_q"}:
                latent = agent.encode(obs)
            _record(events, info)
            event = str(info.get("event", ""))
            if event == "interact_water":
                water_hydrations.append(pre_hydration)
            elif event == "interact_food":
                food_energies.append(pre_energy)
            drives.append(_drive_regulation(info))
            energy += float(info.get("energy", 0.0))
            hydration += float(info.get("hydration", 0.0))
            temp_err += abs(float(info.get("temperature", 0.5)) - 0.5)
            min_health = min(min_health, float(info.get("health", 0.0)))
            steps += 1
        denom = max(steps, 1)
        rows.append({
            "lifespan": steps,
            "drive_regulation": float(np.mean(drives)) if drives else 0.0,
            "mean_energy": energy / denom,
            "mean_hydration": hydration / denom,
            "temperature_error": temp_err / denom,
            "min_health": min_health,
            "final_energy": float(info.get("energy", 0.0)),
            "final_hydration": float(info.get("hydration", 0.0)),
            "final_temperature": float(info.get("temperature", 0.5)),
            "final_health": float(info.get("health", 0.0)),
            "events": events,
            "actions": actions,
            "planner_used": planner_used,
            "water_hydrations": water_hydrations,
            "food_energies": food_energies,
        })
    return _summarise(rows, num_episodes)


def print_stats(label: str, stats: Dict[str, float]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"  lifespan={stats['mean_lifespan']:.1f} +/- {stats['std_lifespan']:.1f} "
        f"(max {stats['max_lifespan']:.0f}) drive={stats['drive_regulation']:.3f}"
    )
    print(
        f"  energy={stats['mean_energy']:.3f} hydration={stats['mean_hydration']:.3f} "
        f"temp_err={stats['temperature_error']:.3f} min_health={stats['min_health']:.3f}"
    )
    print(
        f"  final_energy={stats['final_energy']:.3f} final_hydration={stats['final_hydration']:.3f} "
        f"final_temp={stats['final_temperature']:.3f} final_health={stats['final_health']:.3f}"
    )
    print(
        f"  food={stats['interact_food']:.2f} water={stats['interact_water']:.2f} "
        f"damage={stats['damage']:.2f} health_loss={stats['health_loss']:.2f}"
    )
    print(
        f"  actions: move={stats['action_move']:.1%} interact={stats['action_interact']:.1%} "
        f"rest={stats['action_rest']:.1%} wait={stats['action_wait']:.1%}"
    )
    if stats.get("planner_used", 0.0) > 0.0:
        print(f"  planner_used={stats['planner_used']:.1%}")
    for prefix, label in (("hydration_at_water", "hydration@interact_water"),
                          ("energy_at_food", "energy@interact_food")):
        count = stats.get(f"{prefix}_count")
        if count is None:
            continue
        if count == 0:
            print(f"  {label}: n=0")
            continue
        print(
            f"  {label}: n={count:.0f} mean={stats[f'{prefix}_mean']:.3f} "
            f"median={stats[f'{prefix}_median']:.3f} "
            f"<=0.20: {stats[f'{prefix}_p_le_020']:.1%} "
            f"<=0.35: {stats[f'{prefix}_p_le_035']:.1%} "
            f"<=0.50: {stats[f'{prefix}_p_le_050']:.1%}"
        )


def _causal_feature_names(config: dict) -> list[str]:
    return build_env(config, seed=0).causal_feature_names()


def _causal_event_names(config: dict) -> list[str]:
    return build_env(config, seed=0).causal_event_names()


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


def _feature_index(names: list[str], name: str) -> int | None:
    try:
        return names.index(name)
    except ValueError:
        return None


def _value_bucket_masks(rows: list[dict], feature_names: list[str]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    n = len(rows)
    masks["all"] = np.ones(n, dtype=bool)
    masks["terminal"] = np.asarray([bool(row.get("done", False)) for row in rows], dtype=bool)

    feature_matrix = None
    if rows and all(row.get("causal_features") is not None for row in rows):
        feature_matrix = np.stack([
            np.asarray(row["causal_features"], dtype=np.float32)
            for row in rows
        ])
    if feature_matrix is None:
        return masks

    checks = [
        ("low_energy", "energy", lambda x: x < 0.35),
        ("low_hydration", "hydration", lambda x: x < 0.35),
        ("low_health", "health", lambda x: x < 0.5),
        ("danger_near", "local_danger", lambda x: x > 0.0),
        ("food_signal", "local_food_signal", lambda x: x > 0.0),
        ("water_signal", "local_water_signal", lambda x: x > 0.0),
    ]
    for bucket, feature, predicate in checks:
        idx = _feature_index(feature_names, feature)
        if idx is not None and idx < feature_matrix.shape[1]:
            masks[bucket] = predicate(feature_matrix[:, idx])
    return masks


def _train_test_indices(n: int, seed: int = 12345, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.arange(n))
    split = max(1, min(n - 1, int(n * train_fraction)))
    return indices[:split], indices[split:]


def _linear_probe_regression(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    ridge: float = 1e-3,
) -> None:
    train_idx, test_idx = _train_test_indices(len(x))
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    xs_train = (x_train - x_mean) / x_std
    xs_test = (x_test - x_mean) / x_std
    xb_train = np.concatenate([xs_train, np.ones((len(xs_train), 1), dtype=np.float32)], axis=1)
    xb_test = np.concatenate([xs_test, np.ones((len(xs_test), 1), dtype=np.float32)], axis=1)

    reg = ridge * np.eye(xb_train.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(xb_train.T @ xb_train + reg, xb_train.T @ y_train)
    pred = xb_test @ weights
    baseline = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_test), axis=0)

    print("  causal feature probe:")
    print("  feature              mae    baseline  corr")
    for i, name in enumerate(feature_names):
        mae = float(np.mean(np.abs(pred[:, i] - y_test[:, i])))
        base_mae = float(np.mean(np.abs(baseline[:, i] - y_test[:, i])))
        corr = _pearson(pred[:, i], y_test[:, i])
        print(f"  {name:<18} {mae:6.4f}  {base_mae:8.4f}  {corr:6.3f}")


def _linear_probe_classification(
    x: np.ndarray,
    labels: np.ndarray,
    name: str,
    num_classes: int,
    device: torch.device,
    steps: int = 120,
) -> None:
    train_idx, test_idx = _train_test_indices(len(x), seed=23456)
    x_train = x[train_idx]
    y_train = labels[train_idx]
    x_test = x[test_idx]
    y_test = labels[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_t = torch.as_tensor((x_train - x_mean) / x_std, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.as_tensor((x_test - x_mean) / x_std, dtype=torch.float32, device=device)

    model = torch.nn.Linear(x.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        logits = model(x_train_t)
        loss = torch.nn.functional.cross_entropy(logits, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test_t).argmax(dim=1).cpu().numpy()
    acc = float(np.mean(pred == y_test))
    majority = int(np.bincount(y_train, minlength=num_classes).argmax())
    majority_acc = float(np.mean(y_test == majority))
    recalls = []
    support = np.bincount(y_test, minlength=num_classes)
    for class_idx in range(num_classes):
        if support[class_idx] == 0:
            continue
        class_mask = y_test == class_idx
        recalls.append(float(np.mean(pred[class_mask] == class_idx)))
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0
    pred_counts = np.bincount(pred, minlength=num_classes)
    top_pred = sorted(
        ((count, class_idx) for class_idx, count in enumerate(pred_counts) if count > 0),
        reverse=True,
    )[:3]
    top_pred_text = ",".join(
        f"{class_idx}:{count / max(len(pred), 1):.2f}"
        for count, class_idx in top_pred
    )
    print(
        f"  {name}: acc={acc:.3f} baseline_majority={majority_acc:.3f} "
        f"balanced_acc={balanced_acc:.3f} gain={acc - majority_acc:+.3f} "
        f"classes={num_classes} top_pred={top_pred_text}"
    )


def diagnose_latent_representation(
    config: dict,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 20000,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    rows = [
        row for row in buffer
        if row.get("latent_state") is not None
    ]
    if max_samples > 0 and len(rows) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(rows)), size=max_samples, replace=False)
        rows = [rows[int(i)] for i in indices]
    if len(rows) < 10:
        print("\n=== Latent representation diagnostic ===")
        print("  Not enough latent replay rows.")
        return

    latents = np.stack([
        np.asarray(row["latent_state"], dtype=np.float32)
        for row in rows
    ])
    feature_names = _causal_feature_names(config)
    feature_rows = [
        row for row in rows
        if row.get("causal_features") is not None
    ]
    print("\n=== Latent representation diagnostic ===")
    print(f"  samples={len(rows)} checkpoint={checkpoint}")
    if len(feature_rows) >= 10:
        feature_latents = np.stack([
            np.asarray(row["latent_state"], dtype=np.float32)
            for row in feature_rows
        ])
        features = np.stack([
            np.asarray(row["causal_features"], dtype=np.float32)
            for row in feature_rows
        ])
        _linear_probe_regression(feature_latents, features, feature_names)
    else:
        print("  causal feature probe: unavailable")

    standing_rows = [
        row for row in rows
        if row.get("obs_state") is not None and "standing_entity" in row["obs_state"]
    ]
    if len(standing_rows) >= 10:
        standing_latents = np.stack([
            np.asarray(row["latent_state"], dtype=np.float32)
            for row in standing_rows
        ])
        labels = np.asarray([
            int(row["obs_state"]["standing_entity"])
            for row in standing_rows
        ], dtype=np.int64)
        num_classes = int(max(labels.max() + 1, 1))
        _linear_probe_classification(
            standing_latents, labels, "standing_entity probe",
            num_classes=num_classes, device=device,
        )
    else:
        print("  standing_entity probe: unavailable")


def diagnose_latent_q_alignment(
    config: dict,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 20000,
) -> None:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    rows = [
        row for row in buffer
        if row.get("latent_state") is not None and row.get("obs_state") is not None
    ]
    if max_samples > 0 and len(rows) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(rows)), size=max_samples, replace=False)
        rows = [rows[int(i)] for i in indices]

    print("\n=== LatentQ alignment diagnostic ===")
    if len(rows) < 10:
        print("  Not enough replay rows.")
        return
    if "latent_q_network_state" not in ckpt:
        print("  Checkpoint has no LatentQ state.")
        return

    agent = build_agent(config, seed=0)
    agent.encoder.to(device)
    agent.q_network.to(device)
    agent.encoder.load_state_dict(ckpt["encoder_state"])
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    agent.encoder.eval()
    agent.q_network.eval()

    lpc = config.get("latent_policy", {})
    latent_q_network = LatentQNetwork(
        latent_dim=int(config.get("agent", {}).get("latent_dim", 64)),
        num_actions=len(ACTIONS),
        hidden_dim=int(lpc.get("hidden_dim", 128)),
        num_layers=int(lpc.get("num_layers", 2)),
    ).to(device)
    latent_q_network.load_state_dict(ckpt["latent_q_network_state"])
    latent_q_network.eval()

    q_best = []
    latent_best = []
    taken = []
    q_margins = []
    latent_margins = []
    for row in rows:
        obs = row["obs_state"]
        latent = np.asarray(row["latent_state"], dtype=np.float32)
        scorer = agent.q_network.make_scorer(obs, ACTIONS)
        q_values = np.asarray([float(scorer(action)) for action in ACTIONS], dtype=np.float32)
        lq_values = latent_q_network.q_values(latent)
        q_order = np.argsort(q_values)
        lq_order = np.argsort(lq_values)
        q_best.append(int(q_order[-1]))
        latent_best.append(int(lq_order[-1]))
        taken.append(int(row.get("action_index", -1)))
        q_margins.append(float(q_values[q_order[-1]] - q_values[q_order[-2]]))
        latent_margins.append(float(lq_values[lq_order[-1]] - lq_values[lq_order[-2]]))

    q_best_a = np.asarray(q_best, dtype=np.int64)
    latent_best_a = np.asarray(latent_best, dtype=np.int64)
    taken_a = np.asarray(taken, dtype=np.int64)
    valid_taken = taken_a >= 0
    agreement = float(np.mean(q_best_a == latent_best_a))
    q_taken = float(np.mean(q_best_a[valid_taken] == taken_a[valid_taken])) if np.any(valid_taken) else 0.0
    latent_taken = float(np.mean(latent_best_a[valid_taken] == taken_a[valid_taken])) if np.any(valid_taken) else 0.0

    def _action_distribution(indices: np.ndarray) -> str:
        counts = np.bincount(indices, minlength=len(ACTIONS))
        total = max(int(counts.sum()), 1)
        parts = [
            f"{ACTIONS[i]}:{counts[i] / total:.2f}"
            for i in np.argsort(counts)[::-1][:4]
            if counts[i] > 0
        ]
        return " ".join(parts)

    print(f"  samples={len(rows)} checkpoint={checkpoint}")
    print(
        f"  best_action_agreement={agreement:.3f} "
        f"q_matches_replay={q_taken:.3f} latent_matches_replay={latent_taken:.3f}"
    )
    print(
        f"  q_margin={np.mean(q_margins):.4f} "
        f"latent_margin={np.mean(latent_margins):.4f}"
    )
    print(f"  q_best:      {_action_distribution(q_best_a)}")
    print(f"  latent_best: {_action_distribution(latent_best_a)}")
    if np.any(valid_taken):
        print(f"  replay:      {_action_distribution(taken_a[valid_taken])}")


def diagnose_value_model(
    config: dict,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 20000,
    batch_size: int = 512,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    valid = [
        e for e in buffer
        if e.get("latent_state") is not None
    ]
    if not valid:
        print("\n=== ValueModel diagnostic ===")
        print("  No latent replay rows found.")
        return

    vc = config.get("value_model", {})
    reward_key = str(vc.get("reward_key", config.get("dqn", {}).get("reward_key", "reward_external")))
    gamma = float(vc.get("gamma", config.get("dqn", {}).get("gamma", 0.97)))
    returns_by_idx = _discounted_returns_by_episode(valid, reward_key, gamma)
    rows = [row for idx, row in enumerate(valid) if idx in returns_by_idx]
    returns = np.asarray([returns_by_idx[idx] for idx, _ in enumerate(valid) if idx in returns_by_idx], dtype=np.float32)

    if max_samples > 0 and len(rows) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(rows)), size=max_samples, replace=False)
        rows = [rows[int(i)] for i in indices]
        returns = returns[indices]

    agent = build_agent(config, seed=0)
    if agent.value_model is None or "value_model_state" not in ckpt:
        print("\n=== ValueModel diagnostic ===")
        print("  Checkpoint/config has no ValueModel.")
        return
    agent.value_model.to(device)
    agent.value_model.load_state_dict(ckpt["value_model_state"])
    agent.value_model.eval()

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _chunks(rows, batch_size):
            latents = torch.as_tensor(
                np.stack([np.asarray(row["latent_state"], dtype=np.float32) for row in batch]),
                dtype=torch.float32,
                device=device,
            )
            preds.append(agent.value_model(latents).cpu().numpy())
    values = np.concatenate(preds).astype(np.float32)
    errors = values - returns

    feature_names = _causal_feature_names(config)
    masks = _value_bucket_masks(rows, feature_names)

    print("\n=== ValueModel diagnostic on replay returns ===")
    print(
        f"  samples={len(rows)} reward_key={reward_key} gamma={gamma:.3f} "
        f"checkpoint={checkpoint}"
    )
    print("  bucket              n   value_mean  return_mean  mae     bias    corr")
    priority = [
        "all", "low_energy", "low_hydration", "low_health", "danger_near",
        "food_signal", "water_signal", "terminal",
    ]
    ordered = [bucket for bucket in priority if bucket in masks]
    ordered += sorted(bucket for bucket in masks if bucket not in set(ordered))
    for bucket in ordered:
        mask = masks[bucket]
        n = int(mask.sum())
        if n == 0:
            continue
        bucket_values = values[mask]
        bucket_returns = returns[mask]
        bucket_errors = errors[mask]
        print(
            f"  {bucket:<17} {n:6d} "
            f"{float(np.mean(bucket_values)):10.4f} "
            f"{float(np.mean(bucket_returns)):11.4f} "
            f"{float(np.mean(np.abs(bucket_errors))):7.4f} "
            f"{float(np.mean(bucket_errors)):7.4f} "
            f"{_pearson(bucket_values, bucket_returns):7.3f}"
        )


def diagnose_world_model(
    config: dict,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 20000,
    batch_size: int = 256,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    valid = [
        e for e in buffer
        if e.get("latent_state") is not None
        and e.get("next_latent_state") is not None
        and e.get("action_index") is not None
    ]
    if max_samples > 0 and len(valid) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(valid)), size=max_samples, replace=False)
        valid = [valid[int(i)] for i in indices]

    agent = build_agent(config, seed=0)
    agent.world_model.to(device)
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.world_model.eval()

    event_names = _causal_event_names(config)
    feature_names = _causal_feature_names(config)
    stats = defaultdict(lambda: {
        "n": 0,
        "state_mse": 0.0,
        "reward_mae": 0.0,
        "reward_bias": 0.0,
        "feature_mae": 0.0,
        "feature_count": 0,
        "event_acc": 0.0,
        "event_count": 0,
        "uncertainty": 0.0,
    })

    with torch.no_grad():
        for batch in _chunks(valid, batch_size):
            latents = torch.as_tensor(
                np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            next_latents = torch.as_tensor(
                np.stack([np.asarray(e["next_latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                [int(e["action_index"]) for e in batch],
                dtype=torch.long,
                device=device,
            )
            rewards = torch.as_tensor(
                [float(e.get("reward_external", 0.0)) for e in batch],
                dtype=torch.float32,
                device=device,
            )
            out = agent.world_model.forward_aux(latents, actions)
            state_mse = torch.mean((out["next_state"] - next_latents) ** 2, dim=1).cpu().numpy()
            reward_err = (out["reward"] - rewards).cpu().numpy()
            uncertainty = out["uncertainty"].cpu().numpy()

            feature_err = None
            if "causal_feature_delta" in out and all(
                e.get("causal_features") is not None
                and e.get("next_causal_features") is not None
                for e in batch
            ):
                current_features = np.stack([
                    np.asarray(e["causal_features"], dtype=np.float32)
                    for e in batch
                ])
                next_features = np.stack([
                    np.asarray(e["next_causal_features"], dtype=np.float32)
                    for e in batch
                ])
                target_delta = torch.as_tensor(
                    next_features - current_features,
                    dtype=torch.float32,
                    device=device,
                )
                feature_err = torch.mean(
                    torch.abs(out["causal_feature_delta"] - target_delta),
                    dim=1,
                ).cpu().numpy()

            event_hit = None
            if "event_logits" in out and all(e.get("event_index") is not None for e in batch):
                pred_events = torch.argmax(out["event_logits"], dim=1).cpu().numpy()
                true_events = np.asarray([int(e["event_index"]) for e in batch], dtype=np.int64)
                event_hit = (pred_events == true_events).astype(np.float32)

            for idx, e in enumerate(batch):
                event = str(e.get("event") or "unknown")
                row = stats[event]
                row["n"] += 1
                row["state_mse"] += float(state_mse[idx])
                row["reward_mae"] += abs(float(reward_err[idx]))
                row["reward_bias"] += float(reward_err[idx])
                row["uncertainty"] += float(uncertainty[idx])
                if feature_err is not None:
                    row["feature_mae"] += float(feature_err[idx])
                    row["feature_count"] += 1
                if event_hit is not None:
                    row["event_acc"] += float(event_hit[idx])
                    row["event_count"] += 1

    print("\n=== Micro-Fouloide World Model diagnostic on replay buffer ===")
    print(f"  samples={len(valid)} checkpoint={checkpoint}")
    print(f"  features={','.join(feature_names)}")
    print("  event                 n   state_mse  reward_mae  reward_bias  feature_mae  event_acc  uncertainty")

    priority = [
        "interact_food", "interact_water", "damage", "health_loss", "death",
        "temperature_up", "temperature_down", "move_ok", "move_blocked",
        "rest", "wait", "interact_noop",
    ]
    ordered = [event for event in priority if event in stats]
    ordered += sorted(event for event in stats if event not in set(ordered))
    for event in ordered:
        row = stats[event]
        n = max(int(row["n"]), 1)
        feature_count = int(row["feature_count"])
        event_count = int(row["event_count"])
        feature_mae = row["feature_mae"] / feature_count if feature_count else float("nan")
        event_acc = row["event_acc"] / event_count if event_count else float("nan")
        print(
            f"  {event:<18} {n:6d} "
            f"{row['state_mse'] / n:9.5f} "
            f"{row['reward_mae'] / n:10.5f} "
            f"{row['reward_bias'] / n:11.5f} "
            f"{feature_mae:11.5f} "
            f"{event_acc:9.3f} "
            f"{row['uncertainty'] / n:11.5f}"
        )


def diagnose_world_model_calibration(
    config: dict,
    checkpoint: str,
    device: torch.device,
    max_samples: int = 20000,
    batch_size: int = 256,
) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    buffer = ckpt.get("buffer", {}).get("data", [])
    valid = [
        e for e in buffer
        if e.get("latent_state") is not None
        and e.get("next_latent_state") is not None
        and e.get("action_index") is not None
    ]
    if max_samples > 0 and len(valid) > max_samples:
        rng = np.random.default_rng(12345)
        indices = rng.choice(np.arange(len(valid)), size=max_samples, replace=False)
        valid = [valid[int(i)] for i in indices]

    print("\n=== World Model uncertainty calibration ===")
    if len(valid) < 10:
        print("  Not enough replay rows.")
        return

    agent = build_agent(config, seed=0)
    agent.world_model.to(device)
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.world_model.eval()

    state_errors: list[np.ndarray] = []
    reward_errors: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    feature_errors: list[np.ndarray] = []
    event_misses: list[np.ndarray] = []

    with torch.no_grad():
        for batch in _chunks(valid, batch_size):
            latents = torch.as_tensor(
                np.stack([np.asarray(e["latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            next_latents = torch.as_tensor(
                np.stack([np.asarray(e["next_latent_state"], dtype=np.float32) for e in batch]),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.as_tensor(
                [int(e["action_index"]) for e in batch],
                dtype=torch.long,
                device=device,
            )
            rewards = torch.as_tensor(
                [float(e.get("reward_external", 0.0)) for e in batch],
                dtype=torch.float32,
                device=device,
            )
            out = agent.world_model.forward_aux(latents, actions)
            state_errors.append(
                torch.mean((out["next_state"] - next_latents) ** 2, dim=1).cpu().numpy()
            )
            reward_errors.append(torch.abs(out["reward"] - rewards).cpu().numpy())
            uncertainties.append(out["uncertainty"].cpu().numpy())

            if "causal_feature_delta" in out and all(
                e.get("causal_features") is not None
                and e.get("next_causal_features") is not None
                for e in batch
            ):
                current_features = np.stack([
                    np.asarray(e["causal_features"], dtype=np.float32)
                    for e in batch
                ])
                next_features = np.stack([
                    np.asarray(e["next_causal_features"], dtype=np.float32)
                    for e in batch
                ])
                target_delta = torch.as_tensor(
                    next_features - current_features,
                    dtype=torch.float32,
                    device=device,
                )
                feature_errors.append(
                    torch.mean(
                        torch.abs(out["causal_feature_delta"] - target_delta),
                        dim=1,
                    ).cpu().numpy()
                )
            else:
                feature_errors.append(np.full(len(batch), np.nan, dtype=np.float32))

            if "event_logits" in out and all(e.get("event_index") is not None for e in batch):
                pred_events = torch.argmax(out["event_logits"], dim=1).cpu().numpy()
                true_events = np.asarray([int(e["event_index"]) for e in batch], dtype=np.int64)
                event_misses.append((pred_events != true_events).astype(np.float32))
            else:
                event_misses.append(np.full(len(batch), np.nan, dtype=np.float32))

    state = np.concatenate(state_errors).astype(np.float32)
    reward = np.concatenate(reward_errors).astype(np.float32)
    uncertainty = np.concatenate(uncertainties).astype(np.float32)
    feature = np.concatenate(feature_errors).astype(np.float32)
    event_miss = np.concatenate(event_misses).astype(np.float32)

    components = [_rank01(state), _rank01(reward)]
    feature_mask = np.isfinite(feature)
    if np.any(feature_mask):
        ranked = np.zeros_like(feature, dtype=np.float32)
        ranked[feature_mask] = _rank01(feature[feature_mask])
        components.append(ranked)
    event_mask = np.isfinite(event_miss)
    if np.any(event_mask):
        ranked = np.zeros_like(event_miss, dtype=np.float32)
        ranked[event_mask] = _rank01(event_miss[event_mask])
        components.append(ranked)
    composite_error = np.mean(np.stack(components, axis=0), axis=0)

    print(f"  samples={len(valid)} checkpoint={checkpoint}")
    print("  correlations with uncertainty:")
    print(f"    state_mse:       {_pearson(uncertainty, state):+.3f}")
    print(f"    reward_abs_err:  {_pearson(uncertainty, reward):+.3f}")
    if np.any(feature_mask):
        print(f"    feature_mae:     {_pearson(uncertainty[feature_mask], feature[feature_mask]):+.3f}")
    if np.any(event_mask):
        print(f"    event_miss:      {_pearson(uncertainty[event_mask], event_miss[event_mask]):+.3f}")
    print(f"    composite_rank:  {_pearson(_rank01(uncertainty), composite_error):+.3f}")
    print(
        "  top20 uncertainty captures "
        f"{_top_capture_score(uncertainty, composite_error, fraction=0.2):.1%} "
        "of top20 composite errors"
    )

    order = np.argsort(uncertainty)
    bins = np.array_split(order, 5)
    print("  uncertainty_bin      n  unc_mean  comp_err  state_mse  reward_mae  feature_mae  event_miss")
    for idx, bin_indices in enumerate(bins, start=1):
        if len(bin_indices) == 0:
            continue
        bin_feature = feature[bin_indices]
        bin_event = event_miss[bin_indices]
        feature_mean = float(np.nanmean(bin_feature)) if np.any(np.isfinite(bin_feature)) else float("nan")
        event_mean = float(np.nanmean(bin_event)) if np.any(np.isfinite(bin_event)) else float("nan")
        print(
            f"  q{idx:<1}              {len(bin_indices):5d} "
            f"{float(np.mean(uncertainty[bin_indices])):9.5f} "
            f"{float(np.mean(composite_error[bin_indices])):9.5f} "
            f"{float(np.mean(state[bin_indices])):10.5f} "
            f"{float(np.mean(reward[bin_indices])):11.5f} "
            f"{feature_mean:12.5f} "
            f"{event_mean:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/micro_fouloide_v0.yaml")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    parser.add_argument(
        "--planner-preset",
        choices=["wm-calibrated"],
        default=None,
        help="Apply a validated planner evaluation preset.",
    )
    parser.add_argument("--compare-planner", action="store_true")
    parser.add_argument("--compare-latent-q", action="store_true")
    parser.add_argument("--planning-weight", type=float, default=0.1)
    parser.add_argument("--planner-horizon", type=int, default=3)
    parser.add_argument("--planner-samples", type=int, default=8)
    parser.add_argument(
        "--planner-horizon-sweep",
        default=None,
        help="Comma-separated planner horizons to evaluate, e.g. 2,3,4,5.",
    )
    parser.add_argument(
        "--planner-samples-sweep",
        default=None,
        help="Comma-separated planner sample counts to evaluate, e.g. 4,8,16.",
    )
    parser.add_argument("--planner-uncertainty-threshold", type=float, default=None)
    parser.add_argument(
        "--planner-uncertainty-quantile",
        type=float,
        default=None,
        help="Resolve planner uncertainty threshold from checkpoint replay uncertainty quantile.",
    )
    parser.add_argument(
        "--planner-uncertainty-quantile-sweep",
        default=None,
        help="Comma-separated replay uncertainty quantiles to evaluate, e.g. 0.5,0.6,0.7.",
    )
    parser.add_argument(
        "--planner-uncertainty-quantile-samples",
        type=int,
        default=20000,
        help="Replay rows used to estimate uncertainty quantile thresholds.",
    )
    parser.add_argument("--planner-margin-threshold", type=float, default=0.0)
    parser.add_argument("--planner-q-advantage-threshold", type=float, default=0.0)
    parser.add_argument(
        "--terminal-value-weight",
        type=float,
        default=None,
        help="Override planning.terminal_value_weight during planner evaluation.",
    )
    parser.add_argument(
        "--planner-sweep",
        default=None,
        help="Comma-separated planning weights to evaluate, e.g. 0,0.05,0.1,0.2.",
    )
    parser.add_argument(
        "--terminal-value-sweep",
        default=None,
        help="Comma-separated terminal value weights to evaluate, e.g. 0,0.5,1,2.",
    )
    parser.add_argument(
        "--planner-uncertainty-sweep",
        default=None,
        help="Comma-separated planner uncertainty thresholds to evaluate, e.g. 0.55,0.6,0.65.",
    )
    parser.add_argument(
        "--planner-margin-sweep",
        default=None,
        help="Comma-separated planner best-action margin thresholds to evaluate, e.g. 0,0.01,0.02.",
    )
    parser.add_argument(
        "--planner-q-advantage-sweep",
        default=None,
        help="Comma-separated Q-vs-WM override thresholds on normalised WM advantage, e.g. 0,0.02,0.05.",
    )
    parser.add_argument("--diagnose-world-model", action="store_true")
    parser.add_argument("--diagnose-wm-calibration", action="store_true")
    parser.add_argument("--diagnose-value-model", action="store_true")
    parser.add_argument("--diagnose-latent", action="store_true")
    parser.add_argument("--diagnose-latent-q", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=20000)
    args = parser.parse_args()

    if args.planner_preset == "wm-calibrated":
        args.planning_weight = 0.25
        args.terminal_value_weight = 1.0
        args.planner_uncertainty_quantile = 0.60
        args.planner_margin_threshold = 0.01
        args.planner_q_advantage_threshold = 0.02
        args.planner_horizon = 5
        args.planner_samples = 8

    config = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    if args.diagnose_world_model:
        diagnose_world_model(
            config, args.checkpoint, device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnose_wm_calibration:
        diagnose_world_model_calibration(
            config, args.checkpoint, device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnose_value_model:
        diagnose_value_model(
            config, args.checkpoint, device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnose_latent:
        diagnose_latent_representation(
            config, args.checkpoint, device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnose_latent_q:
        diagnose_latent_q_alignment(
            config, args.checkpoint, device,
            max_samples=args.diagnostic_samples,
        )
    if args.diagnostics_only:
        return
    planner_uncertainty_threshold = args.planner_uncertainty_threshold
    if args.planner_uncertainty_quantile is not None:
        planner_uncertainty_threshold = resolve_uncertainty_threshold_from_replay(
            config,
            args.checkpoint,
            device,
            args.planner_uncertainty_quantile,
            max_samples=args.planner_uncertainty_quantile_samples,
        )
        print(
            "\nResolved planner uncertainty threshold from replay: "
            f"quantile={args.planner_uncertainty_quantile:.3f} "
            f"threshold={planner_uncertainty_threshold:.5f}"
        )
    naive = run_naive(config, args.num_episodes)
    print_stats("Naive", naive)
    trained = run_trained(config, args.checkpoint, args.num_episodes, device, decision_mode="q")
    print_stats("Trained Q-only", trained)
    print(f"\n  Ratio Q/naive lifespan: {trained['mean_lifespan'] / max(naive['mean_lifespan'], 1):.2f}x")
    if args.compare_latent_q:
        latent_q = run_trained(
            config, args.checkpoint, args.num_episodes, device,
            decision_mode="latent_q",
        )
        print_stats("Trained latent Q", latent_q)
        print(
            f"\n  Ratio latent-Q/Q lifespan: "
            f"{latent_q['mean_lifespan'] / max(trained['mean_lifespan'], 1):.2f}x"
        )
    planner_weights = _parse_float_sweep(args.planner_sweep)
    terminal_weights = _parse_float_sweep(args.terminal_value_sweep)
    uncertainty_thresholds = _parse_float_sweep(args.planner_uncertainty_sweep)
    uncertainty_quantiles = _parse_float_sweep(args.planner_uncertainty_quantile_sweep)
    margin_thresholds = _parse_float_sweep(args.planner_margin_sweep)
    q_advantage_thresholds = _parse_float_sweep(args.planner_q_advantage_sweep)
    planner_horizons = _parse_int_sweep(args.planner_horizon_sweep)
    planner_samples = _parse_int_sweep(args.planner_samples_sweep)
    if (
        planner_weights or terminal_weights or uncertainty_thresholds
        or uncertainty_quantiles or margin_thresholds
        or q_advantage_thresholds or planner_horizons or planner_samples
    ):
        if not planner_weights:
            planner_weights = [args.planning_weight]
        if not terminal_weights:
            terminal_weights = [args.terminal_value_weight]
        if not planner_horizons:
            planner_horizons = [args.planner_horizon]
        if not planner_samples:
            planner_samples = [args.planner_samples]
        threshold_labels: list[tuple[float | None, str]] = []
        for quantile in uncertainty_quantiles:
            threshold = resolve_uncertainty_threshold_from_replay(
                config,
                args.checkpoint,
                device,
                quantile,
                max_samples=args.planner_uncertainty_quantile_samples,
            )
            threshold_labels.append((threshold, f"q{quantile:.2f}:{threshold:.3f}"))
        if not uncertainty_thresholds:
            if not threshold_labels:
                threshold_labels = [(
                    planner_uncertainty_threshold,
                    "off" if planner_uncertainty_threshold is None else f"{planner_uncertainty_threshold:.3f}",
                )]
        else:
            for threshold in uncertainty_thresholds:
                threshold_labels.append((threshold, f"{threshold:.3f}"))
        if not margin_thresholds:
            margin_thresholds = [args.planner_margin_threshold]
        if not q_advantage_thresholds:
            q_advantage_thresholds = [args.planner_q_advantage_threshold]
        print("\n=== Planner sweep ===")
        print(
            "  p_weight  tv_weight  horizon  samples  unc_thr  margin  q_adv  "
            "lifespan  planner/Q  drive   food  water  damage  used   max"
        )
        for planning_weight in planner_weights:
            for terminal_weight in terminal_weights:
                for horizon in planner_horizons:
                    for sample_count in planner_samples:
                        for uncertainty_threshold, unc_label in threshold_labels:
                            for margin_threshold in margin_thresholds:
                                for q_advantage_threshold in q_advantage_thresholds:
                                    planned = run_trained(
                                        config, args.checkpoint, args.num_episodes, device,
                                        decision_mode="planner",
                                        planning_weight=planning_weight,
                                        planner_horizon=horizon,
                                        planner_samples=sample_count,
                                        terminal_value_weight=terminal_weight,
                                        planner_uncertainty_threshold=uncertainty_threshold,
                                        planner_margin_threshold=margin_threshold,
                                        planner_q_advantage_threshold=q_advantage_threshold,
                                    )
                                    ratio = planned["mean_lifespan"] / max(trained["mean_lifespan"], 1)
                                    tv_label = "cfg" if terminal_weight is None else f"{terminal_weight:.3f}"
                                    print(
                                        f"  {planning_weight:8.3f}  {tv_label:>9}  "
                                        f"{horizon:7d}  {sample_count:7d}  "
                                        f"{unc_label:>7}  {margin_threshold:6.3f}  "
                                        f"{q_advantage_threshold:5.3f}  "
                                        f"{planned['mean_lifespan']:8.1f}  {ratio:9.2f}  "
                                        f"{planned['drive_regulation']:.3f}  "
                                        f"{planned['interact_food']:5.2f}  {planned['interact_water']:5.2f}  "
                                        f"{planned['damage']:6.2f}  {planned['planner_used']:5.1%}  "
                                        f"{planned['max_lifespan']:3.0f}"
                                    )
    if args.compare_planner:
        planned = run_trained(
            config, args.checkpoint, args.num_episodes, device,
            decision_mode="planner", planning_weight=args.planning_weight,
            planner_horizon=args.planner_horizon, planner_samples=args.planner_samples,
            terminal_value_weight=args.terminal_value_weight,
            planner_uncertainty_threshold=planner_uncertainty_threshold,
            planner_margin_threshold=args.planner_margin_threshold,
            planner_q_advantage_threshold=args.planner_q_advantage_threshold,
        )
        print_stats("Trained Q + WM planner", planned)
        print(
            f"\n  Ratio planner/Q lifespan: "
            f"{planned['mean_lifespan'] / max(trained['mean_lifespan'], 1):.2f}x"
        )


if __name__ == "__main__":
    main()
