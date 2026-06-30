"""SeedMind — Fouloïde homéostatique en apprentissage online (headless).

Un agent vierge (aucun checkpoint) vit dans un monde persistant doux et
apprend en continu via OnlineLearner : WM + DQN + Value mis à jour tous les
K steps, seuil du gate planner rafraîchi en ligne. Validation du pivot avant
le branchement viewer. `OnlineFouloideSession` est aussi consommée par
`demo_fouloides_front.py --source live`.

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
    _spatial_memory_config,
    build_agent,
    build_env,
    causal_event_names,
    load_config,
)
from seedmind.agent.map_memory import MapMemory  # noqa: E402
from seedmind.agent.curiosity import compute_prediction_error_tensor  # noqa: E402
from seedmind.envs.micro_fouloide_world import COMBINE, DROP, PICK, PLANT  # noqa: E402
from seedmind.memory.experience_buffer import make_experience  # noqa: E402
from seedmind.training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from seedmind.training.device import resolve_device  # noqa: E402
from seedmind.training.latent_utils import latent_to_numpy  # noqa: E402
from seedmind.training.online import OnlineLearner  # noqa: E402
from seedmind.training.wellbeing import wellbeing  # noqa: E402


ARTIFACT_ACTIONS = frozenset({PICK, DROP, PLANT, COMBINE})


def _curriculum_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("action_curriculum")
    if cfg is None:
        cfg = config.get("online", {}).get("action_curriculum", {})
    return cfg if isinstance(cfg, dict) else {}


def artifact_actions_unlocked(
    config: dict[str, Any],
    steps: int,
    recent_wellbeing: deque[float],
    recent_events: deque[str],
) -> bool:
    """Return whether object manipulation should be exposed to the policy."""
    cfg = _curriculum_cfg(config)
    if not bool(cfg.get("enabled", False)):
        return True

    if steps < int(cfg.get("unlock_after_steps", 0)):
        return False

    min_samples = int(cfg.get("min_recent_samples", 0))
    if min_samples > 0 and len(recent_wellbeing) < min_samples:
        return False

    min_wellbeing = cfg.get("min_recent_wellbeing")
    if min_wellbeing is not None:
        if not recent_wellbeing:
            return False
        if float(np.mean(recent_wellbeing)) < float(min_wellbeing):
            return False

    counts = Counter(recent_events)
    if counts["interact_hydration"] + counts["interact_water"] < int(
        cfg.get("min_recent_hydration_events", 0)
    ):
        return False
    if counts["interact_energy"] + counts["interact_food"] + counts["interact_berry_bush"] < int(
        cfg.get("min_recent_energy_events", 0)
    ):
        return False
    return True


def curriculum_available_actions(
    actions: list[str],
    config: dict[str, Any],
    steps: int,
    recent_wellbeing: deque[float],
    recent_events: deque[str],
) -> tuple[list[str], bool]:
    """Filter high-level artifact actions until basic homeostasis is visible."""
    unlocked = artifact_actions_unlocked(config, steps, recent_wellbeing, recent_events)
    if unlocked:
        return list(actions), True
    cfg = _curriculum_cfg(config)
    blocked = set(cfg.get("artifact_actions", ARTIFACT_ACTIONS))
    filtered = [action for action in actions if action not in blocked]
    return (filtered if filtered else list(actions)), False


class OnlineFouloideSession:
    """Persistent world + from-scratch agent + continual learner, one step at a time."""

    def __init__(self, config: dict, seed: int, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.comfort = _comfort_config(config)
        cwm = config.get("causal_world_model", {})
        self.causal_wm_enabled = bool(cwm.get("enabled", False))
        self.event_to_index = {event: i for i, event in enumerate(causal_event_names(config))}

        self.agent = build_agent(config, seed=seed)
        self.agent.encoder.to(device)
        self.agent.world_model.to(device)
        self.agent.q_network.to(device)
        if self.agent.value_model is not None:
            self.agent.value_model.to(device)
        if getattr(self.agent, "actor", None) is not None:
            self.agent.actor.to(device)
            self.agent.critic.to(device)
        self.learner = OnlineLearner(self.agent, config, device, seed=seed)

        self.env = build_env(config, seed=seed)
        self.observation = self.env.reset()
        sm_cfg = _spatial_memory_config(config)
        self.map_memory = (
            MapMemory(self.env.size, horizon=int(sm_cfg.get("horizon", 300)))
            if sm_cfg is not None else None
        )
        if self.map_memory is not None:
            self.map_memory.observe(self.observation)
            self.observation = self.map_memory.augment(self.observation)
        self.latent_state = self.agent.encoder.encode_tensor(self.observation)
        self.agent.reset_state()  # fresh recurrent memory at birth
        self.lives = 1
        self.steps = 0
        self.life_steps = 0
        self.best_life_steps = 0
        self.last_info: Dict[str, Any] = {"drives": dict(self.observation_drives()), "event": "reset"}
        self.last_action = "reset"
        self.last_planner_used = False
        self.last_wellbeing = wellbeing(self.last_info["drives"], self.comfort)
        self.recent_wellbeing: deque[float] = deque(maxlen=1000)
        self.recent_events: deque[str] = deque(maxlen=1000)
        self.artifact_actions_unlocked = artifact_actions_unlocked(
            self.config, self.steps, self.recent_wellbeing, self.recent_events
        )

    def observation_drives(self) -> Dict[str, float]:
        return {
            "energy": float(self.observation["energy"]),
            "hydration": float(self.observation["hydration"]),
            "temperature": float(self.observation["temperature"]),
            "health": float(self.observation["health"]),
        }

    def step(self) -> Dict[str, Any]:
        """Act, learn from the transition, handle (rare) death. Returns env info."""
        agent, env = self.agent, self.env
        latent_np = latent_to_numpy(self.latent_state)
        memories = agent.retrieve(latent_np)
        goal = agent.choose_goal(latent_np, memories)
        available_actions, self.artifact_actions_unlocked = curriculum_available_actions(
            env.available_actions(),
            self.config,
            self.steps,
            self.recent_wellbeing,
            self.recent_events,
        )
        agent.advance(latent_np)  # update recurrent state h_t (no-op if feed-forward)
        action = agent.choose_action(
            latent_np, goal, memories, available_actions,
            observation=self.observation,
        )
        self.last_action = action
        self.last_planner_used = bool(getattr(agent, "last_planner_used", False))
        action_index = agent.action_index[action]
        next_obs, reward_ext, done, info = env.step(action)
        if self.map_memory is not None:
            self.map_memory.observe(next_obs)
            next_obs = self.map_memory.augment(next_obs)
        next_latent = agent.encoder.encode_tensor(next_obs)
        event = str(info.get("event", "unknown"))

        if getattr(agent, "_rssm", False) and agent.rssm_state is not None:
            # Curiosity from the RSSM: prior-predicted next embedding vs the real one.
            with torch.no_grad():
                device = next(agent.world_model.parameters()).device
                action_t = torch.as_tensor([action_index], dtype=torch.long, device=device)
                prior = agent.world_model.img_step(agent.rssm_state, action_t)
                predicted = agent.world_model.heads(agent.world_model.get_feat(prior))["recon"].squeeze(0)
        elif agent.recurrent and agent.h is not None:
            # Curiosity from the recurrent WM: how well it predicted z_{t+1}
            # from the current recurrent state h_t and the chosen action.
            action_t = torch.as_tensor([action_index], dtype=torch.long, device=agent.h.device)
            predicted = agent.world_model.forward(agent.h, action_t)[0].squeeze(0)
        else:
            predicted, _, _ = agent.world_model.predict_tensor(self.latent_state, action_index)
        pred_err = float(compute_prediction_error_tensor(predicted, next_latent).item())
        reward_int = agent.curiosity.compute(pred_err)
        reward_learning = _learning_reward(reward_ext, self.observation, info, self.config)

        self.steps += 1
        experience = make_experience(
            episode_id=f"online_life_{self.lives:04d}",
            world_id=env.world_id,
            step=self.steps,
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
            obs_state=_compact_obs(self.observation),
            next_obs_state=_compact_obs(next_obs),
            event=event,
            event_amount=int(info.get("event_amount", 0)),
            event_index=self.event_to_index.get(event) if self.causal_wm_enabled else None,
            causal_features=env.causal_features(self.observation) if self.causal_wm_enabled else None,
            next_causal_features=env.causal_features(next_obs) if self.causal_wm_enabled else None,
        )
        experience["reward_learning"] = reward_learning
        self.learner.observe(experience)
        agent.memory.store_if_important(experience)

        self.observation = next_obs
        self.latent_state = next_latent
        self.last_info = info
        self.last_wellbeing = wellbeing(info.get("drives", {}), self.comfort)
        self.recent_wellbeing.append(float(self.last_wellbeing))
        self.recent_events.append(event)

        self.life_steps += 1
        if done:
            self.best_life_steps = max(self.best_life_steps, self.life_steps)
            self.life_steps = 0
            self.lives += 1
            agent.reset_state()  # the recurrent memory dies with the individual
            self.observation = self.env.reset()
            if self.map_memory is not None:
                # La carte meurt avec l'individu (nouveau layout).
                self.map_memory.reset()
                self.map_memory.observe(self.observation)
                self.observation = self.map_memory.augment(self.observation)
            self.latent_state = agent.encoder.encode_tensor(self.observation)
        return info

    # ------------------------------------------------------------------
    # Persistance du cerveau (le monde se régénère, le vécu persiste)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        learner = self.learner
        metrics = {
            "session_steps": self.steps,
            "lives": self.lives,
            "life_steps": self.life_steps,
            "best_life_steps": self.best_life_steps,
            "env_steps": learner.env_steps,
            "total_q_updates": learner.total_q_updates,
            "total_value_updates": learner.total_value_updates,
            "next_target_sync": learner.next_target_sync,
            "next_value_target_sync": learner.next_value_target_sync,
            "uncertainty_threshold": learner.uncertainty_threshold,
        }
        target = Path(path)
        tmp = target.with_suffix(target.suffix + ".tmp")
        save_checkpoint(
            str(tmp), self.agent,
            optimizer=learner.wm_optimizer,
            buffer=learner.buffer,
            metrics=metrics,
            config=self.config,
            q_optimizer=learner.q_optimizer,
            target_network=learner.target_network,
            value_optimizer=learner.value_optimizer,
            target_value_model=learner.target_value_model,
            actor_optimizer=learner.actor_optimizer,
            critic_optimizer=learner.critic_optimizer,
            target_critic=learner.target_critic,
        )
        tmp.replace(target)

    def resume(self, path: str) -> Dict[str, Any]:
        learner = self.learner
        info = load_checkpoint(
            str(path), self.agent,
            optimizer=learner.wm_optimizer,
            buffer=learner.buffer,
            q_optimizer=learner.q_optimizer,
            target_network=learner.target_network,
            value_optimizer=learner.value_optimizer,
            target_value_model=learner.target_value_model,
            actor_optimizer=learner.actor_optimizer,
            critic_optimizer=learner.critic_optimizer,
            target_critic=learner.target_critic,
        )
        m = info.get("metrics", {})
        self.steps = int(m.get("session_steps", 0))
        self.lives = int(m.get("lives", 1))
        self.life_steps = int(m.get("life_steps", 0))
        self.best_life_steps = int(m.get("best_life_steps", 0))
        learner.env_steps = int(m.get("env_steps", 0))
        learner.total_q_updates = int(m.get("total_q_updates", 0))
        learner.total_value_updates = int(m.get("total_value_updates", 0))
        learner.next_target_sync = int(m.get("next_target_sync", learner.target_update))
        learner.next_value_target_sync = int(
            m.get("next_value_target_sync", learner.value_target_update)
        )
        threshold = m.get("uncertainty_threshold")
        if threshold is not None:
            learner.uncertainty_threshold = float(threshold)
            if getattr(self.agent, "use_planner", False):
                self.agent.planner_uncertainty_threshold = float(threshold)
        # Le latent courant doit venir de l'encodeur restauré.
        self.latent_state = self.agent.encoder.encode_tensor(self.observation)
        self.agent.reset_state()  # recurrent memory restarts from the current obs
        return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micro_fouloide_online_homeostatic.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="sauvegarde du cerveau tous les N steps (0 = off)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="Override imagination.entropy_coef (Dreamer exploration bonus).")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override imagination.horizon (imagined rollout length).")
    parser.add_argument("--resume", default=None,
                        help="checkpoint online à reprendre")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = load_config(args.config)
    if args.entropy_coef is not None:
        config.setdefault("imagination", {})["entropy_coef"] = float(args.entropy_coef)
        print(f"[override] imagination.entropy_coef = {args.entropy_coef}")
    if args.horizon is not None:
        config.setdefault("imagination", {})["horizon"] = int(args.horizon)
        print(f"[override] imagination.horizon = {args.horizon}")
    out_dir = Path(args.out_dir or f"runs/fouloide_online_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    session = OnlineFouloideSession(config, seed=args.seed, device=device)
    checkpoint_path = out_dir / "checkpoint_online.pt"
    if args.resume:
        resumed = session.resume(args.resume)
        print(
            f"Reprise de {args.resume} : {resumed.get('env_steps', 0)} steps vécus, "
            f"vie {session.lives}"
        )

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
        info = session.step()
        event_counts[str(info.get("event", "unknown"))] += 1
        wellbeing_window.append(session.last_wellbeing)
        planner_window.append(int(session.last_planner_used))

        if step % window == 0:
            stats = session.learner.stats()
            row = {
                "step": step,
                "wellbeing": float(np.mean(wellbeing_window)),
                "planner_used_rate": float(np.mean(planner_window)),
                "deaths": session.lives - 1,
                "interact_water": int(
                    event_counts.get("interact_water", 0)
                    + event_counts.get("interact_hydration", 0)
                ),
                "interact_food": int(
                    event_counts.get("interact_food", 0)
                    + event_counts.get("interact_energy", 0)
                ),
                "interact_noop": int(event_counts.get("interact_noop", 0)),
                "health_loss": int(event_counts.get("health_loss", 0)),
                **{k: stats[k] for k in (
                    "wm_loss", "td_loss", "value_loss", "actor_loss", "critic_loss",
                    "imag_entropy", "imag_return",
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
                f"morts {row['deaths']} | eps {row['epsilon']:.2f}"
            )
            metrics_path = out_dir / "metrics_online.json"
            tmp = metrics_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"config": args.config, "seed": args.seed, "windows": history}, f, indent=2)
            tmp.replace(metrics_path)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            session.save(str(checkpoint_path))
            print(f"  cerveau sauvegardé → {checkpoint_path}")

    if args.checkpoint_every > 0:
        session.save(str(checkpoint_path))
        print(f"  cerveau final sauvegardé → {checkpoint_path}")
    print(
        f"Terminé : {args.steps} steps, {session.lives - 1} morts, "
        f"métriques dans {out_dir}/metrics_online.json"
    )


if __name__ == "__main__":
    main()
