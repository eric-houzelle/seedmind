"""SeedMind — Demo front fouloïdes.

Sert le viewer pixel-art fouloïdes dans le navigateur et diffuse l'état d'un
monde via WebSocket. Le mode par défaut reste un **stub** léger, le mode
`--source micro` branche un checkpoint Micro-Fouloïde réel sur le même viewer,
et le mode `--source live` lance un fouloïde **vierge** (aucun checkpoint) qui
apprend en continu pendant que vous le regardez.

Point de branchement moteur : `WorldSource` (méthodes `world_message()` et
`step_message()`). Le stub et le Micro-Fouloïde réel utilisent la même
interface côté viewer.

    python scripts/demo_fouloides_front.py
    python scripts/demo_fouloides_front.py --size 96 --fouloides 14
    python scripts/demo_fouloides_front.py --source micro --device cpu
    python scripts/demo_fouloides_front.py --source live --tick-ms 60

Ouvrir http://localhost:8787 dans un navigateur.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Protocol, Set, Tuple
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_micro_fouloide import resolve_uncertainty_threshold_from_replay  # noqa: E402
from scripts.run_fouloide_online import OnlineFouloideSession  # noqa: E402
from scripts.run_micro_fouloide import build_agent, build_env, load_config  # noqa: E402
from seedmind.agent.goal_generator import GoalGenerator  # noqa: E402
from seedmind.agent.spatial_resource_memory import SpatialResourceMemory  # noqa: E402
from seedmind.training.device import resolve_device  # noqa: E402
from seedmind.training.wellbeing import drive_regulation  # noqa: E402
from websockets.asyncio.server import serve as ws_serve

VIEWER_HTML = (
    Path(__file__).resolve().parents[1]
    / "seedmind" / "visualization" / "fouloides_viewer.html"
)

OBJECTIVE = "AUGMENTEZ LES RANGS DES FOULO\u00cfDES."
MICRO_OBJECTIVE = "SURVIE MICRO-FOULO\u00cfDE : TROUVER EAU ET NOURRITURE."

DEFAULT_MICRO_CONFIG = (
    "configs/micro_fouloide_v0_rough_valueplanner_resource_navigation.yaml"
)
DEFAULT_MICRO_CHECKPOINT = (
    "runs/micro_fouloide_v0_rough_valueplanner_resource_navigation_seed3/"
    "checkpoint_final.pt"
)


# ---------------------------------------------------------------------------
# Interface moteur de monde
# ---------------------------------------------------------------------------

class WorldSource(Protocol):
    """Interface que devra implémenter l'adaptateur du vrai moteur."""

    def world_message(self) -> dict:
        """État statique du monde (envoyé à chaque connexion)."""
        ...

    def step_message(self) -> dict:
        """Avance le monde d'un tick et retourne l'état dynamique."""
        ...


# ---------------------------------------------------------------------------
# Stub : monde fouloïdes en marche aléatoire
# ---------------------------------------------------------------------------

class StubFouloideWorld:
    """Monde de démonstration : terrain procédural + fouloïdes errants.

    Aucune logique d'apprentissage ici — uniquement de quoi alimenter le
    front avec des données plausibles.
    """

    def __init__(self, size: int = 96, num_fouloides: int = 14,
                 num_baths: int = 4, max_apples: int = 60,
                 tick_ms: int = 150, seed: int = 0) -> None:
        self.size = size
        self.tick_ms = tick_ms
        self.rng = random.Random(seed)
        self.step_count = 0

        self.blocked: Set[Tuple[int, int]] = set()
        self.trees: List[Tuple[int, int]] = []
        self.rocks: List[Tuple[int, int]] = []
        self.baths: List[Tuple[int, int]] = []
        self.terrain: List[List[int]] = [[0] * size for _ in range(size)]

        self._generate_terrain()
        self._place_trees()
        self._place_rocks(count=max(8, size // 8))
        self._place_baths(count=num_baths)

        self.max_apples = max_apples
        self.apples: Set[Tuple[int, int]] = set()
        for _ in range(max_apples // 2):
            self._spawn_apple()

        self.fouloides: List[Dict] = []
        for i in range(num_fouloides):
            x, y = self._free_cell()
            self.fouloides.append({"id": i, "x": x, "y": y, "carry": False})

    # -- génération ---------------------------------------------------------

    def _generate_terrain(self) -> None:
        """Patchs d'herbe usée (zones claires comme sur la référence)."""
        for _ in range(self.size // 6):
            cx = self.rng.randrange(4, self.size - 4)
            cy = self.rng.randrange(4, self.size - 4)
            rx = self.rng.randint(2, 6)
            ry = self.rng.randint(2, 5)
            for y in range(max(0, cy - ry), min(self.size, cy + ry + 1)):
                for x in range(max(0, cx - rx), min(self.size, cx + rx + 1)):
                    dx = (x - cx) / rx
                    dy = (y - cy) / ry
                    if dx * dx + dy * dy <= 1.0 + self.rng.uniform(-0.3, 0.1):
                        self.terrain[y][x] = 1

    def _place_trees(self) -> None:
        s = self.size
        # ceinture forestière dense en bordure
        for y in range(s):
            for x in range(s):
                edge = min(x, y, s - 1 - x, s - 1 - y)
                if edge < 3 and self.rng.random() < (0.85 - edge * 0.25):
                    self._add_tree(x, y)
        # bosquets intérieurs
        for _ in range(s // 5):
            cx = self.rng.randrange(5, s - 5)
            cy = self.rng.randrange(5, s - 5)
            for _ in range(self.rng.randint(3, 8)):
                x = cx + self.rng.randint(-3, 3)
                y = cy + self.rng.randint(-2, 2)
                if 0 <= x < s and 0 <= y < s:
                    self._add_tree(x, y)

    def _add_tree(self, x: int, y: int) -> None:
        if (x, y) not in self.blocked:
            self.trees.append((x, y))
            self.blocked.add((x, y))

    def _place_rocks(self, count: int) -> None:
        for _ in range(count):
            x, y = self._free_cell()
            self.rocks.append((x, y))
            self.blocked.add((x, y))

    def _place_baths(self, count: int) -> None:
        for _ in range(count):
            x, y = self._free_cell(margin=6)
            self.baths.append((x, y))
            self.blocked.add((x, y))

    def _free_cell(self, margin: int = 4) -> Tuple[int, int]:
        while True:
            x = self.rng.randrange(margin, self.size - margin)
            y = self.rng.randrange(margin, self.size - margin)
            if (x, y) not in self.blocked:
                return x, y

    def _spawn_apple(self) -> None:
        """Les pommes apparaissent près des arbres, comme sur la référence."""
        for _ in range(20):
            tx, ty = self.rng.choice(self.trees)
            x = tx + self.rng.randint(-2, 2)
            y = ty + self.rng.randint(-1, 2)
            if (0 <= x < self.size and 0 <= y < self.size
                    and (x, y) not in self.blocked
                    and (x, y) not in self.apples):
                self.apples.add((x, y))
                return

    # -- simulation ---------------------------------------------------------

    def _step_fouloide(self, f: Dict) -> None:
        x, y = f["x"], f["y"]
        # attiré par la pomme la plus proche dans un rayon de 6
        target = None
        best = 7
        for ax, ay in self.apples:
            d = abs(ax - x) + abs(ay - y)
            if d < best:
                best, target = d, (ax, ay)
        if target and not f["carry"]:
            dx = (target[0] > x) - (target[0] < x)
            dy = (target[1] > y) - (target[1] < y)
            moves = [(dx, 0), (0, dy)] if self.rng.random() < 0.5 else [(0, dy), (dx, 0)]
        else:
            moves = [self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.size and 0 <= ny < self.size
                    and (nx, ny) not in self.blocked):
                f["x"], f["y"] = nx, ny
                break
        pos = (f["x"], f["y"])
        if pos in self.apples:
            self.apples.discard(pos)
            f["carry"] = True
        elif f["carry"] and self.rng.random() < 0.02:
            f["carry"] = False  # pomme "mangée"

    # -- interface WorldSource ----------------------------------------------

    def world_message(self) -> dict:
        return {
            "type": "world",
            "width": self.size,
            "height": self.size,
            "terrain": self.terrain,
            "trees": [list(t) for t in self.trees],
            "rocks": [list(r) for r in self.rocks],
            "baths": [list(b) for b in self.baths],
            "tick_ms": self.tick_ms,
        }

    def step_message(self) -> dict:
        self.step_count += 1
        for f in self.fouloides:
            self._step_fouloide(f)
        if len(self.apples) < self.max_apples and self.rng.random() < 0.3:
            self._spawn_apple()
        return {
            "type": "step",
            "step": self.step_count,
            "fouloides": self.fouloides,
            "apples": [list(a) for a in self.apples],
            "stats": {"population": len(self.fouloides)},
            "objective": OBJECTIVE,
        }


# ---------------------------------------------------------------------------
# Source réelle : Micro-Fouloïde entraîné
# ---------------------------------------------------------------------------

def _planner_preset_params() -> dict[str, Any]:
    return {
        "planning_weight": 0.25,
        "terminal_value_weight": 1.0,
        "planner_uncertainty_quantile": 0.60,
        "planner_margin_threshold": 0.01,
        "planner_q_advantage_threshold": 0.02,
        "planner_horizon": 5,
        "planner_samples": 8,
    }


def _configure_micro_planner(config: dict, threshold: float) -> dict:
    params = _planner_preset_params()
    configured = dict(config)
    planning = dict(configured.get("planning", {}))
    planning.update({
        "enabled": True,
        "weight": float(params["planning_weight"]),
        "terminal_value_weight": float(params["terminal_value_weight"]),
        "uncertainty_threshold": float(threshold),
        "margin_threshold": float(params["planner_margin_threshold"]),
        "q_advantage_threshold": float(params["planner_q_advantage_threshold"]),
        "horizon": int(params["planner_horizon"]),
        "num_samples": int(params["planner_samples"]),
    })
    configured["planning"] = planning
    return configured


def _configure_micro_runtime(
    config: dict,
    filter_blocked_moves: bool,
    filter_noop_interact: bool,
) -> dict:
    configured = dict(config)
    env_cfg = dict(configured.get("env", {}))
    env_cfg["filter_blocked_moves"] = bool(filter_blocked_moves)
    env_cfg["filter_noop_interact"] = bool(filter_noop_interact)
    configured["env"] = env_cfg
    return configured


def _load_micro_agent(config: dict, checkpoint: str, device: torch.device):
    agent = build_agent(config, seed=0)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.encoder.load_state_dict(ckpt["encoder_state"])
    agent.world_model.load_state_dict(ckpt["world_model_state"])
    agent.q_network.load_state_dict(ckpt["q_network_state"])
    if agent.value_model is not None and "value_model_state" in ckpt:
        agent.value_model.load_state_dict(ckpt["value_model_state"])
    agent.encoder.to(device)
    agent.world_model.to(device)
    agent.q_network.to(device)
    if agent.value_model is not None:
        agent.value_model.to(device)
    agent.policy.epsilon_start = 0.0
    agent.policy.epsilon_end = 0.0
    return agent


def _drive(info: dict[str, Any]) -> float:
    return drive_regulation(info.get("drives", {}))


class MicroFouloideWorldSource:
    """Adapter MicroFouloideWorld + trained agent to the browser protocol."""

    def __init__(
        self,
        config_path: str,
        checkpoint: str,
        seed: int,
        tick_ms: int,
        device_name: str,
        resource_memory: bool,
        filter_blocked_moves: bool,
        filter_noop_interact: bool,
        uncertainty_threshold: float | None,
    ) -> None:
        self.config_path = config_path
        self.checkpoint = checkpoint
        self.tick_ms = int(tick_ms)
        self.seed = int(seed)
        self.device = resolve_device(device_name)
        self.resource_memory_enabled = bool(resource_memory)
        self.resource_memory_used = 0
        self.planner_used = 0
        self.total_actions = 0
        self.episode = 1

        base_config = load_config(config_path)
        runtime_config = _configure_micro_runtime(
            base_config,
            filter_blocked_moves=filter_blocked_moves,
            filter_noop_interact=filter_noop_interact,
        )
        if uncertainty_threshold is None:
            uncertainty_threshold = resolve_uncertainty_threshold_from_replay(
                runtime_config,
                checkpoint,
                self.device,
                _planner_preset_params()["planner_uncertainty_quantile"],
            )
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.config = _configure_micro_planner(runtime_config, self.uncertainty_threshold)
        self.agent = _load_micro_agent(self.config, checkpoint, self.device)
        self.agent.goal_generator = GoalGenerator(seed=self.seed)
        self.agent.planner.rng = np.random.default_rng(self.seed)
        self.env = build_env(self.config, seed=self.seed)
        registry = self.env.registry
        self.resource_memory = SpatialResourceMemory(
            water_ids=registry.drive_signal_ids("hydration"),
            food_ids=registry.drive_signal_ids("energy"),
            solid_ids=registry.solid_ids,
            unknown_id=registry.by_name("unknown").id,
        )
        self.obs = self.env.reset()
        self.latent = self.agent.encode(self.obs)
        self.info: dict[str, Any] = {
            "drives": {
                "energy": float(self.obs["energy"]),
                "hydration": float(self.obs["hydration"]),
                "temperature": float(self.obs["temperature"]),
                "health": float(self.obs["health"]),
            },
            "event": "reset",
            "lifespan": 0,
        }

    def world_message(self) -> dict:
        grid = self.env.grid
        return {
            "type": "world",
            "width": self.env.size,
            "height": self.env.size,
            "terrain": _grid_terrain(grid, self.env.registry),
            "trees": [],
            "rocks": _grid_positions(grid, _render_ids(self.env.registry, "rock")),
            "dangers": _grid_positions(grid, _render_ids(self.env.registry, "danger")),
            "baths": _grid_positions(grid, _render_ids(self.env.registry, "bath")),
            "vision_radius": self.env.visibility_radius,
            "tick_ms": self.tick_ms,
        }

    def step_message(self) -> dict:
        if self.info.get("dead") or self.info.get("timeout"):
            self._reset_episode()

        self._step_agent()
        grid = self.env.grid
        r, c = (int(v) for v in self.env.agent_pos)
        stats = {
            "population": 1,
            "episode": self.episode,
            "lifespan": int(self.info.get("lifespan", self.env.steps)),
            "event": str(self.info.get("event", "unknown")),
            "drive": round(_drive(self.info), 3),
            "energy": round(float(self.info.get("energy", 0.0)), 3),
            "hydration": round(float(self.info.get("hydration", 0.0)), 3),
            "health": round(float(self.info.get("health", 0.0)), 3),
            "planner_used": round(self.planner_used / max(self.total_actions, 1), 3),
            "resource_memory_used": round(
                self.resource_memory_used / max(self.total_actions, 1), 3
            ),
        }
        objective = (
            f"{MICRO_OBJECTIVE}  HP {stats['health']:.2f} "
            f"H2O {stats['hydration']:.2f} E {stats['energy']:.2f} "
            f"step {stats['lifespan']}"
        )
        return {
            "type": "step",
            "step": self.env.steps,
            "fouloides": [{
                "id": 0, "x": c, "y": r, "carry": False,
                "hp": stats["health"],
                "h2o": stats["hydration"],
                "en": stats["energy"],
            }],
            "terrain": _grid_terrain(grid, self.env.registry),
            "apples": _grid_positions(grid, _render_ids(self.env.registry, "apple")),
            "baths": _grid_positions(grid, _render_ids(self.env.registry, "bath")),
            "rocks": _grid_positions(grid, _render_ids(self.env.registry, "rock")),
            "dangers": _grid_positions(grid, _render_ids(self.env.registry, "danger")),
            "stats": stats,
            "objective": objective,
        }

    def _reset_episode(self) -> None:
        self.episode += 1
        self.resource_memory.reset()
        self.resource_memory_used = 0
        self.planner_used = 0
        self.total_actions = 0
        self.env = build_env(self.config, seed=self.seed + self.episode - 1)
        self.obs = self.env.reset()
        self.latent = self.agent.encode(self.obs)
        self.info = {
            "drives": {
                "energy": float(self.obs["energy"]),
                "hydration": float(self.obs["hydration"]),
                "temperature": float(self.obs["temperature"]),
                "health": float(self.obs["health"]),
            },
            "event": "reset",
            "lifespan": 0,
        }

    def _step_agent(self) -> None:
        if self.resource_memory_enabled:
            self.resource_memory.refresh(self.obs)
        memories = self.agent.retrieve(self.latent)
        goal = self.agent.choose_goal(self.latent, memories)
        available = self.env.available_actions()
        action = self.agent.choose_action(
            self.latent,
            goal,
            memories,
            available,
            observation=self.obs,
        )
        memory_action = None
        if self.resource_memory_enabled:
            memory_action = self.resource_memory.choose_action(self.obs, available)
            if memory_action is not None:
                action = memory_action
                self.resource_memory_used += 1
        self.obs, _, _, self.info = self.env.step(action)
        self.latent = self.agent.encode(self.obs)
        self.total_actions += 1
        self.planner_used += int(getattr(self.agent, "last_planner_used", False))



def _grid_positions(grid: np.ndarray, entities: set[int]) -> list[list[int]]:
    if not entities:
        return []
    rows, cols = np.where(np.isin(grid, list(entities)))
    return [[int(c), int(r)] for r, c in zip(rows, cols)]


_KNOWN_RENDERS = {"rock", "danger", "bath", "apple", "terrain_warm", "terrain_cold", "none"}


def _render_ids(registry, render: str) -> set[int]:
    """Entity ids mapped to a viewer sprite category via their render hint.

    Entities with an unknown render hint fall back to the rock sprite so a
    freshly added YAML entity is always visible.
    """
    ids = registry.render_ids(render)
    if render == "rock":
        ids = ids | {
            e.id for e in registry
            if not e.structural and e.render not in _KNOWN_RENDERS
        }
    return ids


def _grid_terrain(grid: np.ndarray, registry) -> list[list[int]]:
    terrain = np.zeros_like(grid, dtype=np.int64)
    warm = list(_render_ids(registry, "terrain_warm"))
    cold = list(_render_ids(registry, "terrain_cold"))
    if warm:
        terrain[np.isin(grid, warm)] = 1
    if cold:
        terrain[np.isin(grid, cold)] = 2
    return terrain.tolist()


# ---------------------------------------------------------------------------
# Source live : fouloïde vierge qui apprend en direct (aucun checkpoint)
# ---------------------------------------------------------------------------

LIVE_OBJECTIVE = "FOULO\u00cfDE VIERGE : APPRENDRE \u00c0 VIVRE EN DIRECT."


class LiveFouloideWorldSource:
    """From-scratch agent learning continually while the viewer watches."""

    def __init__(
        self,
        config_path: str,
        seed: int,
        tick_ms: int,
        device_name: str,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 0,
        fresh: bool = False,
    ) -> None:
        self.tick_ms = int(tick_ms)
        device = resolve_device(device_name)
        torch.manual_seed(seed)
        config = load_config(config_path)
        self.session = OnlineFouloideSession(config, seed=int(seed), device=device)
        self.planner_window: deque = deque(maxlen=500)
        self.wellbeing_window: deque = deque(maxlen=500)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = int(checkpoint_every)
        self.resumed_steps = 0
        self.thirst_threshold = 0.35
        self._thirst_start: int | None = None
        self._thirst_durations: deque = deque(maxlen=20)
        registry = self.session.env.registry
        if self.session.env.property_events:
            self._drink_events = {"interact_hydration"}
        else:
            self._drink_events = {
                registry[i].interact_event
                for i in registry.drive_signal_ids("hydration")
            }
        if (
            not fresh
            and checkpoint_path is not None
            and Path(checkpoint_path).exists()
        ):
            resumed = self.session.resume(checkpoint_path)
            self.resumed_steps = int(resumed.get("env_steps", 0))

    @property
    def env(self):
        return self.session.env

    def world_message(self) -> dict:
        grid = self.env.grid
        return {
            "type": "world",
            "width": self.env.size,
            "height": self.env.size,
            "terrain": _grid_terrain(grid, self.env.registry),
            "trees": [],
            "rocks": _grid_positions(grid, _render_ids(self.env.registry, "rock")),
            "dangers": _grid_positions(grid, _render_ids(self.env.registry, "danger")),
            "baths": _grid_positions(grid, _render_ids(self.env.registry, "bath")),
            "vision_radius": self.env.visibility_radius,
            "tick_ms": self.tick_ms,
        }

    def step_message(self) -> dict:
        info = self.session.step()
        if (
            self.checkpoint_every > 0
            and self.checkpoint_path is not None
            and self.session.steps % self.checkpoint_every == 0
        ):
            self.session.save(self.checkpoint_path)
        self.planner_window.append(int(self.session.last_planner_used))
        self.wellbeing_window.append(float(self.session.last_wellbeing))
        self._track_thirst(info)
        learn = self.session.learner.stats()
        grid = self.env.grid
        r, c = (int(v) for v in self.env.agent_pos)
        threshold = learn["uncertainty_threshold"]
        stats = {
            "population": 1,
            "life": self.session.lives,
            "step": self.session.steps,
            "event": str(info.get("event", "unknown")),
            "wellbeing": round(float(np.mean(self.wellbeing_window)), 3),
            "energy": round(float(info.get("energy", 0.0)), 3),
            "hydration": round(float(info.get("hydration", 0.0)), 3),
            "health": round(float(info.get("health", 0.0)), 3),
            "wm_loss": round(float(learn["wm_loss"]), 4),
            "td_loss": round(float(learn["td_loss"]), 4),
            "planner_used": round(float(np.mean(self.planner_window)), 3),
            "uncertainty_threshold": (
                None if threshold is None else round(float(threshold), 4)
            ),
            "epsilon": round(float(learn["epsilon"]), 3),
            "thirst_to_water_avg": (
                round(float(np.mean(self._thirst_durations)), 1)
                if self._thirst_durations else None
            ),
            "life_steps": self.session.life_steps,
            "best_life_steps": self.session.best_life_steps,
        }
        if self.session.env.inventory_enabled:
            stats["inventory"] = (
                f"{len(self.session.env.inventory)}/{self.session.env.inventory_capacity}"
            )
        thirst_label = (
            f"{stats['thirst_to_water_avg']:.0f}" if stats["thirst_to_water_avg"] is not None
            else "—"
        )
        record_label = (
            f"{stats['best_life_steps']}" if stats["best_life_steps"] > 0 else "—"
        )
        sack_label = f" sac {stats['inventory']}" if "inventory" in stats else ""
        objective = (
            f"{LIVE_OBJECTIVE}  bien-\u00eatre {stats['wellbeing']:.2f} "
            f"HP {stats['health']:.2f} H2O {stats['hydration']:.2f} E {stats['energy']:.2f}"
            f"  |  vie n\u00b0{stats['life']} : {stats['life_steps']} steps "
            f"(record {record_label}){sack_label}"
            f"  |  soif\u2192eau {thirst_label} wm_loss {stats['wm_loss']:.3f} "
            f"planner {stats['planner_used']:.2f} "
            f"eps {stats['epsilon']:.2f} step {stats['step']}"
        )
        return {
            "type": "step",
            "step": self.session.steps,
            "fouloides": [{
                "id": 0, "x": c, "y": r, "carry": False,
                "hp": stats["health"],
                "h2o": stats["hydration"],
                "en": stats["energy"],
            }],
            "terrain": _grid_terrain(grid, self.env.registry),
            "apples": _grid_positions(grid, _render_ids(self.env.registry, "apple")),
            "baths": _grid_positions(grid, _render_ids(self.env.registry, "bath")),
            "rocks": _grid_positions(grid, _render_ids(self.env.registry, "rock")),
            "dangers": _grid_positions(grid, _render_ids(self.env.registry, "danger")),
            "stats": stats,
            "objective": objective,
        }

    def _track_thirst(self, info: dict[str, Any]) -> None:
        """Steps entre le passage sous le seuil de soif et la prochaine gorgée."""
        hydration = float(info.get("hydration", 1.0))
        if self._thirst_start is None:
            if hydration < self.thirst_threshold:
                self._thirst_start = self.session.steps
        elif str(info.get("event", "")) in self._drink_events:
            self._thirst_durations.append(self.session.steps - self._thirst_start)
            self._thirst_start = None
        elif info.get("dead"):
            self._thirst_start = None


# ---------------------------------------------------------------------------
# Serveurs HTTP + WebSocket
# ---------------------------------------------------------------------------

CLIENTS: Set = set()


async def ws_handler(websocket, source: WorldSource):
    CLIENTS.add(websocket)
    try:
        await websocket.send(json.dumps(source.world_message()))
        async for _ in websocket:
            pass  # pas de contrôle client pour l'instant
    finally:
        CLIENTS.discard(websocket)


async def broadcaster(source: WorldSource, tick_ms: int):
    while True:
        msg = json.dumps(source.step_message())
        dead = set()
        for client in CLIENTS.copy():
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        CLIENTS.difference_update(dead)
        await asyncio.sleep(tick_ms / 1000.0)


async def run_ws_server(host: str, port: int, source: WorldSource, tick_ms: int):
    async with ws_serve(lambda ws: ws_handler(ws, source), host, port):
        await broadcaster(source, tick_ms)


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(VIEWER_HTML.read_bytes())

    def log_message(self, format, *args):
        pass


def run_http_server(host: str, port: int):
    HTTPServer((host, port), ViewerHandler).serve_forever()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SeedMind demo front fouloïdes")
    parser.add_argument("--source", choices=["stub", "micro", "live"], default="stub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--size", type=int, default=96, help="côté du monde (tuiles)")
    parser.add_argument("--fouloides", type=int, default=14)
    parser.add_argument("--tick-ms", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--micro-config", default=DEFAULT_MICRO_CONFIG)
    parser.add_argument("--micro-checkpoint", default=DEFAULT_MICRO_CHECKPOINT)
    parser.add_argument("--micro-uncertainty-threshold", type=float, default=None)
    parser.add_argument(
        "--live-config", default="configs/micro_fouloide_online_properties.yaml",
        help="config phare (propriétés + mémoire spatiale + artefacts) ; "
             "ancien cerveau incompatible → premier lancement avec --live-fresh",
    )
    parser.add_argument(
        "--live-checkpoint", default="runs/fouloide_live/checkpoint_live.pt",
        help="cerveau persistant du fouloïde live (auto-repris s'il existe)",
    )
    parser.add_argument(
        "--live-checkpoint-every", type=int, default=5000,
        help="sauvegarde tous les N steps (0 = off)",
    )
    parser.add_argument(
        "--live-fresh", action="store_true",
        help="ignorer le checkpoint existant et repartir d'un cerveau vierge",
    )
    parser.add_argument("--disable-resource-memory", action="store_true")
    parser.add_argument("--allow-blocked-moves", action="store_true")
    parser.add_argument("--allow-noop-interact", action="store_true")
    args = parser.parse_args()

    if args.source == "micro":
        source = MicroFouloideWorldSource(
            config_path=args.micro_config,
            checkpoint=args.micro_checkpoint,
            seed=args.seed,
            tick_ms=args.tick_ms,
            device_name=args.device,
            resource_memory=not args.disable_resource_memory,
            filter_blocked_moves=not args.allow_blocked_moves,
            filter_noop_interact=not args.allow_noop_interact,
            uncertainty_threshold=args.micro_uncertainty_threshold,
        )
        mode = "micro-fouloide réel"
        world_label = (
            f"{source.env.size}x{source.env.size}, checkpoint={args.micro_checkpoint}, "
            f"unc_thr={source.uncertainty_threshold:.5f}, "
            f"resource_memory={not args.disable_resource_memory}"
        )
    elif args.source == "live":
        source = LiveFouloideWorldSource(
            config_path=args.live_config,
            seed=args.seed,
            tick_ms=args.tick_ms,
            device_name=args.device,
            checkpoint_path=args.live_checkpoint,
            checkpoint_every=args.live_checkpoint_every,
            fresh=args.live_fresh,
        )
        mode = "apprentissage live"
        brain = (
            f"cerveau repris ({source.resumed_steps} steps vécus)"
            if source.resumed_steps > 0 else "cerveau vierge"
        )
        world_label = (
            f"{source.env.size}x{source.env.size}, config={args.live_config}, "
            f"{brain}, sauvegarde {args.live_checkpoint}"
        )
    else:
        source = StubFouloideWorld(
            size=args.size, num_fouloides=args.fouloides,
            tick_ms=args.tick_ms, seed=args.seed,
        )
        mode = "monde stub"
        world_label = f"{args.size}x{args.size}, {args.fouloides} fouloïdes"

    threading.Thread(
        target=run_http_server,
        args=(args.host, args.port),
        daemon=True,
    ).start()

    print(f"\n  SeedMind — Demo front fouloïdes ({mode})")
    print(f"  Viewer:    http://{args.host}:{args.port}")
    print(f"  WebSocket: ws://{args.host}:{args.port + 1}")
    print(f"  Monde:     {world_label}")
    print("  Ctrl+C pour arrêter.\n")

    try:
        asyncio.run(run_ws_server(args.host, args.port + 1, source, args.tick_ms))
    except KeyboardInterrupt:
        pass
    finally:
        if (
            args.source == "live"
            and args.live_checkpoint_every > 0
            and source.session.steps > 0
        ):
            source.session.save(args.live_checkpoint)
            print(f"  cerveau sauvegardé → {args.live_checkpoint} ({source.session.steps} steps)")


if __name__ == "__main__":
    main()
