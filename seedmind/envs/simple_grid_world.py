"""SimpleGridWorld — un 2e monde DENSE et trivial pour valider l'universalité.

But (issue W1) : tester que le MÊME cerveau (RSSM + actor-critic en imagination +
boucle online) apprend sur un monde STRUCTURELLEMENT DIFFÉRENT du fouloïde —
ici une pure tâche de navigation vers une cible visible, reward DENSE et IMMÉDIAT,
PAS d'homéostasie, PAS de drives.

  - Le monde est petit (size 6) et entièrement visible dans la fenêtre égocentrée
    11×11 → AUCUNE ambiguïté de perception (on isole 'le stack apprend-il ?').
  - Une cible (GOAL) est posée sur une case ; INTERACT dessus = +1 et la cible
    RÉAPPARAÎT ailleurs (flux continu de reward, comme l'online infini du fouloïde).
  - Petit coût par pas (-0.01) → l'agent a intérêt à y aller VITE.

Ce que ça tranche :
  • si l'agent apprend à foncer sur la cible et INTERACT → le port DreamerV3 est
    bon (la boucle online complète exploite un avantage d'action clair et immédiat) ;
    alors l'échec du fourrage fouloïde est un problème de RÉCOMPENSE/EXPLORATION,
    pas d'architecture.
  • s'il n'apprend MÊME PAS ça → la boucle online a un vrai bug à corriger.

Réutilise le vocabulaire d'entités du fouloïde (EMPTY/OBSTACLE/AGENT/UNKNOWN +
FOOD comme marqueur de cible) pour que TOUTE la perception (encodeur égocentré,
remplissage OOB=OBSTACLE, registre à 9 entités) marche à l'identique. Seules la
DYNAMIQUE et la RÉCOMPENSE diffèrent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from seedmind.envs.base import EnvironmentAdapter
from seedmind.envs.micro_fouloide_world import (
    ACTIONS,
    AGENT,
    EMPTY,
    FOOD,
    INTERACT,
    OBSTACLE,
    UNKNOWN,
)

GOAL = FOOD  # on réutilise l'id FOOD (=3) comme marqueur de cible (rendu pomme)


class SimpleGridWorld(EnvironmentAdapter):
    """Navigation vers une cible visible, reward dense. Monde infini (resets via
    timeout optionnel ; la cible réapparaît à chaque collecte)."""

    def __init__(
        self,
        size: int = 6,
        max_steps: int = 0,           # 0 = pas de timeout (flux online infini)
        num_obstacles: int = 0,
        goal_reward: float = 1.0,
        step_penalty: float = 0.01,
        noop_penalty: float = 0.02,   # INTERACT/REST/WAIT hors cible : léger malus
        visibility_radius: Optional[int] = None,  # None = pas de brouillard (tout visible)
        seed: Optional[int] = None,
    ) -> None:
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.num_obstacles = int(num_obstacles)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.noop_penalty = float(noop_penalty)
        self.visibility_radius = visibility_radius
        self.world_id = "simple_grid"
        self.actions: List[str] = list(ACTIONS)
        self._rng = np.random.default_rng(seed)
        # drives constants : ce monde n'a pas d'homéostasie, mais la perception et
        # la boucle online lisent ces clés → on les fournit, figées.
        self.energy = 1.0
        self.hydration = 1.0
        self.temperature = 0.5
        self.health = 1.0
        self._steps = 0
        self._last_event = "reset"
        self.reset()

    # -- placement ------------------------------------------------------
    def _empty_cells(self) -> np.ndarray:
        return np.argwhere(self.grid == EMPTY)

    def _place_random(self, value: int) -> Tuple[int, int]:
        cells = self._empty_cells()
        idx = int(self._rng.integers(len(cells)))
        r, c = int(cells[idx][0]), int(cells[idx][1])
        self.grid[r, c] = value
        return r, c

    def _respawn_goal(self) -> None:
        cells = self._empty_cells()
        # exclut la case de l'agent
        cells = [tuple(cell) for cell in cells if tuple(cell) != self.agent_pos]
        r, c = cells[int(self._rng.integers(len(cells)))]
        self.grid[r, c] = GOAL
        self.goal_pos = (int(r), int(c))

    # -- API EnvironmentAdapter -----------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)
        for _ in range(min(self.num_obstacles, self.size * self.size // 4)):
            self._place_random(OBSTACLE)
        ar, ac = self._place_random(AGENT)
        self.agent_pos = (int(ar), int(ac))
        self.grid[ar, ac] = EMPTY  # l'agent n'occupe pas la grille (overlay à l'observe)
        self._respawn_goal()
        self._steps = 0
        self._last_event = "reset"
        return self.observe()

    def available_actions(self) -> List[str]:
        return list(self.actions)

    def observe(self) -> Dict[str, Any]:
        view = self.grid.copy()
        r, c = self.agent_pos
        standing_entity = int(self.grid[r, c])
        view[r, c] = AGENT
        if self.visibility_radius is not None:
            for row in range(self.size):
                for col in range(self.size):
                    if abs(row - r) + abs(col - c) > self.visibility_radius:
                        view[row, col] = UNKNOWN
        return {
            "grid": view,
            "agent_pos": self.agent_pos,
            "standing_entity": standing_entity,
            "energy": self.energy,
            "hydration": self.hydration,
            "temperature": self.temperature,
            "health": self.health,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._steps += 1
        r, c = self.agent_pos
        reward = -self.step_penalty
        self._last_event = "move_ok"

        if action in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            dr, dc = {"MOVE_UP": (-1, 0), "MOVE_DOWN": (1, 0),
                      "MOVE_LEFT": (0, -1), "MOVE_RIGHT": (0, 1)}[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != OBSTACLE:
                self.agent_pos = (nr, nc)
            else:
                self._last_event = "move_blocked"
        elif action == INTERACT:
            if self.grid[r, c] == GOAL:
                reward = self.goal_reward
                self._last_event = "interact_goal"
                self.grid[r, c] = EMPTY
                self._respawn_goal()
            else:
                reward = -self.noop_penalty
                self._last_event = "interact_noop"
        else:  # REST / WAIT
            reward = -self.noop_penalty
            self._last_event = action.lower()

        done = self.max_steps > 0 and self._steps >= self.max_steps
        info = {
            "event": self._last_event,
            "drives": {
                "energy": self.energy,
                "hydration": self.hydration,
                "temperature": self.temperature,
                "health": self.health,
            },
        }
        return self.observe(), float(reward), bool(done), info

    def describe_transition(self) -> str:
        return self._last_event
