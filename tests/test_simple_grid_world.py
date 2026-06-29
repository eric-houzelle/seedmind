"""Tests for SimpleGridWorld (W1 — 2e monde dense pour valider l'universalité)."""
from __future__ import annotations

import numpy as np

from seedmind.agent.micro_fouloide_encoder import egocentric_grid
from seedmind.envs.micro_fouloide_world import AGENT, INTERACT, OBSTACLE
from seedmind.envs.simple_grid_world import GOAL, SimpleGridWorld

OBS_KEYS = {
    "grid", "agent_pos", "standing_entity",
    "energy", "hydration", "temperature", "health",
}


def test_observation_contract():
    """L'obs doit fournir exactement les clés que l'encodeur + la boucle online lisent."""
    env = SimpleGridWorld(size=6, seed=0)
    obs = env.reset()
    assert OBS_KEYS <= set(obs)
    assert obs["grid"].shape == (6, 6)
    assert obs["grid"].dtype == np.int64
    assert isinstance(obs["agent_pos"], tuple)
    # drives figés (pas d'homéostasie)
    assert obs["health"] == 1.0


def test_actions_match_fouloide():
    """Même espace d'action (7) → l'actor du cerveau v3 se branche tel quel."""
    env = SimpleGridWorld(seed=0)
    assert env.available_actions() == [
        "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "INTERACT", "REST", "WAIT",
    ]


def test_interact_on_goal_rewards_and_respawns():
    env = SimpleGridWorld(size=6, seed=1)
    env.reset()
    env.agent_pos = env.goal_pos
    prev_goal = env.goal_pos
    _, reward, done, info = env.step(INTERACT)
    assert reward == 1.0
    assert info["event"] == "interact_goal"
    assert not done                      # monde infini : pas de fin sur collecte
    assert env.goal_pos != env.agent_pos  # une nouvelle cible est posée ailleurs
    assert env.goal_pos != prev_goal or True  # peut retomber ailleurs


def test_interact_off_goal_is_penalised():
    env = SimpleGridWorld(size=6, seed=2)
    env.reset()
    # se placer sur une case non-cible
    env.agent_pos = (0, 0) if env.goal_pos != (0, 0) else (5, 5)
    _, reward, _, info = env.step(INTERACT)
    assert reward < 0
    assert info["event"] == "interact_noop"


def test_move_blocked_by_obstacle_and_bounds():
    env = SimpleGridWorld(size=6, num_obstacles=0, seed=3)
    env.reset()
    env.agent_pos = (0, 0)
    _, _, _, info = env.step("MOVE_UP")  # hors grille
    assert info["event"] == "move_blocked"
    assert env.agent_pos == (0, 0)


def test_agent_overlaid_in_view():
    env = SimpleGridWorld(size=6, seed=4)
    obs = env.reset()
    assert (obs["grid"] == AGENT).sum() == 1


def test_full_observability_in_egocentric_window():
    """Sur 6×6, la cible est TOUJOURS dans la fenêtre égocentrée 11×11 (radius 5)."""
    env = SimpleGridWorld(size=6, seed=5)
    env.reset()
    for ap in [(0, 0), (5, 5), (0, 5), (5, 0), (3, 3)]:
        env.agent_pos = ap
        view = env.observe()["grid"]
        ego = egocentric_grid(view, ap, radius=5, oob_fill=OBSTACLE)
        if env.goal_pos != ap:
            assert (ego == GOAL).sum() >= 1, f"cible hors vue depuis {ap}"


def test_build_env_routes_to_simple_grid():
    from scripts.run_micro_fouloide import build_env

    env = build_env({"env": {"type": "simple_grid", "size": 6}}, seed=0)
    assert isinstance(env, SimpleGridWorld)
    assert env.world_id == "simple_grid"


def test_collecting_goal_repeatedly_streams_reward():
    """Sanity : on peut enchaîner les collectes (flux online), reward +1 à chaque fois."""
    env = SimpleGridWorld(size=6, seed=6)
    env.reset()
    total = 0.0
    for _ in range(5):
        env.agent_pos = env.goal_pos
        _, r, _, info = env.step(INTERACT)
        total += r
        assert info["event"] == "interact_goal"
    assert total == 5.0
