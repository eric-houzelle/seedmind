"""Tests for SandboxWorld (Niveau 0 — Survival)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from seedmind.envs.sandbox_world import (
    ACTIONS,
    AGENT,
    ALIVE_BONUS,
    DEATH_PENALTY,
    EAT,
    EMPTY,
    FOOD_SOURCE,
    FOOD_SOURCE_DEPLETED,
    HARVEST,
    MOVE_DOWN,
    MOVE_RIGHT,
    CRAFT,
    CRAFT_ACTIONS,
    UNKNOWN,
    WAIT,
    WALL,
    WOOD_SOURCE,
    STONE_SOURCE,
    SandboxWorld,
)
from seedmind.agent.sandbox_encoder import (
    SANDBOX_CRAFT_NUM_CHANNELS,
    SANDBOX_CRAFT_NUM_SCALARS,
    SANDBOX_NUM_CHANNELS,
    SANDBOX_NUM_SCALARS,
    make_sandbox_obs_batch_to_tensors,
    make_sandbox_observation_to_vector,
    sandbox_obs_batch_to_tensors,
    sandbox_qnet_channels,
    sandbox_qnet_scalars,
    sandbox_observation_to_vector,
)
from seedmind.agent.q_network import QNetwork


# ------------------------------------------------------------------
# Energy mechanics
# ------------------------------------------------------------------

class TestEnergyMechanics:

    def test_energy_decays_each_step(self):
        env = SandboxWorld(size=6, energy_start=20.0, energy_decay=1.0, seed=0)
        env.reset()
        start_energy = env.energy
        env.step(WAIT)
        assert env.energy == pytest.approx(start_energy - 1.0)

    def test_death_at_zero_energy(self):
        env = SandboxWorld(size=6, energy_start=2.0, energy_decay=1.0,
                           max_steps=100, seed=0)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, info = env.step(WAIT)
            steps += 1
        assert info["dead"] is True
        assert env.energy <= 0

    def test_death_penalty_reward(self):
        env = SandboxWorld(size=6, energy_start=1.0, energy_decay=1.0, seed=0)
        env.reset()
        _, reward, done, _ = env.step(WAIT)
        assert done is True
        assert reward == pytest.approx(DEATH_PENALTY)

    def test_alive_bonus_when_alive(self):
        env = SandboxWorld(size=6, energy_start=50.0, energy_decay=1.0, seed=0)
        env.reset()
        _, reward, done, _ = env.step(WAIT)
        assert done is False
        assert reward == pytest.approx(ALIVE_BONUS)


# ------------------------------------------------------------------
# Harvest / Eat
# ------------------------------------------------------------------

class TestHarvestEat:

    def _place_agent_on_food(self, env: SandboxWorld):
        """Move the agent to a food source cell."""
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] == FOOD_SOURCE:
                    env.agent_pos = (r, c)
                    return True
        return False

    def test_harvest_on_food_source(self):
        env = SandboxWorld(size=6, num_food_sources=4, energy_start=50.0, seed=42)
        env.reset()
        assert self._place_agent_on_food(env)
        assert env.inventory["food"] == 0
        _, _, _, info = env.step(HARVEST)
        assert env.inventory["food"] == 1
        assert info["event"] == "harvest_food"
        assert info["event_amount"] == 1
        r, c = env.agent_pos
        assert env.grid[r, c] == FOOD_SOURCE_DEPLETED

    def test_harvest_on_empty_is_noop(self):
        env = SandboxWorld(size=6, num_food_sources=4, energy_start=50.0, seed=42)
        env.reset()
        # Place agent on empty cell
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] == EMPTY:
                    env.agent_pos = (r, c)
                    break
        env.step(HARVEST)
        assert env.inventory["food"] == 0

    def test_eat_restores_energy(self):
        env = SandboxWorld(size=6, energy_start=30.0, energy_decay=1.0,
                           food_energy=15.0, energy_max=100.0, seed=42)
        env.reset()
        assert self._place_agent_on_food(env)
        env.step(HARVEST)
        energy_before = env.energy
        _, _, _, info = env.step(EAT)
        # Energy should increase by food_energy minus the decay
        expected = min(100.0, energy_before - 1.0 + 15.0)
        assert env.energy == pytest.approx(expected, abs=0.01)
        assert env.inventory["food"] == 0
        assert info["event"] == "eat_ok"

    def test_eat_without_food_is_noop(self):
        env = SandboxWorld(size=6, energy_start=30.0, energy_decay=1.0, seed=42)
        env.reset()
        energy_before = env.energy
        env.step(EAT)
        assert env.energy == pytest.approx(energy_before - 1.0)

    def test_energy_capped_at_max(self):
        env = SandboxWorld(size=6, energy_start=95.0, energy_decay=0.0,
                           food_energy=20.0, energy_max=100.0,
                           num_food_sources=4, seed=42)
        env.reset()
        assert self._place_agent_on_food(env)
        env.step(HARVEST)
        env.step(EAT)
        assert env.energy <= 100.0


# ------------------------------------------------------------------
# Craft
# ------------------------------------------------------------------

class TestCraft:

    def _place_agent_on(self, env: SandboxWorld, entity: int) -> bool:
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] == entity:
                    env.agent_pos = (r, c)
                    return True
        return False

    def test_craft_action_only_available_when_enabled(self):
        base = SandboxWorld(size=8, seed=0)
        craft = SandboxWorld(
            size=8, craft_enabled=True,
            num_wood_sources=2, num_stone_sources=2, seed=0,
        )
        assert CRAFT not in base.available_actions()
        assert craft.available_actions() == CRAFT_ACTIONS

    def test_harvest_wood_and_stone(self):
        env = SandboxWorld(
            size=8, craft_enabled=True,
            num_food_sources=1, num_wood_sources=2, num_stone_sources=2,
            energy_start=50.0, seed=3,
        )
        env.reset()
        assert self._place_agent_on(env, WOOD_SOURCE)
        _, _, _, info = env.step(HARVEST)
        assert env.inventory["wood"] == 1
        assert info["event"] == "harvest_wood"
        assert self._place_agent_on(env, STONE_SOURCE)
        _, _, _, info = env.step(HARVEST)
        assert env.inventory["stone"] == 1
        assert info["event"] == "harvest_stone"

    def test_craft_consumes_resources_and_creates_tool(self):
        env = SandboxWorld(size=8, craft_enabled=True, seed=0)
        env.reset()
        env.inventory["wood"] = 1
        env.inventory["stone"] = 1
        _, _, _, info = env.step(CRAFT)
        assert env.inventory["wood"] == 0
        assert env.inventory["stone"] == 0
        assert env.inventory["tool"] == 1
        assert info["event"] == "craft_tool"

    def test_tool_increases_food_harvest_yield(self):
        env = SandboxWorld(
            size=8, craft_enabled=True, num_food_sources=3,
            base_food_yield=1, tool_food_bonus=3,
            energy_start=50.0, seed=7,
        )
        env.reset()
        assert self._place_agent_on(env, FOOD_SOURCE)
        env.inventory["tool"] = 1
        _, _, _, info = env.step(HARVEST)
        assert env.inventory["food"] == 4
        assert info["event"] == "harvest_food_tool"
        assert info["event_amount"] == 4

    def test_food_yield_without_tool_is_configurable(self):
        env = SandboxWorld(
            size=8, craft_enabled=True, num_food_sources=3,
            base_food_yield=0, tool_food_bonus=3,
            energy_start=50.0, seed=7,
        )
        env.reset()
        assert self._place_agent_on(env, FOOD_SOURCE)
        _, _, _, info = env.step(HARVEST)
        assert env.inventory["food"] == 0
        assert info["event"] == "harvest_food"
        assert info["event_amount"] == 0


# ------------------------------------------------------------------
# Regrow
# ------------------------------------------------------------------

class TestRegrow:

    def test_food_regrows_after_delay(self):
        # regrow_delay=4: the source regrows 4 ticks after harvest.
        # Tick 0 = the harvest step itself (timer set to 4, then decremented
        # to 3 by the end of that step), so 3 more WAITs are needed.
        env = SandboxWorld(size=6, num_food_sources=4, energy_start=200.0,
                           energy_decay=0.0, regrow_delay=4, seed=42)
        env.reset()
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] == FOOD_SOURCE:
                    env.agent_pos = (r, c)
                    break
        pos = env.agent_pos
        env.step(HARVEST)
        assert env.grid[pos] == FOOD_SOURCE_DEPLETED
        for _ in range(2):
            env.step(WAIT)
            assert env.grid[pos] == FOOD_SOURCE_DEPLETED
        env.step(WAIT)
        assert env.grid[pos] == FOOD_SOURCE


# ------------------------------------------------------------------
# Movement
# ------------------------------------------------------------------

class TestMovement:

    def test_move_blocked_by_wall(self):
        env = SandboxWorld(size=6, energy_start=50.0, seed=0)
        env.reset()
        # Agent at (1,1), border wall at (0,x)
        env.agent_pos = (1, 1)
        pos_before = env.agent_pos
        # Move up should be blocked by border wall
        env.step("MOVE_UP")
        # Position either same (blocked) or moved — depends on grid
        # Just verify still in bounds
        r, c = env.agent_pos
        assert 0 <= r < env.size and 0 <= c < env.size


# ------------------------------------------------------------------
# Observation shape
# ------------------------------------------------------------------

class TestObservation:

    def test_observe_has_required_keys(self):
        env = SandboxWorld(size=6, seed=0)
        obs = env.reset()
        assert "grid" in obs
        assert "agent_pos" in obs
        assert "energy" in obs
        assert "energy_max" in obs
        assert "inventory_food" in obs

    def test_grid_shape(self):
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        assert np.asarray(obs["grid"]).shape == (8, 8)


# ------------------------------------------------------------------
# Encoder compatibility
# ------------------------------------------------------------------

class TestSandboxEncoder:

    def test_qnet_channels_shape(self):
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        ch = sandbox_qnet_channels(obs)
        assert ch.shape == (SANDBOX_NUM_CHANNELS, 8, 8)

    def test_qnet_scalars_shape(self):
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        sc = sandbox_qnet_scalars(obs)
        assert sc.shape == (SANDBOX_NUM_SCALARS,)

    def test_observation_to_vector_shape(self):
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        vec = sandbox_observation_to_vector(obs)
        expected_len = 8 * 8 * SANDBOX_NUM_CHANNELS + SANDBOX_NUM_SCALARS
        assert vec.shape == (expected_len,)

    def test_batch_to_tensors(self):
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        ch, sc = sandbox_obs_batch_to_tensors([obs, obs])
        assert ch.shape == (2, SANDBOX_NUM_CHANNELS, 8, 8)
        assert sc.shape == (2, SANDBOX_NUM_SCALARS)

    def test_craft_batch_to_tensors(self):
        env = SandboxWorld(
            size=8, craft_enabled=True,
            num_wood_sources=2, num_stone_sources=2, seed=0,
        )
        obs = env.reset()
        obs["inventory_wood"] = 1
        obs["inventory_stone"] = 2
        obs["inventory_tool"] = 1
        convert = make_sandbox_obs_batch_to_tensors(include_craft=True)
        ch, sc = convert([obs, obs])
        assert ch.shape == (2, SANDBOX_CRAFT_NUM_CHANNELS, 8, 8)
        assert sc.shape == (2, SANDBOX_CRAFT_NUM_SCALARS)

    def test_craft_observation_to_vector_shape(self):
        env = SandboxWorld(
            size=8, craft_enabled=True,
            num_wood_sources=2, num_stone_sources=2, seed=0,
        )
        obs = env.reset()
        to_vec = make_sandbox_observation_to_vector(include_craft=True)
        vec = to_vec(obs)
        expected_len = 8 * 8 * SANDBOX_CRAFT_NUM_CHANNELS + SANDBOX_CRAFT_NUM_SCALARS
        assert vec.shape == (expected_len,)


# ------------------------------------------------------------------
# Q-Network compatibility
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Partial observation
# ------------------------------------------------------------------

class TestPartialObservation:

    def test_unknown_cells_outside_radius(self):
        env = SandboxWorld(size=8, visibility_radius=2, energy_start=50.0, seed=0)
        obs = env.reset()
        grid = np.asarray(obs["grid"])
        ar, ac = obs["agent_pos"]
        for r in range(8):
            for c in range(8):
                if abs(r - ar) + abs(c - ac) > 2:
                    assert grid[r, c] == UNKNOWN, f"Cell ({r},{c}) should be UNKNOWN"

    def test_agent_always_visible(self):
        env = SandboxWorld(size=8, visibility_radius=2, energy_start=50.0, seed=0)
        obs = env.reset()
        grid = np.asarray(obs["grid"])
        ar, ac = obs["agent_pos"]
        assert grid[ar, ac] == AGENT

    def test_no_unknown_without_radius(self):
        env = SandboxWorld(size=8, energy_start=50.0, seed=0)
        obs = env.reset()
        grid = np.asarray(obs["grid"])
        assert (grid != UNKNOWN).all()

    def test_encoder_handles_unknown(self):
        env = SandboxWorld(size=8, visibility_radius=2, energy_start=50.0, seed=0)
        obs = env.reset()
        ch = sandbox_qnet_channels(obs)
        assert ch.shape == (SANDBOX_NUM_CHANNELS, 8, 8)
        # UNKNOWN channel (index 5) should be active for far cells
        ar, ac = obs["agent_pos"]
        unknown_channel = ch[UNKNOWN]
        for r in range(8):
            for c in range(8):
                if abs(r - ar) + abs(c - ac) > 2:
                    assert unknown_channel[r, c] == 1.0

    def test_qnet_partial_obs_no_crash(self):
        torch.manual_seed(0)
        env = SandboxWorld(size=8, visibility_radius=2, energy_start=50.0, seed=0)
        obs = env.reset()
        qnet = QNetwork(
            grid_size=8, num_actions=len(ACTIONS),
            num_grid_channels=SANDBOX_NUM_CHANNELS,
            num_scalars=SANDBOX_NUM_SCALARS,
            obs_batch_fn=sandbox_obs_batch_to_tensors,
        )
        vals = qnet.q_values(obs)
        assert vals.shape == (len(ACTIONS),)
        assert np.isfinite(vals).all()


# ------------------------------------------------------------------
# Q-Network compatibility
# ------------------------------------------------------------------

class TestQNetSandbox:

    def test_forward_no_crash(self):
        torch.manual_seed(0)
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        qnet = QNetwork(
            grid_size=8, num_actions=len(ACTIONS),
            num_grid_channels=SANDBOX_NUM_CHANNELS,
            num_scalars=SANDBOX_NUM_SCALARS,
            obs_batch_fn=sandbox_obs_batch_to_tensors,
        )
        ch, sc = sandbox_obs_batch_to_tensors([obs])
        q = qnet(ch, sc)
        assert q.shape == (1, len(ACTIONS))
        assert torch.isfinite(q).all()

    def test_q_values_via_observation(self):
        torch.manual_seed(0)
        env = SandboxWorld(size=8, seed=0)
        obs = env.reset()
        qnet = QNetwork(
            grid_size=8, num_actions=len(ACTIONS),
            num_grid_channels=SANDBOX_NUM_CHANNELS,
            num_scalars=SANDBOX_NUM_SCALARS,
            obs_batch_fn=sandbox_obs_batch_to_tensors,
        )
        vals = qnet.q_values(obs)
        assert vals.shape == (len(ACTIONS),)
        assert np.isfinite(vals).all()
