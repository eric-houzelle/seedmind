"""Agent (SPEC sections 3 & 17).

Orchestrates the modules for a single decision step: encode -> retrieve
memories -> choose goal -> choose action (optionally planning with the World
Model). The agent never reads world rules directly; it only sees observations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from seedmind.agent.curiosity import CuriosityModule
from seedmind.agent.encoder import Encoder
from seedmind.agent.goal_generator import GoalGenerator
from seedmind.agent.planner import Planner
from seedmind.agent.policy import EpsilonGreedyPolicy
from seedmind.agent.q_network import QNetwork
from seedmind.agent.value_model import ValueModel
from seedmind.agent.world_model import WorldModel
from seedmind.memory.persistent_memory import PersistentMemory


class Agent:
    def __init__(
        self,
        encoder: Encoder,
        world_model: WorldModel,
        curiosity: CuriosityModule,
        goal_generator: GoalGenerator,
        policy: EpsilonGreedyPolicy,
        memory: PersistentMemory,
        actions: List[str],
        memory_top_k: int = 5,
        use_planner: bool = True,
        q_network: Optional[QNetwork] = None,
        planning_weight: float = 0.0,
        planner_horizon: int = 4,
        planner_samples: int = 16,
        causal_feature_weights: Optional[np.ndarray] = None,
        causal_feature_targets: Optional[np.ndarray] = None,
        causal_features_fn: Optional[Any] = None,
        value_model: Optional[ValueModel] = None,
        planner_terminal_value_weight: float = 0.0,
        planner_objective_scorer: Optional[Any] = None,
        planner_objective_weight: float = 0.0,
        planner_action_penalties: Optional[Dict[str, float]] = None,
        planner_force_feature_indices: Optional[List[int]] = None,
        planner_force_feature_thresholds: Optional[List[float]] = None,
        planner_seed: Optional[int] = None,
        planner_uncertainty_threshold: Optional[float] = None,
        planner_margin_threshold: float = 0.0,
        planner_q_advantage_threshold: float = 0.0,
    ) -> None:
        self.encoder = encoder
        self.world_model = world_model
        self.curiosity = curiosity
        self.goal_generator = goal_generator
        self.policy = policy
        self.memory = memory
        self.actions = actions
        self.action_index = {a: i for i, a in enumerate(actions)}
        self.memory_top_k = memory_top_k
        self.use_planner = use_planner
        self.planner = Planner(
            world_model, actions, curiosity,
            horizon=planner_horizon, num_samples=planner_samples,
            causal_feature_weights=causal_feature_weights,
            causal_feature_targets=causal_feature_targets,
            value_model=value_model,
            terminal_value_weight=planner_terminal_value_weight,
            objective_scorer=planner_objective_scorer,
            objective_weight=planner_objective_weight,
            action_penalties=planner_action_penalties,
            seed=planner_seed,
        )
        self.q_network = q_network
        self.value_model = value_model
        # When both Q-network and planner are active, the greedy branch
        # uses: score = (1 - planning_weight) * Q + planning_weight * WM
        self.planning_weight = planning_weight
        self.causal_features_fn = causal_features_fn
        self.planner_uncertainty_threshold = planner_uncertainty_threshold
        self.planner_margin_threshold = float(planner_margin_threshold)
        self.planner_q_advantage_threshold = float(planner_q_advantage_threshold)
        self.planner_force_feature_indices = list(planner_force_feature_indices or [])
        self.planner_force_feature_thresholds = [
            float(v) for v in (planner_force_feature_thresholds or [])
        ]
        self.last_planner_used = False

    # ------------------------------------------------------------------
    # Decision pieces (used by the main loop, SPEC section 17)
    # ------------------------------------------------------------------
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        return self.encoder.encode(observation)

    def retrieve(self, latent_state: np.ndarray) -> List[Dict[str, Any]]:
        return self.memory.retrieve(latent_state, top_k=self.memory_top_k)

    def choose_goal(self, latent_state: np.ndarray, memories: List[Dict[str, Any]]) -> str:
        return self.goal_generator.choose(latent_state, memories)

    def choose_action(
        self,
        latent_state: np.ndarray,
        goal: str,
        memories: List[Dict[str, Any]],
        available_actions: List[str],
        observation: Optional[Dict[str, Any]] = None,
    ) -> str:
        scorer = None
        self.last_planner_used = False

        has_q = self.q_network is not None and observation is not None
        has_wm = self.use_planner and self.planning_weight > 0

        if has_q and has_wm:
            q_scorer = self.q_network.make_scorer(observation, available_actions)
            current_features = (
                self.causal_features_fn(observation)
                if self.causal_features_fn is not None and observation is not None
                else None
            )
            wm_values, wm_stats = self.planner.action_values_with_stats(
                latent_state, available_actions, current_features=current_features,
            )
            force_planner = self._planner_force_allows(current_features)
            q_vals = {a: q_scorer(a) for a in available_actions}
            # Normalise each score set to [0, 1] to make the weight meaningful
            q_arr = np.array([q_vals[a] for a in available_actions])
            wm_arr = np.array([wm_values[a] for a in available_actions])
            q_norm = self._normalise(q_arr)
            wm_norm = self._normalise(wm_arr)
            alpha = self.planning_weight
            if not force_planner and not self._planner_gate_allows(available_actions, wm_arr, wm_stats):
                alpha = 0.0
            elif not force_planner and not self._planner_q_advantage_allows(q_norm, wm_norm):
                alpha = 0.0
            else:
                self.last_planner_used = alpha > 0.0
            combined = {a: float((1 - alpha) * q_norm[i] + alpha * wm_norm[i])
                        for i, a in enumerate(available_actions)}
            scorer = lambda action: combined[action]
        elif has_q:
            scorer = self.q_network.make_scorer(observation, available_actions)
        elif self.use_planner:
            current_features = (
                self.causal_features_fn(observation)
                if self.causal_features_fn is not None and observation is not None
                else None
            )
            values = self.planner.action_values(
                latent_state, available_actions, current_features=current_features,
            )
            self.last_planner_used = True
            scorer = lambda action: values.get(action, float("-inf"))

        return self.policy.choose(
            latent_state=latent_state,
            goal=goal,
            memories=memories,
            available_actions=available_actions,
            action_scorer=scorer,
        )

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        rng = arr.max() - arr.min()
        if rng < 1e-8:
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / rng

    def _planner_gate_allows(
        self,
        actions: List[str],
        wm_values: np.ndarray,
        wm_stats: Dict[str, Dict[str, float]],
    ) -> bool:
        if len(actions) == 0:
            return False
        order = np.argsort(wm_values)
        best_idx = int(order[-1])
        if len(order) > 1:
            margin = float(wm_values[order[-1]] - wm_values[order[-2]])
        else:
            margin = float("inf")
        if margin < self.planner_margin_threshold:
            return False
        if self.planner_uncertainty_threshold is None:
            return True
        best_action = actions[best_idx]
        uncertainty = float(wm_stats.get(best_action, {}).get("uncertainty", float("inf")))
        return uncertainty <= self.planner_uncertainty_threshold

    def _planner_q_advantage_allows(self, q_norm: np.ndarray, wm_norm: np.ndarray) -> bool:
        if self.planner_q_advantage_threshold <= 0.0:
            return True
        if q_norm.size == 0 or wm_norm.size == 0:
            return False
        q_best_idx = int(np.argmax(q_norm))
        wm_best_idx = int(np.argmax(wm_norm))
        advantage = float(wm_norm[wm_best_idx] - wm_norm[q_best_idx])
        return advantage >= self.planner_q_advantage_threshold

    def _planner_force_allows(self, current_features: Optional[np.ndarray]) -> bool:
        if current_features is None:
            return False
        if len(self.planner_force_feature_indices) != len(self.planner_force_feature_thresholds):
            return False
        features = np.asarray(current_features, dtype=np.float32)
        for idx, threshold in zip(
            self.planner_force_feature_indices,
            self.planner_force_feature_thresholds,
        ):
            if 0 <= idx < features.shape[0] and float(features[idx]) <= threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        actions: List[str],
        grid_size: int,
        use_planner: bool = True,
        learned_policy: bool = False,
        seed: Optional[int] = None,
    ) -> "Agent":
        agent_cfg = config.get("agent", {})
        wm_cfg = config.get("world_model", {})
        cur_cfg = config.get("curiosity", {})
        pol_cfg = config.get("policy", {})
        dqn_cfg = config.get("dqn", {})

        latent_dim = int(agent_cfg.get("latent_dim", 128))

        encoder = Encoder(grid_size=grid_size, latent_dim=latent_dim, seed=seed or 0)
        world_model = WorldModel(
            latent_dim=latent_dim,
            num_actions=len(actions),
            hidden_dim=int(wm_cfg.get("hidden_dim", 256)),
            num_layers=int(wm_cfg.get("num_layers", 2)),
        )
        curiosity = CuriosityModule(
            weight=float(cur_cfg.get("weight", 0.1)),
            max_reward=float(cur_cfg.get("max_reward", 1.0)),
            enabled=bool(cur_cfg.get("enabled", True)),
        )
        goal_generator = GoalGenerator(seed=seed)
        policy = EpsilonGreedyPolicy(
            epsilon_start=float(pol_cfg.get("epsilon_start", 1.0)),
            epsilon_end=float(pol_cfg.get("epsilon_end", 0.1)),
            epsilon_decay_steps=int(pol_cfg.get("epsilon_decay_steps", 10_000)),
            seed=seed,
        )
        memory = PersistentMemory()

        q_network = None
        if learned_policy:
            q_network = QNetwork(
                grid_size=grid_size,
                num_actions=len(actions),
                conv_channels=int(dqn_cfg.get("conv_channels", 32)),
                hidden_dim=int(dqn_cfg.get("hidden_dim", 128)),
            )

        return cls(
            encoder=encoder,
            world_model=world_model,
            curiosity=curiosity,
            goal_generator=goal_generator,
            policy=policy,
            memory=memory,
            actions=actions,
            memory_top_k=int(agent_cfg.get("memory_top_k", 5)),
            use_planner=use_planner,
            q_network=q_network,
        )
