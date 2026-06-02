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
        self.planner = Planner(world_model, actions, curiosity)
        # V2: a learned action-value network. When present it drives the greedy
        # branch of the epsilon-greedy policy instead of the planner.
        self.q_network = q_network

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
        # Greedy-branch scorer priority: learned Q-network > planner > random.
        if self.q_network is not None and observation is not None:
            scorer = self.q_network.make_scorer(observation, available_actions)
        elif self.use_planner:
            scorer = self.planner.make_scorer(latent_state, available_actions)
        else:
            scorer = None
        return self.policy.choose(
            latent_state=latent_state,
            goal=goal,
            memories=memories,
            available_actions=available_actions,
            action_scorer=scorer,
        )

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
