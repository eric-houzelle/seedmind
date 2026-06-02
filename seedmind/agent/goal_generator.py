"""Goal Generator (SPEC section 14).

The agent sets itself micro-goals. V1 uses a heuristic that scores each
candidate goal with ``score = novelty + expected_utility + uncertainty`` and
samples a goal (softmax) so behaviour stays exploratory yet memory-aware.
This stays generic: it never reads the solution from the grid.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

GOALS: List[str] = [
    "explore_unknown_area",
    "interact_with_unknown_object",
    "reach_visible_reward",
    "avoid_known_danger",
    "test_uncertain_rule",
    "reuse_successful_memory",
]


class GoalGenerator:
    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def score_goals(
        self,
        latent_state: np.ndarray,
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        memories = memories or []
        # Aggregate memory signal.
        if memories:
            mean_utility = float(np.mean([m.get("utility", 0.0) for m in memories]))
            mean_similarity = float(np.mean([m.get("similarity", 0.0) for m in memories]))
            mean_novelty = float(np.mean([m.get("novelty", 0.0) for m in memories]))
        else:
            mean_utility = 0.0
            mean_similarity = 0.0
            mean_novelty = 1.0

        scores: Dict[str, float] = {}
        for goal in GOALS:
            novelty = mean_novelty
            expected_utility = 0.0
            uncertainty = float(self.rng.uniform(0.0, 0.5))

            if goal == "explore_unknown_area":
                novelty = mean_novelty + 0.3
            elif goal == "interact_with_unknown_object":
                novelty = mean_novelty + 0.2
                uncertainty += 0.2
            elif goal == "reach_visible_reward":
                expected_utility = max(0.0, mean_utility) + 0.2
            elif goal == "avoid_known_danger":
                expected_utility = max(0.0, -min(0.0, mean_utility))
            elif goal == "test_uncertain_rule":
                uncertainty += 0.3
            elif goal == "reuse_successful_memory":
                expected_utility = max(0.0, mean_utility)
                # Only attractive when we actually retrieved relevant memories.
                novelty = mean_similarity

            scores[goal] = novelty + expected_utility + uncertainty
        return scores

    def choose(
        self,
        latent_state: np.ndarray,
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        scores = self.score_goals(latent_state, memories)
        goals = list(scores.keys())
        values = np.array([scores[g] for g in goals], dtype=np.float64)
        logits = values / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return str(self.rng.choice(goals, p=probs))
