"""Configurable objective scorers for planner rollouts.

Objectives are intentionally separate from the World Model: the model predicts
what happens, while an objective says which predicted states are valuable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class SurvivalObjective:
    """Score organism drive features for survival-oriented planning."""

    feature_names: Sequence[str]
    weights: Dict[str, float]
    critical_threshold: float = 0.15
    risk_weight: float = 2.0
    temperature_target: float = 0.5
    mode: str = "absolute"
    decline_weight: float = 0.0
    critical_decline_weight: float = 0.0

    def score_batch(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        scores = np.zeros(arr.shape[0], dtype=np.float32)
        total_weight = 0.0
        risks = np.zeros(arr.shape[0], dtype=np.float32)

        for name, weight in self.weights.items():
            if weight == 0.0 or name not in self.feature_names:
                continue
            idx = self.feature_names.index(name)
            values = np.clip(arr[:, idx], 0.0, 1.0)
            if name == "temperature":
                values = np.clip(
                    1.0 - np.abs(values - self.temperature_target) * 2.0,
                    0.0,
                    1.0,
                )
            scores += float(weight) * values
            total_weight += abs(float(weight))
            if name in {"energy", "hydration", "health"}:
                risks += np.square(np.maximum(0.0, self.critical_threshold - values))

        if total_weight > 0.0:
            scores = scores / total_weight
        if self.risk_weight:
            scores -= float(self.risk_weight) * risks
        return scores.astype(np.float32)

    def score_transition_batch(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        before_arr = np.asarray(before, dtype=np.float32)
        after_arr = np.asarray(after, dtype=np.float32)
        if self.mode != "delta":
            return self.score_batch(after_arr)

        delta = self.score_batch(after_arr) - self.score_batch(before_arr)
        decline_penalty = np.zeros(after_arr.shape[0], dtype=np.float32)
        critical_penalty = np.zeros(after_arr.shape[0], dtype=np.float32)
        total_weight = 0.0
        for name, weight in self.weights.items():
            if name not in {"energy", "hydration", "health"}:
                continue
            if weight == 0.0 or name not in self.feature_names:
                continue
            idx = self.feature_names.index(name)
            w = abs(float(weight))
            before_values = np.clip(before_arr[:, idx], 0.0, 1.0)
            after_values = np.clip(after_arr[:, idx], 0.0, 1.0)
            decline = np.maximum(0.0, before_values - after_values)
            decline_penalty += w * decline
            critical_gap = np.maximum(0.0, self.critical_threshold - after_values)
            critical_penalty += w * decline * critical_gap
            total_weight += w
        if total_weight > 0.0:
            decline_penalty /= total_weight
            critical_penalty /= total_weight
        return (
            delta
            - float(self.decline_weight) * decline_penalty
            - float(self.critical_decline_weight) * critical_penalty
        ).astype(np.float32)


def build_objective_scorer(
    config: dict[str, Any],
    feature_names: Sequence[str],
) -> SurvivalObjective | None:
    objective = config.get("objective", {})
    if not bool(objective.get("enabled", False)):
        return None
    objective_type = str(objective.get("type", ""))
    if objective_type != "survival_v0":
        raise ValueError(f"Unknown objective type: {objective_type}")
    return SurvivalObjective(
        feature_names=list(feature_names),
        weights={
            "energy": 1.0,
            "hydration": 2.0,
            "temperature": 0.5,
            "health": 3.0,
            **dict(objective.get("weights", {})),
        },
        critical_threshold=float(objective.get("critical_threshold", 0.15)),
        risk_weight=float(objective.get("risk_weight", 2.0)),
        temperature_target=float(objective.get("temperature_target", 0.5)),
        mode=str(objective.get("mode", "absolute")),
        decline_weight=float(objective.get("decline_weight", 0.0)),
        critical_decline_weight=float(objective.get("critical_decline_weight", 0.0)),
    )
