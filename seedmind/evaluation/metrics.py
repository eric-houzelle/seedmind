"""Metrics logging (SPEC section 25)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EpisodeMetrics:
    episode: int
    episode_reward_external: float = 0.0
    episode_reward_intrinsic: float = 0.0
    steps_survived: int = 0
    success: bool = False
    prediction_error_mean: float = 0.0
    world_model_loss: float = 0.0
    memory_items_count: int = 0
    goal_distribution: Dict[str, int] = field(default_factory=dict)
    exploration_rate: float = 0.0
    repeated_mistakes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "episode_reward_external": self.episode_reward_external,
            "episode_reward_intrinsic": self.episode_reward_intrinsic,
            "steps_survived": self.steps_survived,
            "success": bool(self.success),
            "prediction_error_mean": self.prediction_error_mean,
            "world_model_loss": self.world_model_loss,
            "memory_items_count": self.memory_items_count,
            "goal_distribution": self.goal_distribution,
            "exploration_rate": self.exploration_rate,
            "repeated_mistakes": self.repeated_mistakes,
        }


class MetricsLogger:
    """Accumulates per-episode metrics and exposes series for plotting."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def log(self, metrics: EpisodeMetrics) -> None:
        self.records.append(metrics.to_dict())

    def series(self, key: str) -> List[float]:
        return [r[key] for r in self.records]

    def success_rate(self, window: int = 100) -> float:
        if not self.records:
            return 0.0
        recent = self.records[-window:]
        return sum(1 for r in recent if r["success"]) / len(recent)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
