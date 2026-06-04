"""Universal environment interface for SeedMind.

Every world must implement this interface so that the agent only ever
interacts through ``observation -> action -> consequence`` and never calls
internal world functions directly (see SPEC sections 2 and 7).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class EnvironmentAdapter(ABC):
    """Abstract base class shared by every SeedMind world."""

    @abstractmethod
    def reset(self) -> Any:
        """Reset the world and return the first observation."""
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Any:
        """Return the current observation of the agent."""
        raise NotImplementedError

    @abstractmethod
    def available_actions(self) -> List[str]:
        """Return the actions possible in the current state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply an action.

        Returns ``(next_observation, reward, done, info)``.
        """
        raise NotImplementedError

    def describe_transition(self) -> str:
        """Optional human-readable description of the last transition.

        Useful for debugging, logs and analysis. Default: empty string.
        """
        return ""

    def causal_feature_names(self) -> List[str]:
        """Optional names for observable causal state features.

        Worlds may expose a compact, structured perception vector so a generic
        World Model can learn action consequences outside the opaque latent.
        The agent treats these as anonymous feature indices.
        """
        return []

    def causal_features(self, observation: Any) -> np.ndarray:
        """Optional causal feature vector for an observation."""
        return np.zeros(0, dtype=np.float32)

    def causal_event_names(self) -> List[str]:
        """Optional event vocabulary for transition classification."""
        return []
