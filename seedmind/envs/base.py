"""Universal environment interface for SeedMind.

Every world must implement this interface so that the agent only ever
interacts through ``observation -> action -> consequence`` and never calls
internal world functions directly (see SPEC sections 2 and 7).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


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
