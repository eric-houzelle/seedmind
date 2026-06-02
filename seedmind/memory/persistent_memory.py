"""Persistent long-term memory (SPEC section 10).

Separate from the training buffer, this stores *important* experiences as
memory items with an embedding, and supports simple NumPy vector retrieval.
Later this can be swapped for FAISS or Chroma without changing the interface.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of a matrix."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query.astype(np.float32)
    m = matrix.astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-8
    m_norm = np.linalg.norm(m, axis=1) + 1e-8
    return (m @ q) / (m_norm * q_norm)


class PersistentMemory:
    """Long-term memory with NumPy cosine retrieval and decay."""

    def __init__(
        self,
        novelty_threshold: float = 0.2,
        reward_threshold: float = 0.5,
        decay_rate: float = 0.99,
        min_confidence: float = 0.05,
    ) -> None:
        self.novelty_threshold = novelty_threshold
        self.reward_threshold = reward_threshold
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence

        self._items: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # cached (N, D) matrix
        self._counter = 0

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------
    def _rebuild_cache(self) -> None:
        if self._items:
            self._embeddings = np.stack(
                [np.asarray(it["state_embedding"], dtype=np.float32) for it in self._items]
            )
        else:
            self._embeddings = None

    # ------------------------------------------------------------------
    # Core API (SPEC section 10)
    # ------------------------------------------------------------------
    def store(self, memory_item: Dict[str, Any]) -> str:
        self._counter += 1
        memory_id = memory_item.get("memory_id") or f"mem_{self._counter:05d}"
        item = dict(memory_item)
        item["memory_id"] = memory_id
        item.setdefault("uses", 0)
        item.setdefault("confidence", 0.5)
        item.setdefault("utility", 0.0)
        item.setdefault("novelty", 0.0)
        item["state_embedding"] = np.asarray(item["state_embedding"], dtype=np.float32)
        self._items.append(item)
        self._rebuild_cache()
        return memory_id

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._items:
            return []
        sims = _cosine_similarity(np.asarray(query_embedding), self._embeddings)
        order = np.argsort(-sims)[:top_k]
        results = []
        for i in order:
            item = self._items[int(i)]
            item["uses"] += 1
            result = dict(item)
            result["similarity"] = float(sims[int(i)])
            results.append(result)
        return results

    def update_confidence(self, memory_id: str, delta: float) -> None:
        for item in self._items:
            if item["memory_id"] == memory_id:
                item["confidence"] = float(np.clip(item["confidence"] + delta, 0.0, 1.0))
                return

    def decay_old_memories(self) -> int:
        """Multiply confidences by the decay rate and drop weak memories.

        Returns the number of memories removed.
        """
        before = len(self._items)
        for item in self._items:
            item["confidence"] *= self.decay_rate
        self._items = [it for it in self._items if it["confidence"] >= self.min_confidence]
        self._rebuild_cache()
        return before - len(self._items)

    # ------------------------------------------------------------------
    # Importance heuristic
    # ------------------------------------------------------------------
    def novelty_of(self, embedding: np.ndarray) -> float:
        """Novelty = 1 - max cosine similarity to existing memories."""
        if not self._items:
            return 1.0
        sims = _cosine_similarity(np.asarray(embedding), self._embeddings)
        return float(1.0 - np.max(sims))

    def store_if_important(self, experience: Dict[str, Any]) -> Optional[str]:
        """Store an experience as a memory if it is reward- or novelty-worthy."""
        embedding = experience.get("latent_state")
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32)

        reward = experience.get("reward_external", 0.0)
        novelty = self.novelty_of(embedding)
        important = (
            abs(reward) >= self.reward_threshold
            or novelty >= (1.0 - self.novelty_threshold)
            or experience.get("done", False) and reward > 0
        )
        if not important:
            return None

        item = {
            "world_type": experience.get("world_id", "unknown"),
            "state_embedding": embedding,
            "summary": experience.get("goal", ""),
            "action": experience.get("action", ""),
            "result": experience.get("next_observation_summary", experience.get("goal", "")),
            "utility": float(reward),
            "novelty": float(novelty),
            "confidence": 0.6,
            "uses": 0,
        }
        return self.store(item)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"items": self._items, "counter": self._counter}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._items = state["items"]
        self._counter = state.get("counter", len(self._items))
        for it in self._items:
            it["state_embedding"] = np.asarray(it["state_embedding"], dtype=np.float32)
        self._rebuild_cache()
