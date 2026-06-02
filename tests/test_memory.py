import numpy as np

from seedmind.memory.persistent_memory import PersistentMemory


def _item(vec, utility=0.5):
    return {
        "world_type": "gridworld",
        "state_embedding": np.asarray(vec, dtype=np.float32),
        "summary": "test",
        "action": "INTERACT",
        "result": "door open",
        "utility": utility,
    }


def test_store_increases_length_and_returns_id():
    mem = PersistentMemory()
    mid = mem.store(_item([1.0, 0.0, 0.0]))
    assert mid.startswith("mem_")
    assert len(mem) == 1


def test_retrieve_returns_most_similar_first():
    mem = PersistentMemory()
    mem.store(_item([1.0, 0.0, 0.0]))
    mem.store(_item([0.0, 1.0, 0.0]))
    mem.store(_item([0.0, 0.0, 1.0]))

    results = mem.retrieve(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    # The closest embedding is [1,0,0].
    assert np.allclose(results[0]["state_embedding"], [1.0, 0.0, 0.0])


def test_update_confidence():
    mem = PersistentMemory()
    mid = mem.store(_item([1.0, 0.0, 0.0]))
    mem.update_confidence(mid, 0.3)
    item = mem._items[0]
    assert item["confidence"] > 0.5


def test_decay_removes_weak_memories():
    mem = PersistentMemory(decay_rate=0.0, min_confidence=0.1)
    mem.store(_item([1.0, 0.0, 0.0]))
    removed = mem.decay_old_memories()
    assert removed == 1
    assert len(mem) == 0


def test_store_if_important_on_high_reward():
    mem = PersistentMemory(reward_threshold=0.5)
    experience = {
        "latent_state": np.array([0.2, 0.4, 0.1], dtype=np.float32),
        "reward_external": 1.0,
        "goal": "reach_visible_reward",
        "action": "MOVE_RIGHT",
        "world_id": "gridworld_v1",
        "done": True,
    }
    mid = mem.store_if_important(experience)
    assert mid is not None
    assert len(mem) == 1


def test_save_and_load(tmp_path):
    mem = PersistentMemory()
    mem.store(_item([1.0, 0.0, 0.0]))
    mem.store(_item([0.0, 1.0, 0.0]))
    path = tmp_path / "memory.pkl"
    mem.save(str(path))

    other = PersistentMemory()
    other.load(str(path))
    assert len(other) == 2
    results = other.retrieve(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
