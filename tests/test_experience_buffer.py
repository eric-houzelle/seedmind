import numpy as np

from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience


def _exp(step: int, error: float, reward: float = 0.0) -> dict:
    return make_experience(
        episode_id="episode_000001",
        world_id="gridworld_v1",
        step=step,
        observation="o",
        action="MOVE_RIGHT",
        next_observation="o2",
        reward_external=reward,
        reward_intrinsic=0.1,
        goal="explore_unknown_area",
        prediction_error=error,
        done=False,
        latent_state=np.zeros(4, dtype=np.float32),
        next_latent_state=np.ones(4, dtype=np.float32),
        action_index=3,
    )


def _event_exp(step: int, event: str) -> dict:
    return make_experience(
        episode_id="episode_000001",
        world_id="sandbox",
        step=step,
        observation="o",
        action="HARVEST",
        next_observation="o2",
        reward_external=0.0,
        reward_intrinsic=0.0,
        goal="explore",
        prediction_error=0.0,
        done=False,
        event=event,
        event_amount=1,
    )


def test_make_experience_schema():
    e = _exp(0, 0.5)
    for key in [
        "episode_id", "world_id", "step", "observation", "action",
        "next_observation", "reward_external", "reward_intrinsic", "goal",
        "prediction_error", "memory_used", "done", "timestamp",
    ]:
        assert key in e


def test_make_experience_stores_causal_event():
    e = _event_exp(0, "craft_tool")
    assert e["event"] == "craft_tool"
    assert e["event_amount"] == 1


def test_add_and_len():
    buf = ExperienceBuffer(seed=0)
    for i in range(5):
        buf.add(_exp(i, 0.1 * i))
    assert len(buf) == 5


def test_sample_respects_batch_size():
    buf = ExperienceBuffer(seed=0)
    for i in range(10):
        buf.add(_exp(i, 0.1))
    assert len(buf.sample(4)) == 4
    assert len(buf.sample(100)) == 10


def test_sample_high_error_orders_by_error():
    buf = ExperienceBuffer(seed=0)
    for i in range(10):
        buf.add(_exp(i, error=float(i)))
    top = buf.sample_high_error(3)
    errors = [e["prediction_error"] for e in top]
    assert errors == sorted(errors, reverse=True)
    assert errors[0] == 9.0


def test_capacity_is_a_ring_buffer():
    buf = ExperienceBuffer(capacity=3, seed=0)
    for i in range(5):
        buf.add(_exp(i, 0.1))
    assert len(buf) == 3


def test_save_and_load(tmp_path):
    buf = ExperienceBuffer(seed=0)
    for i in range(6):
        buf.add(_exp(i, 0.2))
    path = tmp_path / "buffer.pkl"
    buf.save(str(path))

    other = ExperienceBuffer()
    other.load(str(path))
    assert len(other) == 6


def test_sample_causal_returns_event_transitions():
    buf = ExperienceBuffer(seed=0)
    for i in range(5):
        buf.add(_event_exp(i, "wait"))
    buf.add(_event_exp(10, "craft_tool"))
    buf.add(_event_exp(11, "harvest_food_tool"))
    sample = buf.sample_causal(10)
    events = {e["event"] for e in sample}
    assert events <= {"craft_tool", "harvest_food_tool"}
    assert events
