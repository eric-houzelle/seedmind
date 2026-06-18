"""Tests for sequence sampling (RSSM stage 2, brick 4a).

Recurrent/BPTT training needs contiguous transition sequences from a single
life, in order, never crossing an episode boundary or a death mid-sequence.
"""
from __future__ import annotations

from seedmind.memory.experience_buffer import ExperienceBuffer, make_experience


def _exp(episode_id, step, done=False):
    return make_experience(
        episode_id=episode_id, world_id="w", step=step, observation=None,
        action="MOVE_UP", next_observation=None, reward_external=0.0,
        reward_intrinsic=0.0, goal="g", prediction_error=0.0, done=done,
        action_index=0,
    )


def _fill(buf, episode_id, n, done_at=None):
    for s in range(1, n + 1):
        buf.add(_exp(episode_id, s, done=(s == done_at)))


class TestSampleSequences:
    def test_empty_buffer(self):
        assert ExperienceBuffer(seed=0).sample_sequences(4, 3) == []

    def test_sequences_have_requested_length(self):
        buf = ExperienceBuffer(seed=0)
        _fill(buf, "life_0001", 50)
        seqs = buf.sample_sequences(batch_size=8, seq_len=5)
        assert seqs, "expected some sequences"
        assert all(len(s) == 5 for s in seqs)

    def test_sequences_are_contiguous_and_in_order(self):
        buf = ExperienceBuffer(seed=1)
        _fill(buf, "life_0001", 40)
        for s in buf.sample_sequences(8, 6):
            steps = [t["step"] for t in s]
            assert steps == list(range(steps[0], steps[0] + 6))  # consecutive

    def test_never_crosses_episode_boundary(self):
        buf = ExperienceBuffer(seed=2)
        _fill(buf, "life_0001", 20)
        _fill(buf, "life_0002", 20)
        for s in buf.sample_sequences(16, 5):
            ids = {t["episode_id"] for t in s}
            assert len(ids) == 1  # single life per sequence

    def test_does_not_span_death_midsequence(self):
        # Life ends (done) at step 10; no full 5-seq may include steps 8-12.
        buf = ExperienceBuffer(seed=3)
        _fill(buf, "life_0001", 10, done_at=10)
        _fill(buf, "life_0002", 10)  # different life continues numbering at 1
        for s in buf.sample_sequences(32, 5):
            done_positions = [i for i, t in enumerate(s) if t["done"]]
            # a done may only appear as the LAST element of a sequence
            assert all(p == len(s) - 1 for p in done_positions)

    def test_seq_len_one_returns_singletons(self):
        buf = ExperienceBuffer(seed=4)
        _fill(buf, "life_0001", 10)
        seqs = buf.sample_sequences(5, 1)
        assert all(len(s) == 1 for s in seqs)

    def test_short_episode_yields_no_full_sequence(self):
        buf = ExperienceBuffer(seed=5)
        _fill(buf, "life_0001", 3)  # only 3 transitions
        assert buf.sample_sequences(8, 5) == []
