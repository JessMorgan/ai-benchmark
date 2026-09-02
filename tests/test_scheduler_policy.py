"""Tests for source benchmark/judge capacity policy."""
import pytest

from benchmark.scheduler_policy import SourceSchedulingPolicy


def test_benchmarks_always_claim_full_model_limit():
    policy = SourceSchedulingPolicy("NAS", 4)
    assert policy.benchmark_slots(3) == 4
    assert policy.benchmark_slots(10) == 4
    assert policy.benchmark_slots(0) == 4
    assert policy.benchmark_has_priority()


def test_judge_slots_are_free_benchmark_model_slots():
    policy = SourceSchedulingPolicy("NAS", 4)
    # No benchmark work -> every model slot is free for judges.
    assert policy.judge_model_slots(0) == 4
    assert policy.can_start_judges(0)
    # A queue that fills the limit leaves no free model slots.
    assert policy.judge_model_slots(4) == 0
    assert not policy.can_start_judges(4)
    # More queued models than the limit still leave no free slots.
    assert policy.judge_model_slots(10) == 0
    # A short queue leaves the unused slots for judges.
    assert policy.judge_model_slots(3) == 1
    assert policy.can_start_judges(3)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_policy_rejects_invalid_model_limits(value):
    with pytest.raises(ValueError, match="positive integer"):
        SourceSchedulingPolicy("NAS", value)


def test_judge_model_slots_tolerates_bad_queue_values():
    policy = SourceSchedulingPolicy("NAS", 4)
    assert policy.judge_model_slots(None) == 4
    assert policy.judge_model_slots("banana") == 4
    assert policy.judge_model_slots(-2) == 4
