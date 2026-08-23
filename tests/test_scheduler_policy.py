"""Tests for source benchmark/judge capacity policy."""
import pytest

from benchmark.scheduler_policy import SourceSchedulingPolicy


def test_active_benchmarks_reserve_all_source_slots():
    policy = SourceSchedulingPolicy("NAS", 4)
    assert policy.capacity(benchmark_active=True) == (4, 0)
    assert policy.benchmark_has_priority(benchmark_active=True)
    assert not policy.can_start_judges(benchmark_active=True)


def test_drained_benchmarks_release_full_judge_capacity():
    policy = SourceSchedulingPolicy("NAS", 4)
    assert policy.capacity(benchmark_active=False) == (4, 4)
    assert not policy.benchmark_has_priority(benchmark_active=False)
    assert policy.can_start_judges(benchmark_active=False)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_policy_rejects_invalid_model_limits(value):
    with pytest.raises(ValueError, match="positive integer"):
        SourceSchedulingPolicy("NAS", value)
