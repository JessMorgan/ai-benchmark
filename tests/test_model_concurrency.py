"""Tests for opt-in per-source target/model concurrency."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from unittest import mock

from benchmark.cli import (
    SourceJudgeWorkerPool,
    SourceModelScheduler,
    _build_runner_queues,
    _configure_judge_source,
    _start_runner_pipeline,
)
from benchmark.core import resolve_model_thread_limit


def test_model_thread_limit_resolution_and_strict_validation():
    sources = {"Local": {"model_thread_limit": 2}, "Cloud": {}}
    assert resolve_model_thread_limit(sources, "Local", 7) == 2
    assert resolve_model_thread_limit(sources, "Cloud", 3) == 3
    assert resolve_model_thread_limit(sources, "Missing", 1) == 1

    for invalid in (0, -1, True, 1.5, "2"):
        with mock.patch.dict(sources, {"Bad": {"model_thread_limit": invalid}}, clear=False):
            try:
                resolve_model_thread_limit(sources, "Bad", 1)
            except ValueError as exc:
                assert "Bad" in str(exc)
                assert "positive integer" in str(exc)
            else:  # pragma: no cover - assertion gives a clearer failure
                raise AssertionError(f"expected ValueError for {invalid!r}")


def test_source_scheduler_calls_completion_after_all_targets():
    events = []
    lock = threading.Lock()

    def run_target(name):
        with lock:
            events.append(("start", name))
        time.sleep(0.01)
        with lock:
            events.append(("end", name))

    scheduler = SourceModelScheduler(
        "Cloud", 2, ["a", "b", "c"], run_target,
        threading.Event(), lambda *_args: None,
        on_complete=lambda source: events.append(("complete", source)),
    )
    scheduler.run_until_drained()

    assert events[-1] == ("complete", "Cloud")
    assert {name for event, name in events if event == "end"} == {"a", "b", "c"}


def test_source_scheduler_is_fifo_and_bounded():
    active = 0
    peak = 0
    calls = []
    lock = threading.Lock()

    def run_target(name):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append(name)
        time.sleep(0.01)
        with lock:
            active -= 1

    scheduler = SourceModelScheduler(
        "Cloud", 2, ["a", "b", "c", "d"], run_target,
        threading.Event(), lambda *_args: None,
    )
    scheduler.run_until_drained()

    assert peak == 2
    assert set(calls) == {"a", "b", "c", "d"}
    assert len(calls) == 4


def test_source_schedulers_have_independent_limits_and_failures_refill():
    active = defaultdict(int)
    peak = defaultdict(int)
    calls = []
    lock = threading.Lock()

    def run_source(source, name):
        with lock:
            active[source] += 1
            peak[source] = max(peak[source], active[source])
            calls.append((source, name))
        try:
            if name == "bad":
                raise RuntimeError("expected target failure")
            time.sleep(0.01)
        finally:
            with lock:
                active[source] -= 1

    stop = threading.Event()
    errors = []
    schedulers = [
        SourceModelScheduler(
            "Local", 1, ["local-a", "local-b"],
            lambda name: run_source("Local", name), stop,
            lambda target, _runner, error: errors.append((target, error)),
        ),
        SourceModelScheduler(
            "Cloud", 2, ["bad", "cloud-b", "cloud-c"],
            lambda name: run_source("Cloud", name), stop,
            lambda target, _runner, error: errors.append((target, error)),
        ),
    ]
    threads = [threading.Thread(target=scheduler.run_until_drained) for scheduler in schedulers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert peak["Local"] == 1
    assert peak["Cloud"] == 2
    assert {name for source, name in calls if source == "Cloud"} == {"bad", "cloud-b", "cloud-c"}
    assert any(item[0] == "bad" for item in errors if isinstance(item, tuple))


def test_source_scheduler_cancellation_stops_new_submissions():
    stop = threading.Event()
    calls = []

    def run_target(name):
        calls.append(name)
        stop.set()
        time.sleep(0.02)

    scheduler = SourceModelScheduler(
        "Local", 1, ["a", "b", "c"], run_target,
        stop, lambda *_args: None,
    )
    scheduler.run_until_drained()

    assert calls == ["a"]


def test_judge_source_reservation_configures_one_then_full_pool():
    stop = threading.Event()
    pool = SourceJudgeWorkerPool("Cloud", 3, lambda _job: None, stop)
    limits = {"Cloud": 3}

    _configure_judge_source(limits, "Cloud", 3, True, pool)
    assert limits["Cloud"] == 2
    assert pool.thread_count == 1
    pool.expand_full()
    assert pool.thread_count == 3
    pool.stop(timeout=1)


def test_judge_pool_expands_after_source_benchmark_completion():
    stop = threading.Event()
    benchmark_release = threading.Event()
    benchmark_started = threading.Event()
    benchmark_both_active = threading.Event()
    judge_started = threading.Event()
    judge_release = threading.Event()
    full_pool_active = threading.Event()
    lock = threading.Lock()
    active_benchmarks = 0
    peak_benchmarks = 0
    active_judges = 0
    peak_judges = 0
    judge_calls = []

    def process_judge(job):
        nonlocal active_judges, peak_judges
        with lock:
            active_judges += 1
            peak_judges = max(peak_judges, active_judges)
            judge_calls.append(job)
            judge_started.set()
            if active_judges == 3:
                full_pool_active.set()
        judge_release.wait(timeout=2)
        with lock:
            active_judges -= 1

    def run_benchmark(_name):
        nonlocal active_benchmarks, peak_benchmarks
        with lock:
            active_benchmarks += 1
            peak_benchmarks = max(peak_benchmarks, active_benchmarks)
            benchmark_started.set()
            if active_benchmarks == 2:
                benchmark_both_active.set()
        benchmark_release.wait(timeout=2)
        with lock:
            active_benchmarks -= 1

    full_limit = 3
    benchmark_limit = full_limit - 1
    pool = SourceJudgeWorkerPool("Cloud", full_limit, process_judge, stop)
    pool.start(1)
    assert pool.thread_count == 1
    for job in range(3):
        pool.enqueue(job)

    benchmark = SourceModelScheduler(
        "Cloud", benchmark_limit, ["model-a", "model-b"], run_benchmark,
        stop, lambda *_args: None,
        on_complete=lambda _source: pool.expand_full(),
    )
    benchmark_thread = threading.Thread(target=benchmark.run_until_drained)
    benchmark_thread.start()

    assert benchmark_started.wait(timeout=1)
    assert benchmark_both_active.wait(timeout=1)
    assert judge_started.wait(timeout=1)
    # The overlap phase uses two benchmark slots plus the one reserved judge
    # slot, never exceeding the source's full limit of three.
    with lock:
        assert active_benchmarks == benchmark_limit
        assert peak_benchmarks == benchmark_limit
        assert active_judges == 1
        assert peak_judges == 1
        assert active_benchmarks + active_judges == full_limit
    assert pool.thread_count == 1

    benchmark_release.set()
    benchmark_thread.join(timeout=2)
    assert not benchmark_thread.is_alive()

    assert full_pool_active.wait(timeout=1)
    with lock:
        assert peak_judges == full_limit
        assert active_benchmarks == 0
        assert active_judges == full_limit
    assert pool.thread_count == full_limit
    assert len(judge_calls) == 3

    judge_release.set()
    pool.stop(timeout=2)
    assert active_judges == 0


def test_runner_pipeline_notifies_source_completion_for_judge_expansion():
    completed = []
    stop = threading.Event()

    threads = _start_runner_pipeline(
        {"Cloud": ["a", "b"]},
        {"Cloud": ["a", "b"]},
        {"Cloud": {"a", "b"}},
        lambda _name, _runner: time.sleep(0.01),
        stop,
        lambda *_args: None,
        model_thread_limits={"Cloud": 2},
        source_complete_callback=completed.append,
    )
    for thread in threads:
        thread.join(timeout=2)

    assert completed == ["Cloud"]


def test_runner_pipeline_applies_source_limit_and_preserves_both_order():
    active = 0
    peak = 0
    calls = []
    lock = threading.Lock()

    def run_target(name, runner):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((name, runner, "start"))
        time.sleep(0.01)
        with lock:
            calls.append((name, runner, "end"))
            active -= 1

    stop = threading.Event()
    peaks = []
    threads = _start_runner_pipeline(
        {"Cloud": ["a", "b", "c"]},
        {"Cloud": ["a", "b", "c"]},
        {"Cloud": {"a", "b", "c"}},
        run_target,
        stop,
        lambda *_args: None,
        model_thread_limits={"Cloud": 2},
        peak_callback=lambda _source, current: peaks.append(current),
    )
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert peak == 2
    assert max(peaks) == 2
    for target in ("a", "b", "c"):
        target_runners = [runner for name, runner, event in calls
                          if name == target and event == "start"]
        assert target_runners == ["opencode", "http"]


def test_both_mode_queue_excludes_targets_with_no_pending_leg():
    targets = {
        "done": {"source": "Cloud"},
        "pending": {"source": "Cloud"},
    }
    snapshot = {
        "done": {"status": "completed"},
        "done [opencode]": {"status": "completed"},
        "pending": {"status": "pending"},
        "pending [opencode]": {"status": "completed"},
    }
    queues = _build_runner_queues(targets, snapshot, "both", {"Cloud": {}})
    targets_by_source, opencode_pending, http_pending = queues
    assert targets_by_source == {"Cloud": ["pending"]}
    assert opencode_pending == {"Cloud": []}
    assert http_pending == {"Cloud": {"pending"}}
