"""Tests for opt-in per-source target/model concurrency."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from unittest import mock

from benchmark.cli import (
    SourceModelScheduler,
    _build_runner_queues,
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
