"""Tests for opt-in per-source target/model concurrency."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from unittest import mock

from benchmark.cli import (
    SourceModelScheduler,
    _build_runner_queues,
    _resolve_judge_plugin_limit,
    _start_runner_pipeline,
)
from benchmark.core import resolve_model_thread_limit
from benchmark.scheduling import (
    PluginSlotGate,
    PluginSlotGateRegistry,
    SourceJudgeWorkerPool,
    _configure_judge_source,
)


def test_resolve_judge_plugin_limit():
    sources = {"Local": {"plugin_thread_limit": 2}, "Cloud": {}}
    assert _resolve_judge_plugin_limit(sources, "Local") == 2
    assert _resolve_judge_plugin_limit(sources, "Cloud") == 1
    assert _resolve_judge_plugin_limit(sources, "Missing") == 1
    with mock.patch.dict(sources, {"Zero": {"plugin_thread_limit": 0}}, clear=False):
        assert _resolve_judge_plugin_limit(sources, "Zero") == 1
    with mock.patch.dict(sources, {"Bad": {"plugin_thread_limit": "banana"}}, clear=False):
        assert _resolve_judge_plugin_limit(sources, "Bad") == 1
    with mock.patch.dict(sources, {"Str": {"plugin_thread_limit": "3"}}, clear=False):
        assert _resolve_judge_plugin_limit(sources, "Str") == 3


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


def test_partial_completed_row_keeps_benchmark_reservation_path_active():
    targets = {"nas-model": {"source": "NAS"}}
    snapshot = {
        "nas-model": {
            "status": "completed",
            "p1_score": 10.0,
            "p2_score": None,
        },
    }
    queues = _build_runner_queues(
        targets,
        snapshot,
        "http",
        {"NAS": {"model_thread_limit": 1}},
        plugin_ids=["p1", "p2"],
    )
    assert queues == {"NAS": ["nas-model"]}


def test_judge_source_reserves_only_slots_benchmarks_cannot_use():
    stop = threading.Event()
    pool = SourceJudgeWorkerPool("Cloud", 3, lambda _job: None, stop)
    limits = {"Cloud": 3}

    # A full benchmark queue leaves no free model slots for regular judges.
    _configure_judge_source(limits, "Cloud", 3, ["a", "b", "c"], pool)
    assert limits["Cloud"] == 3
    assert pool.model_slots == 0

    # A short queue leaves the unused model slots for judges.
    _configure_judge_source(limits, "Cloud", 3, ["a"], pool)
    assert pool.model_slots == 2
    pool.start(3)
    assert pool.model_slots == 3
    pool.stop(timeout=1)


def test_judge_pool_reports_selection_until_judge_finishes():
    stop = threading.Event()
    selected = []
    selected_started = threading.Event()

    def on_selection_change(judge, is_selected):
        selected.append((judge, is_selected))
        if is_selected:
            selected_started.set()

    pool = SourceJudgeWorkerPool(
        "Cloud", 1, lambda _job: time.sleep(0.03), stop,
        on_selection_change=on_selection_change,
    )
    pool.enqueue(("cell", None, None, None, "judge-a", True))
    pool.start(1)
    assert selected_started.wait(timeout=1)
    pool.stop(drain=True)

    assert selected == [("judge-a", True), ("judge-a", False)]


def test_judge_pool_drains_all_jobs_before_stop():
    stop = threading.Event()
    processed = []
    lock = threading.Lock()

    def process(job):
        time.sleep(0.03)
        with lock:
            processed.append(job)

    pool = SourceJudgeWorkerPool("Cloud", 2, process, stop)
    pool.start(2)
    for job in range(8):
        pool.enqueue(job)
    pool.stop(drain=True)
    assert sorted(processed) == list(range(8))
    assert pool.thread_count == 0


def test_judge_pool_runs_one_judge_to_completion_before_another():
    stop = threading.Event()
    processed = []

    def process(job):
        processed.append(job[0])

    pool = SourceJudgeWorkerPool("Cloud", 1, process, stop)
    # Retry work is enqueued first, but a judge always drains its fresh cells
    # before its retries, and one judge runs to completion (never-judged cells
    # then retries) before the next judge is loaded.
    for job in (
        ("a-retry-1", None, None, None, "judge-a", False),
        ("a-fresh", None, None, None, "judge-a", True),
        ("b-retry-1", None, None, None, "judge-b", False),
        ("b-fresh", None, None, None, "judge-b", True),
    ):
        pool.enqueue(job)
    pool.start(1)
    pool.stop(drain=True)

    assert processed == ["a-fresh", "a-retry-1", "b-fresh", "b-retry-1"]
    assert pool.thread_count == 0


def test_judge_pool_runs_at_most_model_limit_judges_concurrently():
    stop = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    both_active = threading.Event()
    release = threading.Event()

    def process(job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_active.set()
        release.wait(timeout=2)
        with lock:
            active -= 1

    pool = SourceJudgeWorkerPool("Cloud", 2, process, stop)
    for judge in ("judge-a", "judge-b", "judge-c"):
        pool.enqueue((judge, None, None, None, judge, True))
    pool.start(2)

    assert both_active.wait(timeout=2)
    with lock:
        assert active == 2
        assert peak == 2
    release.set()
    pool.stop(drain=True)
    with lock:
        assert peak == 2


def test_judge_pool_plugin_limit_scores_cells_in_parallel():
    stop = threading.Event()
    lock = threading.Lock()
    processed = []
    active = 0
    peak = 0
    both_active = threading.Event()
    release = threading.Event()

    def process(job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            processed.append(job[0])
            if active == 2:
                both_active.set()
        release.wait(timeout=2)
        with lock:
            active -= 1

    pool = SourceJudgeWorkerPool("Cloud", 1, process, stop, plugin_limit=2)
    for cell in range(4):
        pool.enqueue((cell, None, None, None, "judge-a", True))
    pool.start(1)

    assert both_active.wait(timeout=2)
    with lock:
        assert peak == 2
    release.set()
    pool.stop(drain=True)
    with lock:
        assert sorted(processed) == list(range(4))
    assert pool.thread_count == 0


def test_judge_pool_plugin_workers_survive_delayed_cells():
    """A quiet queue must not kill a cell worker before later results arrive."""
    stop = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    first_started = threading.Event()
    both_active = threading.Event()
    release = threading.Event()

    def process(_job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 1:
                first_started.set()
            elif active == 2:
                both_active.set()
        release.wait(timeout=2)
        with lock:
            active -= 1

    pool = SourceJudgeWorkerPool("Cloud", 1, process, stop, plugin_limit=2)
    pool.enqueue(("first", None, None, None, "judge-a", True))
    pool.start(1)
    assert first_started.wait(timeout=1)

    # This exceeds the cell worker's polling timeout. The second worker must
    # remain alive and take the result when it is eventually enqueued.
    time.sleep(0.35)
    pool.enqueue(("second", None, None, None, "judge-a", True))
    assert both_active.wait(timeout=1)

    release.set()
    pool.stop(drain=True)
    with lock:
        assert peak == 2
    assert pool.thread_count == 0


def test_judge_pool_continues_after_unexpected_callback_exception():
    stop = threading.Event()
    processed = []

    def process(job):
        if job == "bad":
            raise RuntimeError("unexpected callback failure")
        processed.append(job)

    pool = SourceJudgeWorkerPool("Cloud", 1, process, stop)
    for job in ("bad", "good-a", "good-b"):
        pool.enqueue(job)
    pool.start(1)
    pool.stop(drain=True)

    assert processed == ["good-a", "good-b"]
    assert pool.thread_count == 0


def test_judge_pool_cancellation_discards_queued_work_and_joins():
    stop = threading.Event()
    started = threading.Event()
    processed = []

    def process(job):
        processed.append(job)
        started.set()
        stop.wait(timeout=1)

    pool = SourceJudgeWorkerPool("Cloud", 1, process, stop)
    for job in range(5):
        pool.enqueue(job)
    pool.start(1)
    assert started.wait(timeout=1)
    stop.set()
    pool.stop(timeout=2, drain=False)
    assert pool.thread_count == 0
    assert processed == [0]


def test_judge_pool_uses_free_slots_during_benchmarks_then_expands():
    """Judge workers fill model slots benchmarks cannot use, then take the
    full pool once the source benchmark queue drains."""
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
    pool = SourceJudgeWorkerPool("Cloud", full_limit, process_judge, stop)
    assert pool.model_slots == 0
    for judge in ("judge-0", "judge-1", "judge-2"):
        pool.enqueue((judge, None, None, None, judge, True))

    benchmark = SourceModelScheduler(
        "Cloud", full_limit, ["model-a", "model-b"], run_benchmark,
        stop, lambda *_args: None,
        active_models_callback=lambda _source, models: pool.set_benchmark_models(models),
        on_complete=lambda _source: pool.expand_full(),
    )
    benchmark_thread = threading.Thread(target=benchmark.run_until_drained)
    benchmark_thread.start()

    assert benchmark_started.wait(timeout=1)
    assert benchmark_both_active.wait(timeout=1)
    # One free model slot (3 limit, 2 benchmarks) hosts one judge while the
    # benchmarks are still running; the other two judges wait.
    assert judge_started.wait(timeout=1)
    with lock:
        assert active_benchmarks == 2
        assert peak_benchmarks == 2
        assert active_judges == 1
        assert peak_judges == 1
        assert active_benchmarks + active_judges == full_limit
    assert pool.model_slots == 1

    benchmark_release.set()
    benchmark_thread.join(timeout=2)
    assert not benchmark_thread.is_alive()
    # Queue drained -> the full judge pool runs.
    assert full_pool_active.wait(timeout=1)
    with lock:
        assert peak_judges == full_limit
        assert active_judges == full_limit
    assert pool.model_slots == full_limit
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


def test_benchmark_scheduler_reports_pending_or_running_models():
    """The active-models callback reports queued + running models so the
    judge pool reserves exactly the load benchmarks will actually use."""
    stop = threading.Event()
    seen = []
    release = threading.Event()

    def run_benchmark(_name):
        release.wait(timeout=2)

    scheduler = SourceModelScheduler(
        "Cloud", 2, ["a", "b", "c"], run_benchmark, stop,
        lambda *_args: None,
        active_models_callback=lambda _source, models: seen.append(models),
    )
    thread = threading.Thread(target=scheduler.run_until_drained)
    thread.start()

    deadline = time.time() + 1
    while not seen and time.time() < deadline:
        time.sleep(0.01)
    assert seen
    # The whole queue counts as pending load, not just the running pair.
    assert seen[0] == frozenset({"a", "b", "c"})
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert seen[-1] == frozenset()


def test_dual_role_judge_activates_without_free_model_slot():
    """A benchmarking model that is also a judge may start judging on its
    open plugin slots even when every model slot is taken by benchmarks."""
    stop = threading.Event()
    release = threading.Event()
    processed = []
    lock = threading.Lock()

    def process(job):
        release.wait(timeout=2)
        with lock:
            processed.append(job[4])

    pool = SourceJudgeWorkerPool("Cloud", 2, process, stop, plugin_limit=1)
    # Two benchmarks fill the source's entire model limit.
    pool.set_benchmark_models({"model-a", "model-b"})
    assert pool.model_slots == 0
    # The dual-role judge rides its model's benchmark slot and starts anyway.
    pool.enqueue(("cell", None, None, None, "model-a", True))
    # A regular judge must wait for a free model slot.
    pool.enqueue(("cell", None, None, None, "other", True))
    time.sleep(0.1)
    assert pool.thread_count == 1

    release.set()
    deadline = time.time() + 1
    while pool.thread_count > 0 and time.time() < deadline:
        time.sleep(0.02)
    assert pool.thread_count == 0
    with lock:
        assert processed == ["model-a"]

    # When model-a's benchmark completes, a free slot appears and the
    # regular judge can start.
    pool.set_benchmark_models({"model-b"})
    assert pool.model_slots == 1
    deadline = time.time() + 1
    while "other" not in processed and time.time() < deadline:
        time.sleep(0.02)
    pool.stop(timeout=2)
    with lock:
        assert sorted(processed) == ["model-a", "other"]


def test_plugin_slot_gate_bounds_and_benchmark_priority():
    """Benchmarks get freed slots before waiting judge cells, and combined
    concurrency never exceeds the gate capacity."""
    gate = PluginSlotGate(2)
    gate.acquire(benchmark=True)
    gate.acquire(benchmark=True)
    assert gate.in_use == 2

    judge_got_slot = threading.Event()

    def judge_acquire():
        gate.acquire(benchmark=False)
        judge_got_slot.set()
        gate.release()

    judge_thread = threading.Thread(target=judge_acquire)
    judge_thread.start()
    time.sleep(0.05)
    assert not judge_got_slot.is_set()

    # A benchmark that arrives while the judge waits takes the next freed
    # slot first (benchmark priority), leaving the judge blocked.
    benchmark_got_slot = threading.Event()

    def benchmark_acquire():
        gate.acquire(benchmark=True)
        benchmark_got_slot.set()
        time.sleep(0.05)
        gate.release()

    benchmark_thread = threading.Thread(target=benchmark_acquire)
    benchmark_thread.start()
    time.sleep(0.05)
    gate.release()  # free one slot
    assert benchmark_got_slot.wait(timeout=1)
    assert not judge_got_slot.is_set()
    benchmark_thread.join(timeout=2)
    assert judge_got_slot.wait(timeout=1)
    judge_thread.join(timeout=2)
    gate.release()  # the test's second benchmark-held slot
    assert gate.in_use == 0


def test_dual_role_judge_cells_bounded_by_plugin_gate():
    """Judge cells for a benchmarking model wait for open plugin slots
    instead of exceeding the model's plugin_thread_limit."""
    stop = threading.Event()
    gates = PluginSlotGateRegistry()
    gates.create("Cloud", "model-a", 1)
    processed = []
    lock = threading.Lock()

    def process(job):
        with lock:
            processed.append(job[0])

    pool = SourceJudgeWorkerPool(
        "Cloud", 1, process, stop, plugin_limit=2, plugin_gates=gates,
    )
    pool.set_benchmark_models({"model-a"})
    # The benchmark holds the only plugin slot before any judge cell can
    # start, so no judge cell may run while it is held.
    gate = gates.get("Cloud", "model-a")
    assert gate is not None
    gate.acquire(benchmark=True)
    pool.enqueue(("cell-0", None, None, None, "model-a", True))
    pool.enqueue(("cell-1", None, None, None, "model-a", True))
    time.sleep(0.1)
    with lock:
        assert processed == []
    gate.release()
    deadline = time.time() + 1
    while len(processed) < 2 and time.time() < deadline:
        time.sleep(0.02)
    with lock:
        assert sorted(processed) == ["cell-0", "cell-1"]
    pool.stop(timeout=2)


def test_gate_registry_only_created_for_known_keys():
    gates = PluginSlotGateRegistry()
    gates.create("Cloud", "model-a", 4)
    assert len(gates) == 1
    assert gates.get("Cloud", "model-a") is not None
    assert gates.get("Cloud", "model-b") is None
    assert gates.get("Other", "model-a") is None


def _build_coordinator(targets, judge_models, judge_sources, source_config):
    from benchmark.judge_coordinator import JudgeCoordinator
    from benchmark.scheduling import _FlushGate

    return JudgeCoordinator(
        state=mock.Mock(),
        source_config=source_config,
        targets=targets,
        active_plugins=[],
        judge_models=judge_models,
        judge_contracts={},
        active_judge_contracts={},
        judge_sources=judge_sources,
        judge_model_limits={"Cloud": 3},
        judge_plugin_limits={"Cloud": 2},
        judge_effective_timeout=60.0,
        judge_max_tokens=4096,
        judge_temperature=0.0,
        judge_request_params=None,
        output_dir="/tmp",
        args=mock.Mock(debug_logs=False, storage_profile="compact"),
        model_thread_limits={"Cloud": 3},
        stop_event=threading.Event(),
        raw_targets={},
        run_info={"judge_counts": {}},
        flush_gate=_FlushGate(),
        flusher=mock.Mock(),
    )


def test_judge_coordinator_gates_only_dual_role_models():
    """The coordinator creates plugin gates only for judges that are also
    benchmark targets on the same source, and seeds the pool's benchmark
    reservation from the pending queue."""
    targets = {
        "model-a": {"source": "Cloud", "api_model": "model-a"},
        "model-b": {"source": "Cloud", "api_model": "model-b"},
        "model-c": {"source": "Other", "api_model": "model-c"},
    }
    judge_models = ["model-a", "model-b", "model-c", "judge-only"]
    judge_sources = {
        "model-a": "Cloud", "model-b": "Cloud",
        "model-c": "Cloud", "judge-only": "Cloud",
    }
    coordinator = _build_coordinator(
        targets, judge_models, judge_sources,
        {"Cloud": {"plugin_thread_limit": 2}},
    )
    coordinator.build_pools()
    gates = coordinator.plugin_slot_gates
    # model-a/model-b benchmark on the same source they judge -> shared gate.
    assert len(gates) == 2
    assert gates.get("Cloud", "model-a") is not None
    assert gates.get("Cloud", "model-b") is not None
    # model-c judges on Cloud but benchmarks on Other -> no sharing; a judge
    # that is not a benchmark target at all -> no sharing.
    assert gates.get("Cloud", "model-c") is None
    assert gates.get("Cloud", "judge-only") is None

    pool = coordinator.pools["Cloud"]
    # A queue that fills the limit leaves no free model slots for judges.
    coordinator.start_judge_if_async(
        {"Cloud": 3}, {"Cloud": ["model-a", "model-b", "model-c"]})
    assert pool.model_slots == 0
    # Live benchmark activity updates the reservation.
    coordinator.set_benchmark_active("Cloud", {"model-a"})
    assert pool.model_slots == 2
    coordinator.set_benchmark_active("Cloud", set())
    assert pool.model_slots == 3
    pool.stop(timeout=1)


def test_judge_coordinator_skips_gate_when_plugin_limit_unbounded():
    """A non-positive plugin_thread_limit means benchmarks are unbounded, so
    no gate is created (a gate would wrongly cap benchmark concurrency)."""
    targets = {"model-a": {"source": "Cloud", "api_model": "model-a"}}
    coordinator = _build_coordinator(
        targets, ["model-a"], {"model-a": "Cloud"},
        {"Cloud": {"plugin_thread_limit": 0}},
    )
    coordinator.build_pools()
    assert len(coordinator.plugin_slot_gates) == 0
    coordinator.pools["Cloud"].stop(timeout=1)
