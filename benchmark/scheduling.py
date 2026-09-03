"""Benchmark and judge scheduling primitives."""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from .scheduler_policy import SourceSchedulingPolicy


def _runner_suffix(runner: str) -> str:
    """Return the stable state/artifact suffix for a non-HTTP runner."""
    return "" if runner == "http" else f" [{runner}]"


def _runner_state_key(target_name: str, runner: str) -> str:
    return f"{target_name}{_runner_suffix(runner)}"


def _targets_for_runner(
    targets: dict[str, dict[str, Any]], state_models: Iterable[str], runner: str,
) -> dict[str, dict[str, Any]]:
    """Return targets with a saved/configured identity for ``runner``."""
    suffix = _runner_suffix(runner)
    return {
        name: info
        for name, info in targets.items()
        if f"{name}{suffix}" in state_models
    }


def _mark_preload_failed(state: Any, model_name: str, result: Any,
                         phase_runner: str, runner_mode: str) -> None:
    """Record a failed warm-up in the model's live state only.

    A preload failure means the model produced no per-plugin results, so it
    must not append a row to ``state.results``. A scoreless ``error`` row would
    become the model's latest result and mask later progress, causing a resumed
    run to re-run already-successful plugins. The ``failed`` status in
    ``model_info`` is authoritative for the TUI, queue builder, and resume
    re-queue.
    """
    error = f"preload failed: {result.error or 'empty preload response'}"
    if runner_mode == "both" and phase_runner == "opencode":
        keys = [model_name, _runner_state_key(model_name, phase_runner)]
    elif phase_runner != "http":
        keys = [_runner_state_key(model_name, phase_runner)]
    else:
        keys = [model_name]
    snapshot = state.snapshot()
    for key in keys:
        info = snapshot.get(key)
        if info is None or info.get("status") == "completed":
            continue
        state.update(
            key,
            status="failed",
            error=error,
            last_error=error,
            elapsed=0.0,
            preloading=False,
            preload_start_ts=0,
            preload_status="failed",
            preload_time=result.elapsed,
            preload_error=result.error or "empty preload response",
        )
        state.log(key, error)


def _build_runner_queues(
    targets: dict[str, dict[str, Any]], snapshot: dict[str, Any], runner_mode: str,
    source_config: dict[str, Any], *, rerun_failed: bool = True,
    plugin_ids: Iterable[str] | None = None,
) -> dict[str, list[str]] | tuple[dict[str, list[str]], dict[str, list[str]], dict[str, set[str]]]:
    """Build pending runner queues from the loaded state snapshot.

    ``rerun_failed`` mirrors the resume option. Keeping this decision in the
    queue builder is important: ``BenchmarkState.load_state`` can preserve a
    failed status, but a status-only ``!= completed`` check would immediately
    put that target back on the scheduler queue anyway.
    """
    inferred_plugin_ids = list(plugin_ids or [])
    if not inferred_plugin_ids:
        for state in snapshot.values():
            if not isinstance(state, dict):
                continue
            inferred_plugin_ids.extend(
                key.removesuffix("_score")
                for key, value in state.items()
                if key.endswith("_score") and key != "overall_score_100"
                and value is not None
            )
        inferred_plugin_ids = list(dict.fromkeys(inferred_plugin_ids))

    def needs_run(state: dict[str, Any] | None) -> bool:
        if state is None:
            return False
        if state.get("status") == "failed" and not rerun_failed:
            return False
        if state.get("status") != "completed":
            return True
        return bool(inferred_plugin_ids) and any(
            not isinstance(state.get(f"{pid}_score"), (int, float))
            or isinstance(state.get(f"{pid}_score"), bool)
            for pid in inferred_plugin_ids
        )

    if runner_mode == "both":
        targets_by_source: dict[str, list[str]] = {src: [] for src in source_config}
        opencode_pending: dict[str, list[str]] = {src: [] for src in targets_by_source}
        http_pending: dict[str, set[str]] = {src: set() for src in targets_by_source}
        for name, info in targets.items():
            opencode_state = snapshot.get(f"{name} [opencode]")
            opencode_needed = needs_run(opencode_state)
            http_state = snapshot.get(name)
            http_needed = needs_run(http_state)
            if opencode_needed:
                opencode_pending[info["source"]].append(name)
            if http_needed:
                http_pending[info["source"]].add(name)
            if opencode_needed or http_needed:
                targets_by_source[info["source"]].append(name)
        return targets_by_source, opencode_pending, http_pending

    phase_runner = runner_mode
    source_queues: dict[str, list[str]] = {
        src: [] for src in {info["source"] for info in targets.values()}}
    for name, info in targets.items():
        state_key = _runner_state_key(name, phase_runner)
        if needs_run(snapshot.get(state_key)):
            source_queues[info["source"]].append(name)
    return source_queues


class SourceModelScheduler:
    """Run a FIFO queue of target pipelines with a source-local bound."""

    def __init__(
        self, source: str, max_models: int, target_names: Iterable[str],
        run_target: Callable[[str], None], stop_event: Any,
        on_error: Callable[[str, str, Exception], None], *,
        runner_label: str = "model",
        peak_callback: Callable[[str, int], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
        active_models_callback: Callable[[str, frozenset[str]], None] | None = None,
    ) -> None:
        self.source = source
        self.max_models = max(1, int(max_models))
        self.target_names = list(target_names)
        self.run_target = run_target
        self.stop_event = stop_event
        self.on_error = on_error
        self.runner_label = runner_label
        self.peak_callback = peak_callback
        self.on_complete = on_complete
        self.active_models_callback = active_models_callback

    def run_until_drained(self) -> None:
        """Submit at most ``max_models`` targets and refill as they finish."""
        next_index = 0
        futures: dict[Future[None], str] = {}
        active = 0
        running: set[str] = set()
        executor = ThreadPoolExecutor(max_workers=self.max_models)
        try:
            def pending_benchmark_models() -> frozenset[str]:
                # Models with benchmark work not yet finished: those running
                # plus those still queued. Submission is FIFO, so the queued
                # tail is exactly ``target_names[next_index:]``.
                return frozenset(running) | frozenset(self.target_names[next_index:])

            def report_active_models() -> None:
                if self.active_models_callback:
                    self.active_models_callback(self.source, pending_benchmark_models())

            def submit_next() -> bool:
                nonlocal next_index, active
                if self.stop_event.is_set() or next_index >= len(self.target_names):
                    return False
                target_name = self.target_names[next_index]
                next_index += 1
                # The scheduler's FIFO queue and this one-shot submission
                # path are the claim guard: a target is inserted into exactly
                # one future before any refill can advance the queue.
                futures[executor.submit(self.run_target, target_name)] = target_name
                running.add(target_name)
                active += 1
                if self.peak_callback:
                    self.peak_callback(self.source, active)
                report_active_models()
                return True

            for _ in range(self.max_models):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    target_name = futures.pop(future)
                    running.discard(target_name)
                    active -= 1
                    if self.peak_callback:
                        self.peak_callback(self.source, active)
                    report_active_models()
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate one target failure
                        self.on_error(target_name, self.runner_label, exc)
                    submit_next()
                if self.stop_event.is_set():
                    for future in futures:
                        future.cancel()
                    break
        finally:
            # Do not let the executor context manager wait indefinitely after
            # cancellation: active HTTP/subprocess work is interrupted by the
            # caller before workers are joined. Running futures are not waited
            # on here -- the stop-aware request watchdog bounds any straggler
            # to a fraction of a second, and the orchestrator performs its own
            # bounded shutdown afterwards. Normal completion still shuts down
            # synchronously so no executor thread leaks into output work.
            if self.stop_event.is_set():
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
        if not self.stop_event.is_set() and self.on_complete:
            self.on_complete(self.source)



class _FlushGate:
    """Decide when in-memory changes warrant a full-state flush.

    The judge path used to persist the entire ``benchmark_state.json`` plus
    regenerate every report after each completed vote, which burned ~6 s of
    GIL-bound CPU and ~210 MB of disk writes per vote. ``_FlushGate`` throttles
    that: ``changed()`` is called once per in-memory change (a completed or
    failed judge vote) and returns ``True`` when a flush is due -- either
    ``interval`` seconds have elapsed since the last flush or ``max_changes``
    changes have accumulated, whichever comes first. The caller schedules the
    actual save only when ``changed()`` reports due, then calls ``reset()``.

    ``changed()``/``reset()`` are called from judge cell workers without the
    ``persistence_lock`` (the save itself runs on the background flusher
    thread, which owns the lock). The gate has its own lock so concurrent
    workers cannot lose increments or observe partially updated cadence state;
    ``_BackgroundFlusher.request_flush()`` still coalesces duplicate requests.

    The lock only protects the small in-memory decision; it is never held
    while state serialization or disk I/O runs.

    """

    def __init__(self, interval: float = 60.0, max_changes: int = 10) -> None:
        try:
            self.interval = float(interval)
        except (TypeError, ValueError):
            self.interval = 60.0
        try:
            self.max_changes = max(1, int(max_changes))
        except (TypeError, ValueError):
            self.max_changes = 10
        self._last_flush = time.monotonic()
        self._changes = 0
        self._lock = threading.Lock()

    def changed(self) -> bool:
        """Record one in-memory change; return True when a flush is due."""
        with self._lock:
            self._changes += 1
            return self._due_locked()

    def _due_locked(self) -> bool:
        return (self._changes >= self.max_changes
                or time.monotonic() - self._last_flush >= self.interval)

    def _due(self) -> bool:
        """Return whether a flush is due, for diagnostics and tests."""
        with self._lock:
            return self._due_locked()

    def reset(self) -> None:
        """Mark the current flush as completed, starting a fresh cadence."""
        with self._lock:
            self._last_flush = time.monotonic()
            self._changes = 0


class _BackgroundFlusher:
    """Serialize full-state snapshots on a dedicated thread.

    The judge path used to run the full-state save synchronously in the worker
    thread that completed the vote: ~6 s of GIL-bound deepcopy + JSON dump +
    report regeneration stalled every other judge worker (and the TUI) on the
    GIL, and later finishers queued behind ``persistence_lock`` instead of
    starting their next request. ``_BackgroundFlusher`` moves that
    serialization off the hot path: ``request_flush()`` is non-blocking (it
    only sets a pending flag), and one dedicated daemon thread runs
    ``flush_fn`` -- which must take ``persistence_lock`` itself -- for each
    batch of requests. Requests that arrive while a flush is running are
    coalesced into one follow-up flush, so at most one save is in flight and
    at most one is queued behind it. The flush persists only the state
    snapshot; report files are regenerated once at the end of the run.

    ``stop()`` drains any pending request (one final flush) before exiting.
    The benchmark bounds that wait and performs a synchronous final state and
    journal compaction on the main thread, reporting a shutdown timeout or
    save failure prominently.
    """

    def __init__(self, flush_fn: Callable[[], None],
                 name: str = "background-flusher",
                 failure_callback: Callable[[Exception], None] | None = None) -> None:
        self._flush_fn = flush_fn
        self._failure_callback = failure_callback
        self._condition = threading.Condition()
        self._failure_lock = threading.Lock()
        self._failures: list[Exception] = []
        self._pending = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def request_flush(self) -> None:
        """Request a flush; never blocks the caller."""
        with self._condition:
            self._pending = True
            self._condition.notify()

    def stop(self, timeout: float | None = None) -> bool:
        """Drain pending work and join, returning False if the timeout expires."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    @property
    def failures(self) -> list[Exception]:
        """Return a snapshot of flush exceptions raised by the worker."""
        with self._failure_lock:
            return list(self._failures)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _record_failure(self, exc: Exception) -> None:
        with self._failure_lock:
            self._failures.append(exc)
        if self._failure_callback is not None:
            try:
                self._failure_callback(exc)
                return
            except Exception as callback_exc:  # noqa: BLE001 - reporting must not kill the worker
                print(
                    f"❌ Persistence failure reporter failed: {type(callback_exc).__name__}: "
                    f"{callback_exc}",
                    file=sys.stderr,
                )
        print(
            f"❌ PERSISTENCE FLUSH FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped:
                    self._condition.wait()
                if self._pending:
                    self._pending = False
                elif self._stopped:
                    return
            try:
                self._flush_fn()
            except Exception as exc:  # noqa: BLE001 - keep the flusher alive
                self._record_failure(exc)


def _resolve_judge_plugin_limit(source_config: dict[str, Any], source: str) -> int:
    """Return the per-judge cell concurrency for ``source``.

    Mirrors ``plugin_thread_limit``: how many cells one judge model scores at
    once. Unlike the benchmark's per-target semantics, zero is not an
    unlimited value here -- fanning out an unbounded number of concurrent
    judge requests is a resource hazard, so a non-positive value serializes
    to one cell per judge.
    """
    cfg = source_config.get(source)
    value = cfg.get("plugin_thread_limit", 1) if isinstance(cfg, dict) else 1
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 1
    return value if value > 0 else 1


def _configure_judge_source(benchmark_limits: dict[str, int], source: str,
                            full_limit: int, benchmark_queue: list[str],
                            pool: SourceJudgeWorkerPool) -> None:
    """Configure judge capacity for one source.

    Benchmarks keep priority and claim every configured model slot while work
    is pending or running. Judge workers use the capacity benchmarks cannot:
    the free model slots (``model_limit`` minus the pending benchmark load)
    host regular judges, and a benchmarking model that is also a judge rides
    its model's open plugin slots. ``benchmark_queue`` is the list of model
    names with pending benchmark work for the source; the pool seeds its
    reservation from it and the benchmark scheduler's live
    ``set_benchmark_models`` callbacks keep it current as models start and
    finish. Sources with no benchmark work start their full judge pool
    immediately.
    """
    policy = SourceSchedulingPolicy(source, max(1, int(full_limit)))
    benchmark_limits[source] = policy.benchmark_slots(len(benchmark_queue))
    pool.set_benchmark_models(benchmark_queue)


class _CombinedStopEvent:
    """Expose several cancellation events through the Event interface."""

    def __init__(self, *events: Any) -> None:
        self._events = tuple(events)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is None:
            while not self.is_set():
                time.sleep(0.1)
            return True
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self.is_set()
            time.sleep(min(0.1, remaining))
        return True


class _JudgeQueue:
    """A single judge model's fresh-then-retry FIFO cell queue.

    Each judge owns exactly one queue so a source can run one judge to
    completion before loading another judge (keeping a local model resident)
    instead of round-robin swapping between judges every cell. Within a judge,
    never-judged cells still precede retried cells, and both tiers are served
    in arrival order.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._fresh: deque[Any] = deque()
        self._retry: deque[Any] = deque()
        self._unfinished_tasks = 0
        self._stop_tokens = 0

    @property
    def unfinished_tasks(self) -> int:
        """Expose queue accounting used by tests and shutdown diagnostics."""
        with self._condition:
            return self._unfinished_tasks

    @property
    def pending(self) -> bool:
        """True while the judge still has unstarted cells queued."""
        with self._condition:
            return bool(self._fresh or self._retry)

    @staticmethod
    def _job_is_fresh(job: Any) -> bool:
        # ``expected_added`` is true when this judge has no prior vote for the
        # cell; failed/invalid prior attempts are retry work.
        return not isinstance(job, tuple) or len(job) <= 5 or bool(job[5])

    def put(self, job: Any) -> None:
        bucket = self._fresh if self._job_is_fresh(job) else self._retry
        with self._condition:
            bucket.append(job)
            self._unfinished_tasks += 1
            self._condition.notify()

    def get(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop_tokens:
                    self._stop_tokens -= 1
                    return _JUDGE_QUEUE_STOP
                if self._fresh or self._retry:
                    return self._fresh.popleft() if self._fresh else self._retry.popleft()
                if timeout is not None and deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def task_done(self) -> None:
        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self) -> None:
        with self._condition:
            while self._unfinished_tasks:
                self._condition.wait()

    def cancel_pending(self) -> None:
        """Discard queued, not-yet-started jobs while preserving active jobs."""
        with self._condition:
            pending = len(self._fresh) + len(self._retry)
            self._fresh.clear()
            self._retry.clear()
            self._unfinished_tasks -= pending
            self._condition.notify_all()

    def request_stop(self, count: int) -> None:
        with self._condition:
            self._stop_tokens += count
            self._condition.notify_all()


class PluginSlotGate:
    """Bound combined concurrent requests to one model.

    A model that both benchmarks and judges on a source shares its
    ``plugin_thread_limit`` capacity between the two: benchmark plugin tasks
    acquire with ``benchmark=True`` and are served before waiting judge cells,
    while judge cells acquire with ``benchmark=False`` and only take a slot
    when no benchmark is waiting. This keeps benchmarks in priority while
    letting the model's judge version fill idle plugin capacity.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._used = 0
        self._benchmark_waiters = 0
        self._condition = threading.Condition()

    def acquire(self, *, benchmark: bool) -> None:
        """Block until a plugin slot is available for the requesting side."""
        with self._condition:
            if benchmark:
                self._benchmark_waiters += 1
            try:
                self._condition.wait_for(lambda: self._can_acquire_locked(benchmark))
                self._used += 1
            finally:
                if benchmark:
                    self._benchmark_waiters -= 1

    def _can_acquire_locked(self, benchmark: bool) -> bool:
        if self._used >= self._capacity:
            return False
        if benchmark:
            return True
        # A judge cell may take a free slot only while no benchmark plugin is
        # waiting, so benchmarks always get first claim on freed capacity.
        return self._benchmark_waiters == 0

    def release(self) -> None:
        """Release one plugin slot, waking any waiting acquirers."""
        with self._condition:
            self._used -= 1
            self._condition.notify_all()

    @property
    def in_use(self) -> int:
        with self._condition:
            return self._used


class PluginSlotGateRegistry:
    """Per-(source, model) plugin gates for dual-role models.

    A gate exists only when a judge model is also a benchmark target on the
    same source, so the benchmark pipeline and the judge pool agree on how
    many concurrent requests the model may serve in total.
    """

    def __init__(self) -> None:
        self._gates: dict[tuple[str, str], PluginSlotGate] = {}
        self._lock = threading.Lock()

    def create(self, source: str, model: str, capacity: int) -> None:
        with self._lock:
            self._gates[(source, model)] = PluginSlotGate(capacity)

    def get(self, source: str, model: str) -> PluginSlotGate | None:
        with self._lock:
            return self._gates.get((source, model))

    def __len__(self) -> int:
        with self._lock:
            return len(self._gates)


_JUDGE_QUEUE_STOP = object()
_NO_JUDGE = object()


class SourceJudgeWorkerPool:
    """Run judge jobs with per-source model and plugin concurrency.

    ``model_limit`` bounds how many distinct judge models run concurrently for
    the source; each active judge occupies exactly one model slot, mirroring
    ``model_thread_limit``. ``plugin_limit`` bounds how many cells one judge
    scores at once, mirroring ``plugin_thread_limit``. Judges are run to
    completion in discovery order before another judge is activated, which
    keeps a single local model resident instead of round-robin swapping
    between judges.
    """

    def __init__(
        self, source: str, model_limit: int, process_job: Callable[[Any], None],
        stop_event: Any, plugin_limit: int = 1,
        on_selection_change: Callable[[Any, bool], None] | None = None,
        plugin_gates: PluginSlotGateRegistry | None = None,
    ) -> None:
        self.source = source
        self.model_limit = max(1, int(model_limit))
        self.plugin_limit = max(1, int(plugin_limit))
        self.process_job = process_job
        self.stop_event = stop_event
        self.on_selection_change = on_selection_change
        self.plugin_gates = plugin_gates
        self._condition = threading.Condition()
        self._queues: dict[Any, _JudgeQueue] = {}          # judge -> _JudgeQueue
        self._order: list[Any] = []                        # judge discovery order
        self._active: dict[Any, threading.Thread] = {}     # judge -> judge-runner thread
        self._active_limit = 0     # free model slots for regular judges
        self._benchmark_models: set[Any] = set()  # pending/running benchmark models
        self._stopped = False

    @property
    def thread_count(self) -> int:
        """Number of judge models currently running for this source."""
        with self._condition:
            return len(self._active)

    @property
    def model_slots(self) -> int:
        """Currently allowed number of concurrent judge models (reservation)."""
        with self._condition:
            return self._active_limit

    @staticmethod
    def _job_key(job: Any) -> Any:
        if isinstance(job, tuple) and len(job) > 4:
            return job[4]
        return None

    def _queue_for(self, judge: Any) -> _JudgeQueue:
        queue = self._queues.get(judge)
        if queue is None:
            queue = _JudgeQueue()
            self._queues[judge] = queue
            self._order.append(judge)
        return queue

    def enqueue(self, job: Any) -> None:
        """Queue one judge job, keyed by its judge model."""
        judge = self._job_key(job)
        with self._condition:
            self._queue_for(judge).put(job)
            self._activate_locked()

    def _notify_selection(self, judge: Any, selected: bool) -> None:
        """Publish a judge-runner selection change without affecting workers."""
        if self.on_selection_change is None:
            return
        try:
            self.on_selection_change(judge, selected)
        except Exception as exc:  # noqa: BLE001 - TUI bookkeeping must not kill a judge
            print(
                f"⚠️  Judge selection update ({self.source}/{judge}) failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def _activate_locked(self) -> None:
        """Start judge runners for pending judges while capacity is available.

        A judge whose model is currently benchmarking or queued for it
        (dual-role) rides that model's benchmark slot and its open plugin
        slots, so it may start without a free model slot, capped by the source
        model limit. A regular judge requires a free model slot
        (``_active_limit``). Judges run to completion in discovery order among
        the currently startable ones.
        """
        while not self._stopped and not self.stop_event.is_set():
            if len(self._active) >= self.model_limit:
                break
            candidate = _NO_JUDGE
            for judge in self._order:
                if judge in self._active:
                    continue
                if self._queues[judge].unfinished_tasks <= 0:
                    continue
                if judge in self._benchmark_models:
                    candidate = judge
                    break
                if len(self._active) < self._active_limit:
                    candidate = judge
                    break
            if candidate is _NO_JUDGE:
                break
            thread = threading.Thread(
                target=self._judge_runner,
                args=(candidate,),
                name=f"judge-runner-{self.source}-{candidate}",
                daemon=True,
            )
            self._active[candidate] = thread
            self._notify_selection(candidate, True)
            thread.start()

    def _judge_runner(self, judge: Any) -> None:
        """Run one judge over its queued cells until drained.

        The judge holds a model slot while it still has cells, then tears down
        its cell workers and frees the slot so the next judge (in discovery
        order) can be loaded. Judge runners are daemonized so Ctrl+C cannot
        leave a process permanently stuck behind a provider that ignores
        cancellation; normal completion still drains and joins before exit.
        """
        queue = self._queues[judge]
        workers = []
        for index in range(self.plugin_limit):
            thread = threading.Thread(
                target=self._cell_worker,
                args=(judge, queue),
                name=f"judge-cell-{self.source}-{judge}-{index + 1}",
                daemon=True,
            )
            thread.start()
            workers.append(thread)
        # Drain this judge's currently queued cells before yielding the slot.
        while not self.stop_event.is_set() and queue.unfinished_tasks > 0:
            time.sleep(0.05)
        queue.request_stop(len(workers))
        for thread in workers:
            thread.join()
        with self._condition:
            self._active.pop(judge, None)
            # Clear the old selection before activating its replacement so
            # the live footer never treats a finished runner as selected after
            # another judge has taken its slot.
            self._notify_selection(judge, False)
            self._activate_locked()
            self._condition.notify_all()

    def _cell_worker(self, judge: Any, judge_queue: _JudgeQueue) -> None:
        gate = None
        if self.plugin_gates is not None:
            gate = self.plugin_gates.get(self.source, judge)
        while True:
            try:
                job = judge_queue.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            if job is _JUDGE_QUEUE_STOP:
                return
            if gate is not None:
                # Dual-role model: share its plugin capacity with the running
                # benchmark; benchmarks keep priority over judge cells.
                gate.acquire(benchmark=False)
            try:
                # On cancellation, discard queued work instead of starting
                # another judge request. The active request receives the same
                # stop_event and can terminate cooperatively; every discarded
                # item still receives task_done so queue accounting remains
                # balanced.
                if self.stop_event.is_set():
                    continue
                try:
                    self.process_job(job)
                except Exception as exc:  # noqa: BLE001 - keep one bad job from killing the pool
                    # ``process_judge_job`` normally records its own failure,
                    # but the pool must remain live even if an unexpected
                    # callback bug escapes. Queue accounting is completed in
                    # the finally block and later jobs continue to drain.
                    print(
                        f"⚠️  Judge worker ({self.source}) failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            finally:
                if gate is not None:
                    gate.release()
                judge_queue.task_done()

    def start(self, count: int = 1) -> None:
        """Allow up to ``count`` judge models to run concurrently."""
        with self._condition:
            self._active_limit = min(self.model_limit, max(0, int(count)))
            self._activate_locked()
            self._condition.notify_all()

    def set_benchmark_models(self, models: Any) -> None:
        """Update which models hold pending-or-running benchmark work.

        The judge model-slot budget is the source model limit minus the models
        benchmarks are using; a judge whose model is in ``models`` is dual-role
        and rides that model's benchmark slot (its requests are bounded by the
        shared plugin gate) instead of consuming a free model slot. The
        benchmark scheduler calls this as models start and finish, and
        ``_configure_judge_source`` seeds it from the pending queue.
        """
        with self._condition:
            self._benchmark_models = set(models or ())
            self._active_limit = max(0, self.model_limit - len(self._benchmark_models))
            self._activate_locked()
            self._condition.notify_all()

    def expand_full(self) -> None:
        """Release the benchmark reservation and allow the full judge pool."""
        with self._condition:
            self._benchmark_models = set()
            self._active_limit = self.model_limit
            self._activate_locked()
            self._condition.notify_all()

    def drain(self) -> bool:
        """Wait until every queued job has finished, unless cancellation starts."""
        with self._condition:
            judge_queues = list(self._queues.values())
        for judge_queue in judge_queues:
            while judge_queue.unfinished_tasks:
                if self.stop_event.is_set():
                    return False
                time.sleep(0.05)
        with self._condition:
            while self._active:
                if self.stop_event.is_set():
                    return False
                self._condition.wait(timeout=0.05)
        return True

    def stop(self, timeout: float | None = None, *, drain: bool = False) -> None:
        """Stop judges, optionally draining all queued jobs first.

        Normal completion uses ``drain=True`` and an unbounded join. The
        cancellation path skips the drain so Ctrl+C can save resumable state
        without waiting on new work.
        """
        drained = self.drain() if drain else False
        if not drained:
            with self._condition:
                queues = list(self._queues.values())
            for queue in queues:
                queue.cancel_pending()
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
            active = list(self._active.items())
            if not drain:
                for judge, _thread in active:
                    self._notify_selection(judge, False)
        for _judge, thread in active:
            thread.join(timeout=timeout)



