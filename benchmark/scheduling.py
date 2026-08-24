"""Benchmark and judge scheduling primitives."""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .scheduler_policy import SourceSchedulingPolicy


def _runner_suffix(runner):
    """Return the stable state/artifact suffix for a non-HTTP runner."""
    return "" if runner == "http" else f" [{runner}]"


def _runner_state_key(target_name, runner):
    return f"{target_name}{_runner_suffix(runner)}"


def _targets_for_runner(targets, state_models, runner):
    """Return targets with a saved/configured identity for ``runner``."""
    suffix = _runner_suffix(runner)
    return {
        name: info
        for name, info in targets.items()
        if f"{name}{suffix}" in state_models
    }


def _mark_preload_failed(state, model_name, result, phase_runner, runner_mode):
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


def _build_runner_queues(targets, snapshot, runner_mode, source_config,
                         *, rerun_failed=True, plugin_ids=None):
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

    def needs_run(state):
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
        targets_by_source = {src: [] for src in source_config}
        opencode_pending = {src: [] for src in targets_by_source}
        http_pending = {src: set() for src in targets_by_source}
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
    source_queues = {src: [] for src in {info["source"] for info in targets.values()}}
    for name, info in targets.items():
        state_key = _runner_state_key(name, phase_runner)
        if needs_run(snapshot.get(state_key)):
            source_queues[info["source"]].append(name)
    return source_queues


class SourceModelScheduler:
    """Run a FIFO queue of target pipelines with a source-local bound."""

    def __init__(self, source, max_models, target_names, run_target,
                 stop_event, on_error, *, runner_label="model",
                 peak_callback=None, on_complete=None):
        self.source = source
        self.max_models = max(1, int(max_models))
        self.target_names = list(target_names)
        self.run_target = run_target
        self.stop_event = stop_event
        self.on_error = on_error
        self.runner_label = runner_label
        self.peak_callback = peak_callback
        self.on_complete = on_complete

    def run_until_drained(self):
        """Submit at most ``max_models`` targets and refill as they finish."""
        next_index = 0
        futures = {}
        active = 0
        executor = ThreadPoolExecutor(max_workers=self.max_models)
        try:
            def submit_next():
                nonlocal next_index, active
                if self.stop_event.is_set() or next_index >= len(self.target_names):
                    return False
                target_name = self.target_names[next_index]
                next_index += 1
                # The scheduler's FIFO queue and this one-shot submission
                # path are the claim guard: a target is inserted into exactly
                # one future before any refill can advance the queue.
                futures[executor.submit(self.run_target, target_name)] = target_name
                active += 1
                if self.peak_callback:
                    self.peak_callback(self.source, active)
                return True

            for _ in range(self.max_models):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    target_name = futures.pop(future)
                    active -= 1
                    if self.peak_callback:
                        self.peak_callback(self.source, active)
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

    def __init__(self, interval=60.0, max_changes=10):
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

    def changed(self):
        """Record one in-memory change; return True when a flush is due."""
        with self._lock:
            self._changes += 1
            return self._due_locked()

    def _due_locked(self):
        return (self._changes >= self.max_changes
                or time.monotonic() - self._last_flush >= self.interval)

    def _due(self):
        """Return whether a flush is due, for diagnostics and tests."""
        with self._lock:
            return self._due_locked()

    def reset(self):
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

    def __init__(self, flush_fn, name="background-flusher", failure_callback=None):
        self._flush_fn = flush_fn
        self._failure_callback = failure_callback
        self._condition = threading.Condition()
        self._failure_lock = threading.Lock()
        self._failures = []
        self._pending = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )

    def start(self):
        self._thread.start()

    def request_flush(self):
        """Request a flush; never blocks the caller."""
        with self._condition:
            self._pending = True
            self._condition.notify()

    def stop(self, timeout=None):
        """Drain pending work and join, returning False if the timeout expires."""
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    @property
    def failures(self):
        """Return a snapshot of flush exceptions raised by the worker."""
        with self._failure_lock:
            return list(self._failures)

    @property
    def is_alive(self):
        return self._thread.is_alive()

    def _record_failure(self, exc):
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

    def _run(self):
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


def _resolve_judge_plugin_limit(source_config, source):
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


def _configure_judge_source(benchmark_limits, source, full_limit,
                            benchmark_active, pool):
    """Configure strict benchmark-first scheduling for one source.

    Benchmark work keeps every configured source slot while it is pending or
    running. Judge jobs may be queued during that phase, but judge workers are
    not allocated until the source benchmark scheduler calls
    ``pool.expand_full()`` after draining its queue. Sources with no benchmark
    work can start their full judge pool immediately.
    """
    policy = SourceSchedulingPolicy(source, max(1, int(full_limit)))
    benchmark_slots, judge_slots = policy.capacity(benchmark_active=benchmark_active)
    benchmark_limits[source] = benchmark_slots
    if not policy.can_start_judges(benchmark_active=benchmark_active):
        # Jobs may already be queued, but no judge runner is activated until
        # the benchmark completion callback releases the reserved capacity.
        pool.start(0)
    else:
        pool.start(judge_slots)


class _CombinedStopEvent:
    """Expose several cancellation events through the Event interface."""

    def __init__(self, *events):
        self._events = tuple(events)

    def is_set(self):
        return any(event.is_set() for event in self._events)

    def wait(self, timeout=None):
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

    def __init__(self):
        self._condition = threading.Condition()
        self._fresh = deque()
        self._retry = deque()
        self._unfinished_tasks = 0
        self._stop_tokens = 0

    @property
    def unfinished_tasks(self):
        """Expose queue accounting used by tests and shutdown diagnostics."""
        with self._condition:
            return self._unfinished_tasks

    @property
    def pending(self):
        """True while the judge still has unstarted cells queued."""
        with self._condition:
            return bool(self._fresh or self._retry)

    @staticmethod
    def _job_is_fresh(job):
        # ``expected_added`` is true when this judge has no prior vote for the
        # cell; failed/invalid prior attempts are retry work.
        return not isinstance(job, tuple) or len(job) <= 5 or bool(job[5])

    def put(self, job):
        bucket = self._fresh if self._job_is_fresh(job) else self._retry
        with self._condition:
            bucket.append(job)
            self._unfinished_tasks += 1
            self._condition.notify()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop_tokens:
                    self._stop_tokens -= 1
                    return _JUDGE_QUEUE_STOP
                if self._fresh or self._retry:
                    return self._fresh.popleft() if self._fresh else self._retry.popleft()
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def task_done(self):
        with self._condition:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._condition.notify_all()

    def join(self):
        with self._condition:
            while self._unfinished_tasks:
                self._condition.wait()

    def cancel_pending(self):
        """Discard queued, not-yet-started jobs while preserving active jobs."""
        with self._condition:
            pending = len(self._fresh) + len(self._retry)
            self._fresh.clear()
            self._retry.clear()
            self._unfinished_tasks -= pending
            self._condition.notify_all()

    def request_stop(self, count):
        with self._condition:
            self._stop_tokens += count
            self._condition.notify_all()


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

    def __init__(self, source, model_limit, process_job, stop_event,
                 plugin_limit=1, on_selection_change=None):
        self.source = source
        self.model_limit = max(1, int(model_limit))
        self.plugin_limit = max(1, int(plugin_limit))
        self.process_job = process_job
        self.stop_event = stop_event
        self.on_selection_change = on_selection_change
        self._condition = threading.Condition()
        self._queues = {}          # judge -> _JudgeQueue
        self._order = []           # judge discovery order
        self._active = {}          # judge -> judge-runner thread
        self._active_limit = 0     # currently allowed concurrent judge models
        self._stopped = False

    @property
    def thread_count(self):
        """Number of judge models currently running for this source."""
        with self._condition:
            return len(self._active)

    @property
    def model_slots(self):
        """Currently allowed number of concurrent judge models (reservation)."""
        with self._condition:
            return self._active_limit

    @staticmethod
    def _job_key(job):
        if isinstance(job, tuple) and len(job) > 4:
            return job[4]
        return None

    def _queue_for(self, judge):
        queue = self._queues.get(judge)
        if queue is None:
            queue = _JudgeQueue()
            self._queues[judge] = queue
            self._order.append(judge)
        return queue

    def enqueue(self, job):
        """Queue one judge job, keyed by its judge model."""
        judge = self._job_key(job)
        with self._condition:
            self._queue_for(judge).put(job)
            self._activate_locked()

    def _next_pending_judge_locked(self):
        for judge in self._order:
            if judge in self._active:
                continue
            if self._queues[judge].unfinished_tasks > 0:
                return judge
        return _NO_JUDGE

    def _notify_selection(self, judge, selected):
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

    def _activate_locked(self):
        """Start judge runners for pending judges while model slots are free."""
        while (not self._stopped and not self.stop_event.is_set()
               and len(self._active) < self._active_limit):
            judge = self._next_pending_judge_locked()
            if judge is _NO_JUDGE:
                break
            thread = threading.Thread(
                target=self._judge_runner,
                args=(judge,),
                name=f"judge-runner-{self.source}-{judge}",
                daemon=True,
            )
            self._active[judge] = thread
            self._notify_selection(judge, True)
            thread.start()

    def _judge_runner(self, judge):
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
                args=(queue,),
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

    def _cell_worker(self, judge_queue):
        while True:
            try:
                job = judge_queue.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            if job is _JUDGE_QUEUE_STOP:
                return
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
                judge_queue.task_done()

    def start(self, count=1):
        """Allow up to ``count`` judge models to run concurrently."""
        with self._condition:
            self._active_limit = min(self.model_limit, max(0, int(count)))
            self._activate_locked()
            self._condition.notify_all()

    def expand_full(self):
        """Release the benchmark reservation and allow the full judge pool."""
        self.start(self.model_limit)

    def drain(self):
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

    def stop(self, timeout=None, *, drain=False):
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



