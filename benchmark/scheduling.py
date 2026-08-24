"""Benchmark and judge scheduling primitives."""
from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

from .scheduler_policy import SourceSchedulingPolicy


def _runner_suffix(runner: str) -> str:
    return "" if runner == "http" else f" [{runner}]"


def _runner_state_key(target_name: str, runner: str) -> str:
    return f"{target_name}{_runner_suffix(runner)}"


def _targets_for_runner(targets: dict[str, Any], state_models: dict[str, Any], runner: str) -> dict[str, Any]:
    suffix = _runner_suffix(runner)
    return {name: info for name, info in targets.items() if f"{name}{suffix}" in state_models}


def _build_runner_queues(targets: dict[str, Any], snapshot: dict[str, Any], runner_mode: str,
                         source_config: dict[str, Any], *, rerun_failed: bool = True,
                         plugin_ids: list[str] | None = None) -> Any:
    inferred_plugin_ids = list(plugin_ids or [])
    if not inferred_plugin_ids:
        for state in snapshot.values():
            if isinstance(state, dict):
                inferred_plugin_ids.extend(
                    key.removesuffix("_score") for key, value in state.items()
                    if key.endswith("_score") and key != "overall_score_100" and value is not None
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
        targets_by_source = {src: [] for src in source_config}
        opencode_pending = {src: [] for src in targets_by_source}
        http_pending = {src: set() for src in targets_by_source}
        for name, info in targets.items():
            opencode_needed = needs_run(snapshot.get(f"{name} [opencode]"))
            http_needed = needs_run(snapshot.get(name))
            if opencode_needed:
                opencode_pending[info["source"]].append(name)
            if http_needed:
                http_pending[info["source"]].add(name)
            if opencode_needed or http_needed:
                targets_by_source[info["source"]].append(name)
        return targets_by_source, opencode_pending, http_pending

    source_queues = {src: [] for src in {info["source"] for info in targets.values()}}
    for name, info in targets.items():
        if needs_run(snapshot.get(_runner_state_key(name, runner_mode))):
            source_queues[info["source"]].append(name)
    return source_queues


class SourceModelScheduler:
    """Run a FIFO queue of target pipelines with a source-local bound."""
    def __init__(self, source: str, max_models: int, target_names: list[str], run_target: Callable[[str], Any],
                 stop_event: Any, on_error: Callable[[str, str, Exception], Any], *, runner_label: str = "model",
                 peak_callback: Callable[[str, int], Any] | None = None,
                 on_complete: Callable[[str], Any] | None = None) -> None:
        self.source, self.max_models, self.target_names = source, max(1, int(max_models)), list(target_names)
        self.run_target, self.stop_event, self.on_error = run_target, stop_event, on_error
        self.runner_label, self.peak_callback, self.on_complete = runner_label, peak_callback, on_complete

    def run_until_drained(self) -> None:
        next_index, futures, active = 0, {}, 0
        executor = ThreadPoolExecutor(max_workers=self.max_models)
        try:
            def submit_next() -> bool:
                nonlocal next_index, active
                if self.stop_event.is_set() or next_index >= len(self.target_names):
                    return False
                target_name = self.target_names[next_index]
                next_index += 1
                futures[executor.submit(self.run_target, target_name)] = target_name
                active += 1
                if self.peak_callback:
                    self.peak_callback(self.source, active)
                return True
            for _ in range(self.max_models):
                if not submit_next(): break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    target_name = futures.pop(future); active -= 1
                    if self.peak_callback: self.peak_callback(self.source, active)
                    try: future.result()
                    except Exception as exc: self.on_error(target_name, self.runner_label, exc)
                    submit_next()
                if self.stop_event.is_set():
                    for future in futures: future.cancel()
                    break
        finally:
            if self.stop_event.is_set():
                for future in futures: future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else: executor.shutdown(wait=True)
        if not self.stop_event.is_set() and self.on_complete: self.on_complete(self.source)


class _FlushGate:
    def __init__(self, interval: float = 60.0, max_changes: int = 10) -> None:
        try: self.interval = float(interval)
        except (TypeError, ValueError): self.interval = 60.0
        try: self.max_changes = max(1, int(max_changes))
        except (TypeError, ValueError): self.max_changes = 10
        self._last_flush, self._changes, self._lock = time.monotonic(), 0, threading.Lock()
    def changed(self) -> bool:
        with self._lock:
            self._changes += 1
            return self._due_locked()
    def _due_locked(self) -> bool:
        return self._changes >= self.max_changes or time.monotonic() - self._last_flush >= self.interval
    def _due(self) -> bool:
        with self._lock: return self._due_locked()
    def reset(self) -> None:
        with self._lock: self._last_flush, self._changes = time.monotonic(), 0


class _BackgroundFlusher:
    def __init__(self, flush_fn: Callable[[], Any], name: str = "background-flusher",
                 failure_callback: Callable[[Exception], Any] | None = None) -> None:
        self._flush_fn, self._failure_callback = flush_fn, failure_callback
        self._condition, self._failure_lock = threading.Condition(), threading.Lock()
        self._failures: list[Exception] = []; self._pending = False; self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
    def start(self) -> None: self._thread.start()
    def request_flush(self) -> None:
        with self._condition: self._pending = True; self._condition.notify()
    def stop(self, timeout: float | None = None) -> bool:
        with self._condition: self._stopped = True; self._condition.notify()
        self._thread.join(timeout=timeout); return not self._thread.is_alive()
    @property
    def failures(self) -> list[Exception]:
        with self._failure_lock: return list(self._failures)
    @property
    def is_alive(self) -> bool: return self._thread.is_alive()
    def _record_failure(self, exc: Exception) -> None:
        with self._failure_lock: self._failures.append(exc)
        if self._failure_callback:
            try: self._failure_callback(exc); return
            except Exception as callback_exc: print(f"❌ Persistence failure reporter failed: {callback_exc}", file=sys.stderr)
        print(f"❌ PERSISTENCE FLUSH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopped: self._condition.wait()
                if self._pending: self._pending = False
                elif self._stopped: return
            try: self._flush_fn()
            except Exception as exc: self._record_failure(exc)


def _resolve_judge_plugin_limit(source_config: dict[str, Any], source: str) -> int:
    cfg = source_config.get(source); value = cfg.get("plugin_thread_limit", 1) if isinstance(cfg, dict) else 1
    try: value = int(value)
    except (TypeError, ValueError): value = 1
    return value if value > 0 else 1


def _configure_judge_source(benchmark_limits: dict[str, int], source: str, full_limit: int,
                            benchmark_active: bool, pool: Any) -> None:
    policy = SourceSchedulingPolicy(source, max(1, int(full_limit)))
    benchmark_slots, judge_slots = policy.capacity(benchmark_active=benchmark_active)
    benchmark_limits[source] = benchmark_slots
    pool.start(judge_slots if policy.can_start_judges(benchmark_active=benchmark_active) else 0)


class _CombinedStopEvent:
    def __init__(self, *events: Any) -> None: self._events = tuple(events)
    def is_set(self) -> bool: return any(event.is_set() for event in self._events)
    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline: return self.is_set()
            time.sleep(0.1 if deadline is None else min(0.1, deadline - time.monotonic()))
        return True


_JUDGE_QUEUE_STOP = object(); _NO_JUDGE = object()


class _JudgeQueue:
    def __init__(self) -> None:
        self._condition, self._fresh, self._retry = threading.Condition(), deque(), deque()
        self._unfinished_tasks, self._stop_tokens = 0, 0
    @property
    def unfinished_tasks(self) -> int:
        with self._condition: return self._unfinished_tasks
    @property
    def pending(self) -> bool:
        with self._condition: return bool(self._fresh or self._retry)
    @staticmethod
    def _job_is_fresh(job: Any) -> bool: return not isinstance(job, tuple) or len(job) <= 5 or bool(job[5])
    def put(self, job: Any) -> None:
        with self._condition:
            (self._fresh if self._job_is_fresh(job) else self._retry).append(job)
            self._unfinished_tasks += 1; self._condition.notify()
    def get(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._stop_tokens: self._stop_tokens -= 1; return _JUDGE_QUEUE_STOP
                if self._fresh or self._retry: return self._fresh.popleft() if self._fresh else self._retry.popleft()
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: raise queue.Empty
                    self._condition.wait(remaining)
                else: self._condition.wait()
    def task_done(self) -> None:
        with self._condition:
            if self._unfinished_tasks <= 0: raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0: self._condition.notify_all()
    def join(self) -> None:
        with self._condition:
            while self._unfinished_tasks: self._condition.wait()
    def cancel_pending(self) -> None:
        with self._condition:
            pending = len(self._fresh) + len(self._retry); self._fresh.clear(); self._retry.clear()
            self._unfinished_tasks -= pending; self._condition.notify_all()
    def request_stop(self, count: int) -> None:
        with self._condition: self._stop_tokens += count; self._condition.notify_all()


class SourceJudgeWorkerPool:
    def __init__(self, source: str, model_limit: int, process_job: Callable[[Any], Any], stop_event: Any,
                 plugin_limit: int = 1, on_selection_change: Callable[[Any, bool], Any] | None = None) -> None:
        self.source, self.model_limit, self.plugin_limit = source, max(1, int(model_limit)), max(1, int(plugin_limit))
        self.process_job, self.stop_event, self.on_selection_change = process_job, stop_event, on_selection_change
        self._condition, self._queues, self._order, self._active = threading.Condition(), {}, [], {}
        self._active_limit, self._stopped = 0, False
    @property
    def thread_count(self) -> int:
        with self._condition: return len(self._active)
    @property
    def model_slots(self) -> int:
        with self._condition: return self._active_limit
    @staticmethod
    def _job_key(job: Any) -> Any: return job[4] if isinstance(job, tuple) and len(job) > 4 else None
    def _queue_for(self, judge: Any) -> _JudgeQueue:
        q = self._queues.get(judge)
        if q is None: q = _JudgeQueue(); self._queues[judge] = q; self._order.append(judge)
        return q
    def enqueue(self, job: Any) -> None:
        with self._condition: self._queue_for(self._job_key(job)).put(job); self._activate_locked()
    def _next_pending_judge_locked(self) -> Any:
        for judge in self._order:
            if judge not in self._active and self._queues[judge].unfinished_tasks > 0: return judge
        return _NO_JUDGE
    def _notify_selection(self, judge: Any, selected: bool) -> None:
        if self.on_selection_change is None: return
        try: self.on_selection_change(judge, selected)
        except Exception as exc: print(f"⚠️ Judge selection update ({self.source}/{judge}) failed: {exc}", file=sys.stderr)
    def _activate_locked(self) -> None:
        while not self._stopped and not self.stop_event.is_set() and len(self._active) < self._active_limit:
            judge = self._next_pending_judge_locked()
            if judge is _NO_JUDGE: break
            thread = threading.Thread(target=self._judge_runner, args=(judge,), name=f"judge-runner-{self.source}-{judge}", daemon=True)
            self._active[judge] = thread; self._notify_selection(judge, True); thread.start()
    def _judge_runner(self, judge: Any) -> None:
        q = self._queues[judge]; workers = []
        for index in range(self.plugin_limit):
            thread = threading.Thread(target=self._cell_worker, args=(q,), name=f"judge-cell-{self.source}-{judge}-{index + 1}", daemon=True)
            thread.start(); workers.append(thread)
        while not self.stop_event.is_set() and q.unfinished_tasks > 0: time.sleep(0.05)
        q.request_stop(len(workers))
        for thread in workers: thread.join()
        with self._condition:
            self._active.pop(judge, None); self._notify_selection(judge, False); self._activate_locked(); self._condition.notify_all()
    def _cell_worker(self, judge_queue: _JudgeQueue) -> None:
        while True:
            try: job = judge_queue.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set(): break
                continue
            if job is _JUDGE_QUEUE_STOP: return
            try:
                if not self.stop_event.is_set():
                    try: self.process_job(job)
                    except Exception as exc: print(f"⚠️ Judge worker ({self.source}) failed: {exc}", file=sys.stderr)
            finally: judge_queue.task_done()
    def start(self, count: int = 1) -> None:
        with self._condition: self._active_limit = min(self.model_limit, max(0, int(count))); self._activate_locked(); self._condition.notify_all()
    def expand_full(self) -> None: self.start(self.model_limit)
    def drain(self) -> bool:
        with self._condition: queues = list(self._queues.values())
        for q in queues:
            while q.unfinished_tasks:
                if self.stop_event.is_set(): return False
                time.sleep(0.05)
        with self._condition:
            while self._active:
                if self.stop_event.is_set(): return False
                self._condition.wait(timeout=0.05)
        return True
    def stop(self, timeout: float | None = None, *, drain: bool = False) -> None:
        drained = self.drain() if drain else False
        if not drained:
            with self._condition: queues = list(self._queues.values())
            for q in queues: q.cancel_pending()
        with self._condition:
            self._stopped = True; self._condition.notify_all(); active = list(self._active.items())
            if not drain:
                for judge, _thread in active: self._notify_selection(judge, False)
        for _judge, thread in active: thread.join(timeout=timeout)


__all__ = [name for name in globals() if not name.startswith("__")]
