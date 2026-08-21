"""Single-writer queue for SQLite run persistence."""
from __future__ import annotations

import os
import queue
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from .sqlite_schema import connect_database

SQLiteOperation = Callable[[sqlite3.Connection], Any]


@dataclass
class _WorkItem:
    operation: SQLiteOperation
    future: Future[Any]


class SQLiteWriteQueue:
    """Batch SQLite operations on one dedicated connection-owning thread.

    Worker threads enqueue short database operations and never serialize a
    state snapshot or share a SQLite connection. Each batch commits as one
    transaction. A future lets callers observe commit failures without making
    submission synchronous.
    """

    def __init__(
        self,
        path: str,
        *,
        batch_size: int = 64,
        flush_interval: float = 0.25,
        synchronous: str = "NORMAL",
        failure_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if flush_interval < 0:
            raise ValueError("flush_interval must not be negative")
        self.path = path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.synchronous = synchronous
        self.failure_callback = failure_callback
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue()
        self._state_lock = threading.Lock()
        self._started = False
        self._closing = False
        self._thread: threading.Thread | None = None
        self._failures: list[Exception] = []

    def start(self) -> None:
        """Start the writer thread once."""
        with self._state_lock:
            if self._started:
                return
            if self._closing:
                raise RuntimeError("SQLite writer is already closing")
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="sqlite-writer",
                daemon=True,
            )
            self._thread.start()

    def submit(self, operation: SQLiteOperation) -> Future[Any]:
        """Queue an operation and return a future resolved after commit."""
        if not callable(operation):
            raise TypeError("operation must be callable")
        future: Future[Any] = Future()
        with self._state_lock:
            if self._closing:
                future.set_exception(RuntimeError("SQLite writer is closed"))
                return future
        try:
            self.start()
        except RuntimeError as exc:
            future.set_exception(exc)
            return future
        with self._state_lock:
            if self._closing:
                future.set_exception(RuntimeError("SQLite writer is closed"))
                return future
            self._queue.put(_WorkItem(operation, future))
        return future

    def flush(self, timeout: float | None = None) -> None:
        """Wait until all operations submitted so far have committed."""
        future = self.submit(lambda _connection: None)
        future.result(timeout=timeout)

    def close(self, timeout: float | None = None) -> bool:
        """Drain queued work and stop; return false if timeout expires."""
        with self._state_lock:
            if not self._started:
                self._closing = True
                return True
            if not self._closing:
                self._closing = True
                self._queue.put(None)
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    @property
    def failures(self) -> list[Exception]:
        """Return a snapshot of writer failures."""
        with self._state_lock:
            return list(self._failures)

    @property
    def is_alive(self) -> bool:
        """Whether the writer thread is currently running."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    def _report_failure(self, exc: Exception) -> None:
        with self._state_lock:
            self._failures.append(exc)
        if self.failure_callback is not None:
            try:
                self.failure_callback(exc)
            except Exception:  # noqa: BLE001, S110 - reporting must not kill the writer
                pass

    def _execute_batch(self, connection: sqlite3.Connection,
                       batch: list[_WorkItem]) -> None:
        try:
            connection.execute("BEGIN")
            values = [item.operation(connection) for item in batch]
            connection.commit()
        except Exception as exc:  # noqa: BLE001 - fail the whole transaction
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            self._report_failure(exc)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for item, value in zip(batch, values):
            if not item.future.done():
                item.future.set_result(value)

    def _run(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self.path, synchronous=self.synchronous)
            while True:
                try:
                    first = self._queue.get(timeout=self.flush_interval)
                except queue.Empty:
                    with self._state_lock:
                        if self._closing and self._queue.empty():
                            break
                    continue
                if first is None:
                    self._queue.task_done()
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            self._queue.task_done()
                            continue
                        batch = [item]
                        while len(batch) < self.batch_size:
                            try:
                                item = self._queue.get_nowait()
                            except queue.Empty:
                                break
                            if item is not None:
                                batch.append(item)
                        self._execute_batch(connection, batch)
                        for _ in batch:
                            self._queue.task_done()
                    break

                batch = [first]
                while len(batch) < self.batch_size:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        batch.append(item)
                    else:
                        # Preserve the shutdown marker after this batch.
                        self._queue.put(None)
                        break
                self._execute_batch(connection, batch)
                for _ in batch:
                    self._queue.task_done()
        except Exception as exc:  # noqa: BLE001 - fail queued futures visibly
            self._report_failure(exc)
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None and not item.future.done():
                    item.future.set_exception(exc)
                self._queue.task_done()
        finally:
            if connection is not None:
                connection.close()
