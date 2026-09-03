"""Shared pytest fixtures.

The suite's pytest config treats ``pytest.PytestUnraisableExceptionWarning``
as an error, and on Python 3.13+/3.14 a ``sqlite3.Connection`` that is
garbage-collected without an explicit ``close()`` emits an unraisable
exception at deallocation. Deallocation happens whenever GC next runs —
usually inside a *different* test — so a leaked connection surfaced as
order-dependent flaky failures (``Exception ignored while finalizing
database connection ...`` landing in arbitrary ``test_coverage_gaps*``
tests).

The autouse fixture below closes any sqlite3 connection a test leaves open
at teardown, so a leak cannot survive into a later test's GC cycle.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _close_leaked_sqlite_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Close sqlite3 connections opened during a test and left open.

    ``sqlite3.connect`` is wrapped for the duration of the test so every
    connection created (including inside helper ``_make_store`` methods and
    the write-queue worker thread, which all call the module attribute) is
    tracked. Connections the test already closed are closed again, which is
    a documented no-op for ``sqlite3``.

    Note: only calls through the ``sqlite3.connect`` module attribute are
    intercepted; ``from sqlite3 import connect`` would bind the original
    function at import time and bypass tracking. The codebase has no such
    imports — keep it that way.
    """
    opened: list[sqlite3.Connection] = []
    original_connect = sqlite3.connect

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    yield
    for connection in opened:
        try:
            connection.close()
        except sqlite3.Error:
            pass
