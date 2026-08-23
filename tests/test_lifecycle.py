"""Tests for lifecycle invariants and shutdown supervision."""
import threading
import time

import pytest

from benchmark.lifecycle import (
    LifecycleInvariantError,
    ShutdownSupervisor,
    validate_model_info,
)


def _info(**updates):
    value = {
        "status": "pending",
        "running_pids": [],
        "attempt": 0,
        "elapsed": 0,
        "preloading": False,
    }
    value.update(updates)
    return value


def test_validate_snapshot_and_reject_invalid_model_names():
    from benchmark.lifecycle import validate_snapshot
    validate_snapshot({"m": _info()})
    with pytest.raises(LifecycleInvariantError, match="model name"):
        validate_model_info("", _info())
    with pytest.raises(LifecycleInvariantError, match="snapshot"):
        validate_snapshot([])


def test_shutdown_supervisor_rejects_invalid_timeout_and_deadline_expiry():
    for value in (True, "1", 0):
        with pytest.raises(ValueError):
            ShutdownSupervisor(value)
    supervisor = ShutdownSupervisor(0.01)
    supervisor.results.append(type("Result", (), {"elapsed": 0.02, "completed": True})())
    assert not supervisor.run("late", lambda: True)
    assert supervisor.results[-1].error == "deadline exceeded"


def test_valid_model_lifecycle_record():
    validate_model_info("m", _info(status="running", running_pids=["p"]))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"running_pids": ["p", "p"]}, "duplicates"),
        ({"running_pids": [1]}, "contain strings"),
        ({"attempt": -1}, "attempt"),
        ({"status": "bogus"}, "status"),
        ({"preloading": "yes"}, "preloading"),
    ],
)
def test_invalid_model_lifecycle_records_are_rejected(changes, message):
    with pytest.raises(LifecycleInvariantError, match=message):
        validate_model_info("m", _info(**changes))


def test_shutdown_supervisor_records_success_and_failure():
    supervisor = ShutdownSupervisor(2.0)
    assert supervisor.run("cancel", lambda: True)
    assert not supervisor.run("close", lambda: False)
    assert supervisor.successful is False
    assert [item["name"] for item in supervisor.as_dict()] == ["cancel", "close"]


def test_shutdown_supervisor_records_exception():
    supervisor = ShutdownSupervisor(2.0)

    def fail():
        raise RuntimeError("broken")

    assert not supervisor.run("persist", fail)
    assert supervisor.results[0].error == "RuntimeError: broken"


def test_shutdown_supervisor_bounds_remaining_deadline():
    supervisor = ShutdownSupervisor(0.01)
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.03)
        return True

    assert not supervisor.run("slow", slow)
    assert started.is_set()
    assert "deadline" in (supervisor.results[0].error or "")
