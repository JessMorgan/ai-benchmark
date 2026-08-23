"""Shared subprocess lifecycle helpers for isolated benchmark runners."""
from __future__ import annotations

import contextlib
import subprocess
from typing import Any

import psutil


def terminate_process_tree(process: Any, *, grace_seconds: float = 1.0) -> None:
    """Terminate a process and descendants, escalating to kill if needed."""
    if process is None or process.poll() is not None:
        return
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    processes = [root, *root.children(recursive=True)]
    for child in reversed(processes):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            child.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    for child in reversed(processes):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            child.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)


def close_process_pipes(process: Any) -> None:
    """Close parent-owned subprocess pipes without masking cleanup errors."""
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()


def terminate_and_close(process: Any, *, grace_seconds: float = 1.0) -> None:
    """Terminate a process tree and close all parent-side pipes."""
    if process is None:
        return
    terminate_process_tree(process, grace_seconds=grace_seconds)
    close_process_pipes(process)
