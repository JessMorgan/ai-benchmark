"""Pi SDK runner integration for AI Benchmark.

Pi is deliberately run through one isolated local Node worker per benchmark
cell. The worker owns the JavaScript SDK and emits the small ``pi-worker-v1``
NDJSON protocol; this module owns process lifetime, event extraction, timeout,
cancellation, and diagnostic artifacts.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Any, cast

from .observer import TaskObserver
from .process_supervisor import terminate_process_tree

PI_WORKER_PROTOCOL = "pi-worker-v1"
PI_WORKER_VERSION = "1.0.0"
PI_SDK_VERSION = "0.84.2"
PI_WORKER_ADAPTER_VERSION = "1.0.0"
PI_DEFAULT_NODE = "node"
PI_DEFAULT_TIMEOUT = 300.0
PI_MIN_NODE = (22, 19, 0)


@dataclass(frozen=True)
class PiProcessResult:
    """Captured and normalized result from one Pi worker invocation."""

    text: str
    stderr: str
    elapsed: float
    error: str | None
    returncode: int | None
    think_text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    tool_called: bool = False
    tools: tuple[str, ...] = ()
    provider: str | None = None
    requested_tools: tuple[str, ...] = ()
    permissions: dict[str, str] = field(default_factory=dict)
    sdk_version: str = PI_SDK_VERSION
    worker_version: str = PI_WORKER_VERSION
    adapter_version: str = PI_WORKER_ADAPTER_VERSION


def _worker_path(worker: str | os.PathLike[str] | None = None) -> Path:
    if worker:
        return Path(worker).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "pi-worker" / "worker.mjs"


def _node_version(node_path: str) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            [node_path, "--version"], capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"v?(\d+)\.(\d+)(?:\.(\d+))?", result.stdout or result.stderr)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


@lru_cache(maxsize=8)
def resolve_pi_worker(
    worker: str | Path | None = None,
    *,
    node: str = PI_DEFAULT_NODE,
) -> tuple[str, str]:
    """Resolve, validate, and preflight the project-local Pi worker."""
    worker_path = _worker_path(worker)
    node_path = shutil.which(node) or (node if os.path.isfile(node) else None)
    if node_path is None:
        raise RuntimeError(
            "Pi requires Node.js (>=22.19) but no 'node' executable was found. "
            "Install Node and run 'npm --prefix pi-worker install'."
        )
    version = _node_version(node_path)
    if version is None or version < PI_MIN_NODE:
        required = ".".join(str(part) for part in PI_MIN_NODE)
        actual = ".".join(str(part) for part in version) if version else "unknown"
        raise RuntimeError(f"Pi requires Node.js >= {required}; found {actual}.")
    if not worker_path.is_file():
        raise RuntimeError(f"Pi worker not found at {worker_path}")
    try:
        probe = subprocess.run(
            [node_path, "--check", str(worker_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not validate Pi worker: {type(exc).__name__}") from exc
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        raise RuntimeError(
            "Pi worker syntax check failed" + (f": {detail[-1]}" if detail else "")
        )
    try:
        preflight = subprocess.run(
            [node_path, str(worker_path), "--preflight"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not preflight Pi worker: {type(exc).__name__}") from exc
    if preflight.returncode != 0:
        detail = (preflight.stderr or preflight.stdout).strip().splitlines()
        raise RuntimeError(
            "Pi worker preflight failed" + (f": {detail[-1]}" if detail else "")
        )
    return node_path, str(worker_path)


def run_pi_probe(
    source_config: Mapping[str, Any],
    source: str,
    api_model: str,
    *,
    timeout: float = PI_DEFAULT_TIMEOUT,
    pi_config: Mapping[str, Any] | None = None,
    node: str = PI_DEFAULT_NODE,
    worker: str | Path | None = None,
) -> dict[str, Any]:
    """Run the non-scoring Pi compatibility probe for one target."""
    node_path, worker_path = resolve_pi_worker(worker, node=node)
    result = run_process(
        "Reply with the single word OK.",
        source_config=source_config,
        source=source,
        api_model=api_model,
        max_tokens=64,
        timeout=timeout,
        pi_config=pi_config,
        node=node_path,
        worker=worker_path,
        target_key=api_model,
        plugin_id="pi-probe",
    )
    return {
        "source": source,
        "api_model": api_model,
        "node": pi_version(node_path),
        "worker": worker_path,
        "protocol": PI_WORKER_PROTOCOL,
        "worker_version": result.worker_version,
        "sdk_version": result.sdk_version,
        "text": result.text,
        "thinking_text": result.think_text,
        "error": result.error,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "requested_tools": list(result.requested_tools),
        "actual_tools": list(result.tools),
        "tool_called": result.tool_called,
        "passed": result.error is None and bool(result.text.strip()),
    }


def pi_version(node: str = PI_DEFAULT_NODE) -> str | None:
    """Return the installed Node version for run metadata."""
    try:
        result = subprocess.run(
            [node, "--version"], capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (result.stdout or result.stderr).strip().splitlines()
    return value[0] if value else None


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.()-]+", "_", value)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate the worker and its descendants."""
    terminate_process_tree(process)


def _pump_lines(stream: Any, queue: Queue[tuple[str, str | None]], kind: str) -> None:
    try:
        for line in stream:
            queue.put((kind, line))
    except (OSError, ValueError):
        pass
    finally:
        queue.put((kind, None))


def _source_payload(source_config: Mapping[str, Any], source: str, pi_config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = source_config.get(source) or {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"Source {source!r} must be an object")
    payload: dict[str, Any] = {
        "name": source,
        "api_url": cfg.get("api_url"),
        "api_key": cfg.get("api_key") or cfg.get("apiKey"),
        "headers": cfg.get("headers") or {},
    }
    if pi_config:
        payload["pi"] = dict(pi_config)
    return payload


def run_process(
    prompt: str,
    *,
    source_config: Mapping[str, Any],
    source: str,
    api_model: str,
    max_tokens: int,
    timeout: float,
    attempt: int = 1,
    system_prompt: str | None = None,
    temperature: float | None = None,
    reasoning: bool = False,
    prompt_altered: str = "none",
    pi_config: Mapping[str, Any] | None = None,
    node: str = PI_DEFAULT_NODE,
    worker: str | Path | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    target_key: str = "target",
    plugin_id: str = "plugin",
    stop_event: Any = None,
    observer: TaskObserver | None = None,
) -> PiProcessResult:
    """Run one isolated Pi request and retain partial output on failure."""
    node_path, worker_path = resolve_pi_worker(worker, node=node)
    cfg = source_config.get(source) or {}
    effective_pi = dict(pi_config or (cfg.get("pi") if isinstance(cfg, Mapping) else {}) or {})
    tools = effective_pi.get("tools", [])
    raw_permissions = effective_pi.get("permissions", {})
    if not isinstance(tools, list):
        raise TypeError("Pi tools must be a list")
    if not isinstance(raw_permissions, Mapping):
        raise TypeError("Pi permissions must be an object")
    permissions = dict(raw_permissions)
    request = {
        "protocol": PI_WORKER_PROTOCOL,
        "attempt": attempt,
        "prompt": prompt,
        "api_model": api_model,
        "max_tokens": int(max_tokens),
        "timeout_ms": max(1, int(float(timeout) * 1000)),
        "system_prompt": system_prompt or effective_pi.get("system_prompt"),
        "temperature": temperature,
        "reasoning": bool(reasoning or effective_pi.get("reasoning", False)),
        "prompt_altered": prompt_altered,
        "thinking_budgets": effective_pi.get("thinking_budgets"),
        "max_tool_calls": int(effective_pi.get("max_tool_calls", 50)),
        "tools": [str(value) for value in tools],
        "permissions": {str(key): str(value) for key, value in permissions.items()},
        "cwd": os.getcwd(),
        "source": _source_payload(source_config, source, effective_pi),
    }
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    text_parts: list[str] = []
    think_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    error: str | None = None
    truncated = False
    tool_called = False
    actual_tools: tuple[str, ...] = ()
    requested_tools: tuple[str, ...] = tuple(str(value) for value in tools)
    provider: str | None = None
    worker_version = PI_WORKER_VERSION
    sdk_version = PI_SDK_VERSION
    events: Queue[tuple[str, str | None]] = Queue()
    try:
        process = subprocess.Popen(
            [node_path, worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise OSError("Pi worker did not expose all standard streams")
        process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        process.stdin.close()
        for stream, kind in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            threading.Thread(target=_pump_lines, args=(stream, events, kind), daemon=True).start()

        open_streams = {"stdout", "stderr"}
        deadline = started + max(0.01, float(timeout))
        while open_streams:
            if stop_event is not None and stop_event.is_set():
                _terminate_process(process)
                error = "Cancelled"
                break
            if time.monotonic() >= deadline:
                _terminate_process(process)
                error = f"Pi timed out after {timeout}s"
                break
            try:
                kind, line = events.get(timeout=0.1)
            except Empty:
                if process.poll() is not None and not open_streams:
                    break
                continue
            if line is None:
                open_streams.discard(kind)
                continue
            if kind == "stderr":
                stderr_parts.append(line)
                continue
            stdout_parts.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("protocol") != PI_WORKER_PROTOCOL:
                continue
            data = cast(dict[str, Any], event.get("data")) if isinstance(event.get("data"), dict) else {}
            event_type = event.get("event")
            if event_type == "worker_started":
                worker_version = str(data.get("worker_version") or worker_version)
                sdk_version = str(data.get("sdk_version") or sdk_version).removeprefix(
                    "@earendil-works/pi-coding-agent@"
                )
            elif event_type == "text_delta":
                delta = data.get("text")
                if isinstance(delta, str):
                    text_parts.append(delta)
                    (observer or TaskObserver.noop()).chunk(delta)
            elif event_type == "reasoning_delta":
                delta = data.get("text")
                if isinstance(delta, str):
                    think_parts.append(delta)
                    (observer or TaskObserver.noop()).think_chunk(delta)
            elif event_type == "usage":
                if isinstance(data.get("usage"), dict):
                    usage = data["usage"]
            elif event_type == "finish":
                finish_reason = data.get("finish_reason")
                truncated = bool(data.get("truncated", finish_reason == "length"))
                tool_called = bool(data.get("tool_called", tool_called))
                actual_tools = tuple(str(value) for value in data.get("tools", []) or [])
                provider = data.get("provider")
                if isinstance(data.get("tools"), list):
                    actual_tools = tuple(str(value) for value in data["tools"])
                if isinstance(data.get("usage"), dict):
                    usage = data["usage"]
            elif event_type == "error":
                error = str(data.get("message") or "Pi worker error")
        if process.poll() is None:
            _terminate_process(process)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = f"Could not run Pi worker: {type(exc).__name__}: {exc}"
        if process is not None:
            _terminate_process(process)

    elapsed = time.monotonic() - started
    returncode = process.returncode if process is not None else None
    if error is None and returncode not in (None, 0):
        error = f"Pi worker exited with status {returncode}"
    text = "".join(text_parts)
    think_text = "".join(think_parts)
    if error is None and not text.strip():
        error = "Pi returned an empty response"
    if output_dir:
        log_dir = Path(output_dir) / "logs" / _safe_name(target_key)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{plugin_id}.stdout.ndjson").write_text("".join(stdout_parts), encoding="utf-8")
        (log_dir / f"{plugin_id}.stderr.txt").write_text("".join(stderr_parts), encoding="utf-8")
    return PiProcessResult(
        text=text,
        stderr="".join(stderr_parts),
        elapsed=elapsed,
        error=error,
        returncode=returncode,
        think_text=think_text,
        finish_reason=finish_reason,
        usage=usage,
        truncated=truncated,
        tool_called=tool_called,
        tools=actual_tools,
        provider=provider,
        requested_tools=requested_tools,
        permissions={str(key): str(value) for key, value in permissions.items()},
        sdk_version=sdk_version,
        worker_version=worker_version,
    )
