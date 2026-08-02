"""OpenCode runner integration for AI Benchmark.

This module deliberately keeps the external-process adapter separate from the
existing OpenAI-compatible HTTP transport.  The CLI supplies a generated
OpenCode config and invokes one fresh ``opencode run`` process per plugin task.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


OPENCODE_BINARY = "opencode"

# ``opencode run --format`` accepts exactly ``default`` (formatted) or
# ``json`` (NDJSON event stream).  The adapter uses ``json`` so the final
# assistant answer can be extracted deterministically without ANSI/UI noise;
# ``plain`` does not exist in any released CLI and caused every invocation to
# be rejected with exit status 1.
OPENCODE_RUN_FORMAT = "json"


def validate_cli(binary: str = OPENCODE_BINARY, *, timeout: float = 10) -> None:
    """Validate the minimum non-interactive CLI contract before a run.

    The executable's presence is checked by the caller with ``shutil.which``;
    this probe verifies that the installed release exposes the flags used by
    this adapter and that the ``json`` run-format choice is advertised, so an
    outdated/unsupported CLI fails fast before any benchmark work is started.
    """
    try:
        probe = subprocess.run(
            [binary, "run", "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not validate OpenCode CLI: {type(exc).__name__}") from exc
    help_text = f"{probe.stdout}\n{probe.stderr}"
    if probe.returncode != 0:
        raise RuntimeError(f"OpenCode CLI rejected 'run --help' (status {probe.returncode})")
    missing = [flag for flag in ("--model", "--format", "--agent") if flag not in help_text]
    if missing:
        raise RuntimeError(
            "Installed OpenCode CLI is missing required run options: " + ", ".join(missing)
        )
    if not re.search(r"choices:[^]]*json", help_text, re.IGNORECASE | re.DOTALL):
        raise RuntimeError(
            "Installed OpenCode CLI does not advertise the 'json' run format "
            f"required for the {OPENCODE_RUN_FORMAT!r} output contract"
        )



@dataclass(frozen=True)
class OpenCodeProcessResult:
    """Captured result from one non-interactive OpenCode invocation."""

    text: str
    stderr: str
    elapsed: float
    error: str | None
    returncode: int | None


def slugify_source(source: str) -> str:
    """Turn a benchmark source label into a deterministic provider id."""
    value = re.sub(r"[^a-z0-9]+", "-", str(source).lower())
    return value.strip("-")


def opencode_model_name(source: str, api_model: str) -> str:
    """Build the requested ``{slugified_source}/{api_model}`` identifier."""
    slug = slugify_source(source)
    if not slug:
        raise ValueError(f"Source {source!r} produces an empty OpenCode provider id")
    if not api_model:
        raise ValueError(f"Source {source!r} has an empty api_model")
    return f"{slug}/{api_model}"


def _provider_base_url(api_url: str) -> str:
    """Convert a chat-completions URL into an OpenAI-compatible base URL."""
    parts = urlsplit(api_url)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), parts.query, ""))


def _provider_options(source_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Translate benchmark endpoint/header settings to provider options."""
    options: dict[str, Any] = {}
    api_url = source_cfg.get("api_url")
    if not isinstance(api_url, str) or not api_url:
        raise ValueError("source is missing a non-empty api_url")
    options["baseURL"] = _provider_base_url(api_url)

    headers = source_cfg.get("headers") or {}
    if not isinstance(headers, Mapping):
        raise ValueError("source headers must be an object")
    custom_headers: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        if key.lower() == "authorization" and isinstance(value, str):
            match = re.match(r"Bearer\s+(.+)$", value, re.IGNORECASE)
            if match:
                options["apiKey"] = match.group(1)
                continue
        if key.lower() != "content-type":
            custom_headers[key] = str(value)
    if custom_headers:
        options["headers"] = custom_headers

    # OpenCode's provider options support a request timeout.  The benchmark
    # timeout is injected by generate_config() when available.
    if "timeout" in source_cfg:
        options["timeout"] = source_cfg["timeout"]
    return options


def _model_context_limit(api_model: str, default: int = 131072) -> int:
    """Infer an OpenCode context limit from a benchmark model id.

    Benchmark model ids conventionally carry their context window as a
    ``-NNk`` / ``-NNm`` suffix (e.g. ``qwen3.6:27b-128k``).  OpenCode's
    provider model schema requires ``limit.context`` whenever ``limit`` is
    present, so the generated config always writes it; ids without a suffix
    fall back to a conservative default.
    """
    match = re.search(r"-(\d+)([km])(?:$|[^a-z])", api_model, re.IGNORECASE)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit == "k":
            return amount * 1024
        return amount * 1024 * 1024
    return default


def _agent_id(target_key: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", target_key).strip("-")
    return f"benchmark-{value or 'target'}"


def generate_config(
    source_config: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
    path: str | os.PathLike[str],
    *,
    timeout: int | float | None = None,
    token_levels: Sequence[int] | None = None,
    benchmark_config: Mapping[str, Any] | None = None,
    plugin_temperatures: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and retain an OpenCode config for all OpenCode targets.

    ``targets`` is the resolved target map.  Each entry must contain
    ``source`` and ``api_model`` and may contain ``system_prompt`` and
    ``is_agent``.  The returned config is also written as an exact artifact;
    callers should treat it as credential-bearing data.
    """
    providers: dict[str, Any] = {}
    agents: dict[str, Any] = {}
    agent_ids: dict[str, str] = {}
    mappings: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}

    for target_key, info in targets.items():
        source = info.get("source")
        api_model = info.get("api_model")
        if not isinstance(source, str):
            raise ValueError(f"Target {target_key!r} has no valid source")
        if not isinstance(api_model, str) or not api_model:
            raise ValueError(f"Target {target_key!r} has no valid api_model")
        mapped_model = opencode_model_name(source, api_model)
        mappings.setdefault(mapped_model, []).append(target_key)
        provider_id = slugify_source(source)
        if provider_id not in providers:
            source_cfg = source_config.get(source)
            if source_cfg is None:
                raise ValueError(f"Target {target_key!r} references unknown source {source!r}")
            options = _provider_options(source_cfg)
            if timeout is not None:
                options["timeout"] = timeout * 1000
            providers[provider_id] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": source,
                "options": options,
                "models": {},
            }
        model_options: dict[str, Any] = {"name": api_model}
        # OpenCode requires both ``context`` and ``output`` inside ``limit``;
        # writing only ``output`` makes the whole config fail validation.
        model_options["limit"] = {
            "context": _model_context_limit(api_model),
            "output": max(token_levels) if token_levels else 16384,
        }
        providers[provider_id]["models"][api_model] = model_options

        if info.get("is_agent") and info.get("system_prompt"):
            aid = _agent_id(target_key)
            agent_ids[target_key] = aid
            agents[aid] = {
                "description": f"AI Benchmark agent for {target_key}",
                "mode": "primary",
                "model": mapped_model,
                "prompt": info["system_prompt"],
                "permission": {"edit": "deny", "bash": "deny"},
            }

    duplicate_targets = {key: value for key, value in mappings.items() if len(value) > 1}
    if duplicate_targets:
        details = "; ".join(f"{model}: {', '.join(names)}" for model, names in duplicate_targets.items())
        raise ValueError(f"OpenCode model mapping collision: {details}")

    first_model = next(iter(mappings), None)
    unsupported: list[str] = []
    if any(info.get("drop_params") for info in targets.values()):
        unsupported.append("per-target drop_params")
    if benchmark_config and ("seed" in benchmark_config or benchmark_config.get("seed") is not None):
        unsupported.append("seed")
    if benchmark_config and any(
        key in benchmark_config for key in ("retry_on_429", "max_429_retries", "backoff_seconds", "max_backoff_seconds")
    ):
        unsupported.append("HTTP retry/backoff controls")
    if plugin_temperatures:
        unsupported.append("per-plugin temperature overrides")
    unsupported.append("HTTP streaming/TTFT telemetry")
    projection = {
        "translated": ["source api_url/baseURL", "authorization and custom headers", "resolved api_model", "timeout", "token_levels"],
        "unsupported": unsupported,
        "note": "Unsupported HTTP-only fields are not fabricated in OpenCode results.",
    }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": providers,
        "permission": {"edit": "deny", "bash": "deny"},
    }
    if first_model:
        config["model"] = first_model
    if agents:
        config["agent"] = agents

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    try:
        output_path.chmod(0o600)
    except OSError:
        pass
    return {
        "config": config,
        "agent_ids": agent_ids,
        "mappings": mappings,
        "projection": projection,
        "path": str(output_path),
    }


def _extract_final_text(stdout: bytes) -> tuple[str, str | None]:
    """Extract the final assistant answer from ``opencode run --format json``.

    ``--format json`` emits one NDJSON event per line.  Completed assistant
    text parts arrive as ``{"type": "text", "part": {"type": "text",
    "text": ..., "time": {"end": ...}}}``; the CLI also emits ``tool_use``,
    ``step_start``, ``reasoning``, and ``error`` events, none of which carry
    final answer text.  Concatenate text events in order (the model may emit
    several parts), and fall back to the raw decoded stdout when the stream is
    not NDJSON at all.

    Returns ``(text, error)`` where ``error`` is an ``error`` event payload
    string when the session itself reported one.
    """
    text_parts: list[str] = []
    session_error: str | None = None
    saw_json = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        saw_json = True
        event_type = event.get("type")
        if event_type == "text":
            part = event.get("part") or {}
            if isinstance(part, dict) and part.get("type") == "text":
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text)
        elif event_type == "error":
            payload = event.get("error")
            if isinstance(payload, dict):
                session_error = payload.get("message") or payload.get("name")
            elif isinstance(payload, str):
                session_error = payload
    if saw_json:
        return "\n".join(text_parts).strip(), session_error
    return stdout.decode("utf-8", errors="replace").strip(), None


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate OpenCode and its process group where supported."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass


def run_process(
    prompt: str,
    *,
    config_path: str,
    model: str,
    timeout: int | float,
    binary: str = OPENCODE_BINARY,
    agent: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    target_key: str = "target",
    plugin_id: str = "plugin",
    stop_event: Any = None,
) -> OpenCodeProcessResult:
    """Run one isolated OpenCode task and capture stdout/stderr separately."""
    command = [binary, "run", "--model", model, "--format", OPENCODE_RUN_FORMAT]
    if agent:
        command.extend(["--agent", agent])
    command.append(prompt)
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = os.path.abspath(config_path)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    stdout = b""
    stderr = b""
    error: str | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        deadline = started + max(0, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if stop_event is not None and stop_event.is_set():
                _terminate_process(process)
                error = "Cancelled"
                break
            if remaining <= 0:
                _terminate_process(process)
                error = f"OpenCode timed out after {timeout}s"
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.poll() is None:
            _terminate_process(process)
        if process.returncode is not None and (not stdout and not stderr):
            try:
                stdout, stderr = process.communicate(timeout=1)
            except (subprocess.TimeoutExpired, ValueError):
                pass
        returncode = process.returncode
    except OSError as exc:
        return OpenCodeProcessResult("", "", time.monotonic() - started,
                                    f"Could not start OpenCode: {type(exc).__name__}", None)
    finally:
        if process is not None and process.poll() is None:
            _terminate_process(process)

    text, session_error = _extract_final_text(stdout)
    diagnostic = stderr.decode("utf-8", errors="replace")
    elapsed = time.monotonic() - started
    if output_dir:
        log_dir = Path(output_dir) / "logs" / re.sub(r"[^\w.()-]+", "_", target_key)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{plugin_id}.stdout.txt").write_bytes(stdout)
        (log_dir / f"{plugin_id}.stderr.txt").write_bytes(stderr)
    if error is None and session_error:
        error = f"OpenCode session error: {session_error}"
    if error is None and returncode != 0:
        error = f"OpenCode exited with status {returncode}"
    if error is None and not text:
        error = "OpenCode returned an empty response"
    return OpenCodeProcessResult(text, diagnostic, elapsed, error, returncode)
