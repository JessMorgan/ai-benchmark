"""ChatPlayground.ai interactive-web source adapter (subprocess-isolated).

ChatPlayground.ai (https://web.chatplayground.ai) is a closed, JavaScript-
rendered web app with no public API: it authenticates with a username/password
(Clerk) and renders every chat interaction client-side. This module drives that
UI with Playwright — but Playwright runs in a dedicated worker subprocess
(:mod:`benchmark.chatplayground_worker`), never inside the benchmark runner.

Why a subprocess: Playwright's sync API is not thread-safe and is documented
as main-thread-only. The benchmark runs each model in its own worker thread,
and exercising Playwright from such a thread (greenlets plus a per-thread
asyncio loop) can corrupt the interpreter and segfault the entire run — which
happened in production (a SIGSEGV while a ChatPlayground model was mid-turn).
Isolating the browser in a child process means a native crash there surfaces as
a per-request error instead of taking the benchmark down with it; the next
request simply spawns a fresh worker.

Each model is addressed by the slug used in the site's ``/chat/<slug>`` route
(e.g. ``deepseek-v4-pro``, ``gpt-5.6-terra``, ``gemini-3-flash``). Use
``list_models(cfg)`` (or ``python -m benchmark.chatplayground``) to enumerate
the slugs exposed by the sidebar's "AI MODELS" list.

The parent never imports Playwright. It proxies a JSON-lines protocol over the
worker's stdin/stdout and serializes requests under a module lock; the worker
serializes browser operations under its own lock, so a single logged-in session
is reused across plugin tasks instead of re-authenticating for every request.
"""

import atexit
import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

DEFAULT_BASE_URL = "https://web.chatplayground.ai"

# Selectors for the single-page app, captured against the live site. Every key
# can be overridden through the source's ``selectors`` mapping. ``login_submit``
# and ``send_button`` are accessible names (matched exactly via
# ``get_by_role``), not CSS selectors, because the buttons share a text
# substring with other controls ("Continue with Google", "Send" vs nothing).
DEFAULT_SELECTORS = {
    # Route the Clerk login form lives at (joined to ``base_url``).
    "login_url": "/login",
    "email_input": "input[name=identifier]",
    "password_input": "input[name=password]",
    # Accessible name of the Clerk submit button ("Continue", not "Continue with Google").
    "login_submit": "Continue",
    # Prefix for single-model chat routes; the model slug is appended.
    "chat_path": "/chat",
    # The prompt composer and its submit affordance.
    "prompt_input": "textarea[name=input]",
    "send_button": "Send",
    # Shown only while a response is still generating; its disappearance marks
    # completion.
    "stop_generation": "button:has-text('Stop')",
    # Exact label preceding each assistant answer in the transcript.
    "assistant_label": "ASSISTANT",
    # Extra settle time (ms) after the stop affordance disappears.
    "settle_ms": 1500,
    # Overall selector wait timeout (ms).
    "wait_timeout_ms": 30000,
}

# ─── Worker subprocess lifecycle ─────────────────────────────────────────────
# The browser lives in a child process (``benchmark.chatplayground_worker``).
# A native crash there (Playwright/greenlet/Chromium) is contained: the parent
# detects EOF on the worker's stdout, tears it down, and reports a per-request
# error. The next request spawns a fresh worker, so a single browser crash
# costs one plugin leg instead of the whole run.

# Extra seconds beyond the request timeout before the parent kills a worker
# that stopped responding (a hung browser turn). The worker enforces its own
# generation timeout, so this is only a backstop for a wedged process.
_WORKER_GRACE = 120.0
# Hard ceiling for ops without a request timeout (``list_models`` / ``probe``):
# login + enumeration should never approach this.
_OP_CEILING = 180.0

# All requests are serialized under this lock: only one browser operation is in
# flight at a time, matching the single-session model of the worker.
_lock = threading.RLock()
_proc: subprocess.Popen | None = None
_queue: queue.Queue | None = None
_reader: threading.Thread | None = None
_stderr_chunks: list[str] = []
_next_id = 0


def is_chatplayground(cfg) -> bool:
    """Return whether a source config selects the ChatPlayground protocol."""
    return isinstance(cfg, dict) and cfg.get("api_protocol") == "chatplayground"


def credentials(cfg) -> tuple[str, str]:
    """Return ``(email, password)`` from a ChatPlayground source config."""
    if not isinstance(cfg, dict):
        return "", ""
    email = cfg.get("email") or cfg.get("username") or ""
    password = cfg.get("password") or ""
    return str(email), str(password)


def selectors(cfg) -> dict:
    """Merge the default selectors with any per-source ``selectors`` overrides."""
    merged = dict(DEFAULT_SELECTORS)
    if isinstance(cfg, dict) and isinstance(cfg.get("selectors"), dict):
        merged.update(cfg["selectors"])
    return merged


def _worker_command() -> list[str]:
    """Return the argv that starts the browser worker subprocess."""
    return [sys.executable, "-m", "benchmark.chatplayground_worker"]


def _reader_loop(proc, out_queue) -> None:
    """Pump worker stdout lines onto ``out_queue``; signal EOF on close."""
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            out_queue.put(("msg", msg))
    except (OSError, ValueError):
        pass
    finally:
        out_queue.put(("eof", None))


def _stderr_loop(proc, chunks) -> None:
    """Append worker stderr lines to ``chunks`` (bounded) for crash diagnostics."""
    try:
        for line in proc.stderr:
            chunks.append(line)
            if len(chunks) > 200:
                chunks.pop(0)
    except (OSError, ValueError):
        pass


def _terminate(proc) -> None:
    """Terminate the worker and its process group where supported."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass


def _teardown_worker() -> None:
    """Close the worker's stdin, terminate it, and reset module state."""
    global _proc, _queue, _reader
    proc, _proc = _proc, None
    _queue = None
    _reader = None
    if proc is None:
        return
    if proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.close()
    _terminate(proc)
    with contextlib.suppress(Exception):
        proc.wait(timeout=2)


def _ensure_worker() -> subprocess.Popen:
    """Return a live worker subprocess, spawning one if needed."""
    global _proc, _queue, _reader, _stderr_chunks
    if _proc is not None and _proc.poll() is None:
        return _proc
    _teardown_worker()
    _queue = queue.Queue()
    _stderr_chunks = []
    try:
        _proc = subprocess.Popen(
            _worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        _proc = None
        raise RuntimeError(
            f"Could not start ChatPlayground worker: {type(exc).__name__}: {exc}"
        ) from exc
    _reader = threading.Thread(target=_reader_loop, args=(_proc, _queue), daemon=True)
    _reader.start()
    _stderr_thread = threading.Thread(
        target=_stderr_loop, args=(_proc, _stderr_chunks), daemon=True
    )
    _stderr_thread.start()
    return _proc


def _worker_diag() -> str:
    """Build a crash diagnostic (exit code + stderr tail) for the dead worker."""
    proc = _proc
    rc = proc.poll() if proc is not None else None
    tail = "\n".join(_stderr_chunks[-15:]).strip() if _stderr_chunks else ""
    code = f"exit code {rc}" if rc is not None else "unexpectedly"
    if tail:
        return f"{code}; stderr: {tail[-1000:]}"
    return code


def _send_request(op, cfg, *, stop_event=None, timeout=None, **payload) -> dict:
    """Send one op to the worker and wait for its response.

    Serialized under the module lock (one browser operation at a time). A
    worker death is detected via EOF on its stdout and surfaced as an error; the
    next request spawns a fresh worker. ``stop_event`` (Ctrl+C) or the deadline
    (request ``timeout`` + ``_WORKER_GRACE``) terminates the worker so a hung
    browser turn cannot block the benchmark forever.
    """
    global _next_id
    with _lock:
        proc = _ensure_worker()
        stdin = proc.stdin
        out_queue = _queue
        if stdin is None or out_queue is None:
            _teardown_worker()
            return {"ok": False, "error": "ChatPlayground worker unavailable"}
        _next_id += 1
        req_id = _next_id
        msg = {"id": req_id, "op": op, "config": cfg}
        msg.update(payload)
        try:
            stdin.write(json.dumps(msg) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError):
            _teardown_worker()
            return {"ok": False, "error": "ChatPlayground worker died before handling the request"}
        try:
            timeout = float(timeout) if timeout else 0.0
        except (TypeError, ValueError):
            timeout = 0.0
        ceiling = (timeout + _WORKER_GRACE) if timeout else _OP_CEILING
        deadline = time.monotonic() + ceiling
        while True:
            if stop_event is not None and stop_event.is_set():
                _teardown_worker()
                return {"ok": False, "error": "ChatPlayground request cancelled (worker terminated)"}
            if time.monotonic() >= deadline:
                _teardown_worker()
                return {"ok": False, "error": f"ChatPlayground request timed out after {ceiling:g}s"}
            try:
                kind, data = out_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "eof":
                diag = _worker_diag()
                _teardown_worker()
                return {"ok": False, "error": f"ChatPlayground worker crashed: {diag}"}
            if isinstance(data, dict) and data.get("id") == req_id:
                return data


def request(cfg, model, prompt, *, timeout, stop_event=None, system_prompt=None):
    """Send ``prompt`` through a logged-in ChatPlayground session.

    Returns ``(text, error, elapsed_seconds)``. The whole browser turn runs in
    the worker subprocess; the answer is buffered (no per-token streaming), and
    an optional ``system_prompt`` is folded into the prompt by the worker.
    """
    if stop_event is not None and stop_event.is_set():
        return "", "ChatPlayground request cancelled", 0.0
    started = time.time()
    resp = _send_request(
        "send", cfg, stop_event=stop_event, timeout=timeout,
        model=model, prompt=prompt, system_prompt=system_prompt,
    )
    elapsed = round(time.time() - started, 1)
    if resp.get("ok"):
        return resp.get("text", ""), None, elapsed
    return "", resp.get("error", "ChatPlayground request failed"), elapsed


def list_models(cfg) -> list[str]:
    """Enumerate the model slugs exposed by the sidebar's "AI MODELS" list."""
    resp = _send_request("list_models", cfg)
    if resp.get("ok"):
        return [str(slug) for slug in (resp.get("models") or [])]
    raise RuntimeError(resp.get("error", "ChatPlayground model enumeration failed"))


def probe(cfg) -> dict:
    """Capture diagnostic DOM information for selector finalization."""
    resp = _send_request("probe", cfg)
    if resp.get("ok"):
        return resp.get("probe") or {}
    raise RuntimeError(resp.get("error", "ChatPlayground probe failed"))


def config_from_env() -> dict:
    """Build a ChatPlayground source config from environment variables.

    ``CHATPLAYGROUND_EMAIL``/``CHATPLAYGROUND_PASSWORD`` supply the Clerk
    credentials; ``CHATPLAYGROUND_BASE_URL`` and ``CHATPLAYGROUND_HEADLESS``
    (default ``"1"``) override the site and browser mode.
    """
    return {
        "api_protocol": "chatplayground",
        "base_url": os.environ.get("CHATPLAYGROUND_BASE_URL", DEFAULT_BASE_URL),
        "email": os.environ.get("CHATPLAYGROUND_EMAIL", ""),
        "password": os.environ.get("CHATPLAYGROUND_PASSWORD", ""),
        "headless": os.environ.get("CHATPLAYGROUND_HEADLESS", "1") == "1",
    }


def _complete_source_config(source_cfg: dict) -> dict:
    """Return ``source_cfg`` with the browser-safe defaults benchmark needs.

    Browser work is serialized under a module lock regardless of scheduler
    settings, but pinning both thread limits to 1 (and skipping the HTTP-only
    preload probe) keeps the scheduler from queueing parallel browser turns.
    """
    cfg = dict(source_cfg)
    cfg.setdefault("api_protocol", "chatplayground")
    cfg.setdefault("model_thread_limit", 1)
    cfg.setdefault("plugin_thread_limit", 1)
    cfg.setdefault("preload", False)
    return cfg


def generate_config(source_cfg: dict | None = None, models: list[str] | None = None) -> dict:
    """Return a ready-to-run benchmark config dict for a ChatPlayground source.

    When ``source_cfg`` is omitted it is built from the environment (see
    :func:`config_from_env`). When ``models`` is omitted the model slugs are
    enumerated from the live sidebar's "AI MODELS" list (via the worker
    subprocess). Each discovered slug becomes a ``models`` entry pointing at
    the ``ChatPlayground`` source.
    """
    if source_cfg is None:
        source_cfg = config_from_env()
    if not credentials(source_cfg)[0]:
        raise ValueError(
            "ChatPlayground credentials are missing; set CHATPLAYGROUND_EMAIL "
            "and CHATPLAYGROUND_PASSWORD"
        )
    if models is None:
        models = list_models(source_cfg)
    if not models:
        raise RuntimeError("No ChatPlayground models discovered from the UI")
    return {
        "output_dir": "benchmark-results",
        "timeout": 600,
        "token_levels": [16384],
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {"ChatPlayground": _complete_source_config(source_cfg)},
        "models": {slug: "ChatPlayground" for slug in models},
    }


def _cli_probe() -> dict:
    """Run a probe from a minimal source config built out of environment vars."""
    return probe(config_from_env())


def _close_session() -> None:
    """Tear down the worker subprocess (and its browser), if any."""
    with _lock:
        _teardown_worker()


def _atexit_cleanup() -> None:
    """Best-effort worker teardown at interpreter exit (no lock: shutdown)."""
    _teardown_worker()


atexit.register(_atexit_cleanup)


if __name__ == "__main__":  # pragma: no cover - diagnostic entry point
    result = _cli_probe()  # pragma: no cover
    print(json.dumps(result, indent=2, default=str))  # pragma: no cover
