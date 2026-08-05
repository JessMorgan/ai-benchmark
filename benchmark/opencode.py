"""OpenCode runner integration for AI Benchmark.

This module deliberately keeps the external-process adapter separate from the
existing OpenAI-compatible HTTP transport.  The CLI supplies a generated
OpenCode config and invokes one fresh ``opencode run`` process per plugin task.

When the CLI is not installed, the benchmark can download the official
release into a project-local directory (``.tools/opencode/``) instead of
requiring a manual install; see :func:`resolve_opencode_binary`.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

OPENCODE_BINARY = "opencode"

# ─── Local auto-install constants ───────────────────────────────────────────
# When ``--runner opencode``/``both`` is selected and the CLI is not usable
# from PATH, the benchmark downloads the official release binary into a
# private directory inside the project root. The layout mirrors the official
# installer (``~/.opencode/bin/opencode``) but scoped to the project so the
# tool stays self-contained and never touches user shell config.
OPENCODE_REPO = "anomalyco/opencode"
OPENCODE_GITHUB_URL = f"https://github.com/{OPENCODE_REPO}"
OPENCODE_LATEST_API_URL = (
    f"https://api.github.com/repos/{OPENCODE_REPO}/releases/latest"
)
# Relative to the project root (the directory containing this module).
OPENCODE_INSTALL_SUBDIR = os.path.join(".tools", "opencode")
# Marker file written next to the installed binary so operators can see which
# release was downloaded (best-effort; ``"latest"`` when the GitHub API
# version lookup fails).
OPENCODE_VERSION_MARKER = "version.txt"

# ─── Loop guards / fast-fail ────────────────────────────────────────────────
# OpenCode's agent loop has no internal liveness detection: once it emits a
# step it waits for the provider's next response indefinitely, and a stalled
# or looping task would otherwise burn the full benchmark timeout with zero
# diagnostics. These guards terminate the subprocess early and surface an
# actionable error instead. All three are data-backed from the
# ``2026-08-02-more-tests-more-models-opencode`` run (healthy streams emit
# ``step_start`` within seconds, never exceed 19 steps, and never repeat an
# identical text event); each can be disabled per call by passing 0/None.

# Kill the subprocess when NO bytes have arrived on stdout or stderr for this
# many seconds. Catches silent hangs (provider never returns even a
# ``step_start``) and mid-stream stalls (``step_start`` then silence, or a
# tool round-trip whose follow-up request never returns).
OPENCODE_NO_OUTPUT_GRACE = 300.0


def resolve_opencode_timeout(source_config: Mapping[str, Any], source: str,
                             default: float = OPENCODE_NO_OUTPUT_GRACE) -> float:
    """Return the per-source OpenCode inactivity timeout in seconds.

    ``opencode_timeout`` controls the staleness guard that terminates an
    OpenCode task when neither stdout nor stderr has produced bytes. A value
    of zero disables that guard, matching ``run_process``'s direct-call
    behavior. Invalid or negative values fall back to the default.
    """
    cfg = source_config.get(source) or {}
    value = cfg.get("opencode_timeout", default) if isinstance(cfg, Mapping) else default
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default

# Kill after this many completed agent steps (``step_finish`` events). Catches
# reasoning/tool planning loops (e.g. 683 ``todowrite`` calls with zero final
# text) long before the outer timeout.
OPENCODE_MAX_STEPS = 50
# Kill when the same non-trivial text event appears this many times. Catches
# text-repetition loops (canned continuation strings cycled forever).
OPENCODE_REPEAT_THRESHOLD = 5
# Minimum length (chars) of a text event considered by the repetition guard,
# so trivial acknowledgements cannot false-positive.
OPENCODE_REPEAT_MIN_LEN = 20

# ``opencode run --format`` accepts exactly ``default`` (formatted) or
# ``json`` (NDJSON event stream).  The adapter uses ``json`` so the final
# assistant answer can be extracted deterministically without ANSI/UI noise;
# ``plain`` does not exist in any released CLI and caused every invocation to
# be rejected with exit status 1. ``--pure`` disables external plugins so
# benchmark runs are reproducible and cannot inherit host-local extensions.
OPENCODE_RUN_FORMAT = "json"
OPENCODE_PURE_FLAG = "--pure"
# OpenCode only emits ``reasoning`` NDJSON events (the model's chain-of-
# thought) when this flag is set; non-interactive ``opencode run`` defaults
# ``thinking`` to False, so without it the event stream contains text/tool/
# step events only and thinking-capable models' reasoning is silently lost.
OPENCODE_THINKING_FLAG = "--thinking"

# ─── Neutral agent for plain model targets ──────────────────────────────────
# OpenCode injects its built-in default (Build) agent system prompt — which
# includes "answer concisely with fewer than 4 lines" and exposes every tool
# (webfetch, task, todowrite, ...) — whenever a run does not select a custom
# agent. That prompt is toxic for small function-calling-tuned models (the
# vibethinker family): they fixate on emitting tool-call dicts instead of
# producing the benchmark deliverable, and the "concisely" instruction
# contradicts benchmark prompts that demand full structured output.
#
# Plain model targets (``is_agent=False``) therefore register a neutral agent
# whose system prompt contains no conciseness instruction and whose
# permission map denies every tool key, so the model sees no tool definitions
# at all and simply answers the plugin prompt — the same contract the HTTP
# runner provides. Agent personas keep their own custom system prompts.
OPENCODE_NEUTRAL_AGENT_PROMPT = (
    "You are an AI assistant completing a written benchmark task. Read the "
    "user's request carefully and produce the complete, detailed deliverable "
    "it asks for. Do not truncate or summarize your response."
)
# Every permission key gates a tool family OpenCode could expose (read, edit,
# glob, grep, list, bash, task, external_directory, todowrite, question,
# webfetch, websearch, lsp, doom_loop, skill). Denying all of them removes
# every tool definition from the model's prompt, which is exactly what
# prevents tool-fixation loops on small function-calling-tuned models.
OPENCODE_NEUTRAL_AGENT_PERMISSION: dict[str, str] = {
    "read": "deny",
    "edit": "deny",
    "glob": "deny",
    "grep": "deny",
    "list": "deny",
    "bash": "deny",
    "task": "deny",
    "external_directory": "deny",
    "todowrite": "deny",
    "question": "deny",
    "webfetch": "deny",
    "websearch": "deny",
    "lsp": "deny",
    "doom_loop": "deny",
    "skill": "deny",
}


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
    missing = [
        flag for flag in ("--model", "--format", "--agent", OPENCODE_PURE_FLAG, OPENCODE_THINKING_FLAG)
        if flag not in help_text
    ]
    if missing:
        raise RuntimeError(
            "Installed OpenCode CLI is missing required run options: " + ", ".join(missing)
        )
    if not re.search(r"choices:[^]]*json", help_text, re.IGNORECASE | re.DOTALL):
        raise RuntimeError(
            "Installed OpenCode CLI does not advertise the 'json' run format "
            f"required for the {OPENCODE_RUN_FORMAT!r} output contract"
        )


# ─── Local auto-install ──────────────────────────────────────────────────────


def _local_install_dir() -> Path:
    """Return the project-scoped directory for a locally installed OpenCode.

    The project root is two levels above this module (``benchmark/opencode.py``
    inside the repo root), so the install location is stable regardless of the
    current working directory.
    """
    return Path(__file__).resolve().parent.parent / OPENCODE_INSTALL_SUBDIR


def _local_binary_path(install_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the expected binary path inside a local install directory."""
    install_dir = Path(install_dir) if install_dir else _local_install_dir()
    name = "opencode.exe" if os.name == "nt" else "opencode"
    return install_dir / name


def _cpu_has_avx2() -> bool:
    """Best-effort AVX2 flag check, mirroring the official installer."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("flags"):
                    return "avx2" in line.lower().split()
    except OSError:
        pass
    return True


def _is_musl_libc() -> bool:
    """Detect musl libc (Alpine or an ldd that reports musl)."""
    if os.path.exists("/etc/alpine-release"):
        return True
    try:
        probe = subprocess.run(
            ["ldd", "--version"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        return "musl" in (probe.stdout + probe.stderr).lower()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _darwin_translated() -> bool:
    """True when an x64 process runs under Rosetta 2 on Apple silicon."""
    try:
        probe = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return probe.returncode == 0 and probe.stdout.strip() == "1"
    except (OSError, subprocess.TimeoutExpired):
        return False


def _darwin_avx2() -> bool:
    """Best-effort macOS AVX2 check (``hw.optional.avx2_0`` sysctl)."""
    try:
        probe = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx2_0"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return probe.returncode == 0 and probe.stdout.strip() == "1"
    except (OSError, subprocess.TimeoutExpired):
        return True


def _platform_asset_name() -> str:
    """Build the official release asset filename for this platform.

    Mirrors the official installer's detection exactly: ``opencode-<os>-<arch>
    [-baseline][-musl].tar.gz`` on Linux (``.zip`` on macOS/Windows), with
    ``-baseline`` selected when the CPU lacks AVX2 and ``-musl`` for
    Alpine-style libc builds. macOS x64 processes running under Rosetta 2 get
    the arm64 build.
    """
    raw_os = platform.system().lower()
    if raw_os.startswith("darwin"):
        os_name = "darwin"
    elif raw_os.startswith("linux"):
        os_name = "linux"
    elif raw_os in ("mingw", "msys", "cygwin") or raw_os.startswith("windows"):
        os_name = "windows"
    else:
        raise RuntimeError(
            f"Unsupported OS for OpenCode auto-install: {platform.system()!r}"
        )

    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64", "x64"):
        arch = "x64"
    else:
        raise RuntimeError(
            f"Unsupported architecture for OpenCode auto-install: {machine!r}"
        )

    target = f"{os_name}-{arch}"
    if arch == "x64":
        if os_name == "linux":
            if not _cpu_has_avx2():
                target += "-baseline"
        elif os_name == "darwin":
            if _darwin_translated():
                target = "darwin-arm64"
            elif not _darwin_avx2():
                target += "-baseline"
    if os_name == "linux" and _is_musl_libc():
        target += "-musl"
    extension = ".tar.gz" if os_name == "linux" else ".zip"
    return f"opencode-{target}{extension}"


def _download_to(url: str, dest: Path, *, timeout: float) -> None:
    """Download ``url`` to ``dest``, following redirects (GitHub release
    download URLs redirect to the versioned asset on a CDN)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "ai-benchmark/opencode-installer"}
    )
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        open(dest, "wb") as handle,
    ):
        shutil.copyfileobj(response, handle)


def _latest_opencode_version(*, timeout: float = 15) -> str | None:
    """Return the latest release tag without the leading ``v`` (best effort)."""
    try:
        with urllib.request.urlopen(OPENCODE_LATEST_API_URL, timeout=timeout) as response:
            payload = json.load(response)
        tag = payload.get("tag_name", "")
    except Exception:  # noqa: BLE001 - best-effort version probe; None means unknown
        return None
    if not isinstance(tag, str) or not tag:
        return None
    return tag.removeprefix("v")


def _extract_binary(archive: Path, dest_dir: Path) -> Path:
    """Extract ``archive`` and return the path of the bundled binary."""
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            try:
                # Python >= 3.12: sanitize member metadata on extraction
                # (avoids the 3.14 deprecation for unfiltered extraction).
                tar.extractall(dest_dir, filter="data")
            except TypeError:
                tar.extractall(dest_dir)
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    for candidate in dest_dir.rglob("opencode*"):
        if candidate.is_file() and candidate.name in ("opencode", "opencode.exe"):
            return candidate
    raise RuntimeError(
        f"OpenCode release archive {archive.name} did not contain the opencode binary"
    )


def install_opencode(
    install_dir: str | os.PathLike[str] | None = None,
    *,
    timeout: float = 120,
) -> str:
    """Download the latest OpenCode release into a project-local directory.

    Defaults to ``<project root>/.tools/opencode/``. Returns the absolute
    path of the installed binary. Raises RuntimeError with an actionable
    message when the platform is unsupported, the download fails, or
    extraction fails.
    """
    install_dir = Path(install_dir) if install_dir else _local_install_dir()
    asset = _platform_asset_name()
    url = f"{OPENCODE_GITHUB_URL}/releases/latest/download/{asset}"
    version = _latest_opencode_version()
    try:
        with tempfile.TemporaryDirectory(prefix="opencode-install-") as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / asset
            _download_to(url, archive, timeout=timeout)
            binary = _extract_binary(archive, tmp_path)
            install_dir.mkdir(parents=True, exist_ok=True)
            dest = install_dir / binary.name
            shutil.move(str(binary), str(dest))
            dest.chmod(0o755)
            (install_dir / OPENCODE_VERSION_MARKER).write_text(
                version or "latest", encoding="utf-8"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-install OpenCode from {url}: {type(exc).__name__}: {exc}"
        ) from exc
    return str(dest)


def opencode_version(binary: str, timeout: float = 5) -> str | None:
    """Return the first line of ``<binary> --version``, or None (best effort)."""
    try:
        probe = subprocess.run(
            [binary, "--version"], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (probe.stdout or probe.stderr).strip().splitlines()
    return first[0].strip() if first else None


def resolve_opencode_binary(
    binary: str = OPENCODE_BINARY,
    install_dir: str | os.PathLike[str] | None = None,
    *,
    allow_install: bool = True,
    validate_timeout: float = 10,
    install_timeout: float = 120,
) -> str:
    """Resolve a usable OpenCode CLI binary for a run.

    Priority: (1) an on-PATH install that passes the capability preflight;
    (2) a previously auto-installed local copy; (3) a fresh local install
    (when ``allow_install``). An on-PATH install that exists but fails the
    preflight (e.g. an old release missing ``--thinking``/``--pure``) is
    replaced by the local copy or a fresh install automatically, so a stale
    global binary never blocks a run.

    Raises RuntimeError with an actionable message when no usable binary can
    be found (always the case when ``allow_install`` is False and nothing
    usable exists).
    """
    on_path = shutil.which(binary)
    local = _local_binary_path(install_dir)
    preflight_error: RuntimeError | None = None

    # 1. Prefer a valid on-PATH install.
    if on_path:
        try:
            validate_cli(on_path, timeout=validate_timeout)
            return on_path
        except RuntimeError as exc:
            preflight_error = exc

    # 2. Reuse a previously installed local copy if it still validates.
    if local.is_file():
        try:
            validate_cli(str(local), timeout=validate_timeout)
            return str(local)
        except RuntimeError:
            pass  # stale/corrupt -> reinstall below

    # 3. Install a fresh copy when permitted.
    if allow_install:
        installed = install_opencode(install_dir, timeout=install_timeout)
        try:
            validate_cli(installed, timeout=validate_timeout)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Auto-installed OpenCode at {installed} failed preflight: {exc}"
            ) from exc
        return installed

    if preflight_error is not None:
        raise RuntimeError(
            f"OpenCode on PATH is incompatible: {preflight_error}. Reinstall a "
            "newer OpenCode or re-run without --no-install-opencode so the "
            "benchmark can download a compatible copy into "
            f"{_local_install_dir()}/."
        ) from preflight_error
    if local.is_file():
        raise RuntimeError(
            f"The locally installed OpenCode at {local} does not pass the "
            "preflight. Delete it or re-run without --no-install-opencode to "
            "reinstall the latest release."
        )
    raise RuntimeError(
        "'opencode' was not found on PATH. Install OpenCode or re-run without "
        "--no-install-opencode so the benchmark can download it into "
        f"{_local_install_dir()}/."
    )


@dataclass(frozen=True)
class OpenCodeExtract:
    """Parsed content from one OpenCode NDJSON event stream.

    ``think_text`` holds the concatenated reasoning parts (``{"type":
    "reasoning", "part": {"type": "reasoning", "text": ...}}`` events),
    mirroring the ``think_text`` the HTTP path accumulates from
    ``reasoning_content``. It is empty when the model emitted no thinking or
    when the CLI was invoked without ``--thinking``.
    """

    text: str
    think_text: str
    error: str | None


@dataclass(frozen=True)
class OpenCodeProcessResult:
    """Captured result from one non-interactive OpenCode invocation."""

    text: str
    stderr: str
    elapsed: float
    error: str | None
    returncode: int | None
    think_text: str = ""


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
        # TRY004 would suggest TypeError, but config-validation errors are
        # ValueError throughout this codebase (and tests pin it).
        raise ValueError(  # noqa: TRY004 - config errors are ValueError throughout
            "source headers must be an object"
        )
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
    # timeout is injected by generate_config() when available. The separate
    # per-source ``opencode_timeout`` setting controls the subprocess
    # inactivity guard, not this provider request timeout.
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
    timeout: float | None = None,
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
    mappings: dict[str, list[str]] = {}

    for target_key, info in targets.items():
        source = info.get("source")
        api_model = info.get("api_model")
        if not isinstance(source, str):
            raise TypeError(f"Target {target_key!r} has no valid source")
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
        # Per-target ``token_levels`` (resolved by ``resolve_targets``) beat
        # the global list so thinking-heavy models get a bigger output budget
        # exactly where the operator asked for it.
        per_target_levels = info.get("token_levels")
        effective_levels = per_target_levels or token_levels
        model_options["limit"] = {
            "context": _model_context_limit(api_model),
            "output": max(effective_levels) if effective_levels else 16384,
        }
        providers[provider_id]["models"][api_model] = model_options

        # Every target registers an agent so ``--agent`` always selects
        # explicit context and OpenCode's built-in default agent prompt never
        # applies. ``_agent_id`` collapses non-alphanumerics to ``-``, so two
        # targets that differ only in punctuation (e.g. ``foo:3b`` and
        # ``foo-3b``) would collide on the same agent id and silently
        # overwrite each other; reject that up front.
        aid = _agent_id(target_key)
        if aid in agents:
            raise ValueError(
                f"OpenCode agent id collision: {target_key!r} and an earlier "
                f"target both map to agent {aid!r}"
            )
        agent_ids[target_key] = aid
        if info.get("is_agent") and info.get("system_prompt"):
            # Persona targets keep their explicit system prompt.
            agents[aid] = {
                "description": f"AI Benchmark agent for {target_key}",
                "mode": "primary",
                "model": mapped_model,
                "prompt": info["system_prompt"],
                "permission": {"edit": "deny", "bash": "deny"},
            }
        else:
            # Plain model target: register the neutral agent so OpenCode does
            # NOT fall back to its default Build agent prompt ("answer
            # concisely <4 lines", all tools enabled). The neutral prompt has
            # no conciseness instruction and the deny-all permission removes
            # every tool definition, giving small function-calling-tuned
            # models the same plain "answer the prompt" contract as HTTP.
            agents[aid] = {
                "description": f"AI Benchmark neutral agent for {target_key}",
                "mode": "primary",
                "model": mapped_model,
                "prompt": OPENCODE_NEUTRAL_AGENT_PROMPT,
                "permission": OPENCODE_NEUTRAL_AGENT_PERMISSION,
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


def _extract_final_text(stdout: bytes) -> OpenCodeExtract:
    """Extract final text and reasoning from ``opencode run --format json``.

    ``--format json`` emits one NDJSON event per line.  Completed assistant
    text parts arrive as ``{"type": "text", "part": {"type": "text",
    "text": ..., "time": {"end": ...}}}``; completed reasoning parts
    arrive as ``{"type": "reasoning", "part": {"type": "reasoning",
    "text": ...}}`` (only when the CLI was invoked with ``--thinking``).
    ``tool_use``, ``step_start``, and ``step_finish`` events carry no
    answer/thinking text.  Concatenate text and reasoning events in order
    (the model may emit several parts each), and fall back to the raw
    decoded stdout when the stream is not NDJSON at all.

    Returns an :class:`OpenCodeExtract` whose ``error`` is the ``error``
    event payload string when the session itself reported one.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
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
        elif event_type == "reasoning":
            part = event.get("part") or {}
            if isinstance(part, dict) and part.get("type") == "reasoning":
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    thinking_parts.append(part_text)
        elif event_type == "error":
            payload = event.get("error")
            if isinstance(payload, dict):
                session_error = payload.get("message") or payload.get("name")
            elif isinstance(payload, str):
                session_error = payload
    if saw_json:
        return OpenCodeExtract("\n".join(text_parts).strip(),
                               "\n".join(thinking_parts).strip(),
                               session_error)
    return OpenCodeExtract(stdout.decode("utf-8", errors="replace").strip(), "", None)


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


class _StreamGuard:
    """Incrementally count NDJSON events to enforce the loop guards.

    Fed raw stdout chunks by the reader pump (single writer thread); the
    main ``run_process`` loop reads :attr:`steps_exceeded` / :attr:`repeated`
    after each poll tick. Keeps a partial-line buffer so event boundaries
    that fall across chunk edges are handled correctly.

    Deliberately lock-free: ``feed`` runs on exactly one writer thread while
    the main loop only *reads* :attr:`steps_exceeded` / :attr:`repeated`, so
    under the GIL the worst case is a stale read delayed by one 100 ms poll
    tick — irrelevant for a kill guard. Do not add a lock here; it would
    only add contention to the hot pump path.
    """

    __slots__ = (
        "_pending",
        "_text_counts",
        "repeat_min_len",
        "repeat_threshold",
        "repeated",
        "step_count",
        "step_limit",
    )

    def __init__(self, step_limit: int = 0, repeat_threshold: int = 0,
                 repeat_min_len: int = 20) -> None:
        self.step_limit = step_limit or 0
        self.repeat_threshold = repeat_threshold or 0
        self.repeat_min_len = repeat_min_len or 0
        self._pending = b""
        self._text_counts: dict[str, int] = {}
        self.step_count = 0
        self.repeated = False

    @property
    def steps_exceeded(self) -> bool:
        return bool(self.step_limit) and self.step_count >= self.step_limit

    def feed(self, chunk: bytes) -> None:
        # Once a guard has tripped, stop counting: the main loop will kill
        # the process within one poll tick and a looping stream can emit
        # thousands more events before that happens.
        if self.repeated or self.steps_exceeded:
            return
        self._pending += chunk
        lines = self._pending.split(b"\n")
        self._pending = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "step_finish":
                self.step_count += 1
            elif event_type == "text":
                part = event.get("part")
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str) and len(text.strip()) >= self.repeat_min_len:
                    self._text_counts[text] = self._text_counts.get(text, 0) + 1
                    if (self.repeat_threshold
                            and self._text_counts[text] >= self.repeat_threshold):
                        self.repeated = True


def _pump_stream(stream: Any, sink: list[bytes], guard: _StreamGuard | None) -> None:
    """Read ``stream`` to EOF, appending chunks to ``sink``.

    Runs on a daemon thread so the child can emit megabytes of NDJSON without
    deadlocking the main loop (``Popen.communicate`` would buffer everything
    invisibly; the reader threads are the only way to see data as it flows).
    ``guard``, when given, receives every chunk for incremental loop
    detection. Reading blocks until data or EOF; the main loop terminates the
    process on timeout/cancel, which closes the pipe and unblocks this thread.
    """
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            sink.append(chunk)
            if guard is not None:
                guard.feed(chunk)
    except (OSError, ValueError):
        # Pipe closed under us (process killed, fd closed) — nothing to do.
        pass


def run_process(
    prompt: str,
    *,
    config_path: str,
    model: str,
    timeout: float,
    binary: str = OPENCODE_BINARY,
    agent: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    target_key: str = "target",
    plugin_id: str = "plugin",
    stop_event: Any = None,
    no_output_grace: float = OPENCODE_NO_OUTPUT_GRACE,
    step_limit: int = OPENCODE_MAX_STEPS,
    repeat_threshold: int = OPENCODE_REPEAT_THRESHOLD,
    repeat_min_len: int = OPENCODE_REPEAT_MIN_LEN,
) -> OpenCodeProcessResult:
    """Run one isolated, plugin-free OpenCode task.

    ``--pure`` is deliberately part of every invocation rather than only a
    generated-config setting: it prevents user/project-installed OpenCode
    plugins from changing the benchmark's tools, prompts, or event stream.

    Output is drained by daemon reader threads (``communicate`` would hide
    it until exit). While the process runs, three loop guards abort it early
    instead of burning the full ``timeout``:

    * **Staleness fast-fail** (``no_output_grace``): no bytes on stdout or
      stderr for that many seconds — catches silent hangs and mid-stream /
      tool round-trip stalls where the provider never answers.
    * **Step budget** (``step_limit``): too many completed ``step_finish``
      events — catches reasoning/tool planning loops that never produce a
      final answer.
    * **Text repetition** (``repeat_threshold``/``repeat_min_len``): the same
      non-trivial text event seen repeatedly — catches canned-continuation
      loops.

    All guards are disabled by passing 0/None. The process group is
    terminated (and any partial stdout retained) on timeout, cancellation,
    or a guard trip.
    """
    command = [
        binary, "run", OPENCODE_PURE_FLAG,
        "--model", model, "--format", OPENCODE_RUN_FORMAT,
        OPENCODE_THINKING_FLAG,
    ]
    if agent:
        command.extend(["--agent", agent])
    command.append(prompt)
    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = os.path.abspath(config_path)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return OpenCodeProcessResult("", "", time.monotonic() - started,
                                    f"Could not start OpenCode: {type(exc).__name__}", None)

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    guard = _StreamGuard(step_limit=step_limit,
                         repeat_threshold=repeat_threshold,
                         repeat_min_len=repeat_min_len)
    pumps: list[threading.Thread] = []
    for stream, sink, with_guard in ((process.stdout, stdout_chunks, True),
                                     (process.stderr, stderr_chunks, False)):
        if stream is None:
            continue
        thread = threading.Thread(
            target=_pump_stream,
            args=(stream, sink, guard if with_guard else None),
            daemon=True,
        )
        thread.start()
        pumps.append(thread)

    deadline = started + max(0, float(timeout))
    last_data = started
    prev_total = 0
    error: str | None = None
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                _terminate_process(process)
                error = "Cancelled"
                break
            if process.poll() is not None:
                # Exited on its own — honor that over the deadline and guards.
                break
            if time.monotonic() >= deadline:
                _terminate_process(process)
                error = f"OpenCode timed out after {timeout}s"
                break
            total = sum(len(c) for c in stdout_chunks) + sum(len(c) for c in stderr_chunks)
            if total > prev_total:
                last_data = time.monotonic()
                prev_total = total
            if no_output_grace and time.monotonic() - last_data > no_output_grace:
                _terminate_process(process)
                error = (f"OpenCode produced no output within "
                         f"{no_output_grace:g}s (possible provider/agent stall)")
                break
            if guard.steps_exceeded:
                _terminate_process(process)
                error = (f"OpenCode reached {step_limit} agent steps without "
                         "finishing (possible loop)")
                break
            if guard.repeated:
                _terminate_process(process)
                error = (f"OpenCode repeated the same text {repeat_threshold} "
                         "times (possible loop)")
                break
            if stop_event is not None:
                stop_event.wait(0.1)
            else:
                time.sleep(0.1)
    finally:
        if process.poll() is None:
            _terminate_process(process)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        for thread in pumps:
            thread.join(timeout=2)

    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    returncode = process.returncode

    extract = _extract_final_text(stdout)
    text = extract.text
    think_text = extract.think_text
    session_error = extract.error
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
    return OpenCodeProcessResult(text, diagnostic, elapsed, error, returncode,
                                 think_text=think_text)
