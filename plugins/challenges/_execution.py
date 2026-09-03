"""Podman-backed execution checks for generated Python snippets."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None  # type: ignore[assignment]
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plugins.challenges._validators import extract_fenced_blocks


@dataclass
class ExecutionResult:
    """Bounded execution outcome suitable for rubric diagnostics."""

    status: str
    passed: bool = False
    output: str = ""
    error: str | None = None
    isolation: str = "podman"
    skipped_reason: str | None = None

    def as_evidence(self) -> dict[str, Any]:
        return {
            "kind": "execution",
            "status": self.status,
            "passed": self.passed,
            "output": self.output[-4000:],
            "error": self.error,
            "isolation": self.isolation,
            "skipped_reason": self.skipped_reason,
        }


def _podman_binary() -> str | None:
    return shutil.which("podman")


def _runtime_unavailable(output: str) -> bool:
    """Return whether Podman could not start the requested local container."""
    text = output.lower()
    return any(marker in text for marker in (
        "no such image",
        "image not known",
        "unable to find image",
        "short-name resolution enforced",
        "cannot connect to the podman socket",
        "permission denied",
        "executable file not found",
    ))


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out Podman process and any container child it owns."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()


def _run_local_restricted(source: str, harness: str, *, timeout: float) -> ExecutionResult:
    """Run a check without Podman using resource limits and a clean process.

    This fallback is intentionally reported as ``local-restricted`` rather
    than pretending to provide container isolation. It is useful on developer
    machines and CI hosts without Podman, but it is not a security boundary:
    callers should prefer Podman for untrusted benchmark responses.
    """
    with tempfile.TemporaryDirectory(prefix="ai-benchmark-local-exec-") as tmpdir:
        script = Path(tmpdir) / "check.py"
        execution = source + "\n\n" + harness
        if resource is not None:
            # Keep thread stacks small enough for the address-space limit while
            # still allowing the concurrent challenge harness to run. Compile
            # the generated source separately so a response's future imports
            # retain their normal module semantics.
            execution = (
                "import threading as _threading\n"
                "try:\n"
                "    _threading.stack_size(1024 * 1024)\n"
                "except RuntimeError:\n"
                "    pass\n"
                f"exec(compile({execution!r}, {str(script)!r}, 'exec'))\n"
            )
        script.write_text(execution, encoding="utf-8")

        def limit_resources() -> None:
            if resource is None:
                return
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
            # RLIMIT_NPROC is deliberately NOT set here. On Linux the kernel
            # enforces it against every task (process + thread) already owned
            # by the user, not against this child alone, so a small cap makes
            # correct threaded submissions crash whenever the host user is
            # busy. CPU, address-space, and file-size limits still constrain
            # the single fallback process; Podman remains the security boundary.

        try:
            process = subprocess.Popen(
                [sys.executable, "-I", str(script)],
                cwd=tmpdir,
                env={"PATH": os.environ.get("PATH", "")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                # The child is single-purpose and the fallback is explicitly
                # weaker than Podman; resource limits are still preferable to
                # an unrestricted subprocess.
                preexec_fn=limit_resources if resource is not None else None,  # noqa: PLW1509
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                stdout, stderr = process.communicate()
                return ExecutionResult(
                    "timeout", output=(stdout + stderr).strip(),
                    error=f"execution exceeded {timeout:g}s",
                    isolation="local-restricted",
                )
        except OSError as exc:
            return ExecutionResult("failed", error=f"local execution failed: {exc}", isolation="local-restricted")

    output = (stdout + stderr).strip()
    return ExecutionResult(
        "passed" if process.returncode == 0 else "failed",
        passed=process.returncode == 0,
        output=output,
        error=None if process.returncode == 0 else f"process exited {process.returncode}",
        isolation="local-restricted",
    )


def run_python_check(source: str, harness: str, *, timeout: float = 5.0) -> ExecutionResult:
    """Run source plus a pytest-compatible assertion harness.

    Podman is preferred and provides the network-disabled boundary. When it is
    unavailable, a resource-limited local process is used and the result
    records that weaker isolation mode explicitly.
    """
    podman = _podman_binary()
    if not podman:
        return _run_local_restricted(source, harness, timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="ai-benchmark-exec-") as tmpdir:
        # The container runs as an unprivileged numeric user. Make only the
        # temporary bind source traversable/readable; it contains no host data
        # other than the generated check script.
        os.chmod(tmpdir, 0o755)
        script = Path(tmpdir) / "check.py"
        script.write_text(source + "\n\n" + harness, encoding="utf-8")
        script.chmod(0o644)
        command = [
            podman, "run", "--rm", "--pull=never",
            "--network=none",
            "--read-only",
            "--user=65532:65532",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=128m",
            "--cpus=1",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={script},dst=/check.py,ro",
            "docker.io/library/python:3.12-alpine",
            "python", "/check.py",
        ]
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                stdout, stderr = process.communicate()
                output = (stdout + stderr).strip()
                return ExecutionResult(
                    "timeout",
                    output=output,
                    error=f"execution exceeded {timeout:g}s",
                )
        except OSError:
            return _run_local_restricted(source, harness, timeout=timeout)
    output = (stdout + stderr).strip()
    if process is None:  # pragma: no cover - defensive; Popen either returns or raises
        return _run_local_restricted(source, harness, timeout=timeout)
    if process.returncode == 125 and _runtime_unavailable(output):
        return _run_local_restricted(source, harness, timeout=timeout)
    if "PHASE4_HARNESS_SKIPPED:" in output:
        reason = output.split("PHASE4_HARNESS_SKIPPED:", 1)[1].splitlines()[0].strip()
        return ExecutionResult("skipped", output=output, error=reason, skipped_reason=reason)
    return ExecutionResult(
        "passed" if process.returncode == 0 else "failed",
        passed=process.returncode == 0,
        output=output,
        error=None if process.returncode == 0 else f"container exited {process.returncode}",
    )


def extract_python_source(text: str) -> str | None:
    """Return all fenced Python blocks, or raw text when it is non-empty.

    Multi-step responses intentionally put each function in a separate fenced
    block, so execution must combine the blocks into one module. The caller's
    AST validator remains responsible for deciding whether raw text is valid
    Python; this helper only selects source and never executes it itself.
    """
    blocks = extract_fenced_blocks(text, "python")
    if blocks:
        return "\n\n".join(blocks)
    return text if text.strip() else None


def execution_json(result: ExecutionResult) -> str:
    """Serialize an execution result for debug output/tests."""
    return json.dumps(result.as_evidence(), sort_keys=True)
