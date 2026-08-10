"""Podman-backed execution checks for generated Python snippets."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
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


def run_python_check(source: str, harness: str, *, timeout: float = 5.0) -> ExecutionResult:
    """Run source plus harness in an isolated, network-disabled Podman container.

    The image must already be present locally: ``--pull=never`` prevents an
    unexpected image download from violating the no-network execution
    contract. Missing Podman/image/runtime access is ``skipped``; a started
    container that fails its harness is ``failed``. A timeout remains distinct
    so reports can identify unbounded user code.
    """
    podman = _podman_binary()
    if not podman:
        return ExecutionResult("skipped", error="Podman is not installed")
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
        except OSError as exc:
            return ExecutionResult("skipped", error=f"Podman unavailable: {exc}")
    output = (stdout + stderr).strip()
    if process is None:  # pragma: no cover - defensive; Popen either returns or raises
        return ExecutionResult("skipped", error="Podman process was not started")
    if process.returncode == 125 and _runtime_unavailable(output):
        return ExecutionResult("skipped", output=output, error="container runtime unavailable")
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
