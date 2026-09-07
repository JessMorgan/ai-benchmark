"""Tests for isolated executable plugin checks."""
import subprocess
from unittest import mock

from plugins import discover_plugins
from plugins.challenges._execution import (
    HARNESS_SENTINEL,
    ExecutionResult,
    _sentinel_script,
    extract_python_source,
    run_python_check,
)


def plugin(plugin_id):
    return next(item for item in discover_plugins() if item.id == plugin_id)


def test_missing_podman_uses_restricted_local_fallback():
    process = mock.Mock(returncode=0)
    process.communicate.return_value = ("", "")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value=None), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process):
        result = run_python_check("print('ok')", "assert True")
    assert result.status == "passed"
    assert result.isolation == "local-restricted"


def test_local_restricted_fallback_allows_thread_pools():
    """The local-restricted sandbox must run the challenge-sized thread pool.

    RLIMIT_NPROC is per-user on Linux, so capping it counts every task the
    host user already owns and crashes correct thread-based code. Newer
    Python runtimes also need more virtual address space for thread stacks;
    this uses the 16-thread shape of the rate-limiter execution harness.
    """
    with mock.patch("plugins.challenges._execution.shutil.which", return_value=None):
        result = run_python_check(
            "from concurrent.futures import ThreadPoolExecutor",
            """
def _double(value):
    return value * 2

with ThreadPoolExecutor(max_workers=16) as _pool:
    assert sorted(_pool.map(_double, range(64))) == [v * 2 for v in range(64)]
""",
        )
    assert result.status == "passed", result.output
    assert result.isolation == "local-restricted"


def test_podman_command_is_network_disabled_and_never_pulls():
    process = mock.Mock(returncode=0)
    process.communicate.return_value = ("ok\n", "")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process) as popen:
        result = run_python_check("x = 1", "assert x == 1")
    command = popen.call_args.args[0]
    assert "--network=none" in command
    assert "--pull=never" in command
    assert "--user=65532:65532" in command
    assert popen.call_args.kwargs["start_new_session"] is True
    assert result.status == "passed"
    assert result.isolation == "podman"


def test_runtime_failure_is_failed_not_skipped():
    process = mock.Mock(returncode=1)
    process.communicate.return_value = ("", "AssertionError")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process):
        result = run_python_check("x = 1", "assert x == 2")
    assert result.status == "failed"
    assert not result.passed


def test_runtime_unavailable_falls_back_locally():
    process = mock.Mock(returncode=125)
    process.communicate.return_value = ("", "no such image")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process):
        result = run_python_check("x = 1", "assert x == 1")
    assert result.status == "failed"
    assert result.isolation == "local-restricted"


def test_timeout_is_recorded_separately():
    process = mock.Mock(pid=1234)
    process.communicate.side_effect = [subprocess.TimeoutExpired("podman", 5), ("", "")]
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process), \
            mock.patch("plugins.challenges._execution.os.killpg") as killpg:
        result = run_python_check("while True: pass", "")
    killpg.assert_called_once()
    assert result.status == "timeout"
    assert not result.passed


def test_multiple_python_fences_are_combined():
    source = extract_python_source("```python\nx = 1\n```\n```python\ny = x + 1\n```")
    assert source is not None
    assert "x = 1" in source and "y = x + 1" in source


def test_execution_evidence_is_recorded_for_code_plugins():
    response = """```python
class TokenBucket:
    def __init__(self, limit, window_seconds): pass
    def allow_request(self, client_id, now): return True
    def get_usage_stats(self, client_id): return {}
    def cleanup(self, now): return 0
class SlidingWindowLog(TokenBucket): pass
class FixedWindow(TokenBucket): pass
```"""
    target = plugin("rate-limiter")
    module = __import__(target.__class__.__module__, fromlist=["run_python_check"])
    with mock.patch.object(module, "run_python_check", return_value=ExecutionResult("skipped", error="test")):
        result = target.evaluate(response)
    behavior = next(item for item in result.rubric if item["name"] == "Behavioral strategy tests")
    assert behavior["negative_findings"]
    assert result.diagnostics["errors"] == []


def test_sentinel_script_combines_source_harness_and_sentinel():
    script = _sentinel_script("x = 1", "assert x == 1")
    assert "x = 1" in script
    assert "assert x == 1" in script
    assert script.endswith(f'print("{HARNESS_SENTINEL}")')


def test_harness_ok_requires_pass_and_sentinel():
    assert ExecutionResult("passed", passed=True, output=HARNESS_SENTINEL).harness_ok is True
    assert ExecutionResult("passed", passed=True, output="").harness_ok is False
    assert ExecutionResult("failed", passed=False, output=HARNESS_SENTINEL).harness_ok is False
    assert ExecutionResult("timeout", output=HARNESS_SENTINEL).harness_ok is False
    assert ExecutionResult("skipped", output=HARNESS_SENTINEL, skipped_reason="x").harness_ok is False


def test_exit_zero_without_harness_is_not_harness_ok():
    """A zero-implementation response that exits 0 before the harness runs must
    not masquerade as a clean pass on the local-restricted path.

    ``sys.exit(0)`` raises ``SystemExit`` which propagates out of the combined
    script before the sentinel line, so the process exits 0 (``passed``) but the
    sentinel is never printed (``harness_ok`` is False).
    """
    with mock.patch("plugins.challenges._execution.shutil.which", return_value=None):
        result = run_python_check("import sys\nsys.exit(0)", "assert True")
    assert result.passed is True
    assert result.harness_ok is False
    assert HARNESS_SENTINEL not in result.output


def test_clean_local_run_reports_harness_ok():
    with mock.patch("plugins.challenges._execution.shutil.which", return_value=None):
        result = run_python_check("x = 1", "assert x == 1")
    assert result.harness_ok is True
    assert HARNESS_SENTINEL in result.output
