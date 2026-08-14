"""Tests for isolated executable plugin checks."""
import subprocess
from unittest import mock

from plugins import discover_plugins
from plugins.challenges._execution import ExecutionResult, extract_python_source, run_python_check


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
