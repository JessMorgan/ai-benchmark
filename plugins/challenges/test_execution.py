"""Tests for phase-4 isolated execution checks."""
import subprocess
import sys
from unittest import mock

from plugins import discover_plugins
from plugins.challenges._execution import ExecutionResult, extract_python_source, run_python_check


def plugin(plugin_id):
    return next(item for item in discover_plugins() if item.id == plugin_id)


def test_missing_podman_is_explicitly_skipped():
    with mock.patch("plugins.challenges._execution.shutil.which", return_value=None):
        result = run_python_check("print('ignored')", "assert False")
    assert result.status == "skipped"
    assert not result.passed
    assert "Podman" in result.error


def test_runtime_command_is_network_disabled_and_never_pulls():
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


def test_runtime_failure_is_failed_not_skipped():
    process = mock.Mock(returncode=125)
    process.communicate.return_value = ("", "AssertionError")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process):
        result = run_python_check("x = 1", "assert x == 2")
    assert result.status == "failed"
    assert not result.passed


def test_missing_image_is_explicitly_skipped():
    process = mock.Mock(returncode=125)
    process.communicate.return_value = ("", "no such image")
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process):
        result = run_python_check("x = 1", "assert x == 1")
    assert result.status == "skipped"


def test_timeout_is_recorded_separately():
    process = mock.Mock(pid=1234)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired("podman", 5),
        ("", ""),
    ]
    with mock.patch("plugins.challenges._execution.shutil.which", return_value="podman"), \
            mock.patch("plugins.challenges._execution.subprocess.Popen", return_value=process), \
            mock.patch("plugins.challenges._execution.os.killpg") as killpg:
        result = run_python_check("while True: pass", "")
    killpg.assert_called_once_with(1234, __import__("signal").SIGKILL)
    assert result.status == "timeout"
    assert not result.passed


def test_multiple_python_fences_are_combined():
    source = extract_python_source("```python\nx = 1\n```\n```python\ny = x + 1\n```")
    assert source is not None
    assert "x = 1" in source and "y = x + 1" in source


def test_unavailable_execution_does_not_reduce_rate_limiter_score():
    response = """
```python
class TokenBucket:
    def allow_request(self, client_id: str) -> bool:
        return True
    def get_usage_stats(self, client_id: str) -> dict:
        return {}
```
"""
    target = plugin("rate-limiter")
    module = sys.modules[target.__class__.__module__]
    with mock.patch.object(module, "run_python_check", return_value=ExecutionResult("skipped", error="test")) as check:
        result = target.evaluate(response)
    check.assert_called_once()
    evidence = result.diagnostics["validations"][-1]
    assert evidence["status"] == "skipped"
    assert not evidence["passed"]
    assert not any("execution" in finding["finding"] for criterion in result.rubric for finding in criterion["negative_findings"])


def test_multi_step_and_error_recovery_record_execution_evidence():
    response = "```python\ndef greet_user(name: str) -> str:\n    return f'Hello, {name}! Welcome.'\n```\n```python\ndef validate_name(name: str) -> bool:\n    return bool(name)\n```\n```python\ndef format_greeting(greeting: str, times: int) -> str:\n    return '\\n'.join([greeting] * times)\n```\n[SUMMARY: 3 lines, 3 functions, completed all steps]."
    target = plugin("multi-step")
    module = sys.modules[target.__class__.__module__]
    with mock.patch.object(module, "run_python_check", return_value=ExecutionResult("skipped", error="test")) as check:
        multi_step = target.evaluate(response)
    assert check.call_count == 3
    assert all(item["status"] == "skipped" for item in multi_step.diagnostics["validations"][-3:])

    recovery = "```python\nclass AllProvidersFailedError(Exception):\n    pass\nasync def get_weather_resilient(city: str) -> dict:\n    raise AllProvidersFailedError(city)\n```"
    target = plugin("error-recovery")
    module = sys.modules[target.__class__.__module__]
    with mock.patch.object(module, "run_python_check", return_value=ExecutionResult("skipped", error="test")) as check:
        error_recovery = target.evaluate(recovery)
    check.assert_called_once()
    assert error_recovery.diagnostics["validations"][-1]["status"] == "skipped"
