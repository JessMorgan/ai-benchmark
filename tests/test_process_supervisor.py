"""Tests for shared external-process supervision."""
import subprocess
from unittest import mock

from benchmark.process_supervisor import close_process_pipes, terminate_process_tree


def test_terminate_process_tree_terminates_children_before_parent():
    process = mock.MagicMock()
    process.poll.return_value = None
    process.pid = 42
    child = mock.MagicMock()
    root = mock.MagicMock()
    root.children.return_value = [child]
    with mock.patch("benchmark.process_supervisor.psutil.Process", return_value=root):
        terminate_process_tree(process)
    child.terminate.assert_called_once()
    root.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_terminate_process_tree_escalates_after_timeout():
    process = mock.MagicMock()
    process.poll.return_value = None
    process.pid = 42
    process.wait.side_effect = subprocess.TimeoutExpired("run", 1)
    root = mock.MagicMock()
    root.children.return_value = []
    with mock.patch("benchmark.process_supervisor.psutil.Process", return_value=root):
        terminate_process_tree(process)
    root.terminate.assert_called_once()
    root.kill.assert_called_once()


def test_close_process_pipes_is_best_effort():
    process = mock.MagicMock()
    close_process_pipes(process)
    process.stdin.close.assert_called_once()
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()
