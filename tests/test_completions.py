"""Tests for shell completion generation."""
import shutil
import subprocess
import sys
import unittest

from plugins import discover_plugins
from shell_completion import generate_shell_completion


class TestShellCompletions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugins = discover_plugins()
        cls.plugin_ids = {p.id for p in cls.plugins}

    def test_cli_generate_shell_completion_bash(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "bash"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("_ai_benchmark_complete", output)
        self.assertIn("--plugins-whitelist", output)
        self.assertIn("--plugins-blacklist", output)
        self.assertIn("--generate-shell-completion", output)
        for pid in self.plugin_ids:
            self.assertIn(pid, output)

    def test_cli_generate_shell_completion_zsh(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "zsh"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("#compdef ai-benchmark.py", output)
        self.assertIn("--plugins-whitelist", output)
        self.assertIn("--plugins-blacklist", output)
        self.assertIn("--generate-shell-completion", output)
        for pid in self.plugin_ids:
            self.assertIn(pid, output)

    def test_cli_generate_shell_completion_fish(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "fish"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("complete -c ai-benchmark.py", output)
        self.assertIn("-l plugins-whitelist", output)
        self.assertIn("-l plugins-blacklist", output)
        self.assertIn("-l generate-shell-completion", output)
        for pid in self.plugin_ids:
            self.assertIn(pid, output)

    def test_cli_help_mentions_shell_completion(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--generate-shell-completion", result.stdout)
        self.assertIn("eval \"$(python ai-benchmark.py --generate-shell-completion bash)\"", result.stdout)

    def test_bash_completion_parses(self):
        if not shutil.which("bash"):
            self.skipTest("bash not installed")
        script = generate_shell_completion("bash", self.plugins)
        result = subprocess.run(
            ["bash", "-n", "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zsh_completion_parses(self):
        if not shutil.which("zsh"):
            self.skipTest("zsh not installed")
        script = generate_shell_completion("zsh", self.plugins)
        result = subprocess.run(
            ["zsh", "-n", "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fish_completion_parses(self):
        if not shutil.which("fish"):
            self.skipTest("fish not installed")
        # Verify the installed fish supports --parse-only
        probe = subprocess.run(
            ["fish", "--parse-only", "-c", "echo ok"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            self.skipTest("fish version does not support --parse-only")
        script = generate_shell_completion("fish", self.plugins)
        result = subprocess.run(
            ["fish", "--parse-only", "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
