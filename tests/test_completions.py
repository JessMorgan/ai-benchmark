"""Tests for shell completion generation."""
import shutil
import subprocess
import sys
import unittest

from benchmark.completions import COMMAND_NAMES, generate_shell_completion
from plugins import discover_plugins


class TestCompletionBranches(unittest.TestCase):
    """Direct-call tests for every shell branch.

    The subprocess tests below invoke the CLI end-to-end, but coverage only
    sees in-process calls, so the zsh/fish/unknown branches are exercised
    here by calling ``generate_shell_completion`` directly.
    """

    @classmethod
    def setUpClass(cls):
        cls.plugins = discover_plugins()

    def test_bash_registers_both_command_names(self):
        script = generate_shell_completion("bash", self.plugins)
        self.assertIn("complete -F _shtab_ai_benchmark ai-benchmark", script)
        self.assertIn("complete -F _shtab_ai_benchmark ai-benchmark.py", script)

    def test_zsh_has_compdef_for_both_names_and_plugin_choices(self):
        script = generate_shell_completion("zsh", self.plugins)
        self.assertIn("#compdef ai-benchmark ai-benchmark.py", script)
        self.assertIn("_shtab_ai_benchmark_options", script)
        for flag in (
            "--restart", "--config", "--runner", "--judge-spread-threshold",
            "--no-judge-spread", "--judge-deviation-threshold", "--no-judge-deviation",
        ):
            self.assertIn(flag, script)

    def test_fish_emits_lines_for_both_command_names(self):
        script = generate_shell_completion("fish", self.plugins)
        lines = script.strip().splitlines()
        self.assertTrue(lines)
        self.assertTrue(any(line.startswith("complete -c ai-benchmark ") for line in lines))
        self.assertTrue(any(line.startswith("complete -c ai-benchmark.py ") for line in lines))
        # Every flag appears once per registered command name.
        self.assertEqual(
            sum(1 for line in lines if "-l restart" in line),
            len(COMMAND_NAMES),
        )
        self.assertEqual(
            sum(1 for line in lines if "-l runner" in line),
            len(COMMAND_NAMES),
        )

    def test_unknown_shell_returns_empty_string(self):
        self.assertEqual(generate_shell_completion("tcsh", self.plugins), "")


class TestShellCompletions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugins = discover_plugins()
        cls.plugin_ids = {p.id for p in cls.plugins}

    def test_cli_generate_shell_completion_bash(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "bash"],
            capture_output=True,
            text=True, check=False,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("_shtab_ai_benchmark", output)
        self.assertIn("--plugins-whitelist", output)
        self.assertIn("--plugins-blacklist", output)
        self.assertIn("--generate-shell-completion", output)
        self.assertIn("--judge-spread-threshold", output)
        self.assertIn("--no-judge-deviation", output)
        for pid in self.plugin_ids:
            self.assertIn(pid, output)

    def test_cli_generate_shell_completion_zsh(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "zsh"],
            capture_output=True,
            text=True, check=False,
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout
        self.assertIn("#compdef ai-benchmark ai-benchmark.py", output)
        self.assertIn("--plugins-whitelist", output)
        self.assertIn("--plugins-blacklist", output)
        self.assertIn("--generate-shell-completion", output)
        self.assertIn("--judge-spread-threshold", output)
        self.assertIn("--no-judge-deviation", output)
        for pid in self.plugin_ids:
            self.assertIn(pid, output)

    def test_cli_generate_shell_completion_fish(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--generate-shell-completion", "fish"],
            capture_output=True,
            text=True, check=False,
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
            text=True, check=False,
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
            text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zsh_completion_parses(self):
        if not shutil.which("zsh"):
            self.skipTest("zsh not installed")
        script = generate_shell_completion("zsh", self.plugins)
        result = subprocess.run(
            ["zsh", "-n", "-c", script],
            capture_output=True,
            text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fish_completion_parses(self):
        if not shutil.which("fish"):
            self.skipTest("fish not installed")
        # Verify the installed fish supports --parse-only
        probe = subprocess.run(
            ["fish", "--parse-only", "-c", "echo ok"],
            capture_output=True,
            text=True, check=False,
        )
        if probe.returncode != 0:
            self.skipTest("fish version does not support --parse-only")
        script = generate_shell_completion("fish", self.plugins)
        result = subprocess.run(
            ["fish", "--parse-only", "-c", script],
            capture_output=True,
            text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
