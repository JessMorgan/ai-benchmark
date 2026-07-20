"""Tests for plugin discovery and selection."""
import os
import tempfile
import unittest

from plugins import discover_plugins, discover_output_plugins, format_plugin_list, _discover_plugins_in_dir
from benchmark_plugin import BenchmarkTaskPlugin


class TestPluginDiscovery(unittest.TestCase):
    def test_discovers_all_builtin_plugins(self):
        plugins = discover_plugins()
        ids = [p.id for p in plugins]
        self.assertIn("rate-limiter", ids)
        self.assertIn("moe-dense", ids)
        self.assertIn("tool-calling", ids)
        self.assertIn("orchestration", ids)
        self.assertIn("code-review", ids)
        self.assertIn("structured-output", ids)
        self.assertIn("multi-step", ids)
        self.assertIn("prd-creation", ids)
        self.assertIn("wireframes", ids)
        self.assertIn("software-architecture", ids)

    def test_plugins_have_required_metadata(self):
        plugins = discover_plugins()
        for p in plugins:
            self.assertTrue(p.id)
            self.assertTrue(p.version)
            self.assertTrue(p.name)
            self.assertGreater(p.max_score, 0)

    def test_whitelist_filters_plugins(self):
        plugins = discover_plugins(whitelist=["rate-limiter"])
        self.assertEqual([p.id for p in plugins], ["rate-limiter"])

    def test_blacklist_filters_plugins(self):
        plugins = discover_plugins(blacklist=["moe-dense"])
        self.assertEqual([p.id for p in plugins], ["code-review", "multi-step", "orchestration", "prd-creation", "rate-limiter", "software-architecture", "structured-output", "tool-calling", "wireframes"])

    def test_whitelist_and_blacklist_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            discover_plugins(whitelist=["rate-limiter"], blacklist=["moe-dense"])

    def test_empty_whitelist_returns_all(self):
        plugins = discover_plugins(whitelist=[])
        self.assertEqual(len(plugins), len(discover_plugins()))

    def test_format_plugin_list_empty(self):
        result = format_plugin_list([])
        self.assertEqual(result, "No plugins discovered.")

    def test_format_plugin_list_table(self):
        plugins = discover_plugins(whitelist=["rate-limiter"])
        result = format_plugin_list(plugins)
        self.assertIn("rate-limiter", result)
        self.assertIn("Rate Limiter", result)
        self.assertIn("Use these IDs with --plugins-whitelist or --plugins-blacklist.", result)

    def test_discover_plugins_in_missing_directory(self):
        plugins = _discover_plugins_in_dir("/nonexistent/path", "pkg", BenchmarkTaskPlugin)
        self.assertEqual(plugins, [])

    def test_discover_plugins_instantiation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "bad_plugin.py")
            with open(plugin_path, "w") as f:
                f.write(
                    "from benchmark_plugin import BenchmarkTaskPlugin\n"
                    "class BadPlugin(BenchmarkTaskPlugin):\n"
                    "    @property\n"
                    "    def id(self):\n"
                    "        return 'bad'\n"
                    "    @property\n"
                    "    def version(self):\n"
                    "        return '0.0.1'\n"
                    "    @property\n"
                    "    def name(self):\n"
                    "        return 'Bad'\n"
                    "    @property\n"
                    "    def max_score(self):\n"
                    "        return 1.0\n"
                    "    def __init__(self):\n"
                    "        raise RuntimeError('boom')\n"
                )
            with self.assertRaises(RuntimeError):
                _discover_plugins_in_dir(tmpdir, "testpkg", BenchmarkTaskPlugin)

    def test_discover_output_plugins_basic(self):
        plugins = discover_output_plugins()
        ids = [p.id for p in plugins]
        self.assertIn("output-csv", ids)
        self.assertIn("output-html", ids)
        self.assertIn("output-markdown", ids)
        self.assertIn("output-pdf", ids)

    def test_discover_output_plugins_whitelist(self):
        plugins = discover_output_plugins(whitelist=["output-csv"])
        self.assertEqual([p.id for p in plugins], ["output-csv"])

    def test_discover_output_plugins_blacklist(self):
        plugins = discover_output_plugins(blacklist=["output-csv"])
        self.assertNotIn("output-csv", [p.id for p in plugins])
        self.assertIn("output-html", [p.id for p in plugins])

    def test_discover_output_plugins_whitelist_and_blacklist_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            discover_output_plugins(whitelist=["csv"], blacklist=["pdf"])


if __name__ == "__main__":
    unittest.main()
