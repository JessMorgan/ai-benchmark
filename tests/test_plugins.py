"""Tests for plugin discovery and selection."""
import os
import tempfile
import unittest

from benchmark.plugin import BenchmarkOutputPlugin, BenchmarkTaskPlugin, EvaluationResult
from plugins import (
    PluginDiscoveryError,
    _discover_plugins_in_dir,
    discover_output_plugins,
    discover_plugins,
    format_plugin_list,
    plugin_inventory,
)


class _ConcreteTaskPlugin(BenchmarkTaskPlugin):
    """Minimal concrete subclass exercising the non-abstract defaults."""

    id = "concrete"
    version = "1.0.0"
    name = "Concrete"
    max_score = 20.0

    def get_prompt(self):
        return "prompt"

    def get_temperature(self, global_config):
        return None

    def score(self, response_text):
        return 10.0


class TestPluginBaseDefaults(unittest.TestCase):
    def test_supports_streaming_defaults_to_true(self):
        self.assertTrue(_ConcreteTaskPlugin().supports_streaming)

    def test_evaluate_wraps_score_with_empty_rubric(self):
        result = _ConcreteTaskPlugin().evaluate("some answer")
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.score, 10.0)
        self.assertEqual(result.rubric, [])

    def test_output_plugin_is_abstract(self):
        with self.assertRaises(TypeError):
            BenchmarkOutputPlugin()  # abstract: id/name/extension/generate missing

    def test_output_plugin_concrete_round_trip(self):
        class Out(BenchmarkOutputPlugin):
            id = "output-x"
            name = "X"
            extension = "x"

            def generate(self, results, active_plugins, output_dir=None, session_seed=None):
                return "path"

        self.assertEqual(Out().generate([], []), "path")

    def test_abstract_base_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            BenchmarkTaskPlugin()
        with self.assertRaises(TypeError):
            BenchmarkOutputPlugin()


class TestPluginDiscovery(unittest.TestCase):
    def test_discovers_all_builtin_plugins(self):
        plugins = discover_plugins()
        ids = [p.id for p in plugins]
        self.assertEqual(
            ids,
            [
                "code-review",
                "debug-traversal",
                "error-recovery",
                "moe-dense",
                "multi-step",
                "multi-turn-conversation",
                "orchestration",
                "prd-creation",
                "rate-limiter",
                "software-architecture",
                "structured-output",
                "tool-calling",
                "wireframes",
            ],
        )

    def test_plugins_have_required_metadata(self):
        plugins = discover_plugins()
        inventory = plugin_inventory(plugins)
        self.assertEqual([entry["id"] for entry in inventory], [p.id for p in plugins])
        for p in plugins:
            self.assertTrue(p.id)
            self.assertTrue(p.version)
            self.assertTrue(p.name)
            self.assertGreater(p.max_score, 0)
            self.assertIsInstance(p.supports_streaming, bool)

    def test_discovery_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = (
                "from benchmark.plugin import BenchmarkTaskPlugin\n"
                "class Plugin(BenchmarkTaskPlugin):\n"
                "    @property\n"
                "    def id(self): return 'duplicate'\n"
                "    @property\n"
                "    def version(self): return '1.0.0'\n"
                "    @property\n"
                "    def name(self): return 'Plugin'\n"
                "    @property\n"
                "    def max_score(self): return 1\n"
                "    def get_prompt(self): return ''\n"
                "    def get_temperature(self, global_config): return None\n"
                "    def score(self, response_text): return 0\n"
            )
            for name in ("a.py", "b.py"):
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write(source.replace("class Plugin", f"class Plugin{name[0].upper()}"))
            with self.assertRaises(PluginDiscoveryError) as ctx:
                _discover_plugins_in_dir(tmpdir, "duplicate_test", BenchmarkTaskPlugin)
            self.assertIn("Duplicate plugin id", str(ctx.exception))

    def test_discovery_rejects_invalid_streaming_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "bad_streaming.py")
            with open(plugin_path, "w") as f:
                f.write(
                    "from benchmark.plugin import BenchmarkTaskPlugin\n"
                    "class BadStreaming(BenchmarkTaskPlugin):\n"
                    "    id = 'bad-streaming'\n"
                    "    version = '1.0.0'\n"
                    "    name = 'Bad'\n"
                    "    max_score = 1\n"
                    "    supports_streaming = 'yes'\n"
                    "    def get_prompt(self): return ''\n"
                    "    def get_temperature(self, global_config): return None\n"
                    "    def score(self, response_text): return 0\n"
                )
            with self.assertRaises(PluginDiscoveryError) as ctx:
                _discover_plugins_in_dir(tmpdir, "bad_streaming_test", BenchmarkTaskPlugin)
            self.assertIn("supports_streaming must be a boolean", str(ctx.exception))

    def test_discovery_rejects_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "bad_metadata.py")
            with open(plugin_path, "w") as f:
                f.write(
                    "from benchmark.plugin import BenchmarkTaskPlugin\n"
                    "class BadMetadata(BenchmarkTaskPlugin):\n"
                    "    id = ''\n"
                    "    version = '1.0.0'\n"
                    "    name = 'Bad'\n"
                    "    max_score = 1\n"
                    "    def get_prompt(self): return ''\n"
                    "    def get_temperature(self, global_config): return None\n"
                    "    def score(self, response_text): return 0\n"
                )
            with self.assertRaises(PluginDiscoveryError) as ctx:
                _discover_plugins_in_dir(tmpdir, "bad_metadata_test", BenchmarkTaskPlugin)
            self.assertIn("id must be a non-empty string", str(ctx.exception))

    def test_whitelist_filters_plugins(self):
        plugins = discover_plugins(whitelist=["rate-limiter"])
        self.assertEqual([p.id for p in plugins], ["rate-limiter"])

    def test_blacklist_filters_plugins(self):
        all_ids = {p.id for p in discover_plugins()}
        plugins = discover_plugins(blacklist=["moe-dense"])
        self.assertEqual({p.id for p in plugins}, all_ids - {"moe-dense"})

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
                    "from benchmark.plugin import BenchmarkTaskPlugin\n"
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

    def test_output_plugins_have_required_metadata(self):
        for plugin in discover_output_plugins():
            self.assertTrue(plugin.id)
            self.assertTrue(plugin.name)
            self.assertTrue(plugin.extension)

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
