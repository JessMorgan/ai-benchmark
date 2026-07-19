"""Tests for plugin discovery and selection."""
import unittest

from plugins import discover_plugins


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
        self.assertEqual([p.id for p in plugins], ["code-review", "multi-step", "orchestration", "prd-creation", "rate-limiter", "structured-output", "tool-calling"])

    def test_whitelist_and_blacklist_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            discover_plugins(whitelist=["rate-limiter"], blacklist=["moe-dense"])

    def test_empty_whitelist_returns_all(self):
        plugins = discover_plugins(whitelist=[])
        self.assertEqual(len(plugins), len(discover_plugins()))


if __name__ == "__main__":
    unittest.main()
