import argparse
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.configuration import Configuration


class TestConfiguration(unittest.TestCase):
    def test_loads_file_and_derives_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "timeout": 10,
                "sources": {"Local": {"api_protocol": "openai"}},
                "models": {"demo": "Local"},
            }))
            config = Configuration.from_file(str(path))
        self.assertEqual(config.value("timeout"), 10)
        self.assertEqual(config.targets["demo"]["source"], "Local")

    def test_cli_overrides_are_applied_once(self):
        args = argparse.Namespace(
            retry_on_429=False,
            timeout=30,
            max_tokens=512,
            temperature=0.2,
            plugin_thread_limit=2,
            plugin_temperature=["rate-limiter=0.4"],
        )
        config = Configuration.from_mapping({
            "sources": {"Local": {}},
            "models": {"demo": "Local"},
        }, args)
        self.assertEqual(config.value("timeout"), 30)
        self.assertEqual(config.value("max_tokens"), 512)
        self.assertEqual(config.value("temperature"), 0.2)
        self.assertEqual(config.plugin_temperatures["rate-limiter"], 0.4)
        self.assertEqual(config.source_config()["Local"]["max_429_retries"], 0)
        self.assertEqual(config.source_config()["Local"]["plugin_thread_limit"], 2)

    def test_timeout_retry_defaults_terminal_and_configurable(self):
        """Timeouts are terminal by default; config and CLI can opt in."""
        # No flag, no config key: terminal (key absent -> resolve as False).
        base = {"sources": {"Local": {}}, "models": {"demo": "Local"}}
        config = Configuration.from_mapping(base)
        self.assertNotIn("retry_on_timeout", config.source_config()["Local"])

        # Per-source config opt-in survives a no-op global override.
        config = Configuration.from_mapping({
            "sources": {"Local": {"retry_on_timeout": True}},
            "models": {"demo": "Local"},
        })
        self.assertIs(config.source_config()["Local"]["retry_on_timeout"], True)

        # --retry-on-timeout opts in sources that did not set the key.
        args = argparse.Namespace(retry_on_timeout=True)
        config = Configuration.from_mapping(base, args)
        self.assertIs(config.source_config()["Local"]["retry_on_timeout"], True)

        # Per-source opt-in is preserved when the flag is not passed.
        config = Configuration.from_mapping({
            "sources": {"Local": {"retry_on_timeout": True}},
            "models": {"demo": "Local"},
        }, argparse.Namespace(retry_on_timeout=None))
        self.assertIs(config.source_config()["Local"]["retry_on_timeout"], True)

        # --no-retry-on-timeout is the kill-switch: it overrides opt-ins.
        config = Configuration.from_mapping({
            "sources": {"Local": {"retry_on_timeout": True}},
            "models": {"demo": "Local"},
        }, argparse.Namespace(retry_on_timeout=False))
        self.assertIs(config.source_config()["Local"]["retry_on_timeout"], False)


if __name__ == "__main__":
    unittest.main()
