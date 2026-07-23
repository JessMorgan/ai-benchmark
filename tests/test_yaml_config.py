"""Tests for YAML config support."""
import json
import os
import tempfile
import unittest

from benchmark_core import load_config


class TestLoadConfigYAML(unittest.TestCase):
    def test_load_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                json.dump({"output_dir": "results", "timeout": 100}, f)
            cfg = load_config(path)
            self.assertEqual(cfg["output_dir"], "results")
            self.assertEqual(cfg["timeout"], 100)

    def test_load_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                f.write("output_dir: results\ntimeout: 100\n")
            cfg = load_config(path)
            self.assertEqual(cfg["output_dir"], "results")
            self.assertEqual(cfg["timeout"], 100)

    def test_load_yml_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yml")
            with open(path, "w") as f:
                f.write("output_dir: results\ntimeout: 100\n")
            cfg = load_config(path)
            self.assertEqual(cfg["output_dir"], "results")
            self.assertEqual(cfg["timeout"], 100)

    def test_empty_yaml_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_load_yaml_with_nested_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                f.write(
                    "sources:\n"
                    "  Local:\n"
                    "    api_url: http://localhost:11434/chat/completions\n"
                    "    headers:\n"
                    "      Authorization: Bearer key\n"
                    "models:\n"
                    "  model-a: Local\n"
                    "agents:\n"
                    "  agent-a:\n"
                    "    model: gpt-4\n"
                    "    source: Local\n"
                    "    system_prompt: You are a coder.\n"
                )
            cfg = load_config(path)
            self.assertEqual(cfg["sources"]["Local"]["api_url"], "http://localhost:11434/chat/completions")
            self.assertEqual(cfg["models"]["model-a"], "Local")
            self.assertEqual(cfg["agents"]["agent-a"]["model"], "gpt-4")
            self.assertEqual(cfg["agents"]["agent-a"]["system_prompt"], "You are a coder.")


if __name__ == "__main__":
    unittest.main()
