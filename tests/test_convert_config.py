"""Tests for the --convert-config CLI argument."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import yaml


class TestConvertConfig(unittest.TestCase):
    def test_convert_yaml_to_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                f.write("output_dir: results\ntimeout: 100\n")

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--convert-config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["output_dir"], "results")
            self.assertEqual(parsed["timeout"], 100)

    def test_convert_json_to_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"output_dir": "results", "timeout": 100}, f)

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--convert-config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            parsed = yaml.safe_load(result.stdout)
            self.assertEqual(parsed["output_dir"], "results")
            self.assertEqual(parsed["timeout"], 100)

    def test_convert_missing_config_file(self):
        result = subprocess.run(
            [sys.executable, "ai-benchmark.py", "--convert-config", "/tmp/does-not-exist.yaml"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found", result.stderr)

    def test_convert_yml_to_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yml")
            with open(config_path, "w") as f:
                f.write("output_dir: results\ntimeout: 100\n")

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--convert-config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["output_dir"], "results")
            self.assertEqual(parsed["timeout"], 100)

    def test_convert_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.txt")
            with open(config_path, "w") as f:
                f.write("not a valid config")

            result = subprocess.run(
                [sys.executable, "ai-benchmark.py", "--convert-config", config_path],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported config format", result.stderr)


if __name__ == "__main__":
    unittest.main()
