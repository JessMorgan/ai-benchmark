"""CLI tests for legacy JSON to SQLite conversion."""
import json
import os
import subprocess
import sys
import tempfile
import unittest


class TestSQLiteImportCLI(unittest.TestCase):
    def _write_state(self, directory):
        path = os.path.join(directory, "benchmark_state.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "score_schema": "v1",
                "active_plugins": ["rate-limiter"],
                "model_info": {},
                "results": [],
            }, handle)
        return path

    def test_import_defaults_beside_source_and_preserves_json(self):
        launcher = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai-benchmark.py")
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write_state(tmp)
            with open(source, encoding="utf-8") as handle:
                original = handle.read()
            result = subprocess.run(
                [sys.executable, launcher, "--import-to-sqlite", source],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "run.sqlite3")))
            with open(os.path.join(tmp, "run-info.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["storage"], "sqlite")
            self.assertTrue(manifest["run_id"])
            self.assertEqual(manifest["revision_id"], 1)
            with open(source, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), original)
            report = json.loads(result.stdout)
            self.assertEqual(report["summary"]["imported_attempts"], 0)

    def test_existing_output_requires_explicit_overwrite(self):
        launcher = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai-benchmark.py")
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write_state(tmp)
            output = os.path.join(tmp, "converted.sqlite3")
            with open(output, "wb") as handle:
                handle.write(b"do not replace")
            result = subprocess.run(
                [sys.executable, launcher, "--import-to-sqlite", source,
                 "--sqlite-output", output],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            with open(output, "rb") as handle:
                self.assertEqual(handle.read(), b"do not replace")


if __name__ == "__main__":
    unittest.main()
