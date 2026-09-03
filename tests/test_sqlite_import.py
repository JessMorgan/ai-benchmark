"""Tests for the legacy JSON to SQLite importer."""
import gzip
import json
import os
import tempfile
import unittest

from benchmark.logs import iter_log_members
from benchmark.sqlite_import import LegacySQLiteImporter
from benchmark.sqlite_payloads import SQLitePayloadStore
from benchmark.sqlite_schema import connect_database


class TestSQLiteImport(unittest.TestCase):
    def test_imports_state_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1",
                "runner": "http",
                "session_seed": 7,
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [{
                    "model": "model-a", "source": "Local", "api_model": "model-a",
                    "status": "ok", "rate-limiter_score": 18,
                    "rate-limiter_output_tokens": 20,
                    "rate-limiter_attempts": [
                        {"attempt": 1, "retry_reason": "transport"},
                        {"attempt": 2, "retry_reason": "thinking"},
                    ],
                    "rate-limiter_selected_attempt": 2,
                    "rate-limiter_judge_votes": [{
                        "model": "judge-a", "score": 17, "confidence": "high",
                        "rationale": "good", "usable": True,
                    }],
                }, {
                    "not": "mappable",
                }],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            summary = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(summary.imported_targets, 1)
            self.assertEqual(summary.imported_cells, 1)
            self.assertEqual(summary.imported_attempts, 1)
            self.assertEqual(summary.imported_votes, 1)
            self.assertEqual(summary.ambiguous_records, 1)

            connection = connect_database(database)
            self.addCleanup(connection.close)
            self.assertEqual(connection.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM judge_vote_attempts").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM legacy_import_records").fetchone()[0], 2)

            again = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(again.imported_attempts, 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM benchmark_attempts").fetchone()[0], 1)

    def test_import_uses_latest_duplicate_result_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1",
                "runner": "http",
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [
                    {"model": "model-a", "source": "Local", "api_model": "model-a",
                     "status": "ok", "rate-limiter_score": 4,
                     "rate-limiter_selected_attempt": 1},
                    {"model": "model-a", "source": "Local", "api_model": "model-a",
                     "status": "ok", "rate-limiter_score": 17,
                     "rate-limiter_selected_attempt": 1},
                ],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            summary = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(summary.imported_attempts, 1)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            score = connection.execute(
                "SELECT score FROM benchmark_attempts"
            ).fetchone()[0]
            self.assertEqual(score, 17)

    def test_import_uses_explicit_selected_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1", "runner": "http",
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {"model-a": {}},
                "results": [{
                    "model": "model-a", "source": "Local", "api_model": "model-a",
                    "status": "ok", "rate-limiter_score": 12,
                    "rate-limiter_selected_attempt": 1,
                    "rate-limiter_attempts": [
                        {"attempt": 1, "score": 12, "content": "selected"},
                        {"attempt": 2, "score": 99, "content": "later retry"},
                    ],
                }],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)
            LegacySQLiteImporter.import_path(source, database)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            row = connection.execute(
                """
                SELECT a.attempt_number, p.data, a.score
                FROM benchmark_attempts a
                JOIN payloads p ON p.payload_id = a.content_payload_id
                """
            ).fetchone()
            import gzip
            self.assertEqual((row[0], gzip.decompress(row[1]), row[2]), (1, b"selected", 12))

    def test_import_preserves_scores_from_partial_terminal_rows(self):
        """A cancellation row must not erase score metadata from JSON resume."""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            state = {
                "score_schema": "v1",
                "runner": "http",
                "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {
                    "model-a": {
                        "rate-limiter_score": 18,
                        "rate-limiter_output_tokens": 42,
                        "rate-limiter_tps": 3.5,
                    },
                },
                "results": [
                    {
                        "model": "model-a", "source": "Local", "api_model": "model-a",
                        "status": "ok", "rate-limiter_score": 18,
                        "rate-limiter_output_tokens": 42,
                        "rate-limiter_tps": 3.5,
                    },
                    {
                        "model": "model-a", "source": "Local", "api_model": "model-a",
                        "status": "error", "rate-limiter_score": "fail",
                        "rate-limiter_output_tokens": 7,
                        "rate-limiter_tps": 1.2,
                        "error": "Cancelled",
                    },
                ],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            LegacySQLiteImporter.import_path(source, database)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            row = connection.execute(
                "SELECT score, output_tokens, tps FROM benchmark_attempts"
            ).fetchone()
            self.assertEqual(tuple(row), (18, 7, 1.2))

    def test_imports_judge_sidecar_and_raw_response_as_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            database = os.path.join(tmp, "run.sqlite3")
            os.makedirs(os.path.join(tmp, "judge-inputs", "http", "model-a"))
            os.makedirs(os.path.join(tmp, "http", "responses", "model-a"))
            with open(os.path.join(tmp, "judge-inputs", "http", "model-a", "rate-limiter.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "target": "model-a", "state_key": "model-a", "runner": "http",
                    "plugin": "rate-limiter", "plugin_name": "Rate limiter",
                    "plugin_version": "1.0.0", "max_score": 20,
                    "prompt": "judge prompt with details",
                    "response": "candidate response",
                }, handle)
            response_path = os.path.join(
                tmp, "http", "responses", "model-a",
                "rate-limiter.judge.judge-a.contract-1.txt",
            )
            with open(response_path, "w", encoding="utf-8") as handle:
                handle.write('{"score": 17}')
            state = {
                "runner": "http", "active_plugins": ["rate-limiter"],
                "plugin_versions": {"rate-limiter": "1.0.0"},
                "model_info": {}, "results": [{
                    "model": "model-a", "source": "Local", "api_model": "model-a",
                    "status": "ok", "rate-limiter_score": 18,
                    "rate-limiter_judge_votes": [{
                        "model": "judge-a", "judge_contract_id": "contract-1",
                        "score": 17, "usable": True,
                    }],
                }],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            summary = LegacySQLiteImporter.import_path(source, database)
            self.assertEqual(summary.imported_artifacts, 1)
            connection = connect_database(database)
            self.addCleanup(connection.close)
            attempt = connection.execute(
                "SELECT raw_response_payload_id, request_payload_id FROM judge_attempts"
            ).fetchone()
            self.assertIsNotNone(attempt[0])
            self.assertIsNotNone(attempt[1])
            payloads = SQLitePayloadStore(connection)
            self.assertEqual(payloads.get_text(attempt[0]), '{"score": 17}')
            manifest = json.loads(payloads.get_text(attempt[1]))
            self.assertEqual(payloads.get_text(manifest["prompt_payload_id"]), "judge prompt with details")
            self.assertEqual(payloads.get_text(manifest["response_payload_id"]), "candidate response")

    def test_import_reads_nested_and_flat_judge_response_layouts(self):
        """The importer must accept both layout eras.

        Runs saved after the per-plugin response grouping keep judge raw
        responses inside ``responses/<target>/<plugin>/``; legacy flat runs
        kept them directly under ``responses/<target>/``. Converting either
        era must retain the judge raw response.
        """
        for layout in ("nested", "flat"):
            for state_key, runner, responses_root in (
                ("model-a", "http", "http"),
                # Runner-suffixed state keys must resolve to the SAME plain
                # target directory the writer used (``model-a [opencode]``
                # -> ``responses/model-a/``), never a sanitized-state-key dir.
                ("model-a [opencode]", "opencode", "opencode"),
            ):
                with self.subTest(layout=layout, state_key=state_key):
                    with tempfile.TemporaryDirectory() as tmp:
                        source = os.path.join(tmp, "benchmark_state.json")
                        database = os.path.join(tmp, "run.sqlite3")
                        plugin_dir = os.path.join(tmp, responses_root, "responses", "model-a")
                        # Flat layout: <plugin>.judge.<suffix>.txt beside the
                        # target dir; nested layout: judge.<suffix>.txt inside
                        # the plugin's subdirectory.
                        response_name = (
                            "judge.judge-a.contract-1.txt" if layout == "nested"
                            else "rate-limiter.judge.judge-a.contract-1.txt"
                        )
                        if layout == "nested":
                            plugin_dir = os.path.join(plugin_dir, "rate-limiter")
                        os.makedirs(plugin_dir)
                        with open(os.path.join(plugin_dir, response_name), "w", encoding="utf-8") as handle:
                            handle.write('{"score": 17}')
                        state = {
                            "runner": runner, "active_plugins": ["rate-limiter"],
                            "plugin_versions": {"rate-limiter": "1.0.0"},
                            "model_info": {}, "results": [{
                                "model": "model-a", "state_key": state_key,
                                "source": "Local", "api_model": "model-a",
                                "runner": runner,
                                "status": "ok", "rate-limiter_score": 18,
                                "rate-limiter_judge_votes": [{
                                    "model": "judge-a", "judge_contract_id": "contract-1",
                                    "score": 17, "usable": True,
                                }],
                            }],
                        }
                        with open(source, "w", encoding="utf-8") as handle:
                            json.dump(state, handle)

                        LegacySQLiteImporter.import_path(source, database)
                        connection = connect_database(database)
                        self.addCleanup(connection.close)
                        attempt = connection.execute(
                            "SELECT raw_response_payload_id FROM judge_attempts"
                        ).fetchone()
                        self.assertIsNotNone(
                            attempt[0],
                            f"{layout}/{state_key} layout must retain the raw response",
                        )

    def test_debug_log_import_is_opt_in_and_compresses_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "benchmark_state.json")
            log_path = os.path.join(tmp, "legacy", "model-a.log")
            os.makedirs(os.path.dirname(log_path))
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("legacy request\nlegacy response\n")
            state = {
                "active_plugins": [], "model_info": {}, "results": [],
            }
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(state, handle)

            compact_db = os.path.join(tmp, "compact.sqlite3")
            compact = LegacySQLiteImporter.import_path(source, compact_db)
            self.assertEqual(compact.imported_debug_logs, 0)
            self.assertFalse(os.path.exists(os.path.join(tmp, "logs", "imported")))

            output_dir = os.path.join(tmp, "converted")
            os.makedirs(output_dir)
            debug_db = os.path.join(output_dir, "debug.sqlite3")
            debug = LegacySQLiteImporter.import_path(
                source, debug_db, run_id="debug-run", include_debug_logs=True,
            )
            self.assertEqual(debug.imported_debug_logs, 1)
            imported_path = os.path.join(output_dir, "logs", "imported", "legacy", "model-a.log.gz")
            self.assertTrue(os.path.isfile(imported_path))
            self.assertEqual(list(iter_log_members(imported_path)), [b"legacy request\nlegacy response\n"])
            debug_connection = connect_database(debug_db)
            self.addCleanup(debug_connection.close)
            self.assertEqual(
                debug_connection.execute(
                    "SELECT compression, complete_members FROM debug_log_files"
                ).fetchone()[:],
                ("gzip", 1),
            )
            with gzip.open(imported_path, "rb") as handle:
                self.assertEqual(handle.read(), b"legacy request\nlegacy response\n")

    def test_import_rejects_non_object_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "state.json")
            database = os.path.join(tmp, "run.sqlite3")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump([], handle)
            with self.assertRaises(TypeError):
                LegacySQLiteImporter.import_path(source, database)


if __name__ == "__main__":
    unittest.main()
