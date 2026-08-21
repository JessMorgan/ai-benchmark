"""Tests for canonical compressed SQLite payloads."""
import sqlite3
import tempfile
import unittest

from benchmark.sqlite_payloads import (
    PayloadIntegrityError,
    SQLitePayloadStore,
    build_payload_only_judge_input,
)
from benchmark.sqlite_schema import connect_database


class TestSQLitePayloadStore(unittest.TestCase):
    def _store(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        connection = connect_database(f"{self.tmpdir.name}/run.sqlite3")
        self.addCleanup(connection.close)
        return connection, SQLitePayloadStore(connection)

    def test_text_round_trip_and_deduplication(self):
        connection, store = self._store()
        first = store.put_text("prompt", "hello 🌍")
        second = store.put_text("response", "hello 🌍")
        self.assertEqual(first, second)
        self.assertEqual(store.get_text(first), "hello 🌍")
        self.assertEqual(store.count(), 1)
        metadata = store.metadata(first)
        self.assertEqual(metadata["kind"], "prompt")
        self.assertEqual(metadata["compression"], "gzip")
        self.assertLess(metadata["stored_bytes"], metadata["uncompressed_bytes"] + 30)
        connection.close()

    def test_binary_round_trip_and_empty_payload(self):
        _connection, store = self._store()
        payload = bytes(range(256))
        payload_id = store.put("binary", payload)
        self.assertEqual(store.get(payload_id), payload)
        empty_id = store.put("empty", b"")
        self.assertEqual(store.get(empty_id), b"")

    def test_missing_and_corrupt_payloads_are_visible(self):
        connection, store = self._store()
        with self.assertRaises(KeyError):
            store.get(999)
        payload_id = store.put_text("prompt", "integrity")
        connection.execute(
            "UPDATE payloads SET data = ? WHERE payload_id = ?",
            (sqlite3.Binary(b"not gzip"), payload_id),
        )
        connection.commit()
        with self.assertRaises(PayloadIntegrityError):
            store.get(payload_id)

    def test_payload_hash_mismatch_is_visible(self):
        connection, store = self._store()
        payload_id = store.put_text("prompt", "integrity")
        connection.execute(
            "UPDATE payloads SET sha256 = ? WHERE payload_id = ?",
            ("0" * 64, payload_id),
        )
        connection.commit()
        with self.assertRaises(PayloadIntegrityError):
            store.get(payload_id)

    def test_judge_manifest_contains_ids_not_large_text(self):
        _connection, store = self._store()
        item = {
            "target": "model",
            "plugin": "data-transformation",
            "prompt": "large prompt",
            "response": "large response",
            "response_sha256": "hash",
        }
        manifest = build_payload_only_judge_input(store, item)
        self.assertNotIn("prompt", manifest)
        self.assertNotIn("response", manifest)
        self.assertIsInstance(manifest["prompt_payload_id"], int)
        self.assertIsInstance(manifest["response_payload_id"], int)
        self.assertEqual(store.get_text(manifest["prompt_payload_id"]), "large prompt")
        self.assertEqual(store.get_text(manifest["response_payload_id"]), "large response")


if __name__ == "__main__":
    unittest.main()
