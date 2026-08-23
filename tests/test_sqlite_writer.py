"""Tests for the Stage 3 SQLite write queue."""
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock

from benchmark.sqlite_writer import SQLiteWriteQueue, _WorkItem


class TestSQLiteWriteQueue(unittest.TestCase):
    def test_batches_operations_and_flushes_after_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(f"{tmpdir}/run.sqlite3", batch_size=8)
            futures = [
                writer.submit(
                    lambda connection, value=value: connection.execute(
                        "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                        (f"key-{value}", str(value)),
                    )
                )
                for value in range(5)
            ]
            writer.flush(timeout=5)
            for future in futures:
                future.result(timeout=1)
            self.assertEqual(writer.committed_batches, 1)
            self.assertTrue(writer.close(timeout=5))
            connection = sqlite3.connect(f"{tmpdir}/run.sqlite3")
            count = connection.execute(
                "SELECT count(*) FROM schema_meta WHERE key LIKE 'key-%'"
            ).fetchone()[0]
            self.assertEqual(count, 5)
            connection.close()

    def test_operation_failure_rolls_back_batch_and_notifies(self):
        failures = []
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(
                f"{tmpdir}/run.sqlite3", batch_size=8,
                failure_callback=failures.append,
            )
            good = writer.submit(
                lambda connection: connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('good', '1')"
                )
            )
            bad = writer.submit(
                lambda connection: connection.execute(
                    "INSERT INTO no_such_table VALUES (1)"
                )
            )
            with self.assertRaises(sqlite3.Error):
                bad.result(timeout=5)
            self.assertEqual(good.result(timeout=1).rowcount, 1)
            self.assertTrue(writer.close(timeout=5))
            self.assertEqual(len(failures), 1)
            connection = sqlite3.connect(f"{tmpdir}/run.sqlite3")
            self.assertEqual(
                connection.execute("SELECT count(*) FROM schema_meta WHERE key = 'good'").fetchone()[0],
                1,
            )
            connection.close()

    def test_concurrent_submissions_are_serialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(f"{tmpdir}/run.sqlite3", batch_size=16)
            futures = []
            lock = threading.Lock()

            def submit_range(start):
                local = [
                    writer.submit(
                        lambda connection, value=value: connection.execute(
                            "INSERT INTO schema_meta(key, value) VALUES (?, ?)",
                            (f"thread-{start}-{value}", "ok"),
                        )
                    )
                    for value in range(20)
                ]
                with lock:
                    futures.extend(local)

            threads = [threading.Thread(target=submit_range, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            writer.flush(timeout=5)
            for future in futures:
                future.result(timeout=1)
            writer.close(timeout=5)
            connection = sqlite3.connect(f"{tmpdir}/run.sqlite3")
            self.assertEqual(
                connection.execute("SELECT count(*) FROM schema_meta WHERE key LIKE 'thread-%'").fetchone()[0],
                80,
            )
            connection.close()

    def test_outer_commit_failure_fails_pending_future(self):
        writer = SQLiteWriteQueue(":memory:")
        connection = mock.MagicMock()
        connection.commit.side_effect = sqlite3.Error("commit failed")
        future = writer.submit(lambda _connection: "value")
        # Avoid starting the real writer: exercise the transaction failure
        # branch directly with a controlled connection double.
        writer._execute_batch(connection, [_WorkItem(lambda _connection: "value", future)])
        with self.assertRaisesRegex(sqlite3.Error, "commit failed"):
            future.result()
        self.assertEqual(len(writer.failures), 1)

    def test_close_drains_items_after_shutdown_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(f"{tmpdir}/run.sqlite3", batch_size=1)
            started = threading.Event()
            release = threading.Event()

            def slow(_connection):
                started.set()
                release.wait(5)

            first = writer.submit(slow)
            self.assertTrue(started.wait(5))
            second = writer.submit(lambda connection: connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES ('after', 'marker')"
            ))
            closer = threading.Thread(target=lambda: writer.close(timeout=5))
            closer.start()
            time.sleep(0.02)
            release.set()
            first.result(timeout=5)
            second.result(timeout=5)
            closer.join(timeout=5)
            self.assertFalse(closer.is_alive())

    def test_close_timeout_reports_live_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(f"{tmpdir}/run.sqlite3")
            started = threading.Event()

            def slow(_connection):
                started.set()
                time.sleep(0.2)

            future = writer.submit(slow)
            self.assertTrue(started.wait(5))
            self.assertFalse(writer.close(timeout=0.001))
            future.result(timeout=5)
            self.assertTrue(writer.close(timeout=5))

    def test_submit_after_close_returns_failed_future(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SQLiteWriteQueue(f"{tmpdir}/run.sqlite3")
            self.assertTrue(writer.close(timeout=1))
            future = writer.submit(lambda _connection: None)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                future.result()


if __name__ == "__main__":
    unittest.main()
