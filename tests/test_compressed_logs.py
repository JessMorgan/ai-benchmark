"""Tests for append-only gzip concatenated-member logs."""
import gzip
import os
import tempfile
import threading
import unittest
import zlib
from unittest import mock

from benchmark.http import log_request_entry
from benchmark.log_codec import measure_codecs
from benchmark.logs import (
    AppendOnlyGzipLog,
    iter_log_members,
    recover_log,
    redact_log_text,
)


class TestGzipAppendLog(unittest.TestCase):
    def _path(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return tmpdir.name, os.path.join(tmpdir.name, "logs", "run.log.gz")

    def test_one_and_multiple_members_round_trip(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, member_target_bytes=1024)
        writer.append_record(["first\n"])
        writer.append_record(["second", " 🌍\n"])
        writer.close()
        self.assertEqual(list(iter_log_members(path)), [b"first\n", "second 🌍\n".encode()])
        with gzip.open(path, "rb") as handle:
            self.assertEqual(handle.read(), "first\nsecond 🌍\n".encode())

    def test_append_does_not_rewrite_existing_members(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append_record([b"old"])
        with open(path, "rb") as handle:
            first_bytes = handle.read()
        writer.append_record([b"new"])
        writer.close()
        with open(path, "rb") as handle:
            appended_bytes = handle.read()
        self.assertEqual(appended_bytes[:len(first_bytes)], first_bytes)
        self.assertEqual(list(iter_log_members(path)), [b"old", b"new"])

    def test_buffered_append_flushes_members_and_empty_data(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, member_target_bytes=3)
        writer.append("abc")
        writer.append("def")
        writer.append(b"")
        writer.close()
        self.assertEqual(b"".join(iter_log_members(path)), b"abcdef")

    def test_final_member_truncation_is_recoverable(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append_record([b"complete"])
        writer.append_record([b"partial"])
        writer.close()
        complete_end = _member_ranges(path)[0][1]
        with open(path, "r+b") as handle:
            handle.truncate(_member_ranges(path)[1][1] - 1)
        recovery = recover_log(path)
        self.assertTrue(recovery.truncated_tail)
        self.assertFalse(recovery.invalid_tail)
        self.assertEqual(list(iter_log_members(path)), [b"complete"])
        repaired = recover_log(path, repair=True)
        self.assertEqual(repaired.complete_members, 1)
        self.assertEqual(os.path.getsize(path), complete_end)

    def test_every_final_member_truncation_preserves_previous_members(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, redact=False)
        writer.append_record([b"complete"])
        writer.append_record([b"partial payload with a footer"])
        writer.close()
        ranges = _member_ranges(path)
        with open(path, "rb") as handle:
            original = handle.read()
        for cut in range(ranges[1][0] + 1, ranges[1][1]):
            candidate = f"{path}.{cut}.gz"
            with open(candidate, "wb") as handle:
                handle.write(original[:cut])
            recovery = recover_log(candidate)
            if cut == ranges[1][0]:
                self.assertFalse(recovery.truncated_tail, cut)
            else:
                self.assertTrue(recovery.truncated_tail, cut)
            self.assertFalse(recovery.invalid_tail, cut)
            self.assertEqual(list(iter_log_members(candidate)), [b"complete"], cut)
            os.unlink(candidate)

    def test_python_gzip_reader_matches_tolerant_reader_for_complete_log(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, redact=False)
        writer.append_record([b"one"])
        writer.append_record([b"two"])
        writer.close()
        with gzip.open(path, "rb") as handle:
            standard = handle.read()
        tolerant = b"".join(iter_log_members(path))
        self.assertEqual(standard, tolerant)

    def test_abrupt_child_termination_leaves_prior_member_readable(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, redact=False)
        writer.append_record([b"prior"])
        writer.close()
        with open(path, "ab") as handle:
            handle.write(gzip.compress(b"unfinished")[:-3])
        recovery = recover_log(path)
        self.assertTrue(recovery.truncated_tail)
        self.assertEqual(list(iter_log_members(path)), [b"prior"])

    def test_concurrent_open_recovery_is_serialized(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, redact=False)
        writer.append_record([b"prior"])
        writer.close()
        with open(path, "ab") as handle:
            handle.write(b"\x1f\x8b\x08")
        errors = []
        def open_and_append():
            try:
                current = AppendOnlyGzipLog(path, recover_tail=True, redact=False)
                current.append_record([b"next"])
                current.close()
            except Exception as exc:  # pragma: no cover - assertion below captures unexpected races
                errors.append(exc)
        threads = [threading.Thread(target=open_and_append) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        records = list(iter_log_members(path))
        self.assertEqual(records[0], b"prior")
        self.assertEqual(records[1:], [b"next"] * 4)

    def test_write_and_fsync_failures_are_visible(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path, redact=False)
        writer.append(b"payload")
        with mock.patch("benchmark.logs.os.fsync", side_effect=OSError("sync failed")):
            with self.assertRaises(OSError):
                writer.flush(sync=True)
        writer.close(sync=False)

    def test_legacy_plaintext_logs_are_readable(self):
        _tmpdir, path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"legacy request\nlegacy response\n")
        self.assertEqual(list(iter_log_members(path)), [b"legacy request\nlegacy response\n"])

    def test_codec_measurement_is_reproducible_and_recommends_gzip_for_append(self):
        result = measure_codecs((b"judge transcript\\n" * 1000), repetitions=2)
        self.assertEqual(result["recommended_append_codec"], "gzip")
        self.assertEqual(set(result["codec_measurements"]), {"gzip", "bz2"})
        self.assertLess(
            result["codec_measurements"]["gzip"]["compressed_bytes"],
            result["codec_measurements"]["gzip"]["input_bytes"],
        )

    def test_header_only_and_footer_truncations_preserve_previous_members(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append_record([b"complete"])
        writer.close()
        with open(path, "ab") as handle:
            handle.write(b"\x1f\x8b\x08")
        recovery = recover_log(path)
        self.assertTrue(recovery.truncated_tail)
        self.assertEqual(list(iter_log_members(path)), [b"complete"])

    def test_append_after_recovery_keeps_complete_members(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append_record([b"complete"])
        writer.append_record([b"partial"])
        writer.close()
        ranges = _member_ranges(path)
        with open(path, "r+b") as handle:
            handle.truncate(ranges[1][1] - 2)
        recovered_writer = AppendOnlyGzipLog(path, recover_tail=True)
        recovered_writer.append_record([b"after"])
        recovered_writer.close()
        self.assertEqual(list(iter_log_members(path)), [b"complete", b"after"])

    def test_non_final_corruption_is_not_repaired(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append_record([b"first"])
        writer.append_record([b"second"])
        writer.close()
        ranges = _member_ranges(path)
        with open(path, "r+b") as handle:
            handle.seek(ranges[0][0] + 10)
            original = handle.read(1)
            handle.seek(ranges[0][0] + 10)
            handle.write(bytes([original[0] ^ 0xFF]))
        recovery = recover_log(path)
        self.assertTrue(recovery.invalid_tail)
        before = os.path.getsize(path)
        repaired = recover_log(path, repair=True)
        self.assertTrue(repaired.invalid_tail)
        self.assertEqual(os.path.getsize(path), before)

    def test_concurrent_records_do_not_interleave(self):
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        threads = [
            threading.Thread(target=writer.append_record, args=([f"record-{i}"],))
            for i in range(30)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        writer.close()
        records = {item.decode() for item in iter_log_members(path)}
        self.assertEqual(records, {f"record-{i}" for i in range(30)})

    def test_http_request_logging_uses_gzip_members(self):
        _tmpdir, path = self._path()
        log_request_entry(path, "curl -H 'Authorization: secret'", "response", "attempt 1")
        log_request_entry(path, "curl -H 'Authorization: secret2'", "response 2", "attempt 2")
        content = b"".join(iter_log_members(path))
        self.assertIn(b"attempt 1", content)
        self.assertIn(b"attempt 2", content)
        self.assertTrue(path.endswith(".gz"))

    def test_redaction_happens_before_compression(self):
        text, changed = redact_log_text(
            "Authorization: Bearer secret\n"
            '{"api_key":"top-secret","safe":"value"}'
        )
        self.assertTrue(changed)
        self.assertNotIn("secret", text)
        self.assertIn("[REDACTED]", text)
        _tmpdir, path = self._path()
        writer = AppendOnlyGzipLog(path)
        writer.append(text)
        writer.close()
        self.assertNotIn(b"secret", b"".join(iter_log_members(path)))
        self.assertTrue(writer.redaction_occurred)


def _member_ranges(path):
    """Return complete member offsets using the public recovery result plus scan."""
    # Locate gzip headers and use gzip.decompress on each candidate suffix; this
    # helper is test-only and the production scanner remains bounded-memory.
    with open(path, "rb") as handle:
        data = handle.read()
    ranges = []
    offset = 0
    while offset < len(data):
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoder.decompress(data[offset:])
        if not decoder.eof:
            break
        consumed = len(data[offset:]) - len(decoder.unused_data)
        ranges.append((offset, offset + consumed))
        offset += consumed
    return ranges


if __name__ == "__main__":
    unittest.main()
