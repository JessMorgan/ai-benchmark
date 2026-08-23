"""Tests for the synthetic storage measurement tool."""
import unittest

from benchmark.storage_measure import measure_storage


class TestStorageMeasure(unittest.TestCase):
    def test_measurement_reports_size_deduplication_and_latency(self):
        report = measure_storage(
            targets=2, plugins=2, payload_chars=128, attempts=2,
            flush_interval=0.001,
        )
        self.assertEqual(report["parameters"]["targets"], 2)
        self.assertGreater(report["json_bytes"], 0)
        self.assertGreater(report["sqlite_bytes"], 0)
        self.assertEqual(report["payload_rows"], 1)
        self.assertEqual(report["attempt_rows"], 8)
        self.assertGreaterEqual(report["record_latency_ms"]["p95"], 0)

    def test_invalid_measurement_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            measure_storage(targets=0)


if __name__ == "__main__":
    unittest.main()
