"""Codec measurements for append-only debug-log payloads."""
from __future__ import annotations

import bz2
import gzip
import time
from typing import Any


def measure_codecs(data: bytes, *, repetitions: int = 1) -> dict[str, Any]:
    """Measure compressed size and wall time for standard-library codecs.

    This is intentionally a measurement helper, not a runtime codec switch:
    gzip remains the append-only format because concatenated members provide
    the required crash-recovery semantics.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    measurements: dict[str, dict[str, float | int | str]] = {}
    for name, compressor in (("gzip", gzip.compress), ("bz2", bz2.compress)):
        started = time.perf_counter()
        compressed = b""
        for _ in range(repetitions):
            compressed = compressor(data)
        elapsed = time.perf_counter() - started
        measurements[name] = {
            "input_bytes": len(data),
            "compressed_bytes": len(compressed),
            "ratio": round(len(compressed) / len(data), 6) if data else 0.0,
            "elapsed_ms": round(elapsed * 1000, 4),
            "repetitions": repetitions,
        }
    return {
        "codec_measurements": measurements,
        "recommended_append_codec": "gzip",
        "recommendation_reason": "gzip supports concatenated streaming members in the standard library",
    }
