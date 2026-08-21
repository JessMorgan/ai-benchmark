"""Canonical compressed payload storage for SQLite runs."""
from __future__ import annotations

import gzip
import hashlib
import sqlite3
from typing import Any


class PayloadIntegrityError(RuntimeError):
    """Raised when a stored payload cannot be decoded or fails its hash."""


def build_payload_only_judge_input(store: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Replace embedded judge prompt/response text with canonical payload IDs."""
    prompt = item.get("prompt")
    response = item.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("judge input must contain string prompt and response fields")
    manifest = {
        key: value for key, value in item.items()
        if key not in {"prompt", "response"}
    }
    manifest["prompt_payload_id"] = store.put_text("judge-prompt", prompt)
    manifest["response_payload_id"] = store.put_text("candidate-response", response)
    return manifest


class SQLitePayloadStore:
    """Store each uncompressed payload once as a gzip BLOB."""

    compression = "gzip"

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def put(self, kind: str, data: bytes) -> int:
        """Insert/deduplicate *data* and return its stable payload ID."""
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("payload kind must be a non-empty string")
        if not isinstance(data, bytes):
            raise TypeError("payload data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        row = self.connection.execute(
            "SELECT payload_id FROM payloads WHERE sha256 = ?", (digest,)
        ).fetchone()
        if row is not None:
            return int(row[0])
        compressed = gzip.compress(data, mtime=0)
        cursor = self.connection.execute(
            """
            INSERT INTO payloads(
                sha256, kind, compression, uncompressed_bytes,
                stored_bytes, data, created_at
            ) VALUES (?, ?, 'gzip', ?, ?, ?, strftime('%s', 'now'))
            """,
            (digest, kind, len(data), len(compressed), compressed),
        )
        self.connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a payload ID")
        return int(cursor.lastrowid)

    def put_text(self, kind: str, text: str) -> int:
        """UTF-8 encode and store a text payload."""
        if not isinstance(text, str):
            raise TypeError("text payload must be str")
        return self.put(kind, text.encode("utf-8"))

    def get(self, payload_id: int) -> bytes:
        """Decode and verify one payload by ID."""
        row = self.connection.execute(
            """
            SELECT sha256, compression, uncompressed_bytes, stored_bytes, data
            FROM payloads WHERE payload_id = ?
            """,
            (payload_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown payload ID: {payload_id}")
        if row[1] != self.compression:
            raise PayloadIntegrityError(f"unsupported payload compression: {row[1]!r}")
        compressed = bytes(row[4])
        if len(compressed) != row[3]:
            raise PayloadIntegrityError(f"stored byte count mismatch for payload {payload_id}")
        try:
            data = gzip.decompress(compressed)
        except (OSError, EOFError) as exc:
            raise PayloadIntegrityError(
                f"could not decompress payload {payload_id}: {exc}"
            ) from exc
        if len(data) != row[2]:
            raise PayloadIntegrityError(f"uncompressed byte count mismatch for payload {payload_id}")
        if hashlib.sha256(data).hexdigest() != row[0]:
            raise PayloadIntegrityError(f"SHA-256 mismatch for payload {payload_id}")
        return data

    def get_text(self, payload_id: int) -> str:
        """Decode one UTF-8 text payload after integrity verification."""
        try:
            return self.get(payload_id).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PayloadIntegrityError(
                f"payload {payload_id} is not valid UTF-8 text"
            ) from exc

    def metadata(self, payload_id: int) -> dict[str, Any]:
        """Return compact payload metadata without decompressing the BLOB."""
        row = self.connection.execute(
            """
            SELECT payload_id, sha256, kind, compression,
                   uncompressed_bytes, stored_bytes, created_at
            FROM payloads WHERE payload_id = ?
            """,
            (payload_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown payload ID: {payload_id}")
        return dict(row)

    def count(self) -> int:
        """Return the number of canonical payload rows."""        return int(self.connection.execute("SELECT count(*) FROM payloads").fetchone()[0])
