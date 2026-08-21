"""Append-only gzip concatenated-member logs.

Each completed record is an independent gzip member. Appending therefore never
reads or rewrites existing compressed bytes; recovery can discard only an
incomplete final member left by a crash.
"""
from __future__ import annotations

import gzip
import os
import re
import threading
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LogRecovery:
    """Result of scanning a concatenated gzip log."""

    complete_members: int
    valid_bytes: int
    total_bytes: int
    truncated_tail: bool
    invalid_tail: bool
    discarded_bytes: int


@dataclass(frozen=True)
class _ScanResult:
    recovery: LogRecovery
    members: tuple[tuple[int, int], ...]


_LOG_LOCKS: dict[str, threading.RLock] = {}
_LOG_LOCKS_GUARD = threading.Lock()
_REDACTION_RE = re.compile(
    r"(?im)^(\s*(?:authorization|api[-_]?key|cookie|set-cookie|password|secret)\s*[:=]\s*).*$"
)
_JSON_REDACTION_RE = re.compile(
    r'(?i)("(?:authorization|api[-_]?key|cookie|set-cookie|password|secret)"\s*:\s*")[^"]*(")'
)
_COMMAND_REDACTION_RE = re.compile(
    r'''(?i)((?:authorization|api[-_]?key|cookie|set-cookie|password|secret)\s*:\s*)[^\s'"}]+'''
)


def _path_lock(path: str) -> threading.RLock:
    key = os.path.abspath(path)
    with _LOG_LOCKS_GUARD:
        return _LOG_LOCKS.setdefault(key, threading.RLock())


def redact_log_text(text: str) -> tuple[str, bool]:
    """Redact credential-like headers and JSON fields before compression."""
    redacted, line_count = _REDACTION_RE.subn(r"\1[REDACTED]", text)
    redacted, json_count = _JSON_REDACTION_RE.subn(r"\1[REDACTED]\2", redacted)
    redacted, command_count = _COMMAND_REDACTION_RE.subn(r"\1[REDACTED]", redacted)
    return redacted, bool(line_count or json_count or command_count)


def _normalise_data(data: str | bytes, redact: bool) -> tuple[bytes, bool]:
    if isinstance(data, str):
        text = data
        changed = False
        if redact:
            text, changed = redact_log_text(text)
        return text.encode("utf-8"), changed
    if not isinstance(data, bytes):
        raise TypeError("log data must be str or bytes")
    if not redact:
        return data, False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    text, changed = redact_log_text(text)
    return text.encode("utf-8"), changed


def _scan_log(path: str, chunk_size: int = 64 * 1024) -> _ScanResult:
    """Scan complete gzip members without loading the whole log."""
    total_bytes = 0
    member_start = 0
    current_offset = 0
    members: list[tuple[int, int]] = []
    decoder: Any = None
    pending = b""
    truncated_tail = False
    invalid_tail = False

    with open(path, "rb") as reader:
        eof = False
        while True:
            if not pending and not eof:
                chunk = reader.read(chunk_size)
                if chunk:
                    total_bytes += len(chunk)
                    pending = chunk
                else:
                    eof = True
            if decoder is None:
                if not pending:
                    if eof:
                        break
                    continue
                if not pending.startswith(b"\x1f\x8b"):
                    invalid_tail = True
                    if not eof:
                        total_bytes += len(reader.read())
                    break
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                member_start = current_offset

            if pending:
                try:
                    decoder.decompress(pending)
                except zlib.error:
                    invalid_tail = True
                    if not eof:
                        total_bytes += len(reader.read())
                    break
                if decoder.eof:
                    unused = decoder.unused_data
                    consumed = len(pending) - len(unused)
                    if consumed <= 0:
                        invalid_tail = True
                        break
                    current_offset += consumed
                    members.append((member_start, current_offset))
                    pending = unused
                    decoder = None
                    member_start = current_offset
                    continue
                current_offset += len(pending)
                pending = b""
                continue

            if eof:
                if decoder is not None and not decoder.eof:
                    truncated_tail = True
                break

    if decoder is not None and not decoder.eof and not invalid_tail:
        truncated_tail = True
    total_bytes = max(total_bytes, current_offset)
    valid_bytes = current_offset if not truncated_tail and not invalid_tail else (
        members[-1][1] if members else 0
    )
    discarded = max(0, total_bytes - valid_bytes) if (truncated_tail or invalid_tail) else 0
    return _ScanResult(
        recovery=LogRecovery(
            complete_members=len(members),
            valid_bytes=valid_bytes,
            total_bytes=total_bytes,
            truncated_tail=truncated_tail,
            invalid_tail=invalid_tail,
            discarded_bytes=discarded,
        ),
        members=tuple(members),
    )


def recover_log(path: str, *, repair: bool = False) -> LogRecovery:
    """Scan a log and optionally repair only an incomplete final member."""
    if not os.path.exists(path):
        return LogRecovery(0, 0, 0, False, False, 0)
    scanned = _scan_log(path)
    recovery = scanned.recovery
    if repair and recovery.truncated_tail and not recovery.invalid_tail:
        with open(path, "r+b") as handle:
            handle.truncate(recovery.valid_bytes)
        return LogRecovery(
            recovery.complete_members,
            recovery.valid_bytes,
            recovery.valid_bytes,
            False,
            False,
            recovery.discarded_bytes,
        )
    return recovery


def iter_log_members(path: str, *, tolerate_truncated_tail: bool = True):
    """Yield complete member contents, optionally tolerating a damaged tail."""
    scanned = _scan_log(path)
    recovery = scanned.recovery
    if recovery.invalid_tail:
        raise OSError(f"invalid gzip log tail in {path}")
    if recovery.truncated_tail and not tolerate_truncated_tail:
        raise EOFError(f"truncated final gzip member in {path}")
    with open(path, "rb") as handle:
        for start, end in scanned.members:
            handle.seek(start)
            member = handle.read(end - start)
            yield gzip.decompress(member)


class AppendOnlyGzipLog:
    """Buffered append-only gzip member writer."""

    def __init__(
        self,
        path: str,
        *,
        member_target_bytes: int = 128 * 1024,
        flush_interval: float = 0.5,
        sync_policy: str = "batch",
        recover_tail: bool = True,
        redact: bool = True,
    ) -> None:
        if member_target_bytes <= 0:
            raise ValueError("member_target_bytes must be positive")
        if flush_interval < 0:
            raise ValueError("flush_interval must not be negative")
        if sync_policy not in {"none", "batch", "final"}:
            raise ValueError("sync_policy must be none, batch, or final")
        self.path = path
        self.member_target_bytes = member_target_bytes
        self.flush_interval = flush_interval
        self.sync_policy = sync_policy
        self.redact = redact
        self._lock = _path_lock(path)
        self._buffer = bytearray()
        self._closed = False
        self._last_flush = 0.0
        self.redaction_occurred = False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock:
            if recover_tail:
                recover_log(path, repair=True)
            self._last_flush = 0.0

    def append(self, data: str | bytes) -> None:
        """Buffer data and finish members when the target size is reached."""
        payload, redacted = _normalise_data(data, self.redact)
        with self._lock:
            self._ensure_open()
            self.redaction_occurred |= redacted
            self._buffer.extend(payload)
            if len(self._buffer) >= self.member_target_bytes:
                self._flush_locked(sync=self.sync_policy == "batch")

    def append_record(self, chunks: Iterable[str | bytes]) -> None:
        """Write one logical record as one gzip member."""
        with self._lock:
            self._ensure_open()
            if self._buffer:
                self._flush_locked(sync=self.sync_policy == "batch")
            with open(self.path, "ab") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                    for chunk in chunks:
                        payload, redacted = _normalise_data(chunk, self.redact)
                        self.redaction_occurred |= redacted
                        compressed.write(payload)
                raw.flush()
                if self.sync_policy in {"batch", "final"}:
                    os.fsync(raw.fileno())
            self._last_flush = 0.0

    def flush(self, *, sync: bool = False) -> None:
        """Finish the buffered member and optionally fsync it."""
        with self._lock:
            self._ensure_open()
            self._flush_locked(sync=sync or self.sync_policy == "batch")

    def recover(self) -> LogRecovery:
        """Scan the current file without modifying it."""
        with self._lock:
            return recover_log(self.path, repair=False)

    def close(self, *, sync: bool = True) -> None:
        """Flush the final member and release the writer."""
        with self._lock:
            if self._closed:
                return
            self._flush_locked(sync=sync or self.sync_policy == "final")
            self._closed = True

    def _flush_locked(self, *, sync: bool) -> None:
        if not self._buffer:
            return
        with open(self.path, "ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                compressed.write(bytes(self._buffer))
            raw.flush()
            if sync:
                os.fsync(raw.fileno())
        self._buffer.clear()
        self._last_flush = 0.0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("gzip log is closed")
