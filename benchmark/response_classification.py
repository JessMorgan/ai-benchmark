"""Typed response classification helpers shared by benchmark and judge paths."""
from __future__ import annotations

from typing import Any


def count_tokens(text: str) -> float:
    """Estimate tokens using the benchmark's stable four-characters-per-token rule."""
    return max(0, len(text) / 4)


def classify_empty_reason(
    text: str,
    think_text: str = "",
    finish_reason: str | None = None,
    error: str | None = None,
) -> str | None:
    """Classify why a response has no final content."""
    if text and text.strip():
        return None
    if error:
        return "error"
    if think_text and finish_reason == "length":
        return "thinking-truncation"
    if think_text:
        return "thinking-only"
    if finish_reason == "length":
        return "max-tokens"
    return "empty"


def response_nature(
    *,
    text: str,
    error: str | None,
    finish_reason: str | None,
    repeating: bool = False,
    cancelled: bool = False,
) -> str:
    """Classify the machine-observable end of a response."""
    lowered = str(error or "").lower()
    if cancelled or "cancelled" in lowered or "canceled" in lowered:
        return "cancelled"
    if any(marker in lowered for marker in ("timeout", "timed out", "readtimeout")):
        return "timeout"
    if repeating or "repetition" in lowered or "repeated" in lowered:
        return "repetition_abort"
    if finish_reason == "length" or "token limit" in lowered or "budget exceeded" in lowered:
        return "token_limit"
    if error:
        return "transport_error"
    if not text or not text.strip():
        return "empty"
    return "completed"


def response_has_content(value: Any) -> bool:
    """Return whether a dynamically loaded response contains final text."""
    return isinstance(value, str) and bool(value.strip())
