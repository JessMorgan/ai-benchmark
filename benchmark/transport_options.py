"""Transport-specific option objects used by the shared execution engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HTTPTransportOptions:
    """Options consumed only by the OpenAI-compatible HTTP transport."""

    supports_streaming: bool = True
    max_content_tokens: int | None = None
    max_thinking_tokens: int | None = None
    repetition_guard: int | bool | None = None
    request_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class OpenCodeTransportOptions:
    """Options consumed only by the isolated OpenCode subprocess."""

    config_path: str | None = None
    model: str | None = None
    agent: str | None = None
    binary: str | None = None
    output_dir: str | None = None
    no_output_grace: float | None = None
    target_key: str | None = None
    plugin_id: str | None = None


@dataclass(frozen=True)
class PiTransportOptions:
    """Options consumed only by the isolated Pi worker subprocess."""

    node: str | None = None
    worker: str | None = None
    config: dict[str, Any] | None = None
    target_key: str | None = None
    plugin_id: str | None = None


@dataclass(frozen=True)
class TransportOptions:
    """Runner-specific options grouped behind one transport boundary."""

    http: HTTPTransportOptions = field(default_factory=HTTPTransportOptions)
    opencode: OpenCodeTransportOptions = field(default_factory=OpenCodeTransportOptions)
    pi: PiTransportOptions = field(default_factory=PiTransportOptions)
