"""Typed request models shared by the transport execution boundary.

The request variants are intentionally separate from the legacy ``TransportRequest``
adapter in :mod:`benchmark.transport`. New production code should construct one
of these variants so transport-specific settings cannot be accidentally passed
to another runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .observer import TaskObserver
from .transport_options import HTTPTransportOptions, OpenCodeTransportOptions, PiTransportOptions


@dataclass(frozen=True)
class RequestIdentityFields:
    """Stable request identity used by logs, state, and diagnostics."""

    run_id: str = "unknown-run"
    revision_id: str | int = "unknown-revision"
    target: str = "unknown-target"
    plugin: str = "unknown-plugin"
    attempt: int = 1


@dataclass(frozen=True)
class GenerationFields:
    """Prompt and model-generation settings common to every runner."""

    prompt: str
    max_tokens: int
    source_config: dict[str, Any]
    api_model: str
    source: str
    timeout: float
    temperature: float | None = 0.0
    reasoning: bool = False
    system_prompt: str | None = None
    drop_params: list[Any] | None = None
    session_seed: int = 0
    log_path: str | None = None
    log_label: str = ""
    pid: str = ""
    stop_event: Any = None
    observer: TaskObserver | None = None
    debug_logs: bool = False
    prompt_altered: str = "none"
    identity: RequestIdentityFields | None = None


@dataclass(frozen=True)
class HTTPRequest:
    """HTTP transport request variant."""

    common: GenerationFields
    options: HTTPTransportOptions = field(default_factory=HTTPTransportOptions)
    kind: Literal["http"] = "http"


@dataclass(frozen=True)
class OpenCodeRequest:
    """OpenCode subprocess request variant."""

    common: GenerationFields
    options: OpenCodeTransportOptions = field(default_factory=OpenCodeTransportOptions)
    kind: Literal["opencode"] = "opencode"


@dataclass(frozen=True)
class PiRequest:
    """Pi worker subprocess request variant."""

    common: GenerationFields
    options: PiTransportOptions = field(default_factory=PiTransportOptions)
    kind: Literal["pi"] = "pi"


TransportRequest = HTTPRequest | OpenCodeRequest | PiRequest


def request_identity(fields: GenerationFields, attempt: int | None = None) -> RequestIdentityFields:
    """Return a concrete identity for a request, updating its attempt number."""
    identity = fields.identity or RequestIdentityFields(
        target=fields.api_model,
        plugin=fields.pid or "unknown-plugin",
    )
    return RequestIdentityFields(
        run_id=identity.run_id,
        revision_id=identity.revision_id,
        target=identity.target,
        plugin=identity.plugin,
        attempt=identity.attempt if attempt is None else attempt,
    )
