"""Structured records exchanged between runtime execution and storage backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunContext:
    run_id: str
    revision_id: int
    runner_mode: str = "http"
    session_seed: int | None = None


@dataclass(frozen=True)
class TargetRecord:
    logical_name: str
    runner: str
    source: str
    api_model: str
    target_signature: str
    is_agent: bool = False
    system_prompt: str | None = None
    target_config: dict[str, Any] | None = None
    order_index: int | None = None


@dataclass(frozen=True)
class PluginRecord:
    plugin_id: str
    plugin_version: str
    name: str
    max_score: float
    supports_streaming: bool
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkAttemptRecord:
    attempt_number: int
    prompt: str = ""
    content: str = ""
    thinking: str = ""
    max_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None
    tps: float | None = None
    finish_reason: str | None = None
    response_nature: str | None = None
    retry_reason: str | None = None
    prompt_altered: str | None = None
    truncated: bool | None = None
    truncated_due_to_time: bool | None = None
    failure_cause: str | None = None
    stream_ok: bool | None = None
    repeating: bool | None = None
    empty_reason: str | None = None
    error: str | None = None
    score: float | None = None
    rubric: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class JudgeAttemptRecord:
    judge_model: str
    contract_id: str
    attempt_number: int
    request: str | None = None
    raw_response: str | None = None
    max_tokens: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class JudgeVoteRecord:
    score: float | None = None
    confidence: str | None = None
    rationale: str | None = None
    criteria: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    usable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.__dict__.items()
        }
