"""Core benchmark logic shared by the CLI and tests."""
import contextlib
import copy
import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, cast

from jsonschema import Draft202012Validator

from .http import (  # noqa: F401
    close_active_requests,
    fetch_models_v1,
    nonstream_request,
    stream_request,
)
from .judging import (
    JudgeResult,
    build_judge_prompt,
    judge_contract_id,
    judge_instructions_version,
    judge_sidecar_path,
    parse_judge_response,
    publish_judge_sidecars,
)
from .observer import TaskObserver
from .opencode import (
    run_process,
)
from .outputs import (  # noqa: F401
    _save_outputs,
    gen_csv,
    gen_html,
    gen_markdown,
    gen_pdf,
    sanitize_filename,
)
from .plugin import (
    SCORE_SCHEMA,
    PluginTaskResult,
    normalize_score,
)
from .request_models import (
    GenerationFields,
    HTTPRequest,
)
from .response_classification import (  # noqa: F401 - compatibility exports
    classify_empty_reason,
    count_tokens,
    response_nature,
)
from .state import BenchmarkState  # noqa: F401
from .transport import (
    JUDGE_RETRY_POLICY,
    TransportResult,
    _split_token_budget,
    _thinking_consumed_budget,
    execute_task,
)
from .transport_options import (
    HTTPTransportOptions,
)
from .task_execution import TaskExecutionDependencies
from .task_execution import _run_plugin_task as _execute_plugin_task

PRELOAD_PROMPT = "Reply with the single word OK."
PRELOAD_DEFAULT_TIMEOUT = 300
# Token budget for the warm-up probe. Must be generous enough that a
# thinking/reasoning model (deepseek-r1, qwen3.x, gemma4, ...) can emit at
# least one content token after its reasoning preamble -- the old 16-token
# budget was fully consumed by ``reasoning_content`` for 68% of the probes
# in a prior benchmark run, producing ``content=""`` + ``finish_reason="length"``
# responses that were wrongly classified as ``empty preload response`` and
# skipped the model for the whole benchmark. 256 tokens is comfortably past
# typical reasoning preambles while keeping the probe cheap.
PRELOAD_MAX_TOKENS = 256
JUDGE_PROMPT_VERSION = "judge-v8"
JUDGE_DEFAULT_MAX_TOKENS = 4096
# State persistence is throttled across the whole run: completed judge votes
# and completed benchmark tasks accumulate in memory, and the full state
# snapshot flushes at most every ``FLUSH_INTERVAL_SECONDS`` seconds or
# ``FLUSH_MAX_VOTES`` changes, whichever comes first. A final flush on
# drain/shutdown always persists the tail, so a crash loses at most one
# interval of changes (re-runnable on resume).
FLUSH_INTERVAL_SECONDS = 120.0
FLUSH_MAX_VOTES = 50
# Maximum time the main thread waits for the background state flusher during
# shutdown before reporting a failure and attempting a synchronous final save.
PERSISTENCE_SHUTDOWN_TIMEOUT = 10.0


def _thinking_budget_retry_instruction(token_budget: Any | int, fallback: int = JUDGE_DEFAULT_MAX_TOKENS) -> str:
    """Return retry-only guidance reserving half the budget for the answer."""
    reported, thinking_budget, answer_budget = _split_token_budget(token_budget, fallback)
    return (
        "\n\nRETRY GUIDANCE: On this retry you MUST keep internal thinking or "
        f"reasoning below {thinking_budget} tokens and the entire response below "
        f"{reported} total tokens ({answer_budget} tokens are reserved for the "
        "final answer). Exceeding either limit is considered a failure."
    )


def _judge_system_prompt(total_budget: Any | int) -> str:
    """Return a system prompt that sets token budgets before the judge prompt.

    Thinking models allocate most of their generation budget to
    ``reasoning_content`` before emitting any final answer.  Telling the
    model up-front how much thinking and how much answer content is
    expected gives it a chance to self-regulate, which is more effective
    than appending guidance to the user message after the fact.
    """
    reported, thinking_budget, answer_budget = _split_token_budget(
        total_budget, JUDGE_DEFAULT_MAX_TOKENS,
    )
    return (
        "You are a benchmark semantic evaluator. Your response must be a single "
        "JSON object matching the requested schema — nothing else.\n\n"
        "TOKEN BUDGET:\n"
        f"Total generation budget: {reported} tokens.\n"
        f"Maximum internal thinking/reasoning: {thinking_budget} tokens.\n"
        f"Maximum final answer content: {answer_budget} tokens.\n\n"
        "Spend at most half of the budget on internal reasoning. The remaining "
        "half must be reserved for the final JSON answer. Do not exceed these "
        "limits — a response whose thinking consumes the full budget is "
        "considered a failure because no answer content can follow."
    )


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "rationale": {
            "type": "string",
        },
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "criterion": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["met", "partial", "not_met", "not_applicable"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["id", "criterion", "status", "evidence"],
            },
        },
    },
    "required": ["score", "confidence", "rationale", "criteria"],
}
JUDGE_DEFAULT_REQUEST_PARAMS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "benchmark_judge_result",
            "strict": True,
            "schema": JUDGE_RESPONSE_SCHEMA,
        },
    },
}


SCHEMA_SENTINEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentinel": {
            "type": "string",
            "enum": ["schema-enforced"],
        },
    },
    "required": ["sentinel"],
}
SCHEMA_SENTINEL_PROMPT = """This is a structured-output compatibility probe.

For this probe, deliberately ignore the requested response schema and return
exactly this JSON object, with no markdown or explanation:
{"sentinel":"schema-not-enforced"}
"""


@dataclass(frozen=True)
class PreloadResult:
    """Outcome of a model warm-up probe."""

    success: bool
    elapsed: float
    error: str | None = None
    text: str = ""


def resolve_preload_timeout(source_config: dict[str, Any], source: str, default: int = PRELOAD_DEFAULT_TIMEOUT) -> int:
    """Return a positive per-source preload timeout, or the default."""
    cfg = source_config.get(source) or {}
    value = cfg.get("preload_timeout", default) if isinstance(cfg, dict) else default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def resolve_model_thread_limit(source_config: dict[str, Any], source: str, top_level: int = 1) -> int:
    """Return the validated positive model concurrency for ``source``.

    Unlike ``plugin_thread_limit``, zero is never an unlimited value here:
    source-level model concurrency is an explicit resource-control boundary.
    """
    cfg = source_config.get(source)
    value = cfg.get("model_thread_limit", top_level) if isinstance(cfg, dict) else top_level
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Invalid model_thread_limit for source '{source}': "
            f"expected a positive integer, got {value!r}"
        )
    return value


# Per-response token budgets for the live stream watchdog (``http._StreamGuards``).
# ``reasoning_content`` is capped separately from final content so a thinking
# loop cannot burn the whole ``max_tokens`` allowance before a single content
# token lands, and a content loop cannot drag on for the full 30-60 min a
# 65k-token budget allows on local hardware. The defaults match the
# operator's chosen split: up to 32k thinking tokens, then up to 16k content
# tokens. Per-source ``max_thinking_tokens`` / ``max_content_tokens`` config
# keys override; ``repetition_guard`` (default on) aborts a stream whose
# content or thinking repeats itself.
DEFAULT_MAX_THINKING_TOKENS = 32768
DEFAULT_MAX_CONTENT_TOKENS = 16384


def resolve_stream_guards(source_config: dict[str, Any], source: str) -> tuple[int, int, bool]:
    """Return ``(max_content_tokens, max_thinking_tokens, repetition_guard)``.

    Reads the per-source ``max_content_tokens``, ``max_thinking_tokens`` and
    ``repetition_guard`` keys, falling back to the defaults above. Invalid
    (non-positive / non-int) token values fall back to the default; the
    repetition guard defaults to enabled since its rule (an 80-char block
    repeated 3x) already marks completed responses as ``repeating``.
    """
    cfg = source_config.get(source)
    if not isinstance(cfg, dict):
        return (DEFAULT_MAX_CONTENT_TOKENS, DEFAULT_MAX_THINKING_TOKENS, True)

    def _tokens(key: str, default: int) -> int:
        value = cfg.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return (
        _tokens("max_content_tokens", DEFAULT_MAX_CONTENT_TOKENS),
        _tokens("max_thinking_tokens", DEFAULT_MAX_THINKING_TOKENS),
        bool(cfg.get("repetition_guard", True)),
    )


def preload_model(source_config: dict[str, Any], source: str, api_model: str, timeout: float,
                  session_seed: int = 0, stop_event: threading.Event | None = None, drop_params: list[str] | None = None,
                  log_path: str | None = None) -> PreloadResult:
    """Warm one model with a small, non-scoring HTTP request.

    The probe deliberately disables HTTP 429 retries: a source that cannot
    answer the warm-up request should be reported immediately rather than
    occupying its source worker during backoff. The caller owns the probe's
    timing, so this elapsed time never enters a benchmark result's timers.

    Success is any non-error response that produced *something* -- content
    OR reasoning (``think_text``). Thinking models that burn the entire
    ``PRELOAD_MAX_TOKENS`` budget on ``reasoning_content`` and return empty
    content with ``finish_reason="length"`` still prove the model is warm
    and are treated as preloaded; only a completely empty response (no
    content, no reasoning) is ``empty preload response``.
    """
    started = time.time()
    cfg = source_config.get(source)
    if not isinstance(cfg, dict):
        return PreloadResult(False, round(time.time() - started, 1),
                             f"Unknown source '{source}' — not in SOURCE_CONFIG")

    probe_sources = dict(source_config)
    probe_cfg = dict(cfg)
    probe_cfg["max_429_retries"] = 0
    probe_sources[source] = probe_cfg
    response = nonstream_request(
        probe_sources, timeout, api_model, source, PRELOAD_PROMPT,
        PRELOAD_MAX_TOKENS,
        log_path=log_path,
        log_label=f"Model preload ({source}/{api_model})",
        session_seed=session_seed,
        temperature=0.0,
        drop_params=drop_params or [],
        stop_event=stop_event,
    )
    error = response.error
    # A probe is only a failure when the model produced NOTHING -- no
    # content AND no reasoning. A response whose probe budget was consumed
    # by ``reasoning_content`` (empty ``content``, non-empty
    # ``think_text``, ``finish_reason="length"``) proves the model is warm
    # and responding; treating it as ``empty preload response`` was a false
    # negative that skipped thinking models for the entire benchmark (see
    # an earlier run showed many probes were thinking-truncated). A
    # transport error still fails the probe regardless of ``think_text``
    # (the ``not error`` guard above this check).
    if not error and not response.text.strip() and not response.think_text.strip():
        error = "empty preload response"
    return PreloadResult(
        success=not error,
        elapsed=round(time.time() - started, 1),
        error=error,
        text=response.text,
    )


def _is_exhausted_429(error: Any) -> bool:
    """Return whether an HTTP error is an exhausted rate-limit response."""
    return isinstance(error, str) and error.lstrip().startswith("HTTP 429:")


def _schema_probe_error_status(error: Any) -> str:
    """Classify a sentinel request failure without conflating it with a model score."""
    lowered = str(error or "").lower()
    schema_words = ("schema", "grammar", "response_format", "structured output", "format")
    if any(word in lowered for word in schema_words) and any(
        marker in lowered for marker in ("http 400", "http 422", "bad request", "failed to parse grammar")
    ):
        return "schema_rejected"
    return "schema_transport_error"


def _is_schema_grammar_error(error: Any) -> bool:
    """Return whether a provider failed while compiling a response grammar."""
    lowered = str(error or "").lower()
    compiler_markers = (
        "failed to initialize samplers",
        "grammar sampler",
        "failed to parse grammar",
        "error initializing grammar",
    )
    return any(marker in lowered for marker in compiler_markers)


def _json_object_fallback_params(request_params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Replace a JSON-schema response format with provider JSON mode."""
    if not isinstance(request_params, dict):
        return None
    response_format = request_params.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    fallback = copy.deepcopy(request_params)
    fallback["response_format"] = {"type": "json_object"}
    return fallback


def run_schema_sentinel(source_config: dict[str, Any], source: str, api_model: str, *, timeout: float = 120,
                        session_seed: int = 0, drop_params: list[str] | None = None) -> dict[str, Any]:
    """Probe whether a source accepts and appears to enforce JSON schemas.

    The prompt requests a deliberately schema-invalid value while the schema
    permits only ``schema-enforced``. A valid response with that permitted
    value is evidence of enforcement, but not a cryptographic proof: a model
    could still have followed an unshown instruction. This probe is therefore
    diagnostic and never contributes to benchmark scores.
    """
    cfg = source_config.get(source)
    base = {
        "source": source,
        "model": api_model,
        "schema": copy.deepcopy(SCHEMA_SENTINEL_SCHEMA),
        "response_schema_valid": False,
        "schema_enforcement_verified": False,
    }
    if not isinstance(cfg, dict):
        return {**base, "status": "schema_transport_error", "error": f"Unknown source '{source}'"}
    if cfg.get("api_protocol") in {"1min", "chatplayground"}:
        return {
            **base,
            "status": "schema_not_supported_by_source",
            "error": f"Source protocol {cfg.get('api_protocol')!r} does not use OpenAI response_format",
        }
    request_params = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "benchmark_schema_sentinel",
                "strict": True,
                "schema": copy.deepcopy(SCHEMA_SENTINEL_SCHEMA),
            },
        },
    }
    started = time.time()
    response = nonstream_request(
        source_config, timeout, api_model, source, SCHEMA_SENTINEL_PROMPT,
        128, session_seed=session_seed, temperature=0.0,
        drop_params=drop_params or [], request_params=request_params,
        pid="schema-sentinel",
    )
    base["elapsed"] = round(time.time() - started, 1)
    base["finish_reason"] = response.finish_reason
    base["response"] = (response.text or "")[:2000]
    if response.error:
        return {**base, "status": _schema_probe_error_status(response.error), "error": response.error}
    try:
        value = json.loads(response.text.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        return {**base, "status": "schema_accepted_invalid", "error": f"invalid JSON: {exc}"}
    errors = sorted(
        Draft202012Validator(SCHEMA_SENTINEL_SCHEMA).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        return {
            **base,
            "status": "schema_accepted_invalid",
            "error": "; ".join(error.message for error in errors),
            "schema_errors": [error.message for error in errors],
        }
    base["response_schema_valid"] = True
    enforced = value.get("sentinel") == "schema-enforced"
    base["schema_enforcement_verified"] = enforced
    return {
        **base,
        "status": "schema_likely_enforced" if enforced else "schema_accepted_valid",
        "error": None,
    }


def resolve_judge_request_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return provider-specific request parameters for semantic judges.

    The defaults preserve each model's native behavior and request a JSON
    object. A ``judge`` config block may override or extend these values
    through ``request_params``. Nested dictionaries are merged so explicit
    provider-specific options can be combined safely.
    """
    params = copy.deepcopy(JUDGE_DEFAULT_REQUEST_PARAMS)
    judge_cfg = cfg.get("judge") if isinstance(cfg, dict) else None
    configured = judge_cfg.get("request_params") if isinstance(judge_cfg, dict) else None
    if not isinstance(configured, dict):
        return params
    for key, value in configured.items():
        if isinstance(params.get(key), dict) and isinstance(value, dict):
            params[key].update(copy.deepcopy(value))
        else:
            params[key] = copy.deepcopy(value)
    return params


def _schema_request_metadata(plugin: Any, request_params: dict[str, Any] | None = None, *, response_schema_valid: bool | None = None,
                             error: str | None = None, request_applied: bool = True, schema_fallback_used: bool = False,
                             schema_fallback_error: str | None = None) -> dict[str, Any]:
    """Return non-scoring metadata for a plugin's structured-output contract.

    A completed valid response does not prove that the provider enforced the
    schema, so ``schema_enforcement_verified`` remains false unless the
    separate sentinel probe establishes it. Request failures are classified
    separately from response/schema failures so semantic scores are not
    mistaken for transport compatibility.
    """
    get_schema = getattr(plugin, "get_response_schema", None)
    declared_schema = get_schema() if callable(get_schema) else None
    response_format = request_params.get("response_format") if isinstance(request_params, dict) else None
    requested = bool(declared_schema or (
        isinstance(response_format, dict)
        and response_format.get("type") in {"json", "json_schema"}
    ))
    if not requested:
        return {
            "schema_requested": False,
            "schema_request_status": "schema_not_requested",
            "response_schema_valid": None,
            "schema_enforcement_verified": None,
            "schema_fallback_used": False,
            "schema_fallback_error": None,
        }
    if not request_applied:
        return {
            "schema_requested": True,
            "schema_request_status": "schema_not_applied_by_runner",
            "response_schema_valid": None,
            "schema_enforcement_verified": False,
            "schema_fallback_used": False,
            "schema_fallback_error": None,
        }
    if schema_fallback_used:
        if error:
            status = "schema_fallback_json_object_failed"
        elif response_schema_valid is True:
            status = "schema_fallback_json_object_valid"
        elif response_schema_valid is False:
            status = "schema_fallback_json_object_invalid"
        else:
            status = "schema_fallback_json_object_unknown"
    elif error:
        lowered = str(error).lower()
        schema_words = ("schema", "grammar", "response_format", "structured output", "format")
        if ("http 400" in lowered or "http 422" in lowered) and any(word in lowered for word in schema_words):
            status = "schema_rejected"
        elif "invalid completion response" in lowered or "empty response body" in lowered:
            status = "schema_accepted_invalid"
        else:
            status = "schema_transport_error"
    elif response_schema_valid is True:
        status = "schema_accepted_valid"
    elif response_schema_valid is False:
        status = "schema_accepted_invalid"
    else:
        status = "schema_accepted_unknown"
    return {
        "schema_requested": True,
        "schema_request_status": status,
        "response_schema_valid": response_schema_valid,
        "schema_enforcement_verified": False,
        "schema_fallback_used": schema_fallback_used,
        "schema_fallback_error": schema_fallback_error,
    }


def _judge_response_diagnostics(response: Any, request_params: dict[str, Any] | None, max_tokens: int) -> dict[str, Any]:
    """Summarize request settings and response evidence for judge debugging.

    OpenAI-compatible providers do not consistently expose whether a
    ``chat_template_kwargs`` option was accepted. Recording the exact
    requested values alongside usage, finish reason, and the provider's
    reasoning-token count (when available) lets a completed run distinguish
    an honored thinking cap from a provider that ignored it. The character
    estimate is deliberately labeled as such because it is only a fallback
    when usage details are unavailable.
    """
    params = copy.deepcopy(request_params) if isinstance(request_params, dict) else {}
    chat_template = params.get("chat_template_kwargs")
    requested_budget = (
        chat_template.get("thinking_token_budget")
        if isinstance(chat_template, dict)
        else None
    )
    if isinstance(requested_budget, bool) or not isinstance(requested_budget, (int, float)):
        requested_budget = None

    usage = getattr(response, "usage", {})
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("completion_token_details")
    if not isinstance(details, dict):
        details = {}
    reported_reasoning = None
    reasoning_source = None
    for container, source in ((usage, "usage"), (details, "usage.details")):
        for key in ("reasoning_tokens", "thinking_tokens"):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                reported_reasoning = value
                reasoning_source = f"{source}.{key}"
                break
        if reported_reasoning is not None:
            break

    think_text = getattr(response, "think_text", "") or ""
    if not isinstance(think_text, str):
        think_text = ""
    estimated_reasoning = int(count_tokens(think_text)) if think_text else None
    observed_reasoning = reported_reasoning
    if observed_reasoning is None:
        observed_reasoning = estimated_reasoning
        if observed_reasoning is not None:
            reasoning_source = "estimated_from_reasoning_content"

    budget_honored = None
    if requested_budget is not None and observed_reasoning is not None:
        budget_honored = observed_reasoning <= requested_budget
    return {
        "request_max_tokens": max_tokens,
        "request_params": params,
        "requested_thinking_token_budget": requested_budget,
        "response_finish_reason": (
            getattr(response, "finish_reason", None)
            if isinstance(getattr(response, "finish_reason", None), str)
            else None
        ),
        "response_usage": usage,
        "response_reasoning_tokens": observed_reasoning,
        "response_reasoning_tokens_source": reasoning_source,
        "thinking_budget_honored": budget_honored,
    }


def judge_response(source_config: dict[str, Any], judge_source: str, judge_api_model: str, sidecar: str,
                   *, timeout: float, max_tokens: int | None = None, temperature: float = 0.0,
                   drop_params: list[str] | None = None, request_params: dict[str, Any] | None = None, stop_event: threading.Event | None = None, log_path: str | None = None,
                   plugin: Any = None, progress_callback: Callable[[str, str], None] | None = None,
                   attempt_callback: Callable[[int], None] | None = None) -> JudgeResult:
    """Run one streaming judge request, retrying once when its JSON is invalid."""
    with open(sidecar, encoding="utf-8") as handle:
        item = json.load(handle)
    if plugin is None:
        # Fall back to a name/max_score stub when no plugin instance is
        # supplied (e.g. resume-only judging); the stub has no judge
        # sanitizer, so its prompt is built from the raw sidecar text.
        plugin = type("JudgePlugin", (), {
            "name": item["plugin_name"], "max_score": item["max_score"],
        })()
    prompt = build_judge_prompt(plugin, item["prompt"], item["response"])
    try:
        budget = int(max_tokens if max_tokens is not None else JUDGE_DEFAULT_MAX_TOKENS)
    except (TypeError, ValueError):
        budget = JUDGE_DEFAULT_MAX_TOKENS
    if budget <= 0:
        budget = JUDGE_DEFAULT_MAX_TOKENS

    def report_progress(content_delta: str = "", thinking_delta: str = "") -> None:
        if progress_callback is None:
            return
        # Progress is observational only; a broken TUI/state observer must
        # never terminate a judge stream.
        with contextlib.suppress(Exception):
            progress_callback(content_delta, thinking_delta)

    # The prompt is built by the current code path, so use the versions that
    # actually govern this request rather than stale metadata in a retained
    # sidecar created by an older benchmark run.
    prompt_version = JUDGE_PROMPT_VERSION
    instructions_version = judge_instructions_version(plugin)
    judge_observer = TaskObserver(
        pid=f"judge:{item['plugin']}",
        on_chunk=report_progress,
        on_think_chunk=lambda delta: report_progress("", delta),
    )
    judge_common = GenerationFields(
        prompt=prompt,
        max_tokens=budget,
        source_config=source_config,
        api_model=judge_api_model,
        source=judge_source,
        timeout=timeout,
        temperature=temperature,
        system_prompt=_judge_system_prompt(budget),
        drop_params=drop_params or [],
        stop_event=stop_event,
        observer=judge_observer,
        pid=f"judge:{item['plugin']}",
        log_path=log_path,
        log_label=(
            f"Judge {item['target']} / {item['plugin']} "
            f"(streaming attempt {{attempt}}, "
            f"prompt_version={prompt_version}, "
            f"judge_instructions_version={instructions_version})"
        ),
    )
    judge_request = HTTPRequest(
        judge_common,
        HTTPTransportOptions(supports_streaming=True, request_params=request_params),
    )

    def json_error_prompt_alterer(result: TransportResult) -> str | None:
        parsed = parse_judge_response(result.text)
        if parsed.error is None:
            return None
        diagnostics = _judge_response_diagnostics(result, request_params, budget)
        guidance = (
            _thinking_budget_retry_instruction(budget)
            if _thinking_consumed_budget(diagnostics) else ""
        )
        return (
            "\n\nYour previous response was invalid. Return only the required JSON schema."
            + guidance
        )

    execution = execute_task(
        judge_request,
        retry_policy=JUDGE_RETRY_POLICY,
        base_prompt=prompt,
        json_error_prompt_alterer=json_error_prompt_alterer,
        attempt_callback=attempt_callback,
        stream_request_fn=stream_request,
    )
    if execution.selected is None:
        return JudgeResult(error="cancelled" if stop_event and stop_event.is_set() else "no judge attempt")
    result = execution.selected.result
    diagnostics = _judge_response_diagnostics(result, request_params, budget)
    if result.error:
        # Preserve partial streamed content for cancellation/transport
        # diagnostics, while still treating the attempt as unsuccessful.
        diagnostics["response_json_valid"] = False
        return JudgeResult(
            error=result.error,
            response_text=result.text or None,
            terminal_429=_is_exhausted_429(result.error),
            diagnostics=diagnostics,
        )
    parsed = parse_judge_response(result.text)
    diagnostics["response_json_valid"] = parsed.error is None
    if parsed.error is None:
        return JudgeResult(
            score=parsed.score,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            response_text=result.text,
            diagnostics=diagnostics,
            criteria=parsed.criteria,
        )
    return JudgeResult(
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        error=parsed.error,
        response_text=result.text,
        diagnostics=diagnostics,
        criteria=parsed.criteria,
    )


def _overall_score(result: dict[str, Any], active_plugins: list[Any]) -> int | None:
    """Return the half-up mean of available normalized plugin percentages."""
    scores: list[float] = []
    for plugin in active_plugins:
        value = result.get(f"{plugin.id}_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append(float(value))
    if not scores:
        return None
    return cast(int | None, normalize_score(sum(scores) / len(scores), 100))


def is_repeating(text: str, min_seq: int = 80, repeats: int = 3) -> bool:
    """Detect if text is stuck in a loop."""
    if len(text) < min_seq * repeats:
        return False
    tail = text[-min_seq:]
    return text.count(tail) >= repeats


def _response_reasoning_tokens(response: Any) -> int | None:
    """Prefer provider reasoning usage, falling back to the char/4 estimate."""
    usage = getattr(response, "usage", {})
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details") or usage.get("completion_token_details")
    containers = (usage, details if isinstance(details, dict) else {})
    for container in containers:
        for key in ("reasoning_tokens", "thinking_tokens"):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    thinking = getattr(response, "think_text", "") or ""
    return int(count_tokens(thinking)) if thinking else None


def _source_abbrev(name: str) -> str:
    """Generate a short acronym from a source name using capital letters."""
    tokens = []
    for w in name.split():
        if w.isupper() and 1 < len(w) <= 3:
            tokens.append(w)
        else:
            sub = re.findall(r'[A-Z]?[a-z]+|[A-Z]+', w)
            tokens.extend(sub) if sub else tokens.append(w)
    if not tokens:
        return name[:2].upper()
    ab = ''.join(t[0].upper() for t in tokens if t)
    return ab if len(ab) >= 2 else (name * 2)[:2].upper()


def _unique_source_abbrevs(sources: list[str]) -> dict[str, str]:
    """Return a mapping from source names to short, unique abbreviations."""
    abbrevs = {}
    used = set()
    for src in sources:
        ab = _source_abbrev(src)
        if ab in used:
            for i in range(1, 100):
                candidate = f"{ab}{i}"
                if candidate not in used:
                    ab = candidate
                    break
        abbrevs[src] = ab
        used.add(ab)
    return abbrevs


# ─── Config loading ──────────────────────────────────────────────────────────

# ─── Model execution ─────────────────────────────────────────────────────────




def _run_plugin_task(*args: Any, **kwargs: Any) -> PluginTaskResult:
    """Delegate one plugin execution to the canonical task module."""
    kwargs["dependencies"] = TaskExecutionDependencies(
        stream_request_fn=stream_request,
        nonstream_request_fn=nonstream_request,
        run_process_fn=run_process,
        resolve_stream_guards_fn=resolve_stream_guards,
    )
    return _execute_plugin_task(*args, **kwargs)


def run_model(model_name: str, source: str, state: Any, active_plugins: list[Any], source_config: dict[str, Any], timeout: float,
              max_tokens: int, output_dir: str, session_seed: int = 0, global_cfg: dict[str, Any] | None = None,
              stop_event: threading.Event | None = None, save_responses: bool = False, api_model: str | None = None,
              judge_input_dir: str | None = None, judge_enqueue: Callable[[str, str, str, str], Any] | None = None,
              judge_model: str | None = None, judge_models: list[str] | None = None, judge_prompt_version: str | None = None,
              system_prompt: str | None = None, is_agent: bool = False, runner: str = "http",
              opencode_config_path: str | None = None, opencode_model: str | None = None,
              opencode_agent: str | None = None, opencode_binary: str | None = None,
              pi_node: str | None = None, pi_worker: str | None = None, pi_config: dict[str, Any] | None = None, display_name: str | None = None,
              config_target_name: str | None = None, debug_logs: bool = False) -> dict[str, Any]:
    """Run active plugins for one model or agent through a selected runner."""
    start = time.time()
    target_name = model_name
    display_name = display_name or target_name
    config_target_name = config_target_name or display_name
    api_model = api_model or target_name
    active_judge_contracts = {
        plugin.id: judge_contract_id(plugin)
        for plugin in active_plugins
    } if judge_models or judge_model else {}

    r = {
        "score_schema": SCORE_SCHEMA,
        "model": display_name,
        "state_key": target_name,
        "api_model": api_model,
        "source": source,
        "runner": runner,
        "opencode_model": opencode_model,
        "is_agent": is_agent,
        "system_prompt": system_prompt,
        "judge_model": judge_model,
        "judge_models": list(judge_models or ([judge_model] if judge_model else [])),
        "judge_prompt_version": judge_prompt_version,
        "judge_contracts": active_judge_contracts,
        "judge_status": "disabled" if not (judge_models or judge_model) else "pending",

        "status": "ok",
        "stream_ok": True,
        "ttft": None,
        "prompt_tokens": 0, "completion_tokens": 0,
        "total_time": 0, "error": None,
        "plugin_versions": {p.id: p.version for p in active_plugins},
    }

    state.update(
        target_name,
        status="queued",
        judge_models=list(judge_models or ([judge_model] if judge_model else [])),
        **{
            f"{pid}_judge_selected_contract": contract
            for pid, contract in active_judge_contracts.items()
        },
    )

    cfg = source_config.get(source)
    if runner == "http" and cfg is None:
        r["status"] = "error"
        r["error"] = f"Unknown source '{source}' — not in SOURCE_CONFIG"
        r["total_time"] = round(time.time() - start, 1)  # type: ignore[assignment]
        state.publish_result(
            r, status="failed", error=r["error"], elapsed=r["total_time"],
            last_error=r["error"],
        )
        state.log(target_name, r['error'])
        return r

    latest = {(res.get("state_key", res["model"]), res.get("runner", "http")): res
              for res in state.latest_results()}
    existing = latest.get((target_name, runner))
    # Live per-plugin progress for this target. A previous session may have
    # scored plugins into ``model_info`` but been interrupted before
    # ``add_result`` committed a full row to ``results`` (e.g. a preload error
    # row with no scores is left as the latest row). Fall back to those
    # stranded scores so an interrupted-but-successful plugin is reused
    # instead of silently re-run.
    live_info = state.snapshot().get(target_name, {})

    plugins_to_run = []
    for plugin in active_plugins:
        pid = plugin.id
        score_key = f"{pid}_score"
        # Re-use successful plugin results from a previous run; re-run any
        # plugin that failed or was missing. Prefer the latest committed
        # result row, then fall back to the live model_info score.
        source_row = existing
        if existing is None or score_key not in existing or existing[score_key] == "fail":
            live_score = live_info.get(score_key)
            if live_score is not None and live_score != "fail":
                source_row = live_info
            else:
                source_row = None
        if source_row is not None:
            r[f"{pid}_score"] = source_row[score_key]
            r[f"{pid}_rubric"] = source_row.get(f"{pid}_rubric", [])
            r[f"{pid}_diagnostics"] = source_row.get(f"{pid}_diagnostics", {})
            r[f"{pid}_response_time"] = source_row.get(f"{pid}_response_time")
            r[f"{pid}_output_tokens"] = source_row.get(f"{pid}_output_tokens")
            r[f"{pid}_thinking_tokens"] = source_row.get(f"{pid}_thinking_tokens")
            r[f"{pid}_total_tokens"] = source_row.get(f"{pid}_total_tokens")
            r[f"{pid}_tps"] = source_row.get(f"{pid}_tps")
            r[f"{pid}_stream_ok"] = source_row.get(f"{pid}_stream_ok", True)
            r[f"{pid}_empty_reason"] = source_row.get(f"{pid}_empty_reason")
            for key in (
                "max_tokens", "attempt_count", "retry_count", "retried",
                "retry_reasons", "selected_attempt", "retry_reason",
                "prompt_altered", "response_nature", "finish_reason",
                "truncated", "truncated_due_to_time", "repeating",
                "failure_cause", "attempts", "runner_metadata",
            ):
                r[f"{pid}_{key}"] = source_row.get(f"{pid}_{key}")
            source_contract = source_row.get(f"{pid}_judge_selected_contract")
            current_contract = active_judge_contracts.get(pid)
            projection_matches = (
                not active_judge_contracts or source_contract == current_contract
            )
            r[f"{pid}_judge_score"] = (
                source_row.get(f"{pid}_judge_score") if projection_matches else None
            )
            r[f"{pid}_judge_confidence"] = (
                source_row.get(f"{pid}_judge_confidence") if projection_matches else None
            )
            r[f"{pid}_judge_rationale"] = (
                source_row.get(f"{pid}_judge_rationale") if projection_matches else None
            )
            r[f"{pid}_judge_error"] = (
                source_row.get(f"{pid}_judge_error") if projection_matches else None
            )
            r[f"{pid}_judge_input_sha256"] = source_row.get(f"{pid}_judge_input_sha256")
            r[f"{pid}_judge_votes"] = source_row.get(f"{pid}_judge_votes", [])
            r[f"{pid}_judge_criteria"] = (
                source_row.get(f"{pid}_judge_criteria", []) if projection_matches else []
            )
            r[f"{pid}_judge_consensus_by_contract"] = source_row.get(
                f"{pid}_judge_consensus_by_contract", {}
            )
            r[f"{pid}_judge_selected_contract"] = current_contract
            r[f"{pid}_judge_complete"] = (
                source_row.get(f"{pid}_judge_complete", False) if projection_matches else False
            )
            for key in ("schema_requested", "schema_request_status", "response_schema_valid",
                        "schema_enforcement_verified"):
                r[f"{pid}_{key}"] = source_row.get(f"{pid}_{key}")
        else:
            plugins_to_run.append(plugin)

    if not plugins_to_run:
        r["stream_ok"] = any(r.get(f"{p.id}_stream_ok", True) for p in active_plugins)
        r["ttft"] = existing.get("ttft") if existing else None
        r["overall_score_100"] = _overall_score(r, active_plugins)
        r["overall_scored_plugins"] = sum(
            isinstance(r.get(f"{p.id}_score"), (int, float))
            and not isinstance(r.get(f"{p.id}_score"), bool)
            for p in active_plugins
        )
        r["total_time"] = round(time.time() - start, 1)  # type: ignore[assignment]
        state.publish_result(
            r, status="completed", elapsed=r["total_time"],
        )
        publish_judge_sidecars(
            judge_input_dir, display_name, runner, active_plugins, judge_enqueue,
        )
        return r

    plugin_thread_limit = source_config.get(source, {}).get("plugin_thread_limit", 1)
    try:
        plugin_thread_limit = int(plugin_thread_limit)
    except (TypeError, ValueError):
        plugin_thread_limit = 1
    if plugin_thread_limit <= 0:
        plugin_thread_limit = len(plugins_to_run)

    state.update(target_name, attempt_start=time.monotonic())

    _run_plugins(target_name, api_model, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, max_tokens, output_dir,
                 session_seed, global_cfg, r, start,
                 max_workers=plugin_thread_limit,
                 stop_event=stop_event,
                 save_responses=save_responses,
                 judge_input_dir=judge_input_dir,
                 judge_enqueue=judge_enqueue,
                 system_prompt=system_prompt,
                 is_agent=is_agent,
                 runner=runner,
                 opencode_config_path=opencode_config_path,
                 opencode_model=opencode_model,
                 opencode_agent=opencode_agent,
                 opencode_binary=opencode_binary,
                 pi_node=pi_node,
                 pi_worker=pi_worker,
                 pi_config=pi_config,
                 display_name=display_name,
                 config_target_name=config_target_name,
                 debug_logs=debug_logs)
    return r


def _run_plugins(target_name: str, api_model: str, source: str, state: Any, active_plugins: list[Any], plugins_to_run: list[Any],
                 source_config: dict[str, Any], timeout: float, max_tokens: int, output_dir: str,
                 session_seed: int, global_cfg: dict[str, Any] | None, r: dict[str, Any], start: float, max_workers: int,
                 stop_event: threading.Event | None = None, save_responses: bool = False, judge_input_dir: str | None = None,
                 judge_enqueue: Callable[[str, str, str, str], Any] | None = None,
                 system_prompt: str | None = None, is_agent: bool = False, runner: str = "http", opencode_config_path: str | None = None,
                 opencode_model: str | None = None, opencode_agent: str | None = None, opencode_binary: str | None = None,
                 pi_node: str | None = None, pi_worker: str | None = None, pi_config: dict[str, Any] | None = None,
                 display_name: str | None = None, config_target_name: str | None = None, debug_logs: bool = False) -> None:
    """Run plugins for one model using a thread pool of bounded size.

    A single-worker pool (``max_workers=1``) is equivalent to sequential
    execution, so this helper is used for both sequential and parallel
    plugin execution.
    """
    results = {plugin.id: None for plugin in plugins_to_run}
    errors = {}
    lock = threading.Lock()
    model_stop_event = threading.Event()
    consecutive_429 = 0
    breaker_triggered = False
    breaker_reason = "Cancelled after 2 consecutive exhausted HTTP 429 responses"
    logs_dir = os.path.join(output_dir, "logs")
    log_file = (
        os.path.join(logs_dir, f"{sanitize_filename(display_name or target_name)}.log.gz")
        if debug_logs else None
    )

    def run_one(plugin: Any) -> None:
        nonlocal consecutive_429, breaker_triggered
        pid = plugin.id
        # The model-level circuit breaker is distinct from the process-wide
        # stop event: one rate-limited model must not cancel other models.
        # Queued plugins check it before dispatch, while in-flight HTTP
        # requests receive the same event so their retry sleep can stop.
        if (stop_event and stop_event.is_set()) or model_stop_event.is_set():
            with lock:
                errors[pid] = breaker_reason if breaker_triggered else "Cancelled"
            return
        # Track in-flight plugin tasks via the canonical ``running_pids``
        # list (not a pid-suffix status string) so the live TUI can render
        # each plugin's "[streaming]"/"[requested]" bracket cell and the
        # table's yellow highlight for parallel plugin threads (max_workers > 1).
        # The previous ``state.update(target_name, status=f"running_{pid}")``
        # write left ``running_pids`` empty, which silently broke every
        # downstream visualisation that read it.
        state.start_plugin_run(target_name, pid)
        try:
            task_result = _run_plugin_task(target_name, api_model, source, plugin, source_config,
                                           timeout, max_tokens, session_seed, log_file,
                                           global_cfg or {}, state=state,
                                           stop_event=model_stop_event,
                                           save_responses=save_responses,
                                           output_dir=output_dir,
                                           judge_input_dir=judge_input_dir,
                                           judge_enqueue=judge_enqueue,
                                           system_prompt=system_prompt,
                                           is_agent=is_agent,
                                           runner=runner,
                                           opencode_config_path=opencode_config_path,
                                           opencode_model=opencode_model,
                                           opencode_agent=opencode_agent,
                                           opencode_binary=opencode_binary,
                                           pi_node=pi_node,
                                           pi_worker=pi_worker,
                                           pi_config=pi_config,
                                           artifact_target_name=display_name or target_name,
                                           config_target_name=config_target_name or display_name or target_name,
                                           debug_logs=debug_logs)
            result = task_result.result
            err = task_result.error
        finally:
            # Clear the in-flight marker even on exception/cancellation so
            # parallel plugins aren't stranded in the running list when one
            # of them raises. ``status`` is committed by the outer caller
            # (``run_model``) once all plugins resolve.
            state.finish_plugin_run(target_name, pid)
        with lock:
            results[pid] = result  # type: ignore[assignment]
            if err:
                errors[pid] = err
            # A successful or non-429 test breaks the streak. Cancellation
            # caused by the breaker itself is not a test outcome and must not
            # reset the counter while the remaining futures drain.
            if err == breaker_reason or err == "Cancelled":
                pass
            elif _is_exhausted_429(err):
                consecutive_429 += 1
                if consecutive_429 >= 2:
                    breaker_triggered = True
                    model_stop_event.set()
            else:
                consecutive_429 = 0
        if err or result is None:
            return
        state.update(target_name,                        **{f"{pid}_score": result[f"{pid}_score"],
                     f"{pid}_rubric": result.get(f"{pid}_rubric", []),
                     f"{pid}_diagnostics": result.get(f"{pid}_diagnostics", {}),
                     f"{pid}_tps": result[f"{pid}_tps"],

                        f"{pid}_response_time": result[f"{pid}_response_time"],
                        f"{pid}_output_tokens": result[f"{pid}_output_tokens"],
                        f"{pid}_thinking_tokens": result.get(f"{pid}_thinking_tokens"),
                        f"{pid}_total_tokens": result.get(f"{pid}_total_tokens"),
                     f"{pid}_empty_reason": result.get(f"{pid}_empty_reason"),
                     f"{pid}_max_tokens": result.get(f"{pid}_max_tokens"),
                     f"{pid}_attempt_count": result.get(f"{pid}_attempt_count"),
                     f"{pid}_retry_count": result.get(f"{pid}_retry_count"),
                     f"{pid}_retried": result.get(f"{pid}_retried"),
                     f"{pid}_retry_reasons": result.get(f"{pid}_retry_reasons"),
                     f"{pid}_selected_attempt": result.get(f"{pid}_selected_attempt"),
                     f"{pid}_retry_reason": result.get(f"{pid}_retry_reason"),
                     f"{pid}_prompt_altered": result.get(f"{pid}_prompt_altered"),
                     f"{pid}_response_nature": result.get(f"{pid}_response_nature"),
                     f"{pid}_finish_reason": result.get(f"{pid}_finish_reason"),
                     f"{pid}_truncated": result.get(f"{pid}_truncated"),
                     f"{pid}_truncated_due_to_time": result.get(f"{pid}_truncated_due_to_time"),
                     f"{pid}_repeating": result.get(f"{pid}_repeating"),
                     f"{pid}_failure_cause": result.get(f"{pid}_failure_cause"),
                     f"{pid}_attempts": result.get(f"{pid}_attempts"),
                     f"{pid}_runner_metadata": result.get(f"{pid}_runner_metadata", {}),
                     f"{pid}_schema_requested": result.get(f"{pid}_schema_requested"),
                     f"{pid}_schema_request_status": result.get(f"{pid}_schema_request_status"),
                     f"{pid}_response_schema_valid": result.get(f"{pid}_response_schema_valid"),
                     f"{pid}_schema_enforcement_verified": result.get(f"{pid}_schema_enforcement_verified"),
                     f"{pid}_schema_fallback_used": result.get(f"{pid}_schema_fallback_used"),
                     f"{pid}_schema_fallback_error": result.get(f"{pid}_schema_fallback_error")})
        # After the task finishes, retain the selected logical attempt as the
        # final attempt marker. During execution this field tracks the live
        # attempt; after completion it agrees with the per-attempt token
        # counts stored in the result.
        selected_attempt = result.get(f"{pid}_selected_attempt")
        if selected_attempt is not None:
            state.update(target_name, **{f"{pid}_attempt": selected_attempt})
        # A judge can finish between this plugin's state update and the
        # eventual model-level result append. Queue this plugin immediately;
        # ``BenchmarkState.add_result`` merges any judge fields written during
        # that window into the result row.
        if judge_input_dir and judge_enqueue:
            score = result.get(f"{pid}_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                sidecar = judge_sidecar_path(
                    judge_input_dir, display_name or target_name, runner, pid,
                )
                if os.path.isfile(sidecar):
                    judge_enqueue(
                        sidecar, display_name or target_name, runner, pid,
                    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, plugin): plugin for plugin in plugins_to_run}
        pending = set(futures.keys())
        while pending:
            if stop_event and stop_event.is_set():
                model_stop_event.set()
                for f in pending:
                    f.cancel()
                break
            if breaker_triggered:
                for f in pending:
                    f.cancel()
                break
            done, pending = wait(
                pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - per-plugin errors are collected, not fatal
                    plugin = futures[fut]
                    with lock:
                        errors[plugin.id] = f"{type(exc).__name__}: {exc}"

    for plugin in plugins_to_run:
        pid = plugin.id
        if pid in errors or results.get(pid) is None:
            failed_result: dict[str, Any] = results.get(pid) or {}
            r.update(failed_result if isinstance(failed_result, dict) else {})
            fail_values = {
                f"{pid}_score": "fail",
                f"{pid}_response_time": "fail",
                f"{pid}_output_tokens": "fail",
                f"{pid}_thinking_tokens": "fail",
                f"{pid}_total_tokens": "fail",
                f"{pid}_tps": "fail",
                f"{pid}_stream_ok": False,
            }
            r.update(fail_values)
            state.update(target_name, **{**(failed_result if isinstance(failed_result, dict) else {}), **fail_values})
        else:
            result = results[pid]
            if isinstance(result, dict):
                r.update(result)

    first_tok_time = None
    any_stream_ok = False
    for plugin in active_plugins:
        pid = plugin.id
        if plugin.supports_streaming and r.get(f"{pid}_stream_ok"):
            any_stream_ok = True
            response_time = r.get(f"{pid}_response_time")
            if isinstance(response_time, (int, float)) and (first_tok_time is None or response_time < first_tok_time):
                first_tok_time = response_time

    r["stream_ok"] = any_stream_ok
    if first_tok_time is not None:
        r["ttft"] = round(first_tok_time, 3)

    if breaker_triggered:
        r["status"] = "error"
        r["error"] = breaker_reason
        r["cancelled_after_consecutive_429"] = True
        r["total_time"] = round(time.time() - start, 1)
        state.publish_result(
            r, status="failed", error=r["error"], elapsed=r["total_time"],
            last_error=r["error"],
        )
        return

    if stop_event and stop_event.is_set():
        r["status"] = "error"
        r["error"] = "Cancelled"
        r["total_time"] = round(time.time() - start, 1)
        state.publish_result(
            r, status="failed", error=r["error"], elapsed=r["total_time"],
            last_error=r["error"],
        )
        return

    if errors:
        r["status"] = "error"
        r["error"] = "; ".join(f"{pid}: {err}" for pid, err in errors.items())
        r["total_time"] = round(time.time() - start, 1)
        state.publish_result(
            r, status="failed", error=r["error"], elapsed=r["total_time"],
            last_error=r["error"],
        )
        return

    r["overall_score_100"] = _overall_score(r, active_plugins)
    r["overall_scored_plugins"] = sum(
        isinstance(r.get(f"{p.id}_score"), (int, float))
        and not isinstance(r.get(f"{p.id}_score"), bool)
        for p in active_plugins
    )
    r["total_time"] = round(time.time() - start, 1)
    state.publish_result(
        r, status="completed", elapsed=r["total_time"],
    )
    publish_judge_sidecars(
        judge_input_dir, display_name or target_name, runner, active_plugins, judge_enqueue,
    )
