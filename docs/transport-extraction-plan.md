# Transport & Retry Extraction Plan

## Status: Implemented; typed request variants and retry execution are live

The original extraction is implemented in `benchmark/transport.py`,
`benchmark/request_models.py`, and `benchmark/transport_options.py`. The
remaining work below is limited to follow-up modularization and production
strict-typing cleanup; it is no longer a pending design stage.

## Goal

Extract the transport logic into a reusable module so benchmark tasks, judges,
and any future callers share one consistent execution path. The transport layer
normalizes both HTTP and OpenCode responses to the same shape, returning only
content and thinking text (no transport internals). Callers own retry, scoring,
and storage.

## 1. What Exists Today

### Three distinct call paths

```
Benchmark task (_run_plugin_task)
  ├── HTTP: stream_request / nonstream_request
  └── OpenCode: run_process
      Both followed by: response_nature() → retry decision → evaluate() → metadata

Judge (judge_response)
  └── HTTP: stream_request only
      Followed by: parse_judge_response() → retry decision → score record
```

### Shared behavior (duplicated)

| Behavior | Task path | Judge path | Identical? |
|---|---|---|---|
| HTTP 429 backoff | Yes (`_post_request_context`) | Yes (`stream_request` → `_post_request_context`) | **Yes** — same function |
| Watchdog / timeout | Yes (`_post_request_context`) | Yes (`stream_request`) | **Yes** |
| Stream → nonstream fallback | Yes (`execute_once` checks `plugin.supports_streaming`) | No (always streaming) | Partial |
| Repetition guard | Yes (in HTTP layer) | No | No |
| Reasoning token tracking | Yes (in HTTP layer) | No | No |
| Max-attempt loop | Yes (for attempt in (1,2)) | Yes (for attempt in range(2)) | **Similar** but different retry triggers |
| Prompt alteration | Yes (`_retry_prompt_alteration`) | Yes (JSON error + thinking budget) | **Different logic** |
| Evaluation | Yes (`plugin.evaluate()`) | Yes (`parse_judge_response()`) | **Different** |
| Metadata shape | 30+ fields | 8 fields | **Different** |

### Key structural differences

1. **OpenCode is fundamentally different**: it's a subprocess with NDJSON events,
   step guards, repetition detection, and an output directory. It doesn't use HTTP.
   The extracted layer normalizes its output to the same shape as HTTP.

2. **TUI callbacks are deeply coupled**: `on_chunk`, `on_think_chunk`, `pid`,
   `target_name`, `state` — these are threaded through every HTTP call. Will be
   extracted into a separate Observer interface in a prior commit.

3. **Evaluation is caller-specific**: `plugin.evaluate()` vs
   `parse_judge_response()` vs future callers. The transport layer can't own this.

4. **Retry triggers differ**: benchmark tasks retry on transport_error,
   token_limit, and repetition_abort. Judges retry only on invalid JSON. The retry
   loop stays with the caller.

## 2. Proposed Architecture

### Phase 0 (separate commit): TUI Observer extraction

Before touching transport code, extract the TUI callback threading into a clean
Observer interface. This is a standalone refactor that doesn't change behavior.

```python
# benchmark/observer.py

@dataclass
class TaskObserver:
    """Observer for live streaming progress updates.

    The transport layer calls these methods at the appropriate times.
    Callers create an observer and pass it to the transport layer.
    """
    model_name: str = ""
    pid: str = ""
    on_chunk: Callable[[str], None] | None = None
    on_think_chunk: Callable[[str], None] | None = None
    on_retry: Callable[[], None] | None = None

    def chunk(self, delta: str):
        if self.on_chunk:
            self.on_chunk(delta)

    def think_chunk(self, delta: str):
        if self.on_think_chunk:
            self.on_think_chunk(delta)

    def retry(self):
        if self.on_retry:
            self.on_retry()

    def noop(cls) -> "TaskObserver":
        """Return an observer that does nothing."""
        return cls()
```

Currently these callbacks are threaded through `stream_request` kwargs. This
commit moves them behind the `TaskObserver` interface, which is passed to the
transport layer. The HTTP request functions accept an observer instead of
individual callbacks.

### Phase 1 (separate commit): Transport layer (one attempt, no retry)

Create `benchmark/transport.py` — a stateless execution engine that runs one
prompt through one transport mechanism and returns a normalized result.

#### Data model

```python
@dataclass(frozen=True)
class TransportRequest:
    """What to send and how.

    ``request_params`` may be mutated by the transport layer when a schema
    grammar fallback occurs (replacing ``json_schema`` with ``json_object``).
    The caller should re-read ``request_params`` after ``execute_transport``
    returns if it needs the post-fallback value.
    """
    prompt: str
    max_tokens: int
    source_config: dict
    api_model: str
    source: str
    timeout: float
    temperature: float = 0.0
    system_prompt: str | None = None
    drop_params: list | None = None
    request_params: dict | None = None
    session_seed: int = 0
    log_path: str | None = None
    log_label: str = ""
    pid: str = ""
    stop_event: threading.Event | None = None
    observer: TaskObserver | None = None
    # Stream guards
    max_content_tokens: int | None = None
    max_thinking_tokens: int | None = None
    repetition_guard: int | None = None
    # Transport selection
    transport: Literal["http", "opencode"] = "http"
    supports_streaming: bool = True
    # OpenCode-specific (ignored when transport="http")
    opencode_config_path: str | None = None
    opencode_model: str | None = None
    opencode_agent: str | None = None
    opencode_binary: str | None = None
    opencode_output_dir: str | None = None
    opencode_no_output_grace: float | None = None
    opencode_target_key: str | None = None
    opencode_plugin_id: str | None = None


@dataclass(frozen=True)
class TransportResult:
    """Normalized result from one transport attempt.

    Both HTTP and OpenCode responses are mapped to this shape.
    Only content and thinking text — no transport internals.
    """
    text: str
    think_text: str
    error: str | None
    finish_reason: str | None
    response_time: float    # wall-clock from start to end
    gen_time: float         # generation-only time
    stream_ok: bool         # True if streaming was used and successful
    repeating: bool
    usage: dict[str, Any]
    # Classification (populated by transport layer)
    response_nature: str    # completed, token_limit, transport_error, timeout, cancelled, empty
    empty_reason: str | None
    thinking_tokens: int
    # Prompt tracking
    prompt_sha256: str
    response_sha256: str
    # Schema fallback metadata (HTTP nonstreaming only)
    schema_fallback_used: bool
    schema_fallback_error: str | None
```

#### Core function

```python
def execute_transport(request: TransportRequest) -> TransportResult:
    """Run one prompt through the selected transport mechanism.

    For HTTP:
    - If supports_streaming is True, uses stream_request.
    - If supports_streaming is False, uses nonstream_request with schema
      grammar fallback.
    - Falls back to nonstream only on provider-side streaming rejection.

    For OpenCode:
    - Uses run_process.
    - Normalizes the result to TransportResult shape.

    Returns a normalized TransportResult with classification.
    """
    ...
```

#### Stream → nonstream fallback

Only triggers on explicit provider rejection, not on any error. The
`supports_streaming` flag on the plugin controls the initial path; the fallback
is a safety net.

```python
# Pseudocode for HTTP path
if request.supports_streaming:
    result = stream_request(...)
    if result.error and _is_streaming_not_supported(result.error):
        result = nonstream_request(...)
else:
    result = nonstream_request(...)
```

#### Schema grammar fallback (HTTP nonstreaming only)

This is the most intricate transport-specific behavior. It applies only when:
- `transport="http"`
- `supports_streaming=False` (code-review, moe-dense, structured-output plugins)
- `request_params` contains `response_format: { type: "json_schema", ... }`

**Current flow:**
```
request_nonstream(prompt, label)
  ├── nonstream_request(request_params={response_format: json_schema})
  │   └── provider fails: "failed to parse grammar"
  ├── _is_schema_grammar_error(error) → True
  ├── _json_object_fallback_params(request_params)
  │   └── replaces response_format with {type: "json_object"}
  ├── schema_fallback_used = True
  ├── schema_fallback_error = original error
  └── nonstream_request(request_params={response_format: json_object})
      └── success (or different error)
```

**Key subtlety:** `request_params` is mutated in place
(`request_params = fallback_params`), so the modified params persist across
attempts. The fallback metadata flows into `_schema_request_metadata()` and
ends up in the result dict.

**New architecture — the fallback lives inside `execute_transport`:**

```python
def _execute_http_nonstream(request: TransportRequest) -> TransportResult:
    """Execute a non-streaming HTTP request with schema grammar fallback."""
    schema_fallback_used = False
    schema_fallback_error = None
    request_params = request.request_params  # may be mutated

    def _do_request(prompt, params, label):
        return nonstream_request(
            request.source_config, request.timeout, request.api_model,
            request.source, prompt, request.max_tokens,
            log_path=request.log_path, log_label=label,
            session_seed=request.session_seed, temperature=request.temperature,
            drop_params=request.drop_params, stop_event=request.stop_event,
            system_prompt=request.system_prompt, pid=request.pid,
            on_retry=lambda: request.observer.retry() if request.observer else None,
            max_content_tokens=request.max_content_tokens,
            max_thinking_tokens=request.max_thinking_tokens,
            repetition_guard=request.repetition_guard,
            **({"request_params": params} if params else {}),
        )

    result = _do_request(request.prompt, request_params, request.log_label)

    # Schema grammar fallback: if the provider rejected the grammar,
    # retry with a simpler JSON-object mode.
    if result.error and _is_schema_grammar_error(result.error):
        fallback_params = _json_object_fallback_params(request_params)
        if fallback_params is not None:
            schema_fallback_used = True
            schema_fallback_error = result.error
            request_params = fallback_params  # mutate for this attempt
            result = _do_request(
                request.prompt, request_params,
                f"{request.log_label} (JSON-object schema fallback)",
            )

    return _normalize_http_nonstream(result, request, started,
                                     schema_fallback_used=schema_fallback_used,
                                     schema_fallback_error=schema_fallback_error)
```

**The `TransportResult` carries the fallback metadata:**

```python
@dataclass(frozen=True)
class TransportResult:
    ...
    # Schema fallback metadata (HTTP nonstreaming only)
    schema_fallback_used: bool     # True if json_schema → json_object fallback occurred
    schema_fallback_error: str | None  # The original grammar error (if fallback used)
```

For streaming and OpenCode paths, these fields are always `False` / `None`.

**How the caller consumes it:**

The caller (Phase 2's `execute_task`, or Phase 3's `save_task_result`) uses
`TransportResult.schema_fallback_used` and `TransportResult.schema_fallback_error`
when building the schema metadata dict. No mutation tracking needed — the
result tells you what happened.

```python
# In save_task_result or execute_task:
schema_metadata = _schema_request_metadata(
    plugin,
    request.request_params,  # post-fallback value (mutated by execute_transport)
    response_schema_valid=diagnostics.get("response_schema_valid"),
    request_applied=request_applied,
    schema_fallback_used=result.schema_fallback_used,
    schema_fallback_error=result.schema_fallback_error,
    error=selected_error or score_error,
)
```

**Why the mutation is safe:**

`TransportRequest` is `frozen=True`, but `request_params` is a dict reference.
The mutation happens inside `execute_transport` and is visible to the caller
after the call returns (same dict object). This preserves the current behavior
where the fallback params persist across retry attempts. If this feels too
implicit, an alternative is to return the post-fallback params as a field on
`TransportResult`:

```python
@dataclass(frozen=True)
class TransportResult:
    ...
    effective_request_params: dict | None  # post-fallback request_params
```

This makes the mutation explicit without changing the caller's mental model.

#### Normalization

```python
def _normalize_http_stream(response: StreamResult, started: float) -> TransportResult:
    """Map StreamResult to TransportResult."""
    first = response.first_tok
    stream_end = response.stream_end or time.time()
    gen_time = stream_end - first if first else stream_end - started
    text = response.text or ""
    return TransportResult(
        text=text,
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=response.finish_reason,
        response_time=round(stream_end - started, 1),
        gen_time=max(0.0, gen_time),
        stream_ok=response.error is None and first is not None,
        repeating=is_repeating(text) or "repetition" in str(response.error or "").lower(),
        usage=getattr(response, "usage", {}) or {},
        response_nature=response_nature(text=text, error=response.error, ...),
        empty_reason=classify_empty_reason(text, ...),
        thinking_tokens=_response_reasoning_tokens(response) or 0,
        prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        schema_fallback_used=False,   # streaming never uses schema fallback
        schema_fallback_error=None,
    )

def _normalize_http_nonstream(response, request, started, *,
                               schema_fallback_used=False,
                               schema_fallback_error=None) -> TransportResult:
    """Map nonstream_request result to TransportResult."""
    text = response.text or ""
    return TransportResult(
        text=text,
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=response.finish_reason,
        response_time=round(response.gen_time, 1),
        gen_time=response.gen_time,
        stream_ok=False,
        repeating=is_repeating(text),
        usage=getattr(response, "usage", {}) or {},
        response_nature=response_nature(text=text, error=response.error, ...),
        empty_reason=classify_empty_reason(text, ...),
        thinking_tokens=_response_reasoning_tokens(response) or 0,
        prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        schema_fallback_used=schema_fallback_used,
        schema_fallback_error=schema_fallback_error,
    )

def _normalize_opencode(response, started: float) -> TransportResult:
    """Map OpenCode run_process result to TransportResult."""
    text = response.text or ""
    return TransportResult(
        text=text,
        think_text=response.think_text or "",
        error=response.error,
        finish_reason=None,  # OpenCode doesn't provide this
        response_time=round(response.elapsed, 1),
        gen_time=response.elapsed,
        stream_ok=False,  # OpenCode is never streaming
        repeating=is_repeating(text),
        usage={},  # OpenCode doesn't provide token usage
        response_nature=response_nature(text=text, error=response.error, ...),
        empty_reason=classify_empty_reason(text, ...),
        thinking_tokens=0,  # Would need to be extracted from OpenCode events
        prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        schema_fallback_used=False,   # OpenCode never uses schema fallback
        schema_fallback_error=None,
    )
```

### Phase 2 (separate commit): Full execution engine (retry + classification + selection)

Add `execute_task` to `benchmark/transport.py` — wraps `execute_transport`
with the attempt loop, classification, retry logic, and attempt history.

```python
@dataclass
class TaskAttempt:
    """One attempt in a multi-attempt execution."""
    result: TransportResult
    attempt_number: int
    prompt_altered: str       # none, thinking_50_percent, avoid_repetition, ...
    retry_reason: str | None  # what triggered this attempt
    request_prompt: str       # the actual prompt sent


@dataclass
class TaskExecution:
    """Complete result of up to 2 attempts, with selection."""
    attempts: list[TaskAttempt]
    selected: TaskAttempt
    attempt_count: int
    retry_count: int
    retry_reasons: list[str]
    retry_reason: str | None


def execute_task(
    request: TransportRequest,
    *,
    retry_policy: RetryPolicy,
    base_prompt: str,
    prompt_alterer: Callable[[str, str, int], tuple[str, str]] | None = None,
    json_error_prompt_alterer: Callable[[str], str] | None = None,
) -> TaskExecution:
    """Run up to 2 attempts with classification and retry.

    The caller owns:
    - What to do with the result (evaluate, score, store)
    - How to build the retry prompt (via prompt_alterer)
    - What to store in state/disk

    This function owns:
    - The attempt loop
    - Classification of each attempt
    - Retry decision based on RetryPolicy
    - Prompt alteration dispatch
    - Attempt history and selection
    """
    ...
```

#### Retry policy

```python
@dataclass
class RetryPolicy:
    """Controls what triggers a retry and how prompts are altered."""
    max_attempts: int = 2
    retry_on_transport_error: bool = True
    retry_on_token_limit: bool = True
    retry_on_repetition: bool = True
    retry_on_json_error: bool = False
    retry_on_timeout: bool = False

BENCHMARK_RETRY_POLICY = RetryPolicy(
    retry_on_transport_error=True,
    retry_on_token_limit=True,
    retry_on_repetition=True,
    retry_on_json_error=False,
    retry_on_timeout=False,
)

JUDGE_RETRY_POLICY = RetryPolicy(
    retry_on_transport_error=False,
    retry_on_token_limit=False,
    retry_on_repetition=False,
    retry_on_json_error=True,
    retry_on_timeout=False,
)
```

### Phase 3 (separate commit): Storage helpers

Create `benchmark/results.py` — extract the "write to disk/state" logic from
`_run_plugin_task` and `judge_response` into reusable helpers. This commit
does not change behavior; it moves existing persistence code into a dedicated
module so callers can import and use it independently.

```python
def save_task_result(
    execution: TaskExecution | TransportResult,
    *,
    state: BenchmarkState,
    model_name: str,
    pid: str,
    plugin,
    output_dir: str | None = None,
    save_responses: bool = False,
    judge_input_dir: str | None = None,
    judge_enqueue: Callable | None = None,
    artifact_target: str | None = None,
    runner: str = "http",
    # Schema metadata (populated from TransportResult)
    request_applied: bool = True,
    # Scoring fields (populated by caller after evaluation)
    score: Any = None,
    rubric: list = None,
    diagnostics: dict = None,
    score_error: str | None = None,
) -> dict[str, Any]:
    """Convert execution result to state dict, persist sidecars, return result row.

    Handles:
    - Building the flat result dict for BenchmarkState.add_result()
    - Writing response sidecar files (prompt, content, meta.json)
    - Enqueueing judge input sidecars
    - Schema metadata injection (consumes schema_fallback_used/error from result)
    """
    ...

def save_judge_result(
    result: TransportResult,
    *,
    state: BenchmarkState,
    model_name: str,
    plugin_id: str,
    parsed_judge: dict | None = None,
) -> dict[str, Any]:
    """Persist judge result to state.

    Handles:
    - Building the judge result dict for BenchmarkState.add_result()
    - Writing judge response sidecar files
    """
    ...
```

#### What moves out of `_run_plugin_task`

The following blocks move verbatim into `save_task_result`:

1. **Schema metadata injection** — `_schema_request_metadata()` call and
   `schema_metadata` dict construction (now uses `result.schema_fallback_used`
   and `result.schema_fallback_error` instead of closure-captured variables).
2. **Judge sidecar preparation** — `prepare_judge_sidecar()` call.
3. **Response file persistence** — the `if save_responses and output_dir:` block
   that writes `prompt.txt`, `content.txt`, `response.txt`, `think.txt`, and
   `meta.json` inside a per-plugin subdirectory under `responses/<target>/`.
4. **Result dict construction** — the `result = { f"{pid}_{key}": value ... }`
   block.
5. **Meta dict construction** — the `meta = { ... }` block (returned alongside
   the result dict for callers that need it).

#### What moves out of `judge_response`

Nothing moves out of `judge_response` in this phase — its persistence is
minimal (just state updates). The helper is created in preparation for Phase 4
streaming, where judge results will need the same sidecar/builder pattern.

#### Caller changes in this phase

```python
# _run_plugin_task after Phase 3
def _run_plugin_task(...):
    # ... config resolution, transport request, execute_task ...
    result_dict = save_task_result(
        execution,
        state=state, model_name=target_name, pid=pid, plugin=plugin,
        output_dir=output_dir, save_responses=save_responses,
        judge_input_dir=judge_input_dir, judge_enqueue=judge_enqueue,
        artifact_target=artifact_target, runner=runner,
        request_applied=schema_request_applied,
        score=score, rubric=rubric, diagnostics=diagnostics,
        score_error=score_error,
    )
    return PluginTaskResult(result_dict, task_error)
```

The ~80 lines of metadata construction, file writing, and dict building that
currently live inline in `_run_plugin_task` are replaced by the single
`save_task_result()` call. The function body shrinks by roughly that amount.

### Phase 4 (fast follow, separate commit): Streaming future

Return an object with a streaming response + a future for the metadata,
accessible once the stream is completed.

```python
@dataclass
class StreamingTaskExecution:
    """Streaming version of TaskExecution for in-the-moment scoring."""
    stream: Iterator[str]                      # live tokens
    metadata_future: Future[TaskAttempt]       # resolves after stream completes
    next_attempt: "StreamingTaskExecution | None" = None  # populated if retry triggers
```

The consumer iterates `stream` for live tokens. When iteration completes (or is
canceled), `metadata_future.result()` gives the full `TaskAttempt` with
classification. If a retry is warranted, `next_attempt` is populated with a new
`StreamingTaskExecution` for the retry.

## 3. Phase Ordering

```
Phase 0: Observer extraction       (commit: refactor TUI callbacks into TaskObserver)
    ↓
Phase 1: Transport layer           (commit: benchmark/transport.py — one attempt, normalized output)
    ↓
Phase 2: Full execution engine     (commit: execute_task + RetryPolicy — attempt loop, selection)
    ↓
Phase 3: Storage helpers           (commit: benchmark/results.py — move persistence out of callers)
    ↓
Phase 4: Streaming future          (commit: fast-follow — StreamingTaskExecution + linked-list retry)
```

Each phase is a separate commit. Phases 0-3 are the core extraction.
Phase 4 is the streaming future (fast follow after the base is stable).

### Why storage is its own phase (Phase 3)

Separating storage into its own commit provides three benefits:

1. **Cleaner diffs**: Phase 2 changes execution logic; Phase 3 changes
   persistence logic. Reviewing them separately is easier than reviewing one
   large commit that touches both.

2. **Independent verification**: Phase 2 can be verified by running the
   benchmark and checking that results are identical. Phase 3 can be verified
   by checking that the same files are written with the same content.

3. **Future flexibility**: If the storage layer needs to change (e.g., adding
   new sidecar formats, changing the state dict shape), it's a self-contained
   commit that doesn't touch the execution engine.

## 4. Caller Impact

### Before: `_run_plugin_task` (~400 lines)

```
_run_plugin_task
  ├── resolve config, build prompt
  ├── define execute_once (HTTP/OpenCode branching)
  ├── define request_nonstream (schema fallback)
  ├── define evaluate
  ├── attempt loop (classify, retry, alter prompt)
  ├── select best attempt
  ├── build metadata dict
  ├── persist responses
  └── return PluginTaskResult
```

### After Phase 2: `_run_plugin_task` (~200 lines)

```
_run_plugin_task
  ├── resolve config, build prompt, create observer
  ├── build TransportRequest
  ├── execute_task(request, retry_policy=BENCHMARK_RETRY_POLICY)
  ├── evaluate selected result
  ├── build metadata dict (inline)
  ├── persist responses (inline)
  └── return PluginTaskResult
```

### After Phase 3: `_run_plugin_task` (~100 lines)

```
_run_plugin_task
  ├── resolve config, build prompt, create observer
  ├── build TransportRequest
  ├── execute_task(request, retry_policy=BENCHMARK_RETRY_POLICY)
  ├── evaluate selected result
  ├── save_task_result(execution, state, plugin, ...)
  └── return PluginTaskResult
```

### Before: `judge_response` (~80 lines)

```
judge_response
  ├── read sidecar, build prompt
  ├── attempt loop (stream, parse JSON, retry on invalid)
  └── return JudgeResult
```

### After Phase 2: `judge_response` (~50 lines)

```
judge_response
  ├── read sidecar, build prompt
  ├── execute_task(request, retry_policy=JUDGE_RETRY_POLICY)
  ├── parse selected result
  └── return JudgeResult
```

## 5. Migration Strategy

### Phase 0: Observer extraction (1 commit)

1. Create `benchmark/observer.py` with `TaskObserver` dataclass.
2. Update `stream_request` / `nonstream_request` to accept `observer` instead
   of individual `on_chunk` / `on_think_chunk` / `on_retry` kwargs.
3. Update callers (`_run_plugin_task`, `judge_response`, preload) to create
   a `TaskObserver` and pass it.
4. Verify all tests pass.

### Phase 1: Transport layer (1 commit)

1. Create `benchmark/transport.py` with `TransportRequest`, `TransportResult`,
   and `execute_transport`.
2. Implement `execute_transport` as a direct refactor of `execute_once` + the
   HTTP/OpenCode branching logic.
3. Move `response_nature`, `classify_empty_reason`, `is_repeating`, and
   `_response_reasoning_tokens` into `benchmark/transport.py` (they're
   execution logic, not orchestration).
4. Update `_run_plugin_task` to call `execute_transport` for its single attempt
   (keep the existing retry loop for now).
5. Verify all tests pass.

### Phase 2: Full engine (1 commit)

1. Add `TaskAttempt`, `TaskExecution`, `RetryPolicy` to `benchmark/transport.py`.
2. Implement `execute_task` wrapping `execute_transport` with the attempt loop.
3. Move `_retry_prompt_alteration` and `_thinking_consumed_budget` into
   `benchmark/transport.py`.
4. Rewrite `_run_plugin_task` as a thin wrapper (metadata and persistence
   still inline).
5. Rewrite `judge_response` as a thin wrapper.
6. Verify all tests pass.

### Phase 3: Storage helpers (1 commit)

1. Create `benchmark/results.py` with `save_task_result` and `save_judge_result`.
2. Extract metadata construction, response file writing, sidecar preparation,
   and result dict building from `_run_plugin_task` into `save_task_result`.
3. Update `_run_plugin_task` to call `save_task_result` instead of inline code.
4. Add `save_judge_result` as a forward-looking helper (minimal use now,
   fuller use in Phase 4).
5. Verify all tests pass — confirm identical output files and state dicts.

### Phase 4: Streaming future (1 commit, fast follow)

1. Implement `StreamingTaskExecution` with linked-list retry.
2. Add a streaming variant of `execute_task`.
3. Update callers that want live scoring to use the streaming variant.

## 6. Estimated Impact

| Area | Before | After Phase 2 | After Phase 3 |
|---|---|---|---|
| `_run_plugin_task` lines | ~400 | ~200 | ~100 |
| `judge_response` lines | ~80 | ~50 | ~50 |
| `benchmark/transport.py` | 0 | ~400 | ~400 |
| `benchmark/observer.py` | 0 | ~40 | ~40 |
| `benchmark/results.py` | 0 | 0 | ~150 |
| Total code | ~480 | ~690 | ~740 |
| Shared behavior | Duplicated | Single source of truth | Single source of truth |
| New callers | Must reimplement | Call `execute_task` or `execute_transport` | Same + `save_task_result` |

**Net:** ~260 lines more code, but a cleaner architecture with:
- One execution path instead of two divergent ones
- Normalized results for both HTTP and OpenCode
- Configurable retry policies
- Clean separation of transport, execution, and storage
- Easy to add new transports (e.g., pi-agent) without touching orchestration

## 7. Risks

1. **Regression surface**: Rewriting the core execution path touches every test.
   Mitigated by the phased approach — each phase is a direct refactor.

2. **OpenCode normalization**: Forcing OpenCode into the same result shape may
   lose information (e.g., OpenCode doesn't provide `finish_reason` or `usage`).
   Mitigated by keeping these fields optional with sensible defaults.

3. **Breaking the live TUI**: The TUI relies on callbacks firing at the right
   time during streaming. Phase 0 extracts the observer cleanly; Phases 1-2
   preserve the same callback signatures through the observer.

4. **Schema grammar fallback**: The current fallback logic is inside
   `request_nonstream`. It needs to be preserved in the transport layer without
   leaking into the generic path. Handled by having `execute_transport` own the
   fallback when `transport="http"` and `supports_streaming=False`. The result
   carries `schema_fallback_used` / `schema_fallback_error` so callers don't
   need closure state.

5. **Judge simplification risk**: Making judges use `execute_task` with
   `JUDGE_RETRY_POLICY` keeps their retry behavior separate (JSON-only retry,
   no transport retry). The policy object makes this explicit.

6. **Storage extraction churn**: Phase 3 moves ~80 lines of metadata/file
   writing out of `_run_plugin_task`. If Phase 2 changes the metadata shape
   (e.g., adding new fields), Phase 3's extraction must match. Mitigated by
   running Phase 2's tests before starting Phase 3.

7. **Schema fallback mutation**: `execute_transport` mutates
   `request.request_params` when a grammar fallback occurs. This is safe
   because the dict is passed by reference and the mutation is visible to the
   caller after the call returns. If this feels too implicit, the result can
   carry `effective_request_params` as an explicit post-fallback snapshot.

## 8. Future Extensibility

With this architecture, adding a new transport (e.g., pi-agent) is:

1. Implement `run_pi_agent(...)` in `benchmark/pi_agent.py` (or similar).
2. Add `"pi_agent"` to the `transport` literal type.
3. Add normalization in `execute_transport`:
   ```python
   elif request.transport == "pi_agent":
       response = run_pi_agent(...)
       return _normalize_pi_agent(response, started)
   ```
4. Done. No changes to retry logic, classification, or storage.

Adding a new retry policy is equally simple — just define a new `RetryPolicy`
instance with the desired triggers.
