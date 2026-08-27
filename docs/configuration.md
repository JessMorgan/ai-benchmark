# Configuration Reference

All benchmark configuration lives in a single file (default: `benchmark-config.json`). The file is passed to `ai-benchmark.py` with `--config`. Both JSON and YAML (`.yaml`/`.yml`) formats are supported; the extension is detected automatically.

## Top-Level Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `benchmark-results` | Directory for the SQLite database, reports, logs, and optional state |
| `storage` | string | `sqlite` | Run backend: `sqlite` (default) or explicit legacy `json` |
| `storage_profile` | string | `compact` | Artifact policy: `compact`, `debug`, or `portable` |
| `timeout` | integer | `600` | API request timeout in seconds |
| `max_tokens` | integer | `16384` | One generation budget; each task may retry once with the same budget |
| `model_max_tokens` | object | `{}` | Per-target max-token overrides; keys are target names or `"{source}/{api_model}"` |
| `plugin_thread_limit` | integer | `1` | Top-level fallback for `sources.*.plugin_thread_limit` |
| `model_thread_limit` | positive integer | `1` | Top-level fallback for per-source target/model concurrency; zero is invalid |
| `preload` | boolean | `false` | Per-source model warm-up is opt-in; use `--no-preload` to disable all preload probes for a run |
| `preload_timeout` | integer | `300` | Maximum seconds for a source's warm-up probe; independent of the benchmark request `timeout` |
| `flush_interval_seconds` | number | `60` | Maximum seconds between full-state snapshots; `0` flushes on every change |
| `flush_votes` | integer | `10` | Flush after this many completed changes (judge votes + benchmark tasks) |
| `flush_shutdown_timeout_seconds` | number | `10` | Maximum shutdown wait for the background state flusher before synchronous fallback |
| `plugins_whitelist` | list[string] | `[]` | Run only these plugin IDs (empty = all) |
| `plugins_blacklist` | list[string] | `[]` | Skip these plugin IDs (empty = none) |
| `sources` | object | required | Named API endpoint definitions |
| `models` | object | required | Model-to-source mapping |
| `agents` | object | optional | Agent definitions with system prompts |
| `judge` | object | optional | Semantic-judge timeout, token budget, temperature, and provider-specific request parameters |

SQLite is authoritative for new runs. Existing JSON state is adopted automatically when the default SQLite database is absent; use `--storage json` to preserve the legacy backend explicitly. Full diagnostic transcripts are omitted by the compact profile and enabled only with `--debug-logs` or `storage_profile: debug`; runner transcripts use redacted concatenated gzip members.

State persistence is throttled across the whole run: completed judge votes and
completed benchmark tasks accumulate in memory, and the full state snapshot
(`benchmark_state.json`) is flushed at most every `flush_interval_seconds`
seconds (default `60`) or every `flush_votes` changes (default `10`), whichever
comes first. Each flush also compacts the append-only `results.journal.jsonl`,
which contains compact `result` and `judge` events. On resume, events newer than
the snapshot's `journal_sequence` are replayed; this closes the crash window
between writing the snapshot and truncating the journal. The flush runs on a
dedicated background thread, so workers never stall on serialization, and
requests that arrive mid-flush are coalesced into one follow-up save.

State snapshots and reports are separate: CSV/HTML/Markdown/PDF files are
rebuilt once at the end of the run or when the app is stopped, never per change.
Shutdown waits up to `flush_shutdown_timeout_seconds` (default `10`) for the
background flusher, then reports the timeout and attempts a synchronous final
snapshot/journal compaction. Persistence failures are printed prominently and
also recorded in `run-info.json` under `persistence_failures`. Raise the flush
values on very large runs where each full-state write is expensive; set
`flush_interval_seconds: 0` to restore per-change persistence. The earlier
judge-only keys `judge.flush_interval_seconds` and `judge.flush_votes` are still
honored as fallbacks.

## Sources

Each entry under `sources` defines an API endpoint. The key is the source name used in `models`.

```json
{
  "sources": {
    "Local Server": {
      "api_url": "http://localhost:11434/chat/completions",
      "headers": {
        "Authorization": "Bearer ${LOCAL_API_KEY:sk-fallback}",
        "Content-Type": "application/json"
      },
      "plugin_thread_limit": 1,
      "model_thread_limit": 1
    }
  }
}
```

### 1min.ai Sources

A source can target the native 1min.ai chat endpoint by setting
`api_protocol: "1min"`:

```yaml
sources:
  1min.ai:
    api_protocol: "1min"
    api_url: https://api.1min.ai/api/chat-with-ai
    headers:
      API-KEY: "${ONEMIN_API_KEY:your-1min-api-key}"
      Content-Type: application/json
    model_thread_limit: 1
```

This switches the request/response encoding from the default OpenAI-compatible
format to 1min.ai's native shape. The runner:

- sends `{"type": "UNIFY_CHAT_WITH_AI", "model": ..., "promptObject": {"prompt": ...}}`;
- selects streaming by appending `?isStreaming=true` automatically;
- reads non-streaming text from `aiRecord.aiRecordDetail.resultObject`;
- parses named SSE events (`content`, `result`, `done`, `error`);
- authenticates via the `API-KEY` header.

1min.ai's chat endpoint has no system-message field and no
`max_tokens`/`temperature`/`seed` parameters. A supplied agent system prompt is
folded into the user prompt, and generation knobs (`max_tokens`, temperature,
`seed`, `drop_params`, and judge `request_params`) are ignored for 1min sources.
HTTP 429 retry/backoff still applies. The OpenCode runner and `/v1/models`
discovery are OpenAI-shaped and do not consult `api_protocol`.

### ChatPlayground.ai Sources (Interactive Web)

ChatPlayground.ai is a closed, JavaScript-rendered web app with no public API:
it authenticates with a username/password (Clerk) and renders chat client-side.
A source with `api_protocol: "chatplayground"` drives that UI with Playwright —
log in once, navigate to the model's single-model chat route, submit the prompt,
and read back the completed (buffered) answer.

```yaml
sources:
  ChatPlayground:
    api_protocol: "chatplayground"
    base_url: https://web.chatplayground.ai
    email: "${CHATPLAYGROUND_EMAIL}"
    password: "${CHATPLAYGROUND_PASSWORD}"
    headless: true
    model_thread_limit: 1
    plugin_thread_limit: 1

models:
  gpt-5.6-terra: ChatPlayground
  deepseek-v4-pro: ChatPlayground
```

Each model is addressed by the **slug** used in the site's `/chat/<slug>`
route; the adapter navigates to `{base_url}/chat/{api_model}`. The sidebar's
"AI MODELS" list currently exposes:

| Display name | Slug |
|---|---|
| ChatGPT-5.6 Terra | `gpt-5.6-terra` |
| ChatGPT-5.6 Luna | `gpt-5.6-luna` |
| Gemini (Latest Model) | `gemini-3-flash` |
| Claude (Latest Model) | `claude-sonnet-4-6` |
| Kimi K | `kimi-k2.6` |
| DeepSeek V4 Pro | `deepseek-v4-pro` |
| DeepSeek V4 Flash | `deepseek-v4-flash` |
| Qwen3.7 Plus | `qwen3.7-plus` |
| Llama 4 Scout | `llama-4-scout` |
| Command A | `command-a` |
| Amazon Nova | `nova-2-lite-v1` |
| Grok 4.5 | `grok-4.5` |
| Mistral Large 3 | `mistral-large-3` |

Prerequisites:

- Install Playwright's Chromium once: `uv run playwright install chromium`.
- Supply `email`/`password` via environment variables (any `${VAR}` value is
  expanded by the config loader).
- If the site blocks the headless browser, set `headless: false` to run headed.

Behavior and limitations:

- The answer is **buffered** — there is no per-token streaming, so
  `{pid}_ttft`/`{pid}_tps` are not meaningful for these sources.
- A supplied agent system prompt is folded into the user prompt (the chat UI
  has no separate system field); `max_tokens`, temperature, `seed`,
  `drop_params`, and judge `request_params` are ignored.
- Browser operations are serialized under a module lock and one logged-in
  session is reused across plugin tasks, so keep `model_thread_limit` and
  `plugin_thread_limit` at `1`.
- Playwright runs in an **isolated worker subprocess**
  (`benchmark/chatplayground_worker.py`); the runner module
  (`benchmark/chatplayground.py`) never imports it. Playwright's sync API is
  not thread-safe and is main-thread-only, and the benchmark runs each model in
  its own worker thread — exercising it in-process could corrupt the
  interpreter and segfault the whole run. With the worker, a native browser
  crash surfaces as a per-request error, the worker is torn down, and the next
  request spawns a fresh one.
- The CSS selectors were captured against the live site (see
  `benchmark/chatplayground.py`) and can be overridden per source via a
  `selectors` mapping if the site changes:

```yaml
sources:
  ChatPlayground:
    api_protocol: "chatplayground"
    email: "${CHATPLAYGROUND_EMAIL}"
    password: "${CHATPLAYGROUND_PASSWORD}"
    selectors:
      prompt_input: "textarea[name=input]"
      send_button: "Send"
```

To enumerate the model slugs exposed by the UI and emit a ready-to-run config
in one step, run the CLI flag. Credentials come from the environment — either
inline or from a local `.env` file (see [Dotenv Files](#dotenv-files)):

```sh
CHATPLAYGROUND_EMAIL=you@example.com CHATPLAYGROUND_PASSWORD=... \
  python ai-benchmark.py --chatplayground-config > benchmark-config.json
```

This logs in, reads the sidebar's model slugs, and prints a config whose
`models` maps every slug to the `ChatPlayground` source (with browser-safe
`model_thread_limit`/`plugin_thread_limit` of 1 and `preload: false`). For raw
selector/model diagnostics, run `python -m benchmark.chatplayground` with the
credentials in the environment; it prints a JSON probe including `models`.

### Per-Source Model Concurrency

`model_thread_limit` controls how many complete target pipelines may run at
once against one source. It is separate from `plugin_thread_limit`, which
controls plugin workers inside one target. The effective request burst can be
approximately their product:

```yaml
model_thread_limit: 1        # top-level fallback
sources:
  AI Server:
    model_thread_limit: 1    # keep local hardware serialized
    plugin_thread_limit: 2
  OpenRouter:
    model_thread_limit: 3    # three target pipelines may overlap
    plugin_thread_limit: 1
```

Resolution is per-source value, then top-level `model_thread_limit`, then `1`.
Only positive integers are accepted; `0`, negative values, booleans, floats,
and numeric strings are configuration errors. FIFO target submission is
preserved, and in `--runner both` OpenCode plus HTTP for one target occupy one
slot as an indivisible pipeline. Increasing both limits can create a much
larger request burst, and parallel cold preloads can exhaust local VRAM, so
keep AI Server/Gaming PC at `1` unless deliberately testing higher values.

Semantic judging uses the same two knobs per source. `model_thread_limit`
bounds how many distinct judge models run concurrently; each active judge
occupies one model slot and runs to completion before another judge is
loaded (keeping one local model resident instead of round-robin swapping).
`plugin_thread_limit` bounds how many cells a single judge scores at once.
For judging, a non-positive `plugin_thread_limit` serializes to one cell per
judge rather than meaning "unlimited", since fanning out an unbounded number
of concurrent judge requests is a resource hazard.
The effective limits and observed `peak_active_models` are written to
`run-info.json`. Response times from overlapping targets are not directly
comparable with a serial run.

### OpenCode Timeout

When using `--runner opencode` or `--runner both`, each source can configure how long OpenCode may remain silent before the subprocess is terminated. The setting measures inactivity on OpenCode's stdout/stderr, so it covers cold model loads and provider/agent stalls without changing the separate benchmark-wide request timeout.

```yaml
sources:
  Gaming PC:
    api_url: http://gaming.pc:11434/chat/completions
    opencode_timeout: 600   # seconds; defaults to 300
  Fast Provider:
    api_url: https://api.example.com/v1/chat/completions
    opencode_timeout: 120
```

`opencode_timeout` must be a non-negative number. Invalid or negative values use the 300-second default. Set it to `0` to disable the inactivity guard for a source; the outer benchmark timeout still applies.

### Pi runner

Pi is an opt-in runner backed by the project-local `pi-worker/` Node adapter. It
requires Node.js 22.19+ and an installed, pinned worker dependency tree:

```sh
npm --prefix pi-worker install
```

Select it with `--runner pi`, or include it in an explicit list such as
`--runners http,pi`. The legacy `--runner both` mode remains the OpenCode-then-HTTP
pipeline; explicit multi-runner lists run independent phases with separate state
keys and output namespaces. Pi does not use model preloading because each cell
creates an isolated SDK worker/session.

Pi-specific target settings are deliberately allowlisted:

```yaml
models:
  local-model:
    source: Local
    pi:
      reasoning: true
      tools: [read, grep]
      permissions:
        read: allow
        grep: deny
      max_tokens: 4096
      max_tool_calls: 50
      system_prompt: null
      compat:
        supportsDeveloperRole: false
```

`tools` accepts only `read`, `bash`, `edit`, `write`, `grep`, `find`, and `ls`;
plain targets default to an empty list. Permissions are `allow` or `deny` and
interactive approval is never requested. `pi.max_tokens` is a per-target scalar
budget and follows the same one-retry policy as HTTP/OpenCode. Tool activity,
permissions, SDK/worker versions, prompt alteration, response nature, and token
estimates are recorded in the Pi result metadata. The adapter sends source URL,
model ID, and configured headers through Pi's OpenAI-compatible provider mapping;
credentials are not included in NDJSON events or diagnostic metadata.

### Per-Source Plugin Concurrency

Each source can define `plugin_thread_limit` to control how many plugins run concurrently for models against that source. The top-level `plugin_thread_limit` is used as a fallback for sources that do not define their own value. The CLI `--plugin-thread-limit` overrides all sources.

### Model Preloading

Model preloading is opt-in per source. When enabled, the benchmark sends one direct, non-streaming probe before a source reaches a model's first benchmark leg. The probe uses `Reply with the single word OK.`, a small output budget, and does not contribute to test response times, TTFT, TPS, or `total_time`. A successful probe warms the backend so the real leg does not pay cold-load latency. If the probe fails or returns empty content, both runner legs for that model are recorded as failed with a preload error and the source advances to its next model.

```yaml
sources:
  Gaming PC:
    api_url: http://gaming.pc:11434/chat/completions
    preload: true
    preload_timeout: 300   # seconds; defaults to 300
```

`preload_timeout` is independent of the normal benchmark `timeout` and must be positive. Invalid or non-positive values use the 300-second default. The probe is one-shot and disables HTTP 429 retries so an unavailable model is classified promptly. Preload is performed once per `(source, api_model)` per process; in `--runner both`, the HTTP leg reuses the model warmed before OpenCode.

Pass `--no-preload` to disable all source preload settings for a run. Preload status and timing are shown in the TUI and summarized under `preload` in `run-info.json`.

### HTTP 429 Retry / Backoff

Each source can opt into automatic retries for HTTP 429 (Too Many Requests) responses.

```yaml
sources:
  Google:
    api_url: ...
    headers: ...
    max_429_retries: 2        # default 2; set 0 to fail-fast (legacy)
    backoff_seconds: 30       # first-sleep default
    backoff_factor: 2.0       # exponential growth per retry
    max_backoff_seconds: 300  # hard cap per sleep
```

Defaults are **opt-out**: every source retries up to twice by default. To restore the previous fail-fast behaviour, set `max_429_retries: 0` per source or pass `--no-retry-on-429` globally. Explicit per-source `max_429_retries` values are preserved by the global toggle. When a 429 is returned, the runner sleeps `max(Retry-After, backoff_seconds * backoff_factor ** attempt)` bounded by `max_backoff_seconds`, with ±20 % jitter applied only when `Retry-After` is absent. Each retry is logged to that model's `logs/<model>.log` and `stop_event` cancels the sleep immediately, so Ctrl+C still terminates the runner quickly. During a model's run, two consecutive plugin tests that exhaust their 429 retries trip a model-local circuit breaker: the remaining plugin tests are cancelled, and the model is recorded as failed with the circuit-breaker reason. A successful or non-429 test resets that consecutive count; one isolated 429 does not cancel the model. With parallel plugin workers, already-running requests may finish or observe cancellation, but queued work is prevented from issuing another request once the breaker trips.

### Live Stream Watchdog

Per-source live-stream guardrails abort a request the moment it becomes
pathological, instead of letting it burn its entire `max_tokens` budget
(and the source's model slot) over 30-60 minutes on local hardware.

The watchdog splits the budget by stream: `reasoning_content` (thinking)
and final `content` are tracked separately as the SSE deltas arrive, and
the request is aborted as soon as either exceeds its token budget
(`len(text) / 4`, the same estimator used for final token counts).

The repetition guard is deliberately more selective than the post-hoc
`_repeating` flag: aborting a live stream is destructive (the slot frees,
the partial text is scored), so it only fires on a *dense echo loop* — the
newest 80-char block must appear at least 3 times in the recent ~4K
chars AND the previous repeat must end within ~256 chars of the stream
tail (a real loop re-emits its last phrase immediately). It also ignores
blocks that are mostly typographic decoration (ASCII diagram borders).
Legitimate recurring structure — three generated classes sharing an
`__init__` scaffold, repeated `+---+` diagram borders — is left alone and
is bounded instead by the split token budgets.

```yaml
sources:
  AI Server:
    api_url: ...
    headers: ...
    max_thinking_tokens: 32768    # reasoning_content cap (default 32768)
    max_content_tokens: 16384     # final-content cap (default 16384)
    repetition_guard: true        # abort on repetition loops (default true)
```

An aborted request is recorded as an error on the plugin cell with the
reason in the stream error / `meta.json` (`Content budget exceeded (N
tokens)`, `Thinking budget exceeded (N tokens)`, or `Repetition detected
in content|thinking — stream aborted`); any text streamed before the abort
is retained for scoring and diagnosis. Budget-aborted attempts never trip
the thinking-truncation auto-escalation (the abort is an error, not a
`thinking-truncation` classification). The watchdog applies to the HTTP
benchmark task path only — preload probes and judge requests do not use the
benchmark task watchdog (although judge requests are streamed and honor the
shared shutdown cancellation). OpenCode has its own subprocess timeout and loop guards; those outcomes feed the shared attempt metadata but do not use the HTTP stream watchdog.

### Environment Variable Expansion

Any string value in the config supports `${VAR}` and `${VAR:default}` syntax
(headers, `email`/`password`, etc.), expanded when the config is loaded:

```json
"Authorization": "Bearer ${OPENAI_API_KEY:sk-fallback-key}"
"email": "${CHATPLAYGROUND_EMAIL}"
```

The value is replaced with the environment variable value, or the default if
the variable is unset. The variables may come from the process environment or
from a local `.env` file (see below).

### Dotenv Files

The CLI loads a `.env` file from the current working directory at startup
(using `python-dotenv`), so credentials can live in a file instead of being
exported into the shell:

```sh
# .env
CHATPLAYGROUND_EMAIL=you@example.com
CHATPLAYGROUND_PASSWORD=...
OPENAI_API_KEY=sk-...
```

```sh
python ai-benchmark.py --chatplayground-config > benchmark-config.json
```

- A missing `.env` is ignored.
- Variables already present in the real environment take precedence over the
  `.env` values (dotenv's `override=False` default).
- Only `./.env` in the current directory is read; no parent directories are
  searched.

## Models

The `models` map supports two forms.

### Simple Form

```json
"models": {
  "llama3:8b": "Local Server"
}
```

The value is the source name from the `sources` section.

### Extended Form

```json
"models": {
  "llama3:8b": {
    "source": "Local Server",
    "drop_params": ["seed"]
  }
}
```

| Key | Type | Description |
|---|---|---|
| `source` | string | Source name from `sources` |
| `drop_params` | list[string] | Request body keys to omit for this model |
| `max_tokens` | integer | (optional) Per-model generation budget, beats the global `max_tokens` |

## Per-Model Generation Budget

Thinking-capable models (deepseek-r1/qwen/o1-class) can consume their entire `max_tokens` budget inside `reasoning_content` before a single content token lands, yielding an empty response that scores 0 (see `empty-content-investigation.md`). Give those models a larger budget with a per-target override:

```json
"models": {
  "qwen3.5:9b-32k": {
    "source": "Gaming PC",
    "max_tokens": 32768
  }
}
```

Or keep the `models` map simple and use the top-level `model_max_tokens` map, keyed by target name or `"{source}/{api_model}"`:

```json
{
  "model_max_tokens": {
    "qwen3.5:9b-32k": 32768,
    "Gaming PC/deepseek-v4-flash-free": 32768
  }
}
```

Precedence: per-target `max_tokens` inside the model/agent entry > `pi.max_tokens` for Pi targets or the `model_max_tokens` map (matched by target name, then `{source}/{api_model}`) > global `max_tokens` / `--max-tokens`. The same scalar budget is applied to the target's OpenCode and Pi legs.

### One-retry policy and attempt metadata

Each benchmark task makes at most one benchmark-level retry and reuses the same scalar budget. Transport errors retry with the original prompt. A token-limit response retries with one of `thinking_50_percent`, `thinking_30_percent`, or `response_under_budget`, selected from observed thinking usage; a repetition abort retries with `avoid_repetition`. Timeouts and cancellation do not retry.

The selected attempt is projected into the plugin result fields, while every attempt is retained in `{plugin}_attempts` and in the response `.meta.json`. The machine-readable summary includes `{plugin}_attempt_count`, `{plugin}_retry_count`, `{plugin}_retry_reasons`, `{plugin}_selected_attempt`, `{plugin}_prompt_altered`, `{plugin}_response_nature`, `{plugin}_truncated_due_to_time`, and `{plugin}_failure_cause`. The CSV includes these fields plus the complete attempt history as JSON, allowing reports to distinguish model behavior from transport, timeout, repetition, or evaluator failures without parsing logs.

## Semantic Judge Request Parameters

The optional `judge.request_params` object is merged into every HTTP request
made by a semantic judge. It supports provider-specific OpenAI-compatible
parameters that are not part of the benchmark's normal model request. The
built-in default requests a JSON object matching a strict schema and does
**not** send a thinking-token budget or otherwise change the model's native
thinking behavior:

```yaml
judge:
  max_tokens: 4096
  request_params:
    response_format:
      type: json_schema
      json_schema:
        name: benchmark_judge_result
        strict: true
        schema:
          type: object
          additionalProperties: false
          required: [score, confidence, rationale, criteria]
          properties:
            score: {type: integer, minimum: 0, maximum: 100}
            confidence: {type: string, enum: [high, medium, low]}
            rationale: {type: string}
            criteria:
              type: array
              items:
                type: object
                additionalProperties: false
                required: [id, criterion, status, evidence]
                properties:
                  id: {type: string}
                  criterion: {type: string}
                  status: {type: string, enum: [met, partial, not_met, not_applicable]}
                  evidence: {type: string}
```

The benchmark also does not send `enable_thinking: false`; operators who need
to control thinking explicitly can add provider-supported fields under
`judge.request_params`. The judge prompt (version `judge-v8`) presents the
task and candidate answer as explicitly delimited, quoted data and tells the
judge not to follow candidate instructions, emit tool calls, continue the
embedded task, or reproduce any fragment of it. It also requires exactly one
JSON object with no surrounding text. The prompt asks the judge to keep its rationale, criterion descriptions, and
evidence concise; these guidance limits are intentionally not encoded as grammar
string-length bounds because long bounded strings can exceed llama.cpp's
grammar-construction limits. If a retry follows a response that exhausted its generation budget on reasoning, the retry prompt additionally asks the judge to spend approximately half the budget thinking and reserve the remainder for the JSON answer. Each criterion records the judge's interpretation
of one explicit requirement, whether the candidate met it, and concise evidence
for that determination. Plugins may override
`sanitize_for_judge` (see `docs/plugins.md`) to mask structured fragments -
such as tool-calling's `<tool_call>` blocks - before they reach the judge. Nested dictionaries are merged, so explicit provider-specific options
can be combined safely. Use the relevant provider's supported request fields;
unsupported fields can be removed with the judge model's `drop_params`
configuration.

### Schema portability (llama.cpp and Ollama)

The built-in judge and Data Transformation schemas intentionally use a
conservative intersection of the documented local Ollama and llama.cpp
features: objects, arrays with `items`, required properties,
`additionalProperties: false`, primitive types, enums, anchored patterns, and
bounded integer values. They avoid `$ref`, `oneOf`/`anyOf`, `format`,
`patternProperties`, conditionals, lookarounds, regex shorthand such as `\\d`,
and string length bounds.

Ollama's native `/api/chat` endpoint accepts a schema object in `format`; its
OpenAI-compatible `/v1/chat/completions` endpoint documents `response_format`,
but support for the `json_schema` wrapper is version-dependent. For an Ollama
installation that ignores or rejects the wrapper, use the native endpoint or
configure the proxy to translate `response_format.json_schema.schema` to
Ollama's `format` field. Ollama Cloud does not currently support structured
outputs. Regardless of provider enforcement, the benchmark validates the
returned JSON and applies the semantic/plugin checks after generation.

### Data-transformation schema sentinel

The Data Transformation task records schema compatibility separately from its
semantic score. Its schema contract is worth at most one of the 22 task
points; extraction, normalization, and current-record selection provide the
remaining discrimination. Per-plugin result metadata includes
`schema_requested`, `schema_request_status`, `response_schema_valid`, and
`schema_enforcement_verified`. The last value remains false for ordinary
benchmark cells because a valid response alone cannot prove that the provider
enforced the schema.

Use the separately-run sentinel for an operational check:

```sh
python ai-benchmark.py --config benchmark-config.json --schema-sentinel
```

It does not create benchmark results or affect scores. The sentinel asks for a
value forbidden by the prompt but permitted by the schema, allowing the run to
classify likely enforcement separately from request rejection, transport
failure, or an invalid response. Native 1min.ai and ChatPlayground protocols
are reported as `schema_not_supported_by_source` because they do not use the
OpenAI `response_format` request field.

`judge.max_tokens` controls the total `max_tokens` cap for judge generation;
the default is `4096`. It is independent of any provider-specific thinking
setting an operator adds to `judge.request_params`.

The judge prompt uses a fixed authority hierarchy, a requirement-by-requirement
checklist, a least-restrictive ambiguity policy, and a finalization checklist to
reduce overthinking, speculation, prompt distraction, and score drift. Judge
criterion reports are persisted in each plugin's `*_judge_votes` and
`*_judge_criteria` state fields, exposed as `{plugin}_Judge_Criteria_JSON` in
CSV, rendered in the Markdown/HTML/PDF detailed sections, and summarized in
`run-info.json` under `judge_criteria`. They are diagnostic and do not create a
second score. The legacy per-plugin fields such as `{plugin}_judge_score` are
an explicit projection of the active contract, recorded as
`{plugin}_judge_selected_contract`; historical contracts remain available in
`{plugin}_judge_consensus_by_contract` and the versioned vote list. The run
metadata identifies this policy as `judge_projection: "active-contract"`.

The default `response_format` uses the standard OpenAI-compatible `json_schema`
form, which combines JSON mode with the expected judge-result schema. The judge
score is an integer because the parser rounds fractional scores and llama.cpp
only grammar-enforces numeric bounds for integer schemas. CI also runs
`tests/test_schema_grammar_compat.py`, which validates the schemas against a
conservative Ollama/llama.cpp intersection: no unsupported keywords,
no string-length grammar bounds, anchored backslash-free patterns,
integer-only numeric bounds, and explicit `additionalProperties: false` on
objects. This is stronger than `json_object` alone: the provider can constrain
not only the outer format but also the required fields and value types. A
provider that only supports `json_object` can override `judge.request_params`,
but that removes schema enforcement and should be verified empirically.

Each HTTP request log now records the exact merged request body used for the
`requests.post` call in its replayable curl command, including any explicit
provider-specific parameters. Judge response `.meta.json` files additionally
record `request_params`, `request_max_tokens`, `response_finish_reason`,
`response_usage`, `response_reasoning_tokens`, and `thinking_budget_honored`.
The last field is `true` or `false` only when an explicit thinking budget was
requested and the provider reports reasoning-token usage (or it can be
estimated from `reasoning_content`); otherwise it is `null`.

## Per-plugin capability metadata

Scores are useful for ranking, but a model-selection benchmark should also
retain evidence about *why* a cell succeeded or failed. The current structured
metadata pattern is intended to generalize to every plugin without creating a
second hidden score:

- **Contract metadata:** requested format, parser status, schema/tool-call
  validity, and whether provider enforcement was independently probed.
- **Semantic evidence:** normalized entities, matched requirements, missing
  requirements, contradictions, and negative findings rather than only a
  keyword count.
- **Execution evidence:** syntax/compiler status, test exit code, timeout,
  resource use, generated artifacts, and isolated-test diagnostics.
- **Reliability metadata:** transport errors, EOFs, 429s, retries, truncation,
  repetition aborts, empty-response reason, and whether the result was
  resumed or reused.
- **Interaction metadata:** turn count, state carried between turns, tool-call
  validity, argument correctness, and whether the model recovered after a
  tool/error message.
- **Capability vectors:** record dimensions such as extraction, planning,
  coding, debugging, long-context retrieval, instruction adherence, and
  adversarial robustness so model selection can use strengths and weaknesses
  instead of one global mean.
- **Composition signals:** identify complementary models—for example one with
  strong planning but weak execution, and another with reliable code tests—so
  future agent pipelines can route, verify, or critique rather than simply
  choose the highest aggregate score.

These fields should stay diagnostic and machine-readable. Keep the primary
plugin score task-specific, and aggregate capability rates only when the
underlying evidence is comparable. This prevents an API/schema incompatibility
from being mislabeled as a reasoning failure while still making an unusable
model/source combination visible.

## Thinking-truncation handling


When a streaming response reaches `finish_reason="length"` after spending most of its budget on reasoning, the shared one-retry policy retries with the same `max_tokens` value and adds `prompt_altered: "thinking_50_percent"`. Responses that consume between half and 80% of the budget use `thinking_30_percent`; responses with little or unavailable thinking use `response_under_budget`. The selected response and the original thinking-truncation attempt remain available in the per-plugin metadata and attempt history. The benchmark does not silently double the configured budget.


## Agents

The `agents` block lets you test models with a fixed system prompt. Each agent is a named wrapper around an model and source.

```json
"agents": {
  "my-coding-agent": {
    "model": "gpt-4o",
    "source": "Remote Provider",
    "system_prompt": "You are an expert Python programmer. Be concise and accurate."
  }
}
```

| Key | Type | Description |
|---|---|---|
| `model` | string | Model string sent to the API |
| `source` | string | Source name from `sources` |
| `system_prompt` | string | System prompt prepended to every plugin request |
| `drop_params` | list[string] | (optional) Request body keys to omit for this agent |
| `plugins_blacklist` | list[string] | (optional) Plugin IDs to skip for this agent |

Agents and models coexist in the same config. The benchmark treats each entry as a distinct target. Results include `api_model`, `is_agent`, and `system_prompt` metadata for every target.

### Per-Model Parameter Dropping

Use `drop_params` to omit parameters that a particular model or provider does not support. Common examples:

```json
"models": {
  "model-a": {
    "source": "Remote Provider",
    "drop_params": ["seed"]
  },
  "model-b": {
    "source": "Another Provider",
    "drop_params": ["seed", "temperature"]
  }
}
```

## Per-Plugin Temperature

You can set the temperature for each plugin using either of these config keys:

```json
{
  "rate-limiter_temperature": 0.2,
  "moe-dense_temperature": 0.7,
  "code-review_temperature": 0.3,
  "orchestration_temperature": 0.5,
  "decomposition_temperature": 0.5,
  "tool-calling_temperature": 0.2,
  "data-transformation_temperature": 0.2,
  "instruction-following_temperature": 0.2,
  "reasoning_temperature": 0.1
}
```

## Converting Between Formats

Use the CLI to convert a config between JSON and YAML:

```sh
# Convert YAML config to JSON
python ai-benchmark.py --convert-config benchmark-config.yaml > benchmark-config.json

# Convert JSON config to YAML
python ai-benchmark.py --convert-config benchmark-config.json > benchmark-config.yaml
```

The converted output is printed to stdout. Environment variables in the config are expanded before conversion.

## YAML Configs

YAML configs work the same way as JSON configs. For example:

```yaml
output_dir: benchmark-results
sources:
  Local Server:
    api_url: http://localhost:11434/chat/completions
    headers:
      Authorization: "Bearer ${LOCAL_API_KEY:sk-fallback}"
      Content-Type: application/json
models:
  llama3:8b: Local Server
agents:
  my-coding-agent:
    model: gpt-4o
    source: Local Server
    system_prompt: You are an expert coder.
```

## Complete Example

```json
{
  "output_dir": "benchmark-results",
  "timeout": 600,
  "max_tokens": 16384,
  "rate-limiter_temperature": 0.2,
  "moe-dense_temperature": 0.7,
  "plugins_whitelist": [],
  "plugins_blacklist": [],
  "sources": {
    "Local Server": {
      "api_url": "http://localhost:11434/chat/completions",
      "headers": {
        "Authorization": "Bearer ${LOCAL_API_KEY:sk-fallback}",
        "Content-Type": "application/json"
      },
      "plugin_thread_limit": 1,
      "opencode_timeout": 300
    },
    "Remote Provider": {
      "api_url": "https://api.example.com/v1/chat/completions",
      "headers": {
        "Authorization": "Bearer ${REMOTE_API_KEY}",
        "Content-Type": "application/json"
      }
    }
  },
  "models": {
    "llama3:8b": "Local Server",
    "gpt-oss:120b-128k": {
      "source": "Remote Provider",
      "drop_params": ["seed"]
    }
  }
}
```

## Model/Agent Name Collisions

If a key appears in both `models` and `agents`, the benchmark exits with an error. Rename the model or agent to continue.

## Notes

- `max_tokens` is one scalar generation budget. A task may make at most one benchmark retry with the same budget; token exhaustion, transport errors, repetition aborts, timeout, cancellation, the selected attempt, and any prompt alteration are recorded in per-plugin metadata and attempt history.
- Empty responses are classified (`empty_reason` in `meta.json`, `{pid}_Empty_Reason` in `results.csv`, and a `Reason` column in `results.html`/`results.md`): `error`, `thinking-truncation` (budget consumed by reasoning — retried with the same budget and thinking guidance), `thinking-only`, `max-tokens`, or `empty`.
- `plugin_thread_limit` controls how many plugins run concurrently for each model against a given source. Set to `1` for sequential execution or `0` for maximum parallelism. Define it per-source or as a top-level fallback.
