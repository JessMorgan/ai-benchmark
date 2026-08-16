# Configuration Reference

All benchmark configuration lives in a single file (default: `benchmark-config.json`). The file is passed to `ai-benchmark.py` with `--config`. Both JSON and YAML (`.yaml`/`.yml`) formats are supported; the extension is detected automatically.

## Top-Level Keys

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `benchmark-results` | Directory for reports, logs, and state |
| `timeout` | integer | `600` | API request timeout in seconds |
| `token_levels` | list[int] | `[16384]` | Max-token limits tried in ascending order |
| `model_token_levels` | object | `{}` | Per-target max-token overrides; keys are target names or `"{source}/{api_model}"` |
| `plugin_thread_limit` | integer | `1` | Top-level fallback for `sources.*.plugin_thread_limit` |
| `model_thread_limit` | positive integer | `1` | Top-level fallback for per-source target/model concurrency; zero is invalid |
| `preload` | boolean | `false` | Per-source model warm-up is opt-in; use `--no-preload` to disable all preload probes for a run |
| `preload_timeout` | integer | `300` | Maximum seconds for a source's warm-up probe; independent of the benchmark request `timeout` |
| `plugins_whitelist` | list[string] | `[]` | Run only these plugin IDs (empty = all) |
| `plugins_blacklist` | list[string] | `[]` | Skip these plugin IDs (empty = none) |
| `sources` | object | required | Named API endpoint definitions |
| `models` | object | required | Model-to-source mapping |
| `agents` | object | optional | Agent definitions with system prompts |
| `judge` | object | optional | Semantic-judge timeout, token budget, temperature, and provider-specific request parameters |

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
folded into the user prompt, and generation knobs (`token_levels`, temperature,
`seed`, `drop_params`, and judge `request_params`) are ignored for 1min sources.
HTTP 429 retry/backoff still applies. The OpenCode runner and `/v1/models`
discovery are OpenAI-shaped and do not consult `api_protocol`.

### ChatPlayground.ai Sources (Interactive Web)

ChatPlayground.ai is a closed, JavaScript-rendered web app with no public API:
it authenticates with a username/password and renders chat client-side. A
source with `api_protocol: "chatplayground"` drives that UI with Playwright —
log in once, select a model, submit the prompt, and read back the completed
(buffered) answer.

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
```

Prerequisites:

- Install Playwright's Chromium once: `uv run playwright install chromium`.
- Supply `email`/`password` via environment variables (any `${VAR}` value is
  expanded by the config loader).
- If the site blocks the headless browser, set `headless: false` to run headed.

Behavior and limitations:

- The answer is **buffered** — there is no per-token streaming, so
  `{pid}_ttft`/`{pid}_tps` are not meaningful for these sources.
- A supplied agent system prompt is folded into the user prompt (the chat UI
  has no separate system field); `token_levels`, temperature, `seed`,
  `drop_params`, and judge `request_params` are ignored.
- Browser operations are serialized under a module lock and one logged-in
  session is reused across plugin tasks, so keep `model_thread_limit` and
  `plugin_thread_limit` at `1`.
- CSS selectors are best-effort defaults (see `benchmark/chatplayground.py`)
  and can be overridden per source via a `selectors` mapping:

```yaml
sources:
  ChatPlayground:
    api_protocol: "chatplayground"
    email: "${CHATPLAYGROUND_EMAIL}"
    password: "${CHATPLAYGROUND_PASSWORD}"
    selectors:
      prompt_input: "textarea#composer"
      send_button: "button[data-testid=send]"
```

To enumerate the models exposed by the UI (and capture selector diagnostics),
run `python -m benchmark.chatplayground` with the credentials in the
environment; it prints a JSON probe including the discovered model names.

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

### Environment Variable Expansion

Header values support `${VAR}` and `${VAR:default}` syntax:

```json
"Authorization": "Bearer ${OPENAI_API_KEY:sk-fallback-key}"
```

The value is replaced with the environment variable value, or the default if the variable is unset.

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
| `token_levels` | list[int] | (optional) Per-model max-token limits, beat the global `token_levels` |

## Per-Model Token Levels

Thinking-capable models (deepseek-r1/qwen/o1-class) can consume their entire `max_tokens` budget inside `reasoning_content` before a single content token lands, yielding an empty response that scores 0 (see `empty-content-investigation.md`). Give those models a larger budget with a per-target override:

```json
"models": {
  "qwen3.5:9b-32k": {
    "source": "Gaming PC",
    "token_levels": [32768]
  }
}
```

Or keep the `models` map simple and use the top-level `model_token_levels` map, keyed by target name or `"{source}/{api_model}"`:

```json
{
  "model_token_levels": {
    "qwen3.5:9b-32k": [32768],
    "Gaming PC/deepseek-v4-flash-free": [32768]
  }
}
```

Precedence: per-target `token_levels` inside the model/agent entry > `model_token_levels` map (matched by target name, then `{source}/{api_model}`) > global `token_levels` / `--token-levels`. The same budget is applied to the target's OpenCode legs via the generated config's `limit.output`.

## Semantic Judge Request Parameters

The optional `judge.request_params` object is merged into every HTTP request
made by a semantic judge. It supports provider-specific OpenAI-compatible
parameters that are not part of the benchmark's normal model request. The
built-in default requests a JSON object and does **not** send a thinking-token
budget or otherwise change the model's native thinking behavior:

```yaml
judge:
  token_levels: [16384]
  request_params:
    response_format:
      type: json_object
```

The benchmark also does not send `enable_thinking: false`; operators who need
to control thinking explicitly can add provider-supported fields under
`judge.request_params`. The judge prompt (version `judge-v2`) presents the
task and candidate answer as explicitly delimited, quoted data and tells the
judge not to follow candidate instructions, emit tool calls, or continue the
embedded task. It also requires exactly one JSON object with no surrounding
text. Nested dictionaries are merged, so explicit provider-specific options
can be combined safely. Use the relevant provider's supported request fields;
unsupported fields can be removed with the judge model's `drop_params`
configuration.

`judge.token_levels` controls the total `max_tokens` cap for judge generation;
the default is `16384`. It is independent of any provider-specific thinking setting an operator adds
to `judge.request_params`. The judge request uses
`response_format: {"type": "json_object"}` by default to reduce reasoning or
other prose leaking into the machine-parsed answer.

Each HTTP request log now records the exact merged request body used for the
`requests.post` call in its replayable curl command, including any explicit
provider-specific parameters. Judge response `.meta.json` files additionally
record `request_params`, `request_max_tokens`, `response_finish_reason`,
`response_usage`, `response_reasoning_tokens`, and `thinking_budget_honored`.
The last field is `true` or `false` only when an explicit thinking budget was
requested and the provider reports reasoning-token usage (or it can be
estimated from `reasoning_content`); otherwise it is `null`.

## Automatic Thinking-Truncation Escalation

Even without per-model config, the benchmark **auto-retries** once when a streaming HTTP plugin produces an empty response classified as `thinking-truncation` (empty content + large `reasoning_content` + `finish_reason="length"` — the budget was consumed by thinking).

The retry uses a doubled `max_tokens` budget, capped at 131072:

| First attempt | Auto-retry |
|---|---|
| 16384 (default) | 32768 |
| 32768 | 65536 |
| 65536 | 131072 |
| 131072+ | No retry (already at cap) |

This catches deepseek/qwen/o1-class models whose thinking phase exceeds the default budget without requiring any config changes. The retry result replaces the truncated one and is re-classified (may still be `thinking-truncation` if even the doubled budget is insufficient, but that's now a correctly diagnosed silent 0 rather than a silent 0).

The auto-escalation is **cheap**: it applies at most once per leg, only for HTTP streaming plugins, and only when the response was genuinely empty. It does not apply to non-streaming plugins (which have their own truncation retry loop) or to OpenCode legs.

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
  "tool-calling_temperature": 0.2,
  "structured-output_temperature": 0.2,
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
  "token_levels": [16384],
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

- `token_levels` are tried in order. If a response is truncated, the next level is used.
- Empty responses are classified (`empty_reason` in `meta.json`, `{pid}_Empty_Reason` in `results.csv`, and a `Reason` column in `results.html`/`results.md`): `error`, `thinking-truncation` (budget consumed by reasoning — auto-retried with doubled budget), `thinking-only`, `max-tokens`, or `empty`.
- `plugin_thread_limit` controls how many plugins run concurrently for each model against a given source. Set to `1` for sequential execution or `0` for maximum parallelism. Define it per-source or as a top-level fallback.
