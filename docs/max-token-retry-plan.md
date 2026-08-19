# Max-token handling and retry redesign

**Status:** Planned; not implemented by this document

## Goal

Replace the current list-based token-level retry behavior with one explicit generation budget and a predictable, observable retry policy for the main benchmark task paths. The redesign applies to both HTTP tasks and OpenCode tasks. It should make token exhaustion, transport failure, timeout/cancellation, repetition aborts, prompt alterations, and the selected response easy to distinguish in reports.

Judge-specific retry behavior is separate unless a later change explicitly brings judges under this policy.

## 1. Replace `token_levels` with one scalar `max_tokens`

Use one value everywhere a benchmark task generation budget is configured:

```yaml
max_tokens: 4096
```

Replace list-based configuration such as:

```yaml
token_levels: [4096, 8192, 16384]
```

and list-oriented CLI behavior with a scalar `max_tokens` setting/flag. The same scalar should flow through:

- Global benchmark configuration.
- Per-source, per-model, and per-agent overrides where those overrides currently exist.
- HTTP streaming and non-streaming requests.
- OpenCode task execution and generated limits.
- Default-config output.
- Configuration validation and documentation.

### Migration policy

Remove `token_levels`; do **not** silently convert it, preserve it as a deprecated alias, or infer a value from its list. A configuration containing `token_levels` should fail validation with an actionable message directing the operator to `max_tokens`.

The retry uses the same `max_tokens` value as the initial request. It does not increase the budget and does not derive a second budget from a list.

## 2. Unify the main benchmark attempt policy

For each benchmark task, make at most two benchmark-level attempts:

1. Initial attempt.
2. At most one retry, selected by the failure/end-of-response classification.

HTTP provider-level retries, such as 429 backoff, remain a separate mechanism. They should not consume the benchmark-level retry slot unless the existing policy explicitly says otherwise, and their statistics must remain distinguishable from benchmark attempts.

Schema-compatibility fallbacks, if still supported for a request path, must also remain separately identified rather than being mislabeled as a token or transport retry.

Both streaming and non-streaming HTTP paths should implement the same classification and selection policy. OpenCode should expose equivalent metadata as far as its subprocess interface permits.

## 3. Classify every attempt

Persist an attempt history for every benchmark task. Each attempt should include machine-readable information similar to:

```json
{
  "attempt": 1,
  "max_tokens": 4096,
  "prompt_altered": "none",
  "retry_reason": null,
  "response_nature": "token_limit",
  "finish_reason": "length",
  "transport_error": null,
  "content_tokens": 120,
  "thinking_tokens": 3300,
  "total_tokens": 3420,
  "usable": true,
  "selected": false
}
```

Exact field names should follow existing metadata conventions where possible, but the information must be available without parsing log text.

### Proposed response-nature values

Use a stable enum, or a documented equivalent, for how an attempt ended:

- `completed`
- `token_limit`
- `timeout`
- `cancelled`
- `transport_error`
- `repetition_abort`
- `empty`
- `plugin_error`

The response nature describes the attempt's result. It is separate from `retry_reason`, which describes why a subsequent attempt was made.

Record, where available:

- Requested `max_tokens`.
- Finish reason.
- Content, thinking, and total token estimates or provider-reported usage.
- Whether content was present and whether it was usable.
- Transport/stream error details.
- Timeout or cancellation details.
- Repetition detection details.
- Prompt hash or equivalent identity for comparing the original and altered prompts.
- Whether this attempt was selected for scoring/reporting.

## 4. Retry policy

### Transport error

Retry once when the attempt fails because of a retryable transport or stream error:

- Reuse the exact original prompt.
- Reuse the same `max_tokens`.
- Reuse the same request parameters.
- Set `retry_reason` to `transport_error`.
- Set `prompt_altered` to `none`.

A timeout or cancellation must not be collapsed into this category.

### Token exhaustion

When an attempt ends at the token limit, retry once with the same budget and a prompt alteration chosen from the observed thinking usage:

| Thinking usage | Retry alteration | Meaning |
|---|---|---|
| `>= 80%` of `max_tokens` | `thinking_50_percent` | Ask the model to keep thinking to approximately half of the token budget so room remains for the answer. |
| `> 50%` and `< 80%` | `thinking_30_percent` | Ask the model to keep thinking to approximately 30% of the token budget. |
| `<= 50%`, or unavailable | `response_under_budget` | Ask the model to complete the answer while keeping the total response just below the token limit. |

The retry instruction should be added to the request prompt rather than changing the configured system prompt. It should state the applicable approximate budget clearly while avoiding a provider-specific hard guarantee.

Set `retry_reason` to `token_limit` and use the corresponding `prompt_altered` enum value. The attempt history must retain the original attempt and the altered retry attempt.

### Repetition abort

If repetition detection aborts an attempt and a retry is likely to help, retry once with a no-repetition instruction. Mark this explicitly:

```json
{
  "retry_reason": "repetition_abort",
  "prompt_altered": "avoid_repetition"
}
```

If a no-repetition retry is not beneficial for a particular repetition classification, do not retry. Record that the attempt ended in `repetition_abort` and why no retry was selected.

### Timeout and cancellation

Timeouts and cancellations are non-retryable:

- Do not issue a second benchmark attempt.
- Preserve whatever response content was received.
- Mark the response as truncated due to time when generation stopped before a complete response.
- Use `response_nature: "timeout"` or `"cancelled"` as appropriate.
- Store the timeout/cancellation cause and partial-response status in metadata.

This rule must apply consistently to streaming, non-streaming, and OpenCode execution. A watchdog or subprocess timeout must not fall through to a generic transport retry.

## 5. Select the best usable attempt

Do not automatically prefer the retry. Evaluate all completed attempts and select the best usable response according to the existing benchmark/plugin evaluation rules.

Persist:

- `selected_attempt`: the attempt number chosen for scoring/reporting.
- `selected`: a boolean on each attempt, with exactly one selected when at least one usable attempt exists.
- The selected response's score, timing, token metrics, and output flags.

If the first attempt is usable and better than the retry, select the first attempt. If only the retry is usable, select the retry. If neither is usable, preserve the best available partial/failed result according to existing failure semantics.

## 6. Metadata summary versus attempt history

Store all attempts plus a summary for easy report generation.

### Attempt history

The history should preserve the details above for each attempt, including retry cause and prompt alteration. It must be possible to answer:

- How many attempts were made?
- Why was the retry made?
- Was the prompt altered, and how?
- Did the response end from a token limit, timeout, cancellation, transport error, repetition abort, or normal completion?
- Which attempt was selected?
- How much thinking and content did each attempt use?

### Top-level summary

Top-level fields must describe the selected attempt, not merely the last attempt. In particular:

- `prompt_altered` is the selected attempt's enum value.
- It must be `none` if the first attempt was selected and no alteration affected that response.
- `response_nature`, finish reason, truncation state, token metrics, and failure details describe the selected response.
- Include a top-level retry summary such as `retry_count`, `retried`, and `retry_reason` for report-friendly aggregation.
- Include the selected attempt number and whether the selected response was partial.

Do not claim that the selected response used an altered prompt merely because an unselected retry used one.

Suggested prompt-alteration enum values:

- `none`
- `thinking_50_percent`
- `thinking_30_percent`
- `response_under_budget`
- `avoid_repetition`

If multiple modifications ever become necessary, add a structured list in attempt history rather than overloading the enum. For the initial implementation, one alteration per retry is expected.

## 7. Integration details

### HTTP streaming

- Preserve streaming token/content/thinking accounting.
- Ensure finish reason, stream errors, watchdog timeout, cancellation, and repetition abort are classified before deciding whether to retry.
- Do not convert a timeout into a non-streaming fallback that violates the no-retry rule.
- Keep 429 cleanup/backoff behavior and per-plugin 429 statistics separate from benchmark-attempt metadata.

### HTTP non-streaming

- Normalize response classification to the same attempt schema.
- Preserve provider usage fields when present and use existing estimates only when necessary.
- Apply the same one-retry limit and prompt-alteration policy.

### OpenCode

- Pass the scalar `max_tokens` through the OpenCode runner.
- Classify subprocess output, timeout, cancellation, and transport-like failures into the shared attempt model.
- Retry only when the shared policy permits it.
- Preserve partial output for non-retryable timeout/cancellation.

### Configuration and validation

- Remove `token_levels` from default configuration generation, parsing, validation, and documentation.
- Reject `token_levels` explicitly rather than converting it.
- Update per-source/per-model override validation and any CLI completion/help text.
- Keep explicit `max_tokens` overrides intact.

## 8. Testing plan

Add focused regression tests for:

1. Scalar `max_tokens` is used for the initial request and retry.
2. `token_levels` is rejected and is never silently converted.
3. Transport error retries once with an unchanged prompt and unchanged request parameters.
4. Token exhaustion with thinking `>= 80%` uses `thinking_50_percent`.
5. Token exhaustion with thinking `> 50%` and `< 80%` uses `thinking_30_percent`.
6. Token exhaustion with thinking `<= 50%` or unavailable uses `response_under_budget`.
7. Token exhaustion permits only one benchmark retry.
8. Timeout does not retry and records time truncation.
9. Cancellation does not retry and records cancellation.
10. Repetition abort retries only when configured/useful and records `avoid_repetition`.
11. First attempt can be selected over a usable but worse retry.
12. Retry can be selected when the first attempt is unusable.
13. Top-level metadata reflects the selected attempt rather than the last attempt.
14. Attempt history retains both the original and altered prompts/identities and retry causes.
15. Streaming and non-streaming HTTP paths produce equivalent classifications.
16. OpenCode timeout, cancellation, retry, selection, and metadata behavior follows the same policy.
17. Existing 429 backoff and schema-compatibility diagnostics remain separate.
18. Report/output generation can summarize retry count, retry cause, response nature, prompt alteration, and selected attempt without parsing logs.

Run the complete existing test, typecheck, lint, compilation, and pre-commit suites after implementation.

## 9. Documentation and reporting

Update the configuration, CLI, architecture, OpenCode, plugin, and state/output-schema documentation to distinguish:

- The scalar benchmark generation budget.
- Benchmark-level one-time retries.
- Provider-level HTTP retries, especially 429 backoff.
- Schema-compatibility fallback requests.
- Non-retryable timeout and cancellation.
- Selected-response metadata versus complete attempt history.
- The meaning of each `prompt_altered` enum.

Reports should be able to show, per model/plugin:

- Whether a retry occurred.
- Why it occurred.
- Which prompt alteration was used.
- Whether the selected response was complete, token-limited, timed out, cancelled, repeated, empty, or otherwise failed.
- The token breakdown for the selected attempt and, when useful, each attempt.

## Completion criteria

The redesign is complete when all main benchmark paths use one scalar `max_tokens`, configurations containing `token_levels` fail clearly, each task makes at most one policy-controlled retry, timeout/cancellation never retry, selected-attempt metadata is correct, attempt history is preserved, and report generation can summarize the behavior from machine-readable data alone.
