# Design: Model Pre-Loading (warm-up probe) before benchmark legs

**Status:** Design for review — no code implemented yet.
**Date:** Design note
**Related:** opencode failure root-cause analysis (a prior benchmark run — 21/22 opencode failures were the 120s no-output guard killing legs whose backend was still loading/serving the model; HTTP TTFT for the same models was 20–260s).

---

## 1. Problem statement

Large local models (Ollama via the litellm proxy) take 20–260+ seconds to load into VRAM and produce a first token. When the benchmark switches to a new model on a source, the first leg (usually the OpenCode leg in `both` mode) sends its real request immediately and the OpenCode subprocess emits `step_start` then sits byte-silent while the backend loads the model. The `OPENCODE_NO_OUTPUT_GRACE = 120s` guard cannot distinguish "backend is still loading" from "backend is hung", so it kills the leg and the whole model is marked failed.

**Goal:** when the worker moves to a new model on a source, *optionally* send a trivial warm-up probe first to force the backend to load the model. The probe's time does not count toward any test. If the probe fails, record the failure, skip the model (both runners), and move on — no 30-minute burn on a model that cannot load.

---

## 2. Behavior summary (decisions locked in)

| Decision | Choice |
|---|---|
| Probe transport | Direct HTTP request to the source's `api_url` (reuses `nonstream_request`) |
| Probe failure | Skip the **whole model** — both `[opencode]` and plain legs — record `preload failed`, move to next model |
| Probe timeout | Per-source `preload_timeout` (default **300 seconds**); one attempt, no 429 retry |
| Default state | **Opt-in** — sources only preload when they set `preload: true` |
| CLI escape hatch | `--no-preload` disables preload for the whole run (overrides all sources) |
| Trigger timing | Once per `(source, api_model)` per session, before that model's **first leg**, in every runner mode (`http`, `opencode`, `both`). In `both`, preload precedes the OpenCode leg; the HTTP leg reuses the warm model |
| Resume semantics | Session-only: no preload state persisted for resume-skip purposes. (The failed *result entries* of a skipped model persist like any other failure and follow normal resume rules) |
| TUI | Table: both rows of the model show a preloading glyph with a shared elapsed counter; Live footer shows `Preloading <model> Ns`; bottom footer shows a preload count |
| Reports | Preload-failed models appear as failed rows in results.csv/html/md; run-info.json gains a `preload` summary section |

---

## 3. Config & CLI surface

### 3.1 Per-source config keys (all optional)

```yaml
sources:
  Gaming PC:
    api_url: http://gaming.pc:11434/chat/completions
    preload: true          # NEW — opt-in; default false
    preload_timeout: 300    # optional override; default 300 seconds
```

- `preload: bool` (default `false`) — enable the warm-up probe for this source.
- `preload_timeout: int` (optional, seconds) — maximum time for the probe on this source. The default is **300 seconds**, independent of the much larger benchmark request `timeout`, so a model that cannot load does not block a source worker for the full test timeout. Values must be positive. A configured value overrides the 300-second default for that source.

### 3.2 CLI flag

```
--no-preload    Disable model pre-loading for this run (overrides per-source preload: true)
```

- Mirrors the existing `--no-retry-on-429` / `--no-install-opencode` pattern (`store_false`, default `None`).
- `None` → per-source `preload` decides. `False` (flag given) → all sources skip preload.
- A positive CLI `--preload` flag is **not** added (keep the surface minimal); the config is the opt-in mechanism.

### 3.3 Default-config template

`dump_default_config` emits `preload: false` on each source with a short comment, and `docs/configuration.md` documents both keys plus the CLI flag.

---

## 4. The preload probe

### 4.1 Request shape

- **Transport:** `benchmark_http.nonstream_request(...)` — one buffered request, no streaming machinery, no plugin machinery.
- **Prompt:** module-level constant `PRELOAD_PROMPT = "Reply with the single word OK."` (fixed, trivial; model has to actually generate a token, which forces a full load + inference pass).
- **`max_tokens`:** small fixed value (e.g. `16`) — enough to prove generation, not enough to matter.
- **`temperature`:** `0.0` (deterministic; irrelevant to correctness).
- **`session_seed`:** passed through for consistency with the run's other requests; source `drop_params` still applies (e.g. the gemini sources drop `seed`).
- **Other params:** none of the plugin prompt/agent scaffolding. No system prompt, no tools.

### 4.2 Acceptance criteria

- **Success:** `nonstream_request` returns no error **and** non-empty content.
- **Failure:** any error (timeout, connection refused, HTTP 429/5xx, empty content). Recorded as `preload failed: <error>`.
- **One attempt only:** no retry, no 429 backoff. A rate-limited/failing source gets the model skipped — which is correct, because the legs would fail identically.

### 4.3 Cancellation

The probe respects `stop_event`: it uses the same `stop_event` plumbing as other requests, so Ctrl+C aborts a probe promptly (the underlying request layer already closes active requests on interrupt).

### 4.4 What it does NOT do

- Does **not** create plugin results, score anything, or write `responses/` files.
- Does **not** count toward the model's `total_time`, plugin `response_time`, TTFT, or the worker's per-model `elapsed`. It runs *outside* `run_model` (see §5), so none of the existing timers see it.
- Does **not** warm OpenCode itself (binary startup / session creation) — only the model backend. (Out of scope; opencode binary stays warm enough across legs.)

---

## 5. Pipeline integration

### 5.1 Session-scoped preload cache

A small in-memory registry lives in `main()`'s closure and is passed to the worker closures:

```python
preloaded_ok: set[tuple[str, str]] = set()     # (source, api_model) warmed this session
```

- Preload runs only for `(source, api_model)` pairs **not already in `preloaded_ok`**.
- On success: add to `preloaded_ok`; proceed with the leg.
- On failure: do **not** add; record failure (see §6) and skip the model.

### 5.2 Hook point — `run_target` (both-mode pipeline, `ai-benchmark.py`)

The preload probe runs at the top of `run_target(model_name, phase_runner)`, **before** `run_model(...)`:

```python
def run_target(model_name, phase_runner):
    target_info = targets[model_name]
    key = (target_info["source"], target_info["api_model"])
    if preload_enabled(source_config[target_info["source"]]) and key not in preloaded_ok:
        ok = _preload_model(target_info, phase_runner, ...)   # sets TUI state, waits, times
        if not ok:
            _mark_preload_failed(model_name, phase_runner, ...)  # §6.3
            return                                              # skip this target's legs
        preloaded_ok.add(key)
    ...existing run_model(...) call...
```

Key properties:
- Runs **once per model** because the second leg (`http`) finds `key` already in `preloaded_ok` — in `both` mode the probe precedes the OpenCode leg and the HTTP leg inherits the warm model with zero extra cost.
- In `both` mode the state keys of *both* rows (`model` and `model [opencode]`) are flagged "preloading" so the TUI can show the glyph on both rows (§7).

### 5.3 Hook point — single-runner workers (`http` / `opencode` modes)

The non-pipeline branch (`runner_mode != "both"`, the per-source worker loop at the bottom of `main`) gets the same preload call before its first leg for each model. Same session cache, same failure handling. (`http`-only runs preload too — the probe is runner-agnostic and benefits HTTP TTFT as well.)

### 5.4 Concurrency

No new concurrency: the probe runs inside the source's existing worker thread, so it is naturally serialized per source. Sources preload in parallel with one another — which is exactly the existing parallelism model (one worker per source). No locks beyond the existing state lock.

---

## 6. State & result records

### 6.1 In-memory `model_info` fields (per state key)

Added to both `model_name` and `model_name [opencode]` entries while preloading (set by `state.update(...)`):

| Field | Meaning |
|---|---|
| `preloading` | `True` while the probe is in flight |
| `preload_start_ts` | `time.monotonic()` at probe start (drives the `Ns` counter) |
| `preload_status` | `"ok"` or `"failed"` after the probe completes |
| `preload_time` | probe wall-clock seconds (rounded to 1 decimal) |
| `preload_error` | error string when failed, else absent |

These are **not** persisted as dedicated state (per decision §2) — they are transient session fields the TUI reads from `state.snapshot()`.

### 6.2 Result entries on failure

On preload failure the worker writes **result entries** (the normal `state.add_result` path) so reports surface the skip:

- One entry with `state_key=model_name [opencode]` when the opencode leg would have run (`both`/`opencode` modes), `runner="opencode"`.
- One entry with `state_key=model_name`, `runner="http"` when the http leg would have run (`both`/`http` modes).

Each entry:

```python
{
  "model": "<display>", "state_key": "<key>", "api_model": ..., "source": ...,
  "runner": "opencode" | "http",
  "status": "failed",
  "error": "preload failed: <probe error>",
  "preload_time": <seconds>,
  "total_time": <seconds>,
  "plugin_versions": {...},
}
```

Both entries carry the same `preload_time` and `preload_error`, and `state.update(key, status="failed", error=..., last_error=...)` mirrors it, plus `state.log(key, error)` so it appears in the TUI error area.

### 6.3 Resume semantics (post-skip)

The failed result entries persist in `benchmark_state.json` like any other failure. On resume, normal rules apply:
- Default (`--no-rerun-failed` off): the model is re-attempted → preload re-runs (fresh session) → fails again → re-skipped. Same behavior as any persistently-failing model today.
- `--no-rerun-failed`: the model stays failed and is never touched.
No special-casing added. (`--restart` discards state and re-preloads everything.)

---

## 7. TUI changes

### 7.1 Table (both rows)

- New status glyph `🔄` (U+1F504) shown in the frozen `St` column of **both** `model` and `model [opencode]` rows while `preloading` is set, replacing the pending `⏳`. Both rows show the same live elapsed counter.
- `_format_model_row`: `status_ch = "🔄"` when `s.get("preloading")` (checked before the `running_pids` branch; a preloading model has no plugins in flight).
- Row color: keep the yellow/active treatment (same as running).

### 7.2 Live footer

- New `Preloading:` section under `Live:`, mirroring the existing `429 Sleeping:` block, one line per preloading model:

  ```
  Live:
   🔷 [GP] gemma4:31b-32k [code-review: ...]
  Preloading:
   🔄 [GP] gemma4:31b-256k 8s
  ```

  Each line: `🔄 [<src-abbr>] <model> <elapsed>s`, elapsed from `preload_start_ts` (monotonic).
- The "Live:" header row stays; the preloading section renders between live models and the 429 section (or after, whichever fits — layout already handles variable section counts).

### 7.3 Bottom footer (`_render_footer`)

Extend the status line to include a preload count when any model is preloading:

```
  2 active  |  1 preloading  |  3 queued
```

### 7.4 Summary header (`_render_header_and_summary`)

No change required (Active/Queued counts are unchanged — a preloading model is not yet "active" since it has no plugins in flight). Optional: fold into footer only.

### 7.5 Fallback (non-curses) loop (`_fallback_tui_loop`)

Append `Preloading: <model> Ns` lines in the same style as the existing per-model lines, and include preloading models in the active count:

```
🔄 1 active | ✅ 3/10 completed | HTTP: 1 | 429⏸ 0 | Preloading: gemma4:31b-256k 8s
```

---

## 8. Reports & run-info

### 8.1 results.csv / .html / .md

A preload-failed model already appears as a **failed row** (via the §6.2 result entries) with `-`/empty scores and the error text in the status/error column. No new columns added in v1.

### 8.2 run-info.json

New `preload` section (written on completion/interrupt, like `backoff_429`):

```json
"preload": {
  "enabled_sources": ["Gaming PC", "AI Server"],
  "attempted": 12,
  "succeeded": 10,
  "failed": 2,
  "total_preload_time": 412.3,
  "per_model": {
    "gaming-pc/gemma4:31b-256k": {"status": "ok", "time": 312.4},
    "ai-server/llama3.3:70b-128k": {"status": "failed", "time": 1800.0, "error": "..."}
  }
}
```

Keyed by the resolved `{source}/{api_model}` (consistent with the existing opencode projection naming).

---

## 9. Files touched (implementation plan)

| File | Change |
|---|---|
| `benchmark_core.py` | `PRELOAD_PROMPT` constant; `preload_model(source_config, source, api_model, timeout, session_seed, stop_event)` helper wrapping `nonstream_request`; `dump_default_config` emits `preload: false` per source |
| `ai-benchmark.py` | `--no-preload` arg; per-source `preload` resolution (with CLI override); session `preloaded_ok` cache; preload call + failure skip in `run_target` and the single-runner workers; `_mark_preload_failed` helper writing §6.2 result entries; TUI: `🔄` glyph in `_format_model_row`, `Preloading:` section in `_render_live_activity`, footer count, fallback-loop lines; `run_info["preload"]` aggregation |
| `benchmark_state.py` | (likely none — `state.update` already accepts arbitrary keys; result entries use the existing `add_result`) |
| `docs/configuration.md` | document `preload` / `preload_timeout` / `--no-preload` |
| `docs/cli.md` | `--no-preload` entry |
| `AGENTS.md` | one-line known-feature note |
| `tests/test_cli.py` | preload-on-first-leg, skip-on-failure (both runners), `--no-preload` override, preload runs outside model timing |
| `tests/test_agents.py` | `preload_model` helper unit tests (success / error / empty-content / timeout via mocks) |
| `tests/test_tui_cells.py` | `🔄` glyph when `preloading`, footer preload count, fallback loop line |

---

## 10. Edge cases

1. **Same model in two sources** → two independent preloads (key includes source). Correct — different backends.
2. **Model already warm** (e.g. consecutive targets on the same source) → probe still runs once per model; it's cheap (a few seconds) and proves the backend responds. Acceptable; the session cache prevents per-leg repeats.
3. **Source timeout is huge (1800s)** → the preload still stops after the separate 300-second default (or the source's configured `preload_timeout`), while real tests retain their 1800-second timeout.
4. **Ctrl+C during probe** → `stop_event` aborts; worker exits; state left as `preloading` is re-computed next session (no persistence).
5. **Probe success but legs fail** → normal failure handling; nothing special.
6. **`both` mode + resume mid-model** (opencode done, http pending) → `key` in `preloaded_ok` (same session) → no re-probe; if resumed in a *new* session, the http leg re-probes (fresh cache). Slight cost, correct behavior.
7. **`--no-preload` + `preload: true`** → CLI wins; no probe; model legs run as today.
8. **Remote sources with `preload: true`** → one harmless warm-up call per model (round-trip is fast; may hit provider caches). Operator's choice.

---

## 11. Open questions for the reviewer

1. Should a preload **success** be recorded anywhere visible (e.g. run-info only, as designed) or silently dropped? (Design: run-info summary only.)
2. `preload_timeout` is included in v1. It defaults to 300 seconds and is configurable per source; it is independent of the benchmark request timeout.
3. For a preload-failed model in `both` mode, both rows show `❌ failed`. Acceptable, or should the failure be visually distinct (e.g. `⚠` )?
4. Should the probe reuse the run's `--seed`? (Design: yes, passed through; sources that `drop_params: [seed]` ignore it anyway.)

---

## 12. Validation plan (on implementation)

- Unit: `preload_model` success/error/empty/timeout; `--no-preload` override; skip-on-failure writes both result entries; TUI glyph/footer/fallback rendering; run-info `preload` summary.
- Integration: run `python3 -m unittest discover -s tests -p 'test_*.py'` + plugins suites; `py_compile`; `git diff --check`; code review.
- Manual smoke (no API needed): a fake source whose probe fails → confirm skip + failed rows; one whose probe succeeds → confirm legs proceed and `preload_time` is excluded from `total_time`.
