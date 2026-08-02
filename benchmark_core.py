"""Core benchmark logic shared by the CLI and tests."""
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

import yaml

from benchmark_http import (  # noqa: F401
    close_active_requests,
    fetch_models_v1,
    stream_request,
    nonstream_request,
)
from benchmark_plugin import PluginTaskResult
from opencode_runner import OpenCodeProcessResult, opencode_model_name, run_process
from benchmark_outputs import (  # noqa: F401
    _save_outputs,
    gen_csv,
    gen_html,
    gen_markdown,
    gen_pdf,
    sanitize_filename,
)
from benchmark_state import BenchmarkState  # noqa: F401


def count_tokens(text):
    return max(0, len(text) / 4)


def is_repeating(text, min_seq=80, repeats=3):
    """Detect if text is stuck in a loop."""
    if len(text) < min_seq * repeats:
        return False
    tail = text[-min_seq:]
    return text.count(tail) >= repeats


def _source_abbrev(name):
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


def _unique_source_abbrevs(sources):
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

def _expand_env(val):
    """Recursively expand ${VAR} or ${VAR:default} in strings."""
    if isinstance(val, str):
        parts = []
        raw = val
        i = 0
        while i < len(raw):
            start = raw.find("${", i)
            if start == -1:
                parts.append(raw[i:])
                break
            end = raw.find("}", start)
            if end == -1:
                parts.append(raw[i:])
                break
            expr = raw[start+2:end]
            default = None
            if ":" in expr:
                var, default = expr.split(":", 1)
            else:
                var = expr
            parts.append(raw[i:start])
            parts.append(os.environ.get(var, default or ""))
            i = end + 1
        return "".join(parts)
    if isinstance(val, dict):
        return {k: _expand_env(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_expand_env(v) for v in val]
    return val


def load_config(path):
    """Load benchmark config from a JSON or YAML file. Returns the full config dict."""
    with open(path) as f:
        if path.lower().endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"YAML config file is empty: {path}")
        else:
            data = json.load(f)
    data = _expand_env(data)
    return data


def parse_plugin_temperatures(cfg):
    """Parse per-plugin temperature settings from a config dict.

    Keys ending in ``_temperature`` are mapped to plugin IDs by replacing
    underscores with hyphens (e.g. ``rate-limiter_temperature`` →
    ``rate-limiter``).
    """
    plugin_temperatures = {}
    for key, value in cfg.items():
        if key.endswith("_temperature"):
            plugin_id = key[:-len("_temperature")].replace("_", "-")
            plugin_temperatures[plugin_id] = value
    return plugin_temperatures


def resolve_model_sources(models):
    """Resolve model entries to source strings.

    Model entries may be either a source string or a dict with a
    ``source`` key (and optional per-model settings such as ``drop_params``
    and ``plugins_blacklist``).
    Missing/invalid entries default to ``"Default"``.
    """
    resolved = {}
    for name, val in models.items():
        if isinstance(val, dict):
            resolved[name] = val.get("source", "Default")
        elif isinstance(val, str):
            resolved[name] = val
        else:
            resolved[name] = "Default"
    return resolved


def resolve_targets(cfg):
    """Resolve models and agents into a unified target map.

    Each target contains:
    - ``source``: API source name
    - ``api_model``: actual model string sent to the API
    - ``system_prompt``: optional system prompt for the agent
    - ``is_agent``: whether this target is an agent
    - ``drop_params``: per-target params to drop from API requests
    - ``plugins_blacklist``: per-target plugins to skip
    """
    models = cfg.get("models", {})
    agents = cfg.get("agents", {})
    targets = {}
    for name, val in models.items():
        if isinstance(val, dict):
            targets[name] = {
                "source": val.get("source", "Default"),
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": val.get("drop_params", []),
                "plugins_blacklist": val.get("plugins_blacklist", []),
            }
        elif isinstance(val, str):
            targets[name] = {
                "source": val,
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": [],
                "plugins_blacklist": [],
            }
        else:
            targets[name] = {
                "source": "Default",
                "api_model": name,
                "system_prompt": None,
                "is_agent": False,
                "drop_params": [],
                "plugins_blacklist": [],
            }
    for name, val in agents.items():
        if not isinstance(val, dict):
            raise ValueError(
                f"Agent '{name}' must be an object with at least 'model' and 'system_prompt' keys"
            )
        if "model" not in val:
            raise ValueError(f"Agent '{name}' must specify a 'model' key")
        if "system_prompt" not in val:
            raise ValueError(f"Agent '{name}' must specify a 'system_prompt' key")
        targets[name] = {
            "source": val.get("source", "Default"),
            "api_model": val["model"],
            "system_prompt": val["system_prompt"],
            "is_agent": True,
            "drop_params": val.get("drop_params", []),
            "plugins_blacklist": val.get("plugins_blacklist", []),
        }
    return targets


def get_target_plugins_blacklist(targets, target_name):
    """Get the plugins blacklist for a specific model or agent."""
    val = targets.get(target_name)
    if isinstance(val, dict):
        return val.get("plugins_blacklist", [])
    return []

# Backward-compatible alias.
get_model_plugins_blacklist = get_target_plugins_blacklist


def _apply_http_retry_default(cfg, retry_on_429):
    """Mutate ``cfg`` so HTTP 429 retries align with a global toggle.

    When ``retry_on_429`` is True (the default), this function is a no-op —
    per-source ``max_429_retries`` defaults to 2 inside ``_post_request_context``
    and per-source overrides remain in force. When ``retry_on_429`` is False
    (the user passed ``--no-retry-on-429``), every source that did NOT explicitly
    set ``max_429_retries`` is flipped to ``0`` here so the opt-out propagates
    globally without forcing operators to edit every per-source config block.
    Explicit per-source ``max_429_retries`` values are preserved regardless of
    the global flag — a source that opted in to 5 retries keeps its 5 even
    when the global flag is ``--no-retry-on-429``.

    Mutating ``cfg`` in place is intentional: ``load_config`` returns a fresh
    dict every call, and downstream consumers (``resolve_targets``,
    ``run_model``) read the same object.
    """
    if retry_on_429:
        return
    sources = cfg.get("sources") or {}
    for src_cfg in sources.values():
        if isinstance(src_cfg, dict) and "max_429_retries" not in src_cfg:
            src_cfg["max_429_retries"] = 0


def dump_default_config():
    """Print the default config JSON to stdout."""
    cfg = {
        "output_dir": "benchmark-output-dir",
        "timeout": 1200,
        "token_levels": [16384],
        "rate-limiter_temperature": 0.2,
        "moe-dense_temperature": 0.7,
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {
            "Local Server 1": {
                "api_url": "http://local.server:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${AI_SERVER_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1
            },
            "Local Server 2": {
                "api_url": "http://other.server:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${GAMING_PC_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1
            },
            "Remote Provider 1": {
                "api_url": "http://remote.provider:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${REMOTE_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                },
                "plugin_thread_limit": 1
            }
        },
        "models": {
            "example-model-1": "Local Server 1",
            "example-model-2": "Remote Provider 1",
            "example-model-3": {
                "source": "Local Server 2",
                "drop_params": ["seed"]
            }
        },
        "agents": {
            "example-agent": {
                "model": "gpt-4",
                "source": "Remote Provider 1",
                "system_prompt": "You are a helpful coding assistant. Be concise and accurate."
            }
        }
    }
    print(json.dumps(cfg, indent=2))


def generate_config_from_api(base_url, api_key=None):
    """Build a benchmark config dict by discovering models via the /v1/models endpoint."""
    model_ids = fetch_models_v1(base_url, api_key)
    if not model_ids:
        raise RuntimeError("No models returned by /v1/models endpoint.")

    source_name = "Default"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return {
        "output_dir": "benchmark-results",
        "timeout": 600,
        "token_levels": [16384],
        "plugins_whitelist": [],
        "plugins_blacklist": [],
        "sources": {
            source_name: {
                "api_url": base_url.rstrip("/") + "/chat/completions",
                "headers": headers,
            }
        },
        "models": {mid: source_name for mid in model_ids},
    }


# ─── Model execution ─────────────────────────────────────────────────────────

def _run_plugin_task(target_name, api_model, source, plugin, source_config, timeout,
                     token_levels, session_seed, log_file, global_cfg, state,
                     stop_event=None, save_responses=False, output_dir=None,
                     system_prompt=None, is_agent=False, runner="http",
                     opencode_config_path=None, opencode_model=None,
                     opencode_agent=None, artifact_target_name=None,
                     config_target_name=None) -> PluginTaskResult:
    """Run a single plugin task and return named result/error fields."""
    pid = plugin.id
    cfg = source_config.get(source)
    if runner == "http" and cfg is None:
        return PluginTaskResult(None, f"Unknown source '{source}' — not in SOURCE_CONFIG")
    if runner not in ("http", "opencode"):
        return PluginTaskResult(None, f"Unknown runner {runner!r}")

    if stop_event and stop_event.is_set():
        return PluginTaskResult(None, "Cancelled")

    prompt = plugin.get_prompt()
    temperature = plugin.get_temperature(global_cfg or {})

    config_target_name = config_target_name or target_name
    raw_model_cfg = ((global_cfg or {}).get("models", {}).get(config_target_name)
                     or (global_cfg or {}).get("agents", {}).get(config_target_name))
    drop_params = []
    if isinstance(raw_model_cfg, dict):
        drop_params = raw_model_cfg.get("drop_params", [])
    text = ""
    response_time = 0
    output_tokens = 0
    tps = None
    truncated = False
    repeating = False
    stream_ok = True
    first_tok = None
    gen_time = 0
    think_text = ""
    serr = None
    sfr = None

    if runner == "opencode":
        if not opencode_config_path or not opencode_model:
            return PluginTaskResult(None, "OpenCode runner is missing generated config or model mapping")
        process_result = run_process(
            prompt,
            config_path=opencode_config_path,
            model=opencode_model,
            timeout=timeout,
            agent=opencode_agent,
            output_dir=output_dir,
            target_key=artifact_target_name or config_target_name,
            plugin_id=pid,
            stop_event=stop_event,
        )
        text = process_result.text
        serr = process_result.error
        response_time = round(process_result.elapsed, 1)
        gen_time = process_result.elapsed
        stream_ok = False
        if serr:
            # Preserve a prompt/response/meta sidecar even when OpenCode
            # fails, so a failed subprocess is diagnosable without having
            # to reconstruct the invocation from stderr alone.
            if save_responses and output_dir:
                responses_dir = os.path.join(
                    output_dir, "responses",
                    sanitize_filename(artifact_target_name or config_target_name),
                )
                os.makedirs(responses_dir, exist_ok=True)
                try:
                    with open(os.path.join(responses_dir, f"{pid}.prompt.txt"), "w", encoding="utf-8") as handle:
                        handle.write(prompt)
                    with open(os.path.join(responses_dir, f"{pid}.content.txt"), "w", encoding="utf-8") as handle:
                        handle.write(text)
                    with open(os.path.join(responses_dir, f"{pid}.meta.json"), "w", encoding="utf-8") as handle:
                        json.dump({
                            "plugin": pid,
                            "plugin_version": plugin.version,
                            "target": artifact_target_name or config_target_name,
                            "model": api_model,
                            "runner": runner,
                            "opencode_model": opencode_model,
                            "is_agent": is_agent,
                            "system_prompt": system_prompt,
                            "score": "fail",
                            "rubric": [],
                            "response_time": response_time,
                            "output_tokens": int(count_tokens(text)),
                            "tps": None,
                            "seed": session_seed,
                            "timestamp": datetime.now().isoformat(),
                            "error": serr,
                        }, handle, indent=2, default=str)
                except OSError:
                    pass
            return PluginTaskResult(None, serr)
        output_tokens = int(count_tokens(text))
        if gen_time > 0:
            tps = round(output_tokens / gen_time, 2)
        token_levels = []

    for attempt, max_tok in enumerate(token_levels):
        if stop_event and stop_event.is_set():
            return PluginTaskResult(None, "Cancelled")
        attempt_start = time.time()

        # MUST be defined above both branches -- Python scope analysis
        # binds ``on_retry`` as a local for the entire function because of
        # this ``def``, so the ``else`` branch below would otherwise raise
        # ``UnboundLocalError`` evaluating its ``on_retry=on_retry`` kwarg
        # for every supports_streaming=False plugin. Reset per-plugin
        # timing on every 429 retry to keep the elapsed display honest.
        def on_retry():
            state.start_plugin_run(target_name, pid)

        if plugin.supports_streaming:
            # Per-SSE-delta observer so the live TUI can show a
            # streaming tok ticker ([streaming - N tok] cell +
            # "[name: N tok]" live-footer entry). The callback runs
            # under ``stream_request``'s loop on the worker thread; a
            # buggy observer is swallowed inside ``stream_request`` so
            # it cannot abort the stream read. We measure CHARACTERS
            # (not UTF-8 bytes) so the live ticker matches the post-
            # completion ``count_tokens(text) = max(0, len(text) / 4)`` estimator
            # exactly -- a CJK chunk would otherwise show 3x as many
            # "tokens" during streaming as it does after completion.
            #
            # The closure fires ``mark_first_chunk_seen`` AND
            # ``add_bytes_received`` on every non-empty delta --
            # ``mark_first_chunk_seen`` is idempotent (it only writes
            # ``first_tok_ts`` on the False -> True transition, so
            # subsequent calls preserve the original timestamp); the
            # closure doesn't need a local "fired" flag because the
            # state method owns the gate. This satisfies both
            # downstream consumers: the cell renderer's
            # ``[streaming - N tok]`` real-counter form needs
            # ``first_chunk_seen=True`` (set on first delta),
            # and the live footer's ``[<pid>: N tok]`` per-plugin
            # indicator needs ``first_tok_ts > 0`` (also set on first
            # delta). ``stream_request`` itself only invokes the
            # callback when ``len(text) > prev_text_len`` i.e. on a
            # non-empty content delta -- role-only / heartbeat /
            # ``[DONE]`` / malformed-JSON lines are filtered out
            # inside ``_parse_sse_line`` and never reach us here.
            def on_chunk(delta):
                state.mark_first_chunk_seen(target_name, pid, ts=time.time())
                state.add_bytes_received(target_name, pid, len(delta))
            # Parallel reasoning/thinking callback for thinking-capable
            # models. Fires once per parsed SSE ``reasoning_content``
            # delta so the live TUI can show a tokenised ticker before
            # primary ``content`` starts flowing -- the thinking phase
            # of a deepseek-r1 / Qwen3 / o1-style stream is otherwise
            # indistinguishable from "no first token yet" because
            # ``content`` is still empty. The closure shares the same
            # ``mark_first_chunk_seen`` gate as ``on_chunk`` so the
            # ``first chunk seen`` flag fires on the first reasoning
            # delta (operators do not need to distinguish "first
            # thinking chunk" from "first content chunk" as separate
            # gates -- they only need to know the response has begun).
            # ``add_thinking_bytes_received`` runs the parallel
            # wiring self-check the same way ``add_bytes_received``
            # does, so a wiring bug fails fast at first delta.

            def on_think_chunk(think_delta):
                state.mark_first_chunk_seen(target_name, pid, ts=time.time())
                state.add_thinking_bytes_received(target_name, pid, len(think_delta))

            stream_result = stream_request(
                source_config, timeout, api_model, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (Streaming, attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params, stop_event=stop_event,
                system_prompt=system_prompt,
                on_chunk=on_chunk, on_think_chunk=on_think_chunk, pid=pid, on_retry=on_retry)
            text = stream_result.text
            think_text = stream_result.think_text
            first_tok = stream_result.first_tok
            stream_end = stream_result.stream_end
            serr = stream_result.error
            sfr = stream_result.finish_reason

            if serr or first_tok is None:
                # Streaming attempt failed. If the stream actually opened
                # (``first_tok`` set) OR accumulated ANY characters, KEEP
                # the streamed text instead of clobbering it with a
                # non-streaming retry. A non-stream retry from a
                # "thinking" model that already streamed 40 K chars will
                # likely return empty, and ``count_tokens("")`` now correctly returns
                # 0 rather than a one-token placeholder.
                # into a 1-token placeholder record (operator reported
                # kimi-dev streaming 10 K tokens over 2 000 s then
                # "giving up" with ``_output_tokens = 1``). Only fall
                # through to non-streaming when streaming produced
                # nothing useful; that branch keeps the original
                # behaviour of trying once more to get a response when
                # streaming never opened at all.
                if first_tok is not None or len(text) > 0:
                    # Trust the streamed ``stream_end`` and ``first_tok``
                    # for timing. ``stream_ok`` flips to False because
                    # the request didn't complete normally; ``truncated``
                    # reports ``sfr == "length"``.
                    response_time = round(stream_end - attempt_start, 1)
                    gen_time = stream_end - first_tok if first_tok else 0
                    truncated = (sfr == "length")
                    stream_ok = False
                else:
                    nonstream_result = nonstream_request(
                        source_config, timeout, api_model, source, prompt, max_tok,
                        log_path=log_file,
                        log_label=f"{plugin.name} (Non-Streaming, attempt {attempt + 1})",
                        session_seed=session_seed, temperature=temperature,
                        drop_params=drop_params, stop_event=stop_event,
                        system_prompt=system_prompt, pid=pid, on_retry=on_retry)
                    text = nonstream_result.text
                    think_text = nonstream_result.think_text
                    nsusage = nonstream_result.usage
                    ns_time = nonstream_result.gen_time
                    nserr = nonstream_result.error
                    nsfr = nonstream_result.finish_reason
                    if nserr:
                        return PluginTaskResult(None, f"Stream: {serr or 'no tokens'}. Nostream: {nserr}")
                    stream_ok = False
                    response_time = round(ns_time, 1)
                    gen_time = ns_time
                    truncated = (nsfr == "length")
            else:
                stream_ok = True
                response_time = round(stream_end - attempt_start, 1)
                gen_time = stream_end - first_tok if first_tok else 0
                truncated = (sfr == "length")
        else:
            nonstream_result = nonstream_request(
                source_config, timeout, api_model, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params, stop_event=stop_event,
                system_prompt=system_prompt, pid=pid, on_retry=on_retry)
            text = nonstream_result.text
            think_text = nonstream_result.think_text
            usage = nonstream_result.usage
            gen_time = nonstream_result.gen_time
            gen_err = nonstream_result.error
            gen_fr = nonstream_result.finish_reason

            if gen_err:
                return PluginTaskResult(None, gen_err)
            stream_ok = False
            response_time = round(gen_time, 1)
            truncated = (gen_fr == "length")

            est_tok = count_tokens(text)
            output_tokens = int(est_tok)
            if gen_time > 0:
                tps = round(est_tok / gen_time, 2)

            if not truncated:
                break

            if is_repeating(text):
                repeating = True
                break

            if len(text.strip()) < 50:
                pass

            if attempt < len(token_levels) - 1:
                pass

    # Compute buffered/partial response metrics uniformly for both transports.
    # Streaming failures and OpenCode both arrive here without the HTTP
    # usage-based bookkeeping used by some non-streaming responses.
    output_tokens = int(count_tokens(text))
    if gen_time > 0:
        tps = round(output_tokens / gen_time, 2)

    if save_responses and output_dir:
        responses_dir = os.path.join(output_dir, "responses", sanitize_filename(artifact_target_name or config_target_name))
        os.makedirs(responses_dir, exist_ok=True)
        # 1. Prompt file (unchanged).
        prompt_path = os.path.join(responses_dir, f"{plugin.id}.prompt.txt")
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except OSError:
            pass
        # 2. Joined response — final content, with any thinking content wrapped
        #    in ``<thinking>...</thinking>`` markers so operators can distinguish
        #    the model's chain-of-thought from its final answer without needing
        #    a separate viewer. When there is no thinking content the file is
        #    identical to the previous version (pure final content).
        if think_text:
            joined = f"<thinking>\n{think_text}\n</thinking>\n\n{text}"
        else:
            joined = text
        response_path = os.path.join(responses_dir, f"{plugin.id}.txt")
        try:
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(joined)
        except OSError:
            pass
        # 3. Thinking-only file (only created when thinking content exists).
        if think_text:
            think_path = os.path.join(responses_dir, f"{plugin.id}.think.txt")
            try:
                with open(think_path, "w", encoding="utf-8") as f:
                    f.write(think_text)
            except OSError:
                pass
        # 4. Content-only file — pure final content without thinking markers.
        #    Identical to the original ``{plugin.id}.txt`` format from before
        #    thinking-content separation was added.
        content_path = os.path.join(responses_dir, f"{plugin.id}.content.txt")
        try:
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    # B5: a plugin that crashes mid-evaluation used to silently lose its
    # ``meta.json`` sidecar — only ``prompt.txt`` and ``<plugin>.txt``
    # survived, which forced debuggers to rebuild the failure by hand. We
    # now catch every exception, persist the ``error`` + ``traceback``
    # fields alongside the metrics that WERE successfully gathered, and    #    surface the same named ``PluginTaskResult`` failure contract as the
    #    streaming failure path.
    score = "fail"
    rubric = []
    score_error = None
    score_traceback_text = None
    try:
        evaluation = plugin.evaluate(text)
        score = evaluation.score
        rubric = evaluation.rubric
    except Exception as exc:
        score_error = f"plugin.evaluate raised {type(exc).__name__}: {exc}"
        score_traceback_text = traceback.format_exc()

    if save_responses and output_dir:
        meta_path = os.path.join(responses_dir, f"{plugin.id}.meta.json")
        meta = {
            "plugin": plugin.id,
            "plugin_version": plugin.version,
            "target": artifact_target_name or config_target_name,
            "model": api_model,
            "runner": runner,
            "opencode_model": opencode_model,
            "is_agent": is_agent,
            "system_prompt": system_prompt,
            "score": score,
            "rubric": rubric,
            "response_time": response_time,
            "output_tokens": output_tokens,
            "tps": tps,
            "seed": session_seed,
            "timestamp": datetime.now().isoformat(),
        }
        if score_error is not None:
            meta["error"] = score_error
            meta["traceback"] = score_traceback_text
        # ``stream_error`` is the streaming-layer failure reason when
        # the partial-stream branch (kept-streamed-text on ``serr``)
        # fired. Operators inspecting a meta.json whose ``stream_ok``
        # is False / ``output_tokens`` looks low / ``truncated`` is
        # False get an explicit ``timeout``/``Read timed out``/etc.
        # reason here, rather than having to grep the per-request log.
        # Recorded for streaming-capable plugins only (non-streaming
        # plugins cannot produce ``serr``).
        if plugin.supports_streaming and serr is not None:
            meta["stream_error"] = serr
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except OSError:
            pass

    if score_error is not None:
        return PluginTaskResult(None, score_error)

    result = {
        f"{pid}_score": score,
        f"{pid}_rubric": rubric,
        f"{pid}_response_time": response_time,
        f"{pid}_output_tokens": output_tokens,
        f"{pid}_tps": tps,
        f"{pid}_truncated": truncated,
        f"{pid}_repeating": repeating,
        f"{pid}_stream_ok": stream_ok,
    }
    return PluginTaskResult(result, None)


def run_model(model_name, source, state, active_plugins, source_config, timeout,
              token_levels, output_dir, session_seed=0, global_cfg=None,
              stop_event=None, save_responses=False, api_model=None,
              system_prompt=None, is_agent=False, runner="http",
              opencode_config_path=None, opencode_model=None,
              opencode_agent=None, display_name=None,
              config_target_name=None):
    """Run active plugins for one model or agent through a selected runner."""
    start = time.time()
    target_name = model_name
    display_name = display_name or target_name
    config_target_name = config_target_name or display_name
    api_model = api_model or target_name

    r = {
        "model": display_name,
        "state_key": target_name,
        "api_model": api_model,
        "source": source,
        "runner": runner,
        "opencode_model": opencode_model,
        "is_agent": is_agent,
        "system_prompt": system_prompt,
        "status": "ok",
        "stream_ok": True,
        "ttft": None,
        "prompt_tokens": 0, "completion_tokens": 0,
        "total_time": 0, "error": None,
        "plugin_versions": {p.id: p.version for p in active_plugins},
    }

    state.update(target_name, status="queued")

    cfg = source_config.get(source)
    if runner == "http" and cfg is None:
        r["status"] = "error"
        r["error"] = f"Unknown source '{source}' — not in SOURCE_CONFIG"
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(target_name, status="failed", error=r["error"], elapsed=r["total_time"])
        state.log(target_name, r['error'])
        return

    latest = {(res.get("state_key", res["model"]), res.get("runner", "http")): res
              for res in state.latest_results()}
    existing = latest.get((target_name, runner))

    plugins_to_run = []
    for plugin in active_plugins:
        pid = plugin.id
        score_key = f"{pid}_score"
        # Re-use successful plugin results from a previous run; re-run any
        # plugin that failed or was missing.
        if existing is not None and score_key in existing and existing[score_key] != "fail":
            r[f"{pid}_score"] = existing[score_key]
            r[f"{pid}_response_time"] = existing[f"{pid}_response_time"]
            r[f"{pid}_output_tokens"] = existing[f"{pid}_output_tokens"]
            r[f"{pid}_tps"] = existing[f"{pid}_tps"]
            r[f"{pid}_stream_ok"] = existing.get(f"{pid}_stream_ok", True)
        else:
            plugins_to_run.append(plugin)

    if not plugins_to_run:
        r["stream_ok"] = any(r.get(f"{p.id}_stream_ok", True) for p in active_plugins)
        r["ttft"] = existing.get("ttft") if existing else None
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(target_name, status="completed", elapsed=r["total_time"])
        return

    plugin_thread_limit = source_config.get(source, {}).get("plugin_thread_limit", 1)
    try:
        plugin_thread_limit = int(plugin_thread_limit)
    except (TypeError, ValueError):
        plugin_thread_limit = 1
    if plugin_thread_limit <= 0:
        plugin_thread_limit = len(plugins_to_run)

    state.update(target_name, attempt_start=time.monotonic())

    _run_plugins(target_name, api_model, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start,
                 max_workers=plugin_thread_limit,
                 stop_event=stop_event,
                 save_responses=save_responses,
                 system_prompt=system_prompt,
                 is_agent=is_agent,
                 runner=runner,
                 opencode_config_path=opencode_config_path,
                 opencode_model=opencode_model,
                 opencode_agent=opencode_agent,
                 display_name=display_name,
                 config_target_name=config_target_name)


def _run_plugins(target_name, api_model, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start, max_workers,
                 stop_event=None, save_responses=False, system_prompt=None,
                 is_agent=False, runner="http", opencode_config_path=None,
                 opencode_model=None, opencode_agent=None, display_name=None,
                 config_target_name=None):
    """Run plugins for one model using a thread pool of bounded size.

    A single-worker pool (``max_workers=1``) is equivalent to sequential
    execution, so this helper is used for both sequential and parallel
    plugin execution.
    """
    results = {plugin.id: None for plugin in plugins_to_run}
    errors = {}
    lock = threading.Lock()
    logs_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(logs_dir, f"{sanitize_filename(display_name or target_name)}.log")

    def run_one(plugin):
        pid = plugin.id
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
                                           timeout, token_levels, session_seed, log_file,
                                           global_cfg or {}, state=state,
                                           stop_event=stop_event,
                                           save_responses=save_responses,
                                           output_dir=output_dir,
                                           system_prompt=system_prompt,
                                           is_agent=is_agent,
                                           runner=runner,
                                           opencode_config_path=opencode_config_path,
                                           opencode_model=opencode_model,
                                           opencode_agent=opencode_agent,
                                           artifact_target_name=display_name or target_name,
                                           config_target_name=config_target_name or display_name or target_name)
            result = task_result.result
            err = task_result.error
        finally:
            # Clear the in-flight marker even on exception/cancellation so
            # parallel plugins aren't stranded in the running list when one
            # of them raises. ``status`` is committed by the outer caller
            # (``run_model``) once all plugins resolve.
            state.finish_plugin_run(target_name, pid)
        with lock:
            results[pid] = result
            if err:
                errors[pid] = err
        if err or result is None:
            return
        state.update(target_name,
                     **{f"{pid}_score": result[f"{pid}_score"],
                        f"{pid}_tps": result[f"{pid}_tps"],
                        f"{pid}_response_time": result[f"{pid}_response_time"],
                        f"{pid}_output_tokens": result[f"{pid}_output_tokens"]})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, plugin): plugin for plugin in plugins_to_run}
        pending = set(futures.keys())
        while pending:
            if stop_event and stop_event.is_set():
                for f in pending:
                    f.cancel()
                break
            done, pending = wait(
                pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    fut.result()
                except Exception as exc:
                    plugin = futures[fut]
                    with lock:
                        errors[plugin.id] = f"{type(exc).__name__}: {exc}"

    for plugin in plugins_to_run:
        pid = plugin.id
        if pid in errors or results.get(pid) is None:
            fail_values = {
                f"{pid}_score": "fail",
                f"{pid}_response_time": "fail",
                f"{pid}_output_tokens": "fail",
                f"{pid}_tps": "fail",
                f"{pid}_stream_ok": False,
            }
            r.update(fail_values)
            state.update(target_name, **fail_values)
        else:
            result = results[pid]
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

    if stop_event and stop_event.is_set():
        r["status"] = "error"
        r["error"] = "Cancelled"
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(target_name, status="failed", error=r["error"], elapsed=r["total_time"], last_error=r["error"])
        return

    if errors:
        r["status"] = "error"
        r["error"] = "; ".join(f"{pid}: {err}" for pid, err in errors.items())
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(target_name, status="failed", error=r["error"], elapsed=r["total_time"], last_error=r["error"])
        return

    r["total_time"] = round(time.time() - start, 1)
    state.add_result(r)
    state.update(target_name, status="completed", elapsed=r["total_time"])
