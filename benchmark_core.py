"""Core benchmark logic shared by the CLI and tests."""
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

from benchmark_http import (  # noqa: F401
    close_active_requests,
    fetch_models_v1,
    stream_request,
    nonstream_request,
)
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
    return max(1, len(text) / 4)


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
    """Load benchmark config from a JSON file. Returns the full config dict."""
    with open(path) as f:
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
                     token_levels, session_seed, log_file, global_cfg, stop_event=None,
                     save_responses=False, output_dir=None, system_prompt=None,
                     is_agent=False):
    """Run a single plugin task for a model or agent. Returns (result_dict, error)."""
    pid = plugin.id
    cfg = source_config.get(source)
    if cfg is None:
        return None, f"Unknown source '{source}' — not in SOURCE_CONFIG"

    if stop_event and stop_event.is_set():
        return None, "Cancelled"

    prompt = plugin.get_prompt()
    temperature = plugin.get_temperature(global_cfg or {})

    raw_model_cfg = (global_cfg or {}).get("models", {}).get(target_name) or (global_cfg or {}).get("agents", {}).get(target_name)
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

    for attempt, max_tok in enumerate(token_levels):
        if stop_event and stop_event.is_set():
            return None, "Cancelled"
        attempt_start = time.time()

        if plugin.supports_streaming:
            text, first_tok, stream_end, serr, sfr, _usage = stream_request(
                source_config, timeout, api_model, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (Streaming, attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params, stop_event=stop_event,
                system_prompt=system_prompt)

            if serr or first_tok is None:
                text, nsusage, ns_time, nserr, nsfr = nonstream_request(
                    source_config, timeout, api_model, source, prompt, max_tok,
                    log_path=log_file,
                    log_label=f"{plugin.name} (Non-Streaming, attempt {attempt + 1})",
                    session_seed=session_seed, temperature=temperature,
                    drop_params=drop_params, stop_event=stop_event,
                    system_prompt=system_prompt)
                if nserr:
                    return None, f"Stream: {serr or 'no tokens'}. Nostream: {nserr}"
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
            text, usage, gen_time, gen_err, gen_fr = nonstream_request(
                source_config, timeout, api_model, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params, stop_event=stop_event,
                system_prompt=system_prompt)
            if gen_err:
                return None, gen_err
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

    if save_responses and output_dir:
        responses_dir = os.path.join(output_dir, "responses", sanitize_filename(target_name))
        os.makedirs(responses_dir, exist_ok=True)
        prompt_path = os.path.join(responses_dir, f"{plugin.id}.prompt.txt")
        response_path = os.path.join(responses_dir, f"{plugin.id}.txt")
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except OSError:
            pass
        try:
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    score, rubric = plugin.evaluate(text)

    if save_responses and output_dir:
        meta_path = os.path.join(responses_dir, f"{plugin.id}.meta.json")
        meta = {
            "plugin": plugin.id,
            "plugin_version": plugin.version,
            "target": target_name,
            "model": api_model,
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
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except OSError:
            pass

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
    return result, None


def run_model(model_name, source, state, active_plugins, source_config, timeout,
              token_levels, output_dir, session_seed=0, global_cfg=None,
              stop_event=None, save_responses=False, api_model=None,
              system_prompt=None, is_agent=False):
    """Run active plugins for one model or agent."""
    start = time.time()
    target_name = model_name
    api_model = api_model or target_name

    r = {
        "model": target_name,
        "api_model": api_model,
        "source": source,
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
    if cfg is None:
        r["status"] = "error"
        r["error"] = f"Unknown source '{source}' — not in SOURCE_CONFIG"
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(target_name, status="failed", error=r["error"], elapsed=r["total_time"])
        state.log(target_name, r['error'])
        return

    latest = {res["model"]: res for res in state.latest_results()}
    existing = latest.get(target_name)

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

    state.update(target_name, attempt_start=time.time())

    _run_plugins(target_name, api_model, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start,
                 max_workers=plugin_thread_limit,
                 stop_event=stop_event,
                 save_responses=save_responses,
                 system_prompt=system_prompt,
                 is_agent=is_agent)


def _run_plugins(target_name, api_model, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start, max_workers,
                 stop_event=None, save_responses=False, system_prompt=None,
                 is_agent=False):
    """Run plugins for one model using a thread pool of bounded size.

    A single-worker pool (``max_workers=1``) is equivalent to sequential
    execution, so this helper is used for both sequential and parallel
    plugin execution.
    """
    results = {plugin.id: None for plugin in plugins_to_run}
    errors = {}
    lock = threading.Lock()
    logs_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(logs_dir, f"{sanitize_filename(target_name)}.log")

    def run_one(plugin):
        pid = plugin.id
        state.update(target_name, status=f"running_{pid}")
        result, err = _run_plugin_task(target_name, api_model, source, plugin, source_config,
                                       timeout, token_levels, session_seed, log_file,
                                       global_cfg or {}, stop_event=stop_event,
                                       save_responses=save_responses,
                                       output_dir=output_dir,
                                       system_prompt=system_prompt,
                                       is_agent=is_agent)
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
