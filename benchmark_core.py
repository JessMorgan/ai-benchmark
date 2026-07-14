"""Core benchmark logic shared by the CLI and tests."""
import html
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

import requests


_log_lock = threading.Lock()

# Active HTTP responses so Ctrl+C can close them and unblock plugin threads.
_active_requests_lock = threading.Lock()
_active_requests: set = set()


def close_active_requests():
    """Close all in-flight HTTP responses to unblock worker threads."""
    with _active_requests_lock:
        for resp in list(_active_requests):
            try:
                resp.close()
            except Exception:
                pass


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


def resolve_model_sources(models):
    """Resolve model entries to source strings.

    Model entries may be either a source string or a dict with a
    ``source`` key (and optional per-model settings such as ``drop_params``).
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


def dump_default_config():
    """Print the default config JSON to stdout."""
    cfg = {
        "output_dir": "benchmark-output-dir",
        "timeout": 1200,
        "token_levels": [16384],
        "plugin_thread_limit": 1,
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
                }
            },
            "Local Server 2": {
                "api_url": "http://other.server:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${GAMING_PC_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                }
            },
            "Remote Provider 1": {
                "api_url": "http://remote.provider:11434/chat/completions",
                "headers": {
                    "Authorization": "Bearer ${REMOTE_API_KEY:sk-your-key-here}",
                    "Content-Type": "application/json"
                }
            }
        },
        "models": {
            "example-model-1": "Local Server 1",
            "example-model-2": "Remote Provider 1",
            "example-model-3": {
                "source": "Local Server 2",
                "drop_params": ["seed"]
            }
        }
    }
    print(json.dumps(cfg, indent=2))


def fetch_models_v1(base_url, api_key=None):
    """Call GET {base_url}/v1/models and return a list of model IDs."""
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    return [m["id"] for m in models if "id" in m]


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


# ─── Utility helpers ─────────────────────────────────────────────────────────

def sanitize_filename(name):
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


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


# ─── Logging helpers ────────────────────────────────────────────────────────

def build_curl_cmd(model, prompt, max_tokens, stream, api_url, headers):
    """Build a curl command string for the given API request."""
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream
    }, ensure_ascii=False)
    return (
        f"curl -s -X POST '{api_url}' \\\n"
        f"  -H 'Authorization: Bearer {headers['Authorization']}' \\\n"
        f"  -H 'Content-Type: {headers['Content-Type']}' \\\n"
        f"  -d '{data}'"
    )


def log_request_entry(log_path, curl_cmd, response_body, request_label=None):
    """Append a curl command and response body to the log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _log_lock:
        with open(log_path, 'a') as f:
            if request_label:
                f.write(f"\n# === {request_label} ===\n")
            f.write(f"{curl_cmd}\n\n")
            f.write(f"{response_body}\n")
            f.write("\n" + "-" * 60 + "\n")


# ─── API helpers ────────────────────────────────────────────────────────────

def stream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                   log_path=None, log_label=None, session_seed=0, temperature=None,
                   drop_params=None):
    start = time.time()
    first_tok = None
    text = ""
    error = None
    finish_reason = None
    usage = {}
    cfg = source_config.get(source, {})
    api_url = cfg.get("api_url", "http://localhost:11434/chat/completions")
    headers = cfg.get("headers", {"Content-Type": "application/json"})
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, True, api_url, headers) if log_path else None
    try:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        if session_seed:
            body["seed"] = session_seed
        for p in drop_params or []:
            body.pop(p, None)
        resp = requests.post(
            api_url, headers=headers, json=body, stream=True, timeout=timeout)
        with _active_requests_lock:
            _active_requests.add(resp)
        watchdog = threading.Timer(timeout, resp.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            if resp.status_code != 200:
                if log_path and curl_cmd:
                    log_request_entry(log_path, curl_cmd,
                                      f"HTTP {resp.status_code}: {resp.text[:500]}", log_label)
                return text, first_tok, time.time(), f"HTTP {resp.status_code}: {resp.text[:150]}", finish_reason, usage
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        if first_tok is None:
                            first_tok = time.time()
                        for ch in data.get("choices", []):
                            text += ch.get("delta", {}).get("content", "")
                            fr = ch.get("finish_reason")
                            if fr:
                                finish_reason = fr
                        if "usage" in data:
                            usage = data["usage"]
                    except json.JSONDecodeError:
                        continue
        finally:
            watchdog.cancel()
            with _active_requests_lock:
                _active_requests.discard(resp)
            resp.close()
        if not finish_reason and time.time() - start > timeout:
            error = f"Total timeout ({timeout}s) exceeded"
        if log_path and curl_cmd:
            log_request_entry(log_path, curl_cmd, text or "(empty response)", log_label)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if log_path and curl_cmd:
            log_request_entry(log_path, curl_cmd, f"ERROR: {error}", log_label)
    return text, first_tok, time.time(), error, finish_reason, usage


def nonstream_request(source_config, timeout, model, source, prompt, max_tokens=2048,
                      log_path=None, log_label=None, session_seed=0, temperature=None,
                      drop_params=None):
    start = time.time()
    error = None
    text = ""
    usage = {}
    finish_reason = None
    cfg = source_config.get(source, {})
    api_url = cfg.get("api_url", "http://localhost:11434/chat/completions")
    headers = cfg.get("headers", {"Content-Type": "application/json"})
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, False, api_url, headers) if log_path else None
    raw_resp_text = None
    try:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature
        if session_seed:
            body["seed"] = session_seed
        for p in drop_params or []:
            body.pop(p, None)
        # Use stream=True at the transport layer so the response can be closed
        # early on Ctrl+C, while the server still returns a single JSON object.
        resp = requests.post(
            api_url, headers=headers, json=body, stream=True, timeout=timeout)
        with _active_requests_lock:
            _active_requests.add(resp)
        watchdog = threading.Timer(timeout, resp.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            if resp.status_code != 200:
                raw_resp_text = resp.text[:500]
                if log_path and curl_cmd:
                    log_request_entry(log_path, curl_cmd,
                                      f"HTTP {resp.status_code}: {raw_resp_text}", log_label)
                return text, usage, time.time()-start, f"HTTP {resp.status_code}: {resp.text[:150]}", finish_reason
            # Use resp.content so requests handles decompression/encoding.
            raw_resp_text = resp.content.decode("utf-8", errors="replace")
            data = json.loads(raw_resp_text)
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            finish_reason = data.get("choices", [{}])[0].get("finish_reason")
            if log_path and curl_cmd:
                log_request_entry(log_path, curl_cmd, raw_resp_text, log_label)
        finally:
            watchdog.cancel()
            with _active_requests_lock:
                _active_requests.discard(resp)
            resp.close()
        if not error and time.time() - start > timeout:
            error = f"Total timeout ({timeout}s) exceeded"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if log_path and curl_cmd:
            body = raw_resp_text or f"EXCEPTION: {error}"
            log_request_entry(log_path, curl_cmd, body, log_label)
    return text, usage, time.time()-start, error, finish_reason


# ─── Output generators ────────────────────────────────────────────────────────

def _plugin_total_score(result, active_plugins):
    """Sum numeric plugin scores, ignoring non-numeric/failed values."""
    total = 0
    for plugin in active_plugins:
        score = result.get(f"{plugin.id}_score", 0)
        if isinstance(score, (int, float)):
            total += score
    return total


def _numeric_score(result, plugin_id, default=0):
    """Return a numeric score for sorting, falling back to default for non-numeric values."""
    score = result.get(f"{plugin_id}_score", default)
    if isinstance(score, (int, float)):
        return score
    return default


def gen_markdown(results, active_plugins, output_dir=None):
    ok = [r for r in results if r["status"] == "ok"]
    plugin_names = " | ".join(f"**{p.name}** ({int(p.max_score)} pts)" for p in active_plugins)
    lines = [
        "# AI Benchmark — Plugin-Based",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Tasks:** {plugin_names}",
        f"**Total:** {len(results)} models | **✅ {len(ok)} successful** | **❌ {len(results)-len(ok)} failed**",
        "",
        "## 📋 Complete Results",
        "",
    ]

    header = "| # | Model | Load (s) |"
    for p in active_plugins:
        header += f" {p.name} Resp (s) | {p.name} TPS | {p.name} Tok | {p.name} Score |"
        if output_dir:
            header += f" {p.name} Response |"
    header += " Total | Time | Mode |"
    lines.append(header)

    sep = "|---|---|---|"
    for _p in active_plugins:
        sep += "---|---|---|---|"
        if output_dir:
            sep += "---|"
    sep += "---|---|---|"
    lines.append(sep)

    for idx, r in enumerate(results, 1):
        tot = _plugin_total_score(r, active_plugins)
        m = "stream" if r.get('stream_ok') else "nostream"
        row = f"| {idx} | {r['model']} | {r.get('ttft') or '-'} |"
        for p in active_plugins:
            row += (f" {r.get(f'{p.id}_response_time','-')} | "
                    f"{r.get(f'{p.id}_tps','-')} | "
                    f"{r.get(f'{p.id}_output_tokens','-')} | "
                    f"{r.get(f'{p.id}_score','-')} |")
            if output_dir:
                rel_path = f"responses/{sanitize_filename(r['model'])}/{p.id}.txt"
                row += f" [view]({rel_path}) |"
        row += f" {tot} | {r['total_time']}s | {m} |"
        lines.append(row)

    if ok:
        lines.extend(["", "---", "## 🏆 Leaderboards", ""])

        lines.append("### ⚡ Fastest Cold Load (TTFT)")
        lines.append("| # | Model | Load (s) |")
        lines.append("|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1):
            lines.append(f"| {i} | {r['model']} | {r['ttft']} |")

        for p in active_plugins:
            lines.extend(["", f"### 🧠 Best {p.name} Score (/{int(p.max_score)})"])
            lines.append("| # | Model | Score |")
            lines.append("|---|---|---|")
            for i, r in enumerate(sorted(ok, key=lambda x: _numeric_score(x, p.id), reverse=True)[:10], 1):
                lines.append(f"| {i} | {r['model']} | {r.get(f'{p.id}_score', '-')} |")

        lines.extend(["", "### ⭐ Best Combined"])
        lines.append("| # | Model | Total |")
        lines.append("|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: _plugin_total_score(x, active_plugins), reverse=True)[:10], 1):
            tot = _plugin_total_score(r, active_plugins)
            lines.append(f"| {i} | {r['model']} | {tot} |")

    lines.extend(["", "---", "## 📐 Scoring Rubric", ""])
    for p in active_plugins:
        lines.extend([
            f"### {p.name} (0-{int(p.max_score)})",
            "| Criterion | Max | Description |",
            "|---|---|---|",
        ])
        if p.id == "rate-limiter":
            lines.extend([
                "| Interface design | 3 | ABC/Protocol, clean allow_request/get_usage_stats |",
                "| Token Bucket | 4 | Class, refill logic, consume logic |",
                "| Sliding Window | 3 | Class, timestamp tracking, pruning |",
                "| Thread safety | 3 | Locking, minimal contention |",
                "| Cleanup | 2 | Stale entry eviction |",
                "| Type hints | 2 | Parameter & return annotations |",
                "| Docstrings | 2 | Comprehensive documentation |",
                "| Error handling | 1 | Input validation, exceptions |",
            ])
        elif p.id == "moe-dense":
            lines.extend([
                "| Both architectures covered | 2 | Explicitly discusses MoE and dense |",
                "| Gating/routing mechanism | 2.5 | Top-k routing, softmax gating equations |",
                "| Load-balancing loss | 2.5 | Auxiliary loss formulation |",
                "| Training challenges | 2 | Token dropping, expert collapse, etc. |",
                "| Inference implications | 2 | Memory bandwidth, expert parallelism |",
                "| Specific benchmarks | 2 | MMLU, GSM8K, etc. with comparisons |",
                "| Paper references | 2 | Specific papers, technical reports |",
            ])

    lines.extend(["", "## ❌ Failed Models", ""])
    for r in results:
        if r["status"] != "ok":
            lines.append(f"- **{r['model']}**: {r.get('error','?')}")
    lines.append("")
    return "\n".join(lines)


def gen_csv(results, active_plugins):
    import io
    out = io.StringIO()
    import csv
    w = csv.writer(out)
    headers = ["Model", "Source", "TTFT_s"]
    for p in active_plugins:
        headers.extend([f"{p.id}_Response_s", f"{p.id}_Output_Tokens", f"{p.id}_TPS", f"{p.id}_Score_{int(p.max_score)}"])
    headers.extend(["Total", "Time_s", "Mode", "Status", "Error"])
    w.writerow(headers)

    for r in results:
        tot = _plugin_total_score(r, active_plugins)
        m = "stream" if r.get('stream_ok') else "non-streaming"
        row = [r['model'], r.get('source', ''), r.get('ttft') or '']
        for p in active_plugins:
            row.extend([
                r.get(f"{p.id}_response_time", ''),
                r.get(f"{p.id}_output_tokens", ''),
                r.get(f"{p.id}_tps", ''),
                r.get(f"{p.id}_score", ''),
            ])
        if r["status"] == "ok":
            row.extend([tot, r['total_time'], m, "OK", ""])
        else:
            row.extend([tot, r['total_time'], m, "FAIL", r.get('error', '')])
        w.writerow(row)
    return out.getvalue()


def gen_html(results, active_plugins, output_dir=None):
    ok = [r for r in results if r["status"] == "ok"]
    rows = ""
    for r in results:
        cls = "ok" if r["status"] == "ok" else "fail"
        tot = _plugin_total_score(r, active_plugins)
        m = "str" if r.get('stream_ok') else "ns"
        cells = (f'<td>{r["model"]}</td>'
                 f'<td>{r.get("ttft") or "-"}</td>')
        for p in active_plugins:
            score_val = r.get(f"{p.id}_score", "-")
            if output_dir:
                rel_path = f"responses/{sanitize_filename(r['model'])}/{p.id}.txt"
                score_cell = f'<a href="{rel_path}">{score_val}</a>'
            else:
                score_cell = f"{score_val}"
            cells += (f'<td>{r.get(f"{p.id}_response_time","-")}</td>'
                      f'<td>{r.get(f"{p.id}_tps","-")}</td>'
                      f'<td>{r.get(f"{p.id}_output_tokens","-")}</td>'
                      f'<td><strong>{score_cell}</strong></td>')
        cells += (f'<td><strong>{tot}</strong></td>'
                  f'<td>{r["total_time"]}s</td><td>{m}</td>')
        if r["status"] == "ok":
            cells += '<td class="ok-badge">✅</td>'
        else:
            err = html.escape(str(r.get('error') or '?'))[:50]
            cells += f'<td class="fail-badge" title="{err}">❌ {err}</td>'
        rows += f'<tr class="{cls}">{cells}</tr>\n'

    def lb_ttft():
        out = ""
        for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1):
            out += f'<tr><td>{i}</td><td>{r["model"]}</td><td>{r["ttft"]}s</td></tr>\n'
        return out

    leaderboard_html = ""
    for p in active_plugins:
        def lb_for_plugin(plugin):
            out = ""
            for i, r in enumerate(sorted(ok, key=lambda x: _numeric_score(x, plugin.id), reverse=True)[:10], 1):
                out += f'<tr><td>{i}</td><td>{r["model"]}</td><td><strong>{r.get(f"{plugin.id}_score", "-")}/{int(plugin.max_score)}</strong></td></tr>\n'
            return out
        leaderboard_html += f"""<div class="leaderboard">
<h3>🧠 Best {p.name}</h3>
<table><tr><th>#</th><th>Model</th><th>Score</th></tr>
{lb_for_plugin(p)}
</table></div>"""

    header_cells = "<th>Model</th><th>Load(s)</th>"
    for p in active_plugins:
        header_cells += f"<th>{p.name} Resp(s)</th><th>{p.name} TPS</th><th>{p.name} Tok</th><th>{p.name} Score</th>"
    header_cells += "<th>Total</th><th>Time</th><th>Mode</th><th>Status</th>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Benchmark</title>
<style>
body {{ font-family: -apple-system,'Segoe UI',sans-serif; max-width: 1200px; margin:20px auto; padding:0 20px; background:#0d1117; color:#c9d1d9; }}
h1,h2,h3 {{ color:#58a6ff; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }}
th,td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #30363d; }}
th {{ background:#161b22; color:#8b949e; font-weight:600; }}
tr.ok:hover {{ background:#1c2128; }}
tr.fail {{ color:#8b949e; }}
.ok-badge {{ color:#3fb950; }}
.fail-badge {{ color:#f85149; }}
.leaderboard {{ display:inline-block; vertical-align:top; margin-right:30px; }}
.leaderboard table {{ width:auto; min-width:300px; }}
.subtitle {{ color:#8b949e; font-size:0.9em; }}
</style>
</head>
<body>
<h1>AI Benchmark</h1>
<p class="subtitle">Tasks: {', '.join(p.name for p in active_plugins)}</p>
<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <strong>Total:</strong> {len(results)} | <strong>Successful:</strong> {len(ok)} | <strong>Failed:</strong> {len(results)-len(ok)}</p>

<h2>🏆 Leaderboards</h2>
<div class="leaderboard">
<h3>⚡ Fastest Cold Load</h3>
<table><tr><th>#</th><th>Model</th><th>TTFT</th></tr>
{lb_ttft()}
</table></div>
{leaderboard_html}

<h2>📊 Complete Results</h2>
<table>
<tr>{header_cells}</tr>
{rows}
</table>
</body>
</html>"""


def gen_pdf(results, active_plugins, output_dir):
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Benchmark", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Tasks: {', '.join(p.name for p in active_plugins)}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    ok = [r for r in results if r["status"] == "ok"]
    pdf.cell(0, 6, f"Total: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(results)-len(ok)}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    col_w = [38, 9]
    headers = ["Model", "Load"]
    for p in active_plugins:
        col_w.extend([9, 9, 9, 9])
        headers.extend([f"{p.id[:3].upper()}Rsp", f"{p.id[:3].upper()}TPS", f"{p.id[:3].upper()}Tok", f"{p.id[:3].upper()}Sc"])
    col_w.extend([9, 9])
    headers.extend(["Tot", "Mode"])
    pdf.set_font("Helvetica", "B", 6.5)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 5, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 6.5)
    for r in results:
        tot = _plugin_total_score(r, active_plugins)
        m = "str" if r.get('stream_ok') else "ns"
        vals = [r['model'][:30], str(r.get('ttft') or '-')]
        for p in active_plugins:
            vals.extend([
                str(r.get(f'{p.id}_response_time', '-')),
                str(r.get(f'{p.id}_tps', '-')),
                str(r.get(f'{p.id}_output_tokens', '-')),
                str(r.get(f'{p.id}_score', '-')),
            ])
        if r["status"] == "ok":
            vals.extend([str(tot), m])
        else:
            vals.extend([str(tot), "FAIL"])
        for i, v in enumerate(vals):
            pdf.cell(col_w[i], 4, v, border=1, align="C")
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Leaderboards", new_x="LMARGIN", new_y="NEXT")
    if ok:
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 5, "Fastest TTFT:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7)
        for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:5], 1):
            pdf.cell(0, 4, f"  {i}. {r['model'][:50]}  --  {r['ttft']}s", new_x="LMARGIN", new_y="NEXT")
        for p in active_plugins:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 5, f"Best {p.name}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 7)
            for i, r in enumerate(sorted(ok, key=lambda x: _numeric_score(x, p.id), reverse=True)[:5], 1):
                pdf.cell(0, 4, f"  {i}. {r['model'][:50]}  --  {r.get(f'{p.id}_score', '-')}/{int(p.max_score)}", new_x="LMARGIN", new_y="NEXT")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.pdf")
    pdf.output(path)
    return path


def _save_outputs(state, output_dir, active_plugins):
    """Regenerate CSV/markdown/HTML from latest deduplicated results."""
    results = state.latest_results()
    md = gen_markdown(results, active_plugins, output_dir=output_dir)
    csv_txt = gen_csv(results, active_plugins)
    html = gen_html(results, active_plugins, output_dir=output_dir)
    for fname, content in [("results.md", md), ("results.csv", csv_txt), ("results.html", html)]:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "w") as f:
                f.write(content)
        except OSError:
            pass


# ─── Parallel execution helpers ──────────────────────────────────

class BenchmarkState:
    """Thread-safe shared state for parallel benchmark execution."""
    def __init__(self, models, plugin_ids):
        self._lock = threading.Lock()
        self.results = []
        self._model_info = {}
        self._log = []
        self.plugin_ids = list(plugin_ids)
        for name, source in models.items():
            self._model_info[name] = {
                "source": source,
                "status": "pending",
                "ttft": None,
                "error": None, "elapsed": 0,
                "attempt": 0,
                "max_tok": 0,
                "attempt_start": 0,
                "last_error": "",
                "phase_detail": "",
            }
            for pid in plugin_ids:
                self._model_info[name][f"{pid}_score"] = None
                self._model_info[name][f"{pid}_tps"] = None
                self._model_info[name][f"{pid}_response_time"] = None
                self._model_info[name][f"{pid}_output_tokens"] = None

    def update(self, model_name, **kwargs):
        with self._lock:
            self._model_info[model_name].update(kwargs)

    def add_result(self, result):
        with self._lock:
            self.results.append(result)

    def snapshot(self):
        with self._lock:
            return {k: dict(v) for k, v in self._model_info.items()}

    @property
    def completed(self):
        with self._lock:
            return sum(1 for s in self._model_info.values() if s["status"] == "completed")

    @property
    def total(self):
        return len(self._model_info)

    def log(self, model_name, msg):
        with self._lock:
            self._log.append((time.time(), model_name, msg))
            if len(self._log) > 100:
                self._log = self._log[-100:]

    def recent_log(self, n=5):
        with self._lock:
            return self._log[-n:]

    def save_state(self, path, plugin_versions=None):
        with self._lock:
            data = {
                "model_info": self._model_info,
                "results": self.results,
                "active_plugins": self.plugin_ids,
                "plugin_versions": plugin_versions or {},
            }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def latest_results(self):
        """Return only the most recent result per model (deduplicates across runs)."""
        with self._lock:
            seen = {}
            for r in self.results:
                seen[r["model"]] = r
            return list(seen.values())

    @classmethod
    def load_state(cls, path, models, plugin_ids, *, rerun_failed=True):
        with open(path) as f:
            data = json.load(f)
        state = cls(models, plugin_ids)
        saved_info = data.get("model_info", {})
        for name, info in saved_info.items():
            if name in state._model_info:
                state._model_info[name] = info
        state.results = data.get("results", [])
        for name, info in state._model_info.items():
            if info.get("status") == "completed":
                continue
            if not rerun_failed:
                continue
            # Reset failed (and any other non-completed) models so they are
            # re-run when the benchmark restarts.
            info["status"] = "pending"
            info["last_error"] = ""
            info["error"] = None
            info.setdefault("attempt_start", 0)
        return state


# ─── Model execution ─────────────────────────────────────────────────────────

def _run_plugin_task(model_name, source, plugin, source_config, timeout, token_levels,
                     session_seed, log_file, global_cfg, stop_event=None,
                     save_responses=False, output_dir=None):
    """Run a single plugin task for a model. Returns (result_dict, error)."""
    pid = plugin.id
    cfg = source_config.get(source)
    if cfg is None:
        return None, f"Unknown source '{source}' — not in SOURCE_CONFIG"

    if stop_event and stop_event.is_set():
        return None, "Cancelled"

    prompt = plugin.get_prompt()
    temperature = plugin.get_temperature(global_cfg or {})

    raw_model_cfg = (global_cfg or {}).get("models", {}).get(model_name)
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
                source_config, timeout, model_name, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (Streaming, attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params)

            if serr or first_tok is None:
                text, nsusage, ns_time, nserr, nsfr = nonstream_request(
                    source_config, timeout, model_name, source, prompt, max_tok,
                    log_path=log_file,
                    log_label=f"{plugin.name} (Non-Streaming, attempt {attempt + 1})",
                    session_seed=session_seed, temperature=temperature,
                    drop_params=drop_params)
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
                source_config, timeout, model_name, source, prompt, max_tok,
                log_path=log_file,
                log_label=f"{plugin.name} (attempt {attempt + 1})",
                session_seed=session_seed, temperature=temperature,
                drop_params=drop_params)
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
        responses_dir = os.path.join(output_dir, "responses", sanitize_filename(model_name))
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

    score = plugin.score(text)

    if save_responses and output_dir:
        meta_path = os.path.join(responses_dir, f"{plugin.id}.meta.json")
        meta = {
            "plugin": plugin.id,
            "plugin_version": plugin.version,
            "model": model_name,
            "score": score,
            "response_time": response_time,
            "output_tokens": output_tokens,
            "tps": tps,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except OSError:
            pass

    result = {
        f"{pid}_score": score,
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
              stop_event=None, save_responses=False):
    """Run active plugins for one model."""
    start = time.time()

    r = {
        "model": model_name, "source": source, "status": "ok", "stream_ok": True,
        "ttft": None,
        "prompt_tokens": 0, "completion_tokens": 0,
        "total_time": 0, "error": None,
        "plugin_versions": {p.id: p.version for p in active_plugins},
    }

    state.update(model_name, status="queued")

    cfg = source_config.get(source)
    if cfg is None:
        r["status"] = "error"
        r["error"] = f"Unknown source '{source}' — not in SOURCE_CONFIG"
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(model_name, status="failed", error=r["error"], elapsed=r["total_time"])
        state.log(model_name, r['error'])
        return

    latest = {res["model"]: res for res in state.latest_results()}
    existing = latest.get(model_name)

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
        state.update(model_name, status="completed", elapsed=r["total_time"])
        return

    plugin_thread_limit = (global_cfg or {}).get("plugin_thread_limit", 1)
    try:
        plugin_thread_limit = int(plugin_thread_limit)
    except (TypeError, ValueError):
        plugin_thread_limit = 1
    if plugin_thread_limit <= 0:
        plugin_thread_limit = len(plugins_to_run)

    state.update(model_name, attempt_start=time.time())

    _run_plugins(model_name, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start,
                 max_workers=plugin_thread_limit,
                 stop_event=stop_event,
                 save_responses=save_responses)


def _run_plugins(model_name, source, state, active_plugins, plugins_to_run,
                 source_config, timeout, token_levels, output_dir,
                 session_seed, global_cfg, r, start, max_workers,
                 stop_event=None, save_responses=False):
    """Run plugins for one model using a thread pool of bounded size.

    A single-worker pool (``max_workers=1``) is equivalent to sequential
    execution, so this helper is used for both sequential and parallel
    plugin execution.
    """
    results = {plugin.id: None for plugin in plugins_to_run}
    errors = {}
    lock = threading.Lock()
    logs_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(logs_dir, f"{sanitize_filename(model_name)}.log")

    def run_one(plugin):
        pid = plugin.id
        state.update(model_name, status=f"running_{pid}")
        result, err = _run_plugin_task(model_name, source, plugin, source_config,
                                       timeout, token_levels, session_seed, log_file,
                                       global_cfg or {}, stop_event=stop_event,
                                       save_responses=save_responses,
                                       output_dir=output_dir)
        with lock:
            results[pid] = result
            if err:
                errors[pid] = err
        if err or result is None:
            return
        state.update(model_name,
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
            state.update(model_name, **fail_values)
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
        state.update(model_name, status="failed", error=r["error"], elapsed=r["total_time"], last_error=r["error"])
        return

    if errors:
        r["status"] = "error"
        r["error"] = "; ".join(f"{pid}: {err}" for pid, err in errors.items())
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(model_name, status="failed", error=r["error"], elapsed=r["total_time"], last_error=r["error"])
        return

    r["total_time"] = round(time.time() - start, 1)
    state.add_result(r)
    state.update(model_name, status="completed", elapsed=r["total_time"])
