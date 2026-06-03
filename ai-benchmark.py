#!/usr/bin/env python3
"""
AI Benchmark — Harder tasks, ALL models, retry logic on streaming failures.
Code: Concurrent Rate Limiter (0-20)
General: MoE vs Dense Architecture deep-dive (0-15)
Outputs: Markdown, HTML, CSV, PDF

Configuration: edit benchmark-config.json (or pass --config <path>).
API keys can use ${VAR} or ${VAR:default} syntax for env-var expansion.
"""
import requests, time, json, re, sys, os, csv, io, threading, random, glob, argparse, string
from datetime import datetime

DEFAULT_CONFIG_PATH = "benchmark-config.json"

TIMEOUT = None          # set from config in main()
TOKEN_LEVELS = None     # set from config in main()
SOURCE_CONFIG = None    # loaded from config file
MODELS = None           # loaded from config file


# ─── Config loading ──────────────────────────────────────────────────────────

def _expand_env(val):
    """Recursively expand ${VAR} or ${VAR:default} in strings."""
    if isinstance(val, str):
        t = string.Template(val)
        # safe_substitute leaves ${UNSET} as-is; we want to fall back to default
        parts = []
        for raw in [val]:
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


def dump_default_config():
    """Print the default config JSON to stdout."""
    cfg = {
        "output_dir": "benchmark-output-dir",
        "timeout": 1200,
        "token_levels": [16384],
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
            "example-model-2": "Remote Provider 1"
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
        print("❌ No models returned by /v1/models endpoint.", file=sys.stderr)
        sys.exit(1)

    source_name = "Default"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    cfg = {
        "output_dir": "benchmark-results",
        "timeout": 600,
        "token_levels": [16384],
        "sources": {
            source_name: {
                "api_url": base_url.rstrip("/") + "/chat/completions",
                "headers": headers,
            }
        },
        "models": {mid: source_name for mid in model_ids},
    }
    return cfg


CODE_TASK = (
    "Design and implement a concurrent rate limiter in Python with the following specifications:\n\n"
    "ARCHITECTURE:\n"
    "- Support multiple rate limiting strategies: Token Bucket, Sliding Window Log, and Fixed Window\n"
    "- Use a clean abstract base class or protocol that all strategies implement\n"
    "- Thread-safe throughout with minimal lock contention\n\n"
    "REQUIREMENTS:\n"
    "- Configurable per-client limits (each client_id can have different rate limits)\n"
    "- Method: allow_request(client_id: str) -> bool — returns True if request is allowed\n"
    "- Method: get_usage_stats(client_id: str) -> dict — returns current usage info\n"
    "- Efficient automatic cleanup of stale client entries to prevent memory leaks\n"
    "- Handle edge cases: burst traffic, zero limits, concurrent requests from same client\n\n"
    "IMPLEMENTATION:\n"
    "- Implement at least TokenBucket and SlidingWindowLog strategies fully\n"
    "- Token Bucket: tokens refill at a configurable rate, burst capacity = bucket size\n"
    "- Sliding Window Log: track timestamps of recent requests, O(log n) for inserts\n"
    "- Fixed Window: simple counter per time window, resets at window boundary\n\n"
    "Return the complete implementation with all classes and a usage example."
)

GENERAL_TASK = (
    "Write a detailed technical analysis comparing Mixture-of-Experts (MoE) architecture "
    "(used in Mixtral 8x7B, Qwen3-MoE, DeepSeekMoE) versus dense transformer architecture "
    "(used in Llama 3, Gemma, GPT-4o). Your analysis must cover:\n\n"
    "- The mathematical formulation of the sparse MoE gating/routing mechanism (include the top-k "
    "routing equation and softmax gating)\n"
    "- How the auxiliary load-balancing loss works — include the exact mathematical formulation\n"
    "- At least 2 specific training stability challenges unique to MoE (token dropping, expert "
    "collapse, or others)\n"
    "- Inference implications: memory bandwidth, expert parallelism, vs dense compute patterns\n"
    "- Specific benchmarks or tasks where MoE outperforms dense architectures, and where dense "
    "outperforms MoE (name at least 2 of each)\n"
    "- Reference at least 2 specific papers, technical reports, or model cards\n\n"
    "Be precise and technical — this is for an ML engineering audience. 4-5 paragraphs."
)

OUTPUT_DIR = None       # set from config in main()
STATE_FILE = None       # set from config in main()
SOURCE_CONFIG = None    # loaded from config file
MODELS = None           # loaded from config file


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_code(response):
    """Rate limiter quality (0-20)."""
    t = response
    s = 0.0

    # 1. Interface design (0-3)
    if re.search(r'(?:ABC|abstractmethod|Protocol|ABCMeta)', t): s += 2.0
    elif re.search(r'(?:class\s+\w+RateLimiter|class\s+Base)', t): s += 1.0
    if re.search(r'allow_request', t) and re.search(r'get_usage_stats|get_usage|usage_stats', t): s += 1.0

    # 2. Token Bucket (0-4)
    if re.search(r'(?:class\s+TokenBucket|TokenBucket)', t): s += 1.0
    if re.search(r'(?:refill|last_refill|_refill|tokens\s*[+=-])', t): s += 1.5
    if re.search(r'(?:consume|allow|acquire|try_acquire)', t) and re.search(r'(?:tokens?\s*[>=-]|if\s+\w+\s*[>=-])', t): s += 1.5

    # 3. Sliding Window (0-3)
    if re.search(r'(?:class\s+SlidingWindow|SlidingWindowLog)', t): s += 1.0
    if re.search(r'(?:timestamp|time\.time|deque|list|sorted|bisect)', t) and re.search(r'(?:window|log|history)', t.lower()): s += 1.0
    if re.search(r'(?:prune|clean|remove_old|pop.*while|while.*pop|deque.*popleft)', t): s += 1.0

    # 4. Thread safety (0-3)
    if re.search(r'(?:threading\.Lock|threading\.RLock|from threading import)', t): s += 1.5
    if re.search(r'(?:with\s+.*lock|with\s+.*mutex|\.acquire|\.release)', t, re.IGNORECASE): s += 1.5

    # 5. Cleanup/memory management (0-2)
    if re.search(r'(?:cleanup|clean_up|remove_stale|expire|ttl|timeout)', t.lower()): s += 1.0
    if re.search(r'(?:background|thread.*clean|scheduler|Timer|loop.*clean)', t.lower()): s += 1.0

    # 6. Type hints (0-2)
    if re.search(r'->\s*(?:bool|dict|int|None|str|float)', t): s += 1.0
    if re.search(r':\s*(?:int|str|bool|float|dict|list|Optional|Callable|Type)', t): s += 1.0

    # 7. Docstrings (0-2)
    if '"""' in t: s += 1.0
    if t.count('"""') >= 4 or "'''" in t: s += 1.0

    # 8. Error handling (0-1)
    if re.search(r'(?:raise\s+|try\s*:|except\s+|ValueError|TypeError|Invalid)', t): s += 1.0

    return round(s, 1)


def score_general(response):
    """MoE vs Dense architecture analysis (0-15)."""
    t = response.lower()
    s = 0.0

    # 1. Covers both architectures explicitly (0-2)
    if re.search(r'(?:mixture.of.expert|moe|sparse.*moe)', t): s += 1.0
    if re.search(r'(?:dense\s*(?:transformer|model|architecture)|standard\s*transformer)', t): s += 1.0

    # 2. Gating/routing mechanism (0-2.5)
    if re.search(r'(?:gating|routing|gate|router|top.k|softmax.*gate)', t): s += 1.5
    if re.search(r'(?:expert.*select|which.*expert|rout.*token)', t): s += 1.0

    # 3. Load-balancing loss (0-2.5)
    if re.search(r'(?:load.balanc|auxiliary.*loss|aux.*loss|balance.*loss)', t): s += 1.5
    if re.search(r'(?:importance|loss.*formula|load.*equation|L_aux)', t): s += 1.0

    # 4. Training challenges (0-2)
    if re.search(r'(?:token.dropp|expert.collaps|instability|collapse|dropping)', t): s += 1.0
    if re.search(r'(?:training.*challeng|difficult|problem|issue|stability)', t): s += 1.0

    # 5. Inference implications (0-2)
    if re.search(r'(?:inference|memory.*bandwidth|expert.*parallel|sparse.*compute)', t): s += 1.0
    if re.search(r'(?:throughput|latency|batch.*size|parameter.*efficien)', t): s += 1.0

    # 6. Specific benchmarks / performance comparison (0-2)
    if re.search(r'(?:benchmark|mmlu|gsm8k|human-eval|mbpp|hellaswag|arc|truthful)', t): s += 1.0
    if re.search(r'(?:outperform|better.*than|compared to|vs\.|versus|advantage)', t): s += 1.0

    # 7. Paper references (0-2)
    if re.search(r'(?:paper|report|arxiv|technical.*report)', t): s += 1.0
    if re.search(r'(?:2023|2024|2025|et\s*al|vashwani|shazeer|fedus|lepikhin|du et al)', t): s += 1.0

    return round(s, 1)


def extract_vram(model_name):
    m = re.search(r'(\d+\.?\d*)\s*G(?:B)?', model_name)
    return m.group(1) + 'G' if m else '?'


# ─── Logging helpers ────────────────────────────────────────────────────────

def sanitize_filename(name):
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


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
    with open(log_path, 'a') as f:
        if request_label:
            f.write(f"\n# === {request_label} ===\n")
        f.write(f"{curl_cmd}\n\n")
        f.write(f"{response_body}\n")
        f.write("\n" + "-" * 60 + "\n")


# ─── API helpers ────────────────────────────────────────────────────────────

def stream_request(model, source, prompt, max_tokens=2048, api_url=None, headers=None, log_path=None, log_label=None, session_seed=0):
    start = time.time()
    first_tok = None
    text = ""
    error = None
    finish_reason = None
    usage = {}
    cfg = SOURCE_CONFIG.get(source, {})
    api_url = api_url or cfg.get("api_url", "http://localhost:11434/chat/completions")
    headers = headers or cfg.get("headers", {"Content-Type": "application/json"})
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, True, api_url, headers) if log_path else None
    try:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True}
        if session_seed:
            body["seed"] = session_seed
        resp = requests.post(
            api_url, headers=headers, json=body, stream=True, timeout=TIMEOUT)
        if resp.status_code != 200:
            if log_path and curl_cmd:
                log_request_entry(log_path, curl_cmd,
                                  f"HTTP {resp.status_code}: {resp.text[:500]}", log_label)
            return text, first_tok, time.time(), f"HTTP {resp.status_code}: {resp.text[:150]}", finish_reason, usage
        # ── Total-duration watchdog ────────────────────────────────────
        # The per-recv socket timeout (TIMEOUT) can be defeated by TCP
        # keep-alives or empty chunks from the proxy.  A daemon timer
        # that closes the response after TIMEOUT seconds guarantees the
        # iter_lines() loop terminates.
        watchdog = threading.Timer(TIMEOUT, resp.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line: continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() == "[DONE]": break
                    try:
                        data = json.loads(payload)
                        if first_tok is None: first_tok = time.time()
                        for ch in data.get("choices", []):
                            text += ch.get("delta", {}).get("content", "")
                            fr = ch.get("finish_reason")
                            if fr:
                                finish_reason = fr
                        if "usage" in data:
                            usage = data["usage"]
                    except json.JSONDecodeError: continue
        finally:
            watchdog.cancel()
        if not finish_reason and time.time() - start > TIMEOUT:
            error = f"Total timeout ({TIMEOUT}s) exceeded"
        if log_path and curl_cmd:
            log_request_entry(log_path, curl_cmd, text or "(empty response)", log_label)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if log_path and curl_cmd:
            log_request_entry(log_path, curl_cmd, f"ERROR: {error}", log_label)
    return text, first_tok, time.time(), error, finish_reason, usage


def nonstream_request(model, source, prompt, max_tokens=2048, api_url=None, headers=None, log_path=None, log_label=None, session_seed=0):
    start = time.time()
    error = None; text = ""; usage = {}; finish_reason = None
    cfg = SOURCE_CONFIG.get(source, {})
    api_url = api_url or cfg.get("api_url", "http://localhost:11434/chat/completions")
    headers = headers or cfg.get("headers", {"Content-Type": "application/json"})
    curl_cmd = build_curl_cmd(model, prompt, max_tokens, False, api_url, headers) if log_path else None
    raw_resp_text = None
    try:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": False}
        if session_seed:
            body["seed"] = session_seed
        resp = requests.post(
            api_url, headers=headers, json=body, timeout=TIMEOUT)
        # ── Total-duration watchdog ────────────────────────────────────
        # Same issue as streaming: TCP keep-alives from the proxy can
        # defeat the per-recv timeout when reading the response body.
        watchdog = threading.Timer(TIMEOUT, resp.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            if resp.status_code != 200:
                raw_resp_text = resp.text[:500]
                if log_path and curl_cmd:
                    log_request_entry(log_path, curl_cmd,
                                      f"HTTP {resp.status_code}: {raw_resp_text}", log_label)
                return text, usage, time.time()-start, f"HTTP {resp.status_code}: {resp.text[:150]}", finish_reason
            raw_resp_text = resp.text
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            finish_reason = data.get("choices", [{}])[0].get("finish_reason")
            if log_path and curl_cmd:
                log_request_entry(log_path, curl_cmd, raw_resp_text, log_label)
        finally:
            watchdog.cancel()
        if not error and time.time() - start > TIMEOUT:
            error = f"Total timeout ({TIMEOUT}s) exceeded"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if log_path and curl_cmd:
            body = raw_resp_text or f"EXCEPTION: {error}"
            log_request_entry(log_path, curl_cmd, body, log_label)
    return text, usage, time.time()-start, error, finish_reason


def count_tokens(text):
    return max(1, len(text) / 4)


def is_repeating(text, min_seq=80, repeats=3):
    """Detect if text is stuck in a loop. Checks if the last `min_seq` chars
    appear `repeats+` times in the text — a sign the model is repeating."""
    if len(text) < min_seq * repeats:
        return False
    tail = text[-min_seq:]
    return text.count(tail) >= repeats


# ─── Output generators ──────────────────────────────────────────────────────

def gen_markdown(results):
    ok = [r for r in results if r["status"] == "ok"]
    lines = [
        "# AI Benchmark — Complex Tasks + Retry",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Server:** nas.server:11434 → gaming.pc (16GB VRAM)",
        f"**Code Task:** Concurrent Rate Limiter (0-20) | **General Task:** MoE vs Dense (0-15)",
        f"**Total:** {len(results)} models | **✅ {len(ok)} successful** | **❌ {len(results)-len(ok)} failed**",
        "",
        "## 📋 Complete Results",
        "",
        "| # | Model | VRAM | Load (s) | Code Resp (s) | Gen Resp (s) | Code TPS | Gen TPS | Code Tok | Gen Tok | Code /20 | Gen /15 | Total /35 | Time | Mode |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(results, 1):
        if r["status"] == "ok":
            tot = r.get('code_score',0) + r.get('gen_score',0)
            m = "stream" if r.get('stream_ok') else "nostream"
            lines.append(
                f"| {idx} | {r['model']} | {extract_vram(r['model'])} | {r['ttft'] or '-'} | "
                f"{r.get('code_response_time','-')} | {r.get('gen_response_time','-')} | "
                f"{r.get('output_tps_code','-')} | {r.get('output_tps_gen','-')} | "
                f"{r.get('code_output_tokens','-')} | {r.get('gen_output_tokens','-')} | "
                f"{r['code_score']} | {r['gen_score']} | {tot} | {r['total_time']}s | {m} |")
        else:
            err = (r.get('error') or '?')[:60]
            lines.append(f"| {idx} | {r['model']} | - | - | - | - | - | - | - | - | - | - | - | - | ❌ {err} |")

    # Leaderboards
    if ok:
        lines.extend(["", "---", "## 🏆 Leaderboards", ""])

        lines.append("### ⚡ Fastest Cold Load (TTFT)")
        lines.append("| # | Model | Load (s) |")
        lines.append("|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1):
            lines.append(f"| {i} | {r['model']} | {r['ttft']} |")

        lines.extend(["", "### 💻 Best Code Score (/20)"])
        lines.append("| # | Model | Code | Gen | Total |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0), reverse=True)[:10], 1):
            tot = r.get('code_score',0) + r.get('gen_score',0)
            lines.append(f"| {i} | {r['model']} | {r['code_score']} | {r['gen_score']} | {tot} |")

        lines.extend(["", "### 🧠 Best General Score (/15)"])
        lines.append("| # | Model | Gen /15 |")
        lines.append("|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('gen_score',0), reverse=True)[:10], 1):
            lines.append(f"| {i} | {r['model']} | {r['gen_score']} |")

        lines.extend(["", "### ⭐ Best Combined (/35)"])
        lines.append("| # | Model | Code /20 | Gen /15 | Total /35 |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0)+x.get('gen_score',0), reverse=True)[:10], 1):
            tot = r.get('code_score',0) + r.get('gen_score',0)
            lines.append(f"| {i} | {r['model']} | {r['code_score']} | {r['gen_score']} | {tot} |")

    lines.extend([
        "", "---", "## 📐 Scoring Rubric", "",
        "### Code Task: Concurrent Rate Limiter (0-20)",
        "| Criterion | Max | Description |",
        "|---|---|---|",
        "| Interface design | 3 | ABC/Protocol, clean allow_request/get_usage_stats |",
        "| Token Bucket | 4 | Class, refill logic, consume logic |",
        "| Sliding Window | 3 | Class, timestamp tracking, pruning |",
        "| Thread safety | 3 | Locking, minimal contention |",
        "| Cleanup | 2 | Stale entry eviction |",
        "| Type hints | 2 | Parameter & return annotations |",
        "| Docstrings | 2 | Comprehensive documentation |",
        "| Error handling | 1 | Input validation, exceptions |",
        "",
        "### General Task: MoE vs Dense Architecture (0-15)",
        "| Criterion | Max | Description |",
        "|---|---|---|",
        "| Both architectures covered | 2 | Explicitly discusses MoE and dense |",
        "| Gating/routing mechanism | 2.5 | Top-k routing, softmax gating equations |",
        "| Load-balancing loss | 2.5 | Auxiliary loss formulation |",
        "| Training challenges | 2 | Token dropping, expert collapse, etc. |",
        "| Inference implications | 2 | Memory bandwidth, expert parallelism |",
        "| Specific benchmarks | 2 | MMLU, GSM8K, etc. with comparisons |",
        "| Paper references | 2 | Specific papers, technical reports |",
        "",
        "## ❌ Failed Models",
        "",
    ])
    for r in results:
        if r["status"] != "ok":
            lines.append(f"- **{r['model']}**: {r.get('error','?')}")
    lines.append("")
    return "\n".join(lines)


def gen_csv(results):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Model","Source","VRAM","TTFT_s",
                 "Code_Response_s","Code_Output_Tokens","Output_TPS_Code","Code_Score_20",
                 "Gen_Response_s","Gen_Output_Tokens","Output_TPS_Gen","Gen_Score_15",
                 "Total_35","Time_s","Mode","Status","Error"])
    for r in results:
        if r["status"] == "ok":
            tot = r.get('code_score',0) + r.get('gen_score',0)
            m = "stream" if r.get('stream_ok') else "non-streaming"
            w.writerow([r['model'], r.get('source',''), extract_vram(r['model']), r['ttft'] or '',
                        r.get('code_response_time',''), r.get('code_output_tokens',''),
                        r.get('output_tps_code',''), r['code_score'],
                        r.get('gen_response_time',''), r.get('gen_output_tokens',''),
                        r.get('output_tps_gen',''), r['gen_score'],
                        tot, r['total_time'], m, "OK", ""])
        else:
            cols = 17
            row = [""] * cols
            row[0] = r['model']
            row[1] = r.get('source','')
            row[-3] = "FAIL"
            row[-2] = r.get('error','')
            w.writerow(row)
    return out.getvalue()


def gen_html(results):
    ok = [r for r in results if r["status"] == "ok"]
    rows = ""
    for r in results:
        cls = "ok" if r["status"] == "ok" else "fail"
        if r["status"] == "ok":
            tot = r.get('code_score',0) + r.get('gen_score',0)
            m = "str" if r.get('stream_ok') else "ns"
            rows += (
                f'<tr class="{cls}">'
                f'<td>{r["model"]}</td><td>{extract_vram(r["model"])}</td>'
                f'<td>{r["ttft"] or "-"}</td><td>{r.get("code_response_time","-")}</td>'
                f'<td>{r.get("gen_response_time","-")}</td><td>{r.get("output_tps_code","-")}</td>'
                f'<td>{r.get("output_tps_gen","-")}</td>'
                f'<td>{r.get("code_output_tokens","-")}</td><td>{r.get("gen_output_tokens","-")}</td>'
                f'<td><strong>{r["code_score"]}</strong></td>'
                f'<td>{r["gen_score"]}</td><td><strong>{tot}</strong></td>'
                f'<td>{r["total_time"]}s</td><td>{m}</td>'
                f'<td class="ok-badge">✅</td></tr>\n'
            )
        else:
            err = (r.get('error') or '?')[:50]
            rows += f'<tr class="fail"><td>{r["model"]}</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td class="fail-badge">❌ {err}</td></tr>\n'

    # Leaderboards
    def lb_ttft():
        out = ""
        for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1):
            out += f'<tr><td>{i}</td><td>{r["model"]}</td><td>{r["ttft"]}s</td></tr>\n'
        return out
    def lb_code():
        out = ""
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0), reverse=True)[:10], 1):
            out += f'<tr><td>{i}</td><td>{r["model"]}</td><td><strong>{r["code_score"]}/20</strong></td></tr>\n'
        return out
    def lb_comb():
        out = ""
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0)+x.get('gen_score',0), reverse=True)[:10], 1):
            tot = r.get('code_score',0) + r.get('gen_score',0)
            out += f'<tr><td>{i}</td><td>{r["model"]}</td><td>{r["code_score"]}/20</td><td>{r["gen_score"]}/15</td><td><strong>{tot}/35</strong></td></tr>\n'
        return out

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
<p class="subtitle">Code: Rate Limiter | General: MoE vs Dense Architecture</p>
<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <strong>Total:</strong> {len(results)} | <strong>Successful:</strong> {len(ok)} | <strong>Failed:</strong> {len(results)-len(ok)}</p>

<h2>🏆 Leaderboards</h2>
<div class="leaderboard">
<h3>⚡ Fastest Cold Load</h3>
<table><tr><th>#</th><th>Model</th><th>TTFT</th></tr>
{lb_ttft()}
</table></div>
<div class="leaderboard">
<h3>💻 Best Code /20</h3>
<table><tr><th>#</th><th>Model</th><th>Score</th></tr>
{lb_code()}
</table></div>
<div class="leaderboard">
<h3>⭐ Best Combined /35</h3>
<table><tr><th>#</th><th>Model</th><th>Code</th><th>Gen</th><th>Total</th></tr>
{lb_comb()}
</table></div>

<h2>📊 Complete Results</h2>
<table>
<tr><th>Model</th><th>VRAM</th><th>Load(s)</th><th>Code Resp(s)</th><th>Gen Resp(s)</th><th>Code TPS</th><th>Gen TPS</th><th>Code Tok</th><th>Gen Tok</th><th>Code/20</th><th>Gen/15</th><th>Total/35</th><th>Time</th><th>Mode</th><th>Status</th></tr>
{rows}
</table>

<h2>📐 Scoring</h2>
<h3>Code: Concurrent Rate Limiter (0-20)</h3>
<table><tr><th>Criterion</th><th>Max</th></tr>
<tr><td>Interface design</td><td>3</td></tr><tr><td>Token Bucket</td><td>4</td></tr>
<tr><td>Sliding Window</td><td>3</td></tr><tr><td>Thread safety</td><td>3</td></tr>
<tr><td>Cleanup</td><td>2</td></tr><tr><td>Type hints</td><td>2</td></tr>
<tr><td>Docstrings</td><td>2</td></tr><tr><td>Error handling</td><td>1</td></tr>
</table>
<h3>General: MoE vs Dense (0-15)</h3>
<table><tr><th>Criterion</th><th>Max</th></tr>
<tr><td>Both architectures</td><td>2</td></tr><tr><td>Gating/routing</td><td>2.5</td></tr>
<tr><td>Load-balancing loss</td><td>2.5</td></tr><tr><td>Training challenges</td><td>2</td></tr>
<tr><td>Inference implications</td><td>2</td></tr><tr><td>Specific benchmarks</td><td>2</td></tr><tr><td>Paper references</td><td>2</td></tr>
</table>
</body>
</html>"""


def gen_pdf(results):
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
    pdf.cell(0, 6, f"Rate Limiter Code (0-20) + MoE vs Dense (0-15)  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    ok = [r for r in results if r["status"] == "ok"]
    pdf.cell(0, 6, f"Total: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(results)-len(ok)}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Table
    col_w = [38, 7, 9, 9, 9, 9, 9, 8, 8, 9, 9, 9]
    pdf.set_font("Helvetica", "B", 6.5)
    for i, h in enumerate(["Model","V","Load","CResp","GResp","Ctps","Gtps","CTok","GTok","Code","Gen","Tot"]):
        pdf.cell(col_w[i], 5, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 6.5)
    for r in results:
        if r["status"] == "ok":
            tot = r.get('code_score',0)+r.get('gen_score',0)
            vals = [r['model'][:30], extract_vram(r['model']), str(r['ttft'] or '-'),
                    str(r.get('code_response_time','-')), str(r.get('gen_response_time','-')),
                    str(r.get('output_tps_code','-')), str(r.get('output_tps_gen','-')),
                    str(r.get('code_output_tokens','-')), str(r.get('gen_output_tokens','-')),
                    str(r['code_score']), str(r['gen_score']), str(tot)]
        else:
            vals = [r['model'][:30], '-','-','-','-','-','-','-','-','-','-','FAIL']
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
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 5, "Best Code Score:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7)
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0), reverse=True)[:5], 1):
            pdf.cell(0, 4, f"  {i}. {r['model'][:50]}  --  {r['code_score']}/20", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 5, "Best Combined:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7)
        for i, r in enumerate(sorted(ok, key=lambda x: x.get('code_score',0)+x.get('gen_score',0), reverse=True)[:5], 1):
            tot = r.get('code_score',0)+r.get('gen_score',0)
            pdf.cell(0, 4, f"  {i}. {r['model'][:50]}  --  {tot}/35", new_x="LMARGIN", new_y="NEXT")

    path = os.path.join(OUTPUT_DIR, "results.pdf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf.output(path)
    return path


# ─── Parallel execution helpers ──────────────────────────────────

class BenchmarkState:
    """Thread-safe shared state for parallel benchmark execution."""
    def __init__(self, models):
        self._lock = threading.Lock()
        self.results = []
        self._model_info = {}
        self._log = []
        for name, source in models.items():
            self._model_info[name] = {
                "source": source,
                "status": "pending",
                "code_score": None, "gen_score": None,
                "ttft": None, "code_tps": None, "gen_tps": None,
                "code_response_time": None, "gen_response_time": None,
                "code_time": 0, "gen_time": 0,
                "code_output_tokens": None, "gen_output_tokens": None,
                "error": None, "elapsed": 0,
                "attempt": 0,
                "max_tok": 0,
                "attempt_start": 0,
                "last_error": "",
                "phase_detail": "",
            }

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
            return sum(1 for s in self._model_info.values() if s["status"] in ("completed", "failed"))

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

    def save_state(self, path):
        with self._lock:
            data = {"model_info": self._model_info, "results": self.results}
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
    def load_state(cls, path, models):
        with open(path) as f:
            data = json.load(f)
        state = cls(models)
        # Merge saved state into fresh model_info — models in both use saved status,
        # while models newly added to config keep their fresh "pending" status.
        saved_info = data.get("model_info", {})
        for name, info in saved_info.items():
            if name in state._model_info:
                state._model_info[name] = info
        state.results = data.get("results", [])
        for name, info in state._model_info.items():
            if info.get("status") not in ("completed", "failed"):
                info["status"] = "pending"
                info["last_error"] = ""
            info.pop("attempt_start", None)
        return state


def _save_outputs(state, output_dir):
    """Regenerate CSV/markdown/HTML from latest deduplicated results."""
    results = state.latest_results()
    md = gen_markdown(results)
    csv_txt = gen_csv(results)
    html = gen_html(results)
    for fname, content in [("results.md", md), ("results.csv", csv_txt), ("results.html", html)]:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "w") as f:
                f.write(content)
        except OSError:
            pass


def run_model(model_name, source, state, session_seed=0):
    """Run code + general tasks for one model. Acquires source_lock for API calls."""
    logs_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = os.path.join(logs_dir, f"{sanitize_filename(model_name)}.log")
    global TOKEN_LEVELS
    start = time.time()

    r = {
        "model": model_name, "source": source, "status": "ok", "stream_ok": True,
        "ttft": None, "output_tps_code": None, "output_tps_gen": None,
        "code_output_tokens": 0, "gen_output_tokens": 0,
        "code_score": 0, "gen_score": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "code_response_time": 0, "gen_response_time": 0,
        "total_time": 0, "error": None,
    }

    state.update(model_name, status="queued")

    # Resolve source config — fail gracefully if source not found
    cfg = SOURCE_CONFIG.get(source)
    if cfg is None:
        r["status"] = "error"
        r["error"] = f"Unknown source '{source}' — not in SOURCE_CONFIG"
        r["total_time"] = round(time.time() - start, 1)
        state.add_result(r)
        state.update(model_name, status="failed", error=r["error"], elapsed=r["total_time"])
        state.log(model_name, r['error'])
        return

    # ── Phase 1: Code task with truncation retry ──
    state.update(model_name, status="running_code")

    code_text = ""
    code_truncated = False
    code_repeating = False
    code_gen_time = 0

    for attempt, max_tok in enumerate(TOKEN_LEVELS):
        attempt_start = time.time()
        state.update(model_name, attempt=attempt + 1, max_tok=max_tok,
                     attempt_start=attempt_start,
                     phase_detail="Code", status="running_code")

        text, first_tok, stream_end, serr, sfr, susage = stream_request(
            model_name, source, CODE_TASK, max_tok,
            api_url=cfg["api_url"], headers=cfg["headers"],
            log_path=log_file,
            log_label=f"Code Task (Streaming, attempt {attempt + 1})",
            session_seed=session_seed)

        if serr or first_tok is None:
            # If streaming timed out, non-streaming will hit the same
            # keep-alive wall — fail fast instead of hanging again.
            if serr and "ReadTimeout" in serr:
                r["status"] = "error"
                r["error"] = serr
                r["total_time"] = round(time.time() - start, 1)
                state.add_result(r)
                state.update(model_name, status="failed",
                             error=r["error"], elapsed=r["total_time"],
                             last_error=r["error"])
                return

            text, nsusage, ns_time, nserr, nsfr = nonstream_request(
                model_name, source, CODE_TASK, max_tok,
                api_url=cfg["api_url"], headers=cfg["headers"],
                log_path=log_file,
                log_label=f"Code Task (Non-Streaming, attempt {attempt + 1})",
                session_seed=session_seed)

            seed_retried = False

            if nserr:
                if session_seed and re.search(
                        r"(?i)not support.*seed|unsupported.*seed|unknown.*param.*seed",
                        (serr or '') + ' ' + (nserr or '')):
                    state.log(model_name, "seed parameter rejected, retrying without seed")
                    session_seed = 0
                    seed_retried = True
                    text, ft_r, se_r, serr_r, sfr_r, _ = stream_request(
                        model_name, source, CODE_TASK, max_tok,
                        api_url=cfg["api_url"], headers=cfg["headers"],
                        log_path=log_file,
                        log_label=f"Code Task (Streaming, no-seed retry)",
                        session_seed=0)
                    if serr_r or ft_r is None:
                        # Also skip nonstream if the seed-retry stream timed out
                        if serr_r and "ReadTimeout" in serr_r:
                            r["status"] = "error"
                            r["error"] = serr_r
                            r["total_time"] = round(time.time() - start, 1)
                            state.add_result(r)
                            state.update(model_name, status="failed",
                                         error=r["error"], elapsed=r["total_time"],
                                         last_error=r["error"])
                            return
                        text, _, ns_t_r, nserr_r, nsfr_r = nonstream_request(
                            model_name, source, CODE_TASK, max_tok,
                            api_url=cfg["api_url"], headers=cfg["headers"],
                            log_path=log_file,
                            log_label=f"Code Task (Non-Streaming, no-seed retry)",
                            session_seed=0)
                        if nserr_r:
                            r["status"] = "error"
                            r["error"] = (f"Stream: {serr_r or 'no tokens'}. "
                                          f"Nostream: {nserr_r}")
                            r["total_time"] = round(time.time() - start, 1)
                            state.add_result(r)
                            state.update(model_name, status="failed",
                                         error=r["error"], elapsed=r["total_time"],
                                         last_error=r["error"])
                            return
                        r["stream_ok"] = False
                        r["ttft"] = "N/A (non-streaming)"
                        r["code_response_time"] = round(ns_t_r, 1)
                        code_gen_time = ns_t_r
                        code_text = text
                        code_truncated = (nsfr_r == "length")
                    else:
                        r["stream_ok"] = True
                        r["ttft"] = round(ft_r - attempt_start, 3)
                        r["code_response_time"] = round(se_r - attempt_start, 1)
                        code_gen_time = se_r - ft_r
                        code_text = text
                        code_truncated = (sfr_r == "length")
                elif attempt == 0:
                    r["status"] = "error"
                    r["error"] = (f"Stream: {serr or 'no tokens'}. "
                                  f"Nostream: {nserr}")
                    r["total_time"] = round(time.time() - start, 1)
                    state.add_result(r)
                    state.update(model_name, status="failed",
                                 error=r["error"], elapsed=r["total_time"],
                                 last_error=r["error"])
                    return
                else:
                    state.update(model_name, last_error=nserr)
                    continue

            if not seed_retried:
                # Non-streaming succeeded
                r["stream_ok"] = False
                r["ttft"] = "N/A (non-streaming)"
                r["code_response_time"] = round(ns_time, 1)
                code_gen_time = ns_time
                code_text = text
                code_truncated = (nsfr == "length")
        else:
            # Streaming succeeded
            r["stream_ok"] = True
            r["ttft"] = round(first_tok - attempt_start, 3)
            r["code_response_time"] = round(stream_end - attempt_start, 1)
            code_gen_time = stream_end - first_tok
            code_text = text
            code_truncated = (sfr == "length")

        r["code_score"] = score_code(code_text)
        if code_gen_time > 0:
            est_tok = count_tokens(code_text)
            r["code_output_tokens"] = int(est_tok)
            r["output_tps_code"] = round(est_tok / code_gen_time, 2)

        if not code_truncated:
            state.update(model_name, last_error="")
            break

        # Check for repetition — model is stuck in a loop
        if is_repeating(code_text):
            code_repeating = True
            state.update(model_name, last_error="Repeating, stopping")
            break

        # Check for thinking exhaustion — model burned all tokens on reasoning
        if len(code_text.strip()) < 50:
            state.update(model_name,
                         last_error=f"Thinking exhausted at {max_tok}")
            state.log(model_name, f"Code task exhausted thinking (tok={max_tok}, out={len(code_text.strip())}c)")

        # Retry with larger token limit
        if attempt < len(TOKEN_LEVELS) - 1:
            state.update(model_name,
                         last_error=f"Truncated at {max_tok}, retrying")
            state.log(model_name, f"Code task truncated at {max_tok}, retrying {TOKEN_LEVELS[attempt + 1]}")

    if code_truncated:
        if code_repeating:
            r["code_warning"] = "Truncated at max tokens (repeating)"
        else:
            r["code_warning"] = f"Truncated at max tokens (exhausted {TOKEN_LEVELS[-1]})"

    state.update(model_name, status="running_gen",
                 code_score=r["code_score"], ttft=r["ttft"],
                 code_tps=r["output_tps_code"],
                 code_response_time=r["code_response_time"],
                 code_time=round(time.time() - start, 1),
                 code_output_tokens=r["code_output_tokens"])

    # ── Phase 2: General task with truncation retry ──
    gen_text = ""
    gen_truncated = False
    gen_repeating = False

    for attempt, max_tok in enumerate(TOKEN_LEVELS):
        attempt_start = time.time()
        state.update(model_name, attempt=attempt + 1, max_tok=max_tok,
                     attempt_start=attempt_start,
                     phase_detail="Gen", status="running_gen")

        text, usage, gen_time, gen_err, gen_fr = nonstream_request(
            model_name, source, GENERAL_TASK, max_tok,
            api_url=cfg["api_url"], headers=cfg["headers"],
            log_path=log_file,
            log_label=f"General Task (attempt {attempt + 1})",
            session_seed=session_seed)

        if gen_err:
            seed_retried_gen = False
            if attempt == 0:
                if session_seed and re.search(
                        r"(?i)not support.*seed|unsupported.*seed|unknown.*param.*seed",
                        gen_err):
                    state.log(model_name, "seed parameter rejected (gen), retrying without seed")
                    session_seed = 0
                    seed_retried_gen = True
                    text, usage, gen_time, gen_err, gen_fr = nonstream_request(
                        model_name, source, GENERAL_TASK, max_tok,
                        api_url=cfg["api_url"], headers=cfg["headers"],
                        log_path=log_file,
                        log_label=f"General Task (no-seed retry)",
                        session_seed=0)
                    if not gen_err:
                        pass
                    else:
                        r["gen_response_time"] = round(gen_time, 1)
                        break
                else:
                    r["gen_response_time"] = round(gen_time, 1)
                    break
            if not seed_retried_gen:
                state.update(model_name, last_error=gen_err)
                continue

        gen_text = text
        r["gen_response_time"] = round(gen_time, 1)
        r["gen_score"] = score_general(gen_text)
        r["prompt_tokens"] = usage.get("prompt_tokens", 0)
        r["completion_tokens"] = usage.get("completion_tokens", 0)
        r["gen_output_tokens"] = (r["completion_tokens"]
                                  if r["completion_tokens"] > 0
                                  else int(count_tokens(gen_text)))
        if r["completion_tokens"] > 0 and gen_time > 0:
            r["output_tps_gen"] = round(r["completion_tokens"] / gen_time, 2)

        gen_truncated = (gen_fr == "length")
        if not gen_truncated:
            state.update(model_name, last_error="")
            break

        if is_repeating(gen_text):
            gen_repeating = True
            state.update(model_name, last_error="Repeating, stopping")
            break

        if len(gen_text.strip()) < 50:
            state.update(model_name,
                         last_error=f"Thinking exhausted at {max_tok}")
            state.log(model_name, f"Gen task exhausted thinking (tok={max_tok}, out={len(gen_text.strip())}c)")

        if attempt < len(TOKEN_LEVELS) - 1:
            state.update(model_name,
                         last_error=f"Truncated at {max_tok}, retrying")
            state.log(model_name, f"Gen task truncated at {max_tok}, retrying {TOKEN_LEVELS[attempt + 1]}")

    if gen_truncated:
        if gen_repeating:
            r["gen_warning"] = "Truncated at max tokens (repeating)"
        else:
            r["gen_warning"] = f"Truncated at max tokens (exhausted {TOKEN_LEVELS[-1]})"

    # Source lock released
    r["total_time"] = round(time.time() - start, 1)
    state.add_result(r)
    state.update(model_name, status="completed",
                 gen_score=r["gen_score"], gen_tps=r["output_tps_gen"],
                 gen_response_time=r["gen_response_time"],
                 elapsed=r["total_time"],
                 gen_output_tokens=r["gen_output_tokens"])


def _source_abbrev(name):
    """Generate a short acronym from a source name using capital letters.

    Splits on whitespace and internal capitals (PascalCase), then takes the
    first letter of each resulting token.  All-caps short words (e.g. AI, PC)
    are kept whole.  Returns at least 2 characters.
    """
    tokens = []
    for w in name.split():
        if w.isupper() and 1 < len(w) <= 3:
            tokens.append(w)               # keep all-caps words like "AI", "PC" whole
        else:
            sub = re.findall(r'[A-Z]?[a-z]+|[A-Z]+', w)
            tokens.extend(sub) if sub else tokens.append(w)
    if not tokens:
        return name[:2].upper()
    ab = ''.join(t[0].upper() for t in tokens if t)
    return ab if len(ab) >= 2 else (name * 2)[:2].upper()


def tui_main(state, stop_event, num_sources=0):
    """Run ncurses TUI in a daemon thread. Updates every 200ms."""
    import curses

    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
    except Exception:
        # Fallback: inline progress bar with live details
        while not stop_event.is_set():
            snap = state.snapshot()
            active = sum(1 for s in snap.values() if s["status"]
                         in ("running_code", "running_gen", "queued"))
            done = state.completed
            total = state.total
            parts = [f"🔄 {active} active  |  ✅ {done}/{total} completed"]
            for name, s in snap.items():
                if s["status"] in ("running_code", "running_gen"):
                    elapsed = (time.time() - s["attempt_start"]) if s["attempt_start"] else 0
                    err = s.get("last_error", "")
                    parts.append(f"  {name[:30]}: {s['phase_detail']} "
                                 f"att {s['attempt']}/3 tok {s['max_tok']} "
                                 f"{elapsed:.0f}s{' '+err if err else ''}")
            sys.stdout.write(f"\r{' ' * 80}\r")
            sys.stdout.write(" | ".join(parts))
            sys.stdout.flush()
            time.sleep(1)
        print()
        return

    try:
        LIVE_HEIGHT = max(3, num_sources + 1)  # header line + one line per possible parallel worker
        scroll_offset = 0

        # Build unique source abbreviations (from snapshot to handle dynamic sources)
        src_snap = {s["source"] for s in state.snapshot().values()}
        source_abbrevs = {}
        _used = set()
        for src in sorted(src_snap):
            # First try the heuristic, then disambiguate if needed
            ab = _source_abbrev(src)
            if ab in _used:
                tokens = []
                for w in src.split():
                    if w.isupper() and 1 < len(w) <= 3:
                        tokens.append(w)
                    else:
                        sub = re.findall(r'[A-Z]?[a-z]+|[A-Z]+', w)
                        tokens.extend(sub) if sub else tokens.append(w)
                ab = ''.join(t[:2].upper() for t in tokens if t)
                if ab in _used or len(ab) < 2:
                    ab = (src * 2)[:2].upper()
                    if ab in _used:
                        for i in range(2, min(len(src), 6)):
                            ab = src[:i].upper()
                            if ab not in _used:
                                break
            source_abbrevs[src] = ab
            _used.add(ab)

        while not stop_event.is_set():
            max_y, max_x = stdscr.getmaxyx()
            snap = state.snapshot()
            done = state.completed
            total = state.total
            running_code = [n for n, s in snap.items() if s["status"] == "running_code"]
            running_gen = [n for n, s in snap.items() if s["status"] == "running_gen"]
            queued = [n for n, s in snap.items() if s["status"] == "queued"]
            pending = [n for n, s in snap.items() if s["status"] == "pending"]

            # ── Line writer (clrtoeol avoids leftover chars without flash) ──
            def _wr(y, x, text, attr=0):
                """Clear line then write — no full-screen escape."""
                stdscr.move(y, x)
                stdscr.clrtoeol()
                try:
                    stdscr.addstr(y, x, text[:max_x], attr)
                except:
                    try:
                        stdscr.addstr(y, x, text[:max_x])
                    except:
                        pass

            # ── Layout geometry (fixed: never shifts) ──
            FOOTER_LINE = max_y - 1
            MAX_LOG_ROWS = 3
            LOG_TOP = FOOTER_LINE - MAX_LOG_ROWS
            LIVE_TOP = LOG_TOP - LIVE_HEIGHT
            MODEL_BOTTOM = LIVE_TOP - 1
            MODEL_TOP = 4
            VISIBLE_ROWS = max(0, MODEL_BOTTOM - MODEL_TOP)

            # ── Header ──
            ts = datetime.now().strftime('%H:%M:%S')
            hdr = f"AI Benchmark — Parallel  |  {ts}"
            if max_x > len(hdr):
                _wr(0, 0, hdr, curses.A_BOLD)
            total_models = len(snap)
            failed_count = sum(1 for s in snap.values() if s["status"] == "failed")
            err_indicator = f"  |  \u26a0 {failed_count} failed" if failed_count else ""
            summary = (f"Total: {total}  |  "
                       f"Done: {done}  |  "
                       f"Active: {len(running_code + running_gen)}  |  "
                       f"Queued: {len(queued + pending)}"
                       f"{err_indicator}"
                       f"  |  \u2191\u2195 scroll {scroll_offset + 1}-{min(total_models, scroll_offset + VISIBLE_ROWS)}/{total_models}")
            if max_y > 1 and max_x > len(summary):
                _wr(1, 0, summary)

            # ── Separator ──
            if max_y > 2:
                _wr(2, 0, "\u2500" * min(max_x, 80))

            # ── Column headers ──
            col_hdr = (f"{'#':>3}  {'S':<3} {'Model':<18}  "
                       f"{'St':<3}"
                       f" {'CSc':>4} {'CTok':>5} {'CTm':>5} {'CTPS':>5}"
                       f" {'GSc':>4} {'GTok':>5} {'GTm':>5} {'GTPS':>5}")
            if max_y > 3:
                _wr(3, 0, col_hdr, curses.A_UNDERLINE)

            # ── Handle scroll keys ──
            key = stdscr.getch()
            max_offset = max(0, total_models - VISIBLE_ROWS)
            if key == curses.KEY_UP:
                scroll_offset = max(0, scroll_offset - 1)
            elif key == curses.KEY_DOWN:
                scroll_offset = min(max_offset, scroll_offset + 1)
            elif key == curses.KEY_PPAGE:
                scroll_offset = max(0, scroll_offset - VISIBLE_ROWS)
            elif key == ord(' ') or key == curses.KEY_NPAGE:
                scroll_offset = min(max_offset, scroll_offset + VISIBLE_ROWS)
            elif key == curses.KEY_HOME:
                scroll_offset = 0
            elif key == curses.KEY_END:
                scroll_offset = max_offset
            scroll_offset = max(0, min(max_offset, scroll_offset))

            # ── Model rows (scrollable) ──
            snap_items = list(snap.items())
            def fmt_val(v, fmt=".1f"):
                if v is None: return "-"
                try: return f"{v:{fmt}}"
                except: return str(v)

            for row_idx in range(VISIBLE_ROWS):
                abs_idx = scroll_offset + row_idx
                if abs_idx >= total_models:
                    break
                name, s = snap_items[abs_idx]
                display_idx = abs_idx + 1
                sv = s["status"]
                status_ch = {"pending": "\u23f3", "queued": "\u23f3",
                             "running_code": "\U0001f537", "running_gen": "\U0001f7e2",
                             "completed": "\u2705", "failed": "\u274c"}.get(sv, "?")
                csc = fmt_val(s.get("code_score"))
                gsc = fmt_val(s.get("gen_score"))
                ctok = fmt_val(s.get("code_output_tokens"), "d")
                gtok = fmt_val(s.get("gen_output_tokens"), "d")
                ctime = fmt_val(s.get("code_response_time"))
                gtime = fmt_val(s.get("gen_response_time"))
                ctps = fmt_val(s.get("code_tps"))
                gtps = fmt_val(s.get("gen_tps"))
                model_disp = name[:16]
                src_ab = source_abbrevs.get(s["source"], s["source"][:3])
                line = (f"{display_idx:>3}  {src_ab:<3} {model_disp:<18}  "
                        f"{status_ch:<3}"
                        f" {csc:>4} {ctok:>5} {ctime:>5} {ctps:>5}"
                        f" {gsc:>4} {gtok:>5} {gtime:>5} {gtps:>5}")
                if sv == "failed":
                    err_text = (s.get("error") or s.get("last_error", "") or "")
                    if err_text:
                        line += "  " + err_text
                attr = 0
                if sv == "completed":
                    try: attr = curses.color_pair(1)
                    except: pass
                elif sv == "failed":
                    try: attr = curses.color_pair(3)
                    except: pass
                elif sv in ("running_code", "running_gen"):
                    try: attr = curses.color_pair(2)
                    except: pass
                _wr(MODEL_TOP + row_idx, 0, line, attr)

            # Clear any leftover model rows below the last written one
            # (handles terminal resize or scroll-offset clamp)
            model_end = MODEL_TOP + min(VISIBLE_ROWS, max(0, total_models - scroll_offset))
            for r in range(model_end, MODEL_BOTTOM + 1):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except:
                    pass

            # ── Separator before live section ──
            if MODEL_BOTTOM >= 0:
                _wr(MODEL_BOTTOM, 0, "\u2500" * min(max_x, 60))

            # ── Live status section (fixed height, never shifts) ──
            live_models = running_code + running_gen
            live_row = LIVE_TOP
            _wr(live_row, 0, "Live:", curses.A_BOLD)
            live_row += 1
            for nm in live_models[:LIVE_HEIGHT - 1]:
                if live_row >= LOG_TOP:
                    break
                s = snap[nm]
                src_ab = source_abbrevs.get(s["source"], s["source"][:3])
                elapsed = (time.time() - s["attempt_start"]) if s["attempt_start"] else 0
                err = s.get("last_error", "")
                phase_ch = "\U0001f537" if s["status"] == "running_code" else "\U0001f7e2"
                msg = (f" {phase_ch} [{src_ab}] {nm[:42]}: "
                       f"Att {s['attempt']}/3  Tok {s['max_tok']}  "
                       f"{elapsed:5.0f}s"
                       f"{'  '+err if err else ''}")
                _wr(live_row, 0, msg)
                live_row += 1
            # Clear unused live-section rows
            for r in range(live_row, LOG_TOP):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except:
                    pass

            # ── Error log section (fixed 3 lines, no shifting) ──
            log_row = LOG_TOP
            recent_errors = state.recent_log(2)
            if recent_errors:
                _wr(log_row, 0, "Errors:", curses.A_BOLD)
                log_row += 1
                for ts_entry, model_entry, msg_entry in recent_errors:
                    if log_row >= FOOTER_LINE:
                        break
                    t_str = datetime.fromtimestamp(ts_entry).strftime('%H:%M:%S')
                    err_msg = f"  {t_str} [{model_entry[:20]}]: {msg_entry}"
                    _wr(log_row, 0, err_msg, curses.color_pair(3))
                    log_row += 1
            # Clear unused error-section rows
            for r in range(log_row, FOOTER_LINE):
                try:
                    stdscr.move(r, 0)
                    stdscr.clrtoeol()
                except:
                    pass

            # ── Bottom status line ──
            queuing = queued + pending
            if not live_models and not queuing:
                msg = " All models complete — generating outputs..."
            else:
                q = f"{len(queuing)} queued" if queuing else ""
                a = f"{len(live_models)} active" if live_models else ""
                sep2 = "  |  " if q and a else ""
                msg = f" {a}{sep2}{q}"
            _wr(FOOTER_LINE, 0, msg)

            stdscr.refresh()
            time.sleep(0.2)

    finally:
        curses.echo()
        curses.nocbreak()
        try:
            curses.endwin()
        except:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Reset terminal to sane state no matter what curses did to it before
    import subprocess
    try:
        subprocess.run(['stty', 'sane'], stderr=subprocess.DEVNULL,
                       stdin=sys.stdin, timeout=1)
    except Exception:
        pass
    sys.stderr.write('\033[2J\033[H')
    sys.stderr.flush()

    global SOURCE_CONFIG, MODELS, OUTPUT_DIR, STATE_FILE, TIMEOUT, TOKEN_LEVELS

    parser = argparse.ArgumentParser(
        description="AI Model Benchmark — Run code and general reasoning benchmarks across multiple API sources.",
        epilog="Examples:\n"
               "  python ai-benchmark.py --restart\n"
               "  python ai-benchmark.py --config my-config.json\n"
               "  python ai-benchmark.py --out /tmp/bench-run --timeout 300\n"
               "  python ai-benchmark.py --dump-default-config --base-url http://localhost:11434 > config.json\n"
               "  python ai-benchmark.py --dump-default-config > benchmark-config.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--restart', action='store_true',
                        help='Restart the run from scratch, discarding prior results')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                        help=f'Config file path (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--out', default=None,
                        help='Override output directory from config')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Override request timeout in seconds from config')
    parser.add_argument('--token-levels', type=int, nargs='+', default=None,
                        help='Override token levels (e.g. --token-levels 4096 8192 16384)')
    parser.add_argument('--dump-default-config', action='store_true',
                        help='Print a default config file to stdout and exit')
    parser.add_argument('--base-url', default=None,
                        help='Base URL for model discovery via /v1/models API (used with --dump-default-config)')
    parser.add_argument('--api-key', default=None,
                        help='API key for model discovery (used with --dump-default-config --base-url)')
    args = parser.parse_args()

    if args.dump_default_config:
        if args.base_url:
            cfg = generate_config_from_api(args.base_url, args.api_key)
            print(json.dumps(cfg, indent=2))
        else:
            dump_default_config()
        sys.exit(0)

    # Load config file
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}\n"
              f"   Copy benchmark-config.json or create one with --dump-default-config.",
              file=sys.stderr)
        sys.exit(1)
    cfg = load_config(config_path)
    SOURCE_CONFIG = cfg.get("sources", {})
    MODELS = cfg.get("models", {})
    OUTPUT_DIR = cfg.get("output_dir", "benchmark-results")
    if args.out:
        OUTPUT_DIR = args.out
    STATE_FILE = os.path.join(OUTPUT_DIR, "benchmark_state.json")

    TIMEOUT = cfg.get("timeout", 600)
    if args.timeout is not None:
        TIMEOUT = args.timeout

    TOKEN_LEVELS = cfg.get("token_levels", [16384])
    if args.token_levels is not None:
        TOKEN_LEVELS = args.token_levels

    print(f"📋 Loaded {len(MODELS)} models across {len(SOURCE_CONFIG)} sources "
          f"from {config_path}", file=sys.stderr)
    print(f"📂 Output directory: {OUTPUT_DIR}", file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Restart: wipe everything ──
    if args.restart:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        for f in glob.glob(os.path.join(OUTPUT_DIR, "results.*")):
            try:
                os.remove(f)
            except OSError:
                pass
        logs_dir = os.path.join(OUTPUT_DIR, "logs")
        if os.path.isdir(logs_dir):
            for f in glob.glob(os.path.join(logs_dir, "*.log")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # ── State: resume or fresh ──
    resumed = False
    if not args.restart and os.path.exists(STATE_FILE):
        try:
            state = BenchmarkState.load_state(STATE_FILE, MODELS)
            resumed = True
            completed = state.completed
            total = state.total
            print(f"📂 Resuming — {completed}/{total} models already completed.\n"
                  f"   Remove {STATE_FILE} or use --restart to start fresh.",
                  file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Could not load state file ({e}), starting fresh.",
                  file=sys.stderr)
            state = BenchmarkState(MODELS)
    else:
        state = BenchmarkState(MODELS)

    stop_event = threading.Event()

    # Start ncurses TUI in background
    tui_thread = threading.Thread(target=tui_main, args=(state, stop_event, len(SOURCE_CONFIG)))
    tui_thread.start()

    # Small delay for TUI to initialize
    time.sleep(0.3)

    total = state.total
    worker_errors = 0
    interrupted = False

    session_seed = random.randint(0, 2**31 - 1)

    # Build per-source queues — skip already-completed models on resume
    source_queues = {src: [] for src in set(MODELS.values())}
    for name, src in MODELS.items():
        info = state.snapshot().get(name, {})
        if info.get("status") in ("completed",):
            continue
        source_queues[src].append(name)

    source_threads = {}
    errors_lock = threading.Lock()

    def worker(source, models):
        nonlocal worker_errors
        for model_name in models:
            if stop_event.is_set():
                break
            try:
                run_model(model_name, source, state, session_seed=session_seed)
                state.save_state(STATE_FILE)
                # Keep output files current as models finish
                _save_outputs(state, OUTPUT_DIR)
            except Exception as e:
                with errors_lock:
                    worker_errors += 1
                print(f"\n❌ Worker exception ({model_name}): {type(e).__name__}: {e}",
                      file=sys.stderr)

    for source, queue in source_queues.items():
        if not queue:
            continue
        t = threading.Thread(target=worker, args=(source, queue), daemon=True)
        t.start()
        source_threads[source] = t

    if not source_threads:
        print("✅ All models already completed. Nothing to run.", file=sys.stderr)
    else:
        try:
            for t in source_threads.values():
                t.join()
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            print("\n\n⚠️  Ctrl+C — saving state and shutting down...", file=sys.stderr)
            print("   (Press Ctrl+C again to force exit)", file=sys.stderr)
            for t in source_threads.values():
                t.join(timeout=3)

    # Stop TUI (also set by KeyboardInterrupt path above)
    stop_event.set()
    tui_thread.join(timeout=2)

    # Save state one final time
    try:
        state.save_state(STATE_FILE)
    except Exception:
        pass

    if interrupted:
        done = state.completed
        print(f"✅ Saved state ({done}/{total} done). Re-run without --restart to continue.\n",
              file=sys.stderr)
        return

    # Final output refresh (worker threads already keep them current)
    _save_outputs(state, OUTPUT_DIR)
    final_results = state.latest_results()
    pdf_path = gen_pdf(final_results)
    ok_count = len([r for r in final_results if r["status"] == "ok"])
    print(f"\n{'='*70}")
    print(f"AI BENCHMARK COMPLETE — {ok_count}/{total} successful "
          f"({worker_errors} worker errors)")
    print(f"Outputs: {OUTPUT_DIR}/")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {fname}")
    if pdf_path:
        print(f"  - {os.path.basename(pdf_path)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
