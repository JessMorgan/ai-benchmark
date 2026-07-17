"""Output generators for the AI benchmark.

This module contains the report generators (Markdown, CSV, HTML, PDF) and the
helper used to persist them to disk.
"""
import csv
import html
import io
import os
import re
from datetime import datetime


def sanitize_filename(name):
    """Sanitize a model name for use as a filename."""
    s = re.sub(r'[^\w\-\.\(\) ]', '_', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


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


def gen_markdown(results, active_plugins, output_dir=None, session_seed=None):
    ok = [r for r in results if r["status"] == "ok"]
    plugin_names = " | ".join(f"**{p.name}** ({int(p.max_score)} pts)" for p in active_plugins)
    seed_line = f"**Seed:** {session_seed}" if session_seed is not None else ""
    lines = [
        "# AI Benchmark — Plugin-Based",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Tasks:** {plugin_names}",
        f"**Total:** {len(results)} models | **✅ {len(ok)} successful** | **❌ {len(results)-len(ok)} failed**",
    ]
    if seed_line:
        lines.append(seed_line)
    lines.extend([
        "",
        "## 📋 Complete Results",
        "",
    ])

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

    # Per-response rubric breakdown
    has_rubric = any(isinstance(r.get(f"{p.id}_rubric"), list) and r.get(f"{p.id}_rubric") for p in active_plugins for r in results)
    if has_rubric:
        lines.extend(["", "---", "## 🔍 Detailed Rubric Breakdown", ""])
        for r in results:
            if r["status"] != "ok":
                continue
            for p in active_plugins:
                rubric = r.get(f"{p.id}_rubric")
                if not isinstance(rubric, list) or not rubric:
                    continue
                lines.append(f"### {p.name} — {r['model']}")
                lines.append("| Criterion | Earned | Max | Missed |")
                lines.append("|---|---|---|---|")
                for item in rubric:
                    lines.append(f"| {item['name']} | {item['earned']} | {item['max']} | {item['missed']} |")
                lines.append("")

    lines.extend(["", "## ❌ Failed Models", ""])
    for r in results:
        if r["status"] != "ok":
            lines.append(f"- **{r['model']}**: {r.get('error','?')}")
    lines.append("")
    return "\n".join(lines)


def gen_csv(results, active_plugins):
    out = io.StringIO()
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


def gen_html(results, active_plugins, output_dir=None, session_seed=None):
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

    # Per-response rubric breakdown
    rubric_html = ""
    has_rubric = any(isinstance(r.get(f"{p.id}_rubric"), list) and r.get(f"{p.id}_rubric") for p in active_plugins for r in results)
    if has_rubric:
        rubric_html = "<h2>🔍 Detailed Rubric Breakdown</h2>\n"
        for r in results:
            if r["status"] != "ok":
                continue
            for p in active_plugins:
                rubric = r.get(f"{p.id}_rubric")
                if not isinstance(rubric, list) or not rubric:
                    continue
                rubric_html += f"<h3>{html.escape(p.name)} — {html.escape(r['model'])}</h3>\n"
                rubric_html += '<table><tr><th>Criterion</th><th>Earned</th><th>Max</th><th>Missed</th></tr>\n'
                for item in rubric:
                    rubric_html += f"<tr><td>{html.escape(str(item['name']))}</td><td>{item['earned']}</td><td>{item['max']}</td><td>{item['missed']}</td></tr>\n"
                rubric_html += "</table>\n"

    header_cells = "<th>Model</th><th>Load(s)</th>"
    for p in active_plugins:
        header_cells += f"<th>{p.name} Resp(s)</th><th>{p.name} TPS</th><th>{p.name} Tok</th><th>{p.name} Score</th>"
    header_cells += "<th>Total</th><th>Time</th><th>Mode</th><th>Status</th>"

    seed_html = f"<br><strong>Seed:</strong> {session_seed}" if session_seed is not None else ""
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
<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <strong>Total:</strong> {len(results)} | <strong>Successful:</strong> {len(ok)} | <strong>Failed:</strong> {len(results)-len(ok)}{seed_html}</p>

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
{rubric_html}
</body>
</html>"""


def gen_pdf(results, active_plugins, output_dir, session_seed=None):
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
    seed_part = f"  |  Seed: {session_seed}" if session_seed is not None else ""
    pdf.cell(0, 6, f"Total: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(results)-len(ok)}{seed_part}", align="C", new_x="LMARGIN", new_y="NEXT")
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

    # Per-response rubric breakdown
    has_rubric = any(isinstance(r.get(f"{p.id}_rubric"), list) and r.get(f"{p.id}_rubric") for p in active_plugins for r in results)
    if has_rubric:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Detailed Rubric Breakdown", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7)
        for r in results:
            if r["status"] != "ok":
                continue
            for p in active_plugins:
                rubric = r.get(f"{p.id}_rubric")
                if not isinstance(rubric, list) or not rubric:
                    continue
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 5, f"{p.name} -- {r['model']}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 7)
                for item in rubric:
                    pdf.cell(0, 4, f"  {item['name']}: {item['earned']}/{item['max']} (missed {item['missed']})", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.pdf")
    pdf.output(path)
    return path


def _save_outputs(state, output_dir, active_plugins):
    """Regenerate CSV/markdown/HTML from latest deduplicated results."""
    results = state.latest_results()
    session_seed = getattr(state, "session_seed", None)
    md = gen_markdown(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
    csv_txt = gen_csv(results, active_plugins)
    html = gen_html(results, active_plugins, output_dir=output_dir, session_seed=session_seed)
    for fname, content in [("results.md", md), ("results.csv", csv_txt), ("results.html", html)]:
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "w") as f:
                f.write(content)
        except OSError:
            pass
