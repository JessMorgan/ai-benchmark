import html as html_lib
import os
from datetime import datetime

from benchmark_plugin import BenchmarkOutputPlugin
from benchmark_outputs import sanitize_filename, _plugin_total_score, _numeric_score


class HTMLOutputPlugin(BenchmarkOutputPlugin):
    @property
    def id(self):
        return "output-html"

    @property
    def name(self):
        return "HTML Report"

    @property
    def extension(self):
        return "html"

    def generate(self, results, active_plugins, output_dir=None, session_seed=None):
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
                err = html_lib.escape(str(r.get('error') or '?'))[:50]
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
                    rubric_html += f"<h3>{html_lib.escape(p.name)} — {html_lib.escape(r['model'])}</h3>\n"
                    rubric_html += '<table><tr><th>Criterion</th><th>Earned</th><th>Max</th><th>Missed</th></tr>\n'
                    for item in rubric:
                        rubric_html += f"<tr><td>{html_lib.escape(str(item['name']))}</td><td>{item['earned']}</td><td>{item['max']}</td><td>{item['missed']}</td></tr>\n"
                    rubric_html += "</table>\n"

        header_cells = "<th>Model</th><th>Load(s)</th>"
        for p in active_plugins:
            header_cells += f"<th>{p.name} Resp(s)</th><th>{p.name} TPS</th><th>{p.name} Tok</th><th>{p.name} Score</th>"
        header_cells += "<th>Total</th><th>Time</th><th>Mode</th><th>Status</th>"

        seed_html = f"<br><strong>Seed:</strong> {session_seed}" if session_seed is not None else ""
        content = f"""<!DOCTYPE html>
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

        if output_dir:
            path = os.path.join(output_dir, "results.html")
            try:
                with open(path, "w") as f:
                    f.write(content)
                return path
            except OSError:
                pass
        return content
