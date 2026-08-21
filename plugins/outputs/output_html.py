import html as html_lib
import os
from datetime import datetime, timezone
from pathlib import Path

import jinja2

from benchmark.outputs import (
    _judge_consensus_by_contract,
    _judge_criteria,
    _numeric_score,
    _plugin_token_counts,
    _plugin_total_score,
    _scored_plugin_count,
    sanitize_filename,
)
from benchmark.plugin import BenchmarkOutputPlugin


def _atomic_write(path, content):
    from benchmark.outputs import _atomic_replace_report
    _atomic_replace_report(path, content)

# The document skeleton (head, CSS, page structure) lives in a Jinja2
# template; the data-driven cells are pre-built in Python and passed in
# already escaped, so autoescape is off to avoid double-escaping.
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


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
        judge_enabled = any(
            r.get("judge_models")
            or r.get("judge_model")
            or any(key.endswith(("_judge_score", "_judge_error")) for key in r)
            for r in results
        )
        rows = []
        for r in results:
            cls = "ok" if r["status"] == "ok" else "fail"
            tot = _plugin_total_score(r, active_plugins)
            m = "str" if r.get('stream_ok') else "ns"
            runner = r.get("runner", "http")
            cells = (f'<td>{r["model"]}</td>'
                     f'<td>{html_lib.escape(str(runner))}</td>'
                     f'<td>{r.get("ttft") or "-"}</td>')
            for p in active_plugins:
                score_val = r.get(f"{p.id}_score", "-")
                if output_dir:
                    runner_prefix = f"{runner}/" if runner in ("http", "opencode") else ""
                    rel_path = f"{runner_prefix}responses/{sanitize_filename(r['model'])}/{p.id}.txt"
                    score_cell = f'<a href="{rel_path}">{score_val}</a>'
                else:
                    score_cell = f"{score_val}"
                empty_reason = r.get(f"{p.id}_empty_reason", "")
                empty_cell = f'<td class="empty-reason" title="{html_lib.escape(str(empty_reason))}">{html_lib.escape(str(empty_reason))}</td>' if empty_reason else "<td></td>"
                thinking, content, total = _plugin_token_counts(r, p.id)
                judge_score = r.get(f"{p.id}_judge_score", "-")
                judge_confidence = r.get(f"{p.id}_judge_confidence", "-")
                judge_error = r.get(f"{p.id}_judge_error", "")
                cells += (f'<td>{r.get(f"{p.id}_response_time", "-")}</td>'
                          f'<td>{r.get(f"{p.id}_tps", "-")}</td>'
                          f'<td>{thinking}</td>'
                          f'<td>{content}</td>'
                          f'<td><strong>{total}</strong></td>'
                          f'<td><strong>{score_cell}</strong></td>')
                if judge_enabled:
                    cells += (f'<td class="judge-score">{html_lib.escape(str(judge_score))}</td>'
                              f'<td class="judge-confidence">{html_lib.escape(str(judge_confidence))}</td>'
                              f'<td class="judge-error">{html_lib.escape(str(judge_error))}</td>'
                              f'<td class="judge-votes">{len(r.get(f"{p.id}_judge_votes", []))}</td>')
                cells += empty_cell
                cells += (
                    f'<td>{r.get(f"{p.id}_attempt_count", "-")}</td>'
                    f'<td>{html_lib.escape(", ".join(r.get(f"{p.id}_retry_reasons", []) or []) or "-")}</td>'
                    f'<td>{html_lib.escape(str(r.get(f"{p.id}_prompt_altered", "-")))}</td>'
                    f'<td>{html_lib.escape(str(r.get(f"{p.id}_response_nature", "-")))}</td>'
                    f'<td>{html_lib.escape(str(r.get(f"{p.id}_failure_cause", "-")))}</td>'
                )
            overall = r.get("overall_score_100", tot)
            scored_plugins = r.get("overall_scored_plugins", _scored_plugin_count(r, active_plugins))
            cells += (f'<td><strong>{overall if overall is not None else "-"}</strong></td>'
                      f'<td>{scored_plugins}</td><td>{r["total_time"]}s</td><td>{m}</td>')
            if r["status"] == "ok":
                cells += '<td class="ok-badge">✅</td>'
            else:
                err = html_lib.escape(str(r.get('error') or '?'))[:50]
                cells += f'<td class="fail-badge" title="{err}">❌ {err}</td>'
            rows.append((cls, cells))

        ttft_rows = []
        for i, r in enumerate(
            sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1
        ):
            ttft_rows.append((i, r["model"], r["ttft"]))

        leaderboards = []
        for p in active_plugins:
            lb_rows = []
            for i, r in enumerate(
                sorted(ok, key=lambda x: _numeric_score(x, p.id), reverse=True)[:10], 1
            ):
                lb_rows.append((i, r["model"], r.get(f"{p.id}_score", "-")))
            leaderboards.append((p.name, lb_rows))

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
                    rubric_html += '<table><tr><th>Criterion</th><th>Points</th><th>Total</th></tr>\n'
                    for item in rubric:
                        rubric_html += f"<tr><td>{html_lib.escape(str(item['name']))}</td><td>{item.get('points', '-')}</td><td>{item.get('total', '-')}</td></tr>\n"
                    rubric_html += "</table>\n"

        judge_criteria_html = ""
        if any(_judge_criteria(r, p.id) for r in results for p in active_plugins):
            rows_html = []
            for r in results:
                for p in active_plugins:
                    for judge_report in _judge_criteria(r, p.id):
                        judge_name = html_lib.escape(str(judge_report.get("judge", "-")))
                        for item in judge_report.get("criteria", []):
                            rows_html.append(
                                "<tr>"
                                f"<td>{html_lib.escape(str(r.get('model', '-')))}</td>"
                                f"<td>{html_lib.escape(str(p.name))}</td>"
                                f"<td>{judge_name}</td>"
                                f"<td>{html_lib.escape(str(judge_report.get('judge_contract_id', '-')))}</td>"
                                f"<td>{html_lib.escape(str(item.get('id', '-')))}</td>"
                                f"<td>{html_lib.escape(str(item.get('criterion', '-')))}</td>"
                                f"<td>{html_lib.escape(str(item.get('status', '-')))}</td>"
                                f"<td>{html_lib.escape(str(item.get('evidence', '-')))}</td>"
                                "</tr>"
                            )
            judge_criteria_html = (
                "<h2>🧭 Judge Criteria and Evidence</h2>"
                "<p class=\"subtitle\">The judge's requirement interpretation and evidence; "
                "separate from the deterministic rubric and non-scoring.</p>"
                "<table><tr><th>Model</th><th>Plugin</th><th>Judge</th><th>Contract</th><th>ID</th>"
                "<th>Criterion</th><th>Status</th><th>Evidence</th></tr>"
                + "".join(rows_html) + "</table>"
            )
            consensus_rows = []
            for r in results:
                for p in active_plugins:
                    for contract_id, consensus in _judge_consensus_by_contract(r, p.id).items():
                        consensus_rows.append(
                            "<tr>"
                            f"<td>{html_lib.escape(str(r.get('model', '-')))}</td>"
                            f"<td>{html_lib.escape(str(p.name))}</td>"
                            f"<td>{html_lib.escape(str(contract_id))}</td>"
                            f"<td>{html_lib.escape(str(consensus.get('score', '-')))}</td>"
                            f"<td>{html_lib.escape(str(consensus.get('confidence', '-')))}</td>"
                            f"<td>{html_lib.escape(str(consensus.get('valid_judges', '-')))}</td>"
                            f"<td>{html_lib.escape(str(consensus.get('attempts', '-')))}</td>"
                            "</tr>"
                        )
            if consensus_rows:
                judge_criteria_html += (
                    "<h3>Versioned Judge Consensus</h3>"
                    "<table><tr><th>Model</th><th>Plugin</th><th>Contract</th>"
                    "<th>Score</th><th>Confidence</th><th>Valid Judges</th><th>Attempts</th></tr>"
                    + "".join(consensus_rows) + "</table>"
                )

        header_cells = "<th>Model</th><th>Runner</th><th>Load(s)</th>"
        for p in active_plugins:
            header_cells += (f"<th>{p.name} Resp(s)</th><th>{p.name} TPS</th>"
                             f"<th>{p.name} Think Tok</th><th>{p.name} Cont Tok</th><th>{p.name} Total Tok</th>"
                             f"<th>{p.name} Score (0–100)</th>")
            if judge_enabled:
                header_cells += (f"<th>{p.name} Judge (0–100)</th>"
                                 f"<th>{p.name} Judge Confidence</th><th>{p.name} Judge Error</th>"
                                 f"<th>{p.name} Judge Votes</th>")
            header_cells += (f"<th>{p.name} Reason</th>"
                             f"<th>{p.name} Attempts</th><th>{p.name} Retry</th>"
                             f"<th>{p.name} Prompt Altered</th><th>{p.name} Nature</th>"
                             f"<th>{p.name} Failure</th>")
        header_cells += "<th>Overall Score (0–100)</th><th>Scored Plugins</th><th>Time</th><th>Mode</th><th>Status</th>"

        seed_html = f"<br><strong>Seed:</strong> {session_seed}" if session_seed is not None else ""
        judges_line = html_lib.escape(
            ', '.join(next((r.get('judge_models') for r in results if r.get('judge_models')), [])
                      or ([next((r.get('judge_model') for r in results if r.get('judge_model')), '')]
                          if any(r.get('judge_model') for r in results) else [])) or '—'
        )
        judge_status = html_lib.escape(str(next((r.get('judge_status') for r in results if r.get('judge_status')), '—')))

        content = _ENV.get_template("results.html.j2").render(
            task_names=", ".join(p.name for p in active_plugins),
            now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            total=len(results),
            ok_count=len(ok),
            failed_count=len(results) - len(ok),
            seed_html=seed_html,
            ttft_rows=ttft_rows,
            leaderboards=leaderboards,
            judges_line=judges_line,
            judge_status=judge_status,
            header_cells=header_cells,
            rows=rows,
            rubric_html=rubric_html,
            judge_criteria_html=judge_criteria_html,
        )

        if output_dir:
            path = os.path.join(output_dir, "results.html")
            try:
                _atomic_write(path, content)
                return path
            except OSError:
                pass
        return content
