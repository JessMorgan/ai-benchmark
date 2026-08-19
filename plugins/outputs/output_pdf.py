import os
from datetime import datetime, timezone

from benchmark.outputs import (
    _judge_criteria,
    _numeric_score,
    _plugin_token_counts,
    _plugin_total_score,
    _scored_plugin_count,
)
from benchmark.plugin import BenchmarkOutputPlugin


class PDFOutputPlugin(BenchmarkOutputPlugin):
    @property
    def id(self):
        return "output-pdf"

    @property
    def name(self):
        return "PDF Report"

    @property
    def extension(self):
        return "pdf"

    def generate(self, results, active_plugins, output_dir=None, session_seed=None):
        if not output_dir:
            return None

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
        pdf.cell(0, 6, f"Tasks: {', '.join(p.name for p in active_plugins)}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        ok = [r for r in results if r["status"] == "ok"]
        seed_part = f"  |  Seed: {session_seed}" if session_seed is not None else ""
        pdf.cell(0, 6, f"Total: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(results)-len(ok)}{seed_part}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        has_runner = any(r.get("runner") for r in results)
        col_w = [32, 9]
        headers = ["Model", "Load"]
        if has_runner:
            col_w = [28, 12, 9]
            headers = ["Model", "Runner", "Load"]
        judge_enabled = any(
            r.get("judge_models")
            or r.get("judge_model") is not None
            or r.get("judge_status") not in (None, "disabled")
            or any(key.endswith(("_judge_score", "_judge_error")) for key in r)
            for r in results
        )
        for p in active_plugins:
            # Keep the token columns narrow (6 units) so the extra
            # thinking/content/total columns don't widen the already
            # page-bound PDF table more than necessary.
            col_w.extend([9, 8, 6, 6, 6, 8])
            headers.extend([
                f"{p.id[:3].upper()}Rsp", f"{p.id[:3].upper()}TPS",
                f"{p.id[:3].upper()}Thk", f"{p.id[:3].upper()}Ctn",
                f"{p.id[:3].upper()}Tot", f"{p.id[:3].upper()}Sc",
            ])
            if judge_enabled:
                col_w.extend([8, 8, 7])
                headers.extend([f"{p.id[:3].upper()}Jdg", f"{p.id[:3].upper()}Conf", f"{p.id[:3].upper()}Vts"])
        col_w.extend([9, 9, 9])
        headers.extend(["Overall", "Scored", "Mode"])
        pdf.set_font("Helvetica", "B", 6.5)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 5, h, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 6.5)
        for r in results:
            tot = _plugin_total_score(r, active_plugins)
            m = "str" if r.get('stream_ok') else "ns"
            vals = [r['model'][:24 if has_runner else 30]]
            if has_runner:
                vals.append(str(r.get("runner", "http")))
            vals.append(str(r.get('ttft') or '-'))
            for p in active_plugins:
                thinking, content, total = _plugin_token_counts(r, p.id)
                vals.extend([
                    str(r.get(f'{p.id}_response_time', '-')),
                    str(r.get(f'{p.id}_tps', '-')),
                    str(thinking),
                    str(content),
                    str(total),
                    str(r.get(f'{p.id}_score', '-')),
                ])
                if judge_enabled:
                    vals.extend([
                        str(r.get(f"{p.id}_judge_score", "-")),
                        str(r.get(f"{p.id}_judge_confidence", "-")),
                        str(len(r.get(f"{p.id}_judge_votes", []))),
                    ])
            overall = r.get("overall_score_100", tot)
            scored_plugins = r.get("overall_scored_plugins", _scored_plugin_count(r, active_plugins))
            if r["status"] == "ok":
                vals.extend([str(overall if overall is not None else "-"), str(scored_plugins), m])
            else:
                vals.extend([str(overall if overall is not None else "-"), str(scored_plugins), "FAIL"])
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
                    pdf.cell(0, 4, f"  {i}. {r['model'][:50]}  --  {r.get(f'{p.id}_score', '-')}/100", new_x="LMARGIN", new_y="NEXT")

        has_judge_criteria = any(
            _judge_criteria(r, p.id)
            for r in results for p in active_plugins
        )
        if has_judge_criteria:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Judge Criteria and Evidence", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(
                0, 4,
                "The judge's requirement interpretation and evidence; separate from the deterministic rubric.",
                new_x="LMARGIN", new_y="NEXT",
            )
            for r in results:
                for p in active_plugins:
                    for judge_report in _judge_criteria(r, p.id):
                        judge_name = judge_report.get("judge", "-")
                        for item in judge_report.get("criteria", []):
                            text = (
                                f"{r.get('model', '-')} / {p.name} / {judge_name} / "
                                f"{item.get('id', '-')} [{item.get('status', '-')}]\n"
                                f"Criterion: {item.get('criterion', '-')}\n"
                                f"Evidence: {item.get('evidence', '-')}"
                            )
                            pdf.multi_cell(0, 4, text, new_x="LMARGIN", new_y="NEXT")
                            pdf.ln(1)

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
                        pdf.cell(0, 4, f"  {item['name']}: {item.get('points', '-')} / {item.get('total', '-')}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.pdf")
        pdf.output(path)
        return path
