import os
from datetime import datetime

from benchmark_plugin import BenchmarkOutputPlugin
from benchmark_outputs import _plugin_total_score, _numeric_score


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
