import csv
import io
import os

from benchmark.plugin import BenchmarkOutputPlugin
from benchmark.outputs import _plugin_total_score, _plugin_token_counts


class CSVOutputPlugin(BenchmarkOutputPlugin):
    @property
    def id(self):
        return "output-csv"

    @property
    def name(self):
        return "CSV Data"

    @property
    def extension(self):
        return "csv"

    def generate(self, results, active_plugins, output_dir=None, session_seed=None):
        out = io.StringIO()
        w = csv.writer(out)
        headers = ["Model", "Runner", "Source", "TTFT_s"]
        for p in active_plugins:
            headers.extend([
                f"{p.id}_Response_s", f"{p.id}_Thinking_Tokens",
                f"{p.id}_Content_Tokens", f"{p.id}_Total_Tokens",
                f"{p.id}_TPS",
                f"{p.id}_Score_{int(p.max_score)}", f"{p.id}_Empty_Reason",
            ])
        headers.extend(["Total", "Time_s", "Mode", "Status", "Error"])
        w.writerow(headers)

        for r in results:
            tot = _plugin_total_score(r, active_plugins)
            m = "stream" if r.get('stream_ok') else "non-streaming"
            row = [r['model'], r.get('runner', 'http'), r.get('source', ''), r.get('ttft') or '']
            for p in active_plugins:
                thinking, content, total = _plugin_token_counts(r, p.id)
                row.extend([
                    r.get(f"{p.id}_response_time", ''),
                    thinking,
                    content,
                    total,
                    r.get(f"{p.id}_tps", ''),
                    r.get(f"{p.id}_score", ''),
                    r.get(f"{p.id}_empty_reason", ''),
                ])
            if r["status"] == "ok":
                row.extend([tot, r['total_time'], m, "OK", ""])
            else:
                row.extend([tot, r['total_time'], m, "FAIL", r.get('error', '')])
            w.writerow(row)

        content = out.getvalue()
        if output_dir:
            path = os.path.join(output_dir, "results.csv")
            try:
                with open(path, "w") as f:
                    f.write(content)
                return path
            except OSError:
                pass
        return content
