import csv
import io
import os

from benchmark_plugin import BenchmarkOutputPlugin
from benchmark_outputs import _plugin_total_score


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
