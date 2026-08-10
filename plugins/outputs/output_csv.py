import csv
import io
import json
import os

from benchmark.outputs import (
    _judge_enabled,
    _plugin_token_counts,
    _plugin_total_score,
    _scored_plugin_count,
)
from benchmark.plugin import BenchmarkOutputPlugin


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
        judge_enabled = _judge_enabled(results)
        headers = ["Model", "Runner", "Source", "TTFT_s"]
        for p in active_plugins:
            headers.extend([
                f"{p.id}_Response_s", f"{p.id}_Thinking_Tokens",
                f"{p.id}_Content_Tokens", f"{p.id}_Total_Tokens",
                f"{p.id}_TPS",
                f"{p.id}_Score_100",
                *([
                    f"{p.id}_Judge_Score_100",
                    f"{p.id}_Judge_Confidence",
                    f"{p.id}_Judge_Error",
                    f"{p.id}_Judge_Votes",
                ] if judge_enabled else []),
                f"{p.id}_Empty_Reason",
            ])
        headers.extend(["Overall_Score_100", "Overall_Scored_Plugins", "Time_s", "Mode", "Status", "Error"])
        if judge_enabled:
            headers.insert(3, "Judge_Models")
            headers.insert(4, "Judge_Status")
        w.writerow(headers)

        for r in results:
            tot = _plugin_total_score(r, active_plugins)
            m = "stream" if r.get('stream_ok') else "non-streaming"
            row = [r['model'], r.get('runner', 'http'), r.get('source', ''), r.get('ttft') or '']
            if judge_enabled:
                row[3:3] = [", ".join(r.get("judge_models", []) or ([r.get("judge_model")] if r.get("judge_model") else [])), r.get("judge_status", "")]
            for p in active_plugins:
                thinking, content, total = _plugin_token_counts(r, p.id)
                row.extend([
                    r.get(f"{p.id}_response_time", ''),
                    thinking,
                    content,
                    total,
                    r.get(f"{p.id}_tps", ''),
                    r.get(f"{p.id}_score", ''),
                    *([
                        r.get(f"{p.id}_judge_score", ''),
                        r.get(f"{p.id}_judge_confidence", ''),
                        r.get(f"{p.id}_judge_error", ''),
                        json.dumps(r.get(f"{p.id}_judge_votes", []), ensure_ascii=False),
                    ] if judge_enabled else []),
                    r.get(f"{p.id}_empty_reason", ''),
                ])
            overall = r.get("overall_score_100", tot)
            scored_plugins = r.get("overall_scored_plugins", _scored_plugin_count(r, active_plugins))
            if r["status"] == "ok":
                row.extend([overall if overall is not None else "", scored_plugins, r['total_time'], m, "OK", ""])
            else:
                row.extend([overall if overall is not None else "", scored_plugins, r['total_time'], m, "FAIL", r.get('error', '')])
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
