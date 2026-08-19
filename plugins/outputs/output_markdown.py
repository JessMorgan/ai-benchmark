import os
from datetime import datetime, timezone

from benchmark.outputs import (
    _judge_criteria,
    _numeric_score,
    _plugin_token_counts,
    _plugin_total_score,
    _scored_plugin_count,
    sanitize_filename,
)
from benchmark.plugin import BenchmarkOutputPlugin


class MarkdownOutputPlugin(BenchmarkOutputPlugin):
    @property
    def id(self):
        return "output-markdown"

    @property
    def name(self):
        return "Markdown Report"

    @property
    def extension(self):
        return "md"

    def generate(self, results, active_plugins, output_dir=None, session_seed=None):
        ok = [r for r in results if r["status"] == "ok"]
        judge_enabled = any(
            r.get("judge_models")
            or r.get("judge_model") is not None
            or r.get("judge_status") not in (None, "disabled")
            or any(key.endswith(("_judge_score", "_judge_error")) for key in r)
            for r in results
        )
        plugin_names = " | ".join(f"**{p.name}**" for p in active_plugins)
        seed_line = f"**Seed:** {session_seed}" if session_seed is not None else ""
        lines = [
            "# AI Benchmark — Plugin-Based",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
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

        has_runner = any(r.get("runner") for r in results)
        header = "| # | Model | Runner |"
        if judge_enabled:                header += " Judge Models | Judge Status |"
        header += " Load (s) |" if has_runner else "| # | Model |"
        if not has_runner:
            if judge_enabled:
                header = "| # | Model | Judge Models | Judge Status | Load (s) |"
            else:
                header = "| # | Model | Load (s) |"
        for p in active_plugins:
            header += f" {p.name} Resp (s) | {p.name} TPS | {p.name} Think Tok | {p.name} Cont Tok | {p.name} Total Tok | {p.name} Score (0–100) |"
            if judge_enabled:
                header += f" {p.name} Judge (0–100) | {p.name} Judge Confidence | {p.name} Judge Error | {p.name} Judge Votes |"
            header += f" {p.name} Reason |"
            if output_dir:
                header += f" {p.name} Response |"
        header += " Overall Score (0–100) | Scored Plugins | Time | Mode |"
        lines.append(header)

        separator_columns = header.count("|") - 1
        lines.append("|" + "---|" * separator_columns)

        for idx, r in enumerate(results, 1):
            tot = _plugin_total_score(r, active_plugins)
            m = "stream" if r.get('stream_ok') else "nostream"
            runner = r.get("runner", "")
            if has_runner:
                row = f"| {idx} | {r['model']} | {runner} |"
            else:
                row = f"| {idx} | {r['model']} |"
            if judge_enabled:
                models = r.get("judge_models", []) or ([r.get("judge_model")] if r.get("judge_model") else [])
                row += f" {', '.join(models) or '-'} | {r.get('judge_status', '-')} |"
            row += f" {r.get('ttft') or '-'} |"
            for p in active_plugins:
                empty_reason = r.get(f'{p.id}_empty_reason', '')
                thinking, content, total = _plugin_token_counts(r, p.id)
                row += (f" {r.get(f'{p.id}_response_time','-')} | "
                        f"{r.get(f'{p.id}_tps','-')} | "
                        f"{thinking} | "
                        f"{content} | "
                        f"{total} | "
                        f"{r.get(f'{p.id}_score','-')} | ")
                if judge_enabled:
                    row += (f"{r.get(f'{p.id}_judge_score', '-')} | "
                            f"{r.get(f'{p.id}_judge_confidence', '-')} | "
                            f"{r.get(f'{p.id}_judge_error', '')} | "
                            f"{len(r.get(f'{p.id}_judge_votes', []))} votes | ")
                row += f"{empty_reason} |"
                if output_dir:
                    runner_prefix = f"{runner}/" if runner in ("http", "opencode") else ""
                    rel_path = f"{runner_prefix}responses/{sanitize_filename(r['model'])}/{p.id}.txt"
                    row += f" [view]({rel_path}) |"
            overall = r.get("overall_score_100", tot)
            scored_plugins = r.get("overall_scored_plugins", _scored_plugin_count(r, active_plugins))
            row += f" {overall if overall is not None else '-'} | {scored_plugins} | {r['total_time']}s | {m} |"
            lines.append(row)

        if ok:
            lines.extend(["", "---", "## 🏆 Leaderboards", ""])

            lines.append("### ⚡ Fastest Cold Load (TTFT)")
            lines.append("| # | Model | Load (s) |")
            lines.append("|---|---|---|")
            for i, r in enumerate(sorted(ok, key=lambda x: (x['ttft'] if isinstance(x['ttft'], (int, float)) else 999))[:10], 1):
                lines.append(f"| {i} | {r['model']} | {r['ttft']} |")

            for p in active_plugins:
                lines.extend(["", f"### 🧠 Best {p.name} Score (/100)"])
                lines.append("| # | Model | Score |")
                lines.append("|---|---|---|")
                for i, r in enumerate(sorted(ok, key=lambda x: _numeric_score(x, p.id), reverse=True)[:10], 1):
                    lines.append(f"| {i} | {r['model']} | {r.get(f'{p.id}_score', '-')} |")

            lines.extend(["", "### ⭐ Best Overall"])
            lines.append("| # | Model | Overall Score (0–100) |")
            lines.append("|---|---|---|")
            for i, r in enumerate(sorted(ok, key=lambda x: _plugin_total_score(x, active_plugins), reverse=True)[:10], 1):
                tot = _plugin_total_score(r, active_plugins)
                overall = r.get("overall_score_100", tot)
                lines.append(f"| {i} | {r['model']} | {overall if overall is not None else '-'} |")

        lines.extend(["", "---", "## 📐 Scoring Rubric", ""])
        for p in active_plugins:
            lines.extend([
                f"### {p.name} (native rubric)",
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
                    "| Quantitative trade-offs | 1 | Concrete measurements comparing MoE and dense |",
                ])

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
                    lines.append("| Criterion | Points | Total |")
                    lines.append("|---|---|---|")
                    for item in rubric:
                        lines.append(f"| {item['name']} | {item.get('points', '-')} | {item.get('total', '-')} |")
                    lines.append("")

        judge_criteria_present = any(
            _judge_criteria(r, p.id)
            for r in results for p in active_plugins
        )
        if judge_criteria_present:
            lines.extend(["", "---", "## 🧭 Judge Criteria and Evidence", ""])
            lines.append(
                "The following is the semantic judge's interpretation of each requirement; "
                "it is separate from the deterministic plugin rubric and does not change scores."
            )
            lines.append("")
            lines.append("| Model | Plugin | Judge | ID | Criterion | Status | Evidence |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in results:
                for p in active_plugins:
                    for judge_report in _judge_criteria(r, p.id):
                        judge_name = judge_report.get("judge", "-")
                        for item in judge_report.get("criteria", []):
                            def _cell(value):
                                return str(value or "-").replace("|", "\\|").replace("\n", " ")
                            lines.append(
                                f"| {_cell(r.get('model'))} | {_cell(p.name)} | "
                                f"{_cell(judge_name)} | {_cell(item.get('id'))} | "
                                f"{_cell(item.get('criterion'))} | {_cell(item.get('status'))} | "
                                f"{_cell(item.get('evidence'))} |"
                            )

        lines.extend(["", "## ❌ Failed Models", ""])
        for r in results:
            if r["status"] != "ok":
                lines.append(f"- **{r['model']}**: {r.get('error','?')}")
        lines.append("")

        content = "\n".join(lines)
        if output_dir:
            path = os.path.join(output_dir, "results.md")
            try:
                with open(path, "w") as f:
                    f.write(content)
                return path
            except OSError:
                pass
        return content
