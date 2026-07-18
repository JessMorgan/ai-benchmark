import os
from datetime import datetime

from benchmark_plugin import BenchmarkOutputPlugin
from benchmark_outputs import sanitize_filename, _plugin_total_score, _numeric_score


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
