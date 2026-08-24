"""Structured workflow orchestration challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_workflow_graph


class OrchestrationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "orchestration"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def name(self) -> str:
        return "Orchestration & Workflow"

    @property
    def max_score(self) -> int:
        return int(16.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self):
        return (
            "Plan this pipeline: process 1TB server logs, perform GeoIP lookup, run anomaly "
            "detection, and generate a PDF report. Use exactly four tasks with IDs 1-4. Mark "
            "parallel/sequential status per task, use `[DEPENDS_ON: task_id]`, and provide an "
            "execution trace containing init, running, and complete for every task."
        )

    def get_temperature(self, global_config):
        return global_config.get("orchestration_temperature")

    def evaluate(self, response_text):
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        graph = parse_workflow_graph(text)
        rubric.record_validation(graph)
        task_lines = [
            line for line in text.splitlines()
            if re.search(r"\b(?:task|step)[ _-]?\d+\b", line, re.IGNORECASE)
            and not re.search(r"DEPENDS_ON|depends on", line, re.IGNORECASE)
        ]
        declared_ids = {
            match.group(1)
            for line in task_lines
            if (match := re.search(r"\b(?:task|step)[ _-]?(\d+)\b", line, re.IGNORECASE))
        }
        operations = sum(
            any(re.search(pattern, line, re.IGNORECASE) for pattern in (r"logs?", r"geo.?ip", r"anomal", r"pdf|report"))
            for line in task_lines
        )
        rubric.add_criterion(
            "Task breakdown presence", 4.0,
            4.0 if declared_ids == {"1", "2", "3", "4"} and operations >= 4 else min(4.0, float(len(declared_ids))),
            negative_findings=[] if declared_ids == {"1", "2", "3", "4"} and operations >= 4 else [{"finding": "declare exactly four task operations with IDs 1-4"}],
        )
        edges = graph.value.get("edges", []) if isinstance(graph.value, dict) else []
        rubric.add_criterion("Explicit dependency tagging", 4.0, 4.0 if graph.valid and len(edges) >= 3 else (2.0 if edges else 0.0), negative_findings=[] if graph.valid else [{"finding": "dependency graph is incomplete, cyclic, or references unknown tasks"}])
        labels_ok = True
        for task_id in ("1", "2", "3", "4"):
            lines = [line for line in text.splitlines() if re.search(rf"\b(?:task|step)[ _-]?{task_id}\b", line, re.IGNORECASE)]
            if not lines or not any(re.search(r"parallel|sequential", line, re.IGNORECASE) for line in lines):
                labels_ok = False
            if any(re.search(r"parallel", line, re.IGNORECASE) and re.search(r"sequential", line, re.IGNORECASE) for line in lines):
                labels_ok = False
        rubric.add_criterion("Parallel vs sequential logic", 4.0, 4.0 if labels_ok else 0.0, negative_findings=[] if labels_ok else [{"finding": "each task needs one non-contradictory execution label"}])
        trace_ok = True
        for task_id in ("1", "2", "3", "4"):
            lines = [line for line in text.splitlines() if re.search(rf"\b(?:task|step)[ _-]?{task_id}\b", line, re.IGNORECASE)]
            joined = " ".join(lines)
            if not all(re.search(pattern, joined, re.IGNORECASE) for pattern in (r"init|initialize|pending", r"running|start", r"complete|done|finish")):
                trace_ok = False
        rubric.add_criterion("State / execution trace", 4.0, 4.0 if trace_ok else 0.0, negative_findings=[] if trace_ok else [{"finding": "every task needs init, running, and complete states"}])
        return rubric.results()

    def score(self, response_text):
        return self.evaluate(response_text).score
