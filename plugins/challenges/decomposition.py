"""Design-document decomposition challenge.

Unlike ``orchestration`` (which checks a rigid four-task workflow format), this
task presents a realistic design document and asks the model to decompose it into
a dependency-ordered task plan. It scores the parts that an output-contract check
cannot see: whether the plan actually covers the design document's deliverables
and whether the dependency **direction** is semantically correct (not merely a
valid-but-random acyclic graph).
"""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_workflow_graph

# The design document a candidate must decompose. Cannot be memorized by
# format recall alone: the rubric checks coverage of the document's specific
# deliverables and the semantic direction of the dependency edges.
_DESIGN_DOC = """\
# Design doc: distributed log-ingestion pipeline

You are designing a backend service that ingests 1 TB/day of application logs
from many producers, enriches each line, and exposes them for alerting and
reporting.

Requirements:
- Producers publish log batches over HTTP; the service must accept them under
  load and buffer them durably so no batch is lost on a crash.
- Each log line must be enriched by a GeoIP lookup (IP -> country/region) and
  normalized into a common schema before it can be queried.
- An anomaly-detection job must run over the normalized stream and flag
  suspicious patterns (e.g. login bursts, error spikes).
- Operators need a real-time alert feed for those anomalies and a nightly
  aggregate report.
- The pipeline must be observable: ingestion rate, enrichment lag, and
  anomaly-detector health must be exported to a metrics system.

Constraints: enrichment must not reorder or drop lines; anomaly detection only
ever sees enriched, normalized data; the nightly report is computed from the
stored normalized logs, not from the live stream.
"""

# Reference: which deliverable domains the plan should cover. Keywords are
# ordered most-specific-first so a line like "anomaly detection over normalized
# stream" resolves to the anomaly domain rather than the enrichment domain.
_REFERENCE_DOMAINS = {
    "ingestion": (("ingest", "buffer", "collect", "receive", "http"), "accept/buffer log batches durably"),
    "enrich": (("geoip", "enrich", "normaliz", "geo"), "normalize + GeoIP enrich each line"),
    "anomaly": (("anomal",), "anomaly detection on normalized stream"),
    "alert": (("alert", "notif", "feed", "realtime"), "real-time alert feed for anomalies"),
    "report": (("report", "aggregate", "summary", "nightly"), "nightly aggregate report"),
    "observe": (("metric", "observ", "monitor", "health", "export"), "observability / metrics export"),
}

# Expected edges (task -> depends-on) that must appear with the correct
# direction for the pipeline to be semantically sound.
_REQUIRED_EDGES = {("enrich", "ingestion"), ("anomaly", "enrich"), ("alert", "anomaly")}
# Anomaly detection must not be a prerequisite of enrichment, etc. These are
# edges that, if declared, would indicate a reversed or wrong dependency.
_FORBIDDEN_EDGES = {("ingestion", "enrich"), ("enrich", "anomaly"), ("anomaly", "alert")}


class DecompositionPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "decomposition"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def name(self) -> str:
        return "Design-Doc Decomposition"

    @property
    def max_score(self) -> int:
        return int(20.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            f"{_DESIGN_DOC}\n\n"
            "Break this design document into an ordered task plan. Give each task a "
            "stable ID and a one-line description, then explicitly declare dependencies "
            "between tasks, then state which stages can run in parallel and which must "
            "run sequentially, and finally give a short rationale for the chosen ordering.\n\n"
            "Use this structure:\n"
            "```\n"
            "Task 1: <description>\n"
            "Task 2: <description>\n"
            "Task 3 [DEPENDS_ON: 1]: <description>\n"
            "...\n"
            "```\n"
            "Then a section listing parallel stages and sequential stages, then a short "
            "ordering rationale tied to data flow and prerequisites."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("decomposition_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def get_judge_instructions(self) -> str:
        return (
            "This task asks the candidate to decompose a design document into a "
            "dependency-ordered task plan. Score the QUALITY of the decomposition, "
            "not just whether it contains task-like lines:\n"
            "1. Coverage — does the plan address every major deliverable in the design "
            "doc (the ingestion/buffering layer, the GeoIP+normalization enrichment, "
            "the anomaly detector, the alert feed, the nightly report, and observability)?\n"
            "2. Dependency correctness — are the dependencies semantically correct in "
            "both existence AND direction (e.g. anomaly detection must run after "
            "enrichment, the report must consume stored normalized logs, not the raw "
            "stream)? Penalize reversed or invented dependencies.\n"
            "3. Boundaries & ordering — are tasks sensibly carved and ordered (data "
            "flow, prerequisites respected), with parallel vs sequential stages called "
            "out sensibly?\n"
            "4. Rationale — is there a clear, non-generic justification for the "
            "ordering?\n"
            "Do NOT penalize a valid alternative decomposition just because it differs "
            "from a reference, and do not require a specific task count or naming "
            "scheme. Reward a plan that a competent engineer could execute "
            "end-to-end without re-deriving the architecture."
        )

    def _domain_of(self, line: str) -> str | None:
        best: tuple[str | None, int] = (None, len(line) + 1)
        for domain, (keywords, _label) in _REFERENCE_DOMAINS.items():
            for keyword in keywords:
                pos = re.search(keyword, line, re.IGNORECASE)
                if pos and pos.start() < best[1]:
                    best = (domain, pos.start())
        return best[0]

    def _score_coverage(self, text: str) -> tuple[float, list[str]]:
        found = {self._domain_of(line) for line in text.splitlines()}
        found.discard(None)
        covered = len(found)
        total = len(_REFERENCE_DOMAINS)
        missing = [f"{d} ({_REFERENCE_DOMAINS[d][1]})" for d in _REFERENCE_DOMAINS if d not in found]
        return covered / total, missing

    def evaluate(self, response_text: str) -> EvaluationResult:
        rubric = Rubric(self.max_score)
        if not response_text or not response_text.strip():
            return rubric.results()
        text = response_text.strip()

        # 1. Structural graph validity (parse; reject cycles/unknown refs).
        graph = parse_workflow_graph(text)
        rubric.record_validation(graph)
        valid = graph.valid and len(graph.value.get("edges", [])) >= 1
        rubric.add_criterion(
            "Dependency graph validity", 4.0,
            4.0 if valid else 0.0,
            negative_findings=[] if valid else [{"finding": f"invalid graph: {'; '.join(graph.errors) or 'no edges'}"}],
        )

        coverage, missing = self._score_coverage(text)
        rubric.add_criterion(
            "Coverage of design-doc deliverables", 6.0,
            6.0 * coverage,
            negative_findings=[{"finding": f"missing: {', '.join(missing)}"}] if missing else [],
        )

        declared_edges = graph.value.get("edges", []) if graph.value else []
        domain_by_task: dict[str, str] = {}
        for line in text.splitlines():
            m = re.search(r"task[ _-]?(\d+)", line, re.IGNORECASE)
            if not m:
                continue
            domain = self._domain_of(line)
            if domain:
                domain_by_task[m.group(1)] = domain
        domain_edges = set()
        for src, dst in declared_edges:
            ds, dd = domain_by_task.get(src), domain_by_task.get(dst)
            if ds and dd:
                domain_edges.add((ds, dd))
        correct = sum(1 for e in _REQUIRED_EDGES if e in domain_edges)
        reversed_edges = sorted(domain_edges & _FORBIDDEN_EDGES)
        edge_points = 6.0 * correct / len(_REQUIRED_EDGES)
        if reversed_edges:
            # A reversed dependency is a substantive correctness error over and
            # above a missing edge; cap the criterion at half marks.
            edge_points = min(edge_points, 3.0)
        findings = []
        for e in _REQUIRED_EDGES:
            if e not in domain_edges:
                findings.append(f"missing dependency {e[0]} -> {e[1]}")
        for e in reversed_edges:
            findings.append(f"reversed dependency {e[1]} -> {e[0]}")
        rubric.add_criterion(
            "Semantic dependency direction", 6.0, edge_points,
            negative_findings=[{"finding": f} for f in findings] if findings else [],
        )

        has_parallel = bool(re.search(r"parallel", text, re.IGNORECASE))
        has_sequential = bool(re.search(r"sequential|must run (one )?after", text, re.IGNORECASE))
        rubric.add_criterion(
            "Parallelization reasoning", 2.0, 2.0 if has_parallel and has_sequential else 1.0 if has_parallel or has_sequential else 0.0,
            negative_findings=[] if has_parallel and has_sequential else [{"finding": "parallel vs sequential stages not both identified"}],
        )

        rationale_hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in (
            r"data flows?|data flow", r"prerequisite|pre-requisite", r"depends on|dependency",
            r"order|before|after|first|then",
        ))
        rationale_points = 2.0 if rationale_hits >= 2 else (1.0 if rationale_hits == 1 else 0.0)
        rubric.add_criterion(
            "Ordering rationale", 2.0, rationale_points,
            negative_findings=[] if rationale_points else [{"finding": "no explicit ordering rationale"}],
        )

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
