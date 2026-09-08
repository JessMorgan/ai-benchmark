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

# Reference: which deliverable domains the plan should cover. A line resolves
# to the domain with the most distinct keyword hits (ties: earliest first hit),
# so "anomaly alerts and real-time feed" resolves to the alert domain even
# though "anomaly" appears first.
_REFERENCE_DOMAINS = {
    "ingestion": (("ingest", "buffer", "collect", "receive", "http"), "accept/buffer log batches durably"),
    "enrich": (("geoip", "enrich", "normaliz", r"geo(?=ip|\b)"), "normalize + GeoIP enrich each line"),
    "anomaly": (("anomal",), "anomaly detection on normalized stream"),
    "alert": (("alert", "notif", r"\bfeed\b", "realtime"), "real-time alert feed for anomalies"),
    "report": (("report", "aggregate", "summary", "nightly"), "nightly aggregate report"),
    "observe": (("metric", "observ", "monitor", "health", "export"), "observability / metrics export"),
}

# Expected edges (task -> depends-on) that must appear with the correct
# direction for the pipeline to be semantically sound.
_REQUIRED_EDGES = {("enrich", "ingestion"), ("anomaly", "enrich"), ("alert", "anomaly")}
# Anomaly detection must not be a prerequisite of enrichment, etc. These are
# edges that, if declared, would indicate a reversed or wrong dependency.
_FORBIDDEN_EDGES = {("ingestion", "enrich"), ("enrich", "anomaly"), ("anomaly", "alert")}

# Independence language that justifies running stages in parallel; the bare
# word "parallel" (which the prompt itself elicits) is not enough.
_INDEPENDENCE_RE = (
    r"independent|no dependenc|does not depend|without dependenc"
    r"|run (?:in parallel|concurrently|simultaneously|together|at the same time)"
)

# Named deliverable domains an ordering rationale may reference instead of
# task IDs (each names one task of the reference decomposition).
_RATIONALE_DOMAIN_RE = r"ingest|enrich|normaliz|anomal|alert|report|metric|observ"

# Task IDs in the candidate's own naming: "Task 1" / "task-1" as well as the
# compact "T1" form. The judge must not require a specific naming scheme.
_TASK_ID_RE = r"\b(?:task[ _-]?|t[ _-]?)(\d+)\b"


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
        """Return the deliverable domain a line describes.

        The domain with the most distinct keyword hits wins; ties fall to the
        domain whose first hit appears earliest in the line.
        """
        best_domain: str | None = None
        best_hits = 0
        best_pos = len(line) + 1
        for domain, (keywords, _label) in _REFERENCE_DOMAINS.items():
            hits = 0
            first_pos = len(line) + 1
            for keyword in keywords:
                pos = re.search(keyword, line, re.IGNORECASE)
                if pos:
                    hits += 1
                    first_pos = min(first_pos, pos.start())
            if hits > best_hits or (hits == best_hits and hits and first_pos < best_pos):
                best_domain, best_hits, best_pos = domain, hits, first_pos
        return best_domain

    def _score_coverage(self, text: str) -> tuple[float, list[str]]:
        found = {self._domain_of(line) for line in text.splitlines()}
        found.discard(None)
        covered = len(found)
        total = len(_REFERENCE_DOMAINS)
        missing = [f"{d} ({_REFERENCE_DOMAINS[d][1]})" for d in _REFERENCE_DOMAINS if d not in found]
        return covered / total, missing

    def _task_ids_and_edges(self, text: str) -> tuple[set[str], list[tuple[str, str]]]:
        """Task IDs and dependency edges, accepting both ``Task N`` and ``T<N>``.

        The shared ``parse_workflow_graph`` only recognizes ``task``/``step``
        IDs; this local pass is the fallback for plans that name their tasks
        ``T1``, ``T2``, ... (the judge must not require a specific naming
        scheme).
        """
        id_matches = list(re.finditer(_TASK_ID_RE, text, re.IGNORECASE))
        bracket_matches = list(
            re.finditer(r"\[DEPENDS_ON\s*:\s*(?:task|t)?[ _-]?(\d+)\]", text, re.IGNORECASE)
        )
        edges: list[tuple[str, str]] = []
        referenced_positions: set[int] = set()
        for dependency in bracket_matches:
            current = [
                item.group(1)
                for item in id_matches
                if item.start() < dependency.start()
                and not any(
                    reference.start() <= item.start() < reference.end()
                    for reference in bracket_matches
                )
            ]
            if current:
                edges.append((current[-1], dependency.group(1)))
            referenced_positions.update(
                item.start()
                for item in id_matches
                if dependency.start() <= item.start() < dependency.end()
            )
        for match in re.finditer(
            rf"{_TASK_ID_RE}\s+(?:depends on|requires|after)\s+(?:task|t)[ _-]?(\d+)",
            text,
            re.IGNORECASE,
        ):
            edges.append((match.group(1), match.group(2)))
        for match in re.finditer(
            rf"{_TASK_ID_RE}\s*-->?\s*(?:task|t)[ _-]?(\d+)",
            text,
            re.IGNORECASE,
        ):
            edges.append((match.group(1), match.group(2)))
        task_ids = {
            match.group(1)
            for match in id_matches
            if match.start() not in referenced_positions
        }
        return task_ids, list(dict.fromkeys(edges))

    @staticmethod
    def _graph_problems(task_ids: set[str], edges: list[tuple[str, str]]) -> list[str]:
        """Structural problems of a task graph (empty list = valid)."""
        problems: list[str] = []
        if len(task_ids) < 2:
            problems.append("fewer than two task IDs found")
        if not edges:
            problems.append("no dependency edges found")
        for source, target in edges:
            if source not in task_ids or target not in task_ids:
                problems.append(f"dependency references unknown task {target}")
        if problems:
            return problems
        adjacency: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
        for source, target in edges:
            adjacency[source].add(target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                problems.append("dependency graph contains a cycle")
                return
            if node in visited:
                return
            visiting.add(node)
            for child in adjacency.get(node, ()):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for task_id in adjacency:
            visit(task_id)
        return problems

    def evaluate(self, response_text: str) -> EvaluationResult:
        rubric = Rubric(self.max_score)
        if not response_text or not response_text.strip():
            return rubric.results()
        text = response_text.strip()

        # 1. Structural graph validity (parse; reject cycles/unknown refs).
        graph = parse_workflow_graph(text)
        rubric.record_validation(graph)
        shared_tasks = graph.value.get("tasks", set()) if graph.value else set()
        if len(shared_tasks) >= 2:
            declared_edges = graph.value.get("edges", [])
            valid = graph.valid and len(declared_edges) >= 1
            validity_findings = [] if valid else [{"finding": f"invalid graph: {'; '.join(graph.errors) or 'no edges'}"}]
        else:
            # The shared parser only recognizes "Task N"/"Step N" IDs; a plan
            # that names its tasks "T1", "T2", ... needs a local T-aware parse.
            local_tasks, declared_edges = self._task_ids_and_edges(text)
            problems = self._graph_problems(local_tasks, declared_edges)
            valid = not problems
            validity_findings = [] if valid else [{"finding": f"invalid graph: {'; '.join(problems)}"}]
        rubric.add_criterion(
            "Dependency graph validity", 4.0,
            4.0 if valid else 0.0,
            negative_findings=validity_findings,
        )

        coverage, missing = self._score_coverage(text)
        rubric.add_criterion(
            "Coverage of design-doc deliverables", 6.0,
            6.0 * coverage,
            negative_findings=[{"finding": f"missing: {', '.join(missing)}"}] if missing else [],
        )

        # Bind each task's domain to its own description line: the FIRST line
        # that mentions the task ID wins, so an appended "Domain mapping"
        # section cannot re-declare the tasks' domains (the old last-line-wins
        # binding let a degenerate plan score full marks by mapping the
        # domains itself).
        domain_by_task: dict[str, str] = {}
        seen_tasks: set[str] = set()
        for line in text.splitlines():
            for m in re.finditer(_TASK_ID_RE, line, re.IGNORECASE):
                task_id = m.group(1)
                if task_id in seen_tasks:
                    continue
                seen_tasks.add(task_id)
                domain = self._domain_of(line)
                if domain:
                    domain_by_task[task_id] = domain
        domain_edges = set()
        for src, dst in declared_edges:
            ds, dd = domain_by_task.get(src), domain_by_task.get(dst)
            if ds and dd:
                domain_edges.add((ds, dd))
        correct = sum(1 for e in _REQUIRED_EDGES if e in domain_edges)
        reversed_edges = sorted(domain_edges & _FORBIDDEN_EDGES)
        # Each required edge earns one third of the criterion; every declared
        # forbidden (reversed) edge is penalized individually at the same
        # weight, and the total penalty is capped by the criterion maximum.
        per_edge = 6.0 / len(_REQUIRED_EDGES)
        edge_points = max(0.0, per_edge * (correct - len(reversed_edges)))
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
        has_independence = bool(re.search(_INDEPENDENCE_RE, text, re.IGNORECASE))
        has_sequential = bool(re.search(r"sequential|must run (one )?after", text, re.IGNORECASE))
        parallel_ok = has_parallel and has_independence
        parallel_points = (
            2.0 if (parallel_ok and has_sequential)
            else 1.0 if (parallel_ok or has_sequential)
            else 0.0
        )
        parallel_findings = []
        if not has_parallel:
            parallel_findings.append("parallel stages not identified")
        elif not has_independence:
            parallel_findings.append(
                "parallel stages named but not justified by independence "
                "(e.g. 'independent', 'no dependency')"
            )
        if not has_sequential:
            parallel_findings.append("sequential stages not identified")
        rubric.add_criterion(
            "Parallelization reasoning", 2.0, parallel_points,
            negative_findings=[{"finding": f} for f in parallel_findings] if parallel_findings else [],
        )

        rationale_hits = sum(bool(re.search(p, text, re.IGNORECASE)) for p in (
            r"data flows?|data flow", r"prerequisite|pre-requisite", r"depends on|dependency",
            r"order|before|after|first|then",
        ))
        # The rationale must reference the plan's specific tasks — by task ID
        # ("Task 1 before Task 3") or by named deliverable ("ingestion",
        # "enrichment") — so ordering vocabulary alone cannot earn it.
        rationale_lines = [
            line for line in text.splitlines()
            if re.search(r"rationale", line, re.IGNORECASE)
        ]
        rationale_scope = "\n".join(rationale_lines) if rationale_lines else text
        rationale_specific = bool(
            re.search(_TASK_ID_RE, rationale_scope, re.IGNORECASE)
            or re.search(_RATIONALE_DOMAIN_RE, rationale_scope, re.IGNORECASE)
        )
        rationale_points = (
            2.0 if (rationale_specific and rationale_hits >= 2)
            else 1.0 if (rationale_specific and rationale_hits == 1)
            else 0.0
        )
        if rationale_points:
            rationale_findings = []
        elif not rationale_specific:
            rationale_findings = [
                {"finding": "ordering rationale does not reference specific tasks (IDs or named deliverables)"}
            ]
        else:
            rationale_findings = [{"finding": "no explicit ordering rationale"}]
        rubric.add_criterion(
            "Ordering rationale", 2.0, rationale_points,
            negative_findings=rationale_findings,
        )

        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
