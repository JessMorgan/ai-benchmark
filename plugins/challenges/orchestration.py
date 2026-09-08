"""Structured workflow orchestration challenge."""
from __future__ import annotations

import re

from benchmark.plugin import BenchmarkTaskPlugin, EvaluationResult
from benchmark.types import ConfigMap
from plugins.challenges._rubric import Rubric
from plugins.challenges._validators import parse_workflow_graph

_REQUIRED_TASK_IDS = ("1", "2", "3", "4")
_TASK_DECL_RE = re.compile(r"\b(?:task|step)[ _-]?(\d+)\b", re.IGNORECASE)
# The prompt does not mandate a ``task``/``step`` prefix, so top-level
# numbered-list lines ("1. ...", "2) ...", "3: ...", "4- ...") count as
# task declarations too.
_NUMBERED_DECL_RE = re.compile(r"^\s{0,3}(\d{1,2})[.):\-]\s+\S", re.MULTILINE)
_DEPENDS_ON_RE = re.compile(r"\[DEPENDS_ON\s*:\s*(?:task|step)?[ _-]?(\d+)\]", re.IGNORECASE)
_OPERATION_PATTERNS = (r"logs?", r"geo.?ip", r"anomal", r"pdf|report")
_TRACE_PATTERNS = (r"init|initialize|pending", r"running|start", r"complete|done|finish")


def _task_blocks(text: str) -> dict[str, list[str]]:
    """Group response lines into per-task blocks.

    A block starts at a task declaration — an explicit ``task N``/``step N``
    mention or a numbered-list marker such as ``1.`` — and runs through the
    following lines until the next declaration. This mirrors how
    ``parse_workflow_graph`` binds each line to its nearest declared task, so
    labels and trace states on continuation lines belong to the task they
    follow.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _TASK_DECL_RE.search(line) or _NUMBERED_DECL_RE.match(line)
        if match is not None:
            current = match.group(1)
        if current is not None:
            blocks.setdefault(current, []).append(line)
    return blocks


class OrchestrationPlugin(BenchmarkTaskPlugin):
    @property
    def id(self) -> str:
        return "orchestration"

    @property
    def version(self) -> str:
        return "1.2.0"

    @property
    def name(self) -> str:
        return "Orchestration & Workflow"

    @property
    def max_score(self) -> int:
        return int(16.0)

    @property
    def supports_streaming(self) -> bool:
        return True

    def get_prompt(self) -> str:
        return (
            "Plan this pipeline: process 1TB server logs, perform GeoIP lookup, run anomaly "
            "detection, and generate a PDF report. Use exactly four tasks with IDs 1-4. Mark "
            "parallel/sequential status per task, use `[DEPENDS_ON: task_id]`, and provide an "
            "execution trace containing init, running, and complete for every task."
        )

    def get_temperature(self, global_config: ConfigMap) -> float | None:
        val = global_config.get("orchestration_temperature")
        return float(val) if isinstance(val, (int, float)) else None

    def _numbered_task_ids_and_edges(self, text: str) -> tuple[set[str], list[tuple[str, str]]]:
        """Task IDs and dependency edges for numbered-list plans.

        The shared ``parse_workflow_graph`` only recognizes ``task``/``step``
        IDs; this local pass is the fallback for plans that number their
        tasks (the prompt does not mandate a ``task``/``step`` prefix, so the
        scorer must not require one).
        """
        id_matches = list(_NUMBERED_DECL_RE.finditer(text))
        bracket_matches = list(_DEPENDS_ON_RE.finditer(text))
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
            r"^\s{0,3}(\d{1,2})[.):\-].*?\b(?:depends on|requires|after)\b\s*(?:task|step)?[ _-]?(\d+)",
            text,
            re.IGNORECASE | re.MULTILINE,
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
        text = response_text.strip()
        rubric = Rubric(self.max_score)
        if not text:
            return rubric.results()
        graph = parse_workflow_graph(text)
        rubric.record_validation(graph)
        shared_tasks = graph.value.get("tasks", set()) if isinstance(graph.value, dict) else set()
        shared_edges = graph.value.get("edges", []) if isinstance(graph.value, dict) else []
        # The shared parser is authoritative when it actually finds
        # dependencies. When it sees task IDs but no edges — e.g. a numbered
        # plan whose trace lines happen to use "Task N" — its "no edges"
        # verdict reflects a blind spot, so the local numbered pass decides.
        if len(shared_tasks) >= 2 and shared_edges:
            valid = graph.valid
            edges = shared_edges
            validity_findings = [] if valid else [{"finding": f"invalid graph: {'; '.join(graph.errors) or 'no edges'}"}]
        else:
            # The shared parser only recognizes "Task N"/"Step N" IDs; a plan
            # that numbers its tasks ("1. ...", "2) ...") needs a local pass.
            local_tasks, edges = self._numbered_task_ids_and_edges(text)
            problems = self._graph_problems(local_tasks, edges)
            valid = not problems
            validity_findings = [] if valid else [{"finding": f"invalid graph: {'; '.join(problems)}"}]
        blocks = _task_blocks(text)
        declared_ids = set(blocks)
        covered_ids = declared_ids & set(_REQUIRED_TASK_IDS)
        operations = sum(
            1
            for task_id in covered_ids
            if any(
                re.search(pattern, line, re.IGNORECASE)
                for line in blocks[task_id]
                for pattern in _OPERATION_PATTERNS
            )
        )
        # Scale credit by the lesser of ID coverage and operation coverage:
        # declaring four IDs without naming the pipeline operations earns
        # nothing, and naming operations without the required IDs does not
        # reach full credit.
        id_credit = 4.0 * len(covered_ids) / len(_REQUIRED_TASK_IDS)
        op_credit = 4.0 * operations / len(_REQUIRED_TASK_IDS)
        breakdown = min(id_credit, op_credit)
        rubric.add_criterion(
            "Task breakdown presence", 4.0, breakdown,
            negative_findings=[] if breakdown >= 4.0 else [{"finding": "declare exactly four task operations with IDs 1-4"}],
        )
        if len(declared_ids) > 4:
            rubric.penalize_criterion("Task breakdown presence", 2.0, "declares more than four tasks")
        rubric.add_criterion(
            "Explicit dependency tagging", 4.0,
            4.0 if valid and len(edges) >= 3 else (2.0 if valid and edges else 0.0),
            negative_findings=validity_findings,
        )
        labels_ok = True
        for task_id in _REQUIRED_TASK_IDS:
            lines = blocks.get(task_id, [])
            if not lines or not any(re.search(r"parallel|sequential", line, re.IGNORECASE) for line in lines):
                labels_ok = False
            # Contradiction is judged across the whole task block (union of
            # labels), matching parse_workflow_graph's per-task label check
            # instead of only flagging lines that carry both words.
            joined = " ".join(lines)
            if re.search(r"parallel", joined, re.IGNORECASE) and re.search(r"sequential", joined, re.IGNORECASE):
                labels_ok = False
        rubric.add_criterion("Parallel vs sequential logic", 4.0, 4.0 if labels_ok else 0.0, negative_findings=[] if labels_ok else [{"finding": "each task needs one non-contradictory execution label"}])
        trace_ok = True
        for task_id in _REQUIRED_TASK_IDS:
            joined = " ".join(blocks.get(task_id, []))
            if not all(re.search(pattern, joined, re.IGNORECASE) for pattern in _TRACE_PATTERNS):
                trace_ok = False
        rubric.add_criterion("State / execution trace", 4.0, 4.0 if trace_ok else 0.0, negative_findings=[] if trace_ok else [{"finding": "every task needs init, running, and complete states"}])
        return rubric.results()

    def score(self, response_text: str) -> float:
        return self.evaluate(response_text).score
