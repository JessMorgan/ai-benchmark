"""Tests for the design-doc decomposition challenge."""
from plugins.challenges.decomposition import DecompositionPlugin


def _full_answer():
    return """Task 1: Accept and durably buffer incoming log batches over HTTP ingestion
Task 2 [DEPENDS_ON: 1]: Enrich each line with GeoIP lookup and normalize to common schema
Task 3 [DEPENDS_ON: 2]: Run anomaly detection over normalized stream
Task 4 [DEPENDS_ON: 3]: Emit real-time alert feed for anomalies
Task 5 [DEPENDS_ON: 2]: Compute nightly aggregate report from stored normalized logs
Task 6: Export ingestion rate, enrichment lag and anomaly-detector health to metrics system
Parallel stages: after enrichment, the alert feed, nightly report and observability export can all run in parallel.
Sequential stages: enrichment must run sequentially after ingestion; anomaly detection sequentially after enrichment; the alert feed sequentially after anomaly detection.
Ordering rationale: the pipeline order follows data flow, because data flows from ingestion to enrichment, then on to detection and the consumers, so each task runs after its prerequisite input is ready.
"""


def test_metadata():
    plugin = DecompositionPlugin()
    assert plugin.id == "decomposition"
    assert plugin.version == "0.1.0"
    assert plugin.name == "Design-Doc Decomposition"
    assert plugin.max_score == 20.0
    assert plugin.supports_streaming is True
    assert plugin.get_judge_instructions()


def test_prompt_embeds_the_design_document():
    plugin = DecompositionPlugin()
    prompt = plugin.get_prompt()
    assert "log-ingestion" in prompt
    assert "GeoIP" in prompt
    assert "DEPENDS_ON" in prompt


def test_judge_instructions_cover_quality_dimensions():
    plugin = DecompositionPlugin()
    instructions = plugin.get_judge_instructions()
    assert "Coverage" in instructions
    assert "Dependency correctness" in instructions
    assert "Boundaries" in instructions
    assert "Rationale" in instructions


def test_empty_response_is_zero():
    assert DecompositionPlugin().score("") == 0.0
    assert DecompositionPlugin().score("   ") == 0.0


def test_full_correct_decomposition_scores_maximum():
    result = DecompositionPlugin().evaluate(_full_answer())
    assert result.score == 20.0
    assert all(criterion["earned"] == criterion["max"] for criterion in result.rubric)


def test_semantic_dependency_direction_is_punished_when_reversed():
    reversed_answer = """Task 1: Enrich each log line with GeoIP lookup and normalize it
Task 2 [DEPENDS_ON: 1]: Accept and buffer raw log batches over HTTP first
Task 3 [DEPENDS_ON: 2]: Run anomaly detection on raw stream
Task 4: Real-time alert feed
Task 5: Nightly aggregate report computed from normalized stored logs
Task 6: Export metrics
Parallel stages: report and metrics in parallel.
Sequential stages: mandatory chain.
Ordering rationale: data flows top to bottom, each task follows its prerequisite.
"""
    result = DecompositionPlugin().evaluate(reversed_answer)
    direction = next(c for c in result.rubric if c["name"] == "Semantic dependency direction")
    assert direction["earned"] < direction["max"]
    assert any("reversed dependency" in f["finding"] for f in direction["negative_findings"])


def test_format_only_answer_does_not_reach_full_credit():
    shallow = """Task 1: ingest data
Task 2: enrich
Task 3: detect
Task 4: report
Parallel stages: some.
Sequential stages: some chain.
Ordering rationale: ordered.
"""
    assert DecompositionPlugin().score(shallow) < DecompositionPlugin().max_score / 2


def test_objective_score_matches_evaluate():
    answer = _full_answer()
    plugin = DecompositionPlugin()
    assert plugin.score(answer) == plugin.evaluate(answer).score


def test_duplicate_domain_task_does_not_double_count_coverage():
    redundant = """Task 1: Accept and buffer log batches over HTTP
Task 2 [DEPENDS_ON: 1]: GeoIP enrich the normalized logs
Task 3 [DEPENDS_ON: 2]: Anomaly detection on the enriched stream
Task 4 [DEPENDS_ON: 3]: Real-time alert feed for anomalies
Task 5: Nightly aggregate report
Task 6: Observability and metrics export
Parallel stages: task 4, 5 and 6 in parallel.
Sequential stages: 1 then 2 then 3.
Ordering rationale: data flows in dependency order.
"""
    result = DecompositionPlugin().evaluate(redundant)
    coverage = next(c for c in result.rubric if c["name"] == "Coverage of design-doc deliverables")
    assert coverage["earned"] == coverage["max"]


def test_get_temperature():
    plugin = DecompositionPlugin()
    assert plugin.get_temperature({}) is None
    assert plugin.get_temperature({"decomposition_temperature": "0.5"}) is None
    assert plugin.get_temperature({"decomposition_temperature": 0.4}) == 0.4


def test_unknown_domain_edge_is_ignored_not_scored():
    # An edge whose endpoint is a task that maps to no known domain should be
    # skipped (not crash and not poison the semantic-direction score).
    answer = """Task 1: Accept and buffer log batches over HTTP
Task 2 [DEPENDS_ON: 1]: GeoIP enrich the normalized logs
Task 3 [DEPENDS_ON: 2]: Anomaly detection on the enriched stream
Task 4 [DEPENDS_ON: 3]: Real-time alert feed for anomalies
Task 5: Nightly aggregate report
Task 6: Observability and metrics export
Task 9 [DEPENDS_ON: 1]: completely unrelated helper step that mentions nothing relevant
Parallel stages: 4, 5 and 6 in parallel.
Sequential stages: 1 then 2 then 3.
Ordering rationale: data flows in dependency order.
"""
    result = DecompositionPlugin().evaluate(answer)
    direction = next(c for c in result.rubric if c["name"] == "Semantic dependency direction")
    assert direction["earned"] == direction["max"]
