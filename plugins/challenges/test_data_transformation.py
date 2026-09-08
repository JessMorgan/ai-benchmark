"""Regression tests for data-transformation scoring fixes (DX-1, DX-2)."""
import json

from plugins.challenges.data_transformation import (
    DATA_TRANSFORMATION_EXPECTED_OUTPUT,
    DataTransformationPlugin,
)


def _expected_payload() -> dict:
    return json.loads(json.dumps(DATA_TRANSFORMATION_EXPECTED_OUTPUT))


def _summary_criterion(result) -> dict:
    return next(item for item in result.rubric if item["name"] == "Derived summary")


def test_dx1_fabricated_summary_inconsistent_with_records_scores_lower():
    # Measured pre-fix: records with total=1.0 but a fabricated summary of
    # 570.0 (matching the expected answer) scored 15/22 — the summary
    # criterion only compared against the expected output, never against the
    # response's own records.
    payload = _expected_payload()
    for record in payload["records"]:
        record["total"] = 1.0
    result = DataTransformationPlugin().evaluate(json.dumps(payload))
    assert result.score < 15.0
    assert _summary_criterion(result)["earned"] < 3.0


def test_dx1_top_order_id_inconsistent_with_first_record_scores_lower():
    # Measured pre-fix class: top_order_id matching the expected answer but
    # contradicting the response's own first record scored 21/22 when the
    # summary was the only disagreement. Swapping the top two records keeps
    # the summary answer-key-correct (O-202) while records[0] is O-208:
    # 20.8 pre-fix, must score lower once the cross-check is applied.
    payload = _expected_payload()
    payload["records"][0], payload["records"][1] = payload["records"][1], payload["records"][0]
    result = DataTransformationPlugin().evaluate(json.dumps(payload))
    assert result.score < 20.8
    assert _summary_criterion(result)["earned"] < 3.0
    findings = _summary_criterion(result)["negative_findings"]
    assert any("inconsistent with the emitted records" in finding["finding"] for finding in findings)


def test_dx1_correct_response_summary_still_scores_full():
    # Invariant: a correct, self-consistent response keeps full summary
    # credit (and the full 22/22).
    result = DataTransformationPlugin().evaluate(json.dumps(_expected_payload()))
    assert result.score == 22.0
    assert _summary_criterion(result)["earned"] == 3.0


def _sorting_criterion(result) -> dict:
    return next(item for item in result.rubric if item["name"] == "Sorting and ranking")


def test_dx2_reversed_records_with_positional_ranks_scores_lower():
    # Measured pre-fix: a reversed-order response scored 20.8/22 — the rank
    # sub-credit was position-based (rank at position i == i), so any
    # schema-valid set of 5 records carrying ranks 1-5 earned the full 1.5
    # regardless of which record sat at which position.
    payload = _expected_payload()
    payload["records"] = list(reversed(payload["records"]))
    for index, record in enumerate(payload["records"], 1):
        record["rank"] = index
    result = DataTransformationPlugin().evaluate(json.dumps(payload))
    assert _sorting_criterion(result)["earned"] < 1.8
    assert result.score < 19.8


def test_dx2_correct_response_keeps_full_sorting_credit():
    # Invariant: a correct response (each record at its expected position
    # with its own rank) keeps the full 3.0 sorting credit and 22/22.
    result = DataTransformationPlugin().evaluate(json.dumps(_expected_payload()))
    assert _sorting_criterion(result)["earned"] == 3.0
    assert result.score == 22.0
