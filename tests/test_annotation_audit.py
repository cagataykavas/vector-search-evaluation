from __future__ import annotations

import json

import pytest

from search.annotation_audit import audit_annotations, main


def bundle(grades: list[tuple[str, str, str, int]]) -> dict:
    return {
        "dataset_id": "runbooks-2026-11",
        "rubric_version": "relevance-v2",
        "annotations": [
            {
                "query_id": query_id,
                "document_id": document_id,
                "assessor_id": assessor_id,
                "grade": grade,
            }
            for query_id, document_id, assessor_id, grade in grades
        ],
    }


def reliable_bundle() -> dict:
    return bundle(
        [
            ("q1", "d1", "a1", 3),
            ("q1", "d1", "a2", 3),
            ("q1", "d2", "a1", 1),
            ("q1", "d2", "a2", 1),
            ("q2", "d3", "a1", 0),
            ("q2", "d3", "a2", 0),
        ]
    )


def test_audit_passes_reliable_labels_and_reduces_identifiers() -> None:
    audit = audit_annotations(reliable_bundle())

    assert audit.passed is True
    assert audit.fleiss_kappa == pytest.approx(1.0)
    assert audit.pair_agreement == pytest.approx(1.0)
    assert audit.grade_distribution == (2, 2, 0, 2)
    report = json.dumps(audit.to_dict())
    assert "q1" not in report
    assert "d1" not in report
    assert "a1" not in report


def test_digest_is_independent_of_row_order_but_binds_provenance() -> None:
    payload = reliable_bundle()
    reversed_payload = payload | {"annotations": list(reversed(payload["annotations"]))}
    changed_rubric = payload | {"rubric_version": "relevance-v3"}

    assert (
        audit_annotations(payload).input_digest == audit_annotations(reversed_payload).input_digest
    )
    assert audit_annotations(payload).input_digest != audit_annotations(changed_rubric).input_digest


def test_disagreement_fails_both_gates() -> None:
    audit = audit_annotations(
        bundle(
            [
                ("q1", "d1", "a1", 3),
                ("q1", "d1", "a2", 0),
                ("q1", "d2", "a1", 0),
                ("q1", "d2", "a2", 3),
            ]
        )
    )

    assert audit.passed is False
    assert audit.fleiss_kappa == pytest.approx(-1.0)
    assert audit.findings == (
        "fleiss_kappa_below_threshold",
        "pair_agreement_below_threshold",
    )


def test_constant_labels_report_undefined_chance_corrected_agreement() -> None:
    payload = bundle(
        [
            ("q1", "d1", "a1", 1),
            ("q1", "d1", "a2", 1),
            ("q1", "d2", "a1", 1),
            ("q1", "d2", "a2", 1),
        ]
    )

    audit = audit_annotations(payload)

    assert audit.pair_agreement == 1.0
    assert audit.fleiss_kappa is None
    assert audit.findings == ("fleiss_kappa_undefined_no_grade_variation",)


def test_incomplete_assessor_matrix_is_rejected() -> None:
    payload = reliable_bundle()
    payload["annotations"].pop()

    with pytest.raises(ValueError, match="every assessor"):
        audit_annotations(payload)


@pytest.mark.parametrize("grade", [True, -1, 4, 1.5, "2"])
def test_invalid_grades_are_rejected(grade) -> None:
    payload = reliable_bundle()
    payload["annotations"][0]["grade"] = grade

    with pytest.raises(ValueError, match="grade must be an integer"):
        audit_annotations(payload)


def test_duplicate_item_assessor_rating_is_rejected() -> None:
    payload = reliable_bundle()
    payload["annotations"].append(dict(payload["annotations"][0]))

    with pytest.raises(ValueError, match="duplicates an item-assessor"):
        audit_annotations(payload)


def test_unknown_fields_are_rejected() -> None:
    payload = reliable_bundle() | {"notes": "ignore me"}

    with pytest.raises(ValueError, match="contain exactly"):
        audit_annotations(payload)


def test_cli_writes_report_and_uses_gate_exit_codes(tmp_path) -> None:
    input_path = tmp_path / "annotations.json"
    output_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps(reliable_bundle()), encoding="utf-8")

    assert main([str(input_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True

    failing = bundle(
        [
            ("q1", "d1", "a1", 0),
            ("q1", "d1", "a2", 3),
            ("q1", "d2", "a1", 3),
            ("q1", "d2", "a2", 0),
        ]
    )
    input_path.write_text(json.dumps(failing), encoding="utf-8")
    assert main([str(input_path), "--output", str(output_path)]) == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is False


def test_cli_rejects_duplicate_json_keys_without_overwriting_report(tmp_path) -> None:
    input_path = tmp_path / "annotations.json"
    output_path = tmp_path / "audit.json"
    output_path.write_text("sentinel", encoding="utf-8")
    input_path.write_text(
        '{"dataset_id":"a","dataset_id":"b","rubric_version":"v","annotations":[]}',
        encoding="utf-8",
    )

    assert main([str(input_path), "--output", str(output_path)]) == 2
    assert output_path.read_text(encoding="utf-8") == "sentinel"
