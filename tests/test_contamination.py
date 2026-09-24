from __future__ import annotations

import json

import pytest

from search.contamination import (
    ContaminationInputError,
    ContaminationPolicy,
    ContaminationReason,
    MatchKind,
    TextRecord,
    audit_contamination,
    load_jsonl,
    main,
)


def _record(record_id: str, text: str) -> TextRecord:
    return TextRecord(record_id=record_id, text=text)


def test_clean_partitions_are_accepted() -> None:
    report = audit_contamination(
        [_record("train-1", "reset an expired oauth access token safely")],
        [_record("eval-1", "configure reciprocal rank fusion for search")],
    )

    assert report.accepted is True
    assert report.exact_leaked_queries == 0
    assert report.near_leaked_queries == 0
    assert report.findings == ()


def test_normalized_exact_leakage_is_rejected_without_exposing_text() -> None:
    report = audit_contamination(
        [_record("train-1", "How do I reset OAuth-2 tokens?")],
        [_record("eval-1", "  HOW do I reset oauth 2 tokens  ")],
    )

    assert report.accepted is False
    assert report.exact_leaked_queries == 1
    assert report.matches[0].kind is MatchKind.EXACT
    assert report.matches[0].reference_digest == report.matches[0].evaluation_digest
    encoded = json.dumps(report.as_dict(), allow_nan=False)
    assert "HOW do I" not in encoded
    assert report.findings[0].reason is ContaminationReason.EXACT_LEAKAGE_LIMIT_EXCEEDED


def test_near_duplicate_query_is_detected_by_token_jaccard() -> None:
    report = audit_contamination(
        [_record("train-1", "rotate expired oauth token for upstream service safely")],
        [_record("eval-1", "rotate an expired oauth token for upstream service safely")],
        policy=ContaminationPolicy(near_similarity_threshold=0.75),
    )

    assert report.accepted is False
    assert report.near_leaked_queries == 1
    assert report.matches[0].kind is MatchKind.NEAR
    assert report.matches[0].similarity == 0.888889


def test_short_queries_only_use_exact_matching() -> None:
    report = audit_contamination(
        [_record("train-1", "oauth timeout error")],
        [_record("eval-1", "oauth timeout failure")],
        policy=ContaminationPolicy(min_tokens_for_near_match=4),
    )

    assert report.accepted is True
    assert report.pair_comparisons == 0


def test_leakage_budgets_can_be_explicitly_calibrated() -> None:
    report = audit_contamination(
        [_record("train-1", "rotate expired oauth token for upstream service safely")],
        [
            _record("eval-1", "rotate expired oauth token for upstream service safely"),
            _record("eval-2", "rotate an expired oauth token for upstream service safely"),
            _record("eval-3", "reciprocal rank fusion combines independent rankings"),
        ],
        policy=ContaminationPolicy(
            max_exact_leaked_queries=1,
            max_near_leak_rate=0.34,
            near_similarity_threshold=0.75,
        ),
    )

    assert report.accepted is True
    assert report.exact_leaked_queries == 1
    assert report.near_leak_rate == pytest.approx(1 / 3, abs=1e-6)


def test_evaluation_duplicates_are_a_separate_release_failure() -> None:
    report = audit_contamination(
        [_record("train", "a completely unrelated reference query text")],
        [
            _record("eval-2", "Duplicate evaluation query!"),
            _record("eval-1", "duplicate evaluation query"),
        ],
    )

    assert report.accepted is False
    assert report.duplicate_evaluation_ids == ("eval-1", "eval-2")
    assert report.findings[0].reason is ContaminationReason.DUPLICATE_EVALUATION_TEXT


def test_comparison_budget_exhaustion_fails_closed() -> None:
    references = [
        _record(f"train-{index}", f"shared tokens for query number {index} reference")
        for index in range(3)
    ]
    report = audit_contamination(
        references,
        [_record("eval", "shared tokens for query number x evaluation")],
        policy=ContaminationPolicy(max_pair_comparisons=1),
    )

    assert report.accepted is False
    assert report.pair_comparisons == 1
    assert report.findings[0].reason is ContaminationReason.COMPARISON_BUDGET_EXCEEDED


@pytest.mark.parametrize(
    ("reference", "evaluation", "message"),
    [
        ([], [_record("e", "valid query")], "reference split must contain"),
        ([_record("r", "valid query")], [], "evaluation split must contain"),
        (
            [_record("same", "reference query")],
            [_record("same", "evaluation query")],
            "record IDs occur in both splits",
        ),
        (
            [_record("r", "reference query")],
            [_record("e", "!!!")],
            "has no comparable tokens",
        ),
    ],
)
def test_malformed_split_evidence_is_rejected(
    reference: list[TextRecord],
    evaluation: list[TextRecord],
    message: str,
) -> None:
    with pytest.raises(ContaminationInputError, match=message):
        audit_contamination(reference, evaluation)


def test_duplicate_ids_and_oversized_text_are_rejected() -> None:
    duplicate = _record("r", "valid reference query")
    with pytest.raises(ContaminationInputError, match="duplicate reference record id"):
        audit_contamination([duplicate, duplicate], [_record("e", "valid evaluation query")])

    with pytest.raises(ContaminationInputError, match="exceeds 4 bytes"):
        audit_contamination(
            [_record("r", "large")],
            [_record("e", "tiny")],
            policy=ContaminationPolicy(max_text_bytes=4),
        )

    with pytest.raises(ContaminationInputError, match="invalid record id"):
        audit_contamination(
            [TextRecord(record_id="unsafe id", text="reference query")],
            [_record("e", "evaluation query")],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_exact_leaked_queries": -1}, "must be non-negative"),
        ({"max_near_leak_rate": float("nan")}, "finite and between"),
        ({"near_similarity_threshold": 0}, "must be greater than zero"),
        ({"min_tokens_for_near_match": 0}, "must be at least one"),
        ({"max_pair_comparisons": True}, "must be an integer"),
        ({"reject_evaluation_duplicates": 1}, "must be a boolean"),
    ],
)
def test_invalid_policy_values_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ContaminationPolicy(**kwargs)  # type: ignore[arg-type]


def test_jsonl_loader_rejects_duplicate_fields_and_blank_lines(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"id":"a","id":"b","text":"query"}\n', encoding="utf-8")
    with pytest.raises(ContaminationInputError, match="duplicate JSON field"):
        load_jsonl(duplicate, max_records=10)

    blank = tmp_path / "blank.jsonl"
    blank.write_text('{"id":"a","text":"query"}\n\n', encoding="utf-8")
    with pytest.raises(ContaminationInputError, match="blank line"):
        load_jsonl(blank, max_records=10)


def test_cli_distinguishes_accept_reject_and_malformed(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    output = tmp_path / "report.json"
    reference.write_text('{"id":"r","text":"oauth token rotation procedure"}\n')
    evaluation.write_text('{"id":"e","text":"hybrid retrieval benchmark design"}\n')

    assert main([str(reference), str(evaluation), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True
    capsys.readouterr()

    evaluation.write_text('{"id":"e","text":"oauth token rotation procedure"}\n')
    assert main([str(reference), str(evaluation)]) == 3
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    evaluation.write_text("not-json\n")
    assert main([str(reference), str(evaluation)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "malformed"


def test_report_order_is_deterministic() -> None:
    report = audit_contamination(
        [
            _record("ref-b", "shared benchmark query content exact"),
            _record("ref-a", "shared benchmark query content exact"),
        ],
        [_record("eval", "shared benchmark query content exact")],
    )

    assert [item.reference_id for item in report.matches] == ["ref-a", "ref-b"]
    json.dumps(report.as_dict(), sort_keys=True, allow_nan=False)


def test_input_order_does_not_change_audit_evidence() -> None:
    references = [
        _record("ref-b", "rotate expired oauth token for upstream service safely"),
        _record("ref-a", "another completely unrelated reference query"),
    ]
    evaluations = [
        _record("eval-b", "another independent retrieval benchmark case"),
        _record("eval-a", "rotate an expired oauth token for upstream service safely"),
    ]
    policy = ContaminationPolicy(near_similarity_threshold=0.75)

    forward = audit_contamination(references, evaluations, policy=policy)
    reversed_input = audit_contamination(
        list(reversed(references)),
        list(reversed(evaluations)),
        policy=policy,
    )

    assert forward.as_dict() == reversed_input.as_dict()
