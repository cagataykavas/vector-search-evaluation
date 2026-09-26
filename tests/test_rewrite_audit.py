from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from search.rewrite_audit import (
    ArtifactMalformed,
    RewritePolicy,
    audit_rewrites,
    load_artifact,
)

AS_OF = datetime(2026, 9, 26, 23, 0, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def rewrite(
    *,
    rewrite_id: str = "rewrite-1",
    text: str = "Find 2024 incidents for ERR-42 that are not tagged resolved",
    language: str = "en",
    scope_sha256: str = DIGEST_D,
) -> dict:
    return {
        "rewrite_id": rewrite_id,
        "text": text,
        "language": language,
        "scope_sha256": scope_sha256,
    }


def case(
    *,
    case_id: str = "case-1",
    original: str = 'Find ERR-42 incidents from 2024 not tagged "resolved"',
    language: str = "en",
    protected: list[dict] | None = None,
    allowed: list[str] | None = None,
    rewrites: list[dict] | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "original_query": original,
        "language": language,
        "scope_sha256": DIGEST_D,
        "protected_literals": (
            [
                {"kind": "identifier", "value": "ERR-42"},
                {"kind": "constraint", "value": "2024"},
                {"kind": "phrase", "value": "resolved"},
            ]
            if protected is None
            else protected
        ),
        "allowed_structured_additions": [] if allowed is None else allowed,
        "rewrites": [rewrite()] if rewrites is None else rewrites,
    }


def artifact(cases: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "benchmark_id": "rewrite-release-17",
        "rewriter_model_sha256": DIGEST_A,
        "prompt_sha256": DIGEST_B,
        "retrieval_policy_sha256": DIGEST_C,
        "created_at": "2026-09-26T22:30:00Z",
        "cases": [case()] if cases is None else cases,
    }


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_safe_rewrite_is_accepted() -> None:
    report = audit_rewrites(artifact(), as_of=AS_OF)
    assert report.accepted is True
    assert report.status == "accepted"
    assert report.case_count == 1
    assert report.rewrite_count == 1
    assert report.rejected_rewrite_count == 0


def test_semantically_equivalent_negation_marker_is_accepted() -> None:
    item = rewrite(text="Find ERR-42 incidents from 2024 without resolved tags")
    assert audit_rewrites(artifact([case(rewrites=[item])]), as_of=AS_OF).accepted


def test_noop_rewrite_is_safe_and_accepted() -> None:
    original = 'Find ERR-42 incidents from 2024 not tagged "resolved"'
    item = rewrite(text=original)
    assert audit_rewrites(
        artifact([case(original=original, rewrites=[item])]), as_of=AS_OF
    ).accepted


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (rewrite(text="Find 2024 incidents not tagged resolved"), "PROTECTED_LITERAL_DROPPED"),
        (
            rewrite(text="Find 2024 incidents for ERR-42 tagged resolved"),
            "NEGATION_CHANGED",
        ),
        (
            rewrite(text="Find 2024 and 2025 incidents for ERR-42 not tagged resolved"),
            "STRUCTURED_LITERAL_INTRODUCED",
        ),
        (rewrite(language="tr"), "LANGUAGE_CHANGED"),
        (rewrite(scope_sha256=DIGEST_A), "SCOPE_CHANGED"),
    ],
)
def test_contract_regressions_are_policy_rejections(changed: dict, expected: str) -> None:
    report = audit_rewrites(artifact([case(rewrites=[changed])]), as_of=AS_OF)
    assert report.accepted is False
    assert expected in codes(report)


def test_allowlisted_structured_addition_is_accepted() -> None:
    item = rewrite(text="Find 2024 and 2025 incidents for ERR-42 not tagged resolved")
    report = audit_rewrites(artifact([case(allowed=["2025"], rewrites=[item])]), as_of=AS_OF)
    assert report.accepted is True


def test_structured_literals_are_protected_even_without_manual_labels() -> None:
    item = case(
        protected=[],
        rewrites=[rewrite(text="Find incidents that are not tagged resolved")],
    )
    report = audit_rewrites(artifact([item]), as_of=AS_OF)
    assert "STRUCTURED_LITERAL_DROPPED" in codes(report)


def test_allowlisted_mixed_identifier_covers_its_numeric_component() -> None:
    item = rewrite(text="Find 2024 incidents for ERR-42 and CASE-99 not tagged resolved")
    report = audit_rewrites(artifact([case(allowed=["CASE-99"], rewrites=[item])]), as_of=AS_OF)
    assert report.accepted is True


def test_duplicate_normalized_rewrites_are_rejected() -> None:
    first = rewrite(rewrite_id="one")
    second = rewrite(
        rewrite_id="two",
        text="  FIND  2024 incidents for err-42 that are not tagged resolved  ",
    )
    report = audit_rewrites(artifact([case(rewrites=[first, second])]), as_of=AS_OF)
    assert "DUPLICATE_NORMALIZED_REWRITE" in codes(report)
    assert report.rejected_rewrite_count == 1


def test_token_and_growth_budgets_are_independent_findings() -> None:
    long_text = "ERR-42 2024 not resolved " + "context " * 20
    item = rewrite(text=long_text)
    policy = RewritePolicy(max_rewrite_tokens=10, max_growth_ratio=1.1)
    report = audit_rewrites(artifact([case(rewrites=[item])]), as_of=AS_OF, policy=policy)
    assert {"TOKEN_BUDGET_EXCEEDED", "QUERY_GROWTH_EXCEEDED"} <= codes(report)


def test_turkish_negation_is_preserved() -> None:
    item = case(
        original="ERR-42 için 2024 kayıtları çözülenler hariç bul",
        language="tr",
        protected=[
            {"kind": "identifier", "value": "ERR-42"},
            {"kind": "constraint", "value": "2024"},
        ],
        rewrites=[
            rewrite(
                text="2024 ERR-42 kayıtlarını çözülenlerin dışında bul",
                language="tr",
            )
        ],
    )
    assert audit_rewrites(artifact([item]), as_of=AS_OF).accepted


def test_turkish_negation_drop_is_rejected() -> None:
    item = case(
        original="ERR-42 için 2024 kayıtları çözülenler hariç bul",
        language="tr",
        protected=[{"kind": "identifier", "value": "ERR-42"}],
        rewrites=[rewrite(text="2024 ERR-42 kayıtlarını bul", language="tr")],
    )
    assert codes(audit_rewrites(artifact([item]), as_of=AS_OF)) == {"NEGATION_CHANGED"}


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.update(extra=True), "ROOT_FIELDS"),
        (lambda value: value["cases"][0].update(extra=True), "CASE_FIELDS"),
        (
            lambda value: value["cases"][0]["rewrites"][0].update(extra=True),
            "REWRITE_FIELDS",
        ),
        (lambda value: value.update(rewriter_model_sha256="bad"), "REWRITER_MODEL_SHA256"),
        (lambda value: value["cases"][0].update(language="de"), "ORIGINAL_LANGUAGE"),
        (lambda value: value["cases"][0].update(rewrites=[]), "REWRITES_REQUIRED"),
    ],
)
def test_unknown_or_malformed_fields_fail_closed(mutate, expected: str) -> None:
    value = artifact()
    mutate(value)
    with pytest.raises(ArtifactMalformed, match=expected):
        audit_rewrites(value, as_of=AS_OF)


def test_declared_protected_literal_must_exist_in_original() -> None:
    item = case(protected=[{"kind": "entity", "value": "secret-extra-term"}])
    with pytest.raises(ArtifactMalformed, match="PROTECTED_LITERAL_NOT_IN_ORIGINAL"):
        audit_rewrites(artifact([item]), as_of=AS_OF)


def test_unsafe_unicode_control_character_fails_closed() -> None:
    item = rewrite(text="ERR-42 2024 not resolved\u202e")
    with pytest.raises(ArtifactMalformed, match="UNSAFE_QUERY_CHARACTER"):
        audit_rewrites(artifact([case(rewrites=[item])]), as_of=AS_OF)


def test_duplicate_json_keys_non_finite_and_byte_budget_fail_closed() -> None:
    with pytest.raises(ArtifactMalformed, match="DUPLICATE_JSON_KEY"):
        load_artifact(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ArtifactMalformed, match="NON_FINITE_NUMBER"):
        load_artifact(b'{"value":NaN}')
    with pytest.raises(ArtifactMalformed, match="INPUT_TOO_LARGE"):
        load_artifact(b"{} ", RewritePolicy(max_input_bytes=2))


def test_case_rewrite_and_total_character_budgets_fail_closed() -> None:
    with pytest.raises(ArtifactMalformed, match="CASE_BUDGET_EXCEEDED"):
        audit_rewrites(
            artifact([case(case_id="one"), case(case_id="two")]),
            as_of=AS_OF,
            policy=RewritePolicy(max_cases=1),
        )
    with pytest.raises(ArtifactMalformed, match="REWRITE_BUDGET_EXCEEDED"):
        audit_rewrites(
            artifact(
                [
                    case(
                        rewrites=[
                            rewrite(rewrite_id="one"),
                            rewrite(rewrite_id="two", text="ERR-42 2024 not resolved"),
                        ]
                    )
                ]
            ),
            as_of=AS_OF,
            policy=RewritePolicy(max_rewrites_per_case=1),
        )
    with pytest.raises(ArtifactMalformed, match="TOTAL_QUERY_BUDGET_EXCEEDED"):
        audit_rewrites(
            artifact(),
            as_of=AS_OF,
            policy=RewritePolicy(max_total_query_chars=10),
        )


def test_stale_and_future_artifacts_fail_closed() -> None:
    stale = artifact()
    stale["created_at"] = "2026-09-20T22:30:00Z"
    with pytest.raises(ArtifactMalformed, match="STALE_EVIDENCE"):
        audit_rewrites(stale, as_of=AS_OF)
    future = artifact()
    future["created_at"] = "2026-09-26T23:02:00Z"
    with pytest.raises(ArtifactMalformed, match="EVIDENCE_FROM_FUTURE"):
        audit_rewrites(future, as_of=AS_OF)


def test_canonical_digest_is_independent_of_case_and_rewrite_order() -> None:
    first = case(case_id="a")
    second = case(
        case_id="b",
        rewrites=[
            rewrite(rewrite_id="two", text="ERR-42 2024 not resolved"),
            rewrite(rewrite_id="one"),
        ],
    )
    first_report = audit_rewrites(artifact([first, second]), as_of=AS_OF)
    reordered = artifact([deepcopy(second), deepcopy(first)])
    reordered["cases"][0]["rewrites"].reverse()
    second_report = audit_rewrites(reordered, as_of=AS_OF)
    assert first_report.artifact_sha256 == second_report.artifact_sha256
    assert first_report.findings == second_report.findings


def test_report_does_not_expose_queries_literals_or_ids() -> None:
    report = audit_rewrites(
        artifact([case(rewrites=[rewrite(text="ERR-42 2024 resolved")])]),
        as_of=AS_OF,
    )
    serialized = json.dumps(report.to_dict())
    assert "ERR-42" not in serialized
    assert "resolved" not in serialized
    assert "case-1" not in serialized
    assert "rewrite-1" not in serialized


def test_finding_report_is_bounded_but_total_count_is_preserved() -> None:
    rewrites = [
        rewrite(rewrite_id=f"rewrite-{index}", text=f"unrelated {index + 10_000}")
        for index in range(8)
    ]
    report = audit_rewrites(
        artifact([case(rewrites=rewrites)]),
        as_of=AS_OF,
        policy=RewritePolicy(max_findings=3),
    )
    assert report.finding_count > 3
    assert len(report.findings) == 3
    assert report.findings_truncated is True


def test_thousand_case_maximum_is_processed_deterministically() -> None:
    cases = [case(case_id=f"case-{index}") for index in range(1_000)]
    report = audit_rewrites(artifact(cases), as_of=AS_OF)
    assert report.accepted is True
    assert report.case_count == 1_000
    assert report.rewrite_count == 1_000


def test_cli_exit_codes_and_atomic_output(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    artifact_path.write_text(json.dumps(artifact()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "search.rewrite_audit",
        str(artifact_path),
        "--as-of",
        "2026-09-26T23:00:00Z",
        "--output",
        str(output_path),
    ]
    accepted = subprocess.run(command, check=False, capture_output=True, text=True)
    assert accepted.returncode == 0
    assert json.loads(output_path.read_text())["status"] == "accepted"

    rejected = artifact([case(rewrites=[rewrite(text="ERR-42 2024 resolved")])])
    artifact_path.write_text(json.dumps(rejected), encoding="utf-8")
    policy_rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert policy_rejected.returncode == 2
    assert json.loads(output_path.read_text())["status"] == "policy_rejected"

    artifact_path.write_text("{", encoding="utf-8")
    malformed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert malformed.returncode == 3
    assert json.loads(output_path.read_text())["status"] == "malformed"


def test_policy_digest_covers_every_threshold() -> None:
    default = audit_rewrites(artifact(), as_of=AS_OF)
    changed = audit_rewrites(
        artifact(),
        as_of=AS_OF,
        policy=RewritePolicy(max_growth_ratio=4.0),
    )
    assert default.policy_sha256 != changed.policy_sha256
