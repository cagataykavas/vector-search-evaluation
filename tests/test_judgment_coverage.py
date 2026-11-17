import json

import pytest

from search.judgment_coverage import (
    InvalidJudgmentEvidence,
    JudgmentCoverageCase,
    JudgmentCoveragePolicy,
    audit_judgment_coverage,
    main,
)


def case(query_id: str, judged: set[str]) -> JudgmentCoverageCase:
    return JudgmentCoverageCase(
        query_id=query_id,
        ranking=("a", "b", "c", "d"),
        judged_doc_ids=frozenset(judged),
    )


def test_accepts_well_covered_rankings_and_binds_evidence() -> None:
    policy = JudgmentCoveragePolicy(
        k=4,
        min_mean_judged_fraction=0.75,
        low_coverage_threshold=0.5,
        max_low_coverage_rate=0.0,
    )

    first = audit_judgment_coverage(
        [case("q-2", {"a", "b", "c"}), case("q-1", {"a", "b", "c", "d"})],
        policy,
    )
    reordered = audit_judgment_coverage(
        [case("q-1", {"d", "c", "b", "a"}), case("q-2", {"c", "b", "a"})],
        policy,
    )

    assert first.accepted
    assert first.mean_judged_fraction_at_k == 0.875
    assert first.minimum_judged_fraction_at_k == 0.75
    assert first.evidence_sha256 == reordered.evidence_sha256
    assert all(query.query_ref not in {"q-1", "q-2"} for query in first.queries)


def test_rejects_shallow_judgment_pool_without_calling_unjudged_irrelevant() -> None:
    report = audit_judgment_coverage(
        [case("q-1", {"a"}), case("q-2", {"a", "b", "c", "d"})],
        JudgmentCoveragePolicy(
            k=4,
            min_mean_judged_fraction=0.8,
            low_coverage_threshold=0.5,
            max_low_coverage_rate=0.0,
        ),
    )

    assert not report.accepted
    assert report.reasons == (
        "mean_judgment_coverage_below_minimum",
        "low_coverage_query_rate_above_maximum",
    )
    assert report.low_coverage_query_count == 1
    assert report.queries[0].unjudged_at_k == 3


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        ([], "at least one"),
        ([case("q", {"a"}), case("q", {"a"})], "duplicate query_id"),
        (
            [JudgmentCoverageCase("q", ("a", "a", "b", "c"), frozenset({"a"}))],
            "duplicate documents",
        ),
        ([JudgmentCoverageCase("q", ("a",), frozenset({"a"}))], "at least policy.k"),
        ([JudgmentCoverageCase("q", ("a", "b", "c", "d"), frozenset())], "judged"),
    ],
)
def test_fails_closed_on_invalid_evidence(cases: list[JudgmentCoverageCase], message: str) -> None:
    with pytest.raises(InvalidJudgmentEvidence, match=message):
        audit_judgment_coverage(cases, JudgmentCoveragePolicy(k=4))


def test_cli_has_distinct_policy_and_invalid_exit_codes(tmp_path) -> None:
    payload = {
        "queries": [
            {
                "query_id": "q-1",
                "ranking": ["a", "b"],
                "judged_doc_ids": ["a"],
            }
        ],
        "policy": {
            "k": 2,
            "min_mean_judged_fraction": 1.0,
            "low_coverage_threshold": 1.0,
            "max_low_coverage_rate": 0.0,
        },
    }
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(input_path), "--output", str(output_path)]) == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert not report["accepted"]

    input_path.write_text('{"queries": [], "queries": [], "policy": {}}', encoding="utf-8")
    assert main([str(input_path)]) == 3


def test_cli_rejects_duplicate_and_non_string_judgment_ids(tmp_path) -> None:
    policy = {
        "k": 1,
        "min_mean_judged_fraction": 1.0,
        "low_coverage_threshold": 1.0,
        "max_low_coverage_rate": 0.0,
    }
    input_path = tmp_path / "input.json"

    for judged_doc_ids in (["a", "a"], [1]):
        input_path.write_text(
            json.dumps(
                {
                    "queries": [
                        {
                            "query_id": "q",
                            "ranking": ["a"],
                            "judged_doc_ids": judged_doc_ids,
                        }
                    ],
                    "policy": policy,
                }
            ),
            encoding="utf-8",
        )
        assert main([str(input_path)]) == 3


def test_invalid_query_id_type_fails_closed() -> None:
    with pytest.raises(InvalidJudgmentEvidence, match="query_id"):
        audit_judgment_coverage(
            [
                JudgmentCoverageCase(  # type: ignore[arg-type]
                    7, ("a",), frozenset({"a"})
                )
            ],
            JudgmentCoveragePolicy(k=1),
        )


def test_rejects_non_finite_policy() -> None:
    with pytest.raises(InvalidJudgmentEvidence, match="finite"):
        audit_judgment_coverage(
            [case("q", {"a", "b", "c", "d"})],
            JudgmentCoveragePolicy(k=4, min_mean_judged_fraction=float("nan")),
        )
