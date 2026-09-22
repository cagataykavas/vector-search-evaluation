import json
import math

import pytest

from search.serving_gate import (
    QueryMeasurement,
    ServingGatePolicy,
    evaluate_serving_candidate,
    main,
)


def row(
    query_id: str,
    ranking: tuple[str, ...] = ("relevant", "other"),
    latency: tuple[float, ...] = (10.0, 11.0, 12.0),
) -> QueryMeasurement:
    return QueryMeasurement(query_id, ranking, frozenset({"relevant"}), latency)


def policy(**overrides: object) -> ServingGatePolicy:
    values = {
        "k": 2,
        "min_queries": 2,
        "min_latency_samples_per_query": 3,
        "max_mean_ndcg_drop": 0.05,
        "max_p95_latency_ms": 30.0,
        "max_p95_latency_ratio": 1.5,
        "max_median_latency_ratio": 1.5,
    }
    values.update(overrides)
    return ServingGatePolicy(**values)


def test_quality_improvement_with_bounded_latency_passes() -> None:
    baseline = [row("q2", ("other", "relevant")), row("q1", ("other", "relevant"))]
    candidate = [row("q1", latency=(12.0, 13.0, 14.0)), row("q2")]

    report = evaluate_serving_candidate(baseline, candidate, policy())

    assert report.passed
    assert report.mean_ndcg_delta > 0
    assert [item.query_id for item in report.query_evidence] == ["q1", "q2"]
    assert report.to_dict()["reasons"] == []


def test_slower_and_worse_candidate_is_pareto_dominated() -> None:
    baseline = [row("q1"), row("q2")]
    candidate = [
        row("q1", ("other", "relevant"), (20.0, 21.0, 22.0)),
        row("q2", ("other", "relevant"), (20.0, 21.0, 22.0)),
    ]

    report = evaluate_serving_candidate(baseline, candidate, policy())

    assert not report.passed
    assert "candidate_pareto_dominated" in report.reasons
    assert "mean_ndcg_drop_exceeded" in report.reasons
    assert "p95_latency_ratio_exceeded" in report.reasons


def test_absolute_latency_budget_blocks_fast_relative_regression() -> None:
    baseline = [row("q1", latency=(20.0,) * 3), row("q2", latency=(20.0,) * 3)]
    candidate = [row("q1", latency=(24.0,) * 3), row("q2", latency=(24.0,) * 3)]

    report = evaluate_serving_candidate(
        baseline,
        candidate,
        policy(max_p95_latency_ms=22.0),
    )

    assert report.reasons == ("p95_latency_budget_exceeded",)


def test_insufficient_evidence_fails_closed() -> None:
    report = evaluate_serving_candidate(
        [row("q1", latency=(10.0,))],
        [row("q1", latency=(10.0,))],
        policy(),
    )

    assert report.reasons == ("insufficient_queries", "insufficient_latency_samples")


def test_query_sets_and_relevance_labels_must_be_paired() -> None:
    with pytest.raises(ValueError, match="same query ids"):
        evaluate_serving_candidate([row("q1")], [row("q2")], policy(min_queries=1))

    changed_qrels = QueryMeasurement("q1", ("new", "other"), frozenset({"new"}), (10.0,) * 3)
    with pytest.raises(ValueError, match="relevance labels differ"):
        evaluate_serving_candidate([row("q1")], [changed_qrels], policy(min_queries=1))


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        (row("q1", ranking=("relevant", "relevant")), "ranking has invalid"),
        (row("q1", ranking=("relevant",)), "shorter than k"),
        (row("q1", latency=(10.0, math.nan, 12.0)), "invalid latency"),
        (row("q1", latency=(10.0, "slow", 12.0)), "invalid latency"),
    ],
)
def test_malformed_measurements_are_rejected(measurement: QueryMeasurement, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_serving_candidate([measurement], [row("q1")], policy(min_queries=1))


def test_duplicate_query_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate baseline query_id"):
        evaluate_serving_candidate([row("q1"), row("q1")], [row("q1")], policy())


def test_invalid_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_p95_latency_ratio"):
        policy(max_p95_latency_ratio=0.0)


def test_cli_returns_distinct_policy_and_input_exit_codes(tmp_path, capsys) -> None:
    payload = {
        "policy": {"k": 2, "min_queries": 1, "max_p95_latency_ms": 5.0},
        "baseline": [
            {
                "query_id": "q1",
                "ranking": ["relevant", "other"],
                "relevant_doc_ids": ["relevant"],
                "latency_ms": [10.0, 10.0, 10.0],
            }
        ],
        "candidate": [
            {
                "query_id": "q1",
                "ranking": ["relevant", "other"],
                "relevant_doc_ids": ["relevant"],
                "latency_ms": [10.0, 10.0, 10.0],
            }
        ],
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(artifact), "--require-pass"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["reasons"] == ["p95_latency_budget_exceeded"]

    artifact.write_text("not-json", encoding="utf-8")
    assert main([str(artifact), "--require-pass"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
