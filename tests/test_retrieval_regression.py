import json

import pytest

from search.regression import compare_retrieval_runs


def test_gate_finds_query_regression_hidden_by_unchanged_mean() -> None:
    baseline = {
        "q-1": ["relevant-1", "other-1"],
        "q-2": ["other-2", "relevant-2"],
    }
    candidate = {
        "q-1": ["other-1", "relevant-1"],
        "q-2": ["relevant-2", "other-2"],
    }
    qrels = {"q-1": {"relevant-1"}, "q-2": {"relevant-2"}}

    report = compare_retrieval_runs(baseline, candidate, qrels, k=2)

    assert report.mean_ndcg_delta == pytest.approx(0.0)
    assert report.regressed_queries == 1
    assert report.regression_rate == pytest.approx(0.5)
    assert report.passed is False
    assert report.reasons == ("query_regression_rate_exceeded",)


def test_gate_passes_improvement_and_sorts_query_evidence() -> None:
    baseline = {"z": ["other", "target-z"], "a": ["other", "target-a"]}
    candidate = {"z": ["target-z", "other"], "a": ["target-a", "other"]}
    qrels = {"z": {"target-z"}, "a": {"target-a"}}

    report = compare_retrieval_runs(
        baseline,
        candidate,
        qrels,
        k=2,
        minimum_mean_ndcg_delta=0.1,
    )

    assert report.passed is True
    assert report.reasons == ()
    assert report.mean_recall_delta == pytest.approx(0.0)
    assert report.mean_mrr_delta > 0
    assert report.mean_ndcg_delta > 0
    assert [row.query_id for row in report.query_results] == ["a", "z"]


def test_small_regression_can_be_tolerated_explicitly() -> None:
    baseline = {"q": ["target", "other"]}
    candidate = {"q": ["other", "target"]}
    qrels = {"q": {"target"}}

    report = compare_retrieval_runs(
        baseline,
        candidate,
        qrels,
        k=2,
        regression_tolerance=0.5,
        minimum_mean_ndcg_delta=-1.0,
    )

    assert report.regressed_queries == 0
    assert report.passed is True


def test_report_is_json_ready() -> None:
    values = {"q": ["target"]}

    report = compare_retrieval_runs(values, values, {"q": {"target"}})
    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["passed"] is True
    assert payload["query_results"][0]["query_id"] == "q"
    assert payload["query_results"][0]["ndcg_delta"] == pytest.approx(0.0)


def test_mismatched_query_sets_fail_closed() -> None:
    with pytest.raises(ValueError, match="identical query ids"):
        compare_retrieval_runs(
            {"q-1": ["doc"]},
            {"q-2": ["doc"]},
            {"q-1": {"doc"}},
        )


@pytest.mark.parametrize(
    ("baseline", "candidate", "qrels", "message"),
    [
        ({"q": ["doc", "doc"]}, {"q": ["doc"]}, {"q": {"doc"}}, "duplicate"),
        ({"q": ["doc"]}, {"q": []}, {"q": {"doc"}}, "cannot be empty"),
        ({"q": ["doc"]}, {"q": ["doc"]}, {"q": set()}, "valid document ids"),
    ],
)
def test_invalid_ranking_evidence_is_rejected(
    baseline: dict[str, list[str]],
    candidate: dict[str, list[str]],
    qrels: dict[str, set[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_retrieval_runs(baseline, candidate, qrels)


def test_policy_configuration_is_validated() -> None:
    values = {"q": ["doc"]}
    qrels = {"q": {"doc"}}

    with pytest.raises(ValueError, match="k"):
        compare_retrieval_runs(values, values, qrels, k=0)
    with pytest.raises(ValueError, match="max_regression_rate"):
        compare_retrieval_runs(values, values, qrels, max_regression_rate=1.1)
    with pytest.raises(ValueError, match="regression_tolerance"):
        compare_retrieval_runs(values, values, qrels, regression_tolerance=-0.1)
