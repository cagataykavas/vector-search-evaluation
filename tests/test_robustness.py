import json

import pytest

from search.robustness import PerturbationCase, RobustnessPolicy, evaluate_robustness

POLICY = RobustnessPolicy(
    k=3,
    min_queries=2,
    min_variants_per_query=2,
    min_mean_jaccard=0.5,
    min_mean_rank_overlap=0.5,
    min_relevant_survival=0.75,
    min_worst_query_jaccard=0.4,
)


def stable_cases():
    return [
        PerturbationCase(
            "q1", ("a", "b", "c"), (("a", "b", "d"), ("a", "c", "b")), frozenset({"a", "b"})
        ),
        PerturbationCase(
            "q2", ("x", "y", "z"), (("x", "z", "y"), ("x", "y", "w")), frozenset({"x"})
        ),
    ]


def test_stable_retrieval_passes_with_json_evidence():
    report = evaluate_robustness(stable_cases(), POLICY)
    assert report.passed
    assert report.queries == 2 and report.variants == 4
    assert report.relevant_survival == 1.0
    assert report.top1_retention == 1.0
    assert json.loads(json.dumps(report.to_dict()))["worst_query_id"] == "q1"


def test_fragile_retrieval_reports_all_relevant_failures():
    cases = stable_cases()
    cases[0] = PerturbationCase(
        "q1", ("a", "b", "c"), (("d", "e", "f"), ("g", "h", "i")), frozenset({"a", "b"})
    )
    report = evaluate_robustness(cases, POLICY)
    assert not report.passed
    assert "relevant_survival_below_floor" in report.reasons
    assert "worst_query_jaccard_below_floor" in report.reasons
    assert report.worst_query_id == "q1"


def test_results_are_deterministically_ordered():
    report = evaluate_robustness(list(reversed(stable_cases())), POLICY)
    assert [row.query_id for row in report.query_results] == ["q1", "q2"]


@pytest.mark.parametrize("cases", [stable_cases()[:1], [stable_cases()[0], stable_cases()[0]]])
def test_insufficient_or_duplicate_queries_fail_closed(cases):
    with pytest.raises(ValueError):
        evaluate_robustness(cases, POLICY)


def test_insufficient_variants_fail_closed():
    cases = stable_cases()
    cases[0] = PerturbationCase("q1", ("a", "b", "c"), (("a", "b", "c"),), frozenset({"a"}))
    with pytest.raises(ValueError, match="insufficient perturbation variants"):
        evaluate_robustness(cases, POLICY)


@pytest.mark.parametrize("ranking", [("a", "a", "c"), ("a", "b"), ("", "b", "c")])
def test_invalid_rankings_fail_closed(ranking):
    cases = stable_cases()
    cases[0] = PerturbationCase("q1", ranking, (("a", "b", "c"), ("a", "b", "c")), frozenset({"a"}))
    with pytest.raises(ValueError):
        evaluate_robustness(cases, POLICY)


def test_baseline_must_retrieve_relevant_evidence():
    cases = stable_cases()
    cases[0] = PerturbationCase(
        "q1", ("a", "b", "c"), (("a", "b", "c"), ("a", "b", "c")), frozenset({"missing"})
    )
    with pytest.raises(ValueError, match="no relevant document"):
        evaluate_robustness(cases, POLICY)


@pytest.mark.parametrize(
    "policy",
    [
        RobustnessPolicy(k=0),
        RobustnessPolicy(rank_persistence=0),
        RobustnessPolicy(min_mean_jaccard=1.1),
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(ValueError):
        evaluate_robustness(stable_cases(), policy)
