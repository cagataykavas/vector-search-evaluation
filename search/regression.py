"""Query-level release gate for retrieval ranking changes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from evaluate import ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class QueryRegression:
    """Paired retrieval evidence for one query."""

    query_id: str
    baseline_recall: float
    candidate_recall: float
    recall_delta: float
    baseline_mrr: float
    candidate_mrr: float
    mrr_delta: float
    baseline_ndcg: float
    candidate_ndcg: float
    ndcg_delta: float
    regressed: bool


@dataclass(frozen=True)
class RetrievalGateReport:
    """JSON-ready release decision over matched query cases."""

    queries: int
    k: int
    mean_recall_delta: float
    mean_mrr_delta: float
    mean_ndcg_delta: float
    regressed_queries: int
    regression_rate: float
    max_regression_rate: float
    minimum_mean_ndcg_delta: float
    regression_tolerance: float
    passed: bool
    reasons: tuple[str, ...]
    query_results: tuple[QueryRegression, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_ranking(query_id: str, ranking: Sequence[str], name: str) -> list[str]:
    values = list(ranking)
    if not values:
        raise ValueError(f"{name} ranking for {query_id!r} cannot be empty")
    if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in values):
        raise ValueError(f"{name} ranking for {query_id!r} contains an invalid document id")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} ranking for {query_id!r} contains duplicate document ids")
    return values


def compare_retrieval_runs(
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str] | frozenset[str]],
    *,
    k: int = 10,
    max_regression_rate: float = 0.1,
    minimum_mean_ndcg_delta: float = 0.0,
    regression_tolerance: float = 0.0,
) -> RetrievalGateReport:
    """Compare matched rankings and reject hidden query-level regressions."""

    if k < 1:
        raise ValueError("k must be at least one")
    if not 0.0 <= max_regression_rate <= 1.0:
        raise ValueError("max_regression_rate must be between zero and one")
    if not -1.0 <= minimum_mean_ndcg_delta <= 1.0:
        raise ValueError("minimum_mean_ndcg_delta must be between minus one and one")
    if not 0.0 <= regression_tolerance <= 1.0:
        raise ValueError("regression_tolerance must be between zero and one")
    if not qrels:
        raise ValueError("qrels cannot be empty")

    expected_ids = set(qrels)
    if set(baseline) != expected_ids or set(candidate) != expected_ids:
        raise ValueError("baseline, candidate and qrels must contain identical query ids")

    query_results: list[QueryRegression] = []
    for query_id in sorted(expected_ids):
        relevant = set(qrels[query_id])
        if not relevant or any(
            not isinstance(doc_id, str) or not doc_id.strip() for doc_id in relevant
        ):
            raise ValueError(f"qrels for {query_id!r} must contain valid document ids")

        baseline_ranking = _validate_ranking(query_id, baseline[query_id], "baseline")
        candidate_ranking = _validate_ranking(query_id, candidate[query_id], "candidate")
        baseline_recall = recall_at_k(baseline_ranking, relevant, k)
        candidate_recall = recall_at_k(candidate_ranking, relevant, k)
        baseline_mrr = reciprocal_rank(baseline_ranking, relevant)
        candidate_mrr = reciprocal_rank(candidate_ranking, relevant)
        baseline_ndcg = ndcg_at_k(baseline_ranking, relevant, k)
        candidate_ndcg = ndcg_at_k(candidate_ranking, relevant, k)
        ndcg_delta = candidate_ndcg - baseline_ndcg

        query_results.append(
            QueryRegression(
                query_id=query_id,
                baseline_recall=baseline_recall,
                candidate_recall=candidate_recall,
                recall_delta=candidate_recall - baseline_recall,
                baseline_mrr=baseline_mrr,
                candidate_mrr=candidate_mrr,
                mrr_delta=candidate_mrr - baseline_mrr,
                baseline_ndcg=baseline_ndcg,
                candidate_ndcg=candidate_ndcg,
                ndcg_delta=ndcg_delta,
                regressed=ndcg_delta < -regression_tolerance,
            )
        )

    mean_recall_delta = mean(result.recall_delta for result in query_results)
    mean_mrr_delta = mean(result.mrr_delta for result in query_results)
    mean_ndcg_delta = mean(result.ndcg_delta for result in query_results)
    regressed_queries = sum(result.regressed for result in query_results)
    regression_rate = regressed_queries / len(query_results)
    reasons: list[str] = []
    if mean_ndcg_delta < minimum_mean_ndcg_delta:
        reasons.append("insufficient_mean_ndcg_delta")
    if regression_rate > max_regression_rate:
        reasons.append("query_regression_rate_exceeded")

    return RetrievalGateReport(
        queries=len(query_results),
        k=k,
        mean_recall_delta=mean_recall_delta,
        mean_mrr_delta=mean_mrr_delta,
        mean_ndcg_delta=mean_ndcg_delta,
        regressed_queries=regressed_queries,
        regression_rate=regression_rate,
        max_regression_rate=max_regression_rate,
        minimum_mean_ndcg_delta=minimum_mean_ndcg_delta,
        regression_tolerance=regression_tolerance,
        passed=not reasons,
        reasons=tuple(reasons),
        query_results=tuple(query_results),
    )
