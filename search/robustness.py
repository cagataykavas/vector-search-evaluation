"""Evaluate retrieval stability across controlled query perturbations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean


@dataclass(frozen=True)
class PerturbationCase:
    query_id: str
    baseline: tuple[str, ...]
    variants: tuple[tuple[str, ...], ...]
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class RobustnessPolicy:
    k: int = 10
    min_queries: int = 5
    min_variants_per_query: int = 2
    min_mean_jaccard: float = 0.60
    min_mean_rank_overlap: float = 0.65
    min_relevant_survival: float = 0.80
    min_worst_query_jaccard: float = 0.40
    rank_persistence: float = 0.90


DEFAULT_POLICY = RobustnessPolicy()


@dataclass(frozen=True)
class QueryRobustness:
    query_id: str
    variants: int
    mean_jaccard: float
    mean_rank_overlap: float
    relevant_survival: float
    top1_retention: float


@dataclass(frozen=True)
class RobustnessReport:
    passed: bool
    reasons: tuple[str, ...]
    queries: int
    variants: int
    mean_jaccard: float
    mean_rank_overlap: float
    relevant_survival: float
    top1_retention: float
    worst_query_id: str
    worst_query_jaccard: float
    query_results: tuple[QueryRobustness, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["query_results"] = [asdict(row) for row in self.query_results]
        return result


def evaluate_robustness(
    cases: list[PerturbationCase], policy: RobustnessPolicy = DEFAULT_POLICY
) -> RobustnessReport:
    _validate_policy(policy)
    _validate_cases(cases, policy)
    results = tuple(
        _evaluate_query(case, policy) for case in sorted(cases, key=lambda row: row.query_id)
    )
    total_variants = sum(row.variants for row in results)
    mean_jaccard = _weighted_mean(results, "mean_jaccard")
    mean_rank_overlap = _weighted_mean(results, "mean_rank_overlap")
    relevant_survival = _weighted_mean(results, "relevant_survival")
    top1_retention = _weighted_mean(results, "top1_retention")
    worst = min(results, key=lambda row: (row.mean_jaccard, row.query_id))
    reasons: list[str] = []
    if mean_jaccard < policy.min_mean_jaccard:
        reasons.append("mean_jaccard_below_floor")
    if mean_rank_overlap < policy.min_mean_rank_overlap:
        reasons.append("mean_rank_overlap_below_floor")
    if relevant_survival < policy.min_relevant_survival:
        reasons.append("relevant_survival_below_floor")
    if worst.mean_jaccard < policy.min_worst_query_jaccard:
        reasons.append("worst_query_jaccard_below_floor")
    return RobustnessReport(
        passed=not reasons,
        reasons=tuple(reasons),
        queries=len(results),
        variants=total_variants,
        mean_jaccard=mean_jaccard,
        mean_rank_overlap=mean_rank_overlap,
        relevant_survival=relevant_survival,
        top1_retention=top1_retention,
        worst_query_id=worst.query_id,
        worst_query_jaccard=worst.mean_jaccard,
        query_results=results,
    )


def _evaluate_query(case: PerturbationCase, policy: RobustnessPolicy) -> QueryRobustness:
    baseline = case.baseline[: policy.k]
    baseline_set = set(baseline)
    baseline_relevant = baseline_set & case.relevant_doc_ids
    jaccards, overlaps, survival, top1 = [], [], [], []
    for ranking in case.variants:
        candidate = ranking[: policy.k]
        candidate_set = set(candidate)
        union = baseline_set | candidate_set
        jaccards.append(len(baseline_set & candidate_set) / len(union))
        overlaps.append(_rank_biased_overlap(baseline, candidate, policy.rank_persistence))
        survival.append(len(baseline_relevant & candidate_set) / len(baseline_relevant))
        top1.append(float(candidate[0] == baseline[0]))
    return QueryRobustness(
        case.query_id,
        len(case.variants),
        mean(jaccards),
        mean(overlaps),
        mean(survival),
        mean(top1),
    )


def _rank_biased_overlap(
    left: tuple[str, ...], right: tuple[str, ...], persistence: float
) -> float:
    score = 0.0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    depth = min(len(left), len(right))
    for index in range(depth):
        left_seen.add(left[index])
        right_seen.add(right[index])
        score += (1 - persistence) * persistence**index * len(left_seen & right_seen) / (index + 1)
    return score + persistence**depth * len(left_seen & right_seen) / depth


def _weighted_mean(results: tuple[QueryRobustness, ...], field: str) -> float:
    return sum(getattr(row, field) * row.variants for row in results) / sum(
        row.variants for row in results
    )


def _validate_policy(policy: RobustnessPolicy) -> None:
    if policy.k < 1 or policy.min_queries < 1 or policy.min_variants_per_query < 1:
        raise ValueError("k and evidence minimums must be positive")
    for name in (
        "min_mean_jaccard",
        "min_mean_rank_overlap",
        "min_relevant_survival",
        "min_worst_query_jaccard",
        "rank_persistence",
    ):
        value = getattr(policy, name)
        if not isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")


def _validate_cases(cases: list[PerturbationCase], policy: RobustnessPolicy) -> None:
    if len(cases) < policy.min_queries:
        raise ValueError(f"at least {policy.min_queries} queries are required")
    query_ids: set[str] = set()
    for case in cases:
        if not case.query_id or case.query_id in query_ids:
            raise ValueError("query IDs must be non-empty and unique")
        query_ids.add(case.query_id)
        if len(case.variants) < policy.min_variants_per_query:
            raise ValueError(f"{case.query_id}: insufficient perturbation variants")
        if not case.relevant_doc_ids:
            raise ValueError(f"{case.query_id}: relevance set must not be empty")
        _validate_ranking(case.query_id, "baseline", case.baseline, policy.k)
        if not set(case.baseline[: policy.k]) & case.relevant_doc_ids:
            raise ValueError(f"{case.query_id}: baseline top-k has no relevant document")
        for index, ranking in enumerate(case.variants):
            _validate_ranking(case.query_id, f"variant {index}", ranking, policy.k)


def _validate_ranking(query_id: str, label: str, ranking: tuple[str, ...], k: int) -> None:
    if len(ranking) < k:
        raise ValueError(f"{query_id}: {label} ranking shorter than k")
    if any(not item for item in ranking) or len(set(ranking)) != len(ranking):
        raise ValueError(f"{query_id}: {label} has blank or duplicate document IDs")
