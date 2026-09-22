from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from evaluate import ndcg_at_k


@dataclass(frozen=True)
class QueryMeasurement:
    query_id: str
    ranking: tuple[str, ...]
    relevant_doc_ids: frozenset[str]
    latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class ServingGatePolicy:
    k: int = 10
    min_queries: int = 20
    min_latency_samples_per_query: int = 3
    max_mean_ndcg_drop: float = 0.02
    max_p95_latency_ms: float = 250.0
    max_p95_latency_ratio: float = 1.25
    max_median_latency_ratio: float = 1.15

    def __post_init__(self) -> None:
        if self.k < 1 or self.min_queries < 1 or self.min_latency_samples_per_query < 1:
            raise ValueError("k and evidence minimums must be positive")
        if not math.isfinite(self.max_mean_ndcg_drop) or not 0 <= self.max_mean_ndcg_drop <= 1:
            raise ValueError("max_mean_ndcg_drop must be finite and between 0 and 1")
        for name in (
            "max_p95_latency_ms",
            "max_p95_latency_ratio",
            "max_median_latency_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RunSummary:
    mean_ndcg_at_k: float
    median_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class QueryEvidence:
    query_id: str
    baseline_ndcg_at_k: float
    candidate_ndcg_at_k: float
    baseline_median_latency_ms: float
    candidate_median_latency_ms: float


@dataclass(frozen=True)
class ServingGateReport:
    passed: bool
    reasons: tuple[str, ...]
    query_count: int
    k: int
    baseline: RunSummary
    candidate: RunSummary
    mean_ndcg_delta: float
    p95_latency_ratio: float
    median_latency_ratio: float
    query_evidence: tuple[QueryEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "query_count": self.query_count,
            "k": self.k,
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "mean_ndcg_delta": self.mean_ndcg_delta,
            "p95_latency_ratio": self.p95_latency_ratio,
            "median_latency_ratio": self.median_latency_ratio,
            "query_evidence": [asdict(row) for row in self.query_evidence],
        }


def _validate_run(
    rows: list[QueryMeasurement], *, k: int, label: str
) -> dict[str, QueryMeasurement]:
    indexed: dict[str, QueryMeasurement] = {}
    for row in rows:
        if not isinstance(row, QueryMeasurement):
            raise ValueError(f"{label} rows must be QueryMeasurement instances")
        if (
            not isinstance(row.query_id, str)
            or not row.query_id
            or row.query_id != row.query_id.strip()
        ):
            raise ValueError(f"{label} query_id must be non-empty and trimmed")
        if row.query_id in indexed:
            raise ValueError(f"duplicate {label} query_id: {row.query_id}")
        if not row.relevant_doc_ids or any(
            not isinstance(doc_id, str) or not doc_id for doc_id in row.relevant_doc_ids
        ):
            raise ValueError(f"{label} query {row.query_id} has invalid relevance labels")
        if len(row.ranking) < k:
            raise ValueError(f"{label} query {row.query_id} ranking is shorter than k")
        if len(set(row.ranking)) != len(row.ranking) or any(
            not isinstance(doc_id, str) or not doc_id for doc_id in row.ranking
        ):
            raise ValueError(f"{label} query {row.query_id} ranking has invalid document ids")
        if not row.latency_ms:
            raise ValueError(f"{label} query {row.query_id} has no latency samples")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in row.latency_ms
        ):
            raise ValueError(f"{label} query {row.query_id} has invalid latency samples")
        indexed[row.query_id] = row
    return indexed


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(rows: list[QueryMeasurement], *, k: int) -> RunSummary:
    quality = [ndcg_at_k(list(row.ranking), set(row.relevant_doc_ids), k) for row in rows]
    query_medians = [median(row.latency_ms) for row in rows]
    return RunSummary(mean(quality), median(query_medians), _p95(query_medians))


def evaluate_serving_candidate(
    baseline: list[QueryMeasurement],
    candidate: list[QueryMeasurement],
    policy: ServingGatePolicy | None = None,
) -> ServingGateReport:
    policy = policy or ServingGatePolicy()
    baseline_by_id = _validate_run(baseline, k=policy.k, label="baseline")
    candidate_by_id = _validate_run(candidate, k=policy.k, label="candidate")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate must contain the same query ids")

    query_ids = sorted(baseline_by_id)
    for query_id in query_ids:
        if baseline_by_id[query_id].relevant_doc_ids != candidate_by_id[query_id].relevant_doc_ids:
            raise ValueError(f"relevance labels differ for query {query_id}")

    baseline_rows = [baseline_by_id[query_id] for query_id in query_ids]
    candidate_rows = [candidate_by_id[query_id] for query_id in query_ids]
    baseline_summary = _summary(baseline_rows, k=policy.k)
    candidate_summary = _summary(candidate_rows, k=policy.k)
    ndcg_delta = candidate_summary.mean_ndcg_at_k - baseline_summary.mean_ndcg_at_k
    p95_ratio = candidate_summary.p95_latency_ms / baseline_summary.p95_latency_ms
    median_ratio = candidate_summary.median_latency_ms / baseline_summary.median_latency_ms

    reasons: list[str] = []
    if len(query_ids) < policy.min_queries:
        reasons.append("insufficient_queries")
    if any(
        len(row.latency_ms) < policy.min_latency_samples_per_query
        for row in baseline_rows + candidate_rows
    ):
        reasons.append("insufficient_latency_samples")
    if ndcg_delta < -policy.max_mean_ndcg_drop:
        reasons.append("mean_ndcg_drop_exceeded")
    if candidate_summary.p95_latency_ms > policy.max_p95_latency_ms:
        reasons.append("p95_latency_budget_exceeded")
    if p95_ratio > policy.max_p95_latency_ratio:
        reasons.append("p95_latency_ratio_exceeded")
    if median_ratio > policy.max_median_latency_ratio:
        reasons.append("median_latency_ratio_exceeded")
    if ndcg_delta < 0 and candidate_summary.p95_latency_ms > baseline_summary.p95_latency_ms:
        reasons.append("candidate_pareto_dominated")

    evidence = tuple(
        QueryEvidence(
            query_id=query_id,
            baseline_ndcg_at_k=ndcg_at_k(
                list(baseline_by_id[query_id].ranking),
                set(baseline_by_id[query_id].relevant_doc_ids),
                policy.k,
            ),
            candidate_ndcg_at_k=ndcg_at_k(
                list(candidate_by_id[query_id].ranking),
                set(candidate_by_id[query_id].relevant_doc_ids),
                policy.k,
            ),
            baseline_median_latency_ms=median(baseline_by_id[query_id].latency_ms),
            candidate_median_latency_ms=median(candidate_by_id[query_id].latency_ms),
        )
        for query_id in query_ids
    )
    return ServingGateReport(
        passed=not reasons,
        reasons=tuple(reasons),
        query_count=len(query_ids),
        k=policy.k,
        baseline=baseline_summary,
        candidate=candidate_summary,
        mean_ndcg_delta=ndcg_delta,
        p95_latency_ratio=p95_ratio,
        median_latency_ratio=median_ratio,
        query_evidence=evidence,
    )


def _measurements(rows: list[dict[str, Any]]) -> list[QueryMeasurement]:
    return [
        QueryMeasurement(
            query_id=row["query_id"],
            ranking=tuple(row["ranking"]),
            relevant_doc_ids=frozenset(row["relevant_doc_ids"]),
            latency_ms=tuple(row["latency_ms"]),
        )
        for row in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate a retrieval candidate on quality and latency"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        report = evaluate_serving_candidate(
            _measurements(payload["baseline"]),
            _measurements(payload["candidate"]),
            ServingGatePolicy(**payload.get("policy", {})),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 2 if args.require_pass and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
