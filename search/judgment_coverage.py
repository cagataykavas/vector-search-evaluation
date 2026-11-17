from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_QUERIES = 10_000
MAX_RANKING_LENGTH = 1_000
MAX_IDENTIFIER_LENGTH = 200


class InvalidJudgmentEvidence(ValueError):
    """Raised when a judgment-coverage artifact cannot be trusted."""


@dataclass(frozen=True)
class JudgmentCoverageCase:
    query_id: str
    ranking: tuple[str, ...]
    judged_doc_ids: frozenset[str]


@dataclass(frozen=True)
class JudgmentCoveragePolicy:
    k: int = 10
    min_mean_judged_fraction: float = 0.8
    low_coverage_threshold: float = 0.5
    max_low_coverage_rate: float = 0.1


@dataclass(frozen=True)
class QueryCoverage:
    query_ref: str
    judged_at_k: int
    unjudged_at_k: int
    judged_fraction_at_k: float


@dataclass(frozen=True)
class JudgmentCoverageReport:
    schema_version: int
    accepted: bool
    reasons: tuple[str, ...]
    query_count: int
    k: int
    mean_judged_fraction_at_k: float
    minimum_judged_fraction_at_k: float
    low_coverage_query_count: int
    low_coverage_query_rate: float
    evidence_sha256: str
    queries: tuple[QueryCoverage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reasons": list(self.reasons),
            "queries": [asdict(query) for query in self.queries],
        }


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidJudgmentEvidence(f"{field} must be a non-blank string")
    if value != value.strip():
        raise InvalidJudgmentEvidence(f"{field} must not have surrounding whitespace")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise InvalidJudgmentEvidence(f"{field} must be at most {MAX_IDENTIFIER_LENGTH} characters")


def _validate_policy(policy: JudgmentCoveragePolicy) -> None:
    if isinstance(policy.k, bool) or not isinstance(policy.k, int) or not 1 <= policy.k <= 100:
        raise InvalidJudgmentEvidence("policy.k must be an integer from 1 through 100")
    for field in (
        "min_mean_judged_fraction",
        "low_coverage_threshold",
        "max_low_coverage_rate",
    ):
        value = getattr(policy, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidJudgmentEvidence(f"policy.{field} must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidJudgmentEvidence(f"policy.{field} must be finite and between 0 and 1")


def _query_ref(query_id: str) -> str:
    return hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16]


def audit_judgment_coverage(
    cases: Iterable[JudgmentCoverageCase],
    policy: JudgmentCoveragePolicy = JudgmentCoveragePolicy(),
) -> JudgmentCoverageReport:
    """Fail closed when top-K rankings are insufficiently covered by explicit judgments."""

    _validate_policy(policy)
    rows = list(cases)
    if not rows:
        raise InvalidJudgmentEvidence("at least one query case is required")
    if len(rows) > MAX_QUERIES:
        raise InvalidJudgmentEvidence(f"at most {MAX_QUERIES} query cases are allowed")

    for case in rows:
        _validate_identifier(case.query_id, "query_id")

    seen_queries: set[str] = set()
    normalized: list[dict[str, Any]] = []
    query_reports: list[QueryCoverage] = []
    fractions: list[float] = []

    for case in sorted(rows, key=lambda row: row.query_id):
        if case.query_id in seen_queries:
            raise InvalidJudgmentEvidence(f"duplicate query_id: {case.query_id}")
        seen_queries.add(case.query_id)

        ranking = list(case.ranking)
        if len(ranking) < policy.k:
            raise InvalidJudgmentEvidence(
                f"query {case.query_id!r} ranking must contain at least policy.k documents"
            )
        if len(ranking) > MAX_RANKING_LENGTH:
            raise InvalidJudgmentEvidence(
                f"rankings may contain at most {MAX_RANKING_LENGTH} documents"
            )
        for doc_id in ranking:
            _validate_identifier(doc_id, "ranking doc_id")
        if len(set(ranking)) != len(ranking):
            raise InvalidJudgmentEvidence(
                f"query {case.query_id!r} ranking contains duplicate documents"
            )

        judged = set(case.judged_doc_ids)
        if not judged:
            raise InvalidJudgmentEvidence(
                f"query {case.query_id!r} must include at least one judged document"
            )
        for doc_id in judged:
            _validate_identifier(doc_id, "judged doc_id")

        judged_at_k = sum(doc_id in judged for doc_id in ranking[: policy.k])
        fraction = judged_at_k / policy.k
        fractions.append(fraction)
        query_reports.append(
            QueryCoverage(
                query_ref=_query_ref(case.query_id),
                judged_at_k=judged_at_k,
                unjudged_at_k=policy.k - judged_at_k,
                judged_fraction_at_k=fraction,
            )
        )
        normalized.append(
            {
                "query_id": case.query_id,
                "ranking": ranking,
                "judged_doc_ids": sorted(judged),
            }
        )

    mean_fraction = sum(fractions) / len(fractions)
    low_count = sum(value < policy.low_coverage_threshold for value in fractions)
    low_rate = low_count / len(fractions)
    reasons: list[str] = []
    if mean_fraction < policy.min_mean_judged_fraction:
        reasons.append("mean_judgment_coverage_below_minimum")
    if low_rate > policy.max_low_coverage_rate:
        reasons.append("low_coverage_query_rate_above_maximum")

    evidence = {
        "schema_version": 1,
        "policy": asdict(policy),
        "queries": normalized,
    }
    evidence_sha256 = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return JudgmentCoverageReport(
        schema_version=1,
        accepted=not reasons,
        reasons=tuple(reasons),
        query_count=len(rows),
        k=policy.k,
        mean_judged_fraction_at_k=mean_fraction,
        minimum_judged_fraction_at_k=min(fractions),
        low_coverage_query_count=low_count,
        low_coverage_query_rate=low_rate,
        evidence_sha256=evidence_sha256,
        queries=tuple(query_reports),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidJudgmentEvidence(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_input(path: Path) -> tuple[list[JudgmentCoverageCase], JudgmentCoveragePolicy]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InvalidJudgmentEvidence(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidJudgmentEvidence(f"unable to read input: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"queries", "policy"}:
        raise InvalidJudgmentEvidence("input must contain exactly 'queries' and 'policy'")
    if not isinstance(payload["queries"], list) or not isinstance(payload["policy"], dict):
        raise InvalidJudgmentEvidence("queries must be an array and policy must be an object")

    policy_fields = {field.name for field in JudgmentCoveragePolicy.__dataclass_fields__.values()}
    if set(payload["policy"]) != policy_fields:
        raise InvalidJudgmentEvidence(f"policy must contain exactly {sorted(policy_fields)}")
    try:
        policy = JudgmentCoveragePolicy(**payload["policy"])
    except TypeError as exc:
        raise InvalidJudgmentEvidence(f"invalid policy: {exc}") from exc

    cases: list[JudgmentCoverageCase] = []
    required = {"query_id", "ranking", "judged_doc_ids"}
    for index, row in enumerate(payload["queries"]):
        if not isinstance(row, dict) or set(row) != required:
            raise InvalidJudgmentEvidence(
                f"queries[{index}] must contain exactly {sorted(required)}"
            )
        if not isinstance(row["ranking"], list) or not isinstance(row["judged_doc_ids"], list):
            raise InvalidJudgmentEvidence(
                f"queries[{index}] ranking and judged_doc_ids must be arrays"
            )
        if not all(isinstance(value, str) for value in row["ranking"]):
            raise InvalidJudgmentEvidence(f"queries[{index}] ranking must contain strings")
        if not all(isinstance(value, str) for value in row["judged_doc_ids"]):
            raise InvalidJudgmentEvidence(f"queries[{index}] judged_doc_ids must contain strings")
        if len(set(row["judged_doc_ids"])) != len(row["judged_doc_ids"]):
            raise InvalidJudgmentEvidence(
                f"queries[{index}] judged_doc_ids contains duplicate documents"
            )
        cases.append(
            JudgmentCoverageCase(
                query_id=row["query_id"],
                ranking=tuple(row["ranking"]),
                judged_doc_ids=frozenset(row["judged_doc_ids"]),
            )
        )
    return cases, policy


def _write_report(report: JudgmentCoverageReport, output: Path | None) -> None:
    encoded = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate retrieval metrics on explicit top-K judgment coverage."
    )
    parser.add_argument("input", type=Path, help="JSON judgment-coverage artifact")
    parser.add_argument("--output", type=Path, help="atomically write the JSON report")
    args = parser.parse_args(argv)
    try:
        cases, policy = _load_input(args.input)
        report = audit_judgment_coverage(cases, policy)
        _write_report(report, args.output)
    except (InvalidJudgmentEvidence, OSError) as exc:
        print(f"invalid judgment evidence: {exc}", file=sys.stderr)
        return 3
    return 0 if report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
