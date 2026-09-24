"""Fail-closed contamination audit for retrieval evaluation query sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class ContaminationInputError(ValueError):
    """Raised when an artifact cannot support a trustworthy audit."""


@dataclass(frozen=True, slots=True)
class TextRecord:
    record_id: str
    text: str

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TextRecord:
        record_id = values.get("id")
        text = values.get("text")
        if not isinstance(record_id, str) or not _ID_PATTERN.fullmatch(record_id):
            raise ContaminationInputError(
                "record id must be 1-128 safe ASCII letters, digits or '_.:-'"
            )
        if not isinstance(text, str) or not text.strip():
            raise ContaminationInputError(f"record {record_id!r} must contain non-empty text")
        return cls(record_id=record_id, text=text)


def _finite_rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return float(value)


@dataclass(frozen=True, slots=True)
class ContaminationPolicy:
    max_exact_leaked_queries: int = 0
    max_near_leak_rate: float = 0.0
    near_similarity_threshold: float = 0.85
    min_tokens_for_near_match: int = 5
    max_items_per_split: int = 10_000
    max_text_bytes: int = 16_384
    max_pair_comparisons: int = 1_000_000
    reject_evaluation_duplicates: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "max_exact_leaked_queries",
            "min_tokens_for_near_match",
            "max_items_per_split",
            "max_text_bytes",
            "max_pair_comparisons",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_exact_leaked_queries < 0:
            raise ValueError("max_exact_leaked_queries must be non-negative")
        for name in integer_fields[1:]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        object.__setattr__(
            self,
            "max_near_leak_rate",
            _finite_rate("max_near_leak_rate", self.max_near_leak_rate),
        )
        threshold = _finite_rate("near_similarity_threshold", self.near_similarity_threshold)
        if threshold == 0:
            raise ValueError("near_similarity_threshold must be greater than zero")
        object.__setattr__(self, "near_similarity_threshold", threshold)
        if not isinstance(self.reject_evaluation_duplicates, bool):
            raise TypeError("reject_evaluation_duplicates must be a boolean")

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "max_exact_leaked_queries": self.max_exact_leaked_queries,
            "max_near_leak_rate": self.max_near_leak_rate,
            "near_similarity_threshold": self.near_similarity_threshold,
            "min_tokens_for_near_match": self.min_tokens_for_near_match,
            "max_items_per_split": self.max_items_per_split,
            "max_text_bytes": self.max_text_bytes,
            "max_pair_comparisons": self.max_pair_comparisons,
            "reject_evaluation_duplicates": self.reject_evaluation_duplicates,
        }


class MatchKind(StrEnum):
    EXACT = "exact"
    NEAR = "near"


class ContaminationReason(StrEnum):
    COMPARISON_BUDGET_EXCEEDED = "comparison_budget_exceeded"
    DUPLICATE_EVALUATION_TEXT = "duplicate_evaluation_text"
    EXACT_LEAKAGE_LIMIT_EXCEEDED = "exact_leakage_limit_exceeded"
    NEAR_LEAKAGE_RATE_EXCEEDED = "near_leakage_rate_exceeded"


@dataclass(frozen=True, slots=True)
class LeakageMatch:
    kind: MatchKind
    reference_id: str
    evaluation_id: str
    similarity: float
    reference_digest: str
    evaluation_digest: str

    def as_dict(self) -> dict[str, str | float]:
        return {
            "kind": self.kind.value,
            "reference_id": self.reference_id,
            "evaluation_id": self.evaluation_id,
            "similarity": self.similarity,
            "reference_digest": self.reference_digest,
            "evaluation_digest": self.evaluation_digest,
        }


@dataclass(frozen=True, slots=True)
class AuditFinding:
    reason: ContaminationReason
    detail: str
    evaluation_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "detail": self.detail,
            "evaluation_ids": list(self.evaluation_ids),
        }


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    accepted: bool
    reference_items: int
    evaluation_items: int
    pair_comparisons: int
    exact_leaked_queries: int
    near_leaked_queries: int
    near_leak_rate: float
    duplicate_evaluation_ids: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    matches: tuple[LeakageMatch, ...]
    policy: ContaminationPolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "metrics": {
                "reference_items": self.reference_items,
                "evaluation_items": self.evaluation_items,
                "pair_comparisons": self.pair_comparisons,
                "exact_leaked_queries": self.exact_leaked_queries,
                "near_leaked_queries": self.near_leaked_queries,
                "near_leak_rate": self.near_leak_rate,
            },
            "duplicate_evaluation_ids": list(self.duplicate_evaluation_ids),
            "findings": [item.as_dict() for item in self.findings],
            "matches": [item.as_dict() for item in self.matches],
            "policy": self.policy.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _PreparedRecord:
    record_id: str
    normalized: str
    tokens: frozenset[str]
    digest: str


def audit_contamination(
    reference_records: Sequence[TextRecord],
    evaluation_records: Sequence[TextRecord],
    *,
    policy: ContaminationPolicy | None = None,
) -> ContaminationReport:
    """Compare a tuning/reference query bank with a held-out evaluation split."""

    active_policy = policy or ContaminationPolicy()
    if not isinstance(active_policy, ContaminationPolicy):
        raise TypeError("policy must be a ContaminationPolicy")
    references = _prepare_split("reference", reference_records, active_policy)
    evaluations = _prepare_split("evaluation", evaluation_records, active_policy)
    shared_ids = {item.record_id for item in references} & {item.record_id for item in evaluations}
    if shared_ids:
        raise ContaminationInputError(
            f"record IDs occur in both splits: {', '.join(sorted(shared_ids))}"
        )

    duplicate_evaluation_ids = _duplicate_evaluation_ids(evaluations)
    exact_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, set[int]] = defaultdict(set)
    for index, item in enumerate(references):
        exact_index[item.normalized].append(index)
        for token in item.tokens:
            token_index[token].add(index)

    matches: list[LeakageMatch] = []
    comparisons = 0
    comparison_budget_exceeded = False
    for evaluation in evaluations:
        exact_indices = set(exact_index.get(evaluation.normalized, ()))
        for index in sorted(exact_indices):
            matches.append(_match(MatchKind.EXACT, references[index], evaluation, 1.0))

        if len(evaluation.tokens) < active_policy.min_tokens_for_near_match:
            continue
        candidate_indices: set[int] = set()
        for token in evaluation.tokens:
            candidate_indices.update(token_index.get(token, ()))
        for index in sorted(candidate_indices - exact_indices):
            reference = references[index]
            if len(reference.tokens) < active_policy.min_tokens_for_near_match:
                continue
            if comparisons >= active_policy.max_pair_comparisons:
                comparison_budget_exceeded = True
                break
            comparisons += 1
            similarity = _jaccard(reference.tokens, evaluation.tokens)
            if similarity >= active_policy.near_similarity_threshold:
                matches.append(_match(MatchKind.NEAR, reference, evaluation, similarity))
        if comparison_budget_exceeded:
            break

    matches.sort(key=lambda item: (item.evaluation_id, item.kind.value, item.reference_id))
    exact_ids = tuple(
        sorted({item.evaluation_id for item in matches if item.kind is MatchKind.EXACT})
    )
    near_ids = tuple(
        sorted({item.evaluation_id for item in matches if item.kind is MatchKind.NEAR})
    )
    near_rate = len(near_ids) / len(evaluations)
    findings: list[AuditFinding] = []
    if comparison_budget_exceeded:
        findings.append(
            AuditFinding(
                ContaminationReason.COMPARISON_BUDGET_EXCEEDED,
                f"near-match comparison budget exceeded {active_policy.max_pair_comparisons}",
            )
        )
    if duplicate_evaluation_ids and active_policy.reject_evaluation_duplicates:
        findings.append(
            AuditFinding(
                ContaminationReason.DUPLICATE_EVALUATION_TEXT,
                "normalized duplicate text appears within the evaluation split",
                duplicate_evaluation_ids,
            )
        )
    if len(exact_ids) > active_policy.max_exact_leaked_queries:
        findings.append(
            AuditFinding(
                ContaminationReason.EXACT_LEAKAGE_LIMIT_EXCEEDED,
                f"{len(exact_ids)} exact-leaked query(s) exceed limit "
                f"{active_policy.max_exact_leaked_queries}",
                exact_ids,
            )
        )
    if near_rate > active_policy.max_near_leak_rate:
        findings.append(
            AuditFinding(
                ContaminationReason.NEAR_LEAKAGE_RATE_EXCEEDED,
                f"near-leak rate {near_rate:.6f} exceeds {active_policy.max_near_leak_rate:.6f}",
                near_ids,
            )
        )

    return ContaminationReport(
        accepted=not findings,
        reference_items=len(references),
        evaluation_items=len(evaluations),
        pair_comparisons=comparisons,
        exact_leaked_queries=len(exact_ids),
        near_leaked_queries=len(near_ids),
        near_leak_rate=round(near_rate, 6),
        duplicate_evaluation_ids=duplicate_evaluation_ids,
        findings=tuple(findings),
        matches=tuple(matches),
        policy=active_policy,
    )


def _prepare_split(
    label: str,
    records: Sequence[TextRecord],
    policy: ContaminationPolicy,
) -> tuple[_PreparedRecord, ...]:
    if not records:
        raise ContaminationInputError(f"{label} split must contain at least one record")
    if len(records) > policy.max_items_per_split:
        raise ContaminationInputError(f"{label} split exceeds {policy.max_items_per_split} records")
    seen: set[str] = set()
    output: list[_PreparedRecord] = []
    for record in records:
        if not isinstance(record, TextRecord):
            raise TypeError(f"{label} split entries must be TextRecord instances")
        if not isinstance(record.record_id, str) or not _ID_PATTERN.fullmatch(record.record_id):
            raise ContaminationInputError(f"{label} split contains an invalid record id")
        if not isinstance(record.text, str) or not record.text.strip():
            raise ContaminationInputError(
                f"{label} record {record.record_id!r} must contain non-empty text"
            )
        if record.record_id in seen:
            raise ContaminationInputError(f"duplicate {label} record id: {record.record_id}")
        seen.add(record.record_id)
        encoded = record.text.encode("utf-8")
        if len(encoded) > policy.max_text_bytes:
            raise ContaminationInputError(
                f"{label} record {record.record_id!r} exceeds {policy.max_text_bytes} bytes"
            )
        tokens = tuple(
            _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", record.text).casefold())
        )
        if not tokens:
            raise ContaminationInputError(
                f"{label} record {record.record_id!r} has no comparable tokens"
            )
        normalized = " ".join(tokens)
        output.append(
            _PreparedRecord(
                record_id=record.record_id,
                normalized=normalized,
                tokens=frozenset(tokens),
                digest=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(sorted(output, key=lambda item: item.record_id))


def _duplicate_evaluation_ids(records: Sequence[_PreparedRecord]) -> tuple[str, ...]:
    by_text: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_text[record.normalized].append(record.record_id)
    return tuple(sorted(record_id for ids in by_text.values() if len(ids) > 1 for record_id in ids))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


def _match(
    kind: MatchKind,
    reference: _PreparedRecord,
    evaluation: _PreparedRecord,
    similarity: float,
) -> LeakageMatch:
    return LeakageMatch(
        kind=kind,
        reference_id=reference.record_id,
        evaluation_id=evaluation.record_id,
        similarity=round(similarity, 6),
        reference_digest=reference.digest,
        evaluation_digest=evaluation.digest,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ContaminationInputError(f"duplicate JSON field: {key}")
        values[key] = value
    return values


def load_jsonl(path: Path, *, max_records: int) -> list[TextRecord]:
    records: list[TextRecord] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise ContaminationInputError(f"{path}: blank line at {line_number}")
                try:
                    values = json.loads(raw, object_pairs_hook=_strict_object)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ContaminationInputError(
                        f"{path}: invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(values, dict):
                    raise ContaminationInputError(
                        f"{path}: line {line_number} must be a JSON object"
                    )
                try:
                    records.append(TextRecord.from_dict(values))
                except ContaminationInputError as exc:
                    raise ContaminationInputError(f"{path}: line {line_number}: {exc}") from exc
                if len(records) > max_records:
                    raise ContaminationInputError(
                        f"{path}: exceeds maximum of {max_records} records"
                    )
    except OSError as exc:
        raise ContaminationInputError(f"cannot read {path}") from exc
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_jsonl", type=Path)
    parser.add_argument("evaluation_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--max-near-rate", type=float, default=0.0)
    parser.add_argument("--max-exact", type=int, default=0)
    parser.add_argument("--max-comparisons", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = ContaminationPolicy(
            max_exact_leaked_queries=args.max_exact,
            max_near_leak_rate=args.max_near_rate,
            near_similarity_threshold=args.near_threshold,
            max_pair_comparisons=args.max_comparisons,
        )
        reference = load_jsonl(args.reference_jsonl, max_records=policy.max_items_per_split)
        evaluation = load_jsonl(args.evaluation_jsonl, max_records=policy.max_items_per_split)
        report = audit_contamination(reference, evaluation, policy=policy)
    except (ContaminationInputError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "malformed", "error": str(exc)}, sort_keys=True))
        return 2

    encoded = json.dumps(report.as_dict(), indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        try:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        except OSError:
            print(json.dumps({"status": "malformed", "error": f"cannot write {args.output}"}))
            return 2
    print(encoded)
    return 0 if report.accepted else 3


if __name__ == "__main__":
    sys.exit(main())
