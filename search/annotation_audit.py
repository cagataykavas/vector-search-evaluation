from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_ANNOTATIONS = 1_000_000
MAX_ID_LENGTH = 256


@dataclass(frozen=True, slots=True)
class AnnotationAudit:
    dataset_id: str
    rubric_version: str
    input_digest: str
    items: int
    assessors: int
    ratings: int
    max_grade: int
    grade_distribution: tuple[int, ...]
    pair_agreement: float
    unanimous_item_rate: float
    fleiss_kappa: float | None
    disputed_items: int
    min_pair_agreement: float
    min_fleiss_kappa: float
    passed: bool
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_id(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_ID_LENGTH
        or not value.isprintable()
    ):
        raise ValueError(f"{field} must be a trimmed, printable 1-{MAX_ID_LENGTH} character string")
    return value


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number between 0 and 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a number between 0 and 1")
    return result


def _canonical_digest(
    dataset_id: str,
    rubric_version: str,
    rows: Sequence[tuple[str, str, str, int]],
) -> str:
    canonical = {
        "annotations": [
            {
                "assessor_id": assessor_id,
                "document_id": document_id,
                "grade": grade,
                "query_id": query_id,
            }
            for query_id, document_id, assessor_id, grade in sorted(rows)
        ],
        "dataset_id": dataset_id,
        "rubric_version": rubric_version,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def audit_annotations(
    payload: Mapping[str, Any],
    *,
    max_grade: int = 3,
    min_pair_agreement: float = 0.8,
    min_fleiss_kappa: float = 0.6,
) -> AnnotationAudit:
    """Audit independently assigned qrels without exposing item or assessor identifiers."""
    if not isinstance(payload, Mapping):
        raise TypeError("annotation bundle must be a JSON object")
    if set(payload) != {"dataset_id", "rubric_version", "annotations"}:
        raise ValueError(
            "annotation bundle must contain exactly dataset_id, rubric_version and annotations"
        )
    if type(max_grade) is not int or max_grade < 1 or max_grade > 100:
        raise ValueError("max_grade must be an integer between 1 and 100")
    pair_threshold = _probability(min_pair_agreement, field="min_pair_agreement")
    kappa_threshold = _probability(min_fleiss_kappa, field="min_fleiss_kappa")
    dataset_id = _bounded_id(payload["dataset_id"], field="dataset_id")
    rubric_version = _bounded_id(payload["rubric_version"], field="rubric_version")

    raw_annotations = payload["annotations"]
    if isinstance(raw_annotations, (str, bytes)) or not isinstance(raw_annotations, Sequence):
        raise TypeError("annotations must be a JSON array")
    if not raw_annotations:
        raise ValueError("annotations must not be empty")
    if len(raw_annotations) > MAX_ANNOTATIONS:
        raise ValueError(f"annotations exceeds the {MAX_ANNOTATIONS} rating budget")

    rows: list[tuple[str, str, str, int]] = []
    ratings_by_item: dict[tuple[str, str], dict[str, int]] = {}
    all_assessors: set[str] = set()
    expected_fields = {"query_id", "document_id", "assessor_id", "grade"}
    for position, raw_row in enumerate(raw_annotations, start=1):
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_fields:
            raise ValueError(
                f"annotation {position} must contain exactly query_id, document_id, "
                "assessor_id and grade"
            )
        query_id = _bounded_id(raw_row["query_id"], field=f"annotation {position} query_id")
        document_id = _bounded_id(
            raw_row["document_id"], field=f"annotation {position} document_id"
        )
        assessor_id = _bounded_id(
            raw_row["assessor_id"], field=f"annotation {position} assessor_id"
        )
        grade = raw_row["grade"]
        if type(grade) is not int or not 0 <= grade <= max_grade:
            raise ValueError(
                f"annotation {position} grade must be an integer between 0 and {max_grade}"
            )

        item = (query_id, document_id)
        item_ratings = ratings_by_item.setdefault(item, {})
        if assessor_id in item_ratings:
            raise ValueError(f"annotation {position} duplicates an item-assessor rating")
        item_ratings[assessor_id] = grade
        all_assessors.add(assessor_id)
        rows.append((query_id, document_id, assessor_id, grade))

    if len(ratings_by_item) < 2:
        raise ValueError("agreement audit requires at least two query-document items")
    if len(all_assessors) < 2:
        raise ValueError("agreement audit requires at least two independent assessors")
    for position, item_ratings in enumerate(ratings_by_item.values(), start=1):
        if set(item_ratings) != all_assessors:
            raise ValueError(
                f"item {position} does not have exactly one rating from every assessor"
            )

    assessor_count = len(all_assessors)
    item_count = len(ratings_by_item)
    category_counts = [0] * (max_grade + 1)
    agreement_sum = 0.0
    agreeing_pairs = 0
    pair_count = 0
    disputed_items = 0
    for item_ratings in ratings_by_item.values():
        grades = list(item_ratings.values())
        counts = Counter(grades)
        for grade, count in counts.items():
            category_counts[grade] += count
        agreement_sum += (sum(count * count for count in counts.values()) - assessor_count) / (
            assessor_count * (assessor_count - 1)
        )
        item_pairs = list(combinations(grades, 2))
        agreeing_pairs += sum(left == right for left, right in item_pairs)
        pair_count += len(item_pairs)
        if len(counts) > 1:
            disputed_items += 1

    observed_agreement = agreement_sum / item_count
    rating_count = len(rows)
    expected_agreement = sum((count / rating_count) ** 2 for count in category_counts)
    denominator = 1.0 - expected_agreement
    fleiss_kappa = (
        (observed_agreement - expected_agreement) / denominator if denominator > 1e-12 else None
    )
    pair_agreement = agreeing_pairs / pair_count
    unanimous_item_rate = (item_count - disputed_items) / item_count

    findings: list[str] = []
    if fleiss_kappa is None:
        findings.append("fleiss_kappa_undefined_no_grade_variation")
    elif fleiss_kappa < kappa_threshold:
        findings.append("fleiss_kappa_below_threshold")
    if pair_agreement < pair_threshold:
        findings.append("pair_agreement_below_threshold")

    return AnnotationAudit(
        dataset_id=dataset_id,
        rubric_version=rubric_version,
        input_digest=_canonical_digest(dataset_id, rubric_version, rows),
        items=item_count,
        assessors=assessor_count,
        ratings=rating_count,
        max_grade=max_grade,
        grade_distribution=tuple(category_counts),
        pair_agreement=pair_agreement,
        unanimous_item_rate=unanimous_item_rate,
        fleiss_kappa=fleiss_kappa,
        disputed_items=disputed_items,
        min_pair_agreement=pair_threshold,
        min_fleiss_kappa=kappa_threshold,
        passed=not findings,
        findings=tuple(findings),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("input JSON contains a duplicate object key")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds the {MAX_INPUT_BYTES} byte budget")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, Mapping):
        raise TypeError("annotation bundle must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed when independent qrels annotations lack sufficient agreement."
    )
    parser.add_argument("input", type=Path, help="JSON annotation bundle")
    parser.add_argument("--output", required=True, type=Path, help="atomic JSON audit report")
    parser.add_argument("--max-grade", type=int, default=3)
    parser.add_argument("--min-pair-agreement", type=float, default=0.8)
    parser.add_argument("--min-fleiss-kappa", type=float, default=0.6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit = audit_annotations(
            _load_json(args.input),
            max_grade=args.max_grade,
            min_pair_agreement=args.min_pair_agreement,
            min_fleiss_kappa=args.min_fleiss_kappa,
        )
        _write_json_atomic(args.output, audit.to_dict())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"annotation audit error: {exc}", file=sys.stderr)
        return 2
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
