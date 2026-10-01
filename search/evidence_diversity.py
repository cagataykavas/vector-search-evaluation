from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_IDENTIFIER_CHARS = 256
MAX_TEXT_BYTES = 128 * 1024
MAX_QUERIES = 1_000
MAX_HITS_PER_QUERY = 50
MAX_TOTAL_TEXT_BYTES = 8 * 1024 * 1024


class EvidenceDiversityError(ValueError):
    """Raised when an evidence artifact or policy is malformed."""


@dataclass(frozen=True)
class EvidenceHit:
    rank: int
    doc_id: str
    source_id: str
    text: str


@dataclass(frozen=True)
class QueryEvidence:
    query_id: str
    hits: tuple[EvidenceHit, ...]


@dataclass(frozen=True)
class EvidenceArtifact:
    index_snapshot_digest: str
    queries: tuple[QueryEvidence, ...]


@dataclass(frozen=True)
class DiversityPolicy:
    shingle_size: int = 3
    near_duplicate_threshold: float = 0.82
    max_exact_duplicate_fraction: float = 0.0
    max_redundant_hit_fraction: float = 0.34
    max_largest_cluster_fraction: float = 0.67
    min_effective_evidence: int = 2
    min_independent_sources: int = 2
    max_pair_comparisons: int = 250_000

    def __post_init__(self) -> None:
        if isinstance(self.shingle_size, bool) or not 1 <= self.shingle_size <= 8:
            raise EvidenceDiversityError("shingle_size must be an integer between 1 and 8")
        for name in (
            "near_duplicate_threshold",
            "max_exact_duplicate_fraction",
            "max_redundant_hit_fraction",
            "max_largest_cluster_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvidenceDiversityError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise EvidenceDiversityError(f"{name} must be finite and within [0, 1]")
        if self.near_duplicate_threshold == 0.0:
            raise EvidenceDiversityError("near_duplicate_threshold must be greater than zero")
        for name in ("min_effective_evidence", "min_independent_sources"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvidenceDiversityError(f"{name} must be a positive integer")
            if value > MAX_HITS_PER_QUERY:
                raise EvidenceDiversityError(f"{name} exceeds the per-query hit limit")
        if (
            isinstance(self.max_pair_comparisons, bool)
            or not isinstance(self.max_pair_comparisons, int)
            or self.max_pair_comparisons < 1
        ):
            raise EvidenceDiversityError("max_pair_comparisons must be a positive integer")


@dataclass(frozen=True)
class QueryDiversityAudit:
    query_ref: str
    accepted: bool
    reasons: tuple[str, ...]
    hit_count: int
    distinct_source_count: int
    effective_evidence_count: int
    independent_source_count: int
    exact_duplicate_hits: int
    exact_duplicate_fraction: float
    near_duplicate_pairs: int
    redundant_hit_fraction: float
    largest_cluster_fraction: float


@dataclass(frozen=True)
class DiversityAuditReport:
    schema_version: str
    accepted: bool
    reasons: tuple[str, ...]
    artifact_digest: str
    index_snapshot_digest: str
    query_count: int
    total_hit_count: int
    pair_comparisons: int
    policy: DiversityPolicy
    queries: tuple[QueryDiversityAudit, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["queries"] = [
            {**asdict(query), "reasons": list(query.reasons)} for query in self.queries
        ]
        return payload


def _require_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceDiversityError(f"{path} must be an object")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise EvidenceDiversityError(f"{path} has missing or unexpected fields")


def _require_identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise EvidenceDiversityError(
            f"{path} must be a non-empty string no longer than {MAX_IDENTIFIER_CHARS} characters"
        )
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise EvidenceDiversityError(f"{path} must not contain control characters")
    return value


def _require_rank(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvidenceDiversityError(f"{path} must be a positive integer")
    return value


def _require_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceDiversityError(f"{path} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise EvidenceDiversityError(f"{path} exceeds the per-hit byte limit")
    if "\x00" in value:
        raise EvidenceDiversityError(f"{path} must not contain NUL bytes")
    return value


def parse_artifact(payload: object) -> EvidenceArtifact:
    root = _require_object(payload, "artifact")
    _require_exact_keys(root, {"index_snapshot_digest", "queries"}, "artifact")
    snapshot_digest = root["index_snapshot_digest"]
    if not isinstance(snapshot_digest, str) or not SHA256_RE.fullmatch(snapshot_digest):
        raise EvidenceDiversityError("index_snapshot_digest must be a lowercase SHA-256 digest")

    raw_queries = root["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise EvidenceDiversityError("queries must be a non-empty array")
    if len(raw_queries) > MAX_QUERIES:
        raise EvidenceDiversityError("queries exceeds the artifact limit")

    queries: list[QueryEvidence] = []
    seen_query_ids: set[str] = set()
    total_text_bytes = 0
    for query_index, raw_query in enumerate(raw_queries):
        query_path = f"queries[{query_index}]"
        query = _require_object(raw_query, query_path)
        _require_exact_keys(query, {"query_id", "hits"}, query_path)
        query_id = _require_identifier(query["query_id"], f"{query_path}.query_id")
        if query_id in seen_query_ids:
            raise EvidenceDiversityError("query_id values must be unique")
        seen_query_ids.add(query_id)

        raw_hits = query["hits"]
        if not isinstance(raw_hits, list) or not raw_hits:
            raise EvidenceDiversityError(f"{query_path}.hits must be a non-empty array")
        if len(raw_hits) > MAX_HITS_PER_QUERY:
            raise EvidenceDiversityError(f"{query_path}.hits exceeds the per-query limit")

        hits: list[EvidenceHit] = []
        seen_doc_ids: set[str] = set()
        for hit_index, raw_hit in enumerate(raw_hits):
            hit_path = f"{query_path}.hits[{hit_index}]"
            hit = _require_object(raw_hit, hit_path)
            _require_exact_keys(hit, {"rank", "doc_id", "source_id", "text"}, hit_path)
            rank = _require_rank(hit["rank"], f"{hit_path}.rank")
            doc_id = _require_identifier(hit["doc_id"], f"{hit_path}.doc_id")
            source_id = _require_identifier(hit["source_id"], f"{hit_path}.source_id")
            text = _require_text(hit["text"], f"{hit_path}.text")
            if doc_id in seen_doc_ids:
                raise EvidenceDiversityError("doc_id values must be unique within a query")
            seen_doc_ids.add(doc_id)
            total_text_bytes += len(text.encode("utf-8"))
            if total_text_bytes > MAX_TOTAL_TEXT_BYTES:
                raise EvidenceDiversityError("artifact exceeds the total text byte limit")
            hits.append(EvidenceHit(rank=rank, doc_id=doc_id, source_id=source_id, text=text))
        if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)):
            raise EvidenceDiversityError("hit ranks must be contiguous and ordered from one")
        queries.append(QueryEvidence(query_id=query_id, hits=tuple(hits)))
    return EvidenceArtifact(index_snapshot_digest=snapshot_digest, queries=tuple(queries))


def _normalize_text(text: str) -> str:
    tokens = TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    return " ".join(token for token in tokens if any(character.isalnum() for character in token))


def _features(text: str, shingle_size: int) -> frozenset[tuple[str, ...]]:
    tokens = tuple(_normalize_text(text).split())
    if not tokens:
        raise EvidenceDiversityError("hit text must contain at least one searchable token")
    if len(tokens) < shingle_size:
        return frozenset((token,) for token in tokens)
    return frozenset(
        tokens[index : index + shingle_size] for index in range(len(tokens) - shingle_size + 1)
    )


def _jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _maximum_independent_sources(cluster_sources: list[set[str]]) -> int:
    source_to_cluster: dict[str, int] = {}

    def assign(cluster_index: int, visited: set[str]) -> bool:
        for source_id in sorted(cluster_sources[cluster_index]):
            if source_id in visited:
                continue
            visited.add(source_id)
            incumbent = source_to_cluster.get(source_id)
            if incumbent is None or assign(incumbent, visited):
                source_to_cluster[source_id] = cluster_index
                return True
        return False

    matches = 0
    for cluster_index in range(len(cluster_sources)):
        if assign(cluster_index, set()):
            matches += 1
    return matches


def _artifact_digest(artifact: EvidenceArtifact) -> str:
    canonical_queries = []
    for query in sorted(artifact.queries, key=lambda item: item.query_id):
        canonical_queries.append(
            {
                "query_id": query.query_id,
                "hits": [
                    {
                        "rank": hit.rank,
                        "doc_id": hit.doc_id,
                        "source_id": hit.source_id,
                        "content_digest": hashlib.sha256(
                            _normalize_text(hit.text).encode("utf-8")
                        ).hexdigest(),
                    }
                    for hit in query.hits
                ],
            }
        )
    canonical = json.dumps(
        {
            "index_snapshot_digest": artifact.index_snapshot_digest,
            "queries": canonical_queries,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _audit_query(query: QueryEvidence, policy: DiversityPolicy) -> QueryDiversityAudit:
    count = len(query.hits)
    normalized = [_normalize_text(hit.text) for hit in query.hits]
    content_digests = [hashlib.sha256(text.encode("utf-8")).digest() for text in normalized]
    features = [_features(hit.text, policy.shingle_size) for hit in query.hits]
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    near_duplicate_pairs = 0
    for left in range(count):
        for right in range(left + 1, count):
            if (
                content_digests[left] == content_digests[right]
                or _jaccard(features[left], features[right]) >= policy.near_duplicate_threshold
            ):
                near_duplicate_pairs += 1
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(index)
    cluster_rows = list(clusters.values())
    cluster_sources = [
        {query.hits[index].source_id for index in cluster} for cluster in cluster_rows
    ]
    exact_duplicate_hits = count - len(set(content_digests))
    exact_duplicate_fraction = exact_duplicate_hits / count
    redundant_fraction = (count - len(cluster_rows)) / count
    largest_cluster_fraction = max(len(cluster) for cluster in cluster_rows) / count
    independent_sources = _maximum_independent_sources(cluster_sources)

    reasons: list[str] = []
    if exact_duplicate_fraction > policy.max_exact_duplicate_fraction:
        reasons.append("exact_duplicate_budget_exceeded")
    if redundant_fraction > policy.max_redundant_hit_fraction:
        reasons.append("redundant_hit_budget_exceeded")
    if largest_cluster_fraction > policy.max_largest_cluster_fraction:
        reasons.append("duplicate_cluster_dominates")
    if len(cluster_rows) < policy.min_effective_evidence:
        reasons.append("insufficient_effective_evidence")
    if independent_sources < policy.min_independent_sources:
        reasons.append("insufficient_independent_sources")

    return QueryDiversityAudit(
        query_ref=hashlib.sha256(query.query_id.encode("utf-8")).hexdigest()[:16],
        accepted=not reasons,
        reasons=tuple(sorted(reasons)),
        hit_count=count,
        distinct_source_count=len({hit.source_id for hit in query.hits}),
        effective_evidence_count=len(cluster_rows),
        independent_source_count=independent_sources,
        exact_duplicate_hits=exact_duplicate_hits,
        exact_duplicate_fraction=exact_duplicate_fraction,
        near_duplicate_pairs=near_duplicate_pairs,
        redundant_hit_fraction=redundant_fraction,
        largest_cluster_fraction=largest_cluster_fraction,
    )


def audit_evidence_diversity(
    artifact: EvidenceArtifact | object,
    policy: DiversityPolicy | None = None,
) -> DiversityAuditReport:
    parsed = artifact if isinstance(artifact, EvidenceArtifact) else parse_artifact(artifact)
    effective_policy = policy or DiversityPolicy()
    pair_comparisons = sum(len(query.hits) * (len(query.hits) - 1) // 2 for query in parsed.queries)
    if pair_comparisons > effective_policy.max_pair_comparisons:
        raise EvidenceDiversityError("artifact exceeds the pair comparison budget")
    query_reports = tuple(
        _audit_query(query, effective_policy)
        for query in sorted(parsed.queries, key=lambda item: item.query_id)
    )
    reasons = tuple(sorted({reason for query in query_reports for reason in query.reasons}))
    return DiversityAuditReport(
        schema_version="1.0",
        accepted=not reasons,
        reasons=reasons,
        artifact_digest=_artifact_digest(parsed),
        index_snapshot_digest=parsed.index_snapshot_digest,
        query_count=len(parsed.queries),
        total_hit_count=sum(len(query.hits) for query in parsed.queries),
        pair_comparisons=pair_comparisons,
        policy=effective_policy,
        queries=query_reports,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceDiversityError("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def load_artifact(path: Path) -> EvidenceArtifact:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceDiversityError("input exceeds the JSON byte limit")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceDiversityError("JSON must not contain non-finite numbers")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceDiversityError("input is not valid UTF-8 JSON") from error
    return parse_artifact(payload)


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit diversity of RAG retrieval evidence")
    parser.add_argument("artifact", type=Path, help="retrieval evidence JSON artifact")
    parser.add_argument("--output", type=Path, help="atomically write the JSON report")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--max-exact-duplicate-fraction", type=float, default=0.0)
    parser.add_argument("--max-redundant-hit-fraction", type=float, default=0.34)
    parser.add_argument("--max-largest-cluster-fraction", type=float, default=0.67)
    parser.add_argument("--min-effective-evidence", type=int, default=2)
    parser.add_argument("--min-independent-sources", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        policy = DiversityPolicy(
            near_duplicate_threshold=arguments.near_duplicate_threshold,
            max_exact_duplicate_fraction=arguments.max_exact_duplicate_fraction,
            max_redundant_hit_fraction=arguments.max_redundant_hit_fraction,
            max_largest_cluster_fraction=arguments.max_largest_cluster_fraction,
            min_effective_evidence=arguments.min_effective_evidence,
            min_independent_sources=arguments.min_independent_sources,
        )
        report = audit_evidence_diversity(load_artifact(arguments.artifact), policy)
        _write_json(report.to_dict(), arguments.output)
    except (EvidenceDiversityError, OSError) as error:
        _write_json(
            {"accepted": False, "error": "malformed_artifact", "detail": str(error)},
            arguments.output,
        )
        return 2
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
