from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class QrelsAudit:
    documents: int
    queries: int
    relevance_judgments: int
    unique_relevant_documents: int
    corpus_relevance_coverage: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _bounded_ids(values: Iterable[str], *, label: str, maximum: int) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a collection of IDs, not a string")
    materialized: list[str] = []
    for position, value in enumerate(values, start=1):
        if position > maximum:
            raise ValueError(f"{label} exceeds the {maximum} item budget")
        if not isinstance(value, str) or not 1 <= len(value) <= 256:
            raise ValueError(f"{label} item {position} must be a 1-256 character string")
        materialized.append(value)
    return materialized


def audit_qrels(
    document_ids: Iterable[str],
    qrels: Iterable[tuple[str, Iterable[str]]],
    *,
    max_documents: int = 1_000_000,
    max_queries: int = 10_000,
    max_judgments: int = 1_000_000,
) -> QrelsAudit:
    """Validate positive qrels against an exact corpus before metric calculation."""
    for name, value in (
        ("max_documents", max_documents),
        ("max_queries", max_queries),
        ("max_judgments", max_judgments),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    documents = _bounded_ids(document_ids, label="document IDs", maximum=max_documents)
    if not documents:
        raise ValueError("qrels audit requires at least one document")
    document_set = set(documents)
    if len(document_set) != len(documents):
        raise ValueError("document IDs must be unique")

    query_ids: set[str] = set()
    relevant_documents: set[str] = set()
    query_count = 0
    judgment_count = 0
    for query_count, entry in enumerate(qrels, start=1):
        if query_count > max_queries:
            raise ValueError(f"qrels exceeds the {max_queries} query budget")
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"qrels item {query_count} must be a (query_id, relevant_ids) tuple")
        query_id, relevant_ids = entry
        if not isinstance(query_id, str) or not 1 <= len(query_id) <= 256:
            raise ValueError(f"query ID at position {query_count} must be a 1-256 character string")
        if query_id in query_ids:
            raise ValueError(f"duplicate query ID {query_id!r}")
        query_ids.add(query_id)

        remaining = max_judgments - judgment_count
        if remaining < 1:
            raise ValueError(f"qrels exceeds the {max_judgments} judgment budget")
        relevant = _bounded_ids(
            relevant_ids,
            label=f"relevant IDs for query {query_id!r}",
            maximum=remaining,
        )
        if not relevant:
            raise ValueError(f"query {query_id!r} must have at least one relevant document")
        if len(set(relevant)) != len(relevant):
            raise ValueError(f"query {query_id!r} contains duplicate relevance judgments")
        unknown = sorted(set(relevant) - document_set)
        if unknown:
            rendered = ", ".join(repr(value) for value in unknown[:5])
            suffix = " ..." if len(unknown) > 5 else ""
            raise ValueError(
                f"query {query_id!r} references documents absent from the corpus: "
                f"{rendered}{suffix}"
            )
        judgment_count += len(relevant)
        relevant_documents.update(relevant)

    if query_count == 0:
        raise ValueError("qrels audit requires at least one query")
    return QrelsAudit(
        documents=len(documents),
        queries=query_count,
        relevance_judgments=judgment_count,
        unique_relevant_documents=len(relevant_documents),
        corpus_relevance_coverage=len(relevant_documents) / len(documents),
    )
