from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from evaluate import ndcg_at_k, recall_at_k, reciprocal_rank
from search.engine import Document, HybridSearchEngine


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    text: str
    embedding: tuple[float, ...]
    relevant_doc_ids: frozenset[str] = frozenset()
    relevance_grades: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if bool(self.relevant_doc_ids) == bool(self.relevance_grades):
            raise ValueError(
                "query must define exactly one of relevant_doc_ids or relevance_grades"
            )
        if self.relevance_grades:
            if any(not doc_id.strip() for doc_id, _grade in self.relevance_grades):
                raise ValueError("graded qrels document IDs must not be blank")
            grade_map = {doc_id: float(grade) for doc_id, grade in self.relevance_grades}
            if len(grade_map) != len(self.relevance_grades):
                raise ValueError("graded qrels must contain unique document IDs")
            ndcg_at_k([], grade_map, 1)
            if not any(grade > 0 for grade in grade_map.values()):
                raise ValueError("graded qrels must contain at least one positive grade")

    @property
    def relevance_mode(self) -> str:
        return "graded" if self.relevance_grades else "binary"

    @property
    def grades(self) -> dict[str, float]:
        if self.relevance_grades:
            return {doc_id: float(grade) for doc_id, grade in self.relevance_grades}
        return {doc_id: 1.0 for doc_id in self.relevant_doc_ids}


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    relevance_mode: str
    queries: int
    k: int
    metrics: RankingMetrics


def _metrics(
    rankings: list[list[str]],
    qrels: list[dict[str, float]],
    k: int,
) -> RankingMetrics:
    relevant_sets = [{doc_id for doc_id, grade in grades.items() if grade > 0} for grades in qrels]
    recalls = [
        recall_at_k(ranking, relevant, k) for ranking, relevant in zip(rankings, relevant_sets)
    ]
    reciprocal_ranks = [
        reciprocal_rank(ranking, relevant) for ranking, relevant in zip(rankings, relevant_sets)
    ]
    ndcgs = [ndcg_at_k(ranking, grades, k) for ranking, grades in zip(rankings, qrels)]
    return RankingMetrics(
        recall_at_k=mean(recalls) if recalls else 0.0,
        mrr=mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        ndcg_at_k=mean(ndcgs) if ndcgs else 0.0,
    )


def benchmark(
    documents: Iterable[Document],
    queries: Iterable[QueryCase],
    *,
    k: int = 10,
) -> list[BenchmarkResult]:
    engine = HybridSearchEngine(documents)
    query_rows = list(queries)
    modes = {query.relevance_mode for query in query_rows}
    if len(modes) > 1:
        raise ValueError("a benchmark cannot mix binary and graded qrels")
    relevance_mode = next(iter(modes), "binary")
    qrels = [query.grades for query in query_rows]

    lexical_rankings = [
        [hit.doc_id for hit in engine.bm25.search(query.text, k=k)] for query in query_rows
    ]
    dense_rankings = [
        [hit.doc_id for hit in engine.dense.search(query.embedding, k=k)] for query in query_rows
    ]
    hybrid_rankings = [
        [
            hit.doc_id
            for hit in engine.search(
                query_text=query.text,
                query_embedding=query.embedding,
                k=k,
                candidate_k=max(20, k * 4),
            )
        ]
        for query in query_rows
    ]

    return [
        BenchmarkResult(
            "bm25", relevance_mode, len(query_rows), k, _metrics(lexical_rankings, qrels, k)
        ),
        BenchmarkResult(
            "dense", relevance_mode, len(query_rows), k, _metrics(dense_rankings, qrels, k)
        ),
        BenchmarkResult(
            "hybrid_rrf",
            relevance_mode,
            len(query_rows),
            k,
            _metrics(hybrid_rankings, qrels, k),
        ),
    ]
