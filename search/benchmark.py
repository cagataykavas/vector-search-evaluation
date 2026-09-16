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
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    queries: int
    k: int
    metrics: RankingMetrics


def _metrics(rankings: list[list[str]], qrels: list[set[str]], k: int) -> RankingMetrics:
    recalls = [recall_at_k(ranking, relevant, k) for ranking, relevant in zip(rankings, qrels)]
    reciprocal_ranks = [reciprocal_rank(ranking, relevant) for ranking, relevant in zip(rankings, qrels)]
    ndcgs = [ndcg_at_k(ranking, relevant, k) for ranking, relevant in zip(rankings, qrels)]
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
    qrels = [set(query.relevant_doc_ids) for query in query_rows]

    lexical_rankings = [
        [hit.doc_id for hit in engine.bm25.search(query.text, k=k)]
        for query in query_rows
    ]
    dense_rankings = [
        [hit.doc_id for hit in engine.dense.search(query.embedding, k=k)]
        for query in query_rows
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
        BenchmarkResult("bm25", len(query_rows), k, _metrics(lexical_rankings, qrels, k)),
        BenchmarkResult("dense", len(query_rows), k, _metrics(dense_rankings, qrels, k)),
        BenchmarkResult("hybrid_rrf", len(query_rows), k, _metrics(hybrid_rankings, qrels, k)),
    ]
