from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np

MAX_RELEVANCE_GRADE = 10.0


def cosine_search(query: np.ndarray, documents: np.ndarray, k: int = 10) -> list[int]:
    query = query / (np.linalg.norm(query) + 1e-12)
    docs = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-12)
    scores = docs @ query
    return np.argsort(-scores)[:k].tolist()


def recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _normalized_grades(relevance: set[Any] | Mapping[Any, float]) -> dict[Any, float]:
    if isinstance(relevance, Mapping):
        grades = {doc_id: float(grade) for doc_id, grade in relevance.items()}
    else:
        grades = {doc_id: 1.0 for doc_id in relevance}
    for grade in grades.values():
        if not isfinite(grade) or not 0.0 <= grade <= MAX_RELEVANCE_GRADE:
            raise ValueError(
                f"relevance grades must be finite and between 0 and {MAX_RELEVANCE_GRADE:g}"
            )
    return grades


def ndcg_at_k(
    ranking: list[Any],
    relevance: set[Any] | Mapping[Any, float],
    k: int,
) -> float:
    """Compute linear-gain NDCG for binary sets or graded relevance mappings."""

    grades = _normalized_grades(relevance)
    gains = [grades.get(doc, 0.0) for doc in ranking[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def evaluate(queries, documents, qrels, k: int = 10) -> dict[str, float]:
    recalls, rrs, ndcgs = [], [], []
    for qid, query in enumerate(queries):
        ranking = cosine_search(query, documents, k)
        judgments = qrels[qid]
        if isinstance(judgments, Mapping):
            grades = _normalized_grades(judgments)
            relevant = {doc_id for doc_id, grade in grades.items() if grade > 0}
        else:
            relevant = set(judgments)
            grades = relevant
        recalls.append(recall_at_k(ranking, relevant, k))
        rrs.append(reciprocal_rank(ranking, relevant))
        ndcgs.append(ndcg_at_k(ranking, grades, k))
    return {
        "recall@k": float(np.mean(recalls)),
        "mrr": float(np.mean(rrs)),
        "ndcg@k": float(np.mean(ndcgs)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    docs = rng.normal(size=(100, 32))
    queries = docs[[5, 22, 81]] + rng.normal(scale=0.05, size=(3, 32))
    print(evaluate(queries, docs, {0: {5}, 1: {22}, 2: {81}}, k=10))
