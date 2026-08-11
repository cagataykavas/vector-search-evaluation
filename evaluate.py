from __future__ import annotations

import numpy as np


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


def ndcg_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    gains = [1.0 if doc in relevant else 0.0 for doc in ranking[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = [1.0] * min(len(relevant), k)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def evaluate(queries, documents, qrels, k: int = 10) -> dict[str, float]:
    recalls, rrs, ndcgs = [], [], []
    for qid, query in enumerate(queries):
        ranking = cosine_search(query, documents, k)
        relevant = set(qrels[qid])
        recalls.append(recall_at_k(ranking, relevant, k))
        rrs.append(reciprocal_rank(ranking, relevant))
        ndcgs.append(ndcg_at_k(ranking, relevant, k))
    return {"recall@k": float(np.mean(recalls)), "mrr": float(np.mean(rrs)), "ndcg@k": float(np.mean(ndcgs))}


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    docs = rng.normal(size=(100, 32))
    queries = docs[[5, 22, 81]] + rng.normal(scale=0.05, size=(3, 32))
    print(evaluate(queries, docs, {0: {5}, 1: {22}, 2: {81}}, k=10))
