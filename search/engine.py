from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
import re
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    embedding: tuple[float, ...]
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    rank: int
    source: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document.text) for document in self.documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))
        self.average_document_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens)
            if self.tokens
            else 0.0
        )

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.document_frequency.get(term, 0)
        return log(1.0 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        query_terms = tokenize(query)
        scored: list[tuple[str, float]] = []
        for document, terms, frequencies in zip(
            self.documents,
            self.tokens,
            self.term_frequencies,
        ):
            score = 0.0
            document_length = len(terms)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * document_length
                    / max(self.average_document_length, 1e-12)
                )
                score += self._idf(term) * (frequency * (self.k1 + 1.0)) / denominator
            if score > 0:
                scored.append((document.doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(doc_id=doc_id, score=float(score), rank=index + 1, source="bm25")
            for index, (doc_id, score) in enumerate(scored[:k])
        ]


class DenseIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if self.documents:
            dimensions = {len(document.embedding) for document in self.documents}
            if len(dimensions) != 1:
                raise ValueError("all document embeddings must have the same dimension")
            matrix = np.asarray([document.embedding for document in self.documents], dtype=float)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self.matrix = matrix / np.clip(norms, 1e-12, None)
            self.dimension = matrix.shape[1]
        else:
            self.matrix = np.empty((0, 0), dtype=float)
            self.dimension = 0

    def search(self, query_embedding: Iterable[float], k: int = 10) -> list[SearchHit]:
        query = np.asarray(list(query_embedding), dtype=float)
        if query.ndim != 1 or len(query) != self.dimension:
            raise ValueError(f"query embedding must have dimension {self.dimension}")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.matrix @ query
        order = np.argsort(-scores)[:k]
        return [
            SearchHit(
                doc_id=self.documents[int(index)].doc_id,
                score=float(scores[int(index)]),
                rank=rank,
                source="dense",
            )
            for rank, index in enumerate(order, start=1)
        ]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[SearchHit]],
    *,
    k: int = 10,
    constant: int = 60,
) -> list[SearchHit]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for hit in ranking:
            scores[hit.doc_id] += 1.0 / (constant + hit.rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [
        SearchHit(doc_id=doc_id, score=score, rank=rank, source="rrf")
        for rank, (doc_id, score) in enumerate(ordered, start=1)
    ]


class HybridSearchEngine:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.by_id = {document.doc_id: document for document in self.documents}
        if len(self.by_id) != len(self.documents):
            raise ValueError("document IDs must be unique")
        self.bm25 = BM25Index(self.documents)
        self.dense = DenseIndex(self.documents)

    def search(
        self,
        *,
        query_text: str,
        query_embedding: Iterable[float],
        k: int = 10,
        candidate_k: int = 50,
    ) -> list[SearchHit]:
        lexical = self.bm25.search(query_text, k=candidate_k)
        dense = self.dense.search(query_embedding, k=candidate_k)
        return reciprocal_rank_fusion((lexical, dense), k=k)

    def documents_for_hits(self, hits: Iterable[SearchHit]) -> list[dict]:
        return [
            {
                "doc_id": hit.doc_id,
                "score": hit.score,
                "rank": hit.rank,
                "source": hit.source,
                "text": self.by_id[hit.doc_id].text,
                "metadata": self.by_id[hit.doc_id].metadata or {},
            }
            for hit in hits
        ]
