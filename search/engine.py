from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import log

import numpy as np

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
ACCESS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_ACCESS_GROUPS = 64


def _validate_access_id(name: str, value: object) -> str:
    if not isinstance(value, str) or not ACCESS_ID_RE.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


@dataclass(frozen=True)
class DocumentAccess:
    """Server-owned access metadata attached to one indexed document."""

    visibility: str = "public"
    tenant_id: str | None = None
    groups: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.visibility not in {"public", "tenant"}:
            raise ValueError("visibility must be public or tenant")
        if not isinstance(self.groups, frozenset):
            raise TypeError("document access groups must be a frozenset")
        if len(self.groups) > MAX_ACCESS_GROUPS:
            raise ValueError(f"document access groups cannot exceed {MAX_ACCESS_GROUPS}")
        for group in self.groups:
            _validate_access_id("document access group", group)
        if self.visibility == "public":
            if self.tenant_id is not None or self.groups:
                raise ValueError("public documents cannot declare tenant or group restrictions")
        elif self.tenant_id is None:
            raise ValueError("tenant documents require tenant_id")
        else:
            _validate_access_id("document tenant_id", self.tenant_id)


@dataclass(frozen=True)
class AccessScope:
    """Trusted retrieval scope derived from authenticated request context."""

    tenant_id: str
    groups: frozenset[str] = field(default_factory=frozenset)
    include_public: bool = True

    def __post_init__(self) -> None:
        _validate_access_id("scope tenant_id", self.tenant_id)
        if not isinstance(self.groups, frozenset):
            raise TypeError("scope groups must be a frozenset")
        if len(self.groups) > MAX_ACCESS_GROUPS:
            raise ValueError(f"scope groups cannot exceed {MAX_ACCESS_GROUPS}")
        for group in self.groups:
            _validate_access_id("scope group", group)
        if not isinstance(self.include_public, bool):
            raise TypeError("include_public must be a boolean")


def _is_authorized(access: DocumentAccess, scope: AccessScope | None) -> bool:
    if access.visibility == "public":
        return scope is None or scope.include_public
    if scope is None or access.tenant_id != scope.tenant_id:
        return False
    return not access.groups or bool(access.groups & scope.groups)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    embedding: tuple[float, ...]
    metadata: dict[str, str] | None = None
    access: DocumentAccess | None = None


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
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens) if self.tokens else 0.0
        )

    @staticmethod
    def _idf(term: str, *, n: int, document_frequency: Counter[str]) -> float:
        df = document_frequency.get(term, 0)
        return log(1.0 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        allowed_doc_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be positive")
        query_terms = tokenize(query)
        active_rows = [
            (document, terms, frequencies)
            for document, terms, frequencies in zip(
                self.documents,
                self.tokens,
                self.term_frequencies,
            )
            if allowed_doc_ids is None or document.doc_id in allowed_doc_ids
        ]
        active_frequency: Counter[str] = Counter()
        for _, terms, _ in active_rows:
            active_frequency.update(set(terms))
        active_average_length = (
            sum(len(terms) for _, terms, _ in active_rows) / len(active_rows)
            if active_rows
            else 0.0
        )
        scored: list[tuple[str, float]] = []
        for document, terms, frequencies in active_rows:
            score = 0.0
            document_length = len(terms)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / max(active_average_length, 1e-12)
                )
                score += (
                    self._idf(
                        term,
                        n=len(active_rows),
                        document_frequency=active_frequency,
                    )
                    * (frequency * (self.k1 + 1.0))
                    / denominator
                )
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

    def search(
        self,
        query_embedding: Iterable[float],
        k: int = 10,
        *,
        allowed_doc_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be positive")
        query = np.asarray(list(query_embedding), dtype=float)
        if query.ndim != 1 or len(query) != self.dimension:
            raise ValueError(f"query embedding must have dimension {self.dimension}")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        active_indices = [
            index
            for index, document in enumerate(self.documents)
            if allowed_doc_ids is None or document.doc_id in allowed_doc_ids
        ]
        if not active_indices:
            return []
        scores = self.matrix[active_indices] @ query
        ordered = sorted(
            zip(active_indices, scores),
            key=lambda item: (-float(item[1]), self.documents[item[0]].doc_id),
        )[:k]
        return [
            SearchHit(
                doc_id=self.documents[index].doc_id,
                score=float(score),
                rank=rank,
                source="dense",
            )
            for rank, (index, score) in enumerate(ordered, start=1)
        ]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[SearchHit]],
    *,
    k: int = 10,
    constant: int = 60,
) -> list[SearchHit]:
    scores: defaultdict[str, float] = defaultdict(float)
    score_tiebreakers: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        materialized = list(ranking)
        scale = max((abs(hit.score) for hit in materialized), default=0.0)
        for hit in materialized:
            scores[hit.doc_id] += 1.0 / (constant + hit.rank)
            if scale > 0:
                score_tiebreakers[hit.doc_id] += hit.score / scale
    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], -score_tiebreakers[item[0]], item[0]),
    )[:k]
    return [
        SearchHit(doc_id=doc_id, score=score, rank=rank, source="rrf")
        for rank, (doc_id, score) in enumerate(ordered, start=1)
    ]


class HybridSearchEngine:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if any(not isinstance(document.access, DocumentAccess) for document in self.documents):
            raise TypeError("every document must declare a valid DocumentAccess")
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
        scope: AccessScope | None = None,
    ) -> list[SearchHit]:
        if k < 1 or candidate_k < 1:
            raise ValueError("k and candidate_k must be positive")
        allowed_doc_ids = frozenset(
            document.doc_id for document in self.documents if _is_authorized(document.access, scope)
        )
        lexical = self.bm25.search(
            query_text,
            k=candidate_k,
            allowed_doc_ids=allowed_doc_ids,
        )
        dense = self.dense.search(
            query_embedding,
            k=candidate_k,
            allowed_doc_ids=allowed_doc_ids,
        )
        return reciprocal_rank_fusion((lexical, dense), k=k)

    def documents_for_hits(
        self,
        hits: Iterable[SearchHit],
        *,
        scope: AccessScope | None = None,
    ) -> list[dict]:
        materialized: list[dict] = []
        for hit in hits:
            try:
                document = self.by_id[hit.doc_id]
            except KeyError as exc:
                raise ValueError("search hit references an unknown document") from exc
            if not _is_authorized(document.access, scope):
                raise ValueError("search hit is outside the active access scope")
            materialized.append(
                {
                    "doc_id": hit.doc_id,
                    "score": hit.score,
                    "rank": hit.rank,
                    "source": hit.source,
                    "text": document.text,
                    "metadata": document.metadata or {},
                }
            )
        return materialized
