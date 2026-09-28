from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

MAX_QUERY_BYTES = 16 * 1024
MAX_EMBEDDING_DIMENSIONS = 16_384
MAX_GROUPS = 128
MAX_HITS = 1_000
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SOURCES = frozenset({"bm25", "dense", "rrf", "hybrid"})


class CacheContractError(ValueError):
    """A cache request or value violates the serving contract."""


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """Verified authorization scope used to isolate cache entries."""

    tenant_id: str | None = None
    group_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.tenant_id is None and self.group_ids:
            raise CacheContractError("anonymous scope cannot carry group IDs")
        if self.tenant_id is not None:
            _validate_identifier(self.tenant_id, "tenant_id")
        if len(self.group_ids) > MAX_GROUPS:
            raise CacheContractError("group count exceeds the cache-key budget")
        for group_id in self.group_ids:
            _validate_identifier(group_id, "group_id")

    @property
    def mode(self) -> Literal["public", "tenant"]:
        return "public" if self.tenant_id is None else "tenant"


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query_text: str
    query_embedding: tuple[float, ...]
    scope: RetrievalScope
    index_revision: str
    retriever_config_digest: str
    embedding_model_revision: str
    k: int
    candidate_k: int

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise CacheContractError("query_text must not be empty")
        encoded = self.query_text.encode("utf-8")
        if len(encoded) > MAX_QUERY_BYTES:
            raise CacheContractError("query_text exceeds the byte budget")
        if any(ord(character) < 32 and character not in "\t\n" for character in self.query_text):
            raise CacheContractError("query_text contains unsupported control characters")
        if not 1 <= len(self.query_embedding) <= MAX_EMBEDDING_DIMENSIONS:
            raise CacheContractError("query embedding dimension is outside the accepted range")
        if any(
            isinstance(value, bool) or not math.isfinite(value) for value in self.query_embedding
        ):
            raise CacheContractError("query embedding must contain finite numbers")
        _validate_digest(self.index_revision, "index_revision")
        _validate_digest(self.retriever_config_digest, "retriever_config_digest")
        _validate_identifier(self.embedding_model_revision, "embedding_model_revision")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= MAX_HITS:
            raise CacheContractError("k is outside the accepted range")
        if (
            isinstance(self.candidate_k, bool)
            or not isinstance(self.candidate_k, int)
            or not self.k <= self.candidate_k <= MAX_HITS * 10
        ):
            raise CacheContractError("candidate_k must be at least k and within budget")


@dataclass(frozen=True, slots=True)
class CachedHit:
    doc_id: str
    score: float
    rank: int
    source: str

    def __post_init__(self) -> None:
        _validate_identifier(self.doc_id, "doc_id")
        if isinstance(self.score, bool) or not math.isfinite(self.score):
            raise CacheContractError("hit score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise CacheContractError("hit rank must be a positive integer")
        if self.source not in _SOURCES:
            raise CacheContractError("hit source is not allowlisted")


@dataclass(frozen=True, slots=True)
class CacheLookup:
    status: Literal["hit", "miss", "expired"]
    key_digest: str
    hits: tuple[CachedHit, ...] = ()


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    entries: int
    hits: int
    misses: int
    expirations: int
    evictions: int


@dataclass(frozen=True, slots=True)
class _Entry:
    hits: tuple[CachedHit, ...]
    expires_at: float
    index_revision: str


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CacheContractError(f"{name} is invalid")
    return value


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CacheContractError(f"{name} must be a canonical SHA-256 digest")
    return value


def _embedding_digest(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack(">I", len(values)))
    for value in values:
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def _scope_digest(scope: RetrievalScope) -> str:
    payload = {
        "groups": sorted(scope.group_ids),
        "mode": scope.mode,
        "tenant": scope.tenant_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def cache_key(request: RetrievalRequest) -> str:
    """Return a privacy-preserving key bound to every retrieval input."""

    payload = {
        "candidate_k": request.candidate_k,
        "embedding_digest": _embedding_digest(request.query_embedding),
        "embedding_model_revision": request.embedding_model_revision,
        "index_revision": request.index_revision,
        "k": request.k,
        "query_digest": hashlib.sha256(request.query_text.encode("utf-8")).hexdigest(),
        "retriever_config_digest": request.retriever_config_digest,
        "scope_digest": _scope_digest(request.scope),
        "schema_version": 1,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _materialize_hits(hits: Iterable[CachedHit], *, k: int) -> tuple[CachedHit, ...]:
    materialized = tuple(hits)
    if len(materialized) > k or len(materialized) > MAX_HITS:
        raise CacheContractError("cached hit count exceeds the request budget")
    if len({hit.doc_id for hit in materialized}) != len(materialized):
        raise CacheContractError("cached document IDs must be unique")
    if tuple(hit.rank for hit in materialized) != tuple(range(1, len(materialized) + 1)):
        raise CacheContractError("cached hit ranks must be contiguous and ordered")
    return materialized


class ScopedRetrievalCache:
    """Thread-safe bounded TTL/LRU cache for retrieval identities and scores only."""

    def __init__(
        self,
        *,
        capacity: int = 1_024,
        ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 1 <= capacity <= 100_000
        ):
            raise ValueError("capacity must be between 1 and 100000")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or not 0 < ttl_seconds <= 86_400
        ):
            raise ValueError("ttl_seconds must be finite and between 0 and 86400")
        self._capacity = capacity
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0

    def get(self, request: RetrievalRequest) -> CacheLookup:
        key = cache_key(request)
        now = self._clock()
        if not math.isfinite(now):
            raise RuntimeError("cache clock returned a non-finite value")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return CacheLookup("miss", key)
            if entry.expires_at <= now:
                del self._entries[key]
                self._misses += 1
                self._expirations += 1
                return CacheLookup("expired", key)
            self._entries.move_to_end(key)
            self._hits += 1
            return CacheLookup("hit", key, entry.hits)

    def put(self, request: RetrievalRequest, hits: Iterable[CachedHit]) -> str:
        key = cache_key(request)
        materialized = _materialize_hits(hits, k=request.k)
        now = self._clock()
        if not math.isfinite(now):
            raise RuntimeError("cache clock returned a non-finite value")
        entry = _Entry(materialized, now + self._ttl_seconds, request.index_revision)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1
        return key

    def invalidate_index(self, index_revision: str) -> int:
        _validate_digest(index_revision, "index_revision")
        with self._lock:
            keys = [
                key
                for key, entry in self._entries.items()
                if entry.index_revision == index_revision
            ]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            return CacheSnapshot(
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                expirations=self._expirations,
                evictions=self._evictions,
            )
