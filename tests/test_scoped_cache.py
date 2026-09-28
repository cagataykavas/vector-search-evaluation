from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from search.engine import Document, HybridSearchEngine
from search.scoped_cache import (
    CacheContractError,
    CachedHit,
    RetrievalRequest,
    RetrievalScope,
    ScopedRetrievalCache,
    cache_key,
)

INDEX = "a" * 64
CONFIG = "b" * 64


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def request(**updates: object) -> RetrievalRequest:
    values = {
        "query_text": "vpn outage runbook",
        "query_embedding": (0.4, 0.6),
        "scope": RetrievalScope("tenant-a", frozenset({"operators", "emea"})),
        "index_revision": INDEX,
        "retriever_config_digest": CONFIG,
        "embedding_model_revision": "embed-v3",
        "k": 2,
        "candidate_k": 20,
    }
    values.update(updates)
    return RetrievalRequest(**values)  # type: ignore[arg-type]


def hits() -> tuple[CachedHit, ...]:
    return (
        CachedHit("doc-1", 0.9, 1, "rrf"),
        CachedHit("doc-2", 0.7, 2, "rrf"),
    )


def test_put_and_get_exact_binding() -> None:
    cache = ScopedRetrievalCache()
    key = cache.put(request(), hits())
    result = cache.get(request())
    assert result.status == "hit"
    assert result.key_digest == key
    assert result.hits == hits()
    assert cache.snapshot().hits == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"query_text": "different query"},
        {"query_embedding": (0.6, 0.4)},
        {"scope": RetrievalScope("tenant-b")},
        {"scope": RetrievalScope("tenant-a", frozenset({"operators"}))},
        {"scope": RetrievalScope()},
        {"index_revision": "c" * 64},
        {"retriever_config_digest": "d" * 64},
        {"embedding_model_revision": "embed-v4"},
        {"k": 1},
        {"candidate_k": 30},
    ],
)
def test_every_retrieval_input_is_cache_keyed(changed) -> None:
    cache = ScopedRetrievalCache()
    cache.put(request(), hits())
    assert cache.get(request(**changed)).status == "miss"


def test_group_order_does_not_change_key() -> None:
    first = request(scope=RetrievalScope("tenant-a", frozenset({"a", "b"})))
    second = request(scope=RetrievalScope("tenant-a", frozenset({"b", "a"})))
    assert cache_key(first) == cache_key(second)


def test_key_does_not_disclose_query_or_scope() -> None:
    item = request(query_text="customer-secret", scope=RetrievalScope("private-tenant"))
    key = cache_key(item)
    assert len(key) == 64
    assert "customer" not in key
    assert "tenant" not in key


def test_embedding_uses_exact_ieee_values_and_dimension() -> None:
    base = cache_key(request(query_embedding=(0.0, -0.0)))
    signed_zero = cache_key(request(query_embedding=(-0.0, -0.0)))
    extra_dimension = cache_key(request(query_embedding=(0.0, -0.0, 0.0)))
    assert len({base, signed_zero, extra_dimension}) == 3


def test_ttl_expiry_is_fail_closed() -> None:
    clock = Clock()
    cache = ScopedRetrievalCache(ttl_seconds=5, clock=clock)
    cache.put(request(), hits())
    clock.value = 105.0
    result = cache.get(request())
    assert result.status == "expired"
    assert result.hits == ()
    assert cache.snapshot().expirations == 1


def test_lru_capacity_evicts_oldest_entry() -> None:
    cache = ScopedRetrievalCache(capacity=2)
    first = request(query_text="first")
    second = request(query_text="second")
    third = request(query_text="third")
    cache.put(first, hits())
    cache.put(second, hits())
    assert cache.get(first).status == "hit"
    cache.put(third, hits())
    assert cache.get(second).status == "miss"
    assert cache.get(first).status == "hit"
    assert cache.snapshot().evictions == 1


def test_replacing_same_key_does_not_evict() -> None:
    cache = ScopedRetrievalCache(capacity=1)
    cache.put(request(), hits())
    cache.put(request(), (CachedHit("doc-3", 0.5, 1, "rrf"),))
    assert cache.get(request()).hits[0].doc_id == "doc-3"
    assert cache.snapshot().evictions == 0


def test_index_invalidation_is_revision_scoped() -> None:
    cache = ScopedRetrievalCache()
    first = request(query_text="one")
    second = request(query_text="two", index_revision="c" * 64)
    cache.put(first, hits())
    cache.put(second, hits())
    assert cache.invalidate_index(INDEX) == 1
    assert cache.get(first).status == "miss"
    assert cache.get(second).status == "hit"


def test_clear_returns_removed_entry_count() -> None:
    cache = ScopedRetrievalCache()
    cache.put(request(query_text="one"), hits())
    cache.put(request(query_text="two"), hits())
    assert cache.clear() == 2
    assert cache.snapshot().entries == 0


def test_anonymous_scope_cannot_claim_groups() -> None:
    with pytest.raises(CacheContractError, match="anonymous"):
        RetrievalScope(group_ids=frozenset({"admins"}))


@pytest.mark.parametrize(
    "updates",
    [
        {"query_text": ""},
        {"query_text": "bad\x00query"},
        {"query_embedding": ()},
        {"query_embedding": (math.nan,)},
        {"query_embedding": (math.inf,)},
        {"index_revision": "A" * 64},
        {"retriever_config_digest": "short"},
        {"embedding_model_revision": "bad value"},
        {"k": True},
        {"k": 0},
        {"candidate_k": 1},
    ],
)
def test_invalid_requests_fail_closed(updates) -> None:
    with pytest.raises(CacheContractError):
        request(**updates)


def test_cached_hits_require_unique_contiguous_ranking() -> None:
    cache = ScopedRetrievalCache()
    with pytest.raises(CacheContractError, match="unique"):
        cache.put(
            request(),
            (CachedHit("doc-1", 1.0, 1, "rrf"), CachedHit("doc-1", 0.5, 2, "rrf")),
        )
    with pytest.raises(CacheContractError, match="contiguous"):
        cache.put(request(), (CachedHit("doc-1", 1.0, 2, "rrf"),))


def test_cached_hits_cannot_exceed_requested_k() -> None:
    cache = ScopedRetrievalCache()
    with pytest.raises(CacheContractError, match="count"):
        cache.put(request(k=1), hits())


def test_invalid_hit_fields_fail_closed() -> None:
    with pytest.raises(CacheContractError):
        CachedHit("bad id", 1.0, 1, "rrf")
    with pytest.raises(CacheContractError):
        CachedHit("doc-1", math.nan, 1, "rrf")
    with pytest.raises(CacheContractError):
        CachedHit("doc-1", 1.0, 1, "unknown")


def test_cache_configuration_is_bounded() -> None:
    with pytest.raises(ValueError):
        ScopedRetrievalCache(capacity=0)
    with pytest.raises(ValueError):
        ScopedRetrievalCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        ScopedRetrievalCache(ttl_seconds=math.inf)


def test_non_finite_clock_fails_closed() -> None:
    cache = ScopedRetrievalCache(clock=lambda: math.nan)
    with pytest.raises(RuntimeError, match="clock"):
        cache.get(request())
    with pytest.raises(RuntimeError, match="clock"):
        cache.put(request(), hits())


def test_thread_safe_reads_and_writes_preserve_capacity() -> None:
    cache = ScopedRetrievalCache(capacity=16)

    def operation(index: int) -> str:
        item = request(query_text=f"query-{index % 32}")
        cache.put(item, (CachedHit(f"doc-{index}", float(index), 1, "rrf"),))
        return cache.get(item).status

    with ThreadPoolExecutor(max_workers=16) as executor:
        statuses = list(executor.map(operation, range(2_000)))
    assert set(statuses) <= {"hit", "miss"}
    assert cache.snapshot().entries <= 16


def test_real_hybrid_results_round_trip_without_document_text() -> None:
    documents = [
        Document("vpn", "vpn outage runbook", (1.0, 0.0)),
        Document("mail", "mail delivery guide", (0.0, 1.0)),
    ]
    engine = HybridSearchEngine(documents)
    actual = engine.search(query_text="vpn outage", query_embedding=(1.0, 0.0), k=2, candidate_k=2)
    cached = tuple(CachedHit(hit.doc_id, hit.score, hit.rank, hit.source) for hit in actual)
    cache = ScopedRetrievalCache()
    item = request(k=2, candidate_k=2)
    cache.put(item, cached)
    result = cache.get(item)
    assert [hit.doc_id for hit in result.hits] == [hit.doc_id for hit in actual]
    assert all(not hasattr(hit, "text") for hit in result.hits)
