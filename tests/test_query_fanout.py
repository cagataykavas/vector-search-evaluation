from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace

import pytest

from search.engine import Document, HybridSearchEngine, SearchHit
from search.fanout import (
    AuthorizationScope,
    FanoutAdmissionError,
    FanoutInputError,
    FanoutPolicy,
    QueryVariant,
    hybrid_engine_retriever,
    retrieve_with_query_fanout,
)

DEFAULT_SCOPE = AuthorizationScope(
    tenant_id="tenant-a",
    principal_id="user-a",
    resource_id="knowledge-base-a",
    grants=("documents:read",),
    policy_version="acl-v1",
)


def variant(
    variant_id: str,
    text: str,
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    weight: float = 1.0,
    scope: AuthorizationScope | None = DEFAULT_SCOPE,
) -> QueryVariant:
    return QueryVariant(variant_id, text, embedding, weight, scope)


def hits(*doc_ids: str) -> list[SearchHit]:
    return [
        SearchHit(doc_id, 1.0 / rank, rank, "test") for rank, doc_id in enumerate(doc_ids, start=1)
    ]


def run(coro):
    return asyncio.run(coro)


def test_fuses_expansions_with_document_level_deduplication() -> None:
    rankings = {
        "primary": hits("alpha", "beta", "gamma"),
        "rewrite": hits("beta", "delta", "alpha"),
    }

    async def retriever(query: QueryVariant, candidate_k: int) -> list[SearchHit]:
        assert candidate_k == 3
        return rankings[query.variant_id]

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "original"), variant("rewrite", "expanded")],
            retriever,
            policy=FanoutPolicy(top_k=3, candidate_k=3, min_successful_queries=2),
        )
    )

    assert [hit.doc_id for hit in result.hits] == ["beta", "alpha", "delta"]
    assert result.hits[0].rank == 1
    assert [item.variant_id for item in result.hits[0].contributions] == [
        "primary",
        "rewrite",
    ]
    assert result.successful_queries == 2
    assert result.failed_queries == result.timed_out_queries == 0


def test_query_weight_changes_fusion_without_mutating_source_rank() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        return hits("a", "b") if query.variant_id == "primary" else hits("b", "a")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "first"), variant("rewrite", "second", weight=2.0)],
            retriever,
            policy=FanoutPolicy(top_k=2, candidate_k=2, min_successful_queries=2),
        )
    )

    assert [hit.doc_id for hit in result.hits] == ["b", "a"]
    rewrite_contribution = result.hits[0].contributions[1]
    assert rewrite_contribution.source_rank == 1
    assert rewrite_contribution.weighted_rrf == pytest.approx(2 / 61)


def test_completion_order_does_not_change_ranking_or_evidence_order() -> None:
    async def retrieve_with_delay(query: QueryVariant, _: int) -> list[SearchHit]:
        await asyncio.sleep(0.02 if query.variant_id == "primary" else 0)
        return hits("a", "b") if query.variant_id == "primary" else hits("b", "a")

    queries = [variant("primary", "first"), variant("fast", "second")]
    policy = FanoutPolicy(top_k=2, candidate_k=2, min_successful_queries=2)
    first = run(retrieve_with_query_fanout(queries, retrieve_with_delay, policy=policy))
    second = run(retrieve_with_query_fanout(queries, retrieve_with_delay, policy=policy))

    assert first.hits == second.hits
    assert first.evidence == second.evidence
    assert [row.variant_id for row in first.evidence] == ["primary", "fast"]


def test_partial_expansion_failure_is_reported_without_leaking_message() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id == "bad":
            raise RuntimeError("secret backend hostname and token")
        return hits("safe")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "first"), variant("bad", "second")],
            retriever,
            policy=FanoutPolicy(top_k=1, candidate_k=1),
        )
    )

    assert [hit.doc_id for hit in result.hits] == ["safe"]
    assert result.failed_queries == 1
    assert result.evidence[1].reason_code == "retriever_error"
    assert "secret" not in repr(result)


def test_success_quorum_rejects_too_many_failed_expansions() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id != "primary":
            raise ConnectionError("down")
        return hits("safe")

    with pytest.raises(FanoutAdmissionError, match="insufficient_successful_queries"):
        run(
            retrieve_with_query_fanout(
                [variant("primary", "one"), variant("rewrite", "two")],
                retriever,
                policy=FanoutPolicy(
                    top_k=1,
                    candidate_k=1,
                    min_successful_queries=2,
                ),
            )
        )


def test_primary_failure_is_fail_closed_even_when_expansion_succeeds() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id == "primary":
            raise ConnectionError("down")
        return hits("fallback")

    with pytest.raises(FanoutAdmissionError, match="primary_query_failed"):
        run(
            retrieve_with_query_fanout(
                [variant("primary", "one"), variant("rewrite", "two")],
                retriever,
                policy=FanoutPolicy(top_k=1, candidate_k=1),
            )
        )


def test_primary_requirement_can_be_disabled_explicitly() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id == "primary":
            raise ConnectionError("down")
        return hits("fallback")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "one"), variant("rewrite", "two")],
            retriever,
            policy=FanoutPolicy(
                top_k=1,
                candidate_k=1,
                require_primary=False,
            ),
        )
    )

    assert result.hits[0].doc_id == "fallback"


def test_global_deadline_marks_slow_expansion_and_cancels_it() -> None:
    cancelled = asyncio.Event()

    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id == "primary":
            return hits("safe")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return hits("late")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "one"), variant("slow", "two")],
            retriever,
            policy=FanoutPolicy(top_k=1, candidate_k=1, deadline_ms=20),
        )
    )

    assert result.timed_out_queries == 1
    assert result.evidence[1].status == "timeout"
    assert cancelled.is_set()


def test_blocking_sync_retriever_does_not_block_primary_or_escape_deadline() -> None:
    def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        if query.variant_id == "slow":
            time.sleep(0.05)
            return hits("late")
        return hits("safe")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "one"), variant("slow", "two")],
            retriever,
            policy=FanoutPolicy(
                top_k=1,
                candidate_k=1,
                max_concurrency=2,
                deadline_ms=20,
            ),
        )
    )

    assert result.hits[0].doc_id == "safe"
    assert result.timed_out_queries == 1


def test_concurrency_is_bounded() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.005)
        async with lock:
            active -= 1
        return hits(query.variant_id)

    queries = [variant(f"q-{index}", f"query {index}") for index in range(6)]
    result = run(
        retrieve_with_query_fanout(
            queries,
            retriever,
            policy=FanoutPolicy(
                top_k=6,
                candidate_k=6,
                max_queries=6,
                max_concurrency=2,
                min_successful_queries=6,
            ),
        )
    )

    assert result.successful_queries == 6
    assert peak == 2


@pytest.mark.parametrize(
    ("queries", "code"),
    [
        ([], "missing_queries"),
        ([variant("bad id", "q")], "invalid_variant_id"),
        ([variant("q", "   ")], "empty_query"),
        (
            [variant("q1", "Same Query"), variant("q2", " same   query ")],
            "duplicate_query",
        ),
        (
            [variant("q1", "one"), variant("q1", "two")],
            "duplicate_variant_id",
        ),
        ([variant("q", "query", embedding=())], "empty_embedding"),
        ([variant("q", "query", embedding=(math.nan, 0.0))], "non_finite_embedding"),
        (
            [variant("q1", "one"), variant("q2", "two", embedding=(1.0,))],
            "embedding_dimension_mismatch",
        ),
        ([variant("q", "query", weight=0)], "invalid_query_weight"),
    ],
)
def test_rejects_malformed_queries(queries: list[QueryVariant], code: str) -> None:
    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        return []

    with pytest.raises(FanoutInputError, match=code):
        run(retrieve_with_query_fanout(queries, retriever))


def test_rejects_query_and_quorum_budgets_before_retrieval() -> None:
    called = False

    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        nonlocal called
        called = True
        return []

    queries = [variant("q1", "one"), variant("q2", "two")]
    with pytest.raises(FanoutInputError, match="query_budget_exceeded"):
        run(
            retrieve_with_query_fanout(
                queries,
                retriever,
                policy=FanoutPolicy(max_queries=1),
            )
        )
    with pytest.raises(FanoutInputError, match="unreachable_success_policy"):
        run(
            retrieve_with_query_fanout(
                queries,
                retriever,
                policy=FanoutPolicy(max_queries=3, min_successful_queries=3),
            )
        )
    assert not called


def test_rejects_authorization_scope_drift_before_retrieval() -> None:
    called = False

    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        nonlocal called
        called = True
        return []

    changed_scope = replace(DEFAULT_SCOPE, tenant_id="tenant-b")

    with pytest.raises(FanoutInputError, match="authorization_scope_mismatch"):
        run(
            retrieve_with_query_fanout(
                [
                    variant("primary", "one"),
                    variant("rewrite", "two", scope=changed_scope),
                ],
                retriever,
            )
        )

    assert not called


@pytest.mark.parametrize(
    ("scope", "code"),
    [
        (None, "invalid_authorization_scope"),
        (replace(DEFAULT_SCOPE, tenant_id=""), "invalid_scope_tenant"),
        (replace(DEFAULT_SCOPE, principal_id="bad principal"), "invalid_scope_principal"),
        (replace(DEFAULT_SCOPE, resource_id=""), "invalid_scope_resource"),
        (replace(DEFAULT_SCOPE, policy_version="latest policy"), "invalid_scope_policy_version"),
        (replace(DEFAULT_SCOPE, grants=()), "invalid_scope_grants"),
        (
            replace(DEFAULT_SCOPE, grants=("documents:read", "documents:read")),
            "duplicate_scope_grant",
        ),
        (replace(DEFAULT_SCOPE, grants=("bad grant",)), "invalid_scope_grant"),
        (
            replace(
                DEFAULT_SCOPE,
                grants=tuple(f"grant-{index}" for index in range(33)),
            ),
            "scope_grant_budget_exceeded",
        ),
        (
            replace(
                DEFAULT_SCOPE,
                grants=tuple(f"g-{index:02d}-" + "a" * 120 for index in range(32)),
            ),
            "authorization_scope_too_large",
        ),
    ],
)
def test_rejects_malformed_authorization_scope(
    scope: AuthorizationScope | None,
    code: str,
) -> None:
    called = False

    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        nonlocal called
        called = True
        return []

    with pytest.raises(FanoutInputError, match=code):
        run(retrieve_with_query_fanout([variant("primary", "one", scope=scope)], retriever))

    assert not called


def test_scope_digest_is_order_independent_and_redacts_claims() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        assert query.authorization_scope is not None
        return hits("safe")

    sensitive_scope = AuthorizationScope(
        tenant_id="tenant-secret-8271",
        principal_id="principal-secret-8271",
        resource_id="resource-secret-8271",
        grants=("documents:read", "finance:read"),
        policy_version="acl-v1",
    )
    reordered_scope = replace(
        sensitive_scope,
        grants=("finance:read", "documents:read"),
    )

    first = run(
        retrieve_with_query_fanout(
            [variant("primary", "one", scope=sensitive_scope)],
            retriever,
        )
    )
    reordered = run(
        retrieve_with_query_fanout(
            [variant("primary", "one", scope=reordered_scope)],
            retriever,
        )
    )

    assert first.authorization_scope_digest == reordered.authorization_scope_digest
    assert first.request_digest == reordered.request_digest
    assert "secret-8271" not in repr(first)


def test_authorization_scope_changes_request_and_query_evidence_identity() -> None:
    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        return hits("safe")

    first = run(
        retrieve_with_query_fanout(
            [variant("primary", "one")],
            retriever,
        )
    )
    changed = run(
        retrieve_with_query_fanout(
            [
                variant(
                    "primary",
                    "one",
                    scope=replace(DEFAULT_SCOPE, principal_id="user-b"),
                )
            ],
            retriever,
        )
    )

    assert first.authorization_scope_digest != changed.authorization_scope_digest
    assert first.request_digest != changed.request_digest
    assert first.evidence[0].query_digest != changed.evidence[0].query_digest


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        (
            [SearchHit("a", 1.0, 1, "x"), SearchHit("a", 0.5, 2, "x")],
            "duplicate_document_id",
        ),
        ([SearchHit("a", 1.0, 2, "x")], "invalid_retriever_rank"),
        ([SearchHit("a", math.inf, 1, "x")], "non_finite_retriever_score"),
        ([SearchHit("a\n", 1.0, 1, "x")], "invalid_document_id"),
        ([SearchHit("a", 1.0, 1, "")], "missing_retriever_source"),
    ],
)
def test_malformed_expansion_results_are_quarantined(
    returned: list[SearchHit],
    reason: str,
) -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        return hits("safe") if query.variant_id == "primary" else returned

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "one"), variant("bad", "two")],
            retriever,
            policy=FanoutPolicy(top_k=1, candidate_k=2),
        )
    )

    assert result.hits[0].doc_id == "safe"
    assert result.evidence[1].reason_code == reason


def test_candidate_budget_overflow_is_quarantined() -> None:
    async def retriever(query: QueryVariant, _: int) -> list[SearchHit]:
        return hits("safe") if query.variant_id == "primary" else hits("a", "b", "c")

    result = run(
        retrieve_with_query_fanout(
            [variant("primary", "one"), variant("bad", "two")],
            retriever,
            policy=FanoutPolicy(top_k=1, candidate_k=2),
        )
    )

    assert result.evidence[1].reason_code == "retriever_candidate_budget_exceeded"


def test_request_digest_binds_query_embedding_weight_and_policy() -> None:
    async def retriever(_: QueryVariant, __: int) -> list[SearchHit]:
        return hits("a")

    base = [variant("primary", "sensitive-phrase-8271")]
    first = run(retrieve_with_query_fanout(base, retriever))
    repeated = run(retrieve_with_query_fanout(base, retriever))
    changed_embedding = run(
        retrieve_with_query_fanout(
            [variant("primary", "sensitive-phrase-8271", embedding=(0.0, 1.0))],
            retriever,
        )
    )
    changed_policy = run(
        retrieve_with_query_fanout(
            base,
            retriever,
            policy=FanoutPolicy(rrf_constant=61),
        )
    )

    assert first.request_digest == repeated.request_digest
    assert first.request_digest != changed_embedding.request_digest
    assert first.request_digest != changed_policy.request_digest
    assert "sensitive-phrase-8271" not in repr(first)


def test_real_hybrid_engine_adapter_recovers_expansion_match() -> None:
    engine = HybridSearchEngine(
        [
            Document("timeout", "upstream request deadline", (0.0, 1.0)),
            Document("cache", "cache invalidation procedure", (1.0, 0.0)),
            Document("other", "database backup schedule", (0.5, 0.5)),
        ]
    )
    queries = [
        variant("primary", "request failure", embedding=(0.5, 0.5)),
        variant("domain-term", "upstream timeout deadline", embedding=(0.0, 1.0)),
    ]

    result = run(
        retrieve_with_query_fanout(
            queries,
            hybrid_engine_retriever(engine),
            policy=FanoutPolicy(
                top_k=2,
                candidate_k=3,
                min_successful_queries=2,
            ),
        )
    )

    assert result.hits[0].doc_id == "timeout"
    assert {item.variant_id for item in result.hits[0].contributions} == {
        "primary",
        "domain-term",
    }
