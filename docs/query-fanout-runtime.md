# Bounded query fan-out runtime

Single-query retrieval is brittle when users omit a domain term, use an acronym, or phrase a
request differently from indexed text. `search.fanout` executes a primary query plus admitted
expansions under one resource budget, then combines document rankings with weighted Reciprocal
Rank Fusion (RRF).

This is a runtime component, not an offline audit. It accepts any sync or async retriever and
ships with `hybrid_engine_retriever()` for the repository's `HybridSearchEngine`.

## Safety and determinism

- At most 16 variants, 16 concurrent calls, 1,000 candidates per call, and a 30 second deadline.
- One monotonic request deadline includes time waiting for the concurrency semaphore.
- The primary query must succeed by default. Expansion failures can degrade gracefully only when
  the configured `min_successful_queries` quorum remains satisfied.
- Duplicate normalized queries, mismatched or non-finite embeddings, invalid weights, and
  oversized text are rejected before any retrieval starts.
- Every variant carries an immutable tenant, principal, resource, grant, and policy-version scope.
  Scope drift is rejected before any retriever starts, and query/request evidence binds only the
  canonical scope SHA-256 rather than exposing authorization claims.
- Each retriever ranking must contain unique document IDs, finite scores, and contiguous ranks.
  A malformed expansion is quarantined and reported with a stable reason code.
- Fusion consumes variants in request order and uses document ID as its final tie-breaker, so
  network completion order cannot change the result.
- Evidence contains SHA-256 bindings, variant IDs, counts, and stable reason codes—not query text,
  embeddings, backend exception messages, or document content.

## Example

```python
import asyncio

from search.fanout import (
    AuthorizationScope,
    FanoutPolicy,
    QueryVariant,
    hybrid_engine_retriever,
    retrieve_with_query_fanout,
)

scope = AuthorizationScope(
    tenant_id="tenant-a",
    principal_id="user-a",
    resource_id="knowledge-base-a",
    grants=("documents:read",),
    policy_version="acl-v1",
)

variants = [
    QueryVariant(
        "primary", "request failure", (0.5, 0.5), weight=1.0, authorization_scope=scope
    ),
    QueryVariant(
        "domain-term", "upstream timeout", (0.0, 1.0), weight=1.0, authorization_scope=scope
    ),
]

result = asyncio.run(
    retrieve_with_query_fanout(
        variants,
        hybrid_engine_retriever(engine),
        policy=FanoutPolicy(
            top_k=5,
            candidate_k=20,
            max_concurrency=2,
            deadline_ms=300,
            min_successful_queries=2,
        ),
    )
)
```

Each `FusedHit` lists the contributing variant ID, its source rank, and weighted RRF term. This
makes it possible to explain why a document survived fusion without persisting sensitive query or
document payloads.

## Operational boundaries

- Query generation is deliberately out of scope. Callers must validate model-generated rewrites
  for intent preservation and prompt-injection risk before fan-out. The runtime proves that one
  canonical authorization scope reaches every variant; the retriever must still enforce that scope
  against its backing store.
- A deadline cancels awaitable work, but cancellation does not guarantee that a blocking operation
  running in a worker thread or remote service stops. Production adapters still need transport
  timeouts and cancellation-aware clients.
- SHA-256 digests provide deterministic identity, not producer authenticity. Sign evidence or
  place it in an authenticated audit store when crossing a trust boundary.
- Weighted RRF improves coverage; it does not prove relevance, factuality, or answer faithfulness.
  Tune weights and quorum against held-out qrels and regression slices before release.
- The in-memory engine adapter is suitable for reproducible validation and receives the scope but
  does not implement tenant filtering. A production vector-store adapter must translate the bound
  claims into native, fail-closed ACL filters on every call.

## Next step

Add a rewrite admission adapter that binds each variant to the primary query, model/prompt version,
and a signed intent-preservation decision before this runtime executes it.
