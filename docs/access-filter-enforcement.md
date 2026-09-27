# Access-filter enforcement in hybrid retrieval

The hybrid search API applies document authorization before BM25 or dense scoring.
This prevents an unauthorized document from entering either candidate list, RRF, or
response materialization.

Documents declare one of two policies:

- `public`: available to anonymous and tenant-scoped requests by default;
- `tenant`: available only to the matching tenant, and optionally to members of at
  least one declared group.

The indexing API requires this policy explicitly. Missing access metadata is rejected
instead of silently treating a document as public.

The API intentionally has no scope field in its search body. Its default
`resolve_access_scope` dependency returns anonymous access, which is **public-only**.
Deployments replace that dependency with one that derives `AccessScope` from verified
server-side identity context. Model output, prompt text and client-provided tenant
claims must never select the scope.

```json
{
  "query_text": "payment timeout",
  "query_embedding": [0.91, 0.09]
}
```

BM25 document frequency and average-length statistics are recomputed from the
authorized projection, so excluded documents cannot affect lexical scores. Dense
retrieval filters matrix rows before Top-K selection. `documents_for_hits` repeats
the authorization check before returning text, protecting against forged or stale
hit lists.

## Operational boundaries

The reference implementation performs exact, case-sensitive tenant and group
matching with bounded identifiers and at most 64 groups. An empty document group set
means every principal in that tenant may read the document; otherwise group matching
uses any-of semantics. Setting `include_public` to false creates a tenant-only query.

This code does not authenticate a caller, issue identities, protect the indexing
endpoint, implement deny rules, hide result-count/timing side channels, or invalidate
caches. Production deployments must derive `AccessScope` from verified identity
middleware, include its digest in cache keys, keep index access metadata synchronized
with the source of truth, and apply equivalent filters in the vector database itself.
The next step is an adapter for pgvector/OpenSearch metadata predicates plus
adversarial cache-isolation tests.
