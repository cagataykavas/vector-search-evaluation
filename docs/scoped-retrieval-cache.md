# Access-scoped retrieval cache

Retrieval access control is incomplete if a cache can reuse one principal's result for another. The
same query text can legally return different documents for two tenants, group memberships, index
revisions, embedding models or retriever policies. A query-only cache key turns this into a cross-scope
data exposure.

`search.scoped_cache` provides a dependency-free correctness reference for cache isolation. Each key
binds the exact UTF-8 query, IEEE-754 query embedding, verified tenant/group scope, index revision,
embedding-model revision, retriever configuration digest, requested result count and candidate depth.
The outward key is a canonical SHA-256 digest and does not expose the query or scope identifiers.

## Runtime properties

- anonymous and tenant scopes are distinct; group ordering is canonical;
- changes to any retrieval input produce a different cache key;
- cache values contain document IDs, scores, ranks and source only—not document text;
- values require unique documents, finite scores and contiguous ranks bounded by the request;
- monotonic TTL expiry fails closed and expired values are never returned;
- bounded LRU capacity prevents unbounded memory growth;
- index revisions can be invalidated without flushing unrelated revisions;
- all reads, writes, eviction, invalidation and counters are protected by one process-local lock;
- snapshots expose only low-cardinality operational counters.

The serving path should derive `RetrievalScope` from verified authentication middleware, run access
filtering before retrieval, build a `RetrievalRequest`, then check the cache. On a miss it performs the
authorized search and caches only the ranking identity. Document materialization must repeat the
authorization check, matching the access-enforcement boundary.

## Trust boundaries and limitations

This cache does not authenticate a caller or decide document authorization. A forged scope supplied by
an untrusted request body defeats the boundary; scope must come from server-owned identity state. Cache
keys provide isolation and content identity, not cryptographic authentication. SHA-256 collision
resistance is assumed.

State is process-local, so multiple replicas do not share capacity, invalidations or counters. The
implementation deliberately omits request coalescing: concurrent misses can duplicate retrieval work,
but they cannot receive another scope's entry. Production adapters should use a distributed cache with
atomic TTL, namespace/version invalidation, encryption in transit, admission controls and tenant-aware
rate limits. Never cache raw document text unless storage encryption, retention and deletion semantics
are explicitly designed.

The next increment is to wrap the access-filtered BM25/dense serving path, correlate cache-key digests
with retrieval traces, and add adversarial integration tests for authenticated scope changes and index
rollovers.
