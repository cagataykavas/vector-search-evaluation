# Access-scoped retrieval audit

Multi-tenant RAG must treat document authorization as part of retrieval, not as a presentation
filter applied after ranking. This release audit verifies the relationship between a retriever's raw
candidate window, an authoritative document-scope catalog and the ranking actually served to a
principal.

## What the gate proves

For each query the artifact records metadata only:

- a principal identifier, tenant and group memberships;
- the retriever's ordered candidate IDs before authorization projection;
- the ordered document IDs returned to the RAG pipeline;
- the requested Top-K;
- an authoritative catalog classifying each document as public or tenant-scoped, with optional
  any-of group restrictions.

The gate recomputes the exact authorized projection of the raw ranking. It rejects:

- cross-tenant or group-restricted documents in served results;
- unknown catalog identities;
- injected, reordered or silently truncated served rankings;
- candidate windows that are too shallow to support the policy;
- authorization-aware underfill when enough eligible documents exist in the catalog;
- artifacts that exceed query, document, candidate, byte or authorization-work budgets.

```bash
python -m search.access_audit access-audit.json
```

Exit code `0` means accepted, `2` means malformed evidence and `3` means a well-formed release was
rejected. Duplicate JSON fields and non-finite values are rejected during parsing.

## Evidence privacy

Reports expose counts, stable finding codes and SHA-256 identities. Principal, query and document
IDs in findings are hashed; document text, prompts, embeddings and customer attributes must never be
placed in this artifact. The catalog, query set, policy and report each receive a canonical digest so
configuration or ranking changes cannot reuse the same evidence identity.

Hashes identify exact bytes but do not authenticate the producer. Sign or MAC release artifacts and
bind them to the deployed index snapshot, retriever version and authorization-policy version.

## Trust boundaries and limitations

- The catalog must be a complete, authoritative snapshot for the evaluated index. A forged or stale
  catalog can make unauthorized content look eligible.
- This audit verifies supplied rankings; enforcement still belongs inside the search engine or
  vector-store query. Do not retrieve broadly and send unauthorized text to a later filter or model.
- Any-of group semantics are explicit reference policy. Deployments with deny rules, hierarchies,
  ABAC, row-level security or purpose constraints need a matching authorization adapter.
- A clean result does not prove relevance, factuality or protection against side channels such as
  result counts, latency, cache keys or error differences.
- Underfill indicates insufficient authorized candidates in the evaluated window. It does not prove
  that increasing candidate depth is the right latency/cost tradeoff.

## Next step

Add a real metadata-filter adapter for the BM25 and dense indices, then generate this artifact from
the pre-filter candidate trace and served ranking in CI. Follow with adversarial tenant/group test
cases and cache-key isolation checks.
