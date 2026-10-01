# Retrieval evidence diversity audit

Top-K retrieval can look well populated while carrying only one effective piece of evidence. Chunk
overlap, mirrored pages, boilerplate, or duplicated indexing may put the same claim into several
rows. Counting those rows as independent corroboration inflates downstream confidence and wastes
the model context window.

`search.evidence_diversity` is a dependency-free, fail-closed release audit for the evidence set
served to a RAG prompt. It clusters exact and near-duplicate hits, then checks whether each query
still has enough effective evidence and independently sourced clusters.

## Artifact contract

```json
{
  "index_snapshot_digest": "<lowercase SHA-256>",
  "queries": [
    {
      "query_id": "q-1",
      "hits": [
        {
          "rank": 1,
          "doc_id": "chunk-17",
          "source_id": "policy-manual-v3",
          "text": "Retrieved text used for prompt construction."
        }
      ]
    }
  ]
}
```

The producer must export the final, ordered evidence rows after access filtering and reranking.
`source_id` should identify the authoritative origin, not the chunk. Ranks must be contiguous,
document IDs must be unique per query, and the index snapshot digest must identify the exact index
used for retrieval.

## Decision model

The audit applies Unicode NFKC normalization and case folding, then builds token shingles. Hits are
connected when their normalized content is identical or their shingle Jaccard similarity reaches
the configured threshold. Connected components are treated as one effective evidence cluster.

The report enforces:

- an exact-duplicate fraction budget;
- a redundant-hit fraction budget (`1 - clusters / hits`);
- a maximum fraction occupied by one duplicate cluster;
- a minimum number of effective evidence clusters; and
- a minimum number of independently sourced clusters.

Independent sources are computed with maximum bipartite matching between evidence clusters and
their `source_id` values. This means copied text carrying two source labels still contributes only
one independent evidence unit. The JSON report contains counts, stable reason codes, a canonical
artifact digest, and hashed query references; it does not reproduce query IDs, document IDs, or
retrieved text.

```bash
python -m search.evidence_diversity evidence.json --output diversity-report.json
```

Exit codes are stable for CI: `0` accepted, `2` malformed or over budget, and `3` well-formed but
rejected by policy. Output-file replacement is atomic. JSON duplicate fields, non-finite values,
oversized inputs, non-contiguous ranks, duplicate IDs, and excessive pair comparisons fail closed.

## Trust boundary and limitations

This is a lexical redundancy control, not a factuality or semantic-entailment judge. Paraphrases
can evade shingle similarity; common templates can create false positives. Thresholds must be
calibrated on representative chunk sizes and languages. Transitive clustering can join an A-B-C
chain even when A and C are less similar, which is deliberately conservative for release gating.

The gate trusts the producer to supply complete final evidence, truthful source identities, and the
correct snapshot digest. In production, sign the artifact or generate it inside the retrieval
service, persist the accepted report with the prompt trace, and separately enforce authorization,
source authority, temporal validity, claim support, and answer grounding.
