# RRF fusion evidence

Hybrid retrieval is difficult to debug when only the final document order survives. A result can
move because its BM25 rank changed, its dense rank changed, one source stopped returning it, or a
source-score tie-breaker changed. `include_fusion_evidence` makes those inputs reviewable without
changing the default search response.

```json
{
  "query_text": "upstream timeout",
  "query_embedding": [0.88, 0.12, 0.0],
  "k": 5,
  "candidate_k": 20,
  "include_fusion_evidence": true
}
```

For every returned document, the response records:

- each source name and rank;
- the exact reciprocal-rank contribution;
- the normalized source-score contribution used only for deterministic tie-breaking;
- the fused score and total tie-break score.

The fusion boundary rejects duplicate documents within a source, duplicate source rankings,
non-contiguous ranks, non-finite source scores, and invalid fusion parameters. Contributions are
sorted by source and returned in final rank order, making the evidence deterministic and directly
reconcilable to the public score.

## Limitations

Fusion evidence explains how the implementation produced a ranking; it does not prove relevance,
embedding quality, corpus completeness, or that BM25 and dense candidates were generated from the
same index snapshot. Production evidence should additionally bind the corpus and encoder versions,
query identity, access-control scope, and retrieval timestamp.
