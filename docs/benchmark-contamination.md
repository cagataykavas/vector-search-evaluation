# Retrieval benchmark contamination audit

Offline retrieval metrics are not trustworthy when held-out evaluation queries also
appear in prompt-tuning, query-rewrite, synthetic-example or reference query banks.
`search.contamination` provides a dependency-free, fail-closed split audit before a
benchmark result is admitted to CI.

```bash
python -m search.contamination \
  artifacts/tuning-queries.jsonl \
  artifacts/held-out-queries.jsonl \
  --near-threshold 0.85 \
  --max-near-rate 0 \
  --max-exact 0 \
  --output artifacts/contamination-report.json
```

Each input is UTF-8 JSONL with one record per line:

```json
{"id": "query-17", "text": "How do I rotate an expired OAuth token?"}
```

Exit code `0` means accepted, `2` means malformed or unauditable input, and `3`
means a well-formed artifact violated policy.

## Checks and evidence

- Unicode NFKC, case-folding and token normalization catch superficial formatting
  changes before exact comparison.
- Token-set Jaccard identifies near duplicates above a configurable threshold.
- Short queries use exact matching only to avoid unstable similarity claims.
- Duplicate normalized queries inside the held-out split are reported separately;
  otherwise repeated easy cases can bias aggregate metrics.
- IDs, item counts, text bytes and near-match comparisons are bounded. Exhausting the
  comparison budget rejects the audit instead of returning an incomplete clean bill.
- Reports contain stable reason codes, record IDs, similarity and SHA-256 digests;
  raw query text is not copied into the evidence artifact.
- Exact limits count unique leaked evaluation queries, not the number of matching
  reference rows. Near-leak rate is unique near-leaked queries divided by all held-out
  queries.

The reference side should include every query-bearing artifact that influenced model,
retriever, reranker, rewrite prompt or threshold selection. Run the audit before any
quality comparison so a contaminated benchmark never becomes the approved baseline.

## Limits and calibration

Lexical overlap is neither semantic equivalence nor proof of training-data exposure.
Jaccard can miss paraphrases with different vocabulary and can flag legitimate formulaic
queries. Calibrate thresholds by query length and domain, then manually review bounded
match evidence. Do not weaken the comparison budget merely to force a pass; use a
scalable MinHash/LSH or embedding-based candidate generator for large collections.

This gate does not inspect document-corpus leakage, qrel correctness, temporal leakage
or upstream foundation-model training data. It also cannot establish that a benchmark
represents production traffic. The next step is to bind this report to versioned query,
qrel and index snapshot digests in the retrieval release artifact.
