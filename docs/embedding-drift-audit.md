# Embedding population drift audit

Retrieval metrics from a reviewed benchmark do not show whether production query traffic still
resembles that benchmark. `search.embedding_drift` compares a baseline and candidate population in
the same cosine space before a retrieval release is promoted.

The audit:

- validates bounded, rectangular, finite matrices and rejects zero-norm rows;
- normalizes embeddings because the retrieval engine uses cosine similarity;
- measures multivariate distribution change with the energy distance;
- estimates a deterministic permutation-test p-value;
- requires both statistical significance and a reviewed practical-effect threshold;
- binds the exact input matrices and policy to SHA-256 digests in the JSON report;
- returns exit code `0` for pass, `3` for detected drift, and `2` for invalid evidence.

Input files are JSON arrays of embedding arrays:

```bash
python -m search.embedding_drift baseline.json candidate.json \
  --alpha 0.05 \
  --min-effect-size 0.05 \
  --permutations 499 \
  --output embedding-drift-report.json
```

The sample and dimension caps bound memory and permutation work. Use an upstream sampling policy
that is fixed before observing candidate outcomes, and retain the report with the encoder/index
contract used for both populations.

## Limitations

The permutation test assumes exchangeable observations and does not correct for seasonality,
autocorrelation, traffic-mix changes, or repeated monitoring. It detects distribution change, not
its cause, relevance impact, semantic correctness, or source authority. Baseline and candidate
must use the same reviewed encoder contract; an encoder migration requires a separate retrieval
quality evaluation. Production thresholds need calibration against historical benign windows and
known incidents rather than copying the example values.
