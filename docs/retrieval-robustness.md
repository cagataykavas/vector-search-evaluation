# Retrieval robustness gate

Average Recall@K or MRR can remain healthy while harmless query rewrites or small embedding
perturbations produce a substantially different evidence set. `search.robustness` compares each
baseline ranking with controlled variants using top-K Jaccard similarity, rank-biased overlap,
relevant-document survival, top-result retention, and a worst-query guardrail.

The evaluator requires multiple queries and perturbations, validates ranking integrity, emits a
deterministic JSON-ready report, and fails closed when the baseline itself retrieves no relevant
evidence. Policy thresholds can block a retrieval release with machine-readable reasons.

## Boundaries

- Perturbations must preserve user intent; adversarial or meaning-changing rewrites need separate labels.
- Stability is not correctness. A consistently wrong ranking can be stable, so this gate complements qrels metrics.
- Relevance judgments and embeddings must come from held-out data without benchmark leakage.
- Thresholds should be calibrated by query slice and business impact rather than copied unchanged.
