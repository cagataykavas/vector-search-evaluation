# Hybrid Search Evaluation

A compact retrieval platform for comparing **BM25 lexical search, dense cosine retrieval and Reciprocal Rank Fusion (RRF)** under the same query set and qrels.

This project is intentionally focused on the part of RAG systems that often gets hand-waved: **retrieval quality measurement**. It exposes a runnable API, deterministic local indices, ranking metrics, tests, Docker packaging and CI.

## Architecture

```mermaid
flowchart LR
    DOCS[Documents] --> BM25[BM25 index]
    DOCS --> DENSE[Dense embedding index]
    QUERY[Query text] --> BM25
    QEMB[Query embedding] --> DENSE
    BM25 --> RRF[Reciprocal Rank Fusion]
    DENSE --> RRF
    RRF --> RESULTS[Hybrid ranking]
    QRELS[Relevance labels / qrels] --> EVAL[Recall@K · MRR · NDCG@K]
    RESULTS --> EVAL
    BM25 --> EVAL
    DENSE --> EVAL
```

## Why hybrid retrieval?

Lexical and dense systems fail differently.

- BM25 is strong when exact terms, identifiers, error messages or domain vocabulary matter.
- Dense retrieval can recover semantic matches that share few literal terms.
- Hybrid fusion often improves robustness without requiring scores from incompatible systems to be calibrated onto the same numeric scale.

RRF combines **rank positions** rather than raw similarity scores:

```text
RRF(document) = Σ 1 / (C + rank_i(document))
```

The implementation defaults to `C = 60`, a common stabilizing constant.

## Included components

### BM25

A readable implementation with:

- tokenization;
- document frequencies;
- IDF;
- document-length normalization;
- configurable `k1` and `b`.

### Dense retrieval

- cosine similarity;
- vector normalization;
- embedding-dimension validation;
- top-K ranking.

The service accepts embeddings rather than downloading an embedding model so the retrieval engine remains provider-independent. It can sit behind OpenAI, sentence-transformers, Vertex AI, Bedrock or any custom encoder.

### Hybrid RRF

The top candidate lists from BM25 and dense retrieval are fused by reciprocal rank.

### Evaluation

The repository calculates:

- **Recall@K** — how much of the known relevant set was retrieved;
- **MRR** — how early the first relevant result appeared;
- **NDCG@K** — ranking quality with logarithmic position discount.

### Judgment coverage gate

Relevance metrics are easy to misread when the annotation pool is shallow. A document absent
from qrels may be irrelevant, or it may simply never have been judged. The bundled release gate
measures the fraction of each top-K ranking covered by explicit positive **or negative** judgments
before ranking metrics are trusted.

```bash
judgment-coverage-audit judgment-coverage.json --output coverage-report.json
```

The input separates rankings from the complete set of judged document IDs:

```json
{
  "queries": [
    {
      "query_id": "q-1",
      "ranking": ["doc-1", "doc-2", "doc-3"],
      "judged_doc_ids": ["doc-1", "doc-2", "doc-3"]
    }
  ],
  "policy": {
    "k": 3,
    "min_mean_judged_fraction": 0.8,
    "low_coverage_threshold": 0.5,
    "max_low_coverage_rate": 0.1
  }
}
```

Accepted, policy-rejected and malformed artifacts exit with codes `0`, `2` and `3`. Reports include
aggregate coverage, hashed query references, bounded per-query counts and a canonical evidence
digest. Coverage is an admission check, not a relevance score: a fully judged pool can still contain
incorrect or biased labels, and thresholds must be calibrated to the benchmark's pooling process.

## Run the API

```bash
pip install -e '.[dev]'
uvicorn app.api:app --reload
```

Create an index:

```bash
curl -X PUT http://localhost:8000/index \
  -H 'content-type: application/json' \
  -d '{
    "documents": [
      {
        "doc_id": "doc-1",
        "text": "A timeout error can be caused by an upstream dependency.",
        "embedding": [0.9, 0.1, 0.0],
        "metadata": {"source": "runbook"}
      }
    ]
  }'
```

Search:

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -H 'content-type: application/json' \
  -d '{
    "query_text": "upstream timeout",
    "query_embedding": [0.88, 0.12, 0.0],
    "k": 5,
    "candidate_k": 20
  }'
```

Benchmark retrieval strategies against qrels:

```bash
curl -X POST http://localhost:8000/benchmark \
  -H 'content-type: application/json' \
  -d '{
    "k": 10,
    "queries": [
      {
        "query_id": "q-1",
        "text": "upstream timeout",
        "embedding": [0.88, 0.12, 0.0],
        "relevant_doc_ids": ["doc-1"]
      }
    ]
  }'
```

## Repository layout

```text
vector-search-evaluation/
├── app/
│   └── api.py
├── search/
│   ├── engine.py
│   └── benchmark.py
├── tests/
│   └── test_hybrid_search.py
├── evaluate.py
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Production extensions

- HNSW / FAISS / pgvector / OpenSearch adapters;
- metadata filters;
- sparse learned retrieval;
- cross-encoder reranking;
- query rewriting;
- multi-query retrieval;
- chunking strategy evaluation;
- latency and cost benchmarking;
- RAG answer-faithfulness evaluation after retrieval.

## Interview topics demonstrated

`BM25` · `dense retrieval` · `cosine similarity` · `RRF` · `Recall@K` · `MRR` · `NDCG` · `qrels` · `reranking` · `RAG evaluation` · `vector databases`

The important claim is not “vector search is cool.” The project shows **how to prove whether a retrieval change actually improved ranking quality**.
