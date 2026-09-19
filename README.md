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

## Query-level regression gate

Aggregate retrieval metrics can improve while a subset of queries becomes materially worse. The
paired regression gate compares baseline and candidate rankings over exactly the same query IDs and
qrels:

```python
from search.regression import compare_retrieval_runs

report = compare_retrieval_runs(
    baseline={"q-1": ["doc-a", "doc-b"]},
    candidate={"q-1": ["doc-b", "doc-a"]},
    qrels={"q-1": {"doc-a"}},
    k=10,
    max_regression_rate=0.10,
    minimum_mean_ndcg_delta=0.0,
)
```

The JSON-ready report contains per-query Recall, MRR and NDCG deltas, mean deltas, the exact
regressed-query rate and machine-readable release reasons. Query sets must match exactly; empty
qrels, duplicate result IDs and malformed rankings fail closed. Evidence is sorted by query ID so
CI artifacts are deterministic.

The gate does not remove qrels bias or prove online relevance. Its thresholds must be selected on a
representative, versioned evaluation set. Production promotion should also segment results by query
type, monitor latency and cost, and validate downstream answer quality.

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
