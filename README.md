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

### Serving release gate

Offline relevance gains are not production wins if the candidate violates serving latency or is
strictly worse on both quality and speed. `search.serving_gate` evaluates paired baseline and
candidate artifacts with:

- mean NDCG@K drop tolerance;
- absolute p95 latency and baseline-relative p95/median budgets;
- explicit Pareto-dominance rejection;
- minimum query and per-query timing evidence;
- exact query/qrel pairing and fail-closed malformed-input checks;
- deterministic per-query evidence and JSON output.

Run it in CI with `python -m search.serving_gate artifact.json --require-pass`. Exit code `0`
means accepted, `2` is a valid policy rejection and `1` identifies malformed input. Latency
samples should come from isolated warm runs with consistent concurrency, hardware, index state,
cache policy and network boundaries. This gate does not replace load testing, tail-latency
analysis under saturation or online relevance measurement.

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
│   ├── benchmark.py
│   ├── engine.py
│   └── serving_gate.py
├── tests/
│   ├── test_hybrid_search.py
│   └── test_serving_gate.py
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
- load and cost benchmarking beyond the paired serving gate;
- RAG answer-faithfulness evaluation after retrieval.

## Interview topics demonstrated

`BM25` · `dense retrieval` · `cosine similarity` · `RRF` · `Recall@K` · `MRR` · `NDCG` · `qrels` · `reranking` · `RAG evaluation` · `vector databases`

The important claim is not “vector search is cool.” The project shows **how to prove whether a retrieval change actually improved ranking quality**.
