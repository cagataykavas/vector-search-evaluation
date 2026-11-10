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
- **NDCG@K** — binary or graded ranking quality with linear relevance gains and logarithmic position discount.

### Binary and graded qrels

Binary qrels answer whether a document is relevant. Graded qrels additionally distinguish a highly useful passage from a marginally useful one, so NDCG rewards putting the strongest evidence first even when both rankings have identical Recall@K and MRR.

Each benchmark uses exactly one relevance mode. Binary requests provide `relevant_doc_ids`; graded requests provide a `relevance_grades` mapping with finite grades from 0 through 10 and at least one positive judgment. Mixing modes in one aggregate benchmark is rejected so the result remains interpretable. The response records `relevance_mode` alongside every method.

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

For graded judgments, replace `relevant_doc_ids` with explicit gains:

```json
{
  "query_id": "q-1",
  "text": "upstream timeout",
  "embedding": [0.88, 0.12, 0.0],
  "relevance_grades": {"doc-1": 3, "doc-2": 1}
}
```

The implementation uses linear gain, not `2^grade - 1`, which keeps the scale auditable and avoids making one high grade dominate by exponentiation. Grades still need a documented annotation rubric and consistent assessors; NDCG cannot establish that the judgments themselves are correct.

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
- annotation agreement and adjudication for graded qrels;
- latency and cost benchmarking;
- RAG answer-faithfulness evaluation after retrieval.

## Interview topics demonstrated

`BM25` · `dense retrieval` · `cosine similarity` · `RRF` · `Recall@K` · `MRR` · `NDCG` · `qrels` · `reranking` · `RAG evaluation` · `vector databases`

The important claim is not “vector search is cool.” The project shows **how to prove whether a retrieval change actually improved ranking quality**.
