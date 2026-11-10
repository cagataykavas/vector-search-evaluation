from __future__ import annotations

from dataclasses import asdict
from math import isfinite

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from evaluate import MAX_RELEVANCE_GRADE
from search.benchmark import QueryCase, benchmark
from search.engine import Document, HybridSearchEngine


class DocumentInput(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class SearchRequest(BaseModel):
    query_text: str
    query_embedding: list[float] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=1, le=1000)


class QueryInput(BaseModel):
    query_id: str
    text: str
    embedding: list[float] = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_qrels(self) -> QueryInput:
        if bool(self.relevant_doc_ids) == bool(self.relevance_grades):
            raise ValueError("define exactly one of relevant_doc_ids or relevance_grades")
        if self.relevance_grades:
            if not all(doc_id.strip() for doc_id in self.relevance_grades):
                raise ValueError("graded qrels document IDs must not be blank")
            if not all(
                isfinite(grade) and 0.0 <= grade <= MAX_RELEVANCE_GRADE
                for grade in self.relevance_grades.values()
            ):
                raise ValueError(
                    f"relevance grades must be finite and between 0 and {MAX_RELEVANCE_GRADE:g}"
                )
            if not any(grade > 0 for grade in self.relevance_grades.values()):
                raise ValueError("graded qrels must contain at least one positive grade")
        return self


class BenchmarkRequest(BaseModel):
    queries: list[QueryInput] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def require_consistent_relevance_mode(self) -> BenchmarkRequest:
        modes = {"graded" if row.relevance_grades else "binary" for row in self.queries}
        if len(modes) > 1:
            raise ValueError("a benchmark cannot mix binary and graded qrels")
        return self


def _document(row: DocumentInput) -> Document:
    return Document(
        doc_id=row.doc_id,
        text=row.text,
        embedding=tuple(row.embedding),
        metadata=dict(row.metadata),
    )


app = FastAPI(
    title="Hybrid Search Evaluation",
    version="0.2.0",
    description="BM25, dense cosine and RRF hybrid retrieval with qrels-based evaluation.",
)
engine: HybridSearchEngine | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "indexed_documents": len(engine.documents) if engine else 0}


@app.put("/index")
def create_index(request: IndexRequest) -> dict:
    global engine
    try:
        engine = HybridSearchEngine([_document(row) for row in request.documents])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "documents": len(engine.documents),
        "embedding_dimension": engine.dense.dimension,
        "status": "ready",
    }


def _engine() -> HybridSearchEngine:
    if engine is None:
        raise HTTPException(status_code=409, detail="index not initialized")
    return engine


@app.post("/search/hybrid")
def hybrid_search(request: SearchRequest) -> dict:
    search_engine = _engine()
    try:
        hits = search_engine.search(
            query_text=request.query_text,
            query_embedding=request.query_embedding,
            k=request.k,
            candidate_k=max(request.k, request.candidate_k),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"hits": search_engine.documents_for_hits(hits)}


@app.post("/benchmark")
def run_benchmark(request: BenchmarkRequest) -> dict:
    search_engine = _engine()
    queries = [
        QueryCase(
            query_id=row.query_id,
            text=row.text,
            embedding=tuple(row.embedding),
            relevant_doc_ids=frozenset(row.relevant_doc_ids),
            relevance_grades=tuple(sorted(row.relevance_grades.items())),
        )
        for row in request.queries
    ]
    try:
        results = benchmark(search_engine.documents, queries, k=request.k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"results": [asdict(result) for result in results]}
