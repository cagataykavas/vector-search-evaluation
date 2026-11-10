from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from search.benchmark import QueryCase, benchmark
from search.engine import Document
from search.qrels import audit_qrels


def corpus() -> list[Document]:
    return [
        Document("doc-a", "alpha", (1.0, 0.0)),
        Document("doc-b", "beta", (0.0, 1.0)),
        Document("doc-c", "gamma", (0.5, 0.5)),
    ]


def test_qrels_audit_reports_privacy_reduced_coverage() -> None:
    audit = audit_qrels(
        ["doc-a", "doc-b", "doc-c"],
        [("q-1", ["doc-a", "doc-b"]), ("q-2", ["doc-b"])],
    )

    assert audit.to_dict() == {
        "documents": 3,
        "queries": 2,
        "relevance_judgments": 3,
        "unique_relevant_documents": 2,
        "corpus_relevance_coverage": 2 / 3,
    }


@pytest.mark.parametrize(
    ("qrels", "error", "message"),
    [
        ([], ValueError, "at least one query"),
        ([("q-1", [])], ValueError, "at least one relevant document"),
        (
            [("q-1", ["doc-a"]), ("q-1", ["doc-b"])],
            ValueError,
            "duplicate query ID",
        ),
        ([("q-1", ["doc-a", "doc-a"])], ValueError, "duplicate relevance judgments"),
        ([("q-1", ["missing"])], ValueError, "absent from the corpus"),
        ([("", ["doc-a"])], ValueError, "query ID at position 1"),
        ([("q-1", [""])], ValueError, "relevant IDs for query"),
        ([("q-1", "doc-a")], TypeError, "must be a collection of IDs"),
        (["not-a-pair"], TypeError, r"must be a \(query_id, relevant_ids\) tuple"),
    ],
)
def test_qrels_audit_rejects_invalid_judgments(qrels, error, message) -> None:
    with pytest.raises(error, match=message):
        audit_qrels(["doc-a", "doc-b"], qrels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_documents": 0}, "max_documents must be a positive integer"),
        ({"max_queries": True}, "max_queries must be a positive integer"),
        ({"max_judgments": -1}, "max_judgments must be a positive integer"),
    ],
)
def test_qrels_audit_rejects_invalid_budgets(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_qrels(["doc-a"], [("q-1", ["doc-a"])], **kwargs)


def test_qrels_audit_enforces_resource_budgets() -> None:
    with pytest.raises(ValueError, match="1 item budget"):
        audit_qrels(["doc-a", "doc-b"], [("q-1", ["doc-a"])], max_documents=1)
    with pytest.raises(ValueError, match="1 query budget"):
        audit_qrels(
            ["doc-a"],
            [("q-1", ["doc-a"]), ("q-2", ["doc-a"])],
            max_queries=1,
        )
    with pytest.raises(ValueError, match="1 item budget"):
        audit_qrels(
            ["doc-a", "doc-b"],
            [("q-1", ["doc-a", "doc-b"])],
            max_judgments=1,
        )


def test_benchmark_fails_before_scoring_unknown_qrels() -> None:
    query = QueryCase(
        query_id="q-1",
        text="alpha",
        embedding=(1.0, 0.0),
        relevant_doc_ids=frozenset({"missing"}),
    )

    with pytest.raises(ValueError, match="absent from the corpus"):
        benchmark(corpus(), [query], k=2)


def test_api_returns_qrels_audit_and_rejects_duplicate_judgments() -> None:
    api_module.engine = None
    client = TestClient(api_module.app)
    client.put(
        "/index",
        json={
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "embedding": list(document.embedding),
                }
                for document in corpus()
            ]
        },
    )
    valid = client.post(
        "/benchmark",
        json={
            "k": 2,
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha",
                    "embedding": [1.0, 0.0],
                    "relevant_doc_ids": ["doc-a"],
                }
            ],
        },
    )
    assert valid.status_code == 200
    assert valid.json()["qrels_audit"] == {
        "documents": 3,
        "queries": 1,
        "relevance_judgments": 1,
        "unique_relevant_documents": 1,
        "corpus_relevance_coverage": 1 / 3,
    }

    duplicate = client.post(
        "/benchmark",
        json={
            "k": 2,
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha",
                    "embedding": [1.0, 0.0],
                    "relevant_doc_ids": ["doc-a", "doc-a"],
                }
            ],
        },
    )
    assert duplicate.status_code == 422
    assert "duplicate relevance judgments" in duplicate.json()["detail"]
