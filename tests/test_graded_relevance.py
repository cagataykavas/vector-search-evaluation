from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from evaluate import evaluate, ndcg_at_k
from search.benchmark import QueryCase, benchmark
from search.engine import Document


def documents() -> list[dict]:
    return [
        {
            "doc_id": "primary",
            "text": "Primary deployment runbook for upstream timeouts.",
            "embedding": [1.0, 0.0],
        },
        {
            "doc_id": "secondary",
            "text": "Secondary troubleshooting notes for timeouts.",
            "embedding": [0.8, 0.2],
        },
        {
            "doc_id": "unrelated",
            "text": "Coffee brewing guide.",
            "embedding": [0.0, 1.0],
        },
    ]


def indexed_client() -> TestClient:
    api_module.engine = None
    client = TestClient(api_module.app)
    response = client.put("/index", json={"documents": documents()})
    assert response.status_code == 200
    return client


def test_graded_ndcg_rewards_putting_highest_grade_first() -> None:
    grades = {"primary": 3.0, "secondary": 1.0}

    ideal = ndcg_at_k(["primary", "secondary"], grades, 2)
    reversed_order = ndcg_at_k(["secondary", "primary"], grades, 2)
    binary_reversed = ndcg_at_k(["secondary", "primary"], {"primary", "secondary"}, 2)

    assert ideal == 1.0
    expected_reversed = (1.0 + 3.0 / math.log2(3)) / (3.0 + 1.0 / math.log2(3))
    assert reversed_order == pytest.approx(expected_reversed)
    assert binary_reversed == 1.0


def test_array_evaluator_accepts_graded_qrels() -> None:
    queries = np.asarray([[1.0, 0.0]])
    corpus = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

    metrics = evaluate(queries, corpus, {0: {0: 3.0, 1: 1.0}}, k=2)

    assert metrics == {"recall@k": 1.0, "mrr": 1.0, "ndcg@k": 1.0}


@pytest.mark.parametrize("grade", [-1.0, math.inf, math.nan, 10.1])
def test_graded_ndcg_rejects_invalid_grades(grade: float) -> None:
    with pytest.raises(ValueError, match="finite and between"):
        ndcg_at_k(["doc"], {"doc": grade}, 1)


def test_benchmark_api_reports_graded_relevance_mode() -> None:
    client = indexed_client()

    response = client.post(
        "/benchmark",
        json={
            "k": 2,
            "queries": [
                {
                    "query_id": "timeout-runbook",
                    "text": "upstream timeout runbook",
                    "embedding": [1.0, 0.0],
                    "relevance_grades": {"primary": 3, "secondary": 1},
                }
            ],
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert {row["relevance_mode"] for row in results} == {"graded"}
    assert {row["method"] for row in results} == {"bm25", "dense", "hybrid_rrf"}
    assert all(0.0 <= row["metrics"]["ndcg_at_k"] <= 1.0 for row in results)


@pytest.mark.parametrize(
    "query",
    [
        {"relevant_doc_ids": ["primary"], "relevance_grades": {"primary": 3}},
        {},
        {"relevance_grades": {"primary": -1}},
        {"relevance_grades": {"primary": 11}},
        {"relevance_grades": {"primary": 0}},
        {"relevance_grades": {" ": 1}},
    ],
)
def test_benchmark_api_rejects_ambiguous_or_invalid_qrels(query: dict) -> None:
    client = indexed_client()
    payload = {
        "query_id": "invalid",
        "text": "timeout",
        "embedding": [1.0, 0.0],
    } | query

    response = client.post("/benchmark", json={"queries": [payload]})

    assert response.status_code == 422


def test_benchmark_api_rejects_mixed_relevance_modes() -> None:
    client = indexed_client()

    response = client.post(
        "/benchmark",
        json={
            "queries": [
                {
                    "query_id": "binary",
                    "text": "timeout",
                    "embedding": [1.0, 0.0],
                    "relevant_doc_ids": ["primary"],
                },
                {
                    "query_id": "graded",
                    "text": "notes",
                    "embedding": [0.8, 0.2],
                    "relevance_grades": {"secondary": 2},
                },
            ]
        },
    )

    assert response.status_code == 422
    assert "cannot mix binary and graded" in response.text


def test_core_benchmark_rejects_mixed_relevance_modes() -> None:
    rows = [Document(**row) for row in documents()]
    queries = [
        QueryCase("binary", "timeout", (1.0, 0.0), frozenset({"primary"})),
        QueryCase(
            "graded",
            "notes",
            (0.8, 0.2),
            relevance_grades=(("secondary", 2.0),),
        ),
    ]

    with pytest.raises(ValueError, match="cannot mix binary and graded"):
        benchmark(rows, queries, k=2)
