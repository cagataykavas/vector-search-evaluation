from fastapi.testclient import TestClient

import app.api as api_module


def documents() -> list[dict]:
    return [
        {
            "doc_id": "espresso",
            "text": "Espresso is concentrated coffee brewed under pressure.",
            "embedding": [1.0, 0.0, 0.0],
            "metadata": {"topic": "coffee"},
        },
        {
            "doc_id": "aeropress",
            "text": "AeroPress combines immersion brewing and gentle pressure.",
            "embedding": [0.9, 0.1, 0.0],
            "metadata": {"topic": "coffee"},
        },
        {
            "doc_id": "mythology",
            "text": "Athena is associated with wisdom in Greek mythology.",
            "embedding": [0.0, 1.0, 0.0],
            "metadata": {"topic": "mythology"},
        },
    ]


def test_hybrid_search_returns_expected_document() -> None:
    api_module.engine = None
    client = TestClient(api_module.app)
    indexed = client.put("/index", json={"documents": documents()})
    assert indexed.status_code == 200

    result = client.post(
        "/search/hybrid",
        json={
            "query_text": "coffee pressure brewing",
            "query_embedding": [1.0, 0.0, 0.0],
            "k": 2,
            "candidate_k": 3,
        },
    )
    assert result.status_code == 200
    hits = result.json()["hits"]
    assert hits[0]["doc_id"] == "espresso"
    assert len(hits) == 2


def test_benchmark_compares_three_methods() -> None:
    api_module.engine = None
    client = TestClient(api_module.app)
    client.put("/index", json={"documents": documents()})

    response = client.post(
        "/benchmark",
        json={
            "k": 2,
            "queries": [
                {
                    "query_id": "q1",
                    "text": "coffee under pressure",
                    "embedding": [1.0, 0.0, 0.0],
                    "relevant_doc_ids": ["espresso", "aeropress"],
                },
                {
                    "query_id": "q2",
                    "text": "Athena wisdom",
                    "embedding": [0.0, 1.0, 0.0],
                    "relevant_doc_ids": ["mythology"],
                },
            ],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert {row["method"] for row in results} == {"bm25", "dense", "hybrid_rrf"}
    for row in results:
        assert 0 <= row["metrics"]["recall_at_k"] <= 1
        assert 0 <= row["metrics"]["mrr"] <= 1
        assert 0 <= row["metrics"]["ndcg_at_k"] <= 1
