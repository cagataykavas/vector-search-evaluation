import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from evaluate import cosine_search
from search.engine import DenseIndex, Document


@pytest.mark.parametrize(
    ("document_scale", "query_scale"),
    [
        (1.0, 1.0),
        (1e300, 1e300),
        (1e-300, 1e-300),
        (1e300, 1.0),
        (1.0, 1e300),
        (1e-300, 1.0),
        (1.0, 1e-300),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
        (np.finfo(float).max, np.finfo(float).max),
    ],
)
def test_scale_invariant_cosine_in_serving_and_evaluation(document_scale, query_scale):
    rows = [
        ("orthogonal", (0.0, document_scale)),
        ("opposite", (-document_scale, 0.0)),
        ("aligned", (document_scale, 0.0)),
    ]
    documents = [Document(doc_id, doc_id, embedding) for doc_id, embedding in rows]
    query = np.array([query_scale, 0.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        index = DenseIndex(documents)
        hits = index.search(query, k=3)
        evaluated = cosine_search(query, np.array([embedding for _, embedding in rows]), k=3)
    assert [hit.doc_id for hit in hits] == ["aligned", "orthogonal", "opposite"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.0, -1.0])
    assert evaluated == [2, 0, 1]
    assert np.linalg.norm(index.matrix, axis=1) == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "embedding", [(0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0), (-float("inf"), 1.0), ()]
)
def test_invalid_document_directions_rejected(embedding):
    with pytest.raises(ValueError):
        DenseIndex([Document("bad", "bad", embedding)])
    with pytest.raises(ValueError):
        cosine_search(np.ones(len(embedding)), np.array([embedding]))


@pytest.mark.parametrize(
    "query", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0], [-float("inf"), 1.0]]
)
def test_invalid_query_directions_rejected(query):
    index = DenseIndex([Document("valid", "valid", (1.0, 0.0))])
    with pytest.raises(ValueError):
        index.search(query)
    with pytest.raises(ValueError):
        cosine_search(np.array(query), np.array([[1.0, 0.0]]))


def test_document_and_query_dimension_contracts():
    with pytest.raises(ValueError, match="dimension"):
        DenseIndex([Document("a", "a", (1.0,)), Document("b", "b", (1.0, 0.0))])
    index = DenseIndex([Document("a", "a", (1.0, 0.0))])
    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0])
    with pytest.raises(ValueError, match="dimension"):
        cosine_search(np.array([1.0]), np.array([[1.0, 0.0]]))


def test_empty_dense_index_behavior_preserved():
    assert DenseIndex([]).search([]) == []
    assert cosine_search(np.array([1.0, 0.0]), np.empty((0, 2))) == []


def test_agreement_with_ordinary_float64_reference():
    rng = np.random.default_rng(23)
    matrix = rng.normal(size=(64, 24))
    query = rng.normal(size=24)
    documents = [Document(str(index), "text", tuple(row)) for index, row in enumerate(matrix)]
    expected = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)) @ (
        query / np.linalg.norm(query)
    )
    hits = DenseIndex(documents).search(query, k=64)
    assert [int(hit.doc_id) for hit in hits] == np.argsort(-expected).tolist()
    for hit in hits:
        assert hit.score == pytest.approx(expected[int(hit.doc_id)], abs=1e-14)
        assert -1.0 <= hit.score <= 1.0


def test_independent_row_scales_and_unrepresentable_original_norm():
    maximum = np.finfo(float).max
    documents = [
        Document("huge-diagonal", "diagonal", (maximum, maximum)),
        Document("tiny-aligned", "aligned", (1e-300, 0.0)),
        Document("ordinary-opposite", "opposite", (-2.0, 0.0)),
    ]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        hits = DenseIndex(documents).search([1.0, 0.0], k=3)
    assert [hit.doc_id for hit in hits] == ["tiny-aligned", "huge-diagonal", "ordinary-opposite"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 1 / np.sqrt(2), -1.0])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "engine", None)
    with TestClient(api_module.app) as client:
        response = client.put(
            "/index",
            json={"documents": [{"doc_id": "valid", "text": "valid", "embedding": [1.0, 0.0]}]},
        )
        assert response.status_code == 200
        yield client


def test_failed_reindex_retains_previous_index(client):
    response = client.put(
        "/index",
        json={"documents": [{"doc_id": "bad", "text": "bad", "embedding": [0.0, 0.0]}]},
    )
    assert response.status_code == 422
    result = client.post(
        "/search/hybrid", json={"query_text": "valid", "query_embedding": [1.0, 0.0]}
    )
    assert result.status_code == 200
    assert [hit["doc_id"] for hit in result.json()["hits"]] == ["valid"]


@pytest.mark.parametrize("embedding_json", ["[0.0,0.0]", "[1e309,0.0]"])
def test_api_rejects_undefined_query_direction(client, embedding_json):
    response = client.post(
        "/search/hybrid",
        content='{"query_text":"valid","query_embedding":' + embedding_json + "}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_api_benchmark_rejects_undefined_direction(client):
    response = client.post(
        "/benchmark",
        json={
            "k": 1,
            "queries": [
                {
                    "query_id": "q",
                    "text": "valid",
                    "embedding": [0.0, 0.0],
                    "relevant_doc_ids": ["valid"],
                }
            ],
        },
    )
    assert response.status_code == 422
