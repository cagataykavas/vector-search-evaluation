from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from search.engine import SearchHit, reciprocal_rank_fusion_with_evidence


def ranking(source: str, rows: list[tuple[str, float]]) -> list[SearchHit]:
    return [
        SearchHit(doc_id=doc_id, score=score, rank=rank, source=source)
        for rank, (doc_id, score) in enumerate(rows, start=1)
    ]


def test_fusion_evidence_reconciles_to_returned_score():
    lexical = ranking("bm25", [("a", 0.8), ("b", 0.8)])
    dense = ranking("dense", [("b", 1.0), ("a", 0.5)])

    hits, evidence = reciprocal_rank_fusion_with_evidence((lexical, dense), k=2)

    assert [hit.doc_id for hit in hits] == ["b", "a"]
    assert [item.doc_id for item in evidence] == ["b", "a"]
    for hit, item in zip(hits, evidence, strict=True):
        assert item.fused_score == pytest.approx(hit.score)
        assert sum(row.rrf_score for row in item.contributions) == pytest.approx(hit.score)
        assert item.tie_break_score == pytest.approx(
            sum(row.normalized_source_score for row in item.contributions)
        )
        assert [row.source for row in item.contributions] == ["bm25", "dense"]


@pytest.mark.parametrize(
    "rankings,message",
    [
        (
            ([SearchHit("a", 1.0, 1, "bm25"), SearchHit("a", 0.5, 2, "bm25")],),
            "unique non-empty document IDs",
        ),
        (
            ([SearchHit("a", 1.0, 2, "bm25")],),
            "contiguous from 1",
        ),
        (
            (ranking("bm25", [("a", 1.0)]), ranking("bm25", [("b", 1.0)])),
            "more than one ranking",
        ),
        (
            ([SearchHit("a", float("nan"), 1, "dense")],),
            "finite numbers",
        ),
        (
            ([SearchHit("a", "bad", 1, "dense")],),
            "finite numbers",
        ),
        (
            ([SearchHit("a", 1.0, 1, " dense")],),
            "trimmed strings",
        ),
    ],
)
def test_malformed_component_rankings_fail_closed(rankings, message):
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion_with_evidence(rankings)


@pytest.mark.parametrize("parameters", [{"k": 0}, {"constant": 0}, {"constant": True}])
def test_invalid_fusion_parameters_fail_closed(parameters):
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank_fusion_with_evidence((ranking("dense", [("a", 1.0)]),), **parameters)


def test_api_returns_fusion_evidence_only_when_requested():
    api_module.engine = None
    client = TestClient(api_module.app)
    documents = [
        {
            "doc_id": "one",
            "text": "timeout recovery runbook",
            "embedding": [1.0, 0.0],
        },
        {
            "doc_id": "two",
            "text": "database backup policy",
            "embedding": [0.0, 1.0],
        },
    ]
    assert client.put("/index", json={"documents": documents}).status_code == 200
    request = {
        "query_text": "timeout runbook",
        "query_embedding": [1.0, 0.0],
        "k": 2,
        "candidate_k": 2,
    }

    default_response = client.post("/search/hybrid", json=request)
    assert default_response.status_code == 200
    assert "fusion_evidence" not in default_response.json()

    request["include_fusion_evidence"] = True
    explained_response = client.post("/search/hybrid", json=request)
    assert explained_response.status_code == 200
    payload = explained_response.json()
    assert [row["doc_id"] for row in payload["fusion_evidence"]] == [
        row["doc_id"] for row in payload["hits"]
    ]
    first = payload["fusion_evidence"][0]
    assert sum(row["rrf_score"] for row in first["contributions"]) == pytest.approx(
        first["fused_score"]
    )
