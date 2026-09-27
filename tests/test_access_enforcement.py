from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from search.engine import (
    AccessScope,
    Document,
    DocumentAccess,
    HybridSearchEngine,
    SearchHit,
)


@pytest.fixture(autouse=True)
def reset_api_state():
    api_module.engine = None
    api_module.app.dependency_overrides.clear()
    yield
    api_module.engine = None
    api_module.app.dependency_overrides.clear()


def document(
    doc_id: str,
    text: str,
    embedding: tuple[float, ...],
    *,
    tenant_id: str | None = None,
    groups: frozenset[str] = frozenset(),
) -> Document:
    access = DocumentAccess() if tenant_id is None else DocumentAccess("tenant", tenant_id, groups)
    return Document(doc_id, text, embedding, {"kind": "test"}, access)


def documents() -> list[Document]:
    return [
        document("public", "shared timeout runbook", (0.6, 0.4, 0.0)),
        document("tenant-a", "private timeout runbook", (1.0, 0.0, 0.0), tenant_id="acme"),
        document(
            "finance-a",
            "private finance timeout runbook",
            (0.99, 0.01, 0.0),
            tenant_id="acme",
            groups=frozenset({"finance"}),
        ),
        document("tenant-b", "private timeout runbook", (1.0, 0.0, 0.0), tenant_id="other"),
    ]


def ids(engine: HybridSearchEngine, scope: AccessScope | None) -> list[str]:
    return [
        hit.doc_id
        for hit in engine.search(
            query_text="private timeout runbook",
            query_embedding=(1.0, 0.0, 0.0),
            k=10,
            candidate_k=10,
            scope=scope,
        )
    ]


def test_anonymous_search_is_public_only() -> None:
    assert ids(HybridSearchEngine(documents()), None) == ["public"]


def test_tenant_scope_includes_public_and_unrestricted_tenant_documents() -> None:
    result = ids(HybridSearchEngine(documents()), AccessScope("acme"))
    assert set(result) == {"public", "tenant-a"}
    assert "finance-a" not in result
    assert "tenant-b" not in result


def test_group_membership_unlocks_only_same_tenant_group_documents() -> None:
    result = ids(
        HybridSearchEngine(documents()),
        AccessScope("acme", frozenset({"finance"})),
    )
    assert set(result) == {"public", "tenant-a", "finance-a"}
    assert "tenant-b" not in result


def test_scope_can_exclude_public_documents() -> None:
    result = ids(
        HybridSearchEngine(documents()),
        AccessScope("acme", frozenset({"finance"}), include_public=False),
    )
    assert set(result) == {"tenant-a", "finance-a"}


def test_empty_authorized_projection_returns_no_hits_after_dimension_validation() -> None:
    engine = HybridSearchEngine([document("private", "secret", (1.0, 0.0), tenant_id="other")])
    assert engine.search(query_text="secret", query_embedding=(1.0, 0.0)) == []
    with pytest.raises(ValueError, match="dimension 2"):
        engine.search(query_text="secret", query_embedding=(1.0,))


def test_unauthorized_corpus_does_not_change_authorized_bm25_statistics() -> None:
    allowed = document("allowed", "timeout runbook", (1.0, 0.0))
    baseline = HybridSearchEngine([allowed])
    mixed = HybridSearchEngine(
        [allowed]
        + [
            document(
                f"private-{index}",
                "timeout timeout timeout runbook",
                (1.0, 0.0),
                tenant_id="other",
            )
            for index in range(100)
        ]
    )

    expected = baseline.bm25.search("timeout", allowed_doc_ids=frozenset({"allowed"}))
    actual = mixed.bm25.search("timeout", allowed_doc_ids=frozenset({"allowed"}))
    assert actual == expected


def test_dense_filtering_happens_before_top_k_selection() -> None:
    engine = HybridSearchEngine(documents())
    hits = engine.dense.search(
        (1.0, 0.0, 0.0),
        k=1,
        allowed_doc_ids=frozenset({"public"}),
    )
    assert [hit.doc_id for hit in hits] == ["public"]


def test_equal_dense_scores_are_stable_across_ingestion_order() -> None:
    first = document("a", "a", (1.0, 0.0))
    second = document("b", "b", (1.0, 0.0))
    forward = HybridSearchEngine([first, second]).dense.search((1.0, 0.0), k=2)
    reverse = HybridSearchEngine([second, first]).dense.search((1.0, 0.0), k=2)
    assert [hit.doc_id for hit in forward] == ["a", "b"]
    assert [hit.doc_id for hit in reverse] == ["a", "b"]


def test_materialization_rechecks_scope_for_forged_or_stale_hits() -> None:
    engine = HybridSearchEngine(documents())
    forged = [SearchHit("tenant-b", 1.0, 1, "forged")]
    with pytest.raises(ValueError, match="outside the active access scope"):
        engine.documents_for_hits(forged, scope=AccessScope("acme"))
    with pytest.raises(ValueError, match="unknown document"):
        engine.documents_for_hits([replace(forged[0], doc_id="missing")])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DocumentAccess("private"),
        lambda: DocumentAccess("tenant"),
        lambda: DocumentAccess("public", "acme"),
        lambda: DocumentAccess("public", None, frozenset({"finance"})),
        lambda: DocumentAccess("tenant", "acme", frozenset({"bad group"})),
        lambda: AccessScope("bad tenant"),
        lambda: AccessScope("acme", frozenset(f"g-{index}" for index in range(65))),
    ],
)
def test_invalid_access_contracts_fail_closed(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_direct_engine_rejects_untyped_access_objects() -> None:
    row = document("doc", "text", (1.0,))
    object.__setattr__(row, "access", {"visibility": "public"})
    with pytest.raises(TypeError, match="valid DocumentAccess"):
        HybridSearchEngine([row])


def api_documents() -> list[dict]:
    return [
        {
            "doc_id": "public",
            "text": "public answer",
            "embedding": [0.5, 0.5],
            "access": {"visibility": "public"},
        },
        {
            "doc_id": "private-a",
            "text": "private answer for acme",
            "embedding": [1.0, 0.0],
            "access": {"visibility": "tenant", "tenant_id": "acme"},
        },
        {
            "doc_id": "private-b",
            "text": "private answer for other tenant",
            "embedding": [1.0, 0.0],
            "access": {"visibility": "tenant", "tenant_id": "other"},
        },
    ]


def test_api_never_returns_private_text_without_matching_scope() -> None:
    client = TestClient(api_module.app)
    assert client.put("/index", json={"documents": api_documents()}).status_code == 200

    anonymous = client.post(
        "/search/hybrid",
        json={"query_text": "private answer", "query_embedding": [1.0, 0.0]},
    )
    assert anonymous.status_code == 200
    assert [row["doc_id"] for row in anonymous.json()["hits"]] == ["public"]
    assert "private answer" not in anonymous.text

    api_module.app.dependency_overrides[api_module.resolve_access_scope] = lambda: AccessScope(
        "acme", include_public=False
    )
    scoped = client.post(
        "/search/hybrid",
        json={
            "query_text": "private answer",
            "query_embedding": [1.0, 0.0],
        },
    )
    assert scoped.status_code == 200
    assert [row["doc_id"] for row in scoped.json()["hits"]] == ["private-a"]
    assert "other tenant" not in scoped.text


def test_api_rejects_inconsistent_access_metadata() -> None:
    client = TestClient(api_module.app)
    rows = api_documents()
    rows[0]["access"] = {"visibility": "public", "tenant_id": "acme"}
    response = client.put("/index", json={"documents": rows})
    assert response.status_code == 422
    assert "public documents cannot declare" in response.json()["detail"]


def test_api_rejects_documents_without_explicit_access_policy() -> None:
    client = TestClient(api_module.app)
    row = api_documents()[0]
    row.pop("access")
    response = client.put("/index", json={"documents": [row]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "missing"

    row["access"] = {}
    response = client.put("/index", json={"documents": [row]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "missing"


def test_client_cannot_select_its_own_access_scope() -> None:
    client = TestClient(api_module.app)
    client.put("/index", json={"documents": api_documents()})
    response = client.post(
        "/search/hybrid",
        json={
            "query_text": "answer",
            "query_embedding": [1.0, 0.0],
            "scope": {"tenant_id": "other"},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
