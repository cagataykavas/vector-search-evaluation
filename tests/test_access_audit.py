from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from search.access_audit import (
    AccessAuditError,
    AccessAuditPolicy,
    DocumentScope,
    PrincipalScope,
    ScopedQuery,
    audit_access_scopes,
    audit_artifact,
    load_artifact,
)
from search.engine import Document, HybridSearchEngine


@pytest.fixture
def catalog() -> list[DocumentScope]:
    return [
        DocumentScope("public-1", "public"),
        DocumentScope("tenant-a-general", "tenant", "tenant-a"),
        DocumentScope("tenant-a-risk", "tenant", "tenant-a", ("risk",)),
        DocumentScope("tenant-a-legal", "tenant", "tenant-a", ("legal",)),
        DocumentScope("tenant-b-general", "tenant", "tenant-b"),
    ]


@pytest.fixture
def policy() -> AccessAuditPolicy:
    return AccessAuditPolicy(
        policy_id="acl-v1",
        min_candidate_depth=5,
        max_underfilled_queries=0,
        max_queries=20,
        max_documents=100,
        max_candidates_per_query=100,
        max_authorization_checks=10_000,
    )


@pytest.fixture
def query() -> ScopedQuery:
    return ScopedQuery(
        query_id="q-1",
        principal=PrincipalScope("analyst-1", "tenant-a", ("risk",)),
        requested_k=3,
        raw_candidate_ids=(
            "tenant-b-general",
            "tenant-a-risk",
            "public-1",
            "tenant-a-legal",
            "tenant-a-general",
        ),
        served_ids=("tenant-a-risk", "public-1", "tenant-a-general"),
    )


def test_accepts_exact_authorized_projection(catalog, policy, query):
    report = audit_access_scopes(catalog, [query], policy)

    assert report.accepted is True
    assert report.reason_codes == ()
    assert report.metrics["unauthorized_exposure_count"] == 0
    assert report.metrics["served_fill_rate"] == 1.0
    assert "analyst-1" not in json.dumps(report.to_dict())


def test_public_document_is_visible_across_tenants(catalog, policy):
    query = ScopedQuery(
        "q-public",
        PrincipalScope("user-b", "tenant-b"),
        2,
        (
            "tenant-a-general",
            "public-1",
            "tenant-b-general",
            "tenant-a-risk",
            "tenant-a-legal",
        ),
        ("public-1", "tenant-b-general"),
    )

    assert audit_access_scopes(catalog, [query], policy).accepted is True


def test_audits_real_hybrid_engine_ranking(policy):
    documents = [
        Document("tenant-b", "incident response timeout", (0.99, 0.01)),
        Document("tenant-a", "timeout recovery runbook", (0.95, 0.05)),
        Document("public", "general timeout guidance", (0.90, 0.10)),
    ]
    raw_hits = HybridSearchEngine(documents).search(
        query_text="timeout",
        query_embedding=(1.0, 0.0),
        k=3,
        candidate_k=3,
    )
    raw_ids = tuple(hit.doc_id for hit in raw_hits)
    authorized = tuple(doc_id for doc_id in raw_ids if doc_id != "tenant-b")
    catalog = [
        DocumentScope("tenant-b", "tenant", "tenant-b"),
        DocumentScope("tenant-a", "tenant", "tenant-a"),
        DocumentScope("public", "public"),
    ]
    query = ScopedQuery(
        "q-engine",
        PrincipalScope("user-a", "tenant-a"),
        2,
        raw_ids,
        authorized[:2],
    )
    policy = replace(policy, min_candidate_depth=3)

    report = audit_access_scopes(catalog, [query], policy)

    assert report.accepted is True
    assert report.metrics["served_document_count"] == 2


def test_rejects_cross_tenant_exposure(catalog, policy, query):
    query = replace(query, served_ids=("tenant-b-general", "public-1", "tenant-a-risk"))

    report = audit_access_scopes(catalog, [query], policy)

    assert report.accepted is False
    assert "UNAUTHORIZED_DOCUMENT_EXPOSED" in report.reason_codes
    assert report.metrics["unauthorized_exposure_count"] == 1
    assert "tenant-b-general" not in json.dumps(report.to_dict())


def test_rejects_group_restricted_exposure(catalog, policy, query):
    query = replace(query, served_ids=("tenant-a-legal", "public-1", "tenant-a-general"))

    report = audit_access_scopes(catalog, [query], policy)

    assert "UNAUTHORIZED_DOCUMENT_EXPOSED" in report.reason_codes


def test_any_allowed_group_authorizes_document(catalog, policy, query):
    catalog = [
        *catalog,
        DocumentScope("tenant-a-shared", "tenant", "tenant-a", ("legal", "risk")),
    ]
    query = replace(
        query,
        raw_candidate_ids=(
            "tenant-a-shared",
            "tenant-a-risk",
            "public-1",
            "tenant-b-general",
            "tenant-a-general",
        ),
        served_ids=("tenant-a-shared", "tenant-a-risk", "public-1"),
    )

    assert audit_access_scopes(catalog, [query], policy).accepted is True


def test_rejects_reordered_or_injected_served_projection(catalog, policy, query):
    query = replace(query, served_ids=("public-1", "tenant-a-risk", "tenant-a-general"))

    report = audit_access_scopes(catalog, [query], policy)

    assert report.reason_codes == ("SERVED_PROJECTION_MISMATCH",)
    finding = next(item for item in report.findings if item["code"] == report.reason_codes[0])
    assert finding["expected_count"] == finding["observed_count"] == 3


def test_rejects_silent_post_filter_truncation(catalog, policy, query):
    query = replace(query, served_ids=("tenant-a-risk", "public-1"))

    report = audit_access_scopes(catalog, [query], policy)

    assert "SERVED_PROJECTION_MISMATCH" in report.reason_codes


def test_candidate_underfill_uses_authorized_catalog_not_raw_length(catalog, policy, query):
    query = replace(
        query,
        raw_candidate_ids=(
            "tenant-b-general",
            "tenant-a-risk",
            "public-1",
            "tenant-a-legal",
            "tenant-a-legal",
        ),
    )

    with pytest.raises(AccessAuditError, match="duplicates"):
        audit_access_scopes(catalog, [query], policy)

    underfilled = replace(
        query,
        raw_candidate_ids=(
            "tenant-b-general",
            "tenant-a-risk",
            "public-1",
            "tenant-a-legal",
            "tenant-b-other",
        ),
        served_ids=("tenant-a-risk", "public-1"),
    )
    report = audit_access_scopes(catalog, [underfilled], policy)
    assert "UNDERFILL_BUDGET_EXCEEDED" in report.reason_codes
    assert report.metrics["underfilled_query_count"] == 1


def test_underfill_budget_can_be_explicitly_tolerated(catalog, policy, query):
    policy = replace(policy, max_underfilled_queries=1)
    query = replace(
        query,
        raw_candidate_ids=(
            "tenant-b-general",
            "tenant-a-risk",
            "public-1",
            "tenant-a-legal",
            "tenant-b-missing",
        ),
        served_ids=("tenant-a-risk", "public-1"),
    )

    report = audit_access_scopes(catalog, [query], policy)

    assert "UNDERFILL_BUDGET_EXCEEDED" not in report.reason_codes
    assert "UNKNOWN_CANDIDATE_DOCUMENT" in report.reason_codes


def test_unknown_candidate_and_served_document_fail_closed(catalog, policy, query):
    query = replace(
        query,
        raw_candidate_ids=(
            "missing-doc",
            "tenant-a-risk",
            "public-1",
            "tenant-a-legal",
            "tenant-a-general",
        ),
        served_ids=("missing-doc", "tenant-a-risk", "public-1"),
    )

    report = audit_access_scopes(catalog, [query], policy)

    assert "UNKNOWN_CANDIDATE_DOCUMENT" in report.reason_codes
    assert "UNKNOWN_SERVED_DOCUMENT" in report.reason_codes


def test_insufficient_candidate_depth_is_rejected(catalog, policy, query):
    query = replace(
        query,
        raw_candidate_ids=("tenant-a-risk", "public-1"),
        served_ids=("tenant-a-risk", "public-1"),
    )

    report = audit_access_scopes(catalog, [query], policy)

    assert "INSUFFICIENT_CANDIDATE_DEPTH" in report.reason_codes


def test_work_budget_exhaustion_rejects_without_partial_clean_report(catalog, policy, query):
    report = audit_access_scopes(
        catalog,
        [query],
        replace(policy, max_authorization_checks=1),
    )

    assert report.accepted is False
    assert report.reason_codes == ("AUTHORIZATION_WORK_BUDGET_EXCEEDED",)
    assert report.findings == ()


def test_underfill_budget_cannot_exceed_query_budget(catalog, policy, query):
    with pytest.raises(AccessAuditError, match="cannot exceed"):
        audit_access_scopes(
            catalog,
            [query],
            replace(policy, max_underfilled_queries=21),
        )


def test_deterministic_evidence_ignores_catalog_and_query_order(catalog, policy, query):
    second = replace(
        query,
        query_id="q-2",
        principal=PrincipalScope("analyst-2", "tenant-a", ("risk",)),
    )

    first_report = audit_access_scopes(catalog, [query, second], policy)
    second_report = audit_access_scopes(list(reversed(catalog)), [second, query], policy)

    assert first_report.to_dict() == second_report.to_dict()


@pytest.mark.parametrize(
    "document",
    [
        DocumentScope("bad-public", "public", "tenant-a"),
        DocumentScope("bad-tenant", "tenant"),
        DocumentScope("bad-visibility", "private", "tenant-a"),
    ],
)
def test_invalid_document_scope_is_malformed(catalog, policy, query, document):
    with pytest.raises(AccessAuditError):
        audit_access_scopes([*catalog, document], [query], policy)


@pytest.mark.parametrize(
    "mutation",
    [
        {"requested_k": True},
        {"raw_candidate_ids": ("public-1", "public-1")},
        {"served_ids": ("public-1", "public-1")},
    ],
)
def test_invalid_query_evidence_is_malformed(catalog, policy, query, mutation):
    with pytest.raises(AccessAuditError):
        audit_access_scopes(catalog, [replace(query, **mutation)], policy)


def _artifact() -> dict:
    return {
        "policy": {
            "policy_id": "acl-v1",
            "min_candidate_depth": 3,
            "max_underfilled_queries": 0,
            "max_queries": 10,
            "max_documents": 10,
            "max_candidates_per_query": 10,
            "max_authorization_checks": 100,
            "require_exact_projection": True,
        },
        "catalog": [
            {
                "doc_id": "public-1",
                "visibility": "public",
                "tenant_id": None,
                "allowed_groups": [],
            },
            {
                "doc_id": "tenant-a-1",
                "visibility": "tenant",
                "tenant_id": "tenant-a",
                "allowed_groups": [],
            },
            {
                "doc_id": "tenant-b-1",
                "visibility": "tenant",
                "tenant_id": "tenant-b",
                "allowed_groups": [],
            },
        ],
        "queries": [
            {
                "query_id": "q-1",
                "principal": {
                    "principal_id": "user-a",
                    "tenant_id": "tenant-a",
                    "groups": [],
                },
                "requested_k": 2,
                "raw_candidate_ids": ["tenant-b-1", "tenant-a-1", "public-1"],
                "served_ids": ["tenant-a-1", "public-1"],
            }
        ],
    }


def test_strict_artifact_contract_accepts_valid_evidence():
    assert audit_artifact(_artifact()).accepted is True


def test_duplicate_json_field_and_non_finite_value_are_rejected(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"policy":{},"policy":{}}')
    with pytest.raises(AccessAuditError, match="duplicate"):
        load_artifact(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}')
    with pytest.raises(AccessAuditError, match="non-finite"):
        load_artifact(non_finite)


def test_cli_exit_codes_and_redacted_findings(tmp_path: Path):
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(_artifact()))
    accepted = subprocess.run(
        [sys.executable, "-m", "search.access_audit", str(accepted_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    rejected_artifact = _artifact()
    rejected_artifact["queries"][0]["served_ids"] = ["tenant-b-1", "public-1"]
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected_artifact))
    rejected = subprocess.run(
        [sys.executable, "-m", "search.access_audit", str(rejected_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{broken")
    malformed = subprocess.run(
        [sys.executable, "-m", "search.access_audit", str(malformed_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert accepted.returncode == 0
    assert rejected.returncode == 3
    assert malformed.returncode == 2
    assert "tenant-b-1" not in rejected.stdout
