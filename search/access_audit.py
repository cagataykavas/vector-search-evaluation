"""Fail-closed access-scope audit for multi-tenant retrieval rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MAX_INPUT_BYTES = 2 * 1024 * 1024
_ABSOLUTE_MAX_ITEMS = 100_000


class AccessAuditError(ValueError):
    """Raised when an access-audit artifact is malformed."""


@dataclass(frozen=True)
class DocumentScope:
    doc_id: str
    visibility: str
    tenant_id: str | None = None
    allowed_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrincipalScope:
    principal_id: str
    tenant_id: str
    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopedQuery:
    query_id: str
    principal: PrincipalScope
    requested_k: int
    raw_candidate_ids: tuple[str, ...]
    served_ids: tuple[str, ...]


@dataclass(frozen=True)
class AccessAuditPolicy:
    policy_id: str
    min_candidate_depth: int = 20
    max_underfilled_queries: int = 0
    max_queries: int = 1_000
    max_documents: int = 50_000
    max_candidates_per_query: int = 1_000
    max_authorization_checks: int = 1_000_000
    require_exact_projection: bool = True


@dataclass(frozen=True)
class AccessAuditReport:
    accepted: bool
    reason_codes: tuple[str, ...]
    metrics: dict[str, int | float]
    findings: tuple[dict[str, Any], ...]
    evidence: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "metrics": self.metrics,
            "findings": list(self.findings),
            "evidence": self.evidence,
        }


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise AccessAuditError(f"{field} must be a bounded identifier")
    return value


def _optional_identifier(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field)


def _identifier_array(
    value: object,
    field: str,
    *,
    maximum: int = _ABSOLUTE_MAX_ITEMS,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise AccessAuditError(f"{field} must be an array")
    if (not allow_empty and not value) or len(value) > maximum:
        raise AccessAuditError(f"{field} has an invalid item count")
    items = tuple(_identifier(item, field) for item in value)
    if len(items) != len(set(items)):
        raise AccessAuditError(f"{field} contains duplicates")
    return items


def _bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise AccessAuditError(f"{field} must be between {minimum} and {maximum}")
    return value


def _bounded_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise AccessAuditError(f"{field} must be a boolean")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_from_dict(payload: object) -> DocumentScope:
    if not isinstance(payload, dict):
        raise AccessAuditError("catalog rows must be objects")
    if set(payload) != {"doc_id", "visibility", "tenant_id", "allowed_groups"}:
        raise AccessAuditError("catalog row fields are not exact")
    visibility = payload["visibility"]
    if visibility not in {"public", "tenant"}:
        raise AccessAuditError("visibility must be public or tenant")
    tenant_id = _optional_identifier(payload["tenant_id"], "tenant_id")
    groups = _identifier_array(payload["allowed_groups"], "allowed_groups", maximum=64)
    if visibility == "public" and (tenant_id is not None or groups):
        raise AccessAuditError("public documents cannot carry tenant or group restrictions")
    if visibility == "tenant" and tenant_id is None:
        raise AccessAuditError("tenant documents require tenant_id")
    return DocumentScope(
        doc_id=_identifier(payload["doc_id"], "doc_id"),
        visibility=visibility,
        tenant_id=tenant_id,
        allowed_groups=groups,
    )


def principal_from_dict(payload: object) -> PrincipalScope:
    if not isinstance(payload, dict):
        raise AccessAuditError("principal must be an object")
    if set(payload) != {"principal_id", "tenant_id", "groups"}:
        raise AccessAuditError("principal fields are not exact")
    return PrincipalScope(
        principal_id=_identifier(payload["principal_id"], "principal_id"),
        tenant_id=_identifier(payload["tenant_id"], "tenant_id"),
        groups=_identifier_array(payload["groups"], "groups", maximum=64),
    )


def query_from_dict(payload: object) -> ScopedQuery:
    if not isinstance(payload, dict):
        raise AccessAuditError("queries must be objects")
    required = {
        "query_id",
        "principal",
        "requested_k",
        "raw_candidate_ids",
        "served_ids",
    }
    if set(payload) != required:
        raise AccessAuditError("query fields are not exact")
    requested_k = _bounded_int(payload["requested_k"], "requested_k", 1, 1_000)
    served = _identifier_array(payload["served_ids"], "served_ids", maximum=requested_k)
    return ScopedQuery(
        query_id=_identifier(payload["query_id"], "query_id"),
        principal=principal_from_dict(payload["principal"]),
        requested_k=requested_k,
        raw_candidate_ids=_identifier_array(
            payload["raw_candidate_ids"],
            "raw_candidate_ids",
            maximum=_ABSOLUTE_MAX_ITEMS,
        ),
        served_ids=served,
    )


def policy_from_dict(payload: object) -> AccessAuditPolicy:
    if not isinstance(payload, dict):
        raise AccessAuditError("policy must be an object")
    allowed = {
        "policy_id",
        "min_candidate_depth",
        "max_underfilled_queries",
        "max_queries",
        "max_documents",
        "max_candidates_per_query",
        "max_authorization_checks",
        "require_exact_projection",
    }
    if not set(payload).issubset(allowed) or "policy_id" not in payload:
        raise AccessAuditError("policy contains unknown fields or omits policy_id")
    return AccessAuditPolicy(
        policy_id=_identifier(payload["policy_id"], "policy_id"),
        min_candidate_depth=_bounded_int(
            payload.get("min_candidate_depth", 20), "min_candidate_depth", 1, 10_000
        ),
        max_underfilled_queries=_bounded_int(
            payload.get("max_underfilled_queries", 0),
            "max_underfilled_queries",
            0,
            10_000,
        ),
        max_queries=_bounded_int(payload.get("max_queries", 1_000), "max_queries", 1, 10_000),
        max_documents=_bounded_int(
            payload.get("max_documents", 50_000), "max_documents", 1, 100_000
        ),
        max_candidates_per_query=_bounded_int(
            payload.get("max_candidates_per_query", 1_000),
            "max_candidates_per_query",
            1,
            10_000,
        ),
        max_authorization_checks=_bounded_int(
            payload.get("max_authorization_checks", 1_000_000),
            "max_authorization_checks",
            1,
            10_000_000,
        ),
        require_exact_projection=_bounded_bool(
            payload.get("require_exact_projection", True), "require_exact_projection"
        ),
    )


def _normalize_document(document: DocumentScope) -> DocumentScope:
    return document_from_dict(
        {
            "doc_id": document.doc_id,
            "visibility": document.visibility,
            "tenant_id": document.tenant_id,
            "allowed_groups": list(document.allowed_groups),
        }
    )


def _normalize_query(query: ScopedQuery) -> ScopedQuery:
    return query_from_dict(
        {
            "query_id": query.query_id,
            "principal": {
                "principal_id": query.principal.principal_id,
                "tenant_id": query.principal.tenant_id,
                "groups": list(query.principal.groups),
            },
            "requested_k": query.requested_k,
            "raw_candidate_ids": list(query.raw_candidate_ids),
            "served_ids": list(query.served_ids),
        }
    )


def _normalize_policy(policy: AccessAuditPolicy) -> AccessAuditPolicy:
    return policy_from_dict(
        {
            "policy_id": policy.policy_id,
            "min_candidate_depth": policy.min_candidate_depth,
            "max_underfilled_queries": policy.max_underfilled_queries,
            "max_queries": policy.max_queries,
            "max_documents": policy.max_documents,
            "max_candidates_per_query": policy.max_candidates_per_query,
            "max_authorization_checks": policy.max_authorization_checks,
            "require_exact_projection": policy.require_exact_projection,
        }
    )


def is_authorized(document: DocumentScope, principal: PrincipalScope) -> bool:
    """Return whether a principal may receive a document."""

    if document.visibility == "public":
        return True
    if document.tenant_id != principal.tenant_id:
        return False
    return not document.allowed_groups or bool(
        set(document.allowed_groups).intersection(principal.groups)
    )


def audit_access_scopes(
    catalog: list[DocumentScope] | tuple[DocumentScope, ...],
    queries: list[ScopedQuery] | tuple[ScopedQuery, ...],
    policy: AccessAuditPolicy,
) -> AccessAuditReport:
    """Audit served rankings against an authoritative access catalog."""

    policy = _normalize_policy(policy)
    if policy.max_underfilled_queries > policy.max_queries:
        raise AccessAuditError("max_underfilled_queries cannot exceed max_queries")
    catalog = tuple(_normalize_document(document) for document in catalog)
    queries = tuple(_normalize_query(query) for query in queries)
    if not catalog or len(catalog) > policy.max_documents:
        raise AccessAuditError("catalog is empty or exceeds max_documents")
    if not queries or len(queries) > policy.max_queries:
        raise AccessAuditError("queries are empty or exceed max_queries")
    doc_ids = [document.doc_id for document in catalog]
    if len(doc_ids) != len(set(doc_ids)):
        raise AccessAuditError("catalog contains duplicate doc_id values")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise AccessAuditError("queries contain duplicate query_id values")
    if any(len(query.raw_candidate_ids) > policy.max_candidates_per_query for query in queries):
        raise AccessAuditError("a query exceeds max_candidates_per_query")

    estimated_checks = sum(
        len(catalog) + len(query.raw_candidate_ids) + len(query.served_ids) for query in queries
    )
    catalog_record = [
        {
            "doc_id": document.doc_id,
            "visibility": document.visibility,
            "tenant_id": document.tenant_id,
            "allowed_groups": sorted(document.allowed_groups),
        }
        for document in sorted(catalog, key=lambda item: item.doc_id)
    ]
    query_record = [
        {
            "query_id": query.query_id,
            "principal_id": query.principal.principal_id,
            "tenant_id": query.principal.tenant_id,
            "groups": sorted(query.principal.groups),
            "requested_k": query.requested_k,
            "raw_candidate_ids": list(query.raw_candidate_ids),
            "served_ids": list(query.served_ids),
        }
        for query in sorted(queries, key=lambda item: item.query_id)
    ]
    policy_record = {
        "policy_id": policy.policy_id,
        "min_candidate_depth": policy.min_candidate_depth,
        "max_underfilled_queries": policy.max_underfilled_queries,
        "max_queries": policy.max_queries,
        "max_documents": policy.max_documents,
        "max_candidates_per_query": policy.max_candidates_per_query,
        "max_authorization_checks": policy.max_authorization_checks,
        "require_exact_projection": policy.require_exact_projection,
    }
    evidence_core = {
        "schema_version": "access-scoped-retrieval/v1",
        "policy_id": policy.policy_id,
        "catalog_sha256": _digest(catalog_record),
        "queries_sha256": _digest(query_record),
        "policy_sha256": _digest(policy_record),
    }
    if estimated_checks > policy.max_authorization_checks:
        reasons = ("AUTHORIZATION_WORK_BUDGET_EXCEEDED",)
        metrics: dict[str, int | float] = {
            "query_count": len(queries),
            "catalog_document_count": len(catalog),
            "authorization_checks": estimated_checks,
            "unauthorized_exposure_count": 0,
            "underfilled_query_count": 0,
            "projection_mismatch_count": 0,
        }
        evidence = {
            **evidence_core,
            "report_sha256": _digest(
                {**evidence_core, "reason_codes": reasons, "metrics": metrics, "findings": []}
            ),
        }
        return AccessAuditReport(False, reasons, metrics, (), evidence)

    catalog_by_id = {document.doc_id: document for document in catalog}
    findings: list[dict[str, Any]] = []
    unauthorized_exposures = 0
    unknown_candidates = 0
    unknown_served = 0
    underfilled_queries = 0
    projection_mismatches = 0
    total_served = 0

    for query in sorted(queries, key=lambda item: item.query_id):
        query_hash = _identity_digest(query.query_id)
        principal_hash = _identity_digest(query.principal.principal_id)
        known_candidates: list[DocumentScope] = []
        for rank, doc_id in enumerate(query.raw_candidate_ids, start=1):
            document = catalog_by_id.get(doc_id)
            if document is None:
                unknown_candidates += 1
                findings.append(
                    {
                        "code": "UNKNOWN_CANDIDATE_DOCUMENT",
                        "query_id_sha256": query_hash,
                        "rank": rank,
                        "document_id_sha256": _identity_digest(doc_id),
                    }
                )
            else:
                known_candidates.append(document)

        authorized_catalog_count = sum(
            is_authorized(document, query.principal) for document in catalog
        )
        authorized_candidates = [
            document.doc_id
            for document in known_candidates
            if is_authorized(document, query.principal)
        ]
        expected_target = min(query.requested_k, authorized_catalog_count)
        expected_served = tuple(authorized_candidates[: query.requested_k])
        if len(expected_served) < expected_target:
            underfilled_queries += 1
            findings.append(
                {
                    "code": "AUTHORIZED_CANDIDATE_UNDERFILL",
                    "query_id_sha256": query_hash,
                    "principal_id_sha256": principal_hash,
                    "required_count": expected_target,
                    "available_candidate_count": len(expected_served),
                }
            )

        required_depth = min(policy.min_candidate_depth, len(catalog))
        if len(query.raw_candidate_ids) < required_depth:
            findings.append(
                {
                    "code": "INSUFFICIENT_CANDIDATE_DEPTH",
                    "query_id_sha256": query_hash,
                    "observed_depth": len(query.raw_candidate_ids),
                    "required_depth": required_depth,
                }
            )

        for rank, doc_id in enumerate(query.served_ids, start=1):
            document = catalog_by_id.get(doc_id)
            if document is None:
                unknown_served += 1
                findings.append(
                    {
                        "code": "UNKNOWN_SERVED_DOCUMENT",
                        "query_id_sha256": query_hash,
                        "rank": rank,
                        "document_id_sha256": _identity_digest(doc_id),
                    }
                )
            elif not is_authorized(document, query.principal):
                unauthorized_exposures += 1
                findings.append(
                    {
                        "code": "UNAUTHORIZED_DOCUMENT_EXPOSED",
                        "query_id_sha256": query_hash,
                        "principal_id_sha256": principal_hash,
                        "rank": rank,
                        "document_id_sha256": _identity_digest(doc_id),
                    }
                )

        if policy.require_exact_projection and query.served_ids != expected_served:
            projection_mismatches += 1
            findings.append(
                {
                    "code": "SERVED_PROJECTION_MISMATCH",
                    "query_id_sha256": query_hash,
                    "expected_count": len(expected_served),
                    "observed_count": len(query.served_ids),
                    "expected_sha256": _digest(list(expected_served)),
                    "observed_sha256": _digest(list(query.served_ids)),
                }
            )
        total_served += len(query.served_ids)

    reasons: list[str] = []
    finding_codes = {finding["code"] for finding in findings}
    for code in (
        "UNAUTHORIZED_DOCUMENT_EXPOSED",
        "UNKNOWN_SERVED_DOCUMENT",
        "UNKNOWN_CANDIDATE_DOCUMENT",
        "INSUFFICIENT_CANDIDATE_DEPTH",
        "SERVED_PROJECTION_MISMATCH",
    ):
        if code in finding_codes:
            reasons.append(code)
    if underfilled_queries > policy.max_underfilled_queries:
        reasons.append("UNDERFILL_BUDGET_EXCEEDED")

    metrics = {
        "query_count": len(queries),
        "catalog_document_count": len(catalog),
        "authorization_checks": estimated_checks,
        "served_document_count": total_served,
        "unauthorized_exposure_count": unauthorized_exposures,
        "unknown_candidate_count": unknown_candidates,
        "unknown_served_count": unknown_served,
        "underfilled_query_count": underfilled_queries,
        "projection_mismatch_count": projection_mismatches,
        "served_fill_rate": round(total_served / sum(query.requested_k for query in queries), 6),
    }
    finding_records = tuple(
        sorted(findings, key=lambda item: (str(item["query_id_sha256"]), str(item["code"])))
    )
    evidence = {
        **evidence_core,
        "report_sha256": _digest(
            {
                **evidence_core,
                "reason_codes": reasons,
                "metrics": metrics,
                "findings": finding_records,
            }
        ),
    }
    return AccessAuditReport(not reasons, tuple(reasons), metrics, finding_records, evidence)


def _reject_constant(value: str) -> None:
    raise AccessAuditError(f"non-finite JSON value is not allowed: {value}")


def _without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise AccessAuditError(f"duplicate JSON field: {key}")
        output[key] = value
    return output


def load_artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if not raw or len(raw) > _MAX_INPUT_BYTES:
        raise AccessAuditError("artifact is empty or exceeds the byte budget")
    try:
        payload = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AccessAuditError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AccessAuditError("artifact must be an object")
    return payload


def audit_artifact(payload: dict[str, object]) -> AccessAuditReport:
    if set(payload) != {"policy", "catalog", "queries"}:
        raise AccessAuditError("artifact fields must be policy, catalog and queries")
    if not isinstance(payload["catalog"], list) or not isinstance(payload["queries"], list):
        raise AccessAuditError("catalog and queries must be arrays")
    return audit_access_scopes(
        [document_from_dict(item) for item in payload["catalog"]],
        [query_from_dict(item) for item in payload["queries"]],
        policy_from_dict(payload["policy"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit access-scoped retrieval rankings")
    parser.add_argument("artifact", type=Path, help="JSON access-audit artifact")
    args = parser.parse_args(argv)
    try:
        report = audit_artifact(load_artifact(args.artifact))
    except (AccessAuditError, OSError) as exc:
        print(json.dumps({"accepted": False, "error": "MALFORMED_ARTIFACT", "detail": str(exc)}))
        return 2
    print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.accepted else 3


if __name__ == "__main__":
    sys.exit(main())
