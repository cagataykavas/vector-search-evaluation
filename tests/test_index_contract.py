import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from search.engine import Document, HybridSearchEngine
from search.index_contract import (
    ContractPolicy,
    EncoderContract,
    IndexBuildContract,
    IndexContractError,
    ServingObservation,
    audit_index_compatibility,
    build_index_manifest,
    main,
)

NOW = datetime(2026, 9, 30, 0, 0, tzinfo=UTC)


def documents() -> list[Document]:
    return [
        Document("doc-a", "alpha runbook", (1.0, 0.0, 0.0), {"source": "ops"}),
        Document("doc-b", "beta handbook", (0.0, 1.0, 0.0), {"source": "hr"}),
    ]


def encoder(**overrides: object) -> EncoderContract:
    values = {
        "provider_id": "internal",
        "model_id": "embed-v3",
        "model_revision": "sha256:abc123",
        "dimension": 3,
        "vector_dtype": "float32",
        **overrides,
    }
    return EncoderContract(**values)


def build(**overrides: object) -> IndexBuildContract:
    values = {
        "index_id": "support-kb",
        "index_revision": "idx-20260929.1",
        "corpus_revision": "corpus-42",
        "chunker_id": "recursive",
        "chunker_revision": "chunk-7",
        "tokenizer_id": "cl100k_base",
        "tokenizer_revision": "tok-2025.1",
        "distance_metric": "cosine",
        "normalization": "engine_l2",
        "built_at": NOW - timedelta(hours=1),
        **overrides,
    }
    return IndexBuildContract(**values)


def observation(**overrides: object) -> ServingObservation:
    values = {
        "index_id": "support-kb",
        "index_revision": "idx-20260929.1",
        "distance_metric": "cosine",
        "normalization": "engine_l2",
        "query_encoder": encoder(),
        **overrides,
    }
    return ServingObservation(**values)


def audit(
    loaded: list[Document] | None = None,
    *,
    observed: ServingObservation | None = None,
    now: datetime = NOW,
    expected: str | None = None,
    active_build: IndexBuildContract | None = None,
):
    rows = documents()
    manifest = build_index_manifest(rows, build=active_build or build(), encoder=encoder())
    return audit_index_compatibility(
        manifest=manifest,
        expected_manifest_sha256=expected or manifest.manifest_sha256,
        loaded_documents=rows if loaded is None else loaded,
        observation=observed or observation(),
        now=now,
    )


def test_exact_index_and_query_encoder_contract_is_accepted() -> None:
    report = audit()

    assert report.accepted
    assert report.reason_codes == ()
    assert report.supplied_manifest_sha256 == report.recomputed_manifest_sha256
    assert len(report.evidence_sha256) == 64


def test_real_hybrid_engine_accepts_manifested_documents() -> None:
    rows = documents()
    manifest = build_index_manifest(rows, build=build(), encoder=encoder())
    engine = HybridSearchEngine(rows)
    hits = engine.search(query_text="alpha", query_embedding=(1.0, 0.0, 0.0), k=1)
    report = audit_index_compatibility(
        manifest=manifest,
        expected_manifest_sha256=manifest.manifest_sha256,
        loaded_documents=engine.documents,
        observation=observation(),
        now=NOW,
    )

    assert hits[0].doc_id == "doc-a"
    assert report.accepted


@pytest.mark.parametrize(
    ("observed", "reason"),
    [
        (observation(index_id="other-index"), "INDEX_ID_MISMATCH"),
        (observation(index_revision="idx-20260929.2"), "INDEX_REVISION_MISMATCH"),
        (observation(distance_metric="dot"), "DISTANCE_METRIC_MISMATCH"),
        (observation(normalization="none"), "NORMALIZATION_MISMATCH"),
        (
            observation(query_encoder=encoder(model_revision="sha256:def456")),
            "QUERY_ENCODER_MISMATCH",
        ),
        (observation(query_encoder=encoder(dimension=4)), "QUERY_ENCODER_MISMATCH"),
    ],
)
def test_serving_contract_drift_is_rejected(observed: ServingObservation, reason: str) -> None:
    report = audit(observed=observed)
    assert not report.accepted
    assert report.reason_codes == (reason,)


def test_changed_text_metadata_and_embedding_are_independently_visible() -> None:
    text_changed = documents()
    text_changed[0] = replace(text_changed[0], text="tampered runbook")
    metadata_changed = documents()
    metadata_changed[0] = replace(metadata_changed[0], metadata={"source": "unknown"})
    embedding_changed = documents()
    embedding_changed[0] = replace(embedding_changed[0], embedding=(0.9, 0.1, 0.0))

    text_report = audit(text_changed)
    metadata_report = audit(metadata_changed)
    embedding_report = audit(embedding_changed)

    assert "CORPUS_CONTENT_MISMATCH" in text_report.reason_codes
    assert "INDEX_BYTE_ACCOUNTING_MISMATCH" in text_report.reason_codes
    assert "CORPUS_CONTENT_MISMATCH" in metadata_report.reason_codes
    assert "EMBEDDING_CONTENT_MISMATCH" in embedding_report.reason_codes


def test_added_or_missing_document_changes_count_and_content() -> None:
    report = audit(documents()[:1])
    assert not report.accepted
    assert report.reason_codes == (
        "DOCUMENT_COUNT_MISMATCH",
        "CORPUS_CONTENT_MISMATCH",
        "EMBEDDING_CONTENT_MISMATCH",
        "INDEX_BYTE_ACCOUNTING_MISMATCH",
    )


def test_expected_digest_and_manifest_integrity_are_separate_boundaries() -> None:
    rows = documents()
    manifest = build_index_manifest(rows, build=build(), encoder=encoder())
    wrong_expected = "0" * 64
    expected_report = audit(expected=wrong_expected)
    tampered = replace(manifest, document_count=99)
    integrity_report = audit_index_compatibility(
        manifest=tampered,
        expected_manifest_sha256=manifest.manifest_sha256,
        loaded_documents=rows,
        observation=observation(),
        now=NOW,
    )

    assert expected_report.reason_codes == ("EXPECTED_MANIFEST_MISMATCH",)
    assert integrity_report.reason_codes == ("MANIFEST_INTEGRITY_MISMATCH",)


def test_stale_and_future_manifests_fail_closed() -> None:
    stale = audit(active_build=build(built_at=NOW - timedelta(days=31)))
    future = audit(active_build=build(built_at=NOW + timedelta(minutes=6)))

    assert stale.reason_codes == ("MANIFEST_STALE",)
    assert future.reason_codes == ("MANIFEST_FROM_FUTURE",)


def test_manifest_is_invariant_to_document_and_metadata_order() -> None:
    first = build_index_manifest(documents(), build=build(), encoder=encoder())
    reversed_rows = list(reversed(documents()))
    reversed_rows[0] = replace(reversed_rows[0], metadata={"source": "hr"})
    second = build_index_manifest(reversed_rows, build=build(), encoder=encoder())

    assert first.as_dict() == second.as_dict()


def test_embedding_digest_uses_declared_vector_dtype() -> None:
    float32 = build_index_manifest(documents(), build=build(), encoder=encoder())
    float64 = build_index_manifest(
        documents(), build=build(), encoder=encoder(vector_dtype="float64")
    )

    assert float32.embeddings_sha256 != float64.embeddings_sha256


def test_embedding_must_be_representable_in_declared_dtype() -> None:
    rows = [replace(documents()[0], embedding=(1e100, 0.0, 0.0))]

    with pytest.raises(IndexContractError, match="NON_REPRESENTABLE_DOCUMENT_EMBEDDING"):
        build_index_manifest(rows, build=build(), encoder=encoder(vector_dtype="float16"))


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([], "EMPTY_INDEX"),
        (documents() + [documents()[0]], "DUPLICATE_DOCUMENT_ID"),
        ([replace(documents()[0], embedding=(1.0, 0.0))], "DOCUMENT_DIMENSION_MISMATCH"),
        ([replace(documents()[0], embedding=(0.0, 0.0, 0.0))], "ZERO_DOCUMENT_EMBEDDING"),
        (
            [replace(documents()[0], embedding=(float("nan"), 0.0, 0.0))],
            "NON_FINITE_DOCUMENT_EMBEDDING",
        ),
        ([replace(documents()[0], metadata={"bad": 1})], "INVALID_DOCUMENT_METADATA"),
    ],
)
def test_malformed_document_inventory_is_rejected(rows: list[Document], code: str) -> None:
    with pytest.raises(IndexContractError, match=code):
        build_index_manifest(rows, build=build(), encoder=encoder())


@pytest.mark.parametrize("revision", ["latest", "main", "HEAD"])
def test_mutable_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(IndexContractError, match="INVALID_ENCODER_REVISION"):
        encoder(model_revision=revision)


def test_resource_budgets_fail_closed() -> None:
    with pytest.raises(IndexContractError, match="DOCUMENT_BUDGET_EXCEEDED"):
        build_index_manifest(
            documents(), build=build(), encoder=encoder(), policy=ContractPolicy(max_documents=1)
        )
    with pytest.raises(IndexContractError, match="TEXT_BUDGET_EXCEEDED"):
        build_index_manifest(
            documents(), build=build(), encoder=encoder(), policy=ContractPolicy(max_text_bytes=1)
        )
    with pytest.raises(IndexContractError, match="DIMENSION_BUDGET_EXCEEDED"):
        build_index_manifest(
            documents(), build=build(), encoder=encoder(), policy=ContractPolicy(max_dimensions=2)
        )


def artifact() -> dict[str, object]:
    rows = documents()
    manifest = build_index_manifest(rows, build=build(), encoder=encoder())
    return {
        "manifest": manifest.as_dict(),
        "expected_manifest_sha256": manifest.manifest_sha256,
        "observation": {
            "index_id": "support-kb",
            "index_revision": "idx-20260929.1",
            "distance_metric": "cosine",
            "normalization": "engine_l2",
            "query_encoder": encoder().as_dict(),
        },
        "documents": [
            {
                "doc_id": row.doc_id,
                "text": row.text,
                "embedding": list(row.embedding),
                "metadata": row.metadata,
            }
            for row in rows
        ],
    }


def test_cli_returns_distinct_accept_reject_and_malformed_codes(tmp_path: Path) -> None:
    accepted_input = tmp_path / "accepted-input.json"
    rejected_input = tmp_path / "rejected-input.json"
    malformed_input = tmp_path / "malformed-input.json"
    accepted_output = tmp_path / "accepted-output.json"
    rejected_output = tmp_path / "rejected-output.json"
    malformed_output = tmp_path / "malformed-output.json"
    accepted_input.write_text(json.dumps(artifact()), encoding="utf-8")
    rejected = artifact()
    rejected["observation"]["index_revision"] = "idx-20260929.2"  # type: ignore[index]
    rejected_input.write_text(json.dumps(rejected), encoding="utf-8")
    malformed_input.write_text('{"manifest": 1, "manifest": 2}', encoding="utf-8")

    common = ["--now", "2026-09-30T00:00:00Z"]
    assert main(["--input", str(accepted_input), "--output", str(accepted_output), *common]) == 0
    assert main(["--input", str(rejected_input), "--output", str(rejected_output), *common]) == 2
    assert main(["--input", str(malformed_input), "--output", str(malformed_output), *common]) == 3

    assert json.loads(accepted_output.read_text())["accepted"] is True
    assert json.loads(rejected_output.read_text())["reason_codes"] == ["INDEX_REVISION_MISMATCH"]
    assert json.loads(malformed_output.read_text())["error_code"] == "DUPLICATE_JSON_FIELD"


def test_report_contains_no_raw_document_or_model_identifiers() -> None:
    report = audit()
    serialized = json.dumps(report.as_dict(), sort_keys=True)

    for sensitive in ("doc-a", "doc-b", "support-kb", "embed-v3", "alpha runbook"):
        assert sensitive not in serialized
