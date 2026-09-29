from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .engine import Document

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REVISIONS = {"head", "latest", "main", "master", "stable"}
DISTANCE_METRICS = {"cosine", "dot", "euclidean"}
NORMALIZATION_MODES = {"none", "encoder_l2", "engine_l2"}
VECTOR_DTYPES = {"float16", "float32", "float64"}
VECTOR_PACK_FORMATS = {"float16": ">e", "float32": ">f", "float64": ">d"}


class IndexContractError(ValueError):
    """Malformed or operationally untrustworthy index-contract evidence."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ContractPolicy:
    max_age_seconds: int = 30 * 24 * 60 * 60
    max_future_skew_seconds: int = 300
    max_documents: int = 100_000
    max_dimensions: int = 8_192
    max_text_bytes: int = 128_000_000
    max_metadata_bytes: int = 32_000_000
    max_artifact_bytes: int = 16_000_000

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("max_age_seconds", self.max_age_seconds, 1, 365 * 24 * 60 * 60),
            ("max_future_skew_seconds", self.max_future_skew_seconds, 0, 24 * 60 * 60),
            ("max_documents", self.max_documents, 1, 1_000_000),
            ("max_dimensions", self.max_dimensions, 1, 65_536),
            ("max_text_bytes", self.max_text_bytes, 1, 2_000_000_000),
            ("max_metadata_bytes", self.max_metadata_bytes, 1, 1_000_000_000),
            ("max_artifact_bytes", self.max_artifact_bytes, 1, 256_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")

    def evidence(self) -> dict[str, int]:
        return {
            "max_age_seconds": self.max_age_seconds,
            "max_future_skew_seconds": self.max_future_skew_seconds,
            "max_documents": self.max_documents,
            "max_dimensions": self.max_dimensions,
            "max_text_bytes": self.max_text_bytes,
            "max_metadata_bytes": self.max_metadata_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
        }


@dataclass(frozen=True)
class EncoderContract:
    provider_id: str
    model_id: str
    model_revision: str
    dimension: int
    vector_dtype: str = "float32"

    def __post_init__(self) -> None:
        _validate_identifier(self.provider_id, "INVALID_ENCODER_PROVIDER")
        _validate_identifier(self.model_id, "INVALID_ENCODER_MODEL")
        _validate_revision(self.model_revision, "INVALID_ENCODER_REVISION")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise IndexContractError("INVALID_ENCODER_DIMENSION")
        if self.vector_dtype not in VECTOR_DTYPES:
            raise IndexContractError("INVALID_VECTOR_DTYPE")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "vector_dtype": self.vector_dtype,
        }


@dataclass(frozen=True)
class IndexBuildContract:
    index_id: str
    index_revision: str
    corpus_revision: str
    chunker_id: str
    chunker_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    distance_metric: str
    normalization: str
    built_at: datetime

    def __post_init__(self) -> None:
        _validate_identifier(self.index_id, "INVALID_INDEX_ID")
        _validate_revision(self.index_revision, "INVALID_INDEX_REVISION")
        _validate_revision(self.corpus_revision, "INVALID_CORPUS_REVISION")
        _validate_identifier(self.chunker_id, "INVALID_CHUNKER_ID")
        _validate_revision(self.chunker_revision, "INVALID_CHUNKER_REVISION")
        _validate_identifier(self.tokenizer_id, "INVALID_TOKENIZER_ID")
        _validate_revision(self.tokenizer_revision, "INVALID_TOKENIZER_REVISION")
        if self.distance_metric not in DISTANCE_METRICS:
            raise IndexContractError("INVALID_DISTANCE_METRIC")
        if self.normalization not in NORMALIZATION_MODES:
            raise IndexContractError("INVALID_NORMALIZATION_MODE")
        _require_aware_utc(self.built_at, "INVALID_BUILD_TIMESTAMP")

    def as_dict(self) -> dict[str, object]:
        return {
            "index_id": self.index_id,
            "index_revision": self.index_revision,
            "corpus_revision": self.corpus_revision,
            "chunker_id": self.chunker_id,
            "chunker_revision": self.chunker_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "distance_metric": self.distance_metric,
            "normalization": self.normalization,
            "built_at": _timestamp(self.built_at),
        }


@dataclass(frozen=True)
class ServingObservation:
    index_id: str
    index_revision: str
    distance_metric: str
    normalization: str
    query_encoder: EncoderContract

    def __post_init__(self) -> None:
        _validate_identifier(self.index_id, "INVALID_OBSERVED_INDEX_ID")
        _validate_revision(self.index_revision, "INVALID_OBSERVED_INDEX_REVISION")
        if self.distance_metric not in DISTANCE_METRICS:
            raise IndexContractError("INVALID_OBSERVED_DISTANCE_METRIC")
        if self.normalization not in NORMALIZATION_MODES:
            raise IndexContractError("INVALID_OBSERVED_NORMALIZATION")


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    build: IndexBuildContract
    encoder: EncoderContract
    document_count: int
    text_bytes: int
    metadata_bytes: int
    corpus_sha256: str
    embeddings_sha256: str
    manifest_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "build": self.build.as_dict(),
            "encoder": self.encoder.as_dict(),
            "document_count": self.document_count,
            "text_bytes": self.text_bytes,
            "metadata_bytes": self.metadata_bytes,
            "corpus_sha256": self.corpus_sha256,
            "embeddings_sha256": self.embeddings_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "manifest_sha256": self.manifest_sha256}

    def with_digest(self) -> IndexManifest:
        return replace(self, manifest_sha256=_digest(self.body()))


@dataclass(frozen=True)
class CompatibilityReport:
    schema_version: int
    accepted: bool
    reason_codes: tuple[str, ...]
    expected_manifest_sha256: str
    supplied_manifest_sha256: str
    recomputed_manifest_sha256: str
    policy_sha256: str
    document_count: int
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "supplied_manifest_sha256": self.supplied_manifest_sha256,
            "recomputed_manifest_sha256": self.recomputed_manifest_sha256,
            "policy_sha256": self.policy_sha256,
            "document_count": self.document_count,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> CompatibilityReport:
        return replace(self, evidence_sha256=_digest(self.body()))


def _validate_identifier(value: str, code: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise IndexContractError(code)


def _validate_revision(value: str, code: str) -> None:
    _validate_identifier(value, code)
    if value.casefold() in MUTABLE_REVISIONS:
        raise IndexContractError(code)


def _require_aware_utc(value: datetime, code: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise IndexContractError(code)
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _require_aware_utc(value, "INVALID_TIMESTAMP").isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, code: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise IndexContractError(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise IndexContractError(code) from exc
    return _require_aware_utc(parsed, code)


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise IndexContractError("NON_CANONICAL_VALUE") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_document(
    document: Document,
    encoder: EncoderContract,
) -> tuple[dict[str, object], dict[str, object], int, int]:
    _validate_identifier(document.doc_id, "INVALID_DOCUMENT_ID")
    if not isinstance(document.text, str) or not document.text:
        raise IndexContractError("INVALID_DOCUMENT_TEXT")
    text = document.text.encode("utf-8")
    metadata = document.metadata or {}
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise IndexContractError("INVALID_DOCUMENT_METADATA")
    metadata_bytes = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(document.embedding) != encoder.dimension:
        raise IndexContractError("DOCUMENT_DIMENSION_MISMATCH")
    packed = bytearray()
    squared_norm = 0.0
    for value in document.embedding:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise IndexContractError("NON_FINITE_DOCUMENT_EMBEDDING")
        numeric = float(value)
        squared_norm += numeric * numeric
        try:
            packed.extend(struct.pack(VECTOR_PACK_FORMATS[encoder.vector_dtype], numeric))
        except (OverflowError, struct.error) as exc:
            raise IndexContractError("NON_REPRESENTABLE_DOCUMENT_EMBEDDING") from exc
    if squared_norm <= 0.0:
        raise IndexContractError("ZERO_DOCUMENT_EMBEDDING")
    corpus_row = {
        "doc_id_sha256": _sha256_bytes(document.doc_id.encode("utf-8")),
        "text_sha256": _sha256_bytes(text),
        "metadata_sha256": _sha256_bytes(metadata_bytes),
    }
    embedding_row = {
        "doc_id_sha256": corpus_row["doc_id_sha256"],
        "embedding_sha256": _sha256_bytes(bytes(packed)),
    }
    return corpus_row, embedding_row, len(text), len(metadata_bytes)


def build_index_manifest(
    documents: list[Document],
    *,
    build: IndexBuildContract,
    encoder: EncoderContract,
    policy: ContractPolicy | None = None,
) -> IndexManifest:
    active_policy = policy or ContractPolicy()
    if encoder.dimension > active_policy.max_dimensions:
        raise IndexContractError("DIMENSION_BUDGET_EXCEEDED")
    if not documents:
        raise IndexContractError("EMPTY_INDEX")
    if len(documents) > active_policy.max_documents:
        raise IndexContractError("DOCUMENT_BUDGET_EXCEEDED")
    ids = [document.doc_id for document in documents]
    if len(ids) != len(set(ids)):
        raise IndexContractError("DUPLICATE_DOCUMENT_ID")

    corpus_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    text_bytes = 0
    metadata_bytes = 0
    for document in sorted(documents, key=lambda item: item.doc_id):
        corpus, embedding, text_size, metadata_size = _validate_document(document, encoder)
        corpus_rows.append(corpus)
        embedding_rows.append(embedding)
        text_bytes += text_size
        metadata_bytes += metadata_size
        if text_bytes > active_policy.max_text_bytes:
            raise IndexContractError("TEXT_BUDGET_EXCEEDED")
        if metadata_bytes > active_policy.max_metadata_bytes:
            raise IndexContractError("METADATA_BUDGET_EXCEEDED")

    manifest = IndexManifest(
        schema_version=1,
        build=build,
        encoder=encoder,
        document_count=len(documents),
        text_bytes=text_bytes,
        metadata_bytes=metadata_bytes,
        corpus_sha256=_digest(corpus_rows),
        embeddings_sha256=_digest(embedding_rows),
        manifest_sha256="",
    )
    return manifest.with_digest()


def audit_index_compatibility(
    *,
    manifest: IndexManifest,
    expected_manifest_sha256: str,
    loaded_documents: list[Document],
    observation: ServingObservation,
    now: datetime,
    policy: ContractPolicy | None = None,
) -> CompatibilityReport:
    active_policy = policy or ContractPolicy()
    current_time = _require_aware_utc(now, "INVALID_AUDIT_TIMESTAMP")
    if not SHA256.fullmatch(expected_manifest_sha256):
        raise IndexContractError("INVALID_EXPECTED_MANIFEST_DIGEST")
    if manifest.schema_version != 1:
        raise IndexContractError("UNSUPPORTED_MANIFEST_SCHEMA")

    recomputed = build_index_manifest(
        loaded_documents,
        build=manifest.build,
        encoder=manifest.encoder,
        policy=active_policy,
    )
    reasons: list[str] = []
    if manifest.manifest_sha256 != _digest(manifest.body()):
        reasons.append("MANIFEST_INTEGRITY_MISMATCH")
    if manifest.manifest_sha256 != expected_manifest_sha256:
        reasons.append("EXPECTED_MANIFEST_MISMATCH")
    if recomputed.manifest_sha256 != manifest.manifest_sha256:
        if recomputed.document_count != manifest.document_count:
            reasons.append("DOCUMENT_COUNT_MISMATCH")
        if recomputed.corpus_sha256 != manifest.corpus_sha256:
            reasons.append("CORPUS_CONTENT_MISMATCH")
        if recomputed.embeddings_sha256 != manifest.embeddings_sha256:
            reasons.append("EMBEDDING_CONTENT_MISMATCH")
        if (
            recomputed.text_bytes != manifest.text_bytes
            or recomputed.metadata_bytes != manifest.metadata_bytes
        ):
            reasons.append("INDEX_BYTE_ACCOUNTING_MISMATCH")
    if observation.index_id != manifest.build.index_id:
        reasons.append("INDEX_ID_MISMATCH")
    if observation.index_revision != manifest.build.index_revision:
        reasons.append("INDEX_REVISION_MISMATCH")
    if observation.distance_metric != manifest.build.distance_metric:
        reasons.append("DISTANCE_METRIC_MISMATCH")
    if observation.normalization != manifest.build.normalization:
        reasons.append("NORMALIZATION_MISMATCH")
    if observation.query_encoder != manifest.encoder:
        reasons.append("QUERY_ENCODER_MISMATCH")

    built_at = _require_aware_utc(manifest.build.built_at, "INVALID_BUILD_TIMESTAMP")
    age_seconds = (current_time - built_at).total_seconds()
    if age_seconds < -active_policy.max_future_skew_seconds:
        reasons.append("MANIFEST_FROM_FUTURE")
    if age_seconds > active_policy.max_age_seconds:
        reasons.append("MANIFEST_STALE")

    report = CompatibilityReport(
        schema_version=1,
        accepted=not reasons,
        reason_codes=tuple(reasons),
        expected_manifest_sha256=expected_manifest_sha256,
        supplied_manifest_sha256=manifest.manifest_sha256,
        recomputed_manifest_sha256=recomputed.manifest_sha256,
        policy_sha256=_digest(active_policy.evidence()),
        document_count=len(loaded_documents),
        evidence_sha256="",
    )
    return report.with_digest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise IndexContractError("DUPLICATE_JSON_FIELD")
        output[key] = value
    return output


def _reject_constant(_: str) -> None:
    raise IndexContractError("NON_FINITE_JSON_NUMBER")


def _require_fields(
    value: object,
    *,
    required: set[str],
    code: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != required:
        raise IndexContractError(code)
    return value


def _encoder_from_dict(value: object) -> EncoderContract:
    row = _require_fields(
        value,
        required={"provider_id", "model_id", "model_revision", "dimension", "vector_dtype"},
        code="INVALID_ENCODER_ARTIFACT",
    )
    return EncoderContract(**row)  # type: ignore[arg-type]


def _manifest_from_dict(value: object) -> IndexManifest:
    row = _require_fields(
        value,
        required={
            "schema_version",
            "build",
            "encoder",
            "document_count",
            "text_bytes",
            "metadata_bytes",
            "corpus_sha256",
            "embeddings_sha256",
            "manifest_sha256",
        },
        code="INVALID_MANIFEST_ARTIFACT",
    )
    build_row = _require_fields(
        row["build"],
        required={
            "index_id",
            "index_revision",
            "corpus_revision",
            "chunker_id",
            "chunker_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "distance_metric",
            "normalization",
            "built_at",
        },
        code="INVALID_BUILD_ARTIFACT",
    )
    build = IndexBuildContract(
        **{key: value for key, value in build_row.items() if key != "built_at"},
        built_at=_parse_timestamp(build_row["built_at"], "INVALID_BUILD_TIMESTAMP"),
    )  # type: ignore[arg-type]
    encoder = _encoder_from_dict(row["encoder"])
    for field in ("document_count", "text_bytes", "metadata_bytes"):
        value_field = row[field]
        if isinstance(value_field, bool) or not isinstance(value_field, int) or value_field < 0:
            raise IndexContractError("INVALID_MANIFEST_COUNT")
    for field in ("corpus_sha256", "embeddings_sha256", "manifest_sha256"):
        if not isinstance(row[field], str) or not SHA256.fullmatch(row[field]):
            raise IndexContractError("INVALID_MANIFEST_DIGEST")
    if row["schema_version"] != 1:
        raise IndexContractError("UNSUPPORTED_MANIFEST_SCHEMA")
    return IndexManifest(
        schema_version=1,
        build=build,
        encoder=encoder,
        document_count=row["document_count"],  # type: ignore[arg-type]
        text_bytes=row["text_bytes"],  # type: ignore[arg-type]
        metadata_bytes=row["metadata_bytes"],  # type: ignore[arg-type]
        corpus_sha256=row["corpus_sha256"],  # type: ignore[arg-type]
        embeddings_sha256=row["embeddings_sha256"],  # type: ignore[arg-type]
        manifest_sha256=row["manifest_sha256"],  # type: ignore[arg-type]
    )


def _documents_from_dict(value: object) -> list[Document]:
    if not isinstance(value, list):
        raise IndexContractError("INVALID_DOCUMENT_ARTIFACT")
    documents: list[Document] = []
    for item in value:
        row = _require_fields(
            item,
            required={"doc_id", "text", "embedding", "metadata"},
            code="INVALID_DOCUMENT_ARTIFACT",
        )
        embedding = row["embedding"]
        if not isinstance(embedding, list):
            raise IndexContractError("INVALID_DOCUMENT_ARTIFACT")
        documents.append(
            Document(
                doc_id=row["doc_id"],  # type: ignore[arg-type]
                text=row["text"],  # type: ignore[arg-type]
                embedding=tuple(embedding),  # type: ignore[arg-type]
                metadata=row["metadata"],  # type: ignore[arg-type]
            )
        )
    return documents


def _load_input(path: Path, policy: ContractPolicy) -> dict[str, object]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise IndexContractError("ARTIFACT_READ_FAILED") from exc
    if not content or len(content) > policy.max_artifact_bytes:
        raise IndexContractError("ARTIFACT_BYTE_BUDGET_EXCEEDED")
    try:
        value = json.loads(
            content,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexContractError("INVALID_JSON") from exc
    return _require_fields(
        value,
        required={"manifest", "expected_manifest_sha256", "observation", "documents"},
        code="INVALID_AUDIT_ARTIFACT",
    )


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RAG index/encoder serving compatibility")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--now", required=True)
    parser.add_argument("--max-age-seconds", type=int, default=30 * 24 * 60 * 60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = ContractPolicy(max_age_seconds=args.max_age_seconds)
        artifact = _load_input(args.input, policy)
        observation_row = _require_fields(
            artifact["observation"],
            required={
                "index_id",
                "index_revision",
                "distance_metric",
                "normalization",
                "query_encoder",
            },
            code="INVALID_OBSERVATION_ARTIFACT",
        )
        observation = ServingObservation(
            index_id=observation_row["index_id"],  # type: ignore[arg-type]
            index_revision=observation_row["index_revision"],  # type: ignore[arg-type]
            distance_metric=observation_row["distance_metric"],  # type: ignore[arg-type]
            normalization=observation_row["normalization"],  # type: ignore[arg-type]
            query_encoder=_encoder_from_dict(observation_row["query_encoder"]),
        )
        expected = artifact["expected_manifest_sha256"]
        if not isinstance(expected, str):
            raise IndexContractError("INVALID_EXPECTED_MANIFEST_DIGEST")
        report = audit_index_compatibility(
            manifest=_manifest_from_dict(artifact["manifest"]),
            expected_manifest_sha256=expected,
            loaded_documents=_documents_from_dict(artifact["documents"]),
            observation=observation,
            now=_parse_timestamp(args.now, "INVALID_AUDIT_TIMESTAMP"),
            policy=policy,
        )
        _atomic_write(args.output, report.as_dict())
        return 0 if report.accepted else 2
    except (IndexContractError, ValueError, TypeError) as exc:
        code = exc.code if isinstance(exc, IndexContractError) else "INVALID_AUDIT_INPUT"
        _atomic_write(
            args.output,
            {"accepted": False, "error_code": code, "schema_version": 1},
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
