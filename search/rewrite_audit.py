from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
TOKEN_RE = re.compile(r"\w+(?:[-./:]\w+)*", re.UNICODE)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
UUID_RE = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![0-9a-f])",
    re.IGNORECASE,
)
MIXED_ID_RE = re.compile(r"(?<!\w)(?=[\w./:-]*[A-Za-z])(?=[\w./:-]*\d)[\w./:-]+(?!\w)")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w.])")

NEGATIONS = {
    "en": frozenset(
        {
            "no",
            "not",
            "never",
            "without",
            "exclude",
            "excluding",
            "except",
            "neither",
            "nor",
        }
    ),
    "tr": frozenset(
        {
            "değil",
            "hariç",
            "olmadan",
            "olmayan",
            "yok",
            "dışında",
            "dışındaki",
        }
    ),
}
SUPPORTED_LANGUAGES = frozenset({"en", "tr", "und"})
PROTECTED_KINDS = frozenset({"identifier", "entity", "phrase", "constraint"})


class ArtifactMalformed(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RewritePolicy:
    max_input_bytes: int = 524_288
    max_cases: int = 1_000
    max_rewrites_per_case: int = 8
    max_query_chars: int = 4_096
    max_total_query_chars: int = 1_000_000
    max_rewrite_tokens: int = 256
    max_growth_ratio: float = 3.0
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    max_findings: int = 256

    def __post_init__(self) -> None:
        checks = (
            (1 <= self.max_input_bytes <= 8_388_608, "max_input_bytes"),
            (1 <= self.max_cases <= 10_000, "max_cases"),
            (1 <= self.max_rewrites_per_case <= 64, "max_rewrites_per_case"),
            (1 <= self.max_query_chars <= 65_536, "max_query_chars"),
            (1 <= self.max_total_query_chars <= 20_000_000, "max_total_query_chars"),
            (1 <= self.max_rewrite_tokens <= 4_096, "max_rewrite_tokens"),
            (1.0 <= self.max_growth_ratio <= 20.0, "max_growth_ratio"),
            (1 <= self.max_age_seconds <= 604_800, "max_age_seconds"),
            (0 <= self.max_future_skew_seconds <= 3_600, "max_future_skew_seconds"),
            (1 <= self.max_findings <= 10_000, "max_findings"),
        )
        for valid, name in checks:
            if not valid:
                raise ValueError(f"{name} outside supported range")


@dataclass(frozen=True)
class Finding:
    case_sha256: str
    rewrite_sha256: str
    code: str


@dataclass(frozen=True)
class RewriteAuditReport:
    schema_version: int
    status: str
    accepted: bool
    artifact_sha256: str
    policy_sha256: str
    case_count: int
    rewrite_count: int
    rejected_rewrite_count: int
    finding_count: int
    findings_truncated: bool
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["findings"] = [asdict(finding) for finding in self.findings]
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactMalformed("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ArtifactMalformed("NON_FINITE_NUMBER")


def load_artifact(raw: bytes, policy: RewritePolicy | None = None) -> dict[str, Any]:
    selected_policy = policy or RewritePolicy()
    if len(raw) > selected_policy.max_input_bytes:
        raise ArtifactMalformed("INPUT_TOO_LARGE")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactMalformed("INVALID_UTF8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ArtifactMalformed:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactMalformed("INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactMalformed("ROOT_NOT_OBJECT")
    return value


def audit_rewrites(
    artifact: dict[str, Any],
    *,
    as_of: datetime,
    policy: RewritePolicy | None = None,
) -> RewriteAuditReport:
    selected_policy = policy or RewritePolicy()
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    _expect_keys(
        artifact,
        {
            "schema_version",
            "benchmark_id",
            "rewriter_model_sha256",
            "prompt_sha256",
            "retrieval_policy_sha256",
            "created_at",
            "cases",
        },
        "ROOT_FIELDS",
    )
    if artifact["schema_version"] != 1 or isinstance(artifact["schema_version"], bool):
        raise ArtifactMalformed("SCHEMA_VERSION")
    _identifier(artifact["benchmark_id"], "BENCHMARK_ID")
    _digest(artifact["rewriter_model_sha256"], "REWRITER_MODEL_SHA256")
    _digest(artifact["prompt_sha256"], "PROMPT_SHA256")
    _digest(artifact["retrieval_policy_sha256"], "RETRIEVAL_POLICY_SHA256")
    created_at = _timestamp(artifact["created_at"], "CREATED_AT")
    if created_at > as_of + timedelta(seconds=selected_policy.max_future_skew_seconds):
        raise ArtifactMalformed("EVIDENCE_FROM_FUTURE")
    if as_of - created_at > timedelta(seconds=selected_policy.max_age_seconds):
        raise ArtifactMalformed("STALE_EVIDENCE")

    cases = artifact["cases"]
    if not isinstance(cases, list) or not cases:
        raise ArtifactMalformed("CASES_REQUIRED")
    if len(cases) > selected_policy.max_cases:
        raise ArtifactMalformed("CASE_BUDGET_EXCEEDED")

    seen_cases: set[str] = set()
    all_findings: list[Finding] = []
    rejected_rewrites: set[tuple[str, str]] = set()
    rewrite_count = 0
    total_chars = 0
    canonical_cases: list[dict[str, Any]] = []
    for case in cases:
        case_findings, canonical_case, case_chars, case_rewrites = _audit_case(
            case,
            policy=selected_policy,
            seen_cases=seen_cases,
        )
        all_findings.extend(case_findings)
        rejected_rewrites.update(
            (finding.case_sha256, finding.rewrite_sha256) for finding in case_findings
        )
        total_chars += case_chars
        rewrite_count += case_rewrites
        canonical_cases.append(canonical_case)
    if total_chars > selected_policy.max_total_query_chars:
        raise ArtifactMalformed("TOTAL_QUERY_BUDGET_EXCEEDED")

    all_findings.sort(
        key=lambda finding: (finding.case_sha256, finding.rewrite_sha256, finding.code)
    )
    truncated = len(all_findings) > selected_policy.max_findings
    bounded_findings = tuple(all_findings[: selected_policy.max_findings])
    canonical_artifact = {
        key: artifact[key]
        for key in (
            "schema_version",
            "benchmark_id",
            "rewriter_model_sha256",
            "prompt_sha256",
            "retrieval_policy_sha256",
            "created_at",
        )
    }
    canonical_artifact["cases"] = sorted(canonical_cases, key=lambda item: item["case_id"])
    accepted = not all_findings
    return RewriteAuditReport(
        schema_version=1,
        status="accepted" if accepted else "policy_rejected",
        accepted=accepted,
        artifact_sha256=_sha256(_canonical_json(canonical_artifact)),
        policy_sha256=_sha256(_canonical_json(asdict(selected_policy))),
        case_count=len(cases),
        rewrite_count=rewrite_count,
        rejected_rewrite_count=len(rejected_rewrites),
        finding_count=len(all_findings),
        findings_truncated=truncated,
        findings=bounded_findings,
    )


def _audit_case(
    case: Any,
    *,
    policy: RewritePolicy,
    seen_cases: set[str],
) -> tuple[list[Finding], dict[str, Any], int, int]:
    if not isinstance(case, dict):
        raise ArtifactMalformed("CASE_NOT_OBJECT")
    _expect_keys(
        case,
        {
            "case_id",
            "original_query",
            "language",
            "scope_sha256",
            "protected_literals",
            "allowed_structured_additions",
            "rewrites",
        },
        "CASE_FIELDS",
    )
    case_id = _identifier(case["case_id"], "CASE_ID")
    if case_id in seen_cases:
        raise ArtifactMalformed("DUPLICATE_CASE_ID")
    seen_cases.add(case_id)
    original = _query_text(case["original_query"], policy, "ORIGINAL_QUERY")
    language = _language(case["language"], "ORIGINAL_LANGUAGE")
    scope = _digest(case["scope_sha256"], "SCOPE_SHA256")
    protected = _protected_literals(case["protected_literals"], original, policy)
    allowed_additions = _allowed_additions(case["allowed_structured_additions"], policy)
    rewrites = case["rewrites"]
    if not isinstance(rewrites, list) or not rewrites:
        raise ArtifactMalformed("REWRITES_REQUIRED")
    if len(rewrites) > policy.max_rewrites_per_case:
        raise ArtifactMalformed("REWRITE_BUDGET_EXCEEDED")

    original_tokens = _tokens(original)
    original_structured = _structured_literals(original)
    original_negations = _negation_count(original_tokens, language)
    case_hash = _sha256(case_id.encode())
    seen_rewrite_ids: set[str] = set()
    seen_rewrite_texts: set[str] = set()
    findings: list[Finding] = []
    canonical_rewrites: list[dict[str, Any]] = []
    total_chars = len(original)

    for rewrite in rewrites:
        if not isinstance(rewrite, dict):
            raise ArtifactMalformed("REWRITE_NOT_OBJECT")
        _expect_keys(
            rewrite,
            {"rewrite_id", "text", "language", "scope_sha256"},
            "REWRITE_FIELDS",
        )
        rewrite_id = _identifier(rewrite["rewrite_id"], "REWRITE_ID")
        if rewrite_id in seen_rewrite_ids:
            raise ArtifactMalformed("DUPLICATE_REWRITE_ID")
        seen_rewrite_ids.add(rewrite_id)
        text = _query_text(rewrite["text"], policy, "REWRITE_TEXT")
        rewrite_language = _language(rewrite["language"], "REWRITE_LANGUAGE")
        rewrite_scope = _digest(rewrite["scope_sha256"], "REWRITE_SCOPE_SHA256")
        rewrite_hash = _sha256(f"{case_id}\0{rewrite_id}".encode())
        normalized_text = _normalize(text)
        codes: set[str] = set()
        if normalized_text in seen_rewrite_texts:
            codes.add("DUPLICATE_NORMALIZED_REWRITE")
        seen_rewrite_texts.add(normalized_text)
        if rewrite_language != language:
            codes.add("LANGUAGE_CHANGED")
        if rewrite_scope != scope:
            codes.add("SCOPE_CHANGED")

        rewrite_tokens = _tokens(text)
        if len(rewrite_tokens) > policy.max_rewrite_tokens:
            codes.add("TOKEN_BUDGET_EXCEEDED")
        if len(rewrite_tokens) > max(1, len(original_tokens)) * policy.max_growth_ratio:
            codes.add("QUERY_GROWTH_EXCEEDED")
        rewrite_negations = _negation_count(rewrite_tokens, language)
        if rewrite_negations != original_negations:
            codes.add("NEGATION_CHANGED")
        for literal in protected:
            if not _contains_token_sequence(rewrite_tokens, literal["tokens"]):
                codes.add("PROTECTED_LITERAL_DROPPED")
                break
        rewrite_structured = _structured_literals(text)
        if original_structured - rewrite_structured:
            codes.add("STRUCTURED_LITERAL_DROPPED")
        introduced = rewrite_structured - original_structured - allowed_additions
        if introduced:
            codes.add("STRUCTURED_LITERAL_INTRODUCED")

        findings.extend(Finding(case_hash, rewrite_hash, code) for code in sorted(codes))
        total_chars += len(text)
        canonical_rewrites.append(
            {
                "rewrite_id": rewrite_id,
                "text": text,
                "language": rewrite_language,
                "scope_sha256": rewrite_scope,
            }
        )

    canonical_case = {
        "case_id": case_id,
        "original_query": original,
        "language": language,
        "scope_sha256": scope,
        "protected_literals": sorted(
            ({"kind": item["kind"], "value": item["value"]} for item in protected),
            key=lambda item: (item["kind"], item["value"]),
        ),
        "allowed_structured_additions": sorted(allowed_additions),
        "rewrites": sorted(canonical_rewrites, key=lambda item: item["rewrite_id"]),
    }
    return findings, canonical_case, total_chars, len(rewrites)


def _protected_literals(value: Any, original: str, policy: RewritePolicy) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 64:
        raise ArtifactMalformed("PROTECTED_LITERALS")
    original_tokens = _tokens(original)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ArtifactMalformed("PROTECTED_LITERAL_NOT_OBJECT")
        _expect_keys(item, {"kind", "value"}, "PROTECTED_LITERAL_FIELDS")
        kind = item["kind"]
        if kind not in PROTECTED_KINDS:
            raise ArtifactMalformed("PROTECTED_LITERAL_KIND")
        literal = _query_text(item["value"], policy, "PROTECTED_LITERAL_VALUE")
        literal_tokens = _tokens(literal)
        if not literal_tokens or not _contains_token_sequence(original_tokens, literal_tokens):
            raise ArtifactMalformed("PROTECTED_LITERAL_NOT_IN_ORIGINAL")
        identity = (kind, _normalize(literal))
        if identity in seen:
            raise ArtifactMalformed("DUPLICATE_PROTECTED_LITERAL")
        seen.add(identity)
        result.append({"kind": kind, "value": literal, "tokens": tuple(literal_tokens)})
    return result


def _allowed_additions(value: Any, policy: RewritePolicy) -> set[str]:
    if not isinstance(value, list) or len(value) > 64:
        raise ArtifactMalformed("ALLOWED_STRUCTURED_ADDITIONS")
    result: set[str] = set()
    for item in value:
        text = _query_text(item, policy, "ALLOWED_STRUCTURED_ADDITION")
        extracted = _structured_literals(text)
        if not extracted:
            raise ArtifactMalformed("ALLOWED_ADDITION_NOT_STRUCTURED")
        if result & extracted:
            raise ArtifactMalformed("DUPLICATE_ALLOWED_ADDITION")
        result.update(extracted)
    return result


def _structured_literals(text: str) -> set[str]:
    result: set[str] = set()
    for expression in (URL_RE, EMAIL_RE, UUID_RE, MIXED_ID_RE, NUMBER_RE):
        for match in expression.finditer(text):
            candidate = match.group(0).rstrip(".,;:!?)]}")
            if candidate:
                result.add(_normalize_structured(candidate))
    return result


def _normalize_structured(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(_normalize(value)))


def _contains_token_sequence(tokens: tuple[str, ...], target: tuple[str, ...]) -> bool:
    if len(target) > len(tokens):
        return False
    return any(
        tokens[index : index + len(target)] == target
        for index in range(len(tokens) - len(target) + 1)
    )


def _negation_count(tokens: tuple[str, ...], language: str) -> int:
    vocabulary = NEGATIONS["en"] | NEGATIONS["tr"] if language == "und" else NEGATIONS[language]
    return sum(token in vocabulary for token in tokens)


def _query_text(value: Any, policy: RewritePolicy, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > policy.max_query_chars:
        raise ArtifactMalformed(code)
    for character in value:
        if unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co"}:
            raise ArtifactMalformed("UNSAFE_QUERY_CHARACTER")
    return unicodedata.normalize("NFKC", value)


def _expect_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ArtifactMalformed(code)


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _language(value: Any, code: str) -> str:
    if value not in SUPPORTED_LANGUAGES:
        raise ArtifactMalformed(code)
    return value


def _timestamp(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ArtifactMalformed(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactMalformed(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ArtifactMalformed(code)
    return parsed.astimezone(UTC)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RAG query-rewrite contracts")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", help="UTC RFC3339 timestamp; defaults to current UTC")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = RewritePolicy()
    try:
        artifact = load_artifact(args.artifact.read_bytes(), policy)
        as_of = _timestamp(args.as_of, "AS_OF") if args.as_of else datetime.now(UTC)
        report = audit_rewrites(artifact, as_of=as_of, policy=policy)
        payload = report.to_dict()
        exit_code = 0 if report.accepted else 2
    except (ArtifactMalformed, OSError) as exc:
        error = exc.code if isinstance(exc, ArtifactMalformed) else "ARTIFACT_IO_ERROR"
        payload = {"accepted": False, "error": error, "status": "malformed"}
        exit_code = 3
    if args.output:
        try:
            _write_json(args.output, payload)
        except OSError:
            return 3
    else:
        json.dump(payload, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
