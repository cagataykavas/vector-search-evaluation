from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from search.evidence_diversity import (
    DiversityPolicy,
    EvidenceDiversityError,
    audit_evidence_diversity,
    parse_artifact,
)

SNAPSHOT = "a" * 64


def hit(rank: int, doc_id: str, source_id: str, text: str) -> dict:
    return {"rank": rank, "doc_id": doc_id, "source_id": source_id, "text": text}


def artifact(*hits: dict, query_id: str = "q-1") -> dict:
    return {
        "index_snapshot_digest": SNAPSHOT,
        "queries": [{"query_id": query_id, "hits": list(hits)}],
    }


def diverse_artifact() -> dict:
    return artifact(
        hit(1, "d-1", "manual", "Rotate signing keys every ninety days and record the key ID."),
        hit(2, "d-2", "runbook", "Database restores are rehearsed monthly in an isolated account."),
        hit(3, "d-3", "standard", "Outbound requests require an explicit destination allowlist."),
    )


def test_accepts_diverse_independently_sourced_evidence() -> None:
    report = audit_evidence_diversity(diverse_artifact())

    assert report.accepted
    assert report.reasons == ()
    assert report.query_count == 1
    assert report.total_hit_count == 3
    assert report.pair_comparisons == 3
    query = report.queries[0]
    assert query.effective_evidence_count == 3
    assert query.independent_source_count == 3
    assert query.near_duplicate_pairs == 0
    assert len(query.query_ref) == 16


def test_rejects_exact_duplicate_even_with_different_source_labels() -> None:
    report = audit_evidence_diversity(
        artifact(
            hit(1, "d-1", "mirror-a", "Backups are encrypted before upload."),
            hit(2, "d-2", "mirror-b", "BACKUPS are encrypted before upload!"),
            hit(3, "d-3", "manual", "Restore drills run once each month."),
        )
    )

    query = report.queries[0]
    assert not report.accepted
    assert "exact_duplicate_budget_exceeded" in query.reasons
    assert query.exact_duplicate_hits == 1
    assert query.effective_evidence_count == 2
    assert query.independent_source_count == 2


def test_rejects_near_duplicate_chunk_overlap() -> None:
    common = "the incident commander records the timeline and assigns every recovery action"
    report = audit_evidence_diversity(
        artifact(
            hit(1, "d-1", "runbook", f"{common} during the outage"),
            hit(2, "d-2", "runbook", f"{common} during each outage"),
            hit(3, "d-3", "policy", "Security reviews happen before every production release."),
        ),
        DiversityPolicy(near_duplicate_threshold=0.7, max_redundant_hit_fraction=0.2),
    )

    assert not report.accepted
    assert report.queries[0].near_duplicate_pairs == 1
    assert "redundant_hit_budget_exceeded" in report.reasons


def test_rejects_one_cluster_labeled_as_many_sources() -> None:
    policy = DiversityPolicy(max_exact_duplicate_fraction=1.0, max_redundant_hit_fraction=1.0)
    report = audit_evidence_diversity(
        artifact(
            hit(1, "d-1", "mirror-a", "Use workload identity instead of static cloud keys."),
            hit(2, "d-2", "mirror-b", "Use workload identity instead of static cloud keys."),
        ),
        policy,
    )

    query = report.queries[0]
    assert not report.accepted
    assert query.distinct_source_count == 2
    assert query.effective_evidence_count == 1
    assert query.independent_source_count == 1
    assert "insufficient_independent_sources" in query.reasons
    assert "insufficient_effective_evidence" in query.reasons


def test_maximum_matching_counts_distinct_cluster_source_assignments() -> None:
    # First duplicate cluster can use A or B; the second uses only A. A greedy first choice of A
    # would undercount, while maximum matching correctly assigns B then A.
    policy = DiversityPolicy(
        max_exact_duplicate_fraction=1.0,
        max_redundant_hit_fraction=1.0,
        max_largest_cluster_fraction=1.0,
    )
    report = audit_evidence_diversity(
        artifact(
            hit(1, "d-1", "A", "Rotate all signing keys once per quarter."),
            hit(2, "d-2", "B", "Rotate all signing keys once per quarter."),
            hit(3, "d-3", "A", "Restore drills execute in a clean account every month."),
        ),
        policy,
    )

    assert report.queries[0].independent_source_count == 2
    assert report.accepted


def test_rejects_dominant_duplicate_cluster() -> None:
    policy = DiversityPolicy(
        max_exact_duplicate_fraction=1.0,
        max_redundant_hit_fraction=1.0,
        max_largest_cluster_fraction=0.5,
        min_independent_sources=1,
    )
    report = audit_evidence_diversity(
        artifact(
            hit(1, "d-1", "A", "Tokens expire after fifteen minutes."),
            hit(2, "d-2", "A", "Tokens expire after fifteen minutes."),
            hit(3, "d-3", "A", "Tokens expire after fifteen minutes."),
            hit(4, "d-4", "B", "Audit logs remain immutable for one year."),
        ),
        policy,
    )

    assert "duplicate_cluster_dominates" in report.reasons
    assert report.queries[0].largest_cluster_fraction == 0.75


def test_report_is_deterministic_across_query_order() -> None:
    first = diverse_artifact()["queries"][0]
    second = {**first, "query_id": "q-2"}
    forward = {"index_snapshot_digest": SNAPSHOT, "queries": [first, second]}
    reverse = {"index_snapshot_digest": SNAPSHOT, "queries": [second, first]}

    left = audit_evidence_diversity(forward)
    right = audit_evidence_diversity(reverse)

    assert left.to_dict() == right.to_dict()
    assert left.artifact_digest == right.artifact_digest


def test_digest_changes_when_content_changes_without_exposing_content() -> None:
    original = audit_evidence_diversity(diverse_artifact())
    changed = diverse_artifact()
    changed["queries"][0]["hits"][0]["text"] += " Updated."
    updated = audit_evidence_diversity(changed)
    rendered = json.dumps(original.to_dict())

    assert original.artifact_digest != updated.artifact_digest
    assert "Rotate signing keys" not in rendered
    assert "q-1" not in rendered
    assert "d-1" not in rendered


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(index_snapshot_digest="A" * 64), "lowercase SHA-256"),
        (lambda value: value["queries"].append(value["queries"][0]), "query_id"),
        (
            lambda value: value["queries"][0]["hits"].append(value["queries"][0]["hits"][0]),
            "doc_id",
        ),
        (lambda value: value["queries"][0]["hits"][0].update(rank=2), "contiguous"),
        (lambda value: value["queries"][0].update(extra=True), "missing or unexpected"),
        (lambda value: value["queries"][0]["hits"][0].update(text="---"), "searchable token"),
    ],
)
def test_malformed_artifacts_fail_closed(mutate, message: str) -> None:
    payload = diverse_artifact()
    mutate(payload)

    with pytest.raises(EvidenceDiversityError, match=message):
        audit_evidence_diversity(payload)


def test_pair_comparison_budget_fails_closed() -> None:
    with pytest.raises(EvidenceDiversityError, match="pair comparison budget"):
        audit_evidence_diversity(
            diverse_artifact(),
            replace(DiversityPolicy(), max_pair_comparisons=2),
        )


@pytest.mark.parametrize(
    "policy",
    [
        DiversityPolicy(near_duplicate_threshold=0.5),
        DiversityPolicy(max_exact_duplicate_fraction=1.0),
        DiversityPolicy(max_redundant_hit_fraction=1.0),
        DiversityPolicy(max_largest_cluster_fraction=1.0),
    ],
)
def test_policy_variants_are_serializable(policy: DiversityPolicy) -> None:
    report = audit_evidence_diversity(diverse_artifact(), policy)
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"near_duplicate_threshold": float("nan")},
        {"max_redundant_hit_fraction": 1.1},
        {"shingle_size": 0},
        {"min_effective_evidence": 0},
        {"max_pair_comparisons": True},
    ],
)
def test_invalid_policies_fail_closed(kwargs: dict) -> None:
    with pytest.raises(EvidenceDiversityError):
        DiversityPolicy(**kwargs)


def run_cli(tmp_path: Path, raw: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(raw, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "search.evidence_diversity", str(artifact_path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_accepts_and_atomically_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "audit.json"
    completed = run_cli(tmp_path, json.dumps(diverse_artifact()), "--output", str(output))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert json.loads(output.read_text())["accepted"] is True


def test_cli_returns_three_for_policy_rejection(tmp_path: Path) -> None:
    payload = artifact(
        hit(1, "d-1", "a", "Backups are encrypted before upload."),
        hit(2, "d-2", "b", "Backups are encrypted before upload."),
    )
    completed = run_cli(tmp_path, json.dumps(payload))

    assert completed.returncode == 3
    assert json.loads(completed.stdout)["accepted"] is False


@pytest.mark.parametrize(
    "raw",
    [
        '{"index_snapshot_digest":"' + SNAPSHOT + '","queries":[],"queries":[]}',
        '{"index_snapshot_digest":"' + SNAPSHOT + '","queries":[],"x":NaN}',
        "not-json",
    ],
)
def test_cli_returns_two_for_malformed_json(tmp_path: Path, raw: str) -> None:
    completed = run_cli(tmp_path, raw)

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"] == "malformed_artifact"


def test_parse_artifact_returns_immutable_model() -> None:
    parsed = parse_artifact(diverse_artifact())

    assert parsed.index_snapshot_digest == SNAPSHOT
    assert parsed.queries[0].hits[0].rank == 1
