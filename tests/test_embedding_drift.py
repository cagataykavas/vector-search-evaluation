from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from search.embedding_drift import (
    DRIFT_EXIT_CODE,
    INVALID_INPUT_EXIT_CODE,
    PASS_EXIT_CODE,
    DriftPolicy,
    audit_embedding_drift,
    main,
)


def population(center: tuple[float, ...], *, seed: int, samples: int = 32) -> list[list[float]]:
    random = np.random.default_rng(seed)
    return (np.asarray(center) + random.normal(scale=0.02, size=(samples, len(center)))).tolist()


def test_identical_population_passes_with_reproducible_evidence() -> None:
    baseline = population((1.0, 0.0, 0.0), seed=1)
    candidate = list(reversed(baseline))
    policy = DriftPolicy(permutations=199, min_effect_size=0.05)

    first = audit_embedding_drift(baseline, candidate, policy=policy)
    second = audit_embedding_drift(baseline, candidate, policy=policy)

    assert not first.drift_detected
    assert first.energy_distance == pytest.approx(0.0, abs=1e-12)
    assert first.permutation_p_value == 1.0
    assert first == second
    assert first.baseline_digest != first.candidate_digest


def test_direction_shift_fails_statistical_and_practical_gates() -> None:
    baseline = population((1.0, 0.0, 0.0), seed=2)
    candidate = population((0.0, 1.0, 0.0), seed=3)

    report = audit_embedding_drift(
        baseline,
        candidate,
        policy=DriftPolicy(permutations=199, min_effect_size=0.5),
    )

    assert report.drift_detected
    assert report.energy_distance > 2.0
    assert report.permutation_p_value <= 0.05


def test_practical_threshold_prevents_tiny_effect_alarm() -> None:
    baseline = population((1.0, 0.0, 0.0), seed=4, samples=64)
    candidate = population((0.99, 0.12, 0.0), seed=5, samples=64)

    report = audit_embedding_drift(
        baseline,
        candidate,
        policy=DriftPolicy(permutations=199, min_effect_size=0.5),
    )

    assert report.permutation_p_value <= 0.05
    assert report.energy_distance < 0.5
    assert not report.drift_detected


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ([[1.0, float("nan")]] * 20, "finite"),
        ([[0.0, 0.0]] * 20, "zero-norm"),
        ([[1.0, 0.0, 0.0]] * 20, "same dimension"),
    ],
)
def test_invalid_embedding_evidence_fails_closed(
    candidate: list[list[float]], message: str
) -> None:
    baseline = [[1.0, 0.0]] * 20

    with pytest.raises(ValueError, match=message):
        audit_embedding_drift(baseline, candidate)


def test_permutation_work_is_bounded() -> None:
    baseline = [[1.0, 0.01]] * 256
    candidate = [[0.01, 1.0]] * 256

    with pytest.raises(ValueError, match="work exceeds"):
        audit_embedding_drift(
            baseline,
            candidate,
            policy=DriftPolicy(permutations=999),
        )


def test_pairwise_work_is_bounded() -> None:
    baseline = np.ones((256, 384)).tolist()
    candidate = np.full((256, 384), 2.0).tolist()

    with pytest.raises(ValueError, match="pairwise work exceeds"):
        audit_embedding_drift(baseline, candidate)


def test_cli_writes_report_and_uses_distinct_exit_codes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    stable = tmp_path / "stable.json"
    shifted = tmp_path / "shifted.json"
    invalid = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    baseline.write_text(json.dumps(population((1.0, 0.0), seed=8)), encoding="utf-8")
    stable.write_text(baseline.read_text(encoding="utf-8"), encoding="utf-8")
    shifted.write_text(json.dumps(population((0.0, 1.0), seed=9)), encoding="utf-8")
    invalid.write_text("{}", encoding="utf-8")

    assert main([str(baseline), str(stable), "--output", str(output)]) == PASS_EXIT_CODE
    assert json.loads(output.read_text(encoding="utf-8"))["drift_detected"] is False
    assert main([str(baseline), str(shifted)]) == DRIFT_EXIT_CODE
    assert main([str(baseline), str(invalid)]) == INVALID_INPUT_EXIT_CODE
