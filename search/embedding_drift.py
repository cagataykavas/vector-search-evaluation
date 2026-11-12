from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PASS_EXIT_CODE = 0
INVALID_INPUT_EXIT_CODE = 2
DRIFT_EXIT_CODE = 3
MAX_PERMUTATION_DISTANCE_CELLS = 60_000_000
MAX_PAIRWISE_MULTIPLIES = 100_000_000


@dataclass(frozen=True)
class DriftPolicy:
    alpha: float = 0.05
    min_effect_size: float = 0.05
    permutations: int = 199
    seed: int = 1729
    min_samples: int = 20
    max_samples: int = 256
    max_dimensions: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= self.min_effect_size <= 4.0:
            raise ValueError("min_effect_size must be between 0 and 4")
        if not 99 <= self.permutations <= 9999:
            raise ValueError("permutations must be between 99 and 9999")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must be at least min_samples")
        if self.max_dimensions < 1:
            raise ValueError("max_dimensions must be positive")


@dataclass(frozen=True)
class EmbeddingDriftReport:
    baseline_samples: int
    candidate_samples: int
    dimensions: int
    energy_distance: float
    permutation_p_value: float
    drift_detected: bool
    baseline_digest: str
    candidate_digest: str
    policy_digest: str
    policy: DriftPolicy

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["schema_version"] = "embedding-drift-report-v1"
        return result


def _validate_matrix(
    rows: Iterable[Iterable[float]], *, name: str, policy: DriftPolicy
) -> np.ndarray:
    try:
        materialized = [list(row) for row in rows]
        matrix = np.asarray(materialized, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} embeddings must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} embeddings must be a two-dimensional matrix")
    samples, dimensions = matrix.shape
    if not policy.min_samples <= samples <= policy.max_samples:
        raise ValueError(
            f"{name} sample count must be between {policy.min_samples} and {policy.max_samples}"
        )
    if not 1 <= dimensions <= policy.max_dimensions:
        raise ValueError(f"{name} dimensions must be between 1 and {policy.max_dimensions}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} embeddings must not contain zero-norm rows")
    matrix[matrix == 0.0] = 0.0
    return matrix


def _matrix_digest(matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"embedding-drift-matrix-v1\0")
    digest.update(f"{matrix.shape[0]}:{matrix.shape[1]}\0".encode())
    digest.update(matrix.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _policy_digest(policy: DriftPolicy) -> str:
    payload = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(b"embedding-drift-policy-v1\0" + payload.encode()).hexdigest()


def _energy_statistic(
    distances: np.ndarray, baseline_indices: np.ndarray, candidate_indices: np.ndarray
) -> float:
    cross = distances[np.ix_(baseline_indices, candidate_indices)].mean()
    within_baseline = distances[np.ix_(baseline_indices, baseline_indices)].mean()
    within_candidate = distances[np.ix_(candidate_indices, candidate_indices)].mean()
    return max(0.0, float(2.0 * cross - within_baseline - within_candidate))


def audit_embedding_drift(
    baseline: Iterable[Iterable[float]],
    candidate: Iterable[Iterable[float]],
    *,
    policy: DriftPolicy | None = None,
) -> EmbeddingDriftReport:
    active_policy = policy or DriftPolicy()
    baseline_matrix = _validate_matrix(baseline, name="baseline", policy=active_policy)
    candidate_matrix = _validate_matrix(candidate, name="candidate", policy=active_policy)
    if baseline_matrix.shape[1] != candidate_matrix.shape[1]:
        raise ValueError("baseline and candidate embeddings must have the same dimension")

    baseline_digest = _matrix_digest(baseline_matrix)
    candidate_digest = _matrix_digest(candidate_matrix)
    combined = np.vstack((baseline_matrix, candidate_matrix))
    estimated_work = active_policy.permutations * len(combined) ** 2
    if estimated_work > MAX_PERMUTATION_DISTANCE_CELLS:
        raise ValueError("permutation work exceeds the audit limit; reduce samples or permutations")
    pairwise_work = len(combined) ** 2 * baseline_matrix.shape[1]
    if pairwise_work > MAX_PAIRWISE_MULTIPLIES:
        raise ValueError("pairwise work exceeds the audit limit; reduce samples or dimensions")
    combined /= np.linalg.norm(combined, axis=1, keepdims=True)
    cosine = np.clip(combined @ combined.T, -1.0, 1.0)
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))

    baseline_count = len(baseline_matrix)
    total_count = len(combined)
    baseline_indices = np.arange(baseline_count)
    candidate_indices = np.arange(baseline_count, total_count)
    observed = _energy_statistic(distances, baseline_indices, candidate_indices)

    random = np.random.default_rng(active_policy.seed)
    at_least_observed = 0
    for _ in range(active_policy.permutations):
        shuffled = random.permutation(total_count)
        permuted = _energy_statistic(
            distances,
            shuffled[:baseline_count],
            shuffled[baseline_count:],
        )
        if permuted >= observed - 1e-12:
            at_least_observed += 1
    p_value = (at_least_observed + 1) / (active_policy.permutations + 1)
    drift_detected = p_value <= active_policy.alpha and observed >= active_policy.min_effect_size
    return EmbeddingDriftReport(
        baseline_samples=baseline_count,
        candidate_samples=len(candidate_matrix),
        dimensions=baseline_matrix.shape[1],
        energy_distance=observed,
        permutation_p_value=p_value,
        drift_detected=drift_detected,
        baseline_digest=baseline_digest,
        candidate_digest=candidate_digest,
        policy_digest=_policy_digest(active_policy),
        policy=active_policy,
    )


def _load_matrix(path: Path) -> Sequence[Sequence[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON array of embedding arrays")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit cosine-space embedding population drift.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-effect-size", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-dimensions", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        policy = DriftPolicy(
            alpha=arguments.alpha,
            min_effect_size=arguments.min_effect_size,
            permutations=arguments.permutations,
            seed=arguments.seed,
            min_samples=arguments.min_samples,
            max_samples=arguments.max_samples,
            max_dimensions=arguments.max_dimensions,
        )
        report = audit_embedding_drift(
            _load_matrix(arguments.baseline),
            _load_matrix(arguments.candidate),
            policy=policy,
        )
    except (OSError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"embedding drift audit failed: {exc}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE

    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
    try:
        if arguments.output is None:
            print(rendered, end="")
        else:
            arguments.output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        print(f"embedding drift audit failed: {exc}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE
    return DRIFT_EXIT_CODE if report.drift_detected else PASS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
