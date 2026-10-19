"""Shared float64 direction normalization for serving and offline evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def normalize_rows(values: ArrayLike) -> np.ndarray:
    """Normalize finite, nonzero rows without squaring their original magnitudes."""
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embeddings must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("embeddings must have a positive dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    scales = np.max(np.abs(matrix), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("embeddings must have a nonzero direction")
    # At least one scaled component is +/-1, so its norm cannot underflow.
    # Do not multiply the norm back by the original scale: it can overflow.
    scaled = matrix / scales
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
