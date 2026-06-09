"""Vector similarity helpers shared by selection and evaluation code."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities with zero-vector rows mapped to 0."""

    left_matrix = _as_2d_float_matrix(left, "left")
    right_matrix = _as_2d_float_matrix(right, "right")
    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ValueError(
            "left and right must have the same embedding dimension: "
            f"{left_matrix.shape[1]} != {right_matrix.shape[1]}"
        )

    similarities = left_matrix @ right_matrix.T
    denominator = np.linalg.norm(left_matrix, axis=1)[:, np.newaxis] * np.linalg.norm(
        right_matrix,
        axis=1,
    )[np.newaxis, :]

    return np.divide(
        similarities,
        denominator,
        out=np.zeros_like(similarities, dtype=np.float64),
        where=denominator > 0,
    )


def cosine_similarity_to_selection(
    photos: np.ndarray,
    selected_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return similarity from every photo to each selected photo."""

    photo_matrix = _as_2d_float_matrix(photos, "photos")
    selected = np.asarray(selected_ids, dtype=np.int64)
    if selected.size == 0:
        return np.empty((photo_matrix.shape[0], 0), dtype=np.float64)
    if np.any(selected < 0) or np.any(selected >= photo_matrix.shape[0]):
        raise IndexError("selected_ids contains IDs outside the photo matrix bounds")

    return cosine_similarity_matrix(photo_matrix, photo_matrix[selected])


def _as_2d_float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {matrix.shape}")
    return matrix
