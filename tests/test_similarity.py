import numpy as np
import pytest

from data_reduction import cosine_similarity_matrix, cosine_similarity_to_selection


def test_cosine_similarity_matrix_matches_hand_checked_values():
    left = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    right = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ]
    )

    similarities = cosine_similarity_matrix(left, right)

    np.testing.assert_allclose(
        similarities,
        [
            [1.0, 0.0, np.sqrt(0.5)],
            [0.0, 1.0, np.sqrt(0.5)],
            [np.sqrt(0.5), np.sqrt(0.5), 1.0],
            [-1.0, 0.0, -np.sqrt(0.5)],
        ],
    )
    assert similarities.shape == (4, 3)


def test_cosine_similarity_matrix_handles_zero_vectors_without_nan():
    similarities = cosine_similarity_matrix(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    np.testing.assert_allclose(similarities, [[0.0, 0.0], [0.0, 1.0]])
    assert not np.isnan(similarities).any()


def test_cosine_similarity_matrix_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same embedding dimension"):
        cosine_similarity_matrix(np.ones((2, 3)), np.ones((4, 2)))


def test_cosine_similarity_to_selection_uses_selected_photo_columns():
    photos = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    similarities = cosine_similarity_to_selection(photos, [2, 0])

    np.testing.assert_allclose(
        similarities,
        [
            [np.sqrt(0.5), 1.0],
            [np.sqrt(0.5), 0.0],
            [1.0, np.sqrt(0.5)],
        ],
    )


def test_cosine_similarity_to_empty_selection_returns_empty_columns():
    similarities = cosine_similarity_to_selection(np.ones((3, 2)), [])

    assert similarities.shape == (3, 0)
