import numpy as np
import pytest

from data_reduction import (
    best_candidate_with_id_tiebreak,
    cosine_proxy_utility,
    indepdf_scores,
    jaccard_precision_utility,
    rank_descending_with_id_tiebreak,
)


def test_jaccard_precision_utility_matches_hand_checked_value():
    utility = jaccard_precision_utility([0, 2], [(0, 1), (2,)])

    assert utility == pytest.approx(0.75)


def test_jaccard_precision_utility_empty_selection_is_zero():
    assert jaccard_precision_utility([], [(0, 1), (2,)]) == 0.0


def test_indepdf_scores_match_expected_membership_contributions():
    scores = indepdf_scores(num_photos=4, queries=[(0, 1), (0, 2, 3)])

    np.testing.assert_allclose(scores, [5 / 12, 1 / 4, 1 / 6, 1 / 6])


def test_cosine_proxy_utility_uses_best_retained_similarity_per_query_photo():
    photos = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    utility = cosine_proxy_utility(photos, queries=[(0, 1), (2,)], selected_ids=[0])

    assert utility == pytest.approx((0.5 + np.sqrt(0.5)) / 2)


def test_cosine_proxy_utility_treats_query_rows_as_sets():
    photos = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    utility = cosine_proxy_utility(photos, queries=[(0, 1, 1)], selected_ids=[0])

    assert utility == pytest.approx(0.5)


def test_cosine_proxy_utility_empty_selection_is_zero():
    photos = np.ones((3, 2))

    assert cosine_proxy_utility(photos, [(0, 1)], []) == 0.0


def test_utility_helpers_reject_empty_query_rows():
    with pytest.raises(ValueError, match="empty rows"):
        jaccard_precision_utility([0], [()])
    with pytest.raises(ValueError, match="empty rows"):
        indepdf_scores(3, [()])
    with pytest.raises(ValueError, match="empty rows"):
        cosine_proxy_utility(np.ones((3, 2)), [()], [0])


def test_rank_descending_with_id_tiebreak_prefers_lower_photo_id_on_ties():
    scores = np.array([0.5, 0.7, 0.7, 0.2])

    assert rank_descending_with_id_tiebreak(scores) == (1, 2, 0, 3)
    assert rank_descending_with_id_tiebreak(scores, limit=2) == (1, 2)
    assert best_candidate_with_id_tiebreak(scores) == 1
