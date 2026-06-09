"""Shared utility metrics for query-aware photo selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data_reduction.similarity import cosine_similarity_to_selection

QueryRows = Sequence[Sequence[int]]


def jaccard_precision_utility(
    selected_ids: Sequence[int] | np.ndarray,
    queries: QueryRows,
) -> float:
    """Average retained-query overlap normalized by original query length."""

    selected = set(np.asarray(selected_ids, dtype=np.int64).tolist())
    if not selected:
        return 0.0

    values: list[float] = []
    for query in queries:
        query_set = set(query)
        if not query_set:
            raise ValueError("queries must not contain empty rows")
        values.append(len(selected.intersection(query_set)) / len(query_set))

    return float(np.mean(values)) if values else 0.0


def indepdf_scores(num_photos: int, queries: QueryRows) -> np.ndarray:
    """Return IndepDF scores E[I_q(d) / |q(D)|] for every photo ID."""

    if num_photos <= 0:
        raise ValueError("num_photos must be positive")

    scores = np.zeros(num_photos, dtype=np.float64)
    query_count = 0
    for query in queries:
        query_set = set(query)
        if not query_set:
            raise ValueError("queries must not contain empty rows")

        query_ids = np.asarray(sorted(query_set), dtype=np.int64)
        if np.any(query_ids < 0) or np.any(query_ids >= num_photos):
            raise IndexError("queries contain IDs outside the photo count")
        scores[query_ids] += 1.0 / len(query_set)
        query_count += 1

    if query_count == 0:
        return scores
    return scores / query_count


def cosine_proxy_utility(
    photos: np.ndarray,
    queries: QueryRows,
    selected_ids: Sequence[int] | np.ndarray,
) -> float:
    """Average best retained cosine similarity over all historical query rows."""

    selected = np.asarray(selected_ids, dtype=np.int64)
    if selected.size == 0:
        return 0.0

    similarities = cosine_similarity_to_selection(photos, selected)
    query_values: list[float] = []
    for query in queries:
        query_ids = np.asarray(sorted(set(query)), dtype=np.int64)
        if query_ids.size == 0:
            raise ValueError("queries must not contain empty rows")
        if np.any(query_ids < 0) or np.any(query_ids >= similarities.shape[0]):
            raise IndexError("queries contain IDs outside the photo matrix bounds")

        best_similarities = similarities[query_ids].max(axis=1)
        query_values.append(float(np.mean(best_similarities)))

    return float(np.mean(query_values)) if query_values else 0.0
