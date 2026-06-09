"""Method D: query-aware greedy facility-location selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from data_reduction.evaluation import SelectionResult, measure_peak_memory_and_time
from data_reduction.methods.common import (
    MethodLimits,
    QueryRows,
    coerce_method_limits,
    empty_selection_result,
    query_mass_weights,
    score_summary,
    validate_budget,
)

METHOD = "D"
UTILITY_METRIC = "cosine_proxy_utility"


def select(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Greedily maximize query-mass-weighted clipped cosine coverage."""

    del seed
    method_limits = coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    if effective_budget == 0:
        return empty_selection_result(METHOD, UTILITY_METRIC, budget, effective_budget)

    measured = measure_peak_memory_and_time(
        _select_facility_location,
        np.asarray(photos),
        queries,
        effective_budget,
        method_limits.candidate_chunk_size,
        method_limits.max_facility_similarity_mb,
    )
    selected_ids, utility, diagnostics = measured.value
    return SelectionResult(
        selected_ids=selected_ids,
        utility=utility,
        runtime_seconds=measured.runtime_seconds,
        peak_memory_mb=measured.peak_memory_mb,
        diagnostics={
            "budget": int(budget),
            "effective_budget": effective_budget,
            **diagnostics,
        },
        method=METHOD,
        utility_metric=UTILITY_METRIC,
    )


def _select_facility_location(
    photos: np.ndarray,
    queries: QueryRows,
    effective_budget: int,
    candidate_chunk_size: int,
    max_facility_similarity_mb: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    weights = query_mass_weights(photos.shape[0], queries)
    target_ids = np.flatnonzero(weights > 0.0)
    if target_ids.size == 0:
        return (
            (),
            0.0,
            {
                "facility_objective": 0.0,
                "marginal_gains": [],
                "selected_query_mass_weights": [],
                "weighted_target_count": 0,
                "configured_candidate_chunk_size": candidate_chunk_size,
                "effective_candidate_chunk_size": 0,
                "max_facility_similarity_mb": max_facility_similarity_mb,
                "numeric_dtype": "float64",
                "weight_summary": score_summary(weights),
                "stopped_reason": "no_query_mass",
            },
        )

    target_photos = _normalized_rows_copy(photos[target_ids])
    target_weights = weights[target_ids]
    current_best = np.zeros(target_ids.size, dtype=np.float64)
    effective_chunk_size = _effective_candidate_chunk_size(
        configured_chunk_size=candidate_chunk_size,
        target_count=target_ids.size,
        item_size=np.dtype(np.float64).itemsize,
        max_similarity_mb=max_facility_similarity_mb,
    )
    selected_ids: list[int] = []
    marginal_gains: list[float] = []
    stopped_reason = "budget_reached"

    for _ in range(effective_budget):
        best_candidate, best_gain = _best_facility_candidate(
            photos,
            target_photos,
            target_weights,
            current_best,
            selected_ids,
            effective_chunk_size,
        )
        if best_candidate is None or best_gain <= 0.0:
            stopped_reason = "no_positive_marginal_gain"
            break

        selected_ids.append(best_candidate)
        marginal_gains.append(best_gain)
        selected_photo = _normalized_rows_copy(photos[[best_candidate]])
        candidate_similarity = target_photos @ selected_photo.T
        candidate_similarity = candidate_similarity[:, 0]
        np.maximum(candidate_similarity, 0.0, out=candidate_similarity)
        np.maximum(current_best, candidate_similarity, out=current_best)

    selected_tuple = tuple(selected_ids)
    utility = _cosine_proxy_utility_chunked(
        photos,
        queries,
        selected_tuple,
        max_similarity_mb=max_facility_similarity_mb,
    )
    facility_objective = float(target_weights @ current_best)
    return (
        selected_tuple,
        float(utility),
        {
            "facility_objective": facility_objective,
            "marginal_gains": marginal_gains,
            "selected_query_mass_weights": [
                float(weights[photo_id]) for photo_id in selected_tuple
            ],
            "weighted_target_count": int(target_ids.size),
            "configured_candidate_chunk_size": candidate_chunk_size,
            "effective_candidate_chunk_size": effective_chunk_size,
            "max_facility_similarity_mb": max_facility_similarity_mb,
            "numeric_dtype": "float64",
            "weight_summary": score_summary(weights),
            "coverage_cosine_transform": "max(0, cosine)",
            "candidate_scope": "all_photos",
            "stopped_reason": stopped_reason,
        },
    )


def _best_facility_candidate(
    photos: np.ndarray,
    target_photos: np.ndarray,
    target_weights: np.ndarray,
    current_best: np.ndarray,
    selected_ids: list[int],
    candidate_chunk_size: int,
) -> tuple[int | None, float]:
    best_candidate: int | None = None
    best_gain = -math.inf
    selected_set = set(selected_ids)

    for start in range(0, photos.shape[0], candidate_chunk_size):
        end = min(start + candidate_chunk_size, photos.shape[0])
        candidate_photos = _normalized_rows_copy(photos[start:end])
        similarities = target_photos @ candidate_photos.T
        np.maximum(similarities, 0.0, out=similarities)
        similarities -= current_best[:, np.newaxis]
        np.maximum(similarities, 0.0, out=similarities)
        gains = target_weights @ similarities

        for selected_id in selected_set:
            if start <= selected_id < end:
                gains[selected_id - start] = -math.inf

        local_index = int(np.argmax(gains))
        candidate_id = start + local_index
        candidate_gain = float(gains[local_index])
        if best_candidate is None or candidate_gain > best_gain or (
            candidate_gain == best_gain and candidate_id < best_candidate
        ):
            best_candidate = candidate_id
            best_gain = candidate_gain

    return best_candidate, best_gain


def _normalized_rows_copy(photos: np.ndarray) -> np.ndarray:
    matrix = np.asarray(photos, dtype=np.float64).copy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def _effective_candidate_chunk_size(
    configured_chunk_size: int,
    target_count: int,
    item_size: int,
    max_similarity_mb: int,
) -> int:
    max_similarity_bytes = max_similarity_mb * 1024 * 1024
    bytes_per_candidate = max(target_count * item_size, 1)
    memory_capped_chunk_size = max(max_similarity_bytes // bytes_per_candidate, 1)
    return int(min(configured_chunk_size, memory_capped_chunk_size))


def _cosine_proxy_utility_chunked(
    photos: np.ndarray,
    queries: QueryRows,
    selected_ids: tuple[int, ...],
    max_similarity_mb: int,
) -> float:
    if not selected_ids:
        return 0.0

    photo_matrix = np.asarray(photos)
    selected = np.asarray(selected_ids, dtype=np.int64)
    selected_photos = _normalized_rows_copy(photo_matrix[selected])
    max_similarity_bytes = max_similarity_mb * 1024 * 1024
    query_chunk_size = max(
        int(max_similarity_bytes // (selected.size * np.dtype(np.float64).itemsize)),
        1,
    )

    query_values: list[float] = []
    for query in queries:
        query_ids = np.asarray(sorted(set(query)), dtype=np.int64)
        if query_ids.size == 0:
            raise ValueError("queries must not contain empty rows")
        if np.any(query_ids < 0) or np.any(query_ids >= photo_matrix.shape[0]):
            raise IndexError("queries contain IDs outside the photo matrix bounds")

        best_similarity_sum = 0.0
        for start in range(0, query_ids.size, query_chunk_size):
            query_chunk = query_ids[start : start + query_chunk_size]
            query_photos = _normalized_rows_copy(photo_matrix[query_chunk])
            similarities = query_photos @ selected_photos.T
            best_similarity_sum += float(np.max(similarities, axis=1).sum())
        query_values.append(best_similarity_sum / query_ids.size)

    return float(np.mean(query_values)) if query_values else 0.0
