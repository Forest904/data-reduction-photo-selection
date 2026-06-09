"""Reusable helpers for reproducible experiment batches."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_reduction.evaluation import SelectionResult, measure_peak_memory_and_time
from data_reduction.methods import MethodLimits, select_method
from data_reduction.methods.common import (
    coerce_method_limits,
    query_mass_weights,
    score_summary,
    validate_budget,
)
from data_reduction.methods.facility_location import (
    _best_facility_candidate,
    _cosine_proxy_utility_chunked,
    _effective_candidate_chunk_size,
    _normalized_rows_copy,
)
from data_reduction.utility import cosine_proxy_utility, jaccard_precision_utility

QueryRows = Sequence[Sequence[int]]

_LIMIT_ALIASES = {
    "max_combinations": "max_exhaustive_combinations",
    "max_exhaustive_combinations": "max_exhaustive_combinations",
    "max_exhaustive_photos": "max_exhaustive_photos",
    "max_shapley_photos": "max_shapley_photos",
    "max_shapley_coalitions": "max_shapley_coalitions",
    "candidate_chunk_size": "candidate_chunk_size",
    "max_facility_similarity_mb": "max_facility_similarity_mb",
}


@dataclass(frozen=True)
class ExperimentTask:
    """One method/sample/budget/seed combination from a config grid."""

    method: str
    sample_size: int
    budget: int
    seed: int


@dataclass(frozen=True)
class SampledDataset:
    """A sampled and locally remapped experiment dataset."""

    photos: np.ndarray
    queries: tuple[tuple[int, ...], ...]
    original_photo_ids: tuple[int, ...]
    original_query_indexes: tuple[int, ...]
    diagnostics: dict[str, Any]


def expand_experiment_grid(config: Mapping[str, Any]) -> list[ExperimentTask]:
    """Return the Cartesian method/sample/budget/seed task grid."""

    methods = [str(method) for method in config.get("methods", ["D"])]
    sample_sizes = [_require_positive_int(value, "sample_size") for value in config["sample_sizes"]]
    budgets = [_require_non_negative_int(value, "budget") for value in config.get("budgets", [3])]
    seeds = [_require_non_negative_int(value, "seed") for value in config.get("seeds", [0])]

    return [
        ExperimentTask(method=method, sample_size=sample_size, budget=budget, seed=seed)
        for method in methods
        for sample_size in sample_sizes
        for budget in budgets
        for seed in seeds
    ]


def coerce_experiment_limits(limits: Mapping[str, Any] | None) -> MethodLimits:
    """Coerce experiment config limit aliases into ``MethodLimits``."""

    if limits is None:
        return MethodLimits()
    normalized: dict[str, Any] = {}
    for name, value in limits.items():
        try:
            normalized[_LIMIT_ALIASES[str(name)]] = value
        except KeyError as error:
            raise ValueError(f"unknown method limit field: {name}") from error
    return coerce_method_limits(normalized)


def sample_dataset(
    photos: np.ndarray,
    queries: QueryRows,
    sample_size: int,
    seed: int,
    query_sample_size: int | None = None,
    sample_strategy: str = "query_active",
) -> SampledDataset:
    """Sample photos, project queries, and remap IDs to compact local indexes."""

    photo_matrix = np.asarray(photos)
    if photo_matrix.ndim != 2:
        raise ValueError(f"photos must be 2D, got shape {photo_matrix.shape}")
    original_photo_count = int(photo_matrix.shape[0])
    sample_size = min(_require_positive_int(sample_size, "sample_size"), original_photo_count)
    rng = np.random.default_rng(_require_non_negative_int(seed, "seed"))

    query_rows = tuple(tuple(sorted(set(query))) for query in queries)
    query_indexes = _sample_query_indexes(len(query_rows), query_sample_size, rng)
    sampled_query_rows = tuple(query_rows[index] for index in query_indexes)

    if sample_size == original_photo_count:
        sampled_ids = np.arange(original_photo_count, dtype=np.int64)
    elif sample_strategy == "query_active":
        sampled_ids = _sample_query_active_ids(
            original_photo_count,
            sampled_query_rows,
            sample_size,
            rng,
        )
    else:
        raise ValueError("sample_strategy must be 'query_active'")

    sampled_ids = np.sort(sampled_ids.astype(np.int64))
    id_to_local = {int(original_id): local_id for local_id, original_id in enumerate(sampled_ids)}

    projected_queries: list[tuple[int, ...]] = []
    kept_query_indexes: list[int] = []
    dropped_query_indexes: list[int] = []
    for original_index, query in zip(query_indexes, sampled_query_rows, strict=True):
        projected = tuple(
            id_to_local[photo_id] for photo_id in query if photo_id in id_to_local
        )
        if projected:
            projected_queries.append(tuple(sorted(set(projected))))
            kept_query_indexes.append(int(original_index))
        else:
            dropped_query_indexes.append(int(original_index))

    if not projected_queries:
        raise ValueError("sampling dropped every query; increase sample_size")

    diagnostics = {
        "sample_strategy": sample_strategy,
        "sample_size_requested": int(sample_size),
        "sampled_photo_count": int(sampled_ids.size),
        "original_photo_count": original_photo_count,
        "original_query_count": len(query_rows),
        "query_sample_size_requested": query_sample_size,
        "sampled_query_count": len(projected_queries),
        "dropped_query_count": len(dropped_query_indexes),
        "dropped_query_indexes": dropped_query_indexes,
        "original_query_indexes": kept_query_indexes,
        "sampled_original_photo_ids": [int(photo_id) for photo_id in sampled_ids],
    }
    return SampledDataset(
        photos=photo_matrix[sampled_ids].copy(),
        queries=tuple(projected_queries),
        original_photo_ids=tuple(int(photo_id) for photo_id in sampled_ids),
        original_query_indexes=tuple(kept_query_indexes),
        diagnostics=diagnostics,
    )


def select_experiment_method(
    method: str,
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Run required methods or experiment-only Method D ablations."""

    normalized = method.strip().upper().replace("-", "_")
    if normalized == "D_FREQUENCY_ONLY":
        return _select_frequency_only(photos, queries, budget, limits)
    if normalized == "D_COVERAGE_ONLY":
        return _select_coverage_only(photos, queries, budget, limits)
    return select_method(method, photos, queries, budget, seed=seed, limits=limits)


def evaluate_selection_metrics(
    photos: np.ndarray,
    queries: QueryRows,
    selected_ids: Sequence[int],
) -> dict[str, float | None]:
    """Return cross-method evaluation metrics for a successful selection."""

    return {
        "cosine_proxy_utility_eval": float(
            cosine_proxy_utility(photos, queries, selected_ids)
        ),
        "jaccard_precision_utility_eval": float(
            jaccard_precision_utility(selected_ids, queries)
        ),
    }


def create_synthetic_dataset(config: Mapping[str, Any]) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Create a deterministic clustered embedding/query fixture from config."""

    synthetic = dict(config.get("synthetic", {}))
    photo_count = _require_positive_int(synthetic.get("photo_count", 24), "photo_count")
    dimensions = _require_positive_int(synthetic.get("dimensions", 6), "dimensions")
    clusters = _require_positive_int(synthetic.get("clusters", 3), "clusters")
    query_count = _require_positive_int(synthetic.get("query_count", 18), "query_count")
    query_length = _require_positive_int(synthetic.get("query_length", 4), "query_length")
    seed = _require_non_negative_int(synthetic.get("seed", 0), "seed")

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dimensions))
    assignments = np.arange(photo_count) % clusters
    photos = centers[assignments] + rng.normal(scale=0.05, size=(photo_count, dimensions))

    queries: list[tuple[int, ...]] = []
    for query_index in range(query_count):
        cluster = query_index % clusters
        candidates = np.flatnonzero(assignments == cluster)
        size = min(query_length, int(candidates.size))
        query = np.sort(rng.choice(candidates, size=size, replace=False))
        queries.append(tuple(int(photo_id) for photo_id in query))

    return photos.astype(np.float32), tuple(queries)


def _sample_query_indexes(
    query_count: int,
    query_sample_size: int | None,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if query_count <= 0:
        raise ValueError("queries must contain at least one row")
    if query_sample_size is None:
        return tuple(range(query_count))
    requested = min(_require_positive_int(query_sample_size, "query_sample_size"), query_count)
    return tuple(int(index) for index in np.sort(rng.choice(query_count, requested, replace=False)))


def _sample_query_active_ids(
    photo_count: int,
    queries: QueryRows,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    active_ids = sorted({int(photo_id) for query in queries for photo_id in query})
    if not active_ids:
        raise ValueError("queries do not reference any photos")

    active = np.asarray(active_ids, dtype=np.int64)
    if sample_size <= active.size:
        return rng.choice(active, size=sample_size, replace=False)

    inactive = np.setdiff1d(np.arange(photo_count, dtype=np.int64), active, assume_unique=True)
    extra_size = min(sample_size - active.size, inactive.size)
    if extra_size == 0:
        return active
    extra = rng.choice(inactive, size=extra_size, replace=False)
    return np.concatenate([active, extra])


def _select_frequency_only(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    limits: MethodLimits | Mapping[str, Any] | None,
) -> SelectionResult:
    coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    if effective_budget == 0:
        return SelectionResult(
            selected_ids=(),
            utility=0.0,
            runtime_seconds=0.0,
            peak_memory_mb=0.0,
            diagnostics={"budget": int(budget), "effective_budget": effective_budget},
            method="D_frequency_only",
            utility_metric="cosine_proxy_utility",
        )

    measured = measure_peak_memory_and_time(
        _select_frequency_only_inner,
        np.asarray(photos),
        queries,
        effective_budget,
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
            "ablation": "frequency_only",
            **diagnostics,
        },
        method="D_frequency_only",
        utility_metric="cosine_proxy_utility",
    )


def _select_frequency_only_inner(
    photos: np.ndarray,
    queries: QueryRows,
    effective_budget: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    weights = query_mass_weights(photos.shape[0], queries)
    ids = np.arange(weights.size)
    ranked = np.lexsort((ids, -weights))[:effective_budget]
    selected_ids = tuple(int(photo_id) for photo_id in ranked if weights[photo_id] > 0.0)
    utility = cosine_proxy_utility(photos, queries, selected_ids)
    return (
        selected_ids,
        float(utility),
        {
            "weight_summary": score_summary(weights),
            "selected_query_mass_weights": [
                float(weights[photo_id]) for photo_id in selected_ids
            ],
            "stopped_reason": (
                "budget_reached"
                if len(selected_ids) == effective_budget
                else "no_positive_query_mass"
            ),
        },
    )


def _select_coverage_only(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    limits: MethodLimits | Mapping[str, Any] | None,
) -> SelectionResult:
    method_limits = coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    if effective_budget == 0:
        return SelectionResult(
            selected_ids=(),
            utility=0.0,
            runtime_seconds=0.0,
            peak_memory_mb=0.0,
            diagnostics={"budget": int(budget), "effective_budget": effective_budget},
            method="D_coverage_only",
            utility_metric="cosine_proxy_utility",
        )

    measured = measure_peak_memory_and_time(
        _select_coverage_only_inner,
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
            "ablation": "coverage_only",
            **diagnostics,
        },
        method="D_coverage_only",
        utility_metric="cosine_proxy_utility",
    )


def _select_coverage_only_inner(
    photos: np.ndarray,
    queries: QueryRows,
    effective_budget: int,
    candidate_chunk_size: int,
    max_facility_similarity_mb: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    target_ids = np.asarray(sorted({int(photo_id) for query in queries for photo_id in query}))
    if target_ids.size == 0:
        return (), 0.0, {"stopped_reason": "no_query_targets"}

    target_weights = np.full(target_ids.size, 1.0 / target_ids.size, dtype=np.float64)
    target_photos = _normalized_rows_copy(photos[target_ids])
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
    return (
        selected_tuple,
        float(utility),
        {
            "facility_objective": float(target_weights @ current_best),
            "marginal_gains": marginal_gains,
            "weighted_target_count": int(target_ids.size),
            "configured_candidate_chunk_size": candidate_chunk_size,
            "effective_candidate_chunk_size": effective_chunk_size,
            "max_facility_similarity_mb": max_facility_similarity_mb,
            "numeric_dtype": "float64",
            "coverage_cosine_transform": "max(0, cosine)",
            "candidate_scope": "all_photos",
            "target_weighting": "uniform_over_query_active_photos",
            "stopped_reason": stopped_reason,
        },
    )


def _require_positive_int(value: Any, name: str) -> int:
    value = _require_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    value = _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)
