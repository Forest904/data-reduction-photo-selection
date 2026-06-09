"""Shared helpers for photo-selection methods."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from data_reduction.evaluation import SelectionResult

QueryRows = Sequence[Sequence[int]]


@dataclass(frozen=True)
class MethodLimits:
    """Configurable guardrails and implementation limits for selection methods."""

    max_exhaustive_photos: int = 25
    max_exhaustive_combinations: int = 100_000
    max_shapley_photos: int = 12
    max_shapley_coalitions: int = 4096
    candidate_chunk_size: int = 2048
    max_facility_similarity_mb: int = 64

    def __post_init__(self) -> None:
        _require_int(self.max_exhaustive_photos, "max_exhaustive_photos")
        _require_int(
            self.max_exhaustive_combinations,
            "max_exhaustive_combinations",
        )
        _require_int(self.max_shapley_photos, "max_shapley_photos")
        _require_int(self.max_shapley_coalitions, "max_shapley_coalitions")
        _require_int(self.candidate_chunk_size, "candidate_chunk_size")
        _require_int(
            self.max_facility_similarity_mb,
            "max_facility_similarity_mb",
        )
        if self.max_exhaustive_photos < 0:
            raise ValueError("max_exhaustive_photos must be non-negative")
        if self.max_exhaustive_combinations < 0:
            raise ValueError("max_exhaustive_combinations must be non-negative")
        if self.max_shapley_photos < 0:
            raise ValueError("max_shapley_photos must be non-negative")
        if self.max_shapley_coalitions < 0:
            raise ValueError("max_shapley_coalitions must be non-negative")
        if self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive")
        if self.max_facility_similarity_mb <= 0:
            raise ValueError("max_facility_similarity_mb must be positive")


def coerce_method_limits(
    limits: MethodLimits | Mapping[str, Any] | None,
) -> MethodLimits:
    """Return a ``MethodLimits`` instance from defaults, a dataclass, or mapping."""

    if limits is None:
        return MethodLimits()
    if isinstance(limits, MethodLimits):
        return limits
    if not isinstance(limits, Mapping):
        raise TypeError("limits must be MethodLimits, a mapping, or None")

    valid_names = {field.name for field in fields(MethodLimits)}
    unknown = set(limits) - valid_names
    if unknown:
        raise ValueError(f"unknown method limit fields: {sorted(unknown)}")

    defaults = MethodLimits()
    values = {}
    for field in fields(MethodLimits):
        raw_value = limits.get(field.name, getattr(defaults, field.name))
        values[field.name] = _require_int(raw_value, field.name)
    return MethodLimits(**values)


def validate_budget(photos: np.ndarray, budget: int) -> int:
    """Validate a selection budget and return the capped effective budget."""

    photo_matrix = np.asarray(photos)
    if photo_matrix.ndim != 2:
        raise ValueError(f"photos must be a 2D array, got shape {photo_matrix.shape}")
    if photo_matrix.shape[0] <= 0:
        raise ValueError("photos must contain at least one row")
    budget = _require_int(budget, "budget")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return min(int(budget), int(photo_matrix.shape[0]))


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def empty_selection_result(
    method: str,
    utility_metric: str,
    budget: int,
    effective_budget: int,
    message: str = "",
) -> SelectionResult:
    """Return a successful empty result for zero-budget or no-gain selections."""

    return SelectionResult(
        selected_ids=(),
        utility=0.0,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        diagnostics={
            "budget": int(budget),
            "effective_budget": int(effective_budget),
        },
        status="success",
        message=message,
        method=method,
        utility_metric=utility_metric,
    )


def skipped_selection_result(
    method: str,
    utility_metric: str,
    message: str,
    diagnostics: Mapping[str, Any],
) -> SelectionResult:
    """Return a valid skipped result for infeasible exact runs."""

    return SelectionResult(
        selected_ids=(),
        utility=0.0,
        runtime_seconds=0.0,
        peak_memory_mb=0.0,
        diagnostics=dict(diagnostics),
        status="skipped",
        message=message,
        method=method,
        utility_metric=utility_metric,
    )


def query_mass_weights(num_photos: int, queries: QueryRows) -> np.ndarray:
    """Return normalized query-mass weights over photo IDs.

    Each query contributes total mass 1, split uniformly over its unique returned
    photo IDs. The resulting weights sum to 1 when at least one query is present.
    """

    if num_photos <= 0:
        raise ValueError("num_photos must be positive")

    weights = np.zeros(num_photos, dtype=np.float64)
    query_count = 0
    for query in queries:
        query_set = set(query)
        if not query_set:
            raise ValueError("queries must not contain empty rows")

        query_ids = np.asarray(sorted(query_set), dtype=np.int64)
        if np.any(query_ids < 0) or np.any(query_ids >= num_photos):
            raise IndexError("queries contain IDs outside the photo count")

        weights[query_ids] += 1.0 / len(query_set)
        query_count += 1

    if query_count == 0:
        return weights

    weights /= query_count
    total_weight = float(weights.sum())
    if total_weight > 0.0:
        weights /= total_weight
    return weights


def score_summary(scores: np.ndarray) -> dict[str, float | int]:
    """Return compact JSON-friendly summary statistics for a 1D score vector."""

    if scores.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "nonzero_count": 0,
        }
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "nonzero_count": int(np.count_nonzero(scores)),
    }
