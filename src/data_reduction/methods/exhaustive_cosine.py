"""Method A: exhaustive cosine-proxy subset search."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np

from data_reduction.evaluation import SelectionResult, measure_peak_memory_and_time
from data_reduction.methods.common import (
    MethodLimits,
    QueryRows,
    coerce_method_limits,
    empty_selection_result,
    skipped_selection_result,
    validate_budget,
)
from data_reduction.utility import cosine_proxy_utility

METHOD = "A"
UTILITY_METRIC = "cosine_proxy_utility"


def select(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Select photos by exhaustive search over all feasible budget-sized subsets."""

    del seed
    method_limits = coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    photo_count = int(np.asarray(photos).shape[0])
    if effective_budget == 0:
        return empty_selection_result(METHOD, UTILITY_METRIC, budget, effective_budget)

    candidate_count = math.comb(photo_count, effective_budget)
    guardrail_diagnostics = {
        "num_photos": photo_count,
        "budget": int(budget),
        "effective_budget": effective_budget,
        "candidate_subsets": candidate_count,
        "max_exhaustive_photos": method_limits.max_exhaustive_photos,
        "max_exhaustive_combinations": method_limits.max_exhaustive_combinations,
    }
    if photo_count > method_limits.max_exhaustive_photos:
        return skipped_selection_result(
            METHOD,
            UTILITY_METRIC,
            "exhaustive search skipped because photo count exceeds the configured limit",
            {**guardrail_diagnostics, "reason": "max_exhaustive_photos_exceeded"},
        )
    if candidate_count > method_limits.max_exhaustive_combinations:
        return skipped_selection_result(
            METHOD,
            UTILITY_METRIC,
            "exhaustive search skipped because combination count exceeds the configured limit",
            {**guardrail_diagnostics, "reason": "max_exhaustive_combinations_exceeded"},
        )

    measured = measure_peak_memory_and_time(
        _select_exhaustive,
        np.asarray(photos, dtype=np.float64),
        queries,
        effective_budget,
        candidate_count,
    )
    selected_ids, utility, diagnostics = measured.value
    return SelectionResult(
        selected_ids=selected_ids,
        utility=utility,
        runtime_seconds=measured.runtime_seconds,
        peak_memory_mb=measured.peak_memory_mb,
        diagnostics={
            **guardrail_diagnostics,
            **diagnostics,
        },
        method=METHOD,
        utility_metric=UTILITY_METRIC,
    )


def _select_exhaustive(
    photos: np.ndarray,
    queries: QueryRows,
    effective_budget: int,
    candidate_count: int,
) -> tuple[tuple[int, ...], float, dict[str, int | float]]:
    best_subset: tuple[int, ...] = ()
    best_utility = -math.inf
    evaluated_subsets = 0

    for subset in combinations(range(photos.shape[0]), effective_budget):
        utility = cosine_proxy_utility(photos, queries, subset)
        evaluated_subsets += 1
        if utility > best_utility:
            best_subset = tuple(int(photo_id) for photo_id in subset)
            best_utility = utility

    return (
        best_subset,
        float(best_utility),
        {
            "evaluated_subsets": evaluated_subsets,
            "candidate_subsets": candidate_count,
            "best_utility": float(best_utility),
        },
    )
