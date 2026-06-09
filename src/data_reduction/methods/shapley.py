"""Method C: exact Shapley-value photo ranking for tiny datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from data_reduction.evaluation import (
    SelectionResult,
    measure_peak_memory_and_time,
    rank_descending_with_id_tiebreak,
)
from data_reduction.methods.common import (
    MethodLimits,
    QueryRows,
    coerce_method_limits,
    empty_selection_result,
    score_summary,
    skipped_selection_result,
    validate_budget,
)
from data_reduction.utility import cosine_proxy_utility

METHOD = "C"
UTILITY_METRIC = "cosine_proxy_utility"


def select(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Select top-budget photos by exact Shapley values over cosine utility."""

    del seed
    method_limits = coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    photo_count = int(np.asarray(photos).shape[0])
    if effective_budget == 0:
        return empty_selection_result(METHOD, UTILITY_METRIC, budget, effective_budget)

    coalition_count = 1 << photo_count
    guardrail_diagnostics = {
        "num_photos": photo_count,
        "budget": int(budget),
        "effective_budget": effective_budget,
        "coalition_count": coalition_count,
        "max_shapley_photos": method_limits.max_shapley_photos,
        "max_shapley_coalitions": method_limits.max_shapley_coalitions,
    }
    if photo_count > method_limits.max_shapley_photos:
        return skipped_selection_result(
            METHOD,
            UTILITY_METRIC,
            "exact Shapley skipped because photo count exceeds the configured limit",
            {**guardrail_diagnostics, "reason": "max_shapley_photos_exceeded"},
        )
    if coalition_count > method_limits.max_shapley_coalitions:
        return skipped_selection_result(
            METHOD,
            UTILITY_METRIC,
            "exact Shapley skipped because coalition count exceeds the configured limit",
            {**guardrail_diagnostics, "reason": "max_shapley_coalitions_exceeded"},
        )

    measured = measure_peak_memory_and_time(
        _select_shapley,
        np.asarray(photos, dtype=np.float64),
        queries,
        effective_budget,
        coalition_count,
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


def _select_shapley(
    photos: np.ndarray,
    queries: QueryRows,
    effective_budget: int,
    coalition_count: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    photo_count = photos.shape[0]
    values = np.zeros(coalition_count, dtype=np.float64)
    for mask in range(1, coalition_count):
        values[mask] = cosine_proxy_utility(photos, queries, _mask_to_ids(mask))

    shapley_values = np.zeros(photo_count, dtype=np.float64)
    factorials = [math.factorial(size) for size in range(photo_count + 1)]
    denominator = math.factorial(photo_count)

    for photo_id in range(photo_count):
        photo_bit = 1 << photo_id
        for mask in range(coalition_count):
            if mask & photo_bit:
                continue
            coalition_size = mask.bit_count()
            weight = (
                factorials[coalition_size]
                * factorials[photo_count - coalition_size - 1]
                / denominator
            )
            shapley_values[photo_id] += weight * (
                values[mask | photo_bit] - values[mask]
            )

    selected_ids = rank_descending_with_id_tiebreak(
        shapley_values,
        limit=effective_budget,
    )
    utility = cosine_proxy_utility(photos, queries, selected_ids)
    return (
        selected_ids,
        float(utility),
        {
            "shapley_values": shapley_values,
            "selected_shapley_values": {
                str(photo_id): float(shapley_values[photo_id])
                for photo_id in selected_ids
            },
            "shapley_summary": score_summary(shapley_values),
        },
    )


def _mask_to_ids(mask: int) -> tuple[int, ...]:
    return tuple(
        photo_id for photo_id in range(mask.bit_length()) if mask & (1 << photo_id)
    )
