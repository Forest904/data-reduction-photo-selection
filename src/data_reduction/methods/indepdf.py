"""Method B: IndepDF scoring and top-budget selection."""

from __future__ import annotations

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
    validate_budget,
)
from data_reduction.utility import indepdf_scores, jaccard_precision_utility

METHOD = "B"
UTILITY_METRIC = "jaccard_precision_utility"


def select(
    photos: np.ndarray,
    queries: QueryRows,
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Select the top-budget photos by IndepDF membership contribution."""

    del seed
    coerce_method_limits(limits)
    effective_budget = validate_budget(photos, budget)
    if effective_budget == 0:
        return empty_selection_result(METHOD, UTILITY_METRIC, budget, effective_budget)

    measured = measure_peak_memory_and_time(
        _select_indepdf,
        int(np.asarray(photos).shape[0]),
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
            **diagnostics,
        },
        method=METHOD,
        utility_metric=UTILITY_METRIC,
    )


def _select_indepdf(
    num_photos: int,
    queries: QueryRows,
    effective_budget: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    scores = indepdf_scores(num_photos, queries)
    selected_ids = rank_descending_with_id_tiebreak(scores, limit=effective_budget)
    utility = jaccard_precision_utility(selected_ids, queries)
    return (
        selected_ids,
        float(utility),
        {
            "selected_scores": {
                str(photo_id): float(scores[photo_id]) for photo_id in selected_ids
            },
            "score_summary": score_summary(scores),
        },
    )
