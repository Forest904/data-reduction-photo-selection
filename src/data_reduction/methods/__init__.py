"""Shared interface for required photo-selection methods A-D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from data_reduction.evaluation import SelectionResult
from data_reduction.methods.common import MethodLimits
from data_reduction.methods import exhaustive_cosine, facility_location, indepdf, shapley

_METHOD_MODULES = {
    "A": exhaustive_cosine,
    "B": indepdf,
    "C": shapley,
    "D": facility_location,
}

_METHOD_ALIASES = {
    "A": "A",
    "EXHAUSTIVE": "A",
    "EXHAUSTIVE_COSINE": "A",
    "B": "B",
    "INDEPDF": "B",
    "C": "C",
    "SHAPLEY": "C",
    "D": "D",
    "FACILITY_LOCATION": "D",
    "GREEDY_FACILITY_LOCATION": "D",
}


def select_method(
    method: str,
    photos: np.ndarray,
    queries: list[np.ndarray] | list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    budget: int,
    seed: int | None = None,
    limits: MethodLimits | Mapping[str, Any] | None = None,
) -> SelectionResult:
    """Run one required selection method behind the shared interface."""

    method_key = _normalize_method_name(method)
    return _METHOD_MODULES[method_key].select(
        photos=photos,
        queries=queries,
        budget=budget,
        seed=seed,
        limits=limits,
    )


def _normalize_method_name(method: str) -> str:
    normalized = method.strip().upper().replace("-", "_")
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            f"unknown method {method!r}; expected one of A, B, C, or D"
        ) from error


__all__ = ["MethodLimits", "select_method"]
