"""Evaluation schemas and measurement helpers."""

from __future__ import annotations

import json
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

SelectionStatus = Literal["success", "skipped", "timeout", "error"]

T = TypeVar("T")


@dataclass(frozen=True)
class MeasuredResult[T]:
    """A callable result bundled with elapsed time and peak traced memory."""

    value: T
    runtime_seconds: float
    peak_memory_mb: float


@dataclass(frozen=True)
class SelectionResult:
    """Common result object returned by all selection methods."""

    selected_ids: tuple[int, ...]
    utility: float
    runtime_seconds: float
    peak_memory_mb: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    status: SelectionStatus = "success"
    message: str = ""
    method: str | None = None
    utility_metric: str | None = None
    diagnostics_path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly dictionary representation."""

        data = self.to_json_dict()
        data["selected_ids"] = ",".join(str(photo_id) for photo_id in self.selected_ids)
        data["diagnostics"] = json.dumps(data["diagnostics"], sort_keys=True)
        return data

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""

        return {
            "method": self.method,
            "selected_ids": list(self.selected_ids),
            "utility": float(self.utility),
            "utility_metric": self.utility_metric,
            "runtime_seconds": float(self.runtime_seconds),
            "peak_memory_mb": float(self.peak_memory_mb),
            "diagnostics": _to_jsonable(self.diagnostics),
            "status": self.status,
            "message": self.message,
            "diagnostics_path": (
                str(self.diagnostics_path) if self.diagnostics_path is not None else None
            ),
        }


def measure_peak_memory_and_time(
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> MeasuredResult[T]:
    """Run a callable and return its result, elapsed time, and peak memory."""

    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = function(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        runtime_seconds = time.perf_counter() - start
        tracemalloc.stop()

    del current
    return MeasuredResult(
        value=value,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak / (1024 * 1024),
    )


def rank_descending_with_id_tiebreak(
    scores: Sequence[float] | np.ndarray,
    limit: int | None = None,
) -> tuple[int, ...]:
    """Rank score indexes by descending score, breaking ties by lower index."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {score_array.shape}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    ids = np.arange(score_array.size)
    ranked = np.lexsort((ids, -score_array))
    if limit is not None:
        ranked = ranked[:limit]
    return tuple(int(index) for index in ranked)


def best_candidate_with_id_tiebreak(scores: Sequence[float] | np.ndarray) -> int:
    """Return the highest-scoring index, breaking ties by lower index."""

    ranked = rank_descending_with_id_tiebreak(scores, limit=1)
    if not ranked:
        raise ValueError("scores must contain at least one candidate")
    return ranked[0]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
