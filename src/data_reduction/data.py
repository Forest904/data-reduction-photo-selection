"""CSV loading and validation for the assignment dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.errors import ParserError

IdBase = Literal["auto", "zero", "one"]
ResolvedIdBase = Literal["zero", "one"]


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable validation diagnostic with a human-readable message."""

    code: str
    message: str
    row: int | None = None


@dataclass(frozen=True)
class LoadedQueries:
    """Normalized query rows and ID-base diagnostics."""

    queries: tuple[tuple[int, ...], ...]
    id_base_requested: IdBase
    id_base_resolved: ResolvedIdBase
    diagnostics: tuple[ValidationIssue, ...] = ()

    @property
    def query_count(self) -> int:
        return len(self.queries)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for already-loaded dataset objects."""

    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DataValidationError(ValueError):
    """Raised when CSV loading or dataset validation finds blocking issues."""

    def __init__(self, message: str, issues: list[ValidationIssue]):
        super().__init__(message)
        self.issues = tuple(issues)


def load_photos_csv(path: str | Path) -> np.ndarray:
    """Load a headerless photo embedding CSV into a float32 matrix."""

    csv_path = Path(path)
    try:
        frame = pd.read_csv(
            csv_path,
            header=None,
            dtype=np.float32,
            na_values=[""],
            keep_default_na=True,
        )
    except FileNotFoundError as error:
        raise DataValidationError(
            f"Photo CSV not found: {csv_path}",
            [ValidationIssue("photos_file_not_found", str(csv_path))],
        ) from error
    except ParserError as error:
        raise DataValidationError(
            "Photo CSV has malformed rows or unequal embedding dimensions.",
            [ValidationIssue("photos_malformed_csv", str(error))],
        ) from error
    except ValueError as error:
        raise DataValidationError(
            "Photo CSV contains non-numeric values.",
            [ValidationIssue("photos_non_numeric", str(error))],
        ) from error

    if frame.empty:
        raise DataValidationError(
            "Photo CSV is empty.",
            [ValidationIssue("photos_empty", "Expected at least one photo row.")],
        )

    missing = np.argwhere(frame.isna().to_numpy())
    if missing.size:
        issues = [
            ValidationIssue(
                "photos_missing_value",
                f"Missing value at row {row + 1}, column {column + 1}.",
                row=int(row + 1),
            )
            for row, column in missing[:20]
        ]
        if len(missing) > 20:
            issues.append(
                ValidationIssue(
                    "photos_missing_value_truncated",
                    f"{len(missing) - 20} additional missing values omitted.",
                )
            )
        raise DataValidationError("Photo CSV contains missing values.", issues)

    return frame.to_numpy(dtype=np.float32, copy=True)


def load_queries_csv(
    path: str | Path,
    num_photos: int,
    id_base: IdBase = "auto",
) -> LoadedQueries:
    """Load a headerless query CSV and normalize photo IDs."""

    if id_base not in {"auto", "zero", "one"}:
        raise ValueError("id_base must be one of: auto, zero, one")
    if num_photos <= 0:
        raise ValueError("num_photos must be positive")

    raw_rows = _read_query_rows(Path(path))
    raw_ids = [photo_id for row in raw_rows for photo_id in row]
    resolved_base, diagnostics = _resolve_id_base(raw_ids, num_photos, id_base)

    errors: list[ValidationIssue] = []
    normalized_rows: list[tuple[int, ...]] = []
    offset = 1 if resolved_base == "one" else 0

    for row_number, raw_row in enumerate(raw_rows, start=1):
        normalized = [photo_id - offset for photo_id in raw_row]
        out_of_bounds = [
            photo_id
            for photo_id in normalized
            if photo_id < 0 or photo_id >= num_photos
        ]
        if out_of_bounds:
            errors.append(
                ValidationIssue(
                    "query_id_out_of_bounds",
                    (
                        f"Row {row_number} has normalized IDs outside "
                        f"[0, {num_photos - 1}]: {out_of_bounds[:10]}."
                    ),
                    row=row_number,
                )
            )

        if len(set(normalized)) != len(normalized):
            errors.append(
                ValidationIssue(
                    "duplicate_query_id",
                    f"Row {row_number} contains duplicate photo IDs.",
                    row=row_number,
                )
            )

        normalized_rows.append(tuple(sorted(normalized)))

    if errors:
        raise DataValidationError("Query CSV failed validation.", errors)

    return LoadedQueries(
        queries=tuple(normalized_rows),
        id_base_requested=id_base,
        id_base_resolved=resolved_base,
        diagnostics=tuple(diagnostics),
    )


def validate_dataset(
    photos: np.ndarray,
    queries: LoadedQueries | tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
) -> ValidationReport:
    """Validate already-loaded photos and normalized queries."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if photos.ndim != 2:
        errors.append(
            ValidationIssue(
                "photos_invalid_shape",
                f"Expected a 2D photo embedding matrix, got shape {photos.shape}.",
            )
        )
    elif photos.shape[0] == 0 or photos.shape[1] == 0:
        errors.append(
            ValidationIssue(
                "photos_empty_matrix",
                f"Expected non-empty photo matrix, got shape {photos.shape}.",
            )
        )

    if np.isnan(photos).any():
        errors.append(
            ValidationIssue(
                "photos_missing_value",
                "Photo matrix contains NaN values.",
            )
        )

    if photos.ndim == 2:
        zero_vector_ids = np.flatnonzero(np.linalg.norm(photos, axis=1) == 0)
        if zero_vector_ids.size:
            preview = zero_vector_ids[:20].astype(int).tolist()
            warnings.append(
                ValidationIssue(
                    "zero_photo_vectors",
                    (
                        f"{zero_vector_ids.size} zero-vector photo rows found; "
                        f"first IDs: {preview}."
                    ),
                )
            )

    query_rows = queries.queries if isinstance(queries, LoadedQueries) else queries
    num_photos = photos.shape[0] if photos.ndim == 2 else 0
    for row_number, query in enumerate(query_rows, start=1):
        if not query:
            errors.append(
                ValidationIssue(
                    "empty_query",
                    f"Query row {row_number} is empty.",
                    row=row_number,
                )
            )
        out_of_bounds = [
            photo_id
            for photo_id in query
            if photo_id < 0 or photo_id >= num_photos
        ]
        if out_of_bounds:
            errors.append(
                ValidationIssue(
                    "query_id_out_of_bounds",
                    (
                        f"Query row {row_number} has IDs outside "
                        f"[0, {num_photos - 1}]: {out_of_bounds[:10]}."
                    ),
                    row=row_number,
                )
            )

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def _read_query_rows(path: Path) -> list[list[int]]:
    errors: list[ValidationIssue] = []
    rows: list[list[int]] = []

    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    errors.append(
                        ValidationIssue(
                            "empty_query",
                            f"Query row {row_number} is empty.",
                            row=row_number,
                        )
                    )
                    continue

                parsed_row: list[int] = []
                for column_number, value in enumerate(row, start=1):
                    stripped = value.strip()
                    if not stripped:
                        errors.append(
                            ValidationIssue(
                                "query_missing_value",
                                (
                                    f"Missing query ID at row {row_number}, "
                                    f"column {column_number}."
                                ),
                                row=row_number,
                            )
                        )
                        continue
                    try:
                        parsed_row.append(int(stripped))
                    except ValueError:
                        errors.append(
                            ValidationIssue(
                                "query_non_integer",
                                (
                                    f"Non-integer query ID at row {row_number}, "
                                    f"column {column_number}: {stripped!r}."
                                ),
                                row=row_number,
                            )
                        )

                if parsed_row:
                    rows.append(parsed_row)
    except FileNotFoundError as error:
        raise DataValidationError(
            f"Query CSV not found: {path}",
            [ValidationIssue("queries_file_not_found", str(path))],
        ) from error

    if errors:
        raise DataValidationError("Query CSV failed parsing.", errors)
    if not rows:
        raise DataValidationError(
            "Query CSV is empty.",
            [ValidationIssue("queries_empty", "Expected at least one query row.")],
        )

    return rows


def _resolve_id_base(
    raw_ids: list[int],
    num_photos: int,
    requested: IdBase,
) -> tuple[ResolvedIdBase, list[ValidationIssue]]:
    diagnostics: list[ValidationIssue] = []
    if requested in {"zero", "one"}:
        return requested, diagnostics

    zero_valid = all(0 <= photo_id < num_photos for photo_id in raw_ids)
    one_valid = all(1 <= photo_id <= num_photos for photo_id in raw_ids)

    if zero_valid and not one_valid:
        diagnostics.append(
            ValidationIssue(
                "id_base_auto_zero",
                "Auto-detected zero-based query IDs.",
            )
        )
        return "zero", diagnostics
    if one_valid and not zero_valid:
        diagnostics.append(
            ValidationIssue(
                "id_base_auto_one",
                "Auto-detected one-based query IDs.",
            )
        )
        return "one", diagnostics

    if zero_valid and one_valid:
        diagnostics.append(
            ValidationIssue(
                "id_base_auto_ambiguous",
                "Query IDs fit zero-based and one-based ranges; defaulted to one.",
            )
        )
        return "one", diagnostics

    diagnostics.append(
        ValidationIssue(
            "id_base_auto_unresolved",
            "Query IDs fit neither zero-based nor one-based ranges; defaulted to zero.",
        )
    )
    return "zero", diagnostics
