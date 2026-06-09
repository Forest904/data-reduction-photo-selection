"""Run one photo-selection method on a local photos/query dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_reduction import (
    DataValidationError,
    MethodLimits,
    ValidationIssue,
    load_photos_csv,
    load_queries_csv,
    select_method,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--photos", default="data/raw/photos.csv", type=Path)
    parser.add_argument("--queries", default="data/raw/queries.csv", type=Path)
    parser.add_argument(
        "--id-base",
        choices=["auto", "zero", "one"],
        default="auto",
        help="Photo ID base used by queries.csv.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-exhaustive-photos", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--max-shapley-photos", type=int, default=None)
    parser.add_argument("--max-shapley-coalitions", type=int, default=None)
    parser.add_argument("--candidate-chunk-size", type=int, default=None)
    parser.add_argument("--max-facility-similarity-mb", type=int, default=None)
    args = parser.parse_args()

    try:
        photos = load_photos_csv(args.photos)
        loaded_queries = load_queries_csv(args.queries, len(photos), id_base=args.id_base)
    except DataValidationError as error:
        print("Dataset loading failed.", file=sys.stderr)
        print(f"\n{error}", file=sys.stderr)
        _print_issues("Errors", error.issues)
        return 1

    report = validate_dataset(photos, loaded_queries)
    if not report.is_valid:
        print("Dataset validation failed.", file=sys.stderr)
        _print_issues("Errors", report.errors)
        return 1

    _print_optional_issues("ID normalization diagnostics", loaded_queries.diagnostics)
    _print_optional_issues("Warnings", report.warnings)

    try:
        result = select_method(
            args.method,
            photos,
            loaded_queries.queries,
            budget=args.budget,
            seed=args.seed,
            limits=_limits_from_args(args),
        )
    except (TypeError, ValueError, IndexError) as error:
        print("Method execution failed.", file=sys.stderr)
        print(f"\n{error}", file=sys.stderr)
        return 1

    output = result.to_json_dict()
    output["dataset_diagnostics"] = _dataset_diagnostics(loaded_queries, report)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _limits_from_args(args: argparse.Namespace) -> MethodLimits:
    defaults = MethodLimits()
    return MethodLimits(
        max_exhaustive_photos=(
            args.max_exhaustive_photos
            if args.max_exhaustive_photos is not None
            else defaults.max_exhaustive_photos
        ),
        max_exhaustive_combinations=(
            args.max_combinations
            if args.max_combinations is not None
            else defaults.max_exhaustive_combinations
        ),
        max_shapley_photos=(
            args.max_shapley_photos
            if args.max_shapley_photos is not None
            else defaults.max_shapley_photos
        ),
        max_shapley_coalitions=(
            args.max_shapley_coalitions
            if args.max_shapley_coalitions is not None
            else defaults.max_shapley_coalitions
        ),
        candidate_chunk_size=(
            args.candidate_chunk_size
            if args.candidate_chunk_size is not None
            else defaults.candidate_chunk_size
        ),
        max_facility_similarity_mb=(
            args.max_facility_similarity_mb
            if args.max_facility_similarity_mb is not None
            else defaults.max_facility_similarity_mb
        ),
    )


def _print_issues(title: str, issues: tuple[ValidationIssue, ...]) -> None:
    print(f"\n{title}:", file=sys.stderr)
    for issue in issues:
        row = f" row={issue.row}" if issue.row is not None else ""
        print(f"- [{issue.code}{row}] {issue.message}", file=sys.stderr)


def _print_optional_issues(title: str, issues: tuple[ValidationIssue, ...]) -> None:
    if issues:
        _print_issues(title, issues)


def _dataset_diagnostics(loaded_queries, report) -> dict:
    return {
        "id_base_requested": loaded_queries.id_base_requested,
        "id_base_resolved": loaded_queries.id_base_resolved,
        "id_normalization_diagnostics": [
            _issue_to_dict(issue) for issue in loaded_queries.diagnostics
        ],
        "validation_warnings": [_issue_to_dict(issue) for issue in report.warnings],
    }


def _issue_to_dict(issue: ValidationIssue) -> dict:
    return {
        "code": issue.code,
        "message": issue.message,
        "row": issue.row,
    }


if __name__ == "__main__":
    sys.exit(main())
