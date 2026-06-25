"""Validate the local private assignment dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_reduction import (
    DataValidationError,
    LoadedQueries,
    ValidationIssue,
    ValidationReport,
    load_photos_csv,
    load_queries_csv,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", default="data/raw/photos.csv", type=Path)
    parser.add_argument("--queries", default="data/raw/queries.csv", type=Path)
    parser.add_argument(
        "--id-base",
        choices=["auto", "zero", "one"],
        default="one",
        help="Photo ID base used by queries.csv.",
    )
    args = parser.parse_args()

    try:
        photos = load_photos_csv(args.photos)
        queries = load_queries_csv(args.queries, len(photos), id_base=args.id_base)
    except DataValidationError as error:
        print("Dataset validation failed.")
        print(f"\n{error}")
        _print_issues("Errors", error.issues)
        return 1

    report = validate_dataset(photos, queries)
    _print_summary(photos, queries, report)

    if not report.is_valid:
        print("\nDataset validation failed.")
        _print_issues("Errors", report.errors)
        return 1

    print("\nDataset validation succeeded.")
    if queries.diagnostics:
        _print_issues("ID normalization diagnostics", queries.diagnostics)
    if report.warnings:
        _print_issues("Warnings", report.warnings)
    return 0


def _print_summary(
    photos,
    queries: LoadedQueries,
    report: ValidationReport,
) -> None:
    print("Dataset summary")
    print(f"- Photos: {photos.shape[0]}")
    print(f"- Embedding dimension: {photos.shape[1]}")
    print(f"- Queries: {queries.query_count}")
    print(
        "- ID base: "
        f"requested={queries.id_base_requested}, resolved={queries.id_base_resolved}"
    )
    print(f"- Warning count: {len(report.warnings) + len(queries.diagnostics)}")


def _print_issues(title: str, issues: tuple[ValidationIssue, ...]) -> None:
    print(f"\n{title}:")
    for issue in issues:
        row = f" row={issue.row}" if issue.row is not None else ""
        print(f"- [{issue.code}{row}] {issue.message}")


if __name__ == "__main__":
    sys.exit(main())
