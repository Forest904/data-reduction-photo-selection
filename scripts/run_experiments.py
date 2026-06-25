"""Run a reproducible batch of photo-selection experiments from YAML config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_reduction import (
    DataValidationError,
    coerce_experiment_limits,
    create_synthetic_dataset,
    evaluate_selection_metrics,
    expand_experiment_grid,
    load_experiment_config,
    load_photos_csv,
    load_queries_csv,
    sample_dataset,
    select_experiment_method,
    validate_dataset,
)

_FIELDNAMES = [
    "experiment_id",
    "batch_id",
    "experiment_name",
    "method",
    "dataset_size",
    "num_queries",
    "train_query_count",
    "eval_query_count",
    "evaluation_scope",
    "sample_size_requested",
    "budget",
    "seed",
    "utility_metric",
    "utility",
    "cosine_proxy_utility_eval",
    "jaccard_precision_utility_eval",
    "runtime_seconds",
    "peak_memory_mb",
    "selected_ids",
    "selected_original_ids",
    "status",
    "message",
    "diagnostics_path",
    "config_path",
    "git_commit",
    "git_dirty",
    "python_version",
    "hardware_notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default=Path("experiments/results"), type=Path)
    parser.add_argument("--hardware-notes", default=None)
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
        photos, queries, dataset_diagnostics = _load_dataset(config)
    except (DataValidationError, FileNotFoundError, ValueError, TypeError) as error:
        print(f"Experiment setup failed: {error}", file=sys.stderr)
        return 1

    experiment_name = str(config.get("experiment_name", args.config.stem))
    batch_id = args.batch_id or _default_batch_id(experiment_name)
    batch_dir = args.output / batch_id
    diagnostics_dir = batch_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    metadata = _batch_metadata(
        config_path=args.config,
        batch_id=batch_id,
        experiment_name=experiment_name,
        hardware_notes=(
            args.hardware_notes
            or config.get("hardware_notes")
            or _default_hardware_notes()
        ),
    )
    metadata["dataset_diagnostics"] = dataset_diagnostics
    _write_json(batch_dir / "batch_metadata.json", metadata)

    limits = coerce_experiment_limits(config.get("limits"))
    rows: list[dict[str, Any]] = []
    tasks = expand_experiment_grid(config)
    for index, task in enumerate(tasks, start=1):
        experiment_id = f"{batch_id}_{index:04d}"
        row, diagnostics = _run_task(
            experiment_id=experiment_id,
            task=task,
            photos=photos,
            queries=queries,
            config=config,
            limits=limits,
            metadata=metadata,
        )
        diagnostics_path = diagnostics_dir / f"{experiment_id}_{_safe_name(task.method)}.json"
        _write_json(diagnostics_path, diagnostics)
        row["diagnostics_path"] = str(diagnostics_path)
        rows.append(row)

    csv_path = batch_dir / "results.csv"
    _write_csv(csv_path, rows)
    print(f"Wrote {len(rows)} result rows to {csv_path}")
    print(f"Wrote batch metadata to {batch_dir / 'batch_metadata.json'}")
    return 0


def _load_dataset(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    if config.get("dataset_kind", "raw") == "synthetic":
        photos, queries = create_synthetic_dataset(config)
        return (
            photos,
            queries,
            {
                "dataset_kind": "synthetic",
                "photo_count": int(photos.shape[0]),
                "embedding_dimension": int(photos.shape[1]),
                "query_count": len(queries),
            },
        )

    photos = load_photos_csv(config.get("photos_path", "data/raw/photos.csv"))
    loaded_queries = load_queries_csv(
        config.get("queries_path", "data/raw/queries.csv"),
        len(photos),
        id_base=config.get("id_base", "auto"),
    )
    report = validate_dataset(photos, loaded_queries)
    if not report.is_valid:
        raise DataValidationError("Dataset validation failed.", list(report.errors))
    return (
        photos,
        loaded_queries.queries,
        {
            "dataset_kind": "raw",
            "photo_count": int(photos.shape[0]),
            "embedding_dimension": int(photos.shape[1]),
            "query_count": loaded_queries.query_count,
            "id_base_requested": loaded_queries.id_base_requested,
            "id_base_resolved": loaded_queries.id_base_resolved,
            "id_normalization_diagnostics": [
                issue.__dict__ for issue in loaded_queries.diagnostics
            ],
            "validation_warnings": [issue.__dict__ for issue in report.warnings],
        },
    )


def _run_task(
    experiment_id: str,
    task,
    photos,
    queries,
    config: dict[str, Any],
    limits,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        sampled = sample_dataset(
            photos,
            queries,
            sample_size=task.sample_size,
            seed=task.seed,
            query_sample_size=config.get("query_sample_size"),
            sample_strategy=config.get("sample_strategy", "query_active"),
        )
        train_queries, eval_queries, split_diagnostics = _split_train_eval_queries(
            sampled.queries,
            holdout_fraction=config.get("query_holdout_fraction"),
            seed=task.seed,
        )
        result = select_experiment_method(
            task.method,
            sampled.photos,
            train_queries,
            budget=task.budget,
            seed=task.seed,
            limits=limits,
        )
        eval_metrics = (
            evaluate_selection_metrics(sampled.photos, eval_queries, result.selected_ids)
            if result.status == "success"
            else {
                "cosine_proxy_utility_eval": None,
                "jaccard_precision_utility_eval": None,
            }
        )
        selected_original_ids = [
            sampled.original_photo_ids[local_id] for local_id in result.selected_ids
        ]
        diagnostics = {
            "experiment_id": experiment_id,
            "task": task.__dict__,
            "sample": sampled.diagnostics,
            "query_split": split_diagnostics,
            "result": result.to_json_dict(),
            "selected_original_ids": selected_original_ids,
            "cross_method_metrics": eval_metrics,
        }
        row = _base_row(experiment_id, task, sampled, metadata)
        row.update(
            {
                "method": result.method or task.method,
                "train_query_count": len(train_queries),
                "eval_query_count": len(eval_queries),
                "evaluation_scope": split_diagnostics["evaluation_scope"],
                "utility_metric": result.utility_metric,
                "utility": result.utility,
                "cosine_proxy_utility_eval": eval_metrics["cosine_proxy_utility_eval"],
                "jaccard_precision_utility_eval": eval_metrics[
                    "jaccard_precision_utility_eval"
                ],
                "runtime_seconds": result.runtime_seconds,
                "peak_memory_mb": result.peak_memory_mb,
                "selected_ids": ",".join(str(photo_id) for photo_id in result.selected_ids),
                "selected_original_ids": ",".join(
                    str(photo_id) for photo_id in selected_original_ids
                ),
                "status": result.status,
                "message": result.message,
            }
        )
        return row, diagnostics
    except Exception as error:  # noqa: BLE001 - each run should become a row.
        sampled = sample_dataset(
            photos,
            queries,
            sample_size=task.sample_size,
            seed=task.seed,
            query_sample_size=config.get("query_sample_size"),
            sample_strategy=config.get("sample_strategy", "query_active"),
        )
        row = _base_row(experiment_id, task, sampled, metadata)
        row.update(
            {
                "method": task.method,
                "train_query_count": None,
                "eval_query_count": None,
                "evaluation_scope": "error",
                "utility_metric": None,
                "utility": None,
                "cosine_proxy_utility_eval": None,
                "jaccard_precision_utility_eval": None,
                "runtime_seconds": 0.0,
                "peak_memory_mb": 0.0,
                "selected_ids": "",
                "selected_original_ids": "",
                "status": "error",
                "message": str(error),
            }
        )
        diagnostics = {
            "experiment_id": experiment_id,
            "task": task.__dict__,
            "sample": sampled.diagnostics,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        return row, diagnostics


def _base_row(experiment_id: str, task, sampled, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "batch_id": metadata["batch_id"],
        "experiment_name": metadata["experiment_name"],
        "method": task.method,
        "dataset_size": int(sampled.photos.shape[0]),
        "num_queries": len(sampled.queries),
        "train_query_count": len(sampled.queries),
        "eval_query_count": len(sampled.queries),
        "evaluation_scope": "train",
        "sample_size_requested": task.sample_size,
        "budget": task.budget,
        "seed": task.seed,
        "utility_metric": None,
        "utility": None,
        "cosine_proxy_utility_eval": None,
        "jaccard_precision_utility_eval": None,
        "runtime_seconds": None,
        "peak_memory_mb": None,
        "selected_ids": "",
        "selected_original_ids": "",
        "status": None,
        "message": "",
        "diagnostics_path": "",
        "config_path": metadata["config_path"],
        "git_commit": metadata["git_commit"],
        "git_dirty": metadata["git_dirty"],
        "python_version": metadata["python_version"],
        "hardware_notes": metadata["hardware_notes"],
    }


def _split_train_eval_queries(
    queries,
    holdout_fraction: float | None,
    seed: int,
):
    query_rows = tuple(queries)
    if holdout_fraction is None:
        return (
            query_rows,
            query_rows,
            {
                "evaluation_scope": "train",
                "query_holdout_fraction": None,
                "train_query_count": len(query_rows),
                "eval_query_count": len(query_rows),
                "eval_query_indexes": [],
            },
        )

    if len(query_rows) < 2:
        raise ValueError("query holdout requires at least two projected query rows")

    fraction = float(holdout_fraction)
    eval_count = max(1, int(round(len(query_rows) * fraction)))
    eval_count = min(eval_count, len(query_rows) - 1)
    rng = np.random.default_rng(seed)
    eval_indexes = set(rng.choice(len(query_rows), size=eval_count, replace=False))
    train_queries = tuple(
        query for index, query in enumerate(query_rows) if index not in eval_indexes
    )
    eval_queries = tuple(
        query for index, query in enumerate(query_rows) if index in eval_indexes
    )
    return (
        train_queries,
        eval_queries,
        {
            "evaluation_scope": "holdout",
            "query_holdout_fraction": fraction,
            "train_query_count": len(train_queries),
            "eval_query_count": len(eval_queries),
            "eval_query_indexes": sorted(int(index) for index in eval_indexes),
        },
    )


def _batch_metadata(
    config_path: Path,
    batch_id: str,
    experiment_name: str,
    hardware_notes: str,
) -> dict[str, Any]:
    resolved_config = config_path.resolve()
    return {
        "batch_id": batch_id,
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(resolved_config),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hardware_notes": hardware_notes,
        "git_commit": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_git_output(["status", "--short"])),
    }


def _default_batch_id(experiment_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_name(experiment_name)}_{timestamp}"


def _default_hardware_notes() -> str:
    return f"{platform.platform()}; cpu_count={os.cpu_count()}"


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in safe.split("_") if part)


def _git_output(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(data), file, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > 4096:
            if value > 0 and value & (value - 1) == 0:
                return f"2**{value.bit_length() - 1}"
            decimal_digits = int((value.bit_length() - 1) * 0.30103) + 1
            return (
                f"<large integer: bit_length={value.bit_length()}, "
                f"decimal_digits~{decimal_digits}>"
            )
        return value
    return value


if __name__ == "__main__":
    sys.exit(main())
