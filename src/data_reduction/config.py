"""Small YAML config loader for experiment files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SEQUENCE_FIELDS = {
    "methods",
    "sample_sizes",
    "budgets",
    "seeds",
    "metrics",
}
_MAPPING_FIELDS = {"limits", "synthetic"}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and perform lightweight shape checks."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError("Experiment config root must be a mapping")

    _validate_basic_shape(loaded)
    return loaded


def _validate_basic_shape(config: dict[str, Any]) -> None:
    if "experiment_name" in config and not isinstance(config["experiment_name"], str):
        raise ValueError("experiment_name must be a string when provided")
    if "photos_path" in config and not isinstance(config["photos_path"], str):
        raise ValueError("photos_path must be a string when provided")
    if "queries_path" in config and not isinstance(config["queries_path"], str):
        raise ValueError("queries_path must be a string when provided")
    if "id_base" in config and config["id_base"] not in {"auto", "zero", "one"}:
        raise ValueError("id_base must be one of: auto, zero, one")
    if "dataset_kind" in config and not isinstance(config["dataset_kind"], str):
        raise ValueError("dataset_kind must be a string when provided")
    if "sample_strategy" in config and not isinstance(config["sample_strategy"], str):
        raise ValueError("sample_strategy must be a string when provided")
    if "hardware_notes" in config and not isinstance(config["hardware_notes"], str):
        raise ValueError("hardware_notes must be a string when provided")
    if "include_full_dataset" in config and not isinstance(
        config["include_full_dataset"],
        bool,
    ):
        raise ValueError("include_full_dataset must be a boolean when provided")
    if "query_sample_size" in config and not isinstance(
        config["query_sample_size"],
        int,
    ):
        raise ValueError("query_sample_size must be an integer when provided")
    if "query_holdout_fraction" in config:
        holdout_fraction = config["query_holdout_fraction"]
        if not isinstance(holdout_fraction, int | float):
            raise ValueError("query_holdout_fraction must be numeric when provided")
        if not 0.0 < float(holdout_fraction) < 1.0:
            raise ValueError("query_holdout_fraction must be between 0 and 1")

    for field_name in _SEQUENCE_FIELDS:
        if field_name in config and not isinstance(config[field_name], list):
            raise ValueError(f"{field_name} must be a list when provided")

    for field_name in _MAPPING_FIELDS:
        if field_name in config and not isinstance(config[field_name], dict):
            raise ValueError(f"{field_name} must be a mapping when provided")
