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
_MAPPING_FIELDS = {"limits"}


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

    for field_name in _SEQUENCE_FIELDS:
        if field_name in config and not isinstance(config[field_name], list):
            raise ValueError(f"{field_name} must be a list when provided")

    for field_name in _MAPPING_FIELDS:
        if field_name in config and not isinstance(config[field_name], dict):
            raise ValueError(f"{field_name} must be a mapping when provided")
