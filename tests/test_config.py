from pathlib import Path

import pytest

from data_reduction import load_experiment_config


def test_load_experiment_config_loads_minimal_yaml(tmp_path: Path):
    config_path = tmp_path / "small.yaml"
    config_path.write_text(
        """
experiment_name: small
photos_path: data/raw/photos.csv
queries_path: data/raw/queries.csv
methods: [A, B]
sample_sizes: [6]
budgets: [3]
seeds: [0]
metrics:
  - cosine_proxy_utility
limits:
  max_combinations: 100
""".strip(),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config["experiment_name"] == "small"
    assert config["methods"] == ["A", "B"]
    assert config["limits"] == {"max_combinations": 100}


def test_load_experiment_config_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "missing.yaml")


def test_load_experiment_config_rejects_non_mapping_root(tmp_path: Path):
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- A\n- B\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_wrong_basic_shapes(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("methods: A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="methods must be a list"):
        load_experiment_config(config_path)
