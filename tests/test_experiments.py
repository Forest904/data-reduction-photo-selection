import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

from data_reduction import (
    expand_experiment_grid,
    sample_dataset,
    select_experiment_method,
    select_method,
)


def test_expand_experiment_grid_returns_cartesian_tasks():
    tasks = expand_experiment_grid(
        {
            "methods": ["A", "D"],
            "sample_sizes": [6, 8],
            "budgets": [3],
            "seeds": [0, 1],
        }
    )

    assert len(tasks) == 8
    assert tasks[0].method == "A"
    assert tasks[0].sample_size == 6
    assert tasks[-1].method == "D"
    assert tasks[-1].seed == 1


def test_sample_dataset_is_deterministic_and_records_dropped_queries():
    photos = np.eye(6)
    queries = ((0, 1), (2, 3), (4, 5))

    first = sample_dataset(photos, queries, sample_size=2, seed=7)
    second = sample_dataset(photos, queries, sample_size=2, seed=7)

    assert first.original_photo_ids == second.original_photo_ids
    assert first.queries == second.queries
    assert first.diagnostics["sampled_photo_count"] == 2
    assert first.diagnostics["dropped_query_count"] > 0
    assert all(
        0 <= photo_id < first.photos.shape[0]
        for query in first.queries
        for photo_id in query
    )


def test_experiment_method_d_ablations_are_valid_and_do_not_change_d():
    photos = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    queries = ((0,), (1,))

    direct_d = select_method("D", photos, queries, budget=1)
    experiment_d = select_experiment_method("D", photos, queries, budget=1)
    frequency = select_experiment_method("D_frequency_only", photos, queries, budget=1)
    coverage = select_experiment_method("D_coverage_only", photos, queries, budget=1)

    assert experiment_d.selected_ids == direct_d.selected_ids
    assert frequency.status == "success"
    assert frequency.method == "D_frequency_only"
    assert len(frequency.selected_ids) == 1
    assert coverage.status == "success"
    assert coverage.method == "D_coverage_only"
    assert len(coverage.selected_ids) == 1


def test_run_experiments_writes_tidy_row_and_diagnostics(tmp_path: Path):
    config_path = tmp_path / "exact_skip.yaml"
    output_dir = tmp_path / "results"
    config_path.write_text(
        """
experiment_name: exact_skip_test
dataset_kind: raw
photos_path: tests/fixtures/photos_valid.csv
queries_path: tests/fixtures/queries_zero.csv
id_base: zero
methods: [A]
sample_sizes: [3]
budgets: [2]
seeds: [0]
limits:
  max_exhaustive_photos: 1
""".strip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--batch-id",
            "test_batch",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_dir / "test_batch" / "results.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["git_commit"]
    assert rows[0]["python_version"]
    diagnostics_path = Path(rows[0]["diagnostics_path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["result"]["diagnostics"]["reason"] == (
        "max_exhaustive_photos_exceeded"
    )
    assert diagnostics["sample"]["sampled_original_photo_ids"] == [0, 1, 2]


def test_generate_figures_creates_expected_pngs(tmp_path: Path):
    batch_dir = tmp_path / "results" / "batch"
    batch_dir.mkdir(parents=True)
    csv_path = batch_dir / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "experiment_id,batch_id,experiment_name,method,dataset_size,"
                "num_queries,sample_size_requested,budget,seed,utility_metric,"
                "utility,cosine_proxy_utility_eval,jaccard_precision_utility_eval,"
                "runtime_seconds,peak_memory_mb,selected_ids,selected_original_ids,"
                "status,message,diagnostics_path,config_path,git_commit,git_dirty,"
                "python_version,hardware_notes",
                "e1,b,small,D,6,2,6,3,0,cosine_proxy_utility,0.8,0.8,0.5,"
                "0.01,1.2,1,1,success,,,c.yaml,abc,false,3.12,test",
                "e2,b,small,B,6,2,6,3,0,jaccard_precision_utility,0.5,0.7,0.5,"
                "0.005,0.8,0,0,success,,,c.yaml,abc,false,3.12,test",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_figures.py",
            "--results",
            str(tmp_path / "results"),
            "--output",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for name in (
        "utility.png",
        "runtime.png",
        "memory.png",
        "scalability.png",
        "budget_sensitivity.png",
    ):
        figure_path = output_dir / name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0


def test_json_safe_stringifies_oversized_integers():
    huge_value = 1 << 5000
    namespace = runpy.run_path("scripts/run_experiments.py")

    converted = namespace["_json_safe"]({"coalition_count": huge_value})

    assert converted == {"coalition_count": "2**5000"}
