import json

import numpy as np

from data_reduction import SelectionResult, measure_peak_memory_and_time


def test_selection_result_serializes_to_json_friendly_values():
    result = SelectionResult(
        selected_ids=(2, 0),
        utility=np.float64(0.75),
        runtime_seconds=0.1,
        peak_memory_mb=0.2,
        diagnostics={
            "scores": np.array([0.1, 0.2]),
            "nested": {"ids": (2, 0)},
        },
        method="B",
        utility_metric="jaccard_precision_utility",
    )

    serialized = result.to_json_dict()

    assert serialized["selected_ids"] == [2, 0]
    assert serialized["diagnostics"] == {
        "scores": [0.1, 0.2],
        "nested": {"ids": [2, 0]},
    }
    json.dumps(serialized)


def test_selection_result_to_dict_is_csv_friendly():
    result = SelectionResult(
        selected_ids=(2, 0),
        utility=0.75,
        runtime_seconds=0.1,
        peak_memory_mb=0.2,
        diagnostics={"score": 0.75},
    )

    serialized = result.to_dict()

    assert serialized["selected_ids"] == "2,0"
    assert serialized["diagnostics"] == '{"score": 0.75}'


def test_measure_peak_memory_and_time_returns_value_and_non_negative_metrics():
    measured = measure_peak_memory_and_time(lambda value: [value] * 10, 3)

    assert measured.value == [3] * 10
    assert measured.runtime_seconds >= 0.0
    assert measured.peak_memory_mb >= 0.0
