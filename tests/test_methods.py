import json
import subprocess
import sys

import numpy as np
import pytest

from data_reduction import MethodLimits, cosine_proxy_utility, select_method


def _centroid_fixture() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    return (
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        ((0,), (1,)),
    )


def _assert_valid_selection(result, num_photos: int, budget: int) -> None:
    assert result.status == "success"
    assert len(result.selected_ids) == len(set(result.selected_ids))
    assert len(result.selected_ids) <= min(budget, num_photos)
    assert all(0 <= photo_id < num_photos for photo_id in result.selected_ids)


@pytest.mark.parametrize("method", ["A", "B", "C", "D"])
def test_methods_return_valid_selection_results(method: str):
    photos, queries = _centroid_fixture()

    result = select_method(method, photos, queries, budget=2)

    _assert_valid_selection(result, num_photos=3, budget=2)
    assert result.method == method
    assert result.utility >= 0.0


def test_method_a_exhaustive_selects_best_cosine_subset():
    photos, queries = _centroid_fixture()

    result = select_method("A", photos, queries, budget=1)

    assert result.selected_ids == (2,)
    assert result.utility == pytest.approx(np.sqrt(0.5))
    assert result.utility_metric == "cosine_proxy_utility"
    assert result.diagnostics["candidate_subsets"] == 3
    assert result.diagnostics["evaluated_subsets"] == 3


def test_method_a_skips_when_combination_guardrail_is_exceeded():
    photos = np.eye(4)

    result = select_method(
        "A",
        photos,
        ((0,),),
        budget=2,
        limits=MethodLimits(max_exhaustive_combinations=1),
    )

    assert result.status == "skipped"
    assert result.selected_ids == ()
    assert result.utility == 0.0
    assert result.diagnostics["reason"] == "max_exhaustive_combinations_exceeded"
    assert result.diagnostics["candidate_subsets"] == 6


def test_method_b_indepdf_tie_breaks_by_lower_photo_id():
    photos = np.ones((4, 2))
    queries = ((0, 1), (2, 3))

    result = select_method("B", photos, queries, budget=2)

    assert result.selected_ids == (0, 1)
    assert result.utility == pytest.approx(0.5)
    assert result.utility_metric == "jaccard_precision_utility"
    assert result.diagnostics["selected_scores"] == {"0": 0.25, "1": 0.25}
    assert result.diagnostics["score_formula"] == "mean_query_membership_mass"
    assert result.diagnostics["nonzero_score_count"] == 4
    assert result.diagnostics["tie_breaking"] == "descending score, then lower photo ID"


def test_method_c_exact_shapley_selects_centroid_representative():
    photos, queries = _centroid_fixture()

    result = select_method("C", photos, queries, budget=1)

    assert result.selected_ids == (2,)
    assert result.utility == pytest.approx(np.sqrt(0.5))
    shapley_values = result.diagnostics["shapley_values"]
    assert shapley_values[2] > shapley_values[0]
    assert shapley_values[2] > shapley_values[1]
    assert result.diagnostics["coalition_count"] == 8


def test_method_c_skips_when_shapley_guardrail_is_exceeded():
    photos, queries = _centroid_fixture()

    result = select_method(
        "C",
        photos,
        queries,
        budget=1,
        limits=MethodLimits(max_shapley_coalitions=4),
    )

    assert result.status == "skipped"
    assert result.selected_ids == ()
    assert result.utility == 0.0
    assert result.diagnostics["reason"] == "max_shapley_coalitions_exceeded"
    assert result.diagnostics["coalition_count"] == 8


def test_method_d_can_select_unqueried_centroid_representative():
    photos, queries = _centroid_fixture()

    result = select_method("D", photos, queries, budget=1)

    assert result.selected_ids == (2,)
    assert result.utility == pytest.approx(np.sqrt(0.5))
    assert result.diagnostics["facility_objective"] == pytest.approx(np.sqrt(0.5))
    assert result.diagnostics["selected_query_mass_weights"] == [0.0]
    assert result.diagnostics["weighted_target_count"] == 2
    assert result.diagnostics["candidate_scope"] == "all_photos"


def test_method_d_is_deterministic_and_records_marginal_gains():
    photos, queries = _centroid_fixture()
    limits = MethodLimits(candidate_chunk_size=1)

    first = select_method("D", photos, queries, budget=2, seed=123, limits=limits)
    second = select_method("D", photos, queries, budget=2, seed=999, limits=limits)

    assert first.selected_ids == second.selected_ids
    assert first.diagnostics["marginal_gains"] == pytest.approx(
        second.diagnostics["marginal_gains"]
    )
    assert len(first.diagnostics["marginal_gains"]) == len(first.selected_ids)


def test_method_d_near_ties_still_prefer_higher_gain_over_lower_id():
    photos = np.array(
        [
            [1.0, 1e-5],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    result = select_method("D", photos, ((2,),), budget=1)

    assert result.selected_ids == (1,)


def test_method_d_memory_cap_reduces_effective_chunk_size_without_changing_result():
    photos, queries = _centroid_fixture()
    capped_limits = MethodLimits(
        candidate_chunk_size=1_000_000,
        max_facility_similarity_mb=1,
    )

    capped = select_method("D", photos, queries, budget=1, limits=capped_limits)
    default = select_method("D", photos, queries, budget=1)

    assert capped.selected_ids == default.selected_ids
    assert capped.diagnostics["effective_candidate_chunk_size"] < 1_000_000
    assert capped.diagnostics["configured_candidate_chunk_size"] == 1_000_000
    assert capped.diagnostics["max_facility_similarity_mb"] == 1
    assert capped.diagnostics["numeric_dtype"] == "float64"


def test_method_d_chunked_final_utility_matches_shared_cosine_proxy():
    photos, queries = _centroid_fixture()

    result = select_method("D", photos, queries, budget=2)

    assert result.utility == pytest.approx(
        cosine_proxy_utility(photos, queries, result.selected_ids)
    )


def test_method_d_handles_zero_and_oversized_budgets():
    photos, queries = _centroid_fixture()

    zero_budget = select_method("D", photos, queries, budget=0)
    oversized_budget = select_method("D", photos, queries, budget=99)

    assert zero_budget.selected_ids == ()
    assert zero_budget.status == "success"
    _assert_valid_selection(oversized_budget, num_photos=3, budget=99)
    assert oversized_budget.diagnostics["effective_budget"] == 3


@pytest.mark.parametrize("method", ["A", "B", "C", "D"])
def test_methods_reject_negative_budget(method: str):
    photos, queries = _centroid_fixture()

    with pytest.raises(ValueError, match="budget must be non-negative"):
        select_method(method, photos, queries, budget=-1)


@pytest.mark.parametrize("budget", [1.5, True])
def test_methods_reject_non_integer_budgets(budget):
    photos, queries = _centroid_fixture()

    with pytest.raises(TypeError, match="budget must be an integer"):
        select_method("D", photos, queries, budget=budget)


def test_method_limits_mapping_rejects_non_integer_values():
    photos, queries = _centroid_fixture()

    with pytest.raises(TypeError, match="candidate_chunk_size must be an integer"):
        select_method("D", photos, queries, budget=1, limits={"candidate_chunk_size": 1.5})


def test_run_method_cli_includes_dataset_diagnostics_and_stderr_warning(tmp_path):
    photos_path = tmp_path / "photos.csv"
    photos_path.write_text("1,0\n0,1\n1,1\n-1,0\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_method.py",
            "--method",
            "D",
            "--budget",
            "1",
            "--photos",
            str(photos_path),
            "--queries",
            "tests/fixtures/queries_auto_ambiguous.csv",
            "--id-base",
            "auto",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output = json.loads(completed.stdout)
    diagnostics = output["dataset_diagnostics"]
    assert diagnostics["id_base_requested"] == "auto"
    assert diagnostics["id_base_resolved"] == "one"
    assert diagnostics["id_normalization_diagnostics"][0]["code"] == (
        "id_base_auto_ambiguous"
    )
    assert "id_base_auto_ambiguous" in completed.stderr


def test_run_method_cli_invalid_budget_exits_cleanly_without_traceback():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_method.py",
            "--method",
            "D",
            "--budget",
            "-1",
            "--photos",
            "tests/fixtures/photos_valid.csv",
            "--queries",
            "tests/fixtures/queries_zero.csv",
            "--id-base",
            "zero",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Method execution failed." in completed.stderr
    assert "budget must be non-negative" in completed.stderr
    assert "Traceback" not in completed.stderr
