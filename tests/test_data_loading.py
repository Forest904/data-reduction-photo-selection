from pathlib import Path

import numpy as np
import pytest

from data_reduction import (
    DataValidationError,
    load_photos_csv,
    load_queries_csv,
    validate_dataset,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_photos_csv_parses_headerless_embeddings():
    photos = load_photos_csv(FIXTURES / "photos_valid.csv")

    assert photos.shape == (3, 2)
    assert photos.dtype == np.float32
    np.testing.assert_allclose(photos[2], [1.0, 1.0])


def test_load_queries_csv_parses_zero_based_ids():
    queries = load_queries_csv(FIXTURES / "queries_zero.csv", num_photos=3, id_base="zero")

    assert queries.queries == ((0, 2), (1,))
    assert queries.id_base_resolved == "zero"


def test_load_queries_csv_parses_one_based_ids():
    queries = load_queries_csv(FIXTURES / "queries_one.csv", num_photos=3, id_base="one")

    assert queries.queries == ((0, 2), (1,))
    assert queries.id_base_resolved == "one"


def test_auto_id_base_ambiguous_defaults_to_zero_with_diagnostic():
    queries = load_queries_csv(
        FIXTURES / "queries_auto_ambiguous.csv",
        num_photos=4,
        id_base="auto",
    )

    assert queries.id_base_resolved == "zero"
    assert queries.queries == ((1, 2), (2, 3))
    assert [issue.code for issue in queries.diagnostics] == [
        "id_base_auto_ambiguous"
    ]


@pytest.mark.parametrize(
    ("fixture_name", "expected_code"),
    [
        ("queries_out_of_bounds.csv", "query_id_out_of_bounds"),
        ("queries_empty.csv", "empty_query"),
        ("queries_duplicate.csv", "duplicate_query_id"),
        ("queries_missing.csv", "query_missing_value"),
    ],
)
def test_load_queries_csv_rejects_invalid_queries(fixture_name, expected_code):
    with pytest.raises(DataValidationError) as error:
        load_queries_csv(FIXTURES / fixture_name, num_photos=3, id_base="zero")

    assert expected_code in {issue.code for issue in error.value.issues}


@pytest.mark.parametrize(
    ("fixture_name", "expected_code"),
    [
        ("photos_missing.csv", "photos_missing_value"),
        ("photos_unequal.csv", "photos_malformed_csv"),
    ],
)
def test_load_photos_csv_rejects_invalid_photos(fixture_name, expected_code):
    with pytest.raises(DataValidationError) as error:
        load_photos_csv(FIXTURES / fixture_name)

    assert expected_code in {issue.code for issue in error.value.issues}


def test_validate_dataset_reports_zero_vectors_as_warning():
    photos = load_photos_csv(FIXTURES / "photos_zero.csv")

    report = validate_dataset(photos, ((0, 1),))

    assert report.is_valid
    assert [issue.code for issue in report.warnings] == ["zero_photo_vectors"]
