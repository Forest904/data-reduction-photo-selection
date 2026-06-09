"""Data reduction package."""

from data_reduction.config import load_experiment_config
from data_reduction.data import (
    DataValidationError,
    LoadedQueries,
    ValidationIssue,
    ValidationReport,
    load_photos_csv,
    load_queries_csv,
    validate_dataset,
)
from data_reduction.evaluation import (
    MeasuredResult,
    SelectionResult,
    SelectionStatus,
    best_candidate_with_id_tiebreak,
    measure_peak_memory_and_time,
    rank_descending_with_id_tiebreak,
)
from data_reduction.similarity import (
    cosine_similarity_matrix,
    cosine_similarity_to_selection,
)
from data_reduction.methods import MethodLimits, select_method
from data_reduction.utility import (
    cosine_proxy_utility,
    indepdf_scores,
    jaccard_precision_utility,
)

__all__ = [
    "DataValidationError",
    "LoadedQueries",
    "MeasuredResult",
    "MethodLimits",
    "SelectionResult",
    "SelectionStatus",
    "ValidationIssue",
    "ValidationReport",
    "best_candidate_with_id_tiebreak",
    "cosine_proxy_utility",
    "cosine_similarity_matrix",
    "cosine_similarity_to_selection",
    "indepdf_scores",
    "jaccard_precision_utility",
    "load_experiment_config",
    "load_photos_csv",
    "load_queries_csv",
    "measure_peak_memory_and_time",
    "rank_descending_with_id_tiebreak",
    "select_method",
    "validate_dataset",
]
