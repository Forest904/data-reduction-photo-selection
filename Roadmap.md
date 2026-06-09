# Roadmap: From Zero to Complete Assignment

## Milestone 1: Project Foundation

Status: complete as of 2026-06-09.

Goal: create a reproducible Python project skeleton and trustworthy data-loading layer.

Completed implementation:

- Added `pyproject.toml`, `.python-version`, and `uv.lock` for a Python 3.12.2 `uv` project.
- Added dependencies: NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML; dev dependencies: pytest and Ruff.
- Did not add a `Makefile`; Milestone 1 uses direct documented `uv run ...` commands.
- Created package skeleton under `src/data_reduction/`.
- Created `data/raw/`, `data/processed/`, `experiments/configs/`, `experiments/results/`, `experiments/figures/`, `scripts/`, and `tests/`.
- Added `data/README.md` explaining where to place `photos.csv` and `queries.csv`.
- Moved the private dataset into `data/raw/photos.csv` and `data/raw/queries.csv`.
- Removed the extracted `datareduction_dataset/` folder after moving the CSVs.
- Added `.gitignore` rules so private raw CSVs and extracted dataset folders are not committed.
- Implemented CSV loading for headerless photo embeddings and variable-length query rows.
- Implemented photo ID normalization with `id_base=auto|zero|one`.
- Implemented validation for ID bounds, empty queries, duplicate query IDs, missing values, malformed or unequal embedding rows, non-numeric values, and zero vectors.
- Added tiny synthetic fixtures and initial data-loading tests.

Implemented commands:

```bash
uv sync
uv run python scripts/validate_data.py
uv run pytest
uv run ruff check .
```

Artifacts:

- Working package scaffold in `src/data_reduction/`.
- Data validation script at `scripts/validate_data.py`.
- Synthetic test fixtures under `tests/fixtures/`.
- Passing initial data tests in `tests/test_data_loading.py`.
- Tracked placeholders for data and experiment output directories.

Validation results:

- `uv sync` works.
- `uv run python scripts/validate_data.py` succeeds on the local private dataset.
- Local dataset summary: 41,620 photos, 2,048 embedding dimensions, 443 queries.
- `id_base=auto` is ambiguous for the local dataset, so validation defaults to zero-based IDs and reports `id_base_auto_ambiguous`.
- `uv run pytest` passes with 11 tests.
- `uv run ruff check .` passes.

Exit criteria:

- Complete.

## Milestone 2: Shared Math and Evaluation

Status: complete as of 2026-06-09.

Goal: build the common evaluation layer used by all methods.

Completed implementation:

- Added `src/data_reduction/similarity.py` with vectorized cosine similarity helpers.
- Implemented safe cosine handling for zero vectors, including zero-vs-zero returning `0.0` without NaNs.
- Added `src/data_reduction/utility.py` with Jaccard/precision utility, IndepDF scores, and cosine proxy utility.
- Treat query rows as normalized ID sets in shared utility calculations.
- Added `src/data_reduction/evaluation.py` with shared `SelectionResult`, status typing, timing and peak-memory measurement, deterministic tie-breaking helpers, and CSV/JSON-friendly serialization.
- Added `src/data_reduction/config.py` with a lightweight YAML experiment config loader and basic top-level shape validation.
- Exported the new public helpers from `src/data_reduction/__init__.py`.
- Added hand-verifiable tests for similarity, utility, serialization, measurement helpers, tie-breaking, and config loading.

Implemented commands:

```bash
uv run pytest tests/test_similarity.py tests/test_utility.py tests/test_evaluation.py tests/test_config.py
uv run pytest
uv run ruff check .
```

Artifacts:

- Similarity module at `src/data_reduction/similarity.py`.
- Utility module at `src/data_reduction/utility.py`.
- Evaluation/result schema module at `src/data_reduction/evaluation.py`.
- Config loader at `src/data_reduction/config.py`.
- Unit tests with hand-verifiable expected values in `tests/test_similarity.py`, `tests/test_utility.py`, `tests/test_evaluation.py`, and `tests/test_config.py`.

Validation results:

- `uv run pytest tests/test_similarity.py tests/test_utility.py tests/test_evaluation.py tests/test_config.py` passes with 20 tests.
- `uv run pytest` passes with 31 tests.
- `uv run ruff check .` passes.

Exit criteria:

- Complete.

## Milestone 3: Required Methods

Goal: implement Methods A-D behind one interface.

Implementation tasks:

- Implement Method A exhaustive cosine search.
- Implement Method B IndepDF scoring and top-`B` selection.
- Implement Method C exact Shapley for tiny datasets.
- Implement Method D query-aware greedy facility-location.
- Add optional random and most-frequent baselines if time allows.
- Add infeasibility guardrails for exhaustive and Shapley runs.
- Add method-level tests for deterministic selection and budget validation.
- Add a script to run one method from the command line.

Recommended commands:

```bash
uv run python scripts/run_method.py --method D --budget 3
uv run pytest tests/test_methods.py
uv run ruff check .
```

Artifacts:

- Method modules for A-D.
- Single-method runner script.
- Method tests.

Exit criteria:

- Every method returns a valid `SelectionResult`.
- Selected IDs are unique and within bounds.
- Selected count never exceeds `budget`.
- Method A and Method C skip cleanly when configured limits are exceeded.
- Deterministic methods produce stable results across repeated runs.

## Milestone 4: Experiments

Goal: generate the evidence needed for the report.

Implementation tasks:

- Create experiment configs for synthetic, assignment-literal exact, scalability, budget sensitivity, and stretch ablations.
- Implement `scripts/run_experiments.py`.
- Save each run as one tidy result row.
- Save method-specific diagnostics as JSON artifacts when needed.
- Implement `scripts/generate_figures.py`.
- Run exact 3-photo comparison on feasible sampled data.
- Run larger comparisons for Method B and Method D.
- Run Method D ablations and extra baselines if schedule allows.
- Record Python version, git commit hash, config path, and hardware notes with each experiment batch.

Recommended commands:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/small.yaml
uv run python scripts/run_experiments.py --config experiments/configs/scalability.yaml
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
```

Artifacts:

- Experiment config files.
- Result CSV files.
- Diagnostics JSON files.
- Figures for utility, runtime, memory, scalability, and budget sensitivity.

Exit criteria:

- Required Methods A-D appear in the core comparison.
- Exact Method A and exact Method C are either successfully run on tiny data or skipped with documented infeasibility diagnostics.
- Figures are reproducible from saved results.
- Results support a clear complexity-vs-measurement discussion.

## Milestone 5: Report and Submission

Goal: package the project for grading.

Implementation tasks:

- Write the ACM SIG proceedings-style report in Overleaf.
- Describe Method D clearly, including objective, algorithm, expected strengths, and limitations.
- Compare Methods A-D on utility, runtime, memory, scalability, and theoretical complexity.
- Include experiment setup, dataset handling, hardware notes, and reproducibility instructions.
- Add a compact limitations section explaining the historical-query proxy and exact-method scale limits.
- Share GitHub with `velgias@gmail.com`.
- Share Overleaf with `velgias@gmail.com` and `riccardo.torlone@uniroma3.it`.
- Prepare final submission email.

Recommended commands:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
```

Artifacts:

- Final report.
- Final GitHub repository.
- Final Overleaf project.
- Submission email draft.

Exit criteria:

- Clean clone instructions are accurate.
- Tests and linting pass.
- Report figures match generated artifacts.
- Submission email includes full name, matricola, GitHub link, and Overleaf link.

## Priority Rules

- Required assignment work comes before stretch research additions.
- Do Method A and exact Method C only at tiny scale.
- Keep all experiment choices reproducible with configs and seeds.
- Do not commit private dataset files.
- Document every skipped exact run with the reason, candidate count estimate, and configured limit.
- Keep README concise; put implementation decisions in `PRD.md` and execution sequencing in this file.
