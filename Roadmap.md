# Roadmap: From Zero to Complete Assignment

## Milestone 1: Project Foundation

Status: complete as of 2026-06-09.

Goal: create a reproducible Python project skeleton and trustworthy data-loading layer.

Completed implementation:

- Added `pyproject.toml`, `.python-version`, and `uv.lock` for a Python 3.12.2 `uv` project.
- Added dependencies: NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML; dev dependencies: pytest and Ruff.
- Did not add a `Makefile` during Milestone 1; a reproducibility wrapper was added later in Milestone 4.
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

Status: complete as of 2026-06-09.

Goal: implement Methods A-D behind one interface.

Completed implementation:

- Added `src/data_reduction/methods/` with a shared `select_method(...)` dispatcher and `MethodLimits`.
- Implemented Method A exhaustive cosine-proxy subset search for tiny datasets.
- Implemented Method B IndepDF scoring and top-`B` selection with deterministic lower-ID tie-breaking.
- Implemented Method C exact Shapley-value ranking for tiny datasets using the cosine proxy value function.
- Implemented Method D query-mass-weighted greedy facility location as the main proposed method.
- Method D uses all photos as candidate representatives, clipped cosine coverage `max(0, cosine)`, query-mass weights, and float64 optimization math.
- Added Method D memory-aware chunking with `max_facility_similarity_mb`, an effective chunk-size diagnostic, and chunked final cosine-proxy utility scoring.
- Added infeasibility guardrails for Method A and Method C, returning clean skipped `SelectionResult` objects with diagnostics.
- Added strict public validation for budgets and method limits, rejecting non-integer and boolean budgets/limits.
- Added `scripts/run_method.py` to run one method from the command line.
- Added CLI dataset diagnostics in stdout JSON and human-readable ID-base/warning diagnostics on stderr.
- Added clean CLI error handling for method execution and limit validation failures.
- Added method-level and CLI regression tests for deterministic selection, budget validation, exact-method skips, Method D memory chunking, Method D near-tie ordering, and dataset diagnostics.
- Did not add optional random or most-frequent baselines; required Methods A-D took priority.

Implemented commands:

```bash
uv run python scripts/run_method.py --method D --budget 3
uv run pytest tests/test_methods.py
uv run pytest
uv run ruff check .
```

Artifacts:

- Method package at `src/data_reduction/methods/`.
- Public dispatcher and `MethodLimits` exported from `src/data_reduction/__init__.py`.
- Single-method runner script at `scripts/run_method.py`.
- Method and CLI tests in `tests/test_methods.py`.

Validation results:

- `uv run pytest tests/test_methods.py` passes with 24 tests.
- `uv run pytest` passes with 55 tests.
- `uv run ruff check .` passes.
- `uv run python scripts/run_method.py --method D --budget 3` succeeds on the local private dataset.
- Local Method D run selected normalized photo IDs `11708, 24228, 25519`.
- Local Method D cosine-proxy utility: `0.34930597816451886`.
- Local Method D diagnostics include `id_base_auto_ambiguous`, `weighted_target_count=10937`, `effective_candidate_chunk_size=766`, `max_facility_similarity_mb=64`, and `numeric_dtype=float64`.
- Exact Method A and exact Method C are guarded for infeasible runs by candidate/photo and coalition limits.

Exit criteria:

- Complete.

## Milestone 4: Experiments

Status: complete as of 2026-06-09.

Goal: generate the evidence needed for the report.

Completed implementation:

- Added `src/data_reduction/experiments.py` with deterministic query-active sampling, query projection/remapping, experiment grid expansion, synthetic clustered data generation, cross-method metric evaluation, and experiment-only Method D ablation aliases.
- Added `scripts/run_experiments.py` for YAML-driven experiment batches.
- Added `scripts/generate_figures.py` for reproducible Matplotlib figures generated from saved result CSV files.
- Added experiment configs for synthetic sanity checks, small assignment-literal exact comparisons, scalability, budget sensitivity, Method D ablations, and full-data exact infeasibility documentation.
- Saved each method/sample/budget/seed run as one tidy CSV row.
- Saved per-run diagnostics JSON artifacts under each result batch's `diagnostics/` directory.
- Saved batch metadata including Python version, git commit, git dirty flag, config path, config SHA-256 hash, platform/hardware notes, and dataset diagnostics.
- Added cross-method evaluation columns for `cosine_proxy_utility_eval` and `jaccard_precision_utility_eval`.
- Added Method D ablations for `D_frequency_only` and `D_coverage_only`; random, most-frequent, clustering, and Monte Carlo Shapley baselines remain out of scope.
- Added regression tests for grid expansion, deterministic sampling/remapping, dropped-query diagnostics, experiment-only ablations, CLI result/diagnostics writing, figure generation, and oversized exact-infeasibility diagnostics.
- Added a root `Makefile` with reproducibility targets for validation, tests, linting, core experiments, all experiment configs, figure generation, and experiment cleanup.

Implemented commands:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/small.yaml
uv run python scripts/run_experiments.py --config experiments/configs/exact_infeasibility.yaml --batch-id exact_infeasibility_local
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
uv run pytest
uv run ruff check .
```

Artifacts:

- Experiment config files under `experiments/configs/`: `synthetic.yaml`, `small.yaml`, `scalability.yaml`, `budget_sensitivity.yaml`, `d_ablations.yaml`, and `exact_infeasibility.yaml`.
- Experiment helper module at `src/data_reduction/experiments.py`.
- Batch runner script at `scripts/run_experiments.py`.
- Figure generation script at `scripts/generate_figures.py`.
- Result CSV files and diagnostics JSON artifacts under `experiments/results/`.
- Generated figures under `experiments/figures/`: utility, runtime, memory, scalability, and budget sensitivity.
- Experiment tests in `tests/test_experiments.py`.
- Root `Makefile` for reproducible command shortcuts.

Validation results:

- `uv run pytest` passes with 61 tests.
- `uv run ruff check .` passes.
- `uv run python scripts/run_experiments.py --config experiments/configs/small.yaml` succeeds and produced 48 successful rows covering Methods A-D.
- `uv run python scripts/run_experiments.py --config experiments/configs/exact_infeasibility.yaml --batch-id exact_infeasibility_local` succeeds and produced skipped rows for full-data exact Method A and exact Method C.
- `uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures` regenerates all planned figure PNGs from saved results.

Report evidence run:

- Ran `make check` successfully: 61 tests passed and Ruff passed.
- Ran `make exp-all` successfully to generate the assignment-report evidence batches.
- Latest report evidence batches:
  - `synthetic_20260609T115535Z`: 22 successful rows, 2 expected skipped rows.
  - `small_exact_comparison_20260609T115600Z`: 48 successful rows covering Methods A-D.
  - `scalability_20260609T115625Z`: 10 successful Method B/D scalability rows.
  - `budget_sensitivity_20260609T115734Z`: 10 successful Method B/D budget rows.
  - `d_ablations_20260609T115928Z`: 9 successful Method D ablation rows.
  - `exact_infeasibility_20260609T120233Z`: 2 expected skipped rows for full-data exact Method A and Method C.
- Regenerated `experiments/figures/utility.png`, `runtime.png`, `memory.png`, `scalability.png`, and `budget_sensitivity.png` from saved results.
- Main interpretation for the report: Method D gives the strongest scalable cosine-proxy utility, Method B is the fastest and lightest scalable baseline, and exact Methods A/C are useful tiny-data references but infeasible on full data.

Exit criteria:

- Complete.

## Milestone 5: Report and Submission

Goal: package the project for grading.

Implementation tasks:

- Write the ACM SIG proceedings-style report in Overleaf.
- Describe Method D clearly, including objective, algorithm, expected strengths, and limitations.
- Compare Methods A-D on utility, runtime, memory, scalability, and theoretical complexity.
- Include experiment setup, dataset handling, hardware notes, and reproducibility instructions.
- Add a compact limitations section explaining the historical-query proxy and exact-method scale limits.
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
