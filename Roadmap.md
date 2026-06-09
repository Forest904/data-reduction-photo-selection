# Roadmap: From Zero to Complete Assignment

## Milestone 1: Project Foundation

Goal: create a reproducible Python project skeleton and trustworthy data-loading layer.

Implementation tasks:

- Add `pyproject.toml`, `.python-version`, and `uv.lock`.
- Add a lightweight `Makefile` only if it reduces repeated command typing; otherwise prefer documented `uv run ...` commands.
- Create package skeleton under `src/data_reduction/`.
- Create `data/raw/`, `data/processed/`, `experiments/configs/`, `experiments/results/`, `experiments/figures/`, `scripts/`, and `tests/`.
- Add `data/README.md` explaining where to place `photos.csv` and `queries.csv`.
- Implement CSV loading for photos and queries.
- Implement photo ID normalization with `id_base=auto|zero|one`.
- Implement validation for ID bounds, empty queries, duplicate query IDs, missing values, unequal embedding dimensions, and zero vectors.
- Add tiny synthetic fixtures for tests.

Recommended commands:

```bash
uv init --package
uv add numpy pandas scipy scikit-learn matplotlib pyyaml
uv add --dev pytest ruff
uv run python scripts/validate_data.py
uv run pytest
```

Artifacts:

- Working package scaffold.
- Data validation script.
- Synthetic test fixtures.
- Passing initial data tests.

Exit criteria:

- `uv sync` works.
- Dataset validation produces clear success or failure diagnostics.
- Tests cover photo/query parsing and ID normalization.

## Milestone 2: Shared Math and Evaluation

Goal: build the common evaluation layer used by all methods.

Implementation tasks:

- Implement cosine similarity with safe handling for zero vectors.
- Implement Jaccard/precision-style query utility for Method B.
- Implement cosine proxy utility for Methods A, C, and D.
- Implement shared `SelectionResult`.
- Add timing and peak-memory measurement helpers.
- Add deterministic tie-breaking helpers.
- Add result serialization to CSV/JSON-friendly structures.
- Add a small config loader for YAML experiment files.

Recommended commands:

```bash
uv run pytest tests/test_similarity.py tests/test_utility.py
uv run ruff check .
```

Artifacts:

- Similarity module.
- Utility module.
- Evaluation/result schema module.
- Unit tests with hand-verifiable expected values.

Exit criteria:

- Cosine and Jaccard utilities match expected values on synthetic examples.
- Utility calculation is independent of method implementation.
- Result objects serialize cleanly for experiment output.

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
