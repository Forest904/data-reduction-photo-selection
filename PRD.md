# Product Requirements Document: Data Reduction Photo Selection

## 1. Purpose

This project implements and evaluates data-reduction strategies for query-aware photo selection. Given a full photo collection `D`, represented by embedding vectors in `photos.csv`, and a historical query log in `queries.csv`, the system must select a retained subset `D'` of at most `B` photos that preserves as much query usefulness as possible.

The assignment requires four methods:

- Method A: exhaustive cosine-based selection.
- Method B: IndepDF with Jaccard/precision-style scoring.
- Method C: Shapley-value-based selection.
- Method D: an original proposed method.

The chosen Method D is a query-frequency-weighted greedy facility-location method using cosine similarity.

The implementation must prioritize assignment completion by the June 15, 2026 deadline, while still supporting a fuller research-style comparison where feasible.

## 2. Deliverables

The finished project must include:

- Python implementation of Methods A-D.
- Shared data loading, validation, similarity, utility, and evaluation code.
- Experiment scripts and configuration files.
- Saved experiment results and generated figures.
- Tests for data parsing, utility calculations, and method behavior on small fixtures.
- Reproducibility instructions.
- ACM SIG proceedings-style Overleaf report.
- GitHub repository shared with `velgias@gmail.com`.
- Overleaf project shared with `velgias@gmail.com` and `riccardo.torlone@uniroma3.it`.
- Submission email to `i.velegrakis@uu.nl` and `riccardo.torlone@uniroma3.it` with subject `Roma3 Data Reduction`.

## 3. Technology Stack

- Python 3.12+
- `uv` for environment and dependency management
- NumPy for numerical arrays and vectorized operations
- pandas for result tables and CSV experiment outputs
- SciPy and/or scikit-learn for scientific utilities and validated similarity operations
- Matplotlib for figures
- PyYAML for experiment configuration
- pytest for tests
- Ruff for linting and formatting
- Standard-library `argparse` for scripts and command-line entry points

## 4. Planned Repository Structure

The implementation should use a small, conventional Python package layout:

```text
data-reduction-photo-selection/
  README.md
  PRD.md
  Roadmap.md
  AGENT.md
  pyproject.toml
  uv.lock
  .python-version
  .gitignore
  data/
    README.md
    raw/
      photos.csv
      queries.csv
    processed/
  src/
    data_reduction/
      __init__.py
      cli.py
      config.py
      data.py
      similarity.py
      utility.py
      evaluation.py
      experiments.py
      methods/
        __init__.py
        exhaustive_cosine.py
        indepdf.py
        shapley.py
        proposed.py
  scripts/
    validate_data.py
    run_method.py
    run_experiments.py
    generate_figures.py
  tests/
    conftest.py
    test_data.py
    test_similarity.py
    test_utility.py
    test_methods.py
  experiments/
    configs/
    results/
    figures/
  docs/
    method-d.md
    experimental-design.md
    complexity-analysis.md
```

The `docs/` directory is optional at the start, but it should be added when implementation decisions become too detailed for the README.

## 5. Data Contract

### Inputs

`data/raw/photos.csv`

- One photo embedding per row.
- The photo identifier is the row index after normalization.
- All rows must have the same embedding dimension.
- Missing, non-numeric, or malformed values are validation errors.
- Zero vectors must be reported because cosine similarity is undefined without handling.

`data/raw/queries.csv`

- One historical query result set per row.
- Each value is a photo identifier returned by that historical query.
- Empty query rows must be rejected or explicitly skipped by configuration.
- Duplicate identifiers within one query should be deduplicated once during loading, with diagnostics recorded.

### Photo ID Normalization

Photo identifiers must be normalized in the data-loading layer only. Individual algorithms must never reinterpret identifier bases.

Supported setting:

```text
id_base = auto | zero | one
```

Rules:

- `zero`: query IDs are already zero-based.
- `one`: subtract one from every query ID.
- `auto`: detect clear zero-based or one-based cases from query IDs and number of photo rows.
- If `auto` is ambiguous, default to zero-based and record that decision in validation diagnostics.
- If any normalized query ID is outside `[0, num_photos - 1]`, validation fails.

## 6. Core Interfaces

All selection methods should expose the same logical interface:

```python
def select(
    photos: np.ndarray,
    queries: list[np.ndarray],
    budget: int,
    seed: int | None = None,
) -> SelectionResult:
    ...
```

`photos` is an array with shape `(num_photos, embedding_dim)`.

`queries` is a list of integer arrays containing normalized photo IDs.

`budget` is the maximum number of photos to retain.

`seed` controls stochastic behavior. Deterministic methods should accept it for interface consistency.

`SelectionResult` should contain:

- `selected_ids`: retained normalized photo IDs.
- `utility`: final measured utility under the configured evaluation metric.
- `runtime_seconds`: elapsed execution time.
- `peak_memory_mb`: peak memory usage during selection/evaluation where measurable.
- `diagnostics`: method-specific metadata such as scores, candidate counts, or approximation settings.
- `status`: `success`, `skipped`, `timeout`, or `error`.
- `message`: short human-readable explanation for non-success statuses.

## 7. Utility Definitions

The query log contains historical result IDs, not executable query objects. Therefore, `q(D')` must be approximated consistently across the project.

Primary cosine proxy for Methods A, C, and D:

- For each original query result photo `d` in `q(D)`, find the most similar retained photo `s` in `D'`.
- Similarity is cosine similarity over embeddings.
- Query utility is the average of these best similarities.
- Dataset utility is the empirical average over all query rows.

Jaccard/precision-style utility for Method B:

- Use query membership in the historical result sets.
- IndepDF scores each photo by its empirical expected membership contribution normalized by query length.
- This follows the assignment's required Jaccard/IndepDF direction and the referenced paper's independent data-forgetting setup.

Result reports should make clear which utility is being used for each comparison. Cross-method quality tables should include the primary cosine proxy when comparing A, C, and D, and may include Jaccard/precision as a secondary metric.

## 8. Method Requirements

### Method A: Exhaustive Cosine-Based Selection

Purpose:

- Exact baseline for very small datasets.

Logic:

- Enumerate all subsets of size `B` from the candidate photo set.
- Compute the cosine proxy utility for each subset.
- Return the subset with maximum utility.

Constraints:

- Must include guardrails for infeasible `n choose B` runs.
- Intended mainly for the assignment-literal small exact setting, especially keeping 3 photos.
- If a run exceeds configured candidate, time, or memory limits, return `status="skipped"` or `status="timeout"` with diagnostics.

Diagnostics:

- Number of candidate subsets evaluated.
- Best utility.
- Runtime and peak memory.

### Method B: IndepDF with Jaccard Similarity

Purpose:

- Scalable query-log-based method from the referenced paper and assignment.

Logic:

- For each photo `d`, compute:

```text
score(d) = empirical average over q of I_q(d) / |q(D)|
```

- `I_q(d)` is 1 if `d` appears in query `q`, otherwise 0.
- Return the top `B` photos by score.
- Tie-break deterministically by lower photo ID.

Diagnostics:

- Score per selected photo.
- Optional full score vector path or summary statistics.
- Runtime and peak memory.

### Method C: Shapley-Value-Based Selection

Purpose:

- Principled individual photo importance baseline.

Logic:

- Compute exact Shapley values on tiny datasets using the selected value function.
- Main assignment setting keeps 3 photos.
- Select the top 3 photos by Shapley value.

Default value function:

- Use the same cosine proxy utility as Method A for consistency with the embedding-aware evaluation.

Stretch options:

- Try exact 4-photo selection if computationally feasible.
- Add Monte Carlo Shapley approximation for larger slices if time permits.

Constraints:

- Exact Shapley is exponential and must be guarded by maximum dataset size or coalition count.
- Tie-break deterministically by lower photo ID.

Diagnostics:

- Shapley values for selected photos.
- Coalition count or sampled permutation count.
- Runtime and peak memory.

### Method D: Query-Aware Greedy Facility Location

Purpose:

- Original proposed method that combines query awareness, embedding similarity, popularity, and diversity.

Objective:

```text
F(S) = sum over d in D of w_d * max over s in S cosine(d, s)
```

where `w_d` is the query-derived importance weight for photo `d`.

Weighting:

- Compute `w_d` from normalized frequency in the query log.
- Photos that never appear in the query log receive weight 0 by default.
- Normalize weights so the sum of weights over queried photos is 1.

Logic:

- Start with an empty selected set.
- Maintain each photo's current best similarity to the selected set.
- At each step, add the candidate with largest marginal gain.
- Stop when `B` photos have been selected or no valid candidates remain.
- Tie-break deterministically by lower photo ID.

Expected strengths:

- More scalable than exhaustive search.
- Embedding-aware like Method A.
- Query-aware like Method B.
- Encourages diversity because near-duplicates have diminishing marginal gain after one representative is selected.

Diagnostics:

- Marginal gain per selected photo.
- Final objective value.
- Runtime and peak memory.

## 9. Experiment Requirements

### Required Experiment Groups

Synthetic correctness:

- Use tiny hand-built embeddings and query logs.
- Verify cosine similarity, Jaccard/precision scoring, utility values, and deterministic selections.

Assignment-literal exact comparison:

- Run Methods A, B, C, and D on tiny sampled data where exact A and exact C are feasible.
- Main exact setting should keep 3 photos.
- If exact full-data execution is infeasible, document why using candidate counts and runtime estimates.

Scalability:

- Run scalable methods on increasing photo/query slices.
- Stop Method A and exact Method C once configured limits are exceeded.
- Compare measured runtime and memory to theoretical complexity.

Budget sensitivity:

- Evaluate how utility changes as `B` increases.
- Include at least Method B and Method D; include Method A/C only when feasible.

Stability:

- Repeat stochastic or sampled variants with fixed seeds.
- Deterministic methods should produce identical selected IDs across repeated runs.

Research-fuller stretch:

- Random baseline.
- Most frequently retrieved baseline.
- Method D ablations:
  - frequency-only;
  - embedding-coverage-only;
  - combined frequency plus cosine facility location.

### Metrics

Every experiment row should record:

- experiment ID
- method
- dataset size
- number of queries
- budget
- seed
- utility metric name
- utility value
- runtime seconds
- peak memory MB
- selected IDs or artifact path
- status
- message

### Experiment Configuration

Experiment configurations should be YAML files in `experiments/configs/`.

Minimum useful fields:

```yaml
experiment_name: small_exact_comparison
photos_path: data/raw/photos.csv
queries_path: data/raw/queries.csv
methods: [A, B, C, D]
sample_sizes: [6, 8, 10]
budgets: [3]
seeds: [0, 1, 2, 3, 4]
metrics:
  - cosine_proxy_utility
  - jaccard_precision_utility
  - runtime_seconds
  - peak_memory_mb
limits:
  max_combinations: 100000
  max_shapley_coalitions: 100000
```

### Result Format

Every run should produce one tidy CSV row. Method-specific details may be stored as JSON artifacts referenced by path.

Recommended core schema:

```text
experiment_id,method,dataset_size,num_queries,budget,seed,utility_metric,utility,runtime_seconds,peak_memory_mb,selected_ids,status,message,diagnostics_path
```

## 10. Testing and Quality Requirements

Minimum test coverage:

- CSV parsing.
- Photo ID normalization.
- Query ID bounds validation.
- Duplicate and empty query handling.
- Cosine similarity.
- Jaccard/precision utility.
- Cosine proxy utility.
- Budget validation.
- Deterministic tie-breaking.
- Exact subset selection on tiny fixtures.
- IndepDF score calculation.
- Exact Shapley on a tiny fixture.
- Method D marginal-gain updates.

Quality commands:

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Formatting may be applied during implementation, but generated experiment results should not be manually edited.

## 11. Complexity Expectations

The report must compare measured runtime and memory trends against theoretical expectations.

| Method | Expected behavior | Main bottleneck |
| --- | --- | --- |
| A | Exact but combinatorial | Number of subsets, `n choose B` |
| B | Polynomial independent scoring | Query/result membership counting |
| C | Exact form is exponential | Coalition enumeration |
| D | Greedy and polynomial | Repeated marginal-gain computation |

Exact Method A and exact Method C must include guardrails and should be presented as small-scale baselines, not scalable production methods.

## 12. Report and Submission Requirements

The Overleaf report must use the ACM SIG proceedings template and include:

- Problem statement and assignment context.
- Method D description, objective, algorithm, strengths, and limitations.
- Method A-D implementation summary.
- Experimental setup.
- Quality results.
- Runtime and memory results.
- Scalability discussion.
- Complexity comparison.
- Interpretation, limitations, and future work.
- Reproducibility notes, including Python version, dependency lockfile, random seeds, dataset handling, and hardware details.

Required sharing and submission:

- Share GitHub repository with `velgias@gmail.com`.
- Share Overleaf with `velgias@gmail.com` and `riccardo.torlone@uniroma3.it`.
- Send final email to `i.velegrakis@uu.nl` and `riccardo.torlone@uniroma3.it`.
- Email subject: `Roma3 Data Reduction`.
- Email body must include full name, matricola, GitHub link, and Overleaf link.

## 13. Acceptance Criteria

The project is complete when:

- Data validation rejects malformed inputs and records normalization diagnostics.
- Methods A-D are implemented behind a shared interface.
- Exact methods have clear infeasibility guardrails.
- All deterministic methods are reproducible.
- Experiment outputs are saved as tidy CSV plus optional JSON artifacts.
- Figures are generated from saved experiment outputs, not manual editing.
- Tests pass with `uv run pytest`.
- Ruff passes with `uv run ruff check .`.
- The report explains Method D, compares A-D, and discusses quality, runtime, memory, scalability, and theoretical complexity.
- A clean clone can reproduce the core experiment outputs using documented commands after adding the private dataset.

## 14. Implementation Decisions

Resolved decisions:

- Query result order is ignored. Query rows are treated as sets of returned photo IDs.
- Query IDs are normalized once in the data-loading layer.
- For A, C, and D, `q(D')` is approximated through the nearest retained photo under cosine similarity.
- For B, utility follows the assignment's Jaccard/precision-style IndepDF membership scoring.
- Method C uses the same cosine proxy utility as Method A for coalition values.
- Exact Method A and exact Method C are limited to tiny assignment-scale datasets.
- Method D uses query-frequency weights normalized over queried photos.
- Deterministic tie-breaking uses lower photo ID.

Decisions that may still be refined during implementation:

- Exact numeric guardrails for maximum combinations, Shapley coalitions, runtime, and memory.
- Whether Monte Carlo Shapley is worth adding after required experiments are complete.
- Whether stretch baselines include random, most-frequent, clustering medoids, or only a subset.
- Final experiment sample sizes after local hardware and dataset size are known.

## 15. References

- Yannis Velegrakis, Data Reduction Practical, 2026.
- Ramon Rico, Arno Siebes, and Yannis Velegrakis, "Stochastic Submodular Data Forgetting", SIGMOD 2026 reference paper.
- Lloyd S. Shapley, "A Value for n-Person Games", 1953.

## 16. Assumptions and Defaults

- The private dataset is not committed to git.
- `README.md` remains the broad overview; this PRD is the authoritative implementation spec.
- Optional research-fuller work must not block required assignment completion.
- The project favors simple, well-tested scripts over complex infrastructure.
- If an implementation choice is ambiguous, document the chosen default in the relevant code comments, experiment config, or report notes.
