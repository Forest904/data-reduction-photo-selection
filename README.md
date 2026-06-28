# Data Reduction for Query-Aware Photo Selection

Course project for the Data Reduction Practical. The goal is to decide which photos to keep on a storage-limited device while preserving usefulness for historical search queries.

Each photo is represented by an embedding vector in `photos.csv`. Historical searches are represented by `queries.csv`, where each row lists the photo IDs returned by one query.

> Deadline: June 15, 2026  
> Status: implementation, experiments, figures, and final report completed

## Project Docs

- [AGENT.md](AGENT.md): compact guidance for future coding agents working in this repo.
- [data/README.md](data/README.md): private dataset layout and validation contract.
- [docs/report_draft.md](docs/report_draft.md): Markdown companion to the final report.
- [docs/report/report.tex](docs/report/report.tex): polished LaTeX report source.
- [output/pdf/data_reduction_report.pdf](output/pdf/data_reduction_report.pdf): compiled final PDF report.
- [output/overleaf/README.txt](output/overleaf/README.txt): self-contained Overleaf export instructions.

## Methods

The project implements and compares four required selection methods:

| Method | Summary | Intended role |
| --- | --- | --- |
| A | Exhaustive search maximizing cosine-proxy utility | Exact baseline for tiny datasets |
| B | IndepDF ranking by normalized query-membership mass | Scalable query-log baseline |
| C | Exact Shapley-value ranking under cosine-proxy utility | Principled importance baseline for tiny datasets |
| D | Query-mass-weighted greedy facility location with clipped cosine coverage | Proposed scalable method |

All methods cap the effective budget at the dataset size. Ranking and greedy ties
are resolved by lower photo ID. Methods A and C return documented `skipped`
results when their configurable exact-computation guardrails are exceeded.
Method D evaluates candidates in memory-bounded chunks rather than materializing
a full all-pairs similarity matrix.

## Repository Shape

```text
data-reduction-photo-selection/
  README.md
  AGENT.md
  pyproject.toml
  uv.lock
  data/
  src/data_reduction/
  scripts/
  tests/
  experiments/
  docs/
```

## Private Dataset

The assignment dataset is private and is not committed to git. Place the files here before running validation or experiments:

```text
data/
  raw/
    photos.csv
    queries.csv
```

Both files are headerless CSVs. `photos.csv` contains one numeric embedding
vector per row. Each row of `queries.csv` contains the photo IDs returned by one
historical query. Raw-data commands default to one-based query IDs, matching the
assignment's line-number convention, and normalize them to zero-based internal
IDs during loading. Validation rejects malformed, missing, duplicate, empty, and
out-of-range data. See [data/README.md](data/README.md) for the concise contract.

## Clean Clone Instructions

Install `uv`, clone the repository, then run:

```bash
uv sync
```

Add the private dataset files:

```text
data/raw/photos.csv
data/raw/queries.csv
```

Validate data, run quality checks, and regenerate figures:

```bash
uv run python scripts/validate_data.py --id-base one
uv run pytest
uv run ruff check .
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
```

Optional experiment commands:

```bash
uv run python scripts/run_method.py --method D --budget 3
uv run python scripts/run_experiments.py --config experiments/configs/synthetic.yaml
uv run python scripts/run_experiments.py --config experiments/configs/small.yaml
uv run python scripts/run_experiments.py --config experiments/configs/scalability.yaml
uv run python scripts/run_experiments.py --config experiments/configs/budget_sensitivity.yaml
uv run python scripts/run_experiments.py --config experiments/configs/d_ablations.yaml
uv run python scripts/run_experiments.py --config experiments/configs/query_holdout.yaml
uv run python scripts/run_experiments.py --config experiments/configs/exact_infeasibility.yaml
```

The same workflows are available through the `Makefile`:

```bash
make check
make exp-all
make figures
```

## Experiment Artifacts

Saved result CSVs and diagnostics live under `experiments/results/`. Generated report figures live under `experiments/figures/`:

- `utility.png`
- `runtime.png`
- `memory.png`
- `scalability.png`
- `budget_sensitivity.png`
- `holdout_utility.png`

The report and figure generator use the seven canonical batches under
`experiments/results/`:

- `synthetic_canonical`
- `small_exact_comparison_canonical`
- `scalability_canonical`
- `budget_sensitivity_canonical`
- `d_ablations_canonical`
- `query_holdout_canonical`
- `exact_infeasibility_canonical`

The local private dataset summary is 41,620 photos, 2,048 embedding dimensions,
and 443 query rows. The full-data Method D result for budget 3 selects normalized
IDs `5974`, `39809`, and `24228`, corresponding to one-based assignment line IDs
`5975`, `39810`, and `24229`.

## Report Outputs

`docs/report/report.tex` is the canonical report source and reads figures from
`experiments/figures/`. The committed Overleaf export under `output/overleaf/`
contains the same report text with an export-local `figures/` path. When report
text or figures change, refresh both the Overleaf export and
`output/pdf/data_reduction_report.pdf`.

## Final Deliverables

- GitHub repository containing implementations, tests, experiment scripts, saved results, figures, and reproducibility notes.
- PDF report describing Method D and comparing Methods A-D.
- Generated figures and tables covering utility, runtime, memory, scalability,
  budget sensitivity, query holdout, and complexity.

## References

- Yannis Velegrakis, Data Reduction Practical, 2026.
- Ramon Rico, Arno Siebes, and Yannis Velegrakis, "Stochastic Submodular Data Forgetting", SIGMOD 2026.
- Lloyd S. Shapley, "A Value for n-Person Games", 1953.

## Author

- Luca Foresti
- Matricola: 565562
- Roma Tre
- luc.foresti@stud.uniroma3.it

## License

License to be confirmed after course and dataset requirements are clear. For a private assignment repository, this project may remain unlicensed unless instructed otherwise.
