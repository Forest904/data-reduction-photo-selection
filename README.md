# Data Reduction for Query-Aware Photo Selection

Course project for the Data Reduction Practical. The goal is to decide which photos to keep on a storage-limited device while preserving usefulness for historical search queries.

Each photo is represented by an embedding vector in `photos.csv`. Historical searches are represented by `queries.csv`, where each row lists the photo IDs returned by one query.

> Deadline: June 15, 2026  
> Status: implementation, experiments, figures, and final report completed

## Project Docs

- [AGENT.md](AGENT.md): compact guidance for future coding agents working in this repo.
- [docs/report/report.tex](docs/report/report.tex): polished LaTeX report source.
- [output/pdf/data_reduction_report.pdf](output/pdf/data_reduction_report.pdf): compiled final PDF report.

## Methods

The project implements and compares four required selection methods:

| Method | Summary | Intended role |
| --- | --- | --- |
| A | Exhaustive cosine-based subset selection | Exact baseline for tiny datasets |
| B | IndepDF with Jaccard/precision-style query scoring | Scalable query-log baseline |
| C | Exact Shapley-value-based photo ranking | Principled importance baseline for tiny datasets |
| D | Query-aware greedy facility location with cosine coverage | Proposed scalable method |

Method D is the original project proposal. It weights photos by query frequency and greedily selects representatives that maximize clipped cosine coverage, balancing popularity and diversity.

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

`photos.csv` contains one embedding vector per row. `queries.csv` contains one historical query result set per row. Photo ID normalization and validation rules are implemented in `src/data_reduction/data.py` and summarized in this README and the final report.

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

Latest report evidence batches are saved under `experiments/results/`, including synthetic smoke runs, small exact comparisons, scalability, budget sensitivity, Method D ablations, query-holdout robustness, and exact-method infeasibility checks. The local private dataset summary is 41,620 photos, 2,048 embedding dimensions, and 443 query rows. Raw-data experiments use one-based query IDs to match the assignment's line-number convention.

## Final Deliverables

- GitHub repository containing implementations, tests, experiment scripts, saved results, figures, and reproducibility notes.
- PDF report describing Method D and comparing Methods A-D.
- Generated figures and tables covering utility, runtime, memory, scalability, and complexity.

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
