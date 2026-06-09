# Data Reduction for Query-Aware Photo Selection

Course project for the Data Reduction Practical. The goal is to decide which photos to keep on a storage-limited device while preserving usefulness for historical search queries.

Each photo is represented by an embedding vector in `photos.csv`. Historical searches are represented by `queries.csv`, where each row lists the photo IDs returned by one query.

> Deadline: June 15, 2026  
> Status: documentation scaffold; implementation not started yet

## Project Docs

- [PRD.md](PRD.md): authoritative requirements, method definitions, interfaces, experiment design, and acceptance criteria.
- [Roadmap.md](Roadmap.md): milestone-by-milestone implementation path from scaffold to submission.
- [AGENT.md](AGENT.md): compact guidance for future coding agents working in this repo.
- [Data_Reduction_Lecture_Summary_and_Full_Assignment.md](Data_Reduction_Lecture_Summary_and_Full_Assignment.md): lecture summary and assignment transcription used as project source material.

## Methods

The project will implement and compare four required selection methods:

| Method | Summary | Intended role |
| --- | --- | --- |
| A | Exhaustive cosine-based subset selection | Exact baseline for tiny datasets |
| B | IndepDF with Jaccard/precision-style query scoring | Scalable query-log baseline |
| C | Shapley-value-based photo ranking | Principled importance baseline for tiny datasets |
| D | Query-aware greedy facility-location with cosine coverage | Proposed method |

Method D is the original project proposal. It weights photos by query frequency and greedily selects photos that maximize cosine coverage, balancing popularity and diversity.

## Expected Dataset Layout

The assignment dataset is private and should not be committed.

```text
data/
  raw/
    photos.csv
    queries.csv
  processed/
```

`photos.csv` contains one embedding vector per row. `queries.csv` contains one historical query result set per row. Photo ID normalization and validation rules are specified in [PRD.md](PRD.md).

## Planned Tech Stack

- Python 3.12+
- `uv`
- NumPy
- pandas
- SciPy and/or scikit-learn
- Matplotlib
- PyYAML
- pytest
- Ruff

## Planned Quick Start

These commands describe the intended workflow once the implementation scaffold exists:

```bash
uv sync
uv run python scripts/validate_data.py
uv run python scripts/run_method.py --method D --budget 3
uv run python scripts/run_experiments.py --config experiments/configs/small.yaml
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
uv run pytest
uv run ruff check .
```

Until the package scaffold is created, use [Roadmap.md](Roadmap.md) as the implementation sequence.

## Planned Repository Shape

```text
data-reduction-photo-selection/
  README.md
  PRD.md
  Roadmap.md
  AGENT.md
  Data_Reduction_Lecture_Summary_and_Full_Assignment.md
  data/
  src/data_reduction/
  scripts/
  tests/
  experiments/
  docs/
```

## Final Deliverables

- GitHub repository containing implementations, tests, experiment scripts, results, and reproducibility notes.
- ACM-style Overleaf report describing Method D and comparing Methods A-D.
- Generated figures and tables covering utility, runtime, memory, scalability, and complexity.
- Submission email with full name, matricola, GitHub link, and Overleaf link.

## References

- Yannis Velegrakis, Data Reduction Practical, 2026.
- Ramon Rico, Arno Siebes, and Yannis Velegrakis, "Stochastic Submodular Data Forgetting", SIGMOD 2026 reference paper.
- Lloyd S. Shapley, "A Value for n-Person Games", 1953.

## Author

- Luca Foresti
- Matricola: 565562
- Roma Tre
- luc.foresti@stud.uniroma3.it

## License

License to be confirmed after course and dataset requirements are clear. For a private assignment repository, this project may remain unlicensed unless instructed otherwise.
