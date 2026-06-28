# AGENT.md

This project solves the Data Reduction Practical: given photo embeddings in `photos.csv` and historical query results in `queries.csv`, select which photos to keep under budget `B` while preserving query usefulness. Methods A-D, experiment scripts, figures, and the ACM-style report are implemented. `README.md` is the entrypoint overview, and the final report records the completed project rationale and results.

## Non-Negotiables

- Preserve reproducibility with fixed seeds, configs, saved outputs, and documented commands.
- Keep private dataset files out of git.
- Centralize photo ID normalization in data loading only.
- Do not let methods reinterpret query IDs independently.
- Prefer vectorized NumPy for similarity and utility work.
- Keep exact Method A and exact Method C restricted to tiny datasets with guardrails.
- Tie-break deterministically, preferably by lower photo ID.
- Treat code, tests, experiment configs, canonical result batches, and the
  canonical LaTeX report as the documentation sources of truth.
- Keep `README.md`, `data/README.md`, `docs/report_draft.md`, and
  `docs/report/report.tex` consistent when workflows, data rules, methods, or
  evidence change.
- After report or figure changes, refresh `output/overleaf/` and rebuild
  `output/pdf/data_reduction_report.pdf`.
- Save figures from result files; do not manually edit result data or generated
  figure content.

## Method Reminders

- A: exhaustive cosine-proxy baseline over feasible small combinations.
- B: IndepDF/Jaccard-style score, `E[I_q(d) / |q(D)|]`, then top `B`.
- C: Shapley values, exact for tiny assignment setting; approximation is stretch.
- D: query-mass-weighted greedy facility location with clipped cosine coverage
  and memory-bounded candidate chunks.

## Quality Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
```
