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
- Save figures from result files; do not manually edit result data.
- If docs conflict, follow the implemented code, tests, saved experiment artifacts, and final report; update README for entrypoint-level workflow changes.

## Method Reminders

- A: exhaustive cosine baseline over feasible small combinations.
- B: IndepDF/Jaccard-style score, `E[I_q(d) / |q(D)|]`, then top `B`.
- C: Shapley values, exact for tiny assignment setting; approximation is stretch.
- D: query-aware greedy facility-location with query-frequency weights and cosine coverage.

## Quality Commands

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
