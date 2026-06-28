# Query-Aware Photo Selection for Data Reduction

**Author:** Luca Foresti  
**Matricola:** 565562  
**Course:** Data Reduction, Roma Tre  
**Role:** Markdown companion to the canonical ACM-style LaTeX report

## Abstract

Smartphone photo collections grow quickly, but storage and search quality impose practical limits on how many images can be retained. This project studies query-aware photo selection: given photo embeddings and a historical query log, select a budgeted subset that preserves the usefulness of past search results. I implemented four methods required by the assignment: exhaustive cosine-based selection, IndepDF with Jaccard-style scoring, exact Shapley-value ranking, and a proposed query-frequency-weighted greedy facility-location method. The implementation includes shared loading, validation, utility, experiment, and plotting code. Experiments on the private dataset of 41,620 photos, 2,048-dimensional embeddings, and 443 query rows show the expected trade-off: exact methods are useful tiny-scale references but infeasible on the full collection, IndepDF is fastest and lightest, and Method D gives the strongest scalable cosine-proxy utility while using more runtime and memory. The repository provides reproducible configs, saved results, generated figures, tests, and lint checks.

## 1. Introduction

Global data production keeps increasing while personal devices still face concrete limits: storage, processing time, energy use, and the human cost of searching through large collections. The practical example in this assignment is a smartphone photo library. The phone is running out of space, so only a subset of photos can be kept. A naive deletion policy may save storage but damage future search quality, especially if deleted photos were historically useful answers to past queries.

The project problem is therefore to select retained photos under a fixed budget while preserving the utility of historical search results. Each photo is represented by an embedding vector, and the query log records which photo identifiers were returned by previous searches. Because the original query operators are not available, the implementation evaluates retained subsets through proxy utilities derived from the historical query results and photo embeddings.

This work makes three contributions. First, it implements the three required baselines: Method A, Method B, and Method C. Second, it introduces Method D, a query-frequency-weighted greedy facility-location strategy that combines workload awareness with embedding coverage. Third, it benchmarks all methods where feasible, comparing utility, runtime, memory, scalability, and theoretical complexity.

## 2. Problem Formalization

Let `D` be the full photo collection. Each photo `d in D` is represented as an embedding vector in `R^n`; in the local dataset, `n = 2048`. The file `photos.csv` contains one vector per row. The query log `queries.csv` contains one historical query result per row, represented as a variable-length list of photo identifiers.

The goal is to choose a subset `D' subset D` with `|D'| <= B`, where `B` is the maximum number of photos that can be kept. The ideal objective is to maximize expected query utility:

```text
D* = argmax_{D' subset D, |D'| <= B} E_{q in Q}[S(q(D), q(D'))].
```

Since only historical result identifiers are available, `q(D')` is approximated using the retained photos. For the embedding-aware methods, each original query-result photo is matched to the most similar retained photo under cosine similarity:

```text
S(q(D), D') = (1 / |q(D)|) * sum_{d in q(D)} max_{s in D'} cosine(d, s).
```

Method B also uses the assignment's Jaccard/precision-style query membership view through IndepDF scores. The implementation records both cosine-proxy utility and Jaccard/precision utility for cross-method comparison.

The data-loading layer normalizes query photo identifiers once, validates bounds
and malformed rows, rejects duplicate IDs within a query, and reports
diagnostics. The final raw-data experiments use one-based query IDs to match the
assignment's line-number convention.

## 3. Methodology

### 3.1 Method A: Exhaustive Cosine Search

Method A directly optimizes the cosine-proxy objective by enumerating all candidate subsets of size `B` and returning the subset with maximum utility. In this project, it is implemented as an exact exhaustive baseline for tiny sampled datasets, especially the assignment-literal case with `B=3`.

Its main advantage is interpretability: for a small enough candidate set, it gives the best subset under the chosen cosine proxy. Its main limitation is combinatorial growth. The number of candidate subsets is `binomial(n, B)`, so full-dataset execution is infeasible. The code therefore includes guardrails for maximum photo count and maximum combinations, returning a clean skipped result with diagnostics when the run is too large.

### 3.2 Method B: IndepDF Greedy Scoring

Method B implements IndepDF using the historical query log. For each photo `d`, the score is the empirical average of its normalized membership contribution:

```text
score(d) = E_{q in Q}[I_q(d) / |q(D)|].
```

The method selects the top `B` photos by score, with deterministic lower-ID tie-breaking. In the assignment description, this corresponds to the Jaccard-based simplification that avoids expensive subset search. In this repository, Method B is the scalable query-log baseline.

Method B is very fast and memory efficient because it only needs to count query memberships and sort scores. Its limitation is that it does not use embedding similarity, so it can favor popular photos without explicitly rewarding coverage or diversity in embedding space.

### 3.3 Method C: Shapley Value Ranking

Method C treats each photo as a player in a cooperative game and computes its exact Shapley value under the cosine-proxy utility function. It then selects the top-ranked photos, again using deterministic lower-ID tie-breaking.

This method gives a principled game-theoretic notion of individual contribution because it averages marginal utility over coalitions. However, exact Shapley computation is exponential in the number of photos. The final evidence runs exact Shapley on tiny sampled datasets for `B=3` and `B=4`; full-data Shapley remains guarded.

### 3.4 Method D: Query-Aware Greedy Facility Location

Method D is the proposed strategy. It combines Method B's query awareness with Method A's embedding awareness. First, it assigns each photo a query-derived weight `w_d` from normalized frequency in the historical query log. Photos that never appear in the query log receive weight zero. Then it greedily selects representatives that maximize weighted embedding coverage:

```text
F(S) = sum_{d in D} w_d * max_{s in S} max(0, cosine(d, s)).
```

The algorithm starts with an empty set. At each step it maintains every weighted target photo's best current similarity to the selected set, evaluates candidate marginal gains, and adds the candidate with the largest gain. It stops when `B` photos have been selected. Ties are broken by lower photo ID. The implementation uses float64 math and memory-aware chunking through `max_facility_similarity_mb`.

Expected strengths:

- It is query-aware because weights come from historical queries.
- It is embedding-aware because utility comes from cosine coverage.
- It encourages diversity because once a region is covered, near-duplicate candidates have smaller marginal gains.
- It scales polynomially and can run on the full dataset with chunking.

Limitations:

- It depends on historical queries being a useful proxy for future queries.
- It depends on embedding quality and cosine similarity being meaningful for photos.
- It is slower and heavier than IndepDF because it repeatedly evaluates similarity-based coverage.

## 4. Experimental Evaluation

### 4.1 Setup

The private dataset contains 41,620 photos, each represented by a 2,048-dimensional embedding vector, and 443 historical query rows. The raw CSV files are not committed to the repository. Experiment configs sample query-active subsets where needed and preserve deterministic seeds.

The final evidence batches are:

| Batch | Role | Result |
| --- | --- | --- |
| `synthetic_canonical` | Synthetic sanity checks | 22 success rows, 2 expected Shapley skips |
| `small_exact_comparison_canonical` | A-D exact-scale comparison | 96 success rows |
| `scalability_canonical` | B/D scalability | 10 success rows |
| `budget_sensitivity_canonical` | B/D budget sensitivity | 10 success rows |
| `d_ablations_canonical` | Method D ablations | 9 success rows |
| `query_holdout_canonical` | Held-out query robustness | 12 success rows |
| `exact_infeasibility_canonical` | Full-data exact skip documentation | 4 expected skips |

The recorded hardware environment is Python 3.12.2 on Windows 11, with 16 CPUs and an AMD64 processor. The figure artifacts are generated from saved CSV results using:

```bash
uv run python scripts/generate_figures.py --results experiments/results --output experiments/figures
```

Report figures:

- `../experiments/figures/utility.png`
- `../experiments/figures/runtime.png`
- `../experiments/figures/memory.png`
- `../experiments/figures/scalability.png`
- `../experiments/figures/budget_sensitivity.png`
- `../experiments/figures/holdout_utility.png`

### 4.2 Performance: Runtime and Memory

The measured runtimes match the theoretical expectations. Method B is the
fastest scalable method because it uses direct counting and sorting. In the
scalability batch, Method B averaged about `0.017` seconds and `0.482` MB peak
measured memory. Method D averaged about `11.205` seconds and `305.160` MB
because it computes repeated similarity coverage, but it remained feasible on
the full dataset through chunking.

On the small exact comparison, the average runtimes were:

| Method | Runtime seconds | Peak memory MB |
| --- | ---: | ---: |
| A | 0.110159 | 0.197561 |
| B | 0.000589 | 0.008232 |
| C | 1.167408 | 0.340381 |
| D | 0.002551 | 0.440582 |

Full-data exact Method A and exact Method C were skipped by design. Method A exceeds exhaustive-search limits, and Method C exceeds exact Shapley limits. These skipped results are part of the evidence, not failures.

### 4.3 Quality: Utility Preservation

On the small exact comparison, Methods A and D were nearly tied under cosine-proxy utility, while Method B and Method C remained close:

| Method | Mean cosine-proxy utility | Mean Jaccard/precision utility |
| --- | ---: | ---: |
| A | 0.654818 | 0.518673 |
| B | 0.642168 | 0.518673 |
| C | 0.643034 | 0.518673 |
| D | 0.654377 | 0.518673 |

On larger scalable runs, Method D achieved stronger cosine-proxy utility than Method B. In the scalability batch, Method B averaged `0.293034` cosine-proxy utility, while Method D averaged `0.366595`. In the budget-sensitivity batch, Method B averaged `0.315245`, while Method D averaged `0.391628`.

The Method D ablation batch supports the combined design. Full Method D averaged `0.422695` cosine-proxy utility, coverage-only averaged `0.421382`, and frequency-only averaged `0.352703`. Frequency-only had higher Jaccard/precision utility but lower embedding coverage, which is consistent with its popularity-only behavior.

The query-holdout batch trains on 75% of projected queries and evaluates on the
remaining 25%. Averaged over budgets and seeds, Method D reaches `0.399460`
held-out cosine-proxy utility, compared with `0.295733` for Method B. Because the
query log has no timestamps, this is a deterministic random holdout rather than
a temporal evaluation.

## 5. Discussion and Comparison

The methods show a clear cost-quality trade-off. Exact methods are valuable for small correctness and comparison settings, but they do not scale to the full dataset. IndepDF is the best choice when speed and memory dominate. Method D is the best mobile-deployment candidate when the device can afford an offline selection step and the goal is to preserve embedding-level search quality.

| Method | Utility behavior | Runtime and memory | Theoretical complexity | Scalability |
| --- | --- | --- | --- | --- |
| A | Exact optimum under cosine proxy on tiny samples | Quickly becomes infeasible | `O(binomial(n, B) * evaluation_cost)` | Tiny only |
| B | Strong query-frequency baseline, weaker embedding coverage | Fastest and lightest | Roughly linear in query memberships plus sorting | Full dataset |
| C | Principled individual contribution estimate | Exact version is expensive | Exponential in photo count | Tiny only |
| D | Strongest scalable cosine-proxy utility | Slower and heavier than B | Polynomial greedy marginal-gain computation | Full dataset with chunking |

The most important limitation is that the system uses historical query results as a proxy for future search utility. If future searches differ strongly from past searches, query-frequency weights can overfit old behavior. A second limitation is that exact Method A and exact Method C cannot be used as full-data baselines because their theoretical complexity is too high. Finally, all embedding-aware conclusions depend on cosine similarity over the provided vectors being a meaningful proxy for visual or semantic replacement quality.

## 6. Conclusion and References

This project implements and evaluates all required methods for query-aware photo reduction. The final recommendation is to use Method D when the objective is high-quality offline selection for a mobile device, because it balances popularity, diversity, and embedding-level representativeness. If selection must be extremely fast or memory-constrained, Method B is the practical fallback. Methods A and C should be retained as exact or principled tiny-scale baselines, not as deployable full-dataset methods.

Reproducibility is supported by `uv.lock`, deterministic experiment configs, saved result CSVs, diagnostics JSON files, generated figures, and tests. A clean clone can reproduce the checks after the private dataset is placed in `data/raw/`.

References to include in the ACM Overleaf version:

- Yannis Velegrakis. Data Reduction, 2026.
- Ramon Rico, Arno Siebes, and Yannis Velegrakis. "Stochastic Submodular Data Forgetting." SIGMOD 2026.
- Lloyd S. Shapley. "A Value for n-Person Games." 1953.
