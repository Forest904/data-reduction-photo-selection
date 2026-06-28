# Data Directory

Place the private assignment CSV files here:

```text
data/raw/photos.csv
data/raw/queries.csv
```

Both files are headerless:

- `photos.csv`: one photo embedding per row; every value must be numeric and
  every row must have the same embedding dimension.
- `queries.csv`: one historical query result per row; each field is a photo ID
  and rows may have different lengths.

The raw dataset is private and must not be committed. The project ignores
`data/raw/*.csv` while keeping the directory structure tracked.

Assignment query IDs use one-based `photos.csv` line numbers. The command-line
tools therefore default to `--id-base one` and normalize IDs to zero-based
indexes inside the loader. `--id-base zero` and `--id-base auto` are available
for other datasets.

Validate the local dataset with:

```bash
uv run python scripts/validate_data.py --id-base one
```

Validation rejects missing files, empty or malformed CSVs, nonnumeric photo
values, missing values, empty query rows, noninteger query IDs, duplicate IDs
within a query, and IDs outside the photo matrix. Zero photo vectors are
reported as warnings.
