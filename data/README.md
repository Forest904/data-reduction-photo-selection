# Data Directory

Place the private assignment CSV files here:

```text
data/raw/photos.csv
data/raw/queries.csv
```

The raw dataset is private and should not be committed. The project ignores
`data/raw/*.csv` while keeping this directory structure tracked.

Validate the local dataset with:

```bash
uv run python scripts/validate_data.py
```
