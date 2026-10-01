# SENTINEL Data

Datasets are intentionally excluded from Git.

## Directory structure

- `raw/` - original downloaded datasets
- `interim/` - intermediate transformations
- `processed/` - model-ready datasets

## Rules

1. Do not commit raw datasets.
2. Do not commit processed datasets.
3. Do not upload datasets to GitHub.
4. Each dataset must have a documented source.
5. Record download date, source URL, row count and SHA-256 checksum.
6. Use the dataset's original licensing/terms.
7. Do not replace inaccessible real-world datasets with synthetic data.

Dataset metadata should be recorded in:

`reports/data_manifest.json`
