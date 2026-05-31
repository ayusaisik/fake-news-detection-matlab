# Requirements

## MATLAB Version

- MATLAB R2024a

## Required Toolboxes

- Text Analytics Toolbox
- Statistics and Machine Learning Toolbox

## Recommended Knowledge

- MATLAB table and datastore workflows
- Text preprocessing and tokenization
- TF-IDF feature engineering
- Supervised classification
- SVM model training and evaluation

## Expected Dataset Format

The future implementation should use a labeled news dataset with columns similar to:

| Column | Description |
| --- | --- |
| `text` | Full article text, title, or combined title/body content |
| `label` | Ground-truth class label, such as `fake` or `real` |

Additional metadata columns may be included and ignored or used during analysis.

## Development Notes

- Keep source code modular under `src/`.
- Store raw datasets under `data/raw/`.
- Store cleaned or transformed data under `data/processed/`.
- Store trained models under `models/`.
- Store generated figures, metrics, and outputs under `results/` or `reports/`.
- Add tests under `tests/` as functionality is implemented.

## TODO

- Define the canonical dataset schema.
- Add a data validation checklist.
- Document model performance acceptance criteria.
- Document reproducibility settings such as random seeds and train/test split strategy.

