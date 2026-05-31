# Fake News Detection System

MATLAB R2024a project skeleton for building a fake news detection system using text analytics, TF-IDF features, and Support Vector Machine (SVM) classification.

## Project Overview

The Fake News Detection System is designed as a modular MATLAB machine learning project. The future implementation will load labeled news articles, clean and normalize article text, extract TF-IDF features, train an SVM classifier, evaluate classification quality, and predict whether new article text is likely real or fake.

This repository is intentionally structured for maintainability, reproducibility, and extension.

## Features

- Text preprocessing pipeline for news article content
- TF-IDF feature extraction using Text Analytics Toolbox workflows
- SVM classifier training using Statistics and Machine Learning Toolbox
- Evaluation utilities for accuracy, confusion matrix, precision, recall, and F1-score
- Prediction module for classifying new news text
- Clean folder structure for data, models, reports, results, tests, and application assets

## Folder Structure

```text
fake-news-detection-matlab/
|-- app/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- reports/
|-- results/
|-- src/
|   |-- evaluation/
|   |   `-- evaluateModel.m
|   |-- feature_extraction/
|   |   `-- extractTFIDF.m
|   |-- prediction/
|   |   `-- predictNews.m
|   |-- preprocessing/
|   |   `-- preprocessText.m
|   `-- training/
|       `-- trainSVM.m
|-- tests/
|-- .gitignore
|-- main.m
|-- README.md
`-- requirements.md
```

## Installation

1. Install MATLAB R2024a.
2. Install the required toolboxes listed in `requirements.md`.
3. Clone or download this repository.
4. Open MATLAB and set the current folder to the repository root.
5. Run `main.m` to review the planned pipeline.

```matlab
cd fake-news-detection-matlab
run("main.m")
```

## Dataset Information

Place raw datasets in `data/raw/`. A future implementation should support labeled news datasets with at least:

- Article text or title/body fields
- Binary class label such as `fake` / `real` or `0` / `1`
- Optional metadata such as source, author, publication date, and topic

Processed datasets should be written to `data/processed/`.

## Usage

The intended workflow is:

1. Load labeled news data from `data/raw/`.
2. Preprocess article text with `src/preprocessing/preprocessText.m`.
3. Extract TF-IDF features with `src/feature_extraction/extractTFIDF.m`.
4. Train an SVM model with `src/training/trainSVM.m`.
5. Evaluate model performance with `src/evaluation/evaluateModel.m`.
6. Predict labels for new articles with `src/prediction/predictNews.m`.

## Roadmap

- Add dataset loading and validation utilities
- Implement robust tokenization, stop-word removal, lemmatization, and normalization
- Add TF-IDF vocabulary persistence for repeatable inference
- Implement SVM hyperparameter tuning and cross-validation
- Add model persistence in `models/`
- Add evaluation reports and plots in `reports/`
- Add automated tests in `tests/`
- Build a MATLAB App Designer interface in `app/`

