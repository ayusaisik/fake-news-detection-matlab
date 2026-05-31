# Fake News Detection System

![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-orange)
![Text Analytics](https://img.shields.io/badge/Text%20Analytics-Toolbox-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20SVM-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-brightgreen)

## Project Overview

The **Fake News Detection System** is a MATLAB-based natural language processing and machine learning project that classifies news articles as **Fake** or **Real** using statistical language patterns.

The system uses text preprocessing, TF-IDF feature extraction, and a sparse linear SVM-style classifier trained with MATLAB R2024a. It also includes a programmatic MATLAB GUI for interactive predictions.

Important: this project does **not** perform real-time fact-checking. The model detects patterns learned from historical training data and should be treated as a statistical classifier, not an authority on truth.

## Key Features

- End-to-end fake news classification pipeline
- Dataset loading from separate fake and real CSV files
- Text cleaning and preprocessing
- TF-IDF feature extraction
- Sparse high-dimensional linear SVM-style classification
- Train/test evaluation with accuracy and confusion matrix
- Model persistence with save/load utilities
- Programmatic MATLAB GUI, no `.mlapp` file required
- Sample prediction workflow

## Technology Stack

- MATLAB R2024a
- Text Analytics Toolbox
- Statistics and Machine Learning Toolbox
- TF-IDF feature engineering
- Linear SVM-style classifier using `fitclinear`
- MATLAB programmatic GUI using `uifigure`

## Dataset

This project uses the **Kaggle Fake and Real News Dataset**.

Expected raw files:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Required columns:

```text
title
text
```

During loading, the system creates:

- `Label = 0` for fake news
- `Label = 1` for real news
- `FullText = title + " " + text`

## Machine Learning Pipeline

```text
Raw Dataset
    |
    v
Load Fake.csv and True.csv
    |
    v
Create Labels and FullText
    |
    v
Preprocess Text
    |
    v
Tokenize and Remove Stop Words
    |
    v
Extract TF-IDF Features
    |
    v
Align Valid Documents and Labels
    |
    v
Train Sparse Linear SVM
    |
    v
Evaluate Model
    |
    v
Save Model
    |
    v
Predict New News Text
```

## Project Structure

```text
fake-news-detection-matlab/
|-- app/
|   `-- FakeNewsDetectionApp.m
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- reports/
|-- results/
|   `-- confusion_matrix.png
|-- src/
|   |-- evaluation/
|   |   `-- evaluateModel.m
|   |-- feature_extraction/
|   |   `-- extractTFIDF.m
|   |-- prediction/
|   |   `-- predictNews.m
|   |-- preprocessing/
|   |   |-- loadDataset.m
|   |   `-- preprocessText.m
|   `-- training/
|       |-- loadModel.m
|       |-- saveModel.m
|       `-- trainSVM.m
|-- tests/
|-- .gitignore
|-- main.m
|-- README.md
`-- requirements.md
```

## Model Performance

Final reported model results:

| Metric | Value |
| --- | ---: |
| Training samples | 35,407 |
| Testing samples | 8,851 |
| Accuracy | 99.14% |

The model performs strongly on the held-out test split from the Kaggle dataset. Performance may vary on newer articles, out-of-domain sources, or adversarially written content.

## Confusion Matrix

Class mapping:

- `0 = Fake`
- `1 = Real`

Confusion matrix:

```text
[[4536,   32],
 [  51, 4232]]
```

![Confusion Matrix](results/confusion_matrix.png)

## How to Run

1. Open MATLAB R2024a.
2. Set the current folder to the project root.
3. Place the dataset files in `data/raw/`:

```text
data/raw/Fake.csv
data/raw/True.csv
```

4. Run the full pipeline:

```matlab
run("main.m")
```

The pipeline will:

- Load and validate the dataset
- Preprocess text
- Extract TF-IDF features
- Train the model
- Evaluate performance
- Save the trained model to `models/fake_news_model.mat`
- Run sample predictions

## How to Use GUI

After training and saving the model, launch the programmatic GUI:

```matlab
run app/FakeNewsDetectionApp.m
```

In the GUI:

1. Enter or paste news text into the text area.
2. Click **Predict**.
3. View the predicted class:
   - `FAKE NEWS`
   - `REAL NEWS`
4. View the confidence score.
5. Click **Clear** to reset the input and output fields.

## Example Prediction

Example input:

```text
Government officials announced a new public health policy after a verified press briefing.
```

Example output:

```text
Prediction Result
Predicted Class : Real
Confidence Score: 98.42%
```

The exact confidence score may vary depending on the trained model artifact and feature vocabulary.

## Limitations

- The model is based on statistical language patterns, not real-time fact-checking.
- It does not verify claims against live sources or trusted databases.
- It may perform poorly on topics, writing styles, or news sources not represented in the training data.
- It can be sensitive to dataset bias.
- Confidence scores are probability-like and should not be interpreted as calibrated truth probabilities.
- Highly edited, paraphrased, or adversarial text may reduce prediction reliability.

## Future Improvements

- Add model calibration for more reliable confidence scores
- Add cross-validation and hyperparameter tuning
- Add precision, recall, and F1-score tracking across multiple runs
- Add support for n-grams and alternative feature extraction strategies
- Compare linear SVM with logistic regression, naive Bayes, and transformer-based models
- Add automated tests for each pipeline stage
- Add dataset versioning and experiment tracking
- Add exportable reports for model evaluation
- Improve GUI with model metadata and batch prediction support

## License

This project is provided for educational and portfolio purposes. Add a project-specific license file before distributing or reusing the code in production.

