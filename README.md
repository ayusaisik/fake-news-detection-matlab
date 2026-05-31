# Fake News Detection System

![MATLAB](https://img.shields.io/badge/MATLAB-R2024a-orange)
![Text Analytics Toolbox](https://img.shields.io/badge/Text%20Analytics-Toolbox-blue)
![Machine Learning](https://img.shields.io/badge/Classifier-Linear%20SVM-green)
![GUI](https://img.shields.io/badge/GUI-Programmatic%20MATLAB-lightgrey)

## Project Overview

**Fake News Detection System** is a MATLAB machine learning project that classifies news articles as **Fake** or **Real** using text preprocessing, TF-IDF feature extraction, and a Linear SVM classifier.

The project includes a complete training pipeline, model evaluation utilities, model save/load support, single-news prediction, and a programmatic MATLAB GUI for interactive use.

This system is a statistical text classifier. It does **not** perform real-time fact-checking and does not verify claims against the internet, news databases, or external sources.

## Key Features

- Load `Fake.csv` and `True.csv`
- Merge and label fake/real datasets
- Create combined article text from title and body
- Clean and preprocess raw text
- Extract sparse TF-IDF features
- Train a Linear SVM classifier using `fitclinear`
- Calculate accuracy
- Generate confusion matrix visualization
- Save and load trained model artifacts
- Predict a single news article
- Use a programmatic MATLAB GUI for prediction

## Technology Stack

- MATLAB R2024a
- Text Analytics Toolbox
- Statistics and Machine Learning Toolbox
- TF-IDF
- Linear SVM using `fitclinear`
- Programmatic MATLAB GUI using `uifigure`

## Dataset

This project uses the **Kaggle Fake and Real News Dataset**.

Expected dataset files:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Expected columns:

```text
title
text
```

The loader creates:

- `Label = 0` for fake news
- `Label = 1` for real news
- `FullText = title + " " + text`

Large dataset files may be excluded from GitHub depending on repository settings and `.gitignore` rules. If the raw CSV files are not included, download them from Kaggle and place them in `data/raw/`.

## Machine Learning Pipeline

```text
Fake.csv + True.csv
        |
        v
Load, Validate, Label, Merge
        |
        v
Create FullText
        |
        v
Preprocess Text
        |
        v
Tokenize Documents
        |
        v
Build Bag-of-Words Model
        |
        v
Extract TF-IDF Features
        |
        v
Align Labels With Valid Documents
        |
        v
Train Linear SVM with fitclinear
        |
        v
Evaluate Accuracy + Confusion Matrix
        |
        v
Save Model Artifacts
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
|   `-- fake_news_model.mat
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

Trained model files such as `models/fake_news_model.mat` may be excluded from GitHub if ignored by `.gitignore`. Run the training pipeline to regenerate them.

## Model Performance

Final model performance on the held-out test split:

| Metric | Value |
| --- | ---: |
| Training samples | 35,407 |
| Testing samples | 8,851 |
| Accuracy | approximately 99.14% |

Classes:

- `0 = Fake`
- `1 = Real`

If available, the confusion matrix image is saved at:

![Confusion Matrix](results/confusion_matrix.png)

## How to Run the Full Pipeline

1. Open MATLAB R2024a.
2. Set the current folder to the project root.
3. Place the Kaggle dataset files here:

```text
data/raw/Fake.csv
data/raw/True.csv
```

4. Run:

```matlab
run("main.m")
```

The pipeline will load the dataset, preprocess text, extract TF-IDF features, train the Linear SVM model, evaluate performance, save the model, and run sample predictions.

The saved model is written to:

```text
models/fake_news_model.mat
```

## How to Use the GUI

Run the full pipeline first so the trained model exists at:

```text
models/fake_news_model.mat
```

Then launch the GUI:

```matlab
run app/FakeNewsDetectionApp.m
```

The GUI loads the saved model using `loadModel()`. Enter a news article into the text area, click **Predict**, and the app will display:

- `FAKE NEWS`
- `REAL NEWS`
- Confidence score

Use **Clear** to reset the input and output fields.

## Example Prediction

Example input:

```text
Government officials announced a new public health policy after a verified press briefing.
```

Example output:

```text
Prediction Result
Predicted class : Real
Confidence score: 0.9824
```

Exact predictions and confidence values depend on the trained model artifact, vocabulary, and input text.

## Important Limitations

This model does **not** perform real-time fact-checking. It does not verify whether a news article is objectively true or false using the internet or external sources.

The classifier learns linguistic and statistical patterns from the training dataset. It performs best on full-length news articles that resemble the Kaggle dataset style.

Short inputs, headlines, social media posts, opinion snippets, very recent events, or text from very different domains may produce lower-confidence or less reliable predictions.

Confidence scores are margin-based signals from a Linear SVM-style classifier. They should not be interpreted as calibrated probabilities of truth.

## Future Improvements

- Add calibrated probability estimates
- Add cross-validation and hyperparameter search
- Add precision, recall, and F1-score tracking in the README
- Add automated unit tests for each module
- Add batch prediction support
- Add experiment tracking and dataset versioning
- Compare Linear SVM with logistic regression and transformer-based models
- Improve GUI with model metadata and input diagnostics
- Add support for exporting reports

## License

This project is provided for educational and portfolio purposes. Add a dedicated license file before distributing, publishing, or using the project in production.

