%% Fake News Detection System - End-to-End Pipeline
% MATLAB Version: R2024a
%
% Description:
%   This script runs the complete fake news detection workflow:
%   dataset loading, text preprocessing, TF-IDF feature extraction, sparse
%   linear SVM-style training, evaluation, model persistence, and sample
%   predictions.
%
% Required input files:
%   data/raw/Fake.csv
%   data/raw/True.csv

%% Environment Setup
% Clear the workspace and command window for a reproducible script run.
clear;
clc;

% Add all project folders so modular functions under src/ are discoverable.
addpath(genpath(pwd));

pipelineTimer = tic;

fprintf("\n");
fprintf("============================================================\n");
fprintf(" Fake News Detection System\n");
fprintf(" MATLAB R2024a | TF-IDF + Sparse Linear SVM Pipeline\n");
fprintf("============================================================\n\n");

try
    %% 1. Load Dataset
    % loadDataset reads Fake.csv and True.csv, validates required columns,
    % creates FullText, assigns labels, merges, shuffles, and summarizes the
    % data.
    fprintf("Stage 1/8: Loading dataset...\n");
    dataset = loadDataset();

    %% 2. Preprocess Text
    % preprocessText applies the same cleaning policy used later during
    % inference: lowercase normalization, URL removal, punctuation/number
    % cleanup, tokenization, stop-word removal, and text reconstruction.
    fprintf("Stage 2/8: Preprocessing text...\n");
    processedText = preprocessText(dataset.FullText);

    %% 3. Extract TF-IDF Features
    % extractTFIDF converts processed documents into sparse TF-IDF features
    % and returns validDocumentMask so feature rows stay aligned with labels.
    fprintf("Stage 3/8: Extracting TF-IDF features...\n");
    [X, tfidfModel, validDocumentMask] = extractTFIDF(processedText);

    %% 4. Align Labels
    % Some documents may be removed during feature extraction when they are
    % empty or contain only pruned vocabulary. Apply the returned mask before
    % training to preserve row alignment.
    fprintf("Stage 4/8: Aligning labels with valid TF-IDF rows...\n");
    labels = dataset.Label(validDocumentMask);

    if size(X, 1) ~= numel(labels)
        error("main:FeatureLabelMismatch", ...
            "TF-IDF rows (%d) do not match aligned labels (%d).", ...
            size(X, 1), numel(labels));
    end

    fprintf("Aligned samples: %d\n\n", numel(labels));

    %% 5. Train Model
    % trainSVM uses cvpartition with a holdout split and fitclinear with the
    % SVM learner, which is appropriate for sparse high-dimensional text
    % features.
    fprintf("Stage 5/8: Training sparse linear SVM-style model...\n");
    [svmModel, metrics] = trainSVM(X, labels);

    %% 6. Evaluate Model
    % evaluateModel computes precision, recall, F1-score, displays a
    % confusion chart, and saves results/confusion_matrix.png.
    fprintf("Stage 6/8: Evaluating model...\n");
    evaluationResults = evaluateModel(metrics);

    %% 7. Save Model
    % saveModel persists the trained classifier, TF-IDF vocabulary model,
    % training metrics, evaluation results, and save timestamp.
    fprintf("Stage 7/8: Saving model artifacts...\n");
    saveModel(svmModel, tfidfModel, metrics, evaluationResults);

    %% 8. Run Sample Predictions
    % These examples demonstrate the inference API. Real application code
    % can load the saved model with loadModel and call predictNews directly.
    fprintf("Stage 8/8: Running sample predictions...\n");

    sampleNewsTexts = [
        "Government officials announced a new public health policy after a verified press briefing."
        "Scientists confirm that drinking one glass of miracle water cures every disease overnight."
        "The national election commission released audited vote totals after independent review."
    ];

    for sampleIndex = 1:numel(sampleNewsTexts)
        fprintf("Sample Prediction %d\n", sampleIndex);
        fprintf("-------------------\n");
        fprintf("Input: %s\n", sampleNewsTexts(sampleIndex));

        [predictedLabel, confidenceScore] = predictNews( ...
            sampleNewsTexts(sampleIndex), ...
            svmModel, ...
            tfidfModel);

        if predictedLabel == 1
            predictedClass = "Real";
        else
            predictedClass = "Fake";
        end

        fprintf("Result: %s (%.4f)\n\n", predictedClass, confidenceScore);
    end

    %% Completion Summary
    elapsedSeconds = toc(pipelineTimer);

    fprintf("============================================================\n");
    fprintf(" Pipeline completed successfully.\n");
    fprintf(" Total runtime: %.2f seconds\n", elapsedSeconds);
    fprintf("============================================================\n\n");

catch ME
    %% Error Handling
    % Keep top-level error handling centralized so failures are visible in
    % the command window while preserving the original MATLAB stack trace.
    elapsedSeconds = toc(pipelineTimer);

    fprintf(2, "\n============================================================\n");
    fprintf(2, " Pipeline failed after %.2f seconds.\n", elapsedSeconds);
    fprintf(2, " Error ID: %s\n", ME.identifier);
    fprintf(2, " Message : %s\n", ME.message);
    fprintf(2, "============================================================\n\n");

    rethrow(ME);
end

