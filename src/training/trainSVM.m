function [svmModel, metrics] = trainSVM(X, labels)
%TRAINSVM Train an SVM classifier for fake news detection.
%
% Description:
%   [svmModel, metrics] = trainSVM(X, labels) splits a labeled TF-IDF
%   dataset into training and testing partitions, trains a linear Support
%   Vector Machine classifier, evaluates it on the held-out test set, and
%   returns the trained model with core performance metrics.
%
% Inputs:
%   X      - Numeric TF-IDF feature matrix where rows represent articles and
%            columns represent vocabulary features.
%   labels - Binary class labels for each article:
%            0 = Fake
%            1 = Real
%
% Outputs:
%   svmModel - Trained linear SVM classification model.
%   metrics  - Structure containing:
%              metrics.accuracy
%              metrics.confusionMatrix
%
% TODO:
%   - Add cross-validation for hyperparameter selection.
%   - Add class weighting for imbalanced datasets.
%   - Save trained model artifacts under models/.
%   - Add precision, recall, and F1-score metrics.
%   - Add tests for invalid dimensions and single-class labels.

try
    if nargin < 2
        error("trainSVM:MissingInput", "X and labels inputs are required.");
    end

    %% Validate Feature Matrix
    if isempty(X)
        error("trainSVM:EmptyFeatures", "X must not be empty.");
    end

    if ~isnumeric(X)
        error("trainSVM:InvalidFeatureType", ...
            "X must be a numeric TF-IDF feature matrix.");
    end

    if issparse(X)
        featureValues = nonzeros(X);
    else
        featureValues = X(:);
    end

    if any(~isfinite(featureValues))
        error("trainSVM:InvalidFeatureValues", ...
            "X must not contain NaN or Inf values.");
    end

    %% Validate Labels
    if isempty(labels)
        error("trainSVM:EmptyLabels", "labels must not be empty.");
    end

    if ~(isnumeric(labels) || islogical(labels))
        error("trainSVM:InvalidLabelType", ...
            "labels must be numeric or logical binary values.");
    end

    labels = labels(:);

    if size(X, 1) ~= numel(labels)
        error("trainSVM:DimensionMismatch", ...
            "The number of rows in X (%d) must match the number of labels (%d).", ...
            size(X, 1), numel(labels));
    end

    labels = double(labels);

    validLabelValues = [0; 1];
    observedLabelValues = unique(labels);

    if any(~ismember(observedLabelValues, validLabelValues))
        error("trainSVM:InvalidLabelValues", ...
            "labels must contain only binary values: 0 = Fake, 1 = Real.");
    end

    if numel(observedLabelValues) < 2
        error("trainSVM:SingleClassLabels", ...
            "Both classes are required to train an SVM classifier.");
    end

    %% Split Dataset: 80% Training, 20% Testing
    % cvpartition with labels preserves class proportions as much as
    % possible, which is important for fake/real classification.
    holdoutRatio = 0.20;
    partition = cvpartition(labels, "HoldOut", holdoutRatio);

    trainIdx = training(partition);
    testIdx = test(partition);

    XTrain = X(trainIdx, :);
    yTrain = labels(trainIdx);
    XTest = X(testIdx, :);
    yTest = labels(testIdx);

    %% Train Linear SVM
    % Standardization centers and scales features using the training data,
    % which improves numerical stability for SVM optimization.
    svmModel = fitcsvm( ...
        XTrain, ...
        yTrain, ...
        "KernelFunction", "linear", ...
        "Standardize", true, ...
        "ClassNames", [0; 1]);

    %% Evaluate on Held-Out Test Set
    predictedLabels = predict(svmModel, XTest);

    accuracy = mean(predictedLabels == yTest);
    confusionMatrix = confusionmat(yTest, predictedLabels, "Order", [0; 1]);

    metrics = struct();
    metrics.accuracy = accuracy;
    metrics.confusionMatrix = confusionMatrix;

    %% Print Training Summary
    fprintf("\nSVM Training Summary\n");
    fprintf("--------------------\n");
    fprintf("Training samples: %d\n", sum(trainIdx));
    fprintf("Testing samples : %d\n", sum(testIdx));
    fprintf("Accuracy        : %.4f\n\n", accuracy);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add model training diagnostics and failure artifact capture.
    fprintf(2, "SVM training failed: %s\n", ME.message);
    rethrow(ME);
end

end
