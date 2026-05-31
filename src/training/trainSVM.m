function [svmModel, metrics] = trainSVM(X, labels)
%TRAINSVM Train an SVM classifier for fake news detection.
%
% Description:
%   [svmModel, metrics] = trainSVM(X, labels) splits a labeled TF-IDF
%   dataset into training and testing partitions, trains a linear SVM-style
%   classifier using fitclinear, evaluates it on the held-out test set, and
%   returns the trained model with core performance metrics.
%
% Inputs:
%   X      - Sparse numeric TF-IDF feature matrix where rows represent
%            articles and columns represent vocabulary features.
%   labels - Numeric or categorical binary class labels for each article:
%            0 = Fake
%            1 = Real
%
% Outputs:
%   svmModel - Trained linear classification model using the SVM learner.
%   metrics  - Structure containing:
%              metrics.accuracy
%              metrics.confusionMatrix
%              metrics.trainSampleCount
%              metrics.testSampleCount
%              metrics.predictions
%              metrics.trueLabels
%
% TODO:
%   - Add cross-validation for regularization strength selection.
%   - Add class weighting for imbalanced datasets.
%   - Save trained model artifacts under models/.
%   - Add precision, recall, and F1-score metrics.
%   - Add tests for invalid dimensions, categorical labels, and single-class
%     labels.

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

    if ~(isnumeric(labels) || islogical(labels) || iscategorical(labels))
        error("trainSVM:InvalidLabelType", ...
            "labels must be numeric, logical, or categorical binary values.");
    end

    labels = labels(:);

    if size(X, 1) ~= numel(labels)
        error("trainSVM:DimensionMismatch", ...
            "The number of rows in X (%d) must match the number of labels (%d).", ...
            size(X, 1), numel(labels));
    end

    % Convert labels to categorical for a stable classification contract.
    % Numeric/logical labels are validated before conversion so the expected
    % project mapping remains explicit: 0 = Fake, 1 = Real.
    if iscategorical(labels)
        labelCategories = categories(labels);

        if numel(labelCategories) ~= 2
            error("trainSVM:InvalidCategoricalLabels", ...
                "Categorical labels must contain exactly two categories representing 0 = Fake and 1 = Real.");
        end

        labels = categorical(labels);
    else
        labels = double(labels);

        if any(~isfinite(labels))
            error("trainSVM:InvalidLabelValues", ...
                "Numeric labels must not contain NaN or Inf values.");
        end

        validLabelValues = [0; 1];
        observedNumericLabels = unique(labels);

        if any(~ismember(observedNumericLabels, validLabelValues))
            error("trainSVM:InvalidLabelValues", ...
                "Numeric labels must contain only binary values: 0 = Fake, 1 = Real.");
        end

        labels = categorical(labels, validLabelValues, ["Fake", "Real"]);
    end

    observedLabelValues = categories(removecats(labels));

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

    %% Train Linear SVM-Style Classifier
    % fitclinear is preferred over fitcsvm for sparse, high-dimensional
    % TF-IDF data. It is designed for large linear classification problems
    % and accepts sparse predictor matrices directly, avoiding the memory
    % cost of converting X to a full matrix.
    svmModel = fitclinear( ...
        XTrain, ...
        yTrain, ...
        "Learner", "svm", ...
        "Regularization", "ridge", ...
        "Solver", "dual");

    %% Evaluate on Held-Out Test Set
    predictedLabels = predict(svmModel, XTest);

    accuracy = mean(predictedLabels == yTest);
    classNames = categories(labels);
    classOrder = categorical(classNames, classNames);
    confusionMatrix = confusionmat(yTest, predictedLabels, "Order", classOrder);

    metrics = struct();
    metrics.accuracy = accuracy;
    metrics.confusionMatrix = confusionMatrix;
    metrics.trainSampleCount = sum(trainIdx);
    metrics.testSampleCount = sum(testIdx);
    metrics.predictions = predictedLabels;
    metrics.trueLabels = yTest;

    %% Print Training Summary
    fprintf("\nSVM Training Summary\n");
    fprintf("--------------------\n");
    fprintf("Training samples: %d\n", metrics.trainSampleCount);
    fprintf("Testing samples : %d\n", metrics.testSampleCount);
    fprintf("Accuracy        : %.4f\n\n", accuracy);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add model training diagnostics and failure artifact capture.
    fprintf(2, "SVM training failed: %s\n", ME.message);
    rethrow(ME);
end

end
