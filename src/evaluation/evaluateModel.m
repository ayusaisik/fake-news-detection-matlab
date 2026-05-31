function evaluationResults = evaluateModel(metrics)
%EVALUATEMODEL Evaluate binary fake news classification results.
%
% Description:
%   evaluationResults = evaluateModel(metrics) consumes the metrics
%   structure returned by trainSVM.m, computes binary classification
%   precision, recall, and F1-score, displays a clean evaluation summary,
%   creates a labeled confusion matrix chart, and saves the chart to
%   results/confusion_matrix.png.
%
% Inputs:
%   metrics - Structure returned by trainSVM.m. Required fields:
%             metrics.trueLabels
%             metrics.predictions
%             metrics.confusionMatrix
%             metrics.accuracy
%
% Outputs:
%   evaluationResults - Structure containing:
%                       evaluationResults.accuracy
%                       evaluationResults.precision
%                       evaluationResults.recall
%                       evaluationResults.f1Score
%                       evaluationResults.confusionMatrix
%
% Class Mapping:
%   0 = Fake
%   1 = Real
%
% TODO:
%   - Add macro and weighted metrics if multiclass support is introduced.
%   - Export evaluation metrics as JSON or MAT files under results/.
%   - Add automated tests for zero-division metric edge cases.

try
    %% Validate Input Structure
    if nargin < 1
        error("evaluateModel:MissingInput", "metrics input is required.");
    end

    if ~isstruct(metrics)
        error("evaluateModel:InvalidInputType", ...
            "metrics must be a struct returned by trainSVM.m.");
    end

    requiredFields = ["trueLabels", "predictions", "confusionMatrix", "accuracy"];
    validateRequiredFields(metrics, requiredFields);

    if isempty(metrics.trueLabels) || isempty(metrics.predictions)
        error("evaluateModel:EmptyLabels", ...
            "metrics.trueLabels and metrics.predictions must not be empty.");
    end

    if numel(metrics.trueLabels) ~= numel(metrics.predictions)
        error("evaluateModel:DimensionMismatch", ...
            "trueLabels count (%d) must match predictions count (%d).", ...
            numel(metrics.trueLabels), numel(metrics.predictions));
    end

    if isempty(metrics.confusionMatrix)
        error("evaluateModel:EmptyConfusionMatrix", ...
            "metrics.confusionMatrix must not be empty.");
    end

    %% Normalize Labels for Binary Metric Calculation
    % trainSVM.m may return categorical labels such as Fake/Real. Convert
    % supported label forms to numeric values so the positive class remains
    % explicit: 1 = Real.
    trueLabels = normalizeBinaryLabels(metrics.trueLabels, "trueLabels");
    predictions = normalizeBinaryLabels(metrics.predictions, "predictions");

    %% Compute Binary Classification Metrics
    % For fake news detection, Real is treated as the positive class here
    % because the project label mapping defines 1 = Real.
    positiveClass = 1;

    truePositive = sum(predictions == positiveClass & trueLabels == positiveClass);
    falsePositive = sum(predictions == positiveClass & trueLabels ~= positiveClass);
    falseNegative = sum(predictions ~= positiveClass & trueLabels == positiveClass);

    precision = safeDivide(truePositive, truePositive + falsePositive);
    recall = safeDivide(truePositive, truePositive + falseNegative);
    f1Score = safeDivide(2 * precision * recall, precision + recall);

    evaluationResults = struct();
    evaluationResults.accuracy = metrics.accuracy;
    evaluationResults.precision = precision;
    evaluationResults.recall = recall;
    evaluationResults.f1Score = f1Score;
    evaluationResults.confusionMatrix = metrics.confusionMatrix;

    %% Display Evaluation Summary
    fprintf("\nModel Evaluation Summary\n");
    fprintf("------------------------\n");
    fprintf("Accuracy : %.4f\n", evaluationResults.accuracy);
    fprintf("Precision: %.4f\n", evaluationResults.precision);
    fprintf("Recall   : %.4f\n", evaluationResults.recall);
    fprintf("F1-score : %.4f\n\n", evaluationResults.f1Score);

    %% Create and Save Confusion Matrix Chart
    projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
    resultsDir = fullfile(projectRoot, "results");

    if ~isfolder(resultsDir)
        mkdir(resultsDir);
    end

    confusionMatrixPath = fullfile(resultsDir, "confusion_matrix.png");

    figureHandle = figure("Name", "Fake News Confusion Matrix");
    confusionchart( ...
        metrics.confusionMatrix, ...
        ["Fake", "Real"], ...
        "Title", "Fake News Detection Confusion Matrix", ...
        "RowSummary", "row-normalized", ...
        "ColumnSummary", "column-normalized");

    exportgraphics(figureHandle, confusionMatrixPath, "Resolution", 300);
    fprintf("Confusion matrix saved to: %s\n\n", confusionMatrixPath);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured evaluation logging and report-generation recovery.
    fprintf(2, "Model evaluation failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateRequiredFields(inputStruct, requiredFields)
%VALIDATEREQUIREDFIELDS Ensure the metrics struct contains required fields.

availableFields = string(fieldnames(inputStruct));
missingFields = setdiff(requiredFields, availableFields);

if ~isempty(missingFields)
    error("evaluateModel:MissingRequiredField", ...
        "metrics is missing required field(s): %s", ...
        strjoin(missingFields, ", "));
end

end

function numericLabels = normalizeBinaryLabels(labels, labelName)
%NORMALIZEBINARYLABELS Convert supported binary labels to numeric 0/1 values.

labels = labels(:);

if iscategorical(labels)
    labelStrings = string(labels);
    numericLabels = nan(size(labelStrings));
    numericLabels(labelStrings == "Fake" | labelStrings == "0") = 0;
    numericLabels(labelStrings == "Real" | labelStrings == "1") = 1;
elseif islogical(labels)
    numericLabels = double(labels);
elseif isnumeric(labels)
    numericLabels = double(labels);
else
    error("evaluateModel:InvalidLabelType", ...
        "metrics.%s must contain numeric, logical, or categorical labels.", ...
        labelName);
end

if any(~isfinite(numericLabels)) || any(~ismember(unique(numericLabels), [0; 1]))
    error("evaluateModel:InvalidLabelValues", ...
        "metrics.%s must contain only binary labels where 0 = Fake and 1 = Real.", ...
        labelName);
end

end

function value = safeDivide(numerator, denominator)
%SAFEDIVIDE Return 0 for undefined binary metrics instead of NaN or Inf.

if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end

end
