function [svmModel, tfidfModel, metrics, evaluationResults] = loadModel(modelPath)
%LOADMODEL Load saved fake news detection model artifacts.
%
% Description:
%   [svmModel, tfidfModel, metrics, evaluationResults] =
%   loadModel(modelPath) loads a saved model artifact created by
%   saveModel.m and validates that all required variables are present.
%
% Inputs:
%   modelPath - Optional string scalar or character vector path to the MAT
%               file. Default: models/fake_news_model.mat
%
% Outputs:
%   svmModel          - Trained classifier.
%   tfidfModel        - TF-IDF bag-of-words model.
%   metrics           - Training metrics structure.
%   evaluationResults - Evaluation metrics structure.
%
% TODO:
%   - Validate saved model version once version metadata is introduced.
%   - Add compatibility checks for MATLAB/toolbox versions.
%   - Add optional loading of auxiliary metadata.

try
    %% Resolve Model Path
    if nargin < 1 || isempty(modelPath)
        modelPath = defaultModelPath();
    else
        modelPath = string(modelPath);

        if ~isscalar(modelPath) || strlength(strtrim(modelPath)) == 0
            error("loadModel:InvalidModelPath", ...
                "modelPath must be a non-empty string scalar or character vector.");
        end

        modelPath = char(modelPath);
    end

    %% Validate File Exists
    if ~isfile(modelPath)
        error("loadModel:FileNotFound", ...
            "Model file was not found at: %s", modelPath);
    end

    %% Load and Validate Required Variables
    modelData = load(modelPath);
    requiredVariables = ["svmModel", "tfidfModel", "metrics", "evaluationResults"];
    validateRequiredVariables(modelData, requiredVariables);

    svmModel = modelData.svmModel;
    tfidfModel = modelData.tfidfModel;
    metrics = modelData.metrics;
    evaluationResults = modelData.evaluationResults;

    if isempty(svmModel)
        error("loadModel:EmptySVMModel", ...
            "Loaded svmModel is empty.");
    end

    if isempty(tfidfModel)
        error("loadModel:EmptyTFIDFModel", ...
            "Loaded tfidfModel is empty.");
    end

    if ~isstruct(metrics)
        error("loadModel:InvalidMetrics", ...
            "Loaded metrics variable must be a struct.");
    end

    if ~isstruct(evaluationResults)
        error("loadModel:InvalidEvaluationResults", ...
            "Loaded evaluationResults variable must be a struct.");
    end

    %% Print Loading Summary
    fprintf("\nModel Load Summary\n");
    fprintf("------------------\n");
    fprintf("Model path: %s\n", modelPath);

    if isfield(modelData, "timestamp")
        fprintf("Timestamp : %s\n", string(modelData.timestamp));
    else
        fprintf("Timestamp : Not available\n");
    end

    fprintf("Status    : Loaded successfully\n\n");

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured persistence logging and compatibility diagnostics.
    fprintf(2, "Model loading failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateRequiredVariables(modelData, requiredVariables)
%VALIDATEREQUIREDVARIABLES Ensure the MAT file contains required variables.

availableVariables = string(fieldnames(modelData));
missingVariables = setdiff(requiredVariables, availableVariables);

if ~isempty(missingVariables)
    error("loadModel:MissingRequiredVariable", ...
        "Model file is missing required variable(s): %s", ...
        strjoin(missingVariables, ", "));
end

end

function modelPath = defaultModelPath()
%DEFAULTMODELPATH Build the default repository-local model path.

projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelPath = fullfile(projectRoot, "models", "fake_news_model.mat");

end

