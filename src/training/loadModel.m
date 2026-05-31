function [svmModel, tfidfModel, metrics, evaluationResults] = loadModel(modelPath)
%LOADMODEL Load fake news detection model artifacts.
%
% Description:
%   [svmModel, tfidfModel, metrics, evaluationResults] =
%   loadModel(modelPath) loads a persisted fake news detection model from a
%   MAT file. The trained classifier and TF-IDF model are required. Training
%   metrics and evaluation results are optional and default to empty structs
%   when they are not present in the saved artifact.
%
% Inputs:
%   modelPath - Optional MAT-file path. If omitted or empty, the model is
%               loaded from models/fake_news_model.mat.
%
% Outputs:
%   svmModel          - Loaded trained classifier.
%   tfidfModel        - Loaded bag-of-words / TF-IDF model.
%   metrics           - Loaded training metrics struct, or struct() if not
%                       present.
%   evaluationResults - Loaded evaluation results struct, or struct() if not
%                       present.
%
% TODO:
%   - Add savedAt/version compatibility checks.
%   - Add MATLAB and toolbox version metadata validation.
%   - Add optional integrity validation for saved artifacts.

try
    %% Resolve Input Path
    if nargin < 1 || isempty(modelPath)
        modelPath = defaultModelPath();
    else
        modelPath = normalizeModelPath(modelPath, "loadModel");
    end

    %% Validate File Exists
    if ~isfile(modelPath)
        error("loadModel:FileNotFound", ...
            "Model file was not found at: %s", modelPath);
    end

    %% Load Artifact and Validate Required Variables
    modelData = load(modelPath);
    requiredVariables = ["svmModel", "tfidfModel"];
    validateRequiredVariables(modelData, requiredVariables);

    svmModel = modelData.svmModel;
    tfidfModel = modelData.tfidfModel;

    if isempty(svmModel)
        error("loadModel:EmptySVMModel", ...
            "Loaded svmModel is empty.");
    end

    if isempty(tfidfModel)
        error("loadModel:EmptyTFIDFModel", ...
            "Loaded tfidfModel is empty.");
    end

    %% Load Optional Variables
    % Older or partial model artifacts may not contain metrics. Return empty
    % structs so downstream code can safely check field availability.
    if isfield(modelData, "metrics") && isstruct(modelData.metrics)
        metrics = modelData.metrics;
    else
        metrics = struct();
    end

    if isfield(modelData, "evaluationResults") && isstruct(modelData.evaluationResults)
        evaluationResults = modelData.evaluationResults;
    else
        evaluationResults = struct();
    end

    %% Print Loading Summary
    fprintf("\nModel loaded successfully.\n");
    fprintf("Loaded file: %s\n", modelPath);

    if isfield(modelData, "savedAt")
        fprintf("Saved at   : %s\n", string(modelData.savedAt));
    else
        fprintf("Saved at   : Not available\n");
    end

    fprintf("Metrics loaded           : %s\n", string(~isempty(fieldnames(metrics))));
    fprintf("Evaluation results loaded: %s\n\n", ...
        string(~isempty(fieldnames(evaluationResults))));

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured persistence logging.
    %   - Add diagnostics for incompatible model artifact versions.
    fprintf(2, "Model loading failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateRequiredVariables(modelData, requiredVariables)
%VALIDATEREQUIREDVARIABLES Ensure required variables exist in loaded data.

availableVariables = string(fieldnames(modelData));
missingVariables = setdiff(requiredVariables, availableVariables);

if ~isempty(missingVariables)
    error("loadModel:MissingRequiredVariable", ...
        "Model file is missing required variable(s): %s", ...
        strjoin(missingVariables, ", "));
end

end

function modelPath = defaultModelPath()
%DEFAULTMODELPATH Return the repository-local default model path.

projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelPath = fullfile(projectRoot, "models", "fake_news_model.mat");

end

function modelPath = normalizeModelPath(modelPath, callerName)
%NORMALIZEMODELPATH Validate and convert a path input to a character vector.

errorId = char(string(callerName) + ":InvalidModelPath");

if ~(isstring(modelPath) || ischar(modelPath))
    error(errorId, ...
        "modelPath must be a string scalar or character vector.");
end

modelPath = string(modelPath);

if ~isscalar(modelPath) || strlength(strtrim(modelPath)) == 0
    error(errorId, ...
        "modelPath must be a non-empty string scalar or character vector.");
end

modelPath = char(modelPath);

end
