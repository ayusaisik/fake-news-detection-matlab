function saveModel(svmModel, tfidfModel, metrics, evaluationResults, modelPath)
%SAVEMODEL Save fake news detection model artifacts.
%
% Description:
%   saveModel(svmModel, tfidfModel, metrics, evaluationResults, modelPath)
%   saves the trained classifier, TF-IDF bag-of-words model, optional
%   training metrics, optional evaluation results, and a save timestamp to a
%   MAT file.
%
% Inputs:
%   svmModel          - Required trained classifier returned by trainSVM.m.
%   tfidfModel        - Required bag-of-words model returned by
%                       extractTFIDF.m.
%   metrics           - Optional training metrics struct. If omitted or
%                       empty, an empty struct is saved.
%   evaluationResults - Optional evaluation results struct. If omitted or
%                       empty, an empty struct is saved.
%   modelPath         - Optional output path. If omitted or empty, the model
%                       is saved to models/fake_news_model.mat.
%
% Outputs:
%   None.
%
% TODO:
%   - Add model version metadata.
%   - Add training configuration and dataset checksum metadata.
%   - Add compatibility checks before overwriting existing model artifacts.

try
    %% Validate Required Inputs
    if nargin < 2
        error("saveModel:MissingInput", ...
            "svmModel and tfidfModel are required inputs.");
    end

    if isempty(svmModel)
        error("saveModel:EmptySVMModel", "svmModel must not be empty.");
    end

    if isempty(tfidfModel)
        error("saveModel:EmptyTFIDFModel", "tfidfModel must not be empty.");
    end

    %% Normalize Optional Inputs
    % Metrics are useful for traceability, but a model can still be saved
    % before formal evaluation has been run.
    if nargin < 3 || isempty(metrics)
        metrics = struct();
    elseif ~isstruct(metrics)
        error("saveModel:InvalidMetrics", ...
            "metrics must be a struct when provided.");
    end

    if nargin < 4 || isempty(evaluationResults)
        evaluationResults = struct();
    elseif ~isstruct(evaluationResults)
        error("saveModel:InvalidEvaluationResults", ...
            "evaluationResults must be a struct when provided.");
    end

    %% Resolve Output Path
    if nargin < 5 || isempty(modelPath)
        modelPath = defaultModelPath();
    else
        modelPath = normalizeModelPath(modelPath, "saveModel");
    end

    modelFolder = fileparts(modelPath);

    if strlength(string(modelFolder)) > 0 && ~isfolder(modelFolder)
        mkdir(modelFolder);
    end

    %% Save Model Artifact
    % savedAt records when the artifact was written, independent of file
    % system metadata that can change when files are copied.
    savedAt = datetime("now");

    save(modelPath, ...
        "svmModel", ...
        "tfidfModel", ...
        "metrics", ...
        "evaluationResults", ...
        "savedAt", ...
        "-v7.3");

    %% Print Save Summary
    fprintf("\nModel saved successfully.\n");
    fprintf("Saved file: %s\n\n", modelPath);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured persistence logging.
    %   - Add optional overwrite protection and retry behavior.
    fprintf(2, "Model saving failed: %s\n", ME.message);
    rethrow(ME);
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
