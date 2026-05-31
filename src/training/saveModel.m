function saveModel(svmModel, tfidfModel, metrics, evaluationResults, modelPath)
%SAVEMODEL Save trained fake news detection model artifacts.
%
% Description:
%   saveModel(svmModel, tfidfModel, metrics, evaluationResults, modelPath)
%   saves the trained classifier, TF-IDF bag-of-words model, training
%   metrics, evaluation metrics, and a timestamp to a MAT file.
%
% Inputs:
%   svmModel          - Trained classifier returned by trainSVM.m.
%   tfidfModel        - TF-IDF bag-of-words model returned by extractTFIDF.m.
%   metrics           - Training metrics structure returned by trainSVM.m.
%   evaluationResults - Evaluation metrics structure returned by
%                       evaluateModel.m.
%   modelPath         - Optional string scalar or character vector path to
%                       the output MAT file. Default:
%                       models/fake_news_model.mat
%
% Outputs:
%   None. Artifacts are saved to disk.
%
% TODO:
%   - Add model version metadata.
%   - Add dataset hash or training configuration metadata.
%   - Add optional compression or separate artifact files for large models.

try
    %% Validate Required Inputs
    if nargin < 4
        error("saveModel:MissingInput", ...
            "svmModel, tfidfModel, metrics, and evaluationResults are required.");
    end

    if isempty(svmModel)
        error("saveModel:EmptySVMModel", "svmModel must not be empty.");
    end

    if isempty(tfidfModel)
        error("saveModel:EmptyTFIDFModel", "tfidfModel must not be empty.");
    end

    if ~isstruct(metrics)
        error("saveModel:InvalidMetrics", ...
            "metrics must be a struct returned by trainSVM.m.");
    end

    if ~isstruct(evaluationResults)
        error("saveModel:InvalidEvaluationResults", ...
            "evaluationResults must be a struct returned by evaluateModel.m.");
    end

    %% Resolve Model Path
    if nargin < 5 || isempty(modelPath)
        modelPath = defaultModelPath();
    else
        modelPath = string(modelPath);

        if ~isscalar(modelPath) || strlength(strtrim(modelPath)) == 0
            error("saveModel:InvalidModelPath", ...
                "modelPath must be a non-empty string scalar or character vector.");
        end

        modelPath = char(modelPath);
    end

    modelFolder = fileparts(modelPath);

    if strlength(string(modelFolder)) > 0 && ~isfolder(modelFolder)
        mkdir(modelFolder);
    end

    %% Save Model Artifacts
    % Timestamp is stored with the artifact so future training runs can be
    % traced and compared without relying on file-system metadata.
    timestamp = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");

    save(modelPath, ...
        "svmModel", ...
        "tfidfModel", ...
        "metrics", ...
        "evaluationResults", ...
        "timestamp", ...
        "-v7.3");

    %% Print Save Summary
    fprintf("\nModel Save Summary\n");
    fprintf("------------------\n");
    fprintf("Model path: %s\n", modelPath);
    fprintf("Timestamp : %s\n\n", string(timestamp));

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured persistence logging and save retry behavior.
    fprintf(2, "Model saving failed: %s\n", ME.message);
    rethrow(ME);
end

end

function modelPath = defaultModelPath()
%DEFAULTMODELPATH Build the default repository-local model path.

projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelPath = fullfile(projectRoot, "models", "fake_news_model.mat");

end

