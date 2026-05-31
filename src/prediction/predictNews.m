function [predictedLabel, confidenceScore] = predictNews(newsText, svmModel, tfidfModel)
%PREDICTNEWS Predict whether a news article is fake or real.
%
% Description:
%   [predictedLabel, confidenceScore] = predictNews(newsText, svmModel,
%   tfidfModel) preprocesses a single news article, converts it into a
%   TF-IDF feature vector using the existing training vocabulary, and
%   predicts the class with the trained linear SVM-style model.
%
% Inputs:
%   newsText   - String scalar containing the news article text to classify.
%   svmModel   - Trained model returned by trainSVM.m.
%   tfidfModel - Bag-of-words model returned by extractTFIDF.m.
%
% Outputs:
%   predictedLabel   - Numeric class label:
%                      0 = Fake
%                      1 = Real
%   confidenceScore  - Probability-like confidence score in the range
%                      [0, 1], derived from the model classification score.
%
% Notes:
%   This function must use the same preprocessing policy as training. It
%   calls preprocessText.m before constructing inference features.
%
% TODO:
%   - Calibrate confidence scores with posterior probabilities if required.
%   - Persist and reuse TF-IDF inverse document frequency settings.
%   - Add tests for empty input, unseen vocabulary, and model output types.

try
    %% Validate Inputs
    if nargin < 3
        error("predictNews:MissingInput", ...
            "newsText, svmModel, and tfidfModel inputs are required.");
    end

    if ~isstring(newsText) || ~isscalar(newsText)
        error("predictNews:InvalidTextInput", ...
            "newsText must be a string scalar.");
    end

    if ismissing(newsText) || strlength(strtrim(newsText)) == 0
        error("predictNews:EmptyTextInput", ...
            "newsText must contain non-empty text.");
    end

    if isempty(svmModel)
        error("predictNews:EmptyModel", "svmModel must not be empty.");
    end

    if isempty(tfidfModel)
        error("predictNews:EmptyTFIDFModel", "tfidfModel must not be empty.");
    end

    validateTFIDFModel(tfidfModel);

    %% Apply Training-Time Preprocessing
    % predictNews lives in src/prediction while preprocessText lives in
    % src/preprocessing. Add that folder locally if the caller has not
    % already configured the MATLAB path.
    projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
    preprocessingDir = fullfile(projectRoot, "src", "preprocessing");

    if exist("preprocessText", "file") ~= 2
        addpath(preprocessingDir);
    end

    processedText = preprocessText(newsText);

    if strlength(strtrim(processedText)) == 0
        error("predictNews:EmptyAfterPreprocessing", ...
            "newsText is empty after preprocessing and cannot be classified.");
    end

    %% Convert Text to Existing-Vocabulary TF-IDF Features
    % The inference document must be encoded with the training vocabulary.
    % This keeps the feature order and dimensionality aligned with the model
    % trained by trainSVM.m.
    inferenceDocument = tokenizedDocument(processedText);
    termCounts = encode(tfidfModel, inferenceDocument);

    if nnz(termCounts) == 0
        error("predictNews:NoKnownVocabulary", ...
            "The input text contains no words from the trained TF-IDF vocabulary.");
    end

    XNew = computeTFIDFFromTrainingModel(termCounts, tfidfModel);

    %% Predict with Trained Model
    % fitclinear returns classification scores. These scores are not
    % calibrated probabilities, so the confidence value below is a
    % probability-like score produced by a sigmoid transform.
    [modelLabel, modelScore] = predict(svmModel, XNew);

    predictedLabel = normalizePredictedLabel(modelLabel);
    confidenceScore = computeConfidenceScore(modelScore, predictedLabel);

    %% Print Prediction Summary
    className = "Fake";

    if predictedLabel == 1
        className = "Real";
    end

    fprintf("\nPrediction Result\n");
    fprintf("-----------------\n");
    fprintf("Predicted Class : %s\n", className);
    fprintf("Confidence Score: %.4f\n\n", confidenceScore);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured prediction logging and user-facing recovery tips.
    fprintf(2, "News prediction failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateTFIDFModel(tfidfModel)
%VALIDATETFIDFMODEL Validate the bag-of-words model needed for inference.

requiredProperties = ["Vocabulary", "Counts", "NumDocuments", "NumWords"];

for propertyName = requiredProperties
    if ~isprop(tfidfModel, propertyName)
        error("predictNews:InvalidTFIDFModel", ...
            "tfidfModel must contain the %s property.", propertyName);
    end
end

if tfidfModel.NumDocuments == 0 || tfidfModel.NumWords == 0
    error("predictNews:InvalidTFIDFModel", ...
        "tfidfModel must contain at least one document and one vocabulary word.");
end

end

function XNew = computeTFIDFFromTrainingModel(termCounts, tfidfModel)
%COMPUTETFIDFFROMTRAININGMODEL Build one sparse TF-IDF row for inference.

termCounts = sparse(termCounts);

% Reuse the training document frequencies to approximate the TF-IDF
% weighting used during training while keeping the original vocabulary
% order. The smoothed form avoids division by zero and behaves safely when
% the vocabulary is pruned.
documentFrequency = full(sum(tfidfModel.Counts > 0, 1));
numDocuments = tfidfModel.NumDocuments;
inverseDocumentFrequency = log((numDocuments + 1) ./ (documentFrequency + 1)) + 1;

termFrequency = termCounts ./ max(sum(termCounts, 2), 1);
XNew = termFrequency .* sparse(inverseDocumentFrequency);
XNew = sparse(XNew);

end

function predictedLabel = normalizePredictedLabel(modelLabel)
%NORMALIZEPREDICTEDLABEL Convert model output labels to numeric 0/1 labels.

if iscategorical(modelLabel)
    labelString = string(modelLabel);

    if labelString == "Fake" || labelString == "0"
        predictedLabel = 0;
    elseif labelString == "Real" || labelString == "1"
        predictedLabel = 1;
    else
        error("predictNews:UnexpectedModelLabel", ...
            "Model returned unsupported categorical label: %s", labelString);
    end
elseif islogical(modelLabel) || isnumeric(modelLabel)
    predictedLabel = double(modelLabel);

    if ~isscalar(predictedLabel) || ~ismember(predictedLabel, [0, 1])
        error("predictNews:UnexpectedModelLabel", ...
            "Model returned unsupported numeric label.");
    end
else
    error("predictNews:UnexpectedModelLabel", ...
        "Model returned an unsupported label type.");
end

end

function confidenceScore = computeConfidenceScore(modelScore, predictedLabel)
%COMPUTECONFIDENCESCORE Convert model score output to a bounded score.

if isempty(modelScore)
    confidenceScore = NaN;
    return;
end

modelScore = double(modelScore);

if isvector(modelScore)
    modelScore = modelScore(:).';
end

if size(modelScore, 2) >= 2
    % For binary classifiers, MATLAB commonly returns one score per class.
    % Use the score associated with the predicted class.
    scoreIndex = predictedLabel + 1;
    rawScore = modelScore(1, scoreIndex);
else
    rawScore = modelScore(1);
end

% Sigmoid transformation gives a probability-like confidence while avoiding
% a claim of calibrated posterior probability.
confidenceScore = 1 ./ (1 + exp(-abs(rawScore)));

end
