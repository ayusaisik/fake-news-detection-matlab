function [predictedLabel, confidenceScore] = predictNews(newsText, svmModel, tfidfModel)
%PREDICTNEWS Predict whether a news article is fake or real.
%
% Description:
%   [predictedLabel, confidenceScore] = predictNews(newsText, svmModel,
%   tfidfModel) preprocesses a single news article, tokenizes it, computes
%   TF-IDF features using the trained bag-of-words model vocabulary, and
%   predicts whether the article is fake or real.
%
% Inputs:
%   newsText   - String scalar containing the news article text.
%   svmModel   - Trained linear classifier returned by trainSVM.m.
%   tfidfModel - Trained bag-of-words model returned by extractTFIDF.m.
%
% Outputs:
%   predictedLabel  - Numeric class label:
%                     0 = Fake
%                     1 = Real
%   confidenceScore - Margin-based confidence score in [0, 1]. This is not
%                     a calibrated probability.
%
% Important:
%   Do not create a new bagOfWords model during prediction. This function
%   uses tfidf(tfidfModel, inferenceDocument), which applies MATLAB's TF-IDF
%   workflow with the trained vocabulary and IDF factors.

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
    % This must match the preprocessing used before extractTFIDF.m during
    % training. Keeping preprocessing centralized avoids training/inference
    % drift.
    projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
    preprocessingDir = fullfile(projectRoot, "src", "preprocessing");

    if exist("preprocessText", "file") ~= 2
        addpath(preprocessingDir);
    end

    processedText = strtrim(preprocessText(newsText));

    if strlength(processedText) == 0
        error("predictNews:EmptyAfterPreprocessing", ...
            "newsText is empty after preprocessing and cannot be classified.");
    end

    %% Tokenize and Compute TF-IDF with the Trained Vocabulary
    inferenceDocument = tokenizedDocument(processedText);

    % encode is used only for debug diagnostics and vocabulary-overlap
    % checks. It does not define the model features.
    termCounts = sparse(encode(tfidfModel, inferenceDocument));
    recognizedVocabularyTermCount = nnz(termCounts);
    recognizedTermOccurrences = full(sum(termCounts, 2));
    totalTokenCount = doclength(inferenceDocument);

    if recognizedTermOccurrences == 0
        error("predictNews:NoKnownVocabulary", ...
            "The input text contains no words from the trained TF-IDF vocabulary.");
    end

    vocabularyOverlapRatio = recognizedTermOccurrences / max(totalTokenCount, 1);

    if vocabularyOverlapRatio < 0.50
        fprintf(2, "Warning: The input text has low vocabulary overlap with the trained model.\n");
    end

    % Critical inference step:
    % Use MATLAB's tfidf function with the trained bag model and the new
    % tokenized document. This preserves vocabulary order and uses the same
    % TF-IDF weighting behavior as tfidf(tfidfModel) used during training.
    XNew = sparse(tfidf(tfidfModel, inferenceDocument));

    if size(XNew, 2) ~= tfidfModel.NumWords
        error("predictNews:VocabularyDimensionMismatch", ...
            "Prediction feature columns (%d) do not match tfidfModel.NumWords (%d).", ...
            size(XNew, 2), tfidfModel.NumWords);
    end

    validateModelFeatureCount(svmModel, XNew);

    %% Predict with Trained Model
    confidenceScore = NaN;
    modelScore = [];

    try
        [modelLabel, modelScore] = predict(svmModel, XNew);
    catch scoreME
        % Some model objects may only support one output from predict. In
        % that case, prediction can still proceed but confidence is unknown.
        if strcmp(scoreME.identifier, "MATLAB:maxlhs")
            modelLabel = predict(svmModel, XNew);
        else
            rethrow(scoreME);
        end
    end

    predictedLabel = normalizePredictedLabel(modelLabel);
    predictedClass = labelToClassName(predictedLabel);

    if ~isempty(modelScore)
        scoreMargin = getPredictedClassMargin(modelScore, modelLabel, svmModel);

        % fitclinear with Learner="svm" returns classification scores, not
        % calibrated posterior probabilities. This confidence is a bounded
        % margin-based signal: larger absolute margins indicate stronger
        % separation from the decision boundary.
        confidenceScore = 1 / (1 + exp(-abs(scoreMargin)));
    end

    %% Print Debug-Friendly Prediction Summary
    fprintf("\nPrediction Result\n");
    fprintf("-----------------\n");
    fprintf("Processed text              : %s\n", processedText);
    fprintf("Recognized vocabulary terms : %d\n", recognizedVocabularyTermCount);
    fprintf("Recognized term occurrences : %d of %d\n", ...
        recognizedTermOccurrences, totalTokenCount);
    fprintf("Predicted class             : %s\n", predictedClass);

    if isempty(modelScore)
        fprintf("Model score                 : Unavailable\n");
        fprintf("Confidence score            : Confidence unavailable\n\n");
    else
        fprintf("Model score                 : %s\n", mat2str(modelScore, 4));
        fprintf("Confidence score            : %.4f\n\n", confidenceScore);
    end

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured prediction logging.
    %   - Add compatibility checks for model artifacts saved by older runs.
    fprintf(2, "News prediction failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateTFIDFModel(tfidfModel)
%VALIDATETFIDFMODEL Validate the trained bag-of-words model for inference.

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

function validateModelFeatureCount(svmModel, XNew)
%VALIDATEMODELFEATURECOUNT Ensure inference features match model predictors.

expectedFeatureCount = [];

if isprop(svmModel, "NumPredictors")
    expectedFeatureCount = svmModel.NumPredictors;
elseif isprop(svmModel, "Beta")
    expectedFeatureCount = numel(svmModel.Beta);
else
    fprintf(2, "Warning: Could not verify model feature count from svmModel metadata.\n");
    expectedFeatureCount = size(XNew, 2);
end

if ~isempty(expectedFeatureCount) && size(XNew, 2) ~= expectedFeatureCount
    error("predictNews:FeatureDimensionMismatch", ...
        "Prediction feature columns (%d) do not match expected model predictors (%d).", ...
        size(XNew, 2), expectedFeatureCount);
end

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

function scoreMargin = getPredictedClassMargin(modelScore, modelLabel, svmModel)
%GETPREDICTEDCLASSMARGIN Return the score associated with predicted class.

modelScore = double(modelScore);

if isvector(modelScore)
    modelScore = modelScore(:).';
end

if isempty(modelScore)
    scoreMargin = NaN;
    return;
end

if size(modelScore, 2) == 1
    scoreMargin = modelScore(1);
    return;
end

predictedLabel = normalizePredictedLabel(modelLabel);
scoreIndex = predictedLabel + 1;

% Prefer svmModel.ClassNames when available because it preserves the model's
% actual score-column ordering.
if isprop(svmModel, "ClassNames")
    classNames = svmModel.ClassNames;
    classLabels = strings(size(classNames));

    for i = 1:numel(classNames)
        classLabels(i) = string(classNames(i));
    end

    if predictedLabel == 0
        matchIndex = find(classLabels == "Fake" | classLabels == "0", 1);
    else
        matchIndex = find(classLabels == "Real" | classLabels == "1", 1);
    end

    if ~isempty(matchIndex)
        scoreIndex = matchIndex;
    end
end

if scoreIndex > size(modelScore, 2)
    error("predictNews:ScoreDimensionMismatch", ...
        "Predicted class score index exceeds available model score columns.");
end

scoreMargin = modelScore(1, scoreIndex);

end

function className = labelToClassName(predictedLabel)
%LABELTOCLASSNAME Convert numeric model label to user-facing class name.

if predictedLabel == 0
    className = "Fake";
elseif predictedLabel == 1
    className = "Real";
else
    error("predictNews:UnexpectedPrediction", ...
        "Predicted label must be 0 = Fake or 1 = Real.");
end

end
