function [X, tfidfModel] = extractTFIDF(processedText)
%EXTRACTTFIDF Extract TF-IDF features from preprocessed news text.
%
% Description:
%   [X, tfidfModel] = extractTFIDF(processedText) converts cleaned article
%   text into a sparse TF-IDF feature matrix suitable for machine learning
%   model training and inference. The function tokenizes documents, builds a
%   bag-of-words model, removes empty documents, removes extremely rare
%   words using a minimum document frequency threshold, and computes TF-IDF
%   weights.
%
% Inputs:
%   processedText - String array containing cleaned article text.
%
% Outputs:
%   X          - Sparse TF-IDF feature matrix. Rows represent documents and
%                columns represent vocabulary terms.
%   tfidfModel - Pruned bag-of-words model used to compute the TF-IDF
%                representation.
%
% TODO:
%   - Persist the bag-of-words vocabulary and preprocessing settings.
%   - Add configurable minimum document frequency.
%   - Add optional n-gram feature extraction.
%   - Add tests for empty documents and vocabulary pruning behavior.

try
    if nargin < 1
        error("extractTFIDF:MissingInput", "processedText input is required.");
    end

    if ~isstring(processedText)
        error("extractTFIDF:InvalidInputType", ...
            "processedText must be a string array.");
    end

    if isempty(processedText)
        error("extractTFIDF:EmptyInput", ...
            "processedText must contain at least one document.");
    end

    %% Prepare Text for Tokenization
    % Work with a column vector internally so the returned TF-IDF matrix has
    % one row per document regardless of the caller's input shape.
    processedText = processedText(:);
    processedText(ismissing(processedText)) = "";
    processedText = strtrim(processedText);

    %% Convert Text to tokenizedDocument
    documents = tokenizedDocument(processedText);

    %% Remove Empty Documents
    % Empty documents can appear after preprocessing removes URLs, numbers,
    % punctuation, or stop words. They do not contribute features and should
    % not be used for model training.
    documents = removeEmptyDocuments(documents);

    if isempty(documents)
        error("extractTFIDF:NoValidDocuments", ...
            "No non-empty documents remain after tokenization.");
    end

    %% Create Bag-of-Words Representation
    tfidfModel = bagOfWords(documents);

    if tfidfModel.NumWords == 0
        error("extractTFIDF:EmptyVocabulary", ...
            "The bag-of-words vocabulary is empty.");
    end

    %% Remove Extremely Rare Words
    % Minimum document frequency is the number of documents a word must
    % appear in to remain in the vocabulary. Terms appearing in fewer than
    % two documents are removed to reduce noise and overfitting risk.
    minimumDocumentFrequency = 2;
    documentFrequency = sum(tfidfModel.Counts > 0, 1);
    rareWordMask = documentFrequency < minimumDocumentFrequency;

    if any(rareWordMask)
        rareWords = tfidfModel.Vocabulary(rareWordMask);
        tfidfModel = removeWords(tfidfModel, rareWords);
    end

    if tfidfModel.NumWords == 0
        error("extractTFIDF:VocabularyPrunedToEmpty", ...
            "No vocabulary terms remain after applying minimum document frequency = %d.", ...
            minimumDocumentFrequency);
    end

    % Removing rare words can make some documents empty. Remove them from
    % the final model so every TF-IDF row contains at least one retained
    % vocabulary term.
    tfidfModel = removeEmptyDocuments(tfidfModel);

    if tfidfModel.NumDocuments == 0
        error("extractTFIDF:NoDocumentsAfterPruning", ...
            "No documents remain after rare-word pruning.");
    end

    %% Compute Sparse TF-IDF Matrix
    X = tfidf(tfidfModel);
    X = sparse(X);

    %% Print Feature Extraction Summary
    fprintf("\nTF-IDF Feature Summary\n");
    fprintf("----------------------\n");
    fprintf("Number of documents     : %d\n", tfidfModel.NumDocuments);
    fprintf("Vocabulary size         : %d\n", tfidfModel.NumWords);
    fprintf("TF-IDF matrix dimensions: %d x %d\n\n", size(X, 1), size(X, 2));

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured logging and feature extraction diagnostics.
    %   - Include document identifiers once dataset row IDs are available.
    fprintf(2, "TF-IDF feature extraction failed: %s\n", ME.message);
    rethrow(ME);
end

end
