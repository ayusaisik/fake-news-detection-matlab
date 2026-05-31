function [X, tfidfModel, validDocumentMask] = extractTFIDF(processedText)
%EXTRACTTFIDF Extract TF-IDF features from preprocessed news text.
%
% Description:
%   [X, tfidfModel, validDocumentMask] = extractTFIDF(processedText)
%   converts cleaned article text into a sparse TF-IDF feature matrix
%   suitable for machine learning model training and inference. The function
%   tokenizes documents, builds a bag-of-words model, removes empty
%   documents, removes extremely rare words using a minimum document
%   frequency threshold, and computes TF-IDF weights.
%
%   validDocumentMask preserves row alignment with the original
%   processedText input. Use it to filter labels after feature extraction:
%
%       labels = labels(validDocumentMask);
%
% Inputs:
%   processedText - String array containing cleaned article text.
%
% Outputs:
%   X                 - Sparse TF-IDF feature matrix. Rows represent valid
%                       retained documents and columns represent vocabulary
%                       terms.
%   tfidfModel        - Pruned bag-of-words model used to compute the TF-IDF
%                       representation.
%   validDocumentMask - Logical column vector with one element per original
%                       processedText element. A true value means the
%                       corresponding original document remains aligned with
%                       a row in X after missing-value handling, trimming,
%                       empty document filtering, and vocabulary pruning.
%
% TODO:
%   - Persist the bag-of-words vocabulary and preprocessing settings.
%   - Add configurable minimum document frequency.
%   - Add optional n-gram feature extraction.
%   - Add tests for label alignment after document filtering.

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

    %% Prepare Text and Alignment Mask
    % Work with a column vector internally so each document has a stable
    % row index. validDocumentMask maps retained rows back to the caller's
    % original input order.
    originalDocumentCount = numel(processedText);
    validDocumentMask = false(originalDocumentCount, 1);

    processedText = processedText(:);
    processedText(ismissing(processedText)) = "";
    processedText = strtrim(processedText);

    % Documents that are empty after missing-value handling and trimming are
    % excluded before tokenization. Their labels must also be excluded by
    % applying validDocumentMask to the original label vector.
    nonEmptyTextMask = strlength(processedText) > 0;

    if ~any(nonEmptyTextMask)
        error("extractTFIDF:NoValidDocuments", ...
            "No non-empty documents remain after missing-value handling and trimming.");
    end

    retainedOriginalIndices = find(nonEmptyTextMask);

    %% Convert Text to tokenizedDocument
    documents = tokenizedDocument(processedText(nonEmptyTextMask));

    %% Remove Empty Documents
    % Tokenization can still produce empty documents for inputs that contain
    % no valid tokens. Track those removals explicitly so feature rows and
    % labels stay aligned.
    nonEmptyTokenMask = doclength(documents) > 0;
    documents = documents(nonEmptyTokenMask);
    retainedOriginalIndices = retainedOriginalIndices(nonEmptyTokenMask);

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
    documentFrequency = full(sum(tfidfModel.Counts > 0, 1));
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

    % Removing rare words can make some documents empty. Update the
    % alignment mask before removing those rows from the model. This avoids
    % silently losing feature-label alignment.
    retainedAfterPruningMask = full(sum(tfidfModel.Counts, 2)) > 0;
    retainedOriginalIndices = retainedOriginalIndices(retainedAfterPruningMask);
    validDocumentMask(retainedOriginalIndices) = true;

    tfidfModel = removeEmptyDocuments(tfidfModel);

    if tfidfModel.NumDocuments == 0
        error("extractTFIDF:NoDocumentsAfterPruning", ...
            "No documents remain after rare-word pruning.");
    end

    %% Compute Sparse TF-IDF Matrix
    X = sparse(tfidf(tfidfModel));

    if size(X, 1) ~= nnz(validDocumentMask)
        error("extractTFIDF:AlignmentMismatch", ...
            "TF-IDF rows (%d) do not match valid document count (%d).", ...
            size(X, 1), nnz(validDocumentMask));
    end

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
