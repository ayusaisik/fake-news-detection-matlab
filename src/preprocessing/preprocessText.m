function processedText = preprocessText(rawText)
%PREPROCESSTEXT Clean and normalize raw news article text.
%
% Description:
%   processedText = preprocessText(rawText) prepares raw news article text
%   for TF-IDF feature extraction. The function normalizes text, removes
%   noisy patterns, tokenizes each document, removes English stop words
%   using Text Analytics Toolbox, and reconstructs cleaned documents as a
%   string array.
%
% Inputs:
%   rawText - String array containing raw article text.
%
% Outputs:
%   processedText - String array containing cleaned article text. The output
%                   preserves the input array shape.
%
% TODO:
%   - Add optional lemmatization or stemming.
%   - Add configurable stop-word language support.
%   - Add minimum token length filtering.
%   - Add tests for empty, missing, URL-heavy, and malformed text.

try
    if nargin < 1
        error("preprocessText:MissingInput", "rawText input is required.");
    end

    if ~isstring(rawText)
        error("preprocessText:InvalidInputType", ...
            "rawText must be a string array.");
    end

    % Return an empty string array with the same shape when the caller passes
    % an empty input. This keeps downstream table operations predictable.
    if isempty(rawText)
        processedText = strings(size(rawText));
        return;
    end

    originalSize = size(rawText);
    rawText = rawText(:);

    % Missing values cannot be tokenized reliably. Treat them as empty
    % documents so the number of samples remains unchanged.
    rawText(ismissing(rawText)) = "";

    %% Normalize Surface Text
    % Lowercase first so later text comparisons and tokenization are
    % consistent across articles.
    cleanedText = lower(rawText);

    % Remove common HTTP, HTTPS, and WWW URL patterns before punctuation is
    % stripped, preserving surrounding words.
    cleanedText = regexprep(cleanedText, ...
        "(https?://\S+|www\.\S+)", " ");

    % Remove punctuation and numbers. The whitespace replacement prevents
    % adjacent words from being accidentally concatenated.
    cleanedText = regexprep(cleanedText, "[^\w\s]", " ");
    cleanedText = regexprep(cleanedText, "\d+", " ");

    % Collapse repeated whitespace and trim document boundaries before
    % tokenization.
    cleanedText = regexprep(cleanedText, "\s+", " ");
    cleanedText = strtrim(cleanedText);

    %% Tokenize and Remove Stop Words
    processedText = strings(size(cleanedText));
    nonEmptyDocuments = strlength(cleanedText) > 0;

    if any(nonEmptyDocuments)
        documents = tokenizedDocument(cleanedText(nonEmptyDocuments));

        % removeStopWords uses the default English stop-word list in Text
        % Analytics Toolbox.
        documents = removeStopWords(documents);

        % Convert tokenized documents back to plain strings for TF-IDF and
        % table-based workflows.
        processedText(nonEmptyDocuments) = joinWords(documents);
    end

    % Normalize whitespace again after stop-word removal because removing
    % tokens can leave empty or unevenly spaced documents.
    processedText = regexprep(processedText, "\s+", " ");
    processedText = strtrim(processedText);
    processedText = reshape(processedText, originalSize);

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Add structured logging and context-specific recovery behavior.
    %   - Add dataset row identifiers to improve debugging for bad records.
    fprintf(2, "Text preprocessing failed: %s\n", ME.message);
    rethrow(ME);
end

end
