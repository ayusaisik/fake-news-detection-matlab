function combinedDataset = loadDataset()
%LOADDATASET Load, label, merge, shuffle, and summarize news datasets.
%
% Description:
%   combinedDataset = loadDataset() reads the raw fake and true news CSV
%   files from data/raw/, assigns binary labels, merges the datasets into a
%   single table, randomly shuffles the rows, validates basic dataset
%   quality indicators, and prints a clear dataset summary.
%
% Inputs:
%   None.
%
% Outputs:
%   combinedDataset - Shuffled table containing both fake and true news
%                     samples with FullText and numeric Label columns.
%
% Label Mapping:
%   Fake.csv -> 0
%   True.csv -> 1
%
% TODO:
%   - Add configurable dataset paths.
%   - Add duplicate-row detection and reporting.
%   - Add optional export to data/processed/.
%   - Add unit tests for missing files and malformed CSV files.

try
    %% Configure Dataset Paths
    projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
    rawDataDir = fullfile(projectRoot, "data", "raw");

    fakeFilePath = fullfile(rawDataDir, "Fake.csv");
    trueFilePath = fullfile(rawDataDir, "True.csv");

    %% Validate Input Files
    validateFileExists(fakeFilePath, "Fake news dataset");
    validateFileExists(trueFilePath, "True news dataset");

    %% Read Raw Datasets
    fakeDataset = readtable(fakeFilePath, "TextType", "string");
    trueDataset = readtable(trueFilePath, "TextType", "string");

    %% Validate Required Columns
    requiredColumns = ["title", "text"];
    validateRequiredColumns(fakeDataset, requiredColumns, "Fake.csv");
    validateRequiredColumns(trueDataset, requiredColumns, "True.csv");

    %% Create Full Text Field
    % FullText provides the canonical text input for downstream
    % preprocessing and feature extraction.
    fakeDataset.FullText = fakeDataset.title + " " + fakeDataset.text;
    trueDataset.FullText = trueDataset.title + " " + trueDataset.text;
    fullTextCreationStatus = "Yes";

    %% Assign Labels
    fakeDataset.Label = zeros(height(fakeDataset), 1);
    trueDataset.Label = ones(height(trueDataset), 1);

    %% Merge Datasets
    combinedDataset = [fakeDataset; trueDataset];

    if isempty(combinedDataset)
        error("loadDataset:EmptyDataset", ...
            "The merged dataset is empty. Check the source CSV files.");
    end

    %% Shuffle Dataset
    rng(42);
    shuffledRowOrder = randperm(height(combinedDataset));
    combinedDataset = combinedDataset(shuffledRowOrder, :);

    %% Validate Dataset Statistics
    missingValuesCount = countMissingValues(combinedDataset);
    fakeSamplesCount = sum(combinedDataset.Label == 0);
    realSamplesCount = sum(combinedDataset.Label == 1);
    totalSamplesCount = height(combinedDataset);

    if fakeSamplesCount == 0 || realSamplesCount == 0
        error("loadDataset:MissingClass", ...
            "Both fake and real samples are required for model training.");
    end

    %% Print Dataset Summary
    fprintf("\nDataset Summary\n");
    fprintf("---------------\n");
    fprintf("Fake dataset path : %s\n", fakeFilePath);
    fprintf("True dataset path : %s\n", trueFilePath);
    fprintf("Total samples     : %d\n", totalSamplesCount);
    fprintf("Fake samples      : %d\n", fakeSamplesCount);
    fprintf("Real samples      : %d\n", realSamplesCount);
    fprintf("Missing values    : %d\n", missingValuesCount);
    fprintf("Columns           : %d\n", width(combinedDataset));
    fprintf("FullText created  : %s\n", fullTextCreationStatus);
    fprintf("Dataset shuffled  : Yes\n\n");

catch ME
    % Error handling placeholder:
    % TODO:
    %   - Replace console-only diagnostics with structured project logging.
    %   - Add optional recovery guidance for common dataset schema issues.
    fprintf(2, "Dataset loading failed: %s\n", ME.message);
    rethrow(ME);
end

end

function validateRequiredColumns(inputTable, requiredColumns, datasetName)
%VALIDATEREQUIREDCOLUMNS Ensure a table contains all required variables.

availableColumns = string(inputTable.Properties.VariableNames);
missingColumns = setdiff(requiredColumns, availableColumns);

if ~isempty(missingColumns)
    error("loadDataset:MissingRequiredColumn", ...
        "%s is missing required column(s): %s", ...
        datasetName, strjoin(missingColumns, ", "));
end

end

function validateFileExists(filePath, datasetName)
%VALIDATEFILEEXISTS Raise a clear error when a dataset file is missing.

if ~isfile(filePath)
    error("loadDataset:FileNotFound", ...
        "%s file was not found at: %s", datasetName, filePath);
end

end

function missingValuesCount = countMissingValues(inputTable)
%COUNTMISSINGVALUES Count missing values across all table variables.

missingMask = ismissing(inputTable);
missingValuesCount = sum(missingMask, "all");

end
