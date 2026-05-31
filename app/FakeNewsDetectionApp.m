%% Fake News Detection System - Programmatic MATLAB GUI
% MATLAB Version: R2024a
%
% Description:
%   Programmatic UI for classifying news text as fake or real using the
%   trained Fake News Detection System model. This file does not require App
%   Designer or an .mlapp file.
%
% Run:
%   run app/FakeNewsDetectionApp.m

%% App Startup
clear;
clc;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(projectRoot));

app = struct();
app.ProjectRoot = projectRoot;
app.SVMModel = [];
app.TFIDFModel = [];
app.ModelLoaded = false;

%% Build User Interface
app.Figure = uifigure( ...
    "Name", "Fake News Detection System", ...
    "Position", [100 100 760 560], ...
    "Color", [0.97 0.97 0.97]);

mainLayout = uigridlayout(app.Figure, [7 2]);
mainLayout.RowHeight = {48, 28, "1x", 44, 38, 38, 32};
mainLayout.ColumnWidth = {"1x", "1x"};
mainLayout.Padding = [24 20 24 20];
mainLayout.RowSpacing = 12;
mainLayout.ColumnSpacing = 12;

app.TitleLabel = uilabel(mainLayout, ...
    "Text", "Fake News Detection System", ...
    "FontSize", 24, ...
    "FontWeight", "bold", ...
    "HorizontalAlignment", "center");
app.TitleLabel.Layout.Row = 1;
app.TitleLabel.Layout.Column = [1 2];

app.InputLabel = uilabel(mainLayout, ...
    "Text", "Enter news text", ...
    "FontSize", 14, ...
    "FontWeight", "bold");
app.InputLabel.Layout.Row = 2;
app.InputLabel.Layout.Column = [1 2];

app.NewsTextArea = uitextarea(mainLayout, ...
    "Placeholder", "Paste or type a news article here...", ...
    "FontSize", 13);
app.NewsTextArea.Layout.Row = 3;
app.NewsTextArea.Layout.Column = [1 2];

app.PredictButton = uibutton(mainLayout, ...
    "push", ...
    "Text", "Predict", ...
    "FontSize", 14, ...
    "FontWeight", "bold", ...
    "ButtonPushedFcn", @onPredictButtonPushed);
app.PredictButton.Layout.Row = 4;
app.PredictButton.Layout.Column = 1;

app.ClearButton = uibutton(mainLayout, ...
    "push", ...
    "Text", "Clear", ...
    "FontSize", 14, ...
    "ButtonPushedFcn", @onClearButtonPushed);
app.ClearButton.Layout.Row = 4;
app.ClearButton.Layout.Column = 2;

app.ResultLabel = uilabel(mainLayout, ...
    "Text", "Result: Waiting for input", ...
    "FontSize", 16, ...
    "FontWeight", "bold", ...
    "HorizontalAlignment", "center");
app.ResultLabel.Layout.Row = 5;
app.ResultLabel.Layout.Column = [1 2];

app.ConfidenceLabel = uilabel(mainLayout, ...
    "Text", "Confidence Score: --", ...
    "FontSize", 14, ...
    "HorizontalAlignment", "center");
app.ConfidenceLabel.Layout.Row = 6;
app.ConfidenceLabel.Layout.Column = [1 2];

app.ModelStatusLabel = uilabel(mainLayout, ...
    "Text", "Model Status: Loading...", ...
    "FontSize", 12, ...
    "HorizontalAlignment", "center");
app.ModelStatusLabel.Layout.Row = 7;
app.ModelStatusLabel.Layout.Column = [1 2];

app.Figure.UserData = app;

%% Load Model
% The app loads the persisted classifier and TF-IDF model during startup so
% predictions are ready as soon as the UI appears.
try
    [svmModel, tfidfModel] = loadModel();

    app = app.Figure.UserData;
    app.SVMModel = svmModel;
    app.TFIDFModel = tfidfModel;
    app.ModelLoaded = true;
    app.ModelStatusLabel.Text = "Model Status: Loaded";
    app.ModelStatusLabel.FontColor = [0.00 0.45 0.20];
    app.Figure.UserData = app;

catch ME
    app = app.Figure.UserData;
    app.ModelLoaded = false;
    app.ModelStatusLabel.Text = "Model Status: Failed to load model";
    app.ModelStatusLabel.FontColor = [0.75 0.00 0.00];
    app.PredictButton.Enable = "off";
    app.Figure.UserData = app;

    uialert(app.Figure, ...
        sprintf("Unable to load the trained model.\n\n%s", ME.message), ...
        "Model Loading Error", ...
        "Icon", "error");
end

%% Local Callback Functions
function onPredictButtonPushed(source, ~)
%ONPREDICTBUTTONPUSHED Validate input text and run model inference.

fig = ancestor(source, "figure");
app = fig.UserData;

try
    if ~app.ModelLoaded
        error("FakeNewsDetectionApp:ModelNotLoaded", ...
            "The trained model is not loaded. Train and save the model before predicting.");
    end

    inputLines = string(app.NewsTextArea.Value);
    newsText = strtrim(strjoin(inputLines, newline));

    if strlength(newsText) == 0
        uialert(fig, ...
            "Please enter news text before clicking Predict.", ...
            "Empty Input", ...
            "Icon", "warning");
        return;
    end

    [predictedLabel, confidenceScore] = predictNews( ...
        newsText, ...
        app.SVMModel, ...
        app.TFIDFModel);

    if predictedLabel == 0
        app.ResultLabel.Text = "Result: FAKE NEWS";
        app.ResultLabel.FontColor = [0.75 0.00 0.00];
    elseif predictedLabel == 1
        app.ResultLabel.Text = "Result: REAL NEWS";
        app.ResultLabel.FontColor = [0.00 0.45 0.20];
    else
        error("FakeNewsDetectionApp:UnexpectedPrediction", ...
            "The model returned an unsupported prediction label.");
    end

    app.ConfidenceLabel.Text = sprintf( ...
        "Confidence Score: %.2f%%", ...
        confidenceScore * 100);

    fig.UserData = app;

catch ME
    uialert(fig, ...
        sprintf("Prediction failed.\n\n%s", ME.message), ...
        "Prediction Error", ...
        "Icon", "error");
end

end

function onClearButtonPushed(source, ~)
%ONCLEARBUTTONPUSHED Reset input and output UI elements.

fig = ancestor(source, "figure");
app = fig.UserData;

app.NewsTextArea.Value = "";
app.ResultLabel.Text = "Result: Waiting for input";
app.ResultLabel.FontColor = [0.15 0.15 0.15];
app.ConfidenceLabel.Text = "Confidence Score: --";

fig.UserData = app;

end

