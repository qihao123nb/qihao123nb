clc; clear; close all;
unzip("DigitsData.zip");
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
%%
% Split data into training and testing sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Define image augmentation options (Rotation Augmentation)
augmenter = imageDataAugmenter( ...
    'RandRotation', [-30, 30]);

% Create augmented image datastore
augimdsTrain = augmentedImageDatastore([28,28], imdsTrain, 'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore([28,28], imdsTest);  % No augmentation on test data
%%
% Define CNN Architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Define Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false);
%%
% Train Model without Augmentation
disp('Training Baseline Model...');
baselineNet = trainNetwork(imdsTrain, layers, options);

% Train Model with Rotation Augmentation
disp('Training Model with Rotation Augmentation...');
augNet = trainNetwork(augimdsTrain, layers, options);

% Evaluate Model Performance
disp('Evaluating Baseline Model...');
baselinePred = classify(baselineNet, imdsTest);
baselineAccuracy = mean(baselinePred == imdsTest.Labels) * 100;

disp('Evaluating Augmented Model...');
augPred = classify(augNet, imdsTest);
augAccuracy = mean(augPred == imdsTest.Labels) * 100;

% Display Results
fprintf('Baseline Model Accuracy: %.2f%%\n', baselineAccuracy);
fprintf('Augmented Model Accuracy (Rotation): %.2f%%\n', augAccuracy);

