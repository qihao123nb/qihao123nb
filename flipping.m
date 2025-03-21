clc; clear; close all;
unzip("images.zip")
clear
% Taken from https://uk.mathworks.com/matlabcentral/fileexchange/62990-deep-learning-tutorial-series
tic
% Create an imageDatastore pointing to training data
rootFolderTrain = './images/cifar10Train';
imdsTrain = imageDatastore(rootFolderTrain,"IncludeSubfolders",true,'LabelSource', 'foldernames');

% Create an imageDatastore pointing to testing data
rootFolderTest = "./images/cifar10Test";
imdsTest = imageDatastore(rootFolderTest,"IncludeSubfolders",true,"LabelSource","foldernames");

%%
% Define Image Augmentation with Flipping
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ... % Apply random horizontal flipping
    'RandYReflection', false);   % No vertical flipping

% Create augmented image datastore
augimdsTrain = augmentedImageDatastore([32,32,3], imdsTrain, 'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore([32,32,3], imdsTest);  % No augmentation on test set

%%
% Define CNN Architecture 
layersSimplified = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(3,16,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);  
%%

% Train Baseline Model (No Augmentation)
disp('Training Baseline Model on CIFAR-10...');
baselineNet = trainNetwork(imdsTrain, layers, options);

% Train Model with Flipping Augmentation
disp('Training Model with Flipping Augmentation on CIFAR-10...');
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
fprintf('Flipping-Augmented Model Accuracy: %.2f%%\n', augAccuracy);
