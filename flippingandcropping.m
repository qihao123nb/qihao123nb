clc; clear; close all;
unzip("images.zip");

% Create an imageDatastore for training data
rootFolderTrain = './images/cifar10Train';
imdsTrain = imageDatastore(rootFolderTrain, "IncludeSubfolders", true, 'LabelSource', 'foldernames');

% Create an imageDatastore for testing data
rootFolderTest = "./images/cifar10Test";
imdsTest = imageDatastore(rootFolderTest, "IncludeSubfolders", true, "LabelSource", "foldernames");

%% Define Image Augmentation: Flipping + Cropping
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...  
    'RandYReflection', false);    

% Create augmented image datastore with random cropping
augimdsTrain = augmentedImageDatastore([32,32,3], imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'OutputSizeMode', 'randcrop');

% For testing, simply resize the images without augmentation
augimdsTest = augmentedImageDatastore([32,32,3], imdsTest);

%% Define CNN Architecture 
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

%% Define Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train the Model with Flipping + Cropping Augmentation
disp('Training Model with Flipping + Cropping Augmentation on CIFAR-10...');
augNet = trainNetwork(augimdsTrain, layersSimplified, options);

% Evaluate Augmented Model Performance
disp('Evaluating Augmented Model...');
augPred = classify(augNet, imdsTest);
augAccuracy = mean(augPred == imdsTest.Labels) * 100;

% Display final accuracy
fprintf('Flipping + Cropping Model Accuracy: %.2f%%\n', augAccuracy);
