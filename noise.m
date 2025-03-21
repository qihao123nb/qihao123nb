clc; clear; close all;

%% Create imageDatastores (with Gaussian noise augmentation)
rootFolderTrain = './images/cifar10Train';
imdsTrain = imageDatastore(rootFolderTrain, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @addGaussianNoise);  % Custom ReadFcn adds noise

rootFolderTest = './images/cifar10Test';
imdsTest = imageDatastore(rootFolderTest, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Resize images
augimdsTrain = augmentedImageDatastore([32 32 3], imdsTrain);
augimdsTest = augmentedImageDatastore([32 32 3], imdsTest);

%% CNN Architecture
layers = [
    imageInputLayer([32 32 3])

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train CNN with Gaussian Noise Augmentation
disp('Training model with Gaussian noise augmentation...');
net_noise = trainNetwork(augimdsTrain, layers, options);

%% Evaluate model
preds_noise = classify(net_noise, augimdsTest);
acc_noise = mean(preds_noise == imdsTest.Labels) * 100;
fprintf('Gaussian Noise Model Accuracy: %.2f%%\n', acc_noise);

%% Custom Read Function: Add Gaussian Noise
function imgOut = addGaussianNoise(filename)
    img = imread(filename);
    img = im2double(img); % Convert to double
    noise_sigma = 0.05; % Adjust noise level here (standard deviation)
    imgNoisy = imnoise(img, 'gaussian', 0, noise_sigma^2);
    imgOut = im2uint8(imgNoisy); % Convert back to uint8
end
