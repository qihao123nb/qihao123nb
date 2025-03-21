clc; clear; close all;


%% Create imageDatastores for training and testing data
% Training datastore using a custom ReadFcn to apply brightness & contrast jitter
rootFolderTrain = './images/cifar10Train';
imdsTrain = imageDatastore(rootFolderTrain, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @customReadFcn);

% Testing datastore (no augmentation applied here)
rootFolderTest = './images/cifar10Test';
imdsTest = imageDatastore(rootFolderTest, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Create augmented image datastores for resizing images
augimdsTrain_color = augmentedImageDatastore([32 32 3], imdsTrain);
augimdsTest = augmentedImageDatastore([32 32 3], imdsTest);

%% Define CNN architecture
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

%% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...             % Adjust epochs as needed
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.0001, ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train Model with Brightness & Contrast Augmentation
disp('Training model with brightness & contrast augmentation...');
net_color = trainNetwork(augimdsTrain_color, layers, options);

%% Evaluate the Model
preds_color = classify(net_color, augimdsTest);
acc_color = mean(preds_color == imdsTest.Labels) * 100;
fprintf('Color-Jitter Accuracy (brightness & contrast): %.2f%%\n', acc_color);

%% Custom Read Function for Color Jitter Augmentation
function out = customReadFcn(filename)
    % Read the image
    img = imread(filename);
    % Apply brightness & contrast jitter using jitterColorHSV
    out = jitterColorHSV(img, ...
        'Brightness', [-0.1 0.1], ...  % Random brightness offset between -0.2 and 0.2
        'Contrast', [0.9 1.1], ...      % Random contrast scaling between 0.8 and 1.2
        'Saturation', [1 1], ...        % Keep saturation unchanged
        'Hue', [0 0]);                  % Keep hue unchanged
end
