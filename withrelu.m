%% Experiment: Approximation of x^3 using a network without ReLU

% Define the function to be approximated
fnc = @(x) x.^3;

% Define the training data
xTrain = linspace(-1, 1, 80)';
yTrain = fnc(xTrain);

% Define the network architecture without any activation function
layers = [
    featureInputLayer(1)  
    fullyConnectedLayer(4)
    fullyConnectedLayer(1)
];
%%
% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 1000, ...
    'ExecutionEnvironment', 'cpu', ...
    'Plots', 'none');

% Train the network
trainedNet = trainnet(xTrain, yTrain, layers, "mean-squared-error", options);

% Generate test data and predict outputs
numRand = 100;    
XTest = sort(2 .* rand(numRand, 1) - 1);   
YTest = predict(trainedNet, XTest);
%%
% Plot the exact function vs. the network prediction
figure;
plot(xTrain, yTrain, '-sblack', XTest, YTest, '-vr');
xr = xregion(-1, 1);
legend('Exact', 'Predicted (No ReLU)')
grid on;
xlabel('x')
ylabel('f(x) = x^3')

%%
network with ReLU activation

% Define the function to be approximated
fnc = @(x) x.^3;

% Define the training data
xTrain = linspace(-1, 1, 80)';
yTrain = fnc(xTrain);
%%
% Define the network architecture with a ReLU activation layer
layers = [
    featureInputLayer(1)  
    fullyConnectedLayer(4)
    reluLayer
    fullyConnectedLayer(1)
];

% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 1000, ...
    'ExecutionEnvironment', 'cpu', ...
    'Plots', 'none');
%%
% Train the network
trainedNet_ReLU = trainnet(xTrain, yTrain, layers, "mean-squared-error", options);

% Generate test data and predict outputs
numRand = 100;    
XTest = sort(2 .* rand(numRand, 1) - 1);   
YTest_ReLU = predict(trainedNet_ReLU, XTest);

% Plot the exact function vs. the network prediction
figure;
plot(xTrain, yTrain, '-sblack', XTest, YTest_ReLU, '-vr');
xr = xregion(-1, 1);
legend('Exact', 'Predicted (With ReLU)')
grid on;
xlabel('x')
ylabel('f(x) = x^3')