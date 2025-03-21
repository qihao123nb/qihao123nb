%% Experiment: Approximation of x^3 using Leaky ReLU activation

% Define the function to be approximated
fnc = @(x) x.^3;

% Define the training data
xTrain = linspace(-1, 1, 80)';
yTrain = fnc(xTrain);
%%
% Define the network architecture with a Leaky ReLU activation layer (alpha=0.01)
layers = [
    featureInputLayer(1)  
    fullyConnectedLayer(8)
    leakyReluLayer(0.01)
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
trainedNet_LeakyReLU = trainnet(xTrain, yTrain, layers, "mean-squared-error", options);

% Generate test data and predict outputs
numRand = 100;    
XTest = sort(2 .* rand(numRand, 1) - 1);   
YTest_LeakyReLU = predict(trainedNet_LeakyReLU, XTest);

% Plot the exact function vs. the network prediction
figure;
plot(xTrain, yTrain, '-sblack', XTest, YTest_LeakyReLU, '-vr');
xr = xregion(-1, 1); % For consistent axis range (if needed)
legend('Exact', 'Predicted (Leaky ReLU)')
grid on;
xlabel('x')
ylabel('f(x) = x^3')
