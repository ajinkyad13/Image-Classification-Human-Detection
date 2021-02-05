clear
close all

% load and explore image data
trainCifarDatasetPath = 'cifar-100-data/train';
testCifarDatasetPath = 'cifar-100-data/test';
trainImds = imageDatastore(trainCifarDatasetPath,...
    'IncludeSubfolders', true, 'LabelSource','foldernames');
testImds = imageDatastore(testCifarDatasetPath,...
    'IncludeSubfolders', true, 'LabelSource','foldernames');
    
% Calculate the number of images in each category.
% labelCount is a table that contains the labels and the number of images having each label.
trainLabelCount = countEachLabel(trainImds);
testLabelCount = countEachLabel(testImds);

% get size (all images are the same size)
img = readimage(trainImds,1);
[len, wid, chan] = size(img);

% Define the convolutional neural network architecture.
filter_size = 3;
num_filters = 8;

layers = [
    imageInputLayer([len wid chan]) % specify the image size
    
    convolution2dLayer(filter_size, num_filters, 'Padding', 'same') % convolution layer
    batchNormalizationLayer % normalize the activations and gradients propagating through a network
    reluLayer % nonlinear activation function
    dropoutLayer(.2) % randomly sets input elements to zero with a given probability
    
    maxPooling2dLayer(2,'Stride',2) % returns the maximum values of rectangular regions of inputs
    
    convolution2dLayer(filter_size, num_filters*2, 'Padding', 'same') % convolution layer
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.2)
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(filter_size, num_filters*4, 'Padding', 'same') % convolution layer
    batchNormalizationLayer
    reluLayer
    dropoutLayer(.2)
    
    fullyConnectedLayer(5) % the neurons connect to all the neurons in the preceding layer
    softmaxLayer % normalizes the output of the fully connected layer
    classificationLayer % assign the input to one of the mutually exclusive classes and compute the loss
];

% Specify training options
options = trainingOptions('rmsprop', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',10, ...
    'ValidationData',testImds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress' ...
);

% Train network using training data
net = trainNetwork(trainImds,layers,options);

% Classify Validation Images and Compute Accuracy
YPred = classify(net,testImds);
YValidation = testImds.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);