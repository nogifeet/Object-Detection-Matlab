rng(0);
shuffledIndices = randperm(height(image_bbox));
idx = floor(0.6 * length(shuffledIndices));

trainingDataTbl = image_bbox(shuffledIndices(1:idx), :);
testDataTbl = image_bbox(shuffledIndices(idx+1:end), :);

imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);

bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

validateInputData(trainingData);
validateInputData(testData);

augmentedTrainingData = transform(trainingData, @augmentData);

networkInputSize = [227 227 3];

preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocessData(data, networkInputSize));

rng(0)
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 6;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxMasks = {[1,2,3]
    [4,5,6]
    };

baseNetwork = squeezenet;
lgraph = squeezenetFeatureExtractor(baseNetwork, networkInputSize);

classNames = {'TV','Bottle','TV_Remote','Pen'};
numClasses = size(classNames, 2);
numPredictorsPerAnchor = 5 + numClasses;

lgraph = addFirstDetectionHead(lgraph, anchorBoxMasks{1}, numPredictorsPerAnchor);
lgraph = addSecondDetectionHead(lgraph, anchorBoxMasks{2}, numPredictorsPerAnchor);

lgraph = connectLayers(lgraph, 'fire9-concat', 'conv1Detection1');
lgraph = connectLayers(lgraph, 'relu1Detection1', 'upsample1Detection2');
lgraph = connectLayers(lgraph, 'fire5-concat', 'depthConcat1Detection2/in2');

networkOutputs = ["conv2Detection1"
    "conv2Detection2"
    ];

numEpochs = 70;
miniBatchSize = 8;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);

doTraining=true;

if doTraining
    % Convert layer graph to dlnetwork.
    net = dlnetwork(lgraph);
    
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);

    iteration = 0;
    % Custom training loop.
    for epoch = 1:numEpochs
          
        reset(mbqTrain);
        shuffle(mbqTrain);
        
        while(hasdata(mbqTrain))
            iteration = iteration + 1;
           
            [XTrain, YTrain] = next(mbqTrain);
            
            % Evaluate the model gradients and loss using dlfeval and the
            % modelGradients function.
            [gradients, state, lossInfo] = dlfeval(@modelGradients, net, XTrain, YTrain, anchorBoxes, anchorBoxMasks, penaltyThreshold, networkOutputs);
    
            % Apply L2 regularization.
            gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
    
            % Determine the current learning rate value.
            currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
    
            % Update the network learnable parameters using the SGDM optimizer.
            [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);
    
            % Update the state parameters of dlnetwork.
            net.State = state;
              
            % Display progress.
            displayLossInfo(epoch, iteration, currentLR, lossInfo);  
                
            % Update training plot with new points.
            updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
        end        
    end
end

% Read the datastore.
reset(preprocessedTestData)
data = read(preprocessedTestData);

% Get the image.
I = data{1};

% Convert to dlarray.
XTest = dlarray(I, 'SSCB');

executionEnvironment = "auto";

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest = gpuArray(XTest);
end

[bboxes, scores, labels] = yolov3Detect(net, XTest, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);

% Clear the persistent variables used in the yolov3Detect function to avoid retaining their values in memory.
clear yolov3Detect  

% Display the detections on image.
if ~isempty(scores{1})
    I = insertObjectAnnotation(I, 'rectangle', bboxes{1}, scores{1});
end
figure
imshow(I)