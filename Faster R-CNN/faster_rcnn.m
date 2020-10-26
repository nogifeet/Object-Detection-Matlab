rng(0)
shuffledIndices = randperm(height(image_bbox));
idx = floor(0.6 * height(image_bbox));

trainingIdx = 1:idx;
trainingDataTbl = image_bbox(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = image_bbox(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = image_bbox(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,{'TV','Bottle','TV_Remote','Pen'}));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation  = boxLabelDatastore(validationDataTbl(:,{'TV','Bottle','TV_Remote','Pen'}));


imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,{'TV','Bottle','TV_Remote','Pen'}));


trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

inputSize = [224 224 3];
numClasses=4;

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

augmentedTrainingData = transform(trainingData,@augmentData);

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);

options = trainingOptions('sgdm',...
    'MaxEpochs',20,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);

 %[detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        %'NegativeOverlapRange',[0 0.3], ...
        %'PositiveOverlapRange',[0.6 1]);


%% make_predictions
img = imread("1.jpg");
img_1 = imresize(img,[224,224]);
[bboxes,scores,labels] = detect(detector,img_1);
I = insertObjectAnnotation(img_1,'Rectangle',bboxes,cellstr(labels));
imshow(I)

    
 


