 rng(0);
 shuffledIndices = randperm(height(image_bbox));
 idx = floor(0.8 * length(shuffledIndices) );

 trainingIdx = 1:idx;
 trainingDataTbl = image_bbox(shuffledIndices(trainingIdx),:);

 validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
 validationDataTbl = image_bbox(shuffledIndices(validationIdx),:);

 imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
 bldsTrain = boxLabelDatastore(trainingDataTbl(:,{'TV','Bottle','TV_Remote','Pen'}));

 imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
 bldsValidation = boxLabelDatastore(validationDataTbl(:,{'TV','Bottle','TV_Remote','Pen'}));

 trainingData = combine(imdsTrain,bldsTrain);
 validationData = combine(imdsValidation,bldsValidation);

 inputSize = [128 128 3];
 numClasses = 4;

 trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
 numAnchors = 7;
 [anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

 augmentedTrainingData = transform(trainingData,@augmentData);
 preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
 preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

 inputLayer = imageInputLayer(inputSize,'Name','input','Normalization','none');
 filterSize = [3 3];

 middleLayers = [
     convolution2dLayer(filterSize, 16, 'Padding', 1,'Name','conv_1','WeightsInitializer','narrow-normal')
     batchNormalizationLayer('Name','BN1')
     reluLayer('Name','relu_1')
     maxPooling2dLayer(2, 'Stride',2,'Name','maxpool1')
     convolution2dLayer(filterSize, 32, 'Padding', 1,'Name', 'conv_2','WeightsInitializer','narrow-normal')
     batchNormalizationLayer('Name','BN2')
     reluLayer('Name','relu_2')
     maxPooling2dLayer(2, 'Stride',2,'Name','maxpool2')
     convolution2dLayer(filterSize, 64, 'Padding', 1,'Name','conv_3','WeightsInitializer','narrow-normal')
     batchNormalizationLayer('Name','BN3')
     reluLayer('Name','relu_3')
     maxPooling2dLayer(2, 'Stride',2,'Name','maxpool3')
     convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_4','WeightsInitializer','narrow-normal')
     batchNormalizationLayer('Name','BN4')
     reluLayer('Name','relu_4')];

 lgraph = layerGraph([inputLayer; middleLayers]);

 lgraph = yolov2Layers (inputSize,numClasses,anchorBoxes,lgraph,'relu_4');

 options = trainingOptions('sgdm', ...
         'InitialLearnRate',0.001, ...
         'Verbose',true,'MiniBatchSize',16,'MaxEpochs',80,...
         'Shuffle','every-epoch','VerboseFrequency',50, ...
         'ExecutionEnvironment','gpu','Validationdata',preprocessedValidationData);
    
 [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);


img = imread("test_6.jpg");
img_1 = imresize(img,[128,128]);
[bboxes,scores,labels] = detect(detector,img_1);
I = insertObjectAnnotation(img_1,'Rectangle',bboxes,cellstr(labels));
imshow(I)








