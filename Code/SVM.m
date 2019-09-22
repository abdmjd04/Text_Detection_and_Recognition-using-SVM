clear all; clc; close all;
load('./Lists/English/Img/lists.mat');

trainData=list.ALLnames(list.TRNind(:,end),:);

for i=1:size(trainData,1)
    img=imread(['./EnglishImg/English/Img/', trainData(i,:),'.png']);
    if size(img,3)==3
        img=histeq(rgb2gray(img));
    end
    TrainImgs{i}=imbinarize(imresize(img,[28 28]));
    whiteindexes=find(TrainImgs{i}==1);
    blackindexes=find(TrainImgs{i}==0);
    if size(whiteindexes)<size(blackindexes)
        TrainImgs{i}=imcomplement(TrainImgs{i});
    end
end


cellSize = [4 4];

TrainFeatures = [];
TrainLabels=list.ALLlabels(list.TRNind(:,end),:);

for i = 1:size(trainData,1)
    TrainFeatures(i, :) = extractHOGFeatures(TrainImgs{i}, 'CellSize', cellSize);
end

% SVM 
classifier = fitcecoc(TrainFeatures, TrainLabels);

testingFiles=list.ALLnames(list.TSTind(:,end),:);

for i=1:size(testingFiles,1)
    img=imread(['./English/Img/', testingFiles(i,:),'.png']);
    if size(img,3)==3
        img=histeq(rgb2gray(img));
    end
    xTestImages{i}=imbinarize(imresize(img,[28 28]));
    whiteindexes=find(xTestImages{i}==1);
    blackindexes=find(xTestImages{i}==0);
    if size(whiteindexes)<size(blackindexes)
        xTestImages{i}=imcomplement(xTestImages{i});
    end
end

testingLabels=list.ALLlabels(list.TSTind(:,end),:);

for i = 1:size(testingFiles,1)
    testingFeatures(i, :) = extractHOGFeatures(xTestImages{i}, 'CellSize', cellSize);
end

% class predictions
predictedLabels = predict(classifier, testingFeatures);
 
%confusion matrix.
confMat = confusionmat(testingLabels, predictedLabels);

precision = diag(confMat)./sum(confMat,2);
precisionMean = mean(precision)