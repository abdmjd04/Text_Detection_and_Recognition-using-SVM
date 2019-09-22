%% Detection part of the code was used from mathworks website.
clc;clear all;close all;
load('SVM.mat');
%colorImage = imread('img1.jpg');
colorImage = imread('handicapsign.jpg');
I = rgb2gray(colorImage);
I = histeq(I);

% Detect MSER regions.
[mserRegions, mserConnComp] = detectMSERFeatures(I, ...
    'RegionAreaRange',[200 8000],'ThresholdDelta',4);

figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions')
hold off

%%
% Use regionprops to measure MSER properties
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
    'Solidity', 'Extent', 'Euler', 'Image');

%use of bounding box data to compute aspect ratio
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRatio = w./h;

% Threshold
filterIdx = aspectRatio' > 3;
filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
filterIdx = filterIdx | [mserStats.Solidity] < .3;
filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
filterIdx = filterIdx | [mserStats.EulerNumber] < -8;

% Removing the non-text regions
mserStats(filterIdx) = [];
mserRegions(filterIdx) = [];

% Showing the remaining regions
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Geometric Properties')
hold off


% binarizing the image and padding
regionImage = mserStats(6).Image;
regionImage = padarray(regionImage, [1 1]);

% Computing the stroke width image.
distanceImage = bwdist(~regionImage);
figure; imshow(distanceImage);
skeletonImage = bwmorph(regionImage, 'thin', inf);
figure; imshow(skeletonImage);

strokeWidthImage = distanceImage;
strokeWidthImage(~skeletonImage) = 0;
figure; imshow(strokeWidthImage);

figure
subplot(1,2,1)
imagesc(regionImage)
title('Region Image')

subplot(1,2,2)
imagesc(strokeWidthImage)
title('Stroke Width Image')

% Computing stroke width variation metric
strokeWidthValues = distanceImage(skeletonImage);
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

% Thresholding the stroke width variation metric
strokeWidthThreshold = 0.4;
strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

% Processing the  remaining regions to find similar pattern
for j = 1:numel(mserStats)

    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1], 0);

    distanceImage = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);

    strokeWidthValues = distanceImage(skeletonImage);

    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

end

mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];

% Showing the  remaining regions
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Stroke Width Variation')
hold off


%%

% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);

bboxes = sortrows(bboxes,[1 2]);

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax]
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Clip the bounding boxes
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

% Show the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
expandedBBoxes = ceil(expandedBBoxes);

for i=1:size(expandedBBoxes,1)
    characterImages{i}=imresize(I(expandedBBoxes(i,2):(expandedBBoxes(i,2)+expandedBBoxes(i,4)-1) , expandedBBoxes(i,1):(expandedBBoxes(i,1)+expandedBBoxes(i,3)-1)), [28 28]);
end

IExpandedBBoxes = insertShape(colorImage,'Rectangle',expandedBBoxes,'LineWidth',3);

figure
imshow(IExpandedBBoxes)
title('Expanded Bounding Boxes Text')

%% Recognition part

cellSize = [4 4];
for i = 1:size(characterImages,2)
    characterFeatures(i, :) = extractHOGFeatures(characterImages{i}, 'CellSize', cellSize);
end

% % Prediction
predictedLabels = predict(classifier, characterFeatures);

%%

characters={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};

finalstring=[];
for i=1:size(predictedLabels,1)
    finalstring=[finalstring, characters(predictedLabels(i))]
end
