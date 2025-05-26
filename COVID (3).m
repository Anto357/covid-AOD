% Define the GitHub repository and dataset location
repoURL = "https://github.com/sanjos146/covid/archive/refs/heads/main.zip";
dataFolder = "covid-main/chest_xray";

% Define local paths for downloaded and extracted data
zipFile = "dataset.zip";
extractFolder = "dataset";

% Download and extract the dataset
if ~isfolder(extractFolder)
    fprintf("Downloading dataset...\n");
    websave(zipFile, repoURL);
    unzip(zipFile, extractFolder);
    fprintf("Dataset downloaded and extracted.\n");
end

% Set paths to training and testing folders
trainFolder = fullfile(extractFolder, dataFolder, "train");
testFolder = fullfile(extractFolder, dataFolder, "test");

% Image preprocessing parameters
imageSize = [64, 64]; % Resize images

% Load training data
[trainFeatures, trainLabels, trainCount] = loadData(trainFolder, imageSize);

% Visualize data distribution
categories = {'Normal', 'Pneumonia'};
figure;
bar(trainCount, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', categories, 'FontSize', 12);
xlabel('Category');
ylabel('Number of Images');
title('Data Distribution in Training Set');

% Perform PCA on the training data
[coeff, score, ~, ~, explained] = pca(trainFeatures);

% Visualize the PCA results in 2D
figure;
gscatter(score(:, 1), score(:, 2), trainLabels, 'rgb', 'xo', 8);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA of Training Features');
legend({'Normal', 'Pneumonia'});
grid on;

% Load testing data
[testFeatures, testLabels] = loadData(testFolder, imageSize);

% Train Random Forest
numTrees = 100; % Number of trees in the forest
randomForestModel = TreeBagger(numTrees, trainFeatures, trainLabels, ...
    'Method', 'classification', 'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');

% Test the model
predictedLabels = predict(randomForestModel, testFeatures);
predictedLabels = str2double(predictedLabels); % Convert cell array to numeric

% Evaluate performance
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
confMat = confusionmat(testLabels, predictedLabels);

% Plot confusion matrix
figure;
heatmap(categories, categories, confMat, ...
    'Colormap', jet, 'ColorbarVisible', 'on');
xlabel('Predicted Labels');
ylabel('True Labels');
title('Confusion Matrix');

% Plot out-of-bag error
figure;
oobError = oobError(randomForestModel);
plot(oobError, 'LineWidth', 1.5, 'Color', [0.8, 0.2, 0.2]);
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');
title('Out-of-Bag Error as Trees Grow');

% Feature importance
featureImportance = randomForestModel.OOBPermutedPredictorDeltaError;
figure;
bar(featureImportance, 'FaceColor', [0.2, 0.8, 0.4]);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance');
grid on;

% Helper function to load data
function [features, labels, categoryCounts] = loadData(folderPath, imageSize)
    categories = {'NORMAL', 'PNEUMONIA'};
    numCategories = numel(categories);
    imageData = [];
    imageLabels = [];
    categoryCounts = zeros(1, numCategories);
    for i = 1:numCategories
        categoryFolder = fullfile(folderPath, categories{i});
        imageFiles = dir(fullfile(categoryFolder, '*.jpeg')); % Adjust extension if needed
        categoryCounts(i) = numel(imageFiles);
        for j = 1:numel(imageFiles)
            img = imread(fullfile(categoryFolder, imageFiles(j).name));
            img = imresize(img, imageSize); % Resize image
            if size(img, 3) == 3
                img = rgb2gray(img); % Convert to grayscale if needed
            end
            imageData = [imageData; img(:)']; % Flatten and add to dataset
            imageLabels = [imageLabels; i - 1]; % Assign labels (0 for NORMAL, 1 for PNEUMONIA)
        end
    end
    features = double(imageData);
    labels = double(imageLabels);
end