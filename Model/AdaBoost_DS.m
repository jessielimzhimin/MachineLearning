file = 'creditcard_topfeatures.csv';

% Read the dataset into a table
data = readtable(file);

features = data(:, 1:25);
target = data(:, 26);

X = table2array(features);
y = table2array(target);

cv = cvpartition(size(X, 1), 'Holdout', 0.2);  % 80% training, 20% testing
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Set the number of weak classifiers (decision stumps)
numDS = 100;

% Initialize the ensemble of weak classifiers
classifiers = cell(numDS, 1);

% Initialize weights for the samples
weights = ones(size(X_train, 1), 1) / size(X_train, 1);

for t = 1:numDS
    % Train a decision stump (weak classifier)
    stump = fitctree(X_train, y_train, 'Weights', weights, 'SplitCriterion', 'deviance', 'MaxNumSplits', 1);
    
    % Make predictions on the training data
    y_pred = predict(stump, X_train);
    
    % Compute the weighted error
    penaltyFactor = 2; % Penalty factor when classifier makes the wrong prediction
    error = sum(weights(y_pred ~= y_train));
    
    % Compute the classifier weight
    alpha = 0.5 * log((1 - error) / (error * penaltyFactor));
    
    % Update the sample weights
    weights = weights .* exp(-alpha * y_train .* y_pred);
    
    % Store the weak classifier and its weight in the ensemble
    classifiers{t} = struct('classifier', stump, 'weight', alpha);
end

% Classify the testing data
y_pred_test = zeros(size(y_test));

for i = 1:size(X_test, 1)
    % Initialize the sum of weighted predictions
    sumWeightedPredictions = 0;
    
    % Loop through each weak classifier in the ensemble
    for t = 1:numDS
        % Get the current classifier and its weight
        classifier = classifiers{t}.classifier;
        weight = classifiers{t}.weight;
        
        % Make a prediction using the current classifier
        prediction = predict(classifier, X_test(i, :));
        sumWeightedPredictions = sumWeightedPredictions + weight * prediction;
    end
    
    % Take the sign of the sum of weighted predictions as the final classification result
    y_pred_test(i) = sign(sumWeightedPredictions);
end

% Calculate the predicted scores for each weak classifier
y_scores = zeros(size(X_test, 1), 1);

for i = 1:numDS
    y_scores = y_scores + classifiers{i}.weight * predict(classifiers{i}.classifier, X_test);
end

% Calculate the false positive rate (FPR) and true positive rate (TPR)
[fpr, tpr, ~, auc] = perfcurve(y_test, y_scores, 1);

% Evaluate the performance using F1 score, accuracy, recall, and AUC
tp = sum(y_pred_test(y_test == 1) == 1);
fp = sum(y_pred_test(y_test == 0) == 1);
tn = sum(y_pred_test(y_test == 0) == 0);
fn = sum(y_pred_test(y_test == 1) == 0);

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);
accuracy = sum(y_pred_test == y_test) / numel(y_test);
MCC = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

disp(['F1 Score: ' num2str(f1_score)]);
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Recall: ' num2str(recall)]);
disp(['Precision: ' num2str(precision)]);
disp(['Matthews Correlation Coefficient (MCC): ' num2str(MCC)]);
disp(['Area Under the Curve (AUC): ' num2str(auc)]);

% Plot the ROC curve
plot(fpr, tpr)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Receiver Operating Characteristic (ROC) Curve')
legend(['AUC = ' num2str(auc)])