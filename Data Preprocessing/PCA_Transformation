%% Prepare the data to be used
data = readtable('creditcardori.csv');

% Select features and labels columns from the imported data in data variable
features = data(:, 1:30);
labels = data(:, end);

% Convert data to arrays
X = table2array(features);
y = table2array(labels);

%% Scale columns 1 and 30 to have values between -1 and 1
X_scaled = X;
X_scaled(:, [1, 30]) = rescale(X_scaled(:, [1, 30]), -1, 1);

%% Perform PCA on columns 1 and 30 of the scaled data
[coeff, score] = pca(X_scaled(:, [1, 30]));

% Select the transformed features after PCA
X_pca = score(:, 1:end-1);

% Combine PCA-transformed features with remaining features
X_combined = [X_pca, X(:, [2:29, 31:end])];

% Create a new table with the transformed features
data_pca = array2table(X_combined);

% Get the original variable names from the first 30 columns of the data
original_variable_names = features.Properties.VariableNames;

% Update the variable names of the features
features_with_original_names = [compose('Time'), original_variable_names(2:end)];

% Create a new table with the top features and their original variable names
data_features = [array2table(X_pca), features(:, 2:end)];
data_features.Properties.VariableNames = features_with_original_names;

% Add the 'Class' column from the original dataset to the table
data_features.Class = data.Class;

% Specify the name of the output file
output_file = 'creditcard_pca.csv';

% Write the table to a CSV file
writetable(data_features, output_file);
