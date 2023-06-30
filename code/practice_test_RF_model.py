""" 
Project:        AI4ER MRes
Code:           Random Forest model for training model on S2 satellite data and CORINE reference data
Order:          After add_indices.py and reclassify_CORINE.py 
Time to run:    Hours
To do:          Optionally use CV (deactivate if just testing changes), make file path agnostic
"""

# For data manipulation
import numpy as np
import pandas as pd
from osgeo import gdal
# For modelling
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
# For evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
from wandb.sklearn import plot_confusion_matrix
from sklearn.metrics import classification_report
# For logging metrics to wandb
import wandb
# For saving output csv file
import joblib
# For command-line running
import argparse
import os
# For logging memory useage
import psutil

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # MB
    with open('RF_flat_L1_10b_memory.log', 'a') as f:
        f.write(f'{memory_usage}\n')

### SET UP ###
## Log results to W&B
# Note, entity is not specified, this default saves to a group project instead...
wandb.init(project='land-cover-classification', entity='maplumridge')

## Command-line arguments
parser = argparse.ArgumentParser(description='Land Cover Classification')
parser.add_argument('--satellite_files', nargs='*', help='List of satellite data files to generate time series stack (4 bands, 10 bands, 12 bands)', required=True)
parser.add_argument('--groundtruth_file', help='Ground truth data file (level 1)', required=True)
parser.add_argument('--L2_groundtruth_file', help='Lower level ground truth file for level 2 classification', required=True)
parser.add_argument('--L3_groundtruth_file', help='Lower level ground truth file for level 3 classification', required=True)
parser.add_argument('--pixels_per_class', type=int, help='Number of pixels per class to use for model training', required=False)
parser.add_argument('--label_type', choices=['L1', 'L2', 'L3'], default='L1', help='Labels to use for confusion matrix (level 1, 2 or 3 classes)', required=True)
parser.add_argument('--model_output', help='Name of output pkl file which model will be saved to', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
input_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'
# Location for saving the model and model predictions
output_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'

## LOAD DATA ##
# Load seasonal satellite data with 4, 10 or 12 bands. This will load 4 files for the year 2018.
satellite_files=args.satellite_files
satellite_files = [os.path.join(input_dir, file) for file in args.satellite_files]
# Load CORINE ground truth data with level 1 or 2 classes
groundtruth_file = os.path.join(input_dir, args.groundtruth_file)
# Load additional CORINE ground truth data with level 2 or 3 classes
L2_groundtruth_file = os.path.join(input_dir, args.L2_groundtruth_file)
# Load additional CORINE ground truth data with level 2 or 3 classes
L3_groundtruth_file = os.path.join(input_dir, args.L3_groundtruth_file)

### SATELLITE DATA ###
# Loop over seasonal satellite data and create a time-series stack for the entire year
satellite_data_stack = None
for file in satellite_files:
    satellite_image = gdal.Open(file)
    bands = satellite_image.RasterCount
    band_data = np.stack([satellite_image.GetRasterBand(i).ReadAsArray() for i in range(1, bands + 1)], axis=-1)
    if satellite_data_stack is None:
        satellite_data_stack = band_data
    else:
        satellite_data_stack = np.concatenate((satellite_data_stack, band_data), axis=-1)
# Reshape stacked satellite data into a 2D array, ready for ingestion by the RF model
rows, cols, bands = satellite_data_stack.shape
satellite_data_2d = satellite_data_stack.reshape(rows * cols, bands)

### GROUND TRUTH DATA ###
# Open ground truth data with gdal
groundtruth_image = gdal.Open(groundtruth_file)
# Read ground truth data as array and reshape/size to match satellite data
groundtruth_data = groundtruth_image.GetRasterBand(1).ReadAsArray()
rows, cols = groundtruth_data.shape
groundtruth_data = np.resize(groundtruth_data, (rows, cols))
groundtruth_data_flat = groundtruth_data.flatten()

### ADDITIONAL GROUND TRUTH DATA ###
# Needed for the dataframe for subsequent hierarchical modelling
# Level 2 classes
L2_groundtruth_image = gdal.Open(L2_groundtruth_file)
L2_groundtruth_data = L2_groundtruth_image.GetRasterBand(1).ReadAsArray()
L2_groundtruth_data_flat = L2_groundtruth_data.flatten()

# Level 3 classes
L3_groundtruth_image = gdal.Open(L3_groundtruth_file)
L3_groundtruth_data = L3_groundtruth_image.GetRasterBand(1).ReadAsArray()
L3_groundtruth_data_flat = L3_groundtruth_data.flatten()

### CREATE DATA FRAME ###
# Combine satellite time-series, ground truth data, and L2 ground truth data
df = pd.DataFrame(satellite_data_2d, columns=['band'+str(i) for i in range(1, bands+1)])
df['groundtruth'] = groundtruth_data_flat
df['L2_groundtruth'] = L2_groundtruth_data_flat
df['L3_groundtruth'] = L3_groundtruth_data_flat
# To check
print("Data frame before NaN removal (L1, L2, L3, & satellite data)", df)
# Remove rows with NaN values
df.dropna(inplace=True)
print("Data frame post-NaN removal (L1, L2,L3 &  satellite data):", df)


### ORGANISE DATA FOR MODELLING ###
# DF sampling: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
# DF axes management: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
# DF count: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
# DF drop: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# DF drop NaN: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
# DF grouping: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# DF apply & lambda : https://sparkbyexamples.com/pandas/pandas-apply-with-lambda-examples/

# First, shuffle the data before train/test split, to randomise selection of data for training and testing
# https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
# https://stackoverflow.com/questions/55072435/shuffle-a-dataframe-while-keeping-internal-order
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
shuffled_df = df.sample(frac=1, random_state=42)
# To check (note, index values are maintained)
print("Shuffled data frame:", shuffled_df) 

# I want to use the same number of pixels for each class (oversampling/undersampling), to create a balanced model
# Either specify this as a command-line arg (if wanting to train the model on a small subset while developing iteratively)
# Or calculate the min number in the script and apply
if args.pixels_per_class is not None:
    min_pixels_per_class = args.pixels_per_class
else:
    pixels_per_class = shuffled_df['groundtruth'].value_counts()
    min_pixels_per_class = pixels_per_class.min()

print("Min pixels per class:", min_pixels_per_class)

# Now, group shuffled data by 'groundtruth' class 
# & sample min_pixels_per_class
# don't want to reset index 
# see https://stackoverflow.com/questions/51866908/difference-between-as-index-false-and-reset-index-in-pandas-groupby
df_sample = shuffled_df.groupby('groundtruth', as_index=False).apply(lambda x: x.sample(n=min_pixels_per_class, replace=True, random_state=42)).reset_index(level=0, drop=True)

# Now, automatically choose a random state to initialise the model using the random module
random_state = random.randint(0, 100000)
# Make a note of it for reproducibility
print("Random State:", random_state)

# Split the data into testing and training
# groundtruth is removed from the satellite data
# groundtruth is used for training and comparing predictions
X_train, X_test, y_train, y_test = train_test_split(
    df_sample.drop(['groundtruth'], axis=1),
    df_sample['groundtruth'],
    test_size=0.2,
    random_state=random_state
)

## CHECKS ##
# Training data
print("Shuffled Training Data:")
print("Satellite training data:", X_train)
print("Ground truth training data:", y_train)
# Validation/testing data
print("Shuffled Testing Data:")
print("Satellite testing data:", X_test)
print("Ground truth testing data:", y_test)

# Calculate number of pixels per class, following train/test split to check class balance
train_pixels_per_class = y_train.value_counts()
test_pixels_per_class = y_test.value_counts()
print("Number of pixels per class (Training Data):")
print(train_pixels_per_class)
print("Number of pixels per class (Testing Data):")
print(test_pixels_per_class)

# Remove the 'L2_groundtruth' column from X_train, maintain for X_test
# (this will later be required for hierarchical modelling)
# Create a copy of X_test (with 'L2_groundtruth', for use in CSV file)
X_test_CSV = X_test.copy()
# Ready for modelling
X_test.drop(['L2_groundtruth', 'L3_groundtruth'], axis=1, inplace=True)
X_train.drop(['L2_groundtruth', 'L3_groundtruth'], axis=1, inplace=True)


## MODELING ##
# First, define the parameter grid for 5-fold grid-search cross-validation
# These are the specific parameter values that I want to test

## grid search vs random search : https://www.kdnuggets.com/2022/10/hyperparameter-tuning-grid-search-random-search-python.html
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [5, 15, 30],
    'min_samples_leaf': [1, 3, 5]
}

# Create RF model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Train RF model (note that I am using 5 folds for grid search cross validation)
# Maybe also modify this to be a command-line argument...?
# Adding verbose mode https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(X_train, y_train)
# Find best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
# Make predictions on test data using best model
y_pred = best_model.predict(X_test)


### EVALUATION ###
# Log results to W&B

## Test metrics
# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Hyperparameter metrics from grid search CV
# See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
print("SKlearn metrics:")
# Print all hyperparameter combinations and their results
results = grid_search.cv_results_
for i in range(len(results['params'])):
    print("Hyperparameters:", results['params'][i])
    print("Mean test score:", results['mean_test_score'][i])
    print("Standard deviation test score:", results['std_test_score'][i])
    print()

# Log best parameters
print()
print("Best parameters:", best_params)
wandb.config.update(best_params)

## Log model performance metrics
# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(y_test, y_pred, average='macro')
MiPS = precision_score(y_test, y_pred, average='micro')
WPS = precision_score(y_test, y_pred, average='weighted')

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(y_test, y_pred, average='macro')
MiRS = recall_score(y_test, y_pred, average='micro')
WRS = recall_score(y_test, y_pred, average='weighted')

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(y_test, y_pred, average='macro')
MiFS = f1_score(y_test, y_pred, average='micro')
WFS = f1_score(y_test, y_pred, average='weighted')

# Note, average is macro...
metrics = {
    "Test Accuracy": accuracy,
    "Test Macro Precision": MaPS,
    "Test Micro Precision": MiPS,
    "Test Weighted Precision": WPS,
    "Test Macro Recall": MaRS,
    "Test Micro Recall": MiRS,
    "Test Weighted Recall": WRS,
    "Test Macro F1": MaFS,
    "Test Micro F1": MiFS,
    "Test Weighted F1": WFS,
}

# Log the metrics dictionary to W&B
wandb.log(metrics)

# Log confusion matrix for test results
# https://docs.wandb.ai/guides/integrations/scikit
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
# Labels are selected based on command-line input
# Spectral level 1
if 'SPECTRAL' in args.groundtruth_file:
    L1_classes = ['Artificial surfaces', 'Vegetated areas', 'Water bodies']
# CORINE level 1
else:
    L1_classes = ['Artificial Surfaces', 'Agricultural areas', 'Forest and seminatural areas', 'Wetlands', 'Water bodies']

# Spectral level 2
if 'SPECTRAL' in args.groundtruth_file:
    L2_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Forest', 'Shrub and/or herbaceous vegetation', 'Open spaces with little or no vegetation', 'Wetlands and Water']
# CORINE level 2
else:
    L2_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Forest', 'Shrub and/or herbaceous vegetation', 'Open spaces with little or no vegetation', 'Wetlands', 'Water bodies']

# Spectral level 3
if 'SPECTRAL' in args.groundtruth_file:
    L3_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 'Natural grassland', 'Moors and heathland', 
'Sclerophyllous vegetation', 'Transitional woodland/shrub', 'Open spaces with little or no vegetation', 
'Wetlands and Water']
# CORINE level 3
else:
    L3_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 'Natural grassland', 'Moors and heathland', 
'Sclerophyllous vegetation', 'Transitional woodland/shrub', 'Open spaces with little or no vegetation', 
'Wetlands', 'Water bodies']

# Plot confusion matrix
label_type = args.label_type
if label_type == 'L1':
    labels = L1_classes
elif label_type == 'L2':
    labels = L2_classes
elif label_type == 'L3':
    labels = L3_classes
# Plot matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=labels)

## Additional metrics
# Calculate classification report
classification_metrics = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

# Access precision, recall, F1, and accuracy for each class
for class_name in labels:
    metrics = classification_metrics[class_name]
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score_class = metrics['f1-score']
    
    # Print or use the metrics as needed
    print(f"Class: {class_name}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")

########################################################################

### TRAINING METRICS ###
## Run model on training dataset and compute metrics
y_pred_train = best_model.predict(X_train)

# Overall accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", accuracy_train)

# Precision, recall & F1 score 
precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

metrics = {
    "Train macro precision": precision_train,
    "Train macro recall": recall_train,
    "Train macro F1": f1_train,
}

# Log the metrics dictionary to W&B
wandb.log(metrics)

###########################################################################

## SAVE BEST MODEL AS PKL FILE
# Best model hyperparameters
print("Best model hyperparameters:", best_params)
# Save best model
model_output = args.model_output
model_output_file = os.path.join(output_dir, model_output)
joblib.dump(best_model, model_output_file)

## SAVE PREDICTIONS AS CSV FILE
## Create data frame of predictions
model_predictions = args.model_predictions
predictions_df = pd.DataFrame({'y_pred': y_pred})

# Reset the index of X_test (otherwise subsequent concatination of predictions 
# # from all downstream hierarchical models does not work
# Check before and after
print("X_test before reset index:", X_test_CSV)
X_test_CSV.reset_index(drop=True, inplace=True)
# Create DF of X_test, y_pred, and corresponding L2_groundtruth values
combined_df = pd.concat([X_test_CSV, predictions_df], axis=1)
print("CHECK INDICES MATCH")
print("X test (satellite data) with L2 & L3 ground truth data:", X_test_CSV)
print("Model predictions:", predictions_df)
print("Output CSV data:", combined_df)

## Save DF with predictions, satellite data, L2 & L3 ground truth to CSV file...
# ... for use in downstream hierarchical model
model_predictions_file = os.path.join(output_dir, model_predictions)
combined_df.to_csv(model_predictions_file, index=False)

# log memory usage for entire process
log_memory_usage()

wandb.finish()