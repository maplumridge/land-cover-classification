"""
Project:        AI4ER MRes
Code:           Random Forest model for level 1 classification on unseen test dataset
                and subsequent generation of output predictions
Order:          After L1_train.py
Time to run:    ~1 hour if sampling 100,000 pixels per class
To do:          Merge with L2L3_test.py
"""

# For data manipulation
import numpy as np
import pandas as pd
from osgeo import gdal
# For modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
# For evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
from wandb.sklearn import plot_confusion_matrix
from sklearn.metrics import classification_report
# For logging
import wandb
# For saving
import joblib
# For command-line running
import argparse
import os
# For logging memory requirements
import psutil

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Memory usage in MB
    with open('test_model_memory.log', 'a') as f:
        f.write(f'{memory_usage}\n')

### SET UP ###
## Log results to W&B
# Note, if I don't specify entity, this default saves to a group project instead...
wandb.init(project='land-cover-classification', entity='maplumridge')

## Set up parsing of command line arguments 
parser = argparse.ArgumentParser(description='Land Cover Classification')
parser.add_argument('--test_satellite_files', nargs='*', help='List of satellite data files to generate time series stack (4 bands, 10 bands, 12 bands)', required=True)
parser.add_argument('--test_groundtruth_file', help='Ground truth data file level 1', required=True)
parser.add_argument('--test_L2_groundtruth_file', help='Ground truth data file level 2', required=True)
parser.add_argument('--test_L3_groundtruth_file', help='Ground truth data file level 3', required=True)
parser.add_argument('--input_model', help='Model to use with test data', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
parser.add_argument('--output_tif_file', help='Name of output tif file which model predictions will be saved to')
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
data_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'
# Location of model to load
model_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'
# Where to save CSV file with predictions
output_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'

## LOAD DATA ##
# Load seasonal satellite data with 4, 10 or 12 bands. This will load 4 files for the year 2018.
test_satellite_files = [os.path.join(data_dir, file) for file in args.test_satellite_files]
# Load CORINE ground truth data with level 1, 2 or 3 classes
test_groundtruth_file = os.path.join(data_dir, args.test_groundtruth_file)
# Load additional CORINE ground truth data with 2 or 3 classes
test_L2_groundtruth_file = os.path.join(data_dir, args.test_L2_groundtruth_file)
# Load additional CORINE ground truth data with 2 or 3 classes
test_L3_groundtruth_file = os.path.join(data_dir, args.test_L3_groundtruth_file)

## LOAD MODEL ##
input_model = os.path.join(model_dir, args.input_model)
input_model = joblib.load(input_model)


### SATELLITE DATA ###
# Loop over seasonal satellite data and create a time-series stack for the entire year

# Placeholder to store CRS information for later calculation of coordinates...
# ...for each pixel
crs = None
left = None
top = None
right = None
bottom = None
resolution = None

test_satellite_data_stack = None
for file in test_satellite_files:
    test_satellite_image = gdal.Open(file)
    bands = test_satellite_image.RasterCount
    test_band_data = np.stack([test_satellite_image.GetRasterBand(i).ReadAsArray() for i in range(1, bands + 1)], axis=-1)
    # Check if crs is not yet assigned and retrieve the projection information
    if crs is None:
        crs = test_satellite_image.GetProjection()
        # Bounding box
        bounds = test_satellite_image.GetGeoTransform()
        left = bounds[0]
        top = bounds[3]
        right = bounds[0] + bounds[1] * test_band_data.shape[1]
        bottom = bounds[3] + bounds[5] * test_band_data.shape[0]
        # Resolution, abs takes + value, not -
        resolution = abs(bounds[1]) 
    if test_satellite_data_stack is None:
        test_satellite_data_stack = test_band_data
    else:
        test_satellite_data_stack = np.concatenate((test_satellite_data_stack, test_band_data), axis=-1)
# Reshape stacked satellite data into a 2D array, ready for ingestion by the RF model
test_rows, test_cols, test_bands = test_satellite_data_stack.shape
test_satellite_data_2d = test_satellite_data_stack.reshape(test_rows * test_cols, test_bands)

# Calculate width and height from extent and resolution
width = int((right - left) / resolution)
height = int((top - bottom) / resolution)

### GROUND TRUTH DATA ###
test_groundtruth_image = gdal.Open(test_groundtruth_file)
# Read ground truth data as array and reshape/size to match satellite data
test_groundtruth_data = test_groundtruth_image.GetRasterBand(1).ReadAsArray()
test_groundtruth_data_flat = test_groundtruth_data.flatten()

#### FOR CHECKING ###
## HAVE TO DO THIS AS PANDA SERIES... NOT NP ARRAY
## Error was: AttributeError: 'numpy.ndarray' object has no attribute 'value_counts'
# Convert array to pandas series
test_groundtruth_series = pd.Series(test_groundtruth_data_flat)
# Calculate number of pixels per class
test_pixels_per_class = test_groundtruth_series.value_counts()
print("Number of pixels per class (Test Data):", test_pixels_per_class)


### L2 GROUND TRUTH DATA ###
test_L2_groundtruth_image = gdal.Open(test_L2_groundtruth_file)
test_L2_groundtruth_data = test_L2_groundtruth_image.GetRasterBand(1).ReadAsArray()
test_L2_groundtruth_data_flat = test_L2_groundtruth_data.flatten()

### L3 GROUND TRUTH DATA ###
test_L3_groundtruth_image = gdal.Open(test_L3_groundtruth_file)
test_L3_groundtruth_data = test_L3_groundtruth_image.GetRasterBand(1).ReadAsArray()
test_L3_groundtruth_data_flat = test_L3_groundtruth_data.flatten()

### CREATE DATA FRAME ###
# Combine satellite time-series, with L1, L2 and L3 ground truth data
df_test_NaN = pd.DataFrame(test_satellite_data_2d, columns=['band'+str(i) for i in range(1, test_bands+1)])
df_test_NaN['groundtruth'] = test_groundtruth_data_flat
df_test_NaN['L2_groundtruth'] = test_L2_groundtruth_data_flat
df_test_NaN['L3_groundtruth'] = test_L2_groundtruth_data_flat
print("Data frame before NaN removal (L1, L2, L3 and satellite data)", df_test_NaN)

# Create copy of data frame, to be used in modelling
# DF_test_NaN must be preserved for later use in creating .tif file
df_test = df_test_NaN.copy()
## Remove rows with NaN values
df_test.dropna(inplace=True)
print("Data frame post-NaN removal (L1, L2, L3 and satellite data):", df_test)

# Shuffle the data
df_test_shuffled = df_test.sample(frac=1, random_state=42)
# Sample 100,000 pixels per class
sampled_data = pd.DataFrame()
for class_label, count in test_pixels_per_class.iteritems():
    if count >= 100000:
        sampled_data_per_class = df_test_shuffled[df_test_shuffled['groundtruth'] == class_label].sample(n=100000, random_state=42)
    else:
        sampled_data_per_class = df_test_shuffled[df_test_shuffled['groundtruth'] == class_label]
    sampled_data = sampled_data.append(sampled_data_per_class.head(100000))
# Create a copy of the sampled data for modeling
df_test_sampled = sampled_data.copy()

## In the future, when we want to generate a .tif file full of predictions for the 
# entire region, will need to remove shuffling and sub-sampling of fewer pixels...
    ## DO NOT SHUFFLE 
    # For visualising predicitions after, need to have them in the right location
    ## Use all pixels

# Remove same / NaN rows from ground truth 
# Otherwise code fails because shape of y_pred and ground truth data for measuring
# accuracy do not align...
# Get the indices of remaining rows in the filtered DataFrame
remaining_indices = df_test_sampled.index
# Remove corresponding rows from test_groundtruth_data_flat
test_groundtruth_data_flat_filtered = test_groundtruth_data_flat[remaining_indices]
print("Groundtruth data with NaN rows removed:", test_groundtruth_data_flat_filtered.shape)
print("Groundtruth shape:", test_groundtruth_data_flat_filtered)

## Drop redundant columns for modelling
# Only want satellite data
df_test_sampled.drop(['groundtruth', 'L2_groundtruth', 'L3_groundtruth'], axis=1, inplace=True)

### RUN THE MODEL ###
y_pred_test = input_model.predict(df_test)
print("Predictions:", y_pred_test)

### EVALUATION ###
# Log results to W&B

## Test metrics
# Overall accuracy
accuracy_test = accuracy_score(test_groundtruth_data_flat_filtered, y_pred_test)
print(f"Test accuracy: {accuracy_test}")

# Log confusion matrix for test results
# https://docs.wandb.ai/guides/integrations/scikit
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
# Labels are selected as command-line arg
# Check the value of --test_groundtruth_file
if 'SPECTRAL' in args.test_groundtruth_file:
    L1_classes = ['Artificial surfaces', 'Vegetated areas', 'Water bodies']
else:
    L1_classes = ['Artificial Surfaces', 'Agricultural areas', 'Forest and seminatural areas', 'Wetlands', 'Water bodies']

# Plot matrix
wandb.sklearn.plot_confusion_matrix(test_groundtruth_data_flat_filtered, y_pred_test, labels=L1_classes)

# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')

# Average is macro...
metrics = {
    "Test Accuracy": accuracy_test,
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

# Log metrics to W&B
wandb.log(metrics)

## Evaluation metrics
# Calculate classification report
classification_metrics = classification_report(test_groundtruth_data_flat_filtered, y_pred_test, target_names=L1_classes, output_dict=True)
# Access precision, recall, F1, and accuracy for each class
for class_name in labels:
    metrics = classification_metrics[class_name]
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score_class = metrics['f1-score']
    accuracy = metrics['accuracy']
    
    # Print or use the metrics as needed
    print(f"Class: {class_name}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")
    print(f"Accuracy: {accuracy}")


### SAVE PREDICTIONS TO CSV FILE ###
y_pred_imputed = y_pred_test.copy()
# Create data frame of predictions
y_pred_imputed_df = pd.DataFrame(y_pred_imputed, columns=['Predicted_Value'])  # change column name to fit your needs
# Reset index of df_test and assign to new column
df_test['original_index'] = df_test.index
# Concatenate df_test and y_pred_imputed_df
combined_df = pd.concat([df_test, y_pred_imputed_df], axis=1)
print("Combined data frame with original indices:", combined_df)
### SAVE CSV FILE 
model_predictions = args.model_predictions
model_predictions_file = os.path.join(output_dir, model_predictions)
combined_df.to_csv(model_predictions_file, index=False)

log_memory_usage()

wandb.finish()
