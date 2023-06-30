"""
Project:        AI4ER MRes
Code:           Random Forest model for level 2 or level 3 testing
Order:          After L2L3_train.py
Time to run:    Minutes to hours (due to subsampling of 100,000 pixels)
To do:          TIF file generation
"""

# For data manipulation
import numpy as np
import pandas as pd
from osgeo import gdal
# For modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
import random
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
parser.add_argument('--test_groundtruth_file', help='Ground truth data file level 2 or 3', required=True)
parser.add_argument('--label_type', choices=['L2', 'L3'], default='L1', help='Labels to use for confusion matrix (level 1, 2 or 3 classes)', required=True)
parser.add_argument('--input_model', help='Model to use with test data', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
parser.add_argument('--output_tif_file', help='Name of output tif file which model predictions will be saved to', required=True)
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

# Create a DataFrame with satellite and L2 or L3 ground truth
df_test = pd.DataFrame(test_satellite_data_2d, columns=['band'+str(i) for i in range(1, test_bands+1)])
df_test['groundtruth'] = test_groundtruth_data_flat

# Remove rows with NaN values
df_test.dropna(inplace=True)

# Shuffle data
df_test_shuffled = df_test.sample(frac=1, random_state=42)

# Sample 100,000 pixels per class
sampled_data = pd.DataFrame()
for class_label, count in test_pixels_per_class.iteritems():
    if count >= 100000:
        sampled_data_per_class = df_test_shuffled[df_test_shuffled['groundtruth'] == class_label].sample(n=100000, random_state=42)
    else:
        sampled_data_per_class = df_test_shuffled[df_test_shuffled['groundtruth'] == class_label]
    sampled_data = sampled_data.append(sampled_data_per_class.head(100000))

# Create a copy of sampled data for modeling
df_test_sampled = sampled_data.copy()

# Remove same / NaN rows from ground truth
remaining_indices = df_test_sampled.index
test_groundtruth_data_flat_filtered = test_groundtruth_data_flat[remaining_indices]
print("Groundtruth data with NaN rows removed:", test_groundtruth_data_flat_filtered.shape)

# Drop redundant columns for modeling
df_test_sampled.drop('groundtruth', axis=1, inplace=True)

# Run the model
y_pred_test = input_model.predict(df_test_sampled)

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
    L2_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Forest', 'Shrub and/or herbaceous vegetation', 'Open spaces with little or no vegetation', 'Wetlands and Water']
else:
    L2_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Forest', 'Shrub and/or herbaceous vegetation', 'Open spaces with little or no vegetation', 'Wetlands', 'Water bodies']

# Check the value of --test_groundtruth_file
if 'SPECTRAL' in args.test_groundtruth_file:
    L3_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 'Natural grassland', 'Moors and heathland', 
'Sclerophyllous vegetation', 'Transitional woodland/shrub', 'Open spaces with little or no vegetation', 
'Wetlands and Water']
else:
    L3_classes = ['Urban fabric', 'Artificial, non-agricultural vegetated areas', 'Agricultural areas', 
'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 'Natural grassland', 'Moors and heathland', 
'Sclerophyllous vegetation', 'Transitional woodland/shrub', 'Open spaces with little or no vegetation', 
'Wetlands', 'Water bodies']

label_type = args.label_type

if label_type == 'L2':
    labels = L2_classes
elif label_type == 'L3':
    labels = L3_classes

# Plot matrix
wandb.sklearn.plot_confusion_matrix(test_groundtruth_data_flat_filtered, y_pred_test, labels=labels)

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

# Calculate classification report
classification_metrics = classification_report(test_groundtruth_data_flat_filtered, y_pred_test, target_names=labels, output_dict=True)
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



## SAVE PREDICITONS TO CSV ##
columns_to_drop = ['band' + str(i) for i in range(1, test_bands+1)]
df_test_NaN.drop(columns=columns_to_drop, inplace=True)
# Only left with ground truth (to preserve indices...)
print("CHECK INDICES: Data frame with satellite data removed:", df_test_NaN)

# Impute NaN rows into predictions data frame, so it matches the target shape of test region
y_pred_imputed = y_pred_test.copy()
# Iterate over rows of df_test_NaN
for index, row in df_test_NaN.iterrows():
    if row.isnull().any():
        # Get index position
        pos = index
        # Insert row with value 999 (in correct index location)
        y_pred_imputed = np.insert(y_pred_imputed, pos, 999, axis=0)
# Reshape y_pred_imputed to match the desired shape
print("CHECK INDICES: y_pred_imputed + ground truth:", y_pred_imputed)
y_pred_imputed.drop('groundtruth')
y_pred_imputed = y_pred_imputed.reshape(height, width)
print("CHECK INDICES: y_pred_imputed:", y_pred_imputed)


## SAVE PREDICITONS TO TIF ##

import rasterio
from rasterio.transform import Affine
 
output_tif_file = args.output_tif_file
#output_tif_file = "/home/users/map205/unseen_4bands_predictions.tif"

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": y_pred_imputed.dtype,
    "crs": crs,
    "transform": Affine.translation(left, top) * Affine.scale(resolution, -resolution),
}

with rasterio.open(output_tif_file, "w", **profile) as dst:
    dst.write(y_pred_imputed, 1)

log_memory_usage()

wandb.finish()
