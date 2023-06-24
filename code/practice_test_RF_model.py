"""
To do: 
- save predicitions, satellite data and L2 ground truth for same rows to CSV
- this will feed into hierarchical model

- removed sampling N pixels and shuffling of data

- if input is L1 or L2, save CSV with downstream ground truth (L2/L3)
- else, save CSV without downstream ground truth, only predictions
"""

"""
New updates
- Added precision, recall and F1 metrics
- Added logging of accuracy to W&B
- Added all class labels
- Added command-line args
"""

"""
Project:        AI4ER MRes
Code:           Random Forest model to test in unseen region
Useage:         TBC
Order/workflow: After *_train_RF_model.py...
Prerequisites:  TBC
Time to run:    X hours, X minutes...
Improvements to do: 
- Add command-line arguments to use just one script for all model runs
- Remove original_index 
"""

# Worked on 4bands
# After L1_train_RF_model.py has finished running,
# (with 10 bands and original index removed),
# remove original_index from here and run this using the 10band model
# add more metrics - precision etc.

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
# For logging
import wandb
# For saving
import joblib
# For command-line running
import argparse
import os

### SET UP ###
## Log results to W&B
# Note, if I don't specify entity, this default saves to a group project instead...
wandb.init(project='land-cover-classification', entity='maplumridge')

## Set up parsing of command line arguments 
parser = argparse.ArgumentParser(description='Land Cover Classification')
parser.add_argument('--test_satellite_files', nargs='*', help='List of satellite data files to generate time series stack (4 bands, 10 bands, 12 bands)', required=True)
parser.add_argument('--test_groundtruth_file', help='Ground truth data file (level 1, 2 or 3)', required=True)
parser.add_argument('--test_additional_groundtruth_file', help='Ground truth data file (level 1, 2 or 3)', required=True)
#parser.add_argument('--pixels_per_class', type=int, help='Number of pixels per class to use for model training', required=True)
parser.add_argument('--label_type', choices=['L1', 'L2', 'L3'], default='L1', help='Labels to use for confusion matrix (level 1, 2 or 3 classes)', required=True)
parser.add_argument('--input_model', help='Model to use with test data', required=True)
#parser.add_argument('--model_output', help='Name of output pkl file which model will be saved to', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
data_dir = '/home/users/map205/MRes_Data/'
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
test_additional_groundtruth_file = os.path.join(data_dir, args.test_additional_groundtruth_file)

## LOAD MODEL ##
input_model = os.path.join(model_dir, args.input_model)
input_model = joblib.load(input_model)


### SATELLITE DATA ###
# Loop over seasonal satellite data and create a time-series stack for the entire year
test_satellite_data_stack = None
for file in test_satellite_files:
    test_satellite_image = gdal.Open(file)
    bands = test_satellite_image.RasterCount
    test_band_data = np.stack([test_satellite_image.GetRasterBand(i).ReadAsArray() for i in range(1, bands + 1)], axis=-1)
    if test_satellite_data_stack is None:
        test_satellite_data_stack = test_band_data
    else:
        test_satellite_data_stack = np.concatenate((test_satellite_data_stack, test_band_data), axis=-1)
# Reshape stacked satellite data into a 2D array, ready for ingestion by the RF model
test_rows, test_cols, test_bands = test_satellite_data_stack.shape
test_satellite_data_2d = test_satellite_data_stack.reshape(test_rows * test_cols, test_bands)

##########
## RERUNNING MODELS NOW WITHOUT ORIGINAL_INDEX THEN REMOVE CODE BELOW...
#original_indices = range(len(df_test))
## Add 'original_index' column to data frame
#df_test['original_index'] = original_indices
###########

### GROUND TRUTH DATA ###
# Open ground truth data with gdal, as array
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


### ADDITIONAL GROUND TRUTH DATA ###
# Open additional ground truth data with gdal
test_additional_groundtruth_image = gdal.Open(test_additional_groundtruth_file)
# Read additional ground truth data as array and reshape/size to match satellite data
test_additional_groundtruth_data = test_additional_groundtruth_image.GetRasterBand(1).ReadAsArray()
test_additional_groundtruth_data_flat = test_additional_groundtruth_data.flatten()


### CREATE DATA FRAME ###
# Combine satellite time-series, ground truth data, and L2 ground truth data
df_test = pd.DataFrame(test_satellite_data_2d, columns=['band'+str(i) for i in range(1, test_bands+1)])
df_test['groundtruth'] = test_groundtruth_data_flat
df_test['L2_groundtruth'] = test_additional_groundtruth_data_flat
print("Data frame before NaN removal (L1, L2 and satellite data)", df_test)
## Remove rows with NaN values
df_test.dropna(inplace=True)
print("Data frame post-NaN removal (L1, L2 and satellite data):", df_test)

# Remove same / NaN rows from ground truth 
# Otherwise code fails because shape of y_pred and ground truth data for measuring
# accuracy do not align...
# Get the indices of remaining rows in the filtered DataFrame
remaining_indices = df_test.index
# Remove corresponding rows from test_groundtruth_data_flat
test_groundtruth_data_flat_filtered = test_groundtruth_data_flat[remaining_indices]

# Create a copy of X_test (with L2_groundtruth, for use in CSV file)
df_test_CSV = df_test.copy()
df_test_CSV.drop('groundtruth', axis=1, inplace=True)

## Drop redundant columns for modelling
# Only want satellite data
df_test.drop('L2_groundtruth', axis=1, inplace=True)
df_test.drop('groundtruth', axis=1, inplace=True)

## No shuffling
## Use all pixels

### RUN THE MODEL ###
# Make predictions on satellite data using loaded model
y_pred_test = input_model.predict(df_test)


### EVALUATION ###
# Log results to W&B

## Test metrics
# Overall accuracy
accuracy_test = accuracy_score(test_groundtruth_data_flat_filtered, y_pred_test)
print(f"Test accuracy: {accuracy_test}")
wandb.log({"Test accuracy": accuracy_test})

# Log confusion matrix for test results
# https://docs.wandb.ai/guides/integrations/scikit
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
# Labels are selected as command-line arg
L1_classes = ['Artificial Surfaces', 'Agricultural areas', 'Forest and seminatural areas', 'Wetlands', 'Water bodies']

L2_classes = ['Urban fabric', 'Industrial, comercial and transport units', 'Mine, dump and construction sites', 
'Artificial, non-agricultural vegetated areas','Arable land', 'Permanent crops', 'Pastures', 'Heterogeneous agricultural areas', 
'Forest', 'Shrub and/or herbaceous vegetation associations', 'Open spaces with little or no vegetation', 'Inland wetlands', 
'Coastal wetlands', 'Inland waters', 'Marine waters']

L3_classes = ['Continuous urban fabric', 'Discontinuous urban fabric', 'Industrial, comercial and transport units', 
'Mine, dump and construction sites', 'Artificial, non-agricultural vegetated areas','Arable land', 'Permanent crops', 
'Pastures', 'Heterogeneous agricultural areas', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 
'Natural grassland', 'Moors and heathland', 'Sclerophyllous vegetation', 'Transitional woodland/shrub', 
'Open spaces with little or no vegetation', 'Inland wetlands', 'Coastal wetlands', 'Inland waters', 'Marine waters']

label_type = args.label_type

if label_type == 'L1':
    labels = L1_classes
elif label_type == 'L2':
    labels = L2_classes
elif label_type == 'L3':
    labels = L3_classes

# Plot matrix
wandb.sklearn.plot_confusion_matrix(test_groundtruth_data_flat_filtered, y_pred_test, labels=labels)

## Other metrics
# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WPS = precision_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')
wandb.log({"Test macro precision ": MaPS})
wandb.log({"Test micro precision ": MiPS})
wandb.log({"Test weighted precision ": WPS})

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WRS = recall_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')
wandb.log({"Test macro recall ": MaRS})
wandb.log({"Test micro recall ": MiRS})
wandb.log({"Test weighted recall ": WRS})

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='macro')
MiFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='micro')
WFS = f1_score(test_groundtruth_data_flat_filtered, y_pred_test, average='weighted')
wandb.log({"Test macro F1 ": MaFS})
wandb.log({"Test micro F1 ": MiFS})
wandb.log({"Test weighted F1 ": WFS})


### SAVE PREDICTIONS ###
## Save predictions CSV file, with test data frame containing satellite data and L2 classes for 
# rows. Note, NaN rows are excluded.
print("CHECKING df_test (should have satellite data and L2 data (not L1)):", df_test_CSV)
predictions_df = pd.DataFrame({'y_pred': y_pred_test})
combined_df = pd.concat([df_test_CSV, predictions_df], axis=1)
## To check
print("Output CSV data:", combined_df)
# Save CSV to output_dir
model_predictions = args.model_predictions
model_predictions_file = os.path.join(output_dir, model_predictions)
combined_df.to_csv(model_predictions_file, index=False)

wandb.finish()