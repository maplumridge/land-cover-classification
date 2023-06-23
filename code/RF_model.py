""" 
This finally works!!!! To be cleaned up...

Updates since last push:
- Enforce command-line args
- Added all class labels
- Rearrange CSV generation (order was incorrect so file was not being saved)
- Also added satellite data to CSV file as well, for loading in the next model
- Also added indexing and addition of ground truth level 2 to CSV for the y_pred rows,
so that I can just load the CSV by the next model and it has all the info needed
(the rows which were predicted from this model, with their corresponding 
predicted value, satellite data and L2 class to train the next model)
- Note: I have to generate two versions of X_test, one with the L2 data included
as an additional column (X_test_CSV), so I can save this to CSV with the predictions. 
Another, X_test for use in the model (with L2_groundtruth column removed).
- For every predicition, I have the satellite data and L2 class, ready for the next model.

To do:
- Add file name options for command-line args

Note:
- This is the flat model... still need to configure hierarchical model properly...
"""

# Let's test this with 100 pixels per class and a smaller grid search...
# TO DO: reset to larger config after testing...
# Maybe make more modular with functions...
# Tidy up

# Old updates:
# Added shuffle (with index maintained, to help track pixels)
# - Printing data frame at various stages to confirm this is working
# Added training and testing metrics
# Added command line options, to specify
# - satellite data bands 4, 10 and 12
# - ground truth data levels 1, 2 and 3
# - number of pixels per class
# - where to save output model
# - where to save output predictions
# Removed original_index from data frame before modelling...

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
parser.add_argument('--satellite_files', nargs='*', help='List of satellite data files to generate time series stack (4 bands, 10 bands, 12 bands)', required=True)
parser.add_argument('--groundtruth_file', help='Ground truth data file (level 1 or 2)', required=True)
parser.add_argument('--additional_groundtruth_file', help='Lower level ground truth file for level 2 or 3 classification', required=True)
parser.add_argument('--pixels_per_class', type=int, help='Number of pixels per class to use for model training', required=True)
parser.add_argument('--label_type', choices=['L1', 'L2', 'L3'], default='L1', help='Labels to use for confusion matrix (level 1, 2 or 3 classes)', required=True)
parser.add_argument('--model_output', help='Name of output pkl file which model will be saved to', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
input_dir = '/home/users/map205/MRes_Data/'
# Location for saving the model and model predictions
output_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'
# Checking...
satellite_files=args.satellite_files
print(satellite_files)

## Load data
# Load seasonal satellite data with 4, 10 or 12 bands. This will load 4 files for the year 2018.
satellite_files = [os.path.join(input_dir, file) for file in args.satellite_files]
# Load CORINE ground truth data with level 1 or 2 classes
groundtruth_file = os.path.join(input_dir, args.groundtruth_file)
# Load additional CORINE ground truth data with 2 or 3 classes
additional_groundtruth_file = os.path.join(input_dir, args.additional_groundtruth_file)


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

### No longer want to remove pixels with certain classes, train model on all so that it can be used in multiple settings
## Remove classes we are not interested in (ocean) & NaN values
## Convert ground truth to float
#ground_truth_data = ground_truth_data.astype(float)
## Create mask for "Water bodies" class
#mask = ground_truth_data == 5
## Apply mask to convert class 5 to NaN
#ground_truth_data[mask] = np.nan


### CREATE DATA FRAME ###
## Create a DataFrame combining data and ground truth data
#df = pd.DataFrame(satellite_data_2d, columns=['band'+str(i) for i in range(1, bands+1)])
#df['groundtruth'] = groundtruth_data_flat
## Remove rows with NaN values
#df.dropna(inplace=True)
##print(df.head(10))


### ADDITIONAL GROUND TRUTH DATA ###
# Open additional ground truth data with gdal
L2_groundtruth_image = gdal.Open(additional_groundtruth_file)
# Read additional ground truth data as array and reshape/size to match satellite data
L2_groundtruth_data = L2_groundtruth_image.GetRasterBand(1).ReadAsArray()
L2_groundtruth_data_flat = L2_groundtruth_data.flatten()


### CREATE DATA FRAME ###
# Create a DataFrame combining data, ground truth data, and L2 ground truth data
df = pd.DataFrame(satellite_data_2d, columns=['band'+str(i) for i in range(1, bands+1)])
df['groundtruth'] = groundtruth_data_flat
df['L2_groundtruth'] = L2_groundtruth_data_flat
print("Data frame before NaN removal (L1, L2 and satellite data)", df)
## Remove rows with NaN values
df.dropna(inplace=True)
print("Data frame post-NaN removal (L1, L2 and satellite data):", df)


### ORGANISE DATA FOR MODELLING ###

# New addition:
# First, shuffle the data before train/test split, to randomise selection of data for training and testing
# https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
# https://stackoverflow.com/questions/55072435/shuffle-a-dataframe-while-keeping-internal-order
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
# Note: need to maintain indices for shuffling operation
shuffled_df = df.sample(frac=1, random_state=42)
# This works and shows that the index values are maintained and the pixels are shuffled
print("Shuffled data frame:", shuffled_df) 

# I want to use the same number of pixels for each class (oversampling/undersampling), to create a balanced model
min_pixels_per_class = args.pixels_per_class

## Old sampling code before we had the shuffle...
#df_level1_sample = shuffled_df.groupby('groundtruth_level1').apply(lambda x: x.sample(n=min_pixels_per_class, replace=True, random_state=42)).reset_index(drop=True)
#print("DF with 100 pixels per class:", df_level1_sample.head(10))

# Now, group shuffled data by 'groundtruth' class 
# & sample specified (equal) number of pixels per class
df_sample = shuffled_df.groupby('groundtruth', as_index=False).apply(lambda x: x.sample(n=min_pixels_per_class, replace=True, random_state=42)).reset_index(level=0, drop=True)

## Add original indexes as a column in df_sample <-- do not need this any more...
#df_sample['original_index'] = df_sample.index
#print("DF with 100 pixels per class:", df_sample.head(10))

# Create training and testing datasets with an 80/20 split
# Drop ground truth column from satellite data
# Maintain ground truth column for ground truth data
X_train, X_test, y_train, y_test = train_test_split(
    df_sample.drop(['groundtruth'], axis=1),
    df_sample['groundtruth'],
    test_size=0.2,
    random_state=42
)

## CHECKS ##
# Print shuffled training and testing data to ensure randomness is maintained...
# Should also maintain ground truth level 2 file in this
# as we want to keep this indexed correctly and aligned with the satellite data
# and predictions for the same pixels.
# Train
print("Shuffled Training Data:")
print("Satellite training data:", X_train)
print("Ground truth training data:", y_train)
# Test
print("Shuffled Testing Data:")
print("Satellite testing data:", X_test)
print("Ground truth testing data:", y_test)

# Calculate number of pixels per class, to ensure it's roughly equal
train_pixels_per_class = y_train.value_counts()
test_pixels_per_class = y_test.value_counts()
print("Number of pixels per class (Training Data):")
print(train_pixels_per_class)
print("Number of pixels per class (Testing Data):")
print(test_pixels_per_class)

# Record the indices of the testing/prediction dataset
original_indices_test = X_test.index
print("Test indices:", X_test.index)

# Remove the 'L2_groundtruth' column from X_train, maintain for X_test
X_train.drop('L2_groundtruth', axis=1, inplace=True)
# Create a copy of X_test (with L2_groundtruth, for use in CSV file)
X_test_CSV = X_test.copy()
X_test.drop('L2_groundtruth', axis=1, inplace=True)


## MODELING ##
## Specify model configuration
# First, define the parameter grid for 5-fold grid-search cross-validation
# These are the specific parameter values that I want to test

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 3, 5]
}
"""
For practicing...

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_leaf': [1, 2]
}
"""

# Create RF model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Train RF model (note that I am using 5 folds)
# Maybe also modify this to be a command-line argument...?
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
# Find best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
# Make predictions on test data using the best model
y_pred = best_model.predict(X_test)


### EVALUATION ###
# Log results to W&B

## Test metrics
# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(y_test, y_pred, average='macro')
MiPS = precision_score(y_test, y_pred, average='micro')
WPS = precision_score(y_test, y_pred, average='weighted')
wandb.log({"Test macro precision ": MaPS})
wandb.log({"Test micro precision ": MiPS})
wandb.log({"Test weighted precision ": WPS})

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(y_test, y_pred, average='macro')
MiRS = recall_score(y_test, y_pred, average='micro')
WRS = recall_score(y_test, y_pred, average='weighted')
wandb.log({"Test macro recall ": MaRS})
wandb.log({"Test micro recall ": MiRS})
wandb.log({"Test weighted recall ": WRS})

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(y_test, y_pred, average='macro')
MiFS = f1_score(y_test, y_pred, average='micro')
WFS = f1_score(y_test, y_pred, average='weighted')
wandb.log({"Test macro F1 ": MaFS})
wandb.log({"Test micro F1 ": MiFS})
wandb.log({"Test weighted F1 ": WFS})

## Average precision, recall and F1
precision_test = precision_score(y_test, y_pred, average='macro')
recall_test = recall_score(y_test, y_pred, average='macro')
f1_test = f1_score(y_test, y_pred, average='macro')

# Log testing metrics to W&B
wandb.log({"Test Accuracy": accuracy})
wandb.log({"Test Precision": precision_test})
wandb.log({"Test Recall": recall_test})
wandb.log({"Test F1 Score": f1_test})

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
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=labels)

####

## Training metrics
# Make predictions on training data using best model
y_pred_train = best_model.predict(X_train)

# Overall accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", accuracy_train)

# Precision, recall & F1 score 
precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

# Log training metrics to W&B
wandb.log({"Training Accuracy": accuracy_train})
wandb.log({"Training Precision": precision_train})
wandb.log({"Training Recall": recall_train})
wandb.log({"Training F1 Score": f1_train})

####

## SAVE MODEL & PREDICITIONS ##
# Best model hyperparameters
print("Best model hyperparameters:", best_params)
# Save best model
model_output = args.model_output
model_output_file = os.path.join(output_dir, model_output)
joblib.dump(best_model, model_output_file)

## Create data frame of predictions
#predictions_df = pd.DataFrame({'y_pred': y_pred})
## Save predicitions to CSV file
model_predictions = args.model_predictions
#model_predictions_file = os.path.join(output_dir, model_predictions)
## Do I need to specify 'model_predictions_file' below instead of 'model_predictions'?
#predictions_df.to_csv(model_predictions, index=False)


# Shouldn't need this first command now, since X_test now contains the L2 ground truth data

## Retrieve L2_groundtruth values corresponding to the original indices of the training dataset
#L2_groundtruth_test_rows = X_test.loc[original_indices_test, 'L2_groundtruth']

print("X_test before reset index:", X_test_CSV)
## This assumes that the order of y_pred is the same as the order of X_test
# Create data frame of predictions
predictions_df = pd.DataFrame({'y_pred': y_pred})
# Reset the index of X_test (otherwise concatination does not work)
X_test_CSV.reset_index(drop=True, inplace=True)
# Create DF of X_test, y_pred, and corresponding L2_groundtruth values
combined_df = pd.concat([X_test_CSV, predictions_df], axis=1)
#combined_df['L2_groundtruth'] = L2_groundtruth_test.values
## Add the original indices as a separate column in the DataFrame
#combined_df['original_index'] = shuffled_df.index[X_test.index]

## To check, 
print("CHECK INDICES MATCH")
print("X test (satellite data) with L2 ground truth data:", X_test_CSV)
print("Model predictions:", predictions_df)
print("Output CSV data:", combined_df)

## Drop the 'original_index' column
#combined_df.drop('original_index', axis=1, inplace=True)
## Save the combined DataFrame to a CSV file
model_predictions_file = os.path.join(output_dir, model_predictions)
# Export the combined DataFrame to a CSV file
combined_df.to_csv(model_predictions_file, index=False)

## Create a DataFrame of predictions and satellite data
#predictions_df = pd.DataFrame({'y_pred': y_pred})
#combined_df = pd.concat([X_test, predictions_df], axis=1)
## Save the combined DataFrame to a CSV file
#model_predictions_file = os.path.join(output_dir, model_predictions)
#combined_df.to_csv(model_predictions_file, index=False)

wandb.finish()