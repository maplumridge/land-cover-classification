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
from sklearn.model_selection import train_test_split, GridSearchCV
# For evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
# For logging
import wandb
from wandb.sklearn import plot_confusion_matrix
# For saving
import joblib

# Set up Weights and Biases (W&B) logging
wandb.init(project='land-cover-classification')

### Load data
# Load seasonal satellite data with 4, 10 or 12 bands
unseen_satellite_files = ['/home/users/map205/MRes_Data/Braga_Winter_4bands.tif', '/home/users/map205/MRes_Data/Braga_Spring_4bands.tif', '/home/users/map205/MRes_Data/Braga_Summer_4bands.tif', '/home/users/map205/MRes_Data/Braga_Autumn_4bands.tif']
#unseen_satellite_files = ['/home/users/map205/MRes_Data/Braga_Winter_10bands.tif', '/home/users/map205/MRes_Data/Braga_Spring_10bands.tif', '/home/users/map205/MRes_Data/Braga_Summer_10bands.tif', '/home/users/map205/MRes_Data/Braga_Autumn_10bands.tif']

# Load CORINE ground truth data with level 1, 2 or 3 classes
unseen_groundtruth_file = '/home/users/map205/MRes_Data/Braga_CORINE_Cropped_L1.tif'


### Load the model that was generated during training
model_file = '/gws/nopw/j04/ai4er/users/map205/mres/L1_model_4bands_more_params.pkl'
#model_file = '/gws/nopw/j04/ai4er/users/map205/mres/L1_model_10bands_more_params.pkl'
loaded_model = joblib.load(model_file)


### SATELLITE DATA ###
# Loop over seasonal satellite data and create a time-series stack for the entire year
unseen_satellite_data_stack = None
for file in unseen_satellite_files:
    unseen_satellite_image = gdal.Open(file)
    bands = unseen_satellite_image.RasterCount
    unseen_band_data = np.stack([unseen_satellite_image.GetRasterBand(i).ReadAsArray() for i in range(1, bands + 1)], axis=-1)
    if unseen_satellite_data_stack is None:
        unseen_satellite_data_stack = unseen_band_data
    else:
        unseen_satellite_data_stack = np.concatenate((unseen_satellite_data_stack, unseen_band_data), axis=-1)
# Reshape stacked satellite data into a 2D array, ready for ingestion by the RF model
unseen_rows, unseen_cols, unseen_bands = unseen_satellite_data_stack.shape
unseen_satellite_data_2d = unseen_satellite_data_stack.reshape(unseen_rows * unseen_cols, unseen_bands)
# Create data frame
df_unseen = pd.DataFrame(unseen_satellite_data_2d, columns=['band'+str(i) for i in range(1, unseen_bands+1)])

##########
## RERUN MODELS WITHOUT ORIGINAL_INDEX THEN REMOVE CODE BELOW...
original_indices = range(len(df_unseen))
# Add 'original_index' column to the DataFrame
df_unseen['original_index'] = original_indices
###########

### GROUND TRUTH DATA ###
# Open ground truth data with gdal, as array
unseen_groundtruth_image = gdal.Open(unseen_groundtruth_file)
unseen_groundtruth_data = unseen_groundtruth_image.GetRasterBand(1).ReadAsArray()
# Flatten ground truth data
unseen_groundtruth_data_flat = unseen_groundtruth_data.flatten()
# Remove rows with NaN values from both satellite data frame and ground truth data
nan_indices = df_unseen.isnull().any(axis=1)
df_unseen = df_unseen[~nan_indices]
unseen_groundtruth_data_flat = unseen_groundtruth_data_flat[~nan_indices]


#### FOR CHECKING ###
## HAVE TO DO THIS AS PANDA SERIES... NOT NP ARRAY
# Convert numpy array to pandas series
unseen_groundtruth_series = pd.Series(unseen_groundtruth_data_flat)
# Calculate number of pixels per class
unseen_test_pixels_per_class = unseen_groundtruth_series.value_counts()
print("Number of pixels per class (Unseen Test Data):")
print(unseen_test_pixels_per_class)


### RUN THE MODEL ###
# Make predictions on satellite data using pre-trained model
y_pred_unseen = loaded_model.predict(df_unseen)


### RESULTS ###
# Overall accuracy
accuracy_unseen = accuracy_score(unseen_groundtruth_data_flat, y_pred_unseen)
print(f"Accuracy on Unseen Data: {accuracy_unseen}")
# Confusion matrix (unseen_groundtruth_data_flat is the 'real' situation)
L1_classes = ['Artificial Surfaces', 'Agricultural areas', 'Forest and seminatural areas', 'Wetlands', 'Water bodies']
confusion_matrix_fig = plot_confusion_matrix(unseen_groundtruth_data_flat, y_pred_unseen, labels=L1_classes)
wandb.log({"Confusion Matrix": confusion_matrix_fig})