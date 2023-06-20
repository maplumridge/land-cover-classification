"""
Project:        AI4ER MRes
Code:           Random Forest model for time-series data
Useage:         TBC
Order/workflow: After RF_read_data.py...
Prerequisites:  TBC
Time to run:    X hours, X minutes...
Improvements to do: 
- Add command-line arguments to use just one script for all model runs
- Add more evaluation metrics
- Save the model to then load and use on unseen data
- Set up heirarchical classification?
"""

# For data handling
import numpy as np
import pandas as pd
from osgeo import gdal
# For modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
# For evaluation
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
# For logging
import wandb


## Log results to W&B
wandb.init(project='land-cover-classification', entity='maplumridge')


## Load data
# Load seasonal satellite data with 4, 10 or 12 bands
satellite_files = ['/home/users/map205/MRes_Data/Winter_10bands.tif', '/home/users/map205/MRes_Data/Spring_10bands.tif', '/home/users/map205/MRes_Data/Summer_10bands.tif', '/home/users/map205/MRes_Data/Autumn_10bands.tif']
# Load CORINE ground truth data with level 1, 2 or 3 classes
ground_truth_file = '/home/users/map205/MRes_Data/L1_CORINE_32629_10m_Cropped.tif'


## Satellite data processing
# Loop over seasonal satellite data and create a time-series stack for the entire year
satellite_data_stack = None
for file in satellite_files:
    satellite_image = gdal.Open(file) # Open the file
    bands = satellite_image.RasterCount # Count the number of bands in the file
    band_data = np.stack([satellite_image.GetRasterBand(i).ReadAsArray() for i in range(1, bands + 1)], axis=-1) 
    if satellite_data_stack is None:
        satellite_data_stack = band_data
    else:
        satellite_data_stack = np.concatenate((satellite_data_stack, band_data), axis=-1)
# Reshape stacked satellite data into a 2D array, ready for ingestion by the RF model
rows, cols, bands = satellite_data_stack.shape
satellite_data_2d = satellite_data_stack.reshape(rows * cols, bands)


## Ground truth data processing
# Open ground truth data with gdal
ground_truth_image = gdal.Open(ground_truth_file)
# Read ground truth data as array and reshape to match satellite data
ground_truth_data = ground_truth_image.GetRasterBand(1).ReadAsArray()
ground_truth_data = np.resize(ground_truth_data, (rows, cols))


## Remove classes we are not interested in (ocean) & NaN values
# Convert ground truth to float
ground_truth_data = ground_truth_data.astype(float)
# Create mask for "Water bodies" class
mask = ground_truth_data == 5
# Apply mask to convert class 5 to NaN
ground_truth_data[mask] = np.nan
# Flatten ground truth data
ground_truth_data_flat = ground_truth_data.flatten()
# Create DataFrame combining satellite and ground truth data
df = pd.DataFrame(satellite_data_2d, columns=['band'+str(i) for i in range(1, bands+1)])
df['ground_truth'] = ground_truth_data_flat
# Remove rows with NaN values (& water bodies)
df.dropna(inplace=True)


## Organise data for modelling
# I want to use the same number of pixels for each class, to create a balanced model
# First, I need to specify the minimum number of pixels among all classes
min_pixels_per_class = 129007 # <-- Class 4: Wetlands
# Now, use this same number of pixels for each class to calculate the overall number of pixels I will use
num_classes = len(df['ground_truth'].unique())
sample_size = min_pixels_per_class * num_classes
# Now, sample an equal number of pixels from each class
df_sample = df.groupby('ground_truth').apply(lambda x: resample(x, n_samples=min_pixels_per_class, replace=True, random_state=42)).reset_index(drop=True)
# Create training and testing datasets with an 80/20 split
X = df_sample.drop('ground_truth', axis=1)
y = df_sample['ground_truth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Specify model configuration
# First, define the parameter grid for grid search cross validation
# These are the specific parameter values that I want to test
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 3]
}


## Run the model
# Create RF model
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Train RF model (note that I am using 5 cross-validation folds)
# Maybe modify this to be a command-line argument...?
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
# Find best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
# Make predictions on test data using the best model
y_pred = best_model.predict(X_test)


## Evaluate model performance & log results to W&B
# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
wandb.log({"Accuracy": accuracy})
# Best hyperparameters found through grid-search cross-validation
print("Best Hyperparameters:", best_params)
wandb.config.update(best_params)
wandb.finish()