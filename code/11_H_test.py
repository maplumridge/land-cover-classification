"""
Project:        AI4ER MRes
Code:           Random Forest model to perform hierarchical classification on test dataset
Order:          After H_train.py (for input model) AND L1_test.py (for input predictions)
Time to run:    Depends on number of pixels tested in L1_test. Minutes (thousands) to hours (millions)
To do:          Handle spectral and CORINE in one automatically. Generate .tif file.
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
from sklearn.metrics import classification_report
# For logging
import wandb
from wandb.sklearn import plot_confusion_matrix
# For saving
import joblib
# For command-line running
import argparse
import os

## Set up Weights and Biases (W&B) logging
wandb.init(project='land-cover-classification', entity='maplumridge')

## Set up parsing of command line arguments 
parser = argparse.ArgumentParser(description='Land Cover Classification')
parser.add_argument('--input_class', choices=['1', '2', '3'], help='Classes we want to reclassify', required=True)
parser.add_argument('--input_model', help='Name of output pkl file which model will be saved to', required=True)
parser.add_argument('--input_predictions', help='Name of CSV file containing higher level predictions', required=True)
parser.add_argument('--model_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
input_dir = '/home/users/map205/MRes_Data/'
# Location for saving the model and model predictions
output_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'

## Load hierarchical model
input_model = os.path.join(output_dir, args.input_model)
input_model = joblib.load(input_model)

# Load the level 1 predictions from previous test that generated a CSV file
input_predictions = os.path.join(output_dir, args.input_predictions)
input_class = int(args.input_class)
predictions_with_NaNs_df = pd.read_csv(input_predictions, na_values=['NaN', 'N/A'])
print("Input predictions column name:", predictions_with_NaNs_df.columns)
print("Input predictions data info:", predictions_with_NaNs_df.info())
print("Input predictions data frame (check indices, L2 and L3):", predictions_with_NaNs_df)

# Drop NaN rows
predictions_df = predictions_with_NaNs_df.copy()
predictions_df.dropna(inplace=True)
print("Predictions without NaNs (check indices):", predictions_df)
# Name of column containing predictions should remain the same, but keep in place in case other code changes
if input_class == 1:
    input_y_pred = predictions_df['Predicted_Value']
elif input_class == 3:
    input_y_pred = predictions_df['Predicted_Value']
elif input_class == 4:
    input_y_pred = predictions_df['Predicted_Value']
elif input_class == 5:
    input_y_pred = predictions_df['Predicted_Value']

# Select only rows containing predicted values of the class we want to reclassify into subclasses
input_class = int(args.input_class)
mask = (input_y_pred == input_class)
num_rows_y_pred_class = np.sum(mask)
print("Number of rows for the loaded class:", num_rows_y_pred_class)
# Apply mask to remove rows / load only rows of interest
predictions_df_filtered = predictions_df[mask]
# Map the subclasses we want to predict for each input primary class
if input_class == 1:
    df_combined_for_modelling = predictions_df_filtered[predictions_df_filtered['L2_groundtruth'].isin([1, 2])]
elif input_class == 3:
    df_combined_for_modelling = predictions_df_filtered[predictions_df_filtered['L2_groundtruth'].isin([4, 5, 6])]
elif input_class == 4:
    df_combined_for_modelling = predictions_df_filtered[predictions_df_filtered['L3_groundtruth'].isin([4, 5, 6])]    
elif input_class == 5:
    df_combined_for_modelling = predictions_df_filtered[predictions_df_filtered['L3_groundtruth'].isin([7, 8, 9, 10])]
else:
    raise ValueError("Invalid input_class")
    
# Again, as with train code, if input class is 1 or 3, we want to load the L2 ground truth data...
if input_class == 1 or input_class == 3:
    # Group data frame by 'L2_groundtruth'
    class_counts = df_for_modelling['L2_groundtruth'].value_counts()
    min_pixels_per_class = class_counts.min()
    print("Pixels per class:", class_counts)
    # Create copy of data frame for logging results and CSV
    df_for_results = df_for_modelling[['L2_groundtruth']].copy()
    df_for_CSV = df_for_modelling.copy()
    df_for_CSV.drop('L2_groundtruth', axis=1, inplace=True)
    # Remove redundant columns, ready for modelling
    df_for_modelling.drop(['L2_groundtruth', 'L3_groundtruth', 'y_pred'], axis=1, inplace=True)

# If input class is 4 or 5, we want to load the L3 ground tuth data
elif input_class == 4 or input_class == 5:
    # Group by 'L3_groundtruth'
    class_counts = df_for_modelling['L3_groundtruth'].value_counts()
    min_pixels_per_class = class_counts.min()
    print("Pixels per class:", class_counts)
    # Create copy of data frame for logging results and CSV
    df_for_results = df_for_modelling[['L3_groundtruth']].copy()
    df_for_CSV = df_for_modelling.copy()
    df_for_CSV.drop('L3_groundtruth', axis=1, inplace=True)
    # Remove redundant columns, ready for modelling
    df_for_modelling.drop(['L3_groundtruth', 'y_pred'], axis=1, inplace=True)

# Check
print("Data frame for comparing with upcoming predictions:", df_for_results)
print("Data frame for CSV (satellite & L3):", df_for_CSV)
print("Data frame for modelling (satellite data):", df_for_modelling)


### RUN THE MODEL ###
y_pred_test = input_model.predict(df_for_modelling)

## L2 TESTING METRICS ##

# Evaluate the model
accuracy_test = accuracy_score(y_pred_test, df_for_results)
print("Accuracy:", accuracy_test)
wandb.log({"Test Accuracy": accuracy_test})
wandb.finish()

class1_L2_subclasses = ['Urban fabric', 'Artificial, non-agricultural vegetated areas']
class3_L2_subclasses = ['Forest', 'Shrub and/or herbaceous vegetation associations', 'Open spaces with little or no vegetation']

class3_L3_subclasses_4 = ['Broad-leaved forest', 'Coniferous forest', 'Mixed forest']
class3_L3_subclasses_5 = ['Natural grassland', 'Moors and heathland', 'Sclerophyllous vegetation', 'Transitional woodland/shrub']

if input_class == 1:
    labels = class1_L2_subclasses
elif input_class == 3:
    labels = class3_L2_subclasses
elif input_class == 4:
    labels = class3_L3_subclasses_4
elif input_class == 5:
    labels = class3_L3_subclasses_5

# Plot matrix
wandb.sklearn.plot_confusion_matrix(y_pred_test, df_for_results, labels=labels)

# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(df_for_results, y_pred_test, average='macro')
MiPS = precision_score(df_for_results, y_pred_test, average='micro')
WPS = precision_score(df_for_results, y_pred_test, average='weighted')

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(df_for_results, y_pred_test, average='macro')
MiRS = recall_score(df_for_results, y_pred_test, average='micro')
WRS = recall_score(df_for_results, y_pred_test, average='weighted')

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(df_for_results, y_pred_test, average='macro')
MiFS = f1_score(df_for_results, y_pred_test, average='micro')
WFS = f1_score(df_for_results, y_pred_test, average='weighted')

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

# Log the metrics dictionary to W&B
wandb.log(metrics)

# Calculate classification report
classification_metrics = classification_report(y_pred_test, df_for_results, target_names=labels, output_dict=True)
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


### SAVE TO CSV  ###
print("CHECK INDICES df_test_NaN (should have satellite data, L2 & L3 (not L1)):", df_for_CSV)
y_pred_imputed = y_pred_test.copy()
# Concatenate df_test_NaN (satellite data anad ground truth labels) and y_pred_imputed (predictions from this model)
combined_df = pd.concat([df_for_CSV, y_pred_imputed], axis=1)
# Check
print("Combined data frame with original indices:", combined_df)

### SAVE CSV FILE 
model_predictions = args.model_predictions
model_predictions_file = os.path.join(output_dir, model_predictions)
combined_df.to_csv(model_predictions_file, index=False)


log_memory_usage()

wandb.finish()


