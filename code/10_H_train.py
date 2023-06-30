"""
Project:        AI4ER MRes
Code:           Train a random forest model to perfrom hierarchical classifcation 
                based on predictions from previous level 1 model (L1_train.py)
                and subsequent generation of output predictions
Order:          After L1_train.py
Time to run:    Minutes
To do:          Handle spectral and CORINE at the same time
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import resample
from sklearn.metrics import classification_report
# For logging
import wandb
from wandb.sklearn import plot_confusion_matrix
import plotly.graph_objects as go
import itertools
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
    with open('train_H_model_memory.log', 'a') as f:
        f.write(f'{memory_usage}\n')

## Set up Weights and Biases (W&B) logging
wandb.init(project='land-cover-classification', entity='maplumridge')

## Set up parsing of command line arguments 
parser = argparse.ArgumentParser(description='Land Cover Classification')
parser.add_argument('--input_class', choices=['1', '3', '4', '5', '6'], help='Classes we want to reclassify', required=True)
parser.add_argument('--model_output', help='Name of output pkl file which model will be saved to', required=True)
parser.add_argument('--input_predictions', help='Name of CSV file containing higher level predictions', required=True)
parser.add_argument('--output_predictions', help='Name of output CSV file which model predictions will be saved to', required=True)
args = parser.parse_args()

# Location of input satellite files and ground truth file
# Note: remember the '/' after the directory...
input_dir = '/gws/nopw/j04/ai4er/users/map205/'
# Location for saving the model and model predictions
output_dir = '/gws/nopw/j04/ai4er/users/map205/mres/'

# Load model predictions from level 1 model training
input_predictions = os.path.join(output_dir, args.input_predictions)
input_class = int(args.input_class)
predictions_df = pd.read_csv(input_predictions)
print("Input predictions column name:", predictions_df.columns)
print("Input predictions data info:", predictions_df.info())
# Need to modify the model that feeds into this... column names are not consistently applied
if input_class == 1:
    input_y_pred = predictions_df['y_pred']
elif input_class == 3:
    input_y_pred = predictions_df['y_pred']
elif input_class == 4:
    input_y_pred = predictions_df['test_predictions']
elif input_class == 5:
    input_y_pred = predictions_df['test_predictions']

# Creat mask based on input predictions of interest
mask = (input_y_pred == input_class)
num_rows_y_pred_class = np.sum(mask)
print("Number of rows for the loaded class:", num_rows_y_pred_class)
# Apply mask to remove rows we are not interested in
predictions_df_filtered = predictions_df[mask]
# Remove rows with NaN values
predictions_df_filtered.dropna(inplace=True)

# Further filter and select only the rows where the ground truth pixel belongs to 
# one of the subclasses we are interested in
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

# Check dsta frame contains predictions from level 1 model in addition to satellite data, level 2 and level 3 classes
print("Filtered Data Frame:", df_combined_for_modelling.head(10))
# Shuffle
shuffled_df = df_combined_for_modelling.sample(frac=1, random_state=42)
# This works and shows that the index values are maintained and the pixels are shuffled
print("Shuffled Data Frame:", shuffled_df.head(10)) 

# Now...
# Classes 1 and 3 are level 2 classes
# We can still sample a balanced number for each subclass since this should contain
# a relatively high number of pixels...
if input_class == 1 or input_class == 3:
    class_counts = shuffled_df['L2_groundtruth'].value_counts()
    min_pixels_per_class = class_counts.min()
    print("Pixels per class:", class_counts)

    df_combined_for_modelling_sample = shuffled_df.groupby('L2_groundtruth', as_index=False).apply(
        lambda x: x.sample(n=min_pixels_per_class, replace=True, random_state=42)
    ).reset_index(level=0, drop=True)

    random_state = random.randint(0, 100000)
    print("Random State:", random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        df_combined_for_modelling_sample.drop(['y_pred', 'L2_groundtruth'], axis=1),
        df_combined_for_modelling_sample['L2_groundtruth'],
        test_size=0.2,
        random_state=random_state
    )

# However, classes 4 and 5 are level 3 subclasses
# After several rounds of training, so few samples are left 
# Have removed the equal sampling of pixels per class and instead just sample all pixels
# for thsi final stage of classification 
elif input_class == 4 or input_class == 5:
    class_counts = shuffled_df['L3_groundtruth'].value_counts()
    print("Pixels per class:", class_counts)

    df_combined_for_modelling_sample = shuffled_df.groupby('L3_groundtruth', as_index=False).apply(
        lambda x: x.sample(frac=1, replace=True, random_state=42)  
    ).reset_index(level=0, drop=True)

    random_state = random.randint(0, 100000)
    print("Random State:", random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        df_combined_for_modelling_sample.drop(['test_predictions'], axis=1),
        df_combined_for_modelling_sample['L3_groundtruth'],
        test_size=0.2,
        random_state=random_state
    )

# Remove 'L3_groundtruth' column from X_train, maintain for X_test
# Ready for modelling
X_train.drop(['L3_groundtruth'], axis=1, inplace=True)
# Create a copy of X_test (with 'L3_groundtruth', for use in CSV file
# and downstream modelling/training)
X_test_CSV = X_test.copy()
# Ready for modelling
X_test.drop(['L3_groundtruth'], axis=1, inplace=True)

param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 15, 30],
    'min_samples_leaf': [1, 3, 5]
}

# Train second random forest model
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
# Adding verbose mode https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://stackoverflow.com/questions/24121018/sklearn-gridsearch-how-to-print-out-progress-during-the-execution
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# Get best model and hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the test set
predictions_test = best_model.predict(X_test)

## L2 TESTING METRICS ##
# Evaluate the model
accuracy_test = accuracy_score(y_test, predictions_test)
print("Accuracy:", accuracy_test)
wandb.log({"L2 Test Accuracy": accuracy_test})

# Metrics from grid search CV
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
### END OF NEW CODE ###
# Precision
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
MaPS = precision_score(y_test, predictions_test, average='macro')
MiPS = precision_score(y_test, predictions_test, average='micro')
WPS = precision_score(y_test, predictions_test, average='weighted')

# Recall
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
MaRS = recall_score(y_test, predictions_test, average='macro')
MiRS = recall_score(y_test, predictions_test, average='micro')
WRS = recall_score(y_test, predictions_test, average='weighted')

# F1 score
# See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
MaFS = f1_score(y_test, predictions_test, average='macro')
MiFS = f1_score(y_test, predictions_test, average='micro')
WFS = f1_score(y_test, predictions_test, average='weighted')

### TO BE REVIEWED ###
# Average is macro...
metrics = {
    #"Test Accuracy": accuracy_test,
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

""" # Print hyperparameter combinations and accuracy values
print("Hyperparameter Combinations and Accuracy:")
for i, combination in enumerate(param_combinations):
    print("Combination:", combination)
    print("Accuracy:", accuracy_values[i])
    print()
# Create a 3D scatter plot
fig = go.Figure(data=go.Scatter3d(
    x=[combo[0] for combo in param_combinations],
    y=[combo[1] for combo in param_combinations],
    z=[combo[2] for combo in param_combinations],
    mode='markers',
    marker=dict(
        size=accuracy_values,
        color=accuracy_values,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=['Accuracy: {:.3f}'.format(acc) for acc in accuracy_values]
))

fig.update_layout(
    title='Accuracy vs. Grid Search Parameters',
    scene=dict(
        xaxis_title='n_estimators',
        yaxis_title='max_depth',
        zaxis_title='min_samples_leaf'
    )
) """

""" # Log the 3D scatter plot
wandb.log({"Accuracy vs. Grid Search Parameters": fig})
# Log best parameters
wandb.config.update(best_params) """

### CONTINUE FROM HERE ###

# Log confusion matrix
# https://docs.wandb.ai/guides/integrations/scikit
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
# Labels are selected as command-line arg

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
wandb.sklearn.plot_confusion_matrix(y_test, predictions_test, labels=labels)

# Calculate classification report
classification_metrics = classification_report(y_test, predictions_test, target_names=labels, output_dict=True)

# Access precision, recall, F1, and accuracy for each class
for class_name in labels:
    metrics = classification_metrics[class_name]
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score_class = metrics['f1-score']
    #accuracy = metrics['accuracy']
    
    # Print or use the metrics as needed
    print(f"Class: {class_name}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score_class}")
    #print(f"Accuracy: {accuracy}")

#########################
## Training metrics
# Make predictions on training data
y_pred_train = best_model.predict(X_train)

# Overall accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", accuracy_train)

# Precision, recall & F1 score 
precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

# Average is macro...
metrics = {
    "Train macro precision": precision_train,
    "Train macro recall": recall_train,
    "Train macro F1": f1_train,
    "Train accuracy": accuracy_train
}

# Log metrics to W&B
wandb.log(metrics)

## SAVE MODEL & PREDICITIONS TO CSV FILE ##
# Best model hyperparameters
print("Best model hyperparameters:", best_params)
# Save best model
model_output = args.model_output
model_output_file = os.path.join(output_dir, model_output)
joblib.dump(best_model, model_output_file)

## Load input CSV with level 1 predictions
model_predictions = args.output_predictions
print("X_test before reset index:", X_test_CSV)
## This assumes that the order of y_pred is the same as the order of X_test
# Create data frame of predictions
predictions_df = pd.DataFrame({'test_predictions': predictions_test})
# Reset index of X_test (otherwise concatination does not work)
X_test_CSV.reset_index(drop=True, inplace=True)
# Create data frame of X_test (satellite data), y_pred (predictions from this hierachical model), and corresponding L3_groundtruth values
combined_df = pd.concat([X_test_CSV, predictions_df], axis=1)

## To check, 
print("CHECK INDICES MATCH")
print("X test (satellite data) with L3 ground truth data:", X_test_CSV)
print("Model predictions:", predictions_df)
print("Output CSV data:", combined_df)

## Save DF with predictions, satellite data and L3 ground truth to CSV file
# For use in downstream hierarchical model
model_predictions_file = os.path.join(output_dir, model_predictions)
# Export to CSV file
combined_df.to_csv(model_predictions_file, index=False)

# memory usage
log_memory_usage()

wandb.finish()





""" # Reshape the satellite data to match the shape of the mask
satellite_data_reshaped = satellite_data_2d[mask]
groundtruth_data_flat_reshaped = groundtruth_data_flat[mask]

## Reshape the satellite data to match the shape of the mask
#satellite_data_reshaped = satellite_data_2d[np.where(mask)]
#groundtruth_data_flat_reshaped = groundtruth_data_flat[np.where(mask)]

# Create the DataFrame combining the reshaped satellite data, level 1 predictions and level 2 ground truth data
df_second_model = pd.DataFrame(satellite_data_reshaped, columns=['band'+str(i) for i in range(1, bands+1)])
df_second_model['input_prediction'] = input_y_pred[np.where(mask)]
df_second_model['groundtruth'] = groundtruth_data_flat_reshaped """

""" # Remove rows with NaN values
df_second_model.dropna(inplace=True)
df_second_model= df_second_model[df_second_model['groundtruth'].isin([9, 10, 11])]
# Print the data frame
print("Filtered Data Frame:", df_second_model.head(10))
# Shuffle data frame
shuffled_df = df_second_model.sample(frac=1, random_state=42)
# This works and shows that the index values are maintained and the pixels are shuffled
print("Shuffled Data Frame:", shuffled_df.head(10)) 
# Group the shuffled data frame by 'groundtruth_leve21', sample all pixels
df_second_model_sample = shuffled_df.groupby('groundtruth', as_index=False).apply(lambda x: x.sample(frac=1.0, replace=True, random_state=42)).reset_index(level=0, drop=True)
# Add the original indexes as a column in df_level1_sample
df_second_model_sample['original_index'] = df_second_model_sample.index
print("Shuffled data frame grouped by class", df_second_model_sample.head(10))

# Remove the 'original_index' column from the DataFrame
df_second_model_sample.drop('original_index', axis=1, inplace=True) """

""" # Remove rows with NaN values
predictions_df.dropna(inplace=True)
# Need to find the correct column name for 'groundtruth'
predictions_df_model= predictions_df[predictions_df['L2_groundtruth'].isin([9, 10, 11])]
# Print the data frame
print("Filtered Data Frame:", predictions_df_model.head(10))
# Shuffle data frame
shuffled_df = predictions_df_model.sample(frac=1, random_state=42)
# This works and shows that the index values are maintained and the pixels are shuffled
print("Shuffled Data Frame:", shuffled_df.head(10)) 
# Group the shuffled data frame by 'groundtruth_leve21', sample all pixels
predictions_df_model_sample = shuffled_df.groupby('L2_groundtruth', as_index=False).apply(lambda x: x.sample(frac=1.0, replace=True, random_state=42)).reset_index(level=0, drop=True)

## Filter out the pixels with predicted classes 1, 2, 4, and 5
## shouldn't need this since I use the mask above...
#filtered_df = df_second_model[~df_second_model['level1_prediction'].isin([1, 2, 4, 5])]

# Split the data into training and testing sets for the second model
X_train_second_model, X_test_second_model, y_train_second_model, y_test_second_model = train_test_split(
    df_second_model.drop(['y_pred', 'L2_groundtruth'], axis=1),
    df_second_model['L2_groundtruth'],
    test_size=0.2,
    random_state=42
) """

""" # Calculate the number of pixels per class
train_pixels_per_class = y_train_second_model.value_counts()
test_pixels_per_class = y_test_second_model.value_counts()
print("L2 number of pixels per class (Training Data):")
print(train_pixels_per_class)
print("L2 number of pixels per class (Testing Data):")
print(test_pixels_per_class)


# Define the parameter grid for the second model's grid search
param_grid_level1 = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 3, 5]
}

# Train the second random forest model
rf_second_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search_second_model = GridSearchCV(estimator=rf_second_model, param_grid=param_grid_second_model, cv=5)
### THIS BIT FAILS WITH ERROR :  grid_search_second_model.fit(X_train_second_model, y_train_second_model)
# Removed NaN rows from data frame...
grid_search_second_model.fit(X_train_second_model, y_train_second_model)

# Get the best model and its hyperparameters
best_model_second_model = grid_search_second_model.best_estimator_
best_params_second_model = grid_search_second_model.best_params_

# Make predictions on the test set
predictions_test = best_model_second_model.predict(X_test_second_model)

## L2 TESTING METRICS ##
# Evaluate the model
accuracy_test = accuracy_score(y_test_second_model, predictions_test)

# Calculate precision, recall, and F1 score on the training set
precision_test = precision_score(y_test_second_model, predictions_test, average='macro')
recall_test = recall_score(y_test_second_model, predictions_test, average='macro')
f1_test = f1_score(y_test_second_model, predictions_test, average='macro')

# Log the metrics to Weights and Biases
wandb.log({"L2 Test Accuracy": accuracy_test})
wandb.log({"L2 Test Precision": precision_test})
wandb.log({"L2 Test Recall": recall_test})
wandb.log({"L2 Test F1 Score": f1_test})

## L2 TESTING METRICS ##
# Make predictions on the training set using the best model
y_pred_train = best_model_second_model.predict(X_train_second_model)

# Calculate accuracy on the training set
accuracy_train = accuracy_score(y_train_second_model, y_pred_train)
print("Training Accuracy:", accuracy_train)

# Calculate precision, recall, and F1 score on the training set
precision_train = precision_score(y_train_second_model, y_pred_train, average='macro')
recall_train = recall_score(y_train_second_model, y_pred_train, average='macro')
f1_train = f1_score(y_train_second_model, y_pred_train, average='macro')

# Log the metrics to Weights and Biases
wandb.log({"L2 Training Accuracy": accuracy_train})
wandb.log({"L2 Training Precision": precision_train})
wandb.log({"L2 Training Recall": recall_train})
wandb.log({"L2 Training F1 Score": f1_train})

# Confusion matrix
L2_classes = ['Forest', 'Shrub and/or herbaceous vegetation associations', 'Open spaces with little or no vegetation']

L3_classes = ['TBC']

label_type = args.label_type

if label_type == 'L2':
    labels = L2_classes
elif label_type == 'L3':
    labels = L3_classes

# Plot matrix
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=labels)


## SAVE MODEL & PREDICITIONS ##
# Best model hyperparameters
print("Best model hyperparameters:", best_params_second_model)
# Save best model
model_output = args.model_output
model_output_file = os.path.join(output_dir, model_output)
joblib.dump(best_model_second_model, model_output_file)

### WORK ON THIS LATER ###
# Combine X_test, y_pred, and corresponding L2_groundtruth values
predictions_df = pd.DataFrame({'y_pred': y_pred})
combined_df = pd.concat([X_test, predictions_df], axis=1)
## To check, 
print("CHECK INDICES MATCH")
print("Shuffled testing data:", X_test)
print("Output CSV data:", combined_df)
## Save the combined DataFrame to a CSV file
model_predictions_file = os.path.join(output_dir, model_predictions)
# Export the combined DataFrame to a CSV file
combined_df.to_csv(model_predictions_file, index=False) """
