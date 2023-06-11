"""
Project:        AI4ER MRes
Code:           Reclassify level 3 classes of the ground truth dataset (CORINE) to level 1 and level 2 classes 
Useage:         TBC
Order/workflow: After CORINE data has been cropped and reprojected in QGIS (see workflow.md)
Prerequisites:  TBC
Time:           Seconds
To do:          Remove specific file paths from code. Merge L1 and L2 together in one file.
"""

import rasterio
import numpy as np

# Open CORINE ground truth data (already cropped and reprojected)
with rasterio.open('/Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/Groundtruth_data/CORINE_32629_10m_Cropped.tif') as groundtruth:
    # Read as numpy array
    array = groundtruth.read(1)
    # Map target classes (level 1 or level 2) to the original input classes (level 3)
    # Refer to https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html
    new_to_original = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], #e.g. classes 1 to 11 from the original CORINE data will be reclassified as class 1
        2: list(range(12, 23)),
        3: list(range(23, 35)),
        4: list(range(35, 40)),
        5: list(range(40, 45))
    }
    ## Apply reclassification
    # Create array for data with 'new' classes
    modified_array = np.zeros_like(array, dtype=np.uint8)
    # Assign 'new' classes
    for new_classes, original_classes in new_to_original.items():
        modified_array[np.isin(array, original_classes)] = new_classes

    ## Save results
    # Load metadata of input CORINE/ground truth file
    metadata = groundtruth.profile
    # Modify metadata to contain the new classes for each pixel
    metadata.update(count=1, dtype=rasterio.uint8)
    # Create & save a new .tif ground truth file containing the new classes for each pixel
    output_file = '/Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/Groundtruth_data/L1_CORINE_32629_10m_Cropped_test.tif'
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(modified_array, 1)
    
    # Confirm creation of the new ground truth file
    print(f"Modified classification saved as {output_file}")