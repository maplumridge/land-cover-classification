"""
Project:        AI4ER MRes
Code:           Reclassify level 3 classes of the ground truth dataset (CORINE) to remapped level 1, level 2 and level 3 classes 
Order:          After CORINE data has been cropped and reprojected in QGIS (see workflow.md)
Time:           Seconds 
To do:          Make file path agnostic. Remove hard-coded input file.
"""

import argparse
import rasterio
import numpy as np

parser = argparse.ArgumentParser(description="reclassify CORINE")
parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Reclassify to level 1, 2 or 3 classes")
parser.add_argument("--output_file", type=str, help="Output file path AND name for reclassified file")
parser.add_argument("--classes", type=str, choices=['CORINE', 'spectral'], help="Reclassify using to adapted CORINE or adapted spectral classes")
args = parser.parse_args()

# CORINE Level 2
if args.level == 2 and args.classes == 'CORINE':
    hybrid_classes = {
        # Urban
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        # Agri
        3: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        # Forest
        4: [23, 24, 25],
        5: [26, 27, 28, 29],
        6: [30, 31, 32, 33, 34],
        # Wetlands
        7: [35, 36, 37, 38, 39],
        # Water bodies
        8: [40, 41, 42, 43, 44],
    }
# CORINE Level 3
elif args.level == 3 and args.classes == 'CORINE': 
    hybrid_classes = {
        # Urban
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        # Agri
        3: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        # Forest
        4: [23],
        5: [24],
        6: [25],
        7: [26],
        8: [27],
        9: [28],
        10: [29],
        11: [30, 31, 32, 33, 34],
        # Wetlands
        12: [35, 36, 37, 38, 39],
        # Water bodies
        13: [40, 41, 42, 43, 44],
    }

### FOR SPECTRAL CLASSES
# Level 3
elif args.level == 3 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        3: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        4: [23],
        5: [24],
        6: [25],
        7: [26],
        8: [27],
        9: [28],
        10: [29],
        11: [30, 31, 32, 33, 34],
        12: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    }

# Level 2
elif args.level == 2 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        3: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        4: [23, 24, 25],
        5: [26, 27, 28, 29],
        6: [30, 31, 32, 33, 34],
        7: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    }

# Level 1
elif args.level == 1 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    }
    
# Load file for reclassification (original file is preserved)
with rasterio.open('/gws/nopw/j04/ai4er/users/map205/mres/L3_CORINE_32629_10m_Cropped.tif') as groundtruth:
#with rasterio.open('/gws/nopw/j04/ai4er/users/map205/mres/Braga_CORINE_Cropped_L3.tif') as groundtruth:
    array = groundtruth.read(1)
    modified_array = np.zeros_like(array, dtype=np.uint8)
    # Assign new classes
    for new_classes, original_classes in hybrid_classes.items():
        modified_array[np.isin(array, original_classes)] = new_classes
    # Load metadata of input CORINE/ground truth file
    metadata = groundtruth.profile
    metadata.update(count=1, dtype=rasterio.uint8)
    # Save reclassified .tif file as new file
    output_file = args.output_file
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(modified_array, 1)
