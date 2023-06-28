"""
Project:        AI4ER MRes
Code:           Reclassify level 3 classes of the ground truth dataset (CORINE) to level 1 and level 2 classes 
Useage:         TBC
Order/workflow: After CORINE data has been cropped and reprojected in QGIS (see workflow.md)
Prerequisites:  TBC
Time:           Seconds
To do:          Remove specific file paths from code. Merge L1 and L2 together in one file.
"""

import argparse
import rasterio
import numpy as np

parser = argparse.ArgumentParser(description="Reclassify CORINE")
parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Reclassify to level 1, 2 or 3 classes")
parser.add_argument("--output_file", type=str, help="Output file path and name for reclassified file")
parser.add_argument("--classes", type=str, choices=['CORINE', 'spectral'], help="Reclassify using the CORINE scheme or adapted spectral scheme")
args = parser.parse_args()

# CORINE Level 2
if args.level == 2 and args.classes == 'CORINE':
    hybrid_classes = {
        # Urban
        1: [1, 2],
        2: [3, 4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11],
        # Agri
        5: [12, 13, 14],
        6: [15, 16, 17],
        7: [18],
        8: [19, 20, 21, 22],
        # Forest
        9: [23, 24, 25],
        10: [26, 27, 28, 29],
        11: [30, 31, 32, 33, 34],
        # Wetlands
        12: [35, 36, 37, 38, 39],
        # Water bodies
        13: [40, 41, 42, 43, 44],
    }
# CORINE Level 3
elif args.level == 3 and args.classes == 'CORINE': 
    hybrid_classes = {
        # Urban
        1: [1, 2],
        2: [3, 4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11],
        # Agri
        5: [12, 13, 14],
        6: [15, 16, 17],
        7: [18],
        8: [19, 20, 21, 22],
        # Forest
        9: [23],
        10: [24],
        11: [25],
        12: [26],
        13: [27],
        14: [28],
        15: [29],
        16: [30, 31, 32, 33, 34],
        # Wetlands
        17: [35, 36, 37, 38, 39],
        # Water bodies
        18: [40, 41, 42, 43, 44],
    }

### FOR SPECTRAL CLASSES
# Level 3
elif args.level == 3 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        3: [12, 13, 14], 
        4: [15, 16, 17], 
        5: [18],
        6: [19, 20, 21, 22],
        7: [22],
        8: [23],
        9: [24],
        10: [25],
        11: [26],
        12: [27],
        13: [28],
        14: [29, 30, 31, 32, 33],
        15: [35, 36, 37, 38, 39],
        16: [40, 41, 42, 43, 44, 45]
    }

# Level 2
elif args.level == 2 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11],
        3: [12, 13, 14], 
        4: [15, 16, 17], 
        5: [18],
        6: [19, 20, 21, 22],
        7: [23, 24, 25],
        8: [26, 27, 28, 29],
        9: [30, 31, 32, 33, 34],
        10: [35, 36, 37, 38, 39],
        11: [40, 41, 42, 43, 44, 45]
    }

# Level 1
elif args.level == 1 and args.classes == 'spectral':
    hybrid_classes = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39],
        4: [40, 41, 42, 43, 44, 45]
    }
    
# Open input CORINE file to reclassify
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
    output_file = args.output_file if args.output_file else "modified_classification.tif"
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(modified_array, 1)

print(f"Reclassified file saved as {output_file}")
