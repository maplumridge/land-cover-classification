import os
import rasterio
from rasterio.merge import merge
import csv

# Set the input directory path
input_dir = '/gws/nopw/j04/ai4er/users/map205/BigEarthNet-v1.0/'

# Initialize list of source files to merge
src_files_to_merge = []

# Collect all subdirectories within the input directory
subdirs = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]

# Loop through subdirectories and add first .tif file to list of source files
for subdir in subdirs:
    tif_files = [os.path.join(subdir, name) for name in os.listdir(subdir) if name.endswith('.tif')]
    if len(tif_files) > 0:
        src_files_to_merge.append(tif_files[0])

# Divide the input files into batches of 1,000
file_batches = [src_files_to_merge[i:i+1000] for i in range(0, len(src_files_to_merge), 1000)]

# Loop through each batch and merge the files
for i, batch in enumerate(file_batches):
    # Open all .tif files in the current batch using rasterio
    src_files = [rasterio.open(f) for f in batch]

    # Merge the .tif files using rasterio.merge
    merged, out_transform = merge(src_files)

    # Set the output file path and metadata
    output_file = f'/gws/nopw/j04/ai4er/users/map205/mres/global_merged_file_{i}.tif'
    out_meta = src_files[0].meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': merged.shape[1],
                     'width': merged.shape[2],
                     'transform': out_transform})

    # Write the merged .tif file to disk
    with rasterio.open(output_file, 'w', **out_meta) as dst:
        dst.write(merged)
    
    # Close all the opened source files
    for src_file in src_files:
        src_file.close()

# Save the filenames of the original .tif files for this batch in a .csv file
    filenames_file = f'/gws/nopw/j04/ai4er/users/map205/mres/global_merged_file_{i}_filenames.csv'
    with open(filenames_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for src_file in src_files:
            writer.writerow([src_file.name])

