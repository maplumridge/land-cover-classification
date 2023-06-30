"""
Project:        AI4ER MRes
Code:           Calculate spectral indices, NDVI, NBDI and GCVI, and add these as two additional bands to the satellite data
Order:          After S2_data_download.js, before modelling.
Time:           ~5 minutes
Future:          Perform calculation and modification on ALL 10band files, handle file path, check content before writing.
"""

import rasterio
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Process satellite data and create indices")
parser.add_argument("--input_file", help="Input satellite file with 10 bands")
parser.add_argument("--output_file", help="Output satellite file with indices added")
args = parser.parse_args()

dir = '/gws/nopw/j04/ai4er/users/map205/mres/'

input_file = os.path.join(dir, args.input_file)
output_file = os.path.join(dir, args.output_file)

# Run this on each of the exported '10bands' files from Google Earth Engine
# Winter, Spring, Summer, Autumn
with rasterio.open(input_file) as satellite_data:
    # Read the bands required for calculating NDVI and NDBI
    # See https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
    # See formulae https://www.researchgate.net/publication/327971920_NDVI_NDBI_NDWI_Calculation_Using_Landsat_7_8
    bands = satellite_data.read() 
    band3 = satellite_data.read(2) # Band 3 = green
    band4 = satellite_data.read(3)  # Band 4 = red
    band8 = satellite_data.read(7)  # Band 8 = near infrared (NIR)
    band11 = satellite_data.read(9)  # Band 11 = short-wave infrared (SWIR)

    # Calculate NDVI - normalised difference vegetation index
    # NDVI = (NIR-Red) / (NIR+Red)
    ndvi_denominator = (band8 + band4)
    ndvi_denominator[ndvi_denominator == 0] = 1
    ndvi = (band8 - band4) / ndvi_denominator

    # Calculate NDBI - normalised difference built-up index
    # NDBI = (SWIR-NIR) / (SWIR+NIR)
    ndbi_denominator = (band11 + band8)
    ndbi_denominator[ndbi_denominator == 0] = 1
    ndbi = (band11 - band8) / ndbi_denominator
    
    # Calculate GCVI 
    # GCVI = (NIR/G) - 1
    # See https://www.nlaf.uk/Library/GetDoc.axd?ctID=ZWVhNzBlY2QtZWJjNi00YWZiLWE1MTAtNWExOTFiMjJjOWU1&rID=MjEwMDc=&pID=MjI5&attchmnt=False&uSesDM=False&rIdx=MTExNjk=&rCFU=
    gcvi = ((band8 / band3)) - 1

    # Set any missing data pixels to NaN value
    ndvi[np.isnan(ndvi)] = -9999
    ndbi[np.isnan(ndbi)] = -9999
    gcvi[np.isnan(gcvi)] = -9999

    # Modify metadata of satellite data to accept the new indices
    # Note: need to convert to float
    metadata = satellite_data.profile 
    metadata.update(count=satellite_data.count + 3, dtype=np.float32)  
    metadata.update(nodata=-9999)  

    with rasterio.open(output_file, 'w', **metadata) as dst:
        # Write original bands
        for band_idx in range(satellite_data.count):
            dst.write(bands[band_idx], band_idx + 1)
        # Write NDVI and NDBI bands/indices
        dst.write(ndvi, satellite_data.count + 1)
        dst.write(ndbi, satellite_data.count + 2)
        dst.write(gcvi, satellite_data.count + 3)

        print(dst)
