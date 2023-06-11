"""
Project:        AI4ER MRes
Code:           Calculate spectral indices, NDVI and NBI, and add these as two additional bands to the satellite data
Useage:         TBC
Order/workflow: After S2_data_download.js, before modelling.
Prerequisites:  TBC
Time:           5 minutes
To do:          Perform calculation and modification on ALL 10band files, handle file path, make use of functions, add exceptions/checks
"""

import rasterio
import numpy as np

# Run this on each of the exported '10bands' files from Google Earth Engine.
with rasterio.open('/Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/GEE/Autumn_10bands.tif') as satellite_data:
    # Read the bands required for calculating NDVI and NDBI
    # See https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
    # See formulae https://www.researchgate.net/publication/327971920_NDVI_NDBI_NDWI_Calculation_Using_Landsat_7_8
    bands = satellite_data.read() 
    band4 = satellite_data.read(3)  # Band 4 = red
    band8 = satellite_data.read(7)  # Band 8 = near infrared (NIR)
    band11 = satellite_data.read(10)  # Band 11 = short-wave infrared (SWIR)

    ## To check (although this will also show bands with NaN values)
    #print('Band 8:', band8)
    #print('Band 4:', band4)
    #print('Band 11:', band11)
    
    ## To print non-zero/non-NaN data, see example below for band 4:
    #non_zero_indices = np.nonzero(band4)
    #for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
    #    print(f'Band 4 value at ({i}, {j}): {band4[i, j]}')

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

    # Set any missing data pixels to NaN value
    ndvi[np.isnan(ndvi)] = -9999
    ndbi[np.isnan(ndbi)] = -9999

    # Load and modify metadata of input file to include the two additional indices
    metadata = satellite_data.profile 
    metadata.update(count=satellite_data.count + 2, dtype=np.float32)  # Update band count and convert data type to float"
    metadata.update(nodata=-9999)  # Set NaN value

    with rasterio.open('/Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/GEE/Autumn_10bands_2indices.tif', 'w', **metadata) as dst:
        # Write original bands
        for band_idx in range(satellite_data.count):
            dst.write(bands[band_idx], band_idx + 1)
        # Write NDVI and NDBI bands/indices
        dst.write(ndvi, satellite_data.count + 1)
        dst.write(ndbi, satellite_data.count + 2)