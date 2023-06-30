#!/bin/bash

# shell script to perfom merge operation on the two 10 band satellite images that are exported from Google Earth Engine

# Spring 10 bands, cloud masked 
gdal_merge.py -o SCL_Spring_10bands.tif SCL_Spring_10bands-0000000000-0000007424.tif SCL_Spring_10bands-0000000000-0000000000.tif
# Summer 10 bands, cloud masked
gdal_merge.py -o SCL_Summer_10bands.tif SCL_Summer_10bands-0000000000-0000007424.tif SCL_Summer_10bands-0000000000-0000000000.tif
# Winter 10 bands, cloud masked
gdal_merge.py -o SCL_Winter_10bands.tif SCL_Winter_10bands-0000000000-0000007424.tif SCL_Winter_10bands-0000000000-0000000000.tif
# Autumn 10 bands, cloud masked
gdal_merge.py -o SCL_Autumn_10bands.tif SCL_Autumn_10bands-0000000000-0000007424.tif SCL_Autumn_10bands-0000000000-0000000000.tif


# Winter 10 bands, low cloud cover
gdal_merge.py -o NoCloud_Winter_10bands.tif NoCloud_Winter_10bands-0000000000-0000007424.tif NoCloud_Winter_10bands-0000000000-0000000000.tif
# Spring 10 bands, low cloud cover
gdal_merge.py -o NoCloud_Spring_10bands.tif NoCloud_Spring_10bands-0000000000-0000007424.tif NoCloud_Spring_10bands-0000000000-0000000000.tif
# Summer 10 bands, low cloud cover
gdal_merge.py -o NoCloud_Summer_10bands.tif NoCloud_Summer_10bands-0000000000-0000007424.tif NoCloud_Summer_10bands-0000000000-0000000000.tif
# Autumn 10 bands, low cloud cover
gdal_merge.py -o NoCloud_Autumn_10bands.tif NoCloud_Autumn_10bands-0000000000-0000007424.tif NoCloud_Autumm_10bands-0000000000-0000000000.tif

# Test data, winter 10 bands, cloud masked
gdal_merge.py -o Braga_SCL_Winter_10bands.tif Braga_SCL_Winter_10bands-0000000000-0000007424.tif Braga_SCL_Winter_10bands-0000000000-0000000000.tif
# Test data, spring 10 bands, cloud masked
gdal_merge.py -o Braga_SCL_Spring_10bands.tif Braga_SCL_Spring_10bands-0000000000-0000007424.tif Braga_SCL_Spring_10bands-0000000000-0000000000.tif
# Test data, autumn 10 bands, cloud masked
gdal_merge.py -o Braga_SCL_Autumn_10bands.tif Braga_SCL_Autumn_10bands-0000000000-0000007424.tif Braga_SCL_Autumn_10bands-0000000000-0000000000.tif
# Test data, summer 10 bands, cloud masked
gdal_merge.py -o Braga_SCL_Summer_10bands.tif Braga_SCL_Summer_10bands-0000000000-0000007424.tif Braga_SCL_Summer_10bands-0000000000-0000000000.tif