#!/bin/bash

gdal_merge.py -o SCL_Spring_10bands.tif SCL_Spring_10bands-0000000000-0000007424.tif SCL_Spring_10bands-0000000000-0000000000.tif
gdal_merge.py -o SCL_Summer_10bands.tif SCL_Summer_10bands-0000000000-0000007424.tif SCL_Summer_10bands-0000000000-0000000000.tif
gdal_merge.py -o SCL_Winter_10bands.tif SCL_Winter_10bands-0000000000-0000007424.tif SCL_Winter_10bands-0000000000-0000000000.tif


gdal_merge.py -o NoCloud_Winter_10bands.tif NoCloud_Winter_10bands-0000000000-0000007424.tif NoCloud_Winter_10bands-0000000000-0000000000.tif
gdal_merge.py -o NoCloud_Spring_10bands.tif NoCloud_Spring_10bands-0000000000-0000007424.tif NoCloud_Spring_10bands-0000000000-0000000000.tif
gdal_merge.py -o NoCloud_Summer_10bands.tif NoCloud_Summer_10bands-0000000000-0000007424.tif NoCloud_Summer_10bands-0000000000-0000000000.tif
gdal_merge.py -o NoCloud_Autumn_10bands.tif NoCloud_Autumn_10bands-0000000000-0000007424.tif NoCloud_Autumm_10bands-0000000000-0000000000.tif


gdal_merge.py -o Braga_SCL_Winter_10bands.tif Braga_SCL_Winter_10bands-0000000000-0000007424.tif Braga_SCL_Winter_10bands-0000000000-0000000000.tif
gdal_merge.py -o Braga_SCL_Spring_10bands.tif Braga_SCL_Spring_10bands-0000000000-0000007424.tif Braga_SCL_Spring_10bands-0000000000-0000000000.tif
gdal_merge.py -o Braga_SCL_Autumn_10bands.tif Braga_SCL_Autumn_10bands-0000000000-0000007424.tif Braga_SCL_Autumn_10bands-0000000000-0000000000.tif
gdal_merge.py -o Braga_SCL_Summer_10bands.tif Braga_SCL_Summer_10bands-0000000000-0000007424.tif Braga_SCL_Summer_10bands-0000000000-0000000000.tif
