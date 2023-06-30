#!/bin/bash

# command to reproject and upsample CORINE dataset
gdalwarp -t_srs EPSG:32629 -tr 10.0 10.0 -r near -of GTiff Level1_CORINE.tif OUTPUT.tif