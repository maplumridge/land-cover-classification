# Project Workflow
## AI4ER MRes 2023 
### Land Cover Classification in the Wildland-Urban Interface

# About
The purpose of this workflow document is to facilitate reproducibility of the results obtained during this 12-week project. Please reach out to me (<map205@cam.ac.uk>) if you encounter any issues or have any questions.

# Prerequisites
1. Create a JASMIN account. See [1].
2. Create a Google Earth Engine account. See [2].
3. Create a Weights and Biases account. See [3].
4. Access to QGIS software. See [4].

# Steps

## _Sentienl-2 data_

> **_Note:_**
Follow steps 1 to 4 if you wish to reproduce the Google Earth Engine steps. Alternatively, you can download the ready-made data from [my Google Drive](https://drive.google.com/drive/u/0/folders/1_iOAiCIQxiLXGo_sndqt5mg8I8ObrhCm) and jump ahead to step 5.

### Step 1: Download Portugal shapefile
Download the shapefile containing district boundaries in Portugal from the [Metabolism of Cities Data Hub](https://data.metabolismofcities.org/dashboards/lisbon/maps/35879/) [4].

### Step 2: Upload Portugal shapefile to Google Earth Engine 
Go to https://code.earthengine.google.com. 

Select the 'Assets' tab --> 'NEW' --> 'Shape files'. 

Select the dbf, shx, and shp files. 

The asset will be located under users/YOUR_USER_ID/PRT_regions.

### Step 3: Run Google Earth Engine Code
Copy the [S2_data_download.js](https://github.com/maplumridge/land-cover-classification/blob/dev/code/S2_data_download.js) code into the Google Earth Engine code editor.

'Run' the code to generate the export tasks.

### Step 4: Trigger export tasks
Select the 'Tasks' tab --> Select RUN for 'January_4bands' and 'January_10bands'. 

The data will be exported to your Google Drive account. 

<b>Note: </b> The total volume of data is 7.3 GB. Free Google Drive accounts are limited to 15 GB of storage.

### Step 5: Download data & copy to JASMIN HPC
Once you have downloaded the data from Google Drive to your local storage, run the following command:

```
scp -r /PATH/TO/GEE/DATA <userID>@login1.jasmin.ac.uk:/PATH/TO/HOME/DIRECTORY
```

In my case, the code used was:
```
scp -r /Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/GEE map205@login1.jasmin.ac.uk:/home/users/map205/MRes_Data/
```

<b>Note: </b> This takes about 15 minutes. The files are transferred to the home directory on the JASMIN login node and synchronised to the compute nodes. Home directories on JASMIN have a 100 GB storage limit. To check usage:
```
pdu -sh </path/to/your/home/directory>
```

### Step 6: Merge 10 band satellite data
The data downloaded from Google Earth Engine will be output as two separate files (this is because the Coimbra region happens to exist at the boundary between two satellite paths). To merge the files together for each season, use the merge.sh script in this repository
```
./merge.sh
```
### Step 7: Add spectral indices
Each sesaonal 10 band satellite image can now be updated to include additional spectral indices. Indices NDVI, NDBI and GCVI were used in this project, but the code can easily be adapted to compute other indices. Refer to Refer to [7](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/) for details of Sentinel-2 spectral bands. Use the script 2_add_indicies.py. An example is shown below.

```
python 2_add_indices.py --input_file SCL_Winter_10bands.tif --output_file SCL_Winter_10bands_3indices.tif
```

------------------------------------------------------------------------------------------------------------------------

## _CORINE reference data_

### Step 1: Download CORINE data
Download European-wide data for 2018 (CLC2018) from the [Copernicus Land Monitoring Service](https://land.copernicus.eu/pan-european/corine-land-cover) [8]. <b>Note: </b> Ensure that you download the data in raster format.

### Step 2: Reproject and resample 
The reference data needs to be reprojected to the same coordinate reference system as the satellite data (CRS 32629). The pixel resolution should also be upsampled to 10 metres to match the resolution of the satellite data.

Run the script <b> reproject.sh </b>
```
./reproject.sh
```
This will output a new .tif file at the target resoltuion and projection. 

### Step 3: Crop to region of interest
Load the data in QGIS along with one of the satellite images.
Raster
--> Extraction 
    --> Clip raster by extent 
        --> Clipping extent 
            --> Calcuate from layer (select the satellite data)

These steps will crop the CORINE data to the bounds of the satellite data. Now the data are aligned and can be used for modelling.

### Step 4: Reclassify to level 1, 2 and 3 class nomenclature
Finally, land cover values in the CORINE dataset need to be mapped to the classes of interest. See the [CORINE website](https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html) for a list of all 44 classes and their hierarchy.

Run the script <b> reclassify_CORINE.py </b>, which will output a new .tif file with the re-mapped classes. An example is shown below.

```
python 3_reclassify_CORINE.py --level 2 --output_file FINAL_L2_SPECTRAL_Braga.tif --classes spectral
```

------------------------------------------------------------------------------------------------------------------------

## _Model training_
The <b>script train_models.sh</b> file contains command examples, complete with command-line arguments, for a variety of model training options.
### Options
Choose the desired classification level
- CORINE Level 1
- CORINE Level 2
- CORINE Level 3
- spectral Level 1
- spectral Level 2
- spectral Level 3 

Choose which bands to sample from the satellite imagery
- 4
- 10
- 13 (10 + 3 indices)

Choose which cloud management mechanism you wish to use
- SCL - cloud-masked satellite data
- NoCloud - low cloud cover satellite data

To train models in parallel, execute the following command on JASMIN:
```
sbatch train_models.sh
``` 

<b>IMPORTANT</b> you must be connected to an interactive node (e.g sci6), not a login node, to submit jobs to the Lotus batch claster. To test performance on different queues, modify memory or CPU requirements, refer to the [JASMIN documentation](https://help.jasmin.ac.uk/category/4889-slurm)

------------------------------------------------------------------------------------------------------------------------

## _Model testing_
After completing model training and saving the best models, the test models can also be run in parallel on the Lotus cluster. The file <b>test_models.sh</b> contains command examples, complete with command-line arguments, for a variety of model testing options.
```
sbatch test_models.sh
```

## REFERENCES

[1] https://accounts.jasmin.ac.uk/application/new/

[2] https://code.earthengine.google.com/register

[3] https://wandb.ai/site

[4] https://qgis.org/en/site/

[5] https://data.metabolismofcities.org/dashboards/lisbon/maps/35879/

[6] https://help.jasmin.ac.uk/article/176-storage

[7] https://help.jasmin.ac.uk/article/3810-data-transfer-tools-rsync-scp-sftp 

[8] https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/

[9] https://land.copernicus.eu/pan-european/corine-land-cover

[10] https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html

[11] https://help.jasmin.ac.uk/category/4889-slurm
