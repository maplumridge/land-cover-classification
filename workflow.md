# Project Workflow
## AI4ER MRes 2023 
### Land Cover Classification in the Wildland-Urban Interface

# About
XXX

# Prerequisites
1. Create a JASMIN account. See [1].
2. Create a Google Earth Engine account. See [2].
3. Create a Weights and Biases account. See [3]

# Steps

## DATA&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Sentinel-2_

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

<b>Note: </b> Free Google Drive accounts are limited to 15 GB of storage.

### Step 5: Download data & copy to JASMIN HPC
Once you have downloaded the data from Google Drive to your local storage, run the following command:

```
scp -r /PATH/TO/GEE/DATA <userID>@login1.jasmin.ac.uk:/PATH/TO/HOME/DIRECTORY
```

In my case, the code used was:
```
scp -r /Users/meghanplumridge/Library/CloudStorage/OneDrive-UniversityofCambridge/MRes_Data/GEE map205@login1.jasmin.ac.uk:/home/users/map205/MRes_Data/
```

<b>Note: </b> The files are transferred to the home directory on the JASMIN login node and synchronised to the compute nodes. Home directories on JASMIN have a 100 GB storage limit. 

------------------------------------------------------------------------------------------------------------------------

# MODIFY BELOW

## DATA &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_BigEarthNet_
### Step 1: Download the data from BigEarthNet
Download the file "BigEarthNet-S2" from the following URL https://bigearth.net/ 
The dataset is 66GB. 

### Step 2: Copy the tar file to JASMIN
Note: The files are transferred to my home directory on the login node, but this is synchronised to the compute nodes. However, I will NOT unzip the file in my home directory because I only have a 100GB allowance.
```
scp /Users/meghanplumridge/Downloads/BigEarthNet-S2-v1.0.tar.gz map205@login1.jasmin.ac.uk://home/users/map205/
```

### Step 3: Login to JASMIN and unzip the file
Note: I will unzip the data in the AI4ER group workspace on JASMIN because my personal workspace has a limit of 100 GB. See [5]. This takes > 24 hours to run.

First, create a directory in the AI4ER group workspace:
```
cd /gws/nopw/j04/ai4er/users/
mkdir map205
```
Then unzip the files and output them to this directory:
```
cd /home/users/map205
nohup tar -zxvf BigEarthNet-S2-v1.0.tar.gz -C /gws/nopw/j04/ai4er/users/BigEarthNet-v1.0/ --skip-old-files > /home/myusername/logs/tar_output.log 2>&1 &
```

To check the status of the task:
```
ps -ef | grep map205
```

### Step 4: Merge the files together to visualise in QGIS
Note: to check the number of files you can have open at once, I ran the following command in my personal workspace (/home/users/map205) and the group workspace (/gws/nopw/j04/ai4er/users/map205)
```
ulimit -a
```
Which shows that I can have >4000 files open in my home workspace:
```
open files                      (-n) 4096
```
But only 1000 files open in the group workspace
```
open files                      (-n) 1024
```

I decided to run the merge_files.py script in my home directory (/home/users/map205) since I can process more files at once here.

```
conda activate sea-ice-classification
nohup python merge_files.py &
```

### Step 5: Copy the files to my computer to visualise in QGIS
See instructions at [6]

```
scp -r "map205@login1.jasmin.ac.uk:/home/users/map205/mres/*" /Users/meghanplumridge/Desktop/merged_files/
```
Note: I needed to add parantheses around the jasmin login path otherwise it failed with error zsh: no matches found: map205@login1.jasmin.ac.uk:/home/users/map205/mres/*

<img width="975" alt="Screenshot 2023-04-24 at 15 14 38" src="https://user-images.githubusercontent.com/114443493/234087070-f1fb2929-ec37-4dde-95e4-c7230b337d62.png">

There are 98 'global_merged' tif files covering Portugal (and wider). These each contain 1000 smaller raw .tif files, so altogether there are 98,000 files to review.

### Step 6: Copy the 98,000 directories containing data covering Portugal to a new directory

```
cd /gws/nopw/j04/ai4er/users/map205/mres/
mkdir portugal_data
```

Use the script portugal_data.py, which copies the directories to /gws/nopw/j04/ai4er/users/map205/mres/portugal_data

The total volume of data is X GB.


### Step 7: Identify the files the extend beyond Portugal and discard them

TBC

### Step 8: Move the .tif files of 10m x 10m resolution, and their corresponding .json file to a new directory

Use the script 10m_portugal_data.py, which moves the files with 10 metre spatial resolution to /gws/nopw/j04/ai4er/users/map205/mres/10m_portugal_data

```
cd /gws/nopw/j04/ai4er/users/map205/mres/
mkdir 10m_portugal_data
```

The total number of files is X
The total volume is Y

### Step 9: Copy the .tif and .json files to 


------------------------------------------------------------------------------------------------------------------------

## MODELLING&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Random Forest_

## MODELLING&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_CNN_

## RESULTS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Data Availability_

## REFERENCES

[1] Create a JASMIN account | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/application/new/ (Accessed: March 16, 2023). 

[2] https://code.earthengine.google.com/register

[3] https://wandb.ai/site

[4] https://data.metabolismofcities.org/dashboards/lisbon/maps/35879/

[5] https://help.jasmin.ac.uk/article/176-storage

[6] https://help.jasmin.ac.uk/article/3810-data-transfer-tools-rsync-scp-sftp 

