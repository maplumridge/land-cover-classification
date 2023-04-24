# Project Workflow
### AI4ER MRes 2023: Fuel Classification

## About
XXX

## Prerequisites
1. Create a JASMIN account. See [1].

## Steps
### Step 1: Download the data from BigEarthNet
Download the file "BigEarthNet-S2" from the following URL https://bigearth.net/ 
The dataset is 66GB. 

### Step 2: Copy the tar file to JASMIN
Note: The files are transferred to my home directory on the login node, but this is synchronised to the compute nodes. However, I will NOT unzip the file in my home directory because I only have a 100GB allowance.
```
scp /Users/meghanplumridge/Downloads/BigEarthNet-S2-v1.0.tar.gz map205@login1.jasmin.ac.uk://home/users/map205/
```

### Step 3: Login to JASMIN and unzip the file
Note: I will unzip the data in the AI4ER group workspace on JASMIN because my personal workspace has a limit of 100 GB. See [2]. This takes > 24 hours to run.

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

I decied to run the merge_files.py script in my home directory (/home/users/map205) since I can process more files at once here.

```
conda activate sea-ice-classification
nohup python merge_files.py &
```

### Step 5: Copy the files to my computer to visualise in QGIS
See instructions at [3]

```
scp -r "map205@login1.jasmin.ac.uk:/home/users/map205/mres/*" /Users/meghanplumridge/Desktop/merged_files/
```
Note: I needed to add parantheses around the jasmin login path otherwise it failed with error zsh: no matches found: map205@login1.jasmin.ac.uk:/home/users/map205/mres/*


## References

[1] Create a JASMIN account | JASMIN Accounts Portal. Available at: https://accounts.jasmin.ac.uk/application/new/ (Accessed: March 16, 2023). 

[2] https://help.jasmin.ac.uk/article/176-storage

[3] https://help.jasmin.ac.uk/article/3810-data-transfer-tools-rsync-scp-sftp 

