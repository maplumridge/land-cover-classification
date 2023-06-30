
<br></br>
<img align="left" src="https://www.cmes.info/img/logos/ai4er_logo_2048px.png" width="75" height="75">
<img align="left" src="https://www.cam.ac.uk/sites/www.cam.ac.uk/files/inner-images/logo.jpg" width="300" height="75">
<br><br>
<br><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# AI4ER MRes 2023: Land cover classification

## Project Description

This goal of this 12-week project was to classify land cover in wildland-urban interface (WUI) regions in Portugal, which has historically experienced extreme wildfires. Land cover information in WUI regions is critical for wildfire management. WUI is a human-nature interface, where urban settlements are in close proximity to flammable vegetation [1]. WUI land cover is highly changeable, with flammable vegetation juxtaposed against fire-resistant urban material. The connectivity of flammable materials is critical for fire propagation. This project seeks to classify land cover in WUI regions as a first step towards a broader research goal of high-resolution, automated mapping of fuel and flammability for use in WUI-specific fire propagation models such as FireSPIN [2]. 


## Data
This project uses two publicly available datasets:
- CORINE land cover data from the [Copernicus Land Monitoring Service](https://land.copernicus.eu/pan-european/corine-land-cover/clc2018?tab=download); free and open acess is granted by the [Copernicus data and information policy Regulation](https://land.copernicus.eu/faq/about-data-access)
- Sentinel-2 L2A satellite imagery, available on [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)). Data is available to use in accordance with the [Copernicus Sentinel Data Terms and Conditions](https://sentinels.copernicus.eu/web/sentinel/terms-conditions#:~:text=Users%20may%20not%20modify%2C%20publish,without%20obtaining%20prior%20written%20authorisation)

## Code Structure

```
|──────1_S2_data_download.js         <-- JavaScript Google Earth Engine task to select and preprocess satellite data for the region of interest
|──────2_merge.sh                    <-- Shell script to merge satellite data together with GDAL
|──────3_add_indices.py              <-- Python script to add spectral indices to satellite data
|──────4_reproject.sh                <-- Shell script to reproject and resample CORINE reference data
|──────5_reclassify_CORINE.py        <-- Python script to map land cover classes to new values
|──────6_L1_train.py                 <-- Python script to train random forest models to perfrom level 1 class classifcation
|──────7_L2L3_train.py               <-- Python script to train random forest models to perfrom level 2 and 3 class classifcation
|──────8_L1_test.py                  <-- Python script to test the level 1 random forest models
|──────9_L2L3_test.py                <-- Python script to test the level 2 and 3 random forest models
|──────10_H_train.py                 <-- Python script to train random forest models to perfrom hierarchical classification
|──────11_H_test.py                  <-- Python script to test hierarchical classification
|──────12_train_jobs.sh              <-- Shell script to submit model training runs for batch processing on JASMIN
|──────13_test_jobs.sh               <-- Shell script to submit model testing runs for batch processing on JASMIN

```

## Workflow
<p align="center">
    <img src="MRes_workflow.png" width="70%"\>
</p>

<p align="center">
    <img src="MRes_model_workflow.png" width="90%"\>
</p>

## Usage
To test the code and recreate the results of this project, follow the steps below: 
1. Clone this repository (for the latest version) or access the release for June 30th (MRes submission date)
2. Follow the steps in workflow.md to download and preprocess the datasets and subsequently train and test the random forest models. 

## Contributors
The contents of this repository were created by [Meghan Plumridge](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student, University of Cambridge, for a 12-week MRes project. [AI4ER](https://ai4er-cdt.esc.cam.ac.uk) is the the UKRI Centre for Doctoral Training (CDT) in the Application of Artificial Intelligence to the study of Environmental Risks at the [University of Cambridge](https://www.cam.ac.uk).

## References
[1] A. Bar-Massada, F. Alcasena, F. Schug and V. C. Radeloff, “The wildland – urban interface in Europe: Spatial patterns and associations with socioeconomic and demographic variables,” Landscape and Urban Planning, 2023. 

[2] E. Mastorakos, S. Gkantonas, G. Efstathiou and A. Giusti, “A hybrid stochastic Lagrangian – cellular automata framework for modelling fire propagation in inhomogeneous terrains,” Proceedings of the Combustion Institute, vol. 39, no. 3, pp. 3853-3862, 2023.





