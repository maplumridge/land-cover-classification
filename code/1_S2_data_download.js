// AI4ER MRes
// Code to export L2A Sentinel-2 satellite data
// Steps:
// 1. Selects Sentinel-2 data from January, February and March 2018 (winter)
// 2. Filter for Coimbra, Portugal and images with less than 50% cloud cover
// 3. Applies a cloud mask to any individual satellite images that contain cloud cover
// 4. Reprojects the data to coordinate reference system EPSG:32629
// 5. Merges the data for individual dates together to create a seasonal average
// 6. Creates two export options:
//    a) Export satellite data with 4 bands (2, 3, 4 & 8) at 10 metre resolution
//    b) Export satellite data with 10 bands (2, 3, 4, 5, 6, 7, 8, 8a, 11 & 12) at 10 metre resoltion

// TO DO
// 1. Add data for other months/seasons - DONE
// 2. Increase CC % - DONE
// 3. Modify cloud mask scheme - DONE (to be reviewed)
// 4. Review export option 3 (or remove) - DONE (removed)
// 5. Also corrected data selection (S2_SR not S2) - DONE


// 1. Select Sentinel 2A data (not 1C)
// Specify bounding box, date range, and cloud cover %
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(ee.Geometry.Polygon([
    [-9.6863, 41.8802],
    [-9.6863, 36.8386],
    [-6.1897, 36.8386],
    [-6.1897, 41.8802]
  ]))
//  .filterDate("2018-01-01", "2018-03-31")
//  .filterDate("2018-04-01", "2018-06-30")
//  .filterDate("2018-07-01", "2018-09-30")
  .filterDate("2018-10-01", "2018-12-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5));
  
// 2. Filter for Coimbra, Portugal
// Load Portugal shapefile (which contains boundaries for each district in Portugal)
// Note: Upload shapefile to 'Assests' first
var portugal = ee.FeatureCollection("users/map205/PRT_regions");
// Filter on Portugal shapefile to extract only the Coimbra region
var coimbraRegion = portugal.filter(ee.Filter.eq("NAME_1", "Coimbra"));
var coimbraGeometry = coimbraRegion.geometry();
// Extract bounding box of Coimbra
var coimbraBounds = coimbraGeometry.bounds();
// Define & apply a rectangular buffer (metres)
var bufferDistance = 100;
var bufferedBounds = coimbraBounds.buffer(bufferDistance);
var bufferedGeometry = ee.Geometry(bufferedBounds);
// Filter the Sentinel-2 data and specific bands for Coimbra
var filteredCollection = s2
  .filterBounds(bufferedGeometry)
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'QA60', 'SCL']);
  
// 3. Apply a cloud mask for images with cloud-covered and cloud-shadowed pixels

//METHOD 1: Updated based on 
// - https://github.com/fitoprincipe/geetools-code-editor/wiki/Cloud-Masks
// - https://gis.stackexchange.com/questions/417799/removing-clouds-from-sentinel2-image-collection-in-google-earth-engine
// - https://gis.stackexchange.com/questions/423823/mask-sentinel-2-image-using-scl-product-in-google-earth-engine
// Import cloud_masks module
//##var cld = require('users/fitoprincipe/geetools:cloud_masks');
// Apply cloud and cloud shadow mask
//##var maskedCollection = filteredCollection.map(function(image) {
//##var masked = cld.sclMask(['cloud_low', 'cloud_medium', 'cloud_high', 'shadow'])(image);
//##return masked;
//##});

// METHOD 2
// Function to apply cloud and cloud shadow mask
var applyCloudMask = function(image) {
  var qa = image.select('QA60');
  var cloudMask = qa.bitwiseAnd(1 << 10).eq(0); // Cloud mask bit
  var cloudShadowMask = qa.bitwiseAnd(1 << 11).eq(0); // Cloud shadow mask bit
  return image.updateMask(cloudMask).updateMask(cloudShadowMask);
};
// Apply cloud and cloud shadow mask to the collection
var maskedCollection = filteredCollection.map(applyCloudMask);

// 4. Reproject S2 data to CRS EPSG:32629
var desiredCRS = 'EPSG:32629';

// 5 & 6: Merge and export data
// DOWNLOAD OPTION 1: Exporting bands 2, 3, 4, and 8 only
// Average Volume: 647.5 MB
var exportBands1 = ['B2', 'B3', 'B4', 'B8'];
// Crop the data with selected bands
var croppedCollection1 = maskedCollection.select(exportBands1).map(function(image) {
  return image.clip(bufferedGeometry);
});
// Export TIFF file to my Google drive
Export.image.toDrive({
  image: croppedCollection1.median(),
  scale: 10,
//  description: 'Winter_4bands',
//  description: 'Spring_4bands',
//  description: 'Summer_4bands',
  description: 'Autumn_4bands',
  region: bufferedGeometry,
  crs: desiredCRS,
  folder: 'GEE_Data',
  fileFormat: 'GeoTIFF'
});
// List S2 filenames for reference
var imageList = croppedCollection1.aggregate_array("system:index");
print("Sentinel-2 Image List:", imageList);

/////////////////////////////////////////////////////////////////////////////////

// DOWNLOAD OPTION 2: Exporting all bands excluding 1, 9 & 10 (atmopsheric bands)
// Average Volume: 1.2 GB
var exportBands2 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
// Crop the data with selected bands
var croppedCollection2 = maskedCollection.select(exportBands2).map(function(image) {
  return image.clip(bufferedGeometry);
});
// Export TIFF file to my Google drive
// Specify croppedCollection2 (not 1 from the previous export)
var imageCollection = croppedCollection2;

// Due to larger dataset size, export the data in batches and merge the data back together
// Batch export parameters
var exportConfig = {
  descriptionPrefix: 'Batch_Export_',
  scale: 10,
  crs: desiredCRS,
  region: bufferedGeometry,
  folder: 'GEE_Data',
  fileFormat: 'GeoTIFF'
};

// List image IDs
var imageList = croppedCollection2.aggregate_array('system:index');

// Convert imageList to JavaScript array
imageList.evaluate(function(ids) {
  // Create an empty image collection to store the merged images
  var mergedCollection = ee.ImageCollection([]);

  // Iterate & export
  for (var i = 0; i < ids.length; i++) {
    var imageId = ids[i];
    var exportName = exportConfig.descriptionPrefix + imageId;
    var image = ee.Image(croppedCollection2.filterMetadata('system:index', 'equals', imageId).first())
      .reproject({
        crs: exportConfig.crs,
        scale: exportConfig.scale
      });

    // Add each image to merged collection
    mergedCollection = mergedCollection.merge(image);

    // Export option for separate files
    Export.image.toDrive({
      image: image,
      description: exportName,
      scale: exportConfig.scale,
      crs: exportConfig.crs,
      region: exportConfig.region,
      folder: exportConfig.folder,
      fileFormat: exportConfig.fileFormat
    });
  }
  
  // Merge images/data to export as single file
  var mergedImage = mergedCollection.mosaic()
    .reproject({
      crs: exportConfig.crs,
      scale: exportConfig.scale
    });
  // Export option for single file
  Export.image.toDrive({
    image: mergedImage,
//    description: 'Winter_10bands',
//    description: 'Spring_10bands',
//    description: 'Summer_10bands',
    description: 'Autumn_10bands',
    scale: exportConfig.scale,
    crs: exportConfig.crs,
    region: exportConfig.region,
    folder: exportConfig.folder,
    fileFormat: exportConfig.fileFormat
  });
});

///////////////////////////////////////////////////////////////////////////

/// VISUALISE DATA ON THE GEE MAP ///

var visualizationParams = { bands: ['B4', 'B3', 'B2'], min: 0, max: 3000 };
Map.addLayer(croppedCollection1.median(), visualizationParams, "Sentinel-2");
Map.centerObject(bufferedGeometry, 11);


///////////////////////////////////////////////////////////////////////////

/// OPTIONAL INFORMATION 
// Check bands and data types at various processing steps

// Checking an image in the 'fitlered collection'
var filteredImage = filteredCollection.first();
// Get band names and data type(s)
var bandNames = filteredImage.bandNames();
var bandDataTypes = filteredImage.bandTypes();
print("Filtered Band Names:", bandNames);
print("Filtered Band Data Types:", bandDataTypes);

// Checking an image in the cloud masked collection
var cloudImage = maskedCollection.first();
// Get band names and data type(s)
var bandNames = cloudImage.bandNames();
var bandDataTypes = cloudImage.bandTypes();
print("Masked Band Names:", bandNames);
print("Masked Band Data Types:", bandDataTypes);
