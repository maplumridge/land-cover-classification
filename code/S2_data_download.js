// AI4ER MRes
// Code to export L2A Sentinel-2 satellite data
// Steps:
// 1. Selects Sentinel-2 data from January 2018
// 2. Filter for Coimbra, Portugal
// 3. Applies a cloud mask to any individual satellite images that contain cloud cover
// 4. Reprojects the data to coordinate reference system EPSG:32629
// 5. Merges the data for individual dates together to create a one-month average
// 6. Creates two export options:
//    a) Export satellite data with 4 bands (2, 3, 4 & 8) at 10 metre resolution
//    b) Export satellite data with 10 bands (2, 3, 4, 5, 6, 7, 8, 8a, 11 & 12) at 10 metre resoltion

// TO DO
// 1. Add data for other months/seasons
// 2. Increase CC %
// 3. Modify cloud mask scheme
// 4. Review export option 3 (or remove)


// 1. Select Sentinel 2A data (not 1C)
// Specify bounding box, date range, and cloud cover %
var s2 = ee.ImageCollection("COPERNICUS/S2")
  .filterBounds(ee.Geometry.Polygon([
    [-9.6863, 41.8802],
    [-9.6863, 36.8386],
    [-6.1897, 36.8386],
    [-6.1897, 41.8802]
  ]))
  .filterDate("2018-01-29", "2018-01-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 1));
  
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
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'QA60']);
  
// 3. Apply a cloud mask for images with cloud-covered and cloud-shadowed pixels
var applyCloudMask = function(image) {
  var qa = image.select('QA60');
  var cloudMask = qa.bitwiseAnd(1 << 10).eq(0); // Cloud mask bit
  var cloudShadowMask = qa.bitwiseAnd(1 << 11).eq(0); // Cloud shadow mask bit
  return image.updateMask(cloudMask).updateMask(cloudShadowMask);
};
// Apply masks
var maskedCollection = filteredCollection.map(applyCloudMask);

// 4. Reproject S2 data to CRS EPSG:32629
var desiredCRS = 'EPSG:32629';

// 5 & 6: Merge and export data
// DOWNLOAD OPTION 1: Exporting bands 2, 3, 4, and 8 only
// 644 MB
var exportBands1 = ['B2', 'B3', 'B4', 'B8'];
// Crop the data with selected bands
var croppedCollection1 = maskedCollection.select(exportBands1).map(function(image) {
  return image.clip(bufferedGeometry);
});
// Export TIFF file to my Google drive
Export.image.toDrive({
  image: croppedCollection1.median(),
  scale: 10,
  description: 'January_4bands',
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
// 1.14 GB
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
    description: 'January_10bands',
    scale: exportConfig.scale,
    crs: exportConfig.crs,
    region: exportConfig.region,
    folder: exportConfig.folder,
    fileFormat: exportConfig.fileFormat
  });
});


///////////////////////////////////////////////////////////////////////////
// REVIEW LATER - COULD NOT GET WORKING //

// DOWNLOAD OPTION 3: Exporting bands and indices
// XX GB
var exportBands3 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
// Calculate NDVI and NDBI
// See https://developers.google.com/earth-engine/tutorials/tutorial_api_06 (bands 8 and 4)
// See https://www.lifeingis.com/computing-ndbi-in-google-earth-engine/ (bands 11 and 8)
// Note: .toFloat() ensures that these indices have data type Float32 (same as S2 bands)

// DOWNLOAD OPTION 3: Exporting bands AND indices
var exportBands3 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

// Convert bands to float (so that S2 bands and indices are in the same format)
var convertToFloat = function(image) {
  var floatBands = image.select(exportBands3).toFloat();
  return floatBands.copyProperties(image, image.propertyNames());
};

// Calculate NDVI and NDBI indices
var addIndices = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI');
  return image.addBands([ndvi, ndbi]);
};

// Crop, convert to float, calculate indices
var croppedCollection3 = maskedCollection
  .select(exportBands3)
  .map(convertToFloat)
  .map(addIndices)
  .map(function(image) {
    return image.clip(bufferedGeometry);
  });

// Merge images
var mergedImage3 = croppedCollection3.mean().toFloat()
  .reproject({
    crs: desiredCRS,
    scale: 10
  });

// Export option for single merged file
Export.image.toDrive({
  image: mergedImage3,
  description: 'Last_Try',
  scale: 10,
  crs: desiredCRS,
  region: bufferedGeometry,
  folder: 'GEE_Data',
  fileFormat: 'GeoTIFF'
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

// Checking computation of indices (export option 3)
var indicesImage = croppedCollection3.median();
// Get band names and data type(s)
var indicesBandNames = indicesImage.bandNames();
var indicesBandDataTypes = indicesImage.bandTypes();
print("Indices Band Names:", indicesBandNames);
print("Indices Band Data Types:", indicesBandDataTypes);

