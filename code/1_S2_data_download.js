// AI4ER MRes
// Code to export L2A Sentinel-2 satellite data
// Steps:
// 1. Selects Sentinel-2 data for each season
// 2. Filter for Coimbra, Portugal
// 3. Applies a cloud mask to any individual satellite images that contain cloud cover
// 4. Reprojects the data to coordinate reference system EPSG:32629
// 5. Merges the data for individual dates together to create a seasonal average
// 6. Creates two export options:
//    a) Export satellite data with 4 bands (2, 3, 4 & 8) at 10 metre resolution
//    b) Export satellite data with 10 bands (2, 3, 4, 5, 6, 7, 8, 8a, 11 & 12) at 10 metre resoltion

// DATA SELECITON //
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
  

// CLOUD MASK //
// 3. Apply a cloud mask for images with cloud-covered and cloud-shadowed pixels
// Updated to use 'better' SCL method:
// - See https://www.mdpi.com/2072-4292/13/4/816
// - https://github.com/fitoprincipe/geetools-code-editor/wiki/Cloud-Masks
// - https://gis.stackexchange.com/questions/417799/removing-clouds-from-sentinel2-image-collection-in-google-earth-engine
// - https://gis.stackexchange.com/questions/423823/mask-sentinel-2-image-using-scl-product-in-google-earth-engine
// Extract band SCL
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // Values 3, 8, 9 and 10 represent cloud shadows and different types of clouds
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask);
}
// Apply mask to image collection
var s2CloudMasked = filteredCollection.map(maskS2clouds);
// Create a MEDIAN composite image
// Median is more robust to noise and outliers
// Changing this to .mean shows worse performance
var medianImage = s2CloudMasked.median().clip(coimbraGeometry);

// VISUALISE //
// Visualise median composited data
Map.centerObject(coimbraGeometry, 10);
Map.addLayer(medianImage, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Median RGB');


// EXPORT SEASONAL SATELLITE DATA //
// 4. Reproject S2 data to CRS EPSG:32629
var desiredCRS = 'EPSG:32629';
// 5 & 6: Merge and export data
// DOWNLOAD OPTION 1: Exporting 10m bands 2, 3, 4, and 8 only
// Average Volume: 647.5 MB
var exportBands1 = ['B2', 'B3', 'B4', 'B8'];
// Crop the data with selected bands
var croppedCollection1 = s2CloudMasked.select(exportBands1).map(function(image) {
  return image.clip(bufferedGeometry);
});
// Export TIFF file to my Google drive (20GB limit)
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

// DOWNLOAD OPTION 2: Exporting all 20m bands excluding 1, 9 & 10 (atmopsheric bands)
// Average Volume: 1.2 GB
var exportBands2 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
// Crop the data with selected bands
var croppedCollection2 = s2CloudMasked.select(exportBands2).map(function(image) {
  return image.clip(bufferedGeometry);
});
// Export TIFF file to my Google drive (20GB limit)
Export.image.toDrive({
  image: croppedCollection2.median(),
  scale: 10,
//  description: 'SCL_Winter_10bands',
//  description: 'SCL_Spring_10bands',
//  description: 'SCL_Summer_10bands',
  description: 'SCL_Autumn_10bands',
  region: bufferedGeometry,
  crs: desiredCRS,
  folder: 'GEE_Data',
  fileFormat: 'GeoTIFF'
});

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
var cloudImage = s2CloudMasked.first();
// Get band names and data type(s)
var bandNames = cloudImage.bandNames();
var bandDataTypes = cloudImage.bandTypes();
print("Masked Band Names:", bandNames);
print("Masked Band Data Types:", bandDataTypes);
