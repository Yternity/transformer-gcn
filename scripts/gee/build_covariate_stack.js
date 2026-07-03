// Google Earth Engine covariate-stack extraction for the Transformer-GCN paper.
//
// This script builds the 26 base public-source predictors documented in
// data/README.md and exports five zonal summary statistics for each predictor.
// The resulting 26 x 5 = 130 covariate columns match the manuscript feature
// stack. Replace the placeholder assets in CONFIG with local vegetation
// reference units and prediction-grid cells before running.

var CONFIG = {
  year: 2024,
  scaleMeters: 1000,
  exportFolder: 'transformer_gcn_covariates',

  // Replace these placeholders with user-owned assets.
  // Required properties for trainingUnits include vegetation label fields and
  // coordinates or geometries. predictionGrid should contain the grid units to
  // classify with the trained model.
  trainingUnits: ee.FeatureCollection('users/your_username/training_units'),
  predictionGrid: ee.FeatureCollection('users/your_username/prediction_grid'),
  region: ee.Geometry.Rectangle([73.0, 18.0, 135.0, 54.0], null, false)
};

var start = ee.Date.fromYMD(CONFIG.year, 1, 1);
var end = start.advance(1, 'year');

function maskLandsatL2(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadow = qa.bitwiseAnd(1 << 4).eq(0);
  var clouds = qa.bitwiseAnd(1 << 3).eq(0);
  var snow = qa.bitwiseAnd(1 << 5).eq(0);
  return image.updateMask(cloudShadow.and(clouds).and(snow));
}

function prepLandsat(image) {
  var optical = image
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275)
    .add(-0.2)
    .rename(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']);

  var ndvi = optical.normalizedDifference(['B5', 'B4']).rename('NDVI');
  var evi = optical.expression(
    '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))',
    {
      nir: optical.select('B5'),
      red: optical.select('B4'),
      blue: optical.select('B2')
    }
  ).rename('EVI');
  var savi = optical.expression(
    '1.5 * ((nir - red) / (nir + red + 0.5))',
    {
      nir: optical.select('B5'),
      red: optical.select('B4')
    }
  ).rename('SAVI');

  return optical.addBands([ndvi, evi, savi]).copyProperties(image, image.propertyNames());
}

function annualLandsatComposite() {
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(start, end)
    .filterBounds(CONFIG.region)
    .map(maskLandsatL2)
    .map(prepLandsat);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(start, end)
    .filterBounds(CONFIG.region)
    .map(maskLandsatL2)
    .map(prepLandsat);

  return l8.merge(l9).median().select(
    ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'EVI', 'SAVI']
  );
}

function annualSentinel1Composite() {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate(start, end)
    .filterBounds(CONFIG.region)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .median();

  var vv = s1.select('VV').rename('S1_VV');
  var vh = s1.select('VH').rename('S1_VH');
  var diff = vv.subtract(vh).rename('S1_VV_minus_VH');
  return vv.addBands([vh, diff]);
}

function gediRh98Composite() {
  return ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
    .filterDate(start, end)
    .filterBounds(CONFIG.region)
    .select('rh98')
    .median()
    .rename('GEDI_RH98');
}

function terrainPredictors() {
  var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation');
  var terrain = ee.Terrain.products(elevation);
  var slope = terrain.select('slope').rename('slope');
  var aspect = terrain.select('aspect').rename('aspect');
  var radians = aspect.multiply(Math.PI / 180.0);
  var eastness = radians.sin().rename('eastness');
  var northness = radians.cos().rename('northness');
  var roughness = elevation
    .reduceNeighborhood({
      reducer: ee.Reducer.stdDev(),
      kernel: ee.Kernel.square({radius: 1, units: 'pixels'})
    })
    .rename('roughness');

  return elevation.addBands([slope, aspect, eastness, northness, roughness]);
}

function humanActivityPredictors() {
  var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterDate(start, end)
    .filterBounds(CONFIG.region)
    .select('avg_rad')
    .median()
    .rename('VIIRS_avg_rad');

  var population = ee.Image('JRC/GHSL/P2023A/GHS_POP/2020')
    .select('population_count')
    .rename('GHSL_population');

  var built = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
    .select('built_surface')
    .rename('GHSL_built_surface');

  return viirs.addBands([population, built]);
}

function soilGridMean(assetId, bands, outputName) {
  return ee.Image(assetId).select(bands).reduce(ee.Reducer.mean()).rename(outputName);
}

function soilPredictors() {
  var soc = soilGridMean(
    'projects/soilgrids-isric/soc_mean',
    ['soc_0-5cm_mean', 'soc_5-15cm_mean', 'soc_15-30cm_mean'],
    'soil_soc_0_30cm'
  );
  var ph = soilGridMean(
    'projects/soilgrids-isric/phh2o_mean',
    ['phh2o_0-5cm_mean', 'phh2o_5-15cm_mean', 'phh2o_15-30cm_mean'],
    'soil_phh2o_0_30cm'
  );
  var clay = soilGridMean(
    'projects/soilgrids-isric/clay_mean',
    ['clay_0-5cm_mean', 'clay_5-15cm_mean', 'clay_15-30cm_mean'],
    'soil_clay_0_30cm'
  );

  return soc.addBands([ph, clay]);
}

function buildPredictorImage() {
  return annualLandsatComposite()
    .addBands(annualSentinel1Composite())
    .addBands(gediRh98Composite())
    .addBands(terrainPredictors())
    .addBands(humanActivityPredictors())
    .addBands(soilPredictors())
    .clip(CONFIG.region);
}

function zonalReducer() {
  return ee.Reducer.min()
    .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});
}

function exportCovariates(featureCollection, description) {
  var predictors = buildPredictorImage();
  print('Base predictor bands', predictors.bandNames());
  print('Expected exported covariate count', predictors.bandNames().size().multiply(5));

  var table = predictors.reduceRegions({
    collection: featureCollection,
    reducer: zonalReducer(),
    scale: CONFIG.scaleMeters,
    tileScale: 8
  });

  Export.table.toDrive({
    collection: table,
    description: description,
    folder: CONFIG.exportFolder,
    fileNamePrefix: description,
    fileFormat: 'CSV'
  });
}

exportCovariates(CONFIG.trainingUnits, 'train_set_covariates_' + CONFIG.year);
exportCovariates(CONFIG.predictionGrid, 'prediction_grid_covariates_' + CONFIG.year);
