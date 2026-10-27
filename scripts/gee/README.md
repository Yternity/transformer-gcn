# Google Earth Engine covariate extraction

This folder contains the public Google Earth Engine (GEE) preprocessing script
used to document and reproduce the environmental covariate stack described in
the manuscript.

## Script

- `build_covariate_stack.js`: builds a 26-band public-source predictor image
  and exports zonal summaries for user-supplied vegetation reference units or
  prediction grid cells.

The manuscript model uses 130 environmental covariates. These are generated as
five summary statistics (`min`, `max`, `mean`, `median`, and `stdDev`) for each
of 26 base predictors:

- Landsat 8/9 reflectance bands B1-B7;
- NDVI, EVI, and SAVI;
- GEDI RH98 canopy height;
- Sentinel-1 VV, VH, and VV_minus_VH;
- elevation, slope, aspect, eastness, northness, and roughness;
- VIIRS nighttime lights, GHSL population, and GHSL built-up surface;
- SoilGrids soil organic carbon, pH, and clay content.

## How to run

1. Open `build_covariate_stack.js` in the Google Earth Engine Code Editor.
2. Replace the placeholder assets in the `CONFIG` block:
   - `trainingUnits`: vegetation-community reference polygons or points;
   - `predictionGrid`: national prediction grid cells;
   - `region`: study-area geometry.
3. Run the script and start the Drive export tasks.
4. Save the exported CSV files under the local `data/` directory using the
   layout documented in `data/README.md`.

The script uses only public Earth Engine datasets plus user-supplied reference
units or grid cells. It does not require manuscript data for import, but full
numerical reproduction requires the same vegetation-label harmonization and
spatial sampling units used in the manuscript.
