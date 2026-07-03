# Reproducibility Notes

This document records the computational environment, expected local data
layout, and command sequence used by the released Transformer-GCN code package.

## Repository Scope

This public repository contains:

- model definitions;
- training and prediction entry points;
- the Google Earth Engine covariate extraction script;
- dependency files;
- a synthetic quick test;
- documentation for data sources and local data layout.

It intentionally excludes:

- large tabular data files;
- raw and processed raster/vector geospatial data;
- trained weights and generated predictions;
- figures, manuscripts, Office/PDF files, and archives.

## Tested Environment

The local smoke test was verified with:

```text
Python 3.11.15
PyTorch 2.6.0+cu126
PyTorch Geometric 2.7.0
```

CUDA support is optional for the quick test, but recommended for full training.
Install PyTorch and PyTorch Geometric using the official instructions that
match your Python, operating system, and CUDA version, then install the
remaining dependencies from `requirements.txt`.

## Installation

Using pip:

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows PowerShell users may use .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Using conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate transformer-gcn
```

If `torch` or `torch-geometric` cannot be resolved automatically, install them
from the official PyTorch and PyTorch Geometric channels first, then install
the remaining packages.

## Quick Test

Run the synthetic-data smoke test:

```bash
python quick_test.py
```

Expected output:

```text
quick_test passed: Transformer-GCN forward pass output shape = (6, 4)
```

The quick test does not require manuscript data.

## Full Training Workflow

Prepare the following local file:

```text
data/ours_data.pt
```

Then run:

```bash
python train.py --model-name ours --seed 42 --epochs 1000 --patience 50
```

The default workflow:

1. loads `data/ours_data.pt`;
2. filters vegetation classes with fewer than 1,000 samples;
3. constructs a spatial k-nearest-neighbor graph from longitude and latitude;
4. trains the Transformer-GCN model;
5. saves training diagnostics and best model weights to `outputs/results/`.

The main output needed for prediction is:

```text
outputs/results/best_model_seed_42.pt
```

Baseline models can be trained with:

```bash
python train.py --model-name mlp
python train.py --model-name gcn
python train.py --model-name sage
python train_rf.py
```

## Full Prediction Workflow

Prepare the following local files:

```text
data/prediction_set.csv
outputs/results/best_model_seed_42.pt
```

Then run:

```bash
python predict.py --chunk-size 50000 --grid-size 0.4 --buffer-cells 1
```

The prediction workflow:

1. infers the model input dimension from the saved state dictionary;
2. detects longitude and latitude columns;
3. selects numeric covariate columns in the same order used for training;
4. splits the national prediction table into spatial blocks;
5. builds local buffered graphs for each block;
6. writes predictions, confidence scores, and rejection flags to the output CSV.

The default output is:

```text
data/prediction_complete.csv
```

## Main Public Data Sources

The local data build used public products accessed primarily through Google
Earth Engine and NOAA/ISRIC data services:

- USGS Landsat 8 Level 2 Collection 2 Tier 1:
  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
- Sentinel-1 SAR GRD:
  https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- GEDI L2A monthly raster canopy top height:
  https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY
- VIIRS monthly nighttime lights:
  https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG
- GHSL population:
  https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP
- GHSL built-up surface:
  https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_S
- SRTM elevation:
  https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- SoilGrids:
  https://isric.org/explore/soilgrids
- NOAA/NCEI Global Summary of the Day:
  https://www.ncei.noaa.gov/data/global-summary-of-the-day/
- NOAA ETOPO Global Relief Model:
  https://www.ncei.noaa.gov/products/etopo-global-relief-model

See `data/README.md` for the distinction between online source products and
author-generated local files.

## Earth Engine Covariate Build

The public GEE preprocessing script is:

```text
scripts/gee/build_covariate_stack.js
```

It builds the 26 base predictors from public source products and exports five
zonal statistics for each predictor (`min`, `max`, `mean`, `median`, and
`stdDev`). This produces the 130 environmental covariates used by the model.

Before running, replace the placeholder assets in the script `CONFIG` block
with user-owned vegetation reference units and prediction-grid features. The
script does not redistribute manuscript-specific spatial units or derived
tables.

## Randomness and Splits

The default configuration uses:

```text
DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 11111
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.1
```

The remaining samples are used for testing after stratified per-class splitting.
PyTorch, NumPy, and Python random seeds are set in `core/train.py`.

## Known Limits

The released quick test verifies model import and a forward pass. It does not
verify full scientific reproducibility because the manuscript data are not
redistributed in this repository.

Full numerical reproduction requires rebuilding the local derived files with
the same vegetation-label harmonization, Earth Engine source products, spatial
sampling units, covariate statistics, and class-filtering threshold documented
in `data/README.md`.
