# Data Documentation

This repository does not redistribute the manuscript data files. The released
code is intended to be run after the user places locally prepared data under
`data/` and trained model artifacts under `outputs/`.

The data are omitted for three reasons:

- the derived national prediction tables and model artifacts are large;
- several source layers are third-party geospatial products with their own
  access and citation requirements;
- some intermediate files contain project-specific administrative boundary,
  sampling, or label-harmonization material that should be obtained from the
  original sources rather than copied through this code repository.

## Required Runtime Files

Place the following files under the repository root when running the full
training and prediction workflow:

```text
data/
  ours_data.pt
  prediction_set.csv
outputs/
  results/
    best_model_seed_42.pt
```

`data/ours_data.pt` is the training tensor file loaded by `core/train.py`.
It is a PyTorch Geometric `Data` object with:

- `x`: node features. In the local manuscript run, this had shape
  `[64102, 132]`.
- `y`: vegetation-community labels. In the local manuscript run, this had
  64,102 samples and 54 raw label IDs before filtering.
- the final two columns of `x` are longitude and latitude. The graph builder
  moves these columns into `data.pe` and removes them from the model feature
  matrix before training.

The default training workflow applies `filter_small_classes(min_samples=1000)`.
In the local manuscript run, this retained 53,819 samples and 22 vegetation
classes.

`data/prediction_set.csv` is the national prediction table used by
`core/predict.py`. It should contain:

- longitude and latitude columns named either `lon`/`lat`,
  `longitude`/`latitude`, or Chinese equivalents `经度`/`纬度`;
- the same numeric covariate order used during training;
- optional metadata columns such as `zone_id`, `province`, and
  `province_code`.

In the local manuscript run, `prediction_set.csv` had 132 columns: longitude,
latitude, and 130 environmental covariate summaries.

`outputs/results/best_model_seed_42.pt` is produced by `python train.py` and
is required for `python predict.py`.

The public Earth Engine preprocessing script for constructing the 130
covariate summaries is provided in `scripts/gee/build_covariate_stack.js`.
It uses placeholder user assets for the vegetation reference units and
prediction grid, because these spatial units are manuscript-specific and are
not redistributed in this repository.

## Feature Groups

The model uses 130 environmental covariate summaries plus longitude and
latitude for graph construction. The 130 covariates are grouped as follows:

| Group | Number of features | Main variables |
|---|---:|---|
| Optical bands | 35 | Landsat 8 B1-B7 summaries |
| Vegetation indices | 15 | NDVI, EVI, SAVI summaries |
| Canopy structure | 5 | GEDI RH98 summaries |
| Radar backscatter | 15 | Sentinel-1 VV, VH, and VV-VH summaries |
| Topography | 30 | elevation, slope, aspect, eastness, northness, roughness |
| Human activity | 15 | VIIRS night lights, GHS population, GHS built-up |
| Soil properties | 15 | soil organic carbon, pH, clay content at 0-30 cm |

For most raster-derived variables, the per-unit statistics are maximum, mean,
median, minimum, and standard deviation.

## External Online Data Sources

The following externally sourced datasets were used to build the local training
and prediction tables or to support post hoc climate interpretation. They are
not copied into this repository.

| Source group | Source / Earth Engine ID | Time period used | Role in this project |
|---|---|---:|---|
| Landsat optical reflectance | `LANDSAT/LC08/C02/T1_L2` | 2024-01-01 to 2024-12-31 | B1-B7 annual summaries; NDVI, EVI, and SAVI derived by the authors |
| Sentinel-1 SAR | `COPERNICUS/S1_GRD` | 2024-01-01 to 2024-12-31 | VV, VH, and VV-VH dB-difference summaries |
| GEDI canopy structure | `LARSE/GEDI/GEDI02_A_002_MONTHLY` | 2024-01-01 to 2024-12-31 | RH98 canopy-height summaries |
| VIIRS night-time lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | 2024-01-01 to 2024-12-31 | human-activity covariates |
| GHSL population | `JRC/GHSL/P2023A/GHS_POP/2020` | 2020 | population covariates |
| GHSL built-up surface | `JRC/GHSL/P2023A/GHS_BUILT_S/2020` | 2020 | built-up covariates |
| SRTM elevation | `USGS/SRTMGL1_003` | static product | elevation, slope, aspect, roughness |
| SoilGrids | ISRIC SoilGrids layers, including SOC, pH, and clay | long-term/static soil product | 0-30 cm soil covariates |
| NOAA GSOD | NOAA/NCEI Global Summary of the Day | 2020-2024 | station-based climate analysis and rule-based bioclimatic zones; not required for model training |
| ETOPO relief model | NOAA ETOPO / ETOPO1 local DEM asset | static relief product | background relief and elevation support for climate-zone figures; not required for model training |
| Koppen climate zones | external Koppen-Geiger climate raster used locally | 1901-2000 in the local analysis | post hoc climate correspondence; not required for model training |

Official source pages for the main public products are listed in
`docs/reproducibility.md`.

## Author-Generated Data Products

The following files are derived by the authors from the external products,
reference vegetation labels, and project-specific spatial units:

| Local file | Generated by | Description |
|---|---|---|
| `train_set.xlsx` | local sampling and covariate extraction workflow | tabular training samples with vegetation labels, coordinates, and environmental summaries |
| `ours_data.pt` | local tensor conversion workflow | PyTorch Geometric training object consumed by `core/train.py` |
| `prediction_set.csv` | local national gridding/covariate extraction workflow | national prediction input table consumed by `core/predict.py` |
| `outputs/results/*.pt` | `python train.py` | trained model weights and training diagnostics |
| `prediction_complete.csv` | `python predict.py` | prediction output with class IDs, confidence, and rejection flags |
| climate-zone cross tables and maps | local climate-analysis scripts | post hoc ecological/climatic interpretation products |

These author-generated products are not committed because they are either
large, derived from third-party geospatial assets, or specific to the
manuscript's local data build.

## Reference Vegetation Labels

The supervised labels in `ours_data.pt` come from an author-harmonized
vegetation-community reference layer. The local workflow converts the source
vegetation classes into integer class IDs, filters classes with fewer than
1,000 samples by default, and remaps retained classes to contiguous IDs for
training.

The reference label layer itself is not redistributed here. Users who wish to
reproduce the full manuscript workflow should obtain the vegetation reference
data from the original map/atlas or institutional source cited in the
manuscript, then reproduce the same class harmonization and sampling steps.

## Recreating the Expected Runtime Files

At a high level, the local data build followed these steps:

1. Obtain or prepare vegetation-community reference units and their class IDs.
2. Use `scripts/gee/build_covariate_stack.js` and the public Earth Engine
   products listed above to extract annual or static covariate summaries for
   each training unit.
3. Build `train_set.xlsx` with labels, coordinates, and covariates.
4. Convert the table to a PyTorch Geometric `Data` object and save it as
   `data/ours_data.pt`.
5. Build `prediction_set.csv` on the national prediction grid using the same
   GEE script, covariate definitions, and column order.
6. Run the released training and prediction code.

The released `quick_test.py` does not require any of these files; it uses a
small synthetic graph only to verify that the model code imports and executes.
