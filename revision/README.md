# Revision tables (GLOPLACHA-D-26-00945)

Tables added during the major revision of the manuscript
*Modern vegetation-climate associations across China's three topographic steps: Implications for global-change ecology and paleoclimate reconstruction*
(Global and Planetary Change). The complete data of the study are released with the article as Supplementary
Data S1–S10.

## `reference_samples_corrected.csv`

One row per reference polygon of the 1:1,000,000 Vegetation Map of China that belongs to the 22 modelled
classes and has complete covariates (53,612 rows). The sample point is the representative (interior) point
of the polygon.

| Column | Meaning |
|---|---|
| `zone_id` | Polygon identifier of the digital vegetation map |
| `label_submitted` | Subclass number used as the label in the submitted manuscript |
| `lon`, `lat` | Representative point (WGS84) |
| `elev_mean` | Mean elevation of the 1-km modelling unit (m) |
| `植被名称` | Formation name in the map attribute table |
| `植被大类` | Vegetation type group |
| `植被亚类` | Vegetation subclass (the label source) |
| `sub_code` | Subclass code as written in the attribute table (`l9` = subclass 19) |
| `code` | Community code V01–V22 after correction |
| `managed` | `True` for the managed communities V20–V22 |
| `terrace` | Topographic step at `elev_mean`: I ≥ 4000 m, II 1000–4000 m, III < 1000 m |

Correction made during the revision: subclass 19 is written `(l9)` in the attribute table and had been merged
with subclass 9 under V05. V05 now denotes subclass 19 only; the 207 subclass-9 samples are excluded because
they fall below the 1,000-sample threshold applied to all classes.

## `class_table_corrected.csv`

The 22 classes with their source map subclass, number of samples per topographic step, and Chinese and
English community names.

Source of both tables: Editorial Committee of Vegetation Map of China, Chinese Academy of Sciences, 2001.
Vegetation Atlas of China (1:1,000,000). Science Press, Beijing. Digital version distributed by the Resource
and Environment Science and Data Center, Chinese Academy of Sciences (https://www.resdc.cn). Users should cite
the original map and observe the data center's terms of use.

## `station_climate_1991_2020.csv`

The 348 NOAA GSOD stations used in the manuscript, with mean annual temperature (`mat_c`, °C) and mean annual
precipitation (`map_mm`, mm) over 1991–2020, the number of valid years (`valid_years`), station elevation
(`elevation_m`) and the Köppen-Geiger class at the station from the 0.1° map of Peel et al. (2007) (`koppen_code`,
`koppen_group`). A station-year is valid when daily temperature and precipitation are both available on at
least 90% of days; stations with at least 24 valid years are included. GSOD precipitation for Chinese stations
is largely missing in 1999.

Sources: NOAA National Centers for Environmental Information, Global Surface Summary of the Day (GSOD);
Peel, M.C., Finlayson, B.L., McMahon, T.A., 2007. Updated world map of the Köppen-Geiger climate
classification. Hydrol. Earth Syst. Sci. 11, 1633–1644. https://doi.org/10.5194/hess-11-1633-2007
