# AZ-Hydro

Maintainers: [Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu], [Ryan G. Smith](https://www.engr.colostate.edu/ce/ryan-g-smith/) [ryan.g.smith@colostate.edu]

<img src="../Readme_Figures/DRITaglineLogoTransparentBackground.png" height="45"/> &nbsp; <img src="../Readme_Figures/CSU-Signature-C-357.png" height="55"/> 

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). The Arizona Water Use Dataset (1896–2099): Withdrawals, consumptive use, and irrigation efficiency partitioned by source. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Two Centuries of Groundwater and Surface Water Withdrawals, Consumptive Use, and Irrigation Efficiency (1896–2099). _In prep. for AGU Earth's Future_.

---

## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If Anaconda or miniconda is already installed, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. It may be necessary to restart the current shell session to add conda to path.

The conda package manager can be updated by running the following command: ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help in installing the correct packages and Python version to run the codes.

### 2. Clone or download the repository

Download the repository from the compressed file link at the top right of the repository webpage, or clone the repository using Git.
Unzip all zipped files.  Several of the input datasets in this repository are zipped for efficient storage and must be unzipped before they can be used to run this project.

### 3. Creating the conda environment and installing packages
Open Linux/Mac terminal or Windows PowerShell and run the following:
```
conda create -y -n azhydro python=3.12
conda activate azhydro
conda install -y -c conda-forge gdal rioxarray geopandas lightgbm py-xgboost earthengine-api rasterstats seaborn openpyxl optuna optuna-dashboard scikit-explain catboost dask-ml dask-jobqueue swifter pyarrow
pip install openet-refet-gee
```

### 4. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) some of the predictor datasets from the GEE
data repository. After completing step 3, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). The Google Cloud CLI tools
may be required for this GEE authentication step. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). You also have to create a gcloud project to use the GEE API. 

### 5. Running AZHydro

From the `azhydro/` directory, run the pipeline with:

```bash
python pipeline.py
```

The `main()` function accepts three keyword arguments (all default to `True`):

| Argument | Default | Description |
|---|---|---|
| `skip_download` | `True` | Skip GEE tile download; use existing tiles on disk. |
| `load_files` | `True` | Skip recreating intermediate rasters/vectors that already exist. |
| `run_data_prep` | `True` | Execute Step 0 (data preparation). Set `False` if all rasters/vectors are already prepared. |

The pipeline executes Steps 0–4 in sequence (details below).

---

## Data sources

The project builds a spatially explicit, multi-decadal (1896–2099) dataset for Arizona by combining satellite-derived products, climate model projections, soil properties, streamflow observations, and USBR modeled streamflow.

### Google Earth Engine (GEE) predictor bands

The [`download_gee_data()`](hydrolibs/dataops.py) function downloads 14 bands of geospatial data from GEE as tiled GeoTIFFs at 2 km resolution over Arizona. Data are harmonized across three temporal eras using overlap-period bias-correction ratios to ensure continuity.

| Band | Description | Units | Source |
|------|-------------|-------|--------|
| `annual_et_ensemble_mm` | Actual evapotranspiration | mm/yr | Reitz (1896–1999), OpenET (2000–2025), MACA ensemble (2026–2099) |
| `annual_eto_mm` | Reference evapotranspiration (Penman-Monteith) | mm/yr | PRISM Hargreaves (1896–1978), gridMET (1979–2025), MACA ensemble (2026–2099) |
| `annual_precip_mm` | Precipitation | mm/yr | PRISM (1896–2025), MACA ensemble (2026–2099) |
| `annual_peff_mm` | Effective precipitation (USDA SCS method) | mm/yr | Computed from harmonized ETo, precipitation, and soil AWC |
| `annual_peff_pcml_mm` | Effective precipitation (PCML obs-based, 2000–2024) | mm/yr | PCML model, climatological mean outside 2000–2024 |
| `annual_tmmx_K` | Annual mean daily max temperature | K | PRISM (1896–2025), MACA (2026–2099) |
| `annual_tmmn_K` | Annual mean daily min temperature | K | PRISM (1896–2025), MACA (2026–2099) |
| `lulc` | Land use/land cover (1=Agriculture, 2=Urban, 3=Surface Water) | categorical | USGS historical (≤1984), NLCD (1985–2025), USGS projections (2026–2099) |
| `annual_crop_fraction` | Cropland fraction | fraction | Derived from LULC |
| `annual_irr_fraction` | Irrigated area fraction | binary | IrrMapper RF v1.2 (1985–2025), LULC-derived outside |
| `annual_gw_fraction` | Groundwater irrigation fraction | fraction | USGS snapshots (2000, 2005, 2010, 2015) |
| `soil_depth_cm` | Soil depth | cm | CSRL (static) |
| `awc_in` | Available water capacity (0–152 cm) | inches | SSURGO (static) |
| `ksat_mean_micromps` | Saturated hydraulic conductivity | μm/s | CSRL (static) |

### Data harmonization

The pipeline stitches disparate sources into a consistent 1896–2099 time series:

- **ET**: Reitz ensemble (1896–1999) → OpenET v2.0/v2.1 (2000–2025) → MACA × EToF crop coefficients (2026–2099)
- **ETo**: PRISM Hargreaves (1896–1978) → gridMET (1979–2025) → MACA 20-model ensemble (2026–2099)
- **LULC**: USGS historical scenario (≤1984) → NLCD (1985–2025) → USGS 4-scenario mode ensemble (2026–2099)
- **Climate projections**: MACA v2 daily data across 20 GCMs × 2 RCPs (RCP 4.5, RCP 8.5) = 40-member ensemble. All MACA queries use a flat-pipeline approach (single filter + reduce) to keep GEE computation graphs small: ETo uses `.sum().divide(40)` per month (computed per-image to preserve nonlinearity), precip uses `.sum().divide(40)`, and temperature uses `.mean()`.

Per-pixel, per-month bias-correction ratios are computed from overlapping observation periods and applied to extend each variable seamlessly. See [`gee/README.md`](../gee/README.md) for asset export details and equations.

### GEE pre-exported assets

Nine custom ImageCollections are pre-computed via scripts in [`gee/`](../gee/) and stored in GEE under `projects/azhydro/assets/`:

| Asset | Description | Years |
|-------|-------------|-------|
| `gridmet_hargreaves_eto_ratio` | gridMET / PRISM Hargreaves monthly ratio (12 images) | Climatology |
| `openet_reitz_et_ratio` | OpenET / Reitz ensemble monthly ratio (12 images) | Climatology |
| `monthly_etof` | Crop coefficient (OpenET / gridMET ETo) | Climatology |
| `prism_hargreaves_eto` | PRISM-based Hargreaves ETo | 1896–1978 |
| `usgs_adjusted_et` | Bias-adjusted Reitz actual ET | 1896–1999 |
| `maca_monthly_eto_v2` | MACA per-model/scenario projected ETo | 2026–2099 |
| `maca_monthly_et_v2` | MACA ensemble projected actual ET | 2026–2099 |
| `lulc_projection_ensemble` | USGS 4-scenario LULC mode | 2026–2099 |
| `monthly_peff_v2` | USDA SCS effective precipitation | 1896–2099 |

### Download architecture

Data are downloaded as tiles using a Dask-parallelized worker pool (40 workers, 1 GB each). Each tile covers an 80 km × 80 km region at 2 km resolution. Tiles are later mosaicked and reprojected for the ML pipeline.

### Streamflow analysis

The [`streamflowops`](hydrolibs/streamflowops.py) module handles streamflow data acquisition and rasterization. It covers all 16 Arizona surface watersheds from 1896 to 2099.

#### Data sources

- **USGS NWIS**: Daily mean discharge (parameter 00060) via the `dataretrieval` Python API, resampled to monthly means
- **USBR CMIP Ensemble**: Monthly modeled streamflow averaged across ~112 climate model runs (scenarios a1b, a2, b1), spanning 1950–2099
- **Historical Ratio Method**: For sites without USBR projections, per-calendar-month scaling ratios are computed against the nearest USBR-gauged reference site and applied to generate synthetic 1950–2099 projections

#### Gauge network (20 sites)

| USGS ID | USBR ID | Site Name | Watershed |
|---------|---------|-----------|----------|
| 09380000 | 00013 | Colorado River at Lees Ferry | Colorado River |
| 09429490 | 00014 | Colorado River above Imperial Dam | Colorado River |
| 09444500 | 00058 | San Francisco River at Clifton | Upper Gila River |
| 09448500 | 00059 | Gila River at Head of Safford Valley nr Solomon | Upper Gila River |
| 09497500 | 00061 | Salt River near Chrysotile | Salt River |
| 09498500 | 00062 | Salt River near Roosevelt | Salt River |
| 09499000 | 00063 | Tonto Creek Abv Gun Creek nr Roosevelt | Salt River |
| 09510000 | 00064 | Verde River below Bartlett Dam | Verde River |
| 09508500 | 00064 | Verde R blw Tangle Creek Abv Horseshoe Dam | Verde River |
| 09402300 | — | Little Colorado River Abv Mouth nr Desert View | Little Colorado River |
| 09426620 | — | Bill Williams River near Parker | Bill Williams River |
| 09512500 | — | Agua Fria River near Mayer | Agua Fria River |
| 09415000 | — | Virgin River at Littlefield | Virgin River |
| 09489000 | — | Santa Cruz River near Laveen | Santa Cruz River |
| 09471000 | — | San Pedro River at Charleston | San Pedro River |
| 09520500 | — | Lower Gila River near Dome | Lower Gila River |
| 09535300 | — | Vamori Wash at Kom Vo | San Simon River |
| 09537500 | — | Whitewater Draw near Douglas | White Water Draw |
| 09537200 | — | Leslie Creek near McNeal | White Water Draw / Rio Yaqui |
| 09426650 | — | CAP Canal at Havasu Pumping Plant | CAP Diversion |

Sites with USBR IDs (9 sites) have direct modeled projections. The remaining 11 sites use the historical ratio method, where monthly scaling ratios are computed from the overlapping USGS observation period between the target site and its nearest USBR-gauged reference.

#### Gap-filling strategy

1. **USGS observations** take priority within their available record
2. **USBR ensemble mean** (or ratio-scaled synthetic) fills months outside the USGS range
3. **Monthly climatology** (mean of each calendar month from all available data) fills any remaining gaps in the 1896–2099 range

#### Streamflow raster creation

[`create_streamflow_rasters()`](hydrolibs/streamflowops.py) generates annual GeoTIFF rasters at 2 km resolution (1896–2099) where each pixel receives area-normalized annual streamflow (mm/yr) of its surface watershed:

1. **Watershed rasterization**: [`Surface_Watershed.geojson`](../Data/Inputs/GW_Data/Surface_Watershed.geojson) (16 polygons) is rasterized by `OBJECTID`. Each pixel is assigned the area-normalized average annual streamflow of all gauges within its watershed.
2. **Area normalization**: Gauge-averaged discharge (m³/s) is converted to mm/yr by dividing by the watershed area (m²): `mm/yr = Q(m³/s) × 86400 × 365.25 / A(m²) × 1000`. This yields units consistent with the other predictor bands (ET, ETo, precipitation, effective precipitation).
3. **CAP overlay**: Pixels within the [CAP Service Area](../Data/Inputs/GW_Data/CAP/CAP_Service_Area.geojson) (Maricopa, Pima, Pinal counties) receive additional Colorado River streamflow from Lees Ferry (09380000) and the CAP Canal at Havasu Pumping Plant (09426650), normalized by the CAP service area. This represents imported water delivered via the Central Arizona Project canal.

The CAP overlay does not double-count local watershed flows. Salt/Verde watershed pixels in the Phoenix AMA correctly receive both their local watershed streamflow (from SRP source gauges) and imported CAP water (from Colorado River gauges), reflecting the dual water supply in those areas.

---

## Pipeline overview (`pipeline.py`)

The pipeline is the top-level orchestrator that chains together data
preparation, ML model evaluation, full-period prediction, and
intercomparison with independent USGS datasets.  It is divided into five
numbered steps plus a cross-strategy summary:

```
Step 0  ─  Data Preparation
Step 1  ─  Create AZ Predictor DataFrame
Step 2  ─  Model Evaluation (3 strategies: Random, Temporal LOO, Spatial LOO)
Step 3  ─  Full-Period XGBoost Prediction (1896–2099)
Step 4  ─  USGS Intercomparison (Withdrawals, CU, IE)
```

### Configuration constants

All paths and modelling parameters are defined once at the top of
`pipeline.py`:

| Constant | Value | Description |
|---|---|---|
| `INPUT_DIR` | `../Data/Inputs/` | Root for all input datasets. |
| `OUTPUT_DIR` | `../Data/Outputs/` | Root for all generated outputs. |
| `WATER_USE` | `'All'` | Well filter (`'All'` or `'Irr_Wells'`). |
| `MOSAIC_RASTER_RES` | `2000` | Raster pixel size (m). |
| `TILE_SIZE` | `80000` | Tile size for GEE export (m). |
| `START_YEAR` | `1896` | First prediction year. |
| `END_YEAR` | `2099` | Last prediction year. |
| `YEAR_LIST` | `1984–2024` | Years with metered pumping data (ADWR). |
| `MAX_GW` | `3000` | Maximum allowed pumping depth (mm ≈ 10 ft). |
| `RANDOM_STATE` | `42` | Seed for reproducibility. |
| `N_TRIALS` | `100` | Optuna hyperparameter-tuning trials. |
| `FOLD_COUNT` | `5` | k-fold cross-validation folds. |
| `N_DASK_WORKERS` | `10` | Dask parallel workers. |
| `USE_OPTUNA` | `True` | Enable TPE-based hyperparameter search. |
| `USE_DASK` | `True` | Enable distributed training via Dask. |
| `USE_AMA_INA` | `True` | Restrict training to AMA/INA management areas. |
| `DROP_GW_BASINS` | `('WILLCOX AMA', …)` | Basins excluded from training. |
| `TEMPORAL_HOLDOUTS` | T1–T6 | Six temporal leave-one-out configurations. |
| `DROP_ATTRS` | (list) | Columns dropped before modelling. |

### Step 0 — Data preparation (`prepare_data()`)

Downloads, mosaics, and aligns all input datasets to a common 2 km grid.

1. **GEE data** — Downloads (or reuses) Google Earth Engine tiles via
   `dataops.download_gee_data()`, then mosaics them into annual predictor
   rasters with `dataops.mosaic_tiles()`.
2. **Groundwater data** — Preprocesses ADWR CSV records into per-year
   shapefiles (`gwops.preprocess_gw_csv()`), reprojects all vectors to a
   consistent CRS (`gwops.reproject_vectors()`), and creates GW volume →
   depth → cropped rasters (`gwops.create_gw_volume_rasters()`,
   `create_gw_depth_rasters()`, `crop_gw_rasters()`).
3. **Streamflow & canal density** — `streamflowops.create_canal_density_raster()`
   and `streamflowops.create_streamflow_rasters()` build predictor layers
   from USGS/USBR gauge data and CAP canal geometry.
4. **Basin & well rasters** — `gwops.create_gw_basin_rasters()` and
   `gwops.create_well_density_raster()` rasterise ADWR basins, sub-basins,
   and the well registry.
5. **GEE reprojection** — `dataops.reproject_gee_mosaics()` aligns all GEE
   mosaics to the GW depth raster grid.

**Outputs:**

| Directory | Contents |
|---|---|
| `GEE_Mosaics_2000m/` | Mosaicked annual GEE predictor rasters. |
| `GW/Vectors/{WNAME}/` | Per-year GW shapefiles. |
| `GW/Rasters/GW_Depths_{WNAME}_2000m/` | Pumping depth rasters (mm). |
| `GW_Data/Vector_Reproj/` | Reprojected vectors (basins, wells, CAP, etc.). |
| `Predictor_Data_{WNAME}_2000m/` | Final predictor stack (Predictor\_YYYY.tif). |

### Step 1 — Create AZ predictor DataFrame (`create_az_data()`)

Reads every year's multi-band predictor raster (1896–2099) plus the
basin, sub-basin, streamflow, canal-density, and well-density rasters into
a single DataFrame via `dataops.create_az_data_csv()`.  Each row represents
one pixel in one year; columns include all GEE predictors, ancillary
layers, basin/sub-basin labels, and (for metered years) observed pumping.

ADWR sub-basin OBJECTID codes are mapped to human-readable names using the
ADWR shapefile.  Exploratory data analysis (EDA) plots are generated via
`vizops.explore_az_data()` and saved to `{MODEL_DIR}EDA/`.

**Returns:** `az_df` — the full predictor DataFrame used by all subsequent
steps.

### Step 2 — Model evaluation

Three complementary strategies assess model performance.  Each strategy
trains all available ensemble tree models (XGBoost, LightGBM, Random Forest,
Extra Trees, Histogram Gradient Boosting, CatBoost, Gradient Boosting,
AdaBoost) using Optuna + Dask hyperparameter optimisation (100 TPE trials,
5-fold CV) and reports R², normalised RMSE (%), normalised MAE (%),
and normalised MBE (%).

#### Step 2a — Random 80/20 split (`evaluate_random()`)

A single randomised 80 %/20 % train/test split across all metered pixels
and years.  Trains and evaluates every model type, generating per-model
prediction plots, scatter diagrams, and residual maps via
`mlops.compare_all_models()` and `mlops.generate_model_visualizations()`.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Random/`

#### Step 2b — Temporal leave-one-out (`evaluate_temporal_loo()`)

Six pre-defined temporal holdout configurations (T1–T6):

| Holdout | Withheld years |
|---|---|
| T1 | 2015–2024 |
| T2 | 1990–1992, 2005–2007, 2022–2024 |
| T3 | 2007–2010 |
| T4 | 1985–1989, 2020–2024 |
| T5 | 2024 |
| T6 | 2010–2020 |

For each holdout, the model trains on the remaining years and is tested on
the held-out period.  Per-holdout metrics are recorded, then averaged across
all six splits.  Heatmaps and bar plots (`vizops.plot_loo_heatmap()`,
`vizops.plot_loo_bar()`) visualise model performance across holdouts.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Temporal_LOO/`

#### Step 2c — Spatial leave-one-out (`evaluate_spatial_loo()`)

Iterates over every ADWR sub-basin within AMA/INA management areas.  For
each sub-basin the model trains on the rest of Arizona and is tested on the
held-out region.  Only sub-basins with metered data in the 1984–2024 period
are included.  Bias correction is applied at level 1 (global).

**Outputs:** `{MODEL_DIR}Model_Evaluation/Spatial_LOO/`

#### Cross-strategy summary

After all three strategies complete,
`vizops.create_cross_strategy_summary()` produces comparison plots and
tables across Random, Temporal LOO, and Spatial LOO results, saved to
`{MODEL_DIR}Model_Evaluation/`.

### Step 3 — Full-period prediction (`predict_full_period()`)

The core production step.  Trains a single XGBoost model on **all**
metered data (1984–2024, no holdout) to maximise the training signal, then
predicts annual pumping for every 2 km pixel from 1896 to 2099.

#### 3a. Model training & interpretability

After training, three interpretability diagnostics are generated and saved
to `{prediction_dir}Model_Interpretability/`:

- **Permutation importance** (`mlops.compute_perm_imp()`)
- **Accumulated Local Effects (ALE) plots** (`mlops.compute_ale_plots()`)
- **SHAP plots** (`mlops.compute_shap_plots()`)

#### 3b. Annual raster prediction loop (1896–2099)

For each year the pipeline:

1. **Predicts** total pumping (mm) across all valid pixels.
2. **Partitions** predictions into eight withdrawal categories via
   `partitionops.partition_predictions()`:
   Irrigation, Non-Irrigation, Irrigation\_GW, Irrigation\_SW,
   Non\_Irrigation\_GW, Non\_Irrigation\_SW, Total\_GW, Total\_SW.
3. **Computes consumptive use (CU):**
   ```
   CU = max(Irrigation_ET − Effective_Precip, 0)
   ```
   Split into Irrigation\_CU, Irrigation\_GW\_CU, Irrigation\_SW\_CU using
   the GW fraction.
4. **Computes irrigation efficiency (IE):**
   ```
   IE = CU / Withdrawal
   ```
   Producing Irrigation\_Efficiency, Irrigation\_GW\_Efficiency,
   Irrigation\_SW\_Efficiency.
5. **Writes rasters** in four units for depth/volume products and as
   dimensionless ratios for IE:

| Product | Units written | File naming |
|---|---|---|
| Total pumping | mm, ft, m³, AF | `Predicted_GW_{year}_{unit}.tif` |
| 8 withdrawal categories | mm, ft, m³, AF | `{Category}_{year}_{unit}.tif` |
| 3 CU categories | mm, ft, m³, AF | `{CU_Category}_{year}_{unit}.tif` |
| 3 IE categories | dimensionless | `{IE_Category}_{year}.tif` |

6. **Accumulates statistics** for AZ-wide, per-basin, and per-sub-basin
   totals (volume in m³ and AF, mean depth in mm) for every category.

Unit conversions:
- Pixel area: 2000² = 4 000 000 m²
- mm → m³: mm × pixel\_area\_m² / 1000
- m³ → AF: m³ / 1233.48
- mm → ft: mm / 304.8

#### 3c. Visualisation

After the prediction loop, the pipeline generates time series, era summary
maps, basin-level plots, and sub-basin-level plots for every product:

| Plot type | Function | Applied to |
|---|---|---|
| Full-period time series (1896–2099) | `vizops.create_full_period_time_series()` | Total, 8 categories, 3 CU, 3 IE |
| Era summary maps | `vizops.create_era_summary_maps()` | Total, 8 categories, 3 CU |
| Per-basin time series | `vizops.create_basin_time_series()` | Total, 8 categories, 3 CU, 3 IE |
| Per-sub-basin time series | `vizops.create_subbasin_time_series()` | Total, 8 categories, 3 CU, 3 IE |

Four temporal eras are distinguished in the plots:

| Era | Years | Description |
|---|---|---|
| Hindcast | 1896–1983 | Pre-metered; predictions only. |
| Historical | 1984–2024 | Metered period; predictions vs. actuals. |
| Forecast | 2025 | Transition year. |
| Projected | 2026–2099 | Future projections. |

#### 3d. Well package

`wellops.create_well_package()` disaggregates pixel-level withdrawal
rasters to individual wells from the ADWR Well Registry and writes a
GeoPackage (`Well_Package.gpkg`).  See the `wellops` module description
below for the capacity-proportional distribution logic.

**Outputs:** `{MODEL_DIR}Full_Prediction_XGB/`

### Step 4 — USGS intercomparison

#### Step 4a — Withdrawal intercomparison (`run_usgs_intercomparison()`)

Compares ML-based irrigation GW and SW withdrawal predictions against two
independent USGS datasets at the Arizona groundwater basin scale:

| Dataset | Native resolution | Source |
|---|---|---|
| ML predictions | 2 km rasters | Pipeline Step 3 |
| USGS NHM | HUC12 polygons (Mgal/d) | `USGS_NHM_Withdrawals/` |
| USGS Reitz | 800 m rasters (m/yr) | `USGS_Reitz_Irrigation/` |

Because the three products live at different native resolutions, each is
aggregated to Arizona groundwater basin totals (volume in AF) for
comparison.  The intercomparison produces:

- Pairwise metrics (RMSD, MAD, Percent Difference) for ML vs NHM, ML vs
  Reitz, NHM vs Reitz in both GW and SW categories.
- Per-basin comparison tables (mm, ft, m³, AF).
- Time series CSVs and per-basin time series plots.
- Pairwise scatter plots with 1:1 lines and linear fits.
- Spatial difference maps (diverging colourmap centred on zero).

All outputs are written to `{prediction_dir}Intercomparison/`.

#### Step 4b — CU / IE intercomparison (`run_cu_ie_usgs_intercomparison()`)

Compares ML-based Irrigation Consumptive Use and Irrigation Efficiency with
USGS NHM HUC12-scale data at the basin scale:

| Product | ML source | USGS source |
|---|---|---|
| **CU** | `Irrigation_CU_Rasters/Depth_mm/` (mm) | `Irr_CU_HUC12_Tot_annual_2000_2020.csv` (Mgal/d) |
| **IE** | `Irrigation_Efficiency_Rasters/` (ratio) | `IR_HUC12_Eff_annual_2000_2020.csv` (ratio) |

CU follows the same volume-based framework as withdrawals (RMSD, MAD, %
Difference in AF, m³, mm).  IE uses dimensionless ratio metrics.  Outputs
include metrics CSVs, per-basin tables, time series, and scatter plots,
written to `{prediction_dir}CU_IE_Intercomparison/`.

---

## Library modules (`hydrolibs/`)

### `dataops.py` — Data acquisition & preparation

Downloads and pre-processes Google Earth Engine predictor datasets and
assembles them into the training DataFrame.

Key functions:
- **`download_gee_data()`** — Exports GEE image collections to Cloud
  Storage, then downloads as tiled GeoTIFFs.
- **`mosaic_tiles()`** — Merges GEE tiles into annual mosaics.
- **`reproject_gee_mosaics()`** — Reprojects mosaics to the GW raster grid.
- **`create_az_data_csv()`** — Reads all years' predictor rasters and
  stacks them with basin labels and observed pumping into a single DataFrame.
- **`create_train_test_data()`** — Splits the DataFrame into train/test
  sets using one of four strategies (temporal, spatial, random ratio,
  random 80/20).

### `mlops.py` — Machine learning operations

Builds, tunes, evaluates, and interprets ensemble tree models.

Key functions:
- **`get_model_param_dict()`** — Returns the hyperparameter search spaces
  for XGB, LGBM, RF, ETR, HGBR, CatBoost, GBR, and AdaBoost.
- **`build_ml_model_optuna_dask()`** — Trains a single model with Optuna
  TPE-based hyperparameter search parallelised across Dask workers.
- **`compare_all_models()`** — Trains all models on a common split and
  ranks them by test R².
- **`get_prediction_results()`** — Makes predictions and applies multi-level
  bias correction.
- **`perform_bias_correction()`** — Applies basin-level or global bias
  correction using linear scaling.
- **`calc_train_test_metrics()`** — Computes R², normalised RMSE, MAE, MBE.
- **`compute_perm_imp()`**, **`compute_ale_plots()`**,
  **`compute_shap_plots()`** — Model interpretability diagnostics.
- **`generate_model_visualizations()`** — Scatter, residual, and time series
  plots per model.

### `visualops.py` — Visualisation

Produces journal-quality figures for every stage of the pipeline.

Key functions:
- **`explore_az_data()`** — Exploratory data analysis (histograms,
  correlation matrices, feature distributions by era).
- **`create_full_period_time_series()`** — Annual pumping line plot (1896–
  2099) with era shading and optional observed-data overlay.
- **`create_era_summary_maps()`** — Spatial maps of mean depth for each era
  (Hindcast, Historical, Forecast, Projected).
- **`create_basin_time_series()`** / **`create_subbasin_time_series()`** —
  Per-basin and per-sub-basin annual trends with AMA/INA colour coding.
- **`plot_loo_heatmap()`** / **`plot_loo_bar()`** — Heatmaps and bar plots
  for leave-one-out evaluation results.
- **`create_cross_strategy_summary()`** — Side-by-side comparison of Random,
  Temporal LOO, and Spatial LOO results.

### `gwops.py` — Groundwater data processing

Handles ADWR groundwater withdrawal records and raster generation.

Key functions:
- **`preprocess_gw_csv()`** — Converts raw ADWR CSVs into per-year
  shapefiles.
- **`reproject_vectors()`** — Reprojects basin, sub-basin, well, CAP, and
  streamflow vectors to a consistent CRS.
- **`create_gw_volume_rasters()`** / **`create_gw_depth_rasters()`** —
  Rasterises pumping volumes (AF) and converts to depth (mm).
- **`crop_gw_rasters()`** — Clips GW rasters to the Arizona boundary.
- **`create_gw_basin_rasters()`** — Rasterises ADWR basin and sub-basin
  polygons.
- **`create_well_density_raster()`** — Creates a well-count-per-pixel
  raster from the Well Registry.

### `streamflowops.py` — Streamflow & canal data

Downloads and processes streamflow data from USGS and USBR sources.

Key functions:
- **`download_streamflow()`** — Downloads monthly streamflow records from
  USGS gauges and retrieves USBR delivery data.
- **`create_streamflow_rasters()`** — Rasterises annual streamflow volumes
  onto the 2 km grid using watershed polygons.
- **`create_canal_density_raster()`** — Rasterises CAP canal geometry into
  a canal-density layer used for SW-fraction estimation.

### `partitionops.py` — Water-budget partitioning

Decomposes total pumping predictions into eight withdrawal categories using
ancillary data already in the predictor stack:

| Category | Derivation |
|---|---|
| **Irrigation** | `total × irr_fraction` (USGS irrigation-fraction raster) |
| **Non_Irrigation** | `total − Irrigation` |
| **Irrigation_GW** | `Irrigation × gw_fraction` (USGS GW-fraction snapshots) |
| **Irrigation_SW** | `Irrigation − Irrigation_GW` |
| **Non_Irrigation_GW** | `Non_Irrigation × (1 − sw_fraction)` |
| **Non_Irrigation_SW** | `Non_Irrigation × sw_fraction` (canal-density proxy) |
| **Total_GW** | `Irrigation_GW + Non_Irrigation_GW` |
| **Total_SW** | `Irrigation_SW + Non_Irrigation_SW` |

Key helpers:
- **`focal_fill_irr_fraction()`** — fills edge-pixel gaps (`irr_frac < 0.05`)
  with a focal mean of valid neighbours, avoiding NaN propagation along
  irrigated-area boundaries.
- **`compute_sw_fraction()`** — normalises canal density to [0, 1] using a
  local-maximum filter (`maximum_filter(size=5)`), so that the pixel with the
  highest canal density in a 5 × 5 window receives `sw_fraction = 1.0`.
- **`partition_predictions()`** — orchestrates all splits, applies well-density
  masking, and returns a dict keyed by the eight category names.

All partitions use subtraction from the parent total (e.g., `nonirr = total − irr`)
to guarantee exact budget closure with no floating-point drift.

### `wellops.py` — Well-level withdrawal package

Disaggregates pixel-level withdrawal rasters to individual wells from the
ADWR Well Registry and writes a GeoPackage (`Well_Package.gpkg`).

**Sampling**: Only the **mm** rasters are read (9 categories per year); ft, m³,
and acre-ft values are computed arithmetically, reducing I/O by 75 %.

**Distribution logic** — when multiple wells share a 2 km pixel, the pixel
total is split using capacity-proportional weights with a three-tier fallback:

1. **Historical pumping** — mean `AF Pumped` across all years a well appears
   in the per-year GW shapefiles (`GW_YYYY.shp`).  These cover metered wells
   within AMA/INA management areas (~3 k wells/year, 1984–2024).
2. **PUMPRATE fallback** — for unmetered wells, the `PUMPRATE` field (GPM)
   from the Well Registry is used (~79 k wells have this attribute).
3. **Equal-share fallback** — wells with neither record receive weight 1.0.

Within each pixel the raw weights are normalised to sum to 1, so the pixel
budget is preserved regardless of which tier each well belongs to.

**Nodata masking**: Wells landing in raster nodata or out-of-bounds pixels
are dropped before weight computation, preventing valid wells from losing
share to neighbours in invalid pixels.

**Zero floor**: A `np.maximum(all_mm, 0)` clamp is applied after sampling to
eliminate any negative model artifacts before unit conversion.

### `intercompops.py` — USGS intercomparison

Basin-scale comparison of ML predictions with independent USGS datasets.

**Withdrawal intercomparison** (`run_intercomparison()`):
- Loads ML, NHM, and Reitz data; aggregates to basin volumes (AF).
- Computes pairwise RMSD, MAD, Percent Difference.
- Produces per-basin time series, scatter plots, and spatial difference maps.

**CU / IE intercomparison** (`run_cu_ie_intercomparison()`):
- Compares ML CU (mm) and IE (ratio) with NHM HUC12 annual data.
- CU: Mgal/d → m³/yr → depth (mm) → basin volumes (AF).
- IE: dimensionless ratio → area-weighted basin means.
- Produces metrics, per-basin tables, time series, and scatter plots.

### `rasterops.py` — Raster I/O utilities

Core raster manipulation operations.

Key functions:
- **`read_raster_as_arr()`** — Reads a raster into a NumPy array.
- **`write_raster()`** — Writes a NumPy array to a GeoTIFF.
- **`crop_raster()`** / **`crop_rasters()`** — Clips rasters to a boundary.
- **`reproject_raster_gdal()`** — Reprojects a raster using GDAL.
- **`get_xy_grids_from_raster()`** — Generates easting/northing coordinate
  grids from a raster's transform.

### `vectorops.py` — Vector utilities

Vector file operations including reprojection, format conversion, and
rasterisation.

Key functions:
- **`reproject_vector()`** — Reprojects a vector file to a target CRS.
- **`csv2shp()`** / **`csvs2shps()`** — Converts CSV tables with
  coordinates to shapefiles.
- **`shp2raster()`** / **`shps2rasters()`** — Rasterises vector features
  onto a reference grid.
- **`add_attribute_well_reg()`** — Joins Well Registry attributes to
  shapefiles.

### `sysops.py` — System utilities

File and directory management helpers.

Key functions:
- **`makedirs()`** — Creates directories recursively (no error if exists).
- **`copy_files()`** / **`copy_file()`** — Copies files between directories.
- **`az_nodata()`** — Returns the standard nodata value for AZ rasters.

---

## Output directory structure

After a full pipeline run, the output tree looks like:

```
Data/Outputs/
├── GEE_Mosaics_2000m/                      # Mosaicked GEE predictor tiles
├── GW/
│   ├── Rasters/GW_Depths_All_Wells_2000m/   # Observed pumping rasters (mm)
│   └── Vectors/All_Wells/                   # Per-year GW shapefiles
├── GW_Data/Vector_Reproj/                   # Reprojected basins, wells, etc.
├── Predictor_Data_All_Wells_2000m/          # Multi-band Predictor_YYYY.tif
│
└── ML_Model_All_Wells_2000m/
    ├── EDA/                                 # Exploratory data analysis plots
    ├── Model_Evaluation/
    │   ├── Random/                          # Step 2a results
    │   ├── Temporal_LOO/                    # Step 2b results (T1–T6)
    │   └── Spatial_LOO/                     # Step 2c results (per sub-basin)
    │
    └── Full_Prediction_XGB/
        ├── Model_Interpretability/          # SHAP, ALE, permutation importance
        ├── Predicted_Rasters/               # Total pumping (4 units)
        │   ├── Depth_mm/
        │   ├── Depth_ft/
        │   ├── Volume_m3/
        │   └── Volume_AF/
        ├── {Category}_Rasters/              # 8 withdrawal categories (4 units)
        ├── Irrigation_CU_Rasters/           # CU (4 units)
        ├── Irrigation_Efficiency_Rasters/   # IE (dimensionless)
        ├── Visualizations/                  # Time series & era summary maps
        ├── Well_Package/                    # Per-well GeoPackage
        ├── Intercomparison/                 # Step 4a — withdrawal comparison
        └── CU_IE_Intercomparison/           # Step 4b — CU/IE comparison
```