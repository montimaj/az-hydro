# AZ-Hydro: ML Pipeline for Arizona Water Use Estimation (1896–2099)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19057936.svg)](https://doi.org/10.5281/zenodo.19057936)

Maintainers [Dr. Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu]

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). The Arizona Water Use Dataset (1896–2099): Withdrawals, consumptive use, and irrigation efficiency partitioned by source. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Two Centuries of Groundwater and Surface Water Withdrawals, Consumptive Use, and Irrigation Efficiency (1896–2099). _In prep. for AGU Earth's Future_.

---

## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.11). If Anaconda or miniconda is already installed, skip this step. 

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

> **First-time run:** The default flags (`--skip-download`, `--load-files`) assume
> GEE tiles and intermediate files already exist on disk. If you are starting
> from scratch, use `--download --recreate` to fetch the data and build all
> intermediate files:
>
> ```bash
> python pipeline.py --download --recreate
> ```

The pipeline supports selective step execution via CLI arguments:

```bash
python pipeline.py                        # run all steps (default)
python pipeline.py --steps 0,1,2a         # run only steps 0, 1, and 2a
python pipeline.py --steps 3              # prediction only
python pipeline.py --steps 3,3b           # prediction + uncertainty quantification
python pipeline.py --download --recreate  # force fresh GEE download and file recreation
python pipeline.py --skip-eda            # skip EDA plot generation
```

Step 0 supports fine-grained sub-step control via `--skip-prep`:

```bash
python pipeline.py --steps 0 --skip-prep gee               # skip GEE download & mosaic
python pipeline.py --steps 0 --skip-prep gw-csv,gw-rasters # skip GW CSV and raster creation
python pipeline.py --recreate --skip-prep streamflow        # recreate everything except streamflow
python pipeline.py --skip-prep gee,vectors,reproject        # skip multiple sub-steps
```

#### Available steps

| Step | Description |
|------|-------------|
| `0`  | Data preparation (GEE download, GW processing, rasterisation) |
| `1`  | Create AZ predictor dataset (Parquet) |
| `2a` | Evaluate random 80/20 train/test split |
| `2b` | Evaluate LOO temporal holdout |
| `2c` | Evaluate LOO spatial holdout |
| `3`  | Full-period XGBoost prediction (1896–2099) |
| `3b` | Hybrid uncertainty quantification |
| `3g` | Raster maps, actual vs predicted, and trend analysis for all output categories |
| `4`  | USGS intercomparison |
| `4b` | CU / IE intercomparison |
| `4c` | CAP/SRP surface-water validation |
| `4d` | Effective precipitation intercomparison |
| `4e` | Non-irrigation vs USGS Public Supply intercomparison |

#### Step 0 sub-steps

| Sub-step | Description |
|----------|-------------|
| `gee` | GEE tile download & mosaic |
| `gw-csv` | GW CSV → per-year shapefiles |
| `vectors` | Reproject vectors |
| `gw-rasters` | GW volume → depth → cropped rasters |
| `streamflow` | Canal density & streamflow rasters |
| `basin-rasters` | GW basin, sub-basin & well density rasters |
| `reproject` | Reproject GEE mosaics to match GW grid |

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `all` | Comma-separated step IDs to run (e.g. `"0,1,2a"`) or `"all"`. |
| `--skip-download` | `True` | Skip GEE tile download; use existing tiles on disk. |
| `--download` | — | Force GEE tile download. |
| `--load-files` | `True` | Skip recreating intermediate files that already exist. |
| `--recreate` | — | Force recreation of intermediate files. |
| `--skip-eda` | `False` | Skip EDA plot generation in Step 1. |
| `--skip-prep` | — | Comma-separated Step 0 sub-steps to skip. |
| `-v`, `--verbose` | `False` | Enable verbose (DEBUG-level) logging. |

The pipeline executes the selected steps in sequence (details below).

---

## Data sources

The project builds a spatially explicit, multi-decadal (1896–2099) dataset for Arizona by combining satellite-derived products, climate model projections, soil properties, streamflow observations, and USBR modeled streamflow.

### Google Earth Engine (GEE) predictor bands

The [`download_gee_data()`](hydrolibs/dataops.py) function downloads 14 bands of geospatial data from GEE ([Gorelick et al., 2017](https://doi.org/10.1016/j.rse.2017.06.031); [Roy et al., 2025](https://doi.org/10.5281/zenodo.17641528)) as tiled GeoTIFFs at 2 km resolution over Arizona. Data are harmonized across three temporal eras using overlap-period bias-correction ratios to ensure continuity.

| Band | Description | Units | Source |
|------|-------------|-------|--------|
| `annual_et_ensemble_mm` | Actual evapotranspiration | mm/yr | [Reitz et al., 2023](https://doi.org/10.1029/2022WR034012) (1896–1999), [OpenET (Melton et al., 2022](https://doi.org/10.1111/1752-1688.12956); [Volk et al., 2024)](https://doi.org/10.1038/s44221-023-00181-7) (2000–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) ensemble (2026–2099) |
| `annual_eto_mm` | Reference evapotranspiration (Penman-Monteith) | mm/yr | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) Hargreaves (1896–1978), [gridMET (Abatzoglou, 2013)](https://doi.org/10.1002/joc.3413) (1979–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) bias-corrected ([Volk et al., 2026](https://doi.org/10.5281/zenodo.18673484)) ensemble (2026–2099) |
| `annual_precip_mm` | Precipitation | mm/yr | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) ensemble (2026–2099) |
| `annual_peff_mm` | Effective precipitation (USDA SCS method) | mm/yr | [USDA SCS, 1993](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf); [Muratoglu et al., 2023](https://doi.org/10.1016/j.watres.2023.120011); [Majumdar et al., 2026](https://doi.org/10.5281/zenodo.18706481) |
| `annual_peff_pcml_mm` | Effective precipitation (PCML obs-based, 2000–2024) | mm/yr | [Hasan et al., 2025](https://doi.org/10.1016/j.agwat.2025.109821), climatological mean outside 2000–2024 |
| `annual_tmmx_K` | Annual mean daily max temperature | K | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) (2026–2099) |
| `annual_tmmn_K` | Annual mean daily min temperature | K | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) (2026–2099) |
| `lulc` | Land use/land cover (1=Agriculture, 2=Urban, 3=Surface Water) | categorical | [USGS historical (Sohl et al., 2016)](https://doi.org/10.1080/1747423X.2016.1147619) (≤1984), [NLCD (USGS, 2024](https://doi.org/10.5066/P94UXNTS); [Fleckenstein et al., 2026)](https://doi.org/10.1016/j.rse.2026.115347) (1985–2025), [USGS LULC Projections (Sohl et al., 2014)](https://doi.org/10.1890/13-1245.1) projections (2026–2099) |
| `annual_crop_fraction` | Cropland fraction | fraction | Derived from USGS LULC |
| `annual_irr_fraction` | Irrigated area fraction | fraction | [IrrMapper (Ketchum et al., 2020](https://doi.org/10.3390/rs12142328); [2023)](https://doi.org/10.1038/s43247-023-01152-2) RF v1.2 (1985–2025), LULC-derived outside |
| `annual_gw_fraction` | Groundwater irrigation fraction | fraction | [Hung et al., 2025](https://doi.org/10.1038/s41597-025-05920-x) snapshots (2000, 2005, 2010, 2015) |
| `soil_depth_mm` | Soil depth | mm | [CSRL (Walkinshaw et al., 2022)](https://casoilresource.lawr.ucdavis.edu/soil-properties/) (static) |
| `awc_mm` | Available water capacity (0–152 cm) | mm | [SSURGO](https://websoilsurvey.nrcs.usda.gov/) (static) |
| `ksat_mean_micromps` | Saturated hydraulic conductivity | μm/s | [CSRL (Walkinshaw et al., 2022)](https://casoilresource.lawr.ucdavis.edu/soil-properties/) (static) |

### Data harmonization

The pipeline stitches disparate sources into a consistent 1896–2099 time series:

- **ET**: Reitz ensemble (1896–1999) → OpenET v2.0/v2.1 (2000–2025) → MACA × EToF crop coefficients (2026–2099)
- **ETo**: PRISM Hargreaves (1896–1978) → gridMET (1979–2025) → MACA 20-model ensemble (2026–2099)
- **LULC**: USGS historical scenario (≤1984) → NLCD (1985–2025) → USGS 4-scenario mode ensemble (2026–2099)
- **Climate projections**: MACA v2 daily data across 20 GCMs × 2 RCPs (RCP 4.5, RCP 8.5) = 40-member ensemble. All MACA queries use a flat-pipeline approach (single filter + reduce) to keep GEE computation graphs small: ETo uses `.sum().divide(40)` per month (computed per-image to preserve nonlinearity), precip uses `.sum().divide(40)`, and temperature uses `.mean()`.

Per-pixel, per-month bias-correction ratios are computed from overlapping observation periods and applied to extend each variable seamlessly. See [`gee/README.md`](../gee/README.md) for asset export details and equations.

### GEE pre-exported assets

Twelve custom ImageCollections are pre-computed via scripts in [`gee/`](../gee/) and stored in GEE under `projects/azhydro/assets/`:

| Asset | Description | Years |
|-------|-------------|-------|
| `gridmet_hargreaves_eto_ratio` | gridMET / PRISM Hargreaves monthly ratio (12 images) | Climatology |
| `openet_reitz_et_ratio` | OpenET / Reitz ensemble monthly ratio (12 images) | Climatology |
| `monthly_etof` | Crop coefficient (OpenET / gridMET ETo) | Climatology |
| `prism_hargreaves_eto` | PRISM-based Hargreaves ETo | 1896–1978 |
| `usgs_adjusted_et` | Bias-adjusted Reitz actual ET | 1896–1999 |
| `maca_monthly_eto_v2` | MACA per-model/scenario projected ETo, bias-corrected using [Volk et al., (2026)](https://doi.org/10.5281/zenodo.18673484) | 2026–2099 |
| `maca_monthly_et_v2` | MACA ensemble projected actual ET | 2026–2099 |
| `lulc_projection_ensemble` | USGS 4-scenario LULC mode | 2026–2099 |
| `monthly_peff_v2` | USDA SCS effective precipitation | 1896–2099 |
| `maca_gcm_annual_eto` | Per-GCM annual ETo for σ_MACA (370 images) | 2026–2099 |
| `maca_gcm_annual_et` | Per-GCM annual ET for σ_MACA (370 images) | 2026–2099 |
| `maca_gcm_annual_peff` | Per-GCM annual Peff for σ_MACA (370 images) | 2026–2099 |

### Download architecture

Data are downloaded as tiles using a Dask-parallelized worker pool (40 workers, 1 GB each). Each tile covers an 80 km × 80 km region at 2 km resolution. Tiles are later mosaicked and reprojected for the ML pipeline.

### Streamflow analysis

The [`streamflowops`](hydrolibs/streamflowops.py) module handles streamflow data acquisition and rasterization. It covers all 16 Arizona surface watersheds from 1896 to 2099.

#### Data sources

- **USGS NWIS**: Daily mean discharge (parameter 00060) via the `dataretrieval` Python API ([Hodson et al., 2023](https://doi.org/10.5066/P94I5TX3)), resampled to monthly means
- **USBR CMIP Ensemble**: Monthly modeled streamflow ([Gangopadhyay & Pruitt, 2011](https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf); [USBR, 2025](https://rise-usbr.opendata.arcgis.com/)) averaged across ~112 climate model runs (scenarios a1b, a2, b1), spanning 1950–2099
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

**Known limitation:** Within each watershed, streamflow depth is assigned
uniformly to all pixels.  In reality, streamflow is concentrated near
channels.  The canal-weighted streamflow variant (`Canal_Weighted_Streamflow_*.tif`)
partially mitigates this by redistributing streamflow proportionally to
canal density (segment count per pixel) derived from the GRAIN dataset
([Suresh et al., 2026](https://doi.org/10.5194/essd-18-1855-2026)),
concentrating flow where delivery infrastructure exists.  A further refinement could use distance-to-NHD
flowlines as a weighting factor, but this data product is not currently
in the pipeline.

---

## Pipeline overview (`pipeline.py`)

The pipeline is the top-level orchestrator that chains together data
preparation, ML model evaluation, full-period prediction, and
intercomparison with independent USGS datasets.  It is divided into five
numbered steps plus a cross-strategy summary:

```
Step 0   ─  Data Preparation
Step 1   ─  Create AZ Predictor DataFrame
Step 2   ─  Model Evaluation (3 strategies: Random, Temporal LOO, Spatial LOO)
Step 3   ─  Full-Period XGBoost Prediction (1896–2099)
Step 3g  ─  Raster Maps & Trend Analysis for All Output Categories
Step 4   ─  USGS Intercomparison (Withdrawals, CU, IE, Peff)
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
| `MAX_GW` | `None` | Maximum allowed pumping depth (mm); defaults to 10,000 mm (~32,400 AF per pixel) when `None`. |
| `AF_MAX_THRESHOLD` | `5000` | Maximum per-well `AF Pumped`; rows exceeding this are dropped from CSVs. |
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
   shapefiles (`gwops.preprocess_gw_csv()`).  Per-well `AF Pumped` values
   exceeding `af_max_threshold` (default **5,000 AF**, roughly the physical
   ceiling for a single high-capacity well at ~3,000 gpm sustained
   year-round) are dropped to remove aggregated well-field totals and data
   entry errors.  Vectors are then reprojected to a
   consistent CRS (`gwops.reproject_vectors()`), and GW volume →
   depth → cropped rasters are created (`gwops.create_gw_volume_rasters()`,
   `create_gw_depth_rasters()`, `crop_gw_rasters()`).  A pixel-level raster
   cap (default **10,000 mm** ≈ 32,400 AF at 4 km² when `MAX_GW=None`)
   catches any remaining `gdal_rasterize` artifacts.
3. **Streamflow & canal density** — `streamflowops.create_canal_density_raster()`
   and `streamflowops.create_streamflow_rasters()` build predictor layers
   from USGS/USBR gauge data and GRAIN canal geometry ([Suresh et al., 2026](https://doi.org/10.5194/essd-18-1855-2026)).
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
a single DataFrame via `dataops.create_az_data_parquet()`.  Each row represents
one pixel in one year; columns include all GEE predictors, ancillary
layers, basin/sub-basin labels, and (for metered years) observed pumping.

ADWR sub-basin OBJECTID codes are mapped to human-readable names using the
ADWR shapefile.  Exploratory data analysis (EDA) plots are generated via
`vizops.explore_az_data()` and saved to `{MODEL_DIR}EDA/`.

**Returns:** `az_df` — the full predictor DataFrame used by all subsequent
steps.

### Step 2 — Model evaluation

Three complementary strategies assess model performance.  Each strategy
trains all available models — baseline linear regressors (Linear Regression,
Ridge, Lasso) and ensemble tree models (XGBoost, LightGBM, Random Forest,
Extra Trees, Histogram Gradient Boosting, CatBoost, Gradient Boosting,
AdaBoost) — using Optuna + Dask hyperparameter optimisation (100 TPE trials,
5-fold CV; 1 trial for parameter-free baselines) and reports R², normalised
RMSE (% of σ), normalised MAE (% of σ), and normalised MBE (%).  NRMSE and
NMAE are normalized by the standard deviation of observed values rather than
the mean, which is more appropriate for the right-skewed pumping distribution
(where mean-normalisation underestimates relative error).  NMBE remains
mean-normalised since bias direction relative to the mean is the meaningful
quantity.  The linear baselines
provide a reference for quantifying the value added by nonlinear models.

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
`vizops.create_cross_strategy_summary()` produces:

- **`Cross_Strategy_Summary.csv`** — all models × all strategies with R²,
  RMSE, MAE, MBE, and Overfit R² columns.
- **`Cross_Strategy_Summary.tex`** — LaTeX table (`\begin{table}`) ready
  for direct inclusion in journal manuscripts (booktabs formatting with
  multi-column strategy headers).
- **`Cross_Strategy_Comparison.png`** — grouped bar chart of R², RMSE, MAE,
  and Overfit R² across strategies.

Saved to `{MODEL_DIR}Model_Evaluation/`.

### Step 3 — Full-period prediction (`predict_full_period()`)

The core production step.  Trains a single XGBoost model on **all**
metered data (1984–2024, no holdout) to maximise the training signal, then
predicts annual pumping for every 2 km pixel from 1896 to 2099.

**Absolute-value post-processing:** All XGBoost predictions are wrapped in
`np.abs()` because groundwater pumping depth is physically non-negative.
Tree-based regressors can produce small negative values near zero
(numerical noise at the leaf level), and `abs()` ensures physical
validity.  The same transform is applied consistently in Optuna CV
scoring, uncertainty ensemble generation, and bias correction, so all
metrics and CIs are evaluated on the same transformed quantity.

**Temporal extrapolation caveat:** The XGBoost model is trained on 1984–
2024 metered data and predicts outside this range (1896–1983 hindcast,
2025 forecast, 2026–2099 projection).  Tree-based regressors cannot
extrapolate beyond the training range of any individual predictor; they
instead plateau at the nearest leaf value.  Outputs outside 1984–2024
should therefore be interpreted as *plausible scenarios under stationary
predictor–response relationships*, not forecasts.  The further a year is
from the training window, the less constrained the prediction.

#### 3a. Model training & interpretability

After training, three interpretability diagnostics are generated on the
training data and saved to `{prediction_dir}Model_Interpretability/`:

- **Permutation importance** (`mlops.compute_perm_imp()`)
- **Accumulated Local Effects (ALE) plots** (`mlops.compute_ale_plots()`)
- **SHAP plots** (`mlops.compute_shap_plots()`)

After the full-period prediction loop completes, the same three diagnostics
are re-computed separately for each temporal era:

| Era | Years | Purpose |
|---|---|---|
| Hindcast | 1896–1983 | Pre-training extrapolation |
| Training | 1984–2024 | In-distribution reference |
| Projection | 2025–2099 | Future extrapolation |

Up to 2 000 pixels are subsampled per year (capped at 10 000 per era) and
passed through SHAP, ALE, and permutation importance analysis.  Comparing
the resulting feature-contribution profiles across eras reveals whether the
model relies on the same physical relationships in extrapolation as during
training.  Stable feature rankings and ALE shapes across eras support the
stationarity assumption; divergent patterns flag features whose
out-of-distribution behaviour may reduce prediction reliability.  Outputs
are saved to `{prediction_dir}Model_Interpretability/{Era}/`.

#### 3b. Annual raster prediction loop (1896–2099)

Before the loop begins, an **out-of-distribution (OOD) detector**
(`mlops.OODDetector`) is fitted on the training feature matrix.  The
detector computes the Mahalanobis distance of each prediction-time pixel
from the training distribution using a regularised covariance matrix.
Pixels exceeding the χ²(n\_features) threshold at α = 0.01 are flagged as
OOD — i.e., their feature vector lies outside the region spanned by the
1984–2024 training data.

A **prediction exceedance check** complements the OOD detector from the
opposite direction.  While OOD flags feature-space extrapolation, the
exceedance check flags *output-space* implausibility: per-pixel predictions
exceeding the training-era maximum (or 99th percentile) pumping depth
indicate physically implausible rates, since modern pump infrastructure
operates near hydraulic efficiency limits (~75–85 %) and volumetric capacity
is unlikely to change substantially over the projection horizon.  Per-year
exceedance statistics are accumulated and written to
`Prediction_Exceedance_Summary.csv` with era-level summaries.

For each year the pipeline:

1. **Predicts** total pumping (mm) across all valid pixels.
2. **Checks prediction exceedance** against the training-era per-pixel
   maximum and P99 pumping depth.  Pixels exceeding these thresholds are
   counted per year.
3. **Flags out-of-distribution pixels** via the OOD detector.  Per-year
   binary flag rasters (`OOD_Flag_{year}.tif`, 1 = OOD, 0 = in-distribution)
   are written to `OOD_Rasters/`.  Per-year statistics (n\_ood, pct\_ood,
   mean/max Mahalanobis d²) are accumulated and written to
   `OOD_Rasters/OOD_Summary.csv` after the loop.  Era-level OOD rates
   (hindcast 1896–1983, training 1984–2024, projection 2025–2099) are
   logged with warnings when the mean OOD rate exceeds 10 %.
4. **Partitions** predictions into eight withdrawal categories via
   `partitionops.partition_predictions()`:
   Irrigation, Non-Irrigation, Irrigation\_GW, Irrigation\_SW,
   Non\_Irrigation\_GW, Non\_Irrigation\_SW, Total\_GW, Total\_SW.
5. **Computes consumptive use (CU):**
   ```
   CU = max(Irrigation_ET − Effective_Precip, 0)
   ```
   Split into Irrigation\_CU, Irrigation\_GW\_CU, Irrigation\_SW\_CU using
   the GW fraction.
6. **Computes irrigation efficiency (IE):**
   ```
   IE = CU / Withdrawal
   ```
   Producing Irrigation\_Efficiency, Irrigation\_GW\_Efficiency,
   Irrigation\_SW\_Efficiency.
7. **Writes rasters** in four units for depth/volume products and as
   dimensionless ratios for IE:

| Product | Units written | File naming |
|---|---|---|
| Total pumping | mm, ft, m³, AF | `Predicted_GW_{year}_{unit}.tif` |
| 8 withdrawal categories | mm, ft, m³, AF | `{Category}_{year}_{unit}.tif` |
| 3 CU categories | mm, ft, m³, AF | `{CU_Category}_{year}_{unit}.tif` |
| 3 IE categories | dimensionless | `{IE_Category}_{year}.tif` |
| OOD flags | binary (0/1) | `OOD_Flag_{year}.tif` |

8. **Accumulates statistics** for AZ-wide, per-basin, and per-sub-basin
   totals (volume in m³ and AF, mean depth in mm) for every category.

Unit conversions:
- Pixel area: 2000² = 4 000 000 m²
- mm → m³: mm × pixel\_area\_m² / 1000
- m³ → AF: m³ / 1233.48
- mm → ft: mm / 304.8

#### 3c. Era summary maps

After the prediction loop, the pipeline generates era summary maps for
every depth/volume product.  All time-series plots (AZ-wide, per-basin,
per-sub-basin) are **deferred to the UQ step** (§3d) so that they include
95 % confidence intervals derived from the augmented 6-band rasters.

| Plot type | Function | Applied to |
|---|---|---|
| Era summary maps | `vizops.create_era_summary_maps()` | Total, 8 categories, 3 CU |

Four temporal eras are distinguished in the plots:

| Era | Years | Description |
|---|---|---|
| Hindcast | 1896–1983 | Pre-metered; predictions only. |
| Historical | 1984–2024 | Metered period; predictions vs. actuals. |
| Forecast | 2025 | Transition year. |
| Projected | 2026–2099 | Future projections. |

#### 3f. Graphical abstract / Figure 1

`vizops.create_graphical_abstract()` produces a two-panel publication figure:

- **Panel (a)**: Spatial map of mean-annual predicted pumping depth (mm)
  across all 204 years (1896–2099), with GW basin boundaries and AMA/INA
  labels overlaid on a YlOrRd colour ramp.
- **Panel (b)**: Time series of total annual AMA/INA pumping (acre-ft) with
  era shading and an inset bar chart of era-averaged volumes.

Saved as `{prediction_dir}Graphical_Abstract_Fig1.png` (600 dpi).

#### 3d. Hybrid uncertainty quantification (`uncertaintyops.run_uncertainty_quantification()`)

Pixel-level uncertainty is quantified for every year (1896–2099) by
computing five independent error components and combining them via
quadrature:

$$\sigma_{\text{total}} = \sqrt{\sigma_{\text{MACA}}^2 + \sigma_{\text{model}}^2 + \sigma_{\text{irr}}^2 + \sigma_{\text{LULC}}^2 + \sigma_{\text{gw}}^2}$$

Each component isolates a specific source of prediction uncertainty.
Every ensemble member's total prediction is also partitioned into the 8
withdrawal categories *before* computing std, yielding per-category σ for
each component.  Per-category σ_total is then obtained by the same
quadrature formula applied category-wise.

**Sample-based vs scenario-based components.**  σ_model (10 random seeds)
and σ_gw (4 Hung et al. temporal snapshots) are *sample-based*: their
ensemble members are random draws from a larger population, so
Student's t-distribution critical values are used instead of z = 1.96
to account for small-N estimation uncertainty
(t₉ = 2.262 for σ_model, t₃ = 3.182 for σ_gw).
The t-correction is applied by inflating σ by t/z *before* quadrature,
so all downstream CI computation uses a single multiplier (z = 1.96).
σ_MACA, σ_LULC, and σ_irr are *scenario-based*: their spread bounds
structural uncertainty rather than estimating population variance, so
they retain z = 1.96 (scale = 1.0).  Ensemble sizes (N) are reported
in all σ_total CSVs.

##### σ_MACA — Inter-GCM climate spread (future only, 2026–2099)

Five representative GCMs spanning the Southwest US climate space are
selected following Rupp et al. (2013):

| GCM | Climate archetype |
|-----|-------------------|
| CCSM4 | Central / median |
| CNRM-CM5 | Cool-wet |
| HadGEM2-ES365 | Hot-dry |
| MIROC-ESM-CHEM | Hot-wet |
| inmcm4 | Cool-dry |

For each future year, per-GCM predictor rasters are downloaded from GEE
(using the same `download_gee_data()` pipeline with `gcm=<model>`),
mosaicked, and stored in `GEE_Mosaics_{res}m_{GCM}/`.  The six
MACA-derived climate columns (ET, ETo, precip, Peff, Tmax, Tmin) from each
GCM's predictor raster replace the ensemble values in the year DataFrame,
the XGBoost model predicts total pumping, and σ_MACA is the per-pixel
sample standard deviation across the 5 predictions:

$$\sigma_{\text{MACA}}(x, y, t) = \text{std}\bigl[\hat{y}_{\text{GCM}_1}, \ldots, \hat{y}_{\text{GCM}_5}\bigr]$$

For historical years (1896–2025), σ_MACA = 0 because observations replace
GCM projections.

**Climate input spread diagnostics.**  During the σ_MACA loop, the AZ-mean
values of ET, ETo, and Peff are recorded for each GCM and year.  A 3-panel
ribbon plot (`Climate_Input_Spread.png`) and per-variable CSVs are saved
to `Sigma_MACA/Climate_Input_Spread/`, showing how the raw climate inputs
diverge across the 5 GCMs before they propagate through the XGBoost model.

##### σ_model — XGBoost seed ensemble (all years, 1896–2099)

Ten XGBoost models are trained on the full metered dataset (1984–2024) with
identical Optuna-tuned hyperparameters but different random seeds:

```
Seeds: 42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144
```

Training is parallelised across Dask workers (100 Optuna trials per seed).
For each year and pixel, σ_model is the sample standard deviation of the
10 seed predictions:

$$\sigma_{\text{model}}(x, y, t) = \text{std}\bigl[\hat{y}_{s_1}, \ldots, \hat{y}_{s_{10}}\bigr]$$

This captures the sensitivity of the model to stochastic training choices
(e.g., feature subsampling, tree-building order) at every pixel and year.

##### σ_irr — Irrigation fraction spread (historical only, 1896–2025)

Two independent estimates of irrigated area fraction are available:

1. **IrrMapper-based** (`annual_irr_fraction`, band 14) — the primary
   predictor used in the model.  Uses binary irrigated-area maps from
   IrrMapper RF v1.2 (1985–2025).
2. **Regression-based** — a simple linear regression predicting
   `irr_fraction ~ crop_fraction` trained on the metered period.

σ_irr is computed in two stages:

1. **Residual RMSE** of the regression on training data measures the
   typical discrepancy between the two fractions.
2. **Finite-difference sensitivity** — the XGBoost model is evaluated at
   `irr_frac ± δ` (where δ = regression RMSE), and σ_irr is taken as
   `|pred_plus − pred_minus| / 2`.

This captures how prediction uncertainty propagates through the model's
sensitivity to the irrigation fraction input:

$$\sigma_{\text{irr}}(x, y, t) = \frac{\bigl|\hat{y}(f_{\text{irr}} + \delta) - \hat{y}(f_{\text{irr}} - \delta)\bigr|}{2}$$

For future years (2026–2099) σ_irr = 0 because the full LULC → crop_frac
→ irr_frac chain is re-derived per scenario in σ_LULC, avoiding
double-counting.

##### σ_LULC — LULC projection spread (future only, 2026–2099)

The USGS has published four LULC projection scenarios for CONUS:

| Scenario | Description |
|----------|-------------|
| B1 | Low growth, high environmental emphasis |
| B2 | Low growth, low environmental emphasis |
| A1B | High growth, balanced |
| A2 | High growth, low environmental emphasis |

The primary LULC predictor (band 8) uses the **mode** across all four
scenarios.  σ_LULC captures the spread when individual scenarios are used
instead.  For each of the 4 scenarios:

1. Per-scenario GEE tiles are downloaded via
   `download_gee_data(lulc_scenario=scenario)`.
2. The LULC class, crop fraction, and irrigation fraction are re-derived
   end-to-end (LULC → AGRI/URBAN via `gwops.create_land_use_data()` →
   `crop_frac` → `irr_frac` via the regression model).
3. The XGBoost model predicts total pumping under each scenario.

σ_LULC is the sample standard deviation across the 4 scenario predictions:

$$\sigma_{\text{LULC}}(x, y, t) = \text{std}\bigl[\hat{y}_{B1}, \hat{y}_{B2}, \hat{y}_{A1B}, \hat{y}_{A2}\bigr]$$

Because σ_LULC re-derives the entire irrigation-fraction chain per
scenario, it fully subsumes σ_irr for future years.

##### σ_gw — GW fraction inter-snapshot spread (all years, 1896–2099)

USGS groundwater irrigation fraction data are available at four snapshots
(2000, 2005, 2010, 2015).  The primary model uses interpolated/extended
values via band 12.  σ_gw evaluates the model at each of the four snapshot
fractions and takes the per-pixel standard deviation:

$$\sigma_{\text{gw}}(x, y, t) = \text{std}\bigl[\hat{y}_{2000}, \hat{y}_{2005}, \hat{y}_{2010}, \hat{y}_{2015}\bigr]$$

This quantifies how sensitive the prediction is to the choice of GW
fraction snapshot at each pixel.

**Known limitation — σ_gw is a lower bound:** Only 4 snapshot years
(2000, 2005, 2010, 2015) with 5–15 year gaps are available, so decadal
oscillations (droughts, extreme-wet periods) between snapshots are
invisible to the ensemble.  The resulting σ_gw underestimates the true
temporal variability of GW sourcing fractions.

**GW fraction sensitivity analysis:** A standalone sensitivity test
(`run_gw_fraction_sensitivity`) perturbs `gw_fraction` by ±0.2 for all
years (1896–2099) and reports the resulting per-category volume changes
(AF) to bound the impact of freezing gw_fraction at snapshot values.
Years < 2005 (all frozen at the 2000 snapshot) and years ≥ 2015 (all
frozen at the 2015 snapshot) are the most affected.  Results are written
to `Uncertainty/Sigma_GW/GW_Fraction_Sensitivity.csv`.

##### σ_total — Quadrature combination

The five components are assumed independent and combined in quadrature.
Sample-based σ (Model, GW) are scaled by t/z before squaring, so the
resulting σ_total already incorporates the t-correction.

**Independence assumption.**  The quadrature formula assumes all five
components are mutually uncorrelated.  In practice, σ_MACA and σ_LULC
both perturb future-year predictors and may share structural correlations
(e.g. a hot-dry climate scenario also affects land use).  The combined
σ_total should therefore be interpreted as an approximate bound; true
combined uncertainty could be modestly larger or smaller depending on
the sign of inter-component correlations.

For each year the output is a 2-band raster:

| Band | Description |
|------|-------------|
| 1 | σ_total (mm) |
| 2 | CV = σ_total / \|prediction\| |

A temporal average `Mean_CV.tif` is also computed across all years.

**Outputs:**

```
Full_Prediction_XGB/Uncertainty/
├── Sigma_MACA/
│   ├── Rasters/Sigma_MACA_mm_{year}.tif
│   ├── Climate_Input_Spread/
│   │   ├── Climate_Input_Spread.png         # 3-panel ribbon plot (ET, ETo, Peff)
│   │   ├── annual_et_ensemble_mm.csv        # Per-GCM AZ-mean ET by year
│   │   ├── annual_eto_mm.csv                # Per-GCM AZ-mean ETo by year
│   │   └── annual_peff_mm.csv               # Per-GCM AZ-mean Peff by year
│   ├── Basin_Sigma_MACA.csv
│   ├── Subbasin_Sigma_MACA.csv
│   └── Uncertainty_Summary_MACA.csv
├── Sigma_Model/
│   ├── Rasters/Sigma_Model_mm_{year}.tif
│   ├── Basin_Sigma_Model.csv
│   ├── Subbasin_Sigma_Model.csv
│   └── Uncertainty_Summary_Model.csv
├── Sigma_Irr/
│   ├── Rasters/Sigma_Irr_mm_{year}.tif
│   ├── Basin_Sigma_Irr.csv
│   ├── Subbasin_Sigma_Irr.csv
│   └── Uncertainty_Summary_Irr.csv
├── Sigma_LULC/
│   ├── Rasters/Sigma_LULC_mm_{year}.tif
│   ├── Basin_Sigma_LULC.csv
│   ├── Subbasin_Sigma_LULC.csv
│   └── Uncertainty_Summary_LULC.csv
├── Sigma_GW/
│   ├── Rasters/Sigma_GW_mm_{year}.tif
│   ├── Basin_Sigma_GW.csv
│   ├── Subbasin_Sigma_GW.csv
│   └── Uncertainty_Summary_GW.csv
├── Sigma_Total/
│   ├── Rasters/Sigma_Total_mm_{year}.tif   (2-band: σ, CV)
│   ├── Mean_CV.tif
│   ├── Basin_Sigma_Total.csv
│   ├── Subbasin_Sigma_Total.csv
│   └── Uncertainty_Summary_Total.csv
├── Sigma_CU/
│   ├── Rasters/Sigma_{CU_cat}_mm_{year}.tif
│   └── Uncertainty_Summary_CU.csv
└── Plots/
    ├── {Component}_time_series.png
    ├── Combined_uncertainty_time_series.png
    └── Basin_Sigma/
        ├── Basin_{region}_Sigma.png                  # per-basin: mean±CI + σ_total (twinx m³/AF)
        ├── Subbasin_{region}_Sigma.png               # per-sub-basin
        ├── Basin_All_Sigma_Summary.png               # all basins overlaid
        ├── Subbasin_All_Sigma_Summary.png            # all sub-basins overlaid
        ├── Basin_{region}_Sigma_{Component}.png      # per-basin per-component σ
        ├── Subbasin_{region}_Sigma_{Component}.png   # per-sub-basin per-component σ
        ├── Basin_All_Sigma_{Component}_Summary.png   # all basins overlaid per-component
        └── Subbasin_All_Sigma_{Component}_Summary.png
```

##### σ_CU — Consumptive-use inter-GCM spread (future only, 2026–2099)

Consumptive use is defined as CU = max(ET_irr − Peff_irr, 0), which
depends on ET and Peff — both climate-model-dependent quantities.  σ_CU
captures how CU varies across the 5 representative GCMs:

1. For each GCM, per-GCM ET (band 1) and Peff (band 4) are read from the
   per-GCM predictor rasters (already built during σ_MACA).
2. Ensemble `irr_frac` and `gw_frac` (from the ensemble predictor) are
   applied to derive per-GCM `CU = max(ET × irr_frac − Peff × irr_frac, 0)`.
3. CU is split into CU_GW (`CU × gw_frac`) and CU_SW (`CU − CU_GW`).
4. σ_CU is the per-pixel sample standard deviation across the 5 GCMs for
   each CU category (Irrigation_CU, Irrigation_GW_CU, Irrigation_SW_CU).

For historical years (1896–2025), σ_CU = 0 because ET and Peff are
observation-derived.

##### Augmented prediction rasters (6-band)

After σ_total and σ_CU are computed, all prediction rasters are
**augmented in-place** from 1-band to 6-band GeoTIFFs:

| Band | Description |
|------|-------------|
| 1 | Prediction (original units) |
| 2 | σ (uncertainty in same units) |
| 3 | CV = σ / \|prediction\| (dimensionless) |
| 4 | SNR = \|prediction\| / σ (dimensionless) |
| 5 | Lower 95% CI = prediction − 1.96·σ |
| 6 | Upper 95% CI = prediction + 1.96·σ |

This augmentation is applied to:

| Product | σ source | Units augmented | Details |
|---------|----------|----------------|---------|
| **Total pumping** (32 rasters/yr) | σ_total × unit scale | mm, ft, m³, AF | σ_total computed in mm, scaled by conversion factor per unit |
| **8 withdrawal categories** (256 rasters/yr) | σ_cat via quadrature of per-category ensemble spreads | mm, ft, m³, AF | Each ensemble member is partitioned before computing std |
| **3 CU categories** (48 rasters/yr) | σ_CU (inter-GCM spread) | mm, ft, m³, AF | σ_CU in mm, scaled to target unit |
| **3 IE categories** (12 rasters/yr) | Ratio error propagation | dimensionless | $\sigma_{\text{IE}} = \text{IE} \times \sqrt{\text{CV}_{\text{CU}}^2 + \text{CV}_{\text{wd}}^2}$ |

**Unit conversion for σ:** σ_total is natively in mm.  For other units
the same conversion factors as the predictions are applied:

| Target unit | Scale factor |
|-------------|-------------|
| mm | 1.0 |
| ft | 1 / 304.8 |
| m³ | pixel_area_m² / 1000 = 4,000,000 / 1000 = 4000 |
| AF | 4000 / 1233.48 ≈ 3.2428 |

**Category uncertainty:** Each UQ ensemble function partitions every
member’s total prediction into the 8 withdrawal categories *before*
computing std.  This produces per-category σ for each component (MACA,
Model, Irr, LULC, GW), which are then combined via quadrature:

$$\sigma_{\text{cat}} = \sqrt{\sum_i \sigma_{\text{cat},i}^2}$$

where $\sigma_{\text{cat},i}$ is the per-category std from the $i$-th
UQ component’s ensemble.  This correctly propagates partition-fraction
uncertainty (σ_irr and σ_gw perturb `irr_fraction` and `gw_fraction`
respectively, so per-category spreads differ from simple linear scaling
of the total σ).

**IE uncertainty:** IE = CU / withdrawal is a ratio of two uncertain
quantities.  Standard ratio error propagation gives:

$$\frac{\sigma_{\text{IE}}}{\text{IE}} = \sqrt{\left(\frac{\sigma_{\text{CU}}}{\text{CU}}\right)^2 + \left(\frac{\sigma_{\text{wd}}}{\text{wd}}\right)^2} = \sqrt{\text{CV}_{\text{CU}}^2 + \text{CV}_{\text{wd}}^2}$$

CV_CU and CV_wd are read from band 3 of the already-augmented CU and
withdrawal category rasters respectively.

**Execution order** (dependencies require sequential processing):

1. Compute σ_total → augment total prediction rasters (all 4 units)
2. Augment category rasters (reads augmented total rasters for σ)
3. Compute σ_CU → augment CU rasters
4. Augment IE rasters (reads augmented CU + category rasters for CV)

##### Basin / sub-basin scale uncertainty

Pixel-level σ values cannot be naively summed to obtain basin-scale σ
because spatial correlations between pixels are non-negligible (shared
GCM forcings, model parameters, and land-use projections).  Instead, the
**aggregate-then-spread** approach is used:

1. For each σ component, every ensemble member's per-pixel prediction
   (mm) is summed within each groundwater basin and sub-basin to obtain
   the member's total volume (AF).
2. Basin-scale σ is the sample standard deviation of these member volumes.

This preserves intra-basin spatial correlation because each member volume
already reflects the correlated pixel-level response.

Each `compute_sigma_*` function writes per-component CSVs:

| File | Contents |
|------|----------|
| `{Sigma_X}/Basin_Sigma_{X}.csv` | Per-basin, per-year σ (m³ and AF) for component X |
| `{Sigma_X}/Subbasin_Sigma_{X}.csv` | Per-sub-basin, per-year σ (m³ and AF) |

CSV columns: `Year, Region, Mean_Volume_m3, Sigma_Volume_m3,
Mean_Volume_AF, Sigma_Volume_AF, CV, Lower_95CI_m3, Upper_95CI_m3,
Lower_95CI_AF, Upper_95CI_AF, N_Members`.

After all components are computed, `compute_basin_sigma_total` combines
them via quadrature at the basin/sub-basin level:

$$\sigma_{\text{total,basin}} = \sqrt{\sum_i \sigma_{i,\text{basin}}^2}$$

and writes `Basin_Sigma_Total.csv` / `Subbasin_Sigma_Total.csv` into
`Sigma_Total/`.  These include per-component σ columns (in both m³ and
AF) alongside the combined total, enabling direct attribution of
basin-scale uncertainty to individual sources.

Per-region time-series plots are generated in `Plots/Basin_Sigma/` with
dual y-axes (m³ on the left, AF on the right):

- **Per-region PNGs** — Two panels: (1) mean prediction ± 95 % CI,
  (2) σ_total time series, both with era shading and twinx.
- **Summary PNGs** — All regions overlaid on a single plot for σ_total
  and CV, with twinx for σ.

##### Per-component basin / sub-basin σ time series

In addition to the combined σ_total basin plots, per-component σ time
series are generated for each of the five uncertainty components (MACA,
Model, Irr, LULC, GW).  For each component,
`_plot_component_basin_sigma()` reads the component's
`Basin_Sigma_{X}.csv` and `Subbasin_Sigma_{X}.csv` and produces:

- **Per-region plots** — Two panels: (1) mean volume ± 95 % CI,
  (2) component σ — both with twinx for m³ ↔ AF, era shading.
- **Summary overlays** — All basins (or sub-basins) overlaid for σ and
  CV.

These plots are saved alongside the σ_total plots in
`Plots/Basin_Sigma/` with the component name in the filename.

##### Uncertainty-bounded time series from augmented rasters

After all rasters have been augmented to 6 bands, the UQ step
regenerates **every** time-series plot (AZ-wide, per-basin, per-sub-basin)
directly from the augmented rasters using zonal statistics.  This replaces
the time-series plots that were previously generated in Step 3c without
uncertainty bounds.

`_replot_from_augmented_rasters()` performs the following for each of the
15 product groups (total, 8 categories, 3 CU, 3 IE):

1. **AZ-wide statistics** (`_az_wide_stats`) — Reads bands 1/2/5/6 from
   each year's 6-band raster to compute mean depth, total volume (m³/AF),
   σ_depth, and σ_volume.  Volume-level σ is derived from the CI bands:
   $$\sigma_V = \frac{|\Sigma_{\text{upper CI}} - \Sigma_{\text{lower CI}}|}{2 \times 1.96}$$
2. **Zonal statistics** (`_zone_stats`) — Clips each raster to every
   basin and sub-basin polygon (via `rasterio.mask`) and computes
   per-zone prediction, σ, and CV.
3. **Observed actuals** — Where a pre-existing time-series CSV from
   Step 3c contains observed data (metered period 1984–2024), actuals are
   extracted and overlaid on the predicted time series.
4. **Plot generation** — Calls `vizops.create_full_period_time_series()`,
   `vizops.create_basin_time_series()`, and
   `vizops.create_subbasin_time_series()` with `sigma_data`,
   `sigma_basin_yearly`, and `sigma_subbasin_yearly` arguments to render
   95 % CI shading on all time-series plots.

For the 3 IE products, `_process_ie_group()` uses the dimensionless
efficiency values directly (mean ± 1.96 σ) rather than volume conversions.

All plots are written to the `Visualizations/` directory, overwriting any
earlier plots from Step 3c that lacked uncertainty bounds.

#### 3g. Raster maps for all output categories (`create_all_raster_maps()`)

Generates publication-quality spatial maps for **every** raster output
product in the pipeline.  Three types of maps are produced:

**Era-mean maps** (`vizops.create_era_raster_maps()`) — A 2×2 panel figure
for each raster category showing the temporal mean within each of the four
eras (Hindcast, Historical, Forecast, Projection).  Groundwater basin
boundaries (thin gray) and AMA/INA basins (bold dark + labels) are overlaid
on every panel.  No-data pixels appear as gray background.

| Category group | Colormap | Count |
|---|---|---|
| Total predicted GW + 8 partition categories + 3 CU | `YlOrRd` | 12 figures |
| 3 Irrigation Efficiency categories | `YlGn` | 3 figures |
| OOD flags (mean fraction) | `RdYlGn_r` | 1 figure |
| 6 sigma components — band 1 (σ in mm) | `Purples` | 6 figures |
| 6 sigma components — band 2 (CV) | `inferno` | 6 figures |
| Augmented prediction CV (band 3) and SNR (band 4) | `inferno` / `viridis` | 2 figures |

**Actual vs predicted comparison** (`vizops.create_actual_vs_predicted_maps()`)
— A 1×3 panel figure comparing the metered (1984–2024) mean with the
predicted mean over the same period:

- Panel (a): **Actual** (meter-based GW raster mean).  Unmetered areas
  outside AMA/INA appear as gray no-data.
- Panel (b): **Predicted** (ML raster mean).
- Panel (c): **Difference** (Predicted − Actual) with a diverging `RdBu_r`
  colormap centred on zero.

**Trend analysis** (`vizops.create_trend_maps()`) — Pixel-wise
Mann-Kendall trend test (via `scipy.stats.kendalltau`) and Sen's slope
(via `scipy.stats.theilslopes`) for each period (full 1896–2099 and
per-era).  Each figure shows:

- Sen's slope (unit/year) with a diverging `RdBu_r` colormap (blue =
  decreasing, red = increasing).
- Gray stippling on pixels where the Mann-Kendall test is **not
  significant** (p ≥ 0.05), so statistically significant trends appear
  clean.
- Inset text showing the percentage of domain pixels with significant
  increasing (↑), decreasing (↓), and non-significant trends.
- Basin boundaries and AMA/INA labels overlaid.

Trend maps are generated for: total predicted GW, 8 partition categories,
3 CU categories, and 3 IE categories — each with up to 5 periods (full +
4 eras), yielding ~60–75 trend figures.

**Zonal trend statistics** — For each category × period, per-basin and
per-sub-basin CSV files are written alongside the trend maps.  Each CSV
contains one row per zone with columns: `Category`, `Period`, `Region`,
`N_Pixels`, `Median_Slope`, `Mean_Slope`, `Mean_Slope_Sig`,
`Pct_Sig_Increase`, `Pct_Sig_Decrease`, `Pct_Not_Sig`, `P10_Slope`,
`P90_Slope`, `Median_P_Value`.  Basin zones are rasterized from the
groundwater basin shapefile; sub-basin zones from the ADWR sub-basin
shapefile.

All outputs are saved to `{prediction_dir}Raster_Maps/` (era maps and
actual vs predicted) and `{prediction_dir}Raster_Maps/Trend_Analysis/`
(trend maps and zonal statistics CSVs).

#### 3e. Well package

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
| USGS NHM | HUC12 polygons (Mgal/d) | [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM); [Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ) |
| USGS Reitz | 800 m rasters (m/yr) | [Reitz et al., 2023](https://doi.org/10.5066/P9EZ3VAS) |

Because the three products live at different native resolutions, each is
aggregated to Arizona groundwater basin totals (volume in AF) for
comparison.  The intercomparison produces:

- Pairwise metrics (RMSD, MAD, Percent Difference) for ML vs NHM, ML vs
  Reitz, NHM vs Reitz in both GW and SW categories.
- Temporal agreement metrics (per-basin Pearson r and NSE) quantifying
  interannual variability agreement across overlapping years.
- Per-basin comparison tables (mm, ft, m³, AF).
- Time series CSVs and per-basin time series plots.
- Pairwise scatter plots with 1:1 lines and linear fits.
- Spatial difference maps (diverging colourmap centred on zero).
- Temporal agreement visualizations (`Temporal_Agreement/`):
  - **Heatmaps** — Basin × pair grids coloured by Pearson r and NSE.
  - **Box/violin plots** — Distribution of per-basin r/NSE across pairs.
  - **Taylor diagrams** — Correlation vs normalised std dev in polar
    coordinates, with centred RMSD contours.
  - **r vs NSE scatter** — Paired scatter with quadrant annotations
    identifying basins with good/mixed/poor agreement.

All outputs are written to `{prediction_dir}Intercomparison/`.

#### Step 4b — CU / IE intercomparison (`run_cu_ie_usgs_intercomparison()`)

Compares ML-based Irrigation Consumptive Use and Irrigation Efficiency with
USGS NHM HUC12-scale data ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)) at the basin scale:

| Product | ML source | USGS source |
|---|---|---|
| **CU** | `Irrigation_CU_Rasters/Depth_mm/` (mm) | `Irr_CU_HUC12_Tot_annual_2000_2020.csv` (Mgal/d) |
| **IE** | `Irrigation_Efficiency_Rasters/` (ratio) | `IR_HUC12_Eff_annual_2000_2020.csv` (ratio) |

CU follows the same volume-based framework as withdrawals (RMSD, MAD, %
Difference in AF, m³, mm).  IE uses dimensionless ratio metrics.  Outputs
include metrics CSVs, per-basin tables, time series, and scatter plots,
written to `{prediction_dir}CU_IE_Intercomparison/`.

#### Step 4c — CAP/SRP surface-water validation (`run_cap_srp_sw_validation()`)

Validates ML `Total_SW` predictions at the basin scale against observed
delivery records from the Central Arizona Project (CAP) and Salt River
Project (SRP). CAP delivery data were obtained from CAP via a public data
request; SRP delivery data were obtained from ADWR.

| Source | Basins covered | Years | Filter |
|---|---|---|---|
| **CAP** | Phoenix, Tucson, Pinal AMA; Harquahala INA; Ranegras Plain; Parker | 1985–2024 | `Recharge Facility` is null (direct use only) |
| **SRP** | Phoenix AMA only | 1984–2023 | `Parent Water Type == 'SURFACE WATER'` |

For Phoenix AMA the CAP and SRP totals are summed; all other basins use
CAP data only.

**Data exclusions and caveats:**

- **CAP "Multiple" AMA records** (25 rows, ~15,600 AF total) and **NaN-AMA
  records** (16 rows, ~86,300 AF) are excluded because they cannot be
  attributed to a single groundwater basin.
- **Recharge filtering**: CAP rows where `Recharge Facility` is non-null are
  excluded.  This removes managed aquifer recharge deliveries so that only
  direct surface-water use is compared against the ML `Total_SW` estimate.
  Some direct-use deliveries may be partially classified as recharge, so
  the observed series is a conservative lower bound.
- **Non-CAP/non-SRP surface water**: Surface-water sources not captured by
  CAP or SRP (e.g., local canal diversions, non-SRP irrigation districts)
  are not represented in the observed comparison series.  The ML predictions
  encompass all surface water, so the observed total is expected to be
  lower than the ML estimate, particularly for basins outside the Phoenix
  AMA.
- **Temporal alignment**: Both datasets use calendar years (CAP `Year`,
  SRP `Water Move Year`).  The common overlap period is 1985–2023.
- **Spill water sensitivity**: SRP also reports ``SPILL WATER`` deliveries
  (19–366,000 AF/yr, highly variable).  A sensitivity series that includes
  spill water alongside surface water is plotted as a third time series for
  comparison.

Outputs include per-basin time series plots (with spill-water sensitivity
overlay), a scatter plot of ML vs observed AF per basin-year with 1:1 line
and R², metrics CSV, and time series CSV, written to
`{prediction_dir}CAP_SRP_Validation/`.

#### Step 4d — Effective precipitation intercomparison (`run_peff_usgs_intercomparison()`)

Compares irrigated effective precipitation across three sources:

| Source | Description | Years | Resolution |
|---|---|---|---|
| **ML Peff (SCS)** | Predictor band 4 × `irr_fraction` | 2000–2024 | 2 km rasters |
| **ML Peff (PCML)** | Predictor band 5 × `irr_fraction` | 2000–2023 | 2 km rasters |
| **NHM PPTeff** | USGS NHM HUC12 data ([Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)) (Mgal/d) | 2000–2020 | HUC12 polygons |

All three datasets are scaled by `annual_irr_fraction` so that volumes
represent only the irrigated-area contribution.  NHM PPTeff follows the
same CSV → rasterise → basin-aggregate pipeline as NHM CU, with irrigated-
area scaling for the volume-to-depth conversion.

The intercomparison produces:
- Pairwise metrics (RMSD, MAD, Percent Difference) in AF, m³, and mm.
- Per-basin comparison table.
- Time series CSVs and per-basin time series plots (depth and volume).
- Pairwise scatter plots with 1:1 lines and linear fits.

All outputs are written to `{prediction_dir}Peff_Intercomparison/`.

#### Step 4e — Non-irrigation vs USGS Public Supply (`run_ps_intercomparison()`)

Compares ML non-irrigation withdrawal predictions with USGS Public Supply
(PS) reanalysis data (Alzraiee et al. 2024, *Water Resources Research*;
DOI: 10.5066/P9FUL880).  The PS dataset provides monthly public supply
GW and SW withdrawals by HUC12 for the CONUS (2000–2020).

Public supply is a **subset** of total non-irrigation water use (it
excludes industrial, mining, thermoelectric, and livestock self-supply).
Therefore ML Non_Irrigation predictions should be ≥ PS estimates in most
basins.  The PS / ML ratio quantifies the fraction of non-irrigation use
attributable to public supply — an independently derived quantity.

Three categories are compared at the basin level:

| ML category | PS category | Expected relationship |
|---|---|---|
| Non_Irrigation | PS Total | ML ≥ PS |
| Non_Irrigation_GW | PS GW | ML ≥ PS (validates GW partition) |
| Non_Irrigation_SW | PS SW | ML ≥ PS (validates SW partition) |

**Outputs:**
- Metrics CSV with RMSD, MAD, and percent difference per category.
- Per-basin comparison table with ML and PS volumes (AF, mm) and the
  PS-as-fraction-of-ML percentage.
- Temporal agreement metrics (Pearson r, NSE) per basin.
- Time series plots and CSVs per basin and category.
- Temporal agreement visualisation (heatmap, box/violin, r-vs-NSE).

All outputs are written to `{prediction_dir}PS_Intercomparison/`.

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
- **`create_az_data_parquet()`** — Reads all years' predictor rasters and
  stacks them with basin labels and observed pumping into a single DataFrame.
- **`create_train_test_data()`** — Splits the DataFrame into train/test
  sets using one of four strategies (temporal, spatial, random ratio,
  random 80/20).

### `mlops.py` — Machine learning operations

Builds, tunes, evaluates, and interprets ML models including baseline linear
regressors (LR, Ridge, Lasso) and ensemble tree models.

Key functions:
- **`get_model_param_dict()`** — Returns model objects and hyperparameter
  search spaces for LR, Ridge, Lasso, XGB, LGBM, RF, ETR, HGBR, CatBoost,
  GBR, and AdaBoost.
- **`build_ml_model_optuna_dask()`** — Trains a single model with Optuna
  TPE-based hyperparameter search parallelised across Dask workers.
- **`compare_all_models()`** — Trains all models on a common split and
  ranks them by test R².
- **`get_prediction_results()`** — Makes predictions and applies multi-level
  bias correction.
- **`perform_bias_correction()`** — Applies basin-level or global bias
  correction using linear scaling.
- **`calc_train_test_metrics()`** — Computes R², normalised RMSE (% of σ),
  normalised MAE (% of σ), and normalised MBE (% of mean).
- **`compute_perm_imp()`**, **`compute_ale_plots()`**,
  **`compute_shap_plots()`** — Model interpretability diagnostics.
- **`generate_model_visualizations()`** — Scatter, residual, and time series
  plots per model.
- **`OODDetector`** — Mahalanobis distance-based out-of-distribution
  detector.  Fitted on training features, it flags prediction-time pixels
  whose feature vectors exceed the χ²(n\_features) threshold at α = 0.01.
  Used in `predict_full_period()` to write per-year OOD flag rasters and
  a summary CSV with era-level OOD rate diagnostics.

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
  Temporal LOO, and Spatial LOO results with R², RMSE, MAE, MBE, and
  Overfit R².  Produces CSV, LaTeX (`booktabs`), and grouped bar chart.
- **`create_graphical_abstract()`** — Two-panel Figure 1: (a) mean-annual
  pumping depth map with GW basin boundaries, (b) annual pumping time series
  with era shading and inset bar chart.
- **`create_era_raster_maps()`** — 2×2 panel era-mean spatial maps with
  basin/AMA/INA overlays and shared colorbar.  Supports multi-band rasters
  (e.g., band 2 for CV in sigma rasters).
- **`create_actual_vs_predicted_maps()`** — 3-panel actual vs predicted
  comparison with gray no-data for unmetered areas and diverging difference
  colormap.
- **`create_trend_maps()`** — Pixel-wise Mann-Kendall + Sen's slope trend
  maps per period with significance stippling and summary statistics inset.

**Basin block bootstrap CI:** For AZ-wide time series sums,
`aggregate_yearly_data()` uses a basin-level block bootstrap that resamples
entire groundwater basins with replacement (via `_basin_block_bootstrap`),
preserving within-basin spatial autocorrelation and producing more
conservative CIs.  Per-basin plots fall back to pixel-level resampling
since data is already filtered to a single basin.

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
- **`create_land_use_data()`** — Gaussian-filters the LULC raster to
  produce continuous AGRI, SW, and URBAN density features.  **Known design
  choice:** The density features are independently min–max normalised to
  [0, 1] within each year, so the model sees only within-year spatial
  patterns, not temporal magnitude trends.  This is partially mitigated by
  `annual_crop_fraction` and `annual_irr_fraction`, which provide
  year-to-year magnitude signals.
- **`create_well_density_raster()`** — Creates a well-count-per-pixel
  raster from the Well Registry.  **Known limitation:** A single raster
  is computed from the current (2024) ADWR Well Registry and replicated
  for all years (1896–2099).  In reality, well density has changed over
  time — fewer wells existed in early hindcast years, and future
  projections may see wells retired, decommissioned, or newly drilled.
  Because the pipeline uses well density primarily as a spatial
  mask (pixels with zero wells are set to NaN in the partitioning step),
  the main effect is that hindcast years may include predictions at pixels
  where wells did not yet exist, while projection years may retain
  predictions at pixels where wells have been retired, or miss pixels
  where new wells are drilled.  If historical well drilling records or
  future well siting projections become available, time-varying well
  density rasters could be created.

### `streamflowops.py` — Streamflow & canal data

Downloads and processes streamflow data from USGS ([Hodson et al., 2023](https://doi.org/10.5066/P94I5TX3)) and USBR ([Gangopadhyay & Pruitt, 2011](https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf); [USBR, 2025](https://rise-usbr.opendata.arcgis.com/)) sources.

Key functions:
- **`download_streamflow()`** — Downloads monthly streamflow records from
  USGS gauges and retrieves USBR delivery data.
- **`create_streamflow_rasters()`** — Rasterises annual streamflow volumes
  onto the 2 km grid using watershed polygons.
- **`create_canal_density_raster()`** — Rasterises canal geometry from the
  GRAIN dataset ([Suresh et al., 2026](https://doi.org/10.5194/essd-18-1855-2026))
  into a canal-density layer used for SW-fraction estimation.

### `partitionops.py` — Water-budget partitioning

Decomposes total pumping predictions into eight withdrawal categories using
ancillary data already in the predictor stack:

| Category | Derivation |
|---|---|
| **Irrigation** | `total × irr_fraction` (USGS irrigation-fraction raster) |
| **Non_Irrigation** | `total − Irrigation` |
| **Irrigation_GW** | `Irrigation × gw_fraction` (USGS GW-fraction snapshots) |
| **Irrigation_SW** | `Irrigation − Irrigation_GW` |

**Known limitation (GW fraction frozen at 2015):** `annual_gw_fraction` uses
the 2015 USGS Hung et al. snapshot for all years ≥ 2015.  This freezes the
irrigation GW/SW partitioning for 84 years of projections regardless of
scenario-driven changes in water supply.  A GW-fraction sensitivity analysis
(`run_gw_fraction_sensitivity`, ±0.2 perturbation) quantifies the volume
impact of this assumption across all years (1896–2099); results are in
`Uncertainty/Sigma_GW/GW_Fraction_Sensitivity.csv`.  If future USGS
snapshots or scenario-driven GW fraction projections become available,
they can be integrated to make the partition dynamic.

| **Non_Irrigation_GW** | `Non_Irrigation × (1 − sw_fraction)` |
| **Non_Irrigation_SW** | `Non_Irrigation × sw_fraction` (canal-density proxy) |
| **Total_GW** | `Irrigation_GW + Non_Irrigation_GW` |
| **Total_SW** | `Irrigation_SW + Non_Irrigation_SW` |

**Known limitation (SW fraction proxy):** `compute_sw_fraction()` uses
canal density normalised by the local maximum as a proxy for the
surface-water share of non-irrigation withdrawals.  Canal density
(canal segments per pixel) is not a validated proxy for municipal or
industrial SW sourcing.  Where canal infrastructure is sparse, all
non-irrigation withdrawal is assigned to groundwater, which may
overestimate GW dependence in areas served by surface-water utilities.
When municipal water-delivery records or alternative data become
available, they can replace this proxy.

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

### `uncertaintyops.py` — Hybrid uncertainty quantification

Computes pixel-level prediction uncertainty for all products (total
pumping, withdrawal categories, consumptive use, irrigation efficiency)
and writes augmented 6-band GeoTIFFs.

Key functions:
- **`run_uncertainty_quantification()`** — Master orchestrator.  Computes
  all σ components, combines them via quadrature, writes σ rasters and
  summary CSVs, generates time-series plots, augments all prediction
  rasters with uncertainty bands, and regenerates all time-series plots
  with uncertainty bounds via zonal statistics.  Accepts `basin_shp` and
  `subbasin_shp` parameters for basin/sub-basin shapefiles.
- **`compute_sigma_maca()`** — Inter-GCM climate spread (5 GCMs, future
  only).  Returns per-year total σ, per-category σ, and per-GCM mosaic
  directories (reused by σ_CU).  Also records per-GCM AZ-mean values of
  ET, ETo, and Peff and generates input spread CSVs and a 3-panel ribbon
  plot (`Climate_Input_Spread/`).
- **`compute_sigma_model()`** — XGBoost 10-seed ensemble spread (all
  years).  Parallelised via Dask + Optuna.  Returns per-year total σ and
  per-category σ.
- **`compute_sigma_irr()`** — Irrigation fraction sensitivity (historical
  only, 1896–2025).  Uses IrrMapper vs regression finite-difference with
  half-range mode.  Returns per-year total σ and per-category σ.
- **`compute_sigma_lulc()`** — LULC projection spread (4 USGS scenarios,
  future only, 2026–2099).  Re-derives the full LULC → crop_frac →
  irr_frac chain per scenario.  Returns per-year total σ and per-category σ.
- **`compute_sigma_gw()`** — GW fraction snapshot spread (4 USGS
  snapshots, all years).  Returns per-year total σ and per-category σ.
- **`compute_sigma_total()`** — Quadrature combination of all five
  components for both total and per-category σ.  Writes 2-band total σ
  rasters (σ, CV), per-category σ rasters, and temporal Mean_CV.tif.
- **`compute_basin_sigma_total()`** — Reads per-component basin/sub-basin
  σ CSVs and combines via quadrature into `Basin_Sigma_Total.csv` /
  `Subbasin_Sigma_Total.csv`.
- **`compute_sigma_cu()`** — CU inter-GCM spread (5 GCMs, future only).
  Writes per-category σ_CU rasters (Irrigation_CU, Irrigation_GW_CU,
  Irrigation_SW_CU).  **Known limitation:** `irr_fraction` and
  `gw_fraction` are fixed from the ensemble-mean predictor raster when
  computing CU = max(ET_irr − Peff_irr, 0), so σ_CU captures only the
  inter-GCM ET/Peff spread and not the partitioning uncertainty already
  quantified in σ_LULC and σ_gw.
- **`run_gw_fraction_sensitivity()`** — Standalone sensitivity analysis
  perturbing `gw_fraction` by ±0.2 for all years (1896–2099) and
  reporting per-category volume changes.  Years < 2005 and ≥ 2015 are
  most affected since they are frozen at a single snapshot.  Writes
  `Sigma_GW/GW_Fraction_Sensitivity.csv`.
- **`augment_prediction_rasters()`** — Rewrites total pumping rasters as
  6-band GeoTIFFs (pred, σ, CV, SNR, lower CI, upper CI) for all 4 units.
- **`augment_category_rasters()`** — Augments 8 withdrawal category rasters
  using per-category σ_total rasters computed directly from ensemble
  spreads (not fraction-scaled from total σ).
- **`augment_cu_rasters()`** — Augments 3 CU category rasters using σ_CU.
- **`augment_ie_rasters()`** — Augments 3 IE rasters using ratio error
  propagation from augmented CU and withdrawal CV bands.
- **`_replot_from_augmented_rasters()`** — Regenerates all time-series
  plots (AZ-wide, per-basin, per-sub-basin) with 95 % CI uncertainty
  bounds by reading the 6-band augmented rasters via zonal statistics
  (`rasterio.mask`).  Replaces the earlier non-uncertainty time-series
  plots from Step 3c.
- **`_plot_component_basin_sigma()`** — Generates per-component (MACA,
  Model, Irr, LULC, GW) basin and sub-basin σ time-series plots with
  dual y-axes (m³/AF) and era shading.

### `intercompops.py` — USGS intercomparison

Basin-scale comparison of ML predictions with independent USGS datasets.

**Withdrawal intercomparison** (`run_intercomparison()`):
- Loads ML, NHM ([Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM); [Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)), and Reitz ([Reitz et al., 2023](https://doi.org/10.5066/P9EZ3VAS)) data; aggregates to basin volumes (AF).
- Computes pairwise RMSD, MAD, Percent Difference.
- Computes interannual temporal agreement (per-basin Pearson r and NSE).
- Produces per-basin time series, scatter plots, spatial difference maps,
  and temporal agreement visualizations (heatmaps, box/violin plots,
  Taylor diagrams, r-vs-NSE scatter).

**CU / IE intercomparison** (`run_cu_ie_intercomparison()`):
- Compares ML CU (mm) and IE (ratio) with NHM HUC12 annual data ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)).
- CU: Mgal/d → m³/yr → depth (mm) → basin volumes (AF).
- IE: dimensionless ratio → area-weighted basin means.
- Produces metrics, per-basin tables, time series, and scatter plots.

**CAP/SRP validation** (`run_cap_srp_validation()`):
- Compares ML Total SW predictions with observed CAP + SRP delivery records.
- Filters CAP to direct-use only; SRP to Surface Water (+ optional Spill
  Water sensitivity).
- Produces per-basin time series, scatter plots, and validation metrics.

**Peff intercomparison** (`run_peff_intercomparison()`):
- Compares ML Peff (SCS, band 4) and Peff PCML (band 5) with NHM PPTeff ([Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)).
- All three scaled by `irr_fraction` to represent irrigated-area Peff.
- NHM PPTeff: Mgal/d → m³/yr → depth (mm) → basin volumes (AF).
- Produces metrics, per-basin tables, time series, and scatter plots.

**Public Supply intercomparison** (`run_ps_intercomparison()`):
- Compares ML Non_Irrigation, Non_Irrigation_GW, and Non_Irrigation_SW
  predictions with USGS Public Supply reanalysis data
  ([Alzraiee et al., 2024](https://doi.org/10.1029/2023WR036632);
  data: [Luukkonen et al., 2023](https://doi.org/10.5066/P9FUL880)).
- PS monthly HUC12 data (Mgal/d, 2000–2020) → annual basin volumes (AF).
- Reports PS/ML ratio per basin (expected ≤ 100% since PS ⊂ non-irrigation).
- Produces metrics, per-basin tables, temporal agreement, and time series.

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
    │   ├── Spatial_LOO/                     # Step 2c results (per sub-basin)
    │   ├── Cross_Strategy_Summary.csv       # All models × all strategies
    │   ├── Cross_Strategy_Summary.tex       # LaTeX table for manuscripts
    │   └── Cross_Strategy_Comparison.png    # Grouped bar chart
    │
    └── Full_Prediction_XGB/
        ├── Model_Interpretability/          # SHAP, ALE, permutation importance
        │   ├── Hindcast/                    #   Era-specific plots (1896-1983)
        │   ├── Training/                    #   Era-specific plots (1984-2024)
        │   └── Projection/                  #   Era-specific plots (2025-2099)
        ├── Predicted_Rasters/               # Total pumping (4 units, 6-band)
        │   ├── Depth_mm/
        │   ├── Depth_ft/
        │   ├── Volume_m3/
        │   └── Volume_AF/
        ├── {Category}_Rasters/              # 8 withdrawal categories (4 units, 6-band)
        ├── Irrigation_CU_Rasters/           # CU (4 units, 6-band)
        ├── Irrigation_Efficiency_Rasters/   # IE (dimensionless, 6-band)
        ├── OOD_Rasters/                     # Out-of-distribution detection
        │   ├── OOD_Flag_{year}.tif          #   Binary flag (1=OOD, 0=in-distribution)
        │   └── OOD_Summary.csv              #   Per-year OOD statistics
        ├── Uncertainty/                     # Hybrid uncertainty quantification
        │   ├── Sigma_MACA/                  #   Inter-GCM climate spread
        │   ├── Sigma_Model/                 #   Seed ensemble spread
        │   ├── Sigma_Irr/                   #   Irrigation fraction spread
        │   ├── Sigma_LULC/                  #   LULC projection spread
        │   ├── Sigma_GW/                    #   GW fraction spread
        │   ├── Sigma_Total/                 #   Quadrature combination (σ, CV, per-category σ)
        │   ├── Sigma_CU/                    #   CU inter-GCM spread
        │   └── Plots/                       #   Time-series plots
        ├── Graphical_Abstract_Fig1.png         # Publication Figure 1 (map + time series)
        ├── Prediction_Exceedance_Summary.csv   # Per-year exceedance stats
        ├── Raster_Maps/                     # Step 3g — spatial maps for all products
        │   ├── Era_Maps_*.png               #   2×2 era-mean panels per category
        │   ├── Actual_vs_Predicted.png      #   Actual vs predicted (1984–2024)
        │   └── Trend_Analysis/              #   Mann-Kendall + Sen's slope maps
        │       ├── Trend_*.png              #   Per-category, per-period trend maps
        │       ├── Basin_Trend_*.csv        #   Per-basin zonal trend statistics
        │       └── Subbasin_Trend_*.csv     #   Per-sub-basin zonal trend statistics
        ├── Visualizations/                  # Time series & era summary maps
        ├── Well_Package/                    # Per-well GeoPackage
        ├── Intercomparison/                 # Step 4a — withdrawal comparison
        │   └── Temporal_Agreement/          #   Heatmaps, box/violin, Taylor, r-vs-NSE
        ├── CU_IE_Intercomparison/           # Step 4b — CU/IE comparison
        ├── CAP_SRP_Validation/              # Step 4c — CAP/SRP SW validation
        ├── Peff_Intercomparison/            # Step 4d — Peff comparison
        └── PS_Intercomparison/              # Step 4e — Non-irrigation vs USGS PS
```

### Data References

Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. _International Journal of Climatology_, _33_(1), 121–131. https://doi.org/10.1002/joc.3413.

Abatzoglou, J. T., & Brown, T. J. (2012). A comparison of statistical downscaling methods suited for wildfire applications. _International Journal of Climatology_, _32_(5), 772–780. https://doi.org/10.1002/joc.2312.

Alzraiee, A., Niswonger, R., Luukkonen, C., Larsen, J., Martin, D., Herbert, D., Buchwald, C., Dieter, C., Miller, L., Stewart, J., Houston, N., Paulinski, S., & Valseth, K. (2024). Next Generation Public Supply Water Withdrawal Estimation for the Conterminous United States Using Machine Learning and Operational Frameworks. _Water Resources Research_, _60_(7). https://doi.org/10.1029/2023WR036632

Daly, C., Halbleib, M., Smith, J. I., Gibson, W. P., Doggett, M. K., Taylor, G. H., Curtis, J., & Pasteris, P. P. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. _International Journal of Climatology_, _28_(15), 2031–2064. https://doi.org/10.1002/joc.1688.

Fleckenstein, R., Wellington, D., Jin, S., Tollerud, H., Brown, J. F., Dewitz, J., Pastick, N. J., Barber, C. P., O’Brien, A., & Spanier, M. (2026). A framework for integrating spatiotemporal deep learning methods with landsat for annual land cover and impervious surface mapping. _Remote Sensing of Environment_, _338_, 115347. https://doi.org/10.1016/j.rse.2026.115347.

Gangopadhyay, S., & Pruitt, T. (2011). West-Wide Climate Risk Assessments:  Bias-Corrected  and Spatially Downscaled  Surface Water Projections (Technical Memorandum No. 86-68210-2011-01). _U.S. Bureau of Reclamation_. https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf.

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. _Remote Sensing of Environment_, _202_, 18–27. https://doi.org/10.1016/j.rse.2017.06.031.

Hasan, M. F., Smith, R. G., Majumdar, S., Huntington, J. L., Alves Meira Neto, A., & Minor, B. A. (2025). Satellite data and physics-constrained machine learning for estimating effective precipitation in the Western United States and application for monitoring groundwater irrigation. _Agricultural Water Management_, _319_, 109821. https://doi.org/10.1016/j.agwat.2025.109821.

Haynes, J.V., Read, A.L., Chan, A.Y., Martin, D.J., Regan, R.S., Henson, W.R., Niswonger, R.G., & Stewart, J.S., 2023, Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000-2020 within the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_, https://doi.org/10.5066/P9LGISUM.

Hodson, T.O., Hariharan, J.A., Black, S., & Horsburgh, J.S.. (2023). dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. _U.S. Geological Survey software release_. https://doi.org/10.5066/P94I5TX3.

Hung, F., Chiarelli, D. D., Famiglietti, J. S., & Müller, M. F. (2025). Downscaled global 60-meter resolution estimates of irrigation water sources (2000–2015). _Scientific Data_, _12_(1), 1632. https://doi.org/10.1038/s41597-025-05920-x.

Ketchum, D., Hoylman, Z. H., Huntington, J., Brinkerhoff, D., & Jencso, K. G. (2023). Irrigation intensification impacts sustainability of streamflow in the Western United States. _Communications Earth & Environment_, _4_(1), 479. https://doi.org/10.1038/s43247-023-01152-2.

Ketchum, D., Jencso, K., Maneta, M. P., Melton, F., Jones, M. O., & Huntington, J. (2020). IrrMapper: A Machine Learning Approach for High Resolution Mapping of Irrigated Agriculture Across the Western U.S. _Remote Sensing_, _12_(14), 2328. https://doi.org/10.3390/rs12142328.

Luukkonen, C.L., Alzraiee, A.H., Larsen, J.D., Martin, D.J., Herbert, D.M., Buchwald, C.A., Houston, N.A., Valseth, K.J., Paulinski, S., Miller, L.D., Niswonger, R.G., Stewart, J.S., & Dieter, C.A. (2023). Public supply water use reanalysis for the 2000-2020 period by HUC12, month, and year for the conterminous United States. _U.S. Geological Survey data release_. https://doi.org/10.5066/P9FUL880

Majumdar, S., ReVelle, P., Pearson, C., Nozari, S., Minor, B. A., Hasan, M. F., Huntington, J. L., & Smith, R. G. (2026). pyCropWat: A Python Package for Computing Effective Precipitation Using Google Earth Engine Climate Data (v1.2.1). _Zenodo_. https://doi.org/10.5281/zenodo.18706481.

Martin, D. J., Niswonger, R. G., Regan, R. S., Huntington, J. L., Ott, T., Morton, C., Senay, G. B., Friedrichs, M., Melton, F. S., Haynes, J., Henson, W., Read, A., Xie, Y., Lark, T., & Rush, M. (2025). Estimating irrigation consumptive use for the conterminous United States: coupling satellite-sourced estimates of actual evapotranspiration with a national hydrologic model. _Journal of Hydrology_, _662_, 133909. https://doi.org/10.1016/j.jhydrol.2025.133909.

Martin, D.J., Regan, R.S., Haynes, J.V., Read, A.L., Henson, W.R., Stewart, J.S., Brandt, J.T., & Niswonger, R.G. (2023). Irrigation water use reanalysis for the 2000-20 period by HUC12, month, and year for the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_. https://doi.org/10.5066/P9YWR0OJ.

Melton, F., Huntington, J., Grimm, R., Herring, J., Hall, M., Rollison, D., Erickson, T., Allen, R., Anderson, M., Fisher, J. B., Kilic, A., Senay, G. B., Volk, J., Hain, C., Johnson, L., Ruhoff, A., Blankenau, P., Bromley, M., Carrara, W., … Anderson, R. G. (2022). OpenET: Filling a Critical Data Gap in Water Management for the Western United States. _JAWRA Journal of the American Water Resources Association_. https://doi.org/10.1111/1752-1688.12956.

Muratoglu, A., Bilgen, G. K., Angin, I., & Kodal, S. (2023). Performance analyses of effective rainfall estimation methods for accurate quantification of agricultural water footprint. _Water Research_, _238_, 120011. https://doi.org/10.1016/j.watres.2023.120011.

Reitz, M., Sanford, W. E., & Saxe, S. (2023). Ensemble Estimation of Historical Evapotranspiration for the Conterminous U.S. _Water Resources Research_, _59_(6). https://doi.org/10.1029/2022WR034012.

Reitz, M., Sanford, W. E., & Saxe, S. (2023). Historical evapotranspiration for the conterminous U.S. _U.S. Geological Survey Data Release_. https://doi.org/10.5066/P9EZ3VAS.

Roy, S., Majumdar, S., & Swetnam, T. (2025).  samapriya/awesome-gee-community-datasets: Community Catalog (3.9.0). _Zenodo_. https://doi.org/10.5281/zenodo.17641528.

Sohl, T. L., Reker, R., Bouchard, M., Sayler, K., Dornbierer, J., Wika, S., Quenzer, R., & Friesz, A. (2016). Modeled historical land use and land cover for the conterminous United States. _Journal of Land Use Science_, _11_(4), 476–499. https://doi.org/10.1080/1747423X.2016.1147619.

Sohl, T. L., Reker, R., Bouchard, M., Sayler, K., Dornbierer, J., Wika, S., Quenzer, R., & Friesz, A. (2018). Modeled historical land use and land cover for the conterminous United States: 1938-1992. _U.S. Geological Survey data release_. https://doi.org/10.5066/F7KK99RR.

Sohl, T. L., Sayler, K. L., Bouchard, M. A., Reker, R. R., Friesz, A. M., Bennett, S. L., Sleeter, B. M., Sleeter, R. R., Wilson, T., Soulard, C., Knuppe, M., & van Hofwegen, T. (2014). Spatially explicit modeling of 1992–2100 land cover and forest stand age for the conterminous United States. _Ecological Applications_, _24_(5), 1015–1036. https://doi.org/10.1890/13-1245.1.

Sohl, T. L., Sayler, K. L., Bouchard, M. A., Reker, R. R., Friesz, A. M., Bennett, S. L., Sleeter, B. M., Sleeter, R. R., Wilson, T., Soulard, C., Knuppe, M., & van Hofwegen, T. (2018). Conterminous United States Land Cover Projections - 1992 to 2100. _U.S. Geological Survey data release_. https://doi.org/10.5066/P95AK9HP.

Soil Survey Staff, Natural Resources Conservation Service, United States Department of Agriculture. _Web Soil Survey_. Available online at https://websoilsurvey.nrcs.usda.gov/. 

Suresh, S., Hossain, F., Mishra, V., & Hossain, N. (2026). GRAIN – a Global Registry of Agricultural Irrigation Networks. _Earth System Science Data_, _18_(3), 1855–1875. https://doi.org/10.5194/essd-18-1855-2026.

USBR. (2025). Reclamation Information Sharing Environment (RISE). https://rise-usbr.opendata.arcgis.com/

USDA SCS. (1993). Chapter 2 Irrigation Water Requirements. In Part 623 National Engineering Handbook. _USDA Soil Conservation Service_. https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf.

USGS. (2024). Annual NLCD Collection 1 Science Products. _U.S. Geological Survey data release_. https://doi.org/10.5066/P94UXNTS.

Volk, J. M., Huntington, J. L., Melton, F. S., Allen, R., Anderson, M., Fisher, J. B., Kilic, A., Ruhoff, A., Senay, G. B., Minor, B., Morton, C., Ott, T., Johnson, L., de Andrade, B., Carrara, W., Doherty, C. T., Dunkerly, C., Friedrichs, M., Guzman, A., … Yang, Y. (2024). Assessing the accuracy of OpenET satellite-based evapotranspiration data to support water resource and land management applications. _Nature Water_, _2_(2), 193–205. https://doi.org/10.1038/s44221-023-00181-7.

Volk, J., Dunkerly, C., Majumdar, S., Huntington, J., Minor, B., Kim, Y., Morton, C., ReVelle, P., Kilic, A., Melton, F., Allen, R., Pearson, C., Purdy, A., & Caldwell, T. (2026). CONUS Gridded Reference Evapotranspiration Bias Correction: Inputs, Station Validation, and Outputs (gridMET/OpenET) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18673484.

Walkinshaw, M., O’Geen, A. T., & Beaudette, D. E. (2022). Soil Properties. _California Soil Resource Lab_. https://casoilresource.lawr.ucdavis.edu/soil-properties/.