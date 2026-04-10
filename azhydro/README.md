# AZ-Hydro: ML Pipeline for Arizona Water Use Estimation (1896–2099)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19057936.svg)](https://doi.org/10.5281/zenodo.19057936)

Maintainers [Dr. Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu]

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Historical and projected groundwater/surface-water withdrawals, irrigation consumptive use, and pumping-induced surface water capture for Arizona, 1896–2099. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Declining Agricultural Dominance and Rising Urban Demand Drive a Two-Century Shift in Withdrawal Patterns (1896–2099). _In prep. for AGU Earth's Future_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). AZ-Hydro — Historical and Projected Arizona Annual Water Use: Software, Input Data, Models, Raster and Well GeoPackage Predictions, and Validation at 2 km Resolution (1896–2099). _Zenodo_. https://doi.org/10.5281/zenodo.19057936.

---

## Running the project

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.11). If Anaconda or miniconda is already installed, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. It may be necessary to restart the current shell session to add conda to path.

The conda package manager can be updated by running the following command: ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help in installing the correct packages and Python version to run the codes.

### 2. Clone the repository and download input data

Clone the repository using Git:
```bash
git clone https://github.com/<owner>/az-hydro.git
cd az-hydro
```

The `Data/` folder is hosted separately on Zenodo due to its size (~14 GB).
Download it from [https://doi.org/10.5281/zenodo.19057936](https://doi.org/10.5281/zenodo.19057936) and extract it at the repository root so that `Data/Inputs/` exists.
Unzip all zipped files — several input datasets are compressed and must be unzipped before running the pipeline.

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

> **First-time run:** Ensure you have downloaded the `Data/` folder from
> [Zenodo](https://doi.org/10.5281/zenodo.19057936) (see step 2). The default
> flags (`--skip-download`, `--load-files`) assume GEE tiles and intermediate
> files already exist on disk. If you are starting from scratch, use
> `--download --recreate` to fetch GEE data and build all intermediate files:
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
python pipeline.py --skip-eda            # skip EDA plot generation (auto-skipped when Step 1 not selected)
python pipeline.py --steps 2s            # regenerate cross-strategy summary from saved results
```

Step 0 supports fine-grained sub-step control via `--skip-prep`:

```bash
python pipeline.py --steps 0 --skip-prep gee               # skip GEE download & mosaic
python pipeline.py --steps 0 --skip-prep gw-csv,gw-rasters # skip GW CSV and raster creation
python pipeline.py --recreate --skip-prep streamflow        # recreate everything except streamflow
python pipeline.py --skip-prep gee,vectors,reproject        # skip multiple sub-steps
```

Evaluation strategies (Steps 2a/2a2/2b/2c) support `--skip-eval`:

```bash
python pipeline.py --skip-eval random,spatial         # skip random and spatial LOO evaluations
python pipeline.py --skip-eval temporal               # skip temporal LOO evaluation
python pipeline.py --skip-eval summary                # skip cross-strategy summary
python pipeline.py --skip-eval pixel,temporal,summary # skip multiple strategies
```

UQ components (Step 3b) support `--skip-uq`:

```bash
python pipeline.py --steps 3b --skip-uq sigma-maca,sigma-lulc   # skip GEE-dependent components
python pipeline.py --steps 3b --skip-uq sigma-maca              # skip only inter-GCM spread
python pipeline.py --steps 3b --skip-uq sigma-model,density-sensitivity # skip seed ensemble + partition sensitivity
```

#### Available steps

| Step | Description |
|------|-------------|
| `0`  | Data preparation (GEE download, GW processing, rasterisation) |
| `1`  | Create AZ predictor dataset (Parquet) |
| `2a` | Evaluate random 80/20 train/test split |
| `2a2` | Evaluate pixel holdout (spatial locations held out across all years) |
| `2b` | Evaluate LOO temporal holdout |
| `2c` | Evaluate LOO spatial holdout (AMA/INA basins) |
| `2c-seed` | Evaluate seeded LOO spatial holdout (10% local calibration) |
| `2s` | Cross-strategy summary (can run standalone from saved results) |
| `3`  | Full-period XGBRF prediction (1896–2099) |
| `3b` | Hybrid uncertainty quantification |
| `3e` | Well package (per-well Parquet + GPKG locations with uncertainty) |
| `3g` | Raster maps, actual vs predicted, and trend analysis for all output categories |
| `4`  | USGS intercomparison |
| `4b` | CU intercomparison |
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
| `streamflow` | Canal density (temporally scaled via HarDWR v2.0) & streamflow rasters |
| `basin-rasters` | GW basin, sub-basin, well density & irr capacity fraction rasters |
| `wtd` | Water table depth raster ([Ma et al., 2026](https://doi.org/10.1038/s43247-025-03094-3)) |
| `rights-rasters` | HarDWR v2.0 SW access year, irr/non-irr SW rights density rasters |
| `reproject` | Reproject GEE mosaics to match GW grid |

#### Evaluation sub-steps

| Sub-step | Description |
|----------|-------------|
| `random` | Skip random 80/20 evaluation (Step 2a) |
| `pixel` | Skip pixel holdout evaluation (Step 2a2) |
| `temporal` | Skip LOO temporal holdout evaluation (Step 2b) |
| `spatial` | Skip LOO spatial holdout evaluation (Step 2c) |
| `spatial-seed` | Skip seeded LOO spatial holdout evaluation (Step 2c-seed) |
| `summary` | Skip cross-strategy summary |

#### UQ sub-steps

| Sub-step | Description |
|----------|-------------|
| `sigma-maca` | Skip σ_MACA — inter-GCM climate spread (requires GEE download) |
| `sigma-model` | Skip σ_model — seed ensemble spread |
| `sigma-irr` | Skip σ_irr — irrigation fraction uncertainty |
| `sigma-lulc` | Skip σ_LULC — LULC projection spread (requires GEE download) |
| `sigma-gw` | Skip σ_gw — well-density feature sensitivity across 5 recent HarDWR snapshots (2020–2024) |
| `density-sensitivity` | Skip partition-level diagnostic (density-ratio ±20% + smoothing-sigma sweep {2, 8}) |
| `sigma-total` | Skip σ_total quadrature, basin σ, visualizations, and raster augmentation |
| `sigma-cu` | Skip σ_CU — consumptive use uncertainty (IE × Withdrawal error propagation) |

> **Note on skipping individual σ components:** Per-category σ arrays (e.g., σ_model for Irrigation, Non_Irrigation, etc.) are only held in memory during computation and are never written to disk as separate rasters. When `sigma-total` runs, it can reload *total-level* per-component σ from disk (e.g., `Sigma_Model_mm_{year}.tif`), but the per-category σ_total rasters (`Sigma_Total_{cat}_mm_{year}.tif`) will be zero if the individual σ steps were skipped. This causes downstream augmented category rasters to have zero σ, which in turn makes σ_CU zero. To get correct per-category uncertainty, run all individual σ components (σ_MACA through σ_gw) without skipping. Only `density-sensitivity` (the partition-level diagnostic) can be safely skipped without affecting downstream products.

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `all` | Comma-separated step IDs to run (e.g. `"0,1,2a"`) or `"all"`. |
| `--skip-download` | `True` | Skip GEE tile download; use existing tiles on disk. |
| `--download` | — | Force GEE tile download. |
| `--load-files` | `True` | Skip recreating intermediate files that already exist. |
| `--recreate` | — | Force recreation of intermediate files. |
| `--skip-eda` | `False` | Skip EDA plot generation in Step 1. EDA is auto-skipped when Step 1 is not explicitly selected. |
| `--skip-prep` | — | Comma-separated Step 0 sub-steps to skip. |
| `--skip-eval` | — | Comma-separated evaluation strategies to skip. |
| `--skip-uq` | — | Comma-separated UQ sub-steps to skip. |
| `-v`, `--verbose` | `False` | Enable verbose (DEBUG-level) logging. |

The pipeline executes the selected steps in sequence (details below).

---

## Data sources

The project builds a spatially explicit, multi-decadal (1896–2099) dataset for Arizona by combining satellite-derived products, climate model projections, soil properties, streamflow observations, and USBR modeled streamflow.

### Google Earth Engine (GEE) predictor bands

The [`download_gee_data()`](hydrolibs/dataops.py) function downloads 15 bands of geospatial data from GEE ([Gorelick et al., 2017](https://doi.org/10.1016/j.rse.2017.06.031); [Roy et al., 2025](https://doi.org/10.5281/zenodo.17641528)) as tiled GeoTIFFs at 2 km resolution over Arizona. Data are harmonized across three temporal eras using overlap-period bias-correction ratios to ensure continuity.

| Band | Description | Units | Source |
|------|-------------|-------|--------|
| `annual_et_ensemble_mm` | Actual evapotranspiration | mm/yr | [Reitz et al., 2023](https://doi.org/10.1029/2022WR034012) (1896–1999), [OpenET (Melton et al., 2022](https://doi.org/10.1111/1752-1688.12956); [Volk et al., 2024)](https://doi.org/10.1038/s44221-023-00181-7) (2000–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) ensemble (2026–2099) |
| `annual_eto_mm` | Reference evapotranspiration (Penman-Monteith) | mm/yr | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) Hargreaves (1896–1978), [OpenET gridMET (Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026)](https://doi.org/10.5281/zenodo.18673484) (1979–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) bias-corrected ([Volk et al., 2026](https://doi.org/10.5281/zenodo.18673484)) ensemble (2026–2099) |
| `annual_precip_mm` | Precipitation | mm/yr | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) ensemble (2026–2099) |
| `annual_peff_mm` | Effective precipitation (USDA SCS method) | mm/yr | [USDA SCS, 1993](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf); [Muratoglu et al., 2023](https://doi.org/10.1016/j.watres.2023.120011); [Majumdar et al., 2026](https://doi.org/10.5281/zenodo.18706481) |
| `annual_peff_pcml_mm` | Effective precipitation (PCML obs-based, 2000–2024) | mm/yr | [Hasan et al., 2025](https://doi.org/10.1016/j.agwat.2025.109821), climatological mean outside 2000–2024 |
| `annual_tmmx_K` | Annual mean daily max temperature | K | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) (2026–2099) |
| `annual_tmmn_K` | Annual mean daily min temperature | K | [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) (1896–2025), [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) (2026–2099) |
| `lulc` | Land use/land cover (1=Agriculture, 2=Urban, 3=Surface Water) | categorical | [USGS historical (Sohl et al., 2016)](https://doi.org/10.1080/1747423X.2016.1147619) (≤1984), [NLCD (USGS, 2024](https://doi.org/10.5066/P94UXNTS); [Fleckenstein et al., 2026)](https://doi.org/10.1016/j.rse.2026.115347) (1985–2025), [USGS LULC Projections (Sohl et al., 2014)](https://doi.org/10.1890/13-1245.1) projections (2026–2099) |
| `annual_crop_fraction` | Cropland fraction (physical density) | fraction | Derived from LULC at native resolution (30 m NLCD 1985–2025, 250 m USGS elsewhere), aggregated to 2 km; basin-delta corrected for off-NLCD years |
| `annual_urban_fraction` | Urban/developed fraction (physical density) | fraction | Derived from LULC at native resolution, aggregated to 2 km; basin-delta corrected for off-NLCD years |
| `annual_irr_fraction` | Irrigated area fraction | fraction | [IrrMapper (Ketchum et al., 2020](https://doi.org/10.3390/rs12142328); [2023)](https://doi.org/10.1038/s43247-023-01152-2) RF v1.2 (1985–2025), LULC-derived outside |
| `annual_gw_fraction` | Groundwater irrigation fraction | fraction | [Hung et al., 2025](https://doi.org/10.1038/s41597-025-05920-x) snapshots (2000, 2005, 2010, 2015) |
| `soil_depth_mm` | Soil depth | mm | [CSRL (Walkinshaw et al., 2022)](https://casoilresource.lawr.ucdavis.edu/soil-properties/) (static) |
| `awc_mm` | Available water capacity (0–152 cm) | mm | [SSURGO](https://websoilsurvey.nrcs.usda.gov/) (static) |
| `ksat_mean_micromps` | Saturated hydraulic conductivity | μm/s | [CSRL (Walkinshaw et al., 2022)](https://casoilresource.lawr.ucdavis.edu/soil-properties/) (static) |

### Data harmonization

The pipeline stitches disparate sources into a consistent 1896–2099 time series:

- **ET**: Reitz ensemble (1896–1999) → OpenET v2.0/v2.1 (2000–2025) → MACA × EToF crop coefficients (2026–2099)
- **ETo**: PRISM Hargreaves (1896–1978) → OpenET gridMET (1979–2025) → MACA 20-model ensemble (2026–2099)
- **LULC**: USGS historical scenario (≤1984) → NLCD (1985–2025) → USGS 4-scenario mode ensemble (2026–2099).
  - **Basin-scale delta correction** (analog to the climate bias correction approach): USGS and NLCD disagree on urban/ag classification at the 2 km grid because NLCD's 30 m resolution captures roads, small towns, and fragmented urban/ag pixels that USGS at 250 m misses. At the basin scale these classification noises average out, so for off-NLCD years we compute per-basin relative change from USGS/FORE-SCE and apply it as a multiplicative delta to NLCD-anchor pixel values (1985 for hindcast, 2025 for projection). Four LULC-derived columns are corrected — `URBAN`, `AGRI`, `annual_crop_fraction`, `annual_urban_fraction`. Baseline years are the non-NLCD years paired with the NLCD anchors: 1984 (USGS Historical, immediately before NLCD era) for hindcast, 2026 (FORE-SCE ensemble, immediately after NLCD era) for projection. This guarantees `delta(1984, B) = 1.0` and `delta(2026, B) = 1.0` exactly, so the sign of relative change is physically correct (a growing basin has `delta(earlier year) < 1` and `delta(later year) > 1`). Per-basin, per-year, per-class multipliers are exported to `Basin_LULC_Deltas.csv` for inspection. `create_az_data_parquet` bakes the correction into the parquet, so ML training, inference, and uncertainty all see consistent NLCD-anchored LULC features.
- **Streamflow**: USGS NWIS observations (variable start dates through 2025) → USBR ensemble projections (2026–2099). Per-month multiplicative **bias-correction factors** (`USGS_mean / USBR_mean` per calendar month) are computed from the USGS/USBR overlap period and applied to all USBR-filled months (both pre-USGS and post-2025). For sites without USBR data, monthly ratios to the nearest USBR-gauged reference site are applied on top of the already-bias-corrected reference projections. This eliminates the systematic +79% step-jump previously observed at the 2025→2026 boundary.
- **Climate projections**: MACA v2 daily data across 20 GCMs × 2 RCPs (RCP 4.5, RCP 8.5) = 40-member ensemble. All MACA queries use a flat-pipeline approach (single filter + reduce) to keep GEE computation graphs small: ETo uses `.sum().divide(40)` per month (computed per-image to preserve nonlinearity), precip uses `.sum().divide(40)`, and temperature uses `.mean()`.

Per-pixel, per-month bias-correction ratios are computed from overlapping observation periods and applied to extend each variable seamlessly. See [`gee/README.md`](../gee/README.md) for asset export details and equations.

### GEE pre-exported assets

Twelve custom ImageCollections are pre-computed via scripts in [`gee/`](../gee/) and stored in GEE under `projects/azhydro/assets/`:

| Asset | Description | Years |
|-------|-------------|-------|
| `gridmet_hargreaves_eto_ratio` | OpenET gridMET / PRISM Hargreaves monthly ratio (12 images) | Climatology |
| `openet_reitz_et_ratio` | OpenET / Reitz ensemble monthly ratio (12 images) | Climatology |
| `monthly_etof` | Crop coefficient (OpenET / OpenET gridMET ETo) | Climatology |
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

> **Tile retry note:** Each tile is retried up to 10 times on transient GEE
> failures. If a tile still fails after all retries, re-running the same
> download step after the remaining tiles have finished will fix the issue—
> already-downloaded tiles are skipped automatically, so only the failed
> tiles are re-attempted.

### Streamflow analysis

The [`streamflowops`](hydrolibs/streamflowops.py) module handles streamflow data acquisition and rasterization. It covers all 16 Arizona surface watersheds from 1896 to 2099.

#### Data sources

- **USGS NWIS**: Daily mean discharge (parameter 00060) via the `dataretrieval` Python API ([Hodson et al., 2023](https://doi.org/10.5066/P94I5TX3)), resampled to monthly means
- **USBR CMIP Ensemble**: Monthly modeled streamflow ([Gangopadhyay & Pruitt, 2011](https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf); [USBR, 2025](https://rise-usbr.opendata.arcgis.com/)) averaged across ~112 climate model runs (scenarios a1b, a2, b1), spanning 1950–2099
- **Historical Ratio Method**: For sites without USBR projections, per-calendar-month scaling ratios are computed against the nearest USBR-gauged reference site and applied to generate synthetic 1950–2099 projections

#### Gauge network (19 sites)

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
| 09537500 | — | Whitewater Draw near Douglas | White Water Draw |
| 09537200 | — | Leslie Creek near McNeal | White Water Draw / Rio Yaqui |
| 09426650 | — | CAP Canal at Havasu Pumping Plant | CAP Diversion |

Sites with USBR IDs (9 sites) have direct modeled projections. The remaining 11 sites use the historical ratio method, where monthly scaling ratios are computed from the overlapping USGS observation period between the target site and its nearest USBR-gauged reference.

#### Gap-filling strategy

1. **USGS observations** take priority within their available record
2. **USBR ensemble mean** (or ratio-scaled synthetic) fills months outside the USGS range. Per-month **multiplicative bias-correction factors** (`USGS_mean / USBR_mean` per calendar month) are computed from the USGS/USBR overlap period and applied to all USBR-filled months, eliminating the step-jump at the USGS→USBR boundary. For non-USBR sites using the ratio method, the reference site's USBR data is already bias-corrected before ratios are applied, so the correction propagates transitively.
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

### Water table depth (WTD)

A static water table depth raster from [Ma et al. (2026)](https://doi.org/10.1038/s43247-025-03094-3)
is mosaicked from state-level tiles (Arizona, Nevada, California),
reprojected from Lambert Conformal Conic to EPSG:26912, and resampled
to 2 km using mean aggregation.  Values are in meters below ground
surface.  The WTD is time-invariant (single snapshot) and is used as
an ML predictor capturing subsurface conditions that influence pumping
patterns (e.g., shallow water table areas near rivers have different
withdrawal characteristics than deep basin-fill aquifers).

### Surface Water Capture Index

The pipeline produces a per-pixel, per-year **Surface Water Capture
Index** quantifying what fraction of GW pumping likely depletes surface
water.  The index combines hydraulic connectivity (exponential decay
with water table depth) and surface water availability (focal-max
normalized canal-weighted streamflow):

```
capture_fraction = exp(-wtd_m / λ) × cw_norm
sw_capture_mm    = Total_GW × capture_fraction
```

Three λ values (5, 10, 20 m) produce lower/central/upper bounds
without tunable parameters.  Volume bounds additionally incorporate
Total_GW uncertainty (σ_total from the UQ framework).

Output rasters are organized by GW pumping pool — Total, Irrigation,
Non-Irrigation — under `SW_Capture/{Total,Irrigation,Non_Irrigation}_SW_Capture_Fraction/`
(3-band, dimensionless [0, 1] for λ = 5/10/20 m) and
`SW_Capture/{Total,Irrigation,Non_Irrigation}_SW_Capture_Rasters/Depth_{unit}/`
(central capture volume in mm/ft/m³/AF).  Reading the directory names:
`Total_SW_Capture_Fraction` is "the fraction of Total GW pumping that
captures surface water," and so on for the irrigation and non-irrigation
splits.  A time series CSV
(`SW_Capture/SW_Capture_Time_Series.csv`) and era-mean maps are also
produced.

**What "SW capture" actually means in this model.** The index measures
only one specific pathway: well-mediated stream depletion in pixels the
partition step has labeled as `_GW`. Three categories of well/canal
interaction are *not* counted in the capture numerator: (1) direct
canal diversions, which never enter the model because the ML target is
ADWR Well Registry pumping (federal Yuma-area deliveries through WMIDD,
YCWUA, and the Gila Project bypass wells entirely and are reconciled
separately as the ~2.26 MAF water-budget gap); (2) wells filed under
HarDWR surface-water rights, which the partition routes into `_SW`
regardless of whether they are physically pumping ambient groundwater
or river-recharged alluvium — those volumes are tracked under
`Irrigation_SW`/`Non_Irrigation_SW` and never reach the capture index;
and (3) any well-mediated SW interaction outside the perennial
canal-delivered footprint (`cw_norm = 0`), since ephemeral stream–
aquifer exchange would require transient groundwater modeling. The
capture fraction is therefore the model's most conservative lower bound
on well-mediated SW depletion: a "low" capture fraction in a
canal-dominated basin like Yuma (~4%) means most of that basin's SW use
is being delivered through canals or through SW-righted wells already
counted in `Total_SW`, not that wells are causing little impact.

**Why this is hard.** Allocating surface-water *withdrawals* across
canal diversions and water-right duties is standard water accounting
and can be done from permits and delivery records. Allocating
*groundwater pumping* into the share that depletes stream baseflow
versus the share that mines aquifer storage is much harder: at the
basin scale it normally requires a transient calibrated groundwater
model coupled to a stream network (e.g. MODFLOW–SFR), which is built
one aquifer at a time and rarely covers entire states or multi-century
time spans. The capture index here uses a process-informed proxy
([Barlow & Leake 2012](https://doi.org/10.3133/cir1376),
[Condon & Maxwell 2019](https://doi.org/10.1126/sciadv.aav4574)) —
exponential connectivity decay with water table depth, modulated by
canal-weighted streamflow availability — applied at 2 km annual
resolution across all of Arizona for 1896–2099, with three λ values
producing physically-bounded uncertainty intervals rather than a single
tuned answer. The contribution is the *coverage* (full state, two
centuries, hindcast plus projection) more than the formula itself. It
is not a substitute for a calibrated transient flow simulation in any
individual basin, but it provides a consistent first-order screen for
where well-mediated stream depletion is plausibly significant.

**Scope limitation:** The index quantifies SW depletion only where
perennial canal-delivered surface water exists (`cw_norm > 0`).
Ephemeral stream–aquifer exchange is excluded because most ephemeral
flow is lost to ET before wells can capture it, and quantifying it
would require transient groundwater flow modeling.

**References:**
[Condon & Maxwell (2019)](https://doi.org/10.1126/sciadv.aav4574),
[de Graaf et al. (2019)](https://doi.org/10.1038/s41586-019-1594-4),
[Barlow & Leake (2012)](https://pubs.usgs.gov/circ/1376/).

---

## Pipeline overview (`pipeline.py`)

The pipeline is the top-level orchestrator that chains together data
preparation, ML model evaluation, full-period prediction, and
intercomparison with independent USGS datasets.  It is divided into five
numbered steps plus a cross-strategy summary:

```
Step 0   ─  Data Preparation
Step 1   ─  Create AZ Predictor DataFrame
Step 2   ─  Model Evaluation (5 strategies: Random, Pixel Holdout, Temporal LOO, Spatial LOO, Seeded Spatial LOO)
Step 3   ─  Full-Period XGBRF Prediction (1896–2099)
Step 3e  ─  Well Package (per-well Parquet + GPKG locations with uncertainty)
Step 3g  ─  Raster Maps & Trend Analysis for All Output Categories
Step 4   ─  USGS Intercomparison (Withdrawals, CU, Peff)
```

### Configuration constants

All paths and modeling parameters are defined once at the top of
`pipeline.py`:

| Constant | Value | Description |
|---|---|---|
| `INPUT_DIR` | `../Data/Inputs/` | Root for all input datasets (downloaded from [Zenodo](https://doi.org/10.5281/zenodo.19057936)). |
| `OUTPUT_DIR` | `../Data/Outputs/` | Root for all generated outputs. |
| `WATER_USE` | `'All'` | Well filter (`'All'` or `'Irr_Wells'`). |
| `MOSAIC_RASTER_RES` | `2000` | Raster pixel size (m). |
| `TILE_SIZE` | `80000` | Tile size for GEE export (m). |
| `START_YEAR` | `1896` | First prediction year. |
| `END_YEAR` | `2099` | Last prediction year. |
| `YEAR_LIST` | `1984–2024` | Years with metered withdrawal data (ADWR). |
| `TRAIN_YEAR_LIST_BASELINE` | `2002–2020` | Training years for direct comparison with [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757). Used only by the `T1_Baseline` temporal holdout. |
| `MIN_GW` | `None` | Minimum per-pixel withdrawal depth (mm). Pixels below this are excluded via outlier processing. The default `None` includes only positive withdrawal pixels (i.e., zero-withdrawal pixels are excluded), which is appropriate when annual irrigation masks are available to identify actively pumped areas. Set to `0` for baselines that include zeros (e.g., `T1_Baseline`). |
| `MAX_GW` | `3000` | Maximum allowed withdrawal depth (mm). A conservative upper bound approximately twice Tukey's extreme fence (Q3 + 3×IQR) and well above P99, designed to remove only `gdal_rasterize` artifacts and physically implausible extremes while retaining legitimately high-withdrawal pixels. Consistent with [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757). Set to `None` to disable (falls back to 10,000 mm). |
| `AF_MAX_THRESHOLD` | `5000` | Maximum per-well `AF Pumped`; rows exceeding this are dropped from CSVs. |
| `LOG_TARGET` | `False` | Apply `log1p` / `expm1` target transform. See [Target transform](#target-transform). |
| `RANDOM_STATE` | `42` | Seed for reproducibility. |
| `N_EVAL_SEEDS` | `5` | Random seeds per test size for random/pixel holdout evaluation. |
| `EVAL_TEST_SIZES` | `(0.10, …, 0.30)` | Test fractions to sweep in the evaluation grid. |
| `N_TRIALS` | `50` | Optuna hyperparameter-tuning trials. |
| `FOLD_COUNT` | `5` | k-fold cross-validation folds. |
| `REPEATS` | `1` | Number of CV repetitions (e.g., `RepeatedKFold`). |
| `N_DASK_WORKERS` | `10` | Dask parallel workers for model training. |
| `N_DASK_WORKERS_DATA_PREP` | `40` | Dask parallel workers for data preparation (independent raster operations). |
| `USE_OPTUNA` | `True` | Enable TPE-based hyperparameter search. |
| `USE_DASK` | `True` | Enable distributed training via Dask. |
| `INCLUDE_ALL_MODELS` | `True` | When `True`, adds optional models (ETR, HGBR, GBR, ADA, BAG, CAT, LR, RIDGE, LASSO) to the 4 core models. |
| `SKIP_PIML` | `True` | When `True`, excludes physics-informed models (PIML_XGB, PIML_LGBM, PIML_XGBRF) from training and evaluation. See [note on PIML models](#note-on-physics-informed-piml-models). |
| `PHYSICS_INTERACTION_CONSTRAINTS` | `False` | When `True`, applies feature interaction constraints to PIML models (only relevant when `SKIP_PIML=False`). |
| `PREDICTION_MODEL` | `'XGBRF'` | Model used for full-period prediction (Step 3+). Valid names: core — `XGB`, `LGBM`, `RF`, `XGBRF`; optional (`INCLUDE_ALL_MODELS=True`) — `ETR`, `HGBR`, `GBR`, `ADA`, `BAG`, `CAT`, `LR`, `RIDGE`, `LASSO`; PIML (`SKIP_PIML=False`) — `PIML_XGB`, `PIML_LGBM`, `PIML_XGBRF`. |
| `USE_AMA_INA` | `True` | Restrict training to AMA/INA management areas. |
| `DROP_GW_BASINS` | `()` | Basins excluded from training. Empty by default (all AMA/INA basins included). For the `T1_Baseline` holdout, WILLCOX AMA and HUALAPAI VALLEY INA are dropped because they were not yet designated as AMA/INA management areas during the 2002–2020 period used by [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757), and therefore had no (or limited) metered withdrawal data at that time. |
| `MIN_SPATIAL_EVAL_SAMPLES` | `30` | Minimum non-zero metered samples for a basin to be included in spatial LOO. |
| `SKIP_SPATIAL_BASINS` | `('WILLCOX AMA',)` | Basins explicitly excluded from spatial LOO evaluation regardless of sample count. WILLCOX AMA is excluded because it has very few metered samples in the current dataset. |
| `SPATIAL_SEED_FRACTION` | `0.1` | Fraction of held-out basin samples randomly seeded into training for the seeded spatial LOO (Step 2c-seed). |
| `TEMPORAL_HOLDOUTS` | T1_Baseline + T1–T7 | Eight temporal leave-one-out configurations. `T1_Baseline` uses `TRAIN_YEAR_LIST_BASELINE` and `MIN_GW=0`. |
| `DROP_ATTRS` | (list) | Columns dropped before modeling. |

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
   cap (default **3,000 mm** when `MAX_GW=3000.`; falls back to
   10,000 mm when `MAX_GW=None`) catches any remaining `gdal_rasterize`
   artifacts and statistically implausible extremes.
3. **Streamflow & canal density** — `streamflowops.create_canal_density_raster()`
   and `streamflowops.create_streamflow_rasters()` build predictor layers
   from USGS/USBR gauge data and GRAIN canal geometry ([Suresh et al., 2026](https://doi.org/10.5194/essd-18-1855-2026)).
   Canal density is **temporally scaled** using watershed-level SW rights
   build-out from HarDWR v2.0 ([Lisk et al., 2024](https://doi.org/10.57931/2475303)).
4. **Basin & well rasters** — `gwops.create_gw_basin_rasters()`,
   `gwops.create_well_density_raster()`, and
   `gwops.create_irr_capacity_fraction_raster()` rasterize ADWR basins,
   sub-basins, per-year well counts, and per-year irrigation pump-capacity
   fractions from the Well Registry. Non-consumptive wells (monitoring,
   test, dewatering, drainage, remediation, mineral exploration, etc.) are
   excluded. Both well density and capacity fraction rasters are
   **temporally varying**: for each year, only wells installed by that year
   are included, based on `INSTALLED` / `APPLICATIO` dates.  Wells with
   missing PUMPRATE are imputed using the per-`WATER_USE` category median.
5. **HarDWR v2.0 water-rights rasters** — `gwops.create_sw_access_year_raster()`,
   `create_irr_sw_rights_density_raster()`, and
   `create_nonirr_sw_rights_density_raster()` produce rasters from the
   HarDWR v2.0 harmonized water rights dataset
   ([Lisk et al., 2024](https://doi.org/10.57931/2475303)).
   `sw_rights_density` (cumulative count of all consumptive SW PODs per
   pixel — irrigation + domestic + industrial + livestock, excluding
   environmental in-stream flow rights) is a time-varying ML predictor
   capturing surface-water availability build-out.
   `sw_access_year` (earliest irrigation SW priority year per pixel)
   temporally adjusts the Hung et al. `gw_frac` — pixels without SW access
   yet are set to 100 % groundwater.  `nonirr_sw_rights_density` (cumulative
   non-irrigation SW POD count, excluding environmental in-stream flow
   rights) provides a temporally varying proxy for the non-irrigation
   GW/SW split, replacing the static canal-density proxy.
6. **GEE reprojection** — `dataops.reproject_gee_mosaics()` aligns all GEE
   mosaics to the GW depth raster grid.

**Outputs:**

| Directory | Contents |
|---|---|
| `GEE_Mosaics_2000m/` | Mosaicked annual GEE predictor rasters. |
| `GW/Vectors/{WNAME}/` | Per-year GW shapefiles. |
| `GW/Rasters/GW_Depths_{WNAME}_2000m/` | Withdrawal depth rasters (mm). |
| `GW_Data/Vector_Reproj/` | Reprojected vectors (basins, wells, CAP, etc.). |
| `Predictor_Data_{WNAME}_2000m/` | Final predictor stack (Predictor\_YYYY.tif). |

### Step 1 — Create AZ predictor DataFrame (`create_az_data()`)

Reads every year's multi-band predictor raster (1896–2099) plus the
basin, sub-basin, streamflow, canal-density, and well-density rasters into
a single DataFrame via `dataops.create_az_data_parquet()`.  Each row represents
one pixel in one year; columns include all GEE predictors, ancillary
layers, basin/sub-basin labels, and (for metered years) observed withdrawals.

ADWR sub-basin OBJECTID codes are mapped to human-readable names using the
ADWR shapefile.  Before writing the Parquet file, `create_az_data_parquet()`
caps actual ET at **Kc_max × ETo** (Kc_max = 2.0) to remove physically
implausible values where ET exceeds reference ET.  ET can legitimately
exceed ETo for high-Kc crops (e.g. alfalfa), urban oasis/advection effects,
and open-water/riparian pixels, so a uniform 2× multiplier is applied rather
than a strict 1:1 cap.  The 2× threshold accommodates urban and surface-water
areas while still catching implausible extremes; empirical analysis shows
ET > ETo exceedance outside AGRI/URBAN/SW land uses is negligible.

Exploratory data analysis (EDA) plots are generated via
`vizops.explore_az_data()` and saved to `{MODEL_DIR}EDA/`.

A targeted withdrawal distribution analysis follows via
`vizops.analyze_pumping_distribution()`.  This logs percentile summaries,
Tukey's fence outlier benchmarks (Q3 + 1.5×IQR mild, Q3 + 3×IQR extreme),
and the percentile rank of `MAX_GW`, then saves an empirical CDF plot with
separate curves for each depth threshold (≤1 000, ≤2 000, … ≤5 000 mm) and
no threshold, overlaid with the Tukey fence and `MAX_GW` lines.

**Returns:** `az_df` — the full predictor DataFrame used by all subsequent
steps.

### Step 2 — Model evaluation

Four complementary strategies assess model performance.  The default
model zoo comprises 13 models — 4 core baselines (XGB, LGBM, RF, XGBRF)
and 9 additional ensemble/linear models (ETR, HGBR, GBR, ADA, BAG, CAT,
LR, RIDGE, LASSO) enabled via `INCLUDE_ALL_MODELS=True`.

All models use Optuna + Dask hyperparameter optimization (50 TPE trials,
5-fold CV) and report R², normalized RMSE (% of mean), normalized MAE
(% of mean), and normalized MBE (%).  All three normalized metrics use
the mean of observed values as the denominator, giving a physically
interpretable percentage error relative to the average withdrawal magnitude.

> **Note on R²:** The R² reported throughout this pipeline uses scikit-learn's
> `r2_score`, which is mathematically equivalent to the Nash–Sutcliffe
> Efficiency (NSE): R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)².  Unlike the
> squared Pearson correlation coefficient (which ranges from 0 to 1), this
> formulation ranges from −∞ to 1.  A value of 1 indicates perfect
> prediction, 0 means the model performs no better than predicting the mean,
> and negative values indicate the model is worse than the mean baseline.

Permutation importance, ALE, and SHAP plots are generated for both train
and test data.  For random and pixel holdout evaluations (which sweep
multiple seeds × test sizes), interpretability plots are generated only
for the first run.  Temporal and spatial LOO evaluations generate
interpretability plots for every holdout.

#### Target transform

When `LOG_TARGET=True`, the pipeline applies a `log1p(y)` transform to
the target variable before training and `expm1(ŷ)` to predictions
afterward.  This stabilizes the right-skewed withdrawal distribution and
improves tree-model performance on low-withdrawal pixels.  Because tree
leaf predictions become means of log-transformed values, the inverse
(`expm1`) yields the geometric mean, which is systematically lower than
the arithmetic mean (Jensen's inequality).  Global linear bias correction
(§3b) compensates for this shift.

#### CV scoring under log-space training

When the model trains in log space, R² is computed **in log space**
during CV scoring because R² is scale-invariant within the model's
operating space but *not* invariant under nonlinear transforms like
`expm1`.  Applying `expm1` before R² would deflate scores via Jensen's
inequality.  RMSE, MAE, and MBE, which have physical units (mm), are
still inverse-transformed to original scale so that CV error metrics
remain interpretable as percentage-of-mean withdrawal.

#### Optuna objective function

The Optuna TPE sampler minimizes a composite objective that balances
predictive accuracy, overfitting control, and fold stability:

```
objective = test_NRMSE × (1 + α × max(test_NRMSE / train_NRMSE − 1, 0)) + β × std(test_NRMSE)
```

where α = 0.1 and β = 0.05.

- **Primary term** (`test_NRMSE`): Mean normalized RMSE across CV folds.
  When `LOG_TARGET=True`, this is computed in **log space** (`log_nrmse`)
  so that the hyperparameter search optimizes in the same space as the
  tree model's internal MSE loss, giving equal weight to low- and
  high-withdrawal pixels.  When `LOG_TARGET=False`, original-scale NRMSE is
  used.
- **Overfitting ratio penalty** (`α`): Penalizes trials where
  `test_NRMSE ≫ train_NRMSE` using the *ratio* rather than the absolute
  difference.  A 20× train/test gap is penalized much more heavily than
  a 2× gap.  The `max(…, 0)` ensures no penalty when test ≤ train.
- **Fold variance penalty** (`β`): Penalizes inconsistent performance
  across CV folds, favoring hyperparameters that generalize uniformly
  across data subsets.

#### Linear bias correction (evaluation)

All five evaluation strategies attempt a **global linear bias correction**
after prediction for all tree-based models (XGB, LGBM, RF, XGBRF, ETR,
HGBR, GBR, ADA, BAG, CAT, and PIML variants).  Linear models (LR,
RIDGE, LASSO) are excluded since a post-hoc linear correction is
redundant with an already-linear predictor.  The correction is fit on
training data using OLS: `y_corrected = |m × y_pred + b|`, where `m`
and `b` minimize the squared residuals.  The absolute value ensures
physical non-negativity.

The correction is only applied if it improves **all** training metrics
(R², RMSE, and MAE) simultaneously.  If any metric worsens, the original
predictions are kept and a warning is logged.  Diagnostic plots and CSVs
are still generated so the BC effect can be inspected regardless.

**Strategy-consistent inner CV:** The inner cross-validation used during
Optuna hyperparameter tuning mirrors the outer evaluation strategy to
prevent optimistic CV scores from data leakage:

| Strategy | Inner CV | Group labels |
|---|---|---|
| Random (2a) | `RepeatedKFold` | — (standard random splits) |
| Pixel holdout (2a2) | `GroupKFold` | Pixel coordinates (easting/northing) |
| Temporal LOO (2b) | `GroupKFold` | Year |
| Spatial LOO (2c) | `RepeatedKFold` | — (standard random splits) |
| Spatial LOO seeded (2c-seed) | `RepeatedKFold` | — (standard random splits) |

For group-based strategies, `GroupKFold` ensures that all samples sharing
the same group label (pixel, year, or sub-basin) stay together in the same
fold, so the inner validation folds never leak spatial or temporal
information that the outer holdout was designed to test.  The number of
folds is capped at `min(FOLD_COUNT, n_unique_groups)`.

#### Step 2a — Random 80/20 split (`evaluate_random()`)

A grid evaluation over `EVAL_TEST_SIZES` (default 10 %–30 %) ×
`N_EVAL_SEEDS` (default 5) random seeds.  Each combination re-splits
the data with a different test fraction and seed, retrains with the
tuned hyperparameters, and evaluates.  Optuna tuning runs only on the
first combination (test_size=10 %, seed=42); all subsequent runs reuse
those hyperparameters.  Results are organized under
`ts{NN}/seed_{S}/` subdirectories, and `Model_Comparison_Averaged.csv`
reports mean ± std grouped by model and test size.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Random/`

#### Step 2a2 — Pixel holdout (`evaluate_pixel_holdout()`)

A basin-stratified pixel-level spatial holdout where a fraction of
unique spatial locations (identified by their easting/northing
coordinates) are held out across **all years**.  Pixels are stratified
by `GW_Basin` so that each basin contributes proportionally to the test
set.  Unlike the random split (where the same pixel may appear in both
train and test for different years), this strategy ensures the model is
evaluated on entirely unseen locations.  It provides a finer-grained
spatial generalization test than the basin-level spatial LOO (Step 2c),
revealing how well the model interpolates to new pixels within known
basins.

Like Step 2a, the evaluation runs over the same
`EVAL_TEST_SIZES` × `N_EVAL_SEEDS` grid with Optuna tuning on the
first combination only.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Pixel_Holdout/`

#### Step 2b — Temporal leave-one-out (`evaluate_temporal_loo()`)

Eight pre-defined temporal holdout configurations (T1_Baseline + T1–T7):

| Holdout | Withheld years | Training years | MIN_GW | Purpose |
|---|---|---|---|---|
| T1_Baseline | 2010–2020 | 2002–2020 (`TRAIN_YEAR_LIST_BASELINE`) | 0 | Backward compatibility with Majumdar et al. (2022) |
| T1 | 2010–2020 | 1984–2024 (`YEAR_LIST`) | `MIN_GW` | Mid-period block (same years, current config) |
| T2 | 2015–2024 | 1984–2024 | `MIN_GW` | Recent decade — forward extrapolation |
| T3 | 1990–1992, 2005–2007, 2022–2024 | 1984–2024 | `MIN_GW` | Scattered gaps — temporal interpolation |
| T4 | 2007–2010 | 1984–2024 | `MIN_GW` | Short mid-period block — interpolation from both sides |
| T5 | 1985–1989, 2020–2024 | 1984–2024 | `MIN_GW` | Both tails removed — no early/late anchoring |
| T6 | 2024 | 1984–2024 | `MIN_GW` | Single most recent year — forward extrapolation |
| T7 | 1984–1994 | 1984–2024 | `MIN_GW` | Early period — backward extrapolation |

`T1_Baseline` reproduces the settings from [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757): training
on 2002–2020 only, holding out 2010–2020, and including zero-withdrawal pixels
(`MIN_GW=0`).  The original study included zeros because publicly available
annual irrigation masks (e.g., [IrrMapper; Ketchum et al., 2020](https://doi.org/10.3390/rs12142328)) spanning the 2002–2020
period were not yet available when the study was conceptualized, so non-irrigated pixels could not be reliably
excluded.  All other holdouts use the full `YEAR_LIST` (1984–2024) and
the pipeline-level `MIN_GW` threshold (`None` by default, which excludes
zero-withdrawal pixels).

For each holdout, the model trains on the remaining years and is tested on
the held-out period.  Per-holdout metrics are recorded, then averaged across
all splits.  Heatmaps and bar plots (`vizops.plot_loo_heatmap()`,
`vizops.plot_loo_bar()`) visualize model performance across holdouts.

**Note on comparison with [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757):** The previous study tuned
hyperparameters directly on the test set (without cross-validation), making
its reported test metrics analogous to validation scores.  The current study
uses Optuna with `GroupKFold` (grouped by year), ensuring strict separation
between tuning and evaluation.  The current CV metrics show comparable or
improved R² and RMSE relative to the previous study's test metrics, while the current test metrics are only
marginally lower, confirming that the model generalizes well without
relying on test-set information during training.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Temporal_LOO/`

#### Step 2c — Spatial leave-one-out (`evaluate_spatial_loo()`)

Iterates over every AMA/INA management area (identified by
``GW_Basin_Type`` 0=AMA or 1=INA in the predictor DataFrame).  For
each basin the model trains on the rest of Arizona and is tested on the
held-out management area.  Basins with fewer than
``MIN_SPATIAL_EVAL_SAMPLES`` (default 30) non-zero metered samples or
listed in ``SKIP_SPATIAL_BASINS`` (currently WILLCOX AMA) are excluded.  Bias correction is **disabled** for spatial strategies because
a linear correction calibrated on training-basin residuals can actively
hurt when applied to a held-out basin with a different management regime.

Steps 2c, 2c-seed, and 2d also produce **stratified error metrics** that
bin test-set predictions by actual pumping magnitude (Low < 500 mm,
High ≥ 500 mm).  For each model and bin, R², RMSE, MAE, and MBE are
computed across holdout folds, revealing whether errors concentrate in a
particular pumping regime.  Results are saved to
`Stratified_Metrics.csv` with grouped bar charts (`Stratified_*.png`)
and a `Stratified_Sample_Counts.csv` showing bin populations.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Spatial_LOO/`

#### Step 2c-seed — Seeded spatial LOO (`evaluate_spatial_loo(seed_fraction=0.1)`)

A variant of the spatial LOO where 10 % of each held-out basin's
samples are randomly moved into the training set as a calibration
anchor; the remaining 90 % serve as the test set.
Motivated by [Asfaw et al. (2025)](https://doi.org/10.1016/j.agwat.2025.109691),
who showed that ML-based groundwater withdrawal predictions can perform
well with limited metering data, this tests whether a small amount of
local data is sufficient to correct the basin-specific pumping magnitude
offset that the pure LOO exposes.

Comparing pure LOO (Step 2c) with seeded LOO demonstrates that the
model captures the climate-driven temporal variability across basins but
requires a minimal local signal to calibrate the management-driven
pumping intensity.  Preliminary results show that adding just 10 % of
local data improves mean test R² from approximately −0.17 (pure LOO) to
+0.43 (seeded LOO) over the Douglas AMA, confirming that the negative R² in pure LOO is
driven almost entirely by a basin-specific magnitude offset — not by a
failure to capture the underlying hydrological process.  This supports
the argument that conventional spatial LOO is overly punitive for
groundwater pumping prediction, where management decisions (pumping
allocations, water rights, land-use regulations) are unobservable from
remote-sensing predictors.

**Outputs:** `{MODEL_DIR}Model_Evaluation/Spatial_LOO_Seed10/`

#### Cross-strategy summary

After at least three strategies complete,
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

The core production step.  Trains a single **XGBoost Random Forest (XGBRF) model** on **all**
metered data (1984–2024, no holdout) to maximize the training signal,
then predicts annual withdrawals for every 2 km pixel from 1896 to 2099.

**Why XGBRF?**  XGBRF is a hybrid of XGBoost and Random Forest.  Standard
XGBoost builds trees sequentially (each tree corrects the errors of the
previous one), which can overfit noisy metered records.  Standard Random
Forest trains independent trees on bootstrap samples (bagging), providing
strong variance reduction but lacking XGBoost's regularized split-finding.
XGBRF combines both: it grows a full forest of trees per boosting round
using bagging (`num_parallel_tree = N`), while each tree still benefits
from XGBoost's histogram-based, L1/L2-regularized split algorithm.  The
result is RF-style variance reduction with XGBoost-grade regularization.
Pixel holdout and temporal leave-one-out evaluations confirm that XGBRF
generalizes better to unseen locations and years than either pure XGBoost
or Random Forest.

**Absolute-value post-processing:** All predictions are wrapped in
`np.abs()` because withdrawal depth is physically non-negative.
Tree-based regressors can produce small negative values near zero
(numerical noise at the leaf level), and `abs()` ensures physical
validity.  The same transform is applied consistently in Optuna CV
scoring, uncertainty ensemble generation, and bias correction, so all
metrics and CIs are evaluated on the same transformed quantity.

**Temporal extrapolation caveat:** The model is trained on 1984–2024
metered data and predicts outside this range (1896–1983 hindcast, 2025
forecast, 2026–2099 projection).  Tree-based regressors cannot
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
| Projection | 2026–2099 | Future extrapolation |

Up to 2 000 pixels are subsampled per year (capped at 10 000 per era) and
passed through SHAP, ALE, and permutation importance analysis.  Comparing
the resulting feature-contribution profiles across eras reveals whether the
model relies on the same physical relationships in extrapolation as during
training.  Stable feature rankings and ALE shapes across eras support the
stationarity assumption; divergent patterns flag features whose
out-of-distribution behavior may reduce prediction reliability.  Outputs
are saved to `{prediction_dir}Model_Interpretability/{Era}/`.

#### 3b. Annual raster prediction loop (1896–2099)

Before the loop begins, an **out-of-distribution (OOD) detector**
(`mlops.OODDetector`) is fitted on a **climate/LULC subset** of the
AMA/INA training features (`x_train`).  Spatially-fixed features
(coordinates, well density, canal density, streamflow) are excluded so
that the detector measures **temporal novelty** (unprecedented climate or
land-use conditions) rather than geographic distance from AMA/INA.
This ensures irrigated non-AMA/INA areas (e.g. Yuma) with similar
climate/LULC profiles show low OOD, while pixels experiencing genuinely
novel conditions (extreme drought, future warming) are flagged regardless
of location.  The detector computes the Mahalanobis distance from the
training distribution and converts it to an OOD probability via the
χ²(n\_features) CDF.

OOD features used: ET, ETo, precipitation, temperature, AGRI density,
URBAN density, crop fraction, irrigation fraction, GW fraction, AWC,
Ksat, soil depth.  Excluded: easting, northing, well density, canal
density, canal-weighted streamflow, streamflow.

A **prediction exceedance check** complements the OOD detector from the
opposite direction.  While OOD flags feature-space extrapolation, the
exceedance check flags *output-space* implausibility: per-pixel predictions
exceeding the training-era maximum (or 99th percentile) withdrawal depth
indicate physically implausible rates, since modern pump infrastructure
operates near hydraulic efficiency limits (~75–85 %) and volumetric capacity
is unlikely to change substantially over the projection horizon.  Per-year
exceedance statistics are accumulated and written to
`Prediction_Exceedance_Summary.csv` with era-level summaries.

Before the loop, a **global linear bias correction** is learned from all
AMA/INA training data using `fit_linear_bc()`: `y_corrected = |m × y_pred + b|`,
where `m` and `b` are fit via OLS on training predictions vs observed values.
The correction is **only applied if it improves all three metrics** (R²,
RMSE, and MAE) on the training data, matching the same conditional logic
used in the LOO evaluation strategies.  A `Global_BC_Summary.csv` records
before/after metrics and a `BC_Applied` flag.

For each year the pipeline:

1. **Predicts** total annual withdrawals (mm) across all valid pixels.
1. **Applies global linear bias correction** via `apply_linear_bc()` (only
   if the correction improved R², RMSE, and MAE on training data).
2. **Checks prediction exceedance** against the training-era per-pixel
   maximum and P99 withdrawal depth.  Pixels exceeding these thresholds are
   counted per year.
3. **Flags out-of-distribution pixels** via the OOD detector.  Per-year
   probability rasters (`OOD_Flag_{year}.tif`, continuous values in [0, 1]
   where 0 = in-distribution and 1 = OOD) are written to `OOD_Rasters/`.
   Probabilities are derived from the χ²(n\_features) CDF of the squared
   Mahalanobis distance.  Per-year statistics (n\_ood, pct\_ood,
   mean/max Mahalanobis d²) are accumulated and written to
   `OOD_Rasters/OOD_Summary.csv` after the loop.  An
   `OOD_TimeSeries.png` plot shows OOD percentage by year with era
   shading.  Era-level OOD rates
   (hindcast 1896–1983, historical 1984–2025, projection 2026–2099) are
   logged with warnings when the mean OOD rate exceeds 10 %.
4. **Partitions** predictions into eight withdrawal categories via
   `partitionops.partition_predictions()`:
   Irrigation, Non-Irrigation, Irrigation\_GW, Irrigation\_SW,
   Non\_Irrigation\_GW, Non\_Irrigation\_SW, Total\_GW, Total\_SW.
   The irrigation / non-irrigation split uses **pump-capacity-weighted
   fractions** derived from the ADWR Well Registry. For each 2 km pixel
   and year, only wells installed by that year contribute:
   `irr_capacity_fraction = sum(PUMPRATE for IRRIGATION wells) /
   sum(PUMPRATE for all consumptive wells)`. Non-consumptive wells
   (monitoring, test, dewatering, drainage, remediation, mineral
   exploration, etc.) are excluded. Wells with missing pump rates are
   imputed using the per-`WATER_USE` category median (e.g. IRRIGATION
   ~900 gal/min, DOMESTIC ~15 gal/min), giving ~72% irrigation statewide.
   The per-year fraction is further adjusted at partition time by scaling
   each side by its area-fraction change relative to 2024:
   `irr_weight = irr_cap × (crop_frac / crop_frac_2024)`,
   `mi_weight = mi_cap × (urban_frac / urban_frac_2024)`,
   `irr_frac(y) = irr_weight / (irr_weight + mi_weight)`.
   This captures the 1950s ag boom (higher crop fraction → higher irr
   share) and future urbanisation (higher urban fraction → higher M&I
   share). Conservation is guaranteed: `Irrigation + Non_Irrigation =
   Total`. The **total** raster is recomputed bottom-up. The LULC
   source-mismatch (1984↔1985 USGS→NLCD and 2025↔2026 NLCD→USGS) is
   handled upstream via basin-scale delta correction baked into the
   parquet (see "Data harmonization" above).
5. **Computes consumptive use (CU)** using USGS NHM basin-level
   irrigation efficiencies:
   ```
   CU = IE × Irrigation_Withdrawal
   ```
   Basin-specific IEs are loaded from USGS NHM data (2000–2020).
   For years within that range, per-year basin IEs are used; for all
   other years, the long-term basin mean IE is applied.  CU is split
   into Irrigation\_CU, Irrigation\_GW\_CU, Irrigation\_SW\_CU using
   the GW fraction.
6. **Writes rasters** in four units for depth/volume products:

| Product | Units written | File naming |
|---|---|---|
| Total annual withdrawal | mm, ft, m³, AF | `Total_Predicted_{year}_{unit}.tif` |
| 8 withdrawal categories | mm, ft, m³, AF | `{Category}_{year}_{unit}.tif` |
| 3 CU categories | mm, ft, m³, AF | `{CU_Category}_{year}_{unit}.tif` |
| OOD probability | continuous [0, 1] | `OOD_Flag_{year}.tif` |

7. **Accumulates statistics** for AZ-wide, per-basin, and per-sub-basin
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
| Era summary maps | `vizops.create_era_summary_maps()` | Total, 8 categories, 3 CU  |

Four temporal eras are distinguished in the plots:

| Era | Years | Description |
|---|---|---|
| Hindcast | 1896–1983 | Pre-metered; predictions only. |
| Historical | 1984–2025 | Metered period; predictions vs. actuals. |
| Projected | 2026–2099 | Future projections. |

#### Graphical abstract / Figure 1

Generated in Step 3g (after UQ) so that augmented rasters and UQ-derived
σ_total are available.  `vizops.create_graphical_abstract()` produces a
three-panel publication figure:

- **Panel (a)**: Spatial map of mean-annual predicted withdrawal depth (mm)
  across all 204 years (1896–2099), with GW basin boundaries and
  highlighted AMA/INA regions on a Spectral\_r color ramp (2 %–98 %
  percentile range).  Also saved as a standalone GeoTIFF
  (`Mean_Annual_Predicted_mm.tif`).
- **Panel (b)**: Time series of total annual withdrawals with era shading
  and a ±1σ confidence band.  When UQ has run (Step 3b), the band uses
  σ\_total from `Uncertainty_Summary_Total.csv`; otherwise falls back to
  inter-basin spatial variability.  Left axis shows ×10⁶ m³, right axis
  shows ×1000 acre-ft.
- **Panel (c)**: Era mean bar chart with 95 % CI error bars, dual volume
  axes (m³ left, acre-ft right), and in-bar value labels.

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
and σ_gw (5 recent HarDWR well-density snapshots, 2020–2024) are
*sample-based*: their ensemble members are random draws from a larger
population, so Student's t-distribution critical values are used instead
of z = 1.96 to account for small-N estimation uncertainty
(t₉ = 2.262 for σ_model, t₄ = 2.776 for σ_gw).
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
the XGBRF model predicts total annual withdrawals, and σ_MACA is the per-pixel
sample standard deviation across the 5 predictions:

$$\sigma_{\text{MACA}}(x, y, t) = \text{std}\bigl[\hat{y}_{\text{GCM}_1}, \ldots, \hat{y}_{\text{GCM}_5}\bigr]$$

For historical years (1896–2025), σ_MACA = 0 because observations replace
GCM projections.

**Climate input spread diagnostics.**  During the σ_MACA loop, the AZ-mean
values of ET, ETo, and Peff are recorded for each GCM and year.  A 3-panel
ribbon plot (`Climate_Input_Spread.png`) and per-variable CSVs are saved
to `Sigma_MACA/Climate_Input_Spread/`, showing how the raw climate inputs
diverge across the 5 GCMs before they propagate through the XGBRF model.

##### σ_model — XGBRF seed ensemble (all years, 1896–2099)

Ten XGBRF models are trained on the full metered dataset (1984–2024) with
different random seeds, reusing the Optuna-tuned hyperparameters from the
full-prediction step (no re-tuning per seed):

```
Seeds: 7, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144
```

This isolates stochastic model uncertainty (random initialization, bagging
sampling) from hyperparameter uncertainty.  For each year and pixel,
σ_model is the sample standard deviation of the 10 seed predictions:

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
2. **Finite-difference sensitivity** — the XGBRF model is evaluated at
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
2. Per-scenario basin-level urban and ag fractions are computed from each
   year's scenario LULC, and per-basin **delta factors** are derived
   relative to each scenario's own 2026 baseline (the FORE-SCE year
   paired with the NLCD 2025 anchor).
3. Each scenario's `URBAN`, `AGRI`, `annual_crop_fraction`, and
   `annual_urban_fraction` columns are set to the **NLCD 2025** pixel
   values × the scenario's basin-scale delta for that year. This keeps
   the spatial pattern consistent with NLCD while capturing the
   scenario's own temporal trajectory (same correction methodology as the
   main pipeline).
4. `annual_irr_fraction` is re-derived from the corrected crop_fraction
   via the regression model.
5. The XGBRF model predicts total annual withdrawals under each scenario.

σ_LULC is the sample standard deviation across the 4 scenario predictions:

$$\sigma_{\text{LULC}}(x, y, t) = \text{std}\bigl[\hat{y}_{B1}, \hat{y}_{B2}, \hat{y}_{A1B}, \hat{y}_{A2}\bigr]$$

Because σ_LULC re-derives the entire irrigation-fraction chain per
scenario, it fully subsumes σ_irr for future years.

In addition to the pixel-level σ, per-scenario **volume projections**
(2026–2099) are saved for total and all 8 category products in
`Sigma_LULC/Scenario_Volumes/`.  A combined
`Scenario_Comparison.csv` provides all four scenarios side by side for
plotting scenario-dependent volume trajectories.

##### σ_gw — Well-density feature sensitivity (all years, 1896–2099)

σ_gw measures ML-feature sensitivity of predicted pumping to recent
year-over-year variability in HarDWR well-registry counts.  For each
prediction year, ``well_density`` — the **#1 feature by mean |SHAP
value|** in the XGBRF model — is swapped with values observed in each
of five recent reference years (``INFRASTRUCTURE_SNAPSHOT_YEARS = [2020,
2021, 2022, 2023, 2024]``), the model is re-run on the modified feature
matrix, and σ_gw is the sample std across the five predictions:

$$\sigma_{\text{gw}}(x, y, t) = \text{std}\bigl[\hat{y}_{\text{wd}=2020}, \hat{y}_{\text{wd}=2021}, \hat{y}_{\text{wd}=2022}, \hat{y}_{\text{wd}=2023}, \hat{y}_{\text{wd}=2024}\bigr]$$

The sample std is computed with ``ddof = 1`` (N − 1 = 4 degrees of
freedom) and scaled by the Student's t correction
``T_SCALE_GW = t₄ / z = 2.776 / 1.96 ≈ 1.416`` before entering the
quadrature, so the final σ_total reflects a t-corrected 95 % CI rather
than a z-based one.

``well_density`` was chosen because it dominates the model's learned
response to GW infrastructure (mean |SHAP| ≈ 24 mm, roughly 8× the
contribution of the Hung ``annual_gw_fraction`` feature that earlier
versions of this component perturbed).  Replacing a perturbation of a
rank-12 feature with a perturbation of the rank-1 feature at the same
input CV gives a substantially larger σ_gw signal across the entire
1896–2099 window — the previous Hung-based implementation collapsed to
~0 outside 2000–2015, whereas the new one is nonzero everywhere because
the same five reference snapshots are loaded and swapped in regardless
of the prediction year.

**Known limitations — σ_gw scope:** ``sw_rights_density`` is *not*
perturbed.  It is rank 15 in SHAP importance (~2 mm, about 12× less
influential than ``well_density``) and is effectively frozen in the
HarDWR record after ~1996: per-pixel std across 2020–2024 is literally
0.000 because SW water rights in Arizona were almost entirely filed by
the mid-1990s and only ~400 new entries have appeared in the 28 years
since.  Probing it would contribute negligibly to σ_total while
requiring counterfactuals that don't exist in the data, so it is
omitted with this rationale documented in the docstring.

σ_gw does not represent new-well drilling, well retirement, or spatial
redistribution of infrastructure beyond what the five reference years
already contain — those are out of scope absent structural scenario
projections for Arizona well infrastructure.  For prediction years far
from 2020–2024 the counterfactuals become increasingly abstract ("what
if 2099 Arizona had 2020-era wells"), but because ``well_density`` is
the dominant SHAP feature, probing it at real observed amounts still
exercises the model's primary sensitivity axis.

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
│   ├── Scenario_Volumes/              # Per-scenario volume projections
│   │   ├── Total_{B1,B2,A1B,A2}.csv  #   Per-scenario total volumes
│   │   ├── {Category}_{scenario}.csv  #   Per-scenario category volumes
│   │   └── Scenario_Comparison.csv    #   All scenarios combined
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

##### σ_CU — Consumptive-use uncertainty (error propagation)

Consumptive use is defined as CU = IE × Withdrawal, where IE (irrigation
efficiency) comes from USGS NHM basin-level data (2000–2020).  σ_CU is
computed via error propagation from two sources:

```
σ_CU = √((IE × σ_wd)² + (wd × σ_IE)²)
```

where σ_wd is the per-category total withdrawal uncertainty (from the
augmented Irrigation rasters, band 2), and σ_IE is the inter-annual
standard deviation of NHM basin-level IE across 2000–2020.

For NHM-covered years (2000–2020), per-year basin IEs are used and
σ_IE = 0 (observed efficiency).  For all other years, the basin mean
IE is applied with σ_IE equal to the basin-level temporal std.

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
| **Total annual withdrawal** (32 rasters/yr) | σ_total × unit scale | mm, ft, m³, AF | σ_total computed in mm, scaled by conversion factor per unit |
| **8 withdrawal categories** (256 rasters/yr) | σ_cat via quadrature of per-category ensemble spreads | mm, ft, m³, AF | Each ensemble member is partitioned before computing std |
| **3 CU categories** (48 rasters/yr) | σ_CU (error propagation) | mm, ft, m³, AF | σ_CU in mm, scaled to target unit |

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
UQ component’s ensemble.  This correctly propagates feature-input
uncertainty (σ_irr perturbs ``irr_fraction`` and σ_gw perturbs
``well_density`` across recent HarDWR snapshots, so per-category
spreads differ from simple linear scaling of the total σ).

**Execution order** (dependencies require sequential processing):

1. Compute σ_total → augment total prediction rasters (all 4 units)
2. Augment category rasters (reads augmented total rasters for σ)
3. Compute σ_CU (reads augmented category rasters for σ_wd) → augment CU rasters

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
12 product groups (total, 8 categories, 3 CU):

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

All plots are written to the `Visualizations/` directory, overwriting any
earlier plots from Step 3c that lacked uncertainty bounds.

#### 3g. Raster maps for all output categories (`create_all_raster_maps()`)

Generates publication-quality spatial maps for **every** raster output
product in the pipeline.  Three types of maps are produced:

**Era-mean maps** (`vizops.create_era_raster_maps()`) — A 2×2 panel figure
for each raster category showing the temporal mean within each of the four
eras (Hindcast, Historical, Projection).  Groundwater basin
boundaries (thin gray) and AMA/INA basins (bold dark + labels) are overlaid
on every panel.  No-data pixels appear as gray background.

| Category group | Colormap | Count |
|---|---|---|
| Total predicted withdrawal + 8 partition categories + 3 CU | `YlOrRd` | 12 figures |
| OOD probability (mean) | `RdYlGn_r` | 1 figure |
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
  colormap centered on zero.

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

Trend maps are generated for: total predicted withdrawal, 8 partition
categories, 3 CU categories, and 3 SW capture categories — each with
up to 5 periods (full + 3 eras).

**Basin-level trend choropleth** — For each category × period, a
choropleth map colors each GW basin by its mean Sen's slope (blue =
decreasing, red = increasing).  Every basin is labeled with its name,
trend direction arrow (↑/↓), and slope value.  AMA/INA basins are bold;
others are normal weight.  These maps provide an at-a-glance view of
which basins are increasing or decreasing across the state.

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
(pixel-level trend maps, basin choropleth maps, and zonal statistics
CSVs).

#### 3e. Well package (`create_well_package_step()`)

`wellops.create_well_package()` disaggregates pixel-level prediction
rasters to individual wells from the ADWR Well Registry and writes four
GeoParquet files — one per unit (mm, ft, m³, AF).  Each file stores
per-well withdrawal predictions and uncertainty with WKB point geometry,
readable by QGIS (≥ 3.28) and GeoPandas.  Writing is chunked (10 years
per batch via PyArrow's `ParquetWriter`) to avoid out-of-memory errors
on the ~35M row dataset.  For each year, only wells that existed by that
year are included (see temporal filtering below).  See the `wellops`
module description below for the capacity-proportional distribution logic.

This step runs **after** UQ augmentation (Step 3b) so that the 6-band
augmented rasters are available.  When augmented rasters are present,
per-well uncertainty columns are included:

| Column pattern | Description |
|---|---|
| `{Cat}_{unit}` | Prediction (capacity-weighted share of pixel value) |
| `{Cat}_{unit}_sigma` | σ (capacity-weighted share of pixel σ) |

Categories include 9 withdrawal categories (Total + 8 partitions),
3 CU categories (Irrigation_CU, Irrigation_GW_CU, Irrigation_SW_CU),
and 3 SW capture categories (Total_SW_Capture, Irrigation_SW_Capture,
Non_Irrigation_SW_Capture) — when the SW Capture rasters are available.
The SW capture category names refer to the surface water captured by
each GW pumping pool: e.g. `Total_SW_Capture` is "SW captured by Total
GW pumping" within the parent `SW_Capture/` folder context.

**Caveat:** Per-well σ assumes pixel-level uncertainty distributes
proportionally to capacity weight.  This is a simplification — true
per-well uncertainty would require well-specific error models.

**Outputs:** `{MODEL_DIR}Full_Prediction_XGB/Well_Package/`

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
- Spatial difference maps (diverging colormap centered on zero).
- Temporal agreement visualizations (`Temporal_Agreement/`):
  - **Heatmaps** — Basin × pair grids colored by Pearson r and NSE.
  - **Box/violin plots** — Distribution of per-basin r/NSE across pairs.
  - **Taylor diagrams** — Correlation vs normalized std dev in polar
    coordinates, with centered RMSD contours.
  - **r vs NSE scatter** — Paired scatter with quadrant annotations
    identifying basins with good/mixed/poor agreement.

All outputs are written to `{prediction_dir}Intercomparison/`.

#### Step 4b — CU intercomparison (`run_cu_usgs_intercomparison()`)

Compares ML-based Irrigation Consumptive Use with USGS NHM HUC12-scale
data ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)) at the basin scale:

| Product | ML source | USGS source |
|---|---|---|
| **CU** | `Irrigation_CU_Rasters/Depth_mm/` (mm) | `Irr_CU_HUC12_Tot_annual_2000_2020.csv` (Mgal/d) |

CU follows the same volume-based framework as withdrawals (RMSD, MAD, %
Difference in AF, m³, mm).  Outputs include metrics CSVs, per-basin
tables, time series, and scatter plots, written to
`{prediction_dir}CU_Intercomparison/`.

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
| **Peff (USDA SCS)** | Predictor band 4 × `irr_fraction` | 2000–2024 | 2 km rasters |
| **ML Peff (PCML)** | Predictor band 5 × `irr_fraction` | 2000–2023 | 2 km rasters |
| **NHM PPTeff** | USGS NHM HUC12 data ([Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)) (Mgal/d) | 2000–2020 | HUC12 polygons |

All three datasets are scaled by `annual_irr_fraction` so that volumes
represent only the irrigated-area contribution.  NHM PPTeff follows the
same CSV → rasterize → basin-aggregate pipeline as NHM CU, with irrigated-
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
- Temporal agreement visualization (heatmap, box/violin, r-vs-NSE).

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
  stacks them with basin labels and observed withdrawals into a single DataFrame.
- **`create_train_test_data()`** — Splits the DataFrame into train/test
  sets using one of four strategies (temporal, spatial, random ratio,
  random 80/20).

### `mlops.py` — Machine learning operations

Builds, tunes, evaluates, and interprets ML models including standard
baselines (XGB, LGBM, RF, XGBRF) and optional ensemble/linear models
(ETR, HGBR, GBR, ADA, BAG, CAT, LR, RIDGE, LASSO) when
`INCLUDE_ALL_MODELS=True`.  Also contains optional physics-informed
wrapper estimators (PIML_XGB, PIML_LGBM, PIML_XGBRF) that are disabled
by default — see [note on PIML models](#note-on-physics-informed-piml-models).

Key functions:
- **`get_model_dict()`** — Returns model objects.  Use `skip_piml=True`
  (default in pipeline) to exclude PIML variants.
- **`build_ml_model_optuna()`** — Trains a single model with Optuna
  TPE-based hyperparameter search parallelized across Dask workers.
- **`compare_all_models()`** — Trains all models on a common split and
  ranks them by test R².  Optionally generates interpretability plots
  (permutation importance, ALE, SHAP) for both train and test data.
- **`generate_interp_plots()`** — Generates permutation importance, ALE,
  and SHAP plots (both train and test) for a trained model.
- **`fit_linear_bc()`** — Fits a linear bias correction (`y = m × pred + b`)
  via OLS on training predictions vs observed values.  Returns slope and
  intercept.
- **`apply_linear_bc()`** — Applies a pre-fitted linear bias correction:
  `|m × predictions + b|`.
- **`get_prediction_results()`** — Makes predictions and optionally applies
  linear bias correction (boolean `apply_bias_correction` flag).  BC is
  applied to all tree-based models; linear models (LR, RIDGE, LASSO) are
  skipped.
- **`perform_bias_correction()`** — Fits and applies global linear bias
  correction using `fit_linear_bc()` / `apply_linear_bc()`.  The
  correction is accepted only if R², RMSE, and MAE all improve on
  training data.
- **`calc_train_test_metrics()`** — Computes R², normalized RMSE (% of mean),
  normalized MAE (% of mean), and normalized MBE (% of mean).
- **`compute_perm_imp()`**, **`compute_ale_plots()`**,
  **`compute_shap_plots()`** — Model interpretability diagnostics.
- **`generate_model_visualizations()`** — Scatter, residual, and time series
  plots per model.
- **`OODDetector`** — Mahalanobis distance-based out-of-distribution
  detector.  Fitted on a climate/LULC subset of AMA/INA training features
  (excluding coordinates, well/canal density, streamflow), it produces
  per-pixel OOD probabilities (continuous [0, 1] via χ² CDF) that
  measure temporal novelty — how different a pixel's climate and land-use
  conditions are from the 1984–2025 training era.  Used in
  `predict_full_period()` to write per-year OOD rasters, a summary CSV
  with era-level diagnostics, and a time-series plot.

### `visualops.py` — Visualization

Produces journal-quality figures for every stage of the pipeline.

Key functions:
- **`explore_az_data()`** — Exploratory data analysis (histograms + KDE,
  boxplots by era/basin type/GW basin, violin plots, time series, and
  boxplots by year for all numeric columns).
- **`analyze_pumping_distribution()`** — Withdrawal depth distribution analysis
  with summary statistics, Tukey's fence outlier benchmarks, and empirical
  CDF plot (separate curves per depth threshold vs. no threshold).
- **`create_full_period_time_series()`** — Annual withdrawal line plot (1896–
  2099) with era shading and optional observed-data overlay.
- **`create_era_summary_maps()`** — Spatial maps of mean depth for each era
  (Hindcast, Historical, Projection).
- **`create_basin_time_series()`** / **`create_subbasin_time_series()`** —
  Per-basin and per-sub-basin annual trends with AMA/INA color coding.
- **`plot_loo_heatmap()`** / **`plot_loo_bar()`** — Heatmaps and bar plots
  for leave-one-out evaluation results.
- **`create_cross_strategy_summary()`** — Side-by-side comparison of Random,
  Temporal LOO, and Spatial LOO results with R², RMSE, MAE, MBE, and
  Overfit R².  Produces CSV, LaTeX (`booktabs`), and grouped bar chart.
- **`create_graphical_abstract()`** — Three-panel Figure 1: (a) mean-annual
  withdrawal depth map with GW basin boundaries, (b) annual withdrawal time series
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
  Rasterizes withdrawal volumes (AF) and converts to depth (mm).
- **`crop_gw_rasters()`** — Clips GW rasters to the Arizona boundary.
- **`create_gw_basin_rasters()`** — Rasterizes ADWR basin and sub-basin
  polygons.
- **`create_land_use_data()`** — Gaussian-filters the LULC raster to
  produce continuous AGRI, SW, and URBAN density features.  **Known design
  choice:** The density features are independently min–max normalized to
  [0, 1] within each year, so the model sees only within-year spatial
  patterns, not temporal magnitude trends.  This is partially mitigated by
  `annual_crop_fraction`, `annual_urban_fraction`, and
  `annual_irr_fraction`, which provide year-to-year magnitude signals.
- **`create_well_density_raster()`** — Creates per-year well-count-per-pixel
  rasters from the Well Registry.  Non-consumptive wells (monitoring, test,
  dewatering, drainage, remediation, mineral exploration, unknown, reserved,
  and wells with no `WATER_USE` attribute) are excluded.  For each year,
  only wells installed by that year are included, using `INSTALLED` date
  (with `APPLICATIO` date fallback; wells missing both dates default to
  `start_year`).  The well density raster is used as a spatial mask in
  partitioning — pixels with zero wells are set to NaN.  The temporal
  filtering ensures that hindcast years do not include predictions at
  pixels where wells did not yet exist.  **Remaining limitation:**
  projection years (post-2024) use the 2024 well inventory, so they
  cannot account for future well retirement or new drilling.
- **`create_irr_capacity_fraction_raster()`** — Creates per-year per-pixel
  irrigation pump-capacity fraction rasters from the Well Registry.
  For each pixel and year, only consumptive wells installed by that year
  contribute: `frac = sum(PUMPRATE for IRRIGATION wells) / sum(PUMPRATE
  for all consumptive wells)`. Wells with missing PUMPRATE are imputed
  using the per-`WATER_USE` category median (~900 gal/min for IRRIGATION,
  ~15 gal/min for DOMESTIC), giving ~72% irrigation statewide.  The
  per-year fraction is further adjusted at partition time by scaling each
  side by its temporal crop/urban area-fraction change relative to 2024
  (see Step 3 partitioning).  **Remaining limitation:** same as well
  density — projection years use the 2024 well inventory.

### `streamflowops.py` — Streamflow & canal data

Downloads and processes streamflow data from USGS ([Hodson et al., 2023](https://doi.org/10.5066/P94I5TX3)) and USBR ([Gangopadhyay & Pruitt, 2011](https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf); [USBR, 2025](https://rise-usbr.opendata.arcgis.com/)) sources.

Key functions:
- **`download_streamflow()`** — Downloads monthly streamflow records from
  USGS gauges and retrieves USBR delivery data.
- **`create_streamflow_rasters()`** — Rasterizes annual streamflow volumes
  onto the 2 km grid using watershed polygons.
- **`create_canal_density_raster()`** — Rasterizes canal geometry from the
  GRAIN dataset ([Suresh et al., 2026](https://doi.org/10.5194/essd-18-1855-2026))
  into a canal-density layer used for SW-fraction estimation.

### `partitionops.py` — Water-budget partitioning

Decomposes total annual withdrawal predictions into eight categories using
ancillary data already in the predictor stack:

| Category | Derivation |
|---|---|
| **Irrigation** | `total × irr_fraction` (pump-capacity-weighted or area-based) |
| **Non_Irrigation** | `(total − Irrigation) × URBAN` (Gaussian-smoothed urban density, 0–1) |
| **Total** | `Irrigation + Non_Irrigation` (recomputed after urban weighting) |
| **Irrigation_GW** | `Irrigation × irr_gw_frac` (density-ratio, see below) |
| **Irrigation_SW** | `Irrigation − Irrigation_GW` |
| **Non_Irrigation_GW** | `Non_Irrigation × nonirr_gw_frac` (density-ratio, see below) |
| **Non_Irrigation_SW** | `Non_Irrigation − Non_Irrigation_GW` |
| **Total_GW** | `Irrigation_GW + Non_Irrigation_GW` |
| **Total_SW** | `Irrigation_SW + Non_Irrigation_SW` |

**Density-ratio GW/SW split with CW-weighted smoothing:** The GW/SW
split uses ADWR well density against HarDWR SW rights density weighted
by canal-weighted streamflow and Gaussian-smoothed (σ=4, ~8 km radius)
to spread each SW POD's influence over its canal service area:

```
sw_smoothed    = gaussian_filter(irr_sw_rights_density × canal_weighted_streamflow, σ=4)
irr_gw_frac    = irr_well_density    / (irr_well_density    + sw_smoothed)
nonirr_gw_frac = nonirr_well_density / (nonirr_well_density + nonirr_sw_smoothed)
```

A POD at a major canal headgate (high cw_streamflow) gets proportionally
more influence than a POD at a dry wash — the product
`POD_count × cw_streamflow` scales the SW rights by actual delivery
capacity.  The Gaussian spread distributes that capacity-weighted
influence across the canal service area, solving the spatial mismatch
between point-source SW PODs and diffuse GW wells.  Both well and
SW-rights densities are per-year rasters gated by installation/priority
dates, so the GW/SW balance evolves as infrastructure builds out.
Where the denominator is zero, `gw_frac` defaults to 1.0 (100 % GW).

The [Hung et al. (2025)](https://doi.org/10.1038/s41597-025-05920-x)
`annual_gw_fraction` is retained as an ML feature but no longer used
for partitioning.  A partition-level sensitivity diagnostic
(`run_density_ratio_sensitivity`) probes two orthogonal knobs and
writes both as sections of one CSV with a `Perturbation_Type` column:

- **Density** — well and SW-rights densities perturbed simultaneously
  with opposite signs (well × (1 ± 0.2), sw_rights × (1 ∓ 0.2)) to
  probe the GW/SW ratio's sensitivity to coordinated scaling of its
  numerator and denominator.
- **Smoothing** — the Gaussian canal-reach kernel `sw_smooth_sigma` is
  swept across {2, 8} (~4 km / ~16 km radius at 2 km resolution) with
  densities held at baseline, to probe the assumed canal service-area
  extent.

Both sections write to `Uncertainty/Sigma_GW/Density_Ratio_Sensitivity.csv`
with time-series ribbon plots in `Density_Ratio_Sensitivity.png` and
`Smoothing_Sigma_Sensitivity.png`.

**Zero-surface-water constraint:** Where `canal_weighted_streamflow_mm`
is zero at a pixel, there is no canal-delivered surface water, and
`gw_frac` is forced to 1.0.  This is a simple per-pixel check — no
basin-median override is needed because canal-weighted streamflow is
precisely located at canal infrastructure with no watershed bleed
(unlike regular streamflow, which is uniform per-watershed and can
assign Colorado River flow to desert basins like Butler Valley).
Affected pixels include all of Willcox AMA, Butler Valley, Parker,
Ranegras Plain, and most of McMullen Valley.  Canal-served pixels
in any basin retain their density-ratio + canal-boost split.

**Physics-constrained input data correction:** Published datasets are
treated as informative priors, not ground truth.  For example, the
[Hung et al. (2025)](https://doi.org/10.1038/s41597-025-05920-x)
GW-fraction snapshots report values as low as 0.7 in Willcox AMA,
implying 30 % surface-water irrigation in a closed basin with no river
or canal infrastructure.  This is physically impossible — Willcox is an
endorheic playa where the only "surface water" visible in LULC is mining
tailings ponds (redistributed groundwater, confirmed as GW-sourced by
HarDWR water rights records).  The density-ratio approach inherently
avoids this issue: Willcox has many GW wells but zero SW rights and
zero canal-weighted streamflow, so `gw_frac` → 1.0 without requiring
any override.

**Surface Water Capture Index:** After partitioning, the pipeline
computes a per-pixel, per-year capture index quantifying how much GW
pumping likely depletes surface water, using water table depth
([Ma et al., 2026](https://doi.org/10.1038/s43247-025-03094-3)) and
canal-weighted streamflow:

```
capture_fraction = exp(-wtd_m / λ) × cw_norm
sw_capture_mm    = GW_withdrawal × capture_fraction
```

Three λ values (5, 10, 20 m) produce lower/central/upper bounds.
Computed separately for Total_GW, Irrigation_GW, and
Non_Irrigation_GW.  Output in all 4 units (mm, ft, m³, AF).

Key helpers:
- **`focal_fill_irr_fraction()`** — fills edge-pixel gaps (`irr_frac < 0.05`)
  with a focal mean of valid neighbors, avoiding NaN propagation along
  irrigated-area boundaries.
- **`compute_sw_fraction()`** — normalizes a density array to [0, 1] using a
  local-maximum filter (`maximum_filter(size=5)`).  Used for focal-max
  normalization of canal-weighted streamflow (`cw_norm`).
- **`compute_sw_capture_index()`** — computes per-pixel SW capture fraction
  and volume at three λ values with uncertainty bounds.
- **`partition_predictions()`** — orchestrates all splits (irr/nonirr,
  GW/SW with density-ratio + canal boost, zero-SW constraint), applies
  well-density masking, and returns a dict keyed by the eight category names.

All partitions use subtraction from the parent total (e.g., `nonirr = total − irr`)
to guarantee exact budget closure with no floating-point drift.

### `uncertaintyops.py` — Hybrid uncertainty quantification

Computes pixel-level prediction uncertainty for all products (total
annual withdrawals, withdrawal categories, consumptive use) and writes augmented
6-band GeoTIFFs.

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
- **`compute_sigma_model()`** — XGBRF 10-seed ensemble spread (all
  years).  Parallelized via Dask + Optuna.  Returns per-year total σ and
  per-category σ.
- **`compute_sigma_irr()`** — Irrigation fraction sensitivity (historical
  only, 1896–2025).  Uses IrrMapper vs regression finite-difference with
  half-range mode.  Returns per-year total σ and per-category σ.
- **`compute_sigma_lulc()`** — LULC projection spread (4 USGS scenarios,
  future only, 2026–2099).  Re-derives the full LULC → crop_frac →
  irr_frac chain per scenario.  Returns per-year total σ and per-category σ.
- **`compute_sigma_gw()`** — ML-feature sensitivity to recent HarDWR
  well-registry variability.  Swaps `well_density` (#1 SHAP feature)
  with values from each of 5 recent reference years (2020–2024) and
  takes the sample std across the 5 predictions.  Does not cover
  `sw_rights_density` sensitivity (the feature is rank 15 in SHAP and
  flat post-1996).  Returns per-year total σ and per-category σ.
- **`compute_sigma_total()`** — Quadrature combination of all five
  components for both total and per-category σ.  Writes 2-band total σ
  rasters (σ, CV), per-category σ rasters, and temporal Mean_CV.tif.
- **`compute_basin_sigma_total()`** — Reads per-component basin/sub-basin
  σ CSVs and combines via quadrature into `Basin_Sigma_Total.csv` /
  `Subbasin_Sigma_Total.csv`.
- **`compute_sigma_cu()`** — CU uncertainty via error propagation from
  CU = IE × Withdrawal.  Reads augmented per-category withdrawal σ (band 2)
  and basin-level NHM IE std (2000–2020).  Writes per-category σ_CU rasters
  (Irrigation_CU, Irrigation_GW_CU, Irrigation_SW_CU).
- **`run_density_ratio_sensitivity()`** — Partition-level diagnostic
  covering two orthogonal knobs: a density-ratio sweep (well and
  SW-rights densities perturbed with opposite signs at ±20%) and a
  smoothing-kernel sweep (`sw_smooth_sigma ∈ {2, 8}`).  Writes one CSV
  with a `Perturbation_Type` column and two ribbon-plot PNGs.
- **`augment_prediction_rasters()`** — Rewrites total annual withdrawal rasters as
  6-band GeoTIFFs (pred, σ, CV, SNR, lower CI, upper CI) for all 4 units.
- **`augment_category_rasters()`** — Augments 8 withdrawal category rasters
  using per-category σ_total rasters computed directly from ensemble
  spreads (not fraction-scaled from total σ).
- **`augment_cu_rasters()`** — Augments 3 CU category rasters using σ_CU.
- **`_replot_from_augmented_rasters()`** — Regenerates all time-series
  plots (AZ-wide, per-basin, per-sub-basin) with 95 % CI uncertainty
  bounds by reading the 6-band augmented rasters via zonal statistics
  (`rasterio.mask`).  Replaces the earlier non-uncertainty time-series
  plots from Step 3c.
- **`_plot_component_basin_sigma()`** — Generates per-component (MACA,
  Model, Irr, LULC, GW) basin and sub-basin σ time-series plots with
  dual y-axes (m³/AF) and era shading.

### `wellops.py` — Well-level withdrawal package

Disaggregates pixel-level withdrawal and CU rasters to individual wells
from the ADWR Well Registry and writes four GeoParquet files — one per
unit (`Well_Package_mm.parquet`, `_ft`, `_m3`, `_AF`) — with WKB point
geometry, compatible with QGIS and GeoPandas.

**Sampling**: Only the **mm** rasters are read (12 categories per year:
9 withdrawal + 3 CU); ft, m³, and acre-ft values are computed
arithmetically, reducing I/O by 75 %.  When augmented 6-band rasters are
available (after Step 3b), band 2 (σ) is also sampled and per-well
uncertainty columns (σ, lower/upper 95 % CI) are included for every
category and unit.

**Temporal filtering** — for each year, only wells that existed by that year
are included in the disaggregation.  A well's start year is determined by:

1. **`INSTALLED` date** (year extracted from the Well Registry).
2. **`APPLICATIO` date fallback** if `INSTALLED` is missing or invalid.
   The application date serves as a lower bound — if a permit was filed, the
   well was likely drilled within a reasonable timeframe, making it a better
   estimate than the full-history default.
3. **Conservative default** (`start_year`) if both are missing — the well is
   included for all years.

Dates equal to the ADWR sentinel value `1899-12-30` (meaning "unknown") are
treated as missing and fall through to the next tier.

Capacity weights are re-normalized per year within each pixel using only the
active wells, so the pixel total is always fully distributed.

**Distribution logic** — when multiple wells share a 2 km pixel, the pixel
total is split using capacity-proportional weights with a three-tier fallback:

1. **Historical pumping** — mean `AF Pumped` across all years a well appears
   in the per-year GW shapefiles (`GW_YYYY.shp`).  These cover metered wells
   within AMA/INA management areas (~3 k wells/year, 1984–2024).
2. **PUMPRATE fallback** — for unmetered wells, the `PUMPRATE` field (GPM)
   from the Well Registry is used (~79 k wells have this attribute).
3. **Equal-share fallback** — wells with neither record receive weight 1.0.

**Nodata masking**: Wells landing in raster nodata or out-of-bounds pixels
are dropped before weight computation, preventing valid wells from losing
share to neighbors in invalid pixels.

**Zero floor**: A `np.maximum(all_mm, 0)` clamp is applied after sampling to
eliminate any negative model artifacts before unit conversion.

### `intercompops.py` — USGS intercomparison

Basin-scale comparison of ML predictions with independent USGS datasets.

**Withdrawal intercomparison** (`run_intercomparison()`):
- Loads ML, NHM ([Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM); [Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)), and Reitz ([Reitz et al., 2023](https://doi.org/10.5066/P9EZ3VAS)) data; aggregates to basin volumes (AF).
- Computes pairwise RMSD, MAD, Percent Difference.
- Computes interannual temporal agreement (per-basin Pearson r and NSE).
- Produces per-basin time series, scatter plots, spatial difference maps,
  and temporal agreement visualizations (heatmaps, box/violin plots,
  Taylor diagrams, r-vs-NSE scatter).

**CU intercomparison** (`run_cu_intercomparison()`):
- Compares ML CU (mm) with NHM HUC12 annual data ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)).
- CU: Mgal/d → m³/yr → depth (mm) → basin volumes (AF).
- Produces metrics, per-basin tables, time series, and scatter plots.

**CAP/SRP validation** (`run_cap_srp_validation()`):
- Compares ML Total SW predictions with observed CAP + SRP delivery records.
- Filters CAP to direct-use only; SRP to Surface Water (+ optional Spill
  Water sensitivity).
- Produces per-basin time series, scatter plots, and validation metrics.

**Peff intercomparison** (`run_peff_intercomparison()`):
- Compares USDA-SCS Peff ([USDA SCS, 1993](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf), band 4) and Peff PCML ([Hasan et al., 2025](https://doi.org/10.1016/j.agwat.2025.109821), band 5) with NHM PPTeff ([Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ)).
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
- **`shp2raster()`** / **`shps2rasters()`** — Rasterizes vector features
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

## Physics-constrained approach

Rather than embedding physics directly into the ML loss function (see
[PIML models](#note-on-piml-models) below), the pipeline enforces physical
knowledge at two natural boundary points — **input-side feature engineering**
and **output-side post-processing** — while leaving the ML model free to
learn flexible nonlinear mappings from physically meaningful features to
total withdrawal depth.

### Input side — hydrology-aware feature engineering

Physical and infrastructure knowledge is encoded into the predictor features
before the model sees the data:

- **Temporally varying well density** — per-year well counts gated by
  ADWR Well Registry `INSTALLED` / `APPLICATIO` dates, with non-consumptive
  wells (monitoring, test, dewatering, drainage, remediation, mineral
  exploration) excluded via keyword matching.
- **Pump-capacity-weighted irrigation fraction** — per-year
  `irr_capacity_fraction` from registry PUMPRATE, temporally scaled by
  crop/urban area-fraction changes relative to 2024.
- **HarDWR v2.0 surface-water rights density** — cumulative SW POD count
  per pixel (all consumptive sectors, excluding environmental in-stream
  flow rights), capturing the progressive build-out of surface-water
  infrastructure over 200 years.
- **Watershed-scaled canal density** — modern GRAIN canal density scaled
  backward in time using per-watershed SW rights build-out from HarDWR v2.0.
- **Multi-era climate harmonization** — no single ET or ETo dataset covers
  1896–2099, so overlapping-period bias-correction ratios stitch together
  three eras for each variable (see [gee/README.md](../gee/README.md)):
  - *ET*: USGS Reitz Ensemble (1896–1999) → OpenET Ensemble (2000–2025)
    → MACA EToF × ETo (2026–2099).  Reitz ET is scaled to OpenET using
    per-pixel, per-month ratios from the 2000–2018 overlap (228 paired
    images).  Future ET uses climatological crop coefficients (EToF)
    applied to scenario-driven ETo.
  - *ETo*: PRISM Hargreaves (1896–1978) → OpenET gridMET (1979–2025)
    → MACA ensemble (2026–2099).  PRISM Hargreaves is corrected to
    Penman-Monteith scale using 1979–2025 overlap ratios (564 paired
    images).  MACA ETo is computed per-GCM via `openet.refetgee` and
    bias-corrected with OpenET gridMET ratios.
  - *Streamflow*: USGS NWIS observations (priority) → USBR CMIP
    ensemble (post-USGS gap-fill, 1950–2099, 112 runs) → monthly
    climatology (remaining gaps).  USBR projections receive per-site,
    per-month multiplicative bias correction from the USGS/USBR overlap.
- **Bias-corrected LULC features** — basin-scale delta correction ensures
  temporal continuity across LULC source transitions (USGS historical →
  NLCD → USGS projections).  Off-NLCD years use the non-NLCD source for
  basin-scale relative change, anchored to NLCD's pixel-level spatial
  pattern at the 1985/2025 training-period boundaries.  Gaussian-smoothed
  agricultural, urban, and surface-water density layers provide spatially
  diffused signals for the ML model.

### Output side — conservation-consistent post-processing

Physical constraints that must hold exactly are enforced after prediction:

- **Conservation-consistent partitioning** — `Irrigation + Non-Irrigation =
  Total` and `GW + SW = Total` hold exactly for every pixel and year.
- **Temporal GW fraction adjustment** — pre-canal pixels set to 100 % GW
  using HarDWR v2.0 irrigation SW priority dates, ensuring the irrigation
  GW/SW split reflects actual infrastructure availability.
- **Non-irrigation GW/SW split** — temporally varying HarDWR v2.0 non-irrigation
  SW rights density (domestic, industrial, livestock) replaces the static
  canal-density proxy.
- **Well-density masking** — pixels with zero wells in a given year receive
  NaN predictions, preventing spurious withdrawals at uninstrumented locations.
- **Urban-fraction weighting** — non-irrigation withdrawals outside AMA/INAs
  scaled by physical urban area fraction.
- **Consumptive use** — `CU = IE × Irrigation_Withdrawal` using USGS NHM
  basin-level irrigation efficiencies, with physics-based error propagation
  `σ_CU = √((IE·σ_wd)² + (wd·σ_IE)²)`.

This separation of concerns is advantageous: each component can be described,
validated, and modified independently.  The ML model does not need to "learn"
conservation (guaranteed structurally), temporal infrastructure changes
(encoded in the data), or that pre-canal pixels are 100 % GW (in the
partitioning logic).  The result is a system where the physics is transparent
and exact where it matters, while the statistical model retains full
flexibility where physical laws are insufficient — namely, predicting the
magnitude of human water-use decisions.

### Note on PIML models

The codebase includes optional physics-informed ML wrappers (`PIML_XGB`,
`PIML_LGBM`, `PIML_XGBRF`) that embed domain knowledge into the training
objective via two tiers: (1) monotone constraints enforcing physically
expected directional relationships, and (2) a custom loss function with a
one-sided irrigation-demand floor penalty that keeps predictions at or
above the physics-estimated demand
`(ET − Peff) × irr_frac × gw_frac / IE`, with penalty strength tuned by
Optuna.

These models are **disabled by default** (`SKIP_PIML=True` in
`pipeline.py`) because they fail to outperform their purely statistical
counterparts.  On random and pixel holdout evaluations the PIML variants
perform comparably to standard XGBoost, but on temporal leave-one-out—the
most demanding test of generalization to unseen years—PIML_XGB exhibits
~20 % higher RMSE than standard XGBoost even after bias correction.  The
root cause is that groundwater withdrawals, in Arizona and elsewhere, are not
purely governed by physical irrigation demand: management decisions
(water rights, fallowing programs, municipal/industrial allocations,
conjunctive-use policies) drive withdrawal patterns that cannot be captured
by a physics-based floor tied to crop water balance.  Unlike groundwater
levels, which follow broadly predictable seasonal and climatic patterns,
withdrawal volumes are a human decision variable with no universal physical
law governing their magnitude.  In this setting, purely statistical
approaches that learn flexible predictor–response mappings from the data
outperform models constrained to respect a simplified physical prior.

To re-enable PIML models, set `SKIP_PIML=False` in `pipeline.py`.  The
full PIML infrastructure—`PhysicsXGBRegressor`, `PhysicsBoundsObjective`,
`compute_irrigation_demand_floor()`, and the `_append_physics_floor()`
pipeline helper—remains intact and tested (see `tests/test_core.py`).

---

## Output directory structure

After a full pipeline run, the output tree looks like:

```
Data/Outputs/
├── GEE_Mosaics_2000m/                      # Mosaicked GEE predictor tiles
├── GW/
│   ├── Rasters/GW_Depths_All_Wells_2000m/   # Observed withdrawal rasters (mm)
│   └── Vectors/All_Wells/                   # Per-year GW shapefiles
├── GW_Data/Vector_Reproj/                   # Reprojected basins, wells, etc.
├── Predictor_Data_All_Wells_2000m/          # Multi-band Predictor_YYYY.tif
│
└── ML_Model_All_Wells_2000m/
    ├── EDA/                                 # Exploratory data analysis plots
    ├── Model_Evaluation/
    │   ├── Random/                          # Step 2a results
    │   │   ├── ts10/seed_42/               #   test_size=10%, seed=42
    │   │   ├── ts10/seed_…/
    │   │   ├── ts15/…, ts20/…, …           #   other test sizes
    │   │   ├── All_Runs.csv                #   All test_size × seed metrics
    │   │   └── Model_Comparison_Averaged.csv #  Mean ± std by model & test_size
    │   ├── Pixel_Holdout/                   # Step 2a2 (same grid structure)
    │   ├── Temporal_LOO/                    # Step 2b results (T1–T7)
    │   ├── Spatial_LOO/                     # Step 2c results (per AMA/INA)
    │   │   ├── Stratified_Metrics.csv       #   per-category (Low/High) metrics
    │   │   └── Stratified_*.png             #   grouped bar charts by pumping bin
    │   ├── Spatial_LOO_Seed10/              # Step 2c-seed (10% local calibration)
    │   ├── Cross_Strategy_Summary.csv       # All models × all strategies
    │   ├── Cross_Strategy_Summary.tex       # LaTeX table for manuscripts
    │   └── Cross_Strategy_Comparison.png    # Grouped bar chart
    │
    └── Full_Prediction_XGB/
        ├── Model_Interpretability/          # SHAP, ALE, permutation importance
        │   ├── Hindcast/                    #   Era-specific plots (1896-1983)
        │   ├── Training/                    #   Era-specific plots (1984-2024)
        │   └── Projection/                  #   Era-specific plots (2026-2099)
        ├── Predicted_Rasters/               # Total annual withdrawal (4 units, 6-band)
        │   ├── Depth_mm/
        │   ├── Depth_ft/
        │   ├── Volume_m3/
        │   └── Volume_AF/
        ├── {Category}_Rasters/              # 8 withdrawal categories (4 units, 6-band)
        ├── Irrigation_CU_Rasters/           # CU (4 units, 6-band)
        ├── Annual_Summaries/                # Cached per-year stats (for fast re-runs)
        │   ├── Total_Predicted.csv          #   AZ-wide total predicted stats
        │   ├── {Category}.csv               #   Per-category stats
        │   ├── {CU_Category}.csv            #   Consumptive use stats
        │   ├── Actual.csv                   #   Metered actual stats (1984–2024)
        │   ├── Basin_Total.csv              #   Per-basin stats
        │   └── Subbasin_Total.csv           #   Per-sub-basin stats
        ├── OOD_Rasters/                     # Out-of-distribution detection
        │   ├── OOD_Flag_{year}.tif          #   OOD probability [0,1] (χ² CDF)
        │   ├── OOD_Summary.csv              #   Per-year OOD statistics
        │   └── OOD_TimeSeries.png           #   OOD % by year with era shading
        ├── Uncertainty/                     # Hybrid uncertainty quantification
        │   ├── Sigma_MACA/                  #   Inter-GCM climate spread
        │   ├── Sigma_Model/                 #   Seed ensemble spread
        │   ├── Sigma_Irr/                   #   Irrigation fraction spread
        │   ├── Sigma_LULC/                  #   LULC projection spread
        │   ├── Sigma_GW/                    #   Well-density feature sensitivity (2020–2024 snapshots)
        │   ├── Sigma_Total/                 #   Quadrature combination (σ, CV, per-category σ)
        │   ├── Sigma_CU/                    #   CU error-propagation σ
        │   └── Plots/                       #   Time-series plots
        ├── Graphical_Abstract_Fig1.png         # Publication Figure 1 (map + ts + bar)
        ├── Mean_Annual_Predicted_mm.tif        # Mean-annual depth GeoTIFF (1896–2099)
        ├── Prediction_Exceedance_Summary.csv   # Per-year exceedance stats
        ├── Raster_Maps/                     # Step 3g — spatial maps for all products
        │   ├── Era_Maps_*.png               #   2×2 era-mean panels per category
        │   ├── Actual_vs_Predicted.png      #   Actual vs predicted (1984–2024)
        │   └── Trend_Analysis/              #   Mann-Kendall + Sen's slope maps
        │       ├── Trend_*.png              #   Per-category, per-period pixel-level trend maps
        │       ├── Basin_Trend_*.png        #   Per-category, per-period basin choropleth maps
        │       ├── Basin_Trend_*.csv        #   Per-basin zonal trend statistics
        │       └── Subbasin_Trend_*.csv     #   Per-sub-basin zonal trend statistics
        ├── Visualizations/                  # Time series & era summary maps
        ├── SW_Capture/                      # Surface Water Capture Index
        │   ├── {Cat}_Capture_Fraction/      #   3-band (lower/central/upper) capture fraction
        │   ├── {Cat}_Capture_Rasters/       #   Central capture volume (mm, ft, m³, AF)
        │   └── SW_Capture_Time_Series.csv   #   AZ-wide annual capture totals
        ├── Well_Package/                    # Step 3e — per-well package
        │   ├── Well_Package_mm.parquet     #   GeoParquet: 15 categories (mm + σ_mm)
        │   ├── Well_Package_ft.parquet     #   GeoParquet: 15 categories (ft + σ_ft)
        │   ├── Well_Package_m3.parquet     #   GeoParquet: 15 categories (m³ + σ_m³)
        │   └── Well_Package_AF.parquet     #   GeoParquet: 15 categories (AF + σ_AF)
        ├── Intercomparison/                 # Step 4a — withdrawal comparison
        │   └── Temporal_Agreement/          #   Heatmaps, box/violin, Taylor, r-vs-NSE
        ├── CU_Intercomparison/              # Step 4b — CU comparison
        ├── CAP_SRP_Validation/              # Step 4c — CAP/SRP SW validation
        ├── Peff_Intercomparison/            # Step 4d — Peff comparison
        └── PS_Intercomparison/              # Step 4e — Non-irrigation vs USGS PS
```

## Results

The pipeline estimates **well-based withdrawals** — groundwater pumping plus
locally diverted surface water that passes through registered well/diversion
infrastructure.  This is a subset of Arizona's total water supply, which also
includes Colorado River imports (CAP aqueduct deliveries, Yuma-area
diversions), reclaimed/effluent water, and other sources not captured by the
ADWR Well Registry.  ADWR reports total statewide water use of ~7.0 MAF
(2017), of which irrigated agriculture consumes approximately 72 % (as per ADWR 2019 data)
([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)).

**Water budget reconciliation (2017):** The model predicts 4.74 MAF vs.
ADWR's 7.0 MAF total, a gap of ~2.25 MAF.  This gap is accounted for by
water sources outside the ADWR Well Registry:

| Source | MAF | Notes |
|--------|-----|-------|
| CAP direct deliveries | 0.71 | Excludes recharge; from CAP delivery records |
| SRP surface water | 0.40 | Phoenix AMA; from ADWR SRP delivery records |
| Yuma-area federal diversions | ~0.79 | Bureau of Reclamation Yuma and Gila Project irrigation districts (WMIDD: 278,000 AF, YCWUA: 254,200 AF, Gila Project Yuma Mesa Division: 250,000 AF, Unit B: 6,800 AF); gravity canal diversions from the Colorado River via the All-American Canal and Gila Gravity Main Canal ([Noble et al., 2015](https://www.azwater.gov/sites/default/files/2022-11/Final%20Yuma%20Report%20021715.pdf)) |
| Reclaimed/effluent water | ~0.35 | ~5 % of total state water supply ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)) |
| **Total gap** | **~2.25** | |

The model thus captures ~68% of Arizona's total water use — specifically,
the portion that flows through registered well and diversion infrastructure.
The remaining ~32 % is delivered by large-scale federal water projects,
(CAP aqueduct, SRP canal system, Yuma-area Colorado River diversions) and
reclaimed water systems that operate outside the ADWR Well Registry.
The Yuma-area irrigation districts hold some of the oldest and most senior
water rights on the Lower Colorado River (dating to the Reclamation Act of
1902) and collectively divert ~0.79 MAF/yr through federal canal infrastructure
with application efficiencies of 80–90 % — substantially higher than the
statewide NHM average of ~60 % used in this pipeline
([Noble et al., 2015](https://www.azwater.gov/sites/default/files/2022-11/Final%20Yuma%20Report%20021715.pdf)).
Combining the model's 4.74 MAF with the estimated non-well sources
(~2.25 MAF) yields ~6.99 MAF, closing to within 0.01 MAF of ADWR's
reported 7.0 MAF total.  USGS independently estimates 3.09 MAF of
GW withdrawals for 2015 ([Dieter et al., 2018](https://doi.org/10.3133/cir1441); Arizona summary in [NGWA, 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf));
the model produces 3.17 MAF for the same year (within 0.08 MAF)
([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)).

**Uncertainty around the reconciliation:** The 2017 AZ-wide σ_total
from the UQ pipeline is **0.66 MAF** (≈ 13.9 % of the 4.74 MAF model
value), giving a 95 % confidence interval of **3.45–6.03 MAF** on the
model alone.  Adding the (constant) ~2.26 MAF federal-delivery
adjustment shifts that interval to **5.71–8.29 MAF** for the
reconciled total, which comfortably brackets both the closure value
(6.99 MAF) and ADWR's reported 7.0 MAF.  The fact that the central
estimate (6.99 MAF) lands within 0.01 MAF of ADWR while the 95 % CI
spans ~2.6 MAF is a useful reminder that the very tight closure is a
consequence of using the same federal-delivery offset that ADWR
implicitly counts — not evidence that the underlying ML prediction is
accurate to ± 0.01 MAF.  The honest precision of the model-side
estimate is the σ_total interval, which 6.99 / 7.0 MAF both fall well
inside.

The USGS 2015 GW comparison tells the same story from an independent
direction: the model predicts 3.17 MAF GW pumping for 2015 with
σ_total ≈ 0.65 MAF, giving a 95 % CI of **1.90–4.44 MAF**, and USGS's
3.09 MAF estimate ([Dieter et al., 2018](https://doi.org/10.3133/cir1441); Arizona summary in [NGWA, 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf))
lands well inside that interval.  Two independent agency totals — ADWR
for 2017 and USGS for 2015, computed from different source data with
different methodologies — both fall within the model's σ_total
intervals despite the model never being trained or tuned against either
of them.

**Spatial scope of "no calibration":** The training dataset
(`USE_AMA_INA = True` in [pipeline.py](pipeline.py#L120)) is restricted
to ADWR-metered pixels inside the ten AMA/INA management areas:
Phoenix, Pinal, Tucson, Prescott, Santa Cruz, Douglas, and Willcox
AMAs plus Joseph City, Harquahala, and Hualapai Valley INAs.  Of these,
the eight legacy AMA/INAs (everything except Willcox and Hualapai
Valley) provide continuous metered records from 1984 onward and
contribute the bulk of the training signal.  **Willcox AMA and
Hualapai Valley INA were designated only recently and have sparse
metering both temporally (records concentrated in the most recent
years) and spatially (fewer reporting wells per pixel)**, so the
effective training signal from those two basins is much smaller than
from the eight legacy areas.  Predictions are then generated for every
2 km pixel in Arizona, including the ~25 unmetered Other basins
(basin type 2) — Yuma, Lower Gila, Parker, Lake Havasu, Bill Williams,
Butler Valley, the Mogollon plateau basins, and others — for which
**no per-well training labels exist anywhere**.  The 4.74 MAF (2017)
model total therefore mixes in-sample-distribution AMA/INA predictions
with out-of-distribution unmetered-basin predictions, and roughly 30 %
of the statewide volume comes from the latter group.  When ADWR's 7.0
MAF and USGS's 3.09 MAF land inside the σ_total intervals, the model
is being validated not just against an independent target year and
methodology, but against a target that explicitly aggregates basins
the ML model has never been trained on.  The agreement is therefore a
genuine out-of-distribution generalization test, not an in-sample
goodness-of-fit check.  This is the strongest possible version of the
"no calibration to reported totals" claim because no training signal
from the unmetered basins flows into the model at any stage —
predictor features (climate, LULC, well density, canal-weighted
streamflow, WTD, etc.) are computed identically inside and outside
AMA/INAs from the same gridded inputs, and the model relies entirely
on the assumption that the learned predictor → pumping mapping
generalizes from metered AMA/INAs to morphologically similar
unmetered basins.

Representative statewide volumes (million acre-feet):

| Year | Total | Irrigation | Non-Irrigation | Total GW | Total SW | Irr % | GW % |
|------|-------|------------|----------------|----------|----------|-------|------|
| 1900 | 0.11 | 0.04 | 0.07 | 0.07 | 0.04 | 39 % | 66 % |
| 1910 | 0.14 | 0.06 | 0.08 | 0.10 | 0.04 | 40 % | 71 % |
| 1920 | 0.26 | 0.13 | 0.13 | 0.16 | 0.09 | 50 % | 63 % |
| 1930 | 0.46 | 0.26 | 0.20 | 0.26 | 0.20 | 56 % | 57 % |
| 1940 | 0.85 | 0.57 | 0.28 | 0.46 | 0.40 | 67 % | 54 % |
| 1950 | 1.74 | 1.32 | 0.43 | 0.89 | 0.86 | 75 % | 51 % |
| 1960 | 2.61 | 2.05 | 0.56 | 1.42 | 1.19 | 79 % | 55 % |
| 1970 | 3.16 | 2.51 | 0.65 | 1.88 | 1.28 | 80 % | 60 % |
| 1980 | 3.82 | 3.03 | 0.80 | 2.23 | 1.59 | 79 % | 58 % |
| 1985 | 4.17 | 3.24 | 0.93 | 2.52 | 1.65 | 78 % | 60 % |
| 1990 | 4.20 | 3.25 | 0.95 | 2.65 | 1.55 | 77 % | 63 % |
| 2000 | 4.35 | 3.22 | 1.12 | 2.80 | 1.55 | 74 % | 64 % |
| 2010 | 4.20 | 3.03 | 1.17 | 2.77 | 1.42 | 72 % | 66 % |
| 2015 | 4.65 | 3.41 | 1.24 | 3.17 | 1.48 | 73 % | 68 % |
| 2017 | 4.74 | 3.46 | 1.28 | 3.25 | 1.50 | 73 % | 68 % |
| 2019 | 4.46 | 3.25 | 1.21 | 3.06 | 1.40 | 73 % | 69 % |
| 2020 | 4.73 | 3.41 | 1.32 | 3.28 | 1.45 | 72 % | 69 % |
| 2024 | 4.74 | 3.37 | 1.36 | 3.33 | 1.40 | 71 % | 70 % |
| 2030 | 4.68 | 3.25 | 1.43 | 3.18 | 1.50 | 69 % | 68 % |
| 2040 | 4.79 | 3.26 | 1.52 | 3.26 | 1.52 | 68 % | 68 % |
| 2050 | 4.99 | 3.33 | 1.65 | 3.42 | 1.56 | 67 % | 69 % |
| 2060 | 5.14 | 3.38 | 1.76 | 3.54 | 1.59 | 66 % | 69 % |
| 2070 | 5.34 | 3.44 | 1.89 | 3.71 | 1.62 | 64 % | 70 % |
| 2080 | 5.42 | 3.46 | 1.96 | 3.79 | 1.64 | 64 % | 70 % |
| 2090 | 5.55 | 3.48 | 2.07 | 3.89 | 1.66 | 63 % | 70 % |
| 2099 | 5.66 | 3.52 | 2.14 | 3.99 | 1.67 | 62 % | 71 % |

Consumptive use (CU = IE × Irrigation Withdrawal) volumes, where IE is the
USGS NHM basin-level irrigation efficiency (million acre-feet):

| Year | Irrigation | Irrigation CU | Irrigation GW CU | Irrigation SW CU | IE |
|------|------------|---------------|-------------------|------------------|----|
| 1900 | 0.04 | 0.03 | 0.02 | 0.01 | 61 % |
| 1910 | 0.06 | 0.03 | 0.03 | 0.01 | 61 % |
| 1920 | 0.13 | 0.08 | 0.05 | 0.03 | 61 % |
| 1930 | 0.26 | 0.16 | 0.08 | 0.07 | 61 % |
| 1940 | 0.57 | 0.35 | 0.17 | 0.17 | 61 % |
| 1950 | 1.32 | 0.80 | 0.38 | 0.42 | 61 % |
| 1960 | 2.05 | 1.25 | 0.66 | 0.59 | 61 % |
| 1970 | 2.51 | 1.53 | 0.89 | 0.64 | 61 % |
| 1980 | 3.03 | 1.84 | 1.05 | 0.79 | 61 % |
| 1985 | 3.24 | 1.96 | 1.16 | 0.80 | 61 % |
| 1990 | 3.25 | 1.97 | 1.22 | 0.75 | 61 % |
| 2000 | 3.22 | 1.94 | 1.22 | 0.72 | 60 % |
| 2010 | 3.03 | 1.83 | 1.19 | 0.64 | 60 % |
| 2015 | 3.41 | 2.06 | 1.40 | 0.66 | 60 % |
| 2017 | 3.46 | 2.10 | 1.43 | 0.67 | 61 % |
| 2019 | 3.25 | 1.97 | 1.35 | 0.62 | 61 % |
| 2020 | 3.41 | 2.05 | 1.41 | 0.64 | 60 % |
| 2024 | 3.37 | 2.04 | 1.44 | 0.60 | 60 % |
| 2030 | 3.25 | 1.97 | 1.33 | 0.63 | 60 % |
| 2040 | 3.26 | 1.97 | 1.34 | 0.63 | 60 % |
| 2050 | 3.33 | 2.01 | 1.37 | 0.64 | 60 % |
| 2060 | 3.38 | 2.04 | 1.40 | 0.64 | 61 % |
| 2070 | 3.44 | 2.08 | 1.43 | 0.65 | 61 % |
| 2080 | 3.46 | 2.10 | 1.44 | 0.65 | 61 % |
| 2090 | 3.48 | 2.11 | 1.46 | 0.65 | 61 % |
| 2099 | 3.52 | 2.13 | 1.48 | 0.65 | 61 % |

The statewide mean IE is ~60 %, meaning roughly 40 % of applied irrigation
water returns to aquifers as deep percolation or runs off as return flow.
IE varies by basin (NHM HUC12-level values) but the statewide aggregate is
stable across years because the same NHM efficiency map is applied to
changing withdrawal volumes.

Key trends:
- **Irrigation share** (irrigation withdrawal as a fraction of total
  well-mediated withdrawal) declines from ~80 % (1960s–1980s) to
  ~73 % (2019) and continues to ~62 % by 2099 as urbanization
  increases M&I demand. The 2019 model value of **72.8 %** matches
  ADWR's reported ~72 % share of agriculture in total Arizona water
  use ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector))
  to within ~0.8 percentage point. The irrigation category includes
  rural dual-purpose wells coded ``IRRIGATION, DOMESTIC`` or
  ``IRRIGATION, STOCK`` in the ADWR Well Registry; recreation / turf
  wells (e.g. golf courses) are coded ``RECREATION`` and routed to
  the non-irrigation category.
- **Irrigation GW share** (`Irrigation_GW / Irrigation_total`) is a
  separate metric that asks "of the irrigation water alone, what
  fraction comes from wells." The model produces ~68 % for 2017,
  ~69 % for 2019, and ~71 % for 2024, all in the metered window. This
  is much higher than the *statewide* GW share (44–46 % in those same
  years, the figures compared to USGS and ADWR above) because the
  statewide number is diluted by the ~2.26 MAF/yr of federal Colorado
  River canal deliveries (CAP, SRP, Yuma-area diversions) that bypass
  wells entirely and are reconciled separately as the federal-delivery
  offset.
- **GW share** declines from 66 % (1900) to ~51 % by the 1950s as canal
  infrastructure (SRP) brought surface water to irrigated areas, then
  rises gradually to ~70 % by 2099 as non-irrigation demand —
  predominantly groundwater-sourced outside canal service areas — grows
  faster than irrigation.  Including unaccounted federal SW
  deliveries (~2.26 MAF), the statewide GW share is ~46 % in 2017,
  consistent with independently reported GW/SW shares: USGS estimates
  46 % GW and 3.09 MAF total GW for 2015 ([NGWA, 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf);
  based on [Dieter et al., 2018](https://doi.org/10.3133/cir1441)), while
  ADWR reports 41 % GW for 2019 ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)).
  The model produces 45.9 % / 3.17 MAF for 2015 (within 0.08 MAF and
  0.1 percentage point of USGS) and 44 % for 2019 (within 3 percentage
  points of ADWR) — converging on both agency estimates across different
  years without any calibration to these targets.

  | Year | Source | GW (MAF) | Statewide GW % |
  |------|--------|----------|----------------|
  | 2015 | USGS/NGWA | 3.09 | 46 % |
  | 2015 | AZ-Hydro | 3.17 | 45.9 % |
  | 2019 | ADWR | — | 41 % |
  | 2019 | AZ-Hydro | 3.06 | 44 % |
- **Total withdrawals** grow from 0.11 MAF (1900) to 4.74 MAF (2024)
  and are projected to reach 5.66 MAF by 2099.  Early growth (1900–1950)
  reflects the build-out of well and canal infrastructure; mid-century
  growth (1950s–1980s) is driven by agricultural expansion; projected
  growth is driven by urbanization and increasing M&I demand.
- **Irrigation** remains relatively stable in the projections (3.37 →
  3.52 MAF, +4 % by 2099), while **non-irrigation** grows by ~57 %
  (1.36 → 2.14 MAF), more than doubling from the 1990s level
  (0.95 MAF), reflecting continued urban and industrial growth
  including data center and energy-sector water demand.
- **Pre-CAP era** (before 1985): GW share declines as SRP canal
  infrastructure expanded, reaching a minimum ~51 % by the 1950s.
  Post-1985 GW share rises as non-irrigation (predominantly GW)
  demand grows relative to canal-served irrigation.
- **Conservation**: Irrigation + Non-Irrigation = Total and GW + SW = Total
  hold exactly for all years.
- **Consumptive use** (CU = IE × Irrigation Withdrawal, with IE the
  USGS NHM HUC12 irrigation efficiency map): irrigation CU rises from
  ~0.03 MAF in 1900 to a 2.10 MAF peak in 2017 — a ~70× increase that
  tracks the parent irrigation withdrawal trajectory closely because
  the statewide IE is stable at ~60 % across all years (60.1 % in 2010,
  60.6 % in 2017, 60.4 % in 2024). The roughly 40 % of applied
  irrigation water that is *not* consumed returns to the aquifer as
  deep percolation or runs off as return flow, which is a substantial
  recharge term that this study quantifies but does not separately
  route through the capture index. The GW share of irrigation CU
  follows the parent GW share of irrigation withdrawal — rising from
  ~50 % in the 1950s (when SRP canal deliveries were near peak) to
  ~70 % by 2017 — so the consumptive component of pumping-induced
  groundwater depletion grows in proportion. In the projection
  (2026–2099) irrigation CU is flat to slightly rising (2.04 → 2.13
  MAF, +5 %) because the parent irrigation withdrawal — which is
  driven by the LULC-projection-derived `annual_irr_fraction` and
  `annual_crop_fraction` features — is roughly stable in the USGS
  scenarios used here. Crop-area expansion or contraction *is*
  captured in the projection through those LULC features, so a future
  in which irrigated acreage changes will move our predicted
  withdrawal (and therefore CU) accordingly. The projected CU
  trajectories should still be interpreted with the **irrigation
  efficiency paradox** in mind: an upward shift in IE may actually
  *increase* basin-scale CU rather than decrease it, through several
  rebound channels that this framework does not represent. See the
  Known Limitations subsection on the irrigation efficiency paradox
  for the full discussion and the supporting [Grafton et al. (2018)](https://doi.org/10.1126/science.aat9314)
  citation.
- **Surface Water Capture Index**: The statewide volume-weighted
  capture fraction is ~0.2 % (central estimate, λ=10 m), translating
  to 0.02–0.04 MAF of GW pumping that physically captures surface
  water via stream depletion.  Capture grows from ~0.001 MAF (1900)
  to a peak of ~0.038 MAF (2015–2024) as pumping infrastructure
  expanded near canal corridors, then declines slightly in projections
  as irrigation stabilizes.  Irrigation GW accounts for ~90 % of total
  capture (~0.034 MAF), with non-irrigation contributing ~0.003 MAF.
  While the statewide impact is small (<1 % of total GW), the capture
  is spatially concentrated in shallow-water-table areas near rivers
  and canal infrastructure.  Basin-level volume-weighted capture
  fractions (2017, central λ=10 m):

  | Basin | Capture Fraction | Mean WTD (m) | Context |
  |-------|-----------------|-------------|---------|
  | Parker | 9.7 % | 51 | Colorado River alluvial aquifer |
  | Lower Gila | 6.9 % | 55 | Gila River corridor |
  | Yuma | 3.7 % | 42 | Colorado River diversions, alluvial wells |
  | Lake Havasu | 3.3 % | 73 | Colorado River adjacent |
  | Safford | 2.1 % | 34 | Upper Gila River valley, shallowest WTD |
  | Phoenix AMA | 0.4 % | 64 | Deep wells dominate; SRP corridor localized |
  | Willcox AMA | 0.0 % | 44 | Negative control: no canals despite moderate WTD |
  | Butler Valley | 0.0 % | 72 | Negative control: no canal infrastructure |

  The spatial pattern is consistent with
  [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757), which
  qualitatively identified regions where GW pumping potentially
  impacts surface water by mapping floodplains within 20 km of the
  Colorado and Gila rivers (catchment area ≥ 40,000 km²) with grade
  < 0.4 % and average well depth ≤ 50 m.  The capture index advances
  this from binary criteria to continuous, physics-based variables:
  `canal_weighted_streamflow` replaces the binary "near river"
  proximity rule, and `exp(−wtd/λ)` replaces the 50 m well-depth
  threshold with a smooth exponential decay — at λ = 10 m, a 50 m
  water table gives `exp(−50/10) ≈ 0.007` (negligible capture),
  consistent with the 2022 study's cutoff.  The highest capture
  fractions (Parker, Lower Gila, Yuma) align with the Colorado and
  Gila River alluvial corridors identified in the 2022 study as the
  primary zones of pumping-induced surface water depletion.

- **Uncertainty**: AZ-wide volume-weighted σ_total is **~14 % of the
  mean predicted withdrawal during the 1984–2025 historical period**
  (≈ 0.62 MAF σ on a 4.38 MAF mean), rising to **~19 % over the
  2026–2099 projection** (≈ 0.98 MAF on 5.18 MAF) as σ_LULC and σ_MACA
  begin to contribute, and reaching **~26 % in the deep hindcast
  (1896–1983)** (≈ 0.32 MAF on 1.40 MAF) where σ_irr is largest because
  pre-IrrMapper irrigation-fraction reconstruction is more uncertain.
  Variance attribution (% of σ_total²) shifts across eras:
  - **Hindcast (1900):** σ_irr 61 % · σ_model 20 % · others ≈ 0
  - **Historical (2017):** σ_model 58 % · σ_irr 4 % · σ_GW 0.1 %
  - **Projection (2099):** σ_model 46 % · σ_LULC 19 % · σ_MACA 4 % · σ_irr 0 % · σ_GW 0 %

  The well-density σ_gw component is small everywhere
  (≈ 0.3 mm depth, < 0.1 % of σ_total²) — the 5-snapshot HarDWR
  perturbation captures real year-over-year variability in the model's
  #1 SHAP feature, but the input CV (~3 %) is small enough that the
  resulting prediction sensitivity is dominated by the seed ensemble
  σ_model. Per-pixel CV maps (`Sigma_Total/Mean_CV.tif`) and
  per-component contribution time series are written to
  `Uncertainty/Sigma_Total/` and `Uncertainty/Plots/`.

  **Per-basin CV structure (2017 example).** The AZ-wide ~14 % figure
  conflates very different per-basin behaviors. Quadrature-aggregated
  σ within each basin (from `Sigma_Total/Basin_Sigma_Total.csv`) gives:

  | Basin group | Examples | Per-basin CV |
  |---|---|---|
  | Largest legacy AMAs (in-sample) | Phoenix 1.1 %, Tucson 1.0 %, Pinal 1.3 % | **1–5 %** |
  | Largest unmetered Other / OOD basins | Yuma 3.1 %, Lower Gila 5.9 %, Parker 5.2 %, Lake Havasu 6.9 %, Hualapai INA 5.7 % | **3–10 %** |
  | Mid-size mixed | Lake Mohave 6.4 %, Bill Williams 10.5 %, LCRP 8.5 % | **5–11 %** |
  | Small low-volume basins | Meadview 48 %, Peach Springs 26 %, Morenci 24 % | **20–50 %** |

  Two things are worth noting. First, the largest unmetered Other
  basins (the river-corridor basins where the model is in pure OOD
  mode) have CVs only modestly higher than the largest legacy AMAs —
  the σ framework does inflate the CI in OOD basins, but the inflation
  is small because the predictor → pumping mapping generalizes well
  from morphologically similar metered basins. The very high CVs are
  concentrated in tiny low-volume basins (Meadview, Peach Springs)
  where the absolute σ is small but the denominator is even smaller,
  not in the OOD basins per se. Second, the AZ-wide ~14 % figure
  reported above is computed from `Mean_Sigma_Total_mm × active_area`
  (effectively assuming maximum spatial correlation across pixels),
  while the per-basin numbers use proper quadrature within each basin
  and would aggregate across basins to a much smaller statewide CV
  (~1 %) under an independence assumption. The truth lives between
  these bounds because pixel errors are partially correlated through
  shared predictors. We report the conservative ~14 % number in the
  reconciliation paragraph because it is the most defensible upper
  bound on the AZ-wide aggregate; per-basin CIs in
  `Basin_Sigma_Total.csv` are the appropriate honest precision for
  any individual-basin claim.

### Known limitations

Five limitations are baked into the framework's structure rather than
into any individual UQ component. Each is unavoidable given current
data and is named explicitly here so it does not have to be inferred
from the methods.

1. **Deep hindcast extrapolation (1896–~1950).** The model is trained
   on 1984–2024 metered records and applied backward to 1896. For the
   most recent decades of the hindcast (1950s onward), the underlying
   infrastructure, climate normals, and crop mix are similar enough to
   the training period that the predictor → pumping mapping is
   plausibly stationary, and the σ_irr component correctly inflates
   the CI in this era (61 % of variance share at 1900) to reflect
   that pre-IrrMapper irrigation reconstruction is uncertain. For the
   deep hindcast (pre-1950, and especially pre-1920), the model is
   predicting 60+ years before its training window in a state with
   very different cropping patterns, no Hoover Dam, no SRP canal
   build-out, and dramatically lower well counts. The 1896–1950
   numbers should be read as a **physics-constrained reconstruction
   consistent with the available historical record**, not as a
   validated quantitative reconstruction of any specific year. The
   1950s onward have substantially stronger empirical anchoring.

2. **Projection structural-change blindness (2026–2099).** The σ_LULC
   and σ_MACA components capture parameter uncertainty within fixed
   ensemble scenarios — they vary climate and land use across the
   provided GCM and USGS LULC scenarios but they do not vary the
   underlying predictor → pumping relationship the model learned on
   1984–2024 data. The projection is therefore blind to structural
   changes that would alter that relationship: new crop types and
   crop-mix shifts at constant area, on-farm irrigation technology
   transitions (treated separately as the irrigation efficiency
   paradox in item 3 below), policy shifts (Senate Bill 1740 basin
   closures, additional metering, mandatory conservation rules),
   step-change events (Colorado River Tier 2/3 declarations, Lake
   Mead/Powell shortage cuts, dam decommissioning), and emerging
   demands (data centers, hydrogen production, atmospheric water
   capture). The 2099 projection should be read as **"what would
   happen if the 1984–2024 model relationships continued forward under
   the provided climate and land-use scenarios"**, not as a forecast
   of actual 2099 water use under all plausible futures.

3. **Irrigation efficiency paradox in CU projections.** Consumptive
   use is computed as `CU = IE × Irrigation_Withdrawal`, where the
   irrigation efficiency (IE) is taken from the USGS NHM HUC12 map
   and held constant across all 204 prediction years. The crop-area
   channel of CU change *is* captured in the projection because
   `annual_irr_fraction` and `annual_crop_fraction` are LULC-projection
   features that vary year to year, so a future in which Arizona's
   irrigated acreage expands or contracts under the USGS scenarios
   moves both the parent withdrawal and the resulting CU. Several
   other channels are not represented, however, and they all share a
   common pattern documented by [Grafton et al. (2018)](https://doi.org/10.1126/science.aat9314)
   as **"the paradox of irrigation efficiency"**: when farmers adopt
   higher-efficiency technologies (drip, sub-surface, precision
   sprinklers), the typical empirical response is *not* to use less
   water on the same crops but to switch to higher-water-demand crops,
   intensify application depth, extend the growing season, or
   introduce double-cropping — each of which raises ET and therefore
   raises CU. The mechanism is conservation of mass: the unrecovered
   "losses" from low-efficiency surface irrigation (deep percolation,
   runoff) are largely *recoverable* return flow that recharges the
   aquifer downstream, while the "savings" from high-efficiency
   systems are largely *non-recoverable* consumptive losses to ET.
   Specifically, our framework does not represent: **(i)** crop-mix
   shifts at constant area, because the model has no per-crop water
   demand profile; **(ii)** growing-season extension or double-cropping,
   because the model targets total annual pumping rather than
   per-event applications; **(iii)** intensification of irrigation
   depth on the same area; and **(iv)** the IE shift itself, since
   the NHM HUC12 IE map is held constant. The deeper conceptual
   limitation is that the LULC scenarios are *exogenous* — they do
   not respond endogenously to a hypothetical IE adoption — whereas
   the Grafton paradox is fundamentally about the *coupling* between
   IE adoption and farmer behavioral response. So a wholesale Arizona
   shift to drip irrigation would likely move our CU trajectory
   upward through channels (i)–(iii) above, even where the LULC-driven
   cropped area stays flat. The flat-to-slightly-rising projected CU
   (2.04 → 2.13 MAF, +5 % over 2024 → 2099) should therefore be read
   as a *mechanistic projection under the assumption that IE and the
   IE → behavior coupling do not change*, not as a forecast of actual
   2099 CU under all plausible technology trajectories.

4. **Sparse metering in Willcox AMA and Hualapai Valley INA.** Both
   basins were designated as AMA/INAs only recently, and ADWR-metered
   records exist there only for the most recent years and at fewer
   reporting wells per pixel than the eight legacy AMA/INAs. The
   training set technically includes them, but the effective signal
   from them is small and concentrated in the latest years of the
   training window. Predictions for these two basins for years before
   their designation are de facto extrapolations from the morphology
   of similar legacy basins. The model's published agreement with
   ADWR's reconciliation does not specifically validate the Willcox
   and Hualapai trajectories; basin-specific results in those two
   basins should be interpreted with the same caution as Other-basin
   predictions.

5. **Static water table depth (WTD) raster.** The capture index and
   the σ framework both treat the [Ma et al. (2026)](https://doi.org/10.1038/s43247-025-03094-3)
   WTD raster as time-invariant — a single snapshot is used for all
   204 prediction years. Yuma's water table near the Colorado has
   dropped meaningfully over recent decades; SRP-corridor water
   tables have risen with canal recharge. Using one snapshot for all
   years means the per-pixel `exp(−wtd/λ)` connectivity term in the
   capture index does not move as the actual water table moves, which
   under-counts capture in basins where WTD is rising and over-counts
   capture in basins where WTD is falling. The hydraulic-connectivity
   proxy is therefore most defensible for years close to the snapshot
   date and progressively more uncertain as the prediction year moves
   away from it. Building a time-varying WTD layer for the full
   1896–2099 window would require a transient regional groundwater
   model and is out of scope for this study; the static-WTD assumption
   is documented here so users of the capture-index outputs know what
   they are inheriting.

We document these limitations alongside the methods so that users of
the published outputs inherit them explicitly rather than discovering
them from the data. They define the scope conditions under which our
results should be interpreted: the framework is most defensible for
predictions inside the 1984–2024 metered window and within
morphologically similar basins, and the σ_total framework already
inflates the confidence intervals where these limitations apply most
strongly. Outputs outside that scope — the deep hindcast, the 2099
projection, the projected CU trajectories under hypothetical
irrigation-efficiency shifts, the two recently designated AMA/INAs,
and the capture-index estimates far from the WTD snapshot date —
should be read as physics-constrained reconstructions and projections
rather than as validated point estimates.


## References

Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modeling. _International Journal of Climatology_, _33_(1), 121–131. https://doi.org/10.1002/joc.3413.

Abatzoglou, J. T., & Brown, T. J. (2012). A comparison of statistical downscaling methods suited for wildfire applications. _International Journal of Climatology_, _32_(5), 772–780. https://doi.org/10.1002/joc.2312.

Alzraiee, A., Niswonger, R., Luukkonen, C., Larsen, J., Martin, D., Herbert, D., Buchwald, C., Dieter, C., Miller, L., Stewart, J., Houston, N., Paulinski, S., & Valseth, K. (2024). Next Generation Public Supply Water Withdrawal Estimation for the Conterminous United States Using Machine Learning and Operational Frameworks. _Water Resources Research_, _60_(7). https://doi.org/10.1029/2023WR036632

Asfaw, D., Smith, R. G., Majumdar, S., Grote, K., Fang, B., Wilson, B. B., Lakshmi, V., & Butler, J. J. (2025). Predicting groundwater withdrawals using machine learning with limited metering data: Assessment of training data requirements. Agricultural Water Management, 318, 109691. https://doi.org/10.1016/j.agwat.2025.109691

Barlow, P. M., & Leake, S. A. (2012). Streamflow Depletion by Wells—Understanding and Managing the Effects of Groundwater Pumping on Streamflow. _U.S. Geological Survey Circular 1376_. https://pubs.usgs.gov/circ/1376/.

Condon, L. E., & Maxwell, R. M. (2019). Simulating the sensitivity of evapotranspiration and streamflow to large-scale groundwater depletion. _Science Advances_, _5_(6), eaav4574. https://doi.org/10.1126/sciadv.aav4574.

Daly, C., Halbleib, M., Smith, J. I., Gibson, W. P., Doggett, M. K., Taylor, G. H., Curtis, J., & Pasteris, P. P. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. _International Journal of Climatology_, _28_(15), 2031–2064. https://doi.org/10.1002/joc.1688.

de Graaf, I. E. M., Gleeson, T., van Beek, L. P. H., Sutanudjaja, E. H., & Bierkens, M. F. P. (2019). Environmental flow limits to global groundwater pumping. _Nature_, _574_, 90–94. https://doi.org/10.1038/s41586-019-1594-4.

Dieter, C. A., Maupin, M. A., Caldwell, R. R., Harris, M. A., Ivahnenko, T. I., Lovelace, J. K., Barber, N. L., & Linsey, K. S. (2018). Estimated use of water in the United States in 2015. _U.S. Geological Survey Circular 1441_. https://doi.org/10.3133/cir1441.

Fleckenstein, R., Wellington, D., Jin, S., Tollerud, H., Brown, J. F., Dewitz, J., Pastick, N. J., Barber, C. P., O’Brien, A., & Spanier, M. (2026). A framework for integrating spatiotemporal deep learning methods with landsat for annual land cover and impervious surface mapping. _Remote Sensing of Environment_, _338_, 115347. https://doi.org/10.1016/j.rse.2026.115347.

Gangopadhyay, S., & Pruitt, T. (2011). West-Wide Climate Risk Assessments:  Bias-Corrected  and Spatially Downscaled  Surface Water Projections (Technical Memorandum No. 86-68210-2011-01). _U.S. Bureau of Reclamation_. https://www.usbr.gov/watersmart/docs/west-wide-climate-risk-assessments.pdf.

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. _Remote Sensing of Environment_, _202_, 18–27. https://doi.org/10.1016/j.rse.2017.06.031.

Grafton, R. Q., Williams, J., Perry, C. J., Molle, F., Ringler, C., Steduto, P., Udall, B., Wheeler, S. A., Wang, Y., Garrick, D., & Allen, R. G. (2018). The paradox of irrigation efficiency. _Science_, _361_(6404), 748–750. https://doi.org/10.1126/science.aat9314.

Hasan, M. F., Smith, R. G., Majumdar, S., Huntington, J. L., Alves Meira Neto, A., & Minor, B. A. (2025). Satellite data and physics-constrained machine learning for estimating effective precipitation in the Western United States and application for monitoring groundwater irrigation. _Agricultural Water Management_, _319_, 109821. https://doi.org/10.1016/j.agwat.2025.109821.

Haynes, J.V., Read, A.L., Chan, A.Y., Martin, D.J., Regan, R.S., Henson, W.R., Niswonger, R.G., & Stewart, J.S., 2023, Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000-2020 within the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_, https://doi.org/10.5066/P9LGISUM.

Hodson, T.O., Hariharan, J.A., Black, S., & Horsburgh, J.S.. (2023). dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. _U.S. Geological Survey software release_. https://doi.org/10.5066/P94I5TX3.

Hung, F., Chiarelli, D. D., Famiglietti, J. S., & Müller, M. F. (2025). Downscaled global 60-meter resolution estimates of irrigation water sources (2000–2015). _Scientific Data_, _12_(1), 1632. https://doi.org/10.1038/s41597-025-05920-x.

Ketchum, D., Hoylman, Z. H., Huntington, J., Brinkerhoff, D., & Jencso, K. G. (2023). Irrigation intensification impacts sustainability of streamflow in the Western United States. _Communications Earth & Environment_, _4_(1), 479. https://doi.org/10.1038/s43247-023-01152-2.

Ketchum, D., Jencso, K., Maneta, M. P., Melton, F., Jones, M. O., & Huntington, J. (2020). IrrMapper: A Machine Learning Approach for High Resolution Mapping of Irrigated Agriculture Across the Western U.S. _Remote Sensing_, _12_(14), 2328. https://doi.org/10.3390/rs12142328.

Lisk, M. D., Grogan, D. S., Proctor, K. L., Naz, B. S., Farmer, W. H., & Bock, A. R. (2024). HarDWR — Harmonized Database of Western U.S. Water Rights (v2.0). _Zenodo_. https://doi.org/10.57931/2475303.

Lisk, M. D., Grogan, D. S., Zuidema, S., Zheng, J., Caccese, R., Peklak, D., Fisher-Vanden, K., Lammers, R. B., Olmstead, S. M., & Fowler, L. (2024). Harmonized Database of Western U.S. Water Rights (HarDWR) v.1. _Scientific Data_, _11_(1), 598. https://doi.org/10.1038/s41597-024-03434-6.

Luukkonen, C.L., Alzraiee, A.H., Larsen, J.D., Martin, D.J., Herbert, D.M., Buchwald, C.A., Houston, N.A., Valseth, K.J., Paulinski, S., Miller, L.D., Niswonger, R.G., Stewart, J.S., & Dieter, C.A. (2023). Public supply water use reanalysis for the 2000-2020 period by HUC12, month, and year for the conterminous United States. _U.S. Geological Survey data release_. https://doi.org/10.5066/P9FUL880

Majumdar, S., ReVelle, P., Pearson, C., Nozari, S., Minor, B. A., Hasan, M. F., Huntington, J. L., & Smith, R. G. (2026). pyCropWat: A Python Package for Computing Effective Precipitation Using Google Earth Engine Climate Data (v1.2.1). _Zenodo_. https://doi.org/10.5281/zenodo.18706481.

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. _Hydrological Processes_, _36_(11), e14757. https://doi.org/10.1002/hyp.14757.

Ma, Y., Condon, L. E., Koch, J., Bennett, A., Defnet, A., Tijerina-Kreuzer, D., Melchior, P., & Maxwell, R. M. (2026). High resolution US water table depth estimates reveal quantity of accessible groundwater. _Communications Earth & Environment_, _7_(1), 45. https://doi.org/10.1038/s43247-025-03094-3.

Martin, D. J., Niswonger, R. G., Regan, R. S., Huntington, J. L., Ott, T., Morton, C., Senay, G. B., Friedrichs, M., Melton, F. S., Haynes, J., Henson, W., Read, A., Xie, Y., Lark, T., & Rush, M. (2025). Estimating irrigation consumptive use for the conterminous United States: coupling satellite-sourced estimates of actual evapotranspiration with a national hydrologic model. _Journal of Hydrology_, _662_, 133909. https://doi.org/10.1016/j.jhydrol.2025.133909.

Martin, D.J., Regan, R.S., Haynes, J.V., Read, A.L., Henson, W.R., Stewart, J.S., Brandt, J.T., & Niswonger, R.G. (2023). Irrigation water use reanalysis for the 2000-20 period by HUC12, month, and year for the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_. https://doi.org/10.5066/P9YWR0OJ.

Melton, F., Huntington, J., Grimm, R., Herring, J., Hall, M., Rollison, D., Erickson, T., Allen, R., Anderson, M., Fisher, J. B., Kilic, A., Senay, G. B., Volk, J., Hain, C., Johnson, L., Ruhoff, A., Blankenau, P., Bromley, M., Carrara, W., … Anderson, R. G. (2022). OpenET: Filling a Critical Data Gap in Water Management for the Western United States. _JAWRA Journal of the American Water Resources Association_. https://doi.org/10.1111/1752-1688.12956.

Muratoglu, A., Bilgen, G. K., Angin, I., & Kodal, S. (2023). Performance analyses of effective rainfall estimation methods for accurate quantification of agricultural water footprint. _Water Research_, _238_, 120011. https://doi.org/10.1016/j.watres.2023.120011.

Noble, W. et al. (2015). A Case Study in Efficiency — Agriculture and Water Use in the Yuma, Arizona Area. _Yuma County Agriculture Water Coalition_. https://www.azwater.gov/sites/default/files/2022-11/Final%20Yuma%20Report%20021715.pdf.

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