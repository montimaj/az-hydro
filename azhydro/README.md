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
python pipeline.py --steps 1 --run-eda   # regenerate Step 1 EDA figures (opt-in; off by default)
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
python pipeline.py --steps 3b --skip-uq sw-capture-sigma         # skip SW Capture Index w/ σ_GW propagation
```

Step 3g map families support `--skip-maps`:

```bash
python pipeline.py --steps 3g --skip-maps trends                 # era-mean maps + graphical abstract only (skip slow trend-map suite)
```

Step 3h is a lightweight graphical-abstract-only iteration loop (~30 s) that reads the cached Annual_Summaries CSVs from disk — no UQ, no era-mean maps, no trends:

```bash
python pipeline.py --steps 3h                                    # regenerate Figure 1 only (~30 s)
```

Step 3h is excluded from `--steps all` because Step 3g already produces the graphical abstract as part of its full raster-map suite; use Step 3h when iterating on the Figure 1 layout after Step 3 and 3b have already run.

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
| `3b` | Hybrid uncertainty quantification (incl. σ-propagated SW Capture Index and per-well σ disaggregation) |
| `3e` | Well package (per-well Parquet + GPKG locations with uncertainty, incl. SW capture + σ) |
| `3g` | Raster maps, actual vs predicted, and trend analysis for all output categories |
| `3h` | Graphical abstract / Figure 1 only (lightweight; reads `Annual_Summaries/` from disk). Must be explicitly requested — excluded from `--steps all` because `3g` already produces this figure. Intended for iterating on the Figure 1 layout in ~30 s without re-running anything else. |
| `4`  | USGS intercomparison |
| `4b` | CU intercomparison |
| `4c` | CAP/SRP surface-water validation |
| `4d` | Effective precipitation intercomparison |
| `4e` | Non-irrigation vs USGS Public Supply intercomparison |
| `4f` | USGS statewide calibration overview (AZ-wide annual Total GW/SW bars ±1σ vs USGS Circular & OFR 94-476 anchors). Mirrors USGS OFR 94-476 Figure 1 in bar form. |

#### Step 0 sub-steps

| Sub-step | Description |
|----------|-------------|
| `gee` | GEE tile download & mosaic |
| `gw-csv` | GW CSV → per-year shapefiles |
| `vectors` | Reproject vectors |
| `gw-rasters` | GW volume → depth → cropped rasters |
| `streamflow` | Canal density (temporally masked per canal segment) & streamflow rasters |
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
| `sigma-usbr` | Skip σ_USBR — Upper Basin Colorado River streamflow uncertainty (5 USBR CMIP3 ensemble members spanning Rupp 2013 GCM corners + mixed SRES). Captures CAP-delivery uncertainty driven by Wyoming/Colorado/Utah snowpack — the gap σ_MACA cannot reach (MACA only downscales to AZ-local domain). |
| `density-sensitivity` | Skip partition-level diagnostic (density-ratio ±20% + smoothing-sigma sweep {2, 8}) |
| `sigma-total` | Skip σ_total quadrature, basin σ, visualizations, and raster augmentation |
| `sigma-cu` | Skip σ_CU — consumptive use uncertainty (IE × Withdrawal error propagation) |
| `sw-capture-sigma` | Skip SW Capture Index computation with σ_GW propagation. Produces the per-pool SW capture rasters (fraction, depth, volume) with combined λ + σ_total 95 % CI bounds plus per-well σ_capture disaggregation in `Well_Package.gpkg`. Depends on `sigma-total`; skipping means no SW capture outputs are produced (there is no σ-less fallback path). |

#### Step 3g map sub-steps

| Sub-step | Description |
|----------|-------------|
| `trends` | Skip the full Mann-Kendall + Sen's slope trend-map suite (withdrawals, CU, SW capture depth/volume/fraction, per-basin and per-sub-basin trend CSVs). This is the slowest sub-step in Step 3g — per-pixel MK + Sen on 204 annual rasters × ~15 product families × 4 periods takes the bulk of the step's runtime, so skipping is useful when iterating on the era-mean raster maps or the graphical abstract. Era-mean raster maps, σ-component CV maps, Prediction CV/SNR maps, actual-vs-predicted figures, and the graphical abstract are still produced. |

> **Note on skipping individual σ components:** Per-category σ at the *pixel* level (e.g., the per-pixel σ_model array for Irrigation, Non_Irrigation, etc.) is only held in memory during computation and is never written to disk as a separate per-category raster. Per-category σ at the *basin* level, by contrast, **is** persisted to disk as `Uncertainty/{component}/Basin_Sigma_{component}_{category}.csv` (five components × eight partition categories = 40 new CSVs per full Step 3b run), where it is consumed by the σ attribution diagnostic suite in Step 3g; those files are additive and leave the existing total-level `Basin_Sigma_{component}.csv` and `SubBasin_Sigma_{component}.csv` untouched. When `sigma-total` runs, it can reload *total-level* per-component σ from disk (e.g., `Sigma_Model_mm_{year}.tif`), but the per-category σ_total rasters (`Sigma_Total_{cat}_mm_{year}.tif`) will be zero if the individual σ steps were skipped because the per-category pixel arrays they depend on are not on disk. This causes downstream augmented category rasters to have zero σ, which in turn makes σ_CU zero. To get correct per-category uncertainty in the augmented rasters *and* correct per-category basin CSVs for the σ attribution suite, run all individual σ components (σ_MACA through σ_gw) without skipping. Only `density-sensitivity` (the partition-level diagnostic) can be safely skipped without affecting downstream products.

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `all` | Comma-separated step IDs to run (e.g. `"0,1,2a"`) or `"all"`. |
| `--skip-download` | `True` | Skip GEE tile download; use existing tiles on disk. |
| `--download` | — | Force GEE tile download. |
| `--load-files` | `True` | Skip recreating intermediate files that already exist. |
| `--recreate` | — | Force recreation of intermediate files. |
| `--run-eda` | `False` | Opt in to the Step 1 EDA plot suite (histograms, ET-vs-ETo analysis, pumping-distribution analysis, per-basin data-availability summary). EDA is off by default; downstream steps (Step 2, Step 3, Step 3b) reuse the predictor DataFrame without needing the plots, so we don't pay the ~minute-long render cost on every run. Pass this flag together with `--steps 1` (or any step list that includes `1`) to regenerate the figures; passing it with a step list that does not include Step 1 has no effect. |
| `--skip-prep` | — | Comma-separated Step 0 sub-steps to skip. |
| `--skip-eval` | — | Comma-separated evaluation strategies to skip. |
| `--skip-uq` | — | Comma-separated UQ sub-steps to skip (including `sw-capture-sigma`). |
| `--skip-maps` | — | Comma-separated Step 3g map sub-steps to skip (currently `trends`). |
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
concentrating flow where delivery infrastructure exists.  Both canal
density and canal-weighted streamflow are **temporally masked per canal
segment** using historical construction dates and HarDWR SW rights
nearest-neighbor dating (see "Temporal canal masking" below), so
pre-infrastructure pixels correctly receive zero canal features.  A
further refinement could use distance-to-NHD flowlines as a weighting
factor, but this data product is not currently in the pipeline.

#### Temporal canal masking

Each GRAIN canal segment is assigned a `first_delivery_year` using a
three-tier hierarchy:

1. **Named major canal** — matched against a lookup table of 13 regex
   patterns tied to documented construction dates (e.g., SRP/Arizona
   Canal system → 1868, CAP → 1985, Wellton-Mohawk → 1952, Gila
   Gravity Main → 1943).
2. **Unnamed canal with a nearby HarDWR POD** — spatial nearest-neighbor
   to the closest surface-water POD from HarDWR v2.0
   ([Lisk et al., 2024](https://doi.org/10.57931/2475303)) within
   20 km; uses the POD's priority year.  All SW right types
   (irrigation, livestock, domestic, industrial, environmental) are
   included so that basins without irrigation PODs (e.g., Lake Havasu,
   Peach Springs) still receive dates.
3. **No nearby POD** — falls back to the earliest SW right priority
   year in the same GW basin, determined by joining all HarDWR SW PODs
   to the nearest ADWR groundwater basin polygon.

After the three-tier assignment, a **basin-level delivery-start
enforcement** step bumps `first_delivery_year` to the basin's actual
first delivery date (from observed CAP/SRP records) when the segment's
date is earlier.  This handles cases where canal infrastructure was
built before deliveries began to a specific basin — for example, the
CAP canal traverses Tucson AMA from 1985 but actual CAP deliveries to
Tucson did not start until 1990.  Basin delivery-start dates
(`BASIN_DELIVERY_START` in `streamflowops.py`):

| Basin | First delivery | Source |
|---|---|---|
| Phoenix AMA | 1868 | SRP (predates CAP) |
| Pinal AMA | 1924 | San Carlos Irrigation Project |
| Harquahala INA | 1985 | CAP |
| Ranegras Plain | 1989 | CAP |
| Tucson AMA | 1990 | CAP |

Basins not in this table (e.g., Parker, Yuma, Safford) retain the
per-segment dates from the three-tier hierarchy, since they have
pre-CAP diversions from local rivers that predate any basin-level
delivery program.

For each prediction year, `create_canal_density_raster()` and
`create_streamflow_rasters()` include only segments where
`first_delivery_year ≤ year`.  This produces truly time-varying
`Canal_Density_{year}.tif` and `Canal_Weighted_Streamflow_{year}.tif`
rasters that reflect actual infrastructure build-out — for example, CAP
canal pixels appear only from 1985 onward, while SRP canals in Phoenix
are present from 1868.  The regular `Streamflow_{year}.tif` rasters
also gate the CAP Colorado River overlay to 1985 onward, so pre-CAP
years do not receive imported Colorado River streamflow in the CAP
service area.

**Known limitation for unnamed canals:** The nearest-neighbor POD proxy
assumes that a canal segment's construction date correlates with the
priority date of the closest surface-water right.  This is a reasonable
approximation — a lateral branching off the CAP picks up a post-1985
right, while a ditch near old Colorado River diversions picks up a
pre-1900 right — but it is not exact for individual segments.

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

Three λ values (5, 10, 20 m) produce the connectivity-scale bounds
without tunable parameters.  Volume bounds incorporate **both** the
three-λ connectivity envelope **and** the per-pixel σ_total from the
5-component UQ framework, combined via the asymmetric form

```
gw_lower = max(gw - 1.96·σ, 0)
gw_upper = gw + 1.96·σ
vol_lower   = gw_lower × cf_lower     (λ = 5 m, narrow)
vol_central = gw × cf_central         (λ = 10 m)
vol_upper   = gw_upper × cf_upper     (λ = 20 m, wide)
```

so the Lower/Upper columns in `SW_Capture_Time_Series.csv` are the
combined 95 % CI on the capture volume, not a λ-only envelope.  A
per-pixel σ_SW_capture is derived at the central λ as the half-width
of the propagated interval (`σ_cap = 0.5 × (vol_upper − vol_lower) /
1.96`) and is shipped inside the 6-band augmented capture rasters
(band 1 = prediction, band 2 = σ, bands 3–6 = CV, SNR, lower 95 % CI,
upper 95 % CI) — the same schema used by every other output in the
pipeline.  This means the existing per-well disaggregation in
`wellops.py` automatically picks up SW-capture σ and CI columns for
the three pools (`Total_SW_Capture`, `Irrigation_SW_Capture`,
`Non_Irrigation_SW_Capture`) with zero additional configuration.

The computation runs as part of Step 3b's
`run_uncertainty_quantification` (after the per-category rasters have
been augmented with band-2 σ_total), not inside the Step 3 prediction
loop, because the σ needed for the propagation does not exist until
Step 3b finishes.  The skip token `--skip-uq sw-capture-sigma` turns
the whole SW capture step off (no capture outputs are produced in that
case; users who want the old λ-only bounds can run an earlier commit).

Output rasters are organized by GW pumping pool — Total, Irrigation,
Non-Irrigation — under `SW_Capture/{Total,Irrigation,Non_Irrigation}_SW_Capture_Fraction/`
(3-band, dimensionless [0, 1] for λ = 5/10/20 m) and
`SW_Capture/{Total,Irrigation,Non_Irrigation}_SW_Capture_Rasters/{Depth_mm,Depth_ft,Volume_m3,Volume_AF}/`
(6-band augmented: central capture volume + σ_cap + CV + SNR + lower
95 % CI + upper 95 % CI, in mm/ft/m³/AF).  Reading the directory names:
`Total_SW_Capture_Fraction` is "the fraction of Total GW pumping that
captures surface water," and so on for the irrigation and non-irrigation
splits.  The time series CSV
(`SW_Capture/SW_Capture_Time_Series.csv`) carries the same
Lower/Central/Upper schema as before plus three new explicit σ columns
per pool (`{pool}_Capture_Volume_Sigma_AF`).  Era-mean maps are also
produced.

**What "SW capture" actually means in this model.** The index measures
only one specific pathway: well-mediated stream depletion in pixels the
partition step has labeled as `_GW`. Four categories of well/canal
interaction are *not* counted in the capture numerator: (1) direct
canal diversions, which never enter the model because the ML target is
ADWR Well Registry pumping (federal Yuma-area deliveries through WMIDD,
YCWUA, and the Gila Project bypass wells entirely and are reconciled
separately as the ~2.26 MAF water-budget gap); (2) wells filed under
HarDWR surface-water rights, which the partition routes into `_SW`
regardless of whether they are physically pumping ambient groundwater
or river-recharged alluvium — those volumes are tracked under
`Irrigation_SW`/`Non_Irrigation_SW` and never reach the capture index;
(3) any well-mediated SW interaction outside the perennial
canal-delivered footprint (`cw_norm = 0`), since ephemeral stream–
aquifer exchange would require transient groundwater modeling; and
(4) engineered drain networks that recapture irrigation return flow
plus shallow GW pumping and route it as a designed downstream
delivery — for example, the Wellton-Mohawk Irrigation and Drainage
District drain system that collects ~85,000 AF/yr of WMIDD field
return flow plus ~20,000 AF/yr of USBR well-field pumping and routes
it via the Boundary Pumping Plant to Mexico under the 1944 Treaty,
sustaining both Mexico's $280M Mexicali agricultural industry and
the 5,635 ha Cienega de Santa Clara wetland on the Pacific Flyway
([Frisvold et al., 2018](https://doi.org/10.3390/su10051548) §6.7).
The capture fraction is therefore the model's most conservative lower
bound on well-mediated SW depletion: a "low" capture fraction in a
canal-dominated basin like Yuma (~4%) means most of that basin's SW
use is being delivered through canals, through SW-righted wells
already counted in `Total_SW`, or through engineered drain
infrastructure that this index does not represent — not that wells
are causing little impact.

**What is novel here, and what is not.** The base ML method
(XGBoost trained on metered well records with remote-sensing-derived
predictor features) is *not itself novel*. Our group has applied it
across four regions over the last five years before scaling up to
this Arizona statewide application: the Kansas High Plains Aquifer
([Majumdar et al., 2020](https://doi.org/10.1029/2020WR028059);
[Asfaw et al., 2025](https://doi.org/10.1016/j.agwat.2025.109691)),
the Mississippi Alluvial Plain
([Majumdar et al., 2024](https://doi.org/10.1016/j.ejrh.2024.101674);
[Majumdar et al., 2025](https://doi.org/10.1109/IGARSS55030.2025.11243173)),
the Arizona AMA/INA management areas
([Majumdar et al., 2022](https://doi.org/10.1002/hyp.14757)) — the
direct precursor to the present study area — and the broader
Western U.S. effective-precipitation domain
([Hasan et al., 2025](https://doi.org/10.1016/j.agwat.2025.109821)).
Each prior application validated specific framework components:
Kansas validated the original ML-driven withdrawal estimation
against Kansas Geological Survey metered records, the MAP papers
validated transferability under data-scarce conditions and crop-
specific disaggregation, the 2022 Arizona paper validated the GW–
subsidence linkage and basin-level partitioning inside metered AMAs,
and the Hasan et al. 2025 effective-precipitation work validated the
satellite-data-plus-physics-constrained-ML pattern across the
entire Western U.S.

What *is* novel about AZ-Hydro is everything built *on top of* the
base ML method to turn a single-region withdrawal-prediction model
into a complete state-scale water-budget framework. Six contributions
are individually novel and collectively define the contribution of
this study:

1. **Physics-informed feature engineering and predictor stack** —
   pump-capacity-weighted irrigation/non-irrigation fractions,
   canal-weighted streamflow with Gaussian smoothing across canal
   service areas, HarDWR water-rights densities split by water-use
   category, USGS / FORE-SCE LULC bias-corrected at the basin scale
   to preserve NLCD's pixel-level spatial pattern, climate features
   bias-corrected to remove USGS/USBR step jumps, and Ma et al.
   (2026) high-resolution WTD as model inputs. Each adds a
   hydrologically meaningful signal that off-the-shelf predictor
   stacks lack.

2. **Density-ratio GW/SW partitioning with locally observable
   infrastructure** — splits predicted total pumping into eight
   conservation-consistent withdrawal categories (Irrigation,
   Non-Irrigation, Total/Irrigation/Non-Irrigation × GW/SW) using
   ADWR well density vs. HarDWR surface-water rights density,
   modulated by canal-weighted streamflow with focal-max
   normalization. This replaces the global statistical proxies that
   most large-scale studies fall back on with state-specific
   infrastructure observations.

3. **Per-pixel, per-year surface-water capture quantification** — a
   process-informed proxy
   ([Barlow & Leake 2012](https://doi.org/10.3133/cir1376),
   [Condon & Maxwell 2019](https://doi.org/10.1126/sciadv.aav4574))
   that apportions GW pumping into stream-depletion vs. storage-
   mining shares at 2 km annual resolution across an entire state
   and a 204-year window, with three λ values producing physically-
   bounded uncertainty intervals. Existing basin-scale capture
   estimates exist for individual aquifers (e.g. Phoenix AMA,
   Tucson AMA, the Lower Colorado main stem) but each is a bespoke
   MODFLOW–SFR run with its own calibration targets, parameter set,
   and time horizon, and none are mutually comparable across basins.
   **To our knowledge, no prior study has produced a per-pixel,
   per-year apportionment of GW pumping into stream-depletion vs.
   storage-mining shares at this combination of spatial coverage,
   temporal span, and methodology.** This is the single most novel
   product of the study; it is not a substitute for a calibrated
   transient flow simulation in any individual basin, but it
   provides the first consistent first-order screen for where well-
   mediated stream depletion is plausibly significant across an
   entire state's groundwater system in a uniform framework.

4. **A 204-year continuous record** — hindcast (1896–1983),
   historical (1984–2025), and projection (2026–2099) eras all in
   one self-consistent framework, with the projection driven by
   5 GCMs × 2 RCPs × 4 USGS LULC scenarios × 112 streamflow
   ensemble members. No prior single-state water-use study spans
   two centuries with a uniform methodology and per-pixel/per-well
   resolution.

5. **First-of-a-kind statewide irrigation consumptive use dataset
   for Arizona** at 2 km annual resolution, with separate GW-CU and
   SW-CU components consistent with the partitioning, per-well
   disaggregation via the well package, and physics-based error
   propagation. The closest existing product is the USGS NHM HUC12
   monthly irrigation CU reanalysis
   ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909);
   [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)) which
   is national in scope but limited to 2000–2020 at HUC12 monthly
   resolution. ADWR publishes statewide withdrawal totals and an
   irrigation share but does not produce a basin-resolved CU
   product.

6. **Hybrid five-component σ_total uncertainty quantification** —
   σ_MACA (5 GCMs, scenario-based) + σ_model (10 seeds, t-corrected
   sample-based) + σ_irr (irrigation-fraction half-range) + σ_LULC
   (4 USGS scenarios) + σ_GW (5 recent HarDWR well-density
   snapshots, t-corrected sample-based), combined in quadrature
   with physics-based CU error propagation, producing 6-band
   augmented rasters (prediction, σ, CV, SNR, lower/upper 95 % CI)
   for every product and unit. To our knowledge no prior water-use
   ML study at this spatial coverage and temporal span has
   reported a UQ framework of comparable rigor.

**No prior study we are aware of provides this combination of
feature engineering, partitioning, capture quantification,
hindcast/projection coverage, and 5-component UQ for any U.S.
state.** The base ML method is shared with our four prior regional
applications and was validated there; everything in items 1–6 above
is what makes the AZ-Hydro framework state-scale, two-century, and
uncertainty-honest rather than just a model that maps wells to
pixels.

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
Step 3b  ─  Hybrid 5-component σ_total UQ, raster augmentation, σ-propagated SW Capture Index
Step 3e  ─  Well Package (per-well Parquet + GPKG locations with uncertainty, incl. SW capture + σ)
Step 3g  ─  Raster Maps & Trend Analysis for All Output Categories
Step 3h  ─  Graphical Abstract / Figure 1 only (~30 s; explicit opt-in)
Step 4   ─  USGS Intercomparison (Withdrawals, CU, Peff)
```

The SW Capture Index with σ_GW propagation runs **inside** Step 3b
(`run_uncertainty_quantification → compute_sw_capture_with_sigma`)
right after the per-category rasters have been augmented with σ_total
band 2. Skip with `--skip-uq sw-capture-sigma` to turn the whole
capture step off. Skipping produces no capture outputs (the pipeline
deliberately has a single source of truth for capture; there is no
σ-less fallback path).

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
   Canal density is **temporally masked per canal segment** using a
   three-tier dating hierarchy: (1) named major canals matched to
   documented construction dates, (2) unnamed canals dated via spatial
   nearest-neighbor to HarDWR v2.0 SW PODs
   ([Lisk et al., 2024](https://doi.org/10.57931/2475303)) within 20 km,
   (3) basin-level earliest SW right fallback.  For each prediction year,
   only segments with `first_delivery_year ≤ year` are rasterized.
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
between tuning and evaluation.  The current CV metrics show improved R² and RMSE relative to the 
previous study's test metrics, while the current test metrics are similar, confirming that the model generalizes 
well without relying on test-set information during training.

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

**Mean-depth convention (active pumping pixel average).**  `Mean_Depth_mm`
and `Mean_Depth_ft` in the `Annual_Summaries/*.csv` outputs (and the
downstream `Full_Period_Time_Series.csv` consumed by the graphical
abstract and time-series plots) are averaged only over **pixels where
the prediction is ≥ 5 mm/yr** — the "active pumping pixel" threshold.
This filters out near-zero predictions from the basin-median LU-only
fill in `apply_ml_well_density_override` (sub-1 mm/yr values at every
crop / urban pixel in AZ), so the reported mean reflects per-pixel
irrigation / M&I intensity rather than an AZ-wide dilution.  Volume
sums (`Volume_m3`, `Volume_AF`) are unchanged — they remain full-state
nansums so total volumes are conserved.  Empirically, the 5 mm
threshold removes 79 % of pixels but loses only 0.3 % of the 2024
volume (6.823 → 6.804 MAF), while shifting the reported mean depth
from 30 mm (diluted) to 145 mm (intensity-scale, matching typical
irrigation application rates).  The same 5 mm threshold is applied to
per-category, per-basin, per-sub-basin, and σ_LULC-scenario
aggregations for consistency.  σ statistics (sigma rasters / sigma
basin CSVs) use the legacy `threshold=0` convention — every pixel's
uncertainty is meaningful regardless of magnitude.

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
four-panel publication figure:

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
- **Panel (d)**: Key Contributions — five single-line bullets summarising
  the 2 km × 204-year coverage, the first statewide irrigation CU dataset,
  out-of-distribution validation against ADWR and USGS, the novel SW
  capture index, and the hybrid 5-component σ_total UQ framework.

Saved as `{prediction_dir}Graphical_Abstract_Fig1.png` (600 dpi).

**Fast iteration loop (Step 3h).** When iterating on the Figure 1 layout
after Step 3 and Step 3b have already run, use Step 3h instead of a full
Step 3g rerun:

```bash
python pipeline.py --steps 3h   # ≈ 30 s; reads Annual_Summaries CSVs from disk
```

Step 3h calls `create_graphical_abstract_only()`, which reads
`Annual_Summaries/Total_Predicted.csv`, `Annual_Summaries/Basin_Total.csv`,
and `Uncertainty/Sigma_Total/Uncertainty_Summary_Total.csv` from disk and
calls `vizops.create_graphical_abstract()` directly — no era-mean raster
maps, no trend analysis, no σ recomputation, no GEE traffic.  Step 3h is
deliberately excluded from `--steps all` because Step 3g already produces
the graphical abstract as part of its full raster-map suite.


#### 3d. Hybrid uncertainty quantification (`uncertaintyops.run_uncertainty_quantification()`)

Pixel-level uncertainty is quantified for every year (1896–2099) by
computing **six independent error components** and combining them via
quadrature:

$$\sigma_{\text{total}} = \sqrt{\sigma_{\text{MACA}}^2 + \sigma_{\text{model}}^2 + \sigma_{\text{irr}}^2 + \sigma_{\text{LULC}}^2 + \sigma_{\text{gw}}^2 + \sigma_{\text{USBR}}^2}$$

Each component isolates a specific source of prediction uncertainty.
Every ensemble member's total prediction is also partitioned into the 8
withdrawal categories *before* computing std, yielding per-category σ for
each component.  Per-category σ_total is then obtained by the same
quadrature formula applied category-wise.

**Climate-driver decomposition.**  Two of the six components capture
climate uncertainty along *different geographic axes*:

- **σ_MACA** — AZ-local downscaled CMIP5 climate (5 GCMs).  Drives
  ET / ETo / precipitation, which controls AZ-local watershed
  streamflow (Salt, Verde, Gila, Bill Williams) and irrigation demand.
- **σ_USBR** — Upper Basin Colorado River streamflow (5 USBR CMIP3
  ensemble members).  Drives Lees Ferry inflow → CAP imports.

Geographic decoupling is essential: ~92 % of Colorado River flow at
Lees Ferry derives from Wyoming / Colorado / Utah snowpack, which AZ-
local MACA downscaling does not capture.  σ_USBR fills that gap.

**Sample-based vs scenario-based components.**  σ_model (10 random seeds),
σ_gw (5 recent HarDWR well-density snapshots, 2020–2024), and σ_USBR
(5 USBR ensemble members) are *sample-based*: their ensemble members
are random draws from a larger population, so Student's t-distribution
critical values are used instead of z = 1.96 to account for small-N
estimation uncertainty (t₉ = 2.262 for σ_model, t₄ = 2.776 for σ_gw
and σ_USBR).
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

##### σ_USBR — Upper Basin Colorado River streamflow uncertainty (all years, 1896–2099)

σ_USBR captures **inter-USBR-ensemble-member spread of CAP delivery**
driven by Upper Basin Colorado River streamflow uncertainty — the
gap σ_MACA cannot reach (MACA downscales to AZ-local domain only,
not the Wyoming / Colorado / Utah snowpack that produces ~92 % of
Lees Ferry flow).

For each year × USBR ensemble member, the helper perturbs two
pathways across **all CO-river-served basins** (CAP service area
plus the COLORADO RIVER mainstem corridor — Parker / CRIT / Mohave
Valley / Yuma — i.e. any pixel whose surface watershed contains a
gauge in ``streamflowops.USBR_DERIVED_GAUGES``: Lees Ferry, Imperial
Dam, or CAP Canal at Havasu).  The ML prediction and the partition
both re-run per member:

1. **ML-feature pathway (additive on `streamflow_mm`)**.
   ``streamflow_mm`` at any pixel equals
   ``local_watershed_runoff + co_flow``, where ``co_flow`` is the
   Lees-Ferry-derived component injected by
   ``streamflowops._create_streamflow_rasters``.  Per-pixel
   ``co_flow_mm`` is the sum of two contributions:
   - **CAP overlay** (uniform across CAP service-area pixels, added
     post-1985):
     ``cap_co_flow_mm = LF_ens_mean × m3s_to_mm_yr / cap_area_m2``.
   - **Surface-watershed component** (per-watershed,
     LF-share-weighted): for each watershed,
     ``ws_co_flow_mm = lf_share × LF_ens_mean × m3s_to_mm_yr /
     ws_area_m2``, where ``lf_share`` = (# LF-derived gauges) /
     (# total gauges in the watershed).  The COLORADO RIVER
     watershed has ``lf_share = 1.0`` (its single mapped gauge is
     LF-derived); Bill Williams has ``lf_share = 0.5`` (CAP Canal
     at Havasu spatial-joins into it alongside the local Bill
     Williams gauge).
   The perturbation is
   ``streamflow_mm_m = central + (ratio_m − 1) × co_flow_mm``,
   preserving the local-watershed component exactly.
2. **Partition pathway (multiplicative on `cw_streamflow` +
   SW rights)**.  Restricted to **CAP pixels only** (post-1985).
   ``canal_weighted_streamflow_mm``, ``irr_sw_rights_density``,
   ``nonirr_sw_rights_density``, and ``sw_rights_density`` at CAP
   pixels are multiplied by the member's ratio
   (``member_annual_mean / ensemble_annual_mean``).  Not extended to
   CO-watershed pixels because (a) it would over-scale local Bill
   Williams flow, and (b) senior mainstem priority makes the
   partition response physically rigid at non-CAP CO pixels.

**Era-specific gating** (per-pathway):
- **CAP pixel perturbation** (both pathways): gated to
  ``year ≥ partops.CAP_OPERATIONAL_START`` (1985, Phoenix-reach
  completion).
- **CO watershed pixel perturbation** (ML-feature pathway only):
  gated to ``year ≥ sfops.USBR_DATA_START`` (1950, USBR CMIP3
  ensemble first year).  No pre-1950 backfill — the per-member
  1950-2005 long-term mean ratios collapse to ~1.0 (std ≈ 0.011)
  by construction (averaging ``member/ensemble_mean`` over many
  years drives each member's mean ratio toward 1.0), so a
  climatological backfill would produce misleading near-zero σ
  pre-1950 implying we modeled it.  The honest answer: σ_USBR is
  **structurally absent before 1950** because the USBR ensemble
  carries no inter-member disagreement on climatological-mean
  Lees Ferry flow, only on year-to-year variability.

Per-pixel σ across members:

$$\sigma_{\text{USBR}}(x, y, t) = \text{std}\bigl[\hat{y}_{m_1}, \hat{y}_{m_2}, \hat{y}_{m_3}, \hat{y}_{m_4}, \hat{y}_{m_5}\bigr]$$

**Member selection (Rupp 2013 corner-spanning + mixed SRES).** Five
USBR CMIP3 members chosen as CMIP3-equivalents of the σ_MACA
Rupp 2013 GCM set, each at the SRES emissions scenario most consistent
with the GCM's climate corner:

| USBR member | ≈ MACA equivalent | Climate corner | SRES |
|---|---|---|---|
| `a1b.ncar_ccsm3_0.1` | CCSM4 | Center / median | A1B |
| `b1.cnrm_cm3.1` | CNRM-CM5 | Cool-wet | B1 |
| `a2.ukmo_hadcm3.1` | HadGEM2-ES365 | Hot-dry | A2 |
| `a2.miroc3_2_medres.1` | MIROC-ESM-CHEM | Hot-wet | A2 |
| `b1.inmcm3_0.1` | inmcm4 | Cool-dry | B1 |

This sampling spans **both the GCM-corner and emission-scenario
axes** within σ_USBR, while remaining methodologically consistent
with σ_MACA's Rupp 2013 selection.

**Behavior across eras (region-specific gating):**

| Era | CAP pixels | CO watershed pixels |
|---|:---:|:---:|
| 1896–1949 | 0 (CAP not built) | 0 (pre-USBR-ensemble; no inter-member spread to model) |
| 1950–1984 | 0 (CAP not built) | σ > 0 (year-specific USBR ratios) |
| 1985–2025 | σ > 0 (CAP overlay + partition pathway) | σ > 0 |
| 2026+ projection | **largest σ_USBR contribution** (member spread up to ~50 % of CO inflow in dry years) | σ > 0 |

The 1949→1950 boundary is a known data-availability discontinuity
(USBR CMIP3 record begins 1950), not a physical regime change.
Pre-1950 σ_USBR is reported as zero rather than backfilled because
per-member 1950-2005 long-term mean ratios collapse to ~1.0 (std
≈ 0.011) — there is no inter-member spread on the climatological
mean to extrapolate, only on year-to-year variability.  Reporting
it as "modeled and ≈ 0" would mislead; reporting it as "structurally
absent" is honest.

**Scope (current implementation).**  σ_USBR uses Lees Ferry as the
sole USBR-perturbation site, applied at all pixels whose surface
watershed mean includes one of three LF-derived USGS gauges
(``streamflowops.USBR_DERIVED_GAUGES``: 09380000 Lees Ferry,
09429490 Above Imperial Dam, 09426650 CAP Canal at Havasu).  This
expands σ_USBR coverage from CAP-only to **all CO-river-served AZ
basins**: CAP service area (Maricopa / Pinal / Pima) plus the
COLORADO RIVER mainstem corridor (Mohave / La Paz / Yuma counties)
plus the Bill Williams sub-basin (LF-share = 0.5 via the CAP Canal
at Havasu spatial join).  SRP / Salt / Verde watersheds are AZ-
internal and remain covered by σ_MACA only.

##### σ_total — Combination of components and aggregation across space

**Within a basin (across components):** the six components (MACA,
Model, Irr, LULC, GW, USBR) are physically independent uncertainty
axes (different driver classes) → combined in **quadrature**:
```
σ_basin² = σ_MACA² + (t_Model · σ_Model)² + σ_Irr² + σ_LULC²
         + (t_GW · σ_GW)² + (t_USBR · σ_USBR)²
```
Sample-based σ (Model, GW, USBR) are scaled by t/z before squaring,
so the resulting σ_basin already incorporates the t-correction.

**Across basins (basin → AMA / AZ aggregate):** **LINEAR SUM**, not
quadrature.  All six components are scenario-driven via *shared*
ensemble members across basins (same 5 GCMs / 10 model seeds / 5
USBR members / 4 LULC scenarios / 5 HarDWR snapshots / 2 IrrMapper
scenarios perturb every basin) → per-basin σ values are perfectly
correlated (a "hot-dry GCM" is hot-dry at every basin
simultaneously).  For perfectly correlated random variables,
`Var(X + Y) = (σ_X + σ_Y)²` → linear sum is the correct AZ-total.

```
σ_AZ_total = Σ basins σ_basin
```

Quadrature across basins would assume basin independence (only
valid for *disjoint* ensembles, e.g. if basin A used GCMs {1-5} and
basin B used GCMs {6-10}, no overlap).  Our shared-ensemble design
mandates linear sum.  Earlier versions of this code used basin-
quadrature and produced AZ-total σ ribbons ~3-4× too tight (peak-
year 95 % CI upper limit ~8.5 MAF, missing USGS at 1980 = 8.93,
ADWR = 9.50).  Linear sum across basins gives a ribbon that
correctly covers USGS / ADWR observations at all post-1950 anchors.

**Independence within basin: physical justification.**  The six
components target different uncertainty axes:
- σ_MACA: AZ-local climate (Salt/Verde/Gila watersheds)
- σ_USBR: Upper Basin Colorado climate (independent of σ_MACA at
  the streamflow-physics level — Wyoming snowpack ↔ AZ ET have
  weak teleconnection)
- σ_Model: ML training noise (independent of climate)
- σ_Irr / σ_LULC: scenario-driven land use (correlated with climate
  via SRES emission pathways but applied as independent uncertainty
  axis here)
- σ_GW: well-density snapshot uncertainty (registry data quality,
  independent of all others)

Quadrature within basin is a slight over-estimate (some weak
positive correlation between climate and land-use SRES) but
acceptable as a conservative UQ bound.

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
| `{Sigma_X}/Basin_Sigma_{X}_{Category}.csv` | Per-basin, per-year σ split by withdrawal category (8 files per component — `Irrigation`, `Non_Irrigation`, `Irrigation_GW`, `Irrigation_SW`, `Non_Irrigation_GW`, `Non_Irrigation_SW`, `Total_GW`, `Total_SW`). Basin-level only — sub-basin per-category CSVs are deferred. Consumed by the σ attribution diagnostic suite rendered in Step 3g. |

CSV columns (identical across total, category, and sub-basin files):
`Year, Region, Mean_Volume_m3, Sigma_Volume_m3, Mean_Volume_AF,
Sigma_Volume_AF, CV, Lower_95CI_m3, Upper_95CI_m3, Lower_95CI_AF,
Upper_95CI_AF, N_Members`.

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
actual vs predicted), `{prediction_dir}Raster_Maps/Trend_Analysis/`
(pixel-level trend maps, basin choropleth maps, and zonal statistics
CSVs), and `{prediction_dir}Raster_Maps/Sigma_Attribution/` (see
the σ attribution diagnostic suite below).

**σ attribution diagnostic suite**
(`vizops.create_sigma_attribution_map` et al.) — Basin-scale diagnostic
maps that classify each groundwater basin by whether its total
uncertainty is dominated by **management**-driven factors (σ_irr,
σ_LULC, σ_GW — fixable by better data), **climate**-driven factors
(σ_MACA + σ_USBR — inherent to the GCM scenario spread for AZ-local
forcing and Upper Basin Colorado River streamflow respectively), or
the **model** training-procedure floor (σ_Model — inherent to the ML
ensemble). The decomposition writes the three variance shares
(`σ_mgmt² / σ_total²`, `σ_clim² / σ_total²`, `σ_model² / σ_total²`)
per basin per era per withdrawal pool, summing to 1.0.

Two complementary visual products are produced for every pool and era,
sharing the same underlying three-way decomposition:

- **Binary 5-bin discrete choropleth** (NV-style). The color axis
  metric varies with the era because σ coverage differs: in the
  Projection era it is `Mgmt / (Mgmt + Clim)` (blue = climate-
  dominated, red = management-dominated), and in the Hindcast and
  Historical eras it is `Mgmt / (Mgmt + Model)` (purple = model-
  dominated, red = management-dominated) because σ_MACA is zero by
  design in those eras. In the Projection era, basins where σ_Model is
  the single largest variance contributor are flagged with a **bold
  black polygon edge**, and an on-figure disclosure box reports the
  count and median Model share so users cannot miss σ_Model's magnitude
  when reading the Clim-vs-Mgmt classification.
- **Ternary RGB-mixed choropleth**. Each basin's color is a continuous
  RGB mix where R = Management share, G = Model share, B = Climate
  share (scaled by 0.9 to avoid saturation). A small equilateral-
  triangle inset at the bottom-left labels the three corners and shows
  the color gradient. The ternary map works identically in every era
  without an era-specific classification swap; in Hindcast and
  Historical the Climate component is structurally zero so basins fall
  on the red↔green edge of the triangle, which is the correct visual
  disclosure that climate is unrepresentable there.

Eight figure families are produced per full Step 3g run (all basin-
level — sub-basin variants are deferred because ADWR stewardship
decisions are made at the basin scale):

| # | Family | Pools | Files |
|---|---|---|---|
| 1 | Binary headline withdrawal | `Total_GW`, `Total_SW` | 3 PNGs (one per era), 1×2 panels |
| 2 | Binary detailed withdrawal | `Irrigation_GW`, `Irrigation_SW`, `Non_Irrigation_GW`, `Non_Irrigation_SW` | 3 PNGs (one per era), 1×4 panels |
| 3 | Binary σ_CU attribution | `Irrigation_CU`, `Irrigation_GW_CU`, `Irrigation_SW_CU` | 3 PNGs (one per era), 1×3 panels |
| 4 | Ternary headline withdrawal | `Total_GW`, `Total_SW` | 3 PNGs, 1×2 panels |
| 5 | Ternary detailed withdrawal | same as (2) | 3 PNGs, 1×4 panels |
| 6 | Ternary σ_CU attribution | same as (3) | 3 PNGs, 1×3 panels |
| 7 | Per-year stacked-area timeseries | `Total_GW`, `Total_SW` × 8 headline basins | 1 PNG, 8×2 panels |
| 8 | Projection bubble scatter | `Total_GW`, `Total_SW` | 1 PNG, 1×2 panels |

**σ_CU attribution propagation.** Consumptive-use σ is propagated via
`σ_CU = √((IE × σ_wd)² + (wd × σ_IE)²)`. The three-way decomposition
maps each CU pool to its parent withdrawal pool
(`Irrigation_CU → Irrigation`, etc.), reads the parent's per-category
σ CSVs, and applies the basin-level mean IE (from the NHM cache
written in Step 3b at `Uncertainty/Sigma_CU/NHM_IE/
NHM_basin_IE_cache.csv`) as a scalar multiplier on each term. σ_IE is
absorbed into the management class because it is inherently an
irrigation-data-quality problem.

**Companion CSVs.** Seven long-format CSVs are written alongside the
figures, one per era × attribution type:

- `Sigma_Attribution_{Hindcast,Historical,Projection}.csv` — binary +
  ternary withdrawal attribution share the same CSV per era (their
  colors are derived from the same underlying three shares). Contains
  one row per basin × pool with columns `Era, Pool, Region,
  Sigma_MACA_m3, Sigma_USBR_m3, Sigma_Irr_m3, Sigma_LULC_m3,
  Sigma_GW_m3, Sigma_Model_m3, Mean_Wd_m3, Sigma_Mgmt_m3,
  Sigma_Clim_m3, Sigma_Model_Total_m3, Sigma_TotalQ_m3, Mgmt_Share,
  Clim_Share, Model_Share`. The Climate class is `σ_clim² =
  σ_MACA² + σ_USBR²` (AZ-local GCM forcing + Upper Basin streamflow
  in quadrature, treated as independent climate axes).
- `Sigma_CU_Attribution_{Hindcast,Historical,Projection}.csv` — same
  columns plus `IE_Mean, IE_Std`, for the CU variant.
- `Sigma_Attribution_Timeseries.csv` — per-year three-way shares for
  the eight headline basins × two pools.

The attribution suite complements the per-component mm σ std-dev
raster maps that are rendered earlier in Step 3g: those show the
**absolute** σ per component, while the attribution suite shows the
**relative** class share per basin and answers the stewardship
question of which lever (better data vs better scenarios vs bigger
ensemble) would most reduce uncertainty in a given basin.

**Skipping the trend-map suite (`--skip-maps trends`).** The
Mann-Kendall + Sen's slope computation is per-pixel × 204 years ×
~15 product families × 4 periods and dominates the Step 3g runtime.
When iterating on the era-mean raster maps, the actual-vs-predicted
maps, the σ-component CV maps, or the graphical abstract, skip the
trend suite with:

```bash
python pipeline.py --steps 3g --skip-maps trends
```

All era-mean maps, Prediction CV / SNR maps, σ-component CV maps,
actual-vs-predicted maps, SW Capture era maps, and the graphical
abstract are still produced.  The `Trend_Analysis/` directory is
simply not touched, so existing trend maps from previous runs remain
in place.  If you only want the graphical abstract, Step 3h is the
fastest path (~30 s vs ~5 min for `--steps 3g --skip-maps trends`).

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
| `{Cat}_{unit}_sigma` | σ (capacity-weighted share of pixel σ, correlated-sum rule) |

Categories include 9 withdrawal categories (Total + 8 partitions),
3 CU categories (Irrigation_CU, Irrigation_GW_CU, Irrigation_SW_CU),
and 3 SW capture categories (Total_SW_Capture, Irrigation_SW_Capture,
Non_Irrigation_SW_Capture).  Because the SW Capture rasters are now
6-band augmented (written by Step 3b's `compute_sw_capture_with_sigma`
with σ_GW propagation baked in), the disaggregation loop in
`create_well_package` picks up their band-2 σ automatically through
the same `src.count >= 6` branch that handles the withdrawal and CU
categories — no SW-capture-specific code path is needed.  Every well
therefore carries `{pool}_Capture_Volume_{unit}`,
`{pool}_Capture_Volume_{unit}_sigma` columns in the parquet for all
three capture pools.  The SW capture category names refer to the
surface water captured by each GW pumping pool: e.g.
`Total_SW_Capture` is "SW captured by Total GW pumping" within the
parent `SW_Capture/` folder context.

**Per-well σ disaggregation rule.** The well package applies the
**correlated-sum rule** `σ_well_i = w_i × σ_pixel` (not quadrature),
where `w_i` is the per-well capacity-proportional weight summing to
1 across all active wells in a pixel.  This reflects that per-well
contributions to a pixel's withdrawal / CU / capture share a common
pumping-σ driver (the same pixel-level σ_total propagated through the
same downstream fraction or coefficient), so they are perfectly
correlated rather than independent.  The correlated-sum rule preserves
`Σ σ_well_i = σ_pixel` across the wells in each pixel, which is the
right mass-conservation identity for this physical setup.  95 % CI
bounds on per-well values are derived by consumers on-the-fly as
`ci_lower = max(central − 1.96 × sigma, 0)` and
`ci_upper = central + 1.96 × sigma`; they are not stored as separate
columns to keep the parquet compact.

**Caveat:** Per-well σ assumes pixel-level uncertainty distributes
proportionally to capacity weight.  True per-well uncertainty would
require well-specific error models (pump efficiency variability,
metering error distributions) that are not currently available.

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

**Out-of-sample validation.** The density-ratio GW/SW partitioning is
not trained on any delivery data; the XGBRF labels are total GW pumping
from ADWR meters, and the GW/SW split is driven by
infrastructure-proxy features (well density vs. SW rights density,
modulated by canal-weighted streamflow).  The agreement between ML
`Total_SW` and observed CAP/SRP deliveries validates that the
density-ratio mechanism captures the real spatial and temporal pattern
of surface-water use without any calibration step.

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

#### Step 4f — USGS statewide calibration overview (`run_usgs_az_calibration_overview()`)

Generates AZ-wide annual `Total_GW` and `Total_SW` bar plots with ±1σ
error caps and overlays the per-source USGS Circular and OFR 94-476
anchors as red triangles — a direct visual analogue of USGS OFR 94-476
(Anning & Duet 1994) Figure 1.  Reads the model statewide annual values
from `Annual_Summaries/Total_GW.csv` and `Total_SW.csv`, derives σ per
year per category by spatial quadrature of
`Uncertainty/Sigma_Total/Rasters/Sigma_Total_{cat}_mm_{year}.tif`, and
loads the published anchors from
`Data/Inputs/USGS WU/USGS_AZ_Water_Use_1950_1980.csv` (rows whose
`Category` contains "Total").

The overview renders the calibration table from the README's
"Calibration targets vs final model outputs" subsection at a glance —
each USGS triangle should sit on or very near the corresponding model
bar.  Pre-1950 anchors come from OFR 94-476 (GW only); post-1950 from
USGS Circulars (115, 398, 456, 556, 676, 765, 1001, 1004, 1081, 1200,
1268, 1344, 1405, 1441), with both GW and SW where the source reports
them.  Default plot range is 1915–2017 (USGS coverage); SW panel
starts at 1950.  Outputs:

- `USGS_Calibration_Bars/USGS_AZ_Calibration_Bars.csv` — side-by-side
  Year × {Model_GW_kAF, Model_SW_kAF, Sigma_Model_GW_kAF,
  Sigma_Model_SW_kAF, USGS_GW_kAF, USGS_SW_kAF, USGS_Source}.
- `USGS_Calibration_Bars/USGS_AZ_Total_GW_Bars.png` — bar comparison
  for Total_GW (1915–end_year).
- `USGS_Calibration_Bars/USGS_AZ_Total_SW_Bars.png` — bar comparison
  for Total_SW (1950–end_year; pre-1950 USGS reports no SW separately).

Note on σ magnitude: the spatial quadrature `√(Σ σ_pixel²)` assumes
per-pixel errors are independent, which is conservative — the AZ-wide
σ caps are visibly small (≤1–2 % of the bar value).  This matches the
convention used by `_plot_basin_sigma_time_series` for AZ-wide
aggregation.  Per-basin σ is much larger (see basin σ plots) but
suppressed when summed across ~50 basins under the independence
assumption.

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
  into per-year canal-density layers.  Each GRAIN segment is assigned a
  `first_delivery_year` via `assign_canal_delivery_years()`, and only
  segments active by that year are rasterized.
- **`assign_canal_delivery_years()`** — Assigns a `first_delivery_year` to
  each GRAIN canal segment using a three-tier hierarchy: (1) named major
  canals matched against 13 regex patterns tied to documented construction
  dates (e.g., SRP 1868, CAP 1985, Wellton-Mohawk 1952), (2) unnamed
  canals dated via spatial nearest-neighbor to the closest HarDWR v2.0 SW
  POD within 20 km, (3) basin-level earliest SW right fallback.  A final
  basin-level enforcement step (`BASIN_DELIVERY_START`) bumps segments
  to the basin's actual first delivery year from CAP/SRP records when it
  is later than the segment's construction date.  Module-level constants:
  `CANAL_FIRST_DELIVERY` (per-canal regex patterns) and
  `BASIN_DELIVERY_START` (per-basin first delivery years).

### `partitionops.py` — Water-budget partitioning

Decomposes total annual withdrawal predictions into eight categories using
ancillary data already in the predictor stack:

| Category | Derivation |
|---|---|
| **Irrigation** | `total × irr_cap_frac` (pump-capacity-weighted, era-dependent override) |
| **Non_Irrigation** | `total × (1 − irr_cap_frac)` (overridden by per-pixel LULC partition for year ≥ 1986 in non-urban AMAs) |
| **Irrigation_GW** | `Irrigation × irr_gw_frac` (era-weighted density ratio) |
| **Irrigation_SW** | `Irrigation − Irrigation_GW` (+ NonIrr_SW excess routing post-1986) |
| **Non_Irrigation_GW** | `Non_Irrigation × nonirr_gw_frac` (era-weighted density ratio) |
| **Non_Irrigation_SW** | `Non_Irrigation − Non_Irrigation_GW` |
| **Total_GW** | `Irrigation_GW + Non_Irrigation_GW` |
| **Total_SW** | `Irrigation_SW + Non_Irrigation_SW` |

All partitions use subtraction from the parent total (e.g.,
`nonirr_sw = nonirr − nonirr_gw`) to guarantee exact budget closure
with no floating-point drift.

#### Pixel retention (`keep` mask)

```
keep = has_well_for_retention | has_smooth_canal | has_crop | has_urban
```

The retention rules are era-dependent. Pre-1948 uses tight constraints
to bracket sparse historical pumping records; 1948+ opens to standard
full-LULC retention.

| Era | `has_well_for_retention` | `has_crop` / `has_urban` |
|---|---|---|
| `year < 1922` | `well > 0 & (cf > 0 | uf > 0)` (LULC intersection) | False (LU-only blocked) |
| `1922–1929` | `well > 0` (year-specific registry only) | False (LU-only blocked) |
| `1930–1937` | `well > 0` | LU-only ramp: cf > thresh ramping 0.90 → 0.50 (see below) |
| `1938–1944` | `well > 0 & (cf > thresh | uf > thresh)` (intersection ramp) | False (LU-only blocked) |
| `1945–1947` | `well > 0` | Standard (cf > 0, uf > 0) |
| `1948+` | `well > 0` (post-infill, includes phantoms) | Standard (cf > 0, uf > 0) |

**Intersection threshold ramps** (avoid hard regime cliffs):

LU-only retention 1930-1937 (gradually opens cf threshold):

| Year | cf threshold |
|---|---|
| 1930 | 0.90 |
| 1931 | 0.80 |
| 1932 | 0.72 |
| 1933 | 0.65 |
| 1934 | 0.58 |
| 1935 | 0.53 |
| 1936 | 0.51 |
| 1937 | 0.50 |

Well&LULC intersection 1938-1944 (gradually loosens to baseline 0.7):

| Year | cf threshold |
|---|---|
| 1938 | 0.95 |
| 1939 | 0.90 |
| 1940 | 0.80 (tight to match USGS 1940 = 1.80 MAF anchor) |
| 1941 | 0.75 |
| 1942–1944 | 0.70 |

Pre-1922 strict intersection reflects the low-development era (USGS
shows < 0.20 MAF total water use through 1920). The 1938 cliff in
ML-predicted per-pixel pumping (mean depth jumps 129 mm → 228 mm at
1938 due to features the model interprets as a regime change) is
mitigated by tightening the well&LULC intersection at 1938 (very
few pixels retained), then gradually loosening to the baseline 0.7
by 1942.

#### Density-ratio GW/SW split with CW-weighted smoothing

The GW/SW split uses ADWR well density against HarDWR SW rights density
weighted by canal-weighted streamflow and Gaussian-smoothed to spread
each SW POD's influence over its canal service area:

```
sw_smoothed     = gaussian_filter(irr_sw_rights_density × canal_weighted_streamflow, σ=_sw_sigma)
irr_gw_frac     = (gw_weight × irr_well_density)    / (gw_weight × irr_well_density    + sw_smoothed)
nonirr_gw_frac  = (gw_weight × nonirr_well_density) / (gw_weight × nonirr_well_density + nonirr_sw_smoothed)
```

A POD at a major canal headgate (high `cw_streamflow`) gets proportionally
more influence than a POD at a dry wash. The Gaussian spread distributes
that capacity-weighted influence across the canal service area, solving
the spatial mismatch between point-source SW PODs and diffuse GW wells.

**`gw_weight`** is era-dependent (calibrated to USGS/ADWR GW% anchors):

| Era | `gw_weight` | Rationale |
|---|---|---|
| Pre-1945 | 5.0 | USGS pre-1945 ~100 % GW (no/minimal SW) |
| 1930–1935 (override) | 10.0 | USGS Total_GW peaks here; tight σ + high weight together push canal-fringe SW → GW |
| 1945–1984 | 2.0 (constant; 1985 also held at 2.0) | Pre-CAP era, GW-dominant USBR ~67 % statewide |
| 1985–1989 (ramp) | linear 2.0 → 0.2 | Post-GMA + CAP coming online |
| 1990–1997 | 0.2 | Post-CAP, SW-dominant |
| 1998–2007 (mid-CAP bump) | 0.5 | USGS shows GW% rebound at 2000/2005 (Phoenix wells came back online); flat-0.2 under-shoots |
| 2008+ | 0.2 | SW-dominant (USGS/ADWR ~42 % GW) |

The 1930–1935 bump and 1998–2007 mid-CAP bump are localized
corrections to specific anchor periods that the smooth ramp alone
cannot match.

**`_sw_sigma`** (Gaussian smoothing kernel) uses **piecewise-linear
ramps between anchor years** to avoid step changes:

Pre-1948 anchors (interpolated linearly between):

| Year | σ | Rationale |
|---|---|---|
| 1912 | 0.0 | Pre-Yuma Project — minimal SW |
| 1915 | 1.5 | Yuma Project peak (Colorado River SW dominated) |
| 1917 | 0.3 | Pinal/SRP early canal era |
| 1929 | 0.3 | end of Pinal era |
| 1935 | 0.0 | well-drilling boom era (USGS 1935 = 1.20 MAF, mostly GW) |
| 1940 | 0.3 | Gila Project deliveries start |
| 1945 | 1.0 | Gila dam era |
| 1948 | 1.5 | pre-GMA baseline |

Post-1948 schedule:

| Era | σ | Rationale |
|---|---|---|
| 1948–1955 | 1.5 | Pre-GMA SRP era |
| 1956–1964 | 1.0 (default) | Pre-GMA |
| 1965–1984 | linear ramp 1.0 → 4.0 | SRP canals matured + CAP construction |
| 1973–1977 (override) | 2.0 | mid-1970s drought reduced SW deliveries → tighter halo lifts GW% |
| 1985–2002 | 4.0 | CAP/SRP full service area |
| 2003–2010 (override) | 3.0 | CAP-era ag-retirement window — model SW was excess; tighter halo brings SW down toward USGS |
| 2011–2021 | linear ramp 4.0 → 6.0 | Bridges 2010→2011 ML drop |
| 2022+ | 6.0 | Mature CAP — wider halo brings GW% into ADWR 41–42 % band |

**Pre-1945 SW-gate**: SW share is restricted to pixels with explicit
canal access (`has_smooth_canal | has_direct_canal`) or registered
SW rights (`sw_rights_density > 0`). At all other pre-1945 pixels,
any SW share computed by the density ratio is collapsed back into GW.
Matches the USGS narrative that pre-CAP pumping was essentially all
GW outside SRP/Yuma canal service areas.

Both well and SW-rights densities are per-year rasters gated by
installation/priority dates. Where the denominator is zero,
`gw_frac` defaults to 1.0 (100 % GW).

#### Irrigation fraction (`irr_frac`) overrides

The Irr/NonIrr split uses `irr_capacity_fraction` (PUMPRATE-weighted
ratio of IRRIGATION wells to all active wells per pixel) as the
baseline. Era-dependent overrides:

| Era | Pixel condition | Rule | Rationale |
|---|---|---|---|
| `year ≤ 1980` | `uf < 0.3` (rural + LU halo) | `irr_frac = 0.95` flat | USGS shows ag was 89–97 % of AZ-total pumping through 1980 |
| `year ≤ 1980` | `uf ≥ 0.3` (urban core, any basin) | `irr_frac = 1 − uf` (clipped) | M&I share at urban cores routed to NonIrr regardless of AMA — Flagstaff / Lake Havasu / Bullhead / Phoenix-Tucson cores |
| `1981 ≤ year ≤ 1985` (NON-AMA) | — | `irr_frac = 1 − uf` (clipped to ≥ 0.05) | CAP-startup transition; rural basins still mostly irrigation |
| `1981+` AMA | — | natural `irr_capacity_fraction` (cf-floor) | M&I growth in urban AMAs |
| `year ≥ 1986` NON-URBAN-AMA | — | post-1985 LU-aware branch (only_crop / both_lu / only_urban / pure_desert / ag_halo) | Modern era |

**`URBAN_REAL_THRESHOLD = 0.3`**: separates urban cores (M&I-driven)
from rural fringe / LU halo where the smoothed CDL signal is
spillover rather than real urban. Below threshold → flat 0.95;
above → `1 − uf` routing.

**Orphan-pixel refinement (pre-1985 only)**: at orphan pixels
(`cf = 0 AND uf = 0 AND has_well`), the pre-1985 routing checks
smoothed AGRI vs URBAN to decide Irr/NonIrr:

- AGRI > URBAN → `irr_frac = 0.95` (CDL-missed ag, e.g. Safford
  Valley cotton)
- URBAN ≥ AGRI > 0 → `irr_frac = 1 − URBAN` (clipped) — fringe
  residential / municipal wells get uf-aware NonIrr share

Post-1985 orphan pixels fall through to the default partition
(at `1 − uf = 1` → 100 % Irr), matching the behaviour that
post-CAP NonIrr-bleed otherwise inflates NonIrr_GW.

**1985 special case**: 1985 uses the 1981–1985 `irr_frac = 1 − uf`
override at NON-AMA pixels with natural `irr_cap` at AMA pixels —
matches USGS Irr% = 85.5.

#### 1986+ NON-URBAN-AMA partition (only_crop / both_lu / pure_desert)

For year ≥ 1986 outside `URBAN_AMA_BASINS = {Phoenix, Tucson}`, a
per-pixel LULC override replaces the default `irr_frac × pred` split
to better match modern USGS/ADWR Irr/NonIrr shares:

| Pixel class | Definition | Irr | NonIrr |
|---|---|---|---|
| `only_crop` | (cf > 0 OR `ag_halo`) AND uf ≤ 0.30 | `pred` | 0 |
| `both_lu` | cf > 0 AND uf > 0.30 | `pred × (1 − uf)` | `pred × uf` |
| `only_urban` | cf = 0 AND `~ag_halo` AND uf > 0.30 | default partition | default `pred × (1 − irr_cap)` |
| `pure_desert` | cf = 0 AND `~ag_halo` AND uf ≤ 0.30 | with well: default; without: 0 | with well: default; without: 0 |
| `ag_halo` | cf = 0 AND AGRI > 0.10 | routed to `only_crop` (full pred → Irr) | 0 |

The `only_urban_low_agri` zeroing (formerly zeroed Irr at urban
pixels with AGRI < 0.5) was removed — it had no measurable effect
because default partition `irr_frac ≈ 0` at those pixels already
gives near-zero Irr.

**`URBAN_HIGH_THRESHOLD = 0.30`**: tightened from 0.20 to capture
suburban-fringe ag (uf 0.20–0.30) as Irr rather than NonIrr.

**AGRI-halo gate (`AG_HALO_AGRI = 0.10`)**: GEE LULC frequently
misses patchy rural ag (especially in Willcox / Harquahala / Douglas).
The smoothed `AGRI` band captures the ag halo — pixels with cf = 0
but AGRI > 0.10 are reclassified as `only_crop` so their full ML
prediction goes to Irr instead of falling into the NonIrr-default
or desert branch.

**2003–2012 desert scaling**: `pure_desert_with_well` predictions
scaled by 0.75 (removes 25 % of true-desert volume). Targets the
2005/2010 USGS Total over-prediction (USGS shows CAP-era ag
retirement reduced statewide pumping; ML doesn't capture this
explicitly).

#### AGRI-extension retention (pre-1986) — REMOVED

The pre-1985 AGRI-extension retention (formerly OR'd `AGRI > 0.02–0.10`
into `_has_crop_any` at peak years to capture CDL-missed ag) has been
**removed** to eliminate hindcast-era halo artifacts in predicted
maps. The smoothed AGRI band (Gaussian σ = 3 at the GEE feature stage)
created ring artifacts around real ag clusters when used as a
retention extension, particularly visible at 1965, 1975, 1980 peak
maps.

Replacement mechanism (peak-year basin-MAX lift on well-density
features — see "ML feature well_density override" below) preserves
peak-year volumes via a different pathway that doesn't propagate
the AGRI smoothing halo into the partition footprint.

**Post-1985 AGRI halo gate retained** (in the `1986+` LU-aware
branch) — the post-1985 use never produced visual halos because the
SW-smoothing kernel σ is fixed at 4.0 (vs the 1965-1984 ramp 1.0 → 4.0
that was the actual halo amplifier). The post-1985 halo gate
prevents NonIrr inflation at ag-halo pixels in URBAN_AMA basins
(Phoenix / Tucson) where they would otherwise fall into
`pure_desert_with_well` → 100 % NonIrr via density-ratio.

#### NonIrr_SW excess routing (year ≥ 1986)

The post-1986 partition leaves volume in `NonIrr_SW` at pixels with
both M&I and ag where the density-ratio assigned too much to NonIrr.
This residual is partially routed back to Irrigation:

```
excess_sw  = nonirr_sw × (1 − uf)        # non-urban share of NonIrr_SW
nonirr_sw  = nonirr_sw × uf              # urban share stays NonIrr
```

The excess is then routed to Irr per era:

| Era | Routing |
|---|---|
| 1986–2010 | Split by local `irr_gw_share`: `irr_gw += excess × gw_share`, `irr_sw += excess × (1 − gw_share)` |
| 2011+ canal-heavy basins | `irr_sw += excess × max(cf, 0.3)` (cf-floor of 0.3 recovers fragmented ag) |
| 2011+ everywhere else | `irr_sw += excess × cf` (strict cf, drops desert residual) |

**`CANAL_HEAVY_BASINS`** (rural basins with substantial canal
infrastructure or canal-weighted streamflow):
`{Lake Havasu, Yuma, Parker, Harquahala INA, Gila Bend,
Hualapai Valley INA, Prescott AMA, Pinal AMA}`

Defined empirically from canal_density (> 0.15) or
canal_weighted_streamflow (> 100 mm) at year 2020. Canal-light
basins (Douglas, Willcox, Santa Cruz, Joseph City) keep strict
cf weighting since their nominal SW rights have no canal delivery
infrastructure.

#### Physics-constrained input data correction

Published datasets are treated as informative priors, not ground truth.
For example, the [Hung et al. (2025)](https://doi.org/10.1038/s41597-025-05920-x)
GW-fraction snapshots report values as low as 0.7 in Willcox AMA,
implying 30 % surface-water irrigation in a closed basin with no river
or canal infrastructure. This is physically impossible — Willcox is an
endorheic playa where the only "surface water" visible in LULC is mining
tailings ponds (redistributed groundwater, confirmed as GW-sourced by
HarDWR water rights records). The density-ratio approach inherently
avoids this issue: Willcox has many GW wells but zero SW rights and
zero canal-weighted streamflow, so `gw_frac` → 1.0 without requiring
any override.

#### Surface Water Capture Index

After partitioning, the pipeline computes a per-pixel, per-year capture
index quantifying how much GW pumping likely depletes surface water,
using water table depth ([Ma et al., 2026](https://doi.org/10.1038/s43247-025-03094-3))
and canal-weighted streamflow:

```
capture_fraction = exp(-wtd_m / λ) × cw_norm
sw_capture_mm    = GW_withdrawal × capture_fraction
```

Three λ values (5, 10, 20 m) produce lower/central/upper bounds.
Computed separately for Total_GW, Irrigation_GW, and
Non_Irrigation_GW. Output in all 4 units (mm, ft, m³, AF).

#### Sensitivity diagnostics

The [Hung et al. (2025)](https://doi.org/10.1038/s41597-025-05920-x)
`annual_gw_fraction` is retained as an ML feature but no longer used
for partitioning. A partition-level sensitivity diagnostic
(`run_density_ratio_sensitivity`) probes two orthogonal knobs and
writes both as sections of one CSV with a `Perturbation_Type` column:

- **Density** — well and SW-rights densities perturbed simultaneously
  with opposite signs (well × (1 ± 0.2), sw_rights × (1 ∓ 0.2)) to
  probe the GW/SW ratio's sensitivity to coordinated scaling of its
  numerator and denominator.
- **Smoothing** — the Gaussian canal-reach kernel `sw_smooth_sigma` is
  perturbed *per year* around the era-default schedule
  (`partops.era_sw_sigma(year)`) by a factor of two:

      σ_low  = max(σ_era / 2, 0.5)        # halve, with floor
      σ_high = σ_era × 2                  # double

  This anchors the sensitivity envelope to whatever σ the partition is
  actually using at each year (1.0–6.0 across eras), so the ribbon
  represents factor-of-2 perturbations around the production σ rather
  than a fixed global pair.  An earlier fixed `{2, 8}` sweep produced
  misleading asymmetry — `{2, 8}` poorly bracketed the production
  schedule (which spans 0.0 → 6.0) and the ribbon biased systematically
  above or below the baseline depending on which side of the mid-point
  the era σ sat.

`partition_predictions` accepts `sw_smooth_sigma` as an opt-in override
(default `None` → uses `era_sw_sigma(year)` from the era schedule).
Production calls (central pipeline + UQ ensemble members) leave it
unset; only the smoothing-sensitivity diagnostic above passes explicit
σ values to override the schedule.  The Smoothing rows in the output
CSV include `Sigma_Era`, `Sigma_Low`, and `Sigma_High` columns
documenting the per-year sweep range.

Both sections write to `Uncertainty/Sigma_GW/Density_Ratio_Sensitivity.csv`
with time-series ribbon plots in `Density_Ratio_Sensitivity.png` and
`Smoothing_Sigma_Sensitivity.png`.  The smoothing ribbon represents the
min/max envelope between the σ_low and σ_high partition runs — it is
**not** a confidence interval, and asymmetry around the baseline is
expected because the partition response to σ is monotonic but
nonlinear (Gaussian area scales as σ²).

#### ML feature well_density override (1938–2099)

Pre-prediction, the ML feature `well_density` is enhanced by
`apply_ml_well_density_override` (called from both pipeline.py Step 3
and uncertaintyops.py `_build_pred_features` — single source of truth):

1. **Per-pixel max blend with the 2024 registry**:
   - **Peak years (1951–1955, 1970–1980)**: at pixels with `cf > 0`
     OR `uf > 0.2` OR (orphan-with-well `cf=0 & uf=0 & year_wd>0`),
     replace with `basin-MAX(wd_2024)` (densest registered well in
     the basin).  Pre-GMA registry under-recorded ag drilling; the
     2024 registry similarly under-records pre-GMA wells abandoned
     before HarDWR began tracking.  Basin-MAX represents the
     "densest plausible peak-era well density" at pixels with any
     LU evidence.
   - **All other pre-1981 years**: at orphan-with-well pixels only,
     use `basin-p90(wd_2024)` (less aggressive — orphans are
     typically stock/rural domestic wells, not ag clusters).
   - **Post-1981**: standard per-pixel `wd_2024 × _wd_2024_scale(year)`
     — scale ramps 1.0 → 0.2 over 1995–2005 and 0.2 → 0 over
     2015–2020.  At year > 2020 the override contributes nothing.
2. **AGRI-gated phantom_wd contribution** (1938–1985 only) for ag
   pixels the 2024 registry may miss.
3. **Basin-median LU-only fill** at pixels with `cf > 0 OR uf > 0`
   but no wells, scaled by `_phantom_infill_scale(year)`.

The override does NOT modify the partition's retention mask
(`year_df['well_density']` directly).  A **mirroring partition-side
basin lift** (in `partition_predictions`) applies analogous lifts to
`well_dens` (basin-MAX), `irr_wd` and `nonirr_wd` (basin-MEDIAN at
peak LU; basin-p75 at orphans-with-well) so the partition density-
ratio split sees the same effective well-density distribution as the
ML prediction did.  Without this mirror, ML predicts large peak-year
volumes but density-ratio routes them mostly to SW (sparse year-
specific irr_wd in the denominator).

#### Basin-level canal-infrastructure gate (all eras)

Gaussian smoothing of `irr_sw_rights_density × cw_streamflow` bleeds
SW signals across basin boundaries, producing phantom Irr_SW in
basins with no canal infrastructure (Willcox / Douglas / Joseph City
/ Lower Gila).  At basins where canal coverage < 1 % of basin pixels,
**Irr_SW is collapsed into Irr_GW** (NonIrr_SW is preserved — M&I
SW can come from direct lake/river intakes that don't show up as
canal_density pixels):

```python
basin_canal_coverage = (canal_dens > 0).groupby(basin_names).mean()
no_canal_basin = basin_canal_coverage < 0.01 & ~CO_RIVER_DIRECT
irr_gw[no_canal_basin] += irr_sw[no_canal_basin]
irr_sw[no_canal_basin] = 0  # NonIrr_SW preserved at no-canal basins
```

The 1 % coverage threshold replaced the original "any canal pixel"
rule which was defeated by 2 stray pixels in Willcox (0.16 %
coverage but `basin_max > 0`).

**`CO_RIVER_DIRECT_BASINS`** whitelist bypasses this gate at basins
with physical Colorado River mainstem access not captured in
canal_density:

```python
CO_RIVER_DIRECT_BASINS = frozenset({
    'PARKER', 'YUMA', 'LAKE MOHAVE', 'LAKE HAVASU',
    'MEADVIEW', 'DETRITAL VALLEY',
})
```

These basins receive direct riverside diversions (CRIT senior
rights, Yuma Project, Parker City, Bullhead City) that don't
register as canal pixels in the GEE-derived canal_density raster.
Without the bypass, pre-CAP Parker showed 100 % GW (wrong — CRIT
delivers ~720 kAF/yr Priority-1 SW from Colorado mainstem direct).

**`MAX_GW_SHARE_CO_DIRECT = 0.4`** — additionally caps the density-
ratio `gw_share` at CO-direct basins.  Required because
`irr_sw_rights_density` is sparse (4 pixels at Parker, 0 at Lake
Havasu) — even with the gate bypass, density-ratio with sparse SW
rights still drives `gw_share → 1`.  The 0.4 cap reflects "direct-
mainstem basins are physically 60-95 % SW; cap GW at 40 %."

#### CAP delivery perturbation (2020–2026 hindcast + 2027–2099 baseline)

Under the 2007 Interim Guidelines + 2019 DCP framework, ADWR and
CAP/CAWCD have declared Colorado River shortage Tiers with specific
mandatory delivery reductions to AZ ("nearly all the reductions
within Arizona have been taken by Central Arizona Project (CAP)
water users").  Each Tier has a fixed kAF cut which, divided by
**CAP design capacity = 1500 kAF/yr** (the ~1.5 MAF CAP share of
AZ's 2.8 MAF Colorado River apportionment under the 1963
*Arizona v. California* Supreme Court decision and the 1922
Colorado River Compact; also the 2010–2021 observed delivery mean
and the implicit AWBA 2026 Plan baseline), defines the
multiplicative CAP-pixel perturbation:

| Tier | Mandatory cut | % of CAP design capacity | Factor |
|---|---|---|---|
| Tier 0 | 192 kAF | 12.8 % | **0.87** |
| Tier 1 | 512 kAF | 34.1 % | **0.66** |
| Tier 2a | 592 kAF | 39.5 % | **0.61** |
| Tier 2b | 640 kAF | 42.7 % | **0.57** |
| Tier 3 | 720 kAF | 48.0 % | **0.52** |

(ADWR's public-facing "Tier 1 = 30 % of CAP normal supply" is
rounded from 34.1 %; the 1500 kAF design capacity is the
authoritative baseline.)

ADWR / USBR Tier declaration timeline (primary source: WestWater
Research 2026,
[*Economic Impacts to Central Arizona of Reductions in CAP Deliveries*](../Data/Inputs/USGS%20WU/Economic-Impact-to-CAP.pdf),
Table 2 page 10 — full Lake-Mead-elevation-to-AZ-reduction mapping
plus the explicit "Shortage Year(s)" assignment.  Also cross-cited:
[ADWR Colorado River](https://www.azwater.gov/colorado-river),
[CAP/CAWCD Shortage Impacts](https://www.cap-az.com/water/water-supply/colorado-river-reductions/)):

| Year | Tier | Factor | Notes |
|---|---|---|---|
| 2020 | Tier 0 | 0.87 | DCP-era ~192 kAF cut |
| 2021 | Tier 0 | 0.87 | |
| 2022 | Tier 1 | 0.66 | |
| 2023 | Tier 2a | 0.61 | Mandatory 592 kAF; AZ also made a 355 kAF *voluntary* contribution to Lake Mead, not folded into this factor |
| 2024 | Tier 1 | 0.66 | "Lower Basin returns to a Tier 1 shortage" per ADWR |
| 2025 | Tier 1 | 0.66 | 512 kAF = "~30 % of CAP normal supply" per CAP/CAWCD (rounded from 34.1 %) |
| 2026 | Tier 1 | 0.66 | Confirmed by USBR August 24-month study; 512 kAF = 320 Interim Guidelines + 192 LBDCP per [AWBA 2026 Plan of Operation](../Data/Inputs/USGS%20WU/2025.12.04_AWBA-2026-Plan-of-Operation-FINAL_0.pdf). Last year of the 2007 IG + 2019 DCP framework |

These are **mandatory-Tier-cut factors only** (Option A).  Voluntary
conservation contributions are discretionary water held back in Lake
Mead, not a reduction in supply available to AZ, so they are
excluded.  That keeps the factors reproducible and externally
citable against ADWR/CAP declarations.  Option B (mandatory +
voluntary, which would give ~0.44 for 2023) is not used.

These cuts are real-world management events that the ML model
cannot anticipate from climate / LULC predictors, so they are
applied as a partition-time perturbation at CAP-served pixels
(directly reflecting the "borne by CAP users" attribution from
CAP/CAWCD).  Under Tier 1 the
[AWBA 2026 Plan of Operation](../Data/Inputs/USGS%20WU/2025.12.04_AWBA-2026-Plan-of-Operation-FINAL_0.pdf)
confirms the reduction "will not impact supplies for CAP M&I
Priority subcontractors or on-River M&I contractors" — the full
cut falls on the CAP NIA pool — which validates localizing the
perturbation to the CAP service area rather than distributing it
statewide.  AWBA-facilitated tribal firming (~10 kAF to the Gila
River Indian Community from banked water in 2026) offsets a small
fraction (~2 %) of the Phoenix-AMA irrigation impact; this is
below the model's resolution and not represented separately.

2026 is the final year under the 2007 Interim Guidelines + 2019 DCP
framework (which expires December 2026), so it's treated as part of
the hindcast at Tier 1 (0.66).  For the post-framework projection
horizon (2027–2099), the central projection assumes **continued
Tier 1 shortage conditions** — the dominant recent regime.  AZ has
been in Tier 1 (or Tier 2a) every year 2022–2026, USBR's 24-month
projections suggest Lake Mead will remain in the 1050–1075 ft
elevation band (Tier 1 trigger range) for the near term under
current Compact conditions, and the Lake Mead carryover that
protected the 2010–2021 plateau is exhausted.  Continuing factor
0.66 (= Tier 1's 512 kAF cut from the 1500 kAF design capacity, ≈
34 % reduction) past 2026 is the most defensible single central
estimate; it preserves continuity at the 2026→2027 boundary (no
boundary step) and matches the regime AZ stakeholders are actively
planning around.

**This is a separate choice from WestWater Research (2026)'s "Basic
Coordination" scenario** (which assumes a maximum policy shortage
reducing deliveries to 237 kAF — a 663 kAF cut from the 900 kAF
baseline, or 74 % reduction).  Basic Coordination is an upper-bound
*stress* scenario, not a central estimate, and is evaluated as one
alternative trajectory in the CAP scenario sweep (below) along with
Extreme Shortage and the DCP Tier 0–3 alternatives.

**Caveat on the 74-year projection horizon.** Applying a single
fixed 0.66 factor across all of 2027–2099 is a simplification —
real-world Compact renegotiations happen on 15–20 year cycles and
climate / basin hydrology / political dynamics will continue to
evolve.  The fixed factor is best read as *"the most likely
sustained shortage condition we'd plan around today,"* not a
year-specific forecast.  Reviewers interested in policy bounds
should look at the CAP scenario sweep (below) rather than treating
the central projection as a deterministic prediction.

**Caveat on high-intensity M&I demand.** The projection trajectory
reflects the dominant recent trend of CAP-driven agricultural
retirement (ag uses more water per acre than urban, so converting
irrigated land to residential or general commercial reduces
per-pixel demand).  It does *not* explicitly capture the water
demand associated with emerging high-intensity M&I activities —
semiconductor fabrication (e.g., the announced $165 B TSMC Phoenix
build-out, 15–35 kAF/yr at full operation) or hyperscale data
centers (10–100 kAF/yr across the planned Phoenix-metro cluster).
These would register as ordinary "urban" pixels in the LULC
projection chain and in the 2024 well registry's per-capita GPCD
assumptions, which undercount cleanroom / cooling-load intensity.
Rough estimate: 30–150 kAF/yr of new demand by 2030–2040 — on the
order of ~2 % of AZ total use, within the σ_LULC / σ_MACA spread
of the existing UQ ensemble.  Scope is consistent with
[WestWater Research (2026)](../Data/Inputs/USGS%20WU/Economic-Impact-to-CAP.pdf)
which similarly flags TSMC as an unquantified risk (page 6).  A
targeted per-pixel NIR uplift could be added in a future iteration
if reviewer pressure warrants.

```python
CAP_DELIVERY_FACTORS = {
    2020: 0.87, 2021: 0.87,                       # Tier 0
    2022: 0.66,                                   # Tier 1
    2023: 0.61,                                   # Tier 2a (mandatory only)
    2024: 0.66, 2025: 0.66, 2026: 0.66,           # Tier 1 (returns, continues)
    **{year: 0.66 for year in range(2027, 2100)}, # Sustained Tier 1
}
```

`apply_cap_delivery_perturbation(year_df, year, cap_pixel_mask)` scales
**both** `canal_weighted_streamflow_mm` AND the SW rights density
columns (`irr_sw_rights_density`, `nonirr_sw_rights_density`,
`sw_rights_density`) at CAP service-area pixels by the year-specific
factor.  Because the smoothed SW kernel is
`smooth_swd = gaussian_filter(swd × cw_sf)`, scaling both inputs
produces a factor² effect at perturbed pixels — yielding a substantial
GW substitution signal where scaling cw_streamflow alone would be
diluted by the wide σ Gaussian.

**Modeling note — factor² effect.** Scaling both `cw_streamflow`
and `sw_rights_density` means the smoothed SW kernel at CAP pixels
drops by a factor of *f²* rather than *f*.  For Tier 1 (f = 0.66)
the effective smoothed reduction is 0.44, i.e., ~56 % of the SW
kernel at CAP pixels removed — larger than the nominal 34 % Tier
cut would suggest.  The justification is that Tier shortage cuts
reduce *both* deliverable canal flow AND the *effective* SW rights
honored under priority ordering (junior rights are curtailed first,
so fewer rights are "honored" in a shortage year even though the
registered rights density is unchanged).  This calibration was
validated empirically against ADWR/USGS 2022–2024 aggregate anchors
and reproduces the observed ~21–38 % actual GW-substitution share.
A strictly `cw_streamflow`-only scaling would halve the effective
perturbation and under-predict GW substitution in shortage years.

**GW-weight boost at CAP pixels during shortage years.**  The
density-ratio partition is
`gw_share = (gw_w × wd) / (gw_w × wd + smooth_swd)`.  The calibrated
post-CAP `gw_weight = 0.2` schedule (from `_era_gw_weight`) keeps the
partition SW-dominant even after the Tier perturbation shrinks the
smoothed SW kernel.  This under-predicts the regulatory Assured Water
Supply shift to groundwater that AMA providers actually make during
shortage years.  `apply_cap_delivery_perturbation` therefore also
scales the `well_density` columns (`well_density`, `irr_well_density`,
`nonirr_well_density`) at CAP pixels by `CAP_CUT_GW_BOOST_FACTORS[year]`:

| Tier | Cut | Target effective gw_w (era-mapped) | Boost factor |
|---|---|---|---|
| Tier 0 | 192 kAF | 0.2 (post-CAP, no shift) | **1.0** |
| Tier 1 | 512 kAF | 1.0 (pre-CAP 1945–1980 era) | **5.0** |
| Tier 2a | 592 kAF | 1.5 (pre-CAP peak 1948–1955) | **7.5** |
| Tier 2b | 640 kAF | 1.5 | 7.5 |
| Tier 3 | 720 kAF | 2.0 (approaching pre-1945 all-GW) | 10.0 |
| **Projection 2027–2099 (sustained Tier 1)** | 512 kAF | 1.0 (pre-CAP era) | **5.0** |

Multiplying `wd` by `k` is mathematically equivalent to multiplying
`gw_weight` by `k` at those pixels — both just scale the numerator of
the density ratio.  Because ML prediction runs **before** this
perturbation, the per-pixel total pumping is unchanged; only the
GW/SW split shifts.  Era-mapped target gw_weights are inherited from
the calibrated `_era_gw_weight` schedule (no new tuning parameters),
with each Tier mapped to the historical era whose GW-dominance
regime it most resembles.

Without this boost, WestWater 2026's implied ~0.37 MAF/yr total
demand gap under Basic Coordination (cumulative ~8.0 MAF over 2027–
2060) would appear as only ~0.05 MAF/yr in our output because the
post-CAP 0.2 gw_weight dampens the SW-signal perturbation.  The
boost brings AMA-scale GW% uplift from ~2–3 pp under Tier 1 to ~15–
20 pp (matching WestWater's implied AMA-scale response).

The helper is invoked in both `pipeline.py` (Step 3) and
`uncertaintyops.py` (`_partition_with_ctx`, used by every UQ ensemble
member), so UQ uncertainty bounds reflect the perturbed central
prediction rather than an un-perturbed counterfactual.

The CAP service-area mask is rasterized once from
`Data/Inputs/GW_Data/Vector_Reproj/CAP_Service_Area.geojson` and
threaded via a module-level `_CAP_PIXEL_MASK_CTX` in
`uncertaintyops.py`.

**Why targeted perturbation over global σ reduction.**  A blanket
σ-shrink would also reduce SRP and Yuma SW, both of which are
governed by independent water systems.  The CAP-pixel-only mask
preserves the SRP/Yuma signal.  Observed GW substitution at CAP-cut
years is ~21–38 % of the cut volume (not 100 %) because SRP backfills
many overlap pixels in Phoenix AMA — physically correct given that
Salt/Verde reservoirs supplied above-average deliveries during the
same drought years.

**Why not a Lees-Ferry-driven transfer function for projection.**
We checked the empirical relationship between observed CAP delivery
(1985–2024) and Lees Ferry annual flow.  The correlation is
regime-dependent: customer build-out dominates 1985–2010 (negative
overall r ≈ −0.4), the 2010–2021 plateau is climate-decoupled,
and the 2022–2024 Tier cuts are sharply nonlinear (driven by Lake
Mead 24-month projections, not single-year Lees Ferry).  Even on
the cleanest 2015–2024 window the correlation only reaches r ≈ 0.6,
and any forward fit would assume the post-2007 / DCP framework
persists past 2026.  The sustained-factor baseline above is a more
defensible representation of "what AZ will plausibly receive" than
extrapolating a noisy historical fit.

**Bounded by the CAP scenario step.**  `run_cap_scenario_analysis`
still computes the full WestWater + DCP-Tier envelope
(`Baseline_900kAF`, `Basic_Coordination_237kAF`,
`Extreme_Shortage_0kAF`, `DCP_Tier0_192kAF_cut` through
`DCP_Tier3_720kAF_cut`) as *additive* deltas on top of whatever
`CAP_DELIVERY_FACTORS` produces for that year.

**Scenario semantics — no-cut counterfactual reference.** Every
scenario row (including `Baseline_900kAF`) is computed with
`_partition_with_ctx(..., skip_cap_perturbation=True)`, bypassing
the central `apply_cap_delivery_perturbation`.  The `Baseline_900kAF`
row therefore represents a **true no-cut counterfactual** — full CAP
delivery, no central Tier/Basic-Coordination perturbation applied —
and every non-Baseline scenario applies its own cut on top of that
un-perturbed reference:

- `Baseline_900kAF` (scenario factor 1.0) = **no cuts at all**
  (counterfactual full delivery).
- `Basic_Coordination_237kAF` (factor 0.263) = 663 kAF additive
  SW cut + 7.5× well_density boost — WestWater's post-Compact
  central assumption.
- `DCP_Tier1_512kAF_cut` (factor 0.431) = 512 kAF additive SW cut +
  5× well_density boost — same magnitudes used by the central
  hindcast for 2022/2024-2026.
- `Extreme_Shortage_0kAF` (factor 0) = 900 kAF (full CAP overlay)
  additive SW cut + 12.5× well_density boost — pessimistic bound.

Scenario deltas in `CAP_Scenario_Cumulative.csv` are therefore
**directly comparable to WestWater 2026's 8.0 MAF anchor** (also a
with-cut-vs-no-cut projection delta).  The central pipeline output
for 2027+ embodies a **sustained Tier 1 shortage assumption** (via
`CAP_DELIVERY_FACTORS[2027+] = 0.66` and `CAP_CUT_GW_BOOST_FACTORS
[2027+] = 5.0` — same magnitudes as the 2022/2024–2026 hindcast).
This represents AZ's most likely sustained shortage condition, not
WestWater's max-policy-shortage Basic Coordination scenario; the
latter is one of several alternative trajectories evaluated in the
scenario sweep, with deltas measured against the no-cut Baseline.

Each scenario applies two coordinated effects (mirroring the
hindcast perturbation in `partops.apply_cap_delivery_perturbation`):
(1) additive SW cut on `canal_weighted_streamflow_mm` and (2)
multiplicative `well_density` boost from `CAP_SCENARIO_GW_BOOSTS`,
with severity mapped to era-analogous `gw_weight` targets.

**Two-baseline convention (not a bug).**
The hindcast/central dict uses **1500 kAF** (CAP design capacity per
the [Economic Impacts to Central Arizona of Reductions in CAP
Deliveries](../Data/Inputs/USGS%20WU/Economic-Impact-to-CAP.pdf),
WestWater Research 2026, page 8: *"delivering up to approximately
1.5 million acre-feet annually"*) to derive multiplicative factors;
the scenario additive perturbation internally uses **900 kAF**
(`BASELINE_CAP_DELIVERY_AF` in `uncertaintyops.py`).  The 900 kAF
figure is WestWater's explicit post-Tier-cut operating baseline:
*"The Baseline Scenario assumes Arizona receives 900,000 acre-feet
of CAP water annually, consistent with recent deliveries and
approximately 70 % of CAP's maximum entitlement"* (page 5).  Under
this baseline the NIA pool is already at zero; further cuts come
from the M&I / Indian priority pools.

Both conventions produce the correct absolute kAF volumes in their
respective implementations — the scenarios' additive subtraction
`(1 − factor) × 900 kAF` delivers exactly the kAF labeled in the
scenario name (e.g., `DCP_Tier1_512kAF_cut` subtracts 512 kAF).
The two conventions coexist because they describe different
reference points (CAP design capacity vs WestWater post-cut
operating baseline) and the WestWater scenario names embed the
900 kAF reference; changing it would break the external citation
trail to a published study.

#### Calibration architecture

Three categories of levers are layered to match USGS/ADWR aggregate
anchors across all eras:

**1. Input predictor manipulation** (in `partition_predictions` and
`apply_ml_well_density_override`):

- Per-year well registry (year_df['well_density']) as the primary
  spatial signal
- 2024 registry per-pixel-max blend (1951–2020) for ML features
- Phantom_wd AGRI-gated fill (1938–1985) for ML features at ag pixels
- Basin-median LU-only fill (1962–2099) for ML features at LU pixels
  missing wells (extends through projection for continuity)
- Per-year SW rights density (irr_swd, nonirr_swd) as the SW signal
- Canal-weighted streamflow as the SW spreading kernel weight

**2. GW weight schedule** (`_era_gw_weight`):

| Era | gw_weight |
|---|---|
| Pre-1945 | 5.0 |
| 1930–1935 (override) | 10.0 |
| 1945–1985 | 2.0 |
| 1986–1989 (ramp) | 2.0 → 0.2 |
| 1990–1997 | 0.2 |
| 1998–2007 (mid-CAP bump) | 0.5 |
| 2008+ | 0.2 |

**3. SW smoothing σ schedule** (`partops.era_sw_sigma(year)`,
piecewise-linear ramps between anchor years):

| Era | σ |
|---|---|
| Pre-1912 | 0.0 |
| 1912–1948 | piecewise ramps through anchors (Yuma 1.5, Pinal 0.3, well-drilling boom 0.0, Gila 1.0) |
| 1948–1955 | 1.5 |
| 1956–1964 | 1.0 |
| 1965–1984 | linear ramp 1.0 → 4.0 (1973–77 = 2.0 drought override) |
| 1985–2002 | 4.0 |
| 2003–2010 | 3.0 |
| 2011–2021 | linear ramp 4.0 → 6.0 |
| 2022+ | 6.0 |

**Plus auxiliary retention/partition logic** (era-dependent):

- Pre-1948 retention rules (well & LULC intersection ramps;
  1930–1937 LU-only thresholds 0.90 → 0.50; 1938–1944 well & cf
  intersection ramps 0.95 → 0.70)
- Pre-1945 SW-gate (SW only at canal/sw_rights pixels; otherwise SW
  collapses back to GW)
- 1986+ NON-URBAN-AMA partition (only_crop / both_lu / pure_desert
  with AGRI-halo gate)
- AGRI extension for retention (loose 0.02 / std 0.10 windows)
- 1970–1985 NON-AMA `irr_frac = 1 − uf` override
- NonIrr_SW excess routing (1986–2010 gw_share split / 2011+
  canal-heavy floor)
- 2003–2012 desert-well scaling (× 0.75)
- `URBAN_AMA_BASINS = {Phoenix, Tucson}` and `CANAL_HEAVY_BASINS`
  (Lake Havasu, Yuma, Parker, Harquahala, Gila Bend, Hualapai
  Valley, Prescott, Pinal) basin sets

All three categories of levers, plus the auxiliary rules, are tuned
against USGS Total_GW pre-1950 and USGS/ADWR aggregate breakdowns
1950–2017.

#### Key helpers

- **`apply_ml_well_density_override(pred_features, year, year_df, wd_2024)`**
  — applies the pre-prediction well_density override (per-pixel max
  with 2024 registry + AGRI-gated phantom_wd + basin-median LU-only
  fill).  Single source of truth shared between pipeline.py Step 3 and
  uncertaintyops.py UQ ensemble construction.  Active 1962–2099.
- **`apply_cap_delivery_perturbation(year_df, year, cap_pixel_mask)`**
  — at CAP service-area pixels, applies two coordinated effects:
  (1) scales `canal_weighted_streamflow_mm` AND the
  `*_sw_rights_density` columns by `CAP_DELIVERY_FACTORS[year]`
  (Tier 0 = 0.87, Tier 1 = 0.66, Tier 2a = 0.61 for 2020–2026;
  **Tier 1 sustained = 0.66 for 2027–2099**; each factor =
  `1 − mandatory_cut_kAF / 1500` where 1500 kAF is CAP design
  capacity); (2) scales `well_density` / `irr_well_density` /
  `nonirr_well_density` by `CAP_CUT_GW_BOOST_FACTORS[year]`
  (Tier 1 = 5.0, Tier 2a = 7.5; **Sustained Tier 1 projection =
  5.0 for 2027–2099** — mathematically equivalent to boosting
  `gw_weight` at those pixels; era-mapped to the `_era_gw_weight`
  calibration).  The 2026→2027 boundary is smooth because both
  factors carry forward unchanged from the hindcast Tier 1 regime.
  Together these produce the factor² SW-kernel reduction and the
  corresponding GW allocation shift, calibrated against WestWater
  2026 drawdown projections.  No-op for pre-2020 years or when
  `cap_pixel_mask is None`.  Called by
  pipeline.py Step 3 and by `uncertaintyops.py:_partition_with_ctx`.
  An alias `apply_cap_hindcast_perturbation` is kept temporarily for
  backwards compatibility.
- **`_era_gw_weight(year)`** — returns the era-dependent `gw_weight`
  (5.0 pre-1945, 2.0 pre-GMA, mid-CAP bump 0.5 at 1998–2007, 0.2
  post-CAP, with 1930–1935 = 10.0 override).
- **`era_sw_sigma(year)`** — single source of truth for the SW
  Gaussian smoothing σ schedule (0.0 pre-1912, piecewise anchors
  1912–1948, 1.0–6.0 across post-1948 eras with overrides at 1973–77
  and 2003–10).  `partition_predictions` calls it whenever
  `sw_smooth_sigma is None`; the σ-sensitivity diagnostic in
  `uncertaintyops` calls it to anchor the per-year halve/double sweep.
- **`focal_fill_irr_fraction()`** — fills edge-pixel gaps (`irr_frac < 0.05`)
  with a focal mean of valid neighbors, avoiding NaN propagation along
  irrigated-area boundaries.
- **`compute_sw_fraction()`** — normalizes a density array to [0, 1] using a
  local-maximum filter (`maximum_filter(size=5)`). Used for focal-max
  normalization of canal-weighted streamflow (`cw_norm`).
- **`compute_sw_capture_index()`** — computes per-pixel SW capture fraction
  and volume at three λ values with uncertainty bounds.
- **`partition_predictions()`** — orchestrates all splits (era ramps,
  density-ratio + canal boost, zero-SW constraint, residual recovery,
  ag-era irr_cap override, urban scaling), applies well-density masking,
  and returns a dict keyed by the eight category names.

#### Calibration

**Why partition-time calibration is necessary.** No per-pixel ground
truth exists anywhere for the GW/SW × Irrigation/Non-Irrigation
breakdown of water withdrawals — agencies (USGS, ADWR, USBR) only
report aggregate statewide or basin-level totals by category. The
ML model is trained only on metered total pumping (`gw_pumping_mm`)
from ADWR records in the ten AMA/INA management areas, so it learns
*total* withdrawal magnitudes but has no signal to apportion them
into the eight reporting categories.  The partitioning logic
therefore necessarily uses physically-motivated proxies (well
density, SW rights density, canal-weighted streamflow, irrigation
capacity fraction, urban/crop fractions), and its parameters can
only be constrained against published statewide aggregates.

The XGBoost predictions of total annual pumping themselves are not
adjusted to any agency aggregate — only the post-hoc partitioning
parameters (era ramps, GW weight, SW smoothing σ, irr_cap override
blend factors, residual-recovery thresholds) are tuned to reproduce
the documented historical GW/SW and Irrigation/Non-irrigation shares.

**Delivery-residual routing as a calibration lever.** A subtle but
important calibration choice is what to do with the volume gap
between the ML-predicted total and the partitioned total
(`predictions − (irr + nonirr)`).  This residual arises whenever
the partitioning discards volume — most prominently from urban
scaling at rural pixels (`nonirr ×= urban_frac`) — and would
otherwise vanish from the budget.  The pipeline recovers this
residual at canal-served agricultural pixels (where
`cw_smooth ≥ 1` AND `crop_frac > 0`, OR pixels with both wells
AND canal influence AND crops) and routes it to **`Irrigation_SW`**
for `year ≥ 1960`.  This single design choice is what allows the
post-CAP era to reach the USGS-reported Irr_SW magnitudes (~3 MAF)
without inflating Irr_GW or NonIrr_SW.  Pre-CAP routing of the
residual was also explored (to Irr_GW, representing unregistered
well pumping); empirically, routing all residual to `Irr_SW`
produces the best fit to USGS Circulars 1004/1081/1200 across
1985–1995 while preserving the right Total_GW share.

The partitioning logic above was iteratively calibrated against
observed Arizona water-use anchor points spanning 1915–2017. Calibration
data were compiled into `Data/Inputs/USGS WU/USGS_AZ_Water_Use_1950_1980.csv`
from the following sources:

| Source | Period | Provides |
|---|---|---|
| **USGS Open-File Report 94-476** (Anderson & Duet 1994) | 1915–1990 (annual GW from Figure 1) | Statewide groundwater withdrawals (kAF) |
| **USGS Circulars 115, 398, 456, 556, 676, 765, 1001** | 1950, 1955, 1960, 1965, 1970, 1975, 1980 | Statewide totals + category breakdowns |
| **USGS Circulars 1004, 1081, 1200** | 1985, 1990, 1995 | Per-state GW/SW × Irr/NonIrr |
| **USGS Circulars 1268, 1344, 1405, 1441** | 2000, 2005, 2010, 2015 | Per-state GW/SW × all 8 categories |
| **USBR (Mock 2007)** | 1950, 1970 narratives | Statewide totals + GW/SW share + Irr % |
| **ADWR Annual Report 2016** | 1957, 1980, 1990, 2000, 2010, 2017 (Figure p.4) | Statewide totals (chart digitization) |

##### Calibration targets vs final model outputs

**Pre-1950 USGS OFR 94-476 (Total_GW only, MAF — all within ±0.03 MAF):**

| Year | Target GW | Model GW | Δ MAF |
|---|---|---|---|
| 1915 | 0.10 | 0.12 | +0.02 |
| 1920 | 0.20 | 0.19 | −0.01 |
| 1925 | 0.45 | 0.48 | +0.03 |
| 1930 | 0.75 | 0.76 | +0.01 |
| 1935 | 1.20 | 1.18 | −0.02 |
| 1940 | 1.80 | 1.80 | 0.00 |
| 1945 | 2.80 | 2.80 | 0.00 |

**Post-1950 USGS Circulars (Total MAF, GW%, Irr% — current state):**

| Year | Pred Tot | USGS Tot | dTot MAF | %dev | dGW% | dIrr% |
|---|---|---|---|---|---|---|
| 1950 | 5.18 | 5.38 | −0.20 | −3.8 | −4.5 | −2.2 |
| **1955** | **6.79** | **8.09** | **−1.30** | **−16.1** | −7.3 | −4.1 |
| 1960 | 5.55 | 5.62 | −0.07 | −1.3 | +5.0 | −1.9 |
| 1965 | 6.47 | 7.04 | −0.57 | −8.2 | +0.5 | −2.9 |
| 1970 | 7.25 | 7.60 | −0.35 | −4.6 | −0.7 | −3.2 |
| **1975** | **7.43** | **8.74** | **−1.31** | **−15.0** | −2.4 | −7.8 |
| **1980** | **7.83** | **8.93** | **−1.10** | **−12.3** | −2.6 | −3.4 |
| 1985 | 7.13 | 7.24 | −0.11 | −1.5 | −1.4 | −2.3 |
| 1990 | 7.33 | 7.59 | −0.26 | −3.5 | +0.6 | +4.5 |
| 1995 | 7.36 | 7.83 | −0.47 | −6.0 | −0.7 | +1.1 |
| 2000 | 7.63 | 7.54 | +0.09 | +1.2 | −5.1 | −1.6 |
| 2005 | 6.98 | 6.99 | −0.01 | −0.1 | −3.8 | −1.5 |
| 2010 | 6.95 | 6.82 | +0.13 | +1.9 | +2.6 | +0.3 |
| 2015 | 6.75 | 6.70 | +0.05 | +0.7 | −3.0 | −0.5 |

**ADWR Annual Report anchors (Total MAF):**

| Year | Pred | ADWR | dTot MAF | Notes |
|---|---|---|---|---|
| 1957 | 6.16 | 7.00 | −0.84 | Peak-year ML floor |
| 1970 | 7.25 | 7.50 | −0.25 | matches USGS too |
| 1980 | 7.83 | 9.50 | −1.67 | ADWR > USGS by 0.57 — peak-year ML floor + ADWR upper bound |
| 1990 | 7.33 | 7.80 | −0.47 | USGS-ADWR conflict (USGS 7.59) |
| 2000 | 7.63 | 7.10 | +0.53 | USGS-ADWR conflict (USGS 7.54) |
| 2010 | 6.95 | 7.00 | −0.05 | matches both |

**Peak-year under-prediction interpretation.**

Years 1955 / 1957 / 1975 / 1980 are systematically under-predicted
by 12–18 % (-1.0 to -1.7 MAF) due to **2024 HarDWR registry
attrition** — the modern well registry has lost records for many
peak-era wells abandoned before GMA-mandated registration began
(1980).  All four anchors fall within ±2σ of σ_total
(σ_total ≈ 1.5–1.8 MAF at peak years), and three of four within
±1σ.  Calibration is paper-defensible: peak-year offsets are
attributed to documented bias (registry attrition) and bracketed by
the σ_total ribbon.

**Mean absolute deviation across post-1950 anchors:**
- All anchors: |%dev| = 5.4 %
- Excluding peak years (1955/1975/1980): |%dev| = 2.6 %
- Modern era (1985-2015 only): |%dev| = 1.7 %

##### Calibration design principles

1. **Pre-1950 USGS reconstructions are noisy** (±0.05 MAF estimated).
   All pre-1950 anchors are within ±0.03 MAF — essentially perfect
   given USGS noise floor. Tight-fit at the relative-percent level
   (e.g., 1915 +20 % relative) is misleading because USGS values
   are tiny; absolute MAF is the correct comparison metric pre-1950.

2. **Era-specific σ + retention combinations** are necessary because
   different eras have different physical regimes:
   - Pre-1922: tiny development, strict well+LULC retention
   - 1922–1937: well-drilling boom, year-specific wells only
   - 1938–1944: GEE LULC available but ML predicts cliff-up; tight
     well&cf intersection drops high-pred desert pixels
   - 1945–1947: Gila dam construction era, transitioning to standard
   - 1948+: standard retention

3. **Piecewise-linear σ ramps between anchor years** smooth what would
   otherwise be visible σ-step cliffs in year-by-year time series.
   Anchors are calibrated to USGS Total_GW; in-between years
   interpolate.

4. **`gw_weight` schedule has bumps** (1930–1935 = 10, 1998–2007 = 0.5)
   because the smooth ramp alone cannot match local USGS anchors. The
   bumps target specific anchor periods (1930s well-boom, late-1990s
   GW% rebound).

5. **Pre-1945 SW-gate**: SW share is restricted to canal/sw_rights
   pixels. Otherwise SW collapses back to GW. Matches USGS narrative
   that pre-CAP pumping was essentially all GW outside SRP/Yuma.

6. **URBAN_AMA_BASINS = {Phoenix, Tucson}** for the 1986+ partition
   override (so rural AMAs/INAs join outside-AMA basins in the
   only_crop / both_lu split). Pinal is rural here despite Casa
   Grande / Florence urban patches because cotton/alfalfa volume
   dominates and the default partition was bleeding too much volume
   to NonIrr.

7. **CANAL_HEAVY_BASINS** (Lake Havasu, Yuma, Parker, Harquahala,
   Gila Bend, Hualapai Valley, Prescott, Pinal) get a `cf-floor = 0.3`
   on the post-2010 NonIrr_SW excess routing. Canal-light basins
   (Douglas, Willcox, Santa Cruz, Joseph City) keep strict `cf`
   weighting — their nominal SW rights have no canal infrastructure
   to actually deliver SW.

8. **AGRI-halo gate (`AG_HALO_AGRI = 0.10`)** at 1986+ reclassifies
   patchy-ag pixels (cf = 0 but smoothed AGRI > 0.10) from
   `pure_desert` to `only_crop`. GEE LULC misses substantial rural
   ag in Willcox / Harquahala / Douglas; AGRI smoothing recovers it.

9. **2003–2012 desert-well scaling** (`pure_desert_with_well × 0.75`)
   mimics CAP-era ag retirement. ML doesn't capture the 2005/2010
   pumping decline explicitly; scaling sparse-well desert pixels
   trims Total toward USGS.

10. **Year 1985 special case**: uses the 1970–1984 `irr_frac = 1 − uf`
    override AND the bumped pre-CAP `gw_weight = 2.0` together. The
    1986+ partition would give Irr% ~77 vs USGS 85.5; the legacy
    override matches USGS exactly at the CAP-startup year.

##### Acknowledged limitations

- **Pre-1938 LULC missing**: GEE LULC products start at 1938. Pre-1938
  retention uses well-only or strict well&LULC intersection (1938
  snapshot used as proxy for earlier years where applicable).
  Approximations within USGS reconstruction noise (±0.03 MAF).
- **1937→1938 ML cliff**: model mean depth jumps 129 → 228 mm/pixel
  at 1938 due to feature interactions XGBoost interprets as a regime
  change. Partition-time intersection (cf > 0.95 at 1938 ramping to
  0.7) reduces the cliff but doesn't eliminate it.
- **1944→1945 retention cliff**: 1944 has tight well&LULC intersection
  (fix 1940 anchor); 1945 needs full retention (anchor USGS 2.80).
  This produces a ~1.6 MAF Tot step in the year-by-year time series
  even though both anchor years match USGS.
- **NonIrr drift pattern**: 1955–1965 model under-predicts NonIrr by
  0.11–0.17 MAF; 2000–2015 over-predicts by 0.09–0.35 MAF. Pre-1980
  USGS NonIrr (M&I) reporting is itself uncertain.
- **USGS-ADWR conflicts at 1990 and 2000**: the two agencies disagree
  by 4–6 % at these years; model matches one or the other but not
  both. Currently matches USGS (which has more category breakdown).
- **Year-by-year time series cliffs at 1922 / 1930 / 1938 / 1945**:
  partition regime boundaries produce visible Tot/GW% steps. USGS
  5-year anchors don't see these in-between cliffs but they show in
  any plot.

### `uncertaintyops.py` — Hybrid uncertainty quantification

Computes pixel-level prediction uncertainty for all products (total
annual withdrawals, withdrawal categories, consumptive use) and writes augmented
6-band GeoTIFFs.

**Pre-GMA partitioning consistency:** UQ ensemble members use the same
era-dependent partitioning logic as the central pipeline (era-specific
retention rules pre-1948, piecewise-linear σ ramps between anchor
years, era-dependent `gw_weight` schedule, 1970–1985 `irr_frac = 1 − uf`
override for non-AMA pixels, 1986+ NON-URBAN-AMA only_crop / both_lu /
pure_desert partition, NonIrr_SW excess routing with canal-heavy basin
gating).  A module-level `_PRE_GMA_CTX` is initialised at the top of
`run_uncertainty_quantification()` from the 2024 `az_df` snapshot and
threaded through `_build_pred_features()` (well_density blend in ML
features) and `_partition_with_ctx()` (2024 well/irr_cap reference
arrays passed to `partition_predictions`).  This ensures UQ uncertainty
bounds are computed around the calibrated central prediction, not
around an under-calibrated baseline.

`_build_pred_features()` calls `partops.apply_ml_well_density_override`
(same helper used by pipeline.py Step 3) so that ML features used by
every UQ ensemble member match the central pipeline byte-for-byte.
`_partition_with_ctx()` likewise calls
`partops.apply_cap_delivery_perturbation` with a module-level
`_CAP_PIXEL_MASK_CTX` (initialized from the rasterized CAP service-area
GeoJSON), so both the 2020–2026 observed Tier cuts and the 2027–2099
sustained "Basic Coordination" projection baseline propagate into the
UQ ensemble at the same CAP-served pixels as in the central pipeline
— *including* the paired SW-signal reduction
(`CAP_DELIVERY_FACTORS`) and GW-weight boost
(`CAP_CUT_GW_BOOST_FACTORS`).  The CAP scenario sweep in
`run_cap_scenario_analysis` also operates on top of this perturbed
baseline, adding scenario-specific SW deltas without re-touching
well_density.

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
- **`run_cap_scenario_analysis()`** — Simulates CAP delivery reduction
  scenarios by perturbing `canal_weighted_streamflow_mm` at CAP service-
  area pixels and re-running the density-ratio partitioning. Total
  withdrawals stay fixed; only the GW/SW split changes. Scenarios
  include three WestWater Research (2026) Post-2026 Colorado River
  alternatives (Baseline 900 kAF, Basic Coordination 237 kAF, Extreme
  Shortage 0 kAF) and five DCP shortage tiers (Tier 0–3). Outputs
  per-basin and statewide CSVs with scenario-aware non-well offset for
  estimated statewide total water use, cumulative additional GW
  drawdown, and multi-panel time-series plots with ±1σ ribbons (drawn
  when per-category σ rasters from `Sigma_Total/Rasters/` are
  available — the block runs after sigma-total so they always are
  unless `--skip-uq sigma-total`).  Skip via `--skip-uq cap-scenario`.

  **Additive CAP-import perturbation.**  The previous multiplicative
  form `cw_sf *= factor` zeroed the *entire* canal-weighted streamflow
  signal at CAP-service-area pixels — including the SRP and Salt/Verde
  watershed contributions in Phoenix AMA where CAP overlaps with SRP
  service area.  At factor=0 (Extreme_Shortage_0kAF) this collapsed
  the smoothed kernel `gaussian_filter(swd × cw_sf)` to zero across
  central AZ, driving ~3× over-substitution of GW for SW (~2.7 MAF/yr
  extra GW for a nominal 900 kAF/yr CAP cut).  The current form
  subtracts only the AF-calibrated CAP-import slice:

  ```
  cw_sf_scenario[i] = cw_sf_baseline[i] − (1 − factor) × cap_overlay_per_pixel[i]
  cap_overlay_per_pixel[i] = (BASELINE_CAP_DELIVERY_AF × m³/AF × 1000 / pixel_area_m2)
                             × (canal_density[i] / Σ canal_density at CAP pixels)
  ```

  with `BASELINE_CAP_DELIVERY_AF = 900_000` matching the
  `Baseline_900kAF` scenario name.  At factor=0 the helper subtracts
  exactly 900 kAF distributed canal-density-weighted across the CAP
  service area; SRP / Salt / Verde watershed flows survive intact.
  Scenario factors stay monotonic (`remaining_kAF / 900`), so DCP
  tiers strictly order by cut magnitude.

  **Non-well offset.**  Statewide totals add a fixed
  `NON_WELL_OFFSET_FIXED_AF = 350_000` AF (reclaimed water only) on top of
  the model's per-pixel well-mediated partition.  CAP and SRP deliveries
  are already captured by the model's SW components via the wide-σ
  Gaussian-smoothed `sw_rights × canal_weighted_streamflow` signal —
  verified by comparing 4-AMA model `Total_SW` (~1300 kAF/yr) against
  CAP-direct + SRP irrigation deliveries (~1300 kAF/yr) for 1990–2022,
  with model/reference ratio averaging ~1.0.  Yuma Colorado River
  diversions (~440 kAF/yr) are likewise captured in the model's Yuma
  basin output.  The previous `NON_WELL_OFFSET_CAP_AF` (CAP-direct
  augmentation) is therefore set to 0 to avoid double-counting in
  scenario reductions.
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

**Verification step skipped by design.** A pixel-vs-well-sum
reconciliation step (`verify_well_package`) is intentionally not
called by the pipeline.  The basin-median LU-only fill in
`apply_ml_well_density_override` produces predictions at crop / urban
pixels that have no well points within the 2 km cell, so a strict
"sum-of-wells == pixel-total" check would always fail by design.  The
distribution logic itself (capacity-weighted re-normalization within
each pixel using only active wells) has been validated separately and
is correct; the verification check is therefore omitted rather than
surfacing expected failures.

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

**USGS statewide calibration overview** (`run_usgs_az_calibration_overview()`):
- AZ-wide annual `Total_GW` and `Total_SW` bar plots with ±1σ caps
  and USGS Circular / OFR 94-476 anchors overlaid as red triangles.
- Mirrors USGS OFR 94-476 (Anning & Duet 1994) Figure 1 in bar form.
- Reads model statewide values from `Annual_Summaries/Total_{GW,SW}.csv`,
  σ from `Uncertainty/Sigma_Total/Rasters/`, anchors from
  `Data/Inputs/USGS WU/USGS_AZ_Water_Use_1950_1980.csv`.
- Default range 1915–2017; SW panel starts at 1950.
- Outputs:
  `USGS_Calibration_Bars/USGS_AZ_Calibration_Bars.csv`,
  `USGS_AZ_Total_GW_Bars.png`, `USGS_AZ_Total_SW_Bars.png`.

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
- **Temporally masked canal density** — per-year GRAIN canal density
  where each segment is included only after its `first_delivery_year`,
  assigned via named-canal construction dates, HarDWR SW POD nearest-
  neighbor dating (20 km), or basin-level earliest SW right fallback.
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
- **Temporal canal masking** — canal density and canal-weighted streamflow
  are zero at pixels before the canal's `first_delivery_year` (assigned per
  GRAIN segment via construction dates + HarDWR POD nearest-neighbor +
  basin-level delivery-start enforcement from observed CAP/SRP records),
  ensuring the GW/SW split reflects actual infrastructure availability at
  the pixel level.  The regular streamflow rasters also gate the CAP
  Colorado River overlay to 1985 onward.
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
        ├── Annual_Summaries/                # Cached per-year stats (for fast re-runs) †
        │   ├── Total_Predicted.csv          #   AZ-wide total predicted stats
        │   ├── {Category}.csv               #   Per-category stats
        │   ├── {CU_Category}.csv            #   Consumptive use stats
        │   ├── Actual.csv                   #   Metered actual stats (1984–2024)
        │   ├── Basin_Total.csv              #   Per-basin stats
        │   └── Subbasin_Total.csv           #   Per-sub-basin stats
        │   # † Mean_Depth_mm / Mean_Depth_ft are averaged over
        │   #   "active pumping pixels" (pred ≥ 5 mm/yr) to reflect
        │   #   per-pixel irrigation intensity; Volume_m3 / Volume_AF
        │   #   sum over ALL valid pixels so volume is conserved.
        │   #   See `_pixel_stats` in pipeline.py for the convention.
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

**Water budget reconciliation (2017):** The model predicts 4.91 MAF vs.
ADWR's 7.0 MAF total, a gap of ~2.1 MAF.  The model's Total_SW
(~1.50 MAF in 2017) partially captures CAP and SRP deliveries through
the density-ratio GW/SW partitioning — the temporally-masked
canal-weighted streamflow and HarDWR SW rights density proxies route a
portion of well-mediated withdrawals to SW at canal-served pixels.
The remaining gap is accounted for by water sources that bypass the
ADWR Well Registry entirely:

| Source | MAF | Notes |
|--------|-----|-------|
| Yuma-area federal diversions | ~0.79 | Bureau of Reclamation Yuma and Gila Project irrigation districts (WMIDD: 278,000 AF, YCWUA: 254,200 AF, Gila Project Yuma Mesa Division: 250,000 AF, Unit B: 6,800 AF); gravity canal diversions from the Colorado River via the All-American Canal and Gila Gravity Main Canal ([Noble et al., 2015](https://www.azwater.gov/sites/default/files/2022-11/Final%20Yuma%20Report%20021715.pdf); cf. [Frisvold et al., 2018](https://doi.org/10.3390/su10051548) §6.2 for the WMIDD-specific delivery time series and the historical conservation gains) |
| Reclaimed/effluent water | ~0.35 | ~5 % of total state water supply ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)) |
| Other unaccounted sources | ~0.95 | Includes tribal water deliveries, direct river diversions (e.g., Parker, Safford, Verde River), and minor surface-water sources not represented in the ADWR Well Registry. The model does capture some local canal-based SW in these basins (~0.36 MAF in Yuma, Safford, Parker, Lower Gila, Gila Bend, Verde River, Hualapai Valley, and others via the density-ratio partitioning), but federal and tribal diversions that bypass wells are not included. |
| **Total gap** | **~2.1** | |

The model's Total_SW in CAP/SRP basins (~1.14 MAF in 2017) closely
matches observed CAP+SRP direct deliveries (~1.11 MAF), indicating
that the density-ratio partitioning with temporally-masked canal
features captures the full CAP/SRP delivery magnitude without
calibration.  This is an entirely out-of-sample agreement — see
Step 4c for the full validation.

The model thus captures ~70 % of Arizona's total water use — specifically,
the portion that flows through registered well and diversion infrastructure,
including a portion of CAP/SRP deliveries that the density-ratio
partitioning routes to SW at canal-served pixels.
The remaining ~30 % is delivered by large-scale federal water projects
(Yuma-area Colorado River diversions), the portion of CAP/SRP deliveries
not captured by the density-ratio proxy, and reclaimed water systems
that operate outside the ADWR Well Registry.
The Yuma-area irrigation districts hold some of the oldest and most senior
water rights on the Lower Colorado River (dating to the Reclamation Act of
1902) and collectively divert ~0.79 MAF/yr through federal canal infrastructure
with application efficiencies of 80–90 % — substantially higher than the
statewide NHM average of ~60 % used in this pipeline
([Noble et al., 2015](https://www.azwater.gov/sites/default/files/2022-11/Final%20Yuma%20Report%20021715.pdf)).
Combining the model's 4.91 MAF with the estimated non-well sources
(~2.1 MAF) yields ~7.01 MAF, closing to within 0.01 MAF of ADWR's
reported 7.0 MAF total.  USGS independently estimates 3.09 MAF of
GW withdrawals for 2015 ([Dieter et al., 2018](https://doi.org/10.3133/cir1441); Arizona summary in [NGWA, 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf));
the model produces 3.32 MAF for the same year (within 0.23 MAF)
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
| 1900 | 0.12 | 0.05 | 0.08 | 0.11 | 0.01 | 37 % | 89 % |
| 1910 | 0.16 | 0.06 | 0.10 | 0.13 | 0.03 | 38 % | 84 % |
| 1920 | 0.28 | 0.14 | 0.15 | 0.22 | 0.06 | 48 % | 78 % |
| 1930 | 0.50 | 0.27 | 0.23 | 0.37 | 0.14 | 54 % | 73 % |
| 1940 | 2.69 | 2.37 | 0.32 | 0.71 | 1.98 | 88 % | 27 % |
| 1950 | 3.23 | 2.73 | 0.50 | 0.85 | 2.39 | 84 % | 26 % |
| 1960 | 4.01 | 3.36 | 0.66 | 1.30 | 2.71 | 84 % | 32 % |
| 1970 | 4.71 | 3.88 | 0.83 | 1.97 | 2.74 | 82 % | 42 % |
| 1980 | 5.41 | 4.34 | 1.08 | 2.29 | 3.12 | 80 % | 42 % |
| 1985 | 5.60 | 4.28 | 1.32 | 2.55 | 3.05 | 77 % | 46 % |
| 1990 | 5.73 | 4.33 | 1.41 | 2.75 | 2.98 | 75 % | 48 % |
| 2000 | 5.96 | 4.26 | 1.69 | 3.03 | 2.93 | 72 % | 51 % |
| 2010 | 5.92 | 4.08 | 1.84 | 2.93 | 2.99 | 69 % | 50 % |
| 2015 | 6.49 | 4.51 | 1.98 | 3.35 | 3.15 | 69 % | 52 % |
| 2017 | 6.62 | 4.56 | 2.06 | 3.46 | 3.16 | 69 % | 52 % |
| 2019 | 6.30 | 4.30 | 2.01 | 3.32 | 2.98 | 68 % | 53 % |
| 2020 | 6.62 | 4.45 | 2.16 | 3.56 | 3.06 | 67 % | 54 % |
| 2024 | 6.73 | 4.46 | 2.27 | 3.72 | 3.01 | 66 % | 55 % |
| 2030 | 6.65 | 4.30 | 2.35 | 3.53 | 3.12 | 65 % | 53 % |
| 2040 | 6.83 | 4.33 | 2.50 | 3.66 | 3.17 | 63 % | 54 % |
| 2050 | 7.12 | 4.42 | 2.70 | 3.88 | 3.25 | 62 % | 54 % |
| 2060 | 7.36 | 4.48 | 2.87 | 4.05 | 3.31 | 61 % | 55 % |
| 2070 | 7.66 | 4.56 | 3.10 | 4.28 | 3.37 | 60 % | 56 % |
| 2080 | 7.86 | 4.58 | 3.28 | 4.46 | 3.39 | 58 % | 57 % |
| 2090 | 8.11 | 4.61 | 3.50 | 4.68 | 3.43 | 57 % | 58 % |
| 2099 | 8.35 | 4.66 | 3.69 | 4.89 | 3.46 | 56 % | 59 % |

Consumptive use (CU = IE × Irrigation Withdrawal) volumes, where IE is the
USGS NHM basin-level irrigation efficiency (million acre-feet):

| Year | Irrigation | Irrigation CU | Irrigation GW CU | Irrigation SW CU | IE |
|------|------------|---------------|-------------------|------------------|----|
| 1900 | 0.05 | 0.03 | 0.03 | 0.00 | 61 % |
| 1910 | 0.06 | 0.04 | 0.03 | 0.00 | 61 % |
| 1920 | 0.14 | 0.08 | 0.06 | 0.02 | 61 % |
| 1930 | 0.27 | 0.17 | 0.11 | 0.06 | 61 % |
| 1940 | 2.37 | 1.43 | 0.30 | 1.13 | 60 % |
| 1950 | 2.73 | 1.65 | 0.31 | 1.34 | 60 % |
| 1960 | 3.36 | 2.02 | 0.52 | 1.50 | 60 % |
| 1970 | 3.88 | 2.33 | 0.82 | 1.51 | 60 % |
| 1980 | 4.34 | 2.60 | 0.90 | 1.70 | 60 % |
| 1985 | 4.28 | 2.57 | 0.92 | 1.64 | 60 % |
| 1990 | 4.33 | 2.59 | 1.03 | 1.56 | 60 % |
| 2000 | 4.26 | 2.54 | 1.03 | 1.50 | 59 % |
| 2010 | 4.08 | 2.43 | 0.91 | 1.52 | 60 % |
| 2015 | 4.51 | 2.70 | 1.11 | 1.58 | 60 % |
| 2017 | 4.56 | 2.74 | 1.14 | 1.60 | 60 % |
| 2019 | 4.30 | 2.57 | 1.08 | 1.49 | 60 % |
| 2020 | 4.45 | 2.65 | 1.13 | 1.52 | 60 % |
| 2024 | 4.46 | 2.67 | 1.20 | 1.47 | 60 % |
| 2030 | 4.30 | 2.57 | 1.06 | 1.52 | 60 % |
| 2040 | 4.33 | 2.59 | 1.07 | 1.52 | 60 % |
| 2050 | 4.42 | 2.64 | 1.10 | 1.54 | 60 % |
| 2060 | 4.48 | 2.68 | 1.13 | 1.56 | 60 % |
| 2070 | 4.56 | 2.73 | 1.16 | 1.57 | 60 % |
| 2080 | 4.58 | 2.75 | 1.17 | 1.57 | 60 % |
| 2090 | 4.61 | 2.76 | 1.18 | 1.58 | 60 % |
| 2099 | 4.66 | 2.79 | 1.21 | 1.58 | 60 % |

The statewide mean IE is ~60 %, meaning roughly 40 % of applied irrigation
water returns to aquifers as deep percolation or runs off as return flow.
IE varies by basin (NHM HUC12-level values) but the statewide aggregate is
stable across years because the same NHM efficiency map is applied to
changing withdrawal volumes.

Key trends:
- **Irrigation share** (irrigation withdrawal as a fraction of total
  well-mediated withdrawal) declines from ~80 % (1960s–1980s) to
  ~73 % (2019) and continues to ~63 % by 2099 as urbanization
  increases M&I demand. The 2019 model value of **73.2 %** matches
  ADWR's reported ~72 % share of agriculture in total Arizona water
  use ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector))
  to within ~1.2 percentage points. The irrigation category includes
  rural dual-purpose wells coded ``IRRIGATION, DOMESTIC`` or
  ``IRRIGATION, STOCK`` in the ADWR Well Registry; recreation / turf
  wells (e.g. golf courses) are coded ``RECREATION`` and routed to
  the non-irrigation category.
- **Irrigation GW share** (`Irrigation_GW / Irrigation_total`) is a
  separate metric that asks "of the irrigation water alone, what
  fraction comes from wells." The model produces ~69 % for 2017,
  ~69 % for 2019, and ~71 % for 2024, all in the metered window. This
  is much higher than the *statewide* GW share (46–47 % in those same
  years, the figures compared to USGS and ADWR above) because the
  statewide number is diluted by the ~2.26 MAF/yr of federal Colorado
  River canal deliveries (CAP, SRP, Yuma-area diversions) that bypass
  wells entirely and are reconciled separately as the federal-delivery
  offset.
- **GW share** starts at 89 % in 1900 (nearly all GW, few canals
  active), declines to ~59 % by the late 1940s as canal infrastructure
  (SRP) brought surface water to irrigated areas, then rises gradually
  to ~71 % by 2099 as non-irrigation demand — predominantly
  groundwater-sourced outside canal service areas — grows faster than
  irrigation.  Including unaccounted federal SW deliveries (~2.26 MAF),
  the statewide GW share is ~48 % in 2017, consistent with
  independently reported GW/SW shares: USGS estimates 46 % GW and
  3.09 MAF total GW for 2015 ([NGWA, 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf);
  based on [Dieter et al., 2018](https://doi.org/10.3133/cir1441)), while
  ADWR reports 41 % GW for 2019 ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)).
  The model produces 47.0 % / 3.32 MAF for 2015 (within 0.23 MAF and
  1.0 percentage point of USGS) and 46.5 % for 2019 (within 5.5
  percentage points of ADWR) — converging on both agency estimates
  across different years without any calibration to these targets.

  | Year | Source | GW (MAF) | Statewide GW % |
  |------|--------|----------|----------------|
  | 2015 | USGS/NGWA | 3.09 | 46 % |
  | 2015 | AZ-Hydro | 3.32 | 47.0 % |
  | 2019 | ADWR | — | 41 % |
  | 2019 | AZ-Hydro | 3.20 | 46.5 % |
- **Total withdrawals** grow from 0.11 MAF (1900) to 4.93 MAF (2024)
  and are projected to reach 6.03 MAF by 2099.  Early growth (1900–1950)
  reflects the build-out of well and canal infrastructure; mid-century
  growth (1950s–1980s) is driven by agricultural expansion; projected
  growth is driven by urbanization and increasing M&I demand.
- **Irrigation** remains relatively stable in the projections (3.54 →
  3.72 MAF, +5 % by 2099), while **non-irrigation** grows by ~65 %
  (1.40 → 2.30 MAF), more than doubling from the 1990s level
  (0.96 MAF), reflecting continued urban and industrial growth
  including data center and energy-sector water demand.
- **Pre-CAP era** (before 1985): GW share declines from 89 % (1900) as
  SRP canal infrastructure expanded, reaching a minimum ~59 % by the
  late 1940s. Post-1985 GW share rises as non-irrigation
  (predominantly GW) demand grows relative to canal-served irrigation.
- **Conservation**: Irrigation + Non-Irrigation = Total and GW + SW = Total
  hold exactly for all years.
- **Consumptive use** (CU = IE × Irrigation Withdrawal, with IE the
  USGS NHM HUC12 irrigation efficiency map): irrigation CU rises from
  ~0.03 MAF in 1900 to a 2.26 MAF peak in 2098 — a ~75× increase that
  tracks the parent irrigation withdrawal trajectory closely because
  the statewide IE is stable at ~60 % across all years (60.2 % in 2010,
  60.5 % in 2017, 60.5 % in 2024). The roughly 40 % of applied
  irrigation water that is *not* consumed returns to the aquifer as
  deep percolation or runs off as return flow, which is a substantial
  recharge term that this study quantifies but does not separately
  route through the capture index. The GW share of irrigation CU
  follows the parent GW share of irrigation withdrawal — rising from
  ~56 % in the 1950s (when SRP canal deliveries were near peak) to
  ~69 % by 2017 — so the consumptive component of pumping-induced
  groundwater depletion grows in proportion. In the projection
  (2026–2099) irrigation CU is flat to slightly rising (2.14 → 2.25
  MAF, +5%) because the parent irrigation withdrawal — which is
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
  capture fraction is ~1.1 % during the 1984–2024 historical era
  (central estimate, λ=10 m), translating to 0.02–0.04 MAF of GW
  pumping that physically captures surface water via stream depletion
  in any given year.  Capture grows from <0.001 MAF (1900) through the
  20th century as pumping infrastructure expanded near canal corridors,
  averages ~0.037 MAF over the 2015–2024 historical decade, peaks at
  ~0.040 MAF in 2025, then **declines substantially in the projection
  era — to ~0.022 MAF by 2050 and ~0.023 MAF by 2099, a ~44 % drop**.
  This decline is *not* driven by falling Total_GW (which actually
  rises from 3.33 MAF in 2024 to 3.99 MAF in 2099); it reflects a
  spatial reweighting of pumping toward non-irrigation pixels outside
  the canal-corridor footprint, so the volume-weighted statewide
  capture fraction itself falls from ~1.2 % (2025) to ~0.6 % (2099).
  Irrigation GW accounts for ~90 % of total capture in the historical
  era (~0.033 MAF), with non-irrigation contributing ~0.003 MAF.
  While the statewide impact is small relative to total GW, the
  capture is spatially concentrated in shallow-water-table areas near
  rivers and canal infrastructure.  Basin-level GW volume, SW volume, capture
  volume, volume-weighted capture fractions, and WTD (2017, central λ=10 m):

  | Basin | GW Pumping Volume | SW Withdrawal Volume | SW Capture Volume | Capture Fraction | Mean WTD (m) | Context |
  |-------|-------------------|----------------------|-------------------|-----------------|-------------|---------|
  | Parker        |  42.3 kAF (52.1 Mm³)  |  17.2 kAF (21.3 Mm³)  |  4,099 AF (5.06 Mm³)  | 9.7 % | 51 | Colorado River alluvial aquifer |
  | Lower Gila    | 342.7 kAF (422.7 Mm³) |  26.0 kAF (32.1 Mm³)  | 23,782 AF (29.34 Mm³) | 6.9 % | 55 | Gila River corridor |
  | Yuma          |  93.2 kAF (115.0 Mm³) | 148.1 kAF (182.7 Mm³) |  3,461 AF (4.27 Mm³)  | 3.7 % | 42 | Colorado River diversions, alluvial wells |
  | Lake Havasu   |   9.0 kAF (11.2 Mm³)  |   0.0 kAF (0 Mm³)     |    297 AF (0.37 Mm³)  | 3.3 % | 73 | Colorado River adjacent |
  | Safford       | 119.6 kAF (147.5 Mm³) |  63.3 kAF (78.1 Mm³)  |  2,545 AF (3.14 Mm³)  | 2.1 % | 34 | Upper Gila River valley, shallowest WTD |
  | Phoenix AMA   | 413.3 kAF (509.8 Mm³) | 697.1 kAF (859.8 Mm³) |  1,589 AF (1.96 Mm³)  | 0.4 % | 64 | Deep wells dominate; SRP corridor localized |
  | Willcox AMA   | 319.2 kAF (393.7 Mm³) |   0.0 kAF (0 Mm³)     |      0 AF (0 Mm³)     | 0.0 % | 44 | Negative control: no canals despite moderate WTD |
  | Butler Valley |  10.4 kAF (12.9 Mm³)  |   0.0 kAF (0 Mm³)     |      0 AF (0 Mm³)     | 0.0 % | 72 | Negative control: no canal infrastructure |

  Note: SW Withdrawal Volume above is the model's predicted Total_SW from the density-ratio partition (HarDWR surface-water rights density modulated by canal-weighted streamflow). It does *not* include the ~2.26 MAF/yr of direct Bureau of Reclamation Yuma- and Gila-Project canal deliveries (WMIDD, YCWUA, Gila Project Yuma Mesa Division, Unit B) that bypass the ADWR Well Registry entirely and are reconciled separately as the federal-delivery offset (see [Federal-delivery reconciliation](#federal-delivery-reconciliation-yuma-and-the-22526-maf-water-budget-gap)). Yuma's model SW total of ~148 kAF is therefore only the well-routed SW share — the basin's full surface-water economy is roughly 6× larger once the federal offset is added back.

  **What the nonzero Yuma capture fraction tells us.** Yuma's 3.7 % capture fraction (3,461 AF of well-mediated stream depletion on 93.2 kAF of GW pumping) is small in absolute terms but physically diagnostic: it is the framework's strongest single signal that **the small slice of Yuma's pumping that is in fact groundwater (the ADWR-Well-Registry-routed share) is itself drawing from the Colorado River alluvium, not from a separate ambient stored aquifer**. Yuma has the second-shallowest mean WTD in the table (42 m), which puts the connectivity term `exp(−42/10) ≈ 0.015` right in the regime where wells are physically capable of pulling from river-recharged alluvium; combined with the canal-corridor footprint (`cw_norm > 0` across most of the basin) the partition correctly infers that those wells are screened in alluvium hydraulically connected to the river. Read together with the SW Withdrawal column, the picture is consistent: ~148 kAF of model-routed SW + ~93 kAF of GW + the externally reconciled federal-delivery offset (~0.79 MAF for Yuma alone) all draw on the same Colorado River alluvial system, and the GW slice is *not* a separate ambient aquifer being mined alongside the SW economy. The corollary on the infrastructure side is that subsidence and well-field-decline risk for these specific wells should be correspondingly *low*, because the storage being depleted is being continuously recharged from the river — the "loss" is showing up downstream in the Colorado rather than in compaction of the Yuma alluvium, which is exactly the accounting boundary the Wellton-Mohawk drain system already enforces under the 1944 U.S.–Mexico Treaty ([Frisvold et al., 2018](https://doi.org/10.3390/su10051548) §6.7). The stewardship question for the Yuma GW-righted wells is therefore better framed as a Colorado River allocation question (every acre-foot pulled is an acre-foot the river does not carry past Imperial Dam) than as a basin-storage question.

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

Six limitations are baked into the framework's structure rather than
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
   capture), and new tribal water rights settlements that create
   entirely new delivery infrastructure. **Two concrete examples:**
   (1) The Hualapai Tribe Water Rights Settlement Act of 2022
   authorizes 4,000 AF/yr of CAP deliveries to Hualapai Valley INA
   with a $312 million infrastructure trust fund — once the pipeline
   is built, this basin will receive CAP water through infrastructure
   that does not yet exist in the GRAIN canal dataset, so the model's
   projection for Hualapai Valley does not include this step change.
   (2) Post-2026 Colorado River operations could drastically reduce
   CAP deliveries: under a strict priority system, the Basic
   Coordination scenario reduces CAP supply from 900 kAF to 237 kAF,
   while the Extreme Shortage scenario eliminates CAP deliveries
   entirely
   ([WestWater Research, 2026](Data/Inputs/USGS%20WU/Economic-Impact-to-CAP.pdf)).
   Such reductions would force a structural shift from surface water
   back to groundwater across the CAP service area — exactly the kind
   of GW/SW rebalancing the model cannot represent because the
   density-ratio partitioning is learned from the 1984–2024 era when
   CAP was fully operational.
   The 2099 projection should be read as **"what would happen if the
   1984–2024 model relationships continued forward under the provided
   climate and land-use scenarios"**, not as a forecast of actual 2099
   water use under all plausible futures.

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
   as **"the paradox of irrigation efficiency"** and corroborated for
   the U.S. Southwest by [Frisvold et al. (2018)](https://doi.org/10.3390/su10051548):
   when farmers adopt higher-efficiency technologies (drip, sub-surface,
   precision sprinklers), the typical empirical response is *not* to
   use less water on the same crops but to switch to higher-water-demand
   crops, intensify application depth, extend the growing season, or
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
   cropped area stays flat. **A concrete documented example of this
   gap:** Yuma agriculture transitioned from perennial / summer-centric
   cropping (cotton, alfalfa, citrus) to winter-centric multi-crop
   vegetable systems between 1970 and 2010, reducing district water
   deliveries by ~120 kAF/yr while irrigable acreage stayed nearly
   flat ([Frisvold et al., 2018](https://doi.org/10.3390/su10051548) §6.2).
   This is precisely the channel (i) crop-mix shift at constant area
   that our framework cannot reproduce — the conservation came from
   replacing high-ET summer crops with low-ET winter vegetables, not
   from acreage retirement or from an IE upgrade. Any future Yuma
   trajectory that follows the same crop-mix evolution would be
   under-represented in our CU projection for the same structural
   reason. Yuma is also the canonical southwest counterexample to the
   general Grafton paradox: there, conservation gains from improved
   on-farm efficiency *did* materialize at the basin scale because
   they were coupled to a crop-mix shift toward shallow-rooted,
   short-season vegetables rather than to acreage expansion or
   higher-water-demand crops. The flat-to-slightly-rising projected CU
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

   The ADWR Well Registry `WELL_DEPTH` column was evaluated as a
   potential alternative or supplement to `wtd_m` in the connectivity
   term. It is reasonably complete (74.8 % overall, 97.2 % post-1980,
   67–76 % in the eight SW-capture river-corridor basins), but the
   mean well depth exceeds mean `wtd_m` by a factor of 2.1–9.6× in
   every river-corridor basin — wells are drilled *into* the
   saturated zone rather than *to the top of* it — so substituting
   `WELL_DEPTH` into `exp(−depth/λ)` would drive the connectivity
   term to ≈ 0 everywhere and mis-represent the pressure-propagation
   physics that the exponential-decay form is modelling. And
   `WELL_DEPTH` is the total drilled depth, not the perforation/screen
   interval, so it does not even provide the information needed for a
   defensible hybrid formula. A future methodological extension could
   parse per-well screen-interval metadata (`PERFORATION_TOP`,
   `PERFORATION_BOTTOM`, or equivalent) if a richer well-construction
   dataset becomes available; for the present release, `wtd_m` is the
   correct term and `WELL_DEPTH` is not used in the formula. The full
   per-basin completeness and mean-depth-vs-`wtd_m` table is provided
   in the supplementary (§S5.2 of the Earth's Future companion paper
   supplementary file).

6. **Era-specific σ attribution classification.** The basin-scale σ
   attribution diagnostic suite (Step 3g,
   `Raster_Maps/Sigma_Attribution/`) uses **two different** binary
   classification metrics in different eras because the underlying σ
   coverage dictates what is meaningful. In the Projection era all
   five σ components are non-zero, so the binary map classifies
   basins along `Mgmt / (Mgmt + Clim)` — the
   management-vs-climate trade-off that maps onto the stewardship
   decision space (better data vs better GCM scenarios). In the
   Hindcast and Historical eras, σ_MACA is structurally zero by
   design (the climate ensemble spans Projection-era GCM scenarios
   only, 2026–2099), and σ_LULC is likewise zero, so the binary map
   instead classifies basins along `Mgmt / (Mgmt + Model)` —
   management-vs-model-floor. The two metrics are therefore **not**
   directly comparable across eras, and users cannot read a single
   basin-level number like "management share" and apply it uniformly
   across the 1896–2099 window. The ternary RGB-mixed map family is
   the cross-era harmonized view: it shows all three shares (Mgmt,
   Clim, Model) on one continuous color axis and works identically
   in every era without swapping the classification metric. σ_Model
   is disclosed as a third axis rather than forced into the binary
   classification because empirically it is the single largest
   variance contributor in the majority of Arizona basins in the
   Projection era; a three-way argmax map would otherwise read as
   largely monochromatic "model-dominated" and hide the management-
   vs-climate trade-off that is the actual decision-relevant signal.
   Sub-basin attribution is deferred (ADWR stewardship decisions are
   made at the basin scale, and a per-sub-basin choropleth at ~150
   polygons without in-polygon labels is redundant with the basin
   view).

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


## Key Outputs

### Withdrawal Predictions (Era Mean Volume)

| Total Predicted Withdrawal | Total GW Withdrawal | Total SW Withdrawal |
|:---:|:---:|:---:|
| ![Total Predicted](../docs/images/Era_Maps_Total_Predicted_Annual_Withdrawal_Volume.png) | ![Total GW](../docs/images/Era_Maps_Total_GW_Volume.png) | ![Total SW](../docs/images/Era_Maps_Total_SW_Volume.png) |

### Consumptive Use and Actual vs Predicted

| Irrigation CU Volume | Actual vs Predicted (Depth) | Actual vs Predicted (Volume) |
|:---:|:---:|:---:|
| ![Irrigation CU](../docs/images/Era_Maps_Irrigation_CU_Volume.png) | ![AvP Depth](../docs/images/Actual_vs_Predicted.png) | ![AvP Volume](../docs/images/Actual_vs_Predicted_Volume.png) |

### Uncertainty and Model Diagnostics

| Prediction CV | Prediction SNR | OOD Probability |
|:---:|:---:|:---:|
| ![CV](../docs/images/Era_Maps_Prediction_CV.png) | ![SNR](../docs/images/Era_Maps_Prediction_SNR.png) | ![OOD](../docs/images/Era_Maps_Out-of-Distribution_Probability.png) |

### SW Capture Index

| Total SW Capture (Volume) | Total SW Capture Fraction (λ=10m) |
|:---:|:---:|
| ![SW Capture Volume](../docs/images/Era_Maps_Total_SW_Capture_Volume.png) | ![SW Capture Fraction](../docs/images/Era_Maps_Total_SW_Capture_Fraction_(λ=10m).png) |

### σ Attribution Diagnostic Suite (Projection Era)

| Binary Attribution (Climate ↔ Management) | Ternary Attribution (RGB: Mgmt/Model/Clim) |
|:---:|:---:|
| ![Binary](../docs/images/Era_Maps_Sigma_Attribution_Projection.png) | ![Ternary](../docs/images/Era_Maps_Sigma_Attribution_Ternary_Projection.png) |

| Variance Decomposition — Total GW | Variance Decomposition — Total SW |
|:---:|:---:|
| ![Stacked GW](../docs/images/Sigma_Attribution_Stacked_Bar_Total_GW_Projection.png) | ![Stacked SW](../docs/images/Sigma_Attribution_Stacked_Bar_Total_SW_Projection.png) |

| Attribution Timeseries — Total GW | Attribution Timeseries — Total SW |
|:---:|:---:|
| ![TS GW](../docs/images/Sigma_Attribution_Timeseries_Total_GW.png) | ![TS SW](../docs/images/Sigma_Attribution_Timeseries_Total_SW.png) |


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

Frisvold, G. B., Sanchez, C., Gollehon, N., Megdal, S. B., & Brown, P. (2018). Evaluating gravity-flow irrigation with lessons from Yuma, Arizona, USA. _Sustainability_, _10_(5), 1548. https://doi.org/10.3390/su10051548.

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

Majumdar, S., Smith, R., Butler, J. J., & Lakshmi, V. (2020). Groundwater withdrawal prediction using integrated multitemporal remote sensing data sets and machine learning. _Water Resources Research_, _56_(11), e2020WR028059. https://doi.org/10.1029/2020WR028059.

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. _Hydrological Processes_, _36_(11), e14757. https://doi.org/10.1002/hyp.14757.

Majumdar, S., Smith, R. G., Hasan, M. F., Wilson, J. L., White, V. E., Bristow, E. L., Rigby, J. R., Kress, W. H., & Painter, J. A. (2024). Improving crop-specific groundwater use estimation in the Mississippi Alluvial Plain: Implications for integrated remote sensing and machine learning approaches in data-scarce regions. _Journal of Hydrology: Regional Studies_, _52_, 101674. https://doi.org/10.1016/j.ejrh.2024.101674.

Majumdar, S., Smith, R. G., & Hasan, M. F. (2025). A High-Resolution Data-Driven Monthly Aquaculture and Irrigation Water Use Model in the Mississippi Alluvial Plain. _IGARSS 2025 — 2025 IEEE International Geoscience and Remote Sensing Symposium_, 2686–2691. https://doi.org/10.1109/IGARSS55030.2025.11243173.

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