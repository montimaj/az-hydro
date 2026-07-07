# AZ-Hydro Data Archive

Companion data archive for the AZ-Hydro pipeline (Majumdar et al. 2026,
*Sci Data* — in prep.).  This Zenodo deposit holds the input datasets
the pipeline consumes plus the full set of generated outputs (predictor
rasters, ML predictions, uncertainty bands, intercomparisons, figures).

The pipeline source code lives in the `azhydro/` directory of the
parent GitHub repository (https://github.com/montimaj/az-hydro) and is
not duplicated here.  See `azhydro/README.md` for runtime instructions
and methodology.

## Total size

| Component | Size | Notes |
|---|---|---|
| **Inputs** (Zenodo) | **~9 GB** | Pipeline-required files only (see per-directory inventory below).  After the user downloads the external USGS NHM / Reitz / PS reanalysis bundle from USGS ScienceBase (see "External datasets" below), on-disk Inputs grows to ~20 GB. |
| &emsp;`Inputs/GEE_Data/` | ~3.4 GB | Raw 2 km Google Earth Engine tiles + AZ HUC12 vector |
| &emsp;`Inputs/GW_Data/` | ~5.8 GB | ADWR meter records, well registry, GW basin / AMA-INA / CAP / SRP / streamflow / USBR vectors, statewide WTD TIFs |
| &emsp;`Inputs/USGS WU/` | <1 MB | Two curated USGS/ADWR rollup CSVs |
| **Outputs** (generated) | **~200 GB** | Full pipeline output suite |
| &emsp;`Outputs/GEE_Mosaics_2000m{,_GCM,_LULC}{,_Reproj}/` | ~14 GB | Mosaicked 2 km predictors + 9 alternate ensemble stacks |
| &emsp;`Outputs/Predictor_Data_All_Wells_2000m/` | ~1.8 GB | Per-year predictor TIF stacks (1896–2099) |
| &emsp;`Outputs/GW/` | ~14 GB | Observed ADWR-metered withdrawal depth/volume rasters + per-year vector shapefiles |
| &emsp;`Outputs/GW_Data/` | ~8 GB | Reprojected vector products (basins, wells, CAP, etc.) |
| &emsp;`Outputs/ML_Model_All_Wells_2000m/` | ~161 GB | ML evaluation (~121 GB), Step 3 prediction + UQ + augmented rasters + figures (~40 GB) |
| **Total** | **~220 GB** | After downloading both the Zenodo deposit and the external USGS data products |

External datasets that the pipeline reads but are **not** redistributed
on Zenodo (must be obtained by the user from the cited source) total
roughly an additional ~5.5 GB on disk — the USGS NHM withdrawals +
CU/IE + Reitz irrigation + PS reanalysis bundle.  See **External
datasets** below for the full inventory.

---

## `Inputs/`

### `Inputs/GEE_Data/`

Raw geographic ancillary data and the Google Earth Engine (GEE) export
tiles that feed the predictor stack.  GEE tiles are downloaded once by
`pipeline.py` Step 0 (`download_gee_data()`) and then mosaicked into
`Outputs/GEE_Mosaics_2000m*/` for the ML pipeline.

| Path | Description |
|---|---|
| `AZ_HUC12.geojson` | Arizona HUC-12 watershed vector, used for HUC-12-resolution USGS NHM intercomparison and Peff regression. |
| `GEE_Tiles_2000m/` | Baseline ML predictor tiles (2 km resolution, 80 km × 80 km tile size; LULC, climate, soil, irrigation, ET, Peff bands per year 1896–2099 wherever the source band exists). |
| `GEE_Tiles_2000m_<GCM>/` | Per-GCM tiles for σ_MACA (5 directories): `CCSM4`, `CNRM-CM5`, `HadGEM2-ES365`, `MIROC-ESM-CHEM`, `inmcm4`.  Each holds the same predictor bands as the baseline tiles but with that GCM's downscaled MACA climate forcing for projection years 2026–2099. |
| `GEE_Tiles_2000m_LULC_<scenario>/` | Per-LULC-scenario tiles for σ_LULC (4 directories): `A1B`, `A2`, `B1`, `B2`.  Same predictor bands as baseline but with the alternative USGS FORE-SCE LULC trajectory through 2099. |

### `Inputs/GW_Data/`

ADWR meter records, well registry, scenario-specific delivery data, and
hydrologic ancillary vectors.

> **Note on ADWR-provided datasets.**  All ADWR datasets bundled below
> (well registry, GW basin / sub-basin polygons, AMA/INA polygons,
> surface watersheds, per-year meter records, etc.) are publicly
> available from the Arizona Department of Water Resources at
> [https://www.azwater.gov/](https://www.azwater.gov/) (Hydrology →
> GIS Data, Maps and Reports → Reports & Data).  They are mirrored
> here for one-click pipeline reproducibility.

| Path | Description |
|---|---|
| `ADWR_Groundwater_Subbasin/` | ADWR groundwater sub-basin shapefile for sub-basin aggregation in time-series outputs. |
| `AMA_and_INA.geojson` | Active Management Areas + Irrigation Non-Expansion Areas polygons. |
| `AZ.geojson` | Arizona state boundary polygon. |
| `AZ_Polygons_80000m.geojson` | 80 km × 80 km tiling grid that GEE tile downloads use. |
| `CAP/CAP_Service_Area.geojson` | Central Arizona Project service area polygon (3 county sub-units: Maricopa / Pima / Pinal).  Drives the per-pixel CAP delivery perturbation in `apply_cap_delivery_perturbation`. |
| `CAP/CAP Delivery Data DRI Request.xlsx` | Annual CAP delivery volumes by sub-contractor / category, supplied by CAP for the AZ-Hydro project (1985–2024). |
| `GRAIN_v.1.0/GeoParquet/us-west_GRAIN_v.1.0.parquet` | Western-US subset of the GRAIN canal-network database from Suresh et al. (2026), *Earth System Science Data* 18(3), 1855–1875.  DOI: [10.5194/essd-18-1855-2026](https://doi.org/10.5194/essd-18-1855-2026).  Bundled here (~55 MB after extracting only the western-US tile) — used to build the `canal_density` predictor.  The full release is available at the source DOI; replace this folder with a fresh download if you need other regions. |
| `Groundwater_Basin/` | Statewide GW basin polygons shapefile (52 basin/management-area polygons spanning Arizona's 51 ADWR groundwater basins) — primary basin aggregation unit. |
| `Meter Data/GW_<year>.csv` | Per-year ADWR meter records (1984–2024, one CSV per year, 41 files).  Each row = one well-year with reported `AF Pumped`, lat/lon, basin, etc.  These are the primary training labels for the ML model. |
| `SRP/SRP WATER DELVS HISTORY.xlsx` | Salt River Project annual delivery history (used to validate Phoenix AMA SW totals). |
| `Streamflow/` | USGS gauge monthly streamflow for the Colorado River system (HCDN reference + USBR sites; 8 stations, 1896-present where available).  Used in σ_USBR and Lees Ferry inflow context. |
| `Surface_Watershed.geojson` | Surface watershed boundaries used for SW capture index aggregation. |
| `USBR/` | USBR / HCDN monthly streamflow zips (compressed mirror of `Streamflow/`) plus extracted CSVs.  Used by σ_USBR ensemble member loader. |
| `Water Rights/stateWaterRightsHarmonized/arizona/arizonaStatePOD.{shp,dbf,shx,cpg,prj}` | HarDWR Western U.S. water-rights Point-of-Diversion shapefile (Arizona subset, ~142 MB) from Lisk et al. (2024).  DOIs: data — [10.57931/2475303](https://doi.org/10.57931/2475303); paper — [10.1038/s41597-024-03434-6](https://doi.org/10.1038/s41597-024-03434-6).  Used as the priority-date source for canal `first_delivery_year` dating and the `irr_sw_rights_density` / `nonirr_sw_rights_density` predictor rasters.  Only the shapefile is needed by the pipeline — the harmonized-records CSVs from the original release are not used and have been excluded from this bundle to save space. |
| `Well_Registry_2024/` | ADWR Well Registry snapshot (Dec 2024) — used to build the per-pixel `well_density` feature (the #1 SHAP-importance predictor). |
| `wtd_states/wtd_<state>.tif` | Water-table-depth statewide TIFs (`wtd_arizona.tif`, `wtd_california.tif`, `wtd_nevada.tif`) from Ma et al. (2026), *High resolution US water table depth estimates reveal quantity of accessible groundwater*, Communications Earth & Environment 7(1), 45.  DOI: [10.1038/s43247-025-03094-3](https://doi.org/10.1038/s43247-025-03094-3).  Bundled in this Zenodo deposit because the per-state TIFs are not separately discoverable on the source data portal.  The CONUS dataset is also available as a Google Earth Engine asset at `projects/nmose-openet/assets/WTD-US` — see the [CONUS WTD explorer](https://code.earthengine.google.com/87c965aae54facde740850b0949fc3b4) for an interactive viewer; users with GEE access can pull the WTD raster directly from there instead of downloading the bundled per-state TIFs. |

### `Inputs/USGS WU/`

USGS / ADWR water-use **anchor** datasets used by Step 4 intercomparison
and the partition calibration.  Only the two curated AZ-statewide rollup
CSVs are bundled on Zenodo; per-HUC12 reanalysis products (NHM, PS,
Reitz) and CDL/FRIS crop surveys must be obtained from USGS ScienceBase
— see **External datasets** below.

| Path | Description |
|---|---|
| `AZ_Annual_WU_Summary.csv` | Curated USGS + ADWR statewide annual water-use rollup (1985–2024).  Total / Irrigation / Public Supply / GW / SW columns, derived from USGS Circulars 1950–2015 + ADWR Annual Reports 2016–2024.  Primary anchor for partition calibration. |
| `USGS_AZ_Water_Use_1950_1980.csv` | USGS Circular legacy Arizona water-use 1950–1980 (5-year cadence).  Used as a back-cast anchor for the partition's pre-1985 era-mapped factors. |

---

## `Outputs/`

All `Outputs/` are generated deterministically by `pipeline.py` from
`Inputs/`.  They are bundled on Zenodo so users can reproduce the
analysis without a 10–20 hour full pipeline run.

### `Outputs/GEE_Mosaics_2000m*/`

Mosaicked 2 km predictor TIFs assembled from `Inputs/GEE_Data/GEE_Tiles_*/`.
There are 19 directories total (1 baseline + 5 GCM + 4 LULC, each in
both raw and reprojected variants):

| Path | Description |
|---|---|
| `GEE_Mosaics_2000m/` | Baseline mosaicked predictor TIFs (one per band per year). |
| `GEE_Mosaics_2000m_<GCM>/` | Per-GCM σ_MACA mosaics for `CCSM4`, `CNRM-CM5`, `HadGEM2-ES365`, `MIROC-ESM-CHEM`, `inmcm4` (5 dirs × ~865 MB). |
| `GEE_Mosaics_2000m_LULC_<scenario>/` | Per-LULC σ_LULC mosaics for `A1B`, `A2`, `B1`, `B2` (4 dirs × ~865 MB). |
| `GEE_Mosaics_2000m_*_Reproj/` | Same content reprojected to working CRS for the ML pipeline (9 dirs × ~367 MB). |

### `Outputs/Predictor_Data_All_Wells_2000m/`

Per-year per-band predictor TIFs (1896–2099 × ~12 bands = ~2,662 files,
~1.8 GB).  Bands include `Canal_Density`, `Crop_Fraction`, `ET`, `ETo`,
`Peff`, `Tmax`, `Tmin`, `Urban_Fraction`, `Well_Density`, etc.  Each
TIF has the band name and year encoded in the filename
(e.g. `Canal_Density_1985.tif`).

### `Outputs/GW/`

Observed-data products derived from `Inputs/GW_Data/Meter Data/`.

| Path | Description |
|---|---|
| `Rasters/GW_Depths_All_Wells_2000m/` | Per-year ADWR-metered withdrawal **depth** rasters (mm).  41 TIFs, 1984–2024. |
| `Rasters/GW_Volumes_All_Wells_2000m/` | Per-year ADWR-metered withdrawal **volume** rasters (m³).  41 TIFs. |
| `Vectors/All_Wells/` | Per-year ADWR-metered well vector shapefiles with `AF Pumped` attribute.  Used by Step 0 well registry merge and σ_GW ensemble. |

### `Outputs/GW_Data/`

Reprojected ancillary vectors (matching the CRS of the predictor stack).

| Path | Description |
|---|---|
| `Vector_Reproj/ADWR_Groundwater_Subbasin.{shp,dbf,...}` | Sub-basin polygons in working CRS. |
| `Vector_Reproj/AMA_and_INA.geojson` | Active Management / Irrigation Non-Expansion Areas in working CRS. |
| `Vector_Reproj/AZ.geojson` | AZ state boundary in working CRS. |
| `Vector_Reproj/AZ_Polygons_80000m.geojson` | Tiling grid in working CRS. |
| `Vector_Reproj/CAP_Service_Area.geojson` | CAP service area in working CRS. |
| `Vector_Reproj/Groundwater_Basin.{shp,dbf,...}` | Statewide GW basin polygons in working CRS. |
| `Vector_Reproj/HUC12_processed.{shp,dbf,...}` | HUC-12 polygons clipped to AZ + reprojected. |
| `Vector_Reproj/Surface_Watershed.geojson` | Surface watershed boundaries in working CRS. |

### `Outputs/ML_Model_All_Wells_2000m/`

ML pipeline outputs — the core scientific product.  ~161 GB total,
partitioned into Step 2 evaluation (~121 GB) and Step 3 full prediction
(~40 GB).

| Path | Description |
|---|---|
| `AZ_Data.parquet` | Master predictor + meter-label DataFrame (1896–2024 metered + 1896–2099 prediction) — the input to `Model_Evaluation/` and `Full_Prediction_XGBRF/`. |
| `Basin_LULC_Deltas.csv` | Per-basin per-year LULC change summary (URBAN / AGRI fraction trajectory used by Step 3 LULC-projection diagnostics). |
| `EDA/` | Exploratory data-analysis figures (predictor distributions, correlations, target histograms, IrrMapper coverage maps). |
| `Model_Evaluation/` | Step 2 outputs — five evaluation strategies (Random / Pixel Holdout / Temporal LOO / Spatial LOO / Seeded Spatial LOO) × four model families (XGB / LGBM / RF / XGBRF) with per-fold predictions, error metrics, calibration plots, and seed ensembles for σ_Model.  ~121 GB. |
| `Full_Prediction_XGBRF/` | Step 3 outputs — see breakdown below.  ~40 GB. |

#### `Outputs/ML_Model_All_Wells_2000m/Full_Prediction_XGBRF/`

Per-year prediction stack from the production XGBRF model, plus all
downstream products.

| Sub-path | Content |
|---|---|
| `Predicted_Rasters/` | Per-year total-withdrawal prediction TIFs (1896–2099, mm depth). |
| `Total_GW_Rasters/`, `Total_SW_Rasters/` | Per-year partition outputs (Total_GW, Total_SW) in mm depth. |
| `Irrigation_GW_Rasters/`, `Irrigation_SW_Rasters/`, `Non_Irrigation_GW_Rasters/`, `Non_Irrigation_SW_Rasters/` | Per-category × per-source partition rasters. |
| `Irrigation_CU_Rasters/`, `Irrigation_GW_CU_Rasters/`, `Irrigation_SW_CU_Rasters/` | Consumptive-use (CU) rasters derived from withdrawal × Irrigation Efficiency. |
| `OOD_Rasters/` | Out-of-distribution probability rasters per year (isolation-forest density on the predictor stack). |
| `Total_GW/`, `Total_SW/`, `Irrigation/`, `Irrigation_GW/`, `Irrigation_SW/`, `Irrigation_CU/`, `Irrigation_GW_CU/`, `Irrigation_SW_CU/`, `Non_Irrigation/`, `Non_Irrigation_GW/`, `Non_Irrigation_SW/` | Per-category augmented rasters: 6-band stacks (prediction, σ, CV, SNR, lower 95 % CI, upper 95 % CI) per year.  |
| `Uncertainty/` | Step 3b UQ outputs — per-component rasters (Sigma_MACA, Sigma_Model, Sigma_Irr, Sigma_LULC, Sigma_GW, Sigma_USBR, Sigma_CU, Sigma_Total), per-basin / per-sub-basin σ CSVs, attribution figures, sensitivity diagnostics, and `CAP_Scenario/` sub-directory with the 7-scenario CAP delivery sweep (Baseline / DCP Tier 0–3 / Basic Coordination / Extreme Shortage). |
| `Raster_Maps/` | Step 3g era-mean raster maps (Hindcast / Historical / Projection × all categories), trend maps (per-pixel Mann-Kendall + Sen slope on annual rasters), ternary attribution figures, and the `CAP_Scenario/` spatial-map suite. |
| `SW_Capture/` | SW Capture Index outputs (per-pool capture fraction / depth / volume rasters with σ propagation, per-well capture disaggregation, surface-watershed summaries). |
| `Well_Package/` | Four GeoParquet files (Parquet with embedded `geometry` column + GeoParquet metadata): `Well_Package_{mm,ft,m3,AF}.parquet`, one per unit convention.  Each holds ~34.7 M rows (every ADWR well × every year 1984–2099) × 34 columns: `REGISTRY_I`, `Year`, `WATER_USE`, `geometry`, plus prediction + σ for the 11 partition / CU categories and 3 SW Capture bands.  Primary user-facing per-well product. |
| `Annual_Summaries/` | Statewide and per-basin annual summary CSVs (Total_Predicted, per-category, σ_Total). |
| `ADWR/` | ADWR partner CSV delivery (Step 3i): 16 long-format CSVs — 8 per-basin (`Basin_<cat>.csv`) + 8 per-sub-basin (`Subbasin_<cat>.csv`) — covering Total_Predicted, Total_GW, Total_SW, Irrigation_GW/SW, Non_Irrigation_GW/SW, and Irrigation_CU.  Plus a `readme.txt` describing the schema and source.  Aggregated from per-basin/sub-basin `<cat>/Basin_Time_Series/*_Annual.csv` and `<cat>/Subbasin_Time_Series/*_Annual.csv` files; each row has `Year, Basin/Subbasin, Mean_Depth_mm, Mean_Depth_ft, Volume_m3, Volume_AF, Era` (sub-basin files additionally have `Parent_Basin`).  Designed for delivery to ADWR / state-agency partners (e.g. Senate Bill 1740 basin assessments) without requiring users to navigate the full per-basin file tree. |
| `Basin_Time_Series/`, `AMA_INA_Time_Series/`, `Subbasin_Time_Series/` | Per-basin / per-AMA / per-sub-basin time series figures + CSVs. |
| `Withdrawal_Intercomparison/`, `CU_Intercomparison/`, `Peff_Intercomparison/`, `PS_Intercomparison/`, `NHM_IE_Basins/`, `USGS_Calibration_Bars/`, `CAP_SRP_Validation/` | Step 4 intercomparison outputs against USGS NHM withdrawals / NHM CU / Reitz Peff / PS reanalysis / CAP+SRP delivery records. |
| `Model/` | Trained model pickles, hyperparameter search history, SHAP cache, calibration curves. |
| `Model_Interpretability/` | SHAP summary plots, partial-dependence plots, permutation importance, feature-importance bars. |
| `Bias_Correction/` | Per-basin bias-correction diagnostics (residual analysis, ECDF matching). |
| `Full_Period_Time_Series.csv` + `.png` | Headline statewide time series (1896–2099). |
| `Era_Summary_Bar.png`, `Mean_Annual_Predicted_mm.tif`, `Graphical_Abstract_Fig1.png`, `Prediction_Exceedance_Summary.csv` | Top-level summary figures and CSV. |
| `data/` | Misc serialized intermediates used by Step 3. |

---

## External datasets

The pipeline reads several external datasets that are **not** included
in this Zenodo deposit (because their authoritative source is a
distinct curated archive).  Users must download these from the cited
source and place them at the indicated paths inside `Data/Inputs/`
before re-running the pipeline.  Naming conventions matter — the
pipeline references these paths by exact name.

### USGS ScienceBase data products → `Data/Inputs/USGS WU/<subdir>/`

#### `USGS_NHM_Withdrawals/`

Source: Haynes et al. (2023, USGS data release) — *Monthly crop
irrigation withdrawals and efficiencies by HUC12 watershed for years
2000–2020 within the conterminous United States* (ver. 2.0, Sep 2024).
DOI: [10.5066/P9LGISUM](https://doi.org/10.5066/P9LGISUM).

Expected files (CSV, one per band, all per-HUC12 monthly 2000–2020):
- `IR_HUC12_GW_WD_monthly_2000_2020.csv` — irrigation GW withdrawal
- `IR_HUC12_SW_WD_monthly_2000_2020.csv` — irrigation SW withdrawal
- `IR_HUC12_Tot_WD_monthly_2000_2020.csv` — irrigation total withdrawal
- `IR_HUC12_Eff_annual_2000_2020.csv` — annual irrigation efficiency
- `CDL_FRIS_crosswalk.csv` — crop crosswalk

#### `USGS_NHM_CUIrr/`

Source: Martin et al. (2023, USGS data release) — *Irrigation water use
reanalysis for the 2000–20 period by HUC12, month, and year for the
conterminous United States* (ver. 2.0, Sep 2024).
DOI: [10.5066/P9YWR0OJ](https://doi.org/10.5066/P9YWR0OJ).
Companion paper: Martin et al. (2025), *J. Hydrology* 662, 133909.
DOI: [10.1016/j.jhydrol.2025.133909](https://doi.org/10.1016/j.jhydrol.2025.133909).

Expected files:
- `Irr_CU_HUC12_Tot_annual_2000_2020.csv` — annual irrigation CU
- `PPTeff_HUC12_Tot_annual_2000_2020.csv` — annual effective precipitation
- `Irrigation_reanalysis_metadata.xml` — FGDC metadata

#### `USGS_PS_Data/`

Source: Luukkonen et al. (2023, USGS data release) — *Public supply
water use reanalysis for the 2000–2020 period by HUC12, month, and
year for the conterminous United States*.
DOI: [10.5066/P9FUL880](https://doi.org/10.5066/P9FUL880).
Companion paper: Alzraiee et al. (2024), *Water Resources Research*
60(7).
DOI: [10.1029/2023WR036632](https://doi.org/10.1029/2023WR036632).

Expected files:
- `PS_HUC12_GW_2000_2020.csv` — public-supply GW withdrawal per HUC12
- `PS_HUC12_SW_2000_2020.csv` — public-supply SW withdrawal per HUC12
- `PS_HUC12_Tot_2000_2020.csv` — public-supply total withdrawal per HUC12
- `PS_WU_reanalysis_v2.xml` — FGDC metadata

#### `USGS_Reitz_Irrigation/`

Source: Reitz et al. (2023, USGS data release) — *Historical
evapotranspiration for the conterminous U.S.*  (per-pixel monthly
gridded ET; 800 m).
DOI: [10.5066/P9EZ3VAS](https://doi.org/10.5066/P9EZ3VAS).
Companion paper: Reitz, Sanford & Saxe (2023), *Water Resources Research*
59(6).
DOI: [10.1029/2022WR034012](https://doi.org/10.1029/2022WR034012).

Expected sub-directories (each is a per-year stack of monthly TIFs
1980–2018, plus a zipped backup):
- `Irrigation_all_1980-2018/` — total irrigation evapotranspiration
- `Irrigation_groundwater_1980-2018/` — GW-supplied irrigation ET
- `Irrigation_surfacewater_1980-2018/` — SW-supplied irrigation ET
- `HistoricalET_metadata.xml` — FGDC metadata

### Other external datasets

Other USGS / agency PDFs (USGS Circulars 1950–2015, ADWR Annual
Reports 2016–2024, AWBA Plan of Operation, CAP / SRP technical reports,
WestWater Economic Impacts report, etc.) are reference materials cited
in `azhydro/README.md` but are not redistributed here.  They remain
freely available on the respective USGS, ADWR, CAP, AWBA, and
WestWater Research portals.

---

## Re-running the pipeline

After cloning the GitHub repository and downloading this Zenodo
deposit:

```bash
git clone https://github.com/montimaj/az-hydro.git
cd az-hydro
# Place the Zenodo data archive contents into ./Data/
# Then download the external datasets above into Data/Inputs/USGS WU/
# and Data/Inputs/GW_Data/ following the directory names listed above
cd azhydro
python pipeline.py --steps all
```

See `azhydro/README.md` for runtime-flag documentation, step-by-step
descriptions, and the full software-environment specification.

## Citation

If you use this dataset, please cite both the data archive and the
companion paper:

**Data archive (this Zenodo deposit):**

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C.
(2026).  *AZ-Hydro — Historical and Projected Arizona Annual Water
Use: Software, Input Data, Models, Raster and Well Package
Predictions, and Validation at 2 km Resolution (1896–2099).*
**Zenodo.** [DOI: 10.5281/zenodo.19057935](https://doi.org/10.5281/zenodo.19057935)

**Companion papers:**

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C.
(2026).  *Historical and projected groundwater/surface-water
withdrawals, irrigation consumptive use, and pumping-induced surface
water capture for Arizona, 1896–2099.*  In prep. for *Nature
Scientific Data*.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C.
(2026).  *Where Arizona's Water Goes: Declining Agricultural
Dominance and Rising Urban Demand Drive a Two-Century Shift in
Withdrawal Patterns (1896–2099).*  In prep. for *AGU Earth's Future*.

## License

This Zenodo deposit (data archives `az-hydro-data.7z` and
`az-hydro-headline.7z`, plus this README) is released under
**CC-BY-4.0** — see [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

The accompanying source code in the
[GitHub repository](https://github.com/montimaj/az-hydro) is released
separately under **BSD 3-Clause "Revised"** — see the `LICENSE` file
at the repo root.

External datasets bundled here (HarDWR water-rights shapefile, GRAIN
canal network, WTD TIFs) retain their original licenses — see the
cited source for terms.
