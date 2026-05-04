# AZ-Hydro Headline Data Archive

This is the **focused subset** of the full AZ-Hydro data deposit
(Majumdar et al. 2026, *Sci Data* — in prep.) — the actual scientific
deliverables highlighted in the paper, sized for reasonable download.

| Archive | Size | Contents |
|---|---|---|
| **`az-hydro-headline.7z`** (this file) | **~8 GB** | Published deliverables only |
| `az-hydro-data.7z` | ~74 GB | Full bit-identical reproducibility archive — raw inputs + Step 2 cross-validation + intermediate predictor stacks + per-component σ rasters |

Both archives are deposited in the same Zenodo record:
[10.5281/zenodo.19057936](https://doi.org/10.5281/zenodo.19057936).
For the per-directory inventory, methodology, and external-dataset
citations, see [`Data/README.md`](Data/README.md) inside this archive
(or in the GitHub repository at https://github.com/montimaj/az-hydro).

## Citation

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C.
(2026). *AZ-Hydro — Historical and Projected Arizona Annual Water
Use: Software, Input Data, Models, Raster and Well GeoPackage
Predictions, and Validation at 2 km Resolution (1896–2099).* Zenodo.
[https://doi.org/10.5281/zenodo.19057936](https://doi.org/10.5281/zenodo.19057936)

---

## What's included

All paths below are under
`Data/Outputs/ML_Model_All_Wells_2000m/Full_Prediction_XGBRF/`.

### 1. Six-band augmented per-category rasters (the published per-pixel per-year product)

Each per-category `*_Rasters/` directory contains per-year multi-band
TIFs (1896–2099) in **four unit-convention sub-directories**
(`Depth_mm/`, `Depth_ft/`, `Volume_m3/`, `Volume_AF/`).  Every TIF is
a 6-band stack with the following band order:

| Band | Content |
|---|---|
| 1 | Prediction (mm depth or AF volume) |
| 2 | σ (1σ uncertainty — quadrature-combined for `Predicted_Rasters/`'s `sigma_total_mm`; per-source σ for the others) |
| 3 | CV (= σ / |prediction|) |
| 4 | SNR (= |prediction| / σ) |
| 5 | Lower 95 % CI (= prediction − 1.96 σ) |
| 6 | Upper 95 % CI (= prediction + 1.96 σ) |

Categories included (12 directories):

- **`Predicted_Rasters/`** — Total Predicted withdrawal (= Total_GW + Total_SW = Irrigation + Non_Irrigation), the headline statewide product.  Also contains `ML_mean_annual_GW_mm.tif` + `ML_mean_annual_SW_mm.tif` era-mean TIFs.
- `Total_GW_Rasters/`, `Total_SW_Rasters/` — totals by source
- `Irrigation_Rasters/`, `Irrigation_GW_Rasters/`, `Irrigation_SW_Rasters/` — irrigation totals + by source
- `Non_Irrigation_Rasters/`, `Non_Irrigation_GW_Rasters/`, `Non_Irrigation_SW_Rasters/` — non-irrigation totals + by source
- `Irrigation_CU_Rasters/`, `Irrigation_GW_CU_Rasters/`, `Irrigation_SW_CU_Rasters/` — irrigation consumptive use totals + by source

Each `*_Rasters/` directory is paired with a sibling `*/` directory
(without the `_Rasters` suffix) holding **per-category summary CSVs
and time-series figures** (statewide trajectory, era bar chart,
basin/AMA/sub-basin breakdowns) — see § 6.

### 2. Per-pixel total uncertainty (`Uncertainty/Sigma_Total/`)

Per-year σ_total rasters and CSVs.  σ_total is the quadrature sum of
the six σ components (σ_MACA + σ_Model + σ_Irr + σ_LULC + σ_GW +
σ_USBR; σ_CU additionally for CU products).  The full per-component
σ rasters are in the larger `az-hydro-data.7z` archive; only the
combined σ_Total is included here.

### 3. Per-well GeoPackage + Parquet (`Well_Package/`)

Per-well predicted withdrawals + σ bands (1984–2099), GPKG and
GeoParquet formats.  Headline product for well-level analysis.

### 4. SW Capture Index (`SW_Capture/`)

Pumping-induced surface-water-capture analysis with σ propagation:

- `Total_SW_Capture_Fraction/`, `Irrigation_SW_Capture_Fraction/`, `Non_Irrigation_SW_Capture_Fraction/` — per-year augmented capture-fraction TIFs
- `Total_SW_Capture_Rasters/`, `Irrigation_SW_Capture_Rasters/`, `Non_Irrigation_SW_Capture_Rasters/` — per-year capture rasters in 4 unit conventions (Depth_mm, Depth_ft, Volume_m3, Volume_AF)
- `Basin_Capture_Fraction.csv`, `Subbasin_Capture_Fraction.csv`, `SW_Capture_Time_Series.csv` — aggregated time series

### 5. Aggregated time series

(Includes the per-category `Total_GW/`, `Total_SW/`, `Irrigation/`,
`Irrigation_GW/`, `Irrigation_SW/`, `Irrigation_CU/`,
`Irrigation_GW_CU/`, `Irrigation_SW_CU/`, `Non_Irrigation/`,
`Non_Irrigation_GW/`, `Non_Irrigation_SW/` directories — sibling to
each `*_Rasters/` in § 1 — plus the cross-category aggregations
below.)

- `Annual_Summaries/` — statewide + per-basin annual rollup CSVs
- `Basin_Time_Series/`, `AMA_INA_Time_Series/`, `Subbasin_Time_Series/` — per-basin / per-AMA / per-sub-basin annual time series CSVs + figures
- `Full_Period_Time_Series.csv` + `.png` — headline statewide trajectory
- `Era_Summary_Bar.png`, `Mean_Annual_Predicted_mm.tif`, `Prediction_Exceedance_Summary.csv`, `Graphical_Abstract_Fig1.png` — top-level summary figures and CSVs

### 6. CAP delivery scenario sweep (`Uncertainty/CAP_Scenario/`)

Eight CAP delivery scenarios (Baseline_900kAF, DCP Tier 0–3, Basic
Coordination, Extreme Shortage) re-partitioned 2026–2099:

- `CAP_Scenario_Statewide.csv`, `CAP_Scenario_Basin.csv`, `CAP_Scenario_Delta.csv`, `CAP_Scenario_Cumulative.csv` — primary outputs
- `CAP_Scenario_DCP_Tiers.png`, `CAP_Scenario_WestWater.png`, `CAP_Scenario_Basin.png`, `CAP_Scenario_Cumulative_Drawdown.png` — time-series figures
- `Basin_Sigma_CAP_Restricted_Total_GW_<window>.csv` — per-window per-basin σ on Total_GW restricted to the basin × CAP-pixel intersection

The matching spatial maps are in `Raster_Maps/CAP_Scenario/` (see § 8).

### 7. Validation / intercomparison outputs

Step 4 outputs comparing AZ-Hydro predictions against independent
agency datasets.  Each intercomparison directory contains time-series
plots, scatter plots, and per-source / per-basin metric CSVs; the
following table summarizes which spatial-diff levels (basin / HUC12 /
pixel) each one renders.

| Directory | Comparison source | Year window | Spatial-diff levels rendered |
|---|---|---|---|
| `Withdrawal_Intercomparison/` | USGS NHM withdrawals (Haynes 2023) + Reitz 2023 historical ET | ML-vs-NHM 2000–2020; ML-vs-Reitz 1980–2018; NHM-vs-Reitz 2000–2018 (pairwise intersection) | basin (`Spatial_Diff/`) + HUC12 (`HUC12_Comparison/Spatial_Diff/`) |
| `CU_Intercomparison/` | Martin et al. (2025) NHM consumptive-use reanalysis | 2000–2020 (common) | **basin + HUC12 + pixel** (all three at `Spatial_Diff/Spatial_Diff_{Basin,HUC12,Pixel}_CU.png`) |
| `Peff_Intercomparison/` | USDA-SCS + PCML + NHM Peff (Reitz/Haynes/Martin) | 2000–2020 (pinned common — see pipeline override) | basin (`Spatial_Diff/`) + HUC12 (`HUC12_Comparison/Spatial_Diff/`) + pixel (`Spatial_Diff/`) |
| `PS_Intercomparison/` | Luukkonen et al. (2023) public-supply reanalysis | 2000–2020 (common) | basin (4 panels: PS total + Non_Irrigation total/GW/SW) |
| `NHM_IE_Basins/` | NHM irrigation efficiency aggregated to basins | 2000–2020 | basin scatter + per-basin tables |
| `USGS_Calibration_Bars/` | Statewide AZ totals vs USGS Circulars 1950–2015 + ADWR Annual Reports 2016–2024 | 1950–2024 | annual statewide bar charts (no spatial diff) |
| `CAP_SRP_Validation/` | CAP delivery + SRP delivery records | 1985–2024 | per-CAP-county time series (no spatial diff) |

CU, Peff, and Withdrawal all use **pairwise common-year windows** so
each diff is computed apples-to-apples — see § 7 of the methodology
section in `azhydro/README.md` for the per-pair window definitions.

### 8. Era-mean and trend spatial figures (`Raster_Maps/`)

Hindcast / Historical / Projection era-mean raster maps for every
category, per-pixel Mann-Kendall + Sen-slope trend maps, ternary RGB
σ-attribution figures, and the CAP scenario spatial-map suite (basin
and pixel cumulative drawdown, σ_cum context, basin and pixel SNR maps,
plus the per-scenario per-window cumulative ΔGW pixel rasters in
`Raster_Maps/CAP_Scenario/Pixel_Rasters/`).

---

## What's NOT included (available in `az-hydro-data.7z`)

| Excluded | Why excluded from headline |
|---|---|
| `Data/Inputs/` | ~9 GB of raw inputs (GEE tiles, ADWR meter records, ancillary vectors, WTD).  Only needed to re-run the pipeline from scratch. |
| Step 2 cross-validation (`ML_Model_All_Wells_2000m/Model_Evaluation/`) | ~121 GB of 5-strategy × 4-model evaluation outputs.  Per-strategy metrics summarized in the paper. |
| Per-component σ rasters (`Sigma_MACA/`, `Sigma_Model/`, `Sigma_Irr/`, `Sigma_LULC/`, `Sigma_GW/`, `Sigma_USBR/`, `Sigma_CU/`) | ~13 GB.  Only the quadrature-combined `Sigma_Total/` is included here.  Per-component σ is needed for the σ-attribution diagnostics suite but not for downstream users. |
| Out-of-distribution rasters (`OOD_Rasters/`) | 68 MB.  Diagnostic, not a primary deliverable. |
| Trained model + interpretability (`Model/`, `Model_Interpretability/`, `EDA/`, `Bias_Correction/`) | Diagnostic intermediates. |
| Intermediate predictor stacks (`Outputs/GEE_Mosaics_*/`, `GW_Data/`, `Predictor_Data_*/`, `GW/`) | ~37 GB derivable from inputs. |

If you need any of the above, download the full
`az-hydro-data.7z` from the same Zenodo record.

---

## Re-creating predictions from scratch

This headline archive contains the *outputs*, not the source inputs
needed to re-run the pipeline.  To reproduce everything from the raw
inputs, download `az-hydro-data.7z` from the same Zenodo record (~74
GB) plus the external USGS NHM / Reitz / PS data products from USGS
ScienceBase (see `Data/README.md` for citations and naming
conventions), then follow the runtime instructions in
`azhydro/README.md` in the GitHub repository.

## License

This Zenodo deposit (data archives `az-hydro-data.7z` and
`az-hydro-headline.7z`, plus this README) is released under
**CC-BY-4.0** — see [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

The accompanying source code in the
[GitHub repository](https://github.com/montimaj/az-hydro) is released
separately under **BSD 3-Clause "Revised"** — see the `LICENSE` file
at the repo root.
