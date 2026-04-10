# AZ-Hydro: Two Centuries of Arizona Water Use — Historical and projected withdrawals, consumptive use, and surface water capture, 1896–2099

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19057936.svg)](https://doi.org/10.5281/zenodo.19057936)

Maintainer: [Dr. Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu]

## TL;DR

**What it is.** A physics-constrained machine-learning pipeline that produces annual groundwater and surface-water withdrawals, irrigation consumptive use, and pumping-induced surface-water capture for every 2 km pixel in Arizona, every year from 1896 through 2099.

**What's new.** The base ML method (XGBoost trained on metered well records with remote-sensing-derived predictor features) is *not itself novel* — our group has applied it across four regions over the last five years: the Kansas High Plains Aquifer ([Majumdar et al., 2020](https://doi.org/10.1029/2020WR028059); [Asfaw et al., 2025](https://doi.org/10.1016/j.agwat.2025.109691)), the Mississippi Alluvial Plain ([Majumdar et al., 2024](https://doi.org/10.1016/j.ejrh.2024.101674); [Majumdar et al., 2025](https://doi.org/10.1109/IGARSS55030.2025.11243173)), Arizona ([Majumdar et al., 2022](https://doi.org/10.1002/hyp.14757)), and the broader Western U.S. for effective precipitation ([Hasan et al., 2025](https://doi.org/10.1016/j.agwat.2025.109821)). What is novel about AZ-Hydro is everything *built on top of* the base ML method to turn a single-region withdrawal-prediction model into a complete state-scale water-budget framework:

1. **Physics-informed feature engineering and predictor stack** — pump-capacity-weighted irrigation/non-irrigation fractions, canal-weighted streamflow with Gaussian smoothing across canal service areas, HarDWR water-rights densities split by water-use category, USGS / FORE-SCE LULC bias-corrected to NLCD spatial pattern, and Ma et al. (2026) high-resolution WTD as model inputs — each adding hydrologically meaningful information that off-the-shelf predictor stacks lack.
2. **Density-ratio GW/SW partitioning** — uses ADWR well density vs. HarDWR surface-water rights density, modulated by canal-weighted streamflow, to split predicted total pumping into eight conservation-consistent withdrawal categories with locally observable infrastructure rather than global statistical proxies.
3. **Per-pixel, per-year surface-water capture quantification** — a process-informed proxy that apportions GW pumping into stream-depletion vs. storage-mining shares at 2 km annual resolution across an entire state and a 204-year window, work that has previously only been done one basin at a time with calibrated transient MODFLOW–SFR simulations.
4. **A 204-year continuous record** — hindcast (1896–1983) + historical (1984–2025) + projection (2026–2099) all in one self-consistent framework, with the projection driven by 5 GCMs × 2 RCPs × 4 USGS LULC scenarios × 112 streamflow ensemble members.
5. **A first-of-a-kind statewide irrigation consumptive use dataset** for Arizona at 2 km annual resolution — no public alternative exists at this combination of spatial resolution, time horizon, and per-pixel / per-well disaggregation.
6. **Hybrid five-component σ_total uncertainty quantification** — σ_MACA (5 GCMs) + σ_model (10 seeds) + σ_irr (irrigation-fraction counterfactuals) + σ_LULC (4 USGS scenarios) + σ_GW (well-density feature sensitivity), combined in t-corrected quadrature with physics-based CU error propagation, producing 6-band augmented rasters (prediction, σ, CV, SNR, lower/upper 95 % CI) for every product and unit. To our knowledge no prior water-use ML study at this scale has reported a UQ framework of comparable rigor.

**No prior study we are aware of provides this combination of feature engineering, partitioning, capture quantification, hindcast/projection coverage, and 5-component UQ for any U.S. state.** The base ML method is shared with our four prior regional applications; everything above is new in AZ-Hydro and is what makes the framework state-scale, two-century, and uncertainty-honest rather than just a model that maps wells to pixels.

**The headline validation.** The model is trained *only* on metered ADWR records from the ten AMA/INA management areas, then applied to every 2 km pixel statewide — including ~25 unmetered basins where it has never seen a single training label. Despite this, four independent agency comparisons all land within ~1 percentage point or 0.1 MAF of the reported values:

1. **2017 statewide reconciled total (ADWR):** model + ~2.26 MAF federal-delivery offset = 6.99 MAF vs ADWR's 7.0 MAF ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)) — **central closure within 0.01 MAF**, with the model-side 95 % CI of 5.71–8.29 MAF comfortably bracketing both.
2. **2015 GW pumping (USGS / [Dieter et al. 2018](https://doi.org/10.3133/cir1441); Arizona summary in [NGWA 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf)):** model 3.17 MAF vs USGS 3.09 MAF — **central closure within 0.08 MAF**, inside the model's 1.90–4.44 MAF 95 % CI.
3. **2015 statewide GW share** ([Dieter et al. 2018](https://doi.org/10.3133/cir1441) / [NGWA 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf))**:** model 45.9 % vs USGS 46 % — **within 0.1 percentage point**.
4. **2019 statewide irrigation share (ADWR):** model 72.8 % vs ADWR's reported ~72 % share of total Arizona water use going to agriculture ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)) — **within ~0.8 percentage point**.

The 2019 ADWR statewide GW share comparison is the loosest of the agency cross-checks at 3 percentage points (model 44 % vs ADWR 41 %, [MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)) and is included for completeness. **No calibration to any of these agency values, no training labels from any of the ~25 unmetered basins that contribute roughly 30 % of the predicted statewide volume.**

**What it doesn't do.** See the [Known Limitations](azhydro/README.md#known-limitations) subsection in the methods README for the five structural caveats (deep-hindcast extrapolation, projection structural-change blindness, the irrigation efficiency paradox in CU projections, sparse metering in Willcox/Hualapai, and the static water table depth raster).

**Where to start.** Methods and CLI usage: [`azhydro/README.md`](azhydro/README.md). GEE export scripts: [`gee/README.md`](gee/README.md). Input data: [Zenodo archive](https://doi.org/10.5281/zenodo.19057936).

## Graphical Abstract

![Graphical Abstract](docs/images/Graphical_Abstract_Fig1.png)

**(a)** Mean annual predicted withdrawal depth (mm) across Arizona (1896-2099) with groundwater basin boundaries and AMA/INA labels. **(b)** Statewide annual withdrawal time series with 95% confidence intervals across three eras: Hindcast (1896-1983), Historical (1984-2025), and Projection (2026-2099). **(c)** Era-average withdrawal volumes with 95% CI error bars, showing the progression from 1,353k acre-ft (Hindcast) through 4,370k (Historical) to 5,491k (Projection).

## Abstract

AZ-Hydro is a physics-constrained machine learning pipeline for estimating annual groundwater and surface water withdrawals, consumptive use, and pumping-induced surface water capture across Arizona at 2 km resolution from 1896 to 2099, building on the foundational Arizona groundwater withdrawal study by [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757). The pipeline fuses satellite-derived and climate-model-projected predictor data — including evapotranspiration, reference ET, precipitation, effective precipitation, temperature, land use/land cover, crop fraction, urban fraction, irrigated fraction, groundwater fraction, water table depth, soil properties, canal-weighted streamflow, canal density, and well density — into a spatially explicit predictor stack via Google Earth Engine. LULC-derived features are bias-corrected at the basin scale so that their temporal trajectory reflects source-specific (USGS / FORE-SCE) change anchored to NLCD's pixel-level spatial pattern, and streamflow data is bias-corrected at the per-site monthly scale to remove systematic offsets between USGS observations and USBR ensemble projections — both analogous to the climate delta-method bias correction. An XGBoost Random Forest (XGBRF) model is tuned with Optuna TPE hyperparameter search (50 trials, 5-fold CV), parallelized with Dask, and trained on metered Arizona Department of Water Resources (ADWR) groundwater withdrawal records (1984–2024) **restricted to the ten AMA/INA management areas** (Phoenix, Pinal, Tucson, Prescott, Santa Cruz, Douglas, Willcox AMAs and Joseph City, Harquahala, Hualapai Valley INAs), which are the only Arizona basins with mandatory metering. The eight legacy AMA/INAs provide continuous training records since 1984; Willcox AMA and Hualapai Valley INA were designated more recently and are sparsely metered both temporally and spatially, contributing much less training signal than the legacy areas. At prediction time the model is applied to every 2 km pixel statewide, including the ~25 unmetered "Other" basins (Yuma, Lower Gila, Parker, Lake Havasu, Bill Williams, the Mogollon plateau, etc.), where no per-well meter records exist anywhere — making every statewide aggregate, every river-corridor capture-index value, and every agency reconciliation an out-of-distribution test of the framework rather than an in-sample fit. Up to 13 models are benchmarked across five evaluation strategies: random holdout, pixel-level spatial holdout, temporal leave-one-out (eight configurations), spatial leave-one-out, and seeded spatial leave-one-out (10 % local calibration). Physical constraints are enforced post-hoc: conservation-consistent withdrawal partitioning (Irrigation + Non-Irrigation = Total, GW + SW = Total) using a density-ratio GW/SW split (ADWR well density vs. HarDWR surface-water rights density, boosted by focal-normalized canal-weighted streamflow), pump-capacity-weighted irrigation/non-irrigation allocation, well density masking, and physics-based consumptive use calculation (CU = IE × Withdrawal) with USGS NHM basin-level irrigation efficiencies. A hybrid uncertainty quantification framework combines five independent error components via quadrature to produce 6-band augmented GeoTIFFs (prediction, σ, CV, SNR, lower/upper 95 % CI) for every product and unit. A Surface Water Capture Index quantifies where GW pumping likely depletes surface water, combining hydraulic connectivity (exponential decay with water table depth from [Ma et al., 2026](https://doi.org/10.1038/s43247-025-03094-3)) and canal-delivered surface water availability, with uncertainty bounds at three characteristic depths (λ = 5, 10, 20 m). A well-level package disaggregates pixel-level rasters to ~170,000 individual ADWR wells using capacity-proportional weighting, including per-well uncertainty bounds and capture index values. Predictions are independently validated against USGS National Hydrologic Model (NHM) HUC12 withdrawals, consumptive use, effective precipitation ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Martin et al., 2023](https://doi.org/10.5066/P9YWR0OJ); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)), and public supply ([Alzraiee et al., 2024](https://doi.org/10.1029/2023WR036632); [Luukkonen et al., 2023](https://doi.org/10.5066/P9FUL880)), as well as USGS Reitz 800 m irrigation water-use rasters ([Reitz et al., 2023a](https://doi.org/10.1029/2022WR034012); [2023b](https://doi.org/10.5066/P9EZ3VAS)), aggregated to ADWR groundwater basin totals.

## Key Contributions

- **204-year continuous water use dataset** — first statewide, spatially resolved (2 km) annual withdrawal estimates spanning hindcast (1896–1983), historical (1984–2025), and projected (2026–2099) eras
- **Density-ratio GW/SW partitioning** — uses ADWR well density vs. HarDWR surface-water rights density with canal-delivery boost, replacing global statistical datasets with locally observable infrastructure records
- **Surface Water Capture Index** — novel per-pixel, per-year quantification of pumping-induced streamflow depletion based on water table depth and canal infrastructure, with physics-based uncertainty bounds
- **Statewide irrigation consumptive use, 1896–2099** — to our knowledge, no publicly available dataset reports irrigation CU over Arizona at this combination of spatial resolution (2 km), temporal coverage (204 years, hindcast through projection), and per-well/per-basin/per-sub-basin/per-pixel disaggregation. The closest existing product is the USGS NHM HUC12 monthly irrigation CU reanalysis ([Martin et al., 2025](https://doi.org/10.1016/j.jhydrol.2025.133909); [Haynes et al., 2023](https://doi.org/10.5066/P9LGISUM)), which is national in scope but limited to 2000–2020 at HUC12 monthly resolution. ADWR publishes statewide withdrawal totals and an irrigation share but does not produce a basin-resolved irrigation CU product. AZ-Hydro provides annual irrigation CU at 2 km resolution for every year from 1896 through 2099, with per-pixel uncertainty bounds via physics-based error propagation (`σ_CU = IE × σ_Withdrawal`), separate GW-CU and SW-CU components consistent with the partitioning, and per-well CU disaggregation for ~170,000 individual wells via the well package. Statewide irrigation CU is reported here for the first time as 0.03 MAF (1900) → 2.10 MAF (2017 peak) → ~2.13 MAF (2099 projection), with the limitations on the projected trajectories and the irrigation efficiency paradox documented in the methods Limitations subsection
- **Trained inside AMA/INAs, predicting statewide** — the ML model is trained *only* on metered ADWR records from the ten AMA/INA management areas (Phoenix, Pinal, Tucson, Prescott, Santa Cruz, Douglas, Willcox AMAs + Joseph City, Harquahala, Hualapai Valley INAs), which are the basins with mandatory metering. The eight legacy AMA/INAs (Phoenix, Pinal, Tucson, Prescott, Santa Cruz, Douglas, Joseph City, Harquahala) provide most of the training signal because they have been metered continuously since 1984; **Willcox AMA and Hualapai Valley INA were designated only recently and are sparsely metered both in time (records concentrated in the last few years) and in space (fewer reporting wells per pixel)**, so they contribute much less training data than the legacy areas. At prediction time the model is applied to **every 2 km pixel in Arizona**, including the ~25 unmetered "Other" basins (basin type 2) — Yuma, Lower Gila, Parker, Lake Havasu, Bill Williams, Butler Valley, Mogollon plateau basins, and others. The model has *never* seen a single labeled withdrawal record from any of those unmetered basins. All headline statewide totals, the SW capture index headline numbers for the river-corridor basins, and the agency reconciliation comparisons therefore depend on out-of-distribution transfer from the metered AMA/INAs to morphologically similar but completely unlabeled regions
- **Full water-budget closure** — model captures ~68% of Arizona's 7.0 MAF total; the remaining ~32% (CAP, SRP, Yuma federal diversions, reclaimed water) is independently accounted for. Adding the constant ~2.26 MAF federal-delivery offset to the model's 4.74 MAF (2017) gives 6.99 MAF, closing to within 0.01 MAF of ADWR's reported 7.0 MAF. **Crucially, the unmetered Other-basin contribution to that 4.74 MAF is itself an out-of-distribution prediction** — the model has never seen labeled pumping data from those basins, yet the statewide aggregate matches an independent agency total within 0.2 %. The σ_total interval on the model side is ±0.66 MAF (≈14% CV), so 7.0 MAF lands well inside the federal-adjusted 95% CI of 5.71–8.29 MAF — the very tight central closure is a feature of the constant offset, while the honest precision of the underlying ML prediction is the σ_total interval
- **Multi-source emergent validation — four independent agency comparisons within ~1 pp or 0.1 MAF, no calibration to any of them** — the model agrees with four independent agency-reported numbers despite never seeing any of them during training and despite ~30 % of the predicted statewide volume coming from basins (the ~25 unmetered Other basins) for which no per-well training labels exist anywhere: (1) **2017 ADWR reconciled total** ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)): model + 2.26 MAF federal offset = 6.99 MAF vs ADWR's 7.0 MAF (central closure within **0.01 MAF**, model 95 % CI 5.71–8.29 MAF brackets both); (2) **2015 USGS GW pumping** ([Dieter et al. 2018](https://doi.org/10.3133/cir1441); Arizona summary in [NGWA 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf)): model 3.17 MAF vs USGS 3.09 MAF (central closure within **0.08 MAF**, model 95 % CI 1.90–4.44 MAF); (3) **2015 USGS GW share** ([Dieter et al. 2018](https://doi.org/10.3133/cir1441) / [NGWA 2020](https://www.ngwa.org/docs/default-source/default-document-library/states/az.pdf)): model **45.9 % vs 46 % (within 0.1 percentage point)**; (4) **2019 ADWR irrigation share** ([MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector)): model **72.8 % vs ~72 % (within ~0.8 percentage point)**. The 2019 ADWR statewide GW share comparison (model 44 % vs ADWR 41 %, [MAP Arizona Dashboard](https://mapazdashboard.arizona.edu/article/arizonas-water-use-sector), within 3 pp) is the loosest of the agency cross-checks and is reported for completeness. The capture index independently reproduces the same SW–GW interaction zones identified qualitatively by [Majumdar et al. (2022)](https://doi.org/10.1002/hyp.14757) using different methodology, cross-validating both studies. The convergence of independent datasets (ADWR wells, [HarDWR](https://doi.org/10.57931/2475303) rights, USGS gauges, [Ma et al.](https://doi.org/10.1038/s43247-025-03094-3) WTD, [GRAIN](https://doi.org/10.5194/essd-18-1855-2026) canals) in a physics-constrained framework provides a unified, self-consistent picture of Arizona's water system
- **Hybrid uncertainty quantification** — five-component σ_total via quadrature with physics-based CU error propagation, producing 6-band augmented rasters for every product
- **Multi-scenario projections** — 5 GCMs × 2 RCPs × 4 LULC scenarios × 112 streamflow ensemble members, with pixel-level uncertainty bounds
- **Well-level disaggregation** — ~170,000 individual wells with per-well withdrawal, CU, capture index, and uncertainty in 4 units via GeoParquet
- **ADWR-ready deliverables** — per-basin and per-sub-basin volume time series, well-level GeoPackage, and fully reproducible open-source code for Senate Bill 1740 basin assessments

## Getting Started

See [azhydro/README.md](azhydro/README.md) for installation instructions (conda environment, GEE authentication) and detailed documentation of the ML pipeline steps, configuration constants, library modules, and output directory structure.

See [gee/README.md](gee/README.md) for documentation of the Google Earth Engine export scripts used to generate the predictor data layers.

## Project Structure

```
az-hydro/
├── README.md                        # This file
├── DISCLAIMER.md                    # Provisional software disclaimer
├── LICENSE
├── environment.yml                  # Conda environment specification
├── ruff.toml                        # Ruff linter configuration
│
├── azhydro/                         # ML pipeline package
│   ├── README.md                    # Methods, CLI usage, and Results documentation
│   ├── pipeline.py                  # Main entry point (CLI + step orchestration)
│   └── hydrolibs/                   # Core library modules
│       ├── __init__.py
│       ├── dataops.py               # GEE download, data prep, ML DataFrame assembly
│       ├── gwops.py                 # Groundwater CSV processing, land-use smoothing
│       ├── intercompops.py          # USGS/NHM/Reitz intercomparison & validation
│       ├── mlops.py                 # Model training, tuning (Optuna/Dask), evaluation
│       ├── partitionops.py          # Withdrawal partitioning by category
│       ├── rasterops.py             # Raster I/O, mosaicking, reprojection utilities
│       ├── streamflowops.py         # USGS streamflow retrieval & rasterization
│       ├── sysops.py                # File-system helpers, directory creation
│       ├── uncertaintyops.py        # 5-component hybrid uncertainty quantification
│       ├── vectorops.py             # Vector reprojection, fishnet creation
│       ├── visualops.py             # Journal-quality time-series & map plotting
│       └── wellops.py               # Well-level disaggregation from pixel rasters
│
├── gee/                             # Google Earth Engine export scripts
│   ├── README.md                    # GEE script documentation
│   ├── config.py                    # Shared GEE constants (bands, models, scales)
│   ├── run_all_exports.py           # CLI to batch-run all export scripts
│   ├── plot_monthly_ratios.py       # Diagnostic plots for monthly ET/ETo ratios
│   ├── export_gridmet_hargreaves_ratio.py
│   ├── export_lulc_ensemble.py
│   ├── export_maca_gcm_annual_et.py
│   ├── export_maca_gcm_annual_eto.py
│   ├── export_maca_gcm_annual_peff.py
│   ├── export_maca_monthly_et.py
│   ├── export_maca_monthly_eto.py
│   ├── export_monthly_etof.py
│   ├── export_monthly_peff.py
│   ├── export_openet_reitz_ratio.py
│   ├── export_prism_hargreaves_eto.py
│   ├── export_usgs_adjusted_et.py
│   └── js/                          # GEE Code Editor visualization scripts
│
├── paper/                           # Paper drafts (Markdown)
│   └── data_paper.md                # Sci Data manuscript draft
│
├── tests/                           # Unit tests
│   ├── conftest.py                  # Shared fixtures
│   └── test_core.py                 # Core pipeline tests
│
└── docs/
    └── images/                      # Logo images and graphical abstract
```

### Input data

The `Data/` folder is **not** included in the Git repository. Download it from the [Zenodo archive](https://doi.org/10.5281/zenodo.19057936) and place it at the repository root so that the path `Data/Inputs/` exists:

```bash
# After cloning the repository
cd az-hydro
# Download and extract the Data folder from Zenodo
# https://doi.org/10.5281/zenodo.19057936
```

### Disk space requirements

A full setup with input data and a complete pipeline run requires approximately **37 GB**:

| Component | Size | Notes |
|---|---|---|
| **Inputs total (Zenodo)** | ~14 GB | Downloaded separately from the Git repository |
| &emsp;GEE tiles | ~1 GB | Raw 2 km tiles (80 km × 80 km each) |
| &emsp;GW data | ~2.5 GB | ADWR metered records, shapefiles, well registry, ancillary vectors |
| &emsp;USGS water-use data | ~11 GB | NHM withdrawals/CU/IE, Reitz 800 m rasters, crop surveys |
| **Outputs (generated)** | ~16 GB | |
| &emsp;GW rasters & vectors | ~14 GB | Observed withdrawal depth rasters + per-year vector shapefiles |
| &emsp;Reprojected vectors | ~900 MB | Basins, wells, CAP, streamflow in consistent CRS |
| &emsp;ML model outputs | ~500 MB+ | Evaluation, predictions, intercomparisons (grows with full run) |
| **Code & figures** | ~2 MB | Python source, GEE scripts, readme figures |

Disk usage will increase if additional model configurations or prediction years are added.

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Historical and projected groundwater/surface-water withdrawals, irrigation consumptive use, and pumping-induced surface water capture for Arizona, 1896–2099. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Declining Agricultural Dominance and Rising Urban Demand Drive a Two-Century Shift in Withdrawal Patterns (1896–2099). _In prep. for AGU Earth's Future_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). AZ-Hydro — Historical and Projected Arizona Annual Water Use: Software, Input Data, Models, Raster and Well GeoPackage Predictions, and Validation at 2 km Resolution (1896–2099). _Zenodo_. https://doi.org/10.5281/zenodo.19057936.

## Acknowledgments
This work was supported by NASA (Grant numbers 80NSSC21K0979 and 80NSSC23K1453) and U.S. Army Corps of Engineers (Grant number W912HZ25C0016). We thank the open-source software and data communities, the OpenET consortium, and the Arizona Department of Water Resources for making their resources and datasets publicly available, and Google Earth Engine for compute and storage support. S.M. and P.R. acknowledge Dr. Justin L. Huntington, Christopher Pearson, Charles G. Morton, Blake A. Minor, Dr. Samapriya Roy at the Desert Research Institute, and Dr. David Ketchum at the University of Montana for their contributions to related projects that informed this work. We also thank Rahel Pommerenke at Colorado State University for presenting preliminary results from this work at the 2025 ESA Living Planet Symposium. The views expressed herein are those of the authors and do not necessarily reflect those of the funding agencies.


<img src="docs/images/DRITaglineLogoTransparentBackground.png" height="60"/> &nbsp; <img src="docs/images/CSU-Signature-C-357.png" height="60"/> &nbsp; <img src="docs/images/ADWR.png" height="60"/> &nbsp; <img src="docs/images/nasa-logo-web-rgb.png" height="60"/> &nbsp; <img src="docs/images/USACE_logo_USACE_RW_line.png" height="60"/>

## AI Usage Disclosure

Portions of this codebase were developed with the assistance of **Claude Code**
(Anthropic, Claude Opus 4.6), an AI-powered coding assistant. The AI was used
for:

- **Code generation and refactoring** — implementing pipeline steps,
  visualization functions, intercomparison workflows, uncertainty
  quantification, and trend analysis routines.
- **Code review and cleanup** — identifying dead code, fixing bugs, improving
  code quality, and resolving linter warnings.
- **Documentation** — drafting and updating this README, docstrings, and inline
  comments.

All AI-generated code was reviewed, tested, and validated by the authors. The
scientific methodology, research design, data interpretation, and manuscript
writing remain entirely the responsibility of the authors.

## References

Alzraiee, A., Niswonger, R., Luukkonen, C., Larsen, J., Martin, D., Herbert, D., Buchwald, C., Dieter, C., Miller, L., Stewart, J., Houston, N., Paulinski, S., & Valseth, K. (2024). Next Generation Public Supply Water Withdrawal Estimation for the Conterminous United States Using Machine Learning and Operational Frameworks. _Water Resources Research_, _60_(7). https://doi.org/10.1029/2023WR036632

Hasan, M. F., Smith, R. G., Majumdar, S., Huntington, J. L., Alves Meira Neto, A., & Minor, B. A. (2025). Satellite data and physics-constrained machine learning for estimating effective precipitation in the Western United States and application for monitoring groundwater irrigation. _Agricultural Water Management_, _319_, 109821. https://doi.org/10.1016/j.agwat.2025.109821.

Haynes, J. V., Read, A. L., Chan, A. Y., Martin, D. J., Regan, R. S., Henson, W. R., Niswonger, R. G., & Stewart, J. S. (2023). Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000–2020 within the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_. https://doi.org/10.5066/P9LGISUM.

Lisk, M. D., Grogan, D. S., Proctor, K. L., Naz, B. S., Farmer, W. H., & Bock, A. R. (2024). HarDWR — Harmonized Database of Western U.S. Water Rights (v2.0). _Zenodo_. https://doi.org/10.57931/2475303.

Luukkonen, C.L., Alzraiee, A.H., Larsen, J.D., Martin, D.J., Herbert, D.M., Buchwald, C.A., Houston, N.A., Valseth, K.J., Paulinski, S., Miller, L.D., Niswonger, R.G., Stewart, J.S., & Dieter, C.A. (2023). Public supply water use reanalysis for the 2000-2020 period by HUC12, month, and year for the conterminous United States. _U.S. Geological Survey data release_. https://doi.org/10.5066/P9FUL880

Ma, Y., Condon, L. E., Koch, J., Bennett, A., Defnet, A., Tijerina-Kreuzer, D., Melchior, P., & Maxwell, R. M. (2026). High resolution US water table depth estimates reveal quantity of accessible groundwater. _Communications Earth & Environment_, _7_(1), 45. https://doi.org/10.1038/s43247-025-03094-3.

Majumdar, S., Smith, R., Butler, J. J., & Lakshmi, V. (2020). Groundwater withdrawal prediction using integrated multitemporal remote sensing data sets and machine learning. _Water Resources Research_, _56_(11), e2020WR028059. https://doi.org/10.1029/2020WR028059.

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. _Hydrological Processes, 36_(11), e14757. https://doi.org/10.1002/hyp.14757.

Majumdar, S., Smith, R. G., Hasan, M. F., Wilson, J. L., White, V. E., Bristow, E. L., Rigby, J. R., Kress, W. H., & Painter, J. A. (2024). Improving crop-specific groundwater use estimation in the Mississippi Alluvial Plain: Implications for integrated remote sensing and machine learning approaches in data-scarce regions. _Journal of Hydrology: Regional Studies_, _52_, 101674. https://doi.org/10.1016/j.ejrh.2024.101674.

Majumdar, S., Smith, R. G., & Hasan, M. F. (2025). A High-Resolution Data-Driven Monthly Aquaculture and Irrigation Water Use Model in the Mississippi Alluvial Plain. _IGARSS 2025 — 2025 IEEE International Geoscience and Remote Sensing Symposium_, 2686–2691. https://doi.org/10.1109/IGARSS55030.2025.11243173.

Martin, D. J., Regan, R. S., Haynes, J. V., Read, A. L., Henson, W. R., Stewart, J. S., Brandt, J. T., & Niswonger, R. G. (2023). Irrigation water use reanalysis for the 2000–20 period by HUC12, month, and year for the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_. https://doi.org/10.5066/P9YWR0OJ.

Martin, D. J., Niswonger, R. G., Regan, R. S., Huntington, J. L., Ott, T., Morton, C., Senay, G. B., Friedrichs, M., Melton, F. S., Haynes, J., Henson, W., Read, A., Xie, Y., Lark, T., & Rush, M. (2025). Estimating irrigation consumptive use for the conterminous United States: coupling satellite-sourced estimates of actual evapotranspiration with a national hydrologic model. _Journal of Hydrology_, _662_, 133909. https://doi.org/10.1016/j.jhydrol.2025.133909.

Ott, T. J., Majumdar, S., Huntington, J. L., Pearson, C., Bromley, M., Minor, B. A., ReVelle, P., Morton, C. G., Sueki, S., Beamer, J. P., & Jasoni, R. L. (2024). Toward field-scale groundwater pumping and improved groundwater management using remote sensing and climate data. _Agricultural Water Management_, _302_, 109000. https://doi.org/10.1016/j.agwat.2024.109000.

Reitz, M., Sanford, W. E., & Saxe, S. (2023a). Ensemble Estimation of Historical Evapotranspiration for the Conterminous U.S. _Water Resources Research_, _59_(6). https://doi.org/10.1029/2022WR034012.

Reitz, M., Sanford, W. E., & Saxe, S. (2023b). Historical evapotranspiration for the conterminous U.S. _U.S. Geological Survey Data Release_. https://doi.org/10.5066/P9EZ3VAS.

Suresh, S., Hossain, F., Mishra, V., & Hossain, N. (2026). GRAIN — a Global Registry of Agricultural Irrigation Networks. _Earth System Science Data_, _18_(3), 1855–1875. https://doi.org/10.5194/essd-18-1855-2026.
