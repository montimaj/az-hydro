# Improving remote sensing and machine learning-driven groundwater withdrawal estimation in Arizona 
Maintainer: [Dr. Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu]

## Abstract

AZ-Hydro is an end-to-end pipeline for estimating annual groundwater pumping, consumptive use, effective precipitation, and irrigation efficiencies across Arizona at 2 km resolution from 1896 to 2099. The pipeline fuses 14 bands of satellite-derived and climate-model-projected predictor data—including evapotranspiration, reference ET, precipitation, effective precipitation, temperature, land use/land cover, irrigated fraction, groundwater fraction, soil properties, streamflow, canal density, and well density—into a spatially explicit predictor stack via Google Earth Engine. An XGBoost model, tuned with Optuna TPE hyperparameter search (100 trials, 5-fold CV) and parallelized with Dask, is trained on metered Arizona Department of Water Resources (ADWR) pumping records (1984–2024). Model performance is evaluated using random, temporal leave-one-out (six configurations), and spatial leave-one-out strategies across eight ensemble tree algorithms. Full-period predictions are partitioned into eight withdrawal categories (Irrigation, Non-Irrigation, and their GW/SW splits), three consumptive-use products, and three irrigation-efficiency products. A hybrid uncertainty quantification framework combines five independent error components (inter-GCM climate spread, model seed ensemble, irrigation fraction sensitivity, LULC projection spread, and GW fraction snapshot spread) via quadrature to produce 6-band augmented GeoTIFFs (prediction, σ, CV, SNR, lower 95 % CI, upper 95 % CI) for every product and unit. All time-series plots—AZ-wide, per-basin, and per-sub-basin—are generated with 95 % confidence interval bounds derived from zonal statistics on these augmented rasters. A well-level package disaggregates pixel-level rasters to individual ADWR wells using capacity-proportional weighting. Predictions are independently validated against USGS National Hydrologic Model (NHM) HUC12 withdrawals, consumptive use, irrigation efficiencies, and effective precipitation, as well as USGS Reitz 800 m irrigation water-use rasters, aggregated to ADWR groundwater basin totals.

## Getting Started

See [azhydro/README.md](azhydro/README.md) for installation instructions (conda environment, GEE authentication) and detailed documentation of the ML pipeline steps, configuration constants, library modules, and output directory structure.

See [gee/README.md](gee/README.md) for documentation of the Google Earth Engine export scripts used to generate the predictor data layers.

## Project Structure

```
az-hydro/
├── README.md                        # This file
├── LICENSE
├── environment.yml                  # Conda environment specification
├── ruff.toml                        # Ruff linter configuration
│
├── azhydro/                         # ML pipeline package
│   ├── README.md                    # Installation & CLI usage docs
│   ├── pipeline.py                  # Main entry point (CLI + step orchestration)
│   └── hydrolibs/                   # Core library modules
│       ├── __init__.py
│       ├── dataops.py               # GEE download, data prep, ML DataFrame assembly
│       ├── gwops.py                 # Groundwater CSV processing, land-use smoothing
│       ├── intercompops.py          # USGS/NHM/Reitz intercomparison & validation
│       ├── mlops.py                 # Model training, tuning (Optuna/Dask), evaluation
│       ├── partitionops.py          # Withdrawal/CU/IE partitioning by category
│       ├── rasterops.py             # Raster I/O, mosaicking, reprojection utilities
│       ├── streamflowops.py         # USGS streamflow retrieval & rasterisation
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
│   └── export_usgs_adjusted_et.py
│
├── tests/                           # Unit tests
│   ├── conftest.py                  # Shared fixtures
│   └── test_core.py                 # Core pipeline tests
│
├── Data/                            # Input data (not tracked in full by git)
│   └── Inputs/
│       ├── GEE_Data/                # Downloaded GEE tiles & HUC12 polygons
│       ├── GW_Data/                 # ADWR records, shapefiles, ancillary vectors
│       └── USGS WU/                 # NHM withdrawals, Reitz rasters, crop surveys
│
└── docs/
    └── images/                   # Logo images for README
```

### Repository disk space requirements

A full clone with all input data and a complete pipeline run requires approximately **37 GB**:

| Component | Size | Notes |
|---|---|---|
| **Inputs total** | ~14 GB | |
| &emsp;GEE tiles | ~1 GB | Raw 2 km tiles (80 km × 80 km each) |
| &emsp;GW data | ~2.5 GB | ADWR metered records, shapefiles, well registry, ancillary vectors |
| &emsp;USGS water-use data | ~11 GB | NHM withdrawals/CU/IE, Reitz 800 m rasters, crop surveys |
| **Outputs total** | ~16 GB | |
| &emsp;GW rasters & vectors | ~14 GB | Observed pumping depth rasters + per-year vector shapefiles |
| &emsp;Reprojected vectors | ~900 MB | Basins, wells, CAP, streamflow in consistent CRS |
| &emsp;ML model outputs | ~500 MB+ | Evaluation, predictions, intercomparisons (grows with full run) |
| **Code & figures** | ~2 MB | Python source, GEE scripts, readme figures |

Disk usage will increase if additional model configurations or prediction years are added.

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). The Arizona Water Use Dataset (1896–2099): Withdrawals, consumptive use, and irrigation efficiency partitioned by source. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Two Centuries of Groundwater and Surface Water Withdrawals, Consumptive Use, and Irrigation Efficiency (1896–2099). _In prep. for AGU Earth's Future_.

## Acknowledgments
We would like to acknowledge funding from NASA (Grant numbers: 80NSSC21K0979 and 80NSSC23K1453). We are grateful to all the opensource software and data communities for making their resources publicly available and also thank the ADWR for providing the necessary data sets related to groundwater withdrawals and other shapefiles used in this research. We also acknowledge compute and storage support from Google Earth Engine. Finally, we would like to convey our gratitude to our colleagues and families for their continuous motivation and support. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies.

<img src="docs/images/DRITaglineLogoTransparentBackground.png" height="100"/> &nbsp;  &nbsp; <img src="docs/images/CSU-Signature-C-357.png" height="130"/> &nbsp; <img src="docs/images/ADWR.png" height="120"/> &nbsp;  &nbsp; <img src="docs/images/nasa-logo-web-rgb.png" height="120"/>

## References

Martin, D. J., Niswonger, R. G., Regan, R. S., Huntington, J. L., Ott, T., Morton, C., Senay, G. B., Friedrichs, M., Melton, F. S., Haynes, J., Henson, W., Read, A., Xie, Y., Lark, T., & Rush, M. (2025). Estimating irrigation consumptive use for the conterminous United States: coupling satellite-sourced estimates of actual evapotranspiration with a national hydrologic model. _Journal of Hydrology_, _662_, 133909. https://doi.org/10.1016/j.jhydrol.2025.133909

Haynes, J.V., Read, A.L., Chan, A.Y., Martin, D.J., Regan, R.S., Henson, W.R., Niswonger, R.G., & Stewart, J.S., 2023, Monthly crop irrigation withdrawals and efficiencies by HUC12 watershed for years 2000-2020 within the conterminous United States (ver. 2.0, September 2024). _U.S. Geological Survey data release_, https://doi.org/10.5066/P9LGISUM.

Reitz, M., Sanford, W. E., & Saxe, S. (2023). Ensemble Estimation of Historical Evapotranspiration for the Conterminous U.S. _Water Resources Research_, _59_(6). https://doi.org/10.1029/2022WR034012

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. _Hydrological Processes, 36_(11), e14757. https://doi.org/10.1002/hyp.14757
