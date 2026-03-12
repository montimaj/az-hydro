# AZ-Hydro

Maintainers: [Sayantan Majumdar](https://www.dri.edu/directory/sayantan-majumdar/) [sayantan.majumdar@dri.edu], [Ryan G. Smith](https://www.engr.colostate.edu/ce/ryan-g-smith/) [ryan.g.smith@colostate.edu]

<img src="../Readme_Figures/DRITaglineLogoTransparentBackground.png" height="45"/> &nbsp; <img src="../Readme_Figures/CSU-Signature-C-357.png" height="55"/> 


Note: This software has been successfully tested on [Alienware M17R1 2020](https://www.dell.com/en-us/gaming/alienware) (Windows 10 Home) and the [Apple MacBook Pro 2023](https://www.apple.com/macbook-pro/) (macOS Sonoma 14.3.1).

## Citations
Majumdar, S., Smith, R.G., Hasan, M.F., Wogenstahl, C., & Conway, B.D. (2025). A long-term database of groundwater pumping, consumptive use, effective precipitation, and irrigation efficiencies in Arizona derived from remote sensing and machine learning. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. _Hydrological Processes, 36_(11), e14757. https://doi.org/10.1002/hyp.14757


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

#### Repository disk space requirements
//TODO

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
//TODO

## Library modules (`hydrolibs/`)

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