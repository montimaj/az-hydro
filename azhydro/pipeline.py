"""
ML Pipeline Script for Arizona Annual Withdrawal Prediction.

This script executes the remaining pipeline:
1. Creates dummy annual predictor data from 1896-2099 for AZ and assigns each
   pixel an ADWR groundwater sub-basin label (``GW_Subbasin``).
2. Evaluates standard and physics-informed tree-based ML models (1984-2024) on
   four splitting strategies:
   a) Random 80/20 train/test split.
   a2) Pixel holdout — 20% of unique spatial locations held out across all years.
   b) Leave-one-out temporal holdout over multiple test-year ranges (T1-T7),
      reporting per-holdout and averaged metrics.
   c) Leave-one-out spatial holdout over every AMA/INA management area,
      reporting per-basin and averaged metrics.
   All strategies use kFolds + Optuna (TPE) + Dask parallelization.
   Optional physics-informed models (PIML_XGB, PIML_LGBM, PIML_XGBRF) are
   available but disabled by default (SKIP_PIML=True) — see azhydro/README.md.
3. Uses the best model (XGBoost Random Forests) to predict annual
   pumping rasters from 1896-2099 with maps and time series highlighting three
   eras: Hindcast (1896-1983), Historical (1984-2025), Projection (2026-2099).
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import argparse

import logging
import os
import pickle
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning, module='ee.*')
warnings.filterwarnings('ignore', message='.*deprecated asset.*')
warnings.filterwarnings('ignore', message='.*Attention required.*')

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydrolibs.dataops as dataops
import hydrolibs.gwops as gwops
import hydrolibs.intercompops as intercompops
import hydrolibs.mlops as mlops
import hydrolibs.partitionops as partops
import hydrolibs.streamflowops as streamflowops
import hydrolibs.uncertaintyops as uncops
import hydrolibs.visualops as vizops
import hydrolibs.wellops as wellops
from hydrolibs.rasterops import read_raster_as_arr, write_raster
from hydrolibs.sysops import makedirs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
INPUT_DIR = '../Data/Inputs/'
OUTPUT_DIR = '../Data/Outputs/'

WATER_USE = 'All'
WNAME = 'All_Wells' if WATER_USE == 'All' else 'Irr_Wells'
VECTOR_DIR = os.path.join(INPUT_DIR, 'GW_Data')
MOSAIC_RASTER_RES = 2000
GEE_MOSAIC_DIR = os.path.join(OUTPUT_DIR, f'GEE_Mosaics_{int(MOSAIC_RASTER_RES)}m')
GW_DEPTH_RASTER_DIR = os.path.join(OUTPUT_DIR, f'GW/Rasters/GW_Depths_{WNAME}_{int(MOSAIC_RASTER_RES)}m')
PRED_DATA_DIR = os.path.join(OUTPUT_DIR, f'Predictor_Data_{WNAME}_{int(MOSAIC_RASTER_RES)}m')
MODEL_DIR = os.path.join(OUTPUT_DIR, f'ML_Model_{WNAME}_{int(MOSAIC_RASTER_RES)}m')
GW_CROPPED_RASTER_DIR = os.path.join(GW_DEPTH_RASTER_DIR, 'GW_Cropped')

AZ_GW_BASIN = os.path.join(OUTPUT_DIR, 'GW_Data', 'Vector_Reproj', 'Groundwater_Basin.shp')
ADWR_SUBBASIN_SHP = os.path.join(OUTPUT_DIR, 'GW_Data', 'Vector_Reproj', 'ADWR_Groundwater_Subbasin.shp')

GCLOUD_PROJECT = 'azhydro'
GCLOUD_BUCKET = 'azhydro'
TILE_SIZE = 80000
FILL_ATTR = 'AF Pumped'
AF_MAX_THRESHOLD = 5000.  # max per-well AF; ~3,000 gpm sustained year-round
MIN_GW = None    # min per-pixel GW pumping depth (mm); pixels below this are excluded
MAX_GW = 3000  # max per-pixel GW pumping depth (mm); pixels above this are excluded
LOG_TARGET = False  # log1p-transform target; metrics reported on original scale via expm1

# AMA_CODE → parent AMA/INA name mapping
AMA_CODE_MAP = {
    'A': 'JOSEPH CITY INA',
    'B': 'PRESCOTT AMA',
    'C': 'PHOENIX AMA',
    'D': 'PINAL AMA',
    'E': 'TUCSON AMA',
    'F': 'DOUGLAS AMA',
    'G': 'HARQUAHALA INA',
    'H': 'SANTA CRUZ AMA',
}

START_YEAR = 1896
END_YEAR = 2099
YEAR_LIST = list(range(1984, 2025))
TRAIN_YEAR_LIST_BASELINE = list(range(2002, 2021))  # for direct comparison with Majumdar et al., 2022
RANDOM_STATE = 42
N_EVAL_SEEDS = 5
EVAL_TEST_SIZES = (0.10, 0.15, 0.20, 0.25, 0.30)
N_TRIALS = 50
FOLD_COUNT = 5
REPEATS = 1
N_DASK_WORKERS = 10
N_DASK_WORKERS_DATA_PREP = 40 # more workers for data prep since it involves many independent raster operations
USE_OPTUNA = True
USE_DASK = True
INCLUDE_ALL_MODELS = True
SKIP_PIML = True
PHYSICS_INTERACTION_CONSTRAINTS = False
PREDICTION_MODEL = 'XGBRF'  # Model used for full-period prediction (Step 3+)

USE_AMA_INA = True
DROP_GW_BASINS = ()
MIN_SPATIAL_EVAL_SAMPLES = 30   # skip sub-basins with fewer non-zero metered samples
SKIP_SPATIAL_BASINS = ('WILLCOX AMA',)  # basins to exclude from spatial LOO (too few samples)
SPATIAL_SEED_FRACTION = 0.1     # fraction of held-out basin samples seeded into training

DROP_ATTRS = (
    'Year',
    'GW_Basin',
    'GW_Subbasin',
    'SW',
    'GW_Basin_Type',
    'annual_peff_pcml_mm',
    'lulc',  # raw LULC class kept for partitionops fallback; AGRI/URBAN/SW provide
             # Gaussian-smoothed land-use signal, annual_urban_fraction and
             # annual_crop_fraction provide physical densities
    'irr_capacity_fraction',  # pump-capacity-weighted irrigation fraction for
                              # partitioning only, not an ML feature
    'crop_frac_ref',   # 2024 reference crop fraction for temporal capacity scaling
    'urban_frac_ref',  # 2024 reference urban fraction for temporal capacity scaling
    'sw_access_year',  # earliest irrigation SW priority year (Part A)
    'irr_sw_rights_density',  # intermediate; combined into sw_rights_density
    'nonirr_sw_rights_density',  # non-irr SW rights density (Part B)
    'irr_well_density',       # irrigation well count density (partitioning only)
    'nonirr_well_density',    # non-irrigation well count density (partitioning only)
    'canal_weighted_streamflow_mm',  # used for partitioning only; it has extreme values in the hindcast period
)

# # Temporal holdout configurations
TEMPORAL_HOLDOUTS = {
    'T1_Baseline': ((2010, 2020),), # same from Majumdar et al., 2022
    'T1': ((2010, 2020),),
    'T2': ((2015, 2024),), 
    'T3': ((1990, 1992), (2005, 2007), (2022, 2024)),
    'T4': ((2007, 2010),),
    'T5': ((1985, 1989), (2020, 2024)),
    'T6': ((2024, 2024),),
    'T7': ((1984, 1994),),  # early-period holdout — backward extrapolation
}

# =============================================================================
# Step 0 — Data preparation (GEE download, GW processing, rasterisation)
# =============================================================================

def prepare_data(
        skip_download: bool = True,
        load_files: bool = True,
        verbose: bool = False,
        skip_prep: set[str] | None = None,
) -> list[str]:
    """
    Download GEE data, preprocess GW CSVs, reproject vectors, and create
    all intermediate rasters needed by the ML pipeline.

    Args:
        skip_download (bool): If *True*, skip the GEE download (use existing tiles).
        load_files (bool): If *True*, skip recreating files that already exist on disk.
        verbose (bool): If *True*, enable verbose output for GEE downloads.
        skip_prep (set[str] or None): Sub-step names to skip: ``gee``, ``gw-csv``, ``vectors``,
            ``gw-rasters``, ``streamflow``, ``basin-rasters``, ``reproject``.

    Returns:
        list[str]: GEE data band names (needed by ``create_az_data``).
    """
    if skip_prep is None:
        skip_prep = set()
    logger.info('='*60)
    logger.info('Step 0: Data preparation')
    logger.info('='*60)

    az_state_raw = os.path.join(VECTOR_DIR, 'AZ.geojson')
    well_reg_file = os.path.join(VECTOR_DIR, 'Well_Registry_2024', 'Well_Registry.shp')
    gw_csv_dir = os.path.join(VECTOR_DIR, 'Meter Data')
    grain_parquet = os.path.join(VECTOR_DIR, 'GRAIN_v.1.0', 'GeoParquet', 'us-west_GRAIN_v.1.0.parquet')
    output_gw_vector_dir = os.path.join(OUTPUT_DIR, f'GW/Vectors/{WNAME}')
    vector_reproj_dir = os.path.join(OUTPUT_DIR, 'GW_Data/Vector_Reproj')
    output_gw_volume_dir = (
        os.path.join(OUTPUT_DIR, f'GW/Rasters/GW_Volumes_{WNAME}_{int(MOSAIC_RASTER_RES)}m')
    )

    # GEE download & mosaic
    skip_gee = load_files or 'gee' in skip_prep
    gee_data_dir, data_band_names = dataops.download_gee_data(
        az_state_raw,
        GCLOUD_PROJECT,
        GCLOUD_BUCKET,
        INPUT_DIR,
        START_YEAR,
        END_YEAR,
        skip_download or 'gee' in skip_prep,
        TILE_SIZE,
        num_workers=N_DASK_WORKERS_DATA_PREP,
        worker_memory='1G',
        gee_scale=MOSAIC_RASTER_RES,
        verbose=verbose,
    )
    dataops.mosaic_tiles(
        gee_data_dir,
        GEE_MOSAIC_DIR,
        START_YEAR,
        END_YEAR,
        already_mosaicked=skip_gee,
        fishnet_file=os.path.join(INPUT_DIR, 'GW_Data',
                                  f'AZ_Polygons_{TILE_SIZE}m.geojson'),
    )

    # GW CSV → per-year shapefiles
    skip_gw_csv = load_files or 'gw-csv' in skip_prep
    ref_gw_file = gwops.preprocess_gw_csv(
        well_reg_file,
        gw_csv_dir,
        output_gw_vector_dir,
        fill_attr=FILL_ATTR,
        use_only_ama_ina=False,
        already_preprocessed=skip_gw_csv,
        af_max_threshold=AF_MAX_THRESHOLD,
        water_use=WATER_USE,
    )

    # Reproject vectors
    az_vector_reproj = gwops.reproject_vectors(
        VECTOR_DIR,
        vector_reproj_dir,
        ref_file=ref_gw_file,
        already_reprojected=load_files or 'vectors' in skip_prep,
    )
    well_reg_file = az_vector_reproj['Well_Registry']
    az_gw_basin = az_vector_reproj['Groundwater_Basin']
    az_sw_watershed = az_vector_reproj['Surface_Watershed']
    cap_service_area = az_vector_reproj['CAP_Service_Area']
    az_state = az_vector_reproj['AZ']

    # GW volume → depth → cropped rasters
    skip_gw_rasters = load_files or 'gw-rasters' in skip_prep
    gwops.create_gw_volume_rasters(
        output_gw_vector_dir,
        output_gw_volume_dir,
        value_field=FILL_ATTR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        already_created=skip_gw_rasters
    )
    gwops.create_gw_depth_rasters(
        output_gw_volume_dir,
        GW_DEPTH_RASTER_DIR,
        already_created=skip_gw_rasters,
    )
    gwops.crop_gw_rasters(
        GW_DEPTH_RASTER_DIR,
        GW_DEPTH_RASTER_DIR,
        az_state_file=az_state,
        already_cropped=skip_gw_rasters,
    )

    # Canal density & streamflow rasters
    skip_streamflow = load_files or 'streamflow' in skip_prep
    pod_shapefile = os.path.join(
        VECTOR_DIR, 'Water Rights', 'stateWaterRightsHarmonized',
        'arizona', 'arizonaStatePOD.shp',
    )
    canal_density_file = streamflowops.create_canal_density_raster(
        grain_parquet=grain_parquet,
        az_boundary_file=az_state,
        output_dir=GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_streamflow,
        pod_shapefile=pod_shapefile,
        watershed_file=az_sw_watershed,
        basin_shp=AZ_GW_BASIN,
    )
    streamflowops.create_streamflow_rasters(
        watershed_geojson=az_sw_watershed,
        cap_service_area_geojson=cap_service_area,
        sites_csv=os.path.join(VECTOR_DIR, 'Streamflow', 'sites.csv'),
        output_dir=GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        canal_density_file=canal_density_file,
        already_created=skip_streamflow,
        verbose=verbose,
    )

    # GW basin, sub-basin & well density rasters
    skip_basin = load_files or 'basin-rasters' in skip_prep
    adwr_subbasin_shp = os.path.join(vector_reproj_dir, 'ADWR_Groundwater_Subbasin.shp')
    gwops.create_gw_basin_rasters(
        az_gw_basin,
        GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_basin,
        subbasin_vector=adwr_subbasin_shp,
    )
    gwops.create_well_density_raster(
        well_reg_file,
        GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_basin,
    )
    # Irrigation well density (wells with WATER_USE containing 'IRRIGATION')
    gwops.create_well_density_raster(
        well_reg_file,
        GEE_MOSAIC_DIR,
        water_use='IRRIGATION',
        output_prefix='Well_Density_Irrigation',
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_basin,
    )
    # Non-irrigation well density (consumptive wells WITHOUT 'IRRIGATION')
    gwops.create_well_density_raster(
        well_reg_file,
        GEE_MOSAIC_DIR,
        water_use='IRRIGATION',
        water_use_exclude=True,
        output_prefix='Well_Density_NonIrrigation',
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_basin,
    )
    gwops.create_irr_capacity_fraction_raster(
        well_reg_file,
        GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_basin,
    )

    # HarDWR v2.0 water-rights rasters
    skip_rights = load_files or 'rights-rasters' in skip_prep
    ref_raster = os.path.join(GEE_MOSAIC_DIR, f'GW_Basin_{START_YEAR}.tif')
    gwops.create_sw_access_year_raster(
        pod_shapefile, GEE_MOSAIC_DIR, ref_raster,
        already_created=skip_rights,
    )
    gwops.create_irr_sw_rights_density_raster(
        pod_shapefile, GEE_MOSAIC_DIR, ref_raster,
        start_year=START_YEAR, end_year=END_YEAR,
        already_created=skip_rights,
    )
    gwops.create_nonirr_sw_rights_density_raster(
        pod_shapefile, GEE_MOSAIC_DIR, ref_raster,
        start_year=START_YEAR, end_year=END_YEAR,
        already_created=skip_rights,
    )

    # Water table depth (Ma et al., 2026) — static raster
    wtd_dir = os.path.join(VECTOR_DIR, 'wtd_states')
    gwops.create_wtd_raster(
        wtd_dir=wtd_dir,
        az_boundary_file=az_state,
        output_dir=GEE_MOSAIC_DIR,
        ref_raster=ref_raster,
        already_created=load_files or 'wtd' in skip_prep,
    )

    # Reproject GEE mosaics to match GW raster grid
    dataops.reproject_gee_mosaics(
        GEE_MOSAIC_DIR,
        PRED_DATA_DIR,
        GW_CROPPED_RASTER_DIR,
        already_reprojected=load_files or 'reproject' in skip_prep,
    )

    logger.info('Step 0 complete.')
    return data_band_names


# =============================================================================
# Step 1 — Create AZ predictor data (1896-2099)
# =============================================================================


def create_az_data(
        data_band_names: list[str],
        load_files: bool = True,
        run_eda: bool = False,
) -> pd.DataFrame:
    """
    Build the AZ predictor dataframe for years START_YEAR to END_YEAR.

    Calls ``dataops.create_az_data_parquet`` which reads each year's
    Predictor, GW_Basin, GW_Subbasin, Streamflow,
    Canal_Weighted_Streamflow, Canal_Density, and Well_Density rasters,
    then maps ADWR sub-basin OBJECTIDs to names and optionally runs EDA.

    Args:
        data_band_names (list[str]): Band/layer names for predictor rasters.
        load_files (bool): If True, load from cached parquet files.
        run_eda (bool): If True, regenerate the EDA figures (histograms,
            ET-vs-ETo analysis, pumping-distribution analysis, per-basin
            data-availability summary). Defaults to False so downstream
            steps that reuse the predictor DataFrame (Step 2, Step 3,
            Step 3b) never repeat the ~minute-long EDA render unless
            explicitly asked via ``--run-eda``.

    Returns:
        pd.DataFrame: Combined predictor dataframe for the full study period.
    """
    logger.info('='*60)
    logger.info('Step 1: Creating AZ predictor data (1896-2099)...')
    logger.info('='*60)

    az_df = dataops.create_az_data_parquet(
        PRED_DATA_DIR,
        GW_CROPPED_RASTER_DIR,
        MODEL_DIR,
        data_band_names,
        AZ_GW_BASIN,
        start_year=START_YEAR,
        end_year=END_YEAR,
        load_parquet=load_files,
        subbasin_vector=ADWR_SUBBASIN_SHP,
    )
    logger.debug(f'AZ data shape: {az_df.shape}')
    logger.debug(f'Year range: {az_df.Year.min()} – {az_df.Year.max()}')
    logger.debug(f'Columns: {list(az_df.columns)}')

    # EDA (opt-in)
    if run_eda:
        vizops.explore_az_data(az_df, os.path.join(MODEL_DIR, 'EDA'))

        # ET vs ETo analysis by land use
        vizops.analyze_et_by_land_use(az_df, os.path.join(MODEL_DIR, 'EDA'))

        # Pumping distribution analysis (metered years only)
        vizops.analyze_pumping_distribution(
            az_df, os.path.join(MODEL_DIR, 'EDA'), YEAR_LIST, MAX_GW,
        )

        # Per-basin data availability summary
        gwops.generate_basin_data_summary(
            az_df, os.path.join(MODEL_DIR, 'EDA'), YEAR_LIST,
        )
    else:
        logger.info('EDA plots not generated (pass --run-eda to regenerate).')

    return az_df


# =============================================================================
# Step 2 — Evaluate tree-based ML models
# =============================================================================

# Module-level cache for NHM basin IE (loaded once, reused across evaluations)
_nhm_basin_ie_cache: dict | None = None


def _load_nhm_basin_ie_cached() -> dict:
    """Load NHM basin irrigation efficiencies (cached after first call)."""
    global _nhm_basin_ie_cache
    if _nhm_basin_ie_cache is not None:
        return _nhm_basin_ie_cache

    nhm_ie_csv = os.path.join(INPUT_DIR, 'USGS WU', 'USGS_NHM_Withdrawals',
                              'IR_HUC12_Eff_annual_2000_2020.csv')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')
    nhm_ie_out = os.path.join(MODEL_DIR, 'Model_Evaluation', 'NHM_IE_Basins')
    ref_raster_file = os.path.join(PRED_DATA_DIR, f'Predictor_{YEAR_LIST[0]}.tif')
    makedirs(nhm_ie_out)

    _nhm_basin_ie_cache = intercompops.load_nhm_basin_ie(
        nhm_ie_csv=nhm_ie_csv,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        ref_raster=ref_raster_file,
        output_dir=nhm_ie_out,
    )
    return _nhm_basin_ie_cache


def _append_physics_floor(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        basin_train: pd.DataFrame,
        basin_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute water-balance floor and append as a smuggled column.

    The floor is a well-density-weighted irrigation demand estimate
    (``__y_phys_log__``).  Per-pixel historical bounds (min for floor
    tightening, max for ceiling) are computed inside the PIML wrapper's
    ``fit()`` method from fold training data, giving Optuna real gradient
    signal for lambda tuning.

    The physics wrappers strip the smuggled column during fit()/predict()
    so it travels transparently through sklearn's ``cross_validate``.

    Returns:
        Updated (x_train, x_test) with floor column appended.
    """
    nhm_ie = _load_nhm_basin_ie_cached()
    gw_basin_col = 'GW_Basin'
    wd_col = 'well_density'

    # Compute well_density_p99 from non-zero training well_density
    wd_train = x_train[wd_col].values
    wd_test = x_test[wd_col].values
    nonzero_wd = wd_train[wd_train > 0]
    well_density_p99 = np.percentile(nonzero_wd, 99) if len(nonzero_wd) > 0 else 1.0

    # Water-balance floor (well-density-weighted)
    y_floor_train = mlops.compute_irrigation_demand_floor(
        x_train, basin_train[gw_basin_col], nhm_ie,
        well_density=wd_train, well_density_p99=well_density_p99,
    )
    y_floor_test = mlops.compute_irrigation_demand_floor(
        x_test, basin_test[gw_basin_col], nhm_ie,
        well_density=wd_test, well_density_p99=well_density_p99,
    )

    # Convert to log1p space if needed (matching LOG_TARGET)
    if LOG_TARGET:
        y_floor_train = np.log1p(y_floor_train)
        y_floor_test = np.log1p(y_floor_test)

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[mlops._PHYS_COL] = y_floor_train
    x_test[mlops._PHYS_COL] = y_floor_test

    n_floor = np.sum(y_floor_train > 0)
    logger.info(f'Physics floor: {n_floor}/{len(y_floor_train)} training samples '
                f'with non-zero water-balance floor '
                f'(p99 well_density={well_density_p99:.1f})')
    return x_train, x_test


# Thin aliases — implementations moved to mlops
_compute_metrics = mlops.compute_metrics
_metrics_from_pred_df = mlops.metrics_from_pred_df
_stratified_test_metrics = mlops.stratified_test_metrics


def _train_and_evaluate(
        x_train: pd.DataFrame, y_train: np.ndarray,
        x_test: pd.DataFrame, y_test: np.ndarray,
        model_name: str, output_dir: str,
        cv_groups: np.ndarray | pd.Series | None = None,
        n_trials: int = N_TRIALS,
) -> dict:
    """Train a single model with Optuna+Dask and return train/test metrics."""
    model, cv_df = mlops.build_ml_model_optuna(
        x_train, y_train, output_dir, model_name,
        random_state=RANDOM_STATE,
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=n_trials,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        cv_groups=cv_groups,
        log_target=LOG_TARGET,
    )
    # PIML wrappers strip physics columns internally; standard models need them removed
    if model_name not in mlops._PIML_MODELS:
        x_train_pred = mlops._drop_phys_cols(x_train)
        x_test_pred = mlops._drop_phys_cols(x_test)
    else:
        x_train_pred = x_train
        x_test_pred = x_test
    y_pred_train = model.predict(x_train_pred)
    y_pred_test = model.predict(x_test_pred)
    if LOG_TARGET:
        y_pred_train = np.expm1(y_pred_train)
        y_pred_test = np.expm1(y_pred_test)
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
    else:
        y_train_orig = y_train
        y_test_orig = y_test
    y_pred_train = np.abs(y_pred_train)
    y_pred_test = np.abs(y_pred_test)
    train_metrics = _compute_metrics(y_train_orig, y_pred_train)
    test_metrics = _compute_metrics(y_test_orig, y_pred_test)

    # Extract CV validation metrics
    cv_val_r2 = cv_val_rmse = float('nan')
    if not cv_df.empty and 'Data' in cv_df.columns:
        val_row = cv_df[cv_df['Data'] == 'VALIDATION']
        if not val_row.empty:
            cv_val_r2 = val_row['R2'].values[0]
            cv_val_rmse = val_row['RMSE (%)'].values[0]

    logger.info(f'    Train R2: {train_metrics["R2"]:.4f}, '
                f'Val R2: {cv_val_r2:.4f}, '
                f'Test R2: {test_metrics["R2"]:.4f}')
    logger.info(f'    Train RMSE: {train_metrics["RMSE_pct"]:.2f}%, '
                f'Val RMSE: {cv_val_rmse:.2f}%, '
                f'Test RMSE: {test_metrics["RMSE_pct"]:.2f}%')
    logger.info(f'    Overfit (R2 gap): {train_metrics["R2"] - test_metrics["R2"]:.4f}')

    return {
        'model': model,
        'cv_df': cv_df,
        'train': train_metrics,
        'test': test_metrics,
    }


# ---- 2a. Random 80/20 evaluation ------------------------------------------
def _evaluate_random_single(
        az_df: pd.DataFrame,
        random_state: int,
        strategy_dir: str,
        tuning_model_dir: str | None = None,
        test_size: float = 0.2,
        create_interp_plots: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Run a single random evaluation with the given seed and test size.

    Returns:
        Tuple of (comparison DataFrame, model comparison dir path).
    """
    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
        skip_piml=SKIP_PIML,
    )

    data_dir = os.path.join(strategy_dir, 'data')
    ret_vals = dataops.create_train_test_data(
        az_df, data_dir,
        drop_attr=DROP_ATTRS,
        random_state=random_state,
        scaling=False, already_created=False,
        year_list=YEAR_LIST, split_strategy=4,
        test_year=True,
        test_size=test_size,
        outlier_op=3,
        min_gw_pumping=MIN_GW if MIN_GW is not None else 1e-10,
        max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        log_target=LOG_TARGET,
    )
    (x_train, x_test, y_train, y_test,
     x_scaler, y_scaler,
     year_train, year_test,
     basin_train, basin_test,
     easting_train, easting_test,
     northing_train, northing_test) = ret_vals

    # Append physics floor column for PIML models
    if not SKIP_PIML:
        x_train, x_test = _append_physics_floor(
            x_train, x_test, basin_train, basin_test,
        )

    comparison_dir = os.path.join(strategy_dir, 'Model_Comparison')
    return mlops.compare_all_models(
        x_train, x_test, y_train, y_test,
        comparison_dir,
        model_names=ml_models,
        random_state=random_state,
        use_optuna=USE_OPTUNA,
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        year_train=year_train,
        year_test=year_test,
        basin_train=basin_train,
        basin_test=basin_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        use_ama_ina=USE_AMA_INA,
        apply_bias_correction=True,
        easting_train=easting_train,
        easting_test=easting_test,
        northing_train=northing_train,
        northing_test=northing_test,
        test_case='Random',
        raster_res=MOSAIC_RASTER_RES,
        create_basin_plots=True,
        log_target=LOG_TARGET,
        tuning_model_dir=tuning_model_dir,
        create_interp_plots=create_interp_plots,
        use_interaction_constraints=PHYSICS_INTERACTION_CONSTRAINTS,
    )


def evaluate_random(az_df: pd.DataFrame) -> dict:
    """Random split — 5 test sizes × 5 seeds grid evaluation.

    Optuna tuning runs once (first seed, first test size).  All other
    combinations reuse the tuned hyperparameters and only re-split,
    retrain, and evaluate.

    Args:
        az_df (pd.DataFrame): Arizona training DataFrame.

    Returns:
        dict: Evaluation results containing averaged comparison DataFrame.
    """
    n_sizes = len(EVAL_TEST_SIZES)
    n_seeds = N_EVAL_SEEDS
    logger.info('='*60)
    logger.info(f'Step 2a: Random evaluation ({n_sizes} test sizes × {n_seeds} seeds)')
    logger.info('='*60)

    base_dir = os.path.join(MODEL_DIR, 'Model_Evaluation/Random')
    rng = np.random.RandomState(RANDOM_STATE)
    seeds = [RANDOM_STATE] + [int(rng.randint(0, 2**31)) for _ in range(n_seeds - 1)]

    all_dfs = []
    tuning_model_dir = None
    for ts in EVAL_TEST_SIZES:
        ts_label = f'ts{int(ts*100):02d}'
        for i, seed in enumerate(seeds):
            logger.info(f'--- Random test_size={ts:.0%} seed {i+1}/{n_seeds} (seed={seed}) ---')
            run_dir = os.path.join(base_dir, ts_label, f'seed_{seed}')
            first_run = tuning_model_dir is None
            df, comp_dir = _evaluate_random_single(
                az_df, seed, run_dir,
                tuning_model_dir=tuning_model_dir,
                test_size=ts,
                create_interp_plots=first_run,
            )
            if tuning_model_dir is None:
                tuning_model_dir = comp_dir  # capture from first run
            df['seed'] = seed
            df['test_size'] = ts
            all_dfs.append(df)

    all_runs_df = pd.concat(all_dfs, ignore_index=True)
    all_runs_df.to_csv(os.path.join(base_dir, 'All_Runs.csv'), index=False)

    # Average across seeds per (Model, test_size)
    group_cols = ['Model', 'test_size']
    numeric_cols = all_runs_df.select_dtypes(include='number').columns.difference(
        ['seed', 'test_size'])
    avg_df = (all_runs_df.groupby(group_cols)[numeric_cols]
              .agg(['mean', 'std']).reset_index())
    avg_df.columns = [c[0] if c[1] == '' else f'{c[0]}_{c[1]}'
                      for c in avg_df.columns]
    avg_df = avg_df.sort_values(['test_size', 'Test_RMSE_mean'])
    avg_df.to_csv(os.path.join(base_dir, 'Model_Comparison_Averaged.csv'), index=False)

    logger.debug(f'\nRandom averaged comparison ({n_sizes}×{n_seeds} grid):\n'
                 f'{avg_df.to_string(index=False)}')

    all_runs_csv = os.path.join(base_dir, 'All_Runs.csv')
    avg_csv = os.path.join(base_dir, 'Model_Comparison_Averaged.csv')
    vizops.plot_grid_heatmap(avg_csv, base_dir, strategy_label='Random')
    vizops.plot_grid_boxplots(all_runs_csv, base_dir, strategy_label='Random')
    vizops.plot_grid_bar_charts(all_runs_csv, base_dir, strategy_label='Random')

    return {'comparison_df': avg_df, 'strategy': 'Random',
            'all_runs_df': all_runs_df}


# ---- 2a2. Pixel holdout evaluation ----------------------------------------
def _evaluate_pixel_holdout_single(
        az_df: pd.DataFrame,
        random_state: int,
        strategy_dir: str,
        tuning_model_dir: str | None = None,
        test_size: float = 0.2,
        create_interp_plots: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Run a single pixel holdout evaluation with the given seed and test size.

    Returns:
        Tuple of (comparison DataFrame, model comparison dir path).
    """
    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
        skip_piml=SKIP_PIML,
    )

    data_dir = os.path.join(strategy_dir, 'data')
    ret_vals = dataops.create_train_test_data(
        az_df, data_dir,
        drop_attr=DROP_ATTRS,
        random_state=random_state,
        scaling=False, already_created=False,
        year_list=YEAR_LIST, split_strategy=5,
        test_year=True,
        test_size=test_size,
        outlier_op=3,
        min_gw_pumping=MIN_GW if MIN_GW is not None else 1e-10,
        max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        log_target=LOG_TARGET,
    )
    (x_train, x_test, y_train, y_test,
     x_scaler, y_scaler,
     year_train, year_test,
     basin_train, basin_test,
     easting_train, easting_test,
     northing_train, northing_test) = ret_vals

    # Append physics floor column for PIML models
    if not SKIP_PIML:
        x_train, x_test = _append_physics_floor(
            x_train, x_test, basin_train, basin_test,
        )

    # Group CV by pixel so inner folds mirror the outer pixel-holdout strategy
    pixel_groups = (
        easting_train['easting_m'].astype(str) + '_'
        + northing_train['northing_m'].astype(str)
    ).values

    comparison_dir = os.path.join(strategy_dir, 'Model_Comparison')
    return mlops.compare_all_models(
        x_train, x_test, y_train, y_test,
        comparison_dir,
        model_names=ml_models,
        random_state=random_state,
        use_optuna=USE_OPTUNA,
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        year_train=year_train,
        year_test=year_test,
        basin_train=basin_train,
        basin_test=basin_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        use_ama_ina=USE_AMA_INA,
        apply_bias_correction=True,
        cv_groups=pixel_groups,
        easting_train=easting_train,
        easting_test=easting_test,
        northing_train=northing_train,
        northing_test=northing_test,
        test_case='Pixel_Holdout',
        raster_res=MOSAIC_RASTER_RES,
        create_basin_plots=True,
        log_target=LOG_TARGET,
        tuning_model_dir=tuning_model_dir,
        create_interp_plots=create_interp_plots,
        use_interaction_constraints=PHYSICS_INTERACTION_CONSTRAINTS,
    )


def evaluate_pixel_holdout(az_df: pd.DataFrame) -> dict:
    """Pixel holdout — 5 test sizes × 5 seeds grid evaluation.

    Optuna tuning runs once (first seed, first test size).  All other
    combinations reuse the tuned hyperparameters and only re-split,
    retrain, and evaluate.

    Args:
        az_df (pd.DataFrame): Arizona training DataFrame.

    Returns:
        dict: Evaluation results containing averaged comparison DataFrame.
    """
    n_sizes = len(EVAL_TEST_SIZES)
    n_seeds = N_EVAL_SEEDS
    logger.info('=' * 60)
    logger.info(f'Step 2a2: Pixel holdout evaluation ({n_sizes} test sizes × {n_seeds} seeds)')
    logger.info('=' * 60)

    base_dir = os.path.join(MODEL_DIR, 'Model_Evaluation/Pixel_Holdout')
    rng = np.random.RandomState(RANDOM_STATE)
    seeds = [RANDOM_STATE] + [int(rng.randint(0, 2**31)) for _ in range(n_seeds - 1)]

    all_dfs = []
    tuning_model_dir = None
    for ts in EVAL_TEST_SIZES:
        ts_label = f'ts{int(ts*100):02d}'
        for i, seed in enumerate(seeds):
            logger.info(f'--- Pixel holdout test_size={ts:.0%} seed {i+1}/{n_seeds} (seed={seed}) ---')
            run_dir = os.path.join(base_dir, ts_label, f'seed_{seed}')
            first_run = tuning_model_dir is None
            df, comp_dir = _evaluate_pixel_holdout_single(
                az_df, seed, run_dir,
                tuning_model_dir=tuning_model_dir,
                test_size=ts,
                create_interp_plots=first_run,
            )
            if tuning_model_dir is None:
                tuning_model_dir = comp_dir
            df['seed'] = seed
            df['test_size'] = ts
            all_dfs.append(df)

    all_runs_df = pd.concat(all_dfs, ignore_index=True)
    all_runs_df.to_csv(os.path.join(base_dir, 'All_Runs.csv'), index=False)

    group_cols = ['Model', 'test_size']
    numeric_cols = all_runs_df.select_dtypes(include='number').columns.difference(
        ['seed', 'test_size'])
    avg_df = (all_runs_df.groupby(group_cols)[numeric_cols]
              .agg(['mean', 'std']).reset_index())
    avg_df.columns = [c[0] if c[1] == '' else f'{c[0]}_{c[1]}'
                      for c in avg_df.columns]
    avg_df = avg_df.sort_values(['test_size', 'Test_RMSE_mean'])
    avg_df.to_csv(os.path.join(base_dir, 'Model_Comparison_Averaged.csv'), index=False)

    logger.debug(f'\nPixel holdout averaged comparison ({n_sizes}×{n_seeds} grid):\n'
                 f'{avg_df.to_string(index=False)}')

    all_runs_csv = os.path.join(base_dir, 'All_Runs.csv')
    avg_csv = os.path.join(base_dir, 'Model_Comparison_Averaged.csv')
    vizops.plot_grid_heatmap(avg_csv, base_dir, strategy_label='Pixel Holdout')
    vizops.plot_grid_boxplots(all_runs_csv, base_dir, strategy_label='Pixel Holdout')
    vizops.plot_grid_bar_charts(all_runs_csv, base_dir, strategy_label='Pixel Holdout')

    return {'comparison_df': avg_df, 'strategy': 'Pixel_Holdout',
            'all_runs_df': all_runs_df}


# ---- 2b. Leave-one-out temporal holdout ------------------------------------
def evaluate_temporal_loo(az_df: pd.DataFrame) -> dict:
    """
    Evaluate each model on every temporal holdout (T1-T7).

    Returns per-holdout and averaged metrics across all holdouts.

    Args:
        az_df (pd.DataFrame): Full predictor dataframe with all years.

    Returns:
        dict: Per-model averaged metrics across all temporal holdouts.
    """
    logger.info('='*60)
    logger.info('Step 2b: LOO Temporal evaluation (T1_Baseline + T1-T7)')
    logger.info('='*60)

    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
        skip_piml=SKIP_PIML,
    )
    temporal_dir = os.path.join(MODEL_DIR, 'Model_Evaluation/Temporal_LOO')
    makedirs(temporal_dir)

    per_holdout_rows = []  # per-holdout per-model metrics

    for holdout_name, test_year_limits in TEMPORAL_HOLDOUTS.items():
        logger.info(f'\n--- Temporal holdout {holdout_name}: {test_year_limits} ---')
        test_years = []
        for s, e in test_year_limits:
            test_years.extend(range(s, e + 1))
        test_years = tuple(test_years)

        holdout_dir = os.path.join(temporal_dir, holdout_name)
        data_dir = os.path.join(holdout_dir, 'data')
        # Baseline holdouts use the full metered year range and no min pumping
        # filter to match the previous study (Majumdar et al., 2022).
        is_baseline = 'Baseline' in holdout_name
        yl = TRAIN_YEAR_LIST_BASELINE if is_baseline else YEAR_LIST
        min_gw = 0 if is_baseline else (MIN_GW if MIN_GW is not None else 1e-10)
        max_gw = MAX_GW if MAX_GW is not None else np.inf
        drop_gw_basins = ('WILLCOX AMA', 'HUALAPAI VALLEY INA') if is_baseline else DROP_GW_BASINS
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=yl, split_strategy=1,
            test_year=test_years,
            outlier_op=3,
            min_gw_pumping=min_gw,
            max_gw_pumping=max_gw,
            test_gw_basins=(),
            use_ama_ina=USE_AMA_INA,
            drop_gw_basins=drop_gw_basins,
            log_target=LOG_TARGET,
        )
        (x_train, x_test, y_train, y_test,
         x_scaler, y_scaler,
         year_train, year_test,
         basin_train, basin_test,
         easting_train, easting_test,
         northing_train, northing_test) = ret_vals

        if len(y_test) == 0:
            logger.warning(f'No test data for {holdout_name}, skipping.')
            continue

        # Append physics floor column for PIML models
        if not SKIP_PIML:
            x_train, x_test = _append_physics_floor(
                x_train, x_test, basin_train, basin_test,
            )

        # Group CV by year so inner folds mirror the outer temporal-holdout strategy
        temporal_cv_groups = year_train.values.ravel()

        for model_name in ml_models:
            model_dir = os.path.join(holdout_dir, model_name)

            # Fast path: reuse cached BC predictions if available
            bc_pq = os.path.join(model_dir, f'Predictions_{model_name}_BC.parquet')
            plain_pq = os.path.join(model_dir, f'Predictions_{model_name}.parquet')
            cached_pq = bc_pq if os.path.isfile(bc_pq) else (
                plain_pq if os.path.isfile(plain_pq) else None)

            if cached_pq is not None:
                logger.info(f'  Loading cached {model_name} for {holdout_name}')
                pred_df = pd.read_parquet(cached_pq)

                # Generate interpretability plots if missing for cached predictions
                interp_dir = os.path.join(model_dir, 'Interpretability')
                interp_exists = (
                    os.path.isdir(interp_dir)
                    and len(os.listdir(interp_dir)) > 0
                )
                if not interp_exists:
                    model_file = os.path.join(model_dir, model_name)
                    if os.path.isfile(model_file):
                        logger.info(f'Interpretability missing for cached {model_name} — '
                                    f'loading model and generating plots')
                        with open(model_file, 'rb') as f:
                            cached_model = pickle.load(f)
                        mlops.generate_interp_plots(
                            cached_model, model_name, x_train, x_test,
                            y_train, y_test, model_dir,
                            RANDOM_STATE, LOG_TARGET,
                        )
            else:
                logger.info(f'  Training {model_name} for {holdout_name}...')
                res = _train_and_evaluate(
                    x_train, y_train, x_test, y_test,
                    model_name, model_dir,
                    cv_groups=temporal_cv_groups,
                )
                pred_df = mlops.get_prediction_results(
                    res['model'], x_train, x_test,
                    y_train, y_test, x_scaler, y_scaler,
                    year_train, year_test,
                    basin_train, basin_test,
                    model_dir, model_name,
                    apply_bias_correction=True,
                    easting_train=easting_train,
                    easting_test=easting_test,
                    northing_train=northing_train,
                    northing_test=northing_test,
                    log_target=LOG_TARGET,
                )

                # Interpretability plots (perm importance, ALE, SHAP)
                mlops.generate_interp_plots(
                    res['model'], model_name, x_train, x_test,
                    y_train, y_test, model_dir,
                    RANDOM_STATE, LOG_TARGET,
                )

            mlops.calc_train_test_metrics(
                pred_df, pd.DataFrame(), model_dir,
                use_ama_ina=USE_AMA_INA, model_name=model_name,
            )
            mlops.generate_model_visualizations(
                pred_df=pred_df,
                output_dir=os.path.join(model_dir, 'Visualizations'),
                model_name=model_name,
                test_case=holdout_name,
                test_year_limits=test_year_limits,
                raster_res=MOSAIC_RASTER_RES,
                use_ama_ina=USE_AMA_INA,
                create_basin_plots=True,
            )

            bc_train, bc_test = _metrics_from_pred_df(pred_df)
            logger.info(f'    [BC] Train R2: {bc_train["R2"]:.4f}, '
                        f'Test R2: {bc_test["R2"]:.4f}, '
                        f'Test RMSE: {bc_test["RMSE_pct"]:.2f}%')

            per_holdout_rows.append({
                'Holdout': holdout_name,
                'Model': model_name,
                'Train_R2': bc_train['R2'],
                'Test_R2': bc_test['R2'],
                'Train_RMSE': bc_train['RMSE_pct'],
                'Test_RMSE': bc_test['RMSE_pct'],
                'Test_MAE': bc_test['MAE_pct'],
                'Test_MBE': bc_test['MBE_pct'],
                'Overfit_R2': bc_train['R2'] - bc_test['R2'],
                'Overfit_RMSE': bc_train['RMSE_pct'] - bc_test['RMSE_pct'],
            })

    # Build results DataFrames
    per_holdout_df = pd.DataFrame(per_holdout_rows).round(4)
    per_holdout_df.to_csv(os.path.join(temporal_dir, 'Per_Holdout_Metrics.csv'), index=False)
    logger.info(f'\nPer-holdout metrics:\n{per_holdout_df.to_string(index=False)}')

    # Exclude T1_Baseline from averages and plots (different training config)
    plot_df = per_holdout_df[~per_holdout_df['Holdout'].str.contains('Baseline')]

    # Averaged metrics per model across holdouts (excluding baseline)
    avg_df = (
        plot_df
        .groupby('Model')
        .agg(
            Mean_Test_R2=('Test_R2', 'mean'),
            Std_Test_R2=('Test_R2', 'std'),
            Mean_Test_RMSE=('Test_RMSE', 'mean'),
            Std_Test_RMSE=('Test_RMSE', 'std'),
            Mean_Test_MAE=('Test_MAE', 'mean'),
            Mean_Test_MBE=('Test_MBE', 'mean'),
            Mean_Overfit_R2=('Overfit_R2', 'mean'),
        )
        .reset_index()
        .sort_values('Mean_Test_RMSE')
        .round(4)
    )
    avg_df.to_csv(os.path.join(temporal_dir, 'Averaged_Metrics.csv'), index=False)
    logger.info(f'\nAveraged temporal metrics:\n{avg_df.to_string(index=False)}')

    # Save filtered CSV for plots, then visualize
    plot_csv = os.path.join(temporal_dir, 'Per_Holdout_Metrics_Plot.csv')
    plot_df.to_csv(plot_csv, index=False)

    vizops.plot_loo_heatmap(
        plot_df, 'Holdout', temporal_dir,
        title='Temporal LOO: Test R² per Holdout',
    )
    vizops.plot_loo_bar(plot_df, 'Holdout', temporal_dir)
    vizops.plot_loo_distribution(
        plot_csv, 'Holdout', temporal_dir, strategy_label='Temporal LOO',
    )

    return {
        'per_holdout_df': per_holdout_df,
        'avg_df': avg_df,
        'strategy': 'Temporal_LOO',
    }


# ---- 2c. Leave-one-out spatial holdout (ADWR sub-basins) -------------------
def _get_all_subbasins() -> list[str]:
    """Return all ADWR sub-basin names (excluding dropped basins)."""
    sub_gdf = gpd.read_file(ADWR_SUBBASIN_SHP)
    parent_drop = set()
    for code, name in AMA_CODE_MAP.items():
        if name in DROP_GW_BASINS:
            parent_drop.add(code)
    if parent_drop:
        sub_gdf = sub_gdf[~sub_gdf['AMA_CODE'].isin(parent_drop)]
    return sorted(sub_gdf['SUBBASIN_N'].dropna().unique().tolist())


def _get_ama_ina_subbasins() -> list[str]:
    """Return the list of ADWR sub-basin names within AMA/INA."""
    sub_gdf = gpd.read_file(ADWR_SUBBASIN_SHP)
    ama_ina_codes = set(AMA_CODE_MAP.keys())
    sub_gdf = sub_gdf[sub_gdf['AMA_CODE'].isin(ama_ina_codes)]
    subbasins = sorted(sub_gdf['SUBBASIN_N'].dropna().unique().tolist())
    # Remove any sub-basins whose parent AMA/INA is dropped
    parent_drop = set()
    for code, name in AMA_CODE_MAP.items():
        if name in DROP_GW_BASINS:
            parent_drop.add(code)
    if parent_drop:
        sub_gdf_filt = sub_gdf[~sub_gdf['AMA_CODE'].isin(parent_drop)]
        subbasins = sorted(sub_gdf_filt['SUBBASIN_N'].dropna().unique().tolist())
    return subbasins


def evaluate_spatial_loo(az_df: pd.DataFrame,
                         seed_fraction: float = 0.0) -> dict:
    """
    Leave-one-out spatial evaluation: hold out each AMA/INA management
    area one at a time, train on the rest, evaluate on the held-out
    basin.

    When ``seed_fraction > 0``, a randomly sampled fraction of each
    held-out basin's samples are moved into the training set as a
    calibration anchor, and only the remaining samples are used for
    evaluation.  This tests whether a small amount of local data is
    sufficient to correct the basin-specific pumping magnitude offset.

    Reports per-basin and averaged metrics.

    Args:
        az_df (pd.DataFrame): Full predictor dataframe with all years.
        seed_fraction (float): Fraction of held-out basin samples to
            move into training (default 0.0 = pure LOO).

    Returns:
        dict: Per-model averaged metrics across all spatial holdouts.
    """
    logger.info('='*60)
    seed_pct = int(seed_fraction * 100)
    label = f'LOO Spatial evaluation (AMA/INA, seed={seed_pct}%)'
    logger.info(f'Step 2c: {label}')
    logger.info('='*60)

    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
        skip_piml=SKIP_PIML,
    )

    # Identify AMA/INA basins from GW_Basin_Type (0=AMA, 1=INA),
    # applying the same pumping filters used during training
    metered_df = az_df[az_df['Year'].isin(YEAR_LIST)]
    ama_ina_df = metered_df[metered_df['GW_Basin_Type'].isin([0, 1])]
    min_gw = MIN_GW if MIN_GW is not None else 1e-10
    max_gw = MAX_GW if MAX_GW is not None else np.inf
    filtered = ama_ina_df[
        (ama_ina_df['gw_pumping_mm'] >= min_gw) &
        (ama_ina_df['gw_pumping_mm'] <= max_gw)
    ]
    basin_counts = filtered.groupby('GW_Basin').size()

    # Exclude basins with too few valid metered samples or explicitly skipped
    basins = sorted(b for b, n in basin_counts.items()
                    if n >= MIN_SPATIAL_EVAL_SAMPLES
                    and b not in SKIP_SPATIAL_BASINS)
    excluded = sorted(b for b, n in basin_counts.items()
                      if n < MIN_SPATIAL_EVAL_SAMPLES
                      or b in SKIP_SPATIAL_BASINS)
    if excluded:
        logger.warning(f'Excluding basins (< {MIN_SPATIAL_EVAL_SAMPLES} '
                       f'valid samples or in SKIP_SPATIAL_BASINS): {excluded}')

    logger.info(f'Basins to evaluate ({len(basins)}): {basins}')

    dir_suffix = f'_Seed{seed_pct}' if seed_fraction > 0 else ''
    spatial_dir = os.path.join(MODEL_DIR, f'Model_Evaluation/Spatial_LOO{dir_suffix}')
    makedirs(spatial_dir)

    per_basin_rows = []
    stratified_rows = []

    for basin in basins:
        logger.info(f'\n--- Spatial holdout: {basin} ---')
        basin_safe = basin.replace(' ', '_').replace('.', '')
        holdout_dir = os.path.join(spatial_dir, basin_safe)

        data_dir = os.path.join(holdout_dir, 'data')
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=YEAR_LIST, split_strategy=3,
            test_year=(),
            outlier_op=3,
            min_gw_pumping=MIN_GW if MIN_GW is not None else 1e-10,
            max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
            test_gw_basins=(basin,),
            gw_basin_col='GW_Basin',
            use_ama_ina=USE_AMA_INA,
            drop_gw_basins=SKIP_SPATIAL_BASINS,
            log_target=LOG_TARGET,
        )
        (x_train, x_test, y_train, y_test,
         x_scaler, y_scaler,
         year_train, year_test,
         basin_train, basin_test,
         easting_train, easting_test,
         northing_train, northing_test) = ret_vals

        if len(y_test) == 0:
            logger.warning(f'No test data for basin {basin}, skipping.')
            continue

        # Move a random fraction of held-out samples into training as a seed
        if seed_fraction > 0:
            rng = np.random.default_rng(RANDOM_STATE)
            n_test = len(y_test)
            n_seed = max(1, int(n_test * seed_fraction))
            seed_idx = rng.choice(n_test, size=n_seed, replace=False)
            keep_idx = np.setdiff1d(np.arange(n_test), seed_idx)

            x_train = pd.concat([x_train, x_test.iloc[seed_idx]], ignore_index=True)
            y_train = np.concatenate([y_train, y_test[seed_idx]])
            year_train = pd.concat([year_train, year_test.iloc[seed_idx]], ignore_index=True)
            basin_train = pd.concat([basin_train, basin_test.iloc[seed_idx]], ignore_index=True)
            easting_train = pd.concat([easting_train, easting_test.iloc[seed_idx]], ignore_index=True)
            northing_train = pd.concat([northing_train, northing_test.iloc[seed_idx]], ignore_index=True)

            x_test = x_test.iloc[keep_idx].reset_index(drop=True)
            y_test = y_test[keep_idx]
            year_test = year_test.iloc[keep_idx].reset_index(drop=True)
            basin_test = basin_test.iloc[keep_idx].reset_index(drop=True)
            easting_test = easting_test.iloc[keep_idx].reset_index(drop=True)
            northing_test = northing_test.iloc[keep_idx].reset_index(drop=True)

            logger.info(f'  Seeded {n_seed} samples ({seed_pct}%) '
                        f'from {basin} into training')

        logger.info(f'  Train: {len(y_train)}, Test: {len(y_test)}')

        # Append physics floor column for PIML models
        if not SKIP_PIML:
            x_train, x_test = _append_physics_floor(
                x_train, x_test, basin_train, basin_test,
            )

        for model_name in ml_models:
            model_dir = os.path.join(holdout_dir, model_name)

            # Fast path: reuse cached BC predictions if available
            bc_pq = os.path.join(model_dir, f'Predictions_{model_name}_BC.parquet')
            plain_pq = os.path.join(model_dir, f'Predictions_{model_name}.parquet')
            cached_pq = bc_pq if os.path.isfile(bc_pq) else (
                plain_pq if os.path.isfile(plain_pq) else None)

            if cached_pq is not None:
                logger.info(f'  Loading cached {model_name} (holdout: {basin})')
                pred_df = pd.read_parquet(cached_pq)

                # Generate interpretability plots if missing for cached predictions
                interp_dir = os.path.join(model_dir, 'Interpretability')
                interp_exists = (
                    os.path.isdir(interp_dir)
                    and len(os.listdir(interp_dir)) > 0
                )
                if not interp_exists:
                    model_file = os.path.join(model_dir, model_name)
                    if os.path.isfile(model_file):
                        logger.info(f'Interpretability missing for cached {model_name} — '
                                    f'loading model and generating plots')
                        with open(model_file, 'rb') as f:
                            cached_model = pickle.load(f)
                        mlops.generate_interp_plots(
                            cached_model, model_name, x_train, x_test,
                            y_train, y_test, model_dir,
                            RANDOM_STATE, LOG_TARGET,
                        )
            else:
                logger.info(f'  Training {model_name} (holdout: {basin})...')
                res = _train_and_evaluate(
                    x_train, y_train, x_test, y_test,
                    model_name, model_dir,
                )
                pred_df = mlops.get_prediction_results(
                    res['model'], x_train, x_test,
                    y_train, y_test, x_scaler, y_scaler,
                    year_train, year_test,
                    basin_train, basin_test,
                    model_dir, model_name,
                    gw_basin_col='GW_Basin',
                    apply_bias_correction=False,
                    easting_train=easting_train,
                    easting_test=easting_test,
                    northing_train=northing_train,
                    northing_test=northing_test,
                    log_target=LOG_TARGET,
                )

                # Interpretability plots (perm importance, ALE, SHAP)
                mlops.generate_interp_plots(
                    res['model'], model_name, x_train, x_test,
                    y_train, y_test, model_dir,
                    RANDOM_STATE, LOG_TARGET,
                )

            mlops.calc_train_test_metrics(
                pred_df, pd.DataFrame(), model_dir,
                use_ama_ina=USE_AMA_INA,
                gw_basin_col='GW_Basin',
                model_name=model_name,
            )
            mlops.generate_model_visualizations(
                pred_df=pred_df,
                output_dir=os.path.join(model_dir, 'Visualizations'),
                model_name=model_name,
                test_case=f'Spatial_LOO_{basin_safe}',
                test_year_limits=(),
                gw_basin_col='GW_Basin',
                use_ama_ina=USE_AMA_INA,
                create_basin_plots=True,
                skip_aggregate_ts=True,
                scatter_axis_max=MAX_GW,
            )

            train_m, test_m = _metrics_from_pred_df(pred_df)
            logger.info(f'    Train R2: {train_m["R2"]:.4f}, '
                        f'Test R2: {test_m["R2"]:.4f}, '
                        f'Test RMSE: {test_m["RMSE_pct"]:.2f}%')

            per_basin_rows.append({
                'Basin': basin,
                'Model': model_name,
                'N_test': len(y_test),
                'Train_R2': train_m['R2'],
                'Test_R2': test_m['R2'],
                'Train_RMSE': train_m['RMSE_pct'],
                'Test_RMSE': test_m['RMSE_pct'],
                'Test_MAE': test_m['MAE_pct'],
                'Test_MBE': test_m['MBE_pct'],
                'Overfit_R2': train_m['R2'] - test_m['R2'],
                'Overfit_RMSE': train_m['RMSE_pct'] - test_m['RMSE_pct'],
            })

            for cat_row in _stratified_test_metrics(pred_df):
                cat_row['Basin'] = basin
                cat_row['Model'] = model_name
                stratified_rows.append(cat_row)

    # Build results DataFrames
    per_basin_df = pd.DataFrame(per_basin_rows).round(4)
    per_basin_df.to_csv(os.path.join(spatial_dir, 'Per_Basin_Metrics.csv'), index=False)
    logger.info(f'\nPer-basin metrics:\n{per_basin_df.to_string(index=False)}')

    # Averaged metrics per model across basins
    avg_df = (
        per_basin_df
        .groupby('Model')
        .agg(
            Mean_Test_R2=('Test_R2', 'mean'),
            Std_Test_R2=('Test_R2', 'std'),
            Mean_Test_RMSE=('Test_RMSE', 'mean'),
            Std_Test_RMSE=('Test_RMSE', 'std'),
            Mean_Test_MAE=('Test_MAE', 'mean'),
            Mean_Test_MBE=('Test_MBE', 'mean'),
            Mean_Overfit_R2=('Overfit_R2', 'mean'),
        )
        .reset_index()
        .sort_values('Mean_Test_RMSE')
        .round(4)
    )
    avg_df.to_csv(os.path.join(spatial_dir, 'Averaged_Metrics.csv'), index=False)
    logger.info(f'\nAveraged spatial metrics:\n{avg_df.to_string(index=False)}')

    # Visualizations
    strategy_label = f'Spatial LOO (seed {seed_pct}%)' if seed_fraction > 0 else 'Spatial LOO'
    vizops.plot_loo_heatmap(
        per_basin_df, 'Basin', spatial_dir,
        title=f'{strategy_label}: Test R² per AMA/INA',
    )
    vizops.plot_loo_bar(per_basin_df, 'Basin', spatial_dir)
    vizops.plot_loo_distribution(
        os.path.join(spatial_dir, 'Per_Basin_Metrics.csv'),
        'Basin', spatial_dir, strategy_label=strategy_label,
    )

    # Stratified metrics by pumping magnitude
    if stratified_rows:
        strat_df = pd.DataFrame(stratified_rows).round(4)
        strat_df.to_csv(
            os.path.join(spatial_dir, 'Stratified_Metrics.csv'), index=False)
        logger.info(f'\nStratified metrics:\n{strat_df.to_string(index=False)}')
        vizops.plot_stratified_metrics(strat_df, spatial_dir,
                                       strategy_label=strategy_label)

    strategy_key = f'Spatial_LOO_Seed{seed_pct}' if seed_fraction > 0 else 'Spatial_LOO'
    return {
        'per_basin_df': per_basin_df,
        'avg_df': avg_df,
        'strategy': strategy_key,
    }


# =============================================================================
# Step 3 — Predict annual pumping rasters 1896-2099 with XGBoost
# =============================================================================
def _valid_pixels_to_raster(
        values: np.ndarray,
        valid_mask: np.ndarray,
        raster_shape: tuple,
) -> np.ndarray:
    """Map valid-pixel values back to a full 2-D raster grid."""
    grid = np.full(valid_mask.shape[0], np.nan, dtype=np.float32)
    grid[valid_mask] = values.astype(np.float32)
    return grid.reshape(raster_shape)


def _write_multi_unit_rasters(
        mm_grid: np.ndarray,
        out_dirs: dict[str, str],
        prefix: str,
        year: int,
        ref_raster_file: str,
        mm_to_ft: float,
        mm_to_m3: float,
        m3_to_af: float,
) -> None:
    """Write a depth grid in mm, ft, m³, and AF."""
    unit_grids = {
        'mm': mm_grid,
        'ft': mm_grid * mm_to_ft,
        'm3': mm_grid * mm_to_m3,
        'AF': mm_grid * mm_to_m3 * m3_to_af,
    }
    for unit, grid in unit_grids.items():
        _, raster_file_obj = read_raster_as_arr(ref_raster_file, get_file=True)
        out_path = os.path.join(out_dirs[unit], f'{prefix}_{year}_{unit}.tif')
        write_raster(
            grid, raster_file_obj,
            raster_file_obj.transform, out_path,
            no_data_value=np.nan,
            num_bands=1,
        )
        raster_file_obj.close()


def predict_full_period(az_df: pd.DataFrame) -> tuple:
    """
    Train the selected model (``PREDICTION_MODEL``) on the full 1984-2024
    metered data and predict groundwater pumping rasters for every year
    from 1896 to 2099.

    Returns:
        tuple: (model, feature_cols, x_train, y_train) — the trained model,
            feature column names, and training data for uncertainty quantification.
    """
    model_name = PREDICTION_MODEL
    logger.info('='*60)
    logger.info(f'Step 3: {model_name} full-period prediction (1896-2099)')
    logger.info('='*60)
    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{model_name}')
    makedirs(prediction_dir)

    # ---- 3a. Train on ALL 1984-2024 metered data (no holdout) ----
    # Step 2 already provides thorough LOO evaluation; here we maximize
    # training data for the best possible full-period predictions.
    data_dir = os.path.join(prediction_dir, 'data')
    ret_vals = dataops.create_train_test_data(
        az_df, data_dir,
        drop_attr=DROP_ATTRS,
        random_state=RANDOM_STATE,
        scaling=False,
        already_created=False,
        year_list=YEAR_LIST,
        split_strategy=1,
        test_year=(),
        outlier_op=3,
        min_gw_pumping=MIN_GW if MIN_GW is not None else 1e-10,
        max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        log_target=LOG_TARGET,
    )
    x_train, y_train = ret_vals[0], ret_vals[2]

    logger.info(f'Training {model_name} on {len(x_train)} samples '
                f'({YEAR_LIST[0]}-{YEAR_LIST[-1]}, all years)...')
    model_path = os.path.join(prediction_dir, 'Model')
    cached_model = os.path.join(model_path, model_name)
    model, _ = mlops.build_ml_model_optuna(
        x_train, y_train,
        model_path,
        model_name, RANDOM_STATE,
        load_model=os.path.isfile(cached_model),
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        log_target=LOG_TARGET,
    )

    # ---- Model interpretability plots (on clean features, no physics col) ----
    interp_dir = os.path.join(prediction_dir, 'Model_Interpretability')
    makedirs(interp_dir)

    # Feature importance + permutation importance
    fimp_png = os.path.join(interp_dir, f'F_IMP_{model_name}.png')
    if not os.path.isfile(fimp_png):
        mlops.compute_perm_imp(
            model_name, x_train, x_train, y_train, y_train,
            model, interp_dir, scoring_metric='scaled_rmse',
            random_state=RANDOM_STATE, create_plots=True,
            log_target=LOG_TARGET,
        )
    else:
        logger.info(f'Skipping feature importance (found {fimp_png})')

    # ALE plots
    ale_png = os.path.join(interp_dir, f'{model_name}_ALE_Train.png')
    if not os.path.isfile(ale_png):
        mlops.compute_ale_plots(
            model_name, model,
            x_train, y_train, x_train, y_train,
            interp_dir, log_target=LOG_TARGET,
        )
    else:
        logger.info(f'Skipping ALE plots (found {ale_png})')

    # SHAP plots (TreeExplainer; SHAP values remain in log1p space)
    shap_png = os.path.join(interp_dir, f'{model_name}_SHAP_Summary.png')
    if not os.path.isfile(shap_png):
        mlops.compute_shap_plots(
            model_name, model, x_train, interp_dir,
        )
    else:
        logger.info(f'Skipping SHAP plots (found {shap_png})')

    # ---- 3b. Predict pumping for each year 1896-2099 ----
    logger.info('Predicting pumping for all years 1896-2099...')

    # Per-pixel prediction range check — training-era ceiling.
    # Modern pump infrastructure sets a physical upper bound on per-pixel
    # withdrawal rates.  Predictions exceeding the training-era maximum
    # are flagged as physically implausible extrapolations.
    y_train_orig = np.expm1(y_train) if LOG_TARGET else y_train
    train_pixel_max_mm = float(np.abs(y_train_orig).max())
    train_pixel_p99_mm = float(np.percentile(np.abs(y_train_orig), 99))
    logger.info(
        'Training-era per-pixel pumping ceiling: max=%.2f mm, '
        'P99=%.2f mm',
        train_pixel_max_mm, train_pixel_p99_mm,
    )
    exceedance_summary = []  # collect per-year exceedance stats

    # ---- Global linear bias correction from training residuals ----
    # A single linear correction (pred_corrected = |m*pred + b|) is learned
    # from ALL AMA/INA training data so it generalizes to non-AMA/INA basins
    # during full-period prediction.
    # Bias correction operates in original (mm) scale.
    train_preds = np.abs(model.predict(x_train))
    if LOG_TARGET:
        train_preds = np.abs(np.expm1(train_preds))

    bc_dir = os.path.join(prediction_dir, 'Bias_Correction')
    makedirs(bc_dir)

    bc_m, bc_b = mlops.fit_linear_bc(train_preds, y_train_orig)
    linear_corrected = mlops.apply_linear_bc(train_preds, bc_m, bc_b)

    # Check if BC improves R2, RMSE, and MAE (same logic as LOO)
    from sklearn.metrics import r2_score
    raw_r2 = r2_score(y_train_orig, train_preds)
    bc_r2 = r2_score(y_train_orig, linear_corrected)
    raw_rmse = mlops.normalized_rmse(y_train_orig, train_preds)
    bc_rmse = mlops.normalized_rmse(y_train_orig, linear_corrected)
    raw_mae = mlops.normalized_mae(y_train_orig, train_preds)
    bc_mae = mlops.normalized_mae(y_train_orig, linear_corrected)

    bc_improved = (bc_r2 >= raw_r2 and bc_rmse <= raw_rmse and bc_mae <= raw_mae)

    bc_summary = pd.DataFrame([{
        'Slope': round(bc_m, 4),
        'Intercept': round(bc_b, 4),
        'N_Samples': len(train_preds),
        'Raw_R2': round(raw_r2, 4),
        'BC_R2': round(bc_r2, 4),
        'Raw_RMSE': round(raw_rmse, 4),
        'BC_RMSE': round(bc_rmse, 4),
        'Raw_MAE': round(raw_mae, 4),
        'BC_MAE': round(bc_mae, 4),
        'BC_Applied': bc_improved,
    }])
    bc_summary.to_csv(os.path.join(bc_dir, 'Global_BC_Summary.csv'), index=False)

    if bc_improved:
        logger.info(
            'BC improved all metrics — will apply. '
            'R2: %.4f->%.4f, RMSE: %.2f->%.2f, MAE: %.2f->%.2f',
            raw_r2, bc_r2, raw_rmse, bc_rmse, raw_mae, bc_mae,
        )
    else:
        logger.warning(
            'BC did NOT improve all metrics — skipping. '
            'R2: %.4f->%.4f, RMSE: %.2f->%.2f, MAE: %.2f->%.2f',
            raw_r2, bc_r2, raw_rmse, bc_rmse, raw_mae, bc_mae,
        )

    feature_cols = list(x_train.columns)

    # Fit OOD detector on climate/LULC features only (exclude spatial
    # coordinates and spatially-fixed features that would cause all
    # non-AMA/INA pixels — even irrigated areas like Yuma — to show
    # OOD=1 purely due to geographic location).
    _ood_exclude = {'easting_m', 'northing_m', 'well_density',
                    'canal_density', 'canal_weighted_streamflow_mm',
                    'streamflow_mm'}
    _ood_cols = [c for c in feature_cols if c not in _ood_exclude]
    ood_detector = mlops.OODDetector(alpha=0.01)
    ood_detector.fit(x_train[_ood_cols])

    raster_dir = os.path.join(prediction_dir, 'Predicted_Rasters')
    raster_dirs = {
        'mm': os.path.join(raster_dir, 'Depth_mm'),
        'ft': os.path.join(raster_dir, 'Depth_ft'),
        'm3': os.path.join(raster_dir, 'Volume_m3'),
        'AF': os.path.join(raster_dir, 'Volume_AF'),
    }
    for d in raster_dirs.values():
        makedirs(d)

    # Category-specific raster directories
    cat_raster_dirs = {}
    for cat in partops.CATEGORIES:
        base = os.path.join(prediction_dir, f'{cat}_Rasters')
        cat_raster_dirs[cat] = {
            'mm': os.path.join(base, 'Depth_mm'),
            'ft': os.path.join(base, 'Depth_ft'),
            'm3': os.path.join(base, 'Volume_m3'),
            'AF': os.path.join(base, 'Volume_AF'),
        }
        for d in cat_raster_dirs[cat].values():
            makedirs(d)

    # Consumptive use raster directories
    CU_CATEGORIES = ('Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU')
    cu_raster_dirs = {}
    for cu_cat in CU_CATEGORIES:
        base = os.path.join(prediction_dir, f'{cu_cat}_Rasters')
        cu_raster_dirs[cu_cat] = {
            'mm': os.path.join(base, 'Depth_mm'),
            'ft': os.path.join(base, 'Depth_ft'),
            'm3': os.path.join(base, 'Volume_m3'),
            'AF': os.path.join(base, 'Volume_AF'),
        }
        for d in cu_raster_dirs[cu_cat].values():
            makedirs(d)

    # Load USGS NHM basin-level irrigation efficiencies
    nhm_ie_csv = os.path.join(INPUT_DIR, 'USGS WU', 'USGS_NHM_Withdrawals',
                              'IR_HUC12_Eff_annual_2000_2020.csv')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')
    nhm_ie_out = os.path.join(prediction_dir, 'NHM_IE_Basins')
    # Reference raster for spatial metadata (CRS, transform)
    ref_raster_file = os.path.join(PRED_DATA_DIR, f'Predictor_{YEAR_LIST[0]}.tif')

    nhm_basin_ie = intercompops.load_nhm_basin_ie(
        nhm_ie_csv=nhm_ie_csv,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        ref_raster=ref_raster_file,
        output_dir=nhm_ie_out,
    )

    # Build a valid-pixel mask from the GW_Basin raster (same for all years).
    # In create_az_data_parquet, pixels with NaN or 0 in GW_Basin are labeled
    # 'OUTSIDE AZ' and dropped.  The remaining rows — in ravel order — are
    # what appears in az_df for each year.
    ref_basin_file = os.path.join(PRED_DATA_DIR, f'GW_Basin_{YEAR_LIST[0]}.tif')
    basin_arr, basin_file_obj = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    basin_file_obj.close()

    # All AZ basin names (for per-basin summary statistics)
    all_basins = sorted(az_df['GW_Basin'].unique().tolist())
    all_basins = [b for b in all_basins if b not in DROP_GW_BASINS]

    # All sub-basins (for summary statistics)
    subbasins = _get_all_subbasins()

    pixel_area_m2 = MOSAIC_RASTER_RES ** 2
    mm_to_m3 = pixel_area_m2 / 1000          # depth mm × pixel → volume m³
    m3_to_af = 1 / 1233.48                   # m³ → acre-ft
    mm_to_ft = 1 / 304.8                     # mm → ft

    # Load 1981 well density for pre-GMA partitioning override.
    # Pre-1981 well registries are incomplete; the 1981 snapshot (first
    # year with full GMA coverage) provides reasonable well density for
    # the pre-GMA era when many wells were unregistered.
    _gma_ref_yr = 2024
    _gma_df = az_df[az_df.Year == _gma_ref_yr]
    _wd_1981 = _gma_df['well_density'].values if 'well_density' in _gma_df.columns else None
    _irr_wd_1981 = _gma_df['irr_well_density'].values if 'irr_well_density' in _gma_df.columns else None
    _nonirr_wd_1981 = _gma_df['nonirr_well_density'].values if 'nonirr_well_density' in _gma_df.columns else None
    _irr_cap_1981 = _gma_df['irr_capacity_fraction'].values if 'irr_capacity_fraction' in _gma_df.columns else None
    logger.info('Loaded %d well density + irr_capacity from parquet for pre-GMA partitioning override', _gma_ref_yr)

    # Load CAP service-area pixel mask for the CAP delivery
    # perturbation (partops.apply_cap_delivery_perturbation): observed
    # 2022-2024 Tier 1/2 cuts plus the 2026-2099 sustained
    # post-Compact-renegotiation baseline.  Mask is rasterized to the
    # same grid as the basin reference.
    _cap_pixel_mask = None
    _cap_geojson = os.path.join(
        OUTPUT_DIR, 'GW_Data', 'Vector_Reproj', 'CAP_Service_Area.geojson',
    )
    if os.path.isfile(_cap_geojson):
        try:
            import geopandas as gpd
            import rasterio
            from rasterio.features import rasterize as rio_rasterize
            _cap_gdf = gpd.read_file(_cap_geojson)
            with rasterio.open(ref_basin_file) as _ref_src:
                _cap_gdf_proj = _cap_gdf.to_crs(_ref_src.crs)
                _cap_arr = rio_rasterize(
                    [(geom, 1) for geom in _cap_gdf_proj.geometry],
                    out_shape=raster_shape,
                    transform=_ref_src.transform,
                    fill=0,
                    dtype='uint8',
                )
            _cap_pixel_mask = _cap_arr.ravel()[valid_mask] > 0
            logger.info(
                'Loaded CAP service-area mask: %d pixels (of %d valid) '
                'for 2022-2024 CAP-cut hindcast perturbation',
                int(_cap_pixel_mask.sum()), int(valid_mask.sum()),
            )
        except Exception as e:
            logger.warning('Could not load CAP service-area mask: %s — '
                           'CAP-cut hindcast perturbation will be skipped', e)
    else:
        logger.warning('CAP_Service_Area.geojson not found at %s — '
                       'CAP-cut hindcast perturbation will be skipped',
                       _cap_geojson)

    def _pixel_stats(pred_vals, min_depth_threshold=5.0):
        """Compute depth and volume stats in multiple units.

        ``Mean_Depth_mm`` / ``Mean_Depth_ft`` are averaged over
        "active pumping" pixels (pred >= ``min_depth_threshold`` mm/yr,
        default 5 mm/yr) so the reported mean represents per-pixel
        pumping intensity rather than an AZ-wide dilution that
        includes basin-median LU-only fill pixels (sub-mm values).

        ``Volume_m3`` / ``Volume_AF`` remain a *sum over all valid
        pixels* regardless of threshold — volume conservation is
        preserved.  Verified empirically: at 2024, ≥5 mm threshold
        drops 79 % of pixels but loses only 0.3 % of volume.

        Returns NaN for all fields when *pred_vals* is empty (no
        valid pixels), so "no data" is distinguishable from "zero
        pumping".
        """
        n = len(pred_vals)
        if n == 0 or np.all(np.isnan(pred_vals)):
            return {
                'Mean_Depth_mm': np.nan,
                'Mean_Depth_ft': np.nan,
                'Volume_m3': np.nan,
                'Volume_AF': np.nan,
            }
        finite = np.isfinite(pred_vals)
        active = finite & (pred_vals >= min_depth_threshold)
        if np.any(active):
            mean_mm = float(pred_vals[active].mean())
        else:
            # No active pixels — fall back to nanmean so we don't
            # return NaN when all pumping is below threshold.
            mean_mm = float(np.nanmean(pred_vals))
        vol_m3 = float(np.nansum(pred_vals)) * mm_to_m3
        return {
            'Mean_Depth_mm': round(mean_mm, 4),
            'Mean_Depth_ft': round(mean_mm * mm_to_ft, 6),
            'Volume_m3': round(vol_m3, 2),
            'Volume_AF': round(vol_m3 * m3_to_af, 2),
        }

    yearly_predictions = {}        # year → {metrics dict}  (total pumping)
    basin_yearly = {}              # year → {basin_name: metrics dict}
    subbasin_yearly = {}           # year → {subbasin_name: metrics dict}

    # Actual (metered) data for available years
    actual_yearly = {}             # year → {metrics dict}
    actual_basin_yearly = {}       # year → {basin_name: metrics dict}
    actual_subbasin_yearly = {}    # year → {subbasin_name: metrics dict}

    # Per-category tracking dicts: cat → {year → ...}
    cat_yearly = {cat: {} for cat in partops.CATEGORIES}
    cat_basin_yearly = {cat: {} for cat in partops.CATEGORIES}
    cat_subbasin_yearly = {cat: {} for cat in partops.CATEGORIES}

    # Consumptive use tracking dicts
    cu_yearly = {cat: {} for cat in CU_CATEGORIES}
    cu_basin_yearly = {cat: {} for cat in CU_CATEGORIES}
    cu_subbasin_yearly = {cat: {} for cat in CU_CATEGORIES}

    # Out-of-distribution detection output
    ood_raster_dir = os.path.join(prediction_dir, 'OOD_Rasters')
    makedirs(ood_raster_dir)
    ood_summary = []  # collect per-year OOD stats

    # Era-specific feature collection for interpretability analysis
    ERA_BOUNDS = {
        'Hindcast': (START_YEAR, 1983),
        'Historical': (1984, 2025),
        'Projection': (2026, END_YEAR),
    }
    era_features: dict[str, list[pd.DataFrame]] = {e: [] for e in ERA_BOUNDS}
    ERA_SAMPLE_PER_YEAR = 2000  # max pixels sampled per year per era

    # CSV paths for cached annual summaries
    _summary_dir = os.path.join(prediction_dir, 'Annual_Summaries')
    makedirs(_summary_dir)
    _total_csv = os.path.join(_summary_dir, 'Total_Predicted.csv')
    _cat_csvs = {cat: os.path.join(_summary_dir, f'{cat}.csv') for cat in partops.CATEGORIES}
    _cu_csvs = {cu: os.path.join(_summary_dir, f'{cu}.csv') for cu in CU_CATEGORIES}
    _actual_csv = os.path.join(_summary_dir, 'Actual.csv')
    _basin_csv = os.path.join(_summary_dir, 'Basin_Total.csv')
    _subbasin_csv = os.path.join(_summary_dir, 'Subbasin_Total.csv')

    def _save_yearly_summary():
        """Save all yearly summary dicts to CSVs."""
        # Total
        rows = [{'Year': y, **v} for y, v in sorted(yearly_predictions.items())]
        if rows:
            pd.DataFrame(rows).to_csv(_total_csv, index=False)
        # Categories
        for cat in partops.CATEGORIES:
            rows = [{'Year': y, **v} for y, v in sorted(cat_yearly[cat].items())]
            if rows:
                pd.DataFrame(rows).to_csv(_cat_csvs[cat], index=False)
        # CU
        for cu in CU_CATEGORIES:
            rows = [{'Year': y, **v} for y, v in sorted(cu_yearly[cu].items())]
            if rows:
                pd.DataFrame(rows).to_csv(_cu_csvs[cu], index=False)
        # Actuals
        rows = [{'Year': y, **v} for y, v in sorted(actual_yearly.items())]
        if rows:
            pd.DataFrame(rows).to_csv(_actual_csv, index=False)
        # Basin totals (flat: Year, Basin, metrics)
        b_rows = []
        for y, btotals in sorted(basin_yearly.items()):
            for b, metrics in btotals.items():
                b_rows.append({'Year': y, 'Basin': b, **metrics})
        if b_rows:
            pd.DataFrame(b_rows).to_csv(_basin_csv, index=False)
        # Sub-basin totals
        sb_rows = []
        for y, sbtotals in sorted(subbasin_yearly.items()):
            for sb, metrics in sbtotals.items():
                sb_rows.append({'Year': y, 'Subbasin': sb, **metrics})
        if sb_rows:
            pd.DataFrame(sb_rows).to_csv(_subbasin_csv, index=False)
        logger.info('Annual summaries saved to %s', _summary_dir)

    def _load_yearly_summary() -> bool:
        """Load yearly summary dicts from CSVs. Returns True if loaded."""
        if not os.path.isfile(_total_csv):
            return False
        df = pd.read_csv(_total_csv)
        for _, row in df.iterrows():
            yearly_predictions[int(row['Year'])] = {
                k: row[k] for k in row.index if k != 'Year'}
        for cat in partops.CATEGORIES:
            if os.path.isfile(_cat_csvs[cat]):
                cdf = pd.read_csv(_cat_csvs[cat])
                for _, row in cdf.iterrows():
                    cat_yearly[cat][int(row['Year'])] = {
                        k: row[k] for k in row.index if k != 'Year'}
        for cu in CU_CATEGORIES:
            if os.path.isfile(_cu_csvs[cu]):
                cudf = pd.read_csv(_cu_csvs[cu])
                for _, row in cudf.iterrows():
                    cu_yearly[cu][int(row['Year'])] = {
                        k: row[k] for k in row.index if k != 'Year'}
        if os.path.isfile(_actual_csv):
            adf = pd.read_csv(_actual_csv)
            for _, row in adf.iterrows():
                actual_yearly[int(row['Year'])] = {
                    k: row[k] for k in row.index if k != 'Year'}
        if os.path.isfile(_basin_csv):
            bdf = pd.read_csv(_basin_csv)
            for y, grp in bdf.groupby('Year'):
                basin_yearly[int(y)] = {
                    row['Basin']: {k: row[k] for k in row.index
                                   if k not in ('Year', 'Basin')}
                    for _, row in grp.iterrows()}
        if os.path.isfile(_subbasin_csv):
            sbdf = pd.read_csv(_subbasin_csv)
            for y, grp in sbdf.groupby('Year'):
                subbasin_yearly[int(y)] = {
                    row['Subbasin']: {k: row[k] for k in row.index
                                      if k not in ('Year', 'Subbasin')}
                    for _, row in grp.iterrows()}
        logger.info('Loaded annual summaries from %s', _summary_dir)
        return True

    # Check if all rasters already exist — load summaries from CSV
    _first_tif = os.path.join(raster_dirs['mm'], f'Total_Predicted_{START_YEAR}_mm.tif')
    _last_tif = os.path.join(raster_dirs['mm'], f'Total_Predicted_{END_YEAR}_mm.tif')
    _all_tifs_exist = os.path.isfile(_first_tif) and os.path.isfile(_last_tif)
    _skip_loop = _all_tifs_exist and _load_yearly_summary()
    if _skip_loop:
        logger.info('All rasters and summaries exist, skipping prediction loop.')

    # LULC source-mismatch smoothing is now handled upstream in
    # dataops._apply_basin_scale_lulc_delta (baked into the parquet). No
    # partition-time URBAN mutation needed here.

    for year in range(START_YEAR, END_YEAR + 1):
        if _skip_loop:
            break
        # Skip year if raster already exists
        year_tif = os.path.join(raster_dirs['mm'], f'Total_Predicted_{year}_mm.tif')
        if os.path.isfile(year_tif):
            logger.info(f'  Year {year}: rasters exist, skipping.')
            continue

        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            logger.warning(f'No data for year {year}, skipping.')
            continue

        # XGBoost features = create_az_data_parquet columns - DROP_ATTRS - target
        drop_list = [a for a in DROP_ATTRS if a in year_df.columns]
        pred_features = year_df.drop(
            columns=drop_list + ['gw_pumping_mm'],
            errors='ignore'
        )
        # Ensure same columns and order as training
        missing_cols = [c for c in feature_cols if c not in pred_features.columns]
        if missing_cols:
            logger.warning(
                'Year %d: %d missing feature(s) imputed to 0: %s',
                year, len(missing_cols), missing_cols,
            )
        for c in missing_cols:
            pred_features[c] = 0
        pred_features = pred_features[feature_cols]
        inf_counts = np.isinf(pred_features.values).sum(axis=0)
        nan_counts = pred_features.isna().sum(axis=0).values
        bad_cols = (inf_counts + nan_counts) > 0
        if bad_cols.any():
            for ci, col in enumerate(feature_cols):
                if bad_cols[ci]:
                    logger.warning(
                        'Year %d feature %s: %d inf, %d NaN (filled with 0)',
                        year, col, int(inf_counts[ci]), int(nan_counts[ci]),
                    )
        pred_features = pred_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Mirror partitionops well-density overrides in ML features.
        # Single source of truth shared with UQ ensemble members
        # (uncertaintyops._build_pred_features).
        pred_features = partops.apply_ml_well_density_override(
            pred_features, year, year_df, _wd_1981,
        )

        # Collect subsampled features for era-specific interpretability
        for era_name, (y1, y2) in ERA_BOUNDS.items():
            if y1 <= year <= y2:
                n_sample = min(ERA_SAMPLE_PER_YEAR, len(pred_features))
                era_features[era_name].append(
                    pred_features.sample(n=n_sample, random_state=RANDOM_STATE)
                )
                break

        predictions = model.predict(pred_features)
        if LOG_TARGET:
            predictions = np.expm1(predictions)
        predictions = np.abs(predictions)

        # Apply global linear bias correction (only if it improved all metrics)
        if bc_improved:
            predictions = mlops.apply_linear_bc(predictions, bc_m, bc_b)

        # Per-pixel prediction range check against training-era ceiling
        n_pixels = len(predictions)
        exceed_max = int((predictions > train_pixel_max_mm).sum())
        exceed_p99 = int((predictions > train_pixel_p99_mm).sum())
        exceedance_summary.append({
            'year': year,
            'n_pixels': n_pixels,
            'n_exceed_max': exceed_max,
            'pct_exceed_max': round(100.0 * exceed_max / n_pixels, 2) if n_pixels else 0.0,
            'n_exceed_P99': exceed_p99,
            'pct_exceed_P99': round(100.0 * exceed_p99 / n_pixels, 2) if n_pixels else 0.0,
            'pred_max_mm': round(float(predictions.max()), 4) if n_pixels else 0.0,
            'pred_mean_mm': round(float(predictions.mean()), 4) if n_pixels else 0.0,
        })

        # Out-of-distribution detection
        ood_tif = os.path.join(ood_raster_dir, f'OOD_Flag_{year}.tif')
        if not os.path.isfile(ood_tif):
            ood_features = pred_features[_ood_cols]
            ood_stats = ood_detector.score_and_summarise(ood_features, year=year)
            ood_stats['year'] = year
            ood_summary.append(ood_stats)
            # Write OOD probability raster (0 = in-distribution, 1 = OOD)
            ood_flags = ood_detector.ood_probability(ood_features).astype(np.float32)
            ood_raster = _valid_pixels_to_raster(ood_flags, valid_mask, raster_shape)
            _, ood_ref_obj = read_raster_as_arr(ref_raster_file, get_file=True)
            write_raster(
                ood_raster, ood_ref_obj,
                ood_ref_obj.transform, ood_tif,
                no_data_value=np.nan,
                num_bands=1,
            )
            ood_ref_obj.close()

        # Partition into irrigation/non-irrigation and GW/SW categories.
        # year_df already carries basin-delta-corrected LULC columns (URBAN,
        # AGRI, annual_crop_fraction, annual_urban_fraction) from the
        # parquet; no partition-time smoothing needed.
        # Apply CAP delivery perturbation: observed Tier 1/2 cuts
        # (2022-2024) and the WestWater "Basic Coordination" central
        # baseline (factor 0.74 sustained 2026-2099).  Scales
        # canal_weighted_streamflow + sw_rights densities at CAP-
        # served pixels to mirror real shortage allocations that the
        # raw Colorado River gauge data does not capture.  No-op for
        # 1896-2021 and 2025 (no scheduled cut).
        year_df_partition = partops.apply_cap_delivery_perturbation(
            year_df, year, _cap_pixel_mask,
        )
        cat_predictions = partops.partition_predictions(
            predictions, year_df_partition, raster_shape, valid_mask, year=year,
            wd_1981=_wd_1981, irr_wd_1981=_irr_wd_1981,
            nonirr_wd_1981=_nonirr_wd_1981,
            irr_cap_1981=_irr_cap_1981,
        )

        predictions = cat_predictions['Irrigation'] + cat_predictions['Non_Irrigation']

        # Reconstruct raster: valid_mask marks the pixels that survived
        # the 'OUTSIDE AZ' filter in create_az_data_parquet.
        pred_mm = _valid_pixels_to_raster(predictions, valid_mask, raster_shape)
        _write_multi_unit_rasters(
            pred_mm, raster_dirs, 'Total_Predicted', year,
            ref_raster_file, mm_to_ft, mm_to_m3, m3_to_af,
        )

        # Write category rasters (irr, non-irr, irr_gw, irr_sw, nonirr_gw, nonirr_sw)
        for cat, cat_pred in cat_predictions.items():
            cat_mm = _valid_pixels_to_raster(cat_pred, valid_mask, raster_shape)
            _write_multi_unit_rasters(
                cat_mm, cat_raster_dirs[cat], cat, year,
                ref_raster_file, mm_to_ft, mm_to_m3, m3_to_af,
            )

        # ---- Consumptive use (CU = IE × Withdrawal) ----
        # Map basin-level USGS NHM irrigation efficiency to each pixel.
        # For 2000-2020 use per-year basin IE; otherwise use long-term mean.
        ie_per_year = nhm_basin_ie['per_year']
        ie_mean = nhm_basin_ie['mean']
        pixel_basins = year_df['GW_Basin'].values
        pixel_ie = np.full(len(pixel_basins), nhm_basin_ie['overall_mean'],
                           dtype=np.float64)
        if year in ie_per_year:
            for i, b in enumerate(pixel_basins):
                val = ie_per_year[year].get(b, np.nan)
                pixel_ie[i] = val if np.isfinite(val) else ie_mean.get(b, nhm_basin_ie['overall_mean'])
        else:
            for i, b in enumerate(pixel_basins):
                val = ie_mean.get(b, np.nan)
                pixel_ie[i] = val if np.isfinite(val) else nhm_basin_ie['overall_mean']
        pixel_ie = np.clip(pixel_ie, 0, 1)

        cu_total = cat_predictions['Irrigation'] * pixel_ie
        cu_gw = cat_predictions['Irrigation_GW'] * pixel_ie
        cu_sw = cat_predictions['Irrigation_SW'] * pixel_ie

        cu_dict = {
            'Irrigation_CU': cu_total,
            'Irrigation_GW_CU': cu_gw,
            'Irrigation_SW_CU': cu_sw,
        }

        # Write consumptive use rasters (multiple units)
        for cu_cat, cu_pred in cu_dict.items():
            cu_mm = _valid_pixels_to_raster(cu_pred, valid_mask, raster_shape)
            _write_multi_unit_rasters(
                cu_mm, cu_raster_dirs[cu_cat], cu_cat, year,
                ref_raster_file, mm_to_ft, mm_to_m3, m3_to_af,
            )

        # SW Capture Index is now computed downstream in Step 3b's
        # run_uncertainty_quantification → compute_sw_capture_with_sigma
        # (gated on --skip-uq sw-capture-sigma) so that the per-pixel
        # σ_total from the 6-component UQ framework can be propagated
        # through the capture volume bounds.  That step reads the
        # augmented per-category GW rasters (band 1 = pred, band 2 = σ)
        # and writes 6-band augmented SW capture rasters in place, plus
        # SW_Capture_Time_Series.csv / Basin_Capture_Fraction.csv /
        # Subbasin_Capture_Fraction.csv.

        # AZ-wide annual total (all valid pixels)
        yearly_predictions[year] = _pixel_stats(predictions)
        for cat, cat_pred in cat_predictions.items():
            cat_yearly[cat][year] = _pixel_stats(cat_pred)
        for cu_cat, cu_pred in cu_dict.items():
            cu_yearly[cu_cat][year] = _pixel_stats(cu_pred)

        # Collect actual meter data for metered years
        if year in YEAR_LIST and 'gw_pumping_mm' in year_df.columns:
            actuals = year_df['gw_pumping_mm'].values
            actual_yearly[year] = _pixel_stats(actuals)
            act_basin_totals = {}
            act_subbasin_totals = {}
            for basin in all_basins:
                bmask = (year_df.GW_Basin == basin).values
                act_basin_totals[basin] = _pixel_stats(actuals[bmask])
            actual_basin_yearly[year] = act_basin_totals
            for sb in subbasins:
                sbmask = (year_df.GW_Subbasin == sb).values
                act_subbasin_totals[sb] = _pixel_stats(actuals[sbmask])
            actual_subbasin_yearly[year] = act_subbasin_totals

        # Per-basin annual stats (all AZ basins)
        basin_totals = {}
        cat_basin_totals = {cat: {} for cat in partops.CATEGORIES}
        cu_basin_totals = {cat: {} for cat in CU_CATEGORIES}
        for basin in all_basins:
            bmask = (year_df.GW_Basin == basin).values
            basin_totals[basin] = _pixel_stats(predictions[bmask])
            for cat, cat_pred in cat_predictions.items():
                cat_basin_totals[cat][basin] = _pixel_stats(cat_pred[bmask])
            for cu_cat, cu_pred in cu_dict.items():
                cu_basin_totals[cu_cat][basin] = _pixel_stats(cu_pred[bmask])
        basin_yearly[year] = basin_totals
        for cat in partops.CATEGORIES:
            cat_basin_yearly[cat][year] = cat_basin_totals[cat]
        for cu_cat in CU_CATEGORIES:
            cu_basin_yearly[cu_cat][year] = cu_basin_totals[cu_cat]

        # Per-sub-basin annual stats
        subbasin_totals = {}
        cat_subbasin_totals = {cat: {} for cat in partops.CATEGORIES}
        cu_subbasin_totals = {cat: {} for cat in CU_CATEGORIES}
        for sb in subbasins:
            sbmask = (year_df.GW_Subbasin == sb).values
            subbasin_totals[sb] = _pixel_stats(predictions[sbmask])
            for cat, cat_pred in cat_predictions.items():
                cat_subbasin_totals[cat][sb] = _pixel_stats(cat_pred[sbmask])
            for cu_cat, cu_pred in cu_dict.items():
                cu_subbasin_totals[cu_cat][sb] = _pixel_stats(cu_pred[sbmask])
        subbasin_yearly[year] = subbasin_totals
        for cat in partops.CATEGORIES:
            cat_subbasin_yearly[cat][year] = cat_subbasin_totals[cat]
        for cu_cat in CU_CATEGORIES:
            cu_subbasin_yearly[cu_cat][year] = cu_subbasin_totals[cu_cat]

        if year % 20 == 0 or year == END_YEAR:
            vol_af = yearly_predictions[year]['Volume_AF']
            irr_gw = cat_yearly['Irrigation_GW'][year]['Volume_AF']
            irr_sw = cat_yearly['Irrigation_SW'][year]['Volume_AF']
            nigw = cat_yearly['Non_Irrigation_GW'][year]['Volume_AF']
            nisw = cat_yearly['Non_Irrigation_SW'][year]['Volume_AF']
            cu_af = cu_yearly['Irrigation_CU'][year]['Volume_AF']
            logger.info(
                f'  Year {year}: total = {vol_af:,.0f} AF'
                f'  |  irr_GW = {irr_gw:,.0f}  irr_SW = {irr_sw:,.0f}'
                f'  |  non-irr_GW = {nigw:,.0f}  non-irr_SW = {nisw:,.0f}'
                f'  |  CU = {cu_af:,.0f} AF'
            )

    # Save annual summaries to CSV for fast reload on re-runs
    if not _skip_loop:
        _save_yearly_summary()

    # Era summary bar charts are deferred to Step 3g (after UQ)
    # so they can incorporate scenario volume ranges.

    # Write OOD summary CSV
    ood_csv = os.path.join(ood_raster_dir, 'OOD_Summary.csv')
    if ood_summary:
        ood_df = pd.DataFrame(ood_summary)
        ood_df.to_csv(ood_csv, index=False)
    elif os.path.isfile(ood_csv):
        ood_df = pd.read_csv(ood_csv)
        ood_summary = ood_df.to_dict('records')
        logger.info('Loaded existing OOD summary from %s', ood_csv)
    if ood_summary:
        total_ood_pct = ood_df['pct_ood'].mean()
        logger.info(
            'OOD summary: mean %.1f%% OOD pixels across %d years. '
            'Details in %s',
            total_ood_pct, len(ood_df), ood_csv,
        )
        # Flag eras with high OOD rates
        for era, (y1, y2) in [
            ('Hindcast (1896-1983)', (1896, 1983)),
            ('Historical (1984-2025)', (1984, 2025)),
            ('Projection (2026-2099)', (2026, 2099)),
        ]:
            era_df = ood_df[(ood_df['year'] >= y1) & (ood_df['year'] <= y2)]
            if not era_df.empty:
                era_pct = era_df['pct_ood'].mean()
                if era_pct > 10:
                    logger.warning(
                        'OOD %s: %.1f%% mean OOD rate — predictions in this '
                        'era extrapolate substantially beyond training features',
                        era, era_pct,
                    )
                else:
                    logger.info('OOD %s: %.1f%% mean OOD rate', era, era_pct)

        # Time-series plot of OOD percentage by year with era shading
        vizops.apply_journal_style()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ood_df['year'], ood_df['pct_ood'], linewidth=1, color='#d62728')
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('OOD Pixels (%)', fontweight='bold')
        ax.set_title('Out-of-Distribution Pixel Fraction by Year', fontweight='bold')
        era_colors = {'Hindcast': '#2ca02c', 'Training': '#1f77b4', 'Projection': '#ff7f0e'}
        for era_name, (y1, y2) in ERA_BOUNDS.items():
            ax.axvspan(y1, y2, alpha=0.1, color=era_colors.get(era_name, 'gray'),
                       label=era_name)
        ax.legend(loc='upper right')
        ax.set_xlim(START_YEAR, END_YEAR)
        plt.tight_layout()
        fig.savefig(os.path.join(ood_raster_dir, 'OOD_TimeSeries.png'), dpi=600)
        plt.close()
        logger.info('OOD time-series plot saved to %s', ood_raster_dir)

    # Write prediction exceedance summary CSV
    if exceedance_summary:
        exc_df = pd.DataFrame(exceedance_summary)
        exc_csv = os.path.join(prediction_dir, 'Prediction_Exceedance_Summary.csv')
        exc_df.to_csv(exc_csv, index=False)
        logger.info(
            'Prediction exceedance summary (training max=%.2f mm, '
            'P99=%.2f mm) saved to %s',
            train_pixel_max_mm, train_pixel_p99_mm, exc_csv,
        )
        # Per-era exceedance report
        for era, (y1, y2) in ERA_BOUNDS.items():
            era_exc = exc_df[(exc_df['year'] >= y1) & (exc_df['year'] <= y2)]
            if era_exc.empty:
                continue
            mean_pct_max = era_exc['pct_exceed_max'].mean()
            mean_pct_p99 = era_exc['pct_exceed_P99'].mean()
            max_pred = era_exc['pred_max_mm'].max()
            if mean_pct_max > 1:
                logger.warning(
                    'Exceedance %s (%d-%d): %.1f%% pixels exceed training '
                    'max (%.2f mm), %.1f%% exceed P99 (%.2f mm), '
                    'peak prediction=%.2f mm',
                    era, y1, y2, mean_pct_max, train_pixel_max_mm,
                    mean_pct_p99, train_pixel_p99_mm, max_pred,
                )
            else:
                logger.info(
                    'Exceedance %s (%d-%d): %.1f%% exceed max, '
                    '%.1f%% exceed P99, peak=%.2f mm',
                    era, y1, y2, mean_pct_max, mean_pct_p99, max_pred,
                )

    # ---- 3e. Era-specific model interpretability ----
    # Generate SHAP, ALE, and permutation importance plots per era to
    # characterize how feature contributions change between the training
    # window and extrapolation periods (hindcast, projection).
    logger.info('Computing era-specific interpretability plots...')
    for era_name, frames in era_features.items():
        if not frames:
            continue
        era_df = pd.concat(frames, ignore_index=True)
        # Cap at 10 000 samples for computational tractability
        if len(era_df) > 10_000:
            era_df = era_df.sample(n=10_000, random_state=RANDOM_STATE)
        era_dir = os.path.join(interp_dir, era_name)
        y1, y2 = ERA_BOUNDS[era_name]
        logger.info(
            f'  {era_name} ({y1}-{y2}): {len(era_df)} samples'
        )

        # SHAP plots (TreeExplainer; SHAP values remain in log1p space)
        mlops.compute_shap_plots(
            model_name, model, era_df, era_dir,
        )

        # ALE plots (use era features for both train/test since we only
        # need the partial-dependence profiles, not goodness-of-fit)
        y_dummy = model.predict(era_df)
        mlops.compute_ale_plots(
            model_name, model,
            era_df, y_dummy, era_df, y_dummy,
            era_dir, log_target=LOG_TARGET,
        )

        # Permutation importance (uses model predictions as pseudo-targets
        # to measure feature reliance, not prediction accuracy)
        mlops.compute_perm_imp(
            model_name, era_df, era_df, y_dummy, y_dummy,
            model, era_dir, scoring_metric='scaled_rmse',
            random_state=RANDOM_STATE, create_plots=True,
            log_target=LOG_TARGET,
        )

    logger.info(f'Full-period prediction complete. Results in {prediction_dir}')
    return model, feature_cols, x_train, y_train


def create_well_package_step() -> None:
    """Create the per-well GeoPackage from (augmented) prediction rasters.

    Runs *after* UQ augmentation (Step 3b) so that the 6-band rasters are
    available and per-well σ / 95 % CI columns can be included.  When the
    rasters are still single-band (e.g. Step 3b was skipped), deterministic
    columns are written without uncertainty.
    """
    logger.info('=' * 60)
    logger.info('Step 3e: Creating well package (per-well GeoPackage)')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    pixel_area_m2 = MOSAIC_RASTER_RES ** 2

    # Reconstruct raster directory paths (same structure as predict_full_period)
    raster_dirs = {
        'mm': os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_mm'),
        'ft': os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_ft'),
        'm3': os.path.join(prediction_dir, 'Predicted_Rasters', 'Volume_m3'),
        'AF': os.path.join(prediction_dir, 'Predicted_Rasters', 'Volume_AF'),
    }
    cat_raster_dirs = {}
    for cat in partops.CATEGORIES:
        base = os.path.join(prediction_dir, f'{cat}_Rasters')
        cat_raster_dirs[cat] = {
            'mm': os.path.join(base, 'Depth_mm'),
            'ft': os.path.join(base, 'Depth_ft'),
            'm3': os.path.join(base, 'Volume_m3'),
            'AF': os.path.join(base, 'Volume_AF'),
        }
    cu_raster_dirs = {}
    for cu_cat in ('Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU'):
        base = os.path.join(prediction_dir, f'{cu_cat}_Rasters')
        cu_raster_dirs[cu_cat] = {
            'mm': os.path.join(base, 'Depth_mm'),
            'ft': os.path.join(base, 'Depth_ft'),
            'm3': os.path.join(base, 'Volume_m3'),
            'AF': os.path.join(base, 'Volume_AF'),
        }
    # SW capture volume rasters (same unit structure as CU)
    sw_cap_base = os.path.join(prediction_dir, 'SW_Capture')
    for cap_cat in ('Total_SW_Capture', 'Irrigation_SW_Capture',
                    'Non_Irrigation_SW_Capture'):
        base = os.path.join(sw_cap_base, f'{cap_cat}_Rasters')
        if os.path.isdir(os.path.join(base, 'Depth_mm')):
            cu_raster_dirs[cap_cat] = {
                'mm': os.path.join(base, 'Depth_mm'),
                'ft': os.path.join(base, 'Depth_ft'),
                'm3': os.path.join(base, 'Volume_m3'),
                'AF': os.path.join(base, 'Volume_AF'),
            }

    ref_raster_file = os.path.join(PRED_DATA_DIR, f'Predictor_{YEAR_LIST[0]}.tif')
    well_registry_file = os.path.join(
        OUTPUT_DIR, 'GW_Data', 'Vector_Reproj', 'Well_Registry.shp',
    )
    gw_vector_dir = os.path.join(OUTPUT_DIR, f'GW/Vectors/{WNAME}')

    well_pkg_dir = os.path.join(prediction_dir, 'Well_Package')
    well_parquet = wellops.create_well_package(
        well_registry_file,
        raster_dirs=raster_dirs,
        cat_raster_dirs=cat_raster_dirs,
        output_dir=well_pkg_dir,
        ref_raster_file=ref_raster_file,
        pixel_area_m2=pixel_area_m2,
        start_year=START_YEAR,
        end_year=END_YEAR,
        water_use='All',
        gw_vector_dir=gw_vector_dir,
        cu_raster_dirs=cu_raster_dirs,
    )

    # Well package verification skipped: the well-mediated parquet only
    # samples raster values at well-point locations, so it intentionally
    # under-counts the LU-only-pixel contributions present in the
    # statewide raster aggregates.  Verification would always show a
    # parquet-vs-raster mismatch by design.  Confirmed working separately;
    # do not re-enable.


def create_all_raster_maps(skip_maps: set[str] | None = None) -> None:
    """Create era-mean raster maps for every predicted output category
    and an actual-vs-predicted comparison for the metered GW period.

    Iterates over all raster output directories (depth, volume
    partitions, CU, OOD, and uncertainty) and produces 2×2
    era-mean panel figures with basin boundaries and AMA/INA labels.

    Args:
        skip_maps: Optional set of map-generation sub-step tokens to
            skip.  Supported tokens:

              ``trends``
                Skip the full Mann-Kendall + Sen's slope trend-map
                suite (withdrawals, CU, SW capture depth/volume/
                fraction, per-basin and per-sub-basin trend CSVs).
                This is the slowest sub-step in Step 3g and is
                worth skipping when only the era-mean raster maps
                and graphical abstract are needed.

    Returns:
        None.
    """
    skip_maps = skip_maps or set()
    logger.info('=' * 60)
    logger.info('Step 3g: Creating raster maps for all output categories')
    if skip_maps:
        logger.info(f'  Skipping map sub-steps: {sorted(skip_maps)}')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    maps_dir = os.path.join(prediction_dir, 'Raster_Maps')

    # ── Depth-based categories (use Depth_mm sub-directory) ──────────
    depth_categories = [
        ('Predicted_Rasters', 'Total Predicted Annual Withdrawal'),
    ]
    for cat in partops.CATEGORIES:
        pretty = cat.replace('_', ' ')
        depth_categories.append((f'{cat}_Rasters', pretty))

    CU_CATEGORIES = ('Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU')
    for cu in CU_CATEGORIES:
        pretty = cu.replace('_', ' ')
        depth_categories.append((f'{cu}_Rasters', pretty))

    for folder, title in depth_categories:
        raster_dir = os.path.join(prediction_dir, folder, 'Depth_mm')
        if not os.path.isdir(raster_dir):
            logger.info(f'  Skipping {folder}/Depth_mm (not found)')
            continue
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=title,
            unit_label='Depth (mm)',
            cmap='Spectral_r',
        )

    # ── Volume-based categories (use Volume_m3 sub-directory) ─────────
    for folder, title in depth_categories:
        raster_dir = os.path.join(prediction_dir, folder, 'Volume_m3')
        if not os.path.isdir(raster_dir):
            continue
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=f'{title} Volume',
            unit_label=r'Volume (m$^3$)',
            cmap='Spectral_r',
        )

    # ── Volume-based σ (band 2 of the same augmented volume rasters) ──
    # The 6-band augmented volume rasters written by
    # augment_prediction_rasters / augment_category_rasters carry
    # band 2 = σ in m³, so a second pass over the same directories
    # with band=2 produces σ volume era maps that pair 1:1 with the
    # central-value Volume era maps above.  The Purples colormap
    # matches the mm σ std-dev maps rendered from the σ-component
    # rasters later in this step.  Uses the default cbar_extend='both'
    # so the horizontal colorbar renders triangular ends on both sides
    # that match the corresponding central-value Volume maps — the
    # previous 'max' setting produced a visually inconsistent
    # flat-ended colorbar pair when placed next to the non-σ map.
    for folder, title in depth_categories:
        raster_dir = os.path.join(prediction_dir, folder, 'Volume_m3')
        if not os.path.isdir(raster_dir):
            continue
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=f'{title} Volume \u2014 Std Dev',
            unit_label=r'Volume (m$^3$)',
            cmap='Purples',
            band=2,
        )

    # ── OOD Rasters (probability, 0 = in-distribution, 1 = OOD) ─────
    # Uses the dedicated OOD era-map renderer: in practice most AZ
    # pixels saturate at mean OOD ≈ 1 across every era, which compresses
    # the interesting <1 variation into a sliver at the bottom of a
    # continuous colorbar. create_ood_era_raster_maps splits the
    # rendering into two disjoint classes — a continuous 0–0.999 color
    # axis for partial-OOD pixels, and a uniform gray "Fully OOD"
    # legend patch for pixels that reach the saturation threshold —
    # so the sub-saturation dynamic range is legible.
    ood_dir = os.path.join(prediction_dir, 'OOD_Rasters')
    if os.path.isdir(ood_dir):
        vizops.create_ood_era_raster_maps(
            raster_dir=ood_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
        )

    # ── Uncertainty (Sigma components: band 1 = σ, band 2 = CV) ────
    unc_dir = os.path.join(prediction_dir, 'Uncertainty')
    sigma_components = [
        'Sigma_Total', 'Sigma_MACA', 'Sigma_Model',
        'Sigma_Irr', 'Sigma_LULC', 'Sigma_GW', 'Sigma_USBR',
    ]
    for comp in sigma_components:
        raster_dir = os.path.join(unc_dir, comp, 'Rasters')
        if not os.path.isdir(raster_dir):
            continue
        pretty = comp.replace('_', ' ')
        # Band 1: σ (standard deviation in mm).  The σ-component
        # rasters under Uncertainty/{component}/Rasters/ are
        # 1-band only (just σ in mm); there is no band-2 CV
        # channel.  CV at the aggregate level is rendered by the
        # Prediction CV map below (band 3 of the 6-band augmented
        # Total_Predicted rasters under Predicted_Rasters/Depth_mm/).
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=f'{pretty} — Std Dev',
            unit_label='σ (mm)',
            cmap='Purples',
            band=1,
        )

    # ── Augmented prediction rasters (band 3 = CV, band 4 = SNR) ────
    pred_mm_dir = os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_mm')
    if os.path.isdir(pred_mm_dir):
        # Prediction CV: same tight P2–P95 auto-range as the σ
        # component CV maps.  The actual CV distribution for
        # Total_Predicted is in [0, ~0.3] so the colorbar settles
        # naturally to that range.
        # Hard [0, 1] colorbar range for Prediction CV — the measured
        # CV distribution for Total_Predicted is heavy-tailed (P50 ≈
        # 0.42, P95 ≈ 7, max > 6000 at near-zero-prediction pixels), and
        # the P2–P95 auto-range was squeezing the informative 0–1
        # region into the bottom of the colorbar. A hard vmax=1 maps
        # the entire "uncertainty is comparable to the prediction" band
        # (σ ≈ |pred|) to the top color, and pixels above that saturate
        # at the top via the extend='max' arrowhead.  Using
        # extend='both' gives both ends of the colorbar a triangular
        # tip matching the σ volume / SW capture maps — the lower
        # triangle is cosmetic (CV ≥ 0 by construction), but drawing
        # it keeps the colorbar shape visually consistent across the
        # entire era-map suite.
        vizops.create_era_raster_maps(
            raster_dir=pred_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Prediction CV',
            unit_label='CV (σ / |prediction|)',
            cmap='inferno',
            band=3,
            mask_nan_only=True,
            vmin=0.0,
            vmax=1.0,
            cbar_extend='both',
        )
        # Prediction SNR: P5–P95 auto-range + extend='both' so both
        # the low-SNR tail (where the framework is under-constrained
        # and σ ≈ |pred|) and the high-SNR tail (where σ → 0 pushes
        # SNR toward the 20+ range observed in metered corridors) are
        # honestly flagged with triangular arrowheads on both ends.
        # Dropping P5 rather than P2 on the low side keeps the bulk of
        # the distribution (P25≈1.1, P50≈2.4, P75≈4.3) well-centered
        # on the viridis ramp.
        vizops.create_era_raster_maps(
            raster_dir=pred_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Prediction SNR',
            unit_label='SNR (|prediction| / σ)',
            cmap='viridis',
            band=4,
            mask_nan_only=True,
            percentile_clip=(5.0, 95.0),
            cbar_extend='both',
        )

    # ── Actual vs Predicted comparison (metered GW, 1984-2024) ──────
    predicted_mm_dir = os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_mm')
    actual_gw_dir = GW_CROPPED_RASTER_DIR
    if os.path.isdir(actual_gw_dir) and os.path.isdir(predicted_mm_dir):
        vizops.create_actual_vs_predicted_maps(
            actual_dir=actual_gw_dir,
            predicted_dir=predicted_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Total Annual Withdrawal',
            unit_label='Depth (mm)',
            start_year=YEAR_LIST[0],
            end_year=YEAR_LIST[-1],
        )

    # ── SW Capture Index era maps ───────────────────────────────────
    # The category names on disk are Total_SW_Capture,
    # Irrigation_SW_Capture, Non_Irrigation_SW_Capture — the full
    # capture-category strings.  Titles strip the "SW" token so the
    # slug reads "Total Capture", "Irrigation Capture", and
    # "Non-Irrigation Capture" rather than "Total SW SW Capture".
    sw_cap_base = os.path.join(prediction_dir, 'SW_Capture')
    _cap_pretty = {
        'Total_SW_Capture': 'Total SW Capture',
        'Irrigation_SW_Capture': 'Irrigation SW Capture',
        'Non_Irrigation_SW_Capture': 'Non-Irrigation SW Capture',
    }
    for cap_cat, pretty in _cap_pretty.items():
        # Capture fraction (band 2 = central λ=10m).  The default
        # branch of _compute_era_means masks zero-valued pixels as
        # no-data, which is what we want for the fraction map: the
        # vast upland / canal-less area has cw_norm = 0 and
        # therefore a structural zero capture fraction that is not
        # interesting to colormap.  With zeros masked the
        # auto-derived P2–P98 range runs over only the positive
        # river-corridor pixels, so the YlOrRd low end actually
        # engages.  Config matches the central capture volume map
        # below (same cmap, same default percentile_clip, same
        # cbar_extend) per the user's guidance: "make the color
        # ramp for the fractions similar to the SW capture volume".
        # All SW Capture era maps use cbar_extend='both' for visual
        # consistency with the withdrawal-volume, CV, SNR, and σ
        # component era maps.  The lower triangle is cosmetic on
        # non-negative quantities (capture fraction, depth, volume,
        # and σ are ≥ 0 by construction) but keeping it present
        # across every family gives the era-map suite a uniform
        # colorbar shape.
        frac_dir = os.path.join(sw_cap_base, f'{cap_cat}_Fraction')
        if os.path.isdir(frac_dir):
            vizops.create_era_raster_maps(
                raster_dir=frac_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=f'{pretty} Fraction (\u03bb=10m)',
                unit_label='Capture Fraction',
                cmap='YlOrRd',
                band=2,
            )
        # 6-band augmented capture depth rasters: band 1 = central,
        # band 2 = σ, band 3 = CV, bands 5/6 = lower/upper 95 % CI.
        depth_dir = os.path.join(sw_cap_base, f'{cap_cat}_Rasters',
                                 'Depth_mm')
        if os.path.isdir(depth_dir):
            # Band 1: central capture depth
            vizops.create_era_raster_maps(
                raster_dir=depth_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=pretty,
                unit_label='Capture (mm)',
                cmap='YlOrRd',
                band=1,
            )
            # Band 2: σ_cap — uses σ-friendly Purples colormap to
            # match the other σ era maps.
            vizops.create_era_raster_maps(
                raster_dir=depth_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=f'{pretty} — Std Dev',
                unit_label='σ (mm)',
                cmap='Purples',
                band=2,
            )
            # Band 3: CV — same hard [0, 1] clamp as the
            # Prediction CV map, so the informative portion of the
            # SW capture CV distribution is legible and anything
            # above 1.0 is honestly flagged by the right-side
            # arrowhead.  The lower arrowhead is cosmetic (CV ≥ 0)
            # but keeps the colorbar shape uniform across the
            # era-map suite.
            vizops.create_era_raster_maps(
                raster_dir=depth_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=f'{pretty} — CV',
                unit_label='CV (σ / |capture|)',
                cmap='inferno',
                band=3,
                mask_nan_only=True,
                vmin=0.0,
                vmax=1.0,
            )

        # 6-band augmented capture volume rasters (m³): band 1 =
        # central, band 2 = σ in m³.  Pair with the Depth_mm
        # central and σ era maps above so the SW capture story
        # is told in both depth and volume units.
        vol_dir = os.path.join(sw_cap_base, f'{cap_cat}_Rasters',
                               'Volume_m3')
        if os.path.isdir(vol_dir):
            # Band 1: central capture volume
            vizops.create_era_raster_maps(
                raster_dir=vol_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=f'{pretty} Volume',
                unit_label=r'Volume (m$^3$)',
                cmap='YlOrRd',
                band=1,
            )
            # Band 2: σ capture volume (Purples, matches other
            # σ volume era maps)
            vizops.create_era_raster_maps(
                raster_dir=vol_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=maps_dir,
                title=f'{pretty} Volume \u2014 Std Dev',
                unit_label=r'Volume (m$^3$)',
                cmap='Purples',
                band=2,
            )

    # ── Trend analysis (Mann-Kendall + Sen's slope) ────────────────
    # Gated on --skip-maps trends because this is the slowest
    # sub-step in Step 3g (per-pixel MK + Sen on 204 annual rasters
    # × ~15 product families × 4 periods).  Skipping is useful when
    # iterating on era-mean maps or the graphical abstract.
    if 'trends' not in skip_maps:
        trend_dir = os.path.join(maps_dir, 'Trend_Analysis')

        # Conversion factors for the trend-map secondary axes
        _MM_TO_FT = 1.0 / 304.8
        _M3_TO_AF = 1.0 / 1233.48

        # Total predicted annual withdrawal
        if os.path.isdir(predicted_mm_dir):
            vizops.create_trend_maps(
                raster_dir=predicted_mm_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=trend_dir,
                title='Total Predicted Annual Withdrawal',
                unit_label='mm',
                secondary_unit_label='ft',
                secondary_unit_factor=_MM_TO_FT,
                subbasin_shp=ADWR_SUBBASIN_SHP,
            )

        # All depth-based partition categories
        for cat in partops.CATEGORIES:
            cat_dir = os.path.join(prediction_dir, f'{cat}_Rasters',
                                   'Depth_mm')
            if not os.path.isdir(cat_dir):
                continue
            vizops.create_trend_maps(
                raster_dir=cat_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=trend_dir,
                title=cat.replace('_', ' '),
                unit_label='mm',
                secondary_unit_label='ft',
                secondary_unit_factor=_MM_TO_FT,
                subbasin_shp=ADWR_SUBBASIN_SHP,
            )

        # Consumptive Use categories
        for cu in CU_CATEGORIES:
            cu_dir = os.path.join(prediction_dir, f'{cu}_Rasters',
                                  'Depth_mm')
            if not os.path.isdir(cu_dir):
                continue
            vizops.create_trend_maps(
                raster_dir=cu_dir,
                basin_shp=AZ_GW_BASIN,
                output_dir=trend_dir,
                title=cu.replace('_', ' '),
                unit_label='mm',
                secondary_unit_label='ft',
                secondary_unit_factor=_MM_TO_FT,
                subbasin_shp=ADWR_SUBBASIN_SHP,
            )

        # SW Capture Index categories — depth, volume, and fraction
        sw_cap_base = os.path.join(prediction_dir, 'SW_Capture')
        for cap_cat in ('Total_SW_Capture', 'Irrigation_SW_Capture',
                        'Non_Irrigation_SW_Capture'):
            pretty = cap_cat.replace('_', ' ')

            # Depth trends (mm raster source, mm/ft colorbar)
            depth_dir = os.path.join(sw_cap_base, f'{cap_cat}_Rasters',
                                     'Depth_mm')
            if os.path.isdir(depth_dir):
                vizops.create_trend_maps(
                    raster_dir=depth_dir,
                    basin_shp=AZ_GW_BASIN,
                    output_dir=trend_dir,
                    title=f'{pretty} Depth',
                    unit_label='mm',
                    secondary_unit_label='ft',
                    secondary_unit_factor=_MM_TO_FT,
                    subbasin_shp=ADWR_SUBBASIN_SHP,
                )

            # Volume trends (m³ raster source, m³/AF colorbar)
            vol_dir = os.path.join(sw_cap_base, f'{cap_cat}_Rasters',
                                   'Volume_m3')
            if os.path.isdir(vol_dir):
                vizops.create_trend_maps(
                    raster_dir=vol_dir,
                    basin_shp=AZ_GW_BASIN,
                    output_dir=trend_dir,
                    title=f'{pretty} Volume',
                    unit_label=r'm$^3$',
                    secondary_unit_label='AF',
                    secondary_unit_factor=_M3_TO_AF,
                    subbasin_shp=ADWR_SUBBASIN_SHP,
                )

            # Fraction trends (dimensionless, band 2 = central λ=10m)
            frac_dir = os.path.join(sw_cap_base, f'{cap_cat}_Fraction')
            if os.path.isdir(frac_dir):
                vizops.create_trend_maps(
                    raster_dir=frac_dir,
                    basin_shp=AZ_GW_BASIN,
                    output_dir=trend_dir,
                    title=f'{pretty} Fraction',
                    unit_label='fraction',
                    band=2,  # central estimate (λ=10m)
                    subbasin_shp=ADWR_SUBBASIN_SHP,
                )
    else:
        logger.info('  Trend-map suite skipped per --skip-maps trends.')

    # ── σ attribution diagnostic suite ──────────────────────────────
    # Runs LAST in create_all_raster_maps so a failure here cannot
    # cascade into the core era-mean, CV/SNR, actual-vs-predicted,
    # SW-capture, or trend outputs above. Wrapped in try/except so a
    # crash in any one figure family is logged and swallowed — the
    # rest of the step has already completed, and the graphical
    # abstract below still runs.
    #
    # All eight figure families write to Raster_Maps/Sigma_Attribution/
    # so they stay isolated from the main era-mean, trend-analysis, and
    # SW-capture outputs. Basin-level only — sub-basin attribution is
    # deferred because ADWR stewardship decisions are basin-scale.
    #
    # Eight figure families sharing the same data pipeline:
    #   (1) Binary headline withdrawal (Total_GW + Total_SW).
    #   (2) Binary detailed withdrawal (Irrigation_GW/SW,
    #       Non_Irrigation_GW/SW), 4-panel per era.
    #   (3) Binary σ_CU attribution via the IE × Withdrawal
    #       error-propagation decomposition.
    #   (4) Ternary headline withdrawal (continuous RGB mix of
    #       Mgmt/Clim/Model shares).
    #   (5) Ternary detailed withdrawal.
    #   (6) Ternary σ_CU attribution.
    #   (7) Per-basin per-year stacked-area timeseries for eight
    #       headline basins × two pools.
    #   (8) Projection-era stacked-bar variance decomposition.
    try:
        attr_dir = os.path.join(maps_dir, 'Sigma_Attribution')
        os.makedirs(attr_dir, exist_ok=True)

        # (1) Binary headline — Total_GW + Total_SW
        vizops.create_sigma_attribution_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=('Total_GW', 'Total_SW'),
            eras=('Hindcast', 'Historical', 'Projection'),
        )

        # (2) Binary detailed — per-use-type breakdown
        vizops.create_sigma_attribution_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=(
                'Irrigation_GW', 'Irrigation_SW',
                'Non_Irrigation_GW', 'Non_Irrigation_SW',
            ),
            eras=('Hindcast', 'Historical', 'Projection'),
            filename_tag='Detailed',
        )

        # (3) Binary σ_CU — IE × Withdrawal error-propagation
        vizops.create_sigma_cu_attribution_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=(
                'Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU',
            ),
            eras=('Hindcast', 'Historical', 'Projection'),
        )

        # (4) Ternary headline — continuous RGB three-way disclosure
        vizops.create_sigma_attribution_ternary_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=('Total_GW', 'Total_SW'),
            eras=('Hindcast', 'Historical', 'Projection'),
        )

        # (5) Ternary detailed
        vizops.create_sigma_attribution_ternary_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=(
                'Irrigation_GW', 'Irrigation_SW',
                'Non_Irrigation_GW', 'Non_Irrigation_SW',
            ),
            eras=('Hindcast', 'Historical', 'Projection'),
            filename_tag='Detailed',
        )

        # (6) Ternary σ_CU — reuses the same IE propagation as (3)
        vizops.create_sigma_attribution_ternary_map(
            unc_dir=unc_dir, basin_shp=AZ_GW_BASIN, output_dir=attr_dir,
            pools=(
                'Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU',
            ),
            eras=('Hindcast', 'Historical', 'Projection'),
            for_cu=True,
        )

        # (7) Per-year stacked-area timeseries for headline basins
        vizops.create_sigma_attribution_timeseries(
            unc_dir=unc_dir, output_dir=attr_dir,
            pools=('Total_GW', 'Total_SW'),
        )

        # (8) Stacked-bar variance decomposition, Projection era only
        vizops.create_sigma_attribution_bubble(
            unc_dir=unc_dir, output_dir=attr_dir,
            pools=('Total_GW', 'Total_SW'),
            era='Projection',
        )
    except Exception:
        logger.exception(
            '  σ attribution diagnostic suite failed — skipping the '
            'remainder of the attribution figures. Core era maps, '
            'CV/SNR, actual-vs-predicted, SW capture, and trend '
            'analysis are unaffected.',
        )

    # ── Graphical abstract / Figure 1 (after UQ for augmented rasters) ──
    summary_dir = os.path.join(prediction_dir, 'Annual_Summaries')
    total_csv = os.path.join(summary_dir, 'Total_Predicted.csv')
    basin_csv = os.path.join(summary_dir, 'Basin_Total.csv')
    yearly_predictions = {}
    basin_yearly = {}
    if os.path.isfile(total_csv):
        tdf = pd.read_csv(total_csv)
        for _, row in tdf.iterrows():
            yearly_predictions[int(row['Year'])] = {
                k: row[k] for k in row.index if k != 'Year'}
    if os.path.isfile(basin_csv):
        bdf = pd.read_csv(basin_csv)
        for y, grp in bdf.groupby('Year'):
            basin_yearly[int(y)] = {
                row['Basin']: {k: row[k] for k in row.index
                               if k not in ('Year', 'Basin')}
                for _, row in grp.iterrows()}
    # Load category and CU summaries from cached CSVs
    CAT_TITLES = {
        'Irrigation':         'Irrigation',
        'Non_Irrigation':     'Non-Irrigation',
        'Irrigation_GW':      'Irrigation GW',
        'Irrigation_SW':      'Irrigation SW',
        'Non_Irrigation_GW':  'Non-Irrigation GW',
        'Non_Irrigation_SW':  'Non-Irrigation SW',
        'Total_GW':           'Total GW',
        'Total_SW':           'Total SW',
    }
    CU_TITLES = {
        'Irrigation_CU':    'Irrigation Consumptive Use',
        'Irrigation_GW_CU': 'Irrigation GW Consumptive Use',
        'Irrigation_SW_CU': 'Irrigation SW Consumptive Use',
    }

    def _load_yearly_csv(csv_path):
        """Load a yearly summary CSV into {year: {metric: value}} dict."""
        result = {}
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                result[int(row['Year'])] = {
                    k: row[k] for k in row.index if k != 'Year'}
        return result

    cat_yearly = {}
    for cat in CAT_TITLES:
        cat_yearly[cat] = _load_yearly_csv(
            os.path.join(summary_dir, f'{cat}.csv'))
    cu_yearly = {}
    for cu in CU_TITLES:
        cu_yearly[cu] = _load_yearly_csv(
            os.path.join(summary_dir, f'{cu}.csv'))

    # ── Load per-scenario volume projections (if 3b has run) ────────────
    sc_vol_dir = os.path.join(prediction_dir, 'Uncertainty', 'Sigma_LULC',
                              'Scenario_Volumes')
    LULC_SCENARIOS = ['B1', 'B2', 'A1B', 'A2']

    def _load_scenario_volumes(prefix):
        """Load per-scenario CSVs into {scenario: [row_dicts]}."""
        sv = {}
        for sc in LULC_SCENARIOS:
            sc_csv = os.path.join(sc_vol_dir, f'{prefix}_{sc}.csv')
            if os.path.isfile(sc_csv):
                sv[sc] = pd.read_csv(sc_csv).to_dict('records')
        return sv or None

    total_sc_vols = _load_scenario_volumes('Total')

    # ── Era summary bar charts ─────────────────────────────────────────
    if yearly_predictions:
        vizops.create_era_summary_maps(
            yearly_predictions, prediction_dir,
            scenario_volumes=total_sc_vols)
    for cat, title in CAT_TITLES.items():
        if cat_yearly[cat]:
            cat_dir = os.path.join(prediction_dir, cat)
            cat_sc_vols = _load_scenario_volumes(cat)
            vizops.create_era_summary_maps(
                cat_yearly[cat], cat_dir, title_prefix=title,
                scenario_volumes=cat_sc_vols)
    for cu_cat, title in CU_TITLES.items():
        if cu_yearly[cu_cat]:
            cu_dir = os.path.join(prediction_dir, cu_cat)
            vizops.create_era_summary_maps(
                cu_yearly[cu_cat], cu_dir, title_prefix=title)

    # Load UQ-derived σ_total volume time series (if 3b has run)
    sigma_yearly = {}
    sigma_csv = os.path.join(prediction_dir, 'Uncertainty', 'Sigma_Total',
                             'Uncertainty_Summary_Total.csv')
    if os.path.isfile(sigma_csv):
        sdf = pd.read_csv(sigma_csv)
        for _, row in sdf.iterrows():
            sigma_yearly[int(row['Year'])] = {
                k: row[k] for k in row.index if k != 'Year'}

    # ── Graphical abstract / Figure 1 ──────────────────────────────────
    if yearly_predictions:
        vizops.create_graphical_abstract(
            raster_dir=os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_mm'),
            basin_shp=AZ_GW_BASIN,
            output_dir=prediction_dir,
            start_year=START_YEAR,
            end_year=END_YEAR,
            yearly_predictions=yearly_predictions,
            basin_yearly=basin_yearly,
            sigma_yearly=sigma_yearly or None,
        )

    logger.info(f'All raster maps saved to {maps_dir}')


def create_graphical_abstract_only() -> None:
    """Regenerate only the graphical abstract / Figure 1.

    This is a lightweight sub-step of Step 3 that reads the cached
    Annual_Summaries CSVs from disk (Total_Predicted.csv,
    Basin_Total.csv, and the per-scenario volume CSVs under
    Sigma_LULC/Scenario_Volumes/) plus the Uncertainty_Summary_Total.csv
    produced by Step 3b, and calls ``vizops.create_graphical_abstract``
    directly without regenerating the era-mean raster maps, the trend
    analysis, or anything else under ``create_all_raster_maps``.

    Intended for iterating on the Figure 1 layout after Step 3 / 3b
    have already completed.  Runs in ≈ 30 seconds against the on-disk
    caches.

    Returns:
        None.
    """
    logger.info('=' * 60)
    logger.info(
        'Step 3h: Regenerating graphical abstract / Figure 1 only',
    )
    logger.info('=' * 60)

    prediction_dir = os.path.join(
        MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}',
    )
    summary_dir = os.path.join(prediction_dir, 'Annual_Summaries')
    total_csv = os.path.join(summary_dir, 'Total_Predicted.csv')
    basin_csv = os.path.join(summary_dir, 'Basin_Total.csv')

    if not os.path.isfile(total_csv):
        logger.warning(
            'Step 3h skipped: %s not found.  Run Step 3 first to '
            'populate Annual_Summaries/.', total_csv,
        )
        return

    # Load statewide totals
    yearly_predictions: dict = {}
    tdf = pd.read_csv(total_csv)
    for _, row in tdf.iterrows():
        yearly_predictions[int(row['Year'])] = {
            k: row[k] for k in row.index if k != 'Year'
        }

    # Load per-basin totals (optional; only Phoenix AMA is used by the
    # current graphical abstract Panel B but basin_yearly is passed for
    # future-proofing)
    basin_yearly: dict = {}
    if os.path.isfile(basin_csv):
        bdf = pd.read_csv(basin_csv)
        for y, grp in bdf.groupby('Year'):
            basin_yearly[int(y)] = {
                row['Basin']: {
                    k: row[k] for k in row.index
                    if k not in ('Year', 'Basin')
                }
                for _, row in grp.iterrows()
            }

    # Load σ_total time series for the 95 % CI ribbon (optional; if
    # Step 3b has not run, the graphical abstract renders without the
    # uncertainty ribbon)
    sigma_yearly: dict = {}
    sigma_csv = os.path.join(
        prediction_dir, 'Uncertainty', 'Sigma_Total',
        'Uncertainty_Summary_Total.csv',
    )
    if os.path.isfile(sigma_csv):
        sdf = pd.read_csv(sigma_csv)
        for _, row in sdf.iterrows():
            sigma_yearly[int(row['Year'])] = {
                k: row[k] for k in row.index if k != 'Year'
            }

    vizops.create_graphical_abstract(
        raster_dir=os.path.join(
            prediction_dir, 'Predicted_Rasters', 'Depth_mm',
        ),
        basin_shp=AZ_GW_BASIN,
        output_dir=prediction_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        yearly_predictions=yearly_predictions,
        basin_yearly=basin_yearly,
        sigma_yearly=sigma_yearly or None,
    )
    logger.info(
        f'Graphical abstract saved to {prediction_dir}/Graphical_Abstract_Fig1.png',
    )


# =============================================================================
# Step 4 — Intercomparison with USGS datasets
# =============================================================================

def run_usgs_intercomparison() -> pd.DataFrame:
    """
    Compare ML-based irrigation withdrawal predictions with USGS NHM
    (HUC12-scale) and USGS Reitz (county-scale raster) datasets across
    Arizona groundwater basins.

    Returns:
        pd.DataFrame: Summary metrics for every pairwise comparison × category.
    """
    logger.info('='*60)
    logger.info('Step 4: USGS Intercomparison')
    logger.info('='*60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    ml_pred_dir = os.path.join(prediction_dir, 'Predicted_Rasters/Depth_mm')
    irr_gw_dir = os.path.join(prediction_dir, 'Irrigation_GW_Rasters/Depth_mm')
    irr_sw_dir = os.path.join(prediction_dir, 'Irrigation_SW_Rasters/Depth_mm')

    nhm_dir = os.path.join(INPUT_DIR, 'USGS WU/USGS_NHM_Withdrawals')
    reitz_base_dir = os.path.join(INPUT_DIR, 'USGS WU/USGS_Reitz_Irrigation')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')

    output_dir = os.path.join(prediction_dir, 'Withdrawal_Intercomparison')

    return intercompops.run_intercomparison(
        ml_pred_dir=ml_pred_dir,
        nhm_dir=nhm_dir,
        reitz_base_dir=reitz_base_dir,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        output_dir=output_dir,
        irr_gw_dir=irr_gw_dir,
        irr_sw_dir=irr_sw_dir,
        predictor_dir=PRED_DATA_DIR,
    )


def run_cu_usgs_intercomparison() -> pd.DataFrame:
    """
    Compare ML-based Irrigation CU predictions with USGS NHM
    HUC12-scale data across Arizona groundwater basins.

    Returns:
        pd.DataFrame: Summary metrics for CU intercomparison.
    """
    logger.info('=' * 60)
    logger.info('Step 4b: CU Intercomparison')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    irr_cu_dir = os.path.join(prediction_dir, 'Irrigation_CU_Rasters/Depth_mm')

    nhm_cu_csv = os.path.join(INPUT_DIR, 'USGS WU', 'USGS_NHM_CUIrr', 'Irr_CU_HUC12_Tot_annual_2000_2020.csv')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')

    output_dir = os.path.join(prediction_dir, 'CU_Intercomparison')

    return intercompops.run_cu_intercomparison(
        irr_cu_dir=irr_cu_dir,
        nhm_cu_csv=nhm_cu_csv,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        output_dir=output_dir,
        predictor_dir=PRED_DATA_DIR,
    )


def run_cap_srp_sw_validation() -> pd.DataFrame:
    """
    Validate ML Total_SW predictions against observed CAP surface-water
    delivery records across Arizona groundwater basins.  SRP is
    excluded because its service-area boundary is not publicly mapped
    and cannot be unambiguously attributed to a single GW basin.

    Returns:
        pd.DataFrame: Per-basin statistics (RMSD, MAD, Pct Diff, Pearson R).
    """
    logger.info('=' * 60)
    logger.info('Step 4c: CAP Total SW Validation')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    total_sw_dir = os.path.join(prediction_dir, 'Total_SW_Rasters/Depth_mm')

    cap_xlsx = os.path.join(VECTOR_DIR, 'CAP', 'CAP Delivery Data DRI Request.xlsx')

    output_dir = os.path.join(prediction_dir, 'CAP_SRP_Validation')

    return intercompops.run_cap_srp_validation(
        cap_xlsx=cap_xlsx,
        srp_xlsx=None,  # SRP service area not publicly mapped — CAP only
        total_sw_dir=total_sw_dir,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        output_dir=output_dir,
    )


def run_peff_usgs_intercomparison() -> pd.DataFrame:
    """
    Compare irrigated effective precipitation from ML predictions
    (SCS-based and PCML-based) with USGS NHM PPTeff data.

    Returns:
        pd.DataFrame: Summary metrics for Peff intercomparisons.
    """
    logger.info('=' * 60)
    logger.info('Step 4d: Effective Precipitation Intercomparison')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    nhm_peff_csv = os.path.join(
        INPUT_DIR, 'USGS WU', 'USGS_NHM_CUIrr',
        'PPTeff_HUC12_Tot_annual_2000_2020.csv',
    )
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')
    output_dir = os.path.join(prediction_dir, 'Peff_Intercomparison')

    return intercompops.run_peff_intercomparison(
        predictor_dir=PRED_DATA_DIR,
        nhm_peff_csv=nhm_peff_csv,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        output_dir=output_dir,
    )


def run_ps_intercomparison() -> pd.DataFrame:
    """
    Compare ML non-irrigation withdrawal predictions with USGS Public
    Supply reanalysis data (Alzraiee et al. 2024, WRR) for each basin.

    Categories compared:
        Non_Irrigation (total) vs PS Total
        Non_Irrigation_GW vs PS GW
        Non_Irrigation_SW vs PS SW

    Returns:
        pd.DataFrame: Summary metrics for PS intercomparisons.
    """
    logger.info('=' * 60)
    logger.info('Step 4e: Non-Irrigation vs USGS Public Supply Intercomparison')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}')
    nonirr_dir = os.path.join(prediction_dir, 'Non_Irrigation_Rasters/Depth_mm')
    nonirr_gw_dir = os.path.join(prediction_dir, 'Non_Irrigation_GW_Rasters/Depth_mm')
    nonirr_sw_dir = os.path.join(prediction_dir, 'Non_Irrigation_SW_Rasters/Depth_mm')
    ps_data_dir = os.path.join(INPUT_DIR, 'USGS WU', 'USGS_PS_Data')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')
    output_dir = os.path.join(prediction_dir, 'PS_Intercomparison')

    return intercompops.run_ps_intercomparison(
        nonirr_dir=nonirr_dir,
        nonirr_gw_dir=nonirr_gw_dir,
        nonirr_sw_dir=nonirr_sw_dir,
        ps_data_dir=ps_data_dir,
        huc12_geojson=huc12_geojson,
        basin_shp=AZ_GW_BASIN,
        basin_col='BASIN_NAME',
        output_dir=output_dir,
    )


def run_usgs_az_calibration_overview() -> pd.DataFrame:
    """
    AZ-wide annual Total GW & SW bar plots ±1σ vs USGS anchors.

    Mirrors USGS OFR 94-476 (Anning & Duet 1994) Figure 1 in bar form
    and overlays the per-source Circular and OFR 94-476 anchors as a
    direct visual calibration check on the model's statewide annual
    time series (1915–2017).
    """
    logger.info('=' * 60)
    logger.info('Step 4f: USGS Statewide Calibration Overview')
    logger.info('=' * 60)

    prediction_dir = os.path.join(
        MODEL_DIR, f'Full_Prediction_{PREDICTION_MODEL}',
    )
    annual_summaries_dir = os.path.join(prediction_dir, 'Annual_Summaries')
    sigma_rasters_dir = os.path.join(
        prediction_dir, 'Uncertainty', 'Sigma_Total', 'Rasters',
    )
    usgs_csv = os.path.join(
        INPUT_DIR, 'USGS WU', 'USGS_AZ_Water_Use_1950_1980.csv',
    )
    output_dir = os.path.join(prediction_dir, 'USGS_Calibration_Bars')

    return intercompops.run_usgs_az_calibration_overview(
        annual_summaries_dir=annual_summaries_dir,
        usgs_csv=usgs_csv,
        sigma_rasters_dir=sigma_rasters_dir,
        output_dir=output_dir,
        pixel_area_m2=MOSAIC_RASTER_RES ** 2,
    )


# =============================================================================
# Main
# =============================================================================
STEP_HELP = """\
Pipeline steps (comma-separated or 'all'):
  0    Data preparation (GEE download, GW processing)
  1    Create AZ dataset (Parquet)
  2a   Evaluate random 80/20 split
  2a2  Evaluate pixel holdout (spatial locations held out across all years)
  2b   Evaluate LOO temporal holdout
  2c   Evaluate LOO spatial holdout
  2s   Cross-strategy summary (can run standalone from saved results)
  3    Full-period XGBoost prediction (1896-2099)
  3b   Hybrid uncertainty quantification
  3e   Well package (per-well GeoPackage with uncertainty)
  3g   Raster maps, actual vs predicted, and trend analysis
  3h   Graphical abstract / Figure 1 only (lightweight; reads
       Annual_Summaries/ from disk). Must be explicitly requested —
       excluded from --steps all because 3g already produces this
       figure. Intended for iterating on the Figure 1 layout.
  4    USGS intercomparison
  4b   CU intercomparison
  4c   CAP/SRP surface-water validation
  4d   Effective precipitation intercomparison
  4e   Non-irrigation vs USGS Public Supply intercomparison
  4f   USGS statewide calibration overview (AZ-wide annual Total GW/SW
       bars ±1σ vs USGS Circular & OFR 94-476 anchors)

Step 0 sub-steps (use with --skip-prep to skip individual sub-steps):
  gee           GEE tile download & mosaic
  gw-csv        GW CSV -> per-year shapefiles
  vectors       Reproject vectors
  gw-rasters    GW volume -> depth -> cropped rasters
  streamflow    Canal density & streamflow rasters
  basin-rasters GW basin, sub-basin & well density rasters
  wtd           Water table depth raster (Ma et al., 2026)
  reproject     Reproject GEE mosaics to match GW grid

Evaluation sub-steps (use with --skip-eval to skip individual strategies):
  random        Skip random 80/20 evaluation (Step 2a)
  pixel         Skip pixel holdout evaluation (Step 2a2)
  temporal      Skip LOO temporal holdout evaluation (Step 2b)
  spatial       Skip LOO spatial holdout evaluation (Step 2c)
  spatial-seed  Skip seeded LOO spatial holdout evaluation (Step 2c-seed)
  summary       Skip cross-strategy summary

UQ sub-steps (use with --skip-uq to skip individual σ components):
  sigma-maca       Skip σ_MACA (inter-GCM climate spread)
  sigma-model      Skip σ_model (seed ensemble)
  sigma-irr        Skip σ_irr (irrigation fraction uncertainty)
  sigma-lulc       Skip σ_LULC (LULC projection spread)
  sigma-gw         Skip σ_gw (GW fraction snapshot spread)
  sigma-usbr       Skip σ_USBR (Upper-Basin CO River streamflow ensemble; 5 USBR CMIP3 members)
  density-sensitivity   Skip density-ratio partitioning sensitivity (±20%)
  cap-scenario     Skip CAP delivery reduction scenario analysis (WestWater/DCP)
  sigma-total      Skip σ_total quadrature, basin σ, visualizations, and raster augmentation
  sigma-cu         Skip σ_CU (consumptive use uncertainty)
  sw-capture-sigma Skip SW Capture Index computation with σ_GW propagation. This produces
                   the per-pool SW capture rasters (fraction, depth, volume) with
                   combined λ + σ_total 95 % CI bounds and the per-well σ_capture
                   disaggregation.  Depends on sigma-total; skipping means no SW
                   capture outputs are produced.

Step 3g sub-steps (use with --skip-maps to skip individual map families):
  trends           Skip the full Mann-Kendall + Sen's slope trend-map suite
                   (withdrawals, CU, SW capture depth/volume/fraction, per-basin
                   and per-sub-basin trend CSVs).  This is the slowest sub-step
                   in Step 3g — per-pixel MK + Sen on 204 annual rasters ×
                   ~15 product families × 4 periods takes the bulk of the step's
                   runtime, so skipping is useful when iterating on the era-mean
                   raster maps or the graphical abstract.  Era-mean raster maps
                   and the graphical abstract are still produced.
"""


def main() -> None:
    """
    Run the AZ-Hydro pipeline.

    Supports selective step execution via ``--steps``:

        python pipeline.py --steps 0,1,2a
        python pipeline.py --steps 3   # prediction only
        python pipeline.py              # runs all steps

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description='ML Pipeline for Arizona Annual Withdrawal Prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=STEP_HELP,
    )
    parser.add_argument(
        '--steps', type=str, default='all',
        help='Comma-separated step IDs to run (e.g. "0,1,2a") or "all".',
    )
    parser.add_argument(
        '--skip-download', action='store_true', default=True,
        help='Skip GEE tile download (use existing tiles). Default: True.',
    )
    parser.add_argument(
        '--download', dest='skip_download', action='store_false',
        help='Force GEE tile download.',
    )
    parser.add_argument(
        '--load-files', action='store_true', default=True,
        help='Skip recreating intermediate files that already exist. Default: True.',
    )
    parser.add_argument(
        '--recreate', dest='load_files', action='store_false',
        help='Force recreation of intermediate files.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Enable verbose (DEBUG-level) logging.',
    )
    parser.add_argument(
        '--run-eda', action='store_true', default=False,
        help=(
            'Opt in to the Step 1 EDA plot generation (histograms, '
            'ET-vs-ETo analysis, pumping-distribution analysis, '
            'per-basin data-availability summary). EDA is skipped by '
            'default regardless of which steps are selected, because '
            'downstream steps (Step 2, Step 3, Step 3b) reuse the '
            'predictor DataFrame without needing the plots. Pass this '
            'flag when you actually want the EDA figures regenerated.'
        ),
    )
    parser.add_argument(
        '--skip-prep', type=str, default='',
        help=(
            'Comma-separated Step 0 sub-steps to skip: '
            'gee, gw-csv, vectors, gw-rasters, streamflow, '
            'basin-rasters, reproject.'
        ),
    )
    parser.add_argument(
        '--skip-eval', type=str, default='',
        help=(
            'Comma-separated evaluation strategies to skip: '
            'random, pixel, temporal, spatial, spatial-seed, '
            'summary.'
        ),
    )
    parser.add_argument(
        '--skip-uq', type=str, default='',
        help=(
            'Comma-separated UQ sub-steps to skip: '
            'sigma-maca, sigma-model, sigma-irr, sigma-lulc, '
            'sigma-gw, sigma-usbr, density-sensitivity, cap-scenario, '
            'sigma-total, sigma-cu, sw-capture-sigma.'
        ),
    )
    parser.add_argument(
        '--skip-maps', type=str, default='',
        help=(
            'Comma-separated Step 3g map-generation sub-steps to '
            'skip: trends (skip the full Mann-Kendall + Sen\'s slope '
            'trend-map suite, which is the slowest Step 3g sub-step).'
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_all = args.steps.lower() == 'all'
    selected = set(s.strip().lower() for s in args.steps.split(',')) if not run_all else set()
    skip_download = args.skip_download
    load_files = args.load_files
    skip_prep = set(s.strip().lower() for s in args.skip_prep.split(',') if s.strip())
    skip_eval = set(s.strip().lower() for s in args.skip_eval.split(',') if s.strip())
    skip_uq = set(s.strip().lower() for s in args.skip_uq.split(',') if s.strip())
    skip_maps = set(s.strip().lower() for s in args.skip_maps.split(',') if s.strip())

    data_band_names = None

    def should_run(step_id: str) -> bool:
        return run_all or step_id in selected

    # Step 0 — Data preparation
    if should_run('0'):
        data_band_names = prepare_data(
            skip_download=skip_download,
            load_files=load_files,
            verbose=args.verbose,
            skip_prep=skip_prep,
        )

    # Ensure band names are available for downstream steps
    if data_band_names is None:
        _, data_band_names = dataops.download_gee_data(
            os.path.join(VECTOR_DIR, 'AZ.geojson'),
            GCLOUD_PROJECT, GCLOUD_BUCKET,
            INPUT_DIR,
            START_YEAR, END_YEAR,
            skip_download=True,
            tile_size=TILE_SIZE,
        )

    az_df = None

    def get_az_df():
        nonlocal az_df
        if az_df is None:
            # Only regenerate EDA when the user explicitly opts in AND
            # Step 1 is actually in the requested step list. Running
            # `--steps 3b --run-eda` is not meaningful (Step 3b doesn't
            # touch the predictor build) — the flag is a Step 1 opt-in.
            run_eda = args.run_eda and should_run('1')
            az_df = create_az_data(
                data_band_names, load_files=load_files, run_eda=run_eda,
            )
        return az_df

    # Step 1
    if should_run('1'):
        get_az_df()

    # Step 2a — Random
    random_results = None
    if should_run('2a') and 'random' not in skip_eval:
        random_results = evaluate_random(get_az_df())
    elif 'random' in skip_eval:
        logger.info('Skipping Step 2a (random) per --skip-eval.')

    # Step 2a2 — Pixel Holdout
    pixel_results = None
    if should_run('2a2') and 'pixel' not in skip_eval:
        pixel_results = evaluate_pixel_holdout(get_az_df())
    elif 'pixel' in skip_eval:
        logger.info('Skipping Step 2a2 (pixel holdout) per --skip-eval.')

    # Step 2b — LOO Temporal
    temporal_results = None
    if should_run('2b') and 'temporal' not in skip_eval:
        temporal_results = evaluate_temporal_loo(get_az_df())
    elif 'temporal' in skip_eval:
        logger.info('Skipping Step 2b (temporal LOO) per --skip-eval.')

    # Step 2c — LOO Spatial (AMA/INA basins)
    spatial_results = None
    if should_run('2c') and 'spatial' not in skip_eval:
        spatial_results = evaluate_spatial_loo(get_az_df())
    elif 'spatial' in skip_eval:
        logger.info('Skipping Step 2c (spatial LOO) per --skip-eval.')

    # Step 2c-seed — LOO Spatial with 10% seed from held-out basin
    spatial_seed_results = None
    if should_run('2c-seed') and 'spatial-seed' not in skip_eval:
        spatial_seed_results = evaluate_spatial_loo(get_az_df(), seed_fraction=SPATIAL_SEED_FRACTION)
    elif 'spatial-seed' in skip_eval:
        logger.info('Skipping Step 2c-seed (spatial LOO seeded) per --skip-eval.')

    # Step 2s — Cross-strategy summary
    eval_strategies = {
        'Random': random_results,
        'Pixel_Holdout': pixel_results,
        'Temporal_LOO': temporal_results,
        'Spatial_LOO': spatial_results,
        'Spatial_LOO_Seed10': spatial_seed_results,
    }
    eval_strategies = {k: v for k, v in eval_strategies.items() if v is not None}

    # When running 2s standalone, reload results from disk
    if should_run('2s') and not eval_strategies:
        eval_dir = os.path.join(MODEL_DIR, 'Model_Evaluation')
        _disk_map = {
            'Random': ('comparison_df', os.path.join(eval_dir, 'Random', 'Model_Comparison_Averaged.csv')),
            'Pixel_Holdout': ('comparison_df', os.path.join(eval_dir, 'Pixel_Holdout', 'Model_Comparison_Averaged.csv')),
            'Temporal_LOO': ('avg_df', os.path.join(eval_dir, 'Temporal_LOO', 'Averaged_Metrics.csv')),
            'Spatial_LOO': ('avg_df', os.path.join(eval_dir, 'Spatial_LOO', 'Averaged_Metrics.csv')),
            'Spatial_LOO_Seed10': ('avg_df', os.path.join(eval_dir, 'Spatial_LOO_Seed10', 'Averaged_Metrics.csv')),
        }
        for name, (key, csv_path) in _disk_map.items():
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                # Model_Comparison_Averaged has _mean/_std suffixed columns;
                # strip _mean suffix so create_cross_strategy_summary finds them
                if key == 'comparison_df':
                    df.columns = [c[:-5] if c.endswith('_mean') else c
                                  for c in df.columns]
                eval_strategies[name] = {key: df}
                logger.info(f'Loaded {name} results from {csv_path}')

    if (should_run('2s') or run_all) and 'summary' not in skip_eval and len(eval_strategies) >= 3:
        vizops.create_cross_strategy_summary(
            eval_strategies,
            os.path.join(MODEL_DIR, 'Model_Evaluation'),
        )
    elif 'summary' in skip_eval:
        logger.info('Skipping cross-strategy summary per --skip-eval.')

    # Step 3
    model = feature_cols = x_train = y_train = None
    if should_run('3'):
        model, feature_cols, x_train, y_train = predict_full_period(get_az_df())

    # Step 3b — Hybrid uncertainty quantification
    if should_run('3b'):
        if model is None:
            logger.warning('Step 3b requires a trained model from step 3. Skipping.')
        else:
            uncops.run_uncertainty_quantification(
                model=model,
                feature_cols=feature_cols,
                x_train=x_train,
                y_train=y_train,
                az_df=get_az_df(),
                drop_attrs=DROP_ATTRS,
                pred_data_dir=PRED_DATA_DIR,
                model_dir=MODEL_DIR,
                input_dir=INPUT_DIR,
                vector_dir=VECTOR_DIR,
                mosaic_res=MOSAIC_RASTER_RES,
                gcloud_project=GCLOUD_PROJECT,
                gcloud_bucket=GCLOUD_BUCKET,
                tile_size=TILE_SIZE,
                start_year=START_YEAR,
                end_year=END_YEAR,
                year_list=YEAR_LIST,
                fold_count=FOLD_COUNT,
                repeats=REPEATS,
                n_trials=N_TRIALS,
                n_dask_workers=N_DASK_WORKERS,
                use_dask=USE_DASK,
                skip_download=skip_download,
                subbasin_shp=ADWR_SUBBASIN_SHP,
                ama_code_map=AMA_CODE_MAP,
                basin_shp=AZ_GW_BASIN,
                prediction_model=PREDICTION_MODEL,
                skip_uq_steps=skip_uq or None,
            )

    # Step 3e — Well package (after UQ so augmented rasters include σ)
    if should_run('3e'):
        create_well_package_step()

    # Step 3g — Era-mean raster maps for all output categories
    if should_run('3g'):
        create_all_raster_maps(skip_maps=skip_maps or None)

    # Step 3h — Graphical abstract / Figure 1 only (lightweight, reads
    # Annual_Summaries/ and Uncertainty_Summary_Total.csv from disk).
    # Must be *explicitly* requested — it is redundant with Step 3g
    # (which produces the same figure as part of its full raster-map
    # suite), so ``--steps all`` intentionally skips it.
    if '3h' in selected:
        create_graphical_abstract_only()

    # Step 4
    if should_run('4'):
        run_usgs_intercomparison()

    # Step 4b — CU intercomparison
    if should_run('4b'):
        run_cu_usgs_intercomparison()

    # Step 4c — CAP/SRP total surface water validation
    if should_run('4c'):
        run_cap_srp_sw_validation()

    # Step 4d — Effective precipitation intercomparison
    if should_run('4d'):
        run_peff_usgs_intercomparison()

    # Step 4e — Non-irrigation vs USGS Public Supply intercomparison
    if should_run('4e'):
        run_ps_intercomparison()

    # Step 4f — USGS statewide calibration overview (bars vs anchors)
    if should_run('4f'):
        run_usgs_az_calibration_overview()

    logger.info('\n' + '='*60)
    logger.info('Pipeline complete!')
    logger.info('='*60)


if __name__ == '__main__':
    main()
