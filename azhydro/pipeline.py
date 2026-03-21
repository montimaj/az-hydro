"""
ML Pipeline Script for Arizona Groundwater Pumping Prediction.

This script executes the remaining pipeline:
1. Creates dummy annual predictor data from 1896-2099 for AZ and assigns each
   pixel an ADWR groundwater sub-basin label (``GW_Subbasin``).
2. Evaluates tree-based ML models (1984-2024) on four splitting strategies:
   a) Random 80/20 train/test split.
   a2) Pixel holdout — 20% of unique spatial locations held out across all years.
   b) Leave-one-out temporal holdout over multiple test-year ranges (T1-T6),
      reporting per-holdout and averaged metrics.
   c) Leave-one-out spatial holdout over every AMA/INA sub-basin (ADWR),
      reporting per-sub-basin and averaged metrics.
   All strategies use kFolds + Optuna (TPE) + Dask parallelisation.
3. Uses the best model (XGBoost) to predict annual pumping rasters from
   1896-2099 with maps and time series highlighting four eras:
       Hindcast (1896-1983), Historical (1984-2024), Forecast (2025),
       Projected (2026-2099).
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import argparse
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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
MAX_GW = 3000  # max per-pixel GW pumping depth (mm); pixels above this are excluded
LOG_TARGET = True  # log1p-transform target; metrics reported on original scale via expm1

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
RANDOM_STATE = 42
N_EVAL_SEEDS = 5
EVAL_TEST_SIZES = (0.10, 0.15, 0.20, 0.25, 0.30)
N_TRIALS = 50
FOLD_COUNT = 5
REPEATS = 3
N_DASK_WORKERS = 10
N_DASK_WORKERS_DATA_PREP = 40 # more workers for data prep since it involves many independent raster operations
USE_OPTUNA = True
USE_DASK = True
INCLUDE_ALL_MODELS = True

USE_AMA_INA = True
DROP_GW_BASINS = ('WILLCOX AMA', 'HUALAPAI VALLEY INA')

DROP_ATTRS = (
    'Year',
    'GW_Basin',
    'GW_Subbasin',
    'SW',
    'GW_Basin_Type',
    'annual_peff_pcml_mm',
    'northing_m',
    'easting_m',
)

# Temporal holdout configurations (from azhydro.py)
TEMPORAL_HOLDOUTS = {
    'T1': ((2015, 2024),),
    'T2': ((1990, 1992), (2005, 2007), (2022, 2024)),
    'T3': ((2007, 2010),),
    'T4': ((1985, 1989), (2020, 2024)),
    'T5': ((2024, 2024),),
    'T6': ((2010, 2020),),
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
    canal_density_file = streamflowops.create_canal_density_raster(
        grain_parquet=grain_parquet,
        az_boundary_file=az_state,
        output_dir=GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=skip_streamflow,
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
        skip_eda: bool = False,
) -> pd.DataFrame:
    """
    Build the AZ predictor dataframe for years START_YEAR to END_YEAR.

    Calls ``dataops.create_az_data_parquet`` which reads each year's
    Predictor, GW_Basin, GW_Subbasin, Streamflow,
    Canal_Weighted_Streamflow, Canal_Density, and Well_Density rasters,
    then maps ADWR sub-basin OBJECTIDs to names and runs EDA.

    Args:
        data_band_names (list[str]): Band/layer names for predictor rasters.
        load_files (bool): If True, load from cached parquet files.
        skip_eda (bool): If True, skip EDA plot generation.

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

    # EDA
    if not skip_eda:
        vizops.explore_az_data(az_df, os.path.join(MODEL_DIR, 'EDA'))
    else:
        logger.info('Skipping EDA plots (--skip-eda)')

    # ET vs ETo analysis by land use
    vizops.analyze_et_by_land_use(az_df, os.path.join(MODEL_DIR, 'EDA'))

    # Pumping distribution analysis (metered years only)
    vizops.analyze_pumping_distribution(
        az_df, os.path.join(MODEL_DIR, 'EDA'), YEAR_LIST, MAX_GW,
    )

    return az_df


# =============================================================================
# Step 2 — Evaluate tree-based ML models
# =============================================================================

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE_pct': mlops.normalized_rmse(y_true, y_pred),
        'MAE_pct': mlops.normalized_mae(y_true, y_pred),
        'MBE_pct': mlops.normalized_mbe(y_true, y_pred),
    }


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute train/test metrics from a (bias-corrected) prediction DataFrame."""
    train_part = pred_df[pred_df['DATA'] == 'TRAIN']
    test_part = pred_df[pred_df['DATA'] == 'TEST']
    train_m = _compute_metrics(train_part['Actual_GW_mm'].values,
                               train_part['Pred_GW_mm'].values)
    test_m = _compute_metrics(test_part['Actual_GW_mm'].values,
                              test_part['Pred_GW_mm'].values)
    return train_m, test_m


def _train_and_evaluate(
        x_train: pd.DataFrame, y_train: np.ndarray,
        x_test: pd.DataFrame, y_test: np.ndarray,
        model_name: str, output_dir: str,
        cv_groups: np.ndarray | pd.Series | None = None,
) -> dict:
    """Train a single model with Optuna+Dask and return train/test metrics."""
    model, cv_df = mlops.build_ml_model_optuna(
        x_train, y_train, output_dir, model_name,
        random_state=RANDOM_STATE,
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        cv_groups=cv_groups,
        log_target=LOG_TARGET,
    )
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
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
) -> tuple[pd.DataFrame, str]:
    """Run a single random evaluation with the given seed and test size.

    Returns:
        Tuple of (comparison DataFrame, model comparison dir path).
    """
    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
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
        outlier_op=3 if MAX_GW is not None else None,
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
            df, comp_dir = _evaluate_random_single(
                az_df, seed, run_dir,
                tuning_model_dir=tuning_model_dir,
                test_size=ts,
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

    logger.info(f'\nRandom averaged comparison ({n_sizes}×{n_seeds} grid):\n'
                f'{avg_df.to_string(index=False)}')

    all_runs_csv = os.path.join(base_dir, 'All_Runs.csv')
    vizops.plot_grid_boxplots(all_runs_csv, base_dir, strategy_label='Random')

    return {'comparison_df': avg_df, 'strategy': 'Random',
            'all_runs_df': all_runs_df}


# ---- 2a2. Pixel holdout evaluation ----------------------------------------
def _evaluate_pixel_holdout_single(
        az_df: pd.DataFrame,
        random_state: int,
        strategy_dir: str,
        tuning_model_dir: str | None = None,
        test_size: float = 0.2,
) -> tuple[pd.DataFrame, str]:
    """Run a single pixel holdout evaluation with the given seed and test size.

    Returns:
        Tuple of (comparison DataFrame, model comparison dir path).
    """
    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
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
        outlier_op=3 if MAX_GW is not None else None,
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
            df, comp_dir = _evaluate_pixel_holdout_single(
                az_df, seed, run_dir,
                tuning_model_dir=tuning_model_dir,
                test_size=ts,
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

    logger.info(f'\nPixel holdout averaged comparison ({n_sizes}×{n_seeds} grid):\n'
                f'{avg_df.to_string(index=False)}')

    all_runs_csv = os.path.join(base_dir, 'All_Runs.csv')
    vizops.plot_grid_boxplots(all_runs_csv, base_dir, strategy_label='Pixel Holdout')

    return {'comparison_df': avg_df, 'strategy': 'Pixel_Holdout',
            'all_runs_df': all_runs_df}


# ---- 2b. Leave-one-out temporal holdout ------------------------------------
def evaluate_temporal_loo(az_df: pd.DataFrame) -> dict:
    """
    Evaluate each model on every temporal holdout (T1-T6).

    Returns per-holdout and averaged metrics across all holdouts.

    Args:
        az_df (pd.DataFrame): Full predictor dataframe with all years.

    Returns:
        dict: Per-model averaged metrics across all temporal holdouts.
    """
    logger.info('='*60)
    logger.info('Step 2b: LOO Temporal evaluation (T1-T6)')
    logger.info('='*60)

    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
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
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=YEAR_LIST, split_strategy=1,
            test_year=test_years,
            outlier_op=3 if MAX_GW is not None else None,
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

        if len(y_test) == 0:
            logger.warning(f'No test data for {holdout_name}, skipping.')
            continue

        # Group CV by year so inner folds mirror the outer temporal-holdout strategy
        temporal_cv_groups = year_train.values.ravel()

        for model_name in ml_models:
            logger.info(f'  Training {model_name} for {holdout_name}...')
            model_dir = os.path.join(holdout_dir, model_name)
            res = _train_and_evaluate(
                x_train, y_train, x_test, y_test,
                model_name, model_dir,
                cv_groups=temporal_cv_groups,
            )

            # Prediction results + visualisation
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
            mlops.calc_train_test_metrics(
                pred_df, res['cv_df'], model_dir,
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
                create_basin_plots=False,
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
            })

    # Build results DataFrames
    per_holdout_df = pd.DataFrame(per_holdout_rows).round(4)
    per_holdout_df.to_csv(os.path.join(temporal_dir, 'Per_Holdout_Metrics.csv'), index=False)
    logger.info(f'\nPer-holdout metrics:\n{per_holdout_df.to_string(index=False)}')

    # Averaged metrics per model across holdouts
    avg_df = (
        per_holdout_df
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

    # Visualisation: heatmap of Test R² per holdout × model
    vizops.plot_loo_heatmap(
        per_holdout_df, 'Holdout', temporal_dir,
        title='Temporal LOO: Test R² per Holdout',
    )
    vizops.plot_loo_bar(per_holdout_df, 'Holdout', temporal_dir)
    vizops.plot_loo_distribution(
        os.path.join(temporal_dir, 'Per_Holdout_Metrics.csv'),
        'Holdout', temporal_dir, strategy_label='Temporal LOO',
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


def evaluate_spatial_loo(az_df: pd.DataFrame) -> dict:
    """
    Leave-one-out spatial evaluation: hold out each ADWR sub-basin
    within AMA/INA one at a time, train on the rest, evaluate on
    the held-out sub-basin.

    Reports per-sub-basin and averaged metrics.

    Args:
        az_df (pd.DataFrame): Full predictor dataframe with all years.

    Returns:
        dict: Per-model averaged metrics across all spatial holdouts.
    """
    logger.info('='*60)
    logger.info('Step 2c: LOO Spatial evaluation (ADWR sub-basins)')
    logger.info('='*60)

    ml_models = mlops.get_model_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
    )
    subbasins = _get_ama_ina_subbasins()

    # Keep only sub-basins that have data in the metered years
    metered_df = az_df[az_df['Year'].isin(YEAR_LIST)]
    subbasins_with_data = set(metered_df['GW_Subbasin'].unique())
    skipped = [s for s in subbasins if s not in subbasins_with_data]
    subbasins = [s for s in subbasins if s in subbasins_with_data]
    if skipped:
        logger.warning(f'Skipping sub-basins with no data: {skipped}')
    logger.info(f'Sub-basins to evaluate ({len(subbasins)}): {subbasins}')

    spatial_dir = os.path.join(MODEL_DIR, 'Model_Evaluation/Spatial_LOO')
    makedirs(spatial_dir)

    per_subbasin_rows = []

    for subbasin in subbasins:
        logger.info(f'\n--- Spatial holdout: {subbasin} ---')
        subbasin_safe = subbasin.replace(' ', '_').replace('.', '')
        holdout_dir = os.path.join(spatial_dir, subbasin_safe)

        data_dir = os.path.join(holdout_dir, 'data')
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=YEAR_LIST, split_strategy=3,
            test_year=(),
            outlier_op=3 if MAX_GW is not None else None,
            max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
            test_gw_basins=(subbasin,),
            gw_basin_col='GW_Subbasin',
            use_ama_ina=False,  # filtering is handled by subbasin list
            drop_gw_basins=(),
            log_target=LOG_TARGET,
        )
        (x_train, x_test, y_train, y_test,
         x_scaler, y_scaler,
         year_train, year_test,
         basin_train, basin_test,
         easting_train, easting_test,
         northing_train, northing_test) = ret_vals

        if len(y_test) == 0:
            logger.warning(f'No test data for sub-basin {subbasin}, skipping.')
            continue

        logger.info(f'  Train: {len(y_train)}, Test: {len(y_test)}')

        # Group CV by sub-basin so inner folds mirror the outer spatial-holdout strategy
        spatial_cv_groups = basin_train.values.ravel()

        for model_name in ml_models:
            logger.info(f'  Training {model_name} (holdout: {subbasin})...')
            model_dir = os.path.join(holdout_dir, model_name)
            res = _train_and_evaluate(
                x_train, y_train, x_test, y_test,
                model_name, model_dir,
                cv_groups=spatial_cv_groups,
            )

            # Prediction results
            pred_df = mlops.get_prediction_results(
                res['model'], x_train, x_test,
                y_train, y_test, x_scaler, y_scaler,
                year_train, year_test,
                basin_train, basin_test,
                model_dir, model_name,
                gw_basin_col='GW_Subbasin',
                apply_bias_correction=True,
                easting_train=easting_train,
                easting_test=easting_test,
                northing_train=northing_train,
                northing_test=northing_test,
                log_target=LOG_TARGET,
            )
            mlops.calc_train_test_metrics(
                pred_df, res['cv_df'], model_dir,
                use_ama_ina=False,
                gw_basin_col='GW_Subbasin',
                model_name=model_name,
            )

            mlops.generate_model_visualizations(
                pred_df=pred_df,
                output_dir=os.path.join(model_dir, 'Visualizations'),
                model_name=model_name,
                test_case=f'Spatial_LOO_{subbasin_safe}',
                test_year_limits=(),
                gw_basin_col='GW_Subbasin',
                use_ama_ina=False,
                create_basin_plots=False,
            )

            bc_train, bc_test = _metrics_from_pred_df(pred_df)
            logger.info(f'    [BC] Train R2: {bc_train["R2"]:.4f}, '
                        f'Test R2: {bc_test["R2"]:.4f}, '
                        f'Test RMSE: {bc_test["RMSE_pct"]:.2f}%')

            per_subbasin_rows.append({
                'Subbasin': subbasin,
                'Model': model_name,
                'N_test': len(y_test),
                'Train_R2': bc_train['R2'],
                'Test_R2': bc_test['R2'],
                'Train_RMSE': bc_train['RMSE_pct'],
                'Test_RMSE': bc_test['RMSE_pct'],
                'Test_MAE': bc_test['MAE_pct'],
                'Test_MBE': bc_test['MBE_pct'],
                'Overfit_R2': bc_train['R2'] - bc_test['R2'],
            })

    # Build results DataFrames
    per_subbasin_df = pd.DataFrame(per_subbasin_rows).round(4)
    per_subbasin_df.to_csv(os.path.join(spatial_dir, 'Per_Subbasin_Metrics.csv'), index=False)
    logger.info(f'\nPer-sub-basin metrics:\n{per_subbasin_df.to_string(index=False)}')

    # Averaged metrics per model across sub-basins
    avg_df = (
        per_subbasin_df
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

    # Visualisations
    vizops.plot_loo_heatmap(
        per_subbasin_df, 'Subbasin', spatial_dir,
        title='Spatial LOO: Test R² per Sub-basin',
    )
    vizops.plot_loo_bar(per_subbasin_df, 'Subbasin', spatial_dir)
    vizops.plot_loo_distribution(
        os.path.join(spatial_dir, 'Per_Subbasin_Metrics.csv'),
        'Subbasin', spatial_dir, strategy_label='Spatial LOO',
    )

    return {
        'per_subbasin_df': per_subbasin_df,
        'avg_df': avg_df,
        'strategy': 'Spatial_LOO',
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
        )
        raster_file_obj.close()


def predict_full_period(az_df: pd.DataFrame) -> tuple:
    """
    Train XGBoost on the full 1984-2024 metered data (temporal split T1)
    and predict groundwater pumping rasters for every year from 1896 to 2099.

    Returns:
        tuple: (model, feature_cols, x_train, y_train) — the trained XGBoost model,
            feature column names, and training data for uncertainty quantification.
    """
    logger.info('='*60)
    logger.info('Step 3: XGBoost full-period prediction (1896-2099)')
    logger.info('='*60)

    model_name = 'XGB'
    prediction_dir = os.path.join(MODEL_DIR, f'Full_Prediction_{model_name}')
    makedirs(prediction_dir)

    # ---- 3a. Train on ALL 1984-2024 metered data (no holdout) ----
    # Step 2 already provides thorough LOO evaluation; here we maximise
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
        outlier_op=3 if MAX_GW is not None else None,
        max_gw_pumping=MAX_GW if MAX_GW is not None else np.inf,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        log_target=LOG_TARGET,
    )
    x_train, y_train = ret_vals[0], ret_vals[2]

    logger.info(f'Training XGBoost on {len(x_train)} samples '
                f'({YEAR_LIST[0]}-{YEAR_LIST[-1]}, all years)...')
    model, _ = mlops.build_ml_model_optuna(
        x_train, y_train,
        os.path.join(prediction_dir, 'Model'),
        model_name, RANDOM_STATE,
        fold_count=FOLD_COUNT,
        repeats=REPEATS,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
        log_target=LOG_TARGET,
    )

    # ---- Model interpretability plots ----
    interp_dir = os.path.join(prediction_dir, 'Model_Interpretability')

    # Feature importance + permutation importance
    mlops.compute_perm_imp(
        model_name, x_train, x_train, y_train, y_train,
        model, interp_dir, scoring_metric='scaled_rmse',
        random_state=RANDOM_STATE, create_plots=True,
        log_target=LOG_TARGET,
    )

    # ALE plots
    mlops.compute_ale_plots(
        model_name, model,
        x_train, y_train, x_train, y_train,
        interp_dir, log_target=LOG_TARGET,
    )

    # SHAP plots (TreeExplainer; SHAP values remain in log1p space)
    mlops.compute_shap_plots(
        model_name, model, x_train, interp_dir,
        log_target=LOG_TARGET,
    )

    # ---- 3b. Predict pumping for each year 1896-2099 ----
    logger.info('Predicting pumping for all years 1896-2099...')

    # Fit out-of-distribution detector on training features
    ood_detector = mlops.OODDetector(alpha=0.01)
    ood_detector.fit(x_train)

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
    # from ALL AMA/INA training data so it generalises to non-AMA/INA basins
    # during full-period prediction.
    # Bias correction operates in original (mm) scale.
    train_preds = np.abs(model.predict(x_train))
    if LOG_TARGET:
        train_preds = np.abs(np.expm1(train_preds))

    bc_dir = os.path.join(prediction_dir, 'Bias_Correction')
    makedirs(bc_dir)

    bc_m, bc_b = mlops.fit_linear_bc(train_preds, y_train_orig)

    raw_rmse = float(np.sqrt(np.mean((y_train_orig - train_preds) ** 2)))
    linear_corrected = mlops.apply_linear_bc(train_preds, bc_m, bc_b)
    linear_rmse = float(np.sqrt(np.mean((y_train_orig - linear_corrected) ** 2)))

    bc_summary = pd.DataFrame([{
        'Slope': round(bc_m, 4),
        'Intercept': round(bc_b, 4),
        'N_Samples': len(train_preds),
        'Raw_RMSE': round(raw_rmse, 4),
        'Linear_RMSE': round(linear_rmse, 4),
    }])
    bc_summary.to_csv(os.path.join(bc_dir, 'Global_BC_Summary.csv'), index=False)
    logger.info(
        'Raw RMSE=%.4f, Linear RMSE=%.4f, N=%d',
        raw_rmse, linear_rmse, len(train_preds),
    )

    feature_cols = list(x_train.columns)
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
    # In create_az_data_parquet, pixels with NaN or 0 in GW_Basin are labelled
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

    def _pixel_stats(pred_vals):
        """Compute depth and volume stats in multiple units.

        Returns NaN for all fields when *pred_vals* is empty (no valid
        pixels), so "no data" is distinguishable from "zero pumping".
        """
        n = len(pred_vals)
        if n == 0:
            return {
                'Mean_Depth_mm': np.nan,
                'Mean_Depth_ft': np.nan,
                'Volume_m3': np.nan,
                'Volume_AF': np.nan,
            }
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
        'Training': (1984, 2024),
        'Projection': (2025, END_YEAR),
    }
    era_features: dict[str, list[pd.DataFrame]] = {e: [] for e in ERA_BOUNDS}
    ERA_SAMPLE_PER_YEAR = 2000  # max pixels sampled per year per era

    for year in range(START_YEAR, END_YEAR + 1):
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

        # Apply global linear bias correction
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
        ood_stats = ood_detector.score_and_summarise(pred_features, year=year)
        ood_stats['year'] = year
        ood_summary.append(ood_stats)
        # Write OOD flag raster (1 = OOD, 0 = in-distribution)
        ood_flags = ood_detector.is_ood(pred_features).astype(np.float32)
        ood_raster = _valid_pixels_to_raster(ood_flags, valid_mask, raster_shape)
        _, ood_ref_obj = read_raster_as_arr(ref_raster_file, get_file=True)
        write_raster(
            ood_raster, ood_ref_obj,
            ood_ref_obj.transform,
            os.path.join(ood_raster_dir, f'OOD_Flag_{year}.tif'),
            no_data_value=np.nan,
        )
        ood_ref_obj.close()

        # Partition into irrigation/non-irrigation and GW/SW categories
        cat_predictions = partops.partition_predictions(
            predictions, year_df, raster_shape, valid_mask,
        )
        predictions = cat_predictions['Irrigation'] + cat_predictions['Non_Irrigation']

        # Reconstruct raster: valid_mask marks the pixels that survived
        # the 'OUTSIDE AZ' filter in create_az_data_parquet.
        pred_mm = _valid_pixels_to_raster(predictions, valid_mask, raster_shape)
        _write_multi_unit_rasters(
            pred_mm, raster_dirs, 'Predicted_GW', year,
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

    # ---- 3c. Era summary maps (time series plots deferred to UQ step) ----
    vizops.create_era_summary_maps(yearly_predictions, prediction_dir)

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
    for cat, title in CAT_TITLES.items():
        cat_dir = os.path.join(prediction_dir, cat)
        vizops.create_era_summary_maps(
            cat_yearly[cat], cat_dir,
            title_prefix=title,
        )

    CU_TITLES = {
        'Irrigation_CU':    'Irrigation Consumptive Use',
        'Irrigation_GW_CU': 'Irrigation GW Consumptive Use',
        'Irrigation_SW_CU': 'Irrigation SW Consumptive Use',
    }
    for cu_cat, title in CU_TITLES.items():
        cu_dir = os.path.join(prediction_dir, cu_cat)
        vizops.create_era_summary_maps(
            cu_yearly[cu_cat], cu_dir,
            title_prefix=title,
        )

    # Write OOD summary CSV
    if ood_summary:
        ood_df = pd.DataFrame(ood_summary)
        ood_csv = os.path.join(ood_raster_dir, 'OOD_Summary.csv')
        ood_df.to_csv(ood_csv, index=False)
        total_ood_pct = ood_df['pct_ood'].mean()
        logger.info(
            'OOD summary: mean %.1f%% OOD pixels across %d years. '
            'Details in %s',
            total_ood_pct, len(ood_df), ood_csv,
        )
        # Flag eras with high OOD rates
        for era, (y1, y2) in [
            ('Hindcast (1896-1983)', (1896, 1983)),
            ('Training (1984-2024)', (1984, 2024)),
            ('Projection (2025-2099)', (2025, 2099)),
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
    # characterise how feature contributions change between the training
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
            log_target=LOG_TARGET,
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

    # ---- 3f. Graphical abstract / Figure 1 ----
    vizops.create_graphical_abstract(
        raster_dir=raster_dirs['mm'],
        basin_shp=AZ_GW_BASIN,
        output_dir=prediction_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        ref_raster=ref_raster_file,
        yearly_predictions=yearly_predictions,
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

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
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

    ref_raster_file = os.path.join(PRED_DATA_DIR, f'Predictor_{YEAR_LIST[0]}.tif')
    well_registry_file = os.path.join(
        OUTPUT_DIR, 'GW_Data', 'Vector_Reproj', 'Well_Registry.shp',
    )
    gw_vector_dir = os.path.join(OUTPUT_DIR, f'GW/Vectors/{WNAME}')

    wellops.create_well_package(
        well_registry_file,
        raster_dirs=raster_dirs,
        cat_raster_dirs=cat_raster_dirs,
        output_dir=os.path.join(prediction_dir, 'Well_Package'),
        ref_raster_file=ref_raster_file,
        pixel_area_m2=pixel_area_m2,
        start_year=START_YEAR,
        end_year=END_YEAR,
        water_use='All' if WATER_USE == 'All' else 'IRRIGATION',
        gw_vector_dir=gw_vector_dir,
        cu_raster_dirs=cu_raster_dirs,
    )


def create_all_raster_maps() -> None:
    """Create era-mean raster maps for every predicted output category
    and an actual-vs-predicted comparison for the metered GW period.

    Iterates over all raster output directories (depth, volume
    partitions, CU, OOD, and uncertainty) and produces 2×2
    era-mean panel figures with basin boundaries and AMA/INA labels.

    Returns:
        None.
    """
    logger.info('=' * 60)
    logger.info('Step 3g: Creating raster maps for all output categories')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
    maps_dir = os.path.join(prediction_dir, 'Raster_Maps')

    # ── Depth-based categories (use Depth_mm sub-directory) ──────────
    depth_categories = [
        ('Predicted_Rasters', 'Total Predicted GW Pumping'),
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
            cmap='YlOrRd',
        )

    # ── OOD Rasters (binary flags) ──────────────────────────────────
    ood_dir = os.path.join(prediction_dir, 'OOD_Rasters')
    if os.path.isdir(ood_dir):
        vizops.create_era_raster_maps(
            raster_dir=ood_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Out-of-Distribution Flag',
            unit_label='Mean OOD Fraction',
            cmap='RdYlGn_r',
        )

    # ── Uncertainty (Sigma components: band 1 = σ, band 2 = CV) ────
    unc_dir = os.path.join(prediction_dir, 'Uncertainty')
    sigma_components = [
        'Sigma_Total', 'Sigma_MACA', 'Sigma_Model',
        'Sigma_Irr', 'Sigma_LULC', 'Sigma_GW',
    ]
    for comp in sigma_components:
        raster_dir = os.path.join(unc_dir, comp, 'Rasters')
        if not os.path.isdir(raster_dir):
            continue
        pretty = comp.replace('_', ' ')
        # Band 1: σ (standard deviation in mm)
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=f'{pretty} — Std Dev',
            unit_label='σ (mm)',
            cmap='Purples',
            band=1,
        )
        # Band 2: CV (coefficient of variation)
        vizops.create_era_raster_maps(
            raster_dir=raster_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title=f'{pretty} — CV',
            unit_label='CV (σ / |prediction|)',
            cmap='inferno',
            band=2,
            mask_nan_only=True,
        )

    # ── Augmented prediction rasters (band 3 = CV, band 4 = SNR) ────
    pred_mm_dir = os.path.join(prediction_dir, 'Predicted_Rasters', 'Depth_mm')
    if os.path.isdir(pred_mm_dir):
        vizops.create_era_raster_maps(
            raster_dir=pred_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Prediction CV',
            unit_label='CV (σ / |prediction|)',
            cmap='inferno',
            band=3,
            mask_nan_only=True,
        )
        vizops.create_era_raster_maps(
            raster_dir=pred_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=maps_dir,
            title='Prediction SNR',
            unit_label='SNR (|prediction| / σ)',
            cmap='viridis',
            band=4,
            mask_nan_only=True,
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
            title='Total GW Pumping',
            unit_label='Depth (mm)',
            start_year=YEAR_LIST[0],
            end_year=YEAR_LIST[-1],
        )

    # ── Trend analysis (Mann-Kendall + Sen's slope) ────────────────
    trend_dir = os.path.join(maps_dir, 'Trend_Analysis')

    # Total predicted GW pumping
    if os.path.isdir(predicted_mm_dir):
        vizops.create_trend_maps(
            raster_dir=predicted_mm_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=trend_dir,
            title='Total Predicted GW Pumping',
            unit_label='mm',
            subbasin_shp=ADWR_SUBBASIN_SHP,
        )

    # All depth-based partition categories
    for cat in partops.CATEGORIES:
        cat_dir = os.path.join(prediction_dir, f'{cat}_Rasters', 'Depth_mm')
        if not os.path.isdir(cat_dir):
            continue
        vizops.create_trend_maps(
            raster_dir=cat_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=trend_dir,
            title=cat.replace('_', ' '),
            unit_label='mm',
            subbasin_shp=ADWR_SUBBASIN_SHP,
        )

    # Consumptive Use categories
    for cu in CU_CATEGORIES:
        cu_dir = os.path.join(prediction_dir, f'{cu}_Rasters', 'Depth_mm')
        if not os.path.isdir(cu_dir):
            continue
        vizops.create_trend_maps(
            raster_dir=cu_dir,
            basin_shp=AZ_GW_BASIN,
            output_dir=trend_dir,
            title=cu.replace('_', ' '),
            unit_label='mm',
            subbasin_shp=ADWR_SUBBASIN_SHP,
        )

    logger.info(f'All raster maps saved to {maps_dir}')


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

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
    ml_pred_dir = os.path.join(prediction_dir, 'Predicted_Rasters/Depth_mm')
    irr_gw_dir = os.path.join(prediction_dir, 'Irrigation_GW_Rasters/Depth_mm')
    irr_sw_dir = os.path.join(prediction_dir, 'Irrigation_SW_Rasters/Depth_mm')

    nhm_dir = os.path.join(INPUT_DIR, 'USGS WU/USGS_NHM_Withdrawals')
    reitz_base_dir = os.path.join(INPUT_DIR, 'USGS WU/USGS_Reitz_Irrigation')
    huc12_geojson = os.path.join(INPUT_DIR, 'GEE_Data', 'AZ_HUC12.geojson')

    output_dir = os.path.join(prediction_dir, 'Intercomparison')

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

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
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
    Validate ML Total_SW predictions against observed CAP and SRP
    surface-water delivery records across Arizona groundwater basins.

    Returns:
        pd.DataFrame: Per-basin statistics (RMSD, MAD, Pct Diff, Pearson R).
    """
    logger.info('=' * 60)
    logger.info('Step 4c: CAP/SRP Total SW Validation')
    logger.info('=' * 60)

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
    total_sw_dir = os.path.join(prediction_dir, 'Total_SW_Rasters/Depth_mm')

    cap_xlsx = os.path.join(VECTOR_DIR, 'CAP', 'CAP Delivery Data DRI Request.xlsx')
    srp_xlsx = os.path.join(VECTOR_DIR, 'SRP', 'SRP WATER DELVS HISTORY.xlsx')

    output_dir = os.path.join(prediction_dir, 'CAP_SRP_Validation')

    return intercompops.run_cap_srp_validation(
        cap_xlsx=cap_xlsx,
        srp_xlsx=srp_xlsx,
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

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
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

    prediction_dir = os.path.join(MODEL_DIR, 'Full_Prediction_XGB')
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
  3    Full-period XGBoost prediction (1896-2099)
  3b   Hybrid uncertainty quantification
  3e   Well package (per-well GeoPackage with uncertainty)
  3g   Raster maps, actual vs predicted, and trend analysis
  4    USGS intercomparison
  4b   CU intercomparison
  4c   CAP/SRP surface-water validation
  4d   Effective precipitation intercomparison
  4e   Non-irrigation vs USGS Public Supply intercomparison

Step 0 sub-steps (use with --skip-prep to skip individual sub-steps):
  gee           GEE tile download & mosaic
  gw-csv        GW CSV -> per-year shapefiles
  vectors       Reproject vectors
  gw-rasters    GW volume -> depth -> cropped rasters
  streamflow    Canal density & streamflow rasters
  basin-rasters GW basin, sub-basin & well density rasters
  reproject     Reproject GEE mosaics to match GW grid

Evaluation sub-steps (use with --skip-eval to skip individual strategies):
  random        Skip random 80/20 evaluation (Step 2a)
  pixel         Skip pixel holdout evaluation (Step 2a2)
  temporal      Skip LOO temporal holdout evaluation (Step 2b)
  spatial       Skip LOO spatial holdout evaluation (Step 2c)
  summary       Skip cross-strategy summary
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
        description='ML Pipeline for Arizona Groundwater Pumping Prediction.',
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
        '--skip-eda', action='store_true', default=False,
        help='Skip EDA plot generation in Step 1.',
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
            'random, pixel, temporal, spatial, summary.'
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
            az_df = create_az_data(data_band_names, load_files=load_files, skip_eda=args.skip_eda)
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

    # Step 2c — LOO Spatial (ADWR sub-basins)
    spatial_results = None
    if should_run('2c') and 'spatial' not in skip_eval:
        spatial_results = evaluate_spatial_loo(get_az_df())
    elif 'spatial' in skip_eval:
        logger.info('Skipping Step 2c (spatial LOO) per --skip-eval.')

    # Cross-strategy summary (only if all evaluations ran)
    eval_strategies = {
        'Random': random_results,
        'Pixel_Holdout': pixel_results,
        'Temporal_LOO': temporal_results,
        'Spatial_LOO': spatial_results,
    }
    eval_strategies = {k: v for k, v in eval_strategies.items() if v is not None}
    if 'summary' not in skip_eval and len(eval_strategies) >= 3:
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
            )

    # Step 3e — Well package (after UQ so augmented rasters include σ)
    if should_run('3e'):
        create_well_package_step()

    # Step 3g — Era-mean raster maps for all output categories
    if should_run('3g'):
        create_all_raster_maps()

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

    logger.info('\n' + '='*60)
    logger.info('Pipeline complete!')
    logger.info('='*60)


if __name__ == '__main__':
    main()
