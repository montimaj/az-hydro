"""
ML Pipeline Script for Arizona Groundwater Pumping Prediction.

This script executes the remaining pipeline:
1. Creates dummy annual predictor data from 1896-2099 for AZ and assigns each
   pixel an ADWR groundwater sub-basin label (``GW_Subbasin``).
2. Evaluates tree-based ML models (1984-2024) on three splitting strategies:
   a) Random 80/20 train/test split.
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

import sys
import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import r2_score

import hydrolibs.dataops as dataops
import hydrolibs.mlops as mlops
import hydrolibs.visualops as vizops
import hydrolibs.partitionops as partops
import hydrolibs.wellops as wellops
import hydrolibs.gwops as gwops
import hydrolibs.streamflowops as streamflowops
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
VECTOR_DIR = f'{INPUT_DIR}GW_Data/'
MOSAIC_RASTER_RES = 2000
GEE_MOSAIC_DIR = f'{OUTPUT_DIR}GEE_Mosaics_{int(MOSAIC_RASTER_RES)}m/'
GW_DEPTH_RASTER_DIR = f'{OUTPUT_DIR}GW/Rasters/GW_Depths_{WNAME}_{int(MOSAIC_RASTER_RES)}m/'
PRED_DATA_DIR = f'{OUTPUT_DIR}Predictor_Data_{WNAME}_{int(MOSAIC_RASTER_RES)}m/'
MODEL_DIR = f'{OUTPUT_DIR}ML_Model_{WNAME}_{int(MOSAIC_RASTER_RES)}m/'
GW_CROPPED_RASTER_DIR = f'{GW_DEPTH_RASTER_DIR}Cropped/'

AZ_GW_BASIN = f'{OUTPUT_DIR}GW_Data/Vector_Reproj/Groundwater_Basin.shp'
ADWR_SUBBASIN_SHP = f'{OUTPUT_DIR}GW_Data/Vector_Reproj/ADWR_Groundwater_Subbasin.shp'

GCLOUD_PROJECT = 'azhydro'
GCLOUD_BUCKET = 'azhydro'
TILE_SIZE = 10000 if MOSAIC_RASTER_RES == 30 else 80000
FILL_ATTR = 'AF Pumped'
MAX_GW = 3000 if WATER_USE == 'All' else None  # 3000 mm (~10 ft)

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
N_TRIALS = 100
FOLD_COUNT = 5
N_DASK_WORKERS = 10
USE_OPTUNA = True
USE_DASK = True
INCLUDE_ALL_MODELS = False

USE_AMA_INA = True
DROP_GW_BASINS = ('WILLCOX AMA', 'HUALAPAI VALLEY INA')

DROP_ATTRS = (
    'Year',
    'GW_Basin',
    'GW_Subbasin',
    'easting_m',
    'SW',
    'GW_Basin_Type',
    'annual_peff_pcml_mm',
    'annual_prism_tmmx_K',
    'annual_prism_tmmn_K',
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

def prepare_data(skip_download: bool = True, load_files: bool = True) -> list[str]:
    """
    Download GEE data, preprocess GW CSVs, reproject vectors, and create
    all intermediate rasters needed by the ML pipeline.

    Parameters
    ----------
    skip_download : bool
        If *True*, skip the GEE download (use existing tiles).
    load_files : bool
        If *True*, skip recreating files that already exist on disk.

    Returns
    -------
    list[str]
        GEE data band names (needed by ``create_az_data``).
    """
    logger.info('='*60)
    logger.info('Step 0: Data preparation')
    logger.info('='*60)

    az_state_raw = f'{VECTOR_DIR}AZ.geojson'
    well_reg_file = f'{VECTOR_DIR}Well_Registry_2024/Well_Registry.shp'
    gw_csv_dir = f'{VECTOR_DIR}Meter Data/'
    grain_parquet = f'{VECTOR_DIR}GRAIN_v.1.0/GeoParquet/us-west_GRAIN_v.1.0.parquet'
    output_gw_vector_dir = f'{OUTPUT_DIR}GW/Vectors/{WNAME}/'
    vector_reproj_dir = f'{OUTPUT_DIR}GW_Data/Vector_Reproj/'
    output_gw_volume_dir = (
        f'{OUTPUT_DIR}GW/Rasters/GW_Volumes_{WNAME}_{int(MOSAIC_RASTER_RES)}m/'
    )

    # GEE download & mosaic
    gee_data_dir, data_band_names = dataops.download_gee_data(
        az_state_raw,
        GCLOUD_PROJECT,
        GCLOUD_BUCKET,
        INPUT_DIR,
        START_YEAR,
        END_YEAR,
        skip_download,
        TILE_SIZE,
        num_workers=40,
        worker_memory='1G',
        gee_scale=MOSAIC_RASTER_RES,
        verbose=False,
    )
    dataops.mosaic_tiles(
        gee_data_dir,
        GEE_MOSAIC_DIR,
        START_YEAR,
        END_YEAR,
        already_mosaicked=load_files,
    )

    # GW CSV → per-year shapefiles
    ref_gw_file = gwops.preprocess_gw_csv(
        well_reg_file,
        gw_csv_dir,
        output_gw_vector_dir,
        fill_attr=FILL_ATTR,
        use_only_ama_ina=False,
        already_preprocessed=load_files,
        water_use=WATER_USE,
    )

    # Reproject vectors
    az_vector_reproj = gwops.reproject_vectors(
        VECTOR_DIR,
        vector_reproj_dir,
        ref_file=ref_gw_file,
        already_reprojected=load_files,
    )
    well_reg_file = az_vector_reproj['Well_Registry']
    az_gw_basin = az_vector_reproj['Groundwater_Basin']
    az_sw_watershed = az_vector_reproj['Surface_Watershed']
    cap_service_area = az_vector_reproj['CAP_Service_Area']
    az_state = az_vector_reproj['AZ']

    # GW volume → depth → cropped rasters
    gwops.create_gw_volume_rasters(
        output_gw_vector_dir,
        output_gw_volume_dir,
        value_field=FILL_ATTR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        already_created=load_files,
        max_gw=MAX_GW,
    )
    gwops.create_gw_depth_rasters(
        output_gw_volume_dir,
        GW_DEPTH_RASTER_DIR,
        already_created=load_files,
    )
    gwops.crop_gw_rasters(
        GW_DEPTH_RASTER_DIR,
        GW_DEPTH_RASTER_DIR,
        az_state_file=az_state,
        already_cropped=load_files,
    )
    load_files = False

    # Canal density & streamflow rasters
    canal_density_file = streamflowops.create_canal_density_raster(
        grain_parquet=grain_parquet,
        az_boundary_file=az_state,
        output_dir=GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=load_files,
    )
    streamflowops.create_streamflow_rasters(
        watershed_geojson=az_sw_watershed,
        cap_service_area_geojson=cap_service_area,
        sites_csv=f'{VECTOR_DIR}Streamflow/sites.csv',
        output_dir=GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        canal_density_file=canal_density_file,
        already_created=load_files,
    )

    # GW basin, sub-basin & well density rasters
    adwr_subbasin_shp = f'{vector_reproj_dir}ADWR_Groundwater_Subbasin.shp'
    gwops.create_gw_basin_rasters(
        az_gw_basin,
        GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=load_files,
        subbasin_vector=adwr_subbasin_shp,
    )
    gwops.create_well_density_raster(
        well_reg_file,
        GEE_MOSAIC_DIR,
        xres=MOSAIC_RASTER_RES,
        yres=MOSAIC_RASTER_RES,
        start_year=START_YEAR,
        end_year=END_YEAR,
        already_created=load_files,
    )

    # Reproject GEE mosaics to match GW raster grid
    dataops.reproject_gee_mosaics(
        GEE_MOSAIC_DIR,
        PRED_DATA_DIR,
        GW_CROPPED_RASTER_DIR,
        already_reprojected=load_files,
    )

    logger.info('Step 0 complete.')
    return data_band_names


# =============================================================================
# Step 1 — Create AZ predictor data (1896-2099)
# =============================================================================


def create_az_data(data_band_names: list[str]) -> pd.DataFrame:
    """
    Build the AZ predictor dataframe for years START_YEAR to END_YEAR.

    Calls ``dataops.create_az_data_csv`` which reads each year's
    Predictor, GW_Basin, GW_Subbasin, Streamflow,
    Canal_Weighted_Streamflow, Canal_Density, and Well_Density rasters,
    then maps ADWR sub-basin OBJECTIDs to names and runs EDA.
    """
    logger.info('='*60)
    logger.info('Step 1: Creating AZ predictor data (1896-2099)...')
    logger.info('='*60)

    az_df = dataops.create_az_data_csv(
        PRED_DATA_DIR,
        GW_CROPPED_RASTER_DIR,
        MODEL_DIR,
        data_band_names,
        AZ_GW_BASIN,
        start_year=START_YEAR,
        end_year=END_YEAR,
        load_csv=True,
        subbasin_vector=ADWR_SUBBASIN_SHP,
    )
    logger.info(f'AZ data shape: {az_df.shape}')
    logger.info(f'Year range: {az_df.Year.min()} – {az_df.Year.max()}')
    logger.info(f'Columns: {list(az_df.columns)}')

    # EDA
    vizops.explore_az_data(az_df, f'{MODEL_DIR}EDA/')
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


def _train_and_evaluate(
        x_train: pd.DataFrame, y_train: np.ndarray,
        x_test: pd.DataFrame, y_test: np.ndarray,
        model_name: str, output_dir: str,
) -> dict:
    """Train a single model with Optuna+Dask and return train/test metrics."""
    model, cv_df = mlops.build_ml_model_optuna_dask(
        x_train, y_train, output_dir, model_name,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
    )
    y_pred_train = np.abs(model.predict(x_train))
    y_pred_test = np.abs(model.predict(x_test))
    train_metrics = _compute_metrics(y_train, y_pred_train)
    test_metrics = _compute_metrics(y_test, y_pred_test)
    return {
        'model': model,
        'cv_df': cv_df,
        'train': train_metrics,
        'test': test_metrics,
    }


# ---- 2a. Random 80/20 evaluation ------------------------------------------
def evaluate_random(az_df: pd.DataFrame) -> dict:
    """Random 80/20 split — single run with compare_all_models."""
    logger.info('='*60)
    logger.info('Step 2a: Random 80/20 evaluation')
    logger.info('='*60)

    ml_models = mlops.get_model_param_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
    )
    strategy_dir = f'{MODEL_DIR}Model_Evaluation/Random/'
    test_year_limits = ((min(YEAR_LIST), max(YEAR_LIST)),)

    data_dir = f'{strategy_dir}data/'
    ret_vals = dataops.create_train_test_data(
        az_df, data_dir,
        drop_attr=DROP_ATTRS,
        random_state=RANDOM_STATE,
        scaling=False, already_created=False,
        year_list=YEAR_LIST, split_strategy=4,
        test_year=True, outlier_op=None,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        water_use=WATER_USE,
    )
    (x_train, x_test, y_train, y_test,
     x_scaler, y_scaler,
     year_train, year_test,
     basin_train, basin_test) = ret_vals

    comparison_dir = f'{strategy_dir}Model_Comparison/'
    comparison_df = mlops.compare_all_models(
        x_train, x_test, y_train, y_test,
        comparison_dir,
        model_names=ml_models,
        random_state=RANDOM_STATE,
        use_optuna=USE_OPTUNA,
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
        apply_bias_correction=2,
    )
    logger.info(f'\nRandom model comparison:\n{comparison_df.to_string(index=False)}')

    # Per-model visualisations
    for model_name in ml_models:
        bc_pq = f'{comparison_dir}{model_name}/Predictions_{model_name}_BC.parquet'
        raw_pq = f'{comparison_dir}{model_name}/Predictions_{model_name}.parquet'
        pq = bc_pq if os.path.exists(bc_pq) else raw_pq
        if os.path.exists(pq):
            pred_df = pd.read_parquet(pq)
            mlops.generate_model_visualizations(
                pred_df=pred_df,
                output_dir=f'{comparison_dir}{model_name}/Visualizations/',
                model_name=model_name,
                test_case='Random',
                test_year_limits=test_year_limits,
                raster_res=MOSAIC_RASTER_RES,
                use_ama_ina=USE_AMA_INA,
                create_basin_plots=True,
            )

    return {'comparison_df': comparison_df, 'strategy': 'Random'}


# ---- 2b. Leave-one-out temporal holdout ------------------------------------
def evaluate_temporal_loo(az_df: pd.DataFrame) -> dict:
    """
    Evaluate each model on every temporal holdout (T1-T6).

    Returns per-holdout and averaged metrics across all holdouts.
    """
    logger.info('='*60)
    logger.info('Step 2b: LOO Temporal evaluation (T1-T6)')
    logger.info('='*60)

    ml_models = mlops.get_model_param_dict(
        get_model_names_only=True,
        include_all_models=INCLUDE_ALL_MODELS,
    )
    temporal_dir = f'{MODEL_DIR}Model_Evaluation/Temporal_LOO/'
    makedirs(temporal_dir)

    per_holdout_rows = []  # per-holdout per-model metrics

    for holdout_name, test_year_limits in TEMPORAL_HOLDOUTS.items():
        logger.info(f'\n--- Temporal holdout {holdout_name}: {test_year_limits} ---')
        test_years = []
        for s, e in test_year_limits:
            test_years.extend(range(s, e + 1))
        test_years = tuple(test_years)

        holdout_dir = f'{temporal_dir}{holdout_name}/'
        data_dir = f'{holdout_dir}data/'
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=YEAR_LIST, split_strategy=1,
            test_year=test_years, outlier_op=None,
            test_gw_basins=(),
            use_ama_ina=USE_AMA_INA,
            drop_gw_basins=DROP_GW_BASINS,
            water_use=WATER_USE,
        )
        (x_train, x_test, y_train, y_test,
         x_scaler, y_scaler,
         year_train, year_test,
         basin_train, basin_test) = ret_vals

        if len(y_test) == 0:
            logger.warning(f'No test data for {holdout_name}, skipping.')
            continue

        for model_name in ml_models:
            logger.info(f'  Training {model_name} for {holdout_name}...')
            model_dir = f'{holdout_dir}{model_name}/'
            res = _train_and_evaluate(
                x_train, y_train, x_test, y_test,
                model_name, model_dir,
            )

            # Prediction results + visualisation
            pred_df = mlops.get_prediction_results(
                res['model'], x_train, x_test,
                y_train, y_test, x_scaler, y_scaler,
                year_train, year_test,
                basin_train, basin_test,
                model_dir, model_name,
                apply_bias_correction=2,
            )
            mlops.calc_train_test_metrics(
                pred_df, res['cv_df'], model_dir,
                use_ama_ina=USE_AMA_INA, model_name=model_name,
            )
            mlops.generate_model_visualizations(
                pred_df=pred_df,
                output_dir=f'{model_dir}Visualizations/',
                model_name=model_name,
                test_case=holdout_name,
                test_year_limits=test_year_limits,
                raster_res=MOSAIC_RASTER_RES,
                use_ama_ina=USE_AMA_INA,
                create_basin_plots=False,
            )

            per_holdout_rows.append({
                'Holdout': holdout_name,
                'Model': model_name,
                'Train_R2': res['train']['R2'],
                'Test_R2': res['test']['R2'],
                'Train_RMSE': res['train']['RMSE_pct'],
                'Test_RMSE': res['test']['RMSE_pct'],
                'Test_MAE': res['test']['MAE_pct'],
                'Test_MBE': res['test']['MBE_pct'],
                'Overfit_R2': res['train']['R2'] - res['test']['R2'],
            })

    # Build results DataFrames
    per_holdout_df = pd.DataFrame(per_holdout_rows).round(4)
    per_holdout_df.to_csv(f'{temporal_dir}Per_Holdout_Metrics.csv', index=False)
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
    avg_df.to_csv(f'{temporal_dir}Averaged_Metrics.csv', index=False)
    logger.info(f'\nAveraged temporal metrics:\n{avg_df.to_string(index=False)}')

    # Visualisation: heatmap of Test R² per holdout × model
    vizops.plot_loo_heatmap(
        per_holdout_df, 'Holdout', temporal_dir,
        title='Temporal LOO: Test R² per Holdout',
    )
    vizops.plot_loo_bar(per_holdout_df, 'Holdout', temporal_dir)

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
    """
    logger.info('='*60)
    logger.info('Step 2c: LOO Spatial evaluation (ADWR sub-basins)')
    logger.info('='*60)

    ml_models = mlops.get_model_param_dict(
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

    spatial_dir = f'{MODEL_DIR}Model_Evaluation/Spatial_LOO/'
    makedirs(spatial_dir)

    per_subbasin_rows = []

    for subbasin in subbasins:
        logger.info(f'\n--- Spatial holdout: {subbasin} ---')
        subbasin_safe = subbasin.replace(' ', '_').replace('.', '')
        holdout_dir = f'{spatial_dir}{subbasin_safe}/'

        data_dir = f'{holdout_dir}data/'
        ret_vals = dataops.create_train_test_data(
            az_df, data_dir,
            drop_attr=DROP_ATTRS,
            random_state=RANDOM_STATE,
            scaling=False, already_created=False,
            year_list=YEAR_LIST, split_strategy=3,
            test_year=(), outlier_op=None,
            test_gw_basins=(subbasin,),
            gw_basin_col='GW_Subbasin',
            use_ama_ina=False,  # filtering is handled by subbasin list
            drop_gw_basins=(),
            water_use=WATER_USE,
        )
        (x_train, x_test, y_train, y_test,
         x_scaler, y_scaler,
         year_train, year_test,
         basin_train, basin_test) = ret_vals

        if len(y_test) == 0:
            logger.warning(f'No test data for sub-basin {subbasin}, skipping.')
            continue

        logger.info(f'  Train: {len(y_train)}, Test: {len(y_test)}')

        for model_name in ml_models:
            logger.info(f'  Training {model_name} (holdout: {subbasin})...')
            model_dir = f'{holdout_dir}{model_name}/'
            res = _train_and_evaluate(
                x_train, y_train, x_test, y_test,
                model_name, model_dir,
            )

            # Prediction results
            pred_df = mlops.get_prediction_results(
                res['model'], x_train, x_test,
                y_train, y_test, x_scaler, y_scaler,
                year_train, year_test,
                basin_train, basin_test,
                model_dir, model_name,
                gw_basin_col='GW_Subbasin',
                apply_bias_correction=1,  # global bias correction for spatial
            )
            mlops.calc_train_test_metrics(
                pred_df, res['cv_df'], model_dir,
                use_ama_ina=False,
                gw_basin_col='GW_Subbasin',
                model_name=model_name,
            )

            per_subbasin_rows.append({
                'Subbasin': subbasin,
                'Model': model_name,
                'N_test': len(y_test),
                'Train_R2': res['train']['R2'],
                'Test_R2': res['test']['R2'],
                'Train_RMSE': res['train']['RMSE_pct'],
                'Test_RMSE': res['test']['RMSE_pct'],
                'Test_MAE': res['test']['MAE_pct'],
                'Test_MBE': res['test']['MBE_pct'],
                'Overfit_R2': res['train']['R2'] - res['test']['R2'],
            })

    # Build results DataFrames
    per_subbasin_df = pd.DataFrame(per_subbasin_rows).round(4)
    per_subbasin_df.to_csv(f'{spatial_dir}Per_Subbasin_Metrics.csv', index=False)
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
    avg_df.to_csv(f'{spatial_dir}Averaged_Metrics.csv', index=False)
    logger.info(f'\nAveraged spatial metrics:\n{avg_df.to_string(index=False)}')

    # Visualisations
    vizops.plot_loo_heatmap(
        per_subbasin_df, 'Subbasin', spatial_dir,
        title='Spatial LOO: Test R² per Sub-basin',
    )
    vizops.plot_loo_bar(per_subbasin_df, 'Subbasin', spatial_dir)

    return {
        'per_subbasin_df': per_subbasin_df,
        'avg_df': avg_df,
        'strategy': 'Spatial_LOO',
    }


# =============================================================================
# Step 3 — Predict annual pumping rasters 1896-2099 with XGBoost
# =============================================================================
def predict_full_period(az_df: pd.DataFrame) -> None:
    """
    Train XGBoost on the full 1984-2024 metered data (temporal split T1)
    and predict groundwater pumping rasters for every year from 1896 to 2099.
    """
    logger.info('='*60)
    logger.info('Step 3: XGBoost full-period prediction (1896-2099)')
    logger.info('='*60)

    model_name = 'XGB'
    prediction_dir = f'{MODEL_DIR}Full_Prediction_{model_name}/'
    makedirs(prediction_dir)

    # ---- 3a. Train on ALL 1984-2024 metered data (no holdout) ----
    # Step 2 already provides thorough LOO evaluation; here we maximise
    # training data for the best possible full-period predictions.
    data_dir = f'{prediction_dir}data/'
    ret_vals = dataops.create_train_test_data(
        az_df, data_dir,
        drop_attr=DROP_ATTRS,
        random_state=RANDOM_STATE,
        scaling=False,
        already_created=False,
        year_list=YEAR_LIST,
        split_strategy=1,
        test_year=(),
        outlier_op=None,
        test_gw_basins=(),
        use_ama_ina=USE_AMA_INA,
        drop_gw_basins=DROP_GW_BASINS,
        water_use=WATER_USE,
    )
    x_train, y_train = ret_vals[0], ret_vals[2]

    logger.info(f'Training XGBoost on {len(x_train)} samples '
                f'({YEAR_LIST[0]}-{YEAR_LIST[-1]}, all years)...')
    model, _ = mlops.build_ml_model_optuna_dask(
        x_train, y_train,
        f'{prediction_dir}Model/',
        model_name, RANDOM_STATE,
        n_trials=N_TRIALS,
        n_dask_workers=N_DASK_WORKERS,
        use_dask=USE_DASK,
    )

    # ---- 3b. Predict pumping for each year 1896-2099 ----
    logger.info('Predicting pumping for all years 1896-2099...')

    feature_cols = list(x_train.columns)
    raster_dir = f'{prediction_dir}Predicted_Rasters/'
    raster_dirs = {
        'mm': f'{raster_dir}Depth_mm/',
        'ft': f'{raster_dir}Depth_ft/',
        'm3': f'{raster_dir}Volume_m3/',
        'AF': f'{raster_dir}Volume_AF/',
    }
    for d in raster_dirs.values():
        makedirs(d)

    # Category-specific raster directories
    cat_raster_dirs = {}
    for cat in partops.CATEGORIES:
        base = f'{prediction_dir}{cat}_Rasters/'
        cat_raster_dirs[cat] = {
            'mm': f'{base}Depth_mm/',
            'ft': f'{base}Depth_ft/',
            'm3': f'{base}Volume_m3/',
            'AF': f'{base}Volume_AF/',
        }
        for d in cat_raster_dirs[cat].values():
            makedirs(d)

    # Build a valid-pixel mask from the GW_Basin raster (same for all years).
    # In create_az_data_csv, pixels with NaN or 0 in GW_Basin are labelled
    # 'OUTSIDE AZ' and dropped.  The remaining rows — in ravel order — are
    # what appears in az_df for each year.
    ref_basin_file = f'{PRED_DATA_DIR}GW_Basin_{YEAR_LIST[0]}.tif'
    basin_arr, basin_file_obj = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    basin_file_obj.close()

    # Reference raster for spatial metadata (CRS, transform)
    ref_raster_file = f'{PRED_DATA_DIR}Predictor_{YEAR_LIST[0]}.tif'

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
        """Compute depth and volume stats in multiple units."""
        n = len(pred_vals)
        mean_mm = float(np.nanmean(pred_vals)) if n > 0 else 0.0
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

    # Per-category tracking dicts: cat → {year → ...}
    cat_yearly = {cat: {} for cat in partops.CATEGORIES}
    cat_basin_yearly = {cat: {} for cat in partops.CATEGORIES}
    cat_subbasin_yearly = {cat: {} for cat in partops.CATEGORIES}

    for year in range(START_YEAR, END_YEAR + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            logger.warning(f'No data for year {year}, skipping.')
            continue

        # Build feature matrix matching training columns
        drop_list = [a for a in DROP_ATTRS if a in year_df.columns]
        pred_features = year_df.drop(
            columns=drop_list + ['gw_pumping_mm', 'GW_Basin', 'Year'],
            errors='ignore'
        )
        # Ensure same columns and order as training
        for c in feature_cols:
            if c not in pred_features.columns:
                pred_features[c] = 0
        pred_features = pred_features[feature_cols]
        pred_features = pred_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        predictions = np.abs(model.predict(pred_features))

        # Partition into irrigation/non-irrigation and GW/SW categories
        cat_predictions = partops.partition_predictions(
            predictions, year_df, raster_shape, valid_mask,
        )
        predictions = cat_predictions['Irrigation'] + cat_predictions['Non_Irrigation']

        # Reconstruct raster: valid_mask marks the pixels that survived
        # the 'OUTSIDE AZ' filter in create_az_data_csv.
        pred_mm = np.full(basin_flat.shape[0], np.nan, dtype=np.float32)
        pred_mm[valid_mask] = predictions.astype(np.float32)
        pred_mm = pred_mm.reshape(raster_shape)

        # Derive grids in other units — total pumping
        pred_ft = pred_mm * mm_to_ft
        pred_m3 = pred_mm * mm_to_m3
        pred_af = pred_mm * mm_to_m3 * m3_to_af

        unit_grids = {
            'mm': pred_mm,
            'ft': pred_ft,
            'm3': pred_m3,
            'AF': pred_af,
        }
        for unit, grid in unit_grids.items():
            _, raster_file_obj = read_raster_as_arr(ref_raster_file, get_file=True)
            out_path = f'{raster_dirs[unit]}Predicted_GW_{year}_{unit}.tif'
            write_raster(
                grid, raster_file_obj,
                raster_file_obj.transform, out_path,
                no_data_value=np.nan,
            )
            raster_file_obj.close()

        # Write category rasters (irr, non-irr, irr_gw, irr_sw, nonirr_gw, nonirr_sw)
        for cat, cat_pred in cat_predictions.items():
            cat_mm = np.full(basin_flat.shape[0], np.nan, dtype=np.float32)
            cat_mm[valid_mask] = cat_pred.astype(np.float32)
            cat_mm = cat_mm.reshape(raster_shape)
            cat_units = {
                'mm': cat_mm,
                'ft': cat_mm * mm_to_ft,
                'm3': cat_mm * mm_to_m3,
                'AF': cat_mm * mm_to_m3 * m3_to_af,
            }
            for unit, grid in cat_units.items():
                _, raster_file_obj = read_raster_as_arr(ref_raster_file, get_file=True)
                out_path = f'{cat_raster_dirs[cat][unit]}{cat}_{year}_{unit}.tif'
                write_raster(
                    grid, raster_file_obj,
                    raster_file_obj.transform, out_path,
                    no_data_value=np.nan,
                )
                raster_file_obj.close()

        # AZ-wide annual total (all valid pixels)
        yearly_predictions[year] = _pixel_stats(predictions)
        for cat, cat_pred in cat_predictions.items():
            cat_yearly[cat][year] = _pixel_stats(cat_pred)

        # Per-basin annual stats (all AZ basins)
        basin_totals = {}
        cat_basin_totals = {cat: {} for cat in partops.CATEGORIES}
        for basin in all_basins:
            bmask = (year_df.GW_Basin == basin).values
            basin_totals[basin] = _pixel_stats(predictions[bmask])
            for cat, cat_pred in cat_predictions.items():
                cat_basin_totals[cat][basin] = _pixel_stats(cat_pred[bmask])
        basin_yearly[year] = basin_totals
        for cat in partops.CATEGORIES:
            cat_basin_yearly[cat][year] = cat_basin_totals[cat]

        # Per-sub-basin annual stats
        subbasin_totals = {}
        cat_subbasin_totals = {cat: {} for cat in partops.CATEGORIES}
        for sb in subbasins:
            sbmask = (year_df.GW_Subbasin == sb).values
            subbasin_totals[sb] = _pixel_stats(predictions[sbmask])
            for cat, cat_pred in cat_predictions.items():
                cat_subbasin_totals[cat][sb] = _pixel_stats(cat_pred[sbmask])
        subbasin_yearly[year] = subbasin_totals
        for cat in partops.CATEGORIES:
            cat_subbasin_yearly[cat][year] = cat_subbasin_totals[cat]

        if year % 20 == 0 or year == END_YEAR:
            vol_af = yearly_predictions[year]['Volume_AF']
            irr_gw = cat_yearly['Irrigation_GW'][year]['Volume_AF']
            irr_sw = cat_yearly['Irrigation_SW'][year]['Volume_AF']
            nigw = cat_yearly['Non_Irrigation_GW'][year]['Volume_AF']
            nisw = cat_yearly['Non_Irrigation_SW'][year]['Volume_AF']
            logger.info(
                f'  Year {year}: total = {vol_af:,.0f} AF'
                f'  |  irr_GW = {irr_gw:,.0f}  irr_SW = {irr_sw:,.0f}'
                f'  |  non-irr_GW = {nigw:,.0f}  non-irr_SW = {nisw:,.0f}'
            )

    # ---- 3c. Time series and map visualisations ----
    # Total pumping
    vizops.create_full_period_time_series(
        yearly_predictions, prediction_dir,
        start_year=START_YEAR, end_year=END_YEAR,
    )
    vizops.create_era_summary_maps(yearly_predictions, prediction_dir)
    vizops.create_basin_time_series(
        basin_yearly, prediction_dir,
        start_year=START_YEAR, end_year=END_YEAR,
    )
    vizops.create_subbasin_time_series(
        subbasin_yearly, prediction_dir,
        subbasin_shp=ADWR_SUBBASIN_SHP,
        ama_code_map=AMA_CODE_MAP,
        start_year=START_YEAR, end_year=END_YEAR,
    )

    # Per-category time series
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
        cat_dir = f'{prediction_dir}{cat}/'
        vizops.create_full_period_time_series(
            cat_yearly[cat], cat_dir,
            start_year=START_YEAR, end_year=END_YEAR,
            title_prefix=title,
        )
        vizops.create_era_summary_maps(
            cat_yearly[cat], cat_dir,
            title_prefix=title,
        )
        vizops.create_basin_time_series(
            cat_basin_yearly[cat], cat_dir,
            start_year=START_YEAR, end_year=END_YEAR,
            title_prefix=title,
        )
        vizops.create_subbasin_time_series(
            cat_subbasin_yearly[cat], cat_dir,
            subbasin_shp=ADWR_SUBBASIN_SHP,
            ama_code_map=AMA_CODE_MAP,
            start_year=START_YEAR, end_year=END_YEAR,
            title_prefix=title,
        )

    # ---- 3d. Well package (per-well annual withdrawals as GeoPackage) ----
    well_registry_file = f'{OUTPUT_DIR}GW_Data/Vector_Reproj/Well_Registry.shp'
    gw_vector_dir = f'{OUTPUT_DIR}GW/Vectors/{WNAME}/'
    wellops.create_well_package(
        well_registry_file,
        raster_dirs=raster_dirs,
        cat_raster_dirs=cat_raster_dirs,
        output_dir=f'{prediction_dir}Well_Package/',
        ref_raster_file=ref_raster_file,
        pixel_area_m2=pixel_area_m2,
        start_year=START_YEAR,
        end_year=END_YEAR,
        water_use='All' if WATER_USE == 'All' else 'IRRIGATION',
        gw_vector_dir=gw_vector_dir,
    )

    logger.info(f'Full-period prediction complete. Results in {prediction_dir}')


# =============================================================================
# Main
# =============================================================================
def main(
        skip_download: bool = True,
        load_files: bool = True,
        run_data_prep: bool = True,
) -> None:
    """
    Run the full AZ-Hydro pipeline.

    Parameters
    ----------
    skip_download : bool
        If *True*, skip the GEE tile download (use existing tiles).
    load_files : bool
        If *True*, skip recreating intermediate files that already exist.
    run_data_prep : bool
        If *True*, execute Step 0 (data preparation).  Set to *False* if
        all rasters / vectors are already prepared.
    """
    # Step 0 — Data preparation
    if run_data_prep:
        data_band_names = prepare_data(
            skip_download=skip_download,
            load_files=load_files,
        )
    else:
        # Band names are still needed — grab from GEE config without
        # triggering a download.
        _, data_band_names = dataops.download_gee_data(
            f'{VECTOR_DIR}AZ.geojson',
            GCLOUD_PROJECT, GCLOUD_BUCKET,
            INPUT_DIR,
            START_YEAR, END_YEAR,
            skip_download=True,
            tile_size=TILE_SIZE,
        )

    # Step 1
    az_df = create_az_data(data_band_names)

    # Step 2a — Random
    random_results = evaluate_random(az_df)

    # Step 2b — LOO Temporal
    temporal_results = evaluate_temporal_loo(az_df)

    # Step 2c — LOO Spatial (ADWR sub-basins)
    spatial_results = evaluate_spatial_loo(az_df)

    # Cross-strategy summary
    vizops.create_cross_strategy_summary(
        {
            'Random': random_results,
            'Temporal_LOO': temporal_results,
            'Spatial_LOO': spatial_results,
        },
        f'{MODEL_DIR}Model_Evaluation/',
    )

    # Step 3
    predict_full_period(az_df)

    logger.info('\n' + '='*60)
    logger.info('Pipeline complete!')
    logger.info('='*60)


if __name__ == '__main__':
    main()
