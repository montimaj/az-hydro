"""
Hybrid Uncertainty Quantification for AZ-Hydro Groundwater Pumping Predictions.

Computes five independent uncertainty components and combines them via
quadrature into a total pixel-level uncertainty (σ_total):

    σ_total = √(σ_MACA² + σ_model² + σ_irr² + σ_gw² + σ_LULC²)

Each component is computed at both the **total** pumping level and for
each of the 8 **withdrawal categories** (Irrigation, Non_Irrigation,
Irrigation_GW, Irrigation_SW, Non_Irrigation_GW, Non_Irrigation_SW,
Total_GW, Total_SW).  Per-category σ is derived by partitioning every
ensemble member's prediction *before* computing std, so partition-fraction
uncertainty (irr_fraction, gw_fraction) is properly propagated.

Components
----------
σ_MACA  : Inter-GCM climate spread (5 representative GCMs), future only.
σ_model : XGBoost seed ensemble (10 random seeds), all years.
σ_irr   : Irrigation fraction spread (IrrMapper-based vs regression-based),
           historical only (1896-2025).  For future years this source is
           subsumed by σ_LULC.
σ_gw    : GW fraction inter-snapshot spread (4 Hung et al. snapshots).
σ_LULC  : Inter-scenario LULC projection spread (B1, B2, A1B, A2),
           future only (2026-2099).  Perturbs AGRI, URBAN, crop fraction,
           and irrigation fraction end-to-end.

Confidence-interval methodology
-------------------------------
Components are classified as **sample-based** or **scenario-based**:

*Sample-based* (σ_model, σ_gw): the ensemble members are random draws
from a larger population (random seeds, temporal snapshots).  Their CIs
use Student's t-distribution critical values (t_{0.025, df}) instead of
the normal z = 1.96, because the small sample size under-estimates the
true population σ.  The t-correction is applied by inflating σ by
t/z *before* quadrature, so all downstream code uses a single multiplier
(CI_Z = 1.96).

*Scenario-based* (σ_MACA, σ_LULC, σ_irr): the ensemble spans
deliberately chosen structural alternatives (GCMs, LULC projections,
irrigation mapping methods).  The spread is a lower bound on structural
uncertainty, not a sample from a random population.  These components
retain z = 1.96 (scale = 1.0).

Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import logging
import os
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# 5 representative GCMs spanning the Southwest US climate space
# (cf. Rupp et al., 2013; same list as gee/config.py)
MACA_REPRESENTATIVE_GCMS = [
    'CCSM4',           # central / median
    'CNRM-CM5',        # cool-wet
    'HadGEM2-ES365',   # hot-dry
    'MIROC-ESM-CHEM',  # hot-wet
    'inmcm4',          # cool-dry
]

MACA_FUTURE_START = 2026

# MACA-derived predictor columns that vary by GCM (future years only)
MACA_CLIMATE_COLS = [
    'annual_et_ensemble_mm',
    'annual_eto_mm',
    'annual_precip_mm',
    'annual_peff_mm',
    'annual_tmmx_K',
    'annual_tmmn_K',
]

# 1-based band indices in Predictor_{year}.tif matching MACA_CLIMATE_COLS
MACA_CLIMATE_BAND_INDICES = [1, 2, 3, 4, 6, 7]

# 10-seed ensemble for model uncertainty
MODEL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# GW fraction band index in Predictor_{year}.tif (1-based)
GW_FRACTION_BAND_INDEX = 12

# GW fraction snapshot years (Hung et al.)
GW_FRACTION_SNAPSHOTS = [2000, 2005, 2010, 2015]

# Unit conversions
MM_TO_FT = 1 / 304.8
M3_TO_AF = 1 / 1233.48

# 4 USGS LULC projection scenarios (same list as gee/config.py)
USGS_LULC_SCENARIOS = ['B1', 'B2', 'A1B', 'A2']

# 1-based band indices in Predictor_{year}.tif for LULC-derived columns
LULC_BAND_INDEX = 8            # integer LULC class
CROP_FRACTION_BAND_INDEX = 13  # annual_crop_fraction
ET_BAND_INDEX = 1              # annual_et_ensemble_mm
PEFF_BAND_INDEX = 4            # annual_peff_mm
IRR_FRACTION_BAND_INDEX = 14   # annual_irr_fraction

# ── 95 % CI multiplier and t-distribution corrections ────────────────────
# The normal approximation z = 1.96 is the default 95 % CI multiplier.
# For *sample-based* components (σ_model, σ_GW), where the ensemble is a
# random draw from a larger population, t-distribution critical values are
# more appropriate given the small sample sizes.
# For *scenario-based* components (σ_MACA, σ_LULC, σ_irr), the ensemble
# members span structural choices rather than a random sample, so z = 1.96
# is retained (the resulting CI is a lower bound on structural range).
CI_Z = 1.96

# t_{0.025, df} critical values (two-tailed 95 %)
T_CRIT_MODEL = 2.2622   # df = len(MODEL_SEEDS) − 1 = 9
T_CRIT_GW = 3.1824      # df = len(GW_FRACTION_SNAPSHOTS) − 1 = 3

# Scale factors applied to sample-based σ before quadrature so that the
# final ± CI_Z × σ_total interval reflects t-corrected CIs.
T_SCALE_MODEL = T_CRIT_MODEL / CI_Z   # ≈ 1.154
T_SCALE_GW = T_CRIT_GW / CI_Z         # ≈ 1.624

COMPONENT_T_SCALE = {
    'MACA': 1.0,           # scenario-based — no correction
    'Model': T_SCALE_MODEL,  # sample-based (10 seeds)
    'Irr': 1.0,             # half-range of 2 scenarios — no correction
    'LULC': 1.0,            # scenario-based — no correction
    'GW': T_SCALE_GW,       # sample-based (4 Hung et al. snapshots)
}

COMPONENT_N = {
    'MACA': len(MACA_REPRESENTATIVE_GCMS),   # 5
    'Model': len(MODEL_SEEDS),               # 10
    'Irr': 2,                                # 2 (half-range, not std)
    'LULC': len(USGS_LULC_SCENARIOS),        # 4
    'GW': len(GW_FRACTION_SNAPSHOTS),         # 4
}

# Category / CU / IE raster groups
CU_CATEGORIES = ('Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU')
IE_CATEGORIES = (
    'Irrigation_Efficiency',
    'Irrigation_GW_Efficiency',
    'Irrigation_SW_Efficiency',
)
IE_WITHDRAWAL_MAP = {
    'Irrigation_Efficiency': 'Irrigation',
    'Irrigation_GW_Efficiency': 'Irrigation_GW',
    'Irrigation_SW_Efficiency': 'Irrigation_SW',
}
IE_CU_MAP = {
    'Irrigation_Efficiency': 'Irrigation_CU',
    'Irrigation_GW_Efficiency': 'Irrigation_GW_CU',
    'Irrigation_SW_Efficiency': 'Irrigation_SW_CU',
}


# ── Helper ───────────────────────────────────────────────────────────────────

def _build_pred_features(
        year_df: pd.DataFrame,
        feature_cols: list[str],
        drop_attrs: tuple[str, ...],
) -> pd.DataFrame:
    """Build a prediction-ready feature matrix from *year_df*.

    XGBoost features = columns from ``create_az_data_parquet`` minus
    *drop_attrs* minus the target (``gw_pumping_mm``).
    """
    drop_list = [a for a in drop_attrs if a in year_df.columns]
    pred = year_df.drop(
        columns=drop_list + ['gw_pumping_mm'],
        errors='ignore',
    )
    for c in feature_cols:
        if c not in pred.columns:
            pred[c] = 0
    pred = pred[feature_cols]
    return pred.replace([np.inf, -np.inf], np.nan).fillna(0)


def _predict_total(model, pred_features, year_df, partops,
                   raster_shape, valid_mask):
    """Predict and partition, returning total pumping and category dict.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        (total_1d, categories) where total = Irrigation + Non_Irrigation
        and categories is the full dict from ``partition_predictions``.
    """
    raw = np.abs(model.predict(pred_features))
    cat = partops.partition_predictions(raw, year_df, raster_shape, valid_mask)
    return cat['Irrigation'] + cat['Non_Irrigation'], cat


def _compute_category_sigmas(
        member_cats: list[dict[str, np.ndarray]],
        mode: str = 'std',
) -> dict[str, np.ndarray]:
    """Compute per-category σ from ensemble member category dicts.

    Parameters
    ----------
    member_cats : list of dicts
        Each dict maps category name → 1-D prediction array.
    mode : {'std', 'half_range'}
        'std' — ``np.nanstd`` with ``ddof=1`` (3+ members).
        'half_range' — ``|a − b| / 2`` (2 counterfactual members).
    """
    cat_sigmas: dict[str, np.ndarray] = {}
    for cat_name in member_cats[0]:
        if mode == 'half_range':
            cat_sigmas[cat_name] = (
                np.abs(member_cats[0][cat_name] - member_cats[1][cat_name])
                / 2.0
            ).astype(np.float32)
        else:
            stack = np.stack(
                [mc[cat_name] for mc in member_cats], axis=0,
            )
            cat_sigmas[cat_name] = np.nanstd(
                stack, axis=0, ddof=1,
            ).astype(np.float32)
    return cat_sigmas


def _pixel_stats(pred_vals, mm_to_m3, m3_to_af):
    """Compute summary statistics in multiple units."""
    n = len(pred_vals)
    mean_mm = float(np.nanmean(pred_vals)) if n > 0 else 0.0
    vol_m3 = float(np.nansum(pred_vals)) * mm_to_m3
    return {
        'Mean_Depth_mm': round(mean_mm, 4),
        'Mean_Depth_ft': round(mean_mm * MM_TO_FT, 6),
        'Volume_m3': round(vol_m3, 2),
        'Volume_AF': round(vol_m3 * M3_TO_AF, 2),
    }


def _write_std_raster(std_vals, basin_flat, valid_mask, raster_shape,
                      ref_raster_file, out_path, read_raster_as_arr,
                      write_raster):
    """Write a standard-deviation raster (mm) to disk."""
    grid = np.full(basin_flat.shape[0], np.nan, dtype=np.float32)
    grid[valid_mask] = std_vals.astype(np.float32)
    grid = grid.reshape(raster_shape)
    _, rfile = read_raster_as_arr(ref_raster_file, get_file=True)
    write_raster(grid, rfile, rfile.transform, out_path, no_data_value=np.nan)
    rfile.close()


def _write_sigma_cv_raster(sigma_grid, cv_grid, ref_raster_file,
                           out_path, read_raster_as_arr):
    """Write a 2-band raster: band 1 = σ (mm), band 2 = CV."""
    _, rfile = read_raster_as_arr(ref_raster_file, get_file=True)
    with rio.open(
            out_path, 'w', driver='GTiff',
            height=sigma_grid.shape[0], width=sigma_grid.shape[1],
            dtype=np.float32, crs=rfile.crs,
            transform=rfile.transform, count=2, nodata=np.nan,
    ) as dst:
        dst.write(sigma_grid, 1)
        dst.write(cv_grid, 2)
        dst.set_band_description(1, 'sigma_total_mm')
        dst.set_band_description(2, 'CV')
    rfile.close()


def _save_summary(yearly_dict, output_dir, label=''):
    """Save yearly uncertainty summary to CSV."""
    df = pd.DataFrame.from_dict(yearly_dict, orient='index')
    df.index.name = 'Year'
    suffix = f'_{label}' if label else ''
    df.to_csv(os.path.join(output_dir, f'Uncertainty_Summary{suffix}.csv'))
    return df


def _write_augmented_raster(pred_arr, sigma_arr, out_path, profile,
                            band_descriptions):
    """Write a 6-band augmented raster (pred, σ, CV, SNR, CI bounds)."""
    abs_pred = np.abs(pred_arr)
    with np.errstate(invalid='ignore', divide='ignore'):
        cv = np.where(
            abs_pred > 0, sigma_arr / abs_pred, np.nan,
        ).astype(np.float32)
        snr = np.where(
            sigma_arr > 0, abs_pred / sigma_arr, np.nan,
        ).astype(np.float32)
    lower_ci = (pred_arr - CI_Z * sigma_arr).astype(np.float32)
    upper_ci = (pred_arr + CI_Z * sigma_arr).astype(np.float32)

    profile.update(count=6, dtype=np.float32, nodata=np.nan)
    with rio.open(out_path, 'w', **profile) as dst:
        dst.write(pred_arr.astype(np.float32), 1)
        dst.write(sigma_arr.astype(np.float32), 2)
        dst.write(cv, 3)
        dst.write(snr, 4)
        dst.write(lower_ci, 5)
        dst.write(upper_ci, 6)
        for i, desc in enumerate(band_descriptions, 1):
            dst.set_band_description(i, desc)


def _aggregate_member_volumes(
        member_preds: list[np.ndarray],
        year_df: pd.DataFrame,
        mm_to_m3: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Aggregate per-member pixel predictions to basin/sub-basin volumes.

    Parameters
    ----------
    member_preds : list of 1-D arrays
        Each array has length = number of valid pixels (same as year_df rows).
    year_df : DataFrame
        Must contain ``GW_Basin`` and ``GW_Subbasin`` string columns.
    mm_to_m3 : float
        Conversion factor: pixel depth (mm) → volume (m³).

    Returns
    -------
    (basin_vols, subbasin_vols)
        Each is ``{region_name: np.ndarray of shape (n_members,)}``
        giving the total volume (AF) per member for that region.
    """
    basins = year_df['GW_Basin'].values
    subbasins = year_df['GW_Subbasin'].values
    n_members = len(member_preds)
    m3_to_af = M3_TO_AF

    unique_basins = np.unique(basins)
    basin_vols: dict[str, np.ndarray] = {}
    for b in unique_basins:
        bmask = basins == b
        vols = np.empty(n_members, dtype=np.float64)
        for i, pred in enumerate(member_preds):
            vols[i] = float(np.nansum(pred[bmask])) * mm_to_m3 * m3_to_af
        basin_vols[b] = vols

    unique_subbasins = np.unique(subbasins)
    subbasin_vols: dict[str, np.ndarray] = {}
    for sb in unique_subbasins:
        if sb == 'NO_SUBBASIN':
            continue
        sbmask = subbasins == sb
        vols = np.empty(n_members, dtype=np.float64)
        for i, pred in enumerate(member_preds):
            vols[i] = float(np.nansum(pred[sbmask])) * mm_to_m3 * m3_to_af
        subbasin_vols[sb] = vols

    return basin_vols, subbasin_vols


def _accumulate_basin_sigma(
        accum: dict,
        year: int,
        basin_vols: dict[str, np.ndarray],
        subbasin_vols: dict[str, np.ndarray],
) -> None:
    """Store per-year basin/sub-basin member volumes into *accum*.

    ``accum`` has structure::

        {
            'basin': {year: {name: np.ndarray(n_members)}},
            'subbasin': {year: {name: np.ndarray(n_members)}},
        }
    """
    accum['basin'][year] = basin_vols
    accum['subbasin'][year] = subbasin_vols


def _write_basin_sigma_csv(
        accum: dict,
        output_dir: str,
        label: str,
) -> None:
    """Write basin-scale and sub-basin-scale σ CSVs from accumulated data.

    For each region, the CSV has columns:
    ``Year, Region, Mean_Volume_m3, Sigma_Volume_m3, Mean_Volume_AF,
    Sigma_Volume_AF, CV, Lower_95CI_m3, Upper_95CI_m3, Lower_95CI_AF,
    Upper_95CI_AF, N_Members``.
    """
    af_to_m3 = 1.0 / M3_TO_AF  # 1233.48
    for level in ('basin', 'subbasin'):
        rows = []
        level_data = accum[level]
        for year in sorted(level_data):
            for region, vols in sorted(level_data[year].items()):
                mean_af = float(np.mean(vols))
                std_af = (
                    float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
                )
                mean_m3 = mean_af * af_to_m3
                std_m3 = std_af * af_to_m3
                cv = std_af / abs(mean_af) if abs(mean_af) > 0 else np.nan
                rows.append({
                    'Year': year,
                    'Region': region,
                    'Mean_Volume_m3': round(mean_m3, 2),
                    'Sigma_Volume_m3': round(std_m3, 2),
                    'Mean_Volume_AF': round(mean_af, 2),
                    'Sigma_Volume_AF': round(std_af, 2),
                    'CV': round(cv, 6),
                    'Lower_95CI_m3': round(mean_m3 - CI_Z * std_m3, 2),
                    'Upper_95CI_m3': round(mean_m3 + CI_Z * std_m3, 2),
                    'Lower_95CI_AF': round(mean_af - CI_Z * std_af, 2),
                    'Upper_95CI_AF': round(mean_af + CI_Z * std_af, 2),
                    'N_Members': len(vols),
                })
        if rows:
            df = pd.DataFrame(rows)
            cap_level = level.replace('subbasin', 'Subbasin').replace(
                'basin', 'Basin'
            )
            df.to_csv(
                os.path.join(output_dir, f'{cap_level}_Sigma_{label}.csv'),
                index=False,
            )



# ═════════════════════════════════════════════════════════════════════════════
# σ_MACA — Inter-GCM climate spread (2026-2099 only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_maca(
        model,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        input_dir: str,
        vector_dir: str,
        mosaic_res: int,
        gcloud_project: str,
        gcloud_bucket: str,
        tile_size: int,
        end_year: int,
        year_list: list[int],
        skip_download: bool = False,
) -> dict[int, np.ndarray]:
    """
    Compute σ_MACA: per-pixel std of predictions across 5 representative
    GCMs for future years (2026-2099).

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]], dict[str, str]]
        (sigma_maca, cat_sigma_maca, gcm_mosaic_dirs) — per-year total σ
        arrays, per-category per-year σ arrays, and the per-GCM mosaic
        directory paths (reused by σ_CU).
    """
    import hydrolibs.dataops as dataops
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_MACA (inter-GCM climate uncertainty)...')
    raster_dir = os.path.join(output_dir, 'Sigma_MACA/Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Download and mosaic per-GCM tiles
    gcm_mosaic_dirs = {}
    for gcm in MACA_REPRESENTATIVE_GCMS:
        logger.info(f'  Preparing per-GCM tiles for {gcm}...')
        gcm_tile_dir, _ = dataops.download_gee_data(
            os.path.join(vector_dir, 'AZ.geojson'),
            gcloud_project, gcloud_bucket, input_dir,
            start_year=MACA_FUTURE_START, end_year=end_year,
            skip_download=skip_download, tile_size=tile_size,
            num_workers=40, worker_memory='1G',
            gee_scale=mosaic_res, verbose=False, gcm=gcm,
        )
        gcm_mosaic_dir = (
            f'{os.path.dirname(pred_data_dir.rstrip("/"))}/'
            f'../GEE_Mosaics_{mosaic_res}m_{gcm}/'
        )
        dataops.mosaic_tiles(
            gcm_tile_dir, gcm_mosaic_dir,
            MACA_FUTURE_START, end_year,
            already_mosaicked=skip_download,
        )
        gcm_mosaic_dirs[gcm] = gcm_mosaic_dir

    sigma_maca = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_sigma_maca: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    for year in range(MACA_FUTURE_START, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        gcm_preds = []
        gcm_cats = []
        for gcm in MACA_REPRESENTATIVE_GCMS:
            gcm_raster = f'{gcm_mosaic_dirs[gcm]}Predictor_{year}.tif'
            gcm_year_df = year_df.copy()
            for col, bidx in zip(MACA_CLIMATE_COLS, MACA_CLIMATE_BAND_INDICES):
                band_arr = read_raster_as_arr(gcm_raster, band=bidx, get_file=False)
                gcm_year_df[col] = band_arr.ravel()[valid_mask]

            pf = _build_pred_features(gcm_year_df, feature_cols, drop_attrs)
            pred, cat = _predict_total(model, pf, gcm_year_df, partops,
                                       raster_shape, valid_mask)
            gcm_preds.append(pred)
            gcm_cats.append(cat)

        gcm_stack = np.stack(gcm_preds, axis=0)
        std = np.nanstd(gcm_stack, axis=0, ddof=1)
        sigma_maca[year] = std

        cat_std = _compute_category_sigmas(gcm_cats)
        for c in partops.CATEGORIES:
            cat_sigma_maca[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(gcm_preds, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_MACA_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3, M3_TO_AF)

        if year % 10 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_MACA = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, os.path.join(output_dir, 'Sigma_MACA'), 'MACA')
    _write_basin_sigma_csv(basin_accum, os.path.join(output_dir, 'Sigma_MACA'), 'MACA')
    logger.info('  σ_MACA complete.')
    return sigma_maca, cat_sigma_maca, gcm_mosaic_dirs


# ═════════════════════════════════════════════════════════════════════════════
# σ_model — XGBoost seed ensemble (all years)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_model(
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
        n_trials: int = 100,
        n_dask_workers: int = 10,
        use_dask: bool = True,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Compute σ_model: per-pixel std of predictions across a 10-seed
    XGBoost ensemble.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]
        (sigma_model, cat_sigma_model) — per-year total σ and
        per-category per-year σ arrays.
    """
    import hydrolibs.mlops as mlops
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_model (seed-ensemble uncertainty)...')
    base_dir = os.path.join(output_dir, 'Sigma_Model')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Train (or load) 10 XGBoost models with different seeds
    models = []
    model_name = 'XGB'
    for seed in MODEL_SEEDS:
        seed_dir = os.path.join(base_dir, f'Model_seed{seed}')
        makedirs(seed_dir)
        model_file = os.path.join(seed_dir, f'{model_name}')
        if os.path.exists(model_file):
            logger.info(f'  Loading seed={seed} model...')
            with open(model_file, 'rb') as f:
                m = pickle.load(f)
        else:
            logger.info(f'  Training seed={seed} model...')
            m, _ = mlops.build_ml_model_optuna_dask(
                x_train, y_train, seed_dir,
                model_name, seed,
                n_trials=n_trials,
                n_dask_workers=n_dask_workers,
                use_dask=use_dask,
            )
        models.append(m)

    # Predict for all years with each model
    sigma_model = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_sigma_model: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    for year in range(start_year, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        pf = _build_pred_features(year_df, feature_cols, drop_attrs)
        seed_preds = []
        seed_cats = []
        for m in models:
            pred, cat = _predict_total(m, pf, year_df, partops,
                                       raster_shape, valid_mask)
            seed_preds.append(pred)
            seed_cats.append(cat)

        seed_stack = np.stack(seed_preds, axis=0)
        std = np.nanstd(seed_stack, axis=0, ddof=1)
        sigma_model[year] = std

        cat_std = _compute_category_sigmas(seed_cats)
        for c in partops.CATEGORIES:
            cat_sigma_model[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(seed_preds, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_Model_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3, M3_TO_AF)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_model = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'Model')
    _write_basin_sigma_csv(basin_accum, base_dir, 'Model')
    logger.info('  σ_model complete.')
    return sigma_model, cat_sigma_model


# ═════════════════════════════════════════════════════════════════════════════
# σ_irr — Irrigation fraction uncertainty (hindcast + projection)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_irr(
        model,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Compute σ_irr: uncertainty from irrigation fraction estimation.

    Covers historical years only (up to 2025).  For future years
    (2026-2099) this source is subsumed by σ_LULC, which perturbs the
    entire LULC → crop fraction → irr fraction chain end-to-end.

    For 1985-2025 (IrrMapper era), the IrrMapper value is taken as truth
    and a regression-based counter-factual is computed; σ_irr is the
    absolute difference in predictions between the two.

    For years outside 1985-2025, only the regression estimate exists, so
    σ_irr is estimated via perturbation (irr ± RMSE), taking the half-range.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]
        (sigma_irr, cat_sigma_irr) — per-year total σ and
        per-category per-year σ arrays.
    """
    from sklearn.linear_model import LinearRegression

    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_irr (irrigation fraction uncertainty)...')
    base_dir = os.path.join(output_dir, 'Sigma_Irr')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Refit the same irr-fraction regression used in create_az_data_parquet
    irr_train = az_df[az_df.Year.between(1985, 2025)].dropna(
        subset=['annual_irr_fraction', 'annual_crop_fraction']
    )
    irr_train = irr_train[irr_train.annual_crop_fraction > 0]
    irr_reg = LinearRegression().fit(
        irr_train[['annual_crop_fraction']].values,
        irr_train['annual_irr_fraction'].values,
    )
    # Regression RMSE on training data
    irr_pred_train = irr_reg.predict(
        irr_train[['annual_crop_fraction']].values
    )
    irr_rmse = float(np.sqrt(np.mean(
        (irr_train['annual_irr_fraction'].values - irr_pred_train) ** 2
    )))
    logger.info(f'  Irr-fraction regression RMSE = {irr_rmse:.4f}')

    sigma_irr = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_sigma_irr: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    # Only cover historical years; future years are handled by σ_LULC
    irr_end_year = min(end_year, 2025)

    for year in range(start_year, irr_end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        if 1985 <= year <= 2025:
            # Two counterfactuals: IrrMapper (original) vs regression

            # Compute regression-based irr fraction
            crop_frac = year_df['annual_crop_fraction'].values
            irr_frac_reg = np.zeros_like(crop_frac)
            nonzero = crop_frac > 0
            if nonzero.any():
                irr_frac_reg[nonzero] = np.clip(
                    irr_reg.predict(crop_frac[nonzero].reshape(-1, 1)), 0, 1
                )

            # Predict with original (IrrMapper) irr fraction
            pf_orig = _build_pred_features(year_df, feature_cols, drop_attrs)
            pred_orig, cat_orig = _predict_total(model, pf_orig, year_df,
                                                 partops, raster_shape,
                                                 valid_mask)

            # Predict with regression irr fraction
            alt_df = year_df.copy()
            alt_df['annual_irr_fraction'] = irr_frac_reg
            pf_alt = _build_pred_features(alt_df, feature_cols, drop_attrs)
            pred_alt, cat_alt = _predict_total(model, pf_alt, alt_df, partops,
                                               raster_shape, valid_mask)

            std = np.abs(pred_orig - pred_alt) / 2.0
            irr_members = [pred_orig, pred_alt]
            cat_std = _compute_category_sigmas(
                [cat_orig, cat_alt], mode='half_range',
            )
        else:
            # Outside IrrMapper era: constant σ from regression RMSE
            # propagated through the model's sensitivity to irr_fraction.
            # Use perturbation: predict with irr±RMSE, take half-range.
            irr_frac = year_df['annual_irr_fraction'].values.copy()
            irr_plus = np.clip(irr_frac + irr_rmse, 0, 1)
            irr_minus = np.clip(irr_frac - irr_rmse, 0, 1)

            df_plus = year_df.copy()
            df_plus['annual_irr_fraction'] = irr_plus
            pf_plus = _build_pred_features(df_plus, feature_cols, drop_attrs)
            pred_plus, cat_plus = _predict_total(model, pf_plus, df_plus,
                                                 partops, raster_shape,
                                                 valid_mask)

            df_minus = year_df.copy()
            df_minus['annual_irr_fraction'] = irr_minus
            pf_minus = _build_pred_features(df_minus, feature_cols, drop_attrs)
            pred_minus, cat_minus = _predict_total(model, pf_minus, df_minus,
                                                   partops, raster_shape,
                                                   valid_mask)

            std = np.abs(pred_plus - pred_minus) / 2.0
            irr_members = [pred_plus, pred_minus]
            cat_std = _compute_category_sigmas(
                [cat_plus, cat_minus], mode='half_range',
            )

        sigma_irr[year] = std

        for c in partops.CATEGORIES:
            cat_sigma_irr[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(irr_members, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_Irr_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3, M3_TO_AF)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_irr = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'Irr')
    _write_basin_sigma_csv(basin_accum, base_dir, 'Irr')
    logger.info('  σ_irr complete.')
    return sigma_irr, cat_sigma_irr


# ═════════════════════════════════════════════════════════════════════════════
# σ_LULC — Inter-scenario LULC projection spread (2026-2099)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_lulc(
        model,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        input_dir: str,
        vector_dir: str,
        mosaic_res: int,
        gcloud_project: str,
        gcloud_bucket: str,
        tile_size: int,
        end_year: int,
        year_list: list[int],
        skip_download: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Compute σ_LULC: per-pixel std of predictions across 4 USGS LULC
    projection scenarios (B1, B2, A1B, A2) for future years (2026-2099).

    For each scenario the full LULC → AGRI/URBAN → crop_fraction →
    irr_fraction chain is re-derived, so this component subsumes σ_irr
    for future years.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]
        (sigma_lulc, cat_sigma_lulc) — per-year total σ and
        per-category per-year σ arrays.
    """
    from sklearn.linear_model import LinearRegression

    import hydrolibs.dataops as dataops
    import hydrolibs.partitionops as partops
    from hydrolibs.gwops import create_land_use_data
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_LULC (inter-scenario LULC uncertainty)...')
    base_dir = os.path.join(output_dir, 'Sigma_LULC')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Refit the irr-fraction regression (same as create_az_data_parquet)
    irr_train = az_df[az_df.Year.between(1985, 2025)].dropna(
        subset=['annual_irr_fraction', 'annual_crop_fraction']
    )
    irr_train = irr_train[irr_train.annual_crop_fraction > 0]
    irr_reg = LinearRegression().fit(
        irr_train[['annual_crop_fraction']].values,
        irr_train['annual_irr_fraction'].values,
    )

    # Download and mosaic per-scenario tiles
    scenario_mosaic_dirs = {}
    for scenario in USGS_LULC_SCENARIOS:
        logger.info(f'  Preparing per-scenario tiles for {scenario}...')
        sc_tile_dir, _ = dataops.download_gee_data(
            os.path.join(vector_dir, 'AZ.geojson'),
            gcloud_project, gcloud_bucket, input_dir,
            start_year=MACA_FUTURE_START, end_year=end_year,
            skip_download=skip_download, tile_size=tile_size,
            num_workers=40, worker_memory='1G',
            gee_scale=mosaic_res, verbose=False,
            lulc_scenario=scenario,
        )
        sc_mosaic_dir = (
            f'{os.path.dirname(pred_data_dir.rstrip("/"))}'
            f'/../GEE_Mosaics_{mosaic_res}m_LULC_{scenario}/'
        )
        dataops.mosaic_tiles(
            sc_tile_dir, sc_mosaic_dir,
            MACA_FUTURE_START, end_year,
            already_mosaicked=skip_download,
        )
        scenario_mosaic_dirs[scenario] = sc_mosaic_dir

    sigma_lulc = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_sigma_lulc: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    for year in range(MACA_FUTURE_START, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        scenario_preds = []
        scenario_cats = []
        for scenario in USGS_LULC_SCENARIOS:
            sc_raster = f'{scenario_mosaic_dirs[scenario]}Predictor_{year}.tif'
            sc_year_df = year_df.copy()

            # Read per-scenario LULC class and crop fraction bands
            lulc_arr = read_raster_as_arr(
                sc_raster, band=LULC_BAND_INDEX, get_file=False
            )
            crop_frac_arr = read_raster_as_arr(
                sc_raster, band=CROP_FRACTION_BAND_INDEX, get_file=False
            )

            # Re-derive Gaussian-smoothed AGRI, SW, URBAN from LULC class
            lu_df = pd.DataFrame()
            lu_df = create_land_use_data(lu_df, lulc_arr)
            sc_year_df['AGRI'] = lu_df['AGRI'].values[valid_mask]
            sc_year_df['URBAN'] = lu_df['URBAN'].values[valid_mask]

            # Update crop fraction
            crop_frac_valid = crop_frac_arr.ravel()[valid_mask]
            sc_year_df['annual_crop_fraction'] = crop_frac_valid

            # Re-derive irr fraction from crop fraction via regression
            irr_frac = np.zeros_like(crop_frac_valid)
            nonzero = crop_frac_valid > 0
            if nonzero.any():
                irr_frac[nonzero] = np.clip(
                    irr_reg.predict(
                        crop_frac_valid[nonzero].reshape(-1, 1)
                    ), 0, 1
                )
            sc_year_df['annual_irr_fraction'] = irr_frac

            pf = _build_pred_features(sc_year_df, feature_cols, drop_attrs)
            pred, cat = _predict_total(model, pf, sc_year_df, partops,
                                       raster_shape, valid_mask)
            scenario_preds.append(pred)
            scenario_cats.append(cat)

        sc_stack = np.stack(scenario_preds, axis=0)
        std = np.nanstd(sc_stack, axis=0, ddof=1)
        sigma_lulc[year] = std

        cat_std = _compute_category_sigmas(scenario_cats)
        for c in partops.CATEGORIES:
            cat_sigma_lulc[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(
            scenario_preds, year_df, mm_to_m3,
        )
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_LULC_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3, M3_TO_AF)

        if year % 10 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_LULC = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'LULC')
    _write_basin_sigma_csv(basin_accum, base_dir, 'LULC')
    logger.info('  σ_LULC complete.')
    return sigma_lulc, cat_sigma_lulc


# ═════════════════════════════════════════════════════════════════════════════
# σ_gw — GW fraction inter-snapshot uncertainty
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_gw(
        model,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Compute σ_gw: uncertainty from the stepped GW-fraction assignment.

    The pipeline assigns one of 4 Hung et al. snapshots (2000, 2005, 2010,
    2015) per year.  σ_gw is the std of predictions when each of the 4
    snapshots is used, capturing temporal variability in irrigation source
    allocation.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]
        (sigma_gw, cat_sigma_gw) — per-year total σ and
        per-category per-year σ arrays.
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_gw (GW fraction inter-snapshot uncertainty)...')
    base_dir = os.path.join(output_dir, 'Sigma_GW')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Load the 4 GW-fraction rasters from the predictor mosaics
    # (closest available snapshot years in the historical period)
    # Band 12 = annual_gw_fraction
    snapshot_gw_fracs = {}
    for snap_year in GW_FRACTION_SNAPSHOTS:
        raster_file = f'{pred_data_dir}Predictor_{snap_year}.tif'
        if not os.path.exists(raster_file):
            # Fall back to nearest available year
            for fallback in year_list:
                rf = f'{pred_data_dir}Predictor_{fallback}.tif'
                if os.path.exists(rf):
                    raster_file = rf
                    break
        gw_arr = read_raster_as_arr(
            raster_file, band=GW_FRACTION_BAND_INDEX, get_file=False
        )
        snapshot_gw_fracs[snap_year] = gw_arr.ravel()[valid_mask]

    sigma_gw = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_sigma_gw: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    for year in range(start_year, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        snapshot_preds = []
        snapshot_cats = []
        for snap_year in GW_FRACTION_SNAPSHOTS:
            alt_df = year_df.copy()
            alt_df['annual_gw_fraction'] = snapshot_gw_fracs[snap_year]
            pf = _build_pred_features(alt_df, feature_cols, drop_attrs)
            pred, cat = _predict_total(model, pf, alt_df, partops,
                                       raster_shape, valid_mask)
            snapshot_preds.append(pred)
            snapshot_cats.append(cat)

        snap_stack = np.stack(snapshot_preds, axis=0)
        std = np.nanstd(snap_stack, axis=0, ddof=1)
        sigma_gw[year] = std

        cat_std = _compute_category_sigmas(snapshot_cats)
        for c in partops.CATEGORIES:
            cat_sigma_gw[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(snapshot_preds, year_df,
                                            mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_GW_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3, M3_TO_AF)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_gw = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'GW')
    _write_basin_sigma_csv(basin_accum, base_dir, 'GW')
    logger.info('  σ_gw complete.')
    return sigma_gw, cat_sigma_gw


# ═════════════════════════════════════════════════════════════════════════════
# σ_total — Quadrature combination
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_total(
        sigma_components: dict[str, dict[int, np.ndarray]],
        pred_data_dir: str,
        output_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
        prediction_raster_dir: str = '',
        cat_sigma_components: dict[str, dict[str, dict[int, np.ndarray]]] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Combine independent uncertainty components via quadrature.

    σ_total = √(Σ σ_i²)

    Each annual raster is written with two bands:
        Band 1 — σ_total (mm)
        Band 2 — CV  (σ_total / prediction, dimensionless)

    Per-category σ_total rasters are also written when
    *cat_sigma_components* is provided.

    A temporal mean-CV raster across all years is also written.

    Parameters
    ----------
    sigma_components : dict[str, dict[int, np.ndarray]]
        Mapping of component name → {year → 1-D std array}.
        E.g. {'MACA': {...}, 'Model': {...}, 'Irr': {...}, 'GW': {...}}
    prediction_raster_dir : str
        Directory containing total-pumping prediction rasters named
        ``Predicted_GW_{year}_mm.tif``.  Required for CV computation.
    cat_sigma_components : dict or None
        Mapping of component name → {category → {year → 1-D std array}}.
        When provided, per-category σ_total is computed via quadrature
        and written as ``Sigma_Total_{cat}_mm_{year}.tif``.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]
        (sigma_total, cat_sigma_total) — per-year total σ arrays and
        per-category per-year σ_total arrays.
    """
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_total (quadrature combination)...')
    base_dir = os.path.join(output_dir, 'Sigma_Total')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = f'{pred_data_dir}GW_Basin_{year_list[0]}.tif'
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = f'{pred_data_dir}Predictor_{year_list[0]}.tif'

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    all_years = set()
    for comp in sigma_components.values():
        all_years.update(comp.keys())
    all_years = sorted(all_years)

    sigma_total = {}
    yearly_stats = {}

    # Per-category σ_total via quadrature
    import hydrolibs.partitionops as partops
    cat_sigma_total: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    # Also track per-component contribution
    contribution_records = []

    # Accumulate CV grids for temporal mean
    cv_sum = None
    cv_count = None

    for year in all_years:
        if year < start_year or year > end_year:
            continue

        n_pixels = None
        variance_sum = None

        year_contributions = {'Year': year}
        for name, comp in sigma_components.items():
            if year in comp:
                std_arr = comp[year]
                if variance_sum is None:
                    n_pixels = len(std_arr)
                    variance_sum = np.zeros(n_pixels, dtype=np.float64)
                # Scale sample-based σ by t/z before quadrature
                t_scale = COMPONENT_T_SCALE.get(name, 1.0)
                variance_sum += (std_arr.astype(np.float64) * t_scale) ** 2
                year_contributions[f'Mean_Sigma_{name}_mm'] = round(
                    float(np.nanmean(std_arr)), 4
                )
            else:
                year_contributions[f'Mean_Sigma_{name}_mm'] = 0.0
        for name in sigma_components:
            year_contributions[f'N_{name}'] = COMPONENT_N.get(name, 0)

        if variance_sum is None:
            continue

        total_std = np.sqrt(variance_sum).astype(np.float32)
        sigma_total[year] = total_std

        # Per-category σ_total via quadrature
        if cat_sigma_components:
            for cat_name in partops.CATEGORIES:
                cat_var = np.zeros(n_pixels, dtype=np.float64)
                for comp_name in cat_sigma_components:
                    cat_comp = cat_sigma_components[comp_name].get(cat_name, {})
                    if year in cat_comp:
                        t_sc = COMPONENT_T_SCALE.get(comp_name, 1.0)
                        cat_var += (cat_comp[year].astype(np.float64) * t_sc) ** 2
                cat_total_std = np.sqrt(cat_var).astype(np.float32)
                cat_sigma_total[cat_name][year] = cat_total_std
                _write_std_raster(
                    cat_total_std, basin_flat, valid_mask, raster_shape,
                    ref_raster_file,
                    os.path.join(
                        raster_dir,
                        f'Sigma_Total_{cat_name}_mm_{year}.tif',
                    ),
                    read_raster_as_arr, write_raster,
                )

        # Build σ_total grid
        sigma_grid = np.full(basin_flat.shape[0], np.nan, dtype=np.float32)
        sigma_grid[valid_mask] = total_std
        sigma_grid = sigma_grid.reshape(raster_shape)

        # Compute CV grid from prediction raster
        pred_file = f'{prediction_raster_dir}Predicted_GW_{year}_mm.tif'
        if os.path.exists(pred_file):
            pred_arr = read_raster_as_arr(pred_file, get_file=False)
            with np.errstate(invalid='ignore', divide='ignore'):
                cv_grid = np.where(
                    np.abs(pred_arr) > 0,
                    sigma_grid / np.abs(pred_arr),
                    np.nan,
                ).astype(np.float32)
            # Accumulate for temporal mean CV
            if cv_sum is None:
                cv_sum = np.zeros(raster_shape, dtype=np.float64)
                cv_count = np.zeros(raster_shape, dtype=np.int32)
            finite = np.isfinite(cv_grid)
            cv_sum[finite] += cv_grid[finite]
            cv_count[finite] += 1
        else:
            cv_grid = np.full(raster_shape, np.nan, dtype=np.float32)

        # Write 2-band raster (σ_total, CV)
        _write_sigma_cv_raster(
            sigma_grid, cv_grid, ref_raster_file,
            os.path.join(raster_dir, f'Sigma_Total_mm_{year}.tif'),
            read_raster_as_arr,
        )

        stats = _pixel_stats(total_std, mm_to_m3, M3_TO_AF)
        yearly_stats[year] = stats
        year_contributions['Mean_Sigma_Total_mm'] = stats['Mean_Depth_mm']
        year_contributions['Volume_AF'] = stats['Volume_AF']
        contribution_records.append(year_contributions)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_total = '
                        f'{stats["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'Total')

    # Per-component contribution table
    contrib_df = pd.DataFrame(contribution_records)
    contrib_df.to_csv(os.path.join(base_dir, 'Component_Contributions.csv'), index=False)

    # Write temporal mean CV raster
    if cv_sum is not None and np.any(cv_count > 0):
        with np.errstate(invalid='ignore'):
            mean_cv = np.where(
                cv_count > 0,
                (cv_sum / cv_count).astype(np.float32),
                np.nan,
            ).astype(np.float32)
        _, rfile = read_raster_as_arr(ref_raster_file, get_file=True)
        write_raster(
            mean_cv, rfile, rfile.transform,
            os.path.join(base_dir, 'Mean_CV.tif'),
            no_data_value=np.nan, num_bands=1,
        )
        rfile.close()
        logger.info(f'  Mean CV raster written to {base_dir}Mean_CV.tif')

    logger.info('  σ_total complete.')
    return sigma_total, cat_sigma_total


def compute_basin_sigma_total(output_dir: str) -> None:
    """Combine per-component basin/sub-basin σ CSVs via quadrature.

    Reads ``{Basin|Subbasin}_Sigma_{comp}.csv`` from each
    ``Sigma_{comp}/`` directory, joins on ``(Year, Region)``, and
    writes ``Basin_Sigma_Total.csv`` / ``Subbasin_Sigma_Total.csv``
    into ``Sigma_Total/``.
    """
    component_labels = ('MACA', 'Model', 'Irr', 'LULC', 'GW')
    total_dir = os.path.join(output_dir, 'Sigma_Total')

    for level in ('Basin', 'Subbasin'):
        merged = None
        found_components = []
        for comp in component_labels:
            csv_path = (
                os.path.join(output_dir, f'Sigma_{comp}/{level}_Sigma_{comp}.csv')
            )
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                'Sigma_Volume_AF': f'Sigma_{comp}_AF',
                'Mean_Volume_AF': f'Mean_{comp}_AF',
                'Sigma_Volume_m3': f'Sigma_{comp}_m3',
                'Mean_Volume_m3': f'Mean_{comp}_m3',
            })[['Year', 'Region',
                f'Sigma_{comp}_AF', f'Mean_{comp}_AF',
                f'Sigma_{comp}_m3', f'Mean_{comp}_m3']]
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=['Year', 'Region'], how='outer')
            found_components.append(comp)

        if merged is None or not found_components:
            continue

        # Quadrature: σ_total = √(Σ (t_scale_i · σ_i)²)
        # Sample-based components (Model, GW) are inflated by t/z before
        # quadrature so the final ± CI_Z × σ_total is t-corrected.
        sigma_af_cols = [f'Sigma_{c}_AF' for c in found_components]
        sigma_m3_cols = [f'Sigma_{c}_m3' for c in found_components]
        merged[sigma_af_cols] = merged[sigma_af_cols].fillna(0)
        merged[sigma_m3_cols] = merged[sigma_m3_cols].fillna(0)

        scaled_var_af = pd.DataFrame(index=merged.index, dtype=np.float64)
        scaled_var_m3 = pd.DataFrame(index=merged.index, dtype=np.float64)
        for comp in found_components:
            t_scale = COMPONENT_T_SCALE.get(comp, 1.0)
            scaled_var_af[comp] = (merged[f'Sigma_{comp}_AF'] * t_scale) ** 2
            scaled_var_m3[comp] = (merged[f'Sigma_{comp}_m3'] * t_scale) ** 2

        merged['Sigma_Total_AF'] = np.sqrt(
            scaled_var_af.sum(axis=1)
        ).round(2)
        merged['Sigma_Total_m3'] = np.sqrt(
            scaled_var_m3.sum(axis=1)
        ).round(2)

        # Mean volume: average of component means (they should be similar)
        mean_af_cols = [f'Mean_{c}_AF' for c in found_components]
        mean_m3_cols = [f'Mean_{c}_m3' for c in found_components]
        merged[mean_af_cols] = merged[mean_af_cols].fillna(0)
        merged[mean_m3_cols] = merged[mean_m3_cols].fillna(0)
        merged['Mean_Volume_AF'] = merged[mean_af_cols].mean(axis=1).round(2)
        merged['Mean_Volume_m3'] = merged[mean_m3_cols].mean(axis=1).round(2)

        with np.errstate(invalid='ignore', divide='ignore'):
            merged['CV'] = np.where(
                np.abs(merged['Mean_Volume_AF']) > 0,
                merged['Sigma_Total_AF'] / np.abs(merged['Mean_Volume_AF']),
                np.nan,
            ).round(6)
        merged['Lower_95CI_m3'] = (
            merged['Mean_Volume_m3'] - CI_Z * merged['Sigma_Total_m3']
        ).round(2)
        merged['Upper_95CI_m3'] = (
            merged['Mean_Volume_m3'] + CI_Z * merged['Sigma_Total_m3']
        ).round(2)
        merged['Lower_95CI_AF'] = (
            merged['Mean_Volume_AF'] - CI_Z * merged['Sigma_Total_AF']
        ).round(2)
        merged['Upper_95CI_AF'] = (
            merged['Mean_Volume_AF'] + CI_Z * merged['Sigma_Total_AF']
        ).round(2)

        # Ensemble size metadata (constant per component)
        n_cols = []
        for comp in found_components:
            col = f'N_{comp}'
            merged[col] = COMPONENT_N.get(comp, 0)
            n_cols.append(col)

        out_cols = (
            ['Year', 'Region']
            + sigma_m3_cols + ['Sigma_Total_m3']
            + sigma_af_cols + ['Sigma_Total_AF']
            + ['Mean_Volume_m3', 'Mean_Volume_AF', 'CV',
               'Lower_95CI_m3', 'Upper_95CI_m3',
               'Lower_95CI_AF', 'Upper_95CI_AF']
            + n_cols
        )
        merged.sort_values(['Year', 'Region'], inplace=True)
        merged[out_cols].to_csv(
            f'{total_dir}{level}_Sigma_Total.csv', index=False,
        )
        logger.info(f'  Wrote {total_dir}{level}_Sigma_Total.csv')


# ═════════════════════════════════════════════════════════════════════════════
# σ_CU — Consumptive-use inter-GCM spread (2026-2099 only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_cu(
        gcm_mosaic_dirs: dict[str, str],
        pred_data_dir: str,
        output_dir: str,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
) -> None:
    """
    Compute σ_CU: per-pixel std of consumptive use across 5 GCMs.

    CU = max(ET_irr − Peff_irr, 0).  Per-GCM ET and Peff are read from
    the per-GCM predictor rasters built during σ_MACA.  Irrigation and
    GW fractions are taken from the ensemble predictor (fixed across GCMs).

    Writes σ rasters for Irrigation_CU, Irrigation_GW_CU, and
    Irrigation_SW_CU for future years (2026-2099).
    """
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_CU (consumptive-use inter-GCM spread)...')
    sigma_cu_dir = os.path.join(output_dir, 'Sigma_CU/Rasters')
    makedirs(sigma_cu_dir)

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    yearly_stats = {}

    for year in range(MACA_FUTURE_START, end_year + 1):
        ensemble_raster = f'{pred_data_dir}Predictor_{year}.tif'
        if not os.path.exists(ensemble_raster):
            continue

        with rio.open(ensemble_raster) as src:
            irr_frac = src.read(IRR_FRACTION_BAND_INDEX).astype(np.float32)
            gw_frac = src.read(GW_FRACTION_BAND_INDEX).astype(np.float32)
        irr_frac = np.clip(np.nan_to_num(irr_frac, nan=0.0), 0, 1)
        gw_frac = np.clip(np.nan_to_num(gw_frac, nan=1.0), 0, 1)

        cu_stack, cu_gw_stack, cu_sw_stack = [], [], []

        for gcm in MACA_REPRESENTATIVE_GCMS:
            gcm_raster = f'{gcm_mosaic_dirs[gcm]}Predictor_{year}.tif'
            if not os.path.exists(gcm_raster):
                continue
            with rio.open(gcm_raster) as src:
                et = src.read(ET_BAND_INDEX).astype(np.float32)
                peff = src.read(PEFF_BAND_INDEX).astype(np.float32)

            et_irr = et * irr_frac
            peff_irr = peff * irr_frac
            cu = np.maximum(et_irr - peff_irr, 0)
            cu_gw = cu * gw_frac
            cu_sw = cu - cu_gw

            cu_stack.append(cu)
            cu_gw_stack.append(cu_gw)
            cu_sw_stack.append(cu_sw)

        if len(cu_stack) < 2:
            continue

        with rio.open(ensemble_raster) as src:
            raster_profile = src.profile.copy()
        raster_profile.update(count=1, dtype=np.float32, nodata=np.nan)

        for label, stack in [
            ('Irrigation_CU', cu_stack),
            ('Irrigation_GW_CU', cu_gw_stack),
            ('Irrigation_SW_CU', cu_sw_stack),
        ]:
            sigma_arr = np.nanstd(
                np.stack(stack), axis=0, ddof=1,
            ).astype(np.float32)
            out_path = f'{sigma_cu_dir}Sigma_{label}_mm_{year}.tif'
            with rio.open(out_path, 'w', **raster_profile) as dst:
                dst.write(sigma_arr, 1)

        sigma_cu_total = np.nanstd(
            np.stack(cu_stack), axis=0, ddof=1,
        )
        yearly_stats[year] = _pixel_stats(
            sigma_cu_total.ravel(), mm_to_m3, M3_TO_AF,
        )

        if year % 10 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_CU = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, os.path.join(output_dir, 'Sigma_CU'), 'CU')
    logger.info('  σ_CU complete.')


# ═════════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_uncertainty_quantification(
        model,
        feature_cols: list[str],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        model_dir: str,
        input_dir: str,
        output_dir: str,
        vector_dir: str,
        mosaic_res: int,
        gcloud_project: str,
        gcloud_bucket: str,
        tile_size: int,
        start_year: int,
        end_year: int,
        year_list: list[int],
        n_trials: int = 100,
        n_dask_workers: int = 10,
        use_dask: bool = True,
        skip_download: bool = False,
        subbasin_shp: str = '',
        ama_code_map: dict | None = None,
        basin_shp: str = '',
) -> None:
    """
    Run the full hybrid uncertainty quantification pipeline.

    Computes σ_MACA (future only), σ_model, σ_irr (historical only),
    σ_LULC (future only), σ_gw, and combines them into σ_total via
    quadrature.  Writes per-component and total uncertainty rasters,
    summary CSVs, and time-series plots.
    """
    import hydrolibs.visualops as vizops
    from hydrolibs.sysops import makedirs

    logger.info('=' * 60)
    logger.info('Step 3b: Hybrid Uncertainty Quantification')
    logger.info('=' * 60)

    unc_dir = os.path.join(model_dir, 'Full_Prediction_XGB/Uncertainty')
    makedirs(unc_dir)

    # ── σ_MACA ──
    sigma_maca, cat_sigma_maca, gcm_mosaic_dirs = compute_sigma_maca(
        model, feature_cols, az_df, drop_attrs,
        pred_data_dir, unc_dir, input_dir, vector_dir,
        mosaic_res, gcloud_project, gcloud_bucket,
        tile_size, end_year, year_list,
        skip_download=skip_download,
    )

    # ── σ_model ──
    sigma_model, cat_sigma_model = compute_sigma_model(
        x_train, y_train, feature_cols, az_df,
        drop_attrs, pred_data_dir, unc_dir,
        start_year, end_year, year_list, mosaic_res,
        n_trials=n_trials, n_dask_workers=n_dask_workers,
        use_dask=use_dask,
    )

    # ── σ_irr (historical only, 1896-2025) ──
    sigma_irr, cat_sigma_irr = compute_sigma_irr(
        model, feature_cols, az_df, drop_attrs,
        pred_data_dir, unc_dir, start_year, end_year,
        year_list, mosaic_res,
    )

    # ── σ_LULC (future only, 2026-2099 — subsumes σ_irr) ──
    sigma_lulc, cat_sigma_lulc = compute_sigma_lulc(
        model, feature_cols, az_df, drop_attrs,
        pred_data_dir, unc_dir, input_dir, vector_dir,
        mosaic_res, gcloud_project, gcloud_bucket,
        tile_size, end_year, year_list,
        skip_download=skip_download,
    )

    # ── σ_gw ──
    sigma_gw, cat_sigma_gw = compute_sigma_gw(
        model, feature_cols, az_df, drop_attrs,
        pred_data_dir, unc_dir, start_year, end_year,
        year_list, mosaic_res,
    )

    # ── σ_total ──
    sigma_components = {
        'MACA': sigma_maca,
        'Model': sigma_model,
        'Irr': sigma_irr,
        'LULC': sigma_lulc,
        'GW': sigma_gw,
    }
    cat_sigma_components = {
        'MACA': cat_sigma_maca,
        'Model': cat_sigma_model,
        'Irr': cat_sigma_irr,
        'LULC': cat_sigma_lulc,
        'GW': cat_sigma_gw,
    }
    prediction_raster_dir = (
        os.path.join(model_dir, 'Full_Prediction_XGB/Predicted_Rasters/Depth_mm')
    )
    compute_sigma_total(
        sigma_components, pred_data_dir, unc_dir,
        start_year, end_year, year_list, mosaic_res,
        prediction_raster_dir=prediction_raster_dir,
        cat_sigma_components=cat_sigma_components,
    )

    # ── Basin / sub-basin σ_total (quadrature of per-component CSVs) ──
    compute_basin_sigma_total(unc_dir)
    _plot_basin_sigma_time_series(unc_dir)

    # ── Visualisations ──
    _plot_uncertainty_time_series(
        sigma_components, unc_dir, start_year, end_year,
        year_list, mosaic_res, pred_data_dir, vizops,
    )

    # ── Augment prediction rasters with uncertainty bands ──
    prediction_base_dir = (
        os.path.join(model_dir, 'Full_Prediction_XGB/Predicted_Rasters')
    )
    full_pred_dir = os.path.join(model_dir, 'Full_Prediction_XGB')
    augment_prediction_rasters(
        sigma_total_raster_dir=f'{unc_dir}Sigma_Total/Rasters/',
        prediction_base_dir=prediction_base_dir,
        start_year=start_year,
        end_year=end_year,
        mosaic_res=mosaic_res,
    )

    # ── Augment category rasters (Irrigation, Non_Irrigation, …) ──
    augment_category_rasters(
        prediction_dir=full_pred_dir,
        sigma_total_raster_dir=os.path.join(
            unc_dir, 'Sigma_Total/Rasters/',
        ),
        start_year=start_year,
        end_year=end_year,
        mosaic_res=mosaic_res,
    )

    # ── σ_CU (inter-GCM spread in consumptive use) ──
    compute_sigma_cu(
        gcm_mosaic_dirs=gcm_mosaic_dirs,
        pred_data_dir=pred_data_dir,
        output_dir=unc_dir,
        end_year=end_year,
        year_list=year_list,
        mosaic_res=mosaic_res,
    )

    # ── Augment CU rasters ──
    augment_cu_rasters(
        sigma_cu_raster_dir=f'{unc_dir}Sigma_CU/Rasters/',
        prediction_dir=full_pred_dir,
        start_year=start_year,
        end_year=end_year,
        mosaic_res=mosaic_res,
    )

    # ── Augment IE rasters (needs augmented CU + category rasters) ──
    augment_ie_rasters(
        prediction_dir=full_pred_dir,
        start_year=start_year,
        end_year=end_year,
    )

    # ── Re-plot prediction time series with uncertainty bounds ──
    # Derive all uncertainty data directly from the augmented 6-band
    # rasters using zonal statistics with basin / sub-basin shapefiles.
    _replot_from_augmented_rasters(
        prediction_dir=full_pred_dir,
        basin_shp=basin_shp,
        subbasin_shp=subbasin_shp,
        ama_code_map=ama_code_map,
        start_year=start_year,
        end_year=end_year,
        mosaic_res=mosaic_res,
    )

    logger.info(f'Uncertainty quantification complete. Results in {unc_dir}')


def augment_prediction_rasters(
        sigma_total_raster_dir: str,
        prediction_base_dir: str,
        start_year: int,
        end_year: int,
        mosaic_res: int | float,
) -> None:
    """
    Augment each annual prediction raster with uncertainty bands.

    Rewrites each ``Predicted_GW_{year}_{unit}.tif`` (for mm, ft, m³, AF)
    as a 6-band GeoTIFF:

        1. Prediction (unit)
        2. σ_total (unit)
        3. CV  (σ / |pred|)
        4. SNR (|pred| / σ)
        5. Lower 95 % CI  (pred − CI_Z·σ)
        6. Upper 95 % CI  (pred + CI_Z·σ)

    σ_total rasters are stored in mm; they are scaled to the target unit
    before writing.
    """
    from hydrolibs.rasterops import read_raster_as_arr

    logger.info('Augmenting prediction rasters with uncertainty bands...')

    unit_subdirs = {
        'mm': 'Depth_mm/',
        'ft': 'Depth_ft/',
        'm3': 'Volume_m3/',
        'AF': 'Volume_AF/',
    }

    pixel_area_m2 = mosaic_res ** 2
    mm_to_ft = 1 / 304.8
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = 1 / 1233.48

    sigma_scale = {
        'mm': 1.0,
        'ft': mm_to_ft,
        'm3': mm_to_m3,
        'AF': mm_to_m3 * m3_to_af,
    }

    for unit, subdir in unit_subdirs.items():
        pred_dir = f'{prediction_base_dir}{subdir}'
        scale = sigma_scale[unit]

        band_descriptions = [
            f'prediction_{unit}', f'sigma_total_{unit}', 'CV', 'SNR',
            f'lower_95CI_{unit}', f'upper_95CI_{unit}',
        ]

        for year in range(start_year, end_year + 1):
            pred_file = os.path.join(pred_dir, f'Predicted_GW_{year}_{unit}.tif')
            sigma_file = (
                f'{sigma_total_raster_dir}Sigma_Total_mm_{year}.tif'
            )

            if not os.path.exists(pred_file) or not os.path.exists(sigma_file):
                continue

            pred_arr = read_raster_as_arr(pred_file, get_file=False)
            # σ_total raster is 2-band (band 1 = σ); read band 1
            sigma_mm = read_raster_as_arr(sigma_file, band=1, get_file=False)
            sigma_arr = (sigma_mm * scale).astype(np.float32)

            with np.errstate(invalid='ignore', divide='ignore'):
                abs_pred = np.abs(pred_arr)
                cv = np.where(
                    abs_pred > 0, sigma_arr / abs_pred, np.nan,
                ).astype(np.float32)
                snr = np.where(
                    sigma_arr > 0, abs_pred / sigma_arr, np.nan,
                ).astype(np.float32)

            lower_ci = (pred_arr - CI_Z * sigma_arr).astype(np.float32)
            upper_ci = (pred_arr + CI_Z * sigma_arr).astype(np.float32)

            # Read spatial metadata from the original prediction raster
            with rio.open(pred_file) as src:
                profile = src.profile.copy()

            profile.update(count=6, dtype=np.float32, nodata=np.nan)

            with rio.open(pred_file, 'w', **profile) as dst:
                dst.write(pred_arr.astype(np.float32), 1)
                dst.write(sigma_arr, 2)
                dst.write(cv, 3)
                dst.write(snr, 4)
                dst.write(lower_ci, 5)
                dst.write(upper_ci, 6)
                for i, desc in enumerate(band_descriptions, 1):
                    dst.set_band_description(i, desc)

            if year % 20 == 0 or year == end_year:
                logger.info(f'  Augmented {unit} year {year}')

    logger.info('  Prediction raster augmentation complete.')


def augment_category_rasters(
        prediction_dir: str,
        sigma_total_raster_dir: str,
        start_year: int,
        end_year: int,
        mosaic_res: int | float,
) -> None:
    """
    Augment per-category rasters with uncertainty bands.

    Per-category σ_total rasters are read from *sigma_total_raster_dir*
    (written by ``compute_sigma_total``).  These are computed via
    quadrature over per-category ensemble spreads, so each category's
    uncertainty is derived directly from the ensemble — not approximated
    by linear scaling of the total σ.
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr

    logger.info('Augmenting category rasters with uncertainty bands...')

    unit_subdirs = {
        'mm': 'Depth_mm/', 'ft': 'Depth_ft/',
        'm3': 'Volume_m3/', 'AF': 'Volume_AF/',
    }

    pixel_area_m2 = mosaic_res ** 2
    mm_to_ft = 1 / 304.8
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = 1 / 1233.48

    sigma_scale = {
        'mm': 1.0,
        'ft': mm_to_ft,
        'm3': mm_to_m3,
        'AF': mm_to_m3 * m3_to_af,
    }

    for cat in partops.CATEGORIES:
        for unit, subdir in unit_subdirs.items():
            cat_dir = os.path.join(prediction_dir, f'{cat}_Rasters/{subdir}')
            scale = sigma_scale[unit]

            band_descriptions = [
                f'prediction_{unit}', f'sigma_{unit}', 'CV', 'SNR',
                f'lower_95CI_{unit}', f'upper_95CI_{unit}',
            ]

            for year in range(start_year, end_year + 1):
                cat_file = os.path.join(cat_dir, f'{cat}_{year}_{unit}.tif')
                sigma_cat_file = os.path.join(
                    sigma_total_raster_dir,
                    f'Sigma_Total_{cat}_mm_{year}.tif',
                )

                if not os.path.exists(cat_file) or \
                        not os.path.exists(sigma_cat_file):
                    continue

                with rio.open(cat_file) as src:
                    cat_pred = src.read(1)
                    profile = src.profile.copy()

                # Read per-category σ_total (mm) and scale to target unit
                sigma_mm = read_raster_as_arr(
                    sigma_cat_file, get_file=False,
                )
                sigma_cat = (sigma_mm * scale).astype(np.float32)

                _write_augmented_raster(
                    cat_pred, sigma_cat, cat_file, profile,
                    band_descriptions,
                )

        logger.info(f'  Augmented {cat} rasters (all units)')

    logger.info('  Category raster augmentation complete.')


def augment_cu_rasters(
        sigma_cu_raster_dir: str,
        prediction_dir: str,
        start_year: int,
        end_year: int,
        mosaic_res: int | float,
) -> None:
    """
    Augment CU rasters with uncertainty bands derived from σ_CU.

    For future years (≥ 2026), σ_CU is the inter-GCM spread in
    max(ET_irr − Peff_irr, 0).  For historical years, σ_CU = 0.
    """
    logger.info('Augmenting CU rasters with uncertainty bands...')

    unit_subdirs = {
        'mm': 'Depth_mm/', 'ft': 'Depth_ft/',
        'm3': 'Volume_m3/', 'AF': 'Volume_AF/',
    }

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = 1 / 1233.48
    sigma_scale = {
        'mm': 1.0,
        'ft': 1 / 304.8,
        'm3': mm_to_m3,
        'AF': mm_to_m3 * m3_to_af,
    }

    for cu_cat in CU_CATEGORIES:
        for unit, subdir in unit_subdirs.items():
            cu_dir = os.path.join(prediction_dir, f'{cu_cat}_Rasters/{subdir}')
            scale = sigma_scale[unit]

            band_descriptions = [
                f'prediction_{unit}', f'sigma_{unit}', 'CV', 'SNR',
                f'lower_95CI_{unit}', f'upper_95CI_{unit}',
            ]

            for year in range(start_year, end_year + 1):
                cu_file = f'{cu_dir}{cu_cat}_{year}_{unit}.tif'
                if not os.path.exists(cu_file):
                    continue

                with rio.open(cu_file) as src:
                    cu_pred = src.read(1)
                    profile = src.profile.copy()

                # Read σ_CU in mm, scale to target unit
                sigma_file = (
                    f'{sigma_cu_raster_dir}Sigma_{cu_cat}_mm_{year}.tif'
                )
                if os.path.exists(sigma_file):
                    with rio.open(sigma_file) as src:
                        sigma_arr = (src.read(1) * scale).astype(np.float32)
                else:
                    sigma_arr = np.zeros_like(cu_pred, dtype=np.float32)

                _write_augmented_raster(
                    cu_pred, sigma_arr, cu_file, profile,
                    band_descriptions,
                )

        logger.info(f'  Augmented {cu_cat} rasters (all units)')

    logger.info('  CU raster augmentation complete.')


def augment_ie_rasters(
        prediction_dir: str,
        start_year: int,
        end_year: int,
) -> None:
    """
    Augment IE (irrigation efficiency) rasters with uncertainty bands.

    IE = CU / withdrawal.  σ_IE is propagated via the standard ratio
    formula::

        σ_IE / IE = √( (σ_CU / CU)² + (σ_wd / wd)² )

    CU and withdrawal rasters must already be augmented (6-band) so
    that band 3 = CV is available.
    """
    logger.info('Augmenting IE rasters with uncertainty bands...')

    band_descriptions = [
        'efficiency', 'sigma', 'CV', 'SNR',
        'lower_95CI', 'upper_95CI',
    ]

    for ie_cat in IE_CATEGORIES:
        cu_cat = IE_CU_MAP[ie_cat]
        wd_cat = IE_WITHDRAWAL_MAP[ie_cat]

        ie_dir = os.path.join(prediction_dir, f'{ie_cat}_Rasters')
        cu_dir = os.path.join(prediction_dir, f'{cu_cat}_Rasters/Depth_mm')
        wd_dir = os.path.join(prediction_dir, f'{wd_cat}_Rasters/Depth_mm')

        for year in range(start_year, end_year + 1):
            ie_file = f'{ie_dir}{ie_cat}_{year}.tif'
            cu_file = f'{cu_dir}{cu_cat}_{year}_mm.tif'
            wd_file = f'{wd_dir}{wd_cat}_{year}_mm.tif'

            if not (os.path.exists(ie_file) and os.path.exists(cu_file)
                    and os.path.exists(wd_file)):
                continue

            with rio.open(ie_file) as src:
                ie_pred = src.read(1)
                profile = src.profile.copy()

            # CU raster (augmented): band 3 = CV_CU
            with rio.open(cu_file) as src:
                cv_cu = (src.read(3).astype(np.float32)
                         if src.count >= 3
                         else np.zeros_like(ie_pred, dtype=np.float32))

            # Withdrawal raster (augmented): band 3 = CV_wd
            with rio.open(wd_file) as src:
                cv_wd = (src.read(3).astype(np.float32)
                         if src.count >= 3
                         else np.zeros_like(ie_pred, dtype=np.float32))

            # σ_IE = IE × √(CV_CU² + CV_wd²)
            with np.errstate(invalid='ignore'):
                cv_cu_sq = np.nan_to_num(cv_cu, nan=0.0) ** 2
                cv_wd_sq = np.nan_to_num(cv_wd, nan=0.0) ** 2
                combined_cv = np.sqrt(cv_cu_sq + cv_wd_sq).astype(np.float32)
            sigma_ie = (np.abs(ie_pred) * combined_cv).astype(np.float32)

            _write_augmented_raster(
                ie_pred, sigma_ie, ie_file, profile, band_descriptions,
            )

        logger.info(f'  Augmented {ie_cat} rasters')

    logger.info('  IE raster augmentation complete.')


def _replot_with_uncertainty(
        unc_dir: str,
        prediction_dir: str,
        start_year: int,
        end_year: int,
        subbasin_shp: str = '',
        ama_code_map: dict | None = None,
) -> None:
    """Re-plot prediction time series with model uncertainty bounds.

    Called after all σ rasters and CSVs are generated.  Reads σ from
    ``Uncertainty_Summary_Total.csv`` (and per-component CSVs for
    basin/sub-basin plots) and re-calls the ``visualops`` plotting
    functions with the ``sigma_data`` / ``sigma_basin_yearly`` /
    ``sigma_subbasin_yearly`` parameters to overlay 95 % CI bands.
    """
    import hydrolibs.visualops as vizops

    logger.info('Re-plotting time series with model uncertainty bounds...')

    # ── helpers ────────────────────────────────────────────────────────────
    def _read_sigma_summary(csv_path):
        """Read Uncertainty_Summary CSV → {year: metrics dict}."""
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        return {
            int(r.Year): {c: float(getattr(r, c))
                          for c in ('Mean_Depth_mm', 'Mean_Depth_ft',
                                    'Volume_m3', 'Volume_AF')
                          if c in df.columns}
            for _, r in df.iterrows()
        }

    def _read_predictions_csv(csv_path):
        """Read Full_Period_Time_Series.csv → (yearly_preds, actual_data)."""
        if not os.path.exists(csv_path):
            return None, None
        df = pd.read_csv(csv_path)
        pred_cols = ('Mean_Depth_mm', 'Mean_Depth_ft', 'Volume_m3',
                     'Volume_AF')
        yearly_preds = {
            int(r.Year): {c: float(getattr(r, c)) for c in pred_cols
                          if c in df.columns}
            for _, r in df.iterrows()
        }
        actual_data = None
        if 'Actual_Depth_mm' in df.columns:
            act = df.dropna(subset=['Actual_Depth_mm'])
            if not act.empty:
                actual_data = {
                    int(r.Year): {
                        'Mean_Depth_mm': float(r.Actual_Depth_mm),
                        'Volume_AF': float(r.Actual_Volume_AF),
                    }
                    for _, r in act.iterrows()
                }
        return yearly_preds, actual_data

    def _read_region_sigma(csv_path):
        """Read Basin/Subbasin_Sigma_Total.csv → {year: {region: dict}}."""
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        result = {}
        for _, r in df.iterrows():
            y = int(r.Year)
            result.setdefault(y, {})[r.Region] = {
                'Sigma_Volume_AF': float(r.Sigma_Total_AF),
                'CV': float(r.CV) if pd.notna(r.CV) else 0.0,
            }
        return result

    def _read_basin_yearly(csv_path, col):
        """Read Basin/Subbasin_Annual_GW.csv → (yearly, actual_yearly)."""
        if not os.path.exists(csv_path):
            return None, None
        df = pd.read_csv(csv_path)
        yearly = {}
        actual_yearly = None
        pred_cols = ('Mean_Depth_mm', 'Mean_Depth_ft', 'Volume_m3',
                     'Volume_AF')
        for _, r in df.iterrows():
            y = int(r.Year)
            name = getattr(r, col)
            yearly.setdefault(y, {})[name] = {
                c: float(getattr(r, c)) for c in pred_cols if c in df.columns
            }
        if 'Actual_Depth_mm' in df.columns:
            act = df.dropna(subset=['Actual_Depth_mm'])
            if not act.empty:
                actual_yearly = {}
                for _, r in act.iterrows():
                    y = int(r.Year)
                    name = getattr(r, col)
                    actual_yearly.setdefault(y, {})[name] = {
                        'Mean_Depth_mm': float(r.Actual_Depth_mm),
                        'Volume_AF': float(r.Actual_Volume_AF),
                    }
        return yearly, actual_yearly

    # ── 1. Total pumping ──────────────────────────────────────────────────
    sigma_total = _read_sigma_summary(
        f'{unc_dir}Sigma_Total/Uncertainty_Summary_Total.csv')
    if sigma_total:
        yearly_preds, actual_data = _read_predictions_csv(
            os.path.join(prediction_dir, 'Full_Period_Time_Series.csv'))
        if yearly_preds:
            vizops.create_full_period_time_series(
                yearly_preds, prediction_dir,
                start_year=start_year, end_year=end_year,
                actual_data=actual_data,
                sigma_data=sigma_total,
            )

    # ── 2. Category products (approximate σ via CV_total) ─────────────────
    if sigma_total:
        yearly_preds_total, _ = _read_predictions_csv(
            os.path.join(prediction_dir, 'Full_Period_Time_Series.csv'))
        cv_total = {}
        if yearly_preds_total:
            for y, s in sigma_total.items():
                if y in yearly_preds_total:
                    vol = yearly_preds_total[y].get('Volume_AF', 0)
                    if abs(vol) > 0:
                        cv_total[y] = s['Volume_AF'] / abs(vol)

        cat_titles = {
            'Irrigation': 'Irrigation',
            'Non_Irrigation': 'Non-Irrigation',
            'Irrigation_GW': 'Irrigation GW',
            'Irrigation_SW': 'Irrigation SW',
            'Non_Irrigation_GW': 'Non-Irrigation GW',
            'Non_Irrigation_SW': 'Non-Irrigation SW',
            'Total_GW': 'Total GW',
            'Total_SW': 'Total SW',
        }
        for cat, title in cat_titles.items():
            cat_dir = os.path.join(prediction_dir, cat)
            cat_preds, _ = _read_predictions_csv(
                os.path.join(cat_dir, 'Full_Period_Time_Series.csv'))
            if not cat_preds:
                continue
            cat_sigma = {}
            for y, cv in cv_total.items():
                if y in cat_preds:
                    p = cat_preds[y]
                    cat_sigma[y] = {
                        'Mean_Depth_mm': cv * abs(p.get('Mean_Depth_mm', 0)),
                        'Volume_AF': cv * abs(p.get('Volume_AF', 0)),
                    }
            if cat_sigma:
                vizops.create_full_period_time_series(
                    cat_preds, cat_dir,
                    start_year=start_year, end_year=end_year,
                    title_prefix=title,
                    sigma_data=cat_sigma,
                )

        # CU categories (use σ_CU for Irrigation_CU, CV approx for others)
        cu_sigma = _read_sigma_summary(
            f'{unc_dir}Sigma_CU/Uncertainty_Summary_CU.csv')
        cu_titles = {
            'Irrigation_CU': 'Irrigation Consumptive Use',
            'Irrigation_GW_CU': 'Irrigation GW Consumptive Use',
            'Irrigation_SW_CU': 'Irrigation SW Consumptive Use',
        }
        for cu_cat, title in cu_titles.items():
            cu_dir = os.path.join(prediction_dir, cu_cat)
            cu_preds, _ = _read_predictions_csv(
                f'{cu_dir}Full_Period_Time_Series.csv')
            if not cu_preds:
                continue
            if cu_cat == 'Irrigation_CU' and cu_sigma:
                sd = cu_sigma
            else:
                sd = {}
                for y, cv in cv_total.items():
                    if y in cu_preds:
                        p = cu_preds[y]
                        sd[y] = {
                            'Mean_Depth_mm': cv * abs(
                                p.get('Mean_Depth_mm', 0)),
                            'Volume_AF': cv * abs(p.get('Volume_AF', 0)),
                        }
            if sd:
                vizops.create_full_period_time_series(
                    cu_preds, cu_dir,
                    start_year=start_year, end_year=end_year,
                    title_prefix=title,
                    sigma_data=sd,
                )

    # ── 3. Basin time series (total only) ─────────────────────────────────
    basin_sigma = _read_region_sigma(
        f'{unc_dir}Sigma_Total/Basin_Sigma_Total.csv')
    if basin_sigma:
        basin_yearly, actual_basin = _read_basin_yearly(
            os.path.join(prediction_dir, 'Basin_Time_Series', 'Basin_Annual_GW.csv'),
            'Basin')
        if basin_yearly:
            vizops.create_basin_time_series(
                basin_yearly, prediction_dir,
                start_year=start_year, end_year=end_year,
                actual_basin_yearly=actual_basin,
                sigma_basin_yearly=basin_sigma,
            )

    # ── 4. Subbasin time series (total only) ──────────────────────────────
    subbasin_sigma = _read_region_sigma(
        f'{unc_dir}Sigma_Total/Subbasin_Sigma_Total.csv')
    if subbasin_sigma and subbasin_shp and os.path.exists(subbasin_shp):
        subbasin_yearly, actual_subbasin = _read_basin_yearly(
            os.path.join(prediction_dir, 'Subbasin_Time_Series', 'Subbasin_Annual_GW.csv'),
            'Subbasin')
        if subbasin_yearly:
            vizops.create_subbasin_time_series(
                subbasin_yearly, prediction_dir,
                subbasin_shp=subbasin_shp,
                ama_code_map=ama_code_map or {},
                start_year=start_year, end_year=end_year,
                actual_subbasin_yearly=actual_subbasin,
                sigma_subbasin_yearly=subbasin_sigma,
            )

    logger.info('  Uncertainty-bounded time series complete.')


def _replot_from_augmented_rasters(
        prediction_dir: str,
        basin_shp: str,
        subbasin_shp: str,
        ama_code_map: dict | None,
        start_year: int,
        end_year: int,
        mosaic_res: int | float,
) -> None:
    """Re-plot all time series with uncertainty bounds derived directly
    from the 6-band augmented rasters via zonal statistics.

    For each raster group (total pumping, categories, CU, IE) the
    augmented 6-band rasters are clipped to each basin / sub-basin
    geometry via ``rasterio.mask`` and summary statistics (mean depth,
    total volume, σ, 95 % CI) are computed from:

        Band 1 — prediction
        Band 2 — σ_total
        Band 5 — lower 95 % CI
        Band 6 — upper 95 % CI

    Volume σ is obtained from the CI bands:
    ``σ_V = (Σ upper_CI − Σ lower_CI) / (2 × CI_Z)``,
    which corresponds to a spatially-correlated (conservative) bound.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    import hydrolibs.visualops as vizops
    from hydrolibs.sysops import makedirs

    logger.info('Re-plotting time series from augmented rasters '
                'via zonal statistics...')

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = M3_TO_AF
    mm_to_ft = MM_TO_FT

    basin_gdf = gpd.read_file(basin_shp)
    subbasin_gdf = gpd.read_file(subbasin_shp)

    # ── zonal-stats helpers ───────────────────────────────────────────────

    def _az_wide_stats(raster_path):
        """AZ-wide prediction and σ from all valid pixels."""
        with rio.open(raster_path) as src:
            if src.count < 6:
                return None, None
            pred = src.read(1).astype(np.float64)
            sigma = src.read(2).astype(np.float64)
            lower_ci = src.read(5).astype(np.float64)
            upper_ci = src.read(6).astype(np.float64)

        valid = np.isfinite(pred)
        if not np.any(valid):
            return None, None

        mean_mm = float(np.nanmean(pred[valid]))
        vol_m3 = float(np.nansum(pred[valid])) * mm_to_m3
        vol_af = vol_m3 * m3_to_af
        sigma_depth = float(np.nanmean(sigma[valid]))
        lower_vol = float(np.nansum(lower_ci[valid])) * mm_to_m3 * m3_to_af
        upper_vol = float(np.nansum(upper_ci[valid])) * mm_to_m3 * m3_to_af
        sigma_vol_af = abs(upper_vol - lower_vol) / (2 * CI_Z)

        prediction = {
            'Mean_Depth_mm': round(mean_mm, 4),
            'Mean_Depth_ft': round(mean_mm * mm_to_ft, 6),
            'Volume_m3': round(vol_m3, 2),
            'Volume_AF': round(vol_af, 2),
        }
        sigma_out = {
            'Mean_Depth_mm': round(sigma_depth, 4),
            'Mean_Depth_ft': round(sigma_depth * mm_to_ft, 6),
            'Volume_m3': round(sigma_vol_af / m3_to_af, 2),
            'Volume_AF': round(sigma_vol_af, 2),
        }
        return prediction, sigma_out

    def _zone_stats(raster_path, zone_gdf, zone_col):
        """Per-zone prediction and σ from a 6-band augmented raster."""
        preds, sigmas = {}, {}
        with rio.open(raster_path) as src:
            if src.count < 6:
                return preds, sigmas
            zone_reproj = zone_gdf.to_crs(src.crs)
            for _, row in zone_reproj.iterrows():
                name = row[zone_col]
                geom = [mapping(row.geometry)]
                try:
                    out_img, _ = rio_mask(
                        src, geom, crop=True, nodata=np.nan,
                        all_touched=True,
                    )
                except Exception:
                    logger.debug('Zonal mask failed for %s=%s', zone_col, name)
                    continue
                pred = out_img[0].astype(np.float64)
                _sigma = out_img[1].astype(np.float64)
                lower = out_img[4].astype(np.float64)
                upper = out_img[5].astype(np.float64)

                valid = np.isfinite(pred)
                if not np.any(valid):
                    continue

                mean_mm = float(np.nanmean(pred[valid]))
                vol_af = float(np.nansum(pred[valid])) * mm_to_m3 * m3_to_af
                vol_m3 = float(np.nansum(pred[valid])) * mm_to_m3
                lower_vol = (
                    float(np.nansum(lower[valid])) * mm_to_m3 * m3_to_af
                )
                upper_vol = (
                    float(np.nansum(upper[valid])) * mm_to_m3 * m3_to_af
                )
                sigma_vol_af = abs(upper_vol - lower_vol) / (2 * CI_Z)
                cv = (
                    sigma_vol_af / abs(vol_af) if abs(vol_af) > 0 else 0.0
                )

                preds[name] = {
                    'Mean_Depth_mm': round(mean_mm, 4),
                    'Mean_Depth_ft': round(mean_mm * mm_to_ft, 6),
                    'Volume_m3': round(vol_m3, 2),
                    'Volume_AF': round(vol_af, 2),
                }
                sigmas[name] = {
                    'Sigma_Volume_AF': round(sigma_vol_af, 2),
                    'Sigma_Volume_m3': round(sigma_vol_af / m3_to_af, 2),
                    'CV': round(cv, 6),
                }
        return preds, sigmas

    def _read_actual_from_csv(csv_path):
        """Read actual observed data from an existing time-series CSV."""
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if 'Actual_Depth_mm' not in df.columns:
            return None
        act = df.dropna(subset=['Actual_Depth_mm'])
        if act.empty:
            return None
        return {
            int(r.Year): {
                'Mean_Depth_mm': float(r.Actual_Depth_mm),
                'Volume_AF': float(r.Actual_Volume_AF),
            }
            for _, r in act.iterrows()
        }

    def _read_actual_region(csv_path, col):
        """Read per-region actual data from an existing CSV."""
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if 'Actual_Depth_mm' not in df.columns:
            return None
        act = df.dropna(subset=['Actual_Depth_mm'])
        if act.empty:
            return None
        result = {}
        for _, r in act.iterrows():
            y = int(r.Year)
            result.setdefault(y, {})[getattr(r, col)] = {
                'Mean_Depth_mm': float(r.Actual_Depth_mm),
                'Volume_AF': float(r.Actual_Volume_AF),
            }
        return result

    # ── process one depth/volume raster group ─────────────────────────────

    def _process_group(label, raster_dir, file_pattern, out_subdir,
                       title_prefix=''):
        """Zonal stats → time-series plots (depth / volume products)."""
        yearly_preds = {}
        sigma_data = {}
        basin_yearly = {}
        sigma_basin_yearly = {}
        subbasin_yearly = {}
        sigma_subbasin_yearly = {}

        for year in range(start_year, end_year + 1):
            raster_file = os.path.join(raster_dir, f'{file_pattern.format(year=year)}')
            if not os.path.exists(raster_file):
                continue

            pred, sig = _az_wide_stats(raster_file)
            if pred:
                yearly_preds[year] = pred
                sigma_data[year] = sig

            bpred, bsig = _zone_stats(
                raster_file, basin_gdf, 'BASIN_NAME',
            )
            if bpred:
                basin_yearly[year] = bpred
                sigma_basin_yearly[year] = bsig

            sbpred, sbsig = _zone_stats(
                raster_file, subbasin_gdf, 'SUBBASIN_N',
            )
            if sbpred:
                subbasin_yearly[year] = sbpred
                sigma_subbasin_yearly[year] = sbsig

            if year % 20 == 0 or year == end_year:
                logger.info(f'    {label}: zonal stats year {year}')

        if not yearly_preds:
            return

        out_dir = os.path.join(prediction_dir, f'{out_subdir}')

        # Read actual observed data before overwriting CSVs
        actual = _read_actual_from_csv(
            os.path.join(out_dir, 'Full_Period_Time_Series.csv'))
        actual_basin = _read_actual_region(
            os.path.join(out_dir, 'Basin_Time_Series', 'Basin_Annual_GW.csv'), 'Basin')
        actual_subbasin = _read_actual_region(
            os.path.join(out_dir, 'Subbasin_Time_Series', 'Subbasin_Annual_GW.csv'),
            'Subbasin')

        vizops.create_full_period_time_series(
            yearly_preds, out_dir,
            start_year=start_year, end_year=end_year,
            actual_data=actual,
            title_prefix=title_prefix,
            sigma_data=sigma_data,
        )

        if basin_yearly:
            vizops.create_basin_time_series(
                basin_yearly, out_dir,
                start_year=start_year, end_year=end_year,
                title_prefix=title_prefix,
                actual_basin_yearly=actual_basin,
                sigma_basin_yearly=sigma_basin_yearly,
            )

        if subbasin_yearly and subbasin_shp:
            vizops.create_subbasin_time_series(
                subbasin_yearly, out_dir,
                subbasin_shp=subbasin_shp,
                ama_code_map=ama_code_map or {},
                start_year=start_year, end_year=end_year,
                title_prefix=title_prefix,
                actual_subbasin_yearly=actual_subbasin,
                sigma_subbasin_yearly=sigma_subbasin_yearly,
            )

        logger.info(f'  {label}: time series with uncertainty complete.')

    # ── process one IE (efficiency) raster group ──────────────────────────

    def _process_ie_group(label, raster_dir, file_pattern, out_subdir,
                          title_prefix=''):
        """Zonal stats → time-series plots for efficiency rasters."""
        yearly_mean, yearly_sigma = {}, {}
        basin_mean, basin_sigma = {}, {}
        subbasin_mean, subbasin_sigma = {}, {}

        for year in range(start_year, end_year + 1):
            raster_file = os.path.join(raster_dir, f'{file_pattern.format(year=year)}')
            if not os.path.exists(raster_file):
                continue
            with rio.open(raster_file) as src:
                if src.count < 6:
                    continue
                pred = src.read(1).astype(np.float64)
                sig = src.read(2).astype(np.float64)
            valid = np.isfinite(pred)
            if not np.any(valid):
                continue
            yearly_mean[year] = float(np.nanmean(pred[valid]))
            yearly_sigma[year] = float(np.nanmean(sig[valid]))

            # Basin zonal stats
            with rio.open(raster_file) as src:
                zr = basin_gdf.to_crs(src.crs)
                for _, row in zr.iterrows():
                    name = row['BASIN_NAME']
                    try:
                        out_img, _ = rio_mask(
                            src, [mapping(row.geometry)],
                            crop=True, nodata=np.nan, all_touched=True,
                        )
                    except Exception:
                        logger.debug('Basin mask failed for %s', name)
                        continue
                    p, s = out_img[0], out_img[1]
                    v = np.isfinite(p)
                    if not np.any(v):
                        continue
                    basin_mean.setdefault(year, {})[name] = float(
                        np.nanmean(p[v]))
                    basin_sigma.setdefault(year, {})[name] = float(
                        np.nanmean(s[v]))

            # Sub-basin zonal stats
            with rio.open(raster_file) as src:
                zr = subbasin_gdf.to_crs(src.crs)
                for _, row in zr.iterrows():
                    name = row['SUBBASIN_N']
                    try:
                        out_img, _ = rio_mask(
                            src, [mapping(row.geometry)],
                            crop=True, nodata=np.nan, all_touched=True,
                        )
                    except Exception:
                        logger.debug('Subbasin mask failed for %s', name)
                        continue
                    p, s = out_img[0], out_img[1]
                    v = np.isfinite(p)
                    if not np.any(v):
                        continue
                    subbasin_mean.setdefault(year, {})[name] = float(
                        np.nanmean(p[v]))
                    subbasin_sigma.setdefault(year, {})[name] = float(
                        np.nanmean(s[v]))

            if year % 20 == 0 or year == end_year:
                logger.info(f'    {label}: zonal stats year {year}')

        if not yearly_mean:
            return

        out_dir = os.path.join(prediction_dir, f'{out_subdir}')
        makedirs(out_dir)

        # ---- AZ-wide efficiency time series with 95 % CI -----------------
        vizops.apply_journal_style()
        years = sorted(yearly_mean.keys())
        means = np.array([yearly_mean[y] for y in years])
        sigs = np.array([yearly_sigma.get(y, 0) for y in years])

        fig, ax = plt.subplots(figsize=(16, 6))
        for era, (s, e) in vizops.ERA_PERIODS.items():
            ax.axvspan(s, e, color=vizops.ERA_COLORS[era], alpha=0.10)
        ax.plot(years, means, color='#2C3E50', linewidth=1.5, marker='.',
                markersize=3, label='Predicted')
        ax.fill_between(
            years, means - CI_Z * sigs, means + CI_Z * sigs,
            alpha=0.2, color='#D5DBDB', label='95 % CI', zorder=1,
        )
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Efficiency', fontweight='bold')
        ax.set_title(
            f'{title_prefix} (1896–2099) ± 95 % CI',
            fontweight='bold', fontsize=14,
        )
        era_handles = [
            mpatches.Patch(
                color=vizops.ERA_COLORS[e], alpha=0.4,
                label=(f'{e} ({vizops.ERA_PERIODS[e][0]}–'
                       f'{vizops.ERA_PERIODS[e][1]})'),
            )
            for e in vizops.ERA_PERIODS
        ]
        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + era_handles,
            loc='upper left', framealpha=0.9,
        )
        ax.set_xlim(start_year - 1, end_year + 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, 'Full_Period_Time_Series.png'), dpi=600,
            bbox_inches='tight',
        )
        plt.close()
        pd.DataFrame({
            'Year': years,
            'Mean_Efficiency': means,
            'Sigma_Efficiency': sigs,
        }).to_csv(os.path.join(out_dir, 'Full_Period_Time_Series.csv'), index=False)

        # ---- Per-basin IE plots -------------------------------------------
        _plot_ie_per_zone(
            basin_mean, basin_sigma, 'Basin',
            os.path.join(out_dir, 'Basin_Time_Series'), title_prefix,
        )

        # ---- Per-sub-basin IE plots ---------------------------------------
        _plot_ie_per_zone(
            subbasin_mean, subbasin_sigma, 'Subbasin',
            os.path.join(out_dir, 'Subbasin_Time_Series'), title_prefix,
        )

        logger.info(f'  {label}: efficiency time series with CI complete.')

    def _plot_ie_per_zone(zone_mean, zone_sigma, level, plot_dir,
                          title_prefix):
        """Plot per-zone efficiency time series with 95 % CI."""
        if not zone_mean:
            return
        makedirs(plot_dir)
        vizops.apply_journal_style()
        zones = sorted(
            set().union(*(zone_mean[y].keys() for y in zone_mean)),
        )
        for zone in zones:
            zyears = sorted(y for y in zone_mean if zone in zone_mean[y])
            if not zyears:
                continue
            zmeans = np.array([zone_mean[y][zone] for y in zyears])
            zsigs = np.array([
                zone_sigma.get(y, {}).get(zone, 0) for y in zyears
            ])

            fig, ax = plt.subplots(figsize=(14, 6))
            for era, (s, e) in vizops.ERA_PERIODS.items():
                ax.axvspan(s, e, color=vizops.ERA_COLORS[era], alpha=0.10)
            ax.plot(zyears, zmeans, color='#2C3E50', linewidth=1.5,
                    marker='.', markersize=3, label='Predicted')
            ax.fill_between(
                zyears, zmeans - CI_Z * zsigs, zmeans + CI_Z * zsigs,
                alpha=0.2, color='#D5DBDB', label='95 % CI',
            )
            ax.set_xlabel('Year', fontweight='bold')
            ax.set_ylabel('Efficiency', fontweight='bold')
            ax.set_title(
                f'{zone} — {title_prefix}',
                fontweight='bold', fontsize=13,
            )
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
            ax.set_xlim(start_year - 1, end_year + 1)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            safe = zone.replace(' ', '_').replace('.', '')
            fig.savefig(
                f'{plot_dir}{safe}_Time_Series.png', dpi=600,
                bbox_inches='tight',
            )
            plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # 1. Total pumping
    # ══════════════════════════════════════════════════════════════════════
    _process_group(
        'Total Pumping',
        os.path.join(prediction_dir, 'Predicted_Rasters/Depth_mm'),
        'Predicted_GW_{year}_mm.tif',
        '',
    )

    # ══════════════════════════════════════════════════════════════════════
    # 2. Category products
    # ══════════════════════════════════════════════════════════════════════
    cat_titles = {
        'Irrigation':         'Irrigation',
        'Non_Irrigation':     'Non-Irrigation',
        'Irrigation_GW':      'Irrigation GW',
        'Irrigation_SW':      'Irrigation SW',
        'Non_Irrigation_GW':  'Non-Irrigation GW',
        'Non_Irrigation_SW':  'Non-Irrigation SW',
        'Total_GW':           'Total GW',
        'Total_SW':           'Total SW',
    }
    for cat, title in cat_titles.items():
        _process_group(
            cat,
            os.path.join(prediction_dir, f'{cat}_Rasters/Depth_mm'),
            f'{cat}_{{year}}_mm.tif',
            f'{cat}/',
            title_prefix=title,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 3. Consumptive use
    # ══════════════════════════════════════════════════════════════════════
    cu_titles = {
        'Irrigation_CU':    'Irrigation Consumptive Use',
        'Irrigation_GW_CU': 'Irrigation GW Consumptive Use',
        'Irrigation_SW_CU': 'Irrigation SW Consumptive Use',
    }
    for cu_cat, title in cu_titles.items():
        _process_group(
            cu_cat,
            os.path.join(prediction_dir, f'{cu_cat}_Rasters/Depth_mm'),
            f'{cu_cat}_{{year}}_mm.tif',
            f'{cu_cat}/',
            title_prefix=title,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 4. Irrigation efficiency
    # ══════════════════════════════════════════════════════════════════════
    ie_titles = {
        'Irrigation_Efficiency':    'Irrigation Efficiency',
        'Irrigation_GW_Efficiency': 'Irrigation GW Efficiency',
        'Irrigation_SW_Efficiency': 'Irrigation SW Efficiency',
    }
    for ie_cat, title in ie_titles.items():
        _process_ie_group(
            ie_cat,
            os.path.join(prediction_dir, f'{ie_cat}_Rasters'),
            f'{ie_cat}_{{year}}.tif',
            f'{ie_cat}/',
            title_prefix=title,
        )

    logger.info('  All augmented-raster time series with '
                'uncertainty complete.')


def _plot_uncertainty_time_series(
        sigma_components, unc_dir, start_year, end_year,
        year_list, mosaic_res, pred_data_dir, vizops,
):
    """Generate per-component and combined time-series plots.

    For each σ component, creates:
    1. AZ-wide σ time series (Mean Depth / Volume).
    2. Per-basin and per-sub-basin σ time series from the existing
       ``{Basin|Subbasin}_Sigma_{comp}.csv`` files.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from hydrolibs.sysops import makedirs
    from hydrolibs.visualops import (
        ERA_COLORS,
        ERA_PERIODS,
        apply_journal_style,
    )

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    comp_titles = {
        'MACA': 'Inter-GCM Climate (σ_MACA)',
        'Model': 'Seed Ensemble (σ_model)',
        'Irr': 'Irrigation Fraction (σ_irr)',
        'LULC': 'LULC Projection (σ_LULC)',
        'GW': 'GW Fraction (σ_gw)',
    }

    for name, comp in sigma_components.items():
        if not comp:
            continue
        comp_dir = f'{unc_dir}Sigma_{name}/'
        makedirs(comp_dir)
        yearly = {}
        for year in sorted(comp.keys()):
            yearly[year] = _pixel_stats(comp[year], mm_to_m3, M3_TO_AF)

        title = comp_titles.get(name, name)
        comp_start = min(comp.keys())
        comp_end = max(comp.keys())
        vizops.create_full_period_time_series(
            yearly, comp_dir,
            start_year=comp_start, end_year=comp_end,
            title_prefix=f'Uncertainty {title}',
        )

        # ── Per-basin and per-sub-basin σ plots for this component ──
        _plot_component_basin_sigma(
            comp_dir, name, title,
            plt, mpatches, apply_journal_style,
            ERA_PERIODS, ERA_COLORS, makedirs,
        )


def _plot_component_basin_sigma(
        comp_dir: str,
        comp_name: str,
        comp_title: str,
        plt,
        mpatches,
        apply_journal_style,
        ERA_PERIODS,
        ERA_COLORS,
        makedirs,
) -> None:
    """Plot per-basin and per-sub-basin σ time series for one component.

    Reads ``{Basin|Subbasin}_Sigma_{comp_name}.csv`` from *comp_dir*
    and writes PNGs into ``{comp_dir}Plots/``.
    """
    apply_journal_style()
    plot_dir = f'{comp_dir}Plots/'
    makedirs(plot_dir)

    for level in ('Basin', 'Subbasin'):
        csv_path = f'{comp_dir}{level}_Sigma_{comp_name}.csv'
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        regions = sorted(df['Region'].unique())

        # ── Per-region plots ──
        for region in regions:
            rdf = df[df['Region'] == region].sort_values('Year')
            if rdf.empty:
                continue

            years = rdf['Year'].values
            sigma_m3 = rdf['Sigma_Volume_m3'].values
            mean_m3 = rdf['Mean_Volume_m3'].values

            fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

            # --- Panel 1: Mean volume with 95% CI ---
            ax1 = axes[0]
            for era, (s, e) in ERA_PERIODS.items():
                ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax1.fill_between(
                years,
                rdf['Lower_95CI_m3'].values,
                rdf['Upper_95CI_m3'].values,
                alpha=0.25, color='#2980B9', label='95 % CI',
            )
            ax1.plot(years, mean_m3, color='#2980B9', linewidth=1.5,
                     marker='.', markersize=2, label='Mean volume')
            ax1.set_ylabel('Volume (m³)', fontweight='bold')
            ax1.set_title(
                f'{level}: {region} — {comp_title}',
                fontweight='bold', fontsize=14,
            )
            ax1.grid(True, alpha=0.3, linestyle='--')

            ax1r = ax1.twinx()
            ax1r.set_ylim(
                ax1.get_ylim()[0] * M3_TO_AF,
                ax1.get_ylim()[1] * M3_TO_AF,
            )
            ax1r.set_ylabel('Volume (AF)', fontweight='bold')

            handles1 = ax1.get_legend_handles_labels()[0]
            era_handles = [
                mpatches.Patch(
                    color=ERA_COLORS[e], alpha=0.4,
                    label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})',
                )
                for e in ERA_PERIODS
            ]
            ax1.legend(
                handles=handles1 + era_handles,
                loc='upper left', framealpha=0.9, fontsize=9,
            )

            # --- Panel 2: σ component ---
            ax2 = axes[1]
            for era, (s, e) in ERA_PERIODS.items():
                ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax2.plot(years, sigma_m3, color='#E74C3C', linewidth=1.5,
                     marker='.', markersize=2, label=f'σ_{comp_name}')
            ax2.set_xlabel('Year', fontweight='bold')
            ax2.set_ylabel('σ (m³)', fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

            ax2r = ax2.twinx()
            ax2r.set_ylim(
                ax2.get_ylim()[0] * M3_TO_AF,
                ax2.get_ylim()[1] * M3_TO_AF,
            )
            ax2r.set_ylabel('σ (AF)', fontweight='bold')

            ax2.set_xlim(years.min() - 1, years.max() + 1)
            plt.tight_layout()

            safe_name = region.replace(' ', '_').replace('/', '_')
            fig.savefig(
                f'{plot_dir}{level}_{safe_name}_Sigma_{comp_name}.png',
                dpi=600, bbox_inches='tight',
            )
            plt.close()

        # ── Summary: all regions on one plot ──
        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        ax1, ax2 = axes
        for era, (s, e) in ERA_PERIODS.items():
            ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

        for region in regions:
            rdf = df[df['Region'] == region].sort_values('Year')
            years = rdf['Year'].values
            ax1.plot(years, rdf['Sigma_Volume_m3'].values, linewidth=1,
                     label=region)
            ax2.plot(years, rdf['CV'].values, linewidth=1, label=region)

        ax1.set_ylabel(f'σ_{comp_name} (m³)', fontweight='bold')
        ax1.set_title(
            f'All {level}s — {comp_title}',
            fontweight='bold', fontsize=14,
        )
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax1r = ax1.twinx()
        ax1r.set_ylim(
            ax1.get_ylim()[0] * M3_TO_AF,
            ax1.get_ylim()[1] * M3_TO_AF,
        )
        ax1r.set_ylabel(f'σ_{comp_name} (AF)', fontweight='bold')

        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('CV', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax1.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(
            f'{plot_dir}{level}_All_Sigma_{comp_name}_Summary.png',
            dpi=600, bbox_inches='tight',
        )
        plt.close()

    logger.info(
        f'  {comp_title} basin/sub-basin σ plots saved to {plot_dir}'
    )


def _plot_basin_sigma_time_series(unc_dir: str) -> None:
    """Generate per-region σ time-series with twinx (m³ / AF).

    Reads ``{Basin|Subbasin}_Sigma_Total.csv`` from ``Sigma_Total/``
    and writes one PNG per unique region into ``Plots/Basin_Sigma/``
    plus a combined all-regions summary PNG.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from hydrolibs.sysops import makedirs
    from hydrolibs.visualops import (
        ERA_COLORS,
        ERA_PERIODS,
        apply_journal_style,
    )

    apply_journal_style()
    plot_dir = f'{unc_dir}Plots/Basin_Sigma/'
    makedirs(plot_dir)

    for level in ('Basin', 'Subbasin'):
        csv_path = f'{unc_dir}Sigma_Total/{level}_Sigma_Total.csv'
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        regions = sorted(df['Region'].unique())

        # ── Per-region plots ──
        for region in regions:
            rdf = df[df['Region'] == region].sort_values('Year')
            if rdf.empty:
                continue

            years = rdf['Year'].values
            sigma_m3 = rdf['Sigma_Total_m3'].values
            _sigma_af = rdf['Sigma_Total_AF'].values
            mean_m3 = rdf['Mean_Volume_m3'].values
            _mean_af = rdf['Mean_Volume_AF'].values

            fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

            # --- Panel 1: Mean volume with 95% CI ---
            ax1 = axes[0]
            for era, (s, e) in ERA_PERIODS.items():
                ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax1.fill_between(
                years,
                rdf['Lower_95CI_m3'].values,
                rdf['Upper_95CI_m3'].values,
                alpha=0.25, color='#2980B9', label='95 % CI',
            )
            ax1.plot(years, mean_m3, color='#2980B9', linewidth=1.5,
                     marker='.', markersize=2, label='Mean volume')
            ax1.set_ylabel('Volume (m³)', fontweight='bold')
            ax1.set_title(
                f'{level}: {region} — Mean Prediction ± 95 % CI',
                fontweight='bold', fontsize=14,
            )
            ax1.grid(True, alpha=0.3, linestyle='--')

            ax1r = ax1.twinx()
            ax1r.set_ylim(
                ax1.get_ylim()[0] * M3_TO_AF,
                ax1.get_ylim()[1] * M3_TO_AF,
            )
            ax1r.set_ylabel('Volume (AF)', fontweight='bold')

            handles1 = ax1.get_legend_handles_labels()[0]
            era_handles = [
                mpatches.Patch(
                    color=ERA_COLORS[e], alpha=0.4,
                    label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})',
                )
                for e in ERA_PERIODS
            ]
            ax1.legend(
                handles=handles1 + era_handles,
                loc='upper left', framealpha=0.9, fontsize=9,
            )

            # --- Panel 2: σ_total ---
            ax2 = axes[1]
            for era, (s, e) in ERA_PERIODS.items():
                ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax2.plot(years, sigma_m3, color='#E74C3C', linewidth=1.5,
                     marker='.', markersize=2, label='σ_total')
            ax2.set_xlabel('Year', fontweight='bold')
            ax2.set_ylabel('σ (m³)', fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

            ax2r = ax2.twinx()
            ax2r.set_ylim(
                ax2.get_ylim()[0] * M3_TO_AF,
                ax2.get_ylim()[1] * M3_TO_AF,
            )
            ax2r.set_ylabel('σ (AF)', fontweight='bold')

            ax2.set_xlim(years.min() - 1, years.max() + 1)
            plt.tight_layout()

            safe_name = region.replace(' ', '_').replace('/', '_')
            fig.savefig(
                f'{plot_dir}{level}_{safe_name}_Sigma.png',
                dpi=600, bbox_inches='tight',
            )
            plt.close()

        # ── Summary: all regions on one plot ──
        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        ax1, ax2 = axes
        for era, (s, e) in ERA_PERIODS.items():
            ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

        for region in regions:
            rdf = df[df['Region'] == region].sort_values('Year')
            years = rdf['Year'].values
            ax1.plot(years, rdf['Sigma_Total_m3'].values, linewidth=1,
                     label=region)
            ax2.plot(years, rdf['CV'].values, linewidth=1, label=region)

        ax1.set_ylabel('σ_total (m³)', fontweight='bold')
        ax1.set_title(
            f'All {level}s — σ_total Time Series',
            fontweight='bold', fontsize=14,
        )
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax1r = ax1.twinx()
        ax1r.set_ylim(
            ax1.get_ylim()[0] * M3_TO_AF,
            ax1.get_ylim()[1] * M3_TO_AF,
        )
        ax1r.set_ylabel('σ_total (AF)', fontweight='bold')

        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('CV', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax1.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(
            f'{plot_dir}{level}_All_Sigma_Summary.png',
            dpi=600, bbox_inches='tight',
        )
        plt.close()

    logger.info(f'  Basin/sub-basin σ time-series plots saved to {plot_dir}')
