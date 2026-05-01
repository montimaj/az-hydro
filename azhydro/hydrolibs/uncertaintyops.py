"""
Hybrid Uncertainty Quantification for AZ-Hydro Annual Withdrawal Predictions.

Computes six independent uncertainty components and combines them via
quadrature into a total pixel-level uncertainty (σ_total):

    σ_total = √(σ_MACA² + σ_model² + σ_irr² + σ_gw² + σ_LULC² + σ_USBR²)

σ_USBR captures Upper Colorado River Basin streamflow uncertainty
(5 USBR CMIP3 ensemble members) — the climate axis σ_MACA cannot
reach, since MACA only downscales to AZ-local domain.

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
import warnings

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

# Subset of MACA_CLIMATE_COLS for which to plot inter-GCM input spread
_INPUT_SPREAD_COLS = ['annual_et_ensemble_mm', 'annual_eto_mm', 'annual_peff_mm']

# 10-seed ensemble for model uncertainty
MODEL_SEEDS = [7, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# Recent HarDWR reference years for infrastructure-density sensitivity.
# σ_gw swaps well_density with values from each of these years and takes the
# sample std of the resulting predictions. Chosen to probe recent year-over-year
# variability in HarDWR well-registry counts for the #1 SHAP feature.
INFRASTRUCTURE_SNAPSHOT_YEARS = [2020, 2021, 2022, 2023, 2024]

# Unit conversions
MM_TO_FT = 1 / 304.8
M3_TO_AF = 1 / 1233.48

# 4 USGS LULC projection scenarios (same list as gee/config.py)
USGS_LULC_SCENARIOS = ['B1', 'B2', 'A1B', 'A2']

# 1-based band indices in Predictor_{year}.tif for LULC-derived columns
LULC_BAND_INDEX = 8             # integer LULC class
CROP_FRACTION_BAND_INDEX = 13   # annual_crop_fraction
URBAN_FRACTION_BAND_INDEX = 14  # annual_urban_fraction

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
T_CRIT_GW = 2.776       # df = len(INFRASTRUCTURE_SNAPSHOT_YEARS) − 1 = 4

# Scale factors applied to sample-based σ before quadrature so that the
# final ± CI_Z × σ_total interval reflects t-corrected CIs.
T_SCALE_MODEL = T_CRIT_MODEL / CI_Z   # ≈ 1.154
T_SCALE_GW = T_CRIT_GW / CI_Z         # ≈ 1.416

COMPONENT_T_SCALE = {
    'MACA': 1.0,           # scenario-based — no correction
    'Model': T_SCALE_MODEL,  # sample-based (10 seeds)
    'Irr': 1.0,             # half-range of 2 scenarios — no correction
    'LULC': 1.0,            # scenario-based — no correction
    'GW': T_SCALE_GW,       # sample-based (5 recent HarDWR snapshots)
    'USBR': T_SCALE_GW,     # sample-based (5 USBR ensemble members; same N as GW)
}

COMPONENT_N = {
    'MACA': len(MACA_REPRESENTATIVE_GCMS),          # 5
    'Model': len(MODEL_SEEDS),                      # 10
    'Irr': 2,                                       # 2 (half-range, not std)
    'LULC': len(USGS_LULC_SCENARIOS),               # 4
    'GW': len(INFRASTRUCTURE_SNAPSHOT_YEARS),       # 5
    'USBR': 5,                                      # 5 USBR ensemble members (Rupp + mixed SRES)
}

# Category / CU raster groups
CU_CATEGORIES = ('Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU')


# ── Pre-GMA partitioning context ─────────────────────────────────────────────
# Mirrors the production pipeline's pre-GMA overrides so UQ ensemble
# members use the same partitioning logic as the central prediction.
# Set by run_uncertainty_quantification() at the start of the run.
# Contains arrays from the 2024 parquet snapshot:
#   wd_1981, irr_wd_1981, nonirr_wd_1981, irr_cap_1981
# All arrays are aligned with the same valid-pixel order used in az_df.

_PRE_GMA_CTX: dict[str, np.ndarray | None] = {
    'wd_1981': None,
    'irr_wd_1981': None,
    'nonirr_wd_1981': None,
    'irr_cap_1981': None,
}

# CAP delivery perturbation context for UQ ensemble.  When set,
# every UQ ensemble member's _partition_with_ctx call will apply the
# same CAP-pixel cw_streamflow + SW rights scaling that the central
# pipeline uses (partops.apply_cap_delivery_perturbation): observed
# 2022-2024 Tier cuts plus the 2026-2099 sustained "Basic
# Coordination" baseline.
_CAP_PIXEL_MASK_CTX: dict[str, np.ndarray | None] = {'mask': None}

# Module-level CAP delivery lookup context for UQ (mirrors the central
# pipeline's per-basin per-year CAP delivery, used by the dynamic
# NonIrr GW share cap inside ``partition_predictions``).
_CAP_DELIVERY_CTX: dict[str, dict | None] = {'lookup': None}

# Module-level pre-CAP SW baseline context for UQ (mirrors the central
# pipeline's 1984 reference cw_sf / sw_rights per pixel, used by
# ``apply_cap_delivery_perturbation`` for additive CAP-pixel SW
# scaling).
_PRE_CAP_SW_BASELINE_CTX: dict[str, dict | None] = {'baseline': None}


def _set_cap_pixel_mask_context(cap_pixel_mask: np.ndarray | None) -> None:
    """Populate the module-level CAP-pixel mask context for UQ.

    Mirrors the central pipeline's CAP-cut hindcast perturbation so
    UQ ensemble members compute σ around the same perturbed central
    value at 2022-2024.
    """
    _CAP_PIXEL_MASK_CTX['mask'] = cap_pixel_mask
    if cap_pixel_mask is not None:
        logger.info(
            'UQ CAP-cut hindcast context loaded: %d pixels',
            int(cap_pixel_mask.sum()),
        )


def _set_cap_delivery_context(cap_delivery_lookup: dict | None) -> None:
    """Populate the module-level CAP delivery lookup context for UQ.

    Used by ``partition_predictions`` to drive the per-year per-basin
    NonIrr GW share cap.  Loaded once at pipeline startup and shared
    by all UQ ensemble members so they see the same dynamic cap as
    the central pipeline.
    """
    _CAP_DELIVERY_CTX['lookup'] = cap_delivery_lookup
    if cap_delivery_lookup is not None:
        logger.info(
            'UQ CAP delivery context loaded: %d basins',
            len(cap_delivery_lookup),
        )


def _set_pre_cap_sw_baseline_context(
        pre_cap_sw_baseline: dict | None,
) -> None:
    """Populate the module-level pre-CAP SW baseline context for UQ.

    Used by ``apply_cap_delivery_perturbation`` for additive CAP-pixel
    SW scaling that preserves non-CAP infrastructure (Phoenix SRP /
    Pinal San Carlos / Tucson Avra Valley).  Loaded once at pipeline
    startup so UQ ensemble members use the same baseline as central.
    """
    _PRE_CAP_SW_BASELINE_CTX['baseline'] = pre_cap_sw_baseline
    if pre_cap_sw_baseline is not None:
        logger.info(
            'UQ pre-CAP SW baseline context loaded: %d columns',
            len(pre_cap_sw_baseline),
        )


def _build_cap_pixel_mask(
        vector_dir: str,
        pred_data_dir: str,
        ref_year: int,
) -> np.ndarray | None:
    """Rasterise the CAP service-area geojson onto the valid-pixel grid.

    Returns a 1-D boolean mask aligned with the order of valid pixels
    in ``year_df`` (i.e. matching ``year_df.index[cap_pixel_mask]``
    used downstream).  Returns None and logs a warning if the geojson
    or reference basin raster is missing or fails to load.
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from hydrolibs.rasterops import read_raster_as_arr
        candidates = [
            os.path.join(vector_dir, 'CAP_Service_Area.geojson'),
            os.path.join(vector_dir, 'CAP', 'CAP_Service_Area.geojson'),
        ]
        cap_geojson = next(
            (p for p in candidates if os.path.isfile(p)), None,
        )
        if cap_geojson is None:
            logger.warning(
                'UQ CAP-cut context: CAP_Service_Area.geojson not '
                'found in any of %s; CAP-cut hindcast perturbation '
                'skipped in UQ ensemble.', candidates,
            )
            return None
        ref_basin_file = os.path.join(
            pred_data_dir, f'GW_Basin_{ref_year}.tif',
        )
        if not os.path.isfile(ref_basin_file):
            logger.warning(
                'UQ CAP-cut context: reference basin raster %s not '
                'found; CAP-cut perturbation skipped.',
                ref_basin_file,
            )
            return None
        basin_arr, bf = read_raster_as_arr(ref_basin_file, get_file=True)
        basin_flat = basin_arr.ravel()
        valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
        raster_shape = basin_arr.shape
        bf.close()
        cap_gdf = gpd.read_file(cap_geojson)
        with rasterio.open(ref_basin_file) as ref_src:
            cap_gdf_proj = cap_gdf.to_crs(ref_src.crs)
            cap_arr = rio_rasterize(
                [(geom, 1) for geom in cap_gdf_proj.geometry],
                out_shape=raster_shape,
                transform=ref_src.transform,
                fill=0,
                dtype='uint8',
            )
        return cap_arr.ravel()[valid_mask] > 0
    except Exception as e:
        logger.warning(
            'UQ CAP-cut context could not be loaded: %s — '
            'CAP-cut hindcast perturbation skipped in UQ ensemble.', e,
        )
        return None


def _build_co_watershed_co_flow_arrays(
        sites_csv: str,
        watershed_geojson: str,
        pred_data_dir: str,
        ref_year: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-pixel Lees-Ferry-share + watershed-area arrays for σ_USBR.

    Rasterises the surface-watershed polygons onto the valid-pixel
    grid (matching the basin reference) and looks up each pixel's
    watershed-level ``lf_share`` and ``ws_area_m2`` via
    ``streamflowops.get_co_river_watershed_lf_shares``.

    Together with the per-year USBR ensemble-mean Lees Ferry flow,
    these arrays produce per-pixel ``co_flow_mm`` for the additive
    σ_USBR perturbation:
        co_flow_mm[pix] = lf_share[pix] × LF_m3s × m3s_to_mm_yr
                         / ws_area_m2[pix]

    Returns ``(lf_share_per_pixel, ws_area_m2_per_pixel)`` — both
    1-D float arrays aligned with the valid-pixel ordering of
    ``year_df.index``.  Pixels in non-LF-derived watersheds have
    ``lf_share = 0`` (no σ_USBR contribution).  Returns ``(None,
    None)`` if any input is missing.
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from hydrolibs.rasterops import read_raster_as_arr
        from hydrolibs import streamflowops as sfops

        if not os.path.isfile(sites_csv):
            logger.warning(
                'σ_USBR CO-watershed context: sites CSV %s not '
                'found; CO-mainstem σ_USBR perturbation skipped.',
                sites_csv,
            )
            return None, None
        if not os.path.isfile(watershed_geojson):
            logger.warning(
                'σ_USBR CO-watershed context: watershed geojson %s '
                'not found; CO-mainstem σ_USBR perturbation skipped.',
                watershed_geojson,
            )
            return None, None
        ref_basin_file = os.path.join(
            pred_data_dir, f'GW_Basin_{ref_year}.tif',
        )
        if not os.path.isfile(ref_basin_file):
            logger.warning(
                'σ_USBR CO-watershed context: reference basin raster '
                '%s not found; CO-mainstem σ_USBR perturbation '
                'skipped.', ref_basin_file,
            )
            return None, None

        basin_arr, bf = read_raster_as_arr(ref_basin_file, get_file=True)
        basin_flat = basin_arr.ravel()
        valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
        raster_shape = basin_arr.shape
        bf.close()

        ws_gdf = gpd.read_file(watershed_geojson)
        with rasterio.open(ref_basin_file) as ref_src:
            ws_gdf_proj = ws_gdf.to_crs(ref_src.crs)
            ws_oid_arr = rio_rasterize(
                [(geom, int(oid)) for geom, oid
                 in zip(ws_gdf_proj.geometry, ws_gdf_proj['OBJECTID'])],
                out_shape=raster_shape,
                transform=ref_src.transform,
                fill=0,
                dtype='int32',
            )
        ws_oid_per_pixel = ws_oid_arr.ravel()[valid_mask]

        shares = sfops.get_co_river_watershed_lf_shares(
            sites_csv, watershed_geojson,
        )
        # Per-watershed → per-pixel mapping via vectorised lookup.
        lf_share_per_pixel = np.zeros(
            ws_oid_per_pixel.shape, dtype=np.float32,
        )
        ws_area_per_pixel = np.ones(
            ws_oid_per_pixel.shape, dtype=np.float64,
        )  # area=1 fallback to avoid divide-by-zero (lf_share=0 there)
        for oid, info in shares.items():
            mask = ws_oid_per_pixel == oid
            lf_share_per_pixel[mask] = info['lf_share']
            if info['ws_area_m2'] > 0:
                ws_area_per_pixel[mask] = info['ws_area_m2']

        n_perturbed = int((lf_share_per_pixel > 0).sum())
        logger.info(
            'σ_USBR CO-watershed context: %d pixels in LF-derived '
            'watersheds (%.1f%% of valid).',
            n_perturbed,
            100.0 * n_perturbed / max(1, len(lf_share_per_pixel)),
        )
        return lf_share_per_pixel, ws_area_per_pixel
    except Exception as e:
        logger.warning(
            'σ_USBR CO-watershed context could not be loaded: %s — '
            'CO-mainstem σ_USBR perturbation skipped.', e,
        )
        return None, None


def _set_pre_gma_context(az_df: pd.DataFrame, ref_year: int = 2024) -> None:
    """Populate the module-level pre-GMA partitioning context from az_df.

    Extracts well density and irrigation capacity arrays from the
    *ref_year* snapshot (default 2024 — the most complete registry).
    Mirrors the pipeline's pre-GMA override behaviour so UQ ensemble
    members partition consistently with the central prediction.
    """
    sub = az_df[az_df.Year == ref_year]
    if len(sub) == 0:
        logger.warning(
            'UQ pre-GMA context: no rows for year %d in az_df; '
            'pre-1981 partitioning will use year-specific arrays only.',
            ref_year,
        )
        return
    _PRE_GMA_CTX['wd_1981'] = (
        sub['well_density'].values
        if 'well_density' in sub.columns else None
    )
    _PRE_GMA_CTX['irr_wd_1981'] = (
        sub['irr_well_density'].values
        if 'irr_well_density' in sub.columns else None
    )
    _PRE_GMA_CTX['nonirr_wd_1981'] = (
        sub['nonirr_well_density'].values
        if 'nonirr_well_density' in sub.columns else None
    )
    _PRE_GMA_CTX['irr_cap_1981'] = (
        sub['irr_capacity_fraction'].values
        if 'irr_capacity_fraction' in sub.columns else None
    )
    logger.info(
        'UQ pre-GMA context loaded from year %d (well_density=%s, '
        'irr_well_density=%s, nonirr_well_density=%s, irr_cap=%s)',
        ref_year,
        _PRE_GMA_CTX['wd_1981'] is not None,
        _PRE_GMA_CTX['irr_wd_1981'] is not None,
        _PRE_GMA_CTX['nonirr_wd_1981'] is not None,
        _PRE_GMA_CTX['irr_cap_1981'] is not None,
    )


# ── Helper ───────────────────────────────────────────────────────────────────

def _build_pred_features(
        year_df: pd.DataFrame,
        feature_cols: list[str],
        drop_attrs: tuple[str, ...],
) -> pd.DataFrame:
    """Build a prediction-ready feature matrix from *year_df*.

    XGBoost features = columns from ``create_az_data_parquet`` minus
    *drop_attrs* minus the target (``gw_pumping_mm``).

    Applies the same pre-1981 well_density override as the central
    pipeline (via ``partops.apply_ml_well_density_override``) so UQ
    ensemble members use identical ML features as the central
    prediction.
    """
    from hydrolibs import partitionops as partops

    drop_list = [a for a in drop_attrs if a in year_df.columns]
    pred = year_df.drop(
        columns=drop_list + ['gw_pumping_mm'],
        errors='ignore',
    )
    for c in feature_cols:
        if c not in pred.columns:
            pred[c] = 0
    pred = pred[feature_cols]

    # Apply the shared well_density override (single source of truth
    # with pipeline.py).  Skipped when no Year column or no wd_2024
    # context loaded.
    yr = int(year_df['Year'].iloc[0]) if 'Year' in year_df.columns and len(year_df) else None
    wd_2024 = _PRE_GMA_CTX.get('wd_1981')  # context key kept for backwards compat
    if yr is not None and wd_2024 is not None and len(pred) == len(wd_2024):
        pred = partops.apply_ml_well_density_override(
            pred, yr, year_df, wd_2024,
        )

    inf_mask = np.isinf(pred.values)
    nan_mask = pred.isna().values
    n_inf = int(inf_mask.sum())
    n_nan = int(nan_mask.sum())
    if n_inf or n_nan:
        logger.warning('UQ feature matrix: %d inf, %d NaN values (filled with 0)', n_inf, n_nan)
    return pred.replace([np.inf, -np.inf], np.nan).fillna(0)


def _safe_nanstd(stack, axis=0, ddof=1):
    """np.nanstd with the DOF RuntimeWarning suppressed.

    Pixels where all ensemble members are NaN (e.g. no-well pixels masked by
    partition_predictions) or have only one non-NaN member make numpy warn
    "Degrees of freedom <= 0 for slice". The resulting NaN is the intended
    value, so we silence only that specific warning.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Degrees of freedom <= 0 for slice',
            category=RuntimeWarning,
        )
        return np.nanstd(stack, axis=axis, ddof=ddof)


def _predict_total(model, pred_features, year_df, partops,
                   raster_shape, valid_mask,
):
    """Predict and partition, returning total pumping and category dict.

    Args:
        model: Trained ML model with a ``predict`` method.
        pred_features: Feature matrix for prediction.
        year_df: Single-year DataFrame with partitioning columns.
        partops: Module providing ``partition_predictions``.
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Boolean mask of valid pixels (ravelled).

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: (total_1d, categories) where
            total = Irrigation + Non_Irrigation and categories is the full
            dict from ``partition_predictions``.
    """
    raw = np.abs(model.predict(pred_features))
    yr = int(year_df['Year'].iloc[0]) if 'Year' in year_df.columns else 0
    cat = _partition_with_ctx(
        partops, raw, year_df, raster_shape, valid_mask, year=yr,
    )
    return cat['Irrigation'] + cat['Non_Irrigation'], cat


def _partition_with_ctx(partops, predictions, year_df, raster_shape,
                         valid_mask, year, sw_smooth_sigma=None,
                         skip_cap_perturbation=False):
    """Call partops.partition_predictions with the pre-GMA context kwargs.

    Centralises the wd_1981/irr_wd_1981/nonirr_wd_1981/irr_cap_1981
    plumbing so all UQ ensemble members partition identically to the
    central pipeline run.

    Also applies the CAP delivery perturbation when the CAP pixel-
    mask context has been set (via _set_cap_pixel_mask_context):
    observed 2022-2026 Tier cuts plus the 2027-2099 sustained "Basic
    Coordination" baseline.  No-op for years not in
    CAP_DELIVERY_FACTORS.

    sw_smooth_sigma=None (default) → use the calibrated era-based
    schedule inside partition_predictions.  Explicit value → override
    the era schedule (used by the σ-sensitivity diagnostic only).

    skip_cap_perturbation=False (default) → apply the central CAP
    delivery perturbation.  True → bypass it (used by
    ``run_cap_scenario_analysis`` so scenario baselines represent
    true "no-cut" counterfactuals rather than central-perturbed
    projections).
    """
    if not skip_cap_perturbation:
        year_df = partops.apply_cap_delivery_perturbation(
            year_df, year, _CAP_PIXEL_MASK_CTX.get('mask'),
        )
    return partops.partition_predictions(
        predictions, year_df, raster_shape, valid_mask,
        sw_smooth_sigma=sw_smooth_sigma, year=year,
        wd_1981=_PRE_GMA_CTX.get('wd_1981'),
        irr_wd_1981=_PRE_GMA_CTX.get('irr_wd_1981'),
        nonirr_wd_1981=_PRE_GMA_CTX.get('nonirr_wd_1981'),
        irr_cap_1981=_PRE_GMA_CTX.get('irr_cap_1981'),
        cap_delivery_lookup=_CAP_DELIVERY_CTX.get('lookup'),
        cap_pixel_mask=_CAP_PIXEL_MASK_CTX.get('mask'),
    )


def _compute_category_sigmas(
        member_cats: list[dict[str, np.ndarray]],
        mode: str = 'std',
) -> dict[str, np.ndarray]:
    """Compute per-category σ from ensemble member category dicts.

    Args:
        member_cats (list[dict[str, np.ndarray]]): Each dict maps category name to
            1-D prediction array.
        mode (str): 'std' — ``np.nanstd`` with ``ddof=1`` (3+ members).
            'half_range' — ``|a - b| / 2`` (2 counterfactual members).

    Returns:
        dict[str, np.ndarray]: Per-category sigma arrays.
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
            cat_sigmas[cat_name] = _safe_nanstd(
                stack, axis=0, ddof=1,
            ).astype(np.float32)
    return cat_sigmas


def _pixel_stats(pred_vals, mm_to_m3, min_depth_threshold=0.0):
    """Compute summary statistics in multiple units.

    When ``min_depth_threshold`` > 0, ``Mean_Depth_mm`` / ``_ft`` are
    averaged only over pixels where ``pred_vals >= min_depth_threshold``
    (the "active pumping pixel" convention — see pipeline.py's
    ``_pixel_stats`` docstring).  Default 0.0 preserves the legacy
    nanmean-over-all-pixels behaviour, which is the right choice for
    sigma statistics where every pixel's uncertainty is meaningful.
    """
    n = len(pred_vals)
    if n == 0:
        mean_mm = 0.0
    elif min_depth_threshold > 0:
        finite = np.isfinite(pred_vals)
        active = finite & (pred_vals >= min_depth_threshold)
        mean_mm = (
            float(pred_vals[active].mean()) if np.any(active)
            else float(np.nanmean(pred_vals))
        )
    else:
        mean_mm = float(np.nanmean(pred_vals))
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
    write_raster(grid, rfile, rfile.transform, out_path,
                 no_data_value=np.nan, num_bands=1)
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
    lower_ci = np.maximum(pred_arr - CI_Z * sigma_arr, 0).astype(np.float32)
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

    Args:
        member_preds (list[np.ndarray]): Each array has length = number of valid pixels
            (same as year_df rows).
        year_df (pd.DataFrame): Must contain ``GW_Basin`` and ``GW_Subbasin`` string columns.
        mm_to_m3 (float): Conversion factor: pixel depth (mm) to volume (m³).

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: (basin_vols, subbasin_vols) —
            each is ``{region_name: np.ndarray of shape (n_members,)}``
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


def _aggregate_category_member_volumes(
        member_cats: list[dict[str, np.ndarray]],
        year_df: pd.DataFrame,
        mm_to_m3: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-category variant of ``_aggregate_member_volumes`` (basin-only).

    For each partition category in the first member's dict, aggregate the
    per-pixel predictions from every member to per-basin volumes in
    acre-feet.

    Sub-basin aggregation is intentionally omitted — ADWR stewardship
    decisions are made at the basin level, and the sub-basin variant is
    deferred (see σ attribution plan).

    Args:
        member_cats (list[dict[str, np.ndarray]]): Ensemble members; each
            dict maps a category name to a 1-D array of length = number
            of valid pixels.
        year_df (pd.DataFrame): Must contain ``GW_Basin``.
        mm_to_m3 (float): Pixel depth (mm) → volume (m³) conversion.

    Returns:
        dict[str, dict[str, np.ndarray]]: ``{category: {basin: np.ndarray(n_members)}}``
            where each inner array holds per-member volume in acre-feet.
    """
    basins = year_df['GW_Basin'].values
    n_members = len(member_cats)
    m3_to_af = M3_TO_AF
    unique_basins = np.unique(basins)

    result: dict[str, dict[str, np.ndarray]] = {}
    for cat_name in member_cats[0]:
        cat_basin_vols: dict[str, np.ndarray] = {}
        for b in unique_basins:
            bmask = basins == b
            vols = np.empty(n_members, dtype=np.float64)
            for i, mc in enumerate(member_cats):
                vols[i] = (
                    float(np.nansum(mc[cat_name][bmask]))
                    * mm_to_m3 * m3_to_af
                )
            cat_basin_vols[b] = vols
        result[cat_name] = cat_basin_vols
    return result


def _accumulate_basin_sigma(
        accum: dict,
        year: int,
        basin_vols: dict[str, np.ndarray],
        subbasin_vols: dict[str, np.ndarray] | None,
) -> None:
    """Store per-year basin/sub-basin member volumes into *accum*.

    ``accum`` has structure::

        {
            'basin': {year: {name: np.ndarray(n_members)}},
            'subbasin': {year: {name: np.ndarray(n_members)}},
        }

    If ``subbasin_vols`` is ``None``, the sub-basin entry for *year* is
    set to an empty dict (used by the per-category basin-only path).
    """
    accum['basin'][year] = basin_vols
    accum.setdefault('subbasin', {})[year] = (
        subbasin_vols if subbasin_vols is not None else {}
    )


def _accumulate_category_basin_sigma(
        cat_accum: dict[str, dict],
        year: int,
        cat_basin_vols: dict[str, dict[str, np.ndarray]],
) -> None:
    """Store per-year per-category basin volumes into *cat_accum*.

    Each key in ``cat_accum`` is a category name mapping to
    ``{'basin': {year: {basin: np.ndarray(n_members)}}}``.
    """
    for cat_name, basin_vols in cat_basin_vols.items():
        cat_accum[cat_name]['basin'][year] = basin_vols


def _write_basin_sigma_csv(
        accum: dict,
        output_dir: str,
        label: str,
        basin_only: bool = False,
) -> None:
    """Write basin-scale and sub-basin-scale σ CSVs from accumulated data.

    For each region, the CSV has columns:
    ``Year, Region, Mean_Volume_m3, Sigma_Volume_m3, Mean_Volume_AF,
    Sigma_Volume_AF, CV, Lower_95CI_m3, Upper_95CI_m3, Lower_95CI_AF,
    Upper_95CI_AF, N_Members``.

    When ``basin_only=True`` only the basin-level CSV is written (used by
    the per-category CSV extension for σ attribution).
    """
    af_to_m3 = 1.0 / M3_TO_AF  # 1233.48
    levels = ('basin',) if basin_only else ('basin', 'subbasin')
    for level in levels:
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
                # Withdrawal volumes are physically non-negative, so the
                # 95% CI lower bound is clipped at 0.
                rows.append({
                    'Year': year,
                    'Region': region,
                    'Mean_Volume_m3': round(mean_m3, 2),
                    'Sigma_Volume_m3': round(std_m3, 2),
                    'Mean_Volume_AF': round(mean_af, 2),
                    'Sigma_Volume_AF': round(std_af, 2),
                    'CV': round(cv, 6),
                    'Lower_95CI_m3': round(max(mean_m3 - CI_Z * std_m3, 0), 2),
                    'Upper_95CI_m3': round(mean_m3 + CI_Z * std_m3, 2),
                    'Lower_95CI_AF': round(max(mean_af - CI_Z * std_af, 0), 2),
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

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        output_dir (str): Base output directory for uncertainty products.
        input_dir (str): Base input directory for GEE downloads.
        vector_dir (str): Directory containing vector shapefiles.
        mosaic_res (int): Raster resolution in meters.
        gcloud_project (str): Google Cloud project ID.
        gcloud_bucket (str): Google Cloud Storage bucket name.
        tile_size (int): GEE export tile size.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        skip_download (bool): If True, skip GEE download step.

    Returns:
        tuple: (sigma_maca, cat_sigma_maca, gcm_mosaic_dirs) — per-year
            total σ arrays, per-category per-year σ arrays, and per-GCM
            mosaic directory paths (reused by σ_CU).
    """
    import hydrolibs.dataops as dataops
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_MACA (inter-GCM climate uncertainty)...')
    raster_dir = os.path.join(output_dir, 'Sigma_MACA/Rasters')
    makedirs(raster_dir)

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

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
        gcm_mosaic_dir = os.path.join(
            os.path.dirname(pred_data_dir.rstrip(os.sep)),
            f'GEE_Mosaics_{int(mosaic_res)}m_{gcm}'
        )
        gcm_mosaic_reproj_dir = gcm_mosaic_dir + '_Reproj'
        # Skip mosaic/reproject only if reproj rasters already exist
        reproj_exists = os.path.exists(os.path.join(
            gcm_mosaic_reproj_dir, f'Predictor_{MACA_FUTURE_START}.tif'))
        dataops.mosaic_tiles(
            gcm_tile_dir, gcm_mosaic_dir,
            MACA_FUTURE_START, end_year,
            already_mosaicked=skip_download or reproj_exists,
        )
        dataops.reproject_gee_mosaics(
            gcm_mosaic_dir, gcm_mosaic_reproj_dir, pred_data_dir,
            already_reprojected=reproj_exists,
        )
        gcm_mosaic_dirs[gcm] = gcm_mosaic_reproj_dir

    sigma_maca = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
    cat_sigma_maca: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }
    # Per-GCM AZ-mean climate values for input-spread plots
    climate_spread: dict[str, dict[int, list[float]]] = {
        col: {} for col in _INPUT_SPREAD_COLS
    }

    for year in range(MACA_FUTURE_START, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        gcm_preds = []
        gcm_cats = []
        for gcm in MACA_REPRESENTATIVE_GCMS:
            gcm_raster = os.path.join(gcm_mosaic_dirs[gcm], f'Predictor_{year}.tif')
            gcm_year_df = year_df.copy()
            for col, bidx in zip(MACA_CLIMATE_COLS, MACA_CLIMATE_BAND_INDICES):
                band_arr = read_raster_as_arr(gcm_raster, band=bidx, get_file=False)
                vals = band_arr.ravel()[valid_mask]
                gcm_year_df[col] = vals
                if col in climate_spread:
                    climate_spread[col].setdefault(year, []).append(
                        float(np.nanmean(vals))
                    )

            pf = _build_pred_features(gcm_year_df, feature_cols, drop_attrs)
            pred, cat = _predict_total(model, pf, gcm_year_df, partops,
                                       raster_shape, valid_mask)
            gcm_preds.append(pred)
            gcm_cats.append(cat)

        gcm_stack = np.stack(gcm_preds, axis=0)
        std = _safe_nanstd(gcm_stack, axis=0, ddof=1)
        sigma_maca[year] = std

        cat_std = _compute_category_sigmas(gcm_cats)
        for c in partops.CATEGORIES:
            cat_sigma_maca[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(gcm_preds, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(gcm_cats, year_df, mm_to_m3)
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_MACA_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 10 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_MACA = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, os.path.join(output_dir, 'Sigma_MACA'), 'MACA')
    _write_basin_sigma_csv(basin_accum, os.path.join(output_dir, 'Sigma_MACA'), 'MACA')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat],
            os.path.join(output_dir, 'Sigma_MACA'),
            f'MACA_{cat}',
            basin_only=True,
        )
    _save_climate_input_spread(climate_spread, os.path.join(output_dir, 'Sigma_MACA'))
    _plot_climate_input_spread(climate_spread, os.path.join(output_dir, 'Sigma_MACA'))
    logger.info('  σ_MACA complete.')
    return sigma_maca, cat_sigma_maca, gcm_mosaic_dirs


def _save_climate_input_spread(
        climate_spread: dict[str, dict[int, list[float]]],
        sigma_maca_dir: str,
) -> None:
    """Save per-GCM AZ-mean climate values to CSV (one file per variable)."""
    from hydrolibs.sysops import makedirs

    out_dir = os.path.join(sigma_maca_dir, 'Climate_Input_Spread')
    makedirs(out_dir)
    for col, year_vals in climate_spread.items():
        rows = []
        for year in sorted(year_vals):
            for i, val in enumerate(year_vals[year]):
                rows.append({
                    'Year': year,
                    'GCM': MACA_REPRESENTATIVE_GCMS[i],
                    'AZ_Mean_mm': round(val, 4),
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, f'{col}.csv'), index=False)


def _plot_climate_input_spread(
        climate_spread: dict[str, dict[int, list[float]]],
        sigma_maca_dir: str,
) -> None:
    """Plot per-GCM input climate spread (ET, ETo, Peff) as ribbon + lines."""
    import matplotlib.pyplot as plt

    from hydrolibs.sysops import makedirs
    from hydrolibs.visualops import apply_journal_style

    apply_journal_style()
    plot_dir = os.path.join(sigma_maca_dir, 'Climate_Input_Spread')
    makedirs(plot_dir)

    labels = {
        'annual_et_ensemble_mm': 'ET (mm)',
        'annual_eto_mm': 'ETo (mm)',
        'annual_peff_mm': 'Peff (mm)',
    }
    gcm_colors = ['#2980B9', '#27AE60', '#E74C3C', '#8E44AD', '#E67E22']

    fig, axes = plt.subplots(len(climate_spread), 1, figsize=(14, 4 * len(climate_spread)),
                             sharex=True)
    if len(climate_spread) == 1:
        axes = [axes]

    for ax, (col, year_vals) in zip(axes, climate_spread.items()):
        years = np.array(sorted(year_vals))
        matrix = np.array([year_vals[y] for y in years])  # (n_years, n_gcms)
        ens_mean = matrix.mean(axis=1)
        ens_min = matrix.min(axis=1)
        ens_max = matrix.max(axis=1)

        ax.fill_between(years, ens_min, ens_max, alpha=0.20, color='#2980B9',
                         label='GCM range')
        for gi, gcm in enumerate(MACA_REPRESENTATIVE_GCMS):
            ax.plot(years, matrix[:, gi], lw=0.7, alpha=0.5,
                    color=gcm_colors[gi], label=gcm)
        ax.plot(years, ens_mean, lw=2, color='k', label='Ensemble mean')

        ax.set_ylabel(labels.get(col, col), fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=7, ncol=3, loc='upper left')

    axes[-1].set_xlabel('Year', fontweight='bold')
    fig.suptitle('Inter-GCM Climate Input Spread (AZ Mean)',
                 fontweight='bold', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'Climate_Input_Spread.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved climate input spread plot and CSVs.')


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
        prediction_model: str = 'XGBRF',
        model_dir: str | None = None,
        fold_count: int = 5,
        repeats: int = 3,
        n_trials: int = 100,
        n_dask_workers: int = 10,
        use_dask: bool = True,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """
    Compute σ_model: per-pixel std of predictions across a 10-seed
    ensemble of the prediction model.

    Each seed model reuses the Optuna-tuned hyperparameters from the
    full-prediction step (no re-tuning), varying only the random seed
    to isolate stochastic model uncertainty.

    Args:
        x_train (pd.DataFrame): Training feature matrix.
        y_train (np.ndarray): Training target array.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        output_dir (str): Base output directory for uncertainty products.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        mosaic_res (int): Raster resolution in meters.
        prediction_model (str): Model name (e.g. 'XGBRF'). Default 'XGBRF'.
        model_dir (str or None): Base model directory containing
            ``Full_Prediction_{model}/`` with the Optuna study DB.
            When provided, tuned hyperparameters are reused for each seed.
        fold_count (int): Number of folds for KFold. Default is 5.
        repeats (int): Number of repeats for RepeatedKFold. Default is 3.
        n_trials (int): Number of Optuna trials (used only if tuning_dir
            is unavailable). Default is 100.
        n_dask_workers (int): Number of Dask workers.
        use_dask (bool): If True, use Dask for parallel tuning.

    Returns:
        tuple: (sigma_model, cat_sigma_model) — per-year total σ and
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

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Train (or load) seed-ensemble models using the same model type
    # and tuned hyperparameters as the full prediction
    models = []
    model_name = prediction_model
    tuning_dir = None
    if model_dir is not None:
        # Search for the Optuna study DB in the full prediction directory
        for subdir in ['Model', '']:
            td = os.path.join(model_dir, f'Full_Prediction_{model_name}', subdir)
            if os.path.exists(os.path.join(td, f'optuna_study_{model_name}.db')):
                tuning_dir = td
                logger.info(f'  Reusing tuned {model_name} hyperparameters '
                            f'from {tuning_dir}')
                break
        if tuning_dir is None:
            logger.warning(f'  No Optuna study found for {model_name} — '
                           f'will run full tuning per seed')

    for seed in MODEL_SEEDS:
        seed_dir = os.path.join(base_dir, f'Model_seed{seed}')
        makedirs(seed_dir)
        model_file = os.path.join(seed_dir, f'{model_name}')
        if os.path.exists(model_file):
            logger.info(f'  Loading seed={seed} {model_name} model...')
            with open(model_file, 'rb') as f:
                m = pickle.load(f)
        else:
            logger.info(f'  Training seed={seed} {model_name} model...')
            m, _ = mlops.build_ml_model_optuna(
                x_train, y_train, seed_dir,
                model_name, seed,
                fold_count=fold_count,
                repeats=repeats,
                n_trials=n_trials,
                n_dask_workers=n_dask_workers,
                use_dask=use_dask,
                tuning_dir=tuning_dir,
            )
        models.append(m)

    # Predict for all years with each model
    sigma_model = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
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
        std = _safe_nanstd(seed_stack, axis=0, ddof=1)
        sigma_model[year] = std

        cat_std = _compute_category_sigmas(seed_cats)
        for c in partops.CATEGORIES:
            cat_sigma_model[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(seed_preds, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(seed_cats, year_df, mm_to_m3)
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_Model_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_model = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'Model')
    _write_basin_sigma_csv(basin_accum, base_dir, 'Model')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat], base_dir, f'Model_{cat}', basin_only=True,
        )
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

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        output_dir (str): Base output directory for uncertainty products.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        mosaic_res (int): Raster resolution in meters.

    Returns:
        tuple: (sigma_irr, cat_sigma_irr) — per-year total σ and
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

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

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
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
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
            irr_cat_members = [cat_orig, cat_alt]
            cat_std = _compute_category_sigmas(
                irr_cat_members, mode='half_range',
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
            irr_cat_members = [cat_plus, cat_minus]
            cat_std = _compute_category_sigmas(
                irr_cat_members, mode='half_range',
            )

        sigma_irr[year] = std

        for c in partops.CATEGORIES:
            cat_sigma_irr[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(irr_members, year_df, mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(
            irr_cat_members, year_df, mm_to_m3,
        )
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_Irr_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_irr = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'Irr')
    _write_basin_sigma_csv(basin_accum, base_dir, 'Irr')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat], base_dir, f'Irr_{cat}', basin_only=True,
        )
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

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        output_dir (str): Base output directory for uncertainty products.
        input_dir (str): Base input directory for GEE downloads.
        vector_dir (str): Directory containing vector shapefiles.
        mosaic_res (int): Raster resolution in meters.
        gcloud_project (str): Google Cloud project ID.
        gcloud_bucket (str): Google Cloud Storage bucket name.
        tile_size (int): GEE export tile size.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        skip_download (bool): If True, skip GEE download step.

    Returns:
        tuple: (sigma_lulc, cat_sigma_lulc) — per-year total σ and
            per-category per-year σ arrays.
    """
    from sklearn.linear_model import LinearRegression

    import hydrolibs.dataops as dataops
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_LULC (inter-scenario LULC uncertainty)...')
    base_dir = os.path.join(output_dir, 'Sigma_LULC')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

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
        sc_mosaic_dir = os.path.join(
            os.path.dirname(pred_data_dir.rstrip(os.sep)),
            f'GEE_Mosaics_{int(mosaic_res)}m_LULC_{scenario}'
        )
        sc_mosaic_reproj_dir = sc_mosaic_dir + '_Reproj'
        # Skip mosaic/reproject only if reproj rasters already exist
        reproj_exists = os.path.exists(os.path.join(
            sc_mosaic_reproj_dir, f'Predictor_{MACA_FUTURE_START}.tif'))
        dataops.mosaic_tiles(
            sc_tile_dir, sc_mosaic_dir,
            MACA_FUTURE_START, end_year,
            already_mosaicked=skip_download or reproj_exists,
        )
        dataops.reproject_gee_mosaics(
            sc_mosaic_dir, sc_mosaic_reproj_dir, pred_data_dir,
            already_reprojected=reproj_exists,
        )
        scenario_mosaic_dirs[scenario] = sc_mosaic_reproj_dir

    sigma_lulc = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
    cat_sigma_lulc: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    # Per-scenario volume tracking
    scenario_volumes: dict[str, list[dict]] = {sc: [] for sc in USGS_LULC_SCENARIOS}
    scenario_cat_volumes: dict[str, dict[str, list[dict]]] = {
        sc: {c: [] for c in partops.CATEGORIES} for sc in USGS_LULC_SCENARIOS
    }

    # --- Per-scenario basin-delta baselines (FORE-SCE 2026 per scenario) ---
    # Mirrors the main pipeline's basin-scale delta correction so that each
    # scenario's URBAN/AGRI/crop_fraction/urban_fraction columns are anchored
    # to NLCD 2025's pixel-level spatial pattern, scaled by the scenario's
    # own basin-level relative change. Baseline is the single FORE-SCE year
    # paired with NLCD 2025 (i.e. 2026) so delta(2026, B) = 1 exactly and
    # later years grow from there.
    foresce_baseline_year = MACA_FUTURE_START  # 2026
    valid_basins = az_df.loc[az_df.Year == 2025, 'GW_Basin'].reset_index(drop=True).values
    unique_basins = np.unique(valid_basins)
    nlcd_anchor_2025 = {}
    for col in ('URBAN', 'AGRI', 'annual_crop_fraction', 'annual_urban_fraction'):
        if col in az_df.columns:
            nlcd_anchor_2025[col] = az_df.loc[
                az_df.Year == 2025, col
            ].reset_index(drop=True).values
    scenario_baselines: dict[str, dict[int, dict]] = {}
    for scenario in USGS_LULC_SCENARIOS:
        sc_raster = os.path.join(
            scenario_mosaic_dirs[scenario],
            f'Predictor_{foresce_baseline_year}.tif',
        )
        lulc_b = read_raster_as_arr(
            sc_raster, band=LULC_BAND_INDEX, get_file=False
        )
        lulc_valid = lulc_b.ravel()[valid_mask]
        scenario_baselines[scenario] = {cls: {} for cls in (1, 2)}
        for b in unique_basins:
            m = valid_basins == b
            if m.sum() == 0:
                scenario_baselines[scenario][1][b] = 0.0
                scenario_baselines[scenario][2][b] = 0.0
                continue
            scenario_baselines[scenario][1][b] = float((lulc_valid[m] == 1).mean())
            scenario_baselines[scenario][2][b] = float((lulc_valid[m] == 2).mean())

    def _scenario_pixel_deltas(sc_lulc_valid, scenario):
        """Return (pixel_delta_ag, pixel_delta_urban) arrays."""
        y_frac_1 = {}
        y_frac_2 = {}
        for b in unique_basins:
            m = valid_basins == b
            if m.sum() == 0:
                y_frac_1[b] = 0.0
                y_frac_2[b] = 0.0
                continue
            y_frac_1[b] = float((sc_lulc_valid[m] == 1).mean())
            y_frac_2[b] = float((sc_lulc_valid[m] == 2).mean())
        base = scenario_baselines[scenario]
        basin_delta_1 = {
            b: (y_frac_1[b] / base[1][b]) if base[1][b] > 0 else 1.0
            for b in unique_basins
        }
        basin_delta_2 = {
            b: (y_frac_2[b] / base[2][b]) if base[2][b] > 0 else 1.0
            for b in unique_basins
        }
        pixel_d_ag = np.array([basin_delta_1[b] for b in valid_basins])
        pixel_d_urban = np.array([basin_delta_2[b] for b in valid_basins])
        return pixel_d_ag, pixel_d_urban

    for year in range(MACA_FUTURE_START, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        scenario_preds = []
        scenario_cats = []
        for scenario in USGS_LULC_SCENARIOS:
            sc_raster = os.path.join(scenario_mosaic_dirs[scenario], f'Predictor_{year}.tif')
            sc_year_df = year_df.copy()

            # Read per-scenario LULC class band for basin-delta computation
            lulc_arr = read_raster_as_arr(
                sc_raster, band=LULC_BAND_INDEX, get_file=False
            )
            lulc_valid = lulc_arr.ravel()[valid_mask]

            # Compute per-scenario pixel-level deltas for ag and urban
            pixel_d_ag, pixel_d_urban = _scenario_pixel_deltas(
                lulc_valid, scenario,
            )

            # Apply per-scenario basin-delta correction to NLCD 2025 anchors.
            # All four columns are bounded in [0, 1] so clip after scaling.
            if 'URBAN' in nlcd_anchor_2025:
                sc_year_df['URBAN'] = np.clip(
                    nlcd_anchor_2025['URBAN'] * pixel_d_urban, 0.0, 1.0,
                )
            if 'AGRI' in nlcd_anchor_2025:
                sc_year_df['AGRI'] = np.clip(
                    nlcd_anchor_2025['AGRI'] * pixel_d_ag, 0.0, 1.0,
                )
            if 'annual_urban_fraction' in nlcd_anchor_2025:
                sc_year_df['annual_urban_fraction'] = np.clip(
                    nlcd_anchor_2025['annual_urban_fraction'] * pixel_d_urban,
                    0.0, 1.0,
                )
            if 'annual_crop_fraction' in nlcd_anchor_2025:
                sc_year_df['annual_crop_fraction'] = np.clip(
                    nlcd_anchor_2025['annual_crop_fraction'] * pixel_d_ag,
                    0.0, 1.0,
                )
            # Expose raw scenario lulc class for partitioning fallbacks
            sc_year_df['lulc'] = lulc_valid

            # Re-derive irr fraction from CORRECTED crop fraction via regression
            crop_frac_valid = sc_year_df['annual_crop_fraction'].values
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

        # Collect per-scenario volumes.  Use 5 mm threshold for mean
        # depth (matches pipeline.py _pixel_stats default) so that the
        # Mean_Depth reflects intensity at active pumping pixels
        # rather than being diluted by LU-only basin-median fill.
        for si, scenario in enumerate(USGS_LULC_SCENARIOS):
            sc_stats = _pixel_stats(
                scenario_preds[si], mm_to_m3, min_depth_threshold=5.0,
            )
            sc_stats['Year'] = year
            scenario_volumes[scenario].append(sc_stats)
            for c in partops.CATEGORIES:
                cat_stats = _pixel_stats(
                    scenario_cats[si][c], mm_to_m3, min_depth_threshold=5.0,
                )
                cat_stats['Year'] = year
                scenario_cat_volumes[scenario][c].append(cat_stats)

        sc_stack = np.stack(scenario_preds, axis=0)
        std = _safe_nanstd(sc_stack, axis=0, ddof=1)
        sigma_lulc[year] = std

        cat_std = _compute_category_sigmas(scenario_cats)
        for c in partops.CATEGORIES:
            cat_sigma_lulc[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(
            scenario_preds, year_df, mm_to_m3,
        )
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(
            scenario_cats, year_df, mm_to_m3,
        )
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_LULC_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 10 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_LULC = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'LULC')
    _write_basin_sigma_csv(basin_accum, base_dir, 'LULC')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat], base_dir, f'LULC_{cat}', basin_only=True,
        )

    # Save per-scenario volume projections
    sc_vol_dir = os.path.join(base_dir, 'Scenario_Volumes')
    makedirs(sc_vol_dir)
    for scenario in USGS_LULC_SCENARIOS:
        if scenario_volumes[scenario]:
            sc_df = pd.DataFrame(scenario_volumes[scenario])
            sc_df.to_csv(os.path.join(sc_vol_dir, f'Total_{scenario}.csv'), index=False)
            for c in partops.CATEGORIES:
                if scenario_cat_volumes[scenario][c]:
                    cat_df = pd.DataFrame(scenario_cat_volumes[scenario][c])
                    cat_df.to_csv(os.path.join(sc_vol_dir, f'{c}_{scenario}.csv'), index=False)

    # Combined scenario comparison CSV (all scenarios side by side)
    combined_rows = []
    for scenario in USGS_LULC_SCENARIOS:
        for row in scenario_volumes[scenario]:
            combined_rows.append({'Scenario': scenario, **row})
    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_df.to_csv(os.path.join(sc_vol_dir, 'Scenario_Comparison.csv'), index=False)
        logger.info(f'  Per-scenario volume projections saved to {sc_vol_dir}')

    logger.info('  σ_LULC complete.')
    return sigma_lulc, cat_sigma_lulc


# ═════════════════════════════════════════════════════════════════════════════
# σ_USBR — Upper Colorado River Basin streamflow ensemble uncertainty
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_usbr(
        model,
        feature_cols: list[str],
        az_df: pd.DataFrame,
        drop_attrs: tuple[str, ...],
        pred_data_dir: str,
        output_dir: str,
        usbr_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int,
        vector_dir: str | None = None,
        sites_csv: str | None = None,
        watershed_geojson: str | None = None,
        members: list[str] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Compute σ_USBR — inter-USBR-member spread of CAP delivery driven
    by Upper Colorado River Basin streamflow uncertainty.

    For each year × USBR ensemble member, perturbs
    ``canal_weighted_streamflow_mm`` AND the SW rights density
    columns at CAP service-area pixels by the member's annual
    Lees-Ferry-flow ratio (member_annual_mean / ensemble_annual_mean).
    Re-runs the partition for each member and computes per-pixel std
    across members → σ_USBR.

    σ_USBR captures the **Upper-Basin-headwater hydrologic uncertainty
    that σ_MACA does not** — MACA's 5-GCM ensemble downscales to AZ-
    local domain, which captures Salt/Verde/Gila watershed climate
    but not the Wyoming/Colorado/Utah snowpack that drives Lees Ferry
    inflow → CAP imports.  σ_USBR uses 5 CMIP3 USBR members chosen as
    CMIP3-equivalents of the σ_MACA Rupp 2013 GCMs, with mixed SRES
    coverage to span both GCM-corner and emission-scenario axes.

    Members ranked by their Lees Ferry mean flow:
        a1b.ncar_ccsm3_0.1     (≈ MACA CCSM4, center)
        b1.cnrm_cm3.1          (≈ CNRM-CM5, cool-wet, low emissions)
        a2.ukmo_hadcm3.1       (≈ HadGEM2-ES, hot-dry, high emissions)
        a2.miroc3_2_medres.1   (≈ MIROC-ESM, hot-wet, high emissions)
        b1.inmcm3_0.1          (≈ inmcm4, cool-dry, low emissions)

    For USGS-observed years where streamflow is bias-corrected back
    to USGS observations, σ_USBR is naturally smaller because the
    raw USBR ensemble spread there reflects model variability that is
    constrained out in the central-pipeline streamflow assembly.

    Args:
        model: Trained ML model.
        feature_cols: Feature column names.
        az_df: Arizona training DataFrame.
        drop_attrs: Columns to drop before prediction.
        pred_data_dir: Directory containing predictor rasters.
        output_dir: Base output directory for uncertainty products.
        usbr_dir: Directory containing USBR ensemble CSVs.
        start_year, end_year, year_list: Prediction year range.
        mosaic_res: Raster resolution in meters.
        members: USBR member names; defaults to
            ``streamflowops.USBR_REPRESENTATIVE_MEMBERS``.

    Returns:
        (sigma_usbr, cat_sigma_usbr) — per-year total σ + per-category
        per-year σ arrays (same shape convention as the other σ
        components for direct quadrature into σ_total).
    """
    import hydrolibs.partitionops as partops
    import hydrolibs.streamflowops as sfops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    if members is None:
        members = sfops.USBR_REPRESENTATIVE_MEMBERS

    logger.info(
        'Computing σ_USBR (inter-USBR-member streamflow uncertainty, '
        '%d members)...', len(members),
    )
    base_dir = os.path.join(output_dir, 'Sigma_USBR')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(
        pred_data_dir, f'Predictor_{year_list[0]}.tif',
    )

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # CAP pixel mask (set earlier in run_uncertainty_quantification via
    # _set_cap_pixel_mask_context).  This mask aligns with the valid-
    # pixel ordering of year_df rows.  Falls back to building it from
    # vector_dir if the orchestrator setup did not run or failed.
    cap_pixel_mask = _CAP_PIXEL_MASK_CTX.get('mask')
    if cap_pixel_mask is None and vector_dir is not None:
        logger.info(
            'σ_USBR: CAP pixel mask not in context; building from %s.',
            vector_dir,
        )
        cap_pixel_mask = _build_cap_pixel_mask(
            vector_dir, pred_data_dir, year_list[0],
        )
        if cap_pixel_mask is not None:
            _set_cap_pixel_mask_context(cap_pixel_mask)
    if cap_pixel_mask is None:
        logger.warning(
            'σ_USBR: CAP pixel mask not set and could not be built; '
            'σ_USBR will be zero. Ensure vector_dir contains '
            'CAP_Service_Area.geojson.'
        )
        return {}, {c: {} for c in partops.CATEGORIES}

    # CO River watershed perturbation context.  Builds per-pixel
    # lf_share + watershed area arrays so σ_USBR can perturb
    # streamflow_mm at any pixel whose surface watershed contains
    # one or more LF-derived gauges (Lees Ferry, Imperial Dam,
    # CAP Canal at Havasu — see streamflowops.USBR_DERIVED_GAUGES).
    # This extends σ_USBR scope to all CO-river-served basins:
    # Parker / CRIT / Mohave / Yuma / mainstem AZ — not just CAP
    # service area.
    if sites_csv is not None and watershed_geojson is not None:
        co_lf_share, co_ws_area = _build_co_watershed_co_flow_arrays(
            sites_csv, watershed_geojson, pred_data_dir, year_list[0],
        )
    else:
        co_lf_share = co_ws_area = None
    co_watershed_active = (
        co_lf_share is not None and (co_lf_share > 0).any()
    )

    # σ_USBR loop bounds (region-specific gating applied per-pixel
    # inside the loop; loop start is the earliest year *any* region
    # has signal):
    #   * CAP pixel perturbation gated to year >= CAP_OPERATIONAL_START
    #     (1985, Phoenix reach completion).
    #   * CO watershed pixel perturbation gated to
    #     year >= USBR_DATA_START (1950, USBR CMIP3 ensemble first
    #     year).  No pre-1950 backfill — per-member 1950-2005 long-
    #     term mean ratios collapse to ~1.0 (std 0.011) by
    #     construction, which would produce misleading near-zero σ
    #     pre-1950 implying we modeled it.  Honest answer is "no
    #     inter-member spread available pre-1950 → σ_USBR absent."
    if co_watershed_active:
        usbr_start = max(start_year, sfops.USBR_DATA_START)
    else:
        usbr_start = max(start_year, partops.CAP_OPERATIONAL_START)
    if usbr_start > start_year:
        reason = (
            'pre-USBR-ensemble' if co_watershed_active
            else 'pre-CAP-operational, no CO-mainstem context'
        )
        logger.info(
            '  σ_USBR: skipping years %d-%d (%s).',
            start_year, usbr_start - 1, reason,
        )

    # Pre-compute per-year per-member ratios + ensemble-mean flow at
    # Lees Ferry (00013).  Two pathways:
    #   (1) ML-feature pathway — additive perturbation on
    #       ``streamflow_mm`` (an ML feature) at CAP pixels (uniform
    #       CAP overlay) AND CO watershed pixels (per-watershed
    #       LF-share × LF flow / ws_area).  Local watershed runoff
    #       baked into streamflow_mm is preserved exactly; only the
    #       LF-attributable component is scaled.
    #   (2) Partition pathway — multiplicative perturbation on
    #       ``cw_streamflow`` + ``sw_rights_density`` columns at CAP
    #       pixels only (NOT extended to CO watershed pixels because
    #       it would over-scale local Bill Williams flow and the
    #       senior mainstem priority makes this pathway physically
    #       small at CO watershed pixels).
    logger.info(
        '  Loading USBR member ratios + ensemble-mean flow for '
        'Lees Ferry (years %d-%d, no pre-1950 backfill)...',
        usbr_start, end_year,
    )
    ratios = sfops.compute_usbr_member_annual_ratios(
        usbr_dir, members, usbr_ids=['00013'],
    )
    lees_ratios = ratios.get('00013', {})
    lees_ens_mean_m3s = sfops.compute_usbr_ensemble_annual_mean(
        usbr_dir, members, usbr_id='00013',
    )

    # CAP service-area in m² from the rasterised pixel mask.  This is
    # the same area the central streamflowops uses to convert Lees
    # Ferry m³/s → CAP-overlay mm/yr (modulo small rasterisation
    # rounding vs the polygon area).
    cap_area_m2 = int(cap_pixel_mask.sum()) * pixel_area_m2
    m3s_to_mm_yr = 31_557_600_000.0  # seconds/year × 1000 mm/m

    sigma_usbr: dict[int, np.ndarray] = {}
    yearly_stats: dict = {}
    basin_accum: dict = {'basin': {}, 'subbasin': {}}
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
    cat_sigma_usbr: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    # Partition-stage columns get the multiplicative perturbation.
    # ``streamflow_mm`` is handled separately (additive, ML feature).
    sw_cols = (
        'canal_weighted_streamflow_mm',
        'irr_sw_rights_density',
        'nonirr_sw_rights_density',
        'sw_rights_density',
    )

    for year in range(usbr_start, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        # Build per-pixel co_flow_mm (sum of two contributions):
        #   * CAP overlay — uniform CAP-area-normalised LF flow at
        #     CAP pixels, gated to year >= CAP_OPERATIONAL_START.
        #   * CO watershed — per-watershed LF-share × LF flow /
        #     ws_area_m2 at pixels in LF-derived watersheds, applied
        #     for all years (with backfill).
        ens_m3s = lees_ens_mean_m3s.get(year, 0.0)
        n_valid = int(valid_mask.sum())
        co_flow_mm_per_pixel = np.zeros(n_valid, dtype=np.float32)

        # CAP overlay contribution.
        if year >= partops.CAP_OPERATIONAL_START and cap_area_m2 > 0:
            cap_overlay_mm = ens_m3s * m3s_to_mm_yr / cap_area_m2
            co_flow_mm_per_pixel[cap_pixel_mask] += cap_overlay_mm

        # CO watershed contribution (LF-share-weighted per watershed).
        if co_watershed_active:
            ws_co_mm = (
                co_lf_share * ens_m3s * m3s_to_mm_yr / co_ws_area
            ).astype(np.float32)
            co_flow_mm_per_pixel += ws_co_mm

        # Pixels with non-zero co_flow get the additive ML-feature
        # perturbation.  Pixels in cap_pixel_mask additionally get
        # the multiplicative partition-stage perturbation.
        perturb_mask = co_flow_mm_per_pixel > 0
        perturb_idx = year_df.index[perturb_mask]
        cap_idx = year_df.index[cap_pixel_mask]
        sf_central = (
            year_df.loc[perturb_idx, 'streamflow_mm'].values
            if 'streamflow_mm' in year_df.columns else None
        )

        member_preds = []
        member_cats = []
        for m in members:
            ratio = lees_ratios.get(m, {}).get(year, 1.0)
            if not np.isfinite(ratio) or ratio <= 0:
                ratio = 1.0

            # ML-feature perturbation: rebuild prediction features
            # with streamflow_mm shifted by the member's per-pixel
            # Upper-Basin delta.  Re-predict per member because
            # streamflow_mm IS an ML feature (unlike cw_streamflow,
            # which is in DROP_ATTRS).
            ml_year_df = year_df.copy()
            if sf_central is not None and perturb_mask.any():
                delta_mm = (
                    (ratio - 1.0) * co_flow_mm_per_pixel[perturb_mask]
                )
                ml_year_df.loc[perturb_idx, 'streamflow_mm'] = (
                    sf_central + delta_mm
                )
            pf_member = _build_pred_features(
                ml_year_df, feature_cols, drop_attrs,
            )
            raw = np.abs(model.predict(pf_member))

            # Partition-stage perturbation: multiplicative on
            # cw_streamflow + sw_rights at CAP pixels only (gated to
            # year >= 1985 by the same CAP-operational reasoning).
            sc_year_df = ml_year_df  # streamflow_mm already updated
            if year >= partops.CAP_OPERATIONAL_START:
                for col in sw_cols:
                    if col in sc_year_df.columns:
                        sc_year_df.loc[cap_idx, col] = (
                            sc_year_df.loc[cap_idx, col].values * ratio
                        )
            cats = _partition_with_ctx(
                partops, raw, sc_year_df, raster_shape, valid_mask,
                year=year, skip_cap_perturbation=True,
            )
            pred = cats['Irrigation'] + cats['Non_Irrigation']
            member_preds.append(pred)
            member_cats.append(cats)

        sc_stack = np.stack(member_preds, axis=0)
        std = _safe_nanstd(sc_stack, axis=0, ddof=1)
        sigma_usbr[year] = std

        cat_std = _compute_category_sigmas(member_cats)
        for c in partops.CATEGORIES:
            cat_sigma_usbr[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(
            member_preds, year_df, mm_to_m3,
        )
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(
            member_cats, year_df, mm_to_m3,
        )
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(
            std, basin_flat, valid_mask, raster_shape,
            ref_raster_file,
            os.path.join(raster_dir, f'Sigma_USBR_mm_{year}.tif'),
            read_raster_as_arr, write_raster,
        )
        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 20 == 0 or year == end_year:
            logger.info(
                f'    Year {year}: mean σ_USBR = '
                f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm'
            )

    _save_summary(yearly_stats, base_dir, 'USBR')
    _write_basin_sigma_csv(basin_accum, base_dir, 'USBR')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat], base_dir,
            f'USBR_{cat}', basin_only=True,
        )

    logger.info('  σ_USBR complete.')
    return sigma_usbr, cat_sigma_usbr


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
    Compute σ_gw: ML-feature sensitivity to recent year-over-year variability
    in HarDWR well-registry counts.

    For each prediction year, ``well_density`` — the #1 feature by mean
    |SHAP value| in the XGBRF model — is swapped with values observed in each
    of five recent reference years (INFRASTRUCTURE_SNAPSHOT_YEARS, 2020–2024).
    The model is re-run on the modified feature matrix and σ_gw is the sample
    std across the five predictions.

    ``well_density`` was chosen because it dominates the model's learned
    response to GW infrastructure (mean |SHAP| ≈ 24 mm, roughly 8× the
    contribution of the Hung ``annual_gw_fraction`` feature that previous
    versions of this function perturbed). ``sw_rights_density`` is *not*
    perturbed: it is rank 15 in SHAP importance (≈2 mm) and is effectively
    frozen in the HarDWR record after ~1996 (per-pixel std across 2020–2024
    is 0.000), so probing it would contribute negligibly to σ_total while
    requiring counterfactuals that don't exist in the data.

    σ_gw does not represent new-well drilling, well retirement, or spatial
    redistribution of infrastructure beyond what the five reference years
    already contain — those are out of scope for this study absent
    structural scenario projections.

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona predictor DataFrame spanning all years.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters (used
            to load the reference basin raster for shape/mask).
        output_dir (str): Base output directory for uncertainty products.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        mosaic_res (int): Raster resolution in meters.

    Returns:
        tuple: (sigma_gw, cat_sigma_gw) — per-year total σ and
            per-category per-year σ arrays.
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info(
        'Computing σ_gw (well_density sensitivity across %d recent HarDWR snapshots)...',
        len(INFRASTRUCTURE_SNAPSHOT_YEARS),
    )
    base_dir = os.path.join(output_dir, 'Sigma_GW')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Load well_density snapshots from az_df for each reference year.
    # Pixel ordering is consistent across years within az_df (each year block
    # is a row-stack of the same valid pixels), so snap_df['well_density']
    # has the same length as year_df and can be assigned directly.
    snapshot_wd: dict[int, np.ndarray] = {}
    for snap_year in INFRASTRUCTURE_SNAPSHOT_YEARS:
        snap_df = az_df[az_df.Year == snap_year]
        if snap_df.empty:
            logger.warning(
                'Infrastructure snapshot year %d not present in az_df; skipping',
                snap_year,
            )
            continue
        snapshot_wd[snap_year] = snap_df['well_density'].values

    if len(snapshot_wd) < 2:
        logger.warning(
            'Fewer than 2 infrastructure snapshots available; returning empty sigma_gw',
        )
        return {}, {c: {} for c in partops.CATEGORIES}

    logger.info('  Loaded well_density snapshots: %s', list(snapshot_wd.keys()))

    sigma_gw = {}
    yearly_stats = {}
    basin_accum = {'basin': {}, 'subbasin': {}}
    cat_basin_accum: dict[str, dict] = {
        c: {'basin': {}} for c in partops.CATEGORIES
    }
    cat_sigma_gw: dict[str, dict[int, np.ndarray]] = {
        c: {} for c in partops.CATEGORIES
    }

    for year in range(start_year, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        snapshot_preds = []
        snapshot_cats = []
        for snap_year in snapshot_wd:
            alt_df = year_df.copy()
            alt_df['well_density'] = snapshot_wd[snap_year]
            pf = _build_pred_features(alt_df, feature_cols, drop_attrs)
            pred, cat = _predict_total(model, pf, alt_df, partops,
                                       raster_shape, valid_mask)
            snapshot_preds.append(pred)
            snapshot_cats.append(cat)

        snap_stack = np.stack(snapshot_preds, axis=0)
        std = _safe_nanstd(snap_stack, axis=0, ddof=1)
        sigma_gw[year] = std

        cat_std = _compute_category_sigmas(snapshot_cats)
        for c in partops.CATEGORIES:
            cat_sigma_gw[c][year] = cat_std[c]

        bv, sbv = _aggregate_member_volumes(snapshot_preds, year_df,
                                            mm_to_m3)
        _accumulate_basin_sigma(basin_accum, year, bv, sbv)
        cat_bv = _aggregate_category_member_volumes(
            snapshot_cats, year_df, mm_to_m3,
        )
        _accumulate_category_basin_sigma(cat_basin_accum, year, cat_bv)

        _write_std_raster(std, basin_flat, valid_mask, raster_shape,
                          ref_raster_file,
                          os.path.join(raster_dir, f'Sigma_GW_mm_{year}.tif'),
                          read_raster_as_arr, write_raster)

        yearly_stats[year] = _pixel_stats(std, mm_to_m3)

        if year % 20 == 0 or year == end_year:
            logger.info(f'    Year {year}: mean σ_gw = '
                        f'{yearly_stats[year]["Mean_Depth_mm"]:.2f} mm')

    _save_summary(yearly_stats, base_dir, 'GW')
    _write_basin_sigma_csv(basin_accum, base_dir, 'GW')
    for cat in partops.CATEGORIES:
        _write_basin_sigma_csv(
            cat_basin_accum[cat], base_dir, f'GW_{cat}', basin_only=True,
        )
    logger.info('  σ_gw complete.')
    return sigma_gw, cat_sigma_gw


# ═════════════════════════════════════════════════════════════════════════════
# Density-ratio partitioning sensitivity analysis
# ═════════════════════════════════════════════════════════════════════════════

def run_density_ratio_sensitivity(
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
        delta: float = 0.2,
        sigma_factor: float = 2.0,
        sigma_floor: float = 0.5,
) -> None:
    """
    Partition-level sensitivity diagnostic covering two orthogonal knobs
    that drive the GW/SW split in ``partition_predictions``:

    1. **Density ratio** — both sides of the ratio are perturbed with
       opposite signs:
           plus  : well × (1+δ), sw_rights × (1−δ)  → more GW, less SW
           minus : well × (1−δ), sw_rights × (1+δ)  → less GW, more SW
       Probes how sensitive the split is to a coordinated scaling of the
       well-count and SW-rights-count inputs.

    2. **Smoothing kernel width** — ``sw_smooth_sigma`` in
       ``partition_predictions`` is perturbed *per year* around the
       era-default σ schedule (``partops.era_sw_sigma(year)``) by a
       factor (default 2×).  At each year:

           σ_low  = max(σ_era / sigma_factor, sigma_floor)
           σ_high = σ_era × sigma_factor

       This anchors the sensitivity envelope to the actual production
       σ at every year (rather than a fixed global pair like
       ``{2, 8}``), so the ribbon represents factor-of-2 perturbations
       around what the partition is actually using.  The earlier
       fixed-pair sweep produced misleading asymmetry because
       ``{2, 8}`` poorly bracketed the production schedule (which spans
       0.0 → 6.0 across eras).

    Writes ``Density_Ratio_Sensitivity.csv`` with one row per
    (Year, Perturbation_Type, Category):
        Year, Perturbation_Type, Category, Baseline_AF, Plus_AF, Minus_AF,
        Delta_Plus_AF, Delta_Minus_AF, Pct_Change_Plus, Pct_Change_Minus,
        Sigma_Era, Sigma_Low, Sigma_High
    where ``Perturbation_Type`` ∈ {'Density', 'Smoothing'} and the
    ``Sigma_*`` columns are populated only for Smoothing rows.

    Writes two PNG plots — one per perturbation section.

    The perturbed density columns (irr/nonirr well_density and
    sw_rights_density) are all in ``DROP_ATTRS``, so the ML prediction is
    identical across members and ``model.predict()`` is called only once
    per year; only the partitioning step is re-run.

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        output_dir (str): Base output directory for uncertainty products.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        mosaic_res (int): Raster resolution in meters.
        delta (float): Density perturbation magnitude (default 0.2 = ±20%).
        sigma_factor (float): Multiplicative perturbation around the era
            σ (default 2.0 → halve / double).
        sigma_floor (float): Minimum σ_low to keep the kernel
            non-degenerate (default 0.5; era values of 0.0 are bumped
            to this floor for the low leg only).

    Returns:
        None.
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.sysops import makedirs

    logger.info(
        'Running partition sensitivity analysis '
        '(density ±%.0f%%, smoothing σ × {1/%.1f, %.1f} per-year '
        'around era schedule, floor=%.1f)...',
        delta * 100, sigma_factor, sigma_factor, sigma_floor,
    )
    sens_dir = os.path.join(output_dir, 'Sigma_GW')
    makedirs(sens_dir)

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    from hydrolibs.rasterops import read_raster_as_arr
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()

    required_cols = (
        'irr_well_density', 'nonirr_well_density',
        'irr_sw_rights_density', 'nonirr_sw_rights_density',
    )

    def _mk_row(yr, ptype, cat, base_af, plus_af, minus_af,
                sigma_era=None, sigma_low=None, sigma_high=None):
        d_plus = plus_af - base_af
        d_minus = minus_af - base_af
        pct_plus = (d_plus / base_af * 100) if base_af > 0 else 0.0
        pct_minus = (d_minus / base_af * 100) if base_af > 0 else 0.0
        return {
            'Year': yr,
            'Perturbation_Type': ptype,
            'Category': cat,
            'Baseline_AF': round(base_af, 2),
            'Plus_AF': round(plus_af, 2),
            'Minus_AF': round(minus_af, 2),
            'Delta_Plus_AF': round(d_plus, 2),
            'Delta_Minus_AF': round(d_minus, 2),
            'Pct_Change_Plus': round(pct_plus, 2),
            'Pct_Change_Minus': round(pct_minus, 2),
            'Sigma_Era': round(sigma_era, 4) if sigma_era is not None else np.nan,
            'Sigma_Low': round(sigma_low, 4) if sigma_low is not None else np.nan,
            'Sigma_High': round(sigma_high, 4) if sigma_high is not None else np.nan,
        }

    rows = []

    for year in range(start_year, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        if any(c not in year_df.columns for c in required_cols):
            continue

        # Predict once per year — all perturbed columns are in DROP_ATTRS
        # (they are partitioning-only inputs), so model.predict() is
        # identical across all members. Only partition_predictions varies.
        pf_base = _build_pred_features(year_df, feature_cols, drop_attrs)
        raw = np.abs(model.predict(pf_base))

        # --- Baseline partition (era-default σ schedule) ---
        cats_base = _partition_with_ctx(
            partops, raw, year_df, raster_shape, valid_mask,
            year=year,
        )

        # --- Density section: both sides of ratio, opposite signs.
        # Use era-default σ so baseline vs density-perturbed isolates
        # the density effect cleanly.
        plus_df = year_df.copy()
        plus_df['irr_well_density'] = year_df['irr_well_density'].values * (1 + delta)
        plus_df['nonirr_well_density'] = year_df['nonirr_well_density'].values * (1 + delta)
        plus_df['irr_sw_rights_density'] = year_df['irr_sw_rights_density'].values * max(1 - delta, 0)
        plus_df['nonirr_sw_rights_density'] = year_df['nonirr_sw_rights_density'].values * max(1 - delta, 0)
        cats_density_plus = _partition_with_ctx(
            partops, raw, plus_df, raster_shape, valid_mask,
            year=year,
        )

        minus_df = year_df.copy()
        minus_df['irr_well_density'] = year_df['irr_well_density'].values * max(1 - delta, 0)
        minus_df['nonirr_well_density'] = year_df['nonirr_well_density'].values * max(1 - delta, 0)
        minus_df['irr_sw_rights_density'] = year_df['irr_sw_rights_density'].values * (1 + delta)
        minus_df['nonirr_sw_rights_density'] = year_df['nonirr_sw_rights_density'].values * (1 + delta)
        cats_density_minus = _partition_with_ctx(
            partops, raw, minus_df, raster_shape, valid_mask,
            year=year,
        )

        # --- Smoothing section: baseline densities, swept sigma ---
        # Per-year sweep anchored to the era schedule:
        #   σ_low  = max(σ_era / sigma_factor, sigma_floor)
        #   σ_high = σ_era × sigma_factor
        # When σ_era == 0 (very-pre-Yuma years), the σ_low leg is
        # bumped to sigma_floor and σ_high to sigma_floor × sigma_factor
        # so the kernel remains non-degenerate and the perturbation
        # actually moves something.
        sigma_era = partops.era_sw_sigma(year)
        if sigma_era <= 0:
            sigma_low = sigma_floor
            sigma_high = sigma_floor * sigma_factor
        else:
            sigma_low = max(sigma_era / sigma_factor, sigma_floor)
            sigma_high = sigma_era * sigma_factor
        cats_smooth_high = _partition_with_ctx(
            partops, raw, year_df, raster_shape, valid_mask,
            year=year, sw_smooth_sigma=sigma_high,
        )
        cats_smooth_low = _partition_with_ctx(
            partops, raw, year_df, raster_shape, valid_mask,
            year=year, sw_smooth_sigma=sigma_low,
        )

        for cat in partops.CATEGORIES:
            base_af = float(np.nansum(cats_base[cat])) * mm_to_m3 * M3_TO_AF

            # Density row
            d_plus_af = float(np.nansum(cats_density_plus[cat])) * mm_to_m3 * M3_TO_AF
            d_minus_af = float(np.nansum(cats_density_minus[cat])) * mm_to_m3 * M3_TO_AF
            rows.append(_mk_row(year, 'Density', cat, base_af, d_plus_af, d_minus_af))

            # Smoothing row ("Plus" = high sigma = wider reach)
            s_plus_af = float(np.nansum(cats_smooth_high[cat])) * mm_to_m3 * M3_TO_AF
            s_minus_af = float(np.nansum(cats_smooth_low[cat])) * mm_to_m3 * M3_TO_AF
            rows.append(_mk_row(
                year, 'Smoothing', cat, base_af, s_plus_af, s_minus_af,
                sigma_era=sigma_era, sigma_low=sigma_low,
                sigma_high=sigma_high,
            ))

        if year % 20 == 0 or year == end_year:
            logger.info('    Sensitivity year %d done', year)

    sens_df = pd.DataFrame(rows)
    out_csv = os.path.join(sens_dir, 'Density_Ratio_Sensitivity.csv')
    sens_df.to_csv(out_csv, index=False)
    logger.info('Partition sensitivity results saved to %s', out_csv)

    # Log summary for key GW/SW categories, per perturbation type
    plot_cats = ('Irrigation_GW', 'Irrigation_SW',
                 'Non_Irrigation_GW', 'Non_Irrigation_SW',
                 'Total_GW', 'Total_SW')
    for ptype in ('Density', 'Smoothing'):
        for cat in plot_cats:
            cdf = sens_df[(sens_df.Perturbation_Type == ptype) & (sens_df.Category == cat)]
            if cdf.empty:
                continue
            mean_pct_plus = cdf['Pct_Change_Plus'].mean()
            mean_pct_minus = cdf['Pct_Change_Minus'].mean()
            logger.info(
                '  [%s] %s: mean %%Δ = %+.1f%% (plus), %+.1f%% (minus)',
                ptype, cat, mean_pct_plus, mean_pct_minus,
            )

    # --- Plots: one PNG per perturbation section ---
    _plot_sens_section(
        sens_df[sens_df.Perturbation_Type == 'Density'], plot_cats,
        title=f'Density Ratio Sensitivity (well ±{delta * 100:.0f}%, SW-rights opposite sign)',
        ribbon_label=f'well density ±{delta * 100:.0f}%, SW-rights density opposite sign',
        out_path=os.path.join(sens_dir, 'Density_Ratio_Sensitivity.png'),
        start_year=start_year, end_year=end_year,
    )
    _plot_sens_section(
        sens_df[sens_df.Perturbation_Type == 'Smoothing'], plot_cats,
        title=(
            f'Smoothing Sigma Sensitivity '
            f'(σ_era × {{1/{sigma_factor:.0f}, {sigma_factor:.0f}}} '
            f'per year, floor={sigma_floor:.1f})'
        ),
        ribbon_label=(
            f'sw_smooth_sigma = era × {{1/{sigma_factor:.0f}, '
            f'{sigma_factor:.0f}}} (per year)'
        ),
        out_path=os.path.join(sens_dir, 'Smoothing_Sigma_Sensitivity.png'),
        start_year=start_year, end_year=end_year,
    )
    logger.info('  Partition sensitivity plots saved to %s', sens_dir)


# ── CAP delivery reduction scenarios ──────────────────────────────────────

CAP_SCENARIOS: dict[str, float] = {
    'Baseline_900kAF': 1.0,
    'Basic_Coordination_237kAF': 0.263,
    'Extreme_Shortage_0kAF': 0.0,
    'DCP_Tier0_192kAF_cut': 0.787,
    'DCP_Tier1_512kAF_cut': 0.431,
    'DCP_Tier2a_592kAF_cut': 0.342,
    'DCP_Tier2b_640kAF_cut': 0.289,
    'DCP_Tier3_720kAF_cut': 0.200,
}

# Scenario-specific well_density boost factors, paralleling the
# hindcast ``CAP_CUT_GW_BOOST_FACTORS`` in partitionops.py.  Applied
# to the same well_density / irr_well_density / nonirr_well_density
# columns at CAP pixels during scenario runs.  Mathematically
# equivalent to boosting ``gw_weight`` at those pixels — shifts the
# density-ratio allocation toward GW without changing the ML-predicted
# total pumping.
#
# Mapping is era-analogous against the recalibrated historical gw_w
# schedule (post the partition-side recalibration: pre-CAP baseline
# 1945-1980 = 10.0, pre-CAP peak 1971-1979 = 15.0).  Each tier maps
# to a historical regime where AZ pumping had a documented GW share:
#   Baseline (no cut)  → 1.0   (post-CAP gw_w 0.2; effective 0.2)
#   Tier 0 (192 kAF)   → 1.0   (no real cut; post-CAP)
#   Tier 1 (512 kAF)   → 10.0  (GMA transition 1981-1984 gw_w=2.0;
#                                effective gw_w = 0.2 × 10 = 2.0)
#   Tier 2a (592 kAF)  → 18.0  (interpolated between Tier 1 and
#                                Basic Coord by cut magnitude)
#   Tier 2b (640 kAF)  → 23.0  (interpolated)
#   Basic Coord (663)  → 25.0  (between GMA and pre-CAP; effective 5.0)
#   Tier 3 (720 kAF)   → 50.0  (pre-CAP baseline 1945-1980 gw_w=10.0;
#                                effective gw_w = 10.0)
#   Extreme Shortage   → 75.0  (pre-CAP peak 1971-1979 gw_w=15.0;
#                                effective gw_w = 15.0)
#
# Justification: a CAP shortage that physically forces AZ users back
# toward GW dominance should reach the SAME effective gw_w as the
# historical era when GW dominated to a comparable degree.  Note that
# our model has no regulatory ceiling — all "lost CAP" is apportioned
# to GW via the density-ratio shift, so these boosts approximate the
# physical substitution magnitude rather than a regulatory shortage
# ledger.  Conservative subset of the full era-analogous mapping;
# applying the literal era equivalence (Basic Coord at 50, Tier 3 at
# 75, etc.) would overshoot WestWater 2026 cumulative drawdown.
CAP_SCENARIO_GW_BOOSTS: dict[str, float] = {
    'Baseline_900kAF': 1.0,
    'Basic_Coordination_237kAF': 25.0,
    'Extreme_Shortage_0kAF': 75.0,
    'DCP_Tier0_192kAF_cut': 1.0,
    'DCP_Tier1_512kAF_cut': 10.0,
    'DCP_Tier2a_592kAF_cut': 18.0,
    'DCP_Tier2b_640kAF_cut': 23.0,
    'DCP_Tier3_720kAF_cut': 50.0,
}

# Non-well offset: only reclaimed water (350 kAF) is added on top of
# the model's per-pixel partition.  CAP and SRP deliveries are already
# captured by the model's SW components via the wide-σ Gaussian-
# smoothed sw_rights × canal_weighted_streamflow signal — verified by
# comparing model 4-AMA Total_SW (~1300 kAF/yr) against CAP-direct +
# SRP irrigation district deliveries (~1300 kAF/yr) for 1990-2022,
# with model/reference ratio averaging ~1.0.  Yuma direct Colorado
# River diversions are also captured in the model's Yuma basin output
# (~440 kAF/yr).  Reclaimed water is the only delivery source not
# captured by any model feature, so it remains as a fixed offset.
NON_WELL_OFFSET_FIXED_AF = 350_000  # Reclaimed water only
NON_WELL_OFFSET_CAP_AF = 0          # Captured by model SW (no double-count)

# Baseline CAP delivery (AF) used for the additive scenario perturbation
# to canal_weighted_streamflow_mm.  Scenario factors represent the
# fraction of this baseline that remains (factor=0 → subtract the full
# 900 kAF overlay equivalent; factor=0.2 → subtract 720 kAF, matching
# DCP_Tier3_720kAF_cut).  Isolating an AF-calibrated CAP-import slice
# preserves the SRP and Salt/Verde watershed signal at CAP-overlap
# pixels in Phoenix AMA — the multiplicative ``cw_sf *= factor``
# alternative would zero those out as well, producing physically
# implausible 100-percent SW collapse and ~3× over-substitution.
BASELINE_CAP_DELIVERY_AF = 900_000


def _compute_cap_overlay_per_pixel(
        year_df: pd.DataFrame,
        cap_pixel_mask: np.ndarray,
        pixel_area_m2: float,
) -> np.ndarray:
    """Per-pixel CAP-import contribution (mm/yr) to canal_weighted_streamflow.

    Distributes the ``BASELINE_CAP_DELIVERY_AF`` volume across CAP
    service-area pixels weighted by ``canal_density`` (matching the
    distribution rule used by streamflowops when the CAP overlay is
    added to canal-weighted streamflow).  Returned array is aligned
    to ``cap_pixel_mask.sum()`` rows in the same order as
    ``year_df.index[cap_pixel_mask]``.
    """
    if 'canal_density' not in year_df.columns:
        return np.zeros(int(cap_pixel_mask.sum()), dtype=np.float32)
    canal_dens_cap = year_df.loc[
        year_df.index[cap_pixel_mask], 'canal_density',
    ].values
    cap_canal_sum = float(canal_dens_cap.sum())
    if cap_canal_sum <= 0:
        return np.zeros_like(canal_dens_cap, dtype=np.float32)
    af_to_m3 = 1.0 / M3_TO_AF
    cap_overlay_total_mm = (
        BASELINE_CAP_DELIVERY_AF * af_to_m3 * 1000.0 / pixel_area_m2
    )
    return (cap_overlay_total_mm * (canal_dens_cap / cap_canal_sum)).astype(
        np.float32
    )


def run_cap_scenario_analysis(
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
        cap_service_area_geojson: str,
        scenarios: dict[str, float] | None = None,
) -> None:
    """Simulate CAP delivery reduction scenarios via partition perturbation.

    **Counterfactual semantics (important for WestWater comparison).**
    Every scenario — *including* ``Baseline_900kAF`` — bypasses the
    central `partops.apply_cap_delivery_perturbation` (via
    ``_partition_with_ctx(..., skip_cap_perturbation=True)``).
    ``Baseline_900kAF`` therefore represents a **true no-cut
    counterfactual** (full CAP delivery, no central Tier perturbation)
    and every non-Baseline scenario applies its own cut on that
    un-perturbed reference.  This makes
    ``CAP_Scenario_Cumulative.csv`` directly comparable to WestWater
    2026's ~8.0 MAF drawdown anchor, which is also a "with-cut vs
    no-cut" delta within a single climate/LULC projection.

    For each scenario two complementary perturbations are applied at
    CAP service-area pixels:

    1. **Additive SW cut** — the CAP-import contribution to
       ``canal_weighted_streamflow_mm`` is reduced by
       ``(1 - factor) × cap_overlay_per_pixel`` (where
       ``cap_overlay_per_pixel`` distributes
       ``BASELINE_CAP_DELIVERY_AF`` canal-density-weighted across the
       CAP service area).  The subtraction preserves the SRP /
       Salt-Verde watershed signal at CAP-overlap pixels — the prior
       multiplicative ``cw_sf *= factor`` form would zero those out
       as well, producing ~3× over-substitution at the
       ``Extreme_Shortage_0kAF`` endpoint.

    2. **Multiplicative GW-weight boost** — the ``well_density``
       columns (`well_density`, `irr_well_density`,
       `nonirr_well_density`) are scaled by
       ``CAP_SCENARIO_GW_BOOSTS[scenario]``.  Mathematically
       equivalent to boosting ``gw_weight`` at CAP pixels, this
       shifts the density-ratio allocation toward GW during shortage
       scenarios without affecting the ML-predicted total pumping.
       Mirrors the hindcast boost logic in
       ``partops.apply_cap_delivery_perturbation``.

    Total withdrawals per pixel stay fixed; only the GW/SW split
    changes.

    Scenarios follow WestWater Research (2026) and USBR DEIS DCP tiers.

    Outputs:
        - ``CAP_Scenario_Basin.csv``   — per-basin per-category volumes
        - ``CAP_Scenario_Statewide.csv`` — statewide totals + reconciled WU
        - ``CAP_Scenario_Delta.csv``   — change vs Baseline
        - ``CAP_Scenario_Cumulative.csv`` — cumulative additional GW
        - Time-series plots (WestWater, DCP tiers, per-basin, cumulative)
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr
    from hydrolibs.sysops import makedirs

    if scenarios is None:
        scenarios = CAP_SCENARIOS

    logger.info('Running CAP delivery reduction scenario analysis...')
    logger.info('  Scenarios: %s', list(scenarios.keys()))
    cap_dir = os.path.join(output_dir, 'CAP_Scenario')
    makedirs(cap_dir)

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Build valid_mask and raster_shape from reference basin raster
    ref_basin_file = os.path.join(
        pred_data_dir, f'GW_Basin_{year_list[0]}.tif',
    )
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()

    # Build CAP pixel mask by rasterizing the CAP service area
    # onto the same grid as the basin raster
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize as rio_rasterize

    cap_gdf = gpd.read_file(cap_service_area_geojson)
    with rasterio.open(ref_basin_file) as ref_src:
        cap_gdf_proj = cap_gdf.to_crs(ref_src.crs)
        cap_arr = rio_rasterize(
            [(geom, 1) for geom in cap_gdf_proj.geometry],
            out_shape=raster_shape,
            transform=ref_src.transform,
            fill=0,
            dtype='uint8',
        )
    cap_flat = cap_arr.ravel()
    # CAP mask aligned to valid pixels (same ordering as year_df rows)
    cap_pixel_mask = cap_flat[valid_mask] > 0
    n_cap_pixels = int(cap_pixel_mask.sum())
    logger.info(
        '  CAP service area: %d pixels (of %d valid)',
        n_cap_pixels, int(valid_mask.sum()),
    )

    # Collect results
    basin_rows = []
    statewide_rows = []
    delta_rows = []

    projection_start = max(start_year, 2026)

    for year in range(projection_start, end_year + 1):
        year_df = az_df[az_df.Year == year].copy()
        if year_df.empty:
            continue

        pf = _build_pred_features(year_df, feature_cols, drop_attrs)
        raw = np.abs(model.predict(pf))

        pixel_basins = year_df['GW_Basin'].values
        all_basins = sorted(set(pixel_basins))

        # Pre-compute per-basin masks once per year
        basin_masks = {b: (pixel_basins == b) for b in all_basins}

        # Per-pixel CAP-import overlay contribution (mm/yr) — additive
        # perturbation magnitude.  Recomputed per year because the rows
        # of year_df are filtered from az_df.
        cap_overlay_per_pixel = _compute_cap_overlay_per_pixel(
            year_df, cap_pixel_mask, pixel_area_m2,
        )

        # Baseline partition = true "no-cut counterfactual" (bypass
        # central CAP perturbation so Baseline_900kAF truly means
        # "no CAP cuts applied").  Every non-Baseline scenario will
        # also set skip_cap_perturbation=True and apply its own
        # scenario-specific perturbation — this keeps all scenario
        # deltas measured against the same no-cut reference, which
        # is what WestWater 2026's 8.0 MAF drawdown anchor represents.
        cats_base = _partition_with_ctx(
            partops, raw, year_df, raster_shape, valid_mask, year=year,
            skip_cap_perturbation=True,
        )
        base_basin_gw: dict[str, float] = {}
        base_basin_sw: dict[str, float] = {}

        # Record baseline statewide + basin stats
        sw_row = {'Year': year, 'Scenario': 'Baseline_900kAF'}
        for cat in partops.CATEGORIES:
            vol_af = float(np.nansum(cats_base[cat])) * mm_to_m3 * M3_TO_AF
            sw_row[f'{cat}_AF'] = round(vol_af, 2)
        well_total = sw_row['Irrigation_AF'] + sw_row['Non_Irrigation_AF']
        offset = NON_WELL_OFFSET_FIXED_AF + NON_WELL_OFFSET_CAP_AF
        sw_row['Well_Mediated_Total_AF'] = round(well_total, 2)
        sw_row['Non_Well_Offset_AF'] = round(offset, 2)
        sw_row['Estimated_Statewide_Total_AF'] = round(
            well_total + offset, 2,
        )
        statewide_rows.append(sw_row)

        for basin, bmask in basin_masks.items():
            b_row = {
                'Year': year, 'Scenario': 'Baseline_900kAF',
                'Basin': basin,
            }
            for cat in partops.CATEGORIES:
                vol_af = (
                    float(np.nansum(cats_base[cat][bmask]))
                    * mm_to_m3 * M3_TO_AF
                )
                b_row[f'{cat}_AF'] = round(vol_af, 2)
            basin_rows.append(b_row)
            base_basin_gw[basin] = (
                float(np.nansum(cats_base['Total_GW'][bmask]))
                * mm_to_m3 * M3_TO_AF
            )
            base_basin_sw[basin] = (
                float(np.nansum(cats_base['Total_SW'][bmask]))
                * mm_to_m3 * M3_TO_AF
            )

        del cats_base

        # Process each non-baseline scenario one at a time.  Each
        # scenario is applied on the un-perturbed year_df (matching
        # the no-cut counterfactual baseline above) so that scenario
        # deltas are directly comparable to WestWater 2026's
        # with-cut-vs-no-cut projections.
        for sc_name, factor in scenarios.items():
            if factor == 1.0:
                continue
            sc_df = year_df.copy()
            cap_idx = sc_df.index[cap_pixel_mask]
            # Additive SW cut: subtract (1 - factor) of the CAP-import
            # overlay from canal_weighted_streamflow at CAP pixels,
            # leaving the SRP / Salt-Verde watershed component
            # intact.  See _compute_cap_overlay_per_pixel docstring.
            new_cw = (
                sc_df.loc[cap_idx, 'canal_weighted_streamflow_mm'].values
                - (1.0 - factor) * cap_overlay_per_pixel
            )
            sc_df.loc[cap_idx, 'canal_weighted_streamflow_mm'] = np.clip(
                new_cw, 0.0, None,
            )
            # GW-weight boost: scale well_density columns at CAP
            # pixels to shift the density-ratio allocation toward GW
            # (mathematically equivalent to boosting gw_weight at
            # those pixels).  Parallels the hindcast boost logic in
            # partops.apply_cap_delivery_perturbation.
            gw_boost = CAP_SCENARIO_GW_BOOSTS.get(sc_name, 1.0)
            if gw_boost != 1.0:
                for col in (
                    'well_density',
                    'irr_well_density',
                    'nonirr_well_density',
                ):
                    if col in sc_df.columns:
                        sc_df.loc[cap_idx, col] *= gw_boost
            cats = _partition_with_ctx(
                partops, raw, sc_df, raster_shape, valid_mask, year=year,
                skip_cap_perturbation=True,
            )

            sw_row = {'Year': year, 'Scenario': sc_name}
            for cat in partops.CATEGORIES:
                vol_af = float(np.nansum(cats[cat])) * mm_to_m3 * M3_TO_AF
                sw_row[f'{cat}_AF'] = round(vol_af, 2)
            well_total = sw_row['Irrigation_AF'] + sw_row['Non_Irrigation_AF']
            sc_offset = (
                NON_WELL_OFFSET_FIXED_AF + NON_WELL_OFFSET_CAP_AF * factor
            )
            sw_row['Well_Mediated_Total_AF'] = round(well_total, 2)
            sw_row['Non_Well_Offset_AF'] = round(sc_offset, 2)
            sw_row['Estimated_Statewide_Total_AF'] = round(
                well_total + sc_offset, 2,
            )
            statewide_rows.append(sw_row)

            for basin, bmask in basin_masks.items():
                b_row = {
                    'Year': year, 'Scenario': sc_name, 'Basin': basin,
                }
                for cat in partops.CATEGORIES:
                    vol_af = (
                        float(np.nansum(cats[cat][bmask]))
                        * mm_to_m3 * M3_TO_AF
                    )
                    b_row[f'{cat}_AF'] = round(vol_af, 2)
                basin_rows.append(b_row)

                sc_gw = (
                    float(np.nansum(cats['Total_GW'][bmask]))
                    * mm_to_m3 * M3_TO_AF
                )
                sc_sw = (
                    float(np.nansum(cats['Total_SW'][bmask]))
                    * mm_to_m3 * M3_TO_AF
                )
                delta_rows.append({
                    'Year': year,
                    'Scenario': sc_name,
                    'Basin': basin,
                    'Baseline_GW_AF': round(base_basin_gw[basin], 2),
                    'Baseline_SW_AF': round(base_basin_sw[basin], 2),
                    'Scenario_GW_AF': round(sc_gw, 2),
                    'Scenario_SW_AF': round(sc_sw, 2),
                    'Delta_GW_AF': round(sc_gw - base_basin_gw[basin], 2),
                    'Delta_SW_AF': round(sc_sw - base_basin_sw[basin], 2),
                })

            del cats, sc_df

        if year % 10 == 0 or year == end_year:
            logger.info('    CAP scenario year %d done', year)

    # Write CSVs
    basin_df = pd.DataFrame(basin_rows)
    basin_df.to_csv(
        os.path.join(cap_dir, 'CAP_Scenario_Basin.csv'), index=False,
    )

    statewide_df = pd.DataFrame(statewide_rows)
    statewide_df.to_csv(
        os.path.join(cap_dir, 'CAP_Scenario_Statewide.csv'), index=False,
    )

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(
        os.path.join(cap_dir, 'CAP_Scenario_Delta.csv'), index=False,
    )

    # Cumulative additional GW drawdown
    if not delta_df.empty:
        cumul_rows = []
        for sc_name in delta_df['Scenario'].unique():
            for basin in delta_df['Basin'].unique():
                sub = delta_df[
                    (delta_df.Scenario == sc_name)
                    & (delta_df.Basin == basin)
                ].sort_values('Year')
                cumsum = sub['Delta_GW_AF'].cumsum()
                for yr, cv in zip(sub['Year'], cumsum):
                    cumul_rows.append({
                        'Year': yr,
                        'Scenario': sc_name,
                        'Basin': basin,
                        'Cumulative_Delta_GW_AF': round(cv, 2),
                    })
        cumul_df = pd.DataFrame(cumul_rows)
        cumul_df.to_csv(
            os.path.join(cap_dir, 'CAP_Scenario_Cumulative.csv'),
            index=False,
        )

    logger.info('  CAP scenario CSVs saved to %s', cap_dir)

    # --- Plots ---
    # Use basin-level σ aggregation (not pixel-level) so ribbons on
    # the CAP scenario plots reflect realistic spatial correlation.
    # Pixel quadrature suppresses AZ-wide σ by ~5× (treating every
    # 2 km pixel as independent), producing ribbons so thin they're
    # invisible on the cumulative drawdown plot.  Basin-level matches
    # the aggregation used in Basin_Sigma_Total.csv.
    az_sigma_per_cat = _load_az_sigma_per_category_basin(
        output_dir, projection_start, end_year,
    )
    _plot_cap_scenarios(
        statewide_df, delta_df,
        scenarios, cap_dir,
        projection_start, end_year,
        az_sigma_per_cat,
    )
    logger.info('  CAP scenario plots saved to %s', cap_dir)
    # NOTE: spatial drawdown maps (basin choropleth, pixel raster,
    # σ-cumulative, signal-to-noise) are produced in Step 3g via
    # ``pipeline.create_cap_scenario_spatial_maps`` so all the
    # raster-style outputs land under ``Raster_Maps/CAP_Scenario/``
    # alongside the rest of the era / σ map suite.


def _load_az_sigma_per_category_basin(
        unc_dir: str,
        start_year: int,
        end_year: int,
) -> dict[str, dict[int, float]]:
    """Statewide σ (AF/yr) per category per year.

    Reads the per-component per-category per-basin σ CSVs written by
    ``compute_sigma_{component}`` (6 components × 8 categories ×
    {Basin, Subbasin}).  Aggregates per basin in QUADRATURE across
    components (σ_basin = √(Σ σ_i²) — components are independent
    uncertainty axes), then across basins in LINEAR SUM
    (AZ-wide σ = Σ basin_σ — components are scenario-driven and
    correlated across basins, so basin σ values move together).

    Returns ``{cat: {year: sigma_af}}``.  Returns an empty dict if
    the per-component CSVs are missing (e.g., Step 3b not run).
    """
    cats = (
        'Total_GW', 'Total_SW',
        'Irrigation', 'Irrigation_GW', 'Irrigation_SW',
        'Non_Irrigation', 'Non_Irrigation_GW', 'Non_Irrigation_SW',
    )
    components = ('MACA', 'Model', 'Irr', 'LULC', 'GW', 'USBR')
    out: dict[str, dict[int, float]] = {cat: {} for cat in cats}

    for cat in cats:
        # Load per-component per-basin σ for this category
        comp_dfs = []
        for comp in components:
            f = os.path.join(
                unc_dir, f'Sigma_{comp}', f'Basin_Sigma_{comp}_{cat}.csv',
            )
            if os.path.isfile(f):
                d = pd.read_csv(f)[['Year', 'Region', 'Sigma_Volume_AF']].rename(
                    columns={'Sigma_Volume_AF': f'sigma_{comp}'},
                )
                comp_dfs.append(d)
        if not comp_dfs:
            continue
        # Merge components per (Year, Region)
        merged = comp_dfs[0]
        for d in comp_dfs[1:]:
            merged = merged.merge(d, on=['Year', 'Region'], how='outer')
        sigma_cols = [c for c in merged.columns if c.startswith('sigma_')]
        merged[sigma_cols] = merged[sigma_cols].fillna(0.0)
        merged['basin_sigma'] = np.sqrt(
            (merged[sigma_cols] ** 2).sum(axis=1),
        )
        # AZ-wide LINEAR SUM across basins per year (basin σ are
        # correlated via shared scenario drivers — see docstring).
        az = merged.groupby('Year')['basin_sigma'].sum()
        for yr, v in az.items():
            yr_int = int(yr)
            if start_year <= yr_int <= end_year:
                out[cat][yr_int] = v
    return out


def _load_az_sigma_per_category(
        sigma_rasters_dir: str,
        start_year: int,
        end_year: int,
        pixel_area_m2: float,
) -> dict[str, dict[int, float]]:
    """Statewide σ (AF/yr) per category per year, by quadrature spatial sum.

    **Pixel-level quadrature** — assumes every 2 km pixel is
    independent, which suppresses AZ-wide σ by a factor of ~5× vs
    basin-level aggregation.  For CAP-scenario / cumulative plots,
    prefer ``_load_az_sigma_per_category_basin`` which handles
    intra-basin correlation more honestly.

    Returns ``{cat: {year: sigma_af}}``.  Returns an empty dict when no
    σ rasters are found (e.g., when sigma-total was skipped) so callers
    can fall back to plotting without ribbons.
    """
    cats = (
        'Total_GW', 'Total_SW',
        'Irrigation_GW', 'Irrigation_SW',
        'Non_Irrigation_GW', 'Non_Irrigation_SW',
    )
    out: dict[str, dict[int, float]] = {cat: {} for cat in cats}
    if not os.path.isdir(sigma_rasters_dir):
        return out
    from hydrolibs.rasterops import read_raster_as_arr
    mm_to_m3 = pixel_area_m2 / 1000.0
    for cat in cats:
        for yr in range(start_year, end_year + 1):
            f = os.path.join(
                sigma_rasters_dir, f'Sigma_Total_{cat}_mm_{yr}.tif',
            )
            if not os.path.isfile(f):
                continue
            arr = read_raster_as_arr(f, get_file=False)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            sigma_m3 = arr * mm_to_m3
            sigma_total_m3 = float(np.sqrt(np.sum(sigma_m3 ** 2)))
            out[cat][yr] = sigma_total_m3 * M3_TO_AF
    return out


def _plot_cap_scenarios(
        statewide_df: pd.DataFrame,
        delta_df: pd.DataFrame,
        scenarios: dict[str, float],
        out_dir: str,
        start_year: int,
        end_year: int,
        az_sigma_per_cat: dict[str, dict[int, float]] | None = None,
) -> None:
    """Render CAP scenario time-series plots.

    Scenario lines are drawn without ±σ shading.  Each scenario is a
    deterministic re-partition of the same central ML prediction with
    only the CAP-pixel multiplicative perturbation differing across
    scenarios, so per-year σ_total values are *identical* across
    scenarios (they reflect the central pipeline's prediction
    uncertainty, not scenario uncertainty).  Plotting them as ribbons
    behind every scenario produced heavy overlap that obscured the
    inter-scenario separation, which is the actual signal of interest
    for a tier comparison.  ``az_sigma_per_cat`` is retained in the
    signature for backwards compatibility but is no longer plotted.
    """
    import matplotlib.pyplot as plt
    from hydrolibs.visualops import apply_journal_style

    apply_journal_style()
    # σ ribbons intentionally suppressed (see docstring) — sigma_lookup
    # retained only for the cumulative-drawdown σ accumulation, which
    # also remains suppressed for visual clarity.
    sigma_lookup = az_sigma_per_cat or {}  # noqa: F841

    cat_pairs = [
        ('Total_GW_AF', 'Total_SW_AF', 'Total', 'Total_GW', 'Total_SW'),
        ('Irrigation_GW_AF', 'Irrigation_SW_AF', 'Irrigation',
         'Irrigation_GW', 'Irrigation_SW'),
        ('Non_Irrigation_GW_AF', 'Non_Irrigation_SW_AF', 'Non-Irrigation',
         'Non_Irrigation_GW', 'Non_Irrigation_SW'),
    ]

    scenario_colors = {
        'Baseline_900kAF': '#2C3E50',
        'Basic_Coordination_237kAF': '#E67E22',
        'Extreme_Shortage_0kAF': '#C0392B',
    }
    # Distinct hue+marker per DCP tier (mild → severe).  The previous
    # all-blue gradient ran together visually; this uses the
    # ColorBrewer YlOrRd 5-class palette starting at the darker end so
    # every line is clearly distinguishable, plus per-tier markers as
    # a second discriminator.
    dcp_styles = {
        'DCP_Tier0_192kAF_cut':  ('#FDD49E', 'o'),  # light orange
        'DCP_Tier1_512kAF_cut':  ('#FDBB84', 's'),  # orange
        'DCP_Tier2a_592kAF_cut': ('#FC8D59', '^'),  # darker orange
        'DCP_Tier2b_640kAF_cut': ('#E34A33', 'D'),  # red
        'DCP_Tier3_720kAF_cut':  ('#B30000', 'v'),  # dark red
    }
    dcp_colors = {k: v[0] for k, v in dcp_styles.items()}
    dcp_markers = {k: v[1] for k, v in dcp_styles.items()}

    def _ribbon(ax, years, values, sigma_cat, color):
        """No-op: σ ribbons suppressed for CAP scenario plots.

        Kept as a stub so the call sites below do not need to change
        and so reintroducing per-scenario ribbons (e.g. with a different
        opacity scheme) is a one-function edit.
        """
        return

    # --- WestWater scenarios (3 main) ---
    westwater = ['Baseline_900kAF', 'Basic_Coordination_237kAF',
                 'Extreme_Shortage_0kAF']
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)

    for row, (gw_col, sw_col, label, gw_key, sw_key) in enumerate(cat_pairs):
        for col, (vol_col, gs_label, sig_key) in enumerate(
            [(gw_col, 'GW', gw_key), (sw_col, 'SW', sw_key)],
        ):
            ax = axes[row, col]
            sigma_cat = sigma_lookup.get(sig_key, {})
            for sc in westwater:
                sdf = statewide_df[
                    statewide_df.Scenario == sc
                ].sort_values('Year')
                if sdf.empty:
                    continue
                color = scenario_colors.get(sc, '#333')
                lbl = sc.replace('_', ' ')
                values = sdf[vol_col].values / 1e6
                _ribbon(ax, sdf['Year'].values, values, sigma_cat, color)
                ax.plot(
                    sdf['Year'], values,
                    label=lbl, color=color, linewidth=1.6,
                )
            ax.set_ylabel('Volume (MAF)', fontweight='bold')
            ax.set_title(f'{label} {gs_label}', fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper left')
    axes[-1, 0].set_xlabel('Year', fontweight='bold')
    axes[-1, 1].set_xlabel('Year', fontweight='bold')
    fig.suptitle(
        'CAP Delivery Reduction Scenarios — GW/SW Partition Shift',
        fontweight='bold', fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        os.path.join(out_dir, 'CAP_Scenario_WestWater.png'), dpi=300,
    )
    plt.close(fig)

    # --- DCP tier scenarios ---
    dcp_names = [k for k in scenarios if k.startswith('DCP_')]
    if dcp_names:
        fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
        all_dcp = ['Baseline_900kAF'] + dcp_names
        for row, (gw_col, sw_col, label, gw_key, sw_key) in enumerate(cat_pairs):
            for col, (vol_col, gs_label, sig_key) in enumerate(
                [(gw_col, 'GW', gw_key), (sw_col, 'SW', sw_key)],
            ):
                ax = axes[row, col]
                sigma_cat = sigma_lookup.get(sig_key, {})
                for sc in all_dcp:
                    sdf = statewide_df[
                        statewide_df.Scenario == sc
                    ].sort_values('Year')
                    if sdf.empty:
                        continue
                    if sc == 'Baseline_900kAF':
                        color = '#2C3E50'
                        marker = None
                        lw = 2.0
                    else:
                        color = dcp_colors.get(sc, '#333')
                        marker = dcp_markers.get(sc)
                        lw = 1.6
                    lbl = sc.replace('_', ' ')
                    values = sdf[vol_col].values / 1e6
                    _ribbon(ax, sdf['Year'].values, values, sigma_cat, color)
                    ax.plot(
                        sdf['Year'], values,
                        label=lbl, color=color, linewidth=lw,
                        marker=marker, markevery=10, markersize=5,
                    )
                ax.set_ylabel('Volume (MAF)', fontweight='bold')
                ax.set_title(f'{label} {gs_label}', fontweight='bold')
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc='upper left')
        axes[-1, 0].set_xlabel('Year', fontweight='bold')
        axes[-1, 1].set_xlabel('Year', fontweight='bold')
        fig.suptitle(
            'DCP Shortage Tier Scenarios — GW/SW Partition Shift',
            fontweight='bold', fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(
            os.path.join(out_dir, 'CAP_Scenario_DCP_Tiers.png'), dpi=300,
        )
        plt.close(fig)

    # --- Per-basin plot (WestWater scenarios, CAP-affected basins) ---
    # Auto-discover the basins that actually receive CAP-driven ΔGW
    # under any non-baseline scenario.  The previous hardcoded list
    # (Phoenix / Tucson / Pinal AMA + Harquahala / Ranegras) missed
    # several real CAP-intersected basins (Gila Bend, McMullen Valley,
    # Lower Gila, Upper San Pedro, Verde River, Safford) and
    # included Ranegras Plain which has Δ = 0 (no CAP delivery).
    # Pixels outside the CAP service-area mask have ΔGW ≡ 0 by
    # construction (see apply_cap_delivery_perturbation), so any
    # basin appearing here is genuinely CAP-affected.  Threshold of
    # 1 AF lifetime cumulative |Δ| filters out floating-point noise
    # without requiring a magnitude-based gate.
    cap_basins: list[str] = []
    if not delta_df.empty:
        impact = (
            delta_df.assign(_abs=delta_df['Delta_GW_AF'].abs())
            .groupby('Basin')['_abs']
            .sum()
        )
        cap_basins = (
            impact[impact > 1.0]
            .sort_values(ascending=False)
            .index.tolist()
        )
    if not delta_df.empty:
        avail = [b for b in cap_basins if b in delta_df['Basin'].unique()]
        if avail:
            n = len(avail)
            fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
            if n == 1:
                axes = [axes]
            for ax, basin in zip(axes, avail):
                for sc in westwater[1:]:
                    bdf = delta_df[
                        (delta_df.Scenario == sc) & (delta_df.Basin == basin)
                    ].sort_values('Year')
                    if bdf.empty:
                        continue
                    color = scenario_colors.get(sc, '#333')
                    lbl = sc.replace('_', ' ')
                    ax.plot(
                        bdf['Year'], bdf['Delta_GW_AF'] / 1e3,
                        label=lbl, color=color, linewidth=1.6,
                    )
                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                ax.set_ylabel('ΔGW (kAF)', fontweight='bold')
                ax.set_title(basin, fontweight='bold')
                if ax is axes[0]:
                    ax.legend(fontsize=8)
            axes[-1].set_xlabel('Year', fontweight='bold')
            fig.suptitle(
                'Additional GW Pumping Under CAP Reduction (vs Baseline)',
                fontweight='bold', fontsize=14,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(
                os.path.join(out_dir, 'CAP_Scenario_Basin.png'), dpi=300,
            )
            plt.close(fig)

    # --- Cumulative drawdown plot ---
    cumul_path = os.path.join(out_dir, 'CAP_Scenario_Cumulative.csv')
    if os.path.isfile(cumul_path):
        cumul_df = pd.read_csv(cumul_path)
        # Statewide cumulative = sum across all basins
        state_cumul = cumul_df.groupby(
            ['Year', 'Scenario'], as_index=False,
        )['Cumulative_Delta_GW_AF'].sum()

        fig, ax = plt.subplots(figsize=(12, 6))
        # σ ribbons are intentionally suppressed here — cumulative σ
        # values overlap heavily across scenarios (since each scenario
        # uses the same per-year σ_Total_GW from the central pipeline,
        # accumulated linearly), which obscured the inter-scenario
        # separation that is the actual signal.  See module docstring
        # for the reintroduction recipe if needed.
        # WestWater-named scenarios (Basic_Coordination, Extreme_Shortage)
        # often produce drawdowns nearly identical to mid-band DCP Tier
        # scenarios because they share boost-factor era mappings.  Use
        # dashed lines for the WestWater scenarios and solid for DCP
        # Tiers so visual overlap is distinguishable.  Plot solid lines
        # first, then dashed on top (otherwise the dashed Basic
        # Coordination line gets covered by the solid Tier 2a/2b lines
        # they overlap with around ~27 MAF).
        westwater_scenarios = (
            'Baseline_900kAF',
            'Basic_Coordination_237kAF',
            'Extreme_Shortage_0kAF',
        )
        sc_order = sorted(
            state_cumul['Scenario'].unique(),
            key=lambda s: 1 if s in westwater_scenarios else 0,
        )
        for sc in sc_order:
            sdf = state_cumul[
                state_cumul.Scenario == sc
            ].sort_values('Year')
            if sc == 'Baseline_900kAF':
                color = '#2C3E50'
                marker = None
                lw = 2.0
            elif sc in dcp_colors:
                color = dcp_colors[sc]
                marker = dcp_markers.get(sc)
                lw = 1.6
            else:
                color = scenario_colors.get(sc, '#333')
                marker = None
                lw = 1.8
            linestyle = '--' if sc in westwater_scenarios else '-'
            lbl = sc.replace('_', ' ')
            years = sdf['Year'].values
            values = sdf['Cumulative_Delta_GW_AF'].values / 1e6
            zorder = 5 if sc in westwater_scenarios else 3
            ax.plot(
                years, values,
                label=lbl, color=color, linewidth=lw,
                linestyle=linestyle,
                marker=marker, markevery=10, markersize=5,
                zorder=zorder,
            )
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Cumulative Additional GW (MAF)', fontweight='bold')
        ax.set_title(
            'Cumulative Additional Groundwater Drawdown Under CAP '
            'Reduction',
            fontweight='bold',
        )
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, 'CAP_Scenario_Cumulative_Drawdown.png'),
            dpi=300,
        )
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CAP scenario spatial drawdown maps (basin and pixel level)
# ──────────────────────────────────────────────────────────────────────


def _clip_basins_to_cap(
        basins_gdf: 'gpd.GeoDataFrame',
        cap_service_area_geojson: 'str | None',
) -> 'gpd.GeoDataFrame | None':
    """Intersect basin polygons with the CAP service area.

    Returns a GDF where each basin's geometry is replaced by its
    intersection with the CAP service area union.  Basins that don't
    intersect the CAP footprint are dropped (their cumulative ΔGW
    is zero by construction).  Returns ``None`` if the geojson is
    missing or unreadable so callers can fall back to rendering full
    basin polygons.
    """
    if not cap_service_area_geojson or not os.path.isfile(
        cap_service_area_geojson,
    ):
        return None
    try:
        cap_gdf = gpd.read_file(cap_service_area_geojson)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            'CAP scenario clip: cannot read %s: %s',
            cap_service_area_geojson, exc,
        )
        return None
    if cap_gdf.empty:
        return None
    if cap_gdf.crs != basins_gdf.crs:
        cap_gdf = cap_gdf.to_crs(basins_gdf.crs)
    cap_union = cap_gdf.geometry.unary_union
    clipped = basins_gdf.copy()
    clipped['geometry'] = clipped.geometry.apply(
        lambda g: g.intersection(cap_union) if g.intersects(cap_union)
        else None,
    )
    clipped = clipped[
        clipped.geometry.notna() & ~clipped.geometry.is_empty
    ]
    return clipped

# Display order for scenarios (mild → severe shortage), with the
# reference Baseline excluded since ΔGW vs Baseline is by construction
# zero everywhere.
_CAP_SCENARIO_PANEL_ORDER: tuple[str, ...] = (
    'DCP_Tier0_192kAF_cut',
    'DCP_Tier1_512kAF_cut',
    'DCP_Tier2a_592kAF_cut',
    'DCP_Tier2b_640kAF_cut',
    'DCP_Tier3_720kAF_cut',
    'Basic_Coordination_237kAF',
    'Extreme_Shortage_0kAF',
)

# Pretty labels for figure panel titles.
_CAP_SCENARIO_PANEL_TITLES: dict[str, str] = {
    'DCP_Tier0_192kAF_cut': 'DCP Tier 0 (−192 kAF cut)',
    'DCP_Tier1_512kAF_cut': 'DCP Tier 1 (−512 kAF cut)',
    'DCP_Tier2a_592kAF_cut': 'DCP Tier 2a (−592 kAF cut)',
    'DCP_Tier2b_640kAF_cut': 'DCP Tier 2b (−640 kAF cut)',
    'DCP_Tier3_720kAF_cut': 'DCP Tier 3 (−720 kAF cut)',
    'Basic_Coordination_237kAF': 'WestWater Basic Coordination',
    'Extreme_Shortage_0kAF': 'WestWater Extreme Shortage',
}


def _plot_cap_scenario_basin_drawdown(
        delta_df: 'pd.DataFrame',
        basin_shp: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> None:
    """Multi-panel basin choropleth of cumulative ΔGW per CAP scenario.

    Renders a 2×4 grid (7 scenario panels + 1 legend cell) showing
    per-basin cumulative additional GW pumping volume vs the
    Baseline_900kAF scenario, summed over *year_window* (default
    2027-2060 to match the WestWater 2026 comparison window).

    Volume space (10⁶ m³ on the primary colorbar, AF on a secondary
    axis) — no hydraulic-head modelling is implied.  Discrete YlOrRd
    bins make scenario-to-scenario differences readable; shared
    colorbar across all 7 panels keeps the visual scale comparable.

    Args:
        delta_df: ``CAP_Scenario_Delta.csv`` as a DataFrame; must have
            columns ``Year``, ``Scenario``, ``Basin``, ``Delta_GW_AF``.
        basin_shp: Path to the AZ groundwater basin shapefile.
        out_dir: Directory for the output PNG.
        year_window: Inclusive ``(start, end)`` year range for the
            cumulative sum.  Default 2027-2060 (matches WestWater).
        basin_col: Basin name column in *basin_shp*.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names,
        add_ama_ina_legend, apply_journal_style,
        overlay_cap_service_area,
    )

    apply_journal_style()
    if delta_df is None or delta_df.empty:
        logger.info('  CAP scenario basin drawdown: empty delta_df, skipping')
        return

    # Cumulative ΔGW per (scenario, basin) over the requested window
    sub = delta_df[
        (delta_df['Year'] >= year_window[0])
        & (delta_df['Year'] <= year_window[1])
    ]
    cum = (
        sub.groupby(['Scenario', 'Basin'])['Delta_GW_AF']
        .sum()
        .reset_index()
    )
    if cum.empty:
        logger.info(
            '  CAP scenario basin drawdown: no data in window %s, '
            'skipping', year_window,
        )
        return
    af_to_m3 = 1.0 / M3_TO_AF
    cum['Delta_GW_m3'] = cum['Delta_GW_AF'] * af_to_m3

    # Load basins, project to a metric CRS only if needed
    basins_gdf = gpd.read_file(basin_shp)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    # Clip basin polygons to the CAP service area so basins like
    # Verde River / McMullen Valley / Harquahala render only the
    # actually-affected sliver (a small CAP-pixel intersection)
    # instead of the entire basin polygon — the basin's cumulative
    # ΔGW is sourced from those CAP pixels only, so the colored
    # area must visually match.
    basins_clipped = _clip_basins_to_cap(
        basins_gdf, cap_service_area_geojson,
    )
    plot_polys = (
        basins_clipped if basins_clipped is not None else basins_gdf
    )

    # Discrete diverging-style bins in 10⁶ m³.  Most basins under most
    # scenarios fall in 0-2000 × 10⁶ m³ cumulative ΔGW; the Phoenix
    # AMA Extreme Shortage hits ~3400 × 10⁶ m³.  Bins chosen to keep
    # rural near-zero basins distinguishable from mid-range and to
    # show the Phoenix saturation at the top.
    boundaries_m3_million = [0, 5, 25, 100, 500, 1000, 2000, 4000]
    boundaries_m3 = [b * 1e6 for b in boundaries_m3_million]
    n_levels = len(boundaries_m3) - 1
    palette = cm.get_cmap('YlOrRd', n_levels)
    discrete_cmap = ListedColormap([palette(i) for i in range(n_levels)])
    discrete_cmap.set_under('#FFFFFF')  # truly-zero basins render white
    norm = BoundaryNorm(boundaries_m3, discrete_cmap.N, clip=False)

    # 2 × 4 grid: 7 scenario panels + 1 legend cell
    fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(
        f'CAP Scenario — Cumulative Additional GW Volume vs Baseline '
        f'({year_window[0]}–{year_window[1]})',
        fontsize=16, fontweight='bold',
    )
    axes_flat = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    for i, sc_key in enumerate(_CAP_SCENARIO_PANEL_ORDER):
        ax = axes_flat[i]
        ax.set_facecolor('#EEEEEE')
        sc_data = cum[cum['Scenario'] == sc_key]
        if sc_data.empty:
            ax.text(
                0.5, 0.5, f'No data for {sc_key}',
                ha='center', va='center', transform=ax.transAxes,
            )
            ax.axis('off')
            continue
        # Map basin → cumulative Δ in m³
        sc_lookup = dict(zip(sc_data['Basin'], sc_data['Delta_GW_m3']))
        plot_gdf = plot_polys.set_index(name_col).copy()
        plot_gdf['delta_m3'] = plot_gdf.index.map(
            lambda b: sc_lookup.get(b, np.nan),
        )
        # Treat true zeros (basins outside CAP service area) as
        # missing so they render white via missing_kwds rather than
        # the lightest YlOrRd bin (BoundaryNorm puts a value exactly
        # at the lowest boundary into bin 0, not the under-color).
        plot_gdf.loc[
            plot_gdf['delta_m3'].fillna(-1).abs() < 1.0, 'delta_m3'
        ] = np.nan
        plot_gdf.plot(
            ax=ax, column='delta_m3', cmap=discrete_cmap, norm=norm,
            edgecolor='none', linewidth=0,
            missing_kwds={'color': '#FFFFFF', 'edgecolor': 'none',
                          'linewidth': 0},
        )
        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col,
            label_fontsize=4.5, label_all=True,
        )
        overlay_cap_service_area(
            ax, cap_service_area_geojson,
            target_crs=basins_gdf.crs,
        )
        title_pretty = _CAP_SCENARIO_PANEL_TITLES.get(
            sc_key, sc_key.replace('_', ' '),
        )
        # AZ-wide cumulative for the panel title context
        az_cum_m3 = sc_data['Delta_GW_m3'].sum()
        ax.set_title(
            f'{panel_labels[i]} {title_pretty}\n'
            f'AZ total: {az_cum_m3 / 1e9:.2f} km³ '
            f'({az_cum_m3 * M3_TO_AF / 1e6:.2f} MAF)',
            fontsize=11, fontweight='bold',
        )

    # Full-height shared colorbar on the right of the entire grid;
    # AMA/INA legend in the row-2-col-4 cell (axes_flat[7]) enlarged.
    ax_legend = axes_flat[7]
    ax_legend.axis('off')
    sm = ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes_flat[:7], orientation='vertical',
        shrink=1.0, pad=0.04, fraction=0.04, aspect=40,
        boundaries=boundaries_m3, ticks=boundaries_m3, extend='both',
    )
    cbar.formatter = mticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:,.0f}',
    )
    cbar.update_ticks()
    cbar.set_label(
        r'Cumulative $\Delta$ GW Volume ($\times$10$^{6}$ m$^{3}$)',
        fontsize=12, fontweight='bold',
    )
    cbar.ax.tick_params(labelsize=11)
    # AF / kAF axis on the LEFT (primary m³ on the right is default).
    secax = cbar.ax.secondary_yaxis(
        'left',
        functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
    )
    secax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e3:,.0f}'),
    )
    secax.set_ylabel(
        'Cumulative Δ GW Volume (kAF)',
        fontsize=12, fontweight='bold',
    )
    secax.tick_params(labelsize=11)
    add_ama_ina_legend(
        ax_legend, loc='center', bbox_to_anchor=(0.5, 0.5),
        fontsize=14, framealpha=1.0, include_cap=True,
    )

    out_path = os.path.join(
        out_dir,
        f'CAP_Scenario_Basin_Drawdown_'
        f'{year_window[0]}_{year_window[1]}.png',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('  CAP scenario basin drawdown map saved to %s', out_path)


def _generate_cap_scenario_pixel_rasters(
        delta_df: 'pd.DataFrame',
        basin_shp: str,
        total_gw_dir: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> dict[str, str]:
    """Distribute basin-level ΔGW to pixels via ML Total_GW share.

    For each (scenario, year) the basin-level ΔGW is partitioned to
    pixels in proportion to that pixel's share of the basin's
    ML-predicted Total_GW.  Pixels are accumulated over *year_window*
    to produce one cumulative-ΔGW raster per scenario, written to
    ``{out_dir}/Pixel_Rasters/CAP_Scenario_Pixel_{scenario}_cum_AF.tif``.

    The pro-rata distribution is an assumption — the basin total is
    well-constrained by the partition, but its sub-basin spatial
    pattern follows the central pipeline's per-pixel demand prediction.
    Reviewers should not interpret pixel-level texture as a
    hydraulic-head response.

    Args:
        delta_df: ``CAP_Scenario_Delta.csv`` DataFrame.
        basin_shp: AZ groundwater basin shapefile.
        total_gw_dir: Directory with ``Total_GW_{year}_mm.tif`` rasters
            (mean annual depth in mm).
        out_dir: Directory under which ``Pixel_Rasters/`` will be created.
        year_window: Inclusive cumulative window.
        basin_col: Basin name column.

    Returns:
        ``{scenario_key: cumulative_raster_path}`` for downstream
        rendering.  Empty dict if the source rasters or delta data
        are unavailable.
    """
    from rasterio.features import geometry_mask
    from hydrolibs.sysops import makedirs
    if delta_df is None or delta_df.empty:
        return {}

    pixel_dir = os.path.join(out_dir, 'Pixel_Rasters')
    makedirs(pixel_dir)

    # Find a reference raster to define grid / CRS
    start_yr, end_yr = year_window
    ref_raster = None
    for yr in range(start_yr, end_yr + 1):
        cand = os.path.join(total_gw_dir, f'Total_GW_{yr}_mm.tif')
        if os.path.isfile(cand):
            ref_raster = cand
            break
    if ref_raster is None:
        logger.warning(
            '  CAP scenario pixel rasters: no Total_GW rasters found in '
            '%s for window %s — skipping', total_gw_dir, year_window,
        )
        return {}

    with rio.open(ref_raster) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_transform = ref_src.transform
        ref_shape = ref_src.shape
        ref_crs = ref_src.crs
        pixel_area_m2 = abs(ref_transform.a * ref_transform.e)

    # Pre-build CAP service area mask (when geojson provided) so basin
    # masks below get intersected to CAP pixels only.  Without this
    # intersection the pro-rata distribution of basin Δ would spread
    # across the full basin polygon, including pixels OUTSIDE the CAP
    # service area — those pixels never see the partition perturbation,
    # so giving them non-zero ΔGW is incorrect (they're a structural
    # zero per ``apply_cap_delivery_perturbation``).
    cap_mask: 'np.ndarray | None' = None
    if cap_service_area_geojson and os.path.isfile(
        cap_service_area_geojson,
    ):
        cap_gdf = gpd.read_file(cap_service_area_geojson)
        if not cap_gdf.empty:
            if cap_gdf.crs != ref_crs:
                cap_gdf = cap_gdf.to_crs(ref_crs)
            cap_mask = geometry_mask(
                list(cap_gdf.geometry), transform=ref_transform,
                invert=True, out_shape=ref_shape,
            )
            if not cap_mask.any():
                cap_mask = None
    if cap_mask is None:
        logger.warning(
            '  CAP scenario pixel rasters: CAP service-area mask not '
            'available — pixel ΔGW will distribute across full basin '
            'polygons (non-CAP pixels will receive non-zero share).'
        )

    # Pre-build basin geometry masks once, intersected with CAP mask
    # when available so the pro-rata distribution stays inside the
    # CAP-affected portion of each basin.
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != ref_crs:
        basins_gdf = basins_gdf.to_crs(ref_crs)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    basin_masks: dict[str, np.ndarray] = {}
    for _, row in basins_gdf.iterrows():
        mask = geometry_mask(
            [row.geometry], transform=ref_transform,
            invert=True, out_shape=ref_shape,
        )
        if cap_mask is not None:
            mask = mask & cap_mask
            if not mask.any():
                continue
        basin_masks[row[name_col]] = mask

    # Pre-load per-year Total_GW rasters in mm and convert to AF per pixel
    # (depth_mm × pixel_area_m² × 1e-3 m / 1233.48 m³/AF)
    af_per_m3 = M3_TO_AF
    mm_to_af_factor = pixel_area_m2 * 1e-3 * af_per_m3
    yearly_gw_af: dict[int, np.ndarray] = {}
    for yr in range(start_yr, end_yr + 1):
        rpath = os.path.join(total_gw_dir, f'Total_GW_{yr}_mm.tif')
        if not os.path.isfile(rpath):
            continue
        with rio.open(rpath) as src:
            arr = src.read(1).astype(np.float64)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        yearly_gw_af[yr] = arr * mm_to_af_factor

    if not yearly_gw_af:
        logger.warning(
            '  CAP scenario pixel rasters: no usable Total_GW rasters '
            'in %s', total_gw_dir,
        )
        return {}

    # Pre-compute basin totals per year for share denominator
    basin_totals: dict[int, dict[str, float]] = {}
    for yr, arr_af in yearly_gw_af.items():
        basin_totals[yr] = {
            b: float(arr_af[m].sum()) for b, m in basin_masks.items()
        }

    # Index delta CSV for fast lookup
    delta_lookup = (
        delta_df.set_index(['Scenario', 'Basin', 'Year'])['Delta_GW_AF']
        .to_dict()
    )

    out_paths: dict[str, str] = {}
    for sc_key in _CAP_SCENARIO_PANEL_ORDER:
        cumulative_af = np.zeros(ref_shape, dtype=np.float64)
        for yr in range(start_yr, end_yr + 1):
            arr_af = yearly_gw_af.get(yr)
            if arr_af is None:
                continue
            for basin, mask in basin_masks.items():
                delta_af = delta_lookup.get((sc_key, basin, yr), 0.0)
                if delta_af == 0:
                    continue
                basin_total = basin_totals[yr].get(basin, 0.0)
                if basin_total <= 0:
                    continue
                # Distribute basin Δ to its pixels in proportion to
                # the per-pixel ML Total_GW share within the basin.
                scale = delta_af / basin_total
                cumulative_af[mask] += arr_af[mask] * scale

        # Save cumulative raster
        out_path = os.path.join(
            pixel_dir,
            f'CAP_Scenario_Pixel_{sc_key}_cum_AF_'
            f'{year_window[0]}_{year_window[1]}.tif',
        )
        prof = ref_profile.copy()
        prof.update(dtype='float32', nodata=np.nan, count=1)
        out_arr = cumulative_af.astype(np.float32)
        out_arr[out_arr == 0] = np.nan
        with rio.open(out_path, 'w', **prof) as dst:
            dst.write(out_arr, 1)
        out_paths[sc_key] = out_path
        logger.info(
            '  CAP scenario pixel raster %s: %s '
            '(AZ-wide cumulative = %.2f MAF)',
            sc_key, out_path,
            np.nansum(cumulative_af) / 1e6,
        )

    return out_paths


def _plot_cap_scenario_pixel_drawdown(
        scenario_raster_paths: dict[str, str],
        basin_shp: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> None:
    """Multi-panel pixel-level cumulative ΔGW drawdown maps.

    Mirrors :func:`_plot_cap_scenario_basin_drawdown` but uses the
    pixel-level cumulative-ΔGW rasters generated by
    :func:`_generate_cap_scenario_pixel_rasters`.  Renders one panel
    per scenario as a continuous imshow with the same discrete bins
    so the basin choropleth and pixel maps are visually comparable.

    Pro-rata distribution caveat (basin Δ × per-pixel demand share)
    is documented in the suptitle to keep reviewers from over-
    interpreting sub-basin texture as a hydraulic-head field.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names,
        add_ama_ina_legend, apply_journal_style,
        overlay_cap_service_area,
    )

    apply_journal_style()
    if not scenario_raster_paths:
        return

    # Bin scheme in m³ per pixel (cumulative over the year window).
    # At a 2 km pixel (4×10⁶ m²) and Phoenix-AMA-equivalent peak Δ ≈
    # 100 mm/yr × 34 years = 3.4 m → 13.6 × 10⁶ m³ per pixel.
    boundaries_m3_million = [0, 0.1, 0.5, 1, 2, 5, 10, 20]
    boundaries_m3 = [b * 1e6 for b in boundaries_m3_million]
    n_levels = len(boundaries_m3) - 1
    palette = cm.get_cmap('YlOrRd', n_levels)
    discrete_cmap = ListedColormap([palette(i) for i in range(n_levels)])
    discrete_cmap.set_under('#FFFFFF')
    norm = BoundaryNorm(boundaries_m3, discrete_cmap.N, clip=False)

    # Read template raster (any scenario) for extent
    template = next(iter(scenario_raster_paths.values()))
    with rio.open(template) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != ref_crs:
        basins_gdf = basins_gdf.to_crs(ref_crs)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(
        f'CAP Scenario — Pixel-Level Cumulative ΔGW Volume vs '
        f'Baseline ({year_window[0]}–{year_window[1]})\n'
        f'(per-pixel volume = basin Δ × pixel ML Total_GW share — '
        f'pro-rata, NOT a hydraulic-head response)',
        fontsize=14, fontweight='bold',
    )
    axes_flat = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    af_to_m3 = 1.0 / M3_TO_AF
    for i, sc_key in enumerate(_CAP_SCENARIO_PANEL_ORDER):
        ax = axes_flat[i]
        ax.set_facecolor('#EEEEEE')
        rpath = scenario_raster_paths.get(sc_key)
        if rpath is None or not os.path.isfile(rpath):
            ax.text(
                0.5, 0.5, f'No raster for {sc_key}',
                ha='center', va='center', transform=ax.transAxes,
            )
            ax.axis('off')
            continue
        with rio.open(rpath) as src:
            arr_af = src.read(1).astype(np.float64)
        arr_m3 = np.where(np.isfinite(arr_af) & (arr_af > 0),
                          arr_af * af_to_m3, np.nan)
        masked = np.ma.masked_invalid(arr_m3)
        ax.imshow(
            masked, extent=extent, origin='upper',
            cmap=discrete_cmap, norm=norm, interpolation='nearest',
        )
        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col,
            label_fontsize=4.5, label_all=True,
        )
        overlay_cap_service_area(
            ax, cap_service_area_geojson,
            target_crs=ref_crs,
        )
        title_pretty = _CAP_SCENARIO_PANEL_TITLES.get(
            sc_key, sc_key.replace('_', ' '),
        )
        az_cum_af = np.nansum(arr_af)
        ax.set_title(
            f'{panel_labels[i]} {title_pretty}\n'
            f'AZ total: {az_cum_af * af_to_m3 / 1e9:.2f} km³ '
            f'({az_cum_af / 1e6:.2f} MAF)',
            fontsize=11, fontweight='bold',
        )

    ax_legend = axes_flat[7]
    ax_legend.axis('off')
    sm = ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes_flat[:7], orientation='vertical',
        shrink=1.0, pad=0.04, fraction=0.04, aspect=40,
        boundaries=boundaries_m3, ticks=boundaries_m3, extend='both',
    )
    cbar.formatter = mticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:,.1f}',
    )
    cbar.update_ticks()
    cbar.set_label(
        r'Per-Pixel Cumulative $\Delta$ GW Volume '
        r'($\times$10$^{6}$ m$^{3}$)',
        fontsize=12, fontweight='bold',
    )
    cbar.ax.tick_params(labelsize=11)
    # AF axis on the LEFT — primary m³ on the right (default).
    secax = cbar.ax.secondary_yaxis(
        'left',
        functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
    )
    secax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'),
    )
    secax.set_ylabel(
        'Per-Pixel Cumulative Δ GW Volume (AF)',
        fontsize=12, fontweight='bold',
    )
    secax.tick_params(labelsize=11)
    add_ama_ina_legend(
        ax_legend, loc='center', bbox_to_anchor=(0.5, 0.5),
        fontsize=14, framealpha=1.0, include_cap=True,
    )

    out_path = os.path.join(
        out_dir,
        f'CAP_Scenario_Pixel_Drawdown_'
        f'{year_window[0]}_{year_window[1]}.png',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('  CAP scenario pixel drawdown map saved to %s', out_path)


# ──────────────────────────────────────────────────────────────────────
# CAP scenario σ-cumulative + signal-to-noise (SNR) maps
# ──────────────────────────────────────────────────────────────────────

# Per-component independent uncertainty sources whose per-basin
# Total_GW Sigma_Volume_AF is quadrature-combined to a per-(basin,year)
# σ_total before linear-time accumulation across the projection window.
# CU is included for parity with the CU-uncertainty extension that
# may exist on some prediction roots; missing components are silently
# skipped at load time.
_BASIN_SIGMA_TOTAL_GW_COMPONENTS: tuple[str, ...] = (
    'MACA', 'Model', 'Irr', 'LULC', 'GW', 'USBR', 'CU',
)


def _load_basin_sigma_total_gw_yearly(
        unc_dir: str,
        year_window: tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Per-basin per-year σ_total_GW (AF) over a year window.

    Reads ``Basin_Sigma_<comp>_Total_GW.csv`` for each component under
    *unc_dir* and combines the per-basin per-year σ in quadrature
    (components are independent ensemble axes by design).

    Returns ``{basin_name: {year: sigma_AF}}``.
    """
    var_per_basin_yr: dict[str, dict[int, float]] = {}
    for comp in _BASIN_SIGMA_TOTAL_GW_COMPONENTS:
        csv_path = os.path.join(
            unc_dir, f'Sigma_{comp}',
            f'Basin_Sigma_{comp}_Total_GW.csv',
        )
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            continue
        if not {'Year', 'Region', 'Sigma_Volume_AF'}.issubset(df.columns):
            continue
        df = df.dropna(subset=['Year', 'Region', 'Sigma_Volume_AF'])
        df = df[df.Year.between(year_window[0], year_window[1])]
        for _, row in df.iterrows():
            b = row['Region']
            yr = int(row['Year'])
            s = float(row['Sigma_Volume_AF'])
            if not np.isfinite(s):
                continue
            var_per_basin_yr.setdefault(b, {})[yr] = (
                var_per_basin_yr.get(b, {}).get(yr, 0.0) + s ** 2
            )
    return {
        b: {yr: float(np.sqrt(v)) for yr, v in d.items()}
        for b, d in var_per_basin_yr.items()
    }


def _compute_basin_sigma_cum(
        basin_sigma_yearly: dict[str, dict[int, float]],
        year_window: tuple[int, int],
) -> dict[str, float]:
    """Cumulative basin σ over *year_window* via linear time-sum
    (perfect year-to-year correlation — conservative upper bound).
    """
    out: dict[str, float] = {}
    for b, yr_dict in basin_sigma_yearly.items():
        s = 0.0
        for yr in range(year_window[0], year_window[1] + 1):
            v = yr_dict.get(yr)
            if v is not None and np.isfinite(v):
                s += v
        if s > 0:
            out[b] = s
    return out


def _save_cap_restricted_basin_sigma_csv(
        sigma_raster_dir: str,
        basin_shp: str,
        cap_service_area_geojson: str,
        year_range: tuple[int, int],
        out_csv: str,
        *,
        basin_col: str = 'BASIN_NAME',
) -> 'pd.DataFrame | None':
    """Per-year per-basin σ_total_GW (AF) restricted to CAP-pixel
    intersection — saved as a permanent UQ artifact alongside the
    other ``CAP_Scenario_*.csv`` files.

    Each row reports the σ aggregated over the basin × CAP-service-
    area intersection pixels for that year, plus the spatial pixel
    count for transparency.  Downstream analyses (cumulative σ over
    arbitrary windows, time-series plots, σ trajectories at CAP-
    affected basins) can read this CSV directly without re-running
    the per-pixel raster aggregation.

    Columns: ``Year, Region, Sigma_Volume_AF, Sigma_Volume_m3,
    N_CAP_Pixels``.

    Returns the DataFrame written, or None if the inputs aren't
    available.
    """
    from rasterio.features import geometry_mask
    if (
        not os.path.isdir(sigma_raster_dir)
        or not os.path.isfile(cap_service_area_geojson)
    ):
        return None
    start_yr, end_yr = year_range
    ref_raster = None
    for yr in range(start_yr, end_yr + 1):
        cand = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if os.path.isfile(cand):
            ref_raster = cand
            break
    if ref_raster is None:
        return None

    with rio.open(ref_raster) as src:
        ref_transform = src.transform
        ref_shape = src.shape
        ref_crs = src.crs
        pixel_area_m2 = abs(ref_transform.a * ref_transform.e)

    cap_gdf = gpd.read_file(cap_service_area_geojson)
    if cap_gdf.empty:
        return None
    if cap_gdf.crs != ref_crs:
        cap_gdf = cap_gdf.to_crs(ref_crs)
    cap_mask = geometry_mask(
        list(cap_gdf.geometry), transform=ref_transform,
        invert=True, out_shape=ref_shape,
    )
    if not cap_mask.any():
        return None

    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != ref_crs:
        basins_gdf = basins_gdf.to_crs(ref_crs)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    basin_cap_masks: dict[str, np.ndarray] = {}
    basin_cap_pixels: dict[str, int] = {}
    for _, row in basins_gdf.iterrows():
        bm = geometry_mask(
            [row.geometry], transform=ref_transform,
            invert=True, out_shape=ref_shape,
        )
        intersect = bm & cap_mask
        n_pix = int(intersect.sum())
        if n_pix > 0:
            basin_cap_masks[row[name_col]] = intersect
            basin_cap_pixels[row[name_col]] = n_pix

    af_to_m3 = 1.0 / M3_TO_AF
    mm_to_af = pixel_area_m2 * 1e-3 * M3_TO_AF
    rows = []
    for yr in range(start_yr, end_yr + 1):
        path = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if not os.path.isfile(path):
            continue
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float64)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        arr_af = arr * mm_to_af
        for b, mask in basin_cap_masks.items():
            sigma_af = float(arr_af[mask].sum())
            if sigma_af <= 0:
                continue
            rows.append({
                'Year': yr,
                'Region': b,
                'Sigma_Volume_AF': round(sigma_af, 2),
                'Sigma_Volume_m3': round(sigma_af * af_to_m3, 2),
                'N_CAP_Pixels': basin_cap_pixels[b],
            })
    if not rows:
        return None
    df_out = pd.DataFrame(rows)
    df_out.sort_values(['Year', 'Region'], inplace=True)
    df_out.to_csv(out_csv, index=False)
    logger.info(
        '  CAP-restricted basin σ CSV saved (%d rows, %d basins) to %s',
        len(df_out), df_out['Region'].nunique(), out_csv,
    )
    return df_out


def _compute_basin_sigma_cum_cap_restricted(
        sigma_raster_dir: str,
        basin_shp: str,
        cap_service_area_geojson: str,
        year_window: tuple[int, int],
        *,
        basin_col: str = 'BASIN_NAME',
) -> dict[str, float]:
    """Per-basin cumulative σ_total_GW (AF) over *year_window*,
    aggregated over **only** the basin × CAP-service-area intersection
    pixels (not the full basin).

    For each year:
      - Read ``Sigma_Total_Total_GW_mm_{year}.tif``
      - Multiply by pixel area to get per-pixel σ in AF
      - Mask to the CAP service area
      - For each basin, linear-time-sum aggregate across years and
        spatial linear-sum within the basin × CAP intersection
        (perfect-correlation upper bound, consistent with the
        AZ-wide cumulative method).

    Used as the denominator for the basin CV map so the noise floor
    at basins with small CAP footprints (Verde River, Harquahala
    INA, McMullen Valley) reflects only the CAP-affected portion
    rather than the full basin's σ_total.

    Returns ``{basin_name: sigma_AF}`` for basins with non-zero
    intersection and σ data, else empty dict.
    """
    from rasterio.features import geometry_mask
    if (
        not os.path.isdir(sigma_raster_dir)
        or not os.path.isfile(cap_service_area_geojson)
    ):
        return {}
    start_yr, end_yr = year_window
    # Locate a reference σ raster
    ref_raster = None
    for yr in range(start_yr, end_yr + 1):
        cand = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if os.path.isfile(cand):
            ref_raster = cand
            break
    if ref_raster is None:
        return {}

    with rio.open(ref_raster) as src:
        ref_transform = src.transform
        ref_shape = src.shape
        ref_crs = src.crs
        pixel_area_m2 = abs(ref_transform.a * ref_transform.e)

    # CAP service area mask
    cap_gdf = gpd.read_file(cap_service_area_geojson)
    if cap_gdf.empty:
        return {}
    if cap_gdf.crs != ref_crs:
        cap_gdf = cap_gdf.to_crs(ref_crs)
    cap_mask = geometry_mask(
        list(cap_gdf.geometry), transform=ref_transform,
        invert=True, out_shape=ref_shape,
    )
    if not cap_mask.any():
        return {}

    # Basin × CAP intersection masks (one per basin, restricted
    # to CAP pixels)
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != ref_crs:
        basins_gdf = basins_gdf.to_crs(ref_crs)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    basin_cap_masks: dict[str, np.ndarray] = {}
    for _, row in basins_gdf.iterrows():
        bm = geometry_mask(
            [row.geometry], transform=ref_transform,
            invert=True, out_shape=ref_shape,
        )
        intersect = bm & cap_mask
        if intersect.any():
            basin_cap_masks[row[name_col]] = intersect
    if not basin_cap_masks:
        return {}

    mm_to_af = pixel_area_m2 * 1e-3 * M3_TO_AF
    cum_per_basin: dict[str, float] = {b: 0.0 for b in basin_cap_masks}
    for yr in range(start_yr, end_yr + 1):
        path = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if not os.path.isfile(path):
            continue
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float64)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        arr_af = arr * mm_to_af
        for b, mask in basin_cap_masks.items():
            cum_per_basin[b] += float(arr_af[mask].sum())
    return {b: v for b, v in cum_per_basin.items() if v > 0}


def _compute_pixel_sigma_cum(
        sigma_raster_dir: str,
        year_window: tuple[int, int],
        cap_service_area_geojson: 'str | None' = None,
) -> 'tuple[np.ndarray, str] | None':
    """Cumulative pixel σ_total_GW (AF) over *year_window*.

    Reads ``Sigma_Total_Total_GW_mm_{year}.tif`` for each year and
    accumulates in linear time-sum (perfect-correlation upper bound).
    When ``cap_service_area_geojson`` is provided, restricts the
    output to CAP-service-area pixels (sets non-CAP pixels to NaN)
    so the σ_cum context map matches the pixel ΔGW footprint.
    Returns ``(cum_AF, ref_raster_path)`` or ``None`` if no rasters
    are available.
    """
    from rasterio.features import geometry_mask
    if not os.path.isdir(sigma_raster_dir):
        return None
    ref_raster = None
    for yr in range(year_window[0], year_window[1] + 1):
        cand = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if os.path.isfile(cand):
            ref_raster = cand
            break
    if ref_raster is None:
        return None
    with rio.open(ref_raster) as src:
        ref_shape = src.shape
        ref_transform = src.transform
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
    mm_to_af = pixel_area_m2 * 1e-3 * M3_TO_AF

    # Optional CAP mask
    cap_mask: 'np.ndarray | None' = None
    if cap_service_area_geojson and os.path.isfile(
        cap_service_area_geojson,
    ):
        cap_gdf = gpd.read_file(cap_service_area_geojson)
        if not cap_gdf.empty:
            if cap_gdf.crs != ref_crs:
                cap_gdf = cap_gdf.to_crs(ref_crs)
            cap_mask = geometry_mask(
                list(cap_gdf.geometry), transform=ref_transform,
                invert=True, out_shape=ref_shape,
            )
            if not cap_mask.any():
                cap_mask = None

    cum = np.zeros(ref_shape, dtype=np.float64)
    n = 0
    for yr in range(year_window[0], year_window[1] + 1):
        path = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_Total_GW_mm_{yr}.tif',
        )
        if not os.path.isfile(path):
            continue
        with rio.open(path) as src:
            arr = src.read(1).astype(np.float64)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        cum += arr * mm_to_af
        n += 1
    if n == 0:
        return None
    if cap_mask is not None:
        cum = np.where(cap_mask, cum, np.nan)
    return cum, ref_raster


def _plot_cap_scenario_sigma_combined(
        basin_sigma_cum: dict[str, float],
        pixel_sigma_cum: 'np.ndarray | None',
        pixel_ref_raster: 'str | None',
        basin_shp: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> None:
    """Single 2-panel figure showing the cumulative σ_total_GW context
    (basin choropleth | pixel raster) that applies to every CAP scenario.

    Renders the per-pixel and per-basin uncertainty bound that the
    central drawdown maps share.  Same colorbar (10⁶ m³ primary, AF
    secondary) as the central figures so reviewers can compare a
    scenario's signal magnitude against the noise floor at a glance.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names,
        add_ama_ina_legend, apply_journal_style,
        overlay_cap_service_area,
    )

    apply_journal_style()
    if not basin_sigma_cum and pixel_sigma_cum is None:
        return

    af_to_m3 = 1.0 / M3_TO_AF

    # Basin σ in 10⁶ m³ — same bin scheme as the central drawdown so
    # the eye can compare σ to ΔGW directly.
    boundaries_m3_million_basin = [0, 5, 25, 100, 500, 1000, 2000, 4000]
    boundaries_basin = [b * 1e6 for b in boundaries_m3_million_basin]
    n_levels = len(boundaries_basin) - 1
    palette_basin = cm.get_cmap('Purples', n_levels)
    cmap_basin = ListedColormap(
        [palette_basin(i) for i in range(n_levels)],
    )
    cmap_basin.set_under('#FFFFFF')
    norm_basin = BoundaryNorm(boundaries_basin, cmap_basin.N, clip=False)

    # Pixel σ in 10⁶ m³ per pixel — finer bins (smaller scale).
    boundaries_m3_million_pix = [0, 0.1, 0.5, 1, 2, 5, 10, 20]
    boundaries_pix = [b * 1e6 for b in boundaries_m3_million_pix]
    n_levels_pix = len(boundaries_pix) - 1
    palette_pix = cm.get_cmap('Purples', n_levels_pix)
    cmap_pix = ListedColormap(
        [palette_pix(i) for i in range(n_levels_pix)],
    )
    cmap_pix.set_under('#FFFFFF')
    norm_pix = BoundaryNorm(boundaries_pix, cmap_pix.N, clip=False)

    basins_gdf = gpd.read_file(basin_shp)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(
        f'Cumulative σ_total on Total_GW over CAP service area '
        f'({year_window[0]}–{year_window[1]})\n'
        f'Same uncertainty applies to every CAP scenario — quadrature '
        f'across components, linear-time-sum (perfect-correlation '
        f'conservative upper bound).  Basin and pixel panels share the '
        f'same CAP-restricted spatial scope and their AZ totals match.',
        fontsize=13, fontweight='bold',
    )

    # --- Panel (a): basin σ choropleth (clipped to CAP service area
    #     so the colored polygon matches the basin × CAP intersection,
    #     consistent with the central drawdown maps)
    ax_basin = axes[0]
    ax_basin.set_facecolor('#EEEEEE')
    basins_clipped = _clip_basins_to_cap(
        basins_gdf, cap_service_area_geojson,
    )
    plot_polys = (
        basins_clipped if basins_clipped is not None else basins_gdf
    )
    sigma_m3_lookup = {
        b: s * af_to_m3 for b, s in basin_sigma_cum.items()
    }
    plot_gdf = plot_polys.set_index(name_col).copy()
    plot_gdf['sigma_m3'] = plot_gdf.index.map(
        lambda b: sigma_m3_lookup.get(b, np.nan),
    )
    plot_gdf.loc[
        plot_gdf['sigma_m3'].fillna(-1).abs() < 1.0, 'sigma_m3'
    ] = np.nan
    plot_gdf.plot(
        ax=ax_basin, column='sigma_m3', cmap=cmap_basin, norm=norm_basin,
        edgecolor='none', linewidth=0,
        missing_kwds={'color': '#FFFFFF', 'edgecolor': 'none',
                      'linewidth': 0},
    )
    _overlay_boundaries(
        ax_basin, basins_gdf, ama_ina, name_col,
        label_fontsize=4.5, label_all=True,
    )
    overlay_cap_service_area(
        ax_basin, cap_service_area_geojson,
        target_crs=basins_gdf.crs,
    )
    # Match panel (b)'s axes extent so both panels share the same
    # bounding box and the basin map doesn't render with a vertical
    # gap from auto-scaled tighter limits.
    if pixel_ref_raster is not None and os.path.isfile(pixel_ref_raster):
        with rio.open(pixel_ref_raster) as src:
            _basin_extent_crs = src.crs
            _b = src.bounds
        if basins_gdf.crs != _basin_extent_crs:
            ext_left, ext_bottom, ext_right, ext_top = (
                gpd.GeoSeries.from_wkt(
                    [f'POLYGON(({_b.left} {_b.bottom}, '
                     f'{_b.right} {_b.bottom}, '
                     f'{_b.right} {_b.top}, '
                     f'{_b.left} {_b.top}, '
                     f'{_b.left} {_b.bottom}))'],
                    crs=_basin_extent_crs,
                ).to_crs(basins_gdf.crs).total_bounds
            )
        else:
            ext_left, ext_bottom, ext_right, ext_top = (
                _b.left, _b.bottom, _b.right, _b.top
            )
        ax_basin.set_xlim(ext_left, ext_right)
        ax_basin.set_ylim(ext_bottom, ext_top)
    az_basin_total = sum(basin_sigma_cum.values()) * af_to_m3
    ax_basin.set_title(
        f'(a) Basin σ_cum\nAZ total: {az_basin_total / 1e9:.2f} km³ '
        f'({sum(basin_sigma_cum.values()) / 1e6:.2f} MAF)',
        fontsize=12, fontweight='bold',
    )
    sm_b = ScalarMappable(cmap=cmap_basin, norm=norm_basin)
    sm_b.set_array([])
    cbar_b = fig.colorbar(
        sm_b, ax=ax_basin, orientation='horizontal',
        shrink=0.8, pad=0.06, aspect=30,
        boundaries=boundaries_basin, ticks=boundaries_basin,
        extend='both',
    )
    cbar_b.formatter = mticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:,.0f}',
    )
    cbar_b.update_ticks()
    cbar_b.set_label(
        r'Basin Cumulative σ_total ($\times$10$^{6}$ m$^{3}$)',
        fontsize=11, fontweight='bold',
    )
    secax_b = cbar_b.ax.secondary_xaxis(
        'top',
        functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
    )
    secax_b.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e3:,.0f}'),
    )
    secax_b.set_xlabel('Basin Cumulative σ_total (kAF)',
                       fontsize=11, fontweight='bold')

    # --- Panel (b): pixel σ raster ---
    ax_pix = axes[1]
    ax_pix.set_facecolor('#EEEEEE')
    if pixel_sigma_cum is not None and pixel_ref_raster is not None:
        with rio.open(pixel_ref_raster) as src:
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]
            ref_crs = src.crs
        sigma_m3_arr = pixel_sigma_cum * af_to_m3
        masked = np.ma.masked_where(
            ~np.isfinite(sigma_m3_arr) | (sigma_m3_arr <= 0),
            sigma_m3_arr,
        )
        ax_pix.imshow(
            masked, extent=extent, origin='upper',
            cmap=cmap_pix, norm=norm_pix, interpolation='nearest',
        )
        basins_pix = (
            basins_gdf.to_crs(ref_crs)
            if basins_gdf.crs != ref_crs else basins_gdf
        )
        _overlay_boundaries(
            ax_pix, basins_pix, ama_ina, name_col,
            label_fontsize=4.5, label_all=True,
        )
        overlay_cap_service_area(
            ax_pix, cap_service_area_geojson,
            target_crs=ref_crs,
        )
        az_pix_total = float(np.nansum(pixel_sigma_cum))
        ax_pix.set_title(
            f'(b) Pixel σ_cum\nAZ total (sum of pixels): '
            f'{az_pix_total * af_to_m3 / 1e9:.2f} km³ '
            f'({az_pix_total / 1e6:.2f} MAF)',
            fontsize=12, fontweight='bold',
        )
    else:
        ax_pix.text(
            0.5, 0.5, 'Pixel σ rasters unavailable',
            ha='center', va='center', transform=ax_pix.transAxes,
        )
        ax_pix.axis('off')
    sm_p = ScalarMappable(cmap=cmap_pix, norm=norm_pix)
    sm_p.set_array([])
    cbar_p = fig.colorbar(
        sm_p, ax=ax_pix, orientation='horizontal',
        shrink=0.8, pad=0.06, aspect=30,
        boundaries=boundaries_pix, ticks=boundaries_pix, extend='both',
    )
    cbar_p.formatter = mticker.FuncFormatter(
        lambda x, _: f'{x / 1e6:,.1f}',
    )
    cbar_p.update_ticks()
    cbar_p.set_label(
        r'Pixel Cumulative σ_total ($\times$10$^{6}$ m$^{3}$)',
        fontsize=11, fontweight='bold',
    )
    secax_p = cbar_p.ax.secondary_xaxis(
        'top',
        functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
    )
    secax_p.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'),
    )
    secax_p.set_xlabel('Pixel Cumulative σ_total (AF)',
                       fontsize=11, fontweight='bold')

    add_ama_ina_legend(ax_basin, include_cap=True)

    out_path = os.path.join(
        out_dir,
        f'CAP_Scenario_Sigma_Cumulative_'
        f'{year_window[0]}_{year_window[1]}.png',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('  CAP scenario σ_cum map saved to %s', out_path)


def _plot_cap_scenario_basin_snr(
        delta_df: 'pd.DataFrame',
        basin_sigma_cum: dict[str, float],
        basin_shp: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> None:
    """Per-scenario basin signal-to-noise (SNR = |ΔGW_cum| / σ_cum) maps.

    Highlights basins where the scenario's cumulative ΔGW signal
    exceeds the central pipeline's σ_total uncertainty.  Discrete bins
    centered on SNR = 1 (signal == 1σ noise): below 0.5 the signal is
    statistically subtle; above 2 the basin response is robust.

    Note: the metric is signal-to-noise (|signal| / σ), not the
    classical coefficient of variation (σ / mean) — earlier versions
    of this code mislabeled it as "CV".
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names,
        add_ama_ina_legend, apply_journal_style,
        overlay_cap_service_area,
    )

    apply_journal_style()
    if delta_df is None or delta_df.empty or not basin_sigma_cum:
        return

    sub = delta_df[
        (delta_df['Year'] >= year_window[0])
        & (delta_df['Year'] <= year_window[1])
    ]
    cum = (
        sub.groupby(['Scenario', 'Basin'])['Delta_GW_AF']
        .sum()
        .reset_index()
    )
    if cum.empty:
        return

    boundaries = [0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_levels = len(boundaries) - 1
    palette = cm.get_cmap('PuBuGn', n_levels)
    discrete_cmap = ListedColormap([palette(i) for i in range(n_levels)])
    discrete_cmap.set_under('#FFFFFF')
    norm = BoundaryNorm(boundaries, discrete_cmap.N, clip=False)

    basins_gdf = gpd.read_file(basin_shp)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()
    basins_clipped = _clip_basins_to_cap(
        basins_gdf, cap_service_area_geojson,
    )
    plot_polys = (
        basins_clipped if basins_clipped is not None else basins_gdf
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(
        f'CAP Scenario — Basin Signal-to-Noise (SNR = |ΔGW_cum| / σ_cum) '
        f'over {year_window[0]}–{year_window[1]}\n'
        f'SNR ≥ 1 → scenario signal exceeds the central-pipeline '
        f'σ_total noise floor at that basin (σ_cum aggregated over '
        f'CAP-service-area pixels only at intersected basins)',
        fontsize=13, fontweight='bold',
    )
    axes_flat = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    for i, sc_key in enumerate(_CAP_SCENARIO_PANEL_ORDER):
        ax = axes_flat[i]
        ax.set_facecolor('#EEEEEE')
        sc_data = cum[cum['Scenario'] == sc_key]
        if sc_data.empty:
            ax.text(0.5, 0.5, f'No data for {sc_key}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        sc_lookup = dict(zip(sc_data['Basin'], sc_data['Delta_GW_AF'].abs()))
        snr_lookup: dict[str, float] = {}
        for b, delta_abs in sc_lookup.items():
            sigma = basin_sigma_cum.get(b, 0.0)
            # Skip basins with zero ΔGW (outside CAP service area)
            # — they should render white, not as SNR = 0 (lightest bin).
            if sigma > 0 and delta_abs > 1.0:
                snr_lookup[b] = float(delta_abs / sigma)
        plot_gdf = plot_polys.set_index(name_col).copy()
        plot_gdf['snr'] = plot_gdf.index.map(
            lambda b: snr_lookup.get(b, np.nan),
        )
        plot_gdf.plot(
            ax=ax, column='snr', cmap=discrete_cmap, norm=norm,
            edgecolor='none', linewidth=0,
            missing_kwds={'color': '#FFFFFF', 'edgecolor': 'none',
                          'linewidth': 0},
        )
        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col,
            label_fontsize=4.5, label_all=True,
        )
        overlay_cap_service_area(
            ax, cap_service_area_geojson,
            target_crs=basins_gdf.crs,
        )
        title_pretty = _CAP_SCENARIO_PANEL_TITLES.get(
            sc_key, sc_key.replace('_', ' '),
        )
        n_robust = sum(1 for v in snr_lookup.values() if v >= 1.0)
        n_total = len([1 for v in snr_lookup.values() if v > 0])
        ax.set_title(
            f'{panel_labels[i]} {title_pretty}\n'
            f'{n_robust}/{n_total} basins with SNR ≥ 1',
            fontsize=11, fontweight='bold',
        )

    ax_legend = axes_flat[7]
    ax_legend.axis('off')
    sm = ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes_flat[:7], orientation='vertical',
        shrink=1.0, pad=0.04, fraction=0.04, aspect=40,
        boundaries=boundaries, ticks=boundaries, extend='both',
    )
    cbar.set_label(
        '|Cumulative ΔGW| / σ_total (signal-to-noise)',
        fontsize=12, fontweight='bold',
    )
    cbar.ax.tick_params(labelsize=11)
    add_ama_ina_legend(
        ax_legend, loc='center', bbox_to_anchor=(0.5, 0.5),
        fontsize=14, framealpha=1.0, include_cap=True,
    )

    out_path = os.path.join(
        out_dir,
        f'CAP_Scenario_Basin_SNR_'
        f'{year_window[0]}_{year_window[1]}.png',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('  CAP scenario basin SNR map saved to %s', out_path)


def _plot_cap_scenario_pixel_snr(
        scenario_raster_paths: dict[str, str],
        pixel_sigma_cum: 'np.ndarray | None',
        pixel_ref_raster: 'str | None',
        basin_shp: str,
        out_dir: str,
        *,
        year_window: tuple[int, int] = (2027, 2060),
        basin_col: str = 'BASIN_NAME',
        cap_service_area_geojson: 'str | None' = None,
) -> None:
    """Per-scenario pixel signal-to-noise (SNR = |ΔGW_cum| / σ_cum) maps.

    Pixel-level analogue of :func:`_plot_cap_scenario_basin_snr`.  The
    per-pixel σ_cum is the linear-time-sum of per-year σ_total_GW
    from ``Sigma_Total/Rasters/`` (perfect correlation, conservative
    upper bound).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import cm
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names,
        add_ama_ina_legend, apply_journal_style,
        overlay_cap_service_area,
    )

    apply_journal_style()
    if (
        not scenario_raster_paths
        or pixel_sigma_cum is None
        or pixel_ref_raster is None
    ):
        return

    boundaries = [0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_levels = len(boundaries) - 1
    palette = cm.get_cmap('PuBuGn', n_levels)
    discrete_cmap = ListedColormap([palette(i) for i in range(n_levels)])
    discrete_cmap.set_under('#FFFFFF')
    norm = BoundaryNorm(boundaries, discrete_cmap.N, clip=False)

    with rio.open(pixel_ref_raster) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        ref_crs = src.crs
        ref_shape = src.shape

    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != ref_crs:
        basins_gdf = basins_gdf.to_crs(ref_crs)
    name_col = (
        basin_col if basin_col in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    sigma_safe = np.where(
        np.isfinite(pixel_sigma_cum) & (pixel_sigma_cum > 0),
        pixel_sigma_cum, np.nan,
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(
        f'CAP Scenario — Pixel Signal-to-Noise (SNR = |ΔGW_cum| / σ_cum) '
        f'over {year_window[0]}–{year_window[1]}\n'
        f'(per-pixel ΔGW = basin Δ × pixel ML Total_GW share — '
        f'pro-rata, NOT a hydraulic-head response; SNR ≥ 1 → signal '
        f'exceeds local σ noise)',
        fontsize=13, fontweight='bold',
    )
    axes_flat = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    for i, sc_key in enumerate(_CAP_SCENARIO_PANEL_ORDER):
        ax = axes_flat[i]
        ax.set_facecolor('#EEEEEE')
        rpath = scenario_raster_paths.get(sc_key)
        if rpath is None or not os.path.isfile(rpath):
            ax.text(0.5, 0.5, f'No raster for {sc_key}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        with rio.open(rpath) as src:
            arr_af = src.read(1).astype(np.float64)
        snr = np.abs(arr_af) / sigma_safe
        # Mask pixels where ΔGW = 0 (outside CAP service area) so
        # they render in the axes face color instead of SNR bin 0
        # (the lightest color, which would falsely imply a faint
        # CAP signal at desert basins).
        snr = np.where(np.isfinite(arr_af) & (np.abs(arr_af) > 1.0),
                       snr, np.nan)
        masked = np.ma.masked_invalid(snr)
        ax.imshow(
            masked, extent=extent, origin='upper',
            cmap=discrete_cmap, norm=norm, interpolation='nearest',
        )
        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col,
            label_fontsize=4.5, label_all=True,
        )
        overlay_cap_service_area(
            ax, cap_service_area_geojson,
            target_crs=ref_crs,
        )
        title_pretty = _CAP_SCENARIO_PANEL_TITLES.get(
            sc_key, sc_key.replace('_', ' '),
        )
        # Pct of pixels with SNR >= 1 among those with finite signal
        finite = np.isfinite(snr)
        if finite.any():
            pct_robust = 100.0 * np.nansum(snr[finite] >= 1.0) / finite.sum()
        else:
            pct_robust = 0.0
        ax.set_title(
            f'{panel_labels[i]} {title_pretty}\n'
            f'{pct_robust:.1f}% of pixels with SNR ≥ 1',
            fontsize=11, fontweight='bold',
        )

    ax_legend = axes_flat[7]
    ax_legend.axis('off')
    sm = ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes_flat[:7], orientation='vertical',
        shrink=1.0, pad=0.04, fraction=0.04, aspect=40,
        boundaries=boundaries, ticks=boundaries, extend='both',
    )
    cbar.set_label(
        '|Cumulative ΔGW| / σ_total (signal-to-noise)',
        fontsize=12, fontweight='bold',
    )
    cbar.ax.tick_params(labelsize=11)
    add_ama_ina_legend(
        ax_legend, loc='center', bbox_to_anchor=(0.5, 0.5),
        fontsize=14, framealpha=1.0, include_cap=True,
    )

    out_path = os.path.join(
        out_dir,
        f'CAP_Scenario_Pixel_SNR_'
        f'{year_window[0]}_{year_window[1]}.png',
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info('  CAP scenario pixel SNR map saved to %s', out_path)


def _plot_sens_section(
        df_section: pd.DataFrame,
        plot_cats: tuple,
        title: str,
        ribbon_label: str,
        out_path: str,
        start_year: int,
        end_year: int,
) -> None:
    """Render a single partition-sensitivity time-series figure.

    Used by ``run_density_ratio_sensitivity`` to plot either the density
    section or the smoothing section with identical styling.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from hydrolibs.visualops import (
        ERA_COLORS,
        ERA_PERIODS,
        _format_volume_axis,
        apply_journal_style,
    )

    apply_journal_style()

    cat_titles = {
        'Irrigation_GW': 'Irrigation GW',
        'Irrigation_SW': 'Irrigation SW',
        'Non_Irrigation_GW': 'Non-Irrigation GW',
        'Non_Irrigation_SW': 'Non-Irrigation SW',
        'Total_GW': 'Total GW',
        'Total_SW': 'Total SW',
    }
    cat_colors = {
        'Irrigation_GW': '#2980B9',
        'Irrigation_SW': '#27AE60',
        'Non_Irrigation_GW': '#8E44AD',
        'Non_Irrigation_SW': '#E67E22',
        'Total_GW': '#2C3E50',
        'Total_SW': '#E74C3C',
    }

    n_cats = len(plot_cats)
    fig, axes = plt.subplots(n_cats, 1, figsize=(16, 4 * n_cats), sharex=True)
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, plot_cats):
        cdf = df_section[df_section.Category == cat].sort_values('Year')
        if cdf.empty:
            continue
        years = cdf['Year'].values
        baseline = cdf['Baseline_AF'].values
        plus_af = cdf['Plus_AF'].values
        minus_af = cdf['Minus_AF'].values
        color = cat_colors.get(cat, '#2C3E50')

        for era, (s, e) in ERA_PERIODS.items():
            ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.08)

        ax.fill_between(years, minus_af, plus_af,
                        alpha=0.20, color=color, label=ribbon_label)
        ax.plot(years, baseline, color=color, linewidth=1.4,
                marker='.', markersize=2, label='Baseline')
        ax.plot(years, plus_af, color=color, linewidth=0.8,
                linestyle='--', alpha=0.6)
        ax.plot(years, minus_af, color=color, linewidth=0.8,
                linestyle='--', alpha=0.6)

        _format_volume_axis(ax, unit='AF', label='Volume')
        ax.set_title(cat_titles.get(cat, cat), fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        handles = ax.get_legend_handles_labels()[0]
        era_handles = [
            mpatches.Patch(
                color=ERA_COLORS[e], alpha=0.4,
                label=f'{e} ({ERA_PERIODS[e][0]}\u2013{ERA_PERIODS[e][1]})',
            )
            for e in ERA_PERIODS
        ]
        ax.legend(handles=handles + era_handles,
                  loc='upper left', framealpha=0.9, fontsize=8)

    axes[-1].set_xlabel('Year', fontweight='bold')
    axes[-1].set_xlim(start_year - 1, end_year + 1)
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.001)
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()


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

    Args:
        sigma_components (dict[str, dict[int, np.ndarray]]): Mapping of component name to
            {year: 1-D std array}.
            E.g. {'MACA': {...}, 'Model': {...}, 'Irr': {...}, 'GW': {...}}
        prediction_raster_dir (str): Directory containing total-pumping prediction rasters named
            ``Total_Predicted_{year}_mm.tif``.  Required for CV computation.
        cat_sigma_components (dict or None): Mapping of component name to
            {category: {year: 1-D std array}}.
            When provided, per-category sigma_total is computed via quadrature
            and written as ``Sigma_Total_{cat}_mm_{year}.tif``.

    Returns:
        tuple[dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]: (sigma_total,
            cat_sigma_total) — per-year total sigma arrays and per-category per-year
            sigma_total arrays.
    """
    from hydrolibs.rasterops import read_raster_as_arr, write_raster
    from hydrolibs.sysops import makedirs

    logger.info('Computing σ_total (quadrature combination)...')
    base_dir = os.path.join(output_dir, 'Sigma_Total')
    raster_dir = os.path.join(base_dir, 'Rasters')
    makedirs(raster_dir)

    ref_basin_file = os.path.join(pred_data_dir, f'GW_Basin_{year_list[0]}.tif')
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(pred_data_dir, f'Predictor_{year_list[0]}.tif')

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Reload per-component σ from disk when in-memory dicts are empty
    # (e.g. when individual σ steps were skipped).
    comp_names = list(sigma_components.keys())
    any_empty = any(not comp for comp in sigma_components.values())
    if any_empty:
        logger.info('  Some σ components are empty in memory; '
                     'reloading from existing rasters on disk...')
        for name in comp_names:
            if sigma_components[name]:
                continue  # already populated
            comp_raster_dir = os.path.join(output_dir, f'Sigma_{name}', 'Rasters')
            if not os.path.isdir(comp_raster_dir):
                continue
            for year in range(start_year, end_year + 1):
                raster_file = os.path.join(
                    comp_raster_dir, f'Sigma_{name}_mm_{year}.tif')
                if not os.path.exists(raster_file):
                    continue
                arr = read_raster_as_arr(raster_file, get_file=False)
                vals = arr.ravel()[valid_mask]
                sigma_components[name][year] = vals
            if sigma_components[name]:
                logger.info(f'    Reloaded σ_{name}: {len(sigma_components[name])} years')

        # Also reload per-category σ if needed
        if cat_sigma_components:
            import hydrolibs.partitionops as _partops
            for name in comp_names:
                cat_comp = cat_sigma_components.get(name, {})
                any_cat_empty = not cat_comp or all(
                    (not v if isinstance(v, dict) else len(v) == 0)
                    for v in cat_comp.values())
                if not any_cat_empty:
                    continue
                comp_raster_dir = os.path.join(
                    output_dir, f'Sigma_{name}', 'Rasters')
                if not os.path.isdir(comp_raster_dir):
                    continue
                if name not in cat_sigma_components:
                    cat_sigma_components[name] = {}
                for cat in _partops.CATEGORIES:
                    cat_sigma_components[name].setdefault(cat, {})
                    for year in range(start_year, end_year + 1):
                        raster_file = os.path.join(
                            comp_raster_dir,
                            f'Sigma_{name}_{cat}_mm_{year}.tif')
                        if not os.path.exists(raster_file):
                            continue
                        arr = read_raster_as_arr(raster_file, get_file=False)
                        vals = arr.ravel()[valid_mask]
                        cat_sigma_components[name][cat][year] = vals

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
        pred_file = os.path.join(prediction_raster_dir, f'Total_Predicted_{year}_mm.tif')
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

        stats = _pixel_stats(total_std, mm_to_m3)
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


def compute_basin_sigma_total(output_dir: str, prediction_dir: str = '') -> None:
    """Combine per-component basin/sub-basin σ CSVs via quadrature.

    Reads ``{Basin|Subbasin}_Sigma_{comp}.csv`` from each
    ``Sigma_{comp}/`` directory, joins on ``(Year, Region)``, and
    writes ``Basin_Sigma_Total.csv`` / ``Subbasin_Sigma_Total.csv``
    into ``Sigma_Total/``.

    Args:
        output_dir: Uncertainty base directory (contains Sigma_*/ subdirs).
        prediction_dir: Full prediction directory containing
            ``Annual_Summaries/{Basin|Subbasin}_Total.csv``.  If provided,
            the actual prediction volumes are used as the mean (instead of
            averaging per-component ensemble means).
    """
    component_labels = ('MACA', 'Model', 'Irr', 'LULC', 'GW', 'USBR')
    total_dir = os.path.join(output_dir, 'Sigma_Total')

    # Load actual prediction volumes from output rasters if available
    pred_volumes = {}  # {('Basin'|'Subbasin'): DataFrame with Year, Region, Volume_m3, Volume_AF}
    for level, fname in (('Basin', 'Basin_Total.csv'), ('Subbasin', 'Subbasin_Total.csv')):
        csv_path = os.path.join(prediction_dir, 'Annual_Summaries', fname)
        if prediction_dir and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            region_col = 'Basin' if level == 'Basin' else 'Subbasin'
            if region_col in df.columns and 'Volume_m3' in df.columns:
                pred_volumes[level] = df.rename(
                    columns={region_col: 'Region'}
                )[['Year', 'Region', 'Volume_m3', 'Volume_AF']]
                logger.info(f'  Loaded actual {level} prediction volumes '
                            f'from {csv_path}')

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

        # Mean volume: use actual prediction volumes from output rasters
        # when available; fall back to averaging component means.
        if level in pred_volumes:
            merged = merged.merge(
                pred_volumes[level].rename(columns={
                    'Volume_m3': 'Mean_Volume_m3',
                    'Volume_AF': 'Mean_Volume_AF',
                }),
                on=['Year', 'Region'], how='left',
            )
            merged['Mean_Volume_m3'] = merged['Mean_Volume_m3'].fillna(0).round(2)
            merged['Mean_Volume_AF'] = merged['Mean_Volume_AF'].fillna(0).round(2)
        else:
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
        # Withdrawal volumes are physically non-negative; clip the 95%
        # CI lower bound at 0 so the on-disk values match what the
        # plotting layer renders.
        merged['Lower_95CI_m3'] = (
            (merged['Mean_Volume_m3'] - CI_Z * merged['Sigma_Total_m3'])
            .clip(lower=0).round(2)
        )
        merged['Upper_95CI_m3'] = (
            merged['Mean_Volume_m3'] + CI_Z * merged['Sigma_Total_m3']
        ).round(2)
        merged['Lower_95CI_AF'] = (
            (merged['Mean_Volume_AF'] - CI_Z * merged['Sigma_Total_AF'])
            .clip(lower=0).round(2)
        )
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
            os.path.join(total_dir, f'{level}_Sigma_Total.csv'), index=False,
        )
        logger.info(f'  Wrote {total_dir}/{level}_Sigma_Total.csv')


# ═════════════════════════════════════════════════════════════════════════════
# σ_CU — Consumptive-use inter-GCM spread (2026-2099 only)
# ═════════════════════════════════════════════════════════════════════════════

def compute_sigma_cu(
        prediction_dir: str,
        output_dir: str,
        basin_shp: str,
        input_dir: str,
        start_year: int,
        end_year: int,
        mosaic_res: int,
) -> None:
    """
    Compute σ_CU via error propagation from CU = IE × Withdrawal.

    Two uncertainty sources are combined in quadrature:

        σ_CU = √((IE × σ_wd)² + (wd × σ_IE)²)

    where σ_wd is the per-category total uncertainty from the augmented
    Irrigation rasters (band 2) and σ_IE is the inter-annual std of
    USGS NHM basin-level irrigation efficiency (2000–2020).

    For NHM-covered years (2000–2020), per-year basin IEs are used and
    σ_IE = 0 (observed efficiency).  For all other years, σ_IE equals
    the basin-level temporal std of IE.

    Writes σ rasters for Irrigation_CU, Irrigation_GW_CU, and
    Irrigation_SW_CU.

    Args:
        prediction_dir (str): Base prediction directory (e.g., ``Full_Prediction_XGB``).
        output_dir (str): Base output directory for uncertainty products.
        basin_shp (str): Path to groundwater basin shapefile.
        input_dir (str): Root input directory (for NHM IE CSV paths).
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        mosaic_res (int): Raster resolution in meters.

    Returns:
        None.
    """
    from hydrolibs.sysops import makedirs
    import hydrolibs.intercompops as intercompops

    logger.info('Computing σ_CU (IE × Withdrawal error propagation)...')
    sigma_cu_dir = os.path.join(output_dir, 'Sigma_CU/Rasters')
    makedirs(sigma_cu_dir)

    pixel_area_m2 = mosaic_res ** 2
    mm_to_m3 = pixel_area_m2 / 1000

    # Load basin-level NHM IEs
    nhm_ie_csv = os.path.join(input_dir, 'USGS WU', 'USGS_NHM_Withdrawals',
                              'IR_HUC12_Eff_annual_2000_2020.csv')
    huc12_geojson = os.path.join(input_dir, 'GEE_Data', 'AZ_HUC12.geojson')
    nhm_ie_out = os.path.join(output_dir, 'Sigma_CU', 'NHM_IE')
    # Find a reference raster
    irr_dir = os.path.join(prediction_dir, 'Irrigation_Rasters/Depth_mm')
    ref_raster = None
    for yr in range(start_year, end_year + 1):
        candidate = os.path.join(irr_dir, f'Irrigation_{yr}_mm.tif')
        if os.path.isfile(candidate):
            ref_raster = candidate
            break
    if ref_raster is None:
        logger.warning('No Irrigation rasters found — skipping σ_CU')
        return

    nhm_basin_ie = intercompops.load_nhm_basin_ie(
        nhm_ie_csv=nhm_ie_csv,
        huc12_geojson=huc12_geojson,
        basin_shp=basin_shp,
        basin_col='BASIN_NAME',
        ref_raster=ref_raster,
        output_dir=nhm_ie_out,
    )

    # Build a per-pixel basin label raster for IE mapping
    basin_gdf = gpd.read_file(basin_shp)
    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        ref_profile = src.profile.copy()
        ref_shape = (src.height, src.width)

    basin_reproj = basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    basin_names = sorted(basin_gdf['BASIN_NAME'].unique().tolist())
    basin_to_idx = {name: i + 1 for i, name in enumerate(basin_names)}

    # Rasterize basin labels
    from rasterio.features import rasterize
    from shapely.geometry import mapping
    shapes = [
        (mapping(row.geometry), basin_to_idx[row['BASIN_NAME']])
        for _, row in basin_reproj.iterrows()
        if row['BASIN_NAME'] in basin_to_idx
    ]
    basin_raster = rasterize(
        shapes, out_shape=ref_shape,
        transform=ref_profile['transform'],
        fill=0, dtype='int32',
    )

    ie_per_year = nhm_basin_ie['per_year']
    ie_mean = nhm_basin_ie['mean']
    ie_std = nhm_basin_ie['std']
    overall_mean = nhm_basin_ie['overall_mean']

    # Category mapping: CU category → withdrawal category raster dir
    cu_wd_map = {
        'Irrigation_CU': 'Irrigation',
        'Irrigation_GW_CU': 'Irrigation_GW',
        'Irrigation_SW_CU': 'Irrigation_SW',
    }

    yearly_stats = {}

    for year in range(start_year, end_year + 1):
        # Build pixel-level IE mean and IE std arrays
        ie_grid = np.full(ref_shape, overall_mean, dtype=np.float32)
        ie_std_grid = np.zeros(ref_shape, dtype=np.float32)

        for basin_name, idx in basin_to_idx.items():
            bmask = basin_raster == idx
            if year in ie_per_year:
                val = ie_per_year[year].get(basin_name, np.nan)
                ie_grid[bmask] = val if np.isfinite(val) else ie_mean.get(
                    basin_name, overall_mean)
                # NHM year: observed IE → σ_IE = 0
            else:
                ie_grid[bmask] = ie_mean.get(basin_name, overall_mean)
                ie_std_grid[bmask] = ie_std.get(basin_name, 0.0)

        ie_grid = np.clip(ie_grid, 0, 1)

        ref_profile.update(count=1, dtype=np.float32, nodata=np.nan)

        for cu_cat, wd_cat in cu_wd_map.items():
            wd_file = os.path.join(
                prediction_dir, f'{wd_cat}_Rasters/Depth_mm',
                f'{wd_cat}_{year}_mm.tif',
            )
            if not os.path.isfile(wd_file):
                continue

            with rio.open(wd_file) as src:
                if src.count >= 2:
                    wd_pred = src.read(1).astype(np.float32)
                    sigma_wd = src.read(2).astype(np.float32)
                else:
                    wd_pred = src.read(1).astype(np.float32)
                    sigma_wd = np.zeros_like(wd_pred)

            # σ_CU = √((IE × σ_wd)² + (wd × σ_IE)²)
            with np.errstate(invalid='ignore'):
                sigma_cu = np.sqrt(
                    (ie_grid * sigma_wd) ** 2
                    + (wd_pred * ie_std_grid) ** 2
                ).astype(np.float32)

            out_path = os.path.join(sigma_cu_dir, f'Sigma_{cu_cat}_mm_{year}.tif')
            with rio.open(out_path, 'w', **ref_profile) as dst:
                dst.write(sigma_cu, 1)

        # Track stats for the total CU sigma
        sigma_total_file = os.path.join(sigma_cu_dir,
                                        f'Sigma_Irrigation_CU_mm_{year}.tif')
        if os.path.isfile(sigma_total_file):
            with rio.open(sigma_total_file) as src:
                sigma_total = src.read(1)
            yearly_stats[year] = _pixel_stats(
                sigma_total.ravel(), mm_to_m3,
            )
            if year % 20 == 0 or year == end_year:
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
        vector_dir: str,
        mosaic_res: int,
        gcloud_project: str,
        gcloud_bucket: str,
        tile_size: int,
        start_year: int,
        end_year: int,
        year_list: list[int],
        fold_count: int = 5,
        repeats: int = 3,
        n_trials: int = 100,
        n_dask_workers: int = 10,
        use_dask: bool = True,
        skip_download: bool = False,
        subbasin_shp: str = '',
        ama_code_map: dict | None = None,
        basin_shp: str = '',
        prediction_model: str = 'XGB',
        skip_uq_steps: set[str] | None = None,
) -> None:
    """
    Run the full hybrid uncertainty quantification pipeline.

    Computes σ_MACA (future only), σ_model, σ_irr (historical only),
    σ_LULC (future only), σ_gw, and combines them into σ_total via
    quadrature.  Writes per-component and total uncertainty rasters,
    summary CSVs, and time-series plots.

    Args:
        model: Trained ML model for prediction.
        feature_cols (list[str]): Feature column names.
        x_train (pd.DataFrame): Training feature matrix.
        y_train (np.ndarray): Training target array.
        az_df (pd.DataFrame): Arizona training DataFrame.
        drop_attrs (tuple[str, ...]): Columns to drop before prediction.
        pred_data_dir (str): Directory containing predictor rasters.
        model_dir (str): Base model output directory.
        input_dir (str): Base input directory for GEE downloads.
        vector_dir (str): Directory containing vector shapefiles.
        mosaic_res (int): Raster resolution in meters.
        gcloud_project (str): Google Cloud project ID.
        gcloud_bucket (str): Google Cloud Storage bucket name.
        tile_size (int): GEE export tile size.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        year_list (list[int]): Full list of prediction years.
        fold_count (int): Number of folds for KFold. Default is 5.
        repeats (int): Number of repeats for RepeatedKFold. Default is 3.
        n_trials (int): Number of Optuna trials per seed.
        n_dask_workers (int): Number of Dask workers.
        use_dask (bool): If True, use Dask for parallel tuning.
        skip_download (bool): If True, skip GEE download step.
        subbasin_shp (str): Path to ADWR sub-basin shapefile.
        ama_code_map (dict or None): AMA/INA code mapping.
        basin_shp (str): Path to GW basin shapefile.

    Returns:
        None.
    """
    import hydrolibs.visualops as vizops
    from hydrolibs.sysops import makedirs

    logger.info('=' * 60)
    logger.info('Step 3b: Hybrid Uncertainty Quantification')
    logger.info('=' * 60)

    pred_base = f'Full_Prediction_{prediction_model}'
    unc_dir = os.path.join(model_dir, pred_base, 'Uncertainty')
    makedirs(unc_dir)

    # Initialise pre-GMA partitioning context from the 2024 az_df snapshot
    # so all UQ ensemble members partition with the same overrides as the
    # central pipeline run (well-density blend in ML features +
    # 2024 well/irr_capacity arrays in partition_predictions).
    _set_pre_gma_context(az_df, ref_year=2024)

    # Initialise CAP delivery pixel mask so UQ ensemble members apply
    # the same CAP perturbation as the central pipeline at observed
    # 2022-2024 cuts and the 2026-2099 sustained baseline (mirrors
    # partops.apply_cap_delivery_perturbation).
    _cap_pixel_mask_uq = _build_cap_pixel_mask(
        vector_dir, pred_data_dir, year_list[0],
    )
    if _cap_pixel_mask_uq is not None:
        _set_cap_pixel_mask_context(_cap_pixel_mask_uq)

    # Initialise CAP per-basin per-year delivery lookup so UQ ensemble
    # members apply the same dynamic NonIrr GW share cap as the central
    # pipeline (partops.CAP_BASIN_NI_CAP_CONFIG / _cap_basin_ni_cap).
    _cap_xlsx_uq = os.path.join(
        vector_dir, 'CAP', 'CAP Delivery Data DRI Request.xlsx',
    )
    if os.path.isfile(_cap_xlsx_uq):
        try:
            _set_cap_delivery_context(
                partops.load_cap_basin_delivery(_cap_xlsx_uq),
            )
        except Exception as _e:
            logger.warning(
                'UQ could not load CAP delivery lookup from %s: %s — '
                'NonIrr GW share cap will fall back to peak-baseline.',
                _cap_xlsx_uq, _e,
            )

    skip = skip_uq_steps or set()
    if skip:
        logger.info(f'Skipping UQ sub-steps: {skip}')

    # ── σ_MACA ──
    sigma_maca = cat_sigma_maca = {}
    if 'sigma-maca' not in skip:
        sigma_maca, cat_sigma_maca, gcm_mosaic_dirs = compute_sigma_maca(
            model, feature_cols, az_df, drop_attrs,
            pred_data_dir, unc_dir, input_dir, vector_dir,
            mosaic_res, gcloud_project, gcloud_bucket,
            tile_size, end_year, year_list,
            skip_download=skip_download,
        )
    else:
        logger.info('  σ_MACA skipped.')

    # ── σ_model ──
    sigma_model = cat_sigma_model = {}
    if 'sigma-model' not in skip:
        sigma_model, cat_sigma_model = compute_sigma_model(
            x_train, y_train, feature_cols, az_df,
            drop_attrs, pred_data_dir, unc_dir,
            start_year, end_year, year_list, mosaic_res,
            prediction_model=prediction_model,
            model_dir=model_dir,
            fold_count=fold_count, repeats=repeats,
            n_trials=n_trials, n_dask_workers=n_dask_workers,
            use_dask=use_dask,
        )
    else:
        logger.info('  σ_model skipped.')

    # ── σ_irr (historical only, 1896-2025) ──
    sigma_irr = cat_sigma_irr = {}
    if 'sigma-irr' not in skip:
        sigma_irr, cat_sigma_irr = compute_sigma_irr(
            model, feature_cols, az_df, drop_attrs,
            pred_data_dir, unc_dir, start_year, end_year,
            year_list, mosaic_res,
        )
    else:
        logger.info('  σ_irr skipped.')

    # ── σ_LULC (future only, 2026-2099 — subsumes σ_irr) ──
    sigma_lulc = cat_sigma_lulc = {}
    if 'sigma-lulc' not in skip:
        sigma_lulc, cat_sigma_lulc = compute_sigma_lulc(
            model, feature_cols, az_df, drop_attrs,
            pred_data_dir, unc_dir, input_dir, vector_dir,
            mosaic_res, gcloud_project, gcloud_bucket,
            tile_size, end_year, year_list,
            skip_download=skip_download,
        )
    else:
        logger.info('  σ_LULC skipped.')

    # ── σ_gw ──
    sigma_gw = cat_sigma_gw = {}
    if 'sigma-gw' not in skip:
        sigma_gw, cat_sigma_gw = compute_sigma_gw(
            model, feature_cols, az_df, drop_attrs,
            pred_data_dir, unc_dir, start_year, end_year,
            year_list, mosaic_res,
        )
    else:
        logger.info('  σ_gw skipped.')

    # ── σ_USBR (Upper Colorado River Basin streamflow uncertainty) ──
    # Captures inter-USBR-member spread of CAP delivery driven by
    # Upper Basin headwater hydrology — the dominant climate-uncertainty
    # axis that σ_MACA does not reach (MACA downscales to AZ-local
    # domain only).  Applies to all years 1896-2099; will be near-zero
    # for USGS-observed years (where streamflow is observed) and
    # substantial for projection (2026+) where USBR ensemble drives
    # the central estimate.
    sigma_usbr = cat_sigma_usbr = {}
    if 'sigma-usbr' not in skip:
        usbr_dir = os.path.join(input_dir, 'GW_Data', 'USBR')
        if os.path.isdir(usbr_dir):
            sites_csv = os.path.join(
                input_dir, 'GW_Data', 'Streamflow', 'sites.csv',
            )
            watershed_geojson = os.path.join(
                input_dir, 'GW_Data', 'Surface_Watershed.geojson',
            )
            sigma_usbr, cat_sigma_usbr = compute_sigma_usbr(
                model, feature_cols, az_df, drop_attrs,
                pred_data_dir, unc_dir, usbr_dir,
                start_year, end_year, year_list, mosaic_res,
                vector_dir=vector_dir,
                sites_csv=sites_csv,
                watershed_geojson=watershed_geojson,
            )
        else:
            logger.warning(
                '  σ_USBR skipped: USBR data dir not found at %s',
                usbr_dir,
            )
    else:
        logger.info('  σ_USBR skipped.')

    # ── Density-ratio partitioning sensitivity (±20%) ──
    if 'density-sensitivity' not in skip:
        run_density_ratio_sensitivity(
            model, feature_cols, az_df, drop_attrs,
            pred_data_dir, unc_dir, start_year, end_year,
            year_list, mosaic_res,
        )
    else:
        logger.info('  Density-ratio sensitivity skipped.')

    # ── σ_total + augmentation ──
    full_pred_dir = os.path.join(model_dir, pred_base)
    if 'sigma-total' not in skip:
        sigma_components = {
            'MACA': sigma_maca,
            'Model': sigma_model,
            'Irr': sigma_irr,
            'LULC': sigma_lulc,
            'GW': sigma_gw,
            'USBR': sigma_usbr,
        }
        cat_sigma_components = {
            'MACA': cat_sigma_maca,
            'Model': cat_sigma_model,
            'Irr': cat_sigma_irr,
            'LULC': cat_sigma_lulc,
            'GW': cat_sigma_gw,
            'USBR': cat_sigma_usbr,
        }
        prediction_raster_dir = (
            os.path.join(model_dir, pred_base, 'Predicted_Rasters', 'Depth_mm')
        )
        compute_sigma_total(
            sigma_components, pred_data_dir, unc_dir,
            start_year, end_year, year_list, mosaic_res,
            prediction_raster_dir=prediction_raster_dir,
            cat_sigma_components=cat_sigma_components,
        )

        # ── Basin / sub-basin σ_total (quadrature of per-component CSVs) ──
        full_prediction_dir = os.path.join(model_dir, pred_base)
        compute_basin_sigma_total(unc_dir, prediction_dir=full_prediction_dir)

        # ── Time-series visualizations (skippable) ──
        # Both basin σ time-series plots and AZ-wide uncertainty
        # time-series plots are wrapped behind ``time-series-plots`` so
        # downstream raster-augmentation can run without spending the
        # ~5-10 minute matplotlib render cost when only the underlying
        # CSVs / rasters are needed.  CSVs (Basin_Sigma_Total.csv etc.)
        # are still produced by compute_basin_sigma_total above.
        if 'time-series-plots' not in skip:
            _plot_basin_sigma_time_series(unc_dir)
            _plot_uncertainty_time_series(
                sigma_components, unc_dir, mosaic_res, vizops,
            )
        else:
            logger.info('  σ_total time-series plots skipped.')

        # ── Augment prediction rasters with uncertainty bands ──
        prediction_base_dir = (
            os.path.join(model_dir, pred_base, 'Predicted_Rasters')
        )
        augment_prediction_rasters(
            sigma_total_raster_dir=os.path.join(unc_dir, 'Sigma_Total', 'Rasters'),
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
    else:
        logger.info('  σ_total + augmentation skipped.')

    # ── CAP delivery reduction scenario analysis ──
    # Runs after sigma-total so per-category σ rasters in
    # Uncertainty/Sigma_Total/Rasters/ are available for plot ribbons.
    if 'cap-scenario' not in skip:
        cap_geojson = os.path.join(
            vector_dir, 'CAP', 'CAP_Service_Area.geojson',
        )
        if not os.path.isfile(cap_geojson):
            cap_geojson = os.path.join(
                os.path.dirname(basin_shp), 'CAP_Service_Area.geojson',
            )
        if os.path.isfile(cap_geojson):
            run_cap_scenario_analysis(
                model, feature_cols, az_df, drop_attrs,
                pred_data_dir, unc_dir, start_year, end_year,
                year_list, mosaic_res,
                cap_service_area_geojson=cap_geojson,
            )
        else:
            logger.warning(
                '  CAP service area geojson not found; '
                'skipping CAP scenario analysis'
            )
    else:
        logger.info('  CAP scenario analysis skipped.')

    # ── Surface Water Capture Index with propagated σ_GW ──
    #
    # Runs after ``augment_category_rasters`` so the per-category
    # rasters contain band-1 prediction and band-2 σ_total that
    # ``compute_sw_capture_with_sigma`` reads directly.  The resulting
    # 6-band augmented capture rasters are automatically picked up by
    # the downstream well-level disaggregation in ``wellops.py`` via
    # its existing ``src.count >= 6`` σ branch, giving per-well
    # capture and σ_capture columns in ``Well_Package.gpkg`` with no
    # wellops changes required.
    if 'sw-capture-sigma' not in skip:
        compute_sw_capture_with_sigma(
            prediction_dir=full_pred_dir,
            az_df=az_df,
            pred_data_dir=pred_data_dir,
            start_year=start_year,
            end_year=end_year,
            year_list=year_list,
            mosaic_res=mosaic_res,
        )
    else:
        logger.info('  SW Capture Index with σ_GW propagation skipped.')

    # ── σ_CU (error propagation: CU = IE × Withdrawal) ──
    if 'sigma-cu' not in skip:
        compute_sigma_cu(
            prediction_dir=full_pred_dir,
            output_dir=unc_dir,
            basin_shp=basin_shp,
            input_dir=input_dir,
            start_year=start_year,
            end_year=end_year,
            mosaic_res=mosaic_res,
        )

        # ── Augment CU rasters ──
        augment_cu_rasters(
            sigma_cu_raster_dir=os.path.join(unc_dir, 'Sigma_CU', 'Rasters'),
            prediction_dir=full_pred_dir,
            start_year=start_year,
            end_year=end_year,
            mosaic_res=mosaic_res,
        )
    else:
        logger.info('  σ_CU skipped.')

    # ── Re-plot prediction time series with uncertainty bounds ──
    # Derive all uncertainty data directly from the augmented 6-band
    # rasters using zonal statistics with basin / sub-basin shapefiles.
    # Gated on the same ``time-series-plots`` skip token as the
    # σ-component time-series block above so a single
    # ``--skip-uq time-series-plots`` skips ALL time-series replot
    # work in Step 3b (this is the slow ~10-minute zonal-stats loop;
    # gating it correctly keeps the underlying augmented rasters and
    # CSVs intact while skipping the rendering cost).
    if 'time-series-plots' not in skip:
        _replot_from_augmented_rasters(
            prediction_dir=full_pred_dir,
            basin_shp=basin_shp,
            subbasin_shp=subbasin_shp,
            ama_code_map=ama_code_map,
            start_year=start_year,
            end_year=end_year,
            mosaic_res=mosaic_res,
        )
    else:
        logger.info(
            '  Augmented-raster time-series replot skipped '
            '(--skip-uq time-series-plots).'
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

    Rewrites each ``Total_Predicted_{year}_{unit}.tif`` (for mm, ft, m³, AF)
    as a 6-band GeoTIFF:

        1. Prediction (unit)
        2. σ_total (unit)
        3. CV  (σ / |pred|)
        4. SNR (|pred| / σ)
        5. Lower 95 % CI  (pred − CI_Z·σ)
        6. Upper 95 % CI  (pred + CI_Z·σ)

    σ_total rasters are stored in mm; they are scaled to the target unit
    before writing.

    Args:
        sigma_total_raster_dir (str): Directory containing Sigma_Total rasters.
        prediction_base_dir (str): Base directory for prediction rasters.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        mosaic_res (int or float): Raster resolution in meters.

    Returns:
        None.
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
        pred_dir = os.path.join(prediction_base_dir, subdir)
        scale = sigma_scale[unit]

        band_descriptions = [
            f'prediction_{unit}', f'sigma_total_{unit}', 'CV', 'SNR',
            f'lower_95CI_{unit}', f'upper_95CI_{unit}',
        ]

        for year in range(start_year, end_year + 1):
            pred_file = os.path.join(pred_dir, f'Total_Predicted_{year}_{unit}.tif')
            sigma_file = os.path.join(
                sigma_total_raster_dir, f'Sigma_Total_mm_{year}.tif'
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

            lower_ci = np.maximum(pred_arr - CI_Z * sigma_arr, 0).astype(np.float32)
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

    Args:
        prediction_dir (str): Base directory for per-category prediction rasters.
        sigma_total_raster_dir (str): Directory containing per-category σ_total rasters.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        mosaic_res (int or float): Raster resolution in meters.

    Returns:
        None.
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

    Args:
        sigma_cu_raster_dir (str): Directory containing σ_CU rasters.
        prediction_dir (str): Base directory for CU prediction rasters.
        start_year (int): First year of prediction period.
        end_year (int): Last year of prediction period.
        mosaic_res (int or float): Raster resolution in meters.

    Returns:
        None.
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
                cu_file = os.path.join(cu_dir, f'{cu_cat}_{year}_{unit}.tif')
                if not os.path.exists(cu_file):
                    continue

                with rio.open(cu_file) as src:
                    cu_pred = src.read(1)
                    profile = src.profile.copy()

                # Read σ_CU in mm, scale to target unit
                sigma_file = os.path.join(
                    sigma_cu_raster_dir, f'Sigma_{cu_cat}_mm_{year}.tif'
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


def compute_sw_capture_with_sigma(
        prediction_dir: str,
        az_df: pd.DataFrame,
        pred_data_dir: str,
        start_year: int,
        end_year: int,
        year_list: list[int],
        mosaic_res: int | float,
) -> None:
    """Compute the Surface Water Capture Index with propagated σ_GW.

    Runs *after* ``augment_category_rasters`` so that the per-category
    augmented rasters contain band-1 prediction and band-2 σ_total
    (from the 6-component UQ framework) in matching depth units.  For
    each year and each of the three GW pumping pools (Total_GW,
    Irrigation_GW, Non_Irrigation_GW), this function:

    1. Reads the augmented Depth_mm category raster for that pool and
       year, extracting band 1 (prediction, mm) and band 2 (σ_total, mm).
    2. Calls ``partitionops.compute_sw_capture_index`` with the real
       ``sigma_gw`` populated, which produces the asymmetric combined
       95 % CI bounds:
           ``vol_lower = max(gw − 1.96 σ, 0) × cf_lower`` (λ = 5 m)
           ``vol_upper = (gw + 1.96 σ) × cf_upper`` (λ = 20 m)
       with ``vol_central = gw × cf_central`` (λ = 10 m) unchanged.
    3. Derives a per-pixel σ_cap at the central λ via
       ``σ_cap = 0.5 × (vol_upper − vol_lower) / 1.96``, keeping σ
       orthogonal to the λ dimension so downstream users can decompose
       the two uncertainty sources.
    4. Writes a 3-band capture-fraction raster (λ = 5/10/20) under
       ``{cap_cat}_Capture_Fraction/`` (unchanged convention).
    5. Writes the central capture volume in four units (mm, ft, m³, AF)
       as a **6-band augmented raster** (band 1 = pred, band 2 = σ_cap,
       bands 3–6 = CV, SNR, lower 95 % CI, upper 95 % CI) under the
       existing ``{cap_cat}_Capture_Rasters/{Depth_mm, Depth_ft,
       Volume_m3, Volume_AF}/`` convention.  This mirrors the 6-band
       layout already used for Total_GW, Irrigation_GW, CU rasters, etc.
       so the existing per-well disaggregation in ``wellops.py`` picks
       up the new σ columns automatically via its ``src.count >= 6``
       branch.
    6. Writes ``SW_Capture_Time_Series.csv``, ``Basin_Capture_Fraction.csv``,
       and ``Subbasin_Capture_Fraction.csv`` using the existing schema
       plus three new per-pool σ columns
       (``{cap_cat}_Capture_Volume_Sigma_AF``).

    The existing ``SW_Capture_Time_Series.csv`` Lower/Central/Upper
    column names are preserved; the meaning of Lower/Upper changes from
    "λ envelope" to "combined σ + λ envelope" as computed by the
    asymmetric bounds in ``compute_sw_capture_index``.

    Args:
        prediction_dir (str): Base directory
            (``Full_Prediction_{model}``) containing ``{pool}_Rasters/``
            (augmented by ``augment_category_rasters``), ``SW_Capture/``,
            and ``Annual_Summaries/``.
        az_df (pd.DataFrame): Arizona training DataFrame with
            per-year ``GW_Basin``, ``GW_Subbasin``, ``wtd_m``, and
            ``canal_weighted_streamflow_mm`` columns.
        pred_data_dir (str): Directory containing the reference basin
            raster (used to build ``valid_mask`` / ``raster_shape``).
        start_year (int): First prediction year.
        end_year (int): Last prediction year.
        year_list (list[int]): Full prediction year list (used to pick
            the reference raster at ``year_list[0]``).
        mosaic_res (int or float): Raster resolution in metres.

    Returns:
        None.
    """
    import hydrolibs.partitionops as partops
    from hydrolibs.rasterops import read_raster_as_arr
    from hydrolibs.sysops import makedirs

    logger.info('=' * 60)
    logger.info('Computing Surface Water Capture Index with σ_GW propagation')
    logger.info('=' * 60)

    # Reference basin raster → valid_mask, raster_shape (same pattern
    # as every other σ_* function in this module)
    ref_basin_file = os.path.join(
        pred_data_dir, f'GW_Basin_{year_list[0]}.tif'
    )
    basin_arr, bfile = read_raster_as_arr(ref_basin_file, get_file=True)
    basin_flat = basin_arr.ravel()
    valid_mask = ~np.isnan(basin_flat) & (basin_flat != 0)
    raster_shape = basin_arr.shape
    bfile.close()
    ref_raster_file = os.path.join(
        pred_data_dir, f'Predictor_{year_list[0]}.tif'
    )

    # Unit conversion factors (identical to pipeline.py)
    pixel_area_m2 = mosaic_res ** 2
    mm_to_ft = 1 / 304.8
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = 1 / 1233.48

    unit_scales = {
        'mm': 1.0,
        'ft': mm_to_ft,
        'm3': mm_to_m3,
        'AF': mm_to_m3 * m3_to_af,
    }
    unit_subdirs = {
        'mm': 'Depth_mm',
        'ft': 'Depth_ft',
        'm3': 'Volume_m3',
        'AF': 'Volume_AF',
    }

    # Map pumping pool → capture-category name.  The capture category
    # names the *captured* SW pool; the pool names the GW pumping that
    # drives it.  Reading the directory layout: ``Total_SW_Capture`` =
    # "fraction of Total_GW pumping that captures surface water".
    pool_to_cap_cat = {
        'Total_GW': 'Total_SW',
        'Irrigation_GW': 'Irrigation_SW',
        'Non_Irrigation_GW': 'Non_Irrigation_SW',
    }

    # All AZ basin names (filter DROP_GW_BASINS — same convention as
    # pipeline.predict_full_period)
    from hydrolibs import partitionops as _partops  # for CATEGORIES
    all_basins = sorted(az_df['GW_Basin'].dropna().unique().tolist())
    # Keep only basins that ever have at least one row with nonzero
    # GW well density (matches predict_full_period's all_basins, minus
    # DROP_GW_BASINS which is a pipeline-level configuration).
    all_basins = [b for b in all_basins if b]
    subbasins = sorted(
        az_df['GW_Subbasin'].dropna().unique().tolist()
        if 'GW_Subbasin' in az_df.columns else []
    )

    sw_cap_dir = os.path.join(prediction_dir, 'SW_Capture')
    makedirs(sw_cap_dir)

    sw_capture_yearly: dict[str, dict] = {}

    for year in range(start_year, end_year + 1):
        year_df = az_df[az_df.Year == year]
        if year_df.empty:
            continue
        if 'wtd_m' not in year_df.columns:
            continue

        wtd_vals = year_df['wtd_m'].values
        cw_sf_vals = (
            year_df['canal_weighted_streamflow_mm'].values
            if 'canal_weighted_streamflow_mm' in year_df.columns
            else np.zeros(len(wtd_vals))
        )

        for pool, cap_cat in pool_to_cap_cat.items():
            # Load prediction (band 1) and σ_total (band 2) from the
            # augmented category Depth_mm raster
            cat_mm_file = os.path.join(
                prediction_dir, f'{pool}_Rasters', 'Depth_mm',
                f'{pool}_{year}_mm.tif',
            )
            if not os.path.exists(cat_mm_file):
                logger.warning(
                    '  Missing %s — skipping %s capture for year %d',
                    cat_mm_file, cap_cat, year,
                )
                continue

            with rio.open(cat_mm_file) as src:
                gw_2d = src.read(1)
                has_sigma = src.count >= 2
                sigma_2d = src.read(2) if has_sigma else None

            gw_flat = gw_2d.ravel()[valid_mask]
            if has_sigma:
                sigma_flat = sigma_2d.ravel()[valid_mask]
                # Floor σ at zero (numerical noise in CI band 5)
                sigma_flat = np.maximum(np.nan_to_num(sigma_flat, nan=0.0),
                                        0.0)
            else:
                sigma_flat = None

            capture = partops.compute_sw_capture_index(
                total_gw=gw_flat,
                sigma_gw=sigma_flat,
                wtd_m=wtd_vals,
                cw_streamflow=cw_sf_vals,
                raster_shape=raster_shape,
                valid_mask=valid_mask,
            )

            # Derive σ_cap at the central λ as the half-width of the
            # propagated 95 % CI.  The asymmetric form of
            # compute_sw_capture_index couples gw ± 1.96 σ with the
            # λ = 5 m (narrow) and λ = 20 m (wide) connectivity
            # scales, so inverting at the central λ (via the full
            # upper − lower range) decomposes cleanly: when σ = 0 the
            # inverted σ_cap reflects only the λ envelope; when σ > 0
            # it adds the pumping-side contribution.
            sigma_cap_flat = (
                0.5 * (capture['Capture_Volume_Upper']
                       - capture['Capture_Volume_Lower']) / CI_Z
            )
            sigma_cap_flat = np.maximum(sigma_cap_flat, 0.0)

            # ── 3-band capture fraction raster (λ = 5/10/20) ──
            frac_dir = os.path.join(
                sw_cap_dir, f'{cap_cat}_Capture_Fraction'
            )
            makedirs(frac_dir)
            cf_grid = np.stack([
                _valid_to_grid(capture[f'Capture_Fraction_{b}'],
                               valid_mask, raster_shape)
                for b in ('Lower', 'Central', 'Upper')
            ])
            with rio.open(ref_raster_file) as ref:
                profile = ref.profile.copy()
            profile.update(count=3, dtype='float32', nodata=np.nan)
            with rio.open(
                os.path.join(frac_dir,
                             f'{cap_cat}_Capture_Fraction_{year}.tif'),
                'w', **profile,
            ) as dst:
                for bi in range(3):
                    dst.write(cf_grid[bi], bi + 1)
                dst.set_band_description(1, 'lower_lambda5m')
                dst.set_band_description(2, 'central_lambda10m')
                dst.set_band_description(3, 'upper_lambda20m')

            # ── 6-band augmented central capture volume rasters in
            # all four units (mm, ft, m³, AF).  σ in band 2 is the
            # σ_cap derived above; bands 3–6 follow the standard
            # (CV, SNR, lower 95 % CI, upper 95 % CI) convention. ──
            cap_central_mm = _valid_to_grid(
                capture['Capture_Volume_Central'], valid_mask,
                raster_shape,
            )
            sigma_cap_mm = _valid_to_grid(
                sigma_cap_flat, valid_mask, raster_shape,
            )

            for unit, subdir in unit_subdirs.items():
                scale = unit_scales[unit]
                out_dir = os.path.join(
                    sw_cap_dir, f'{cap_cat}_Capture_Rasters', subdir,
                )
                makedirs(out_dir)
                out_path = os.path.join(
                    out_dir, f'{cap_cat}_Capture_{year}_{unit}.tif',
                )

                pred_scaled = (cap_central_mm * scale).astype(np.float32)
                sigma_scaled = (sigma_cap_mm * scale).astype(np.float32)

                with rio.open(ref_raster_file) as ref:
                    band_profile = ref.profile.copy()

                band_descriptions = [
                    f'prediction_{unit}', f'sigma_{unit}', 'CV', 'SNR',
                    f'lower_95CI_{unit}', f'upper_95CI_{unit}',
                ]
                _write_augmented_raster(
                    pred_scaled, sigma_scaled, out_path, band_profile,
                    band_descriptions,
                )

            # ── Statewide annual scalars for the time-series CSV ──
            for key in capture:
                ts_key = f'{cap_cat}_{key}'
                if ts_key not in sw_capture_yearly:
                    sw_capture_yearly[ts_key] = {}
                if 'Fraction' in key:
                    # Volume-weighted mean fraction: what fraction of
                    # GW pumping captures surface water statewide.
                    gw_sum = float(np.nansum(gw_flat))
                    cap_vol_key = key.replace(
                        'Capture_Fraction', 'Capture_Volume',
                    )
                    if cap_vol_key in capture and gw_sum > 0:
                        val = (float(np.nansum(capture[cap_vol_key]))
                               / gw_sum)
                    else:
                        val = float(np.nanmean(capture[key]))
                else:
                    # Volume: sum mm depths → m³ → AF
                    val = (float(np.nansum(capture[key]))
                           * mm_to_m3 * m3_to_af)
                sw_capture_yearly[ts_key][year] = val

            # Per-pool σ scalar (AF) for the explicit σ columns
            sigma_scalar_af = (
                float(np.nansum(sigma_cap_flat)) * mm_to_m3 * m3_to_af
            )
            ts_sigma_key = f'{cap_cat}_Capture_Volume_Sigma'
            if ts_sigma_key not in sw_capture_yearly:
                sw_capture_yearly[ts_sigma_key] = {}
            sw_capture_yearly[ts_sigma_key][year] = sigma_scalar_af

            # ── Per-basin and per-sub-basin VW capture fraction
            #    (central λ = 10 m only) ──
            cap_vol_arr = capture['Capture_Volume_Central']
            basins_arr = year_df['GW_Basin'].values
            subbasins_arr = (
                year_df['GW_Subbasin'].values
                if 'GW_Subbasin' in year_df.columns
                else np.full(len(year_df), '', dtype=object)
            )
            for level, zone_arr, zone_list in [
                ('Basin', basins_arr, all_basins),
                ('Subbasin', subbasins_arr, subbasins),
            ]:
                ts_basin_key = f'{cap_cat}_Capture_Fraction_{level}'
                if ts_basin_key not in sw_capture_yearly:
                    sw_capture_yearly[ts_basin_key] = {}
                zone_fracs = {}
                for zone in zone_list:
                    zmask = zone_arr == zone
                    gw_z = float(np.nansum(gw_flat[zmask]))
                    cap_z = float(np.nansum(cap_vol_arr[zmask]))
                    zone_fracs[zone] = cap_z / gw_z if gw_z > 0 else 0.0
                sw_capture_yearly[ts_basin_key][year] = zone_fracs

        if year % 20 == 0 or year == end_year:
            central_af = sw_capture_yearly.get(
                'Total_SW_Capture_Volume_Central', {},
            ).get(year, 0.0)
            sigma_af = sw_capture_yearly.get(
                'Total_SW_Capture_Volume_Sigma', {},
            ).get(year, 0.0)
            logger.info(
                '  Year %d: statewide capture = %.4f MAF, σ_cap = %.4f MAF',
                year, central_af / 1e6, sigma_af / 1e6,
            )

    # ── Write the three CSVs ──
    _write_sw_capture_csvs(sw_cap_dir, sw_capture_yearly)

    logger.info(
        'SW Capture Index with σ_GW propagation complete. Results in %s',
        sw_cap_dir,
    )


def _valid_to_grid(
        values: np.ndarray,
        valid_mask: np.ndarray,
        raster_shape: tuple,
) -> np.ndarray:
    """Map valid-pixel values back to a full 2-D raster grid."""
    grid = np.full(valid_mask.shape[0], np.nan, dtype=np.float32)
    grid[valid_mask] = values.astype(np.float32)
    return grid.reshape(raster_shape)


def _write_sw_capture_csvs(
        sw_cap_dir: str,
        sw_capture_yearly: dict,
) -> None:
    """Write the three SW capture CSVs from the per-year scalar dict.

    Produces:
      - ``SW_Capture_Time_Series.csv`` — statewide per-year scalars
        (fractions, volumes, and explicit σ columns).
      - ``Basin_Capture_Fraction.csv`` — flat per-basin VW fractions.
      - ``Subbasin_Capture_Fraction.csv`` — flat per-sub-basin VW
        fractions.
    """
    if not sw_capture_yearly:
        return

    # Statewide scalars: keys whose values are {year: float} rather
    # than {year: {zone: float}}
    scalar_keys = {
        k: v for k, v in sw_capture_yearly.items()
        if v and not isinstance(next(iter(v.values())), dict)
    }
    if scalar_keys:
        years_present = sorted(next(iter(scalar_keys.values())).keys())
        rows = []
        for y in years_present:
            row = {'Year': y}
            for key, ydict in scalar_keys.items():
                suffix = '' if 'Fraction' in key else '_AF'
                row[f'{key}{suffix}'] = ydict.get(y, 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(sw_cap_dir, 'SW_Capture_Time_Series.csv'),
            index=False,
        )

    # Per-basin and per-sub-basin fraction time series
    for level in ('Basin', 'Subbasin'):
        zone_keys = {
            k: v for k, v in sw_capture_yearly.items()
            if k.endswith(f'_Fraction_{level}') and v
        }
        if not zone_keys:
            continue
        zone_rows = []
        for key, ydict in zone_keys.items():
            cat_label = key.replace(f'_Fraction_{level}', '')
            for y, zone_fracs in sorted(ydict.items()):
                for zone, frac in zone_fracs.items():
                    zone_rows.append({
                        'Year': y,
                        level: zone,
                        'Category': cat_label,
                        'VW_Capture_Fraction': round(frac, 6),
                    })
        if zone_rows:
            pd.DataFrame(zone_rows).to_csv(
                os.path.join(sw_cap_dir, f'{level}_Capture_Fraction.csv'),
                index=False,
            )


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
        """Read Basin/Subbasin_Annual.csv → (yearly, actual_yearly)."""
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
        os.path.join(unc_dir, 'Sigma_Total', 'Uncertainty_Summary_Total.csv'))
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
            os.path.join(unc_dir, 'Sigma_CU', 'Uncertainty_Summary_CU.csv'))
        cu_titles = {
            'Irrigation_CU': 'Irrigation Consumptive Use',
            'Irrigation_GW_CU': 'Irrigation GW Consumptive Use',
            'Irrigation_SW_CU': 'Irrigation SW Consumptive Use',
        }
        for cu_cat, title in cu_titles.items():
            cu_dir = os.path.join(prediction_dir, cu_cat)
            cu_preds, _ = _read_predictions_csv(
                os.path.join(cu_dir, 'Full_Period_Time_Series.csv'))
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
        os.path.join(unc_dir, 'Sigma_Total', 'Basin_Sigma_Total.csv'))
    if basin_sigma:
        basin_yearly, actual_basin = _read_basin_yearly(
            os.path.join(prediction_dir, 'Basin_Time_Series', 'Basin_Annual.csv'),
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
        os.path.join(unc_dir, 'Sigma_Total', 'Subbasin_Sigma_Total.csv'))
    if subbasin_sigma and subbasin_shp and os.path.exists(subbasin_shp):
        subbasin_yearly, actual_subbasin = _read_basin_yearly(
            os.path.join(prediction_dir, 'Subbasin_Time_Series', 'Subbasin_Annual.csv'),
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

    For each raster group (total pumping, categories, CU) the
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

    # 5 mm/yr threshold for "active pumping pixel" mean depth (matches
    # pipeline.py _pixel_stats default).  Volume sums remain over all
    # valid pixels — threshold only filters the Mean_Depth average.
    _ACTIVE_PUMPING_THRESHOLD_MM = 5.0

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

        active = valid & (pred >= _ACTIVE_PUMPING_THRESHOLD_MM)
        if np.any(active):
            mean_mm = float(pred[active].mean())
        else:
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
                if row.geometry is None:
                    continue
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

                active = valid & (pred >= _ACTIVE_PUMPING_THRESHOLD_MM)
                if np.any(active):
                    mean_mm = float(pred[active].mean())
                else:
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
            os.path.join(out_dir, 'Basin_Time_Series', 'Basin_Annual.csv'), 'Basin')
        actual_subbasin = _read_actual_region(
            os.path.join(out_dir, 'Subbasin_Time_Series', 'Subbasin_Annual.csv'),
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

            # ── AMA/INA aggregate time series ──
            from hydrolibs.visualops import get_ama_ina_basin_names
            ama_ina_names = set(get_ama_ina_basin_names())
            ama_preds = {}
            ama_sigma = {}
            pixel_area_m2 = mosaic_res ** 2
            _mm_to_m3 = pixel_area_m2 / 1000
            for year in sorted(basin_yearly.keys()):
                vol_m3 = 0.0
                vol_af = 0.0
                # AMA-aggregate σ via LINEAR SUM across basins.  All
                # σ components (MACA / Model / Irr / LULC / GW / USBR)
                # are scenario-driven via shared drivers (same 5
                # GCMs / 10 model seeds / 5 USBR members perturb every
                # basin), so per-basin σ values are correlated and
                # combine linearly at the AMA aggregate scale.
                # Quadrature here would under-estimate by ~3-4×.
                sigma_af_sum = 0.0
                # Area-weighted depth: total_volume / total_area
                total_npix = 0.0
                for bname, metrics in basin_yearly[year].items():
                    if bname not in ama_ina_names:
                        continue
                    bv_m3 = metrics.get('Volume_m3', 0)
                    bd_mm = metrics.get('Mean_Depth_mm', 0)
                    vol_m3 += bv_m3
                    vol_af += metrics.get('Volume_AF', 0)
                    # Derive pixel count: vol = depth × n_pixels × mm_to_m3
                    if bd_mm > 0 and _mm_to_m3 > 0:
                        total_npix += bv_m3 / (bd_mm * _mm_to_m3)
                    bsig = sigma_basin_yearly.get(year, {}).get(bname, {})
                    sigma_af_sum += bsig.get('Sigma_Volume_AF', 0)
                if vol_af == 0 and vol_m3 == 0:
                    continue
                # Area-weighted mean depth = total_volume / total_area
                mean_depth_mm = (vol_m3 / (total_npix * _mm_to_m3)
                                 if total_npix > 0 else 0)
                ama_preds[year] = {
                    'Mean_Depth_mm': mean_depth_mm,
                    'Mean_Depth_ft': mean_depth_mm * MM_TO_FT,
                    'Volume_m3': vol_m3,
                    'Volume_AF': vol_af,
                }
                s_af = sigma_af_sum
                cv = s_af / abs(vol_af) if vol_af else 0
                ama_sigma[year] = {
                    'Mean_Depth_mm': cv * abs(mean_depth_mm),
                    'Mean_Depth_ft': cv * abs(mean_depth_mm) * MM_TO_FT,
                    'Volume_AF': s_af,
                    'Volume_m3': s_af / M3_TO_AF,
                }
            if ama_preds:
                ama_dir = os.path.join(out_dir, 'AMA_INA_Time_Series')
                makedirs(ama_dir)
                actual_ama = None
                if actual_basin:
                    actual_ama_preds = {}
                    for year, basins in actual_basin.items():
                        a_vol_af = 0.0
                        a_depth = 0.0
                        n = 0
                        for bname, metrics in basins.items():
                            if bname not in ama_ina_names:
                                continue
                            a_vol_af += metrics.get('Volume_AF', 0)
                            a_depth += metrics.get('Mean_Depth_mm', 0)
                            n += 1
                        if n > 0:
                            actual_ama_preds[year] = {
                                'Mean_Depth_mm': a_depth / n,
                                'Volume_AF': a_vol_af,
                            }
                    if actual_ama_preds:
                        actual_ama = actual_ama_preds
                ama_title = f'{title_prefix} ' if title_prefix else ''
                ama_title += 'AMA/INA'
                vizops.create_full_period_time_series(
                    ama_preds, ama_dir,
                    start_year=start_year, end_year=end_year,
                    actual_data=actual_ama,
                    title_prefix=ama_title,
                    sigma_data=ama_sigma,
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

    # ══════════════════════════════════════════════════════════════════════
    # 1. Total annual withdrawal
    # ══════════════════════════════════════════════════════════════════════
    _process_group(
        'Total Annual Withdrawal',
        os.path.join(prediction_dir, 'Predicted_Rasters/Depth_mm'),
        'Total_Predicted_{year}_mm.tif',
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

    logger.info('  All augmented-raster time series with '
                'uncertainty complete.')


def _plot_uncertainty_time_series(
        sigma_components, unc_dir, mosaic_res, vizops,
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
        'GW': 'Well Density (σ_gw)',
        'USBR': 'Upper Basin Streamflow (σ_USBR)',
    }

    for name, comp in sigma_components.items():
        if not comp:
            continue
        comp_dir = os.path.join(unc_dir, f'Sigma_{name}')
        makedirs(comp_dir)
        yearly = {}
        for year in sorted(comp.keys()):
            yearly[year] = _pixel_stats(comp[year], mm_to_m3)

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
    from hydrolibs.visualops import _format_volume_axis
    apply_journal_style()
    plot_dir = os.path.join(comp_dir, 'Plots')
    makedirs(plot_dir)

    for level in ('Basin', 'Subbasin'):
        csv_path = os.path.join(comp_dir, f'{level}_Sigma_{comp_name}.csv')
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
            # Lower CI is clipped at 0 (volumes are non-negative);
            # this is also done at CSV write time but applied here as a
            # belt-and-braces against any pre-clipping CSVs on disk.
            ax1 = axes[0]
            for era, (s, e) in ERA_PERIODS.items():
                ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax1.fill_between(
                years,
                np.maximum(rdf['Lower_95CI_m3'].values, 0),
                rdf['Upper_95CI_m3'].values,
                alpha=0.25, color='#2980B9', label='95 % CI',
            )
            ax1.plot(years, mean_m3, color='#2980B9', linewidth=1.5,
                     marker='.', markersize=2, label='Mean volume')
            _format_volume_axis(ax1, unit='m3', label='Volume')
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
            _format_volume_axis(ax1r, unit='AF', label='Volume')

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
            _format_volume_axis(ax2, unit='m3', label='σ')
            ax2.grid(True, alpha=0.3, linestyle='--')

            ax2r = ax2.twinx()
            ax2r.set_ylim(
                ax2.get_ylim()[0] * M3_TO_AF,
                ax2.get_ylim()[1] * M3_TO_AF,
            )
            _format_volume_axis(ax2r, unit='AF', label='σ')

            ax2.set_xlim(years.min() - 1, years.max() + 1)
            plt.tight_layout()

            safe_name = region.replace(' ', '_').replace('/', '_')
            fig.savefig(
                os.path.join(plot_dir, f'{level}_{safe_name}_Sigma_{comp_name}.png'),
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

        _format_volume_axis(ax1, unit='m3', label=f'σ_{comp_name}')
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
        _format_volume_axis(ax1r, unit='AF', label=f'σ_{comp_name}')

        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('CV', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax1.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f'{level}_All_Sigma_{comp_name}_Summary.png'),
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
        _format_volume_axis,
        apply_journal_style,
    )

    apply_journal_style()
    plot_dir = os.path.join(unc_dir, 'Plots', 'Basin_Sigma')
    makedirs(plot_dir)

    for level in ('Basin', 'Subbasin'):
        csv_path = os.path.join(unc_dir, 'Sigma_Total', f'{level}_Sigma_Total.csv')
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
            # Lower CI clipped at 0 (volumes are non-negative).
            ax1 = axes[0]
            for era, (s, e) in ERA_PERIODS.items():
                ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax1.fill_between(
                years,
                np.maximum(rdf['Lower_95CI_m3'].values, 0),
                rdf['Upper_95CI_m3'].values,
                alpha=0.25, color='#2980B9', label='95 % CI',
            )
            ax1.plot(years, mean_m3, color='#2980B9', linewidth=1.5,
                     marker='.', markersize=2, label='Mean volume')
            _format_volume_axis(ax1, unit='m3', label='Volume')
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
            _format_volume_axis(ax1r, unit='AF', label='Volume')

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
            _format_volume_axis(ax2, unit='m3', label='σ')
            ax2.grid(True, alpha=0.3, linestyle='--')

            ax2r = ax2.twinx()
            ax2r.set_ylim(
                ax2.get_ylim()[0] * M3_TO_AF,
                ax2.get_ylim()[1] * M3_TO_AF,
            )
            _format_volume_axis(ax2r, unit='AF', label='σ')

            ax2.set_xlim(years.min() - 1, years.max() + 1)
            plt.tight_layout()

            safe_name = region.replace(' ', '_').replace('/', '_')
            fig.savefig(
                os.path.join(plot_dir, f'{level}_{safe_name}_Sigma.png'),
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

        _format_volume_axis(ax1, unit='m3', label='σ_total')
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
        _format_volume_axis(ax1r, unit='AF', label='σ_total')

        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('CV', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax1.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f'{level}_All_Sigma_Summary.png'),
            dpi=600, bbox_inches='tight',
        )
        plt.close()

    # ── AZ-wide σ_total time series (sum of basin volumes) ──
    # Aggregation across basins: LINEAR SUM (not quadrature).
    # All σ components (MACA, Model, Irr, LULC, GW, USBR) are
    # scenario-driven and CORRELATED across basins (the same 5 GCMs
    # / 10 model seeds / 5 USBR members perturb every basin), so
    # basin-σ values move together rather than independently.  Linear
    # sum is the correct AZ-total under that correlation.
    # Basin-quadrature (sqrt(Σ σ_basin²)) under-estimates AZ-total σ
    # by ~3-4× because it assumes basin independence — produced
    # ribbons that didn't cover USGS at peak years (~8.5 MAF cap).
    basin_csv = os.path.join(unc_dir, 'Sigma_Total', 'Basin_Sigma_Total.csv')
    if os.path.exists(basin_csv):
        bdf = pd.read_csv(basin_csv)
        if not bdf.empty:
            az_df = bdf.groupby('Year').agg(
                Mean_Volume_m3=('Mean_Volume_m3', 'sum'),
                Mean_Volume_AF=('Mean_Volume_AF', 'sum'),
                Sigma_Total_m3=('Sigma_Total_m3', 'sum'),
                Sigma_Total_AF=('Sigma_Total_AF', 'sum'),
            ).reset_index().sort_values('Year')

            # Withdrawal volumes are physically non-negative; clip the
            # 95% CI lower bound at 0.
            az_df['Lower_95CI_m3'] = (
                az_df['Mean_Volume_m3'] - CI_Z * az_df['Sigma_Total_m3']
            ).clip(lower=0)
            az_df['Upper_95CI_m3'] = az_df['Mean_Volume_m3'] + CI_Z * az_df['Sigma_Total_m3']
            az_df['Lower_95CI_AF'] = (
                az_df['Mean_Volume_AF'] - CI_Z * az_df['Sigma_Total_AF']
            ).clip(lower=0)
            az_df['Upper_95CI_AF'] = az_df['Mean_Volume_AF'] + CI_Z * az_df['Sigma_Total_AF']

            years = az_df['Year'].values
            mean_m3 = az_df['Mean_Volume_m3'].values
            sigma_m3 = az_df['Sigma_Total_m3'].values

            fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

            # --- Panel 1: Mean volume with 95% CI ---
            # Lower CI clipped at 0 (volumes are non-negative).
            ax1 = axes[0]
            for era, (s, e) in ERA_PERIODS.items():
                ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
            ax1.fill_between(
                years,
                np.maximum(az_df['Lower_95CI_m3'].values, 0),
                az_df['Upper_95CI_m3'].values,
                alpha=0.25, color='#2980B9', label='95 % CI',
            )
            ax1.plot(years, mean_m3, color='#2980B9', linewidth=1.5,
                     marker='.', markersize=2, label='Mean volume')
            _format_volume_axis(ax1, unit='m3', label='Volume')
            ax1.set_title(
                'Arizona \u2014 Mean Prediction \u00b1 95 % CI',
                fontweight='bold', fontsize=14,
            )
            ax1.grid(True, alpha=0.3, linestyle='--')

            ax1r = ax1.twinx()
            ax1r.set_ylim(
                ax1.get_ylim()[0] * M3_TO_AF,
                ax1.get_ylim()[1] * M3_TO_AF,
            )
            _format_volume_axis(ax1r, unit='AF', label='Volume')

            handles1 = ax1.get_legend_handles_labels()[0]
            era_handles = [
                mpatches.Patch(
                    color=ERA_COLORS[e], alpha=0.4,
                    label=f'{e} ({ERA_PERIODS[e][0]}\u2013{ERA_PERIODS[e][1]})',
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
                     marker='.', markersize=2, label='\u03c3_total')
            ax2.set_xlabel('Year', fontweight='bold')
            _format_volume_axis(ax2, unit='m3', label='\u03c3')
            ax2.grid(True, alpha=0.3, linestyle='--')

            ax2r = ax2.twinx()
            ax2r.set_ylim(
                ax2.get_ylim()[0] * M3_TO_AF,
                ax2.get_ylim()[1] * M3_TO_AF,
            )
            _format_volume_axis(ax2r, unit='AF', label='\u03c3')

            ax2.set_xlim(years.min() - 1, years.max() + 1)
            plt.tight_layout()

            fig.savefig(
                os.path.join(plot_dir, 'Arizona_Sigma_Total.png'),
                dpi=600, bbox_inches='tight',
            )
            plt.close()
            logger.info(f'  AZ-wide σ_total time-series plot saved to {plot_dir}')

    logger.info(f'  Basin/sub-basin σ time-series plots saved to {plot_dir}')
