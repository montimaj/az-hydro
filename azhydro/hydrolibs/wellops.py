"""
Well-level withdrawal package generation.

Samples predicted withdrawal rasters at well locations from the ADWR Well
Registry and produces a GeoPackage with annual per-well values across all
withdrawal categories (Total, Irrigation, Non-Irrigation, GW/SW splits)
and consumptive-use categories (Irrigation_CU, Irrigation_GW_CU,
Irrigation_SW_CU).

When augmented 6-band rasters are available (after the UQ step), the
package also includes per-well uncertainty metrics (σ, lower/upper 95 % CI)
for every category.  Pixel-level σ (band 2) is distributed to wells using
the same capacity-proportional weight as the prediction itself:

    well_σ = pixel_σ × well_share

This is a simplification: it assumes within-pixel uncertainty distributes
proportionally to capacity weight.  True per-well uncertainty would
require well-specific error models, which are beyond the scope of this
dataset.

Distribution logic
-------------------
Each raster pixel stores the **total** withdrawal for that pixel.  When
multiple wells share a pixel the value is split among them using a
capacity-based weight:

1. **Historical pumping weight** — mean ``AF Pumped`` across all years a
   well appears in the per-year GW shapefiles (``gw_vector_dir``).
2. **Pump-rate fallback** — for wells with no history, the ``PUMPRATE``
   field from the Well Registry is used.
3. **Equal-share fallback** — wells with neither get weight 1.0, so that
   within a pixel the split is capacity-proportional where data exist and
   equal otherwise.

Temporal filtering
------------------
For each year, only wells that existed by that year are included in the
disaggregation.  A well's start year is determined by:

1. ``INSTALLED`` date (year extracted).
2. ``APPLICATIO`` date fallback if ``INSTALLED`` is missing/invalid.
3. Conservative default (``start_year``) if both are missing — the well
   is included for all years.

Capacity weights are re-normalised per year within each pixel using only
the active wells, so the pixel total is always fully distributed.

Wells that fall in raster nodata pixels are excluded before weighting.
All sampled values are floored at zero.
"""

import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from hydrolibs.partitionops import CATEGORIES
from hydrolibs.sysops import NON_CONSUMPTIVE_USES, derive_well_start_year, makedirs

logger = logging.getLogger(__name__)

# Unit conversion constants
_MM_TO_FT = 1 / 304.8
_M3_TO_AF = 1 / 1233.48


# ---------------------------------------------------------------------------
# Capacity-weight helpers
# ---------------------------------------------------------------------------

def _compute_capacity_weights(
        wells: gpd.GeoDataFrame,
        pixel_keys: np.ndarray,
        gw_vector_dir: str | None,
        normalize: bool = True,
) -> np.ndarray:
    """
    Compute per-well weights so that wells sharing a pixel receive a
    capacity-proportional share of the pixel total.

    Priority:
        1. Mean historical ``AF Pumped`` across all available GW years.
        2. ``PUMPRATE`` from the Well Registry (GPM field).
        3. Equal-share fallback (weight = 1.0).

    Args:
        normalize: If True (default), normalise weights within each pixel
            to sum to 1.  If False, return the raw (un-normalised) weights
            so the caller can normalise per year using only active wells.
    """
    n = len(wells)
    raw_weight = np.ones(n, dtype=np.float64)   # fallback

    # --- Historical pumping ---
    hist_mean = np.full(n, np.nan, dtype=np.float64)
    if gw_vector_dir is not None:
        gw_dir = Path(gw_vector_dir)
        shp_files = sorted(gw_dir.glob('GW_*.shp'))
        if shp_files:
            reg_ids = wells['REGISTRY_I'].values
            id_to_idx = {rid: i for i, rid in enumerate(reg_ids)}
            accum = np.zeros(n, dtype=np.float64)
            count = np.zeros(n, dtype=np.int32)
            for shp in shp_files:
                gdf = gpd.read_file(shp, include_fields=['REGISTRY_I', 'AF Pumped'])
                valid = gdf['AF Pumped'].notnull() & (gdf['AF Pumped'] > 0)
                gdf = gdf[valid]
                for rid, af in zip(gdf['REGISTRY_I'].values,
                                   gdf['AF Pumped'].values):
                    idx = id_to_idx.get(rid)
                    if idx is not None:
                        accum[idx] += af
                        count[idx] += 1
            has_hist = count > 0
            hist_mean[has_hist] = accum[has_hist] / count[has_hist]
            n_hist = has_hist.sum()
            logger.info(f'  Capacity weights: {n_hist} wells with '
                        f'historical pumping from {len(shp_files)} years')

    has_hist = np.isfinite(hist_mean)
    raw_weight[has_hist] = hist_mean[has_hist]

    # --- PUMPRATE fallback ---
    if 'PUMPRATE' in wells.columns:
        pump = pd.to_numeric(wells['PUMPRATE'], errors='coerce').values
        use_pump = ~has_hist & np.isfinite(pump) & (pump > 0)
        # Convert GPM to AF/yr to match the AF/yr units of historical pumping
        raw_weight[use_pump] = pump[use_pump] * 1.6133
        logger.info(f'  Capacity weights: {use_pump.sum()} wells using '
                    f'PUMPRATE fallback')

    if not normalize:
        return raw_weight

    # --- Normalise within each pixel ---
    unique_keys, inverse = np.unique(pixel_keys, return_inverse=True)
    well_share = np.empty(n, dtype=np.float64)
    for ui, uk in enumerate(unique_keys):
        mask = inverse == ui
        w = raw_weight[mask]
        well_share[mask] = w / w.sum()

    return well_share


def create_well_package(
        well_registry_file: str,
        raster_dirs: dict[str, str],
        cat_raster_dirs: dict[str, dict[str, str]],
        output_dir: str,
        ref_raster_file: str,
        pixel_area_m2: float = 2000 ** 2,
        start_year: int = 1896,
        end_year: int = 2099,
        water_use: str | None = None,
        gw_vector_dir: str | None = None,
        cu_raster_dirs: dict[str, dict[str, str]] | None = None,
) -> str:
    """
    Sample all withdrawal rasters at well locations and write a GeoPackage.

    Only the **mm** rasters are read; ft, m³, and acre-ft values are
    computed arithmetically, reducing I/O by 75 %.

    When called after the UQ augmentation step, the mm rasters are 6-band
    GeoTIFFs.  Band 1 is the prediction and band 2 is σ.  Both are sampled
    and weighted identically, producing per-well uncertainty columns
    (``{Cat}_{unit}_sigma``, ``{Cat}_{unit}_ci_lower``,
    ``{Cat}_{unit}_ci_upper``).

    Args:
        well_registry_file (str): Path to the (reprojected) ADWR Well Registry shapefile.
        raster_dirs (dict): ``{'mm': dir, 'ft': dir, 'm3': dir, 'AF': dir}`` for total
            pumping rasters.
        cat_raster_dirs (dict): ``{category: {'mm': dir, ...}, ...}`` for each of the 8
            partitioned categories.
        output_dir (str): Directory for the output GeoPackage.
        ref_raster_file (str): Path to a reference raster used to map well coordinates to pixel
            indices (must share the same grid as the withdrawal rasters).
        pixel_area_m2 (float): Area of one raster pixel in m².  Default 2000² = 4 000 000.
        start_year (int): Start of the year range to process.
        end_year (int): End of the year range to process.
        water_use (str or None): If set, filter wells by ``WATER_USE`` attribute before sampling.
        gw_vector_dir (str or None): Directory containing per-year GW shapefiles (``GW_YYYY.shp``)
            with ``AF Pumped`` column.  Used to build capacity-proportional
            weights.  If *None*, falls back to ``PUMPRATE`` and then
            equal-share.
        cu_raster_dirs (dict or None): ``{cu_category: {'mm': dir, ...}, ...}`` for the 3
            CU categories.  If *None*, CU columns are omitted.

    Returns:
        str: Path to the written GeoPackage file.
    """
    makedirs(output_dir)
    out_gpkg = os.path.join(output_dir, 'Well_Package.gpkg')

    # ---- Load well registry ----
    wells = gpd.read_file(well_registry_file)
    # Drop non-consumptive wells (monitoring, test, dewatering, etc.)
    pattern = '|'.join(NON_CONSUMPTIVE_USES)
    has_use = wells['WATER_USE'].notna()
    wells = wells[has_use & ~wells['WATER_USE'].str.contains(pattern, na=False)]
    if water_use:
        wells = wells[wells['WATER_USE'] == water_use]
    wells = wells[wells.geometry.notnull()].copy()
    wells = wells.reset_index(drop=True)
    logger.info(f'Well package: {len(wells)} wells loaded from registry')

    coords = np.column_stack((wells.geometry.x.values,
                              wells.geometry.y.values))

    # ---- Map wells to pixel indices (vectorised) ----
    with rio.open(ref_raster_file) as src:
        transform_inv = ~src.transform          # inverse affine
        cols, rows = transform_inv * (coords[:, 0], coords[:, 1])
        pixel_rc = np.column_stack((rows.astype(int), cols.astype(int)))
        # Mask wells in nodata pixels
        raster_mask = src.read_masks(1)
        n_rows, n_cols = raster_mask.shape
        in_bounds = (
            (pixel_rc[:, 0] >= 0) & (pixel_rc[:, 0] < n_rows) &
            (pixel_rc[:, 1] >= 0) & (pixel_rc[:, 1] < n_cols)
        )
        valid_pixel = np.zeros(len(wells), dtype=bool)
        valid_pixel[in_bounds] = (
            raster_mask[pixel_rc[in_bounds, 0], pixel_rc[in_bounds, 1]] > 0
        )

    n_dropped = (~valid_pixel).sum()
    if n_dropped:
        logger.info(f'  Dropped {n_dropped} wells in nodata / out-of-bounds pixels')
    wells = wells[valid_pixel].reset_index(drop=True)
    coords = coords[valid_pixel]
    pixel_rc = pixel_rc[valid_pixel]
    n_wells = len(wells)

    # ---- Pixel keys for grouping ----
    # Encode (row, col) as a single int: row * 1M + col.
    # This requires the grid to have fewer than 1M columns to avoid collisions.
    n_cols = raster_mask.shape[1] if raster_mask is not None else int(pixel_rc[:, 1].max()) + 1
    if n_cols >= 1_000_000:
        raise ValueError(
            f'Raster has {n_cols} columns (≥ 1M). Pixel key encoding '
            f'(row * 1_000_000 + col) would produce collisions. '
            f'Increase the multiplier or switch to tuple keys.'
        )
    pixel_keys = pixel_rc[:, 0] * 1_000_000 + pixel_rc[:, 1]

    # ---- Derive well start year (INSTALLED → APPLICATIO → conservative) ----
    well_start_year = derive_well_start_year(wells, default_year=start_year)

    # ---- Base capacity weights (before year-filtering) ----
    raw_weight = _compute_capacity_weights(wells, pixel_keys, gw_vector_dir,
                                           normalize=False)
    unique_keys, inverse = np.unique(pixel_keys, return_inverse=True)
    logger.info(f'  Unique pixels occupied: {len(unique_keys)}, '
                f'wells retained: {n_wells}')

    # ---- Unit conversion factors ----
    mm_to_m3 = pixel_area_m2 / 1000

    # ---- Pre-allocate output array ----
    cat_mm_info = [('Total', raster_dirs['mm'], 'Total_Predicted')]
    for cat in CATEGORIES:
        cat_mm_info.append((cat, cat_raster_dirs[cat]['mm'], cat))
    if cu_raster_dirs:
        for cu_cat, unit_dirs in cu_raster_dirs.items():
            cat_mm_info.append((cu_cat, unit_dirs['mm'], cu_cat))

    n_cats = len(cat_mm_info)  # 9 (withdrawal) + up to 3 (CU)
    n_years = end_year - start_year + 1

    all_mm = np.full((n_years, n_wells, n_cats), np.nan, dtype=np.float64)
    all_sigma_mm = np.full((n_years, n_wells, n_cats), np.nan, dtype=np.float64)
    has_sigma = False

    years_sampled = []
    for yi, year in enumerate(range(start_year, end_year + 1)):
        # Only include wells installed by this year
        active = well_start_year <= year
        if not active.any():
            continue

        # Per-year capacity weights: normalise within each pixel
        # using only active wells
        well_share = np.zeros(n_wells, dtype=np.float64)
        for ui in range(len(unique_keys)):
            pix_mask = inverse == ui
            pix_active = pix_mask & active
            if not pix_active.any():
                continue
            w = raw_weight[pix_active]
            well_share[pix_active] = w / w.sum()

        sampled_any = False
        for ci, (cat, mm_dir, prefix) in enumerate(cat_mm_info):
            raster_path = os.path.join(mm_dir, f'{prefix}_{year}_mm.tif')
            try:
                with rio.open(raster_path) as src:
                    vals = np.array(
                        list(src.sample(coords, indexes=1)),
                        dtype=np.float64,
                    ).ravel()
                    all_mm[yi, :, ci] = vals * well_share
                    sampled_any = True
                    # Read σ from band 2 if augmented (6-band) raster
                    if src.count >= 6:
                        sigma_vals = np.array(
                            list(src.sample(coords, indexes=2)),
                            dtype=np.float64,
                        ).ravel()
                        all_sigma_mm[yi, :, ci] = sigma_vals * well_share
                        has_sigma = True
            except FileNotFoundError:
                pass
        if sampled_any:
            years_sampled.append(year)
        if year % 50 == 0 or year == end_year:
            logger.info(f'  Well package sampled through {year}')

    if not years_sampled:
        logger.warning('No raster data found for well package.')
        return out_gpkg

    if has_sigma:
        logger.info('  Uncertainty bands detected — including σ and 95%% CI columns')

    # ---- Floor at zero (remove negative model artifacts) ----
    np.maximum(all_mm, 0, out=all_mm)
    np.maximum(all_sigma_mm, 0, out=all_sigma_mm)

    # ---- Build DataFrame ----
    year_indices = [y - start_year for y in years_sampled]
    mm_data = all_mm[year_indices]  # (n_sampled_years, n_wells, n_cats)
    sigma_mm_data = all_sigma_mm[year_indices]

    n_sampled = len(years_sampled)
    total_rows = n_sampled * n_wells

    # Flatten: year-major order (year0-well0, year0-well1, ...)
    mm_flat = mm_data.reshape(total_rows, n_cats)
    sigma_mm_flat = sigma_mm_data.reshape(total_rows, n_cats)

    col_names = [info[0] for info in cat_mm_info]  # Total, Irrigation, ...
    data = {}
    for ci, cat in enumerate(col_names):
        mm_col = mm_flat[:, ci]
        data[f'{cat}_mm'] = mm_col
        data[f'{cat}_ft'] = mm_col * _MM_TO_FT
        data[f'{cat}_m3'] = mm_col * mm_to_m3
        data[f'{cat}_AF'] = mm_col * mm_to_m3 * _M3_TO_AF

        if has_sigma:
            s_mm = sigma_mm_flat[:, ci]
            data[f'{cat}_mm_sigma'] = s_mm
            data[f'{cat}_ft_sigma'] = s_mm * _MM_TO_FT
            data[f'{cat}_m3_sigma'] = s_mm * mm_to_m3
            data[f'{cat}_AF_sigma'] = s_mm * mm_to_m3 * _M3_TO_AF
            # 95% CI
            for unit, scale in (('mm', 1.0), ('ft', _MM_TO_FT),
                                ('m3', mm_to_m3), ('AF', mm_to_m3 * _M3_TO_AF)):
                s_u = s_mm * scale
                pred_u = data[f'{cat}_{unit}']
                data[f'{cat}_{unit}_ci_lower'] = np.maximum(pred_u - 1.96 * s_u, 0)
                data[f'{cat}_{unit}_ci_upper'] = pred_u + 1.96 * s_u

    data['Year'] = np.repeat(years_sampled, n_wells)

    df = pd.DataFrame(data)

    # Tile well attributes for all years
    df['REGISTRY_I'] = np.tile(wells['REGISTRY_I'].values, n_sampled)
    if 'WATER_USE' in wells.columns:
        df['WATER_USE'] = np.tile(wells['WATER_USE'].values, n_sampled)
    geom = np.tile(wells.geometry.values, n_sampled)

    result = gpd.GeoDataFrame(df, geometry=geom, crs=wells.crs)

    # Reorder columns: ID, year, value columns, geometry
    id_cols = [c for c in ('REGISTRY_I', 'WATER_USE', 'Year') if c in result.columns]
    val_cols = sorted(c for c in result.columns if c not in id_cols + ['geometry'])
    result = result[id_cols + val_cols + ['geometry']]

    result.to_file(out_gpkg, driver='GPKG', layer='well_withdrawals')
    n_val_cols = len([c for c in result.columns if c not in
                      ('REGISTRY_I', 'WATER_USE', 'Year', 'geometry')])
    logger.info(f'Well package written: {out_gpkg}  '
                f'({n_wells} wells × {n_sampled} years, '
                f'{n_val_cols} value columns'
                f'{", incl. σ/CI" if has_sigma else ""})')
    return out_gpkg
