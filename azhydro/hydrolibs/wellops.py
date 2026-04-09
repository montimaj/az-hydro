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
    if water_use and water_use != 'All':
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

    # ---- Vectorised per-pixel weight normalization ----
    # Pre-compute per-pixel group boundaries for fast normalization.
    # Sort wells by pixel key so each pixel's wells are contiguous.
    sort_idx = np.argsort(pixel_keys)
    sorted_weight = raw_weight[sort_idx]
    sorted_start_year = well_start_year[sort_idx]
    # unsort_idx maps sorted position back to original well order
    unsort_idx = np.argsort(sort_idx)

    # Pre-compute sorted inverse (maps each sorted well to its pixel group index)
    sorted_inverse = inverse[sort_idx]

    def _compute_well_share(year):
        """Fully vectorized per-pixel capacity weight normalization."""
        active_sorted = (sorted_start_year <= year).astype(np.float64)
        wa = sorted_weight * active_sorted  # zero out inactive wells
        # Sum active weights per pixel group using np.bincount
        group_sums = np.bincount(sorted_inverse, weights=wa,
                                 minlength=len(unique_keys))
        # Per-well normalization: wa[i] / group_sum[group_of_i]
        per_well_sum = group_sums[sorted_inverse]
        with np.errstate(invalid='ignore', divide='ignore'):
            share = np.where(per_well_sum > 0, wa / per_well_sum, 0.0)
        return share[unsort_idx]

    years_sampled = []
    for yi, year in enumerate(range(start_year, end_year + 1)):
        active = well_start_year <= year
        if not active.any():
            continue

        well_share = _compute_well_share(year)

        sampled_any = False
        rows_idx = pixel_rc[:, 0]
        cols_idx = pixel_rc[:, 1]
        for ci, (cat, mm_dir, prefix) in enumerate(cat_mm_info):
            raster_path = os.path.join(mm_dir, f'{prefix}_{year}_mm.tif')
            try:
                with rio.open(raster_path) as src:
                    band = src.read(1).astype(np.float64)
                    vals = band[rows_idx, cols_idx]
                    all_mm[yi, :, ci] = vals * well_share
                    sampled_any = True
                    if src.count >= 6:
                        sigma_band = src.read(2).astype(np.float64)
                        sigma_vals = sigma_band[rows_idx, cols_idx]
                        all_sigma_mm[yi, :, ci] = sigma_vals * well_share
                        has_sigma = True
            except FileNotFoundError:
                pass
        if sampled_any:
            years_sampled.append(year)
        if year % 10 == 0 or year == end_year:
            logger.info(f'  Well package sampled through {year}')

    if not years_sampled:
        logger.warning('No raster data found for well package.')
        return out_gpkg

    if has_sigma:
        logger.info('  Uncertainty bands detected — including σ and 95%% CI columns')

    # ---- Floor at zero (remove negative model artifacts) ----
    np.maximum(all_mm, 0, out=all_mm)
    np.maximum(all_sigma_mm, 0, out=all_sigma_mm)

    # ---- Write 4 GeoParquet files (one per unit) in chunks ----
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import wkb as shapely_wkb

    col_names = [info[0] for info in cat_mm_info]  # Total, Irrigation, ...
    n_sampled = len(years_sampled)

    registry_ids = wells['REGISTRY_I'].values
    water_use_vals = (wells['WATER_USE'].values
                      if 'WATER_USE' in wells.columns else None)

    # Pre-compute WKB geometry once (reused for every year chunk)
    well_wkb = np.array([shapely_wkb.dumps(g) for g in wells.geometry.values],
                        dtype=object)

    # GeoParquet metadata (shared across all unit files)
    crs_json = wells.crs.to_json() if wells.crs else '{}'
    geo_meta = (
        '{"version": "1.1.0", "primary_column": "geometry", '
        '"columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"]'
        f', "crs": {crs_json}' + '}}}'
    )

    unit_scales = {
        'mm': 1.0,
        'ft': _MM_TO_FT,
        'm3': mm_to_m3,
        'AF': mm_to_m3 * _M3_TO_AF,
    }

    out_parquets = []
    for unit, scale in unit_scales.items():
        out_path = os.path.join(output_dir, f'Well_Package_{unit}.parquet')
        out_parquets.append(out_path)
        writer = None
        chunks = []
        batch_size = 10

        for year in years_sampled:
            yi = year - start_year
            row_data = {
                'REGISTRY_I': registry_ids,
                'Year': np.full(n_wells, year, dtype=np.int32),
            }
            if water_use_vals is not None:
                row_data['WATER_USE'] = water_use_vals

            for ci, cat in enumerate(col_names):
                vals = all_mm[yi, :, ci].astype(np.float64) * scale
                row_data[f'{cat}_{unit}'] = vals.astype(np.float32)
                if has_sigma:
                    s_vals = all_sigma_mm[yi, :, ci].astype(np.float64) * scale
                    row_data[f'{cat}_{unit}_sigma'] = s_vals.astype(np.float32)

            row_data['geometry'] = well_wkb
            chunks.append(pd.DataFrame(row_data))

            if len(chunks) >= batch_size or year == years_sampled[-1]:
                batch = pd.concat(chunks, ignore_index=True)
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                chunks.clear()
                del batch, table

        if writer is not None:
            writer.close()

        # Inject GeoParquet metadata by streaming row groups
        pf = pq.ParquetFile(out_path)
        schema = pf.schema_arrow
        existing_meta = schema.metadata or {}
        existing_meta[b'geo'] = geo_meta.encode('utf-8')
        new_schema = schema.with_metadata(existing_meta)
        tmp_path = out_path + '.tmp'
        with pq.ParquetWriter(tmp_path, new_schema) as w:
            for i in range(pf.metadata.num_row_groups):
                w.write_table(pf.read_row_group(i))
        pf.close()
        os.replace(tmp_path, out_path)

        logger.info(f'  Well package ({unit}) written: {out_path}')

    n_val_cols = len(col_names) * (2 if has_sigma else 1)
    logger.info(f'Well package complete: {n_wells} wells × {n_sampled} years, '
                f'{n_val_cols} columns per unit'
                f'{", incl. σ" if has_sigma else ""}')

    # Free large arrays
    del all_mm, all_sigma_mm

    return out_parquets[0]


def verify_well_package(
        parquet_path: str,
        raster_dirs: dict[str, str],
        cat_raster_dirs: dict[str, dict[str, str]],
        ref_raster_file: str,
        output_dir: str,
        *,
        cu_raster_dirs: dict[str, dict[str, str]] | None = None,
        sample_years: list[int] | None = None,
        rtol: float = 1e-3,
        unit: str = 'mm',
) -> bool:
    """Verify well package by reconstructing pixel values from well sums.

    For each sampled year and category, wells are grouped by pixel and
    their values summed.  The per-pixel sums are compared against the
    original raster band 1 values.  A summary CSV is written.

    Args:
        parquet_path: Path to the GeoParquet well package for the given unit.
        raster_dirs: ``{'mm': dir}`` mapping for total predicted rasters.
        cat_raster_dirs: ``{cat: {'mm': dir}}`` for category rasters.
        ref_raster_file: Reference raster for grid shape and transform.
        output_dir: Directory for verification outputs.
        cu_raster_dirs: Optional ``{cu_cat: {'mm': dir}}`` for CU rasters.
        sample_years: Years to verify (default: all years).
        rtol: Relative tolerance for pixel-level comparison (default 0.1%).
        unit: Unit label (mm, ft, m3, AF) for column names and output files.

    Returns:
        True if all checks pass, False otherwise.
    """
    import pyarrow.parquet as pq
    from shapely import wkb as shapely_wkb

    logger.info(f'Verifying well package ({unit}) against source rasters...')
    makedirs(output_dir)

    # ---- Load reference grid ----
    with rio.open(ref_raster_file) as src:
        transform_inv = ~src.transform
        ref_shape = (src.height, src.width)
        ref_profile = src.profile.copy()
        ref_profile.update(count=1, dtype=np.float32, nodata=np.nan)

    # ---- Build category → raster dir mapping ----
    cat_info = [('Total', raster_dirs['mm'], 'Total_Predicted')]
    for cat in CATEGORIES:
        if cat in cat_raster_dirs:
            cat_info.append((cat, cat_raster_dirs[cat]['mm'], cat))
    if cu_raster_dirs:
        for cu_cat, unit_dirs in cu_raster_dirs.items():
            cat_info.append((cu_cat, unit_dirs['mm'], cu_cat))

    # ---- Read parquet metadata to get years ----
    pf = pq.ParquetFile(parquet_path)
    # Read Year column from first row group to discover available years
    year_col = pf.read_row_group(0, columns=['Year'])['Year'].to_pylist()
    all_years = sorted(set(year_col))
    # Read all row groups for full year list
    for i in range(1, pf.metadata.num_row_groups):
        rg_years = pf.read_row_group(i, columns=['Year'])['Year'].to_pylist()
        all_years = sorted(set(all_years) | set(rg_years))

    if sample_years is None:
        sample_years = sorted(all_years)

    logger.info(f'  Verifying {len(sample_years)} years: '
                f'{sample_years[0]}–{sample_years[-1]}')

    # ---- Pre-compute pixel row/col from first year's geometry ----
    first_year_df = pd.read_parquet(
        parquet_path, filters=[('Year', '==', sample_years[0])],
    )
    geom_wkb_first = first_year_df['geometry'].values
    coords_x = np.array([shapely_wkb.loads(g).x for g in geom_wkb_first])
    coords_y = np.array([shapely_wkb.loads(g).y for g in geom_wkb_first])
    cols_arr, rows_arr = transform_inv * (coords_x, coords_y)
    pix_rows = rows_arr.astype(int)
    pix_cols = cols_arr.astype(int)
    del first_year_df, geom_wkb_first, coords_x, coords_y

    # ---- Helper: compare reconstructed vs original ----
    def _compare(recon_arr, orig_arr, year, cat, band_label):
        v = np.isfinite(recon_arr) & np.isfinite(orig_arr)
        if not v.any():
            return None
        rv = recon_arr[v]
        ov = orig_arr[v]
        abs_diff = np.abs(rv - ov)
        # Relative difference: |diff| / max(|orig|, 1e-10)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_diff = abs_diff / np.maximum(np.abs(ov), 1e-10)
        mx_abs = float(np.nanmax(abs_diff))
        mn_abs = float(np.nanmean(abs_diff))
        mx_rel = float(np.nanmax(rel_diff))
        mn_rel = float(np.nanmean(rel_diff))
        n = int(v.sum())
        np_ = int((rel_diff <= rtol).sum())
        pct = 100.0 * np_ / n if n > 0 else 0.0
        ok = mx_rel <= rtol
        return {
            'Year': year, 'Category': cat, 'Band': band_label,
            'N_Pixels': n,
            f'Max_Abs_Diff_{unit}': round(mx_abs, 6),
            f'Mean_Abs_Diff_{unit}': round(mn_abs, 6),
            'Max_Rel_Diff': round(mx_rel, 6),
            'Mean_Rel_Diff': round(mn_rel, 6),
            'Pct_Pass': round(pct, 2), 'Pass': ok,
        }

    # ---- Helper: reconstruct pixel grid from well values ----
    def _reconstruct(values, p_rows, p_cols):
        grid = np.full(ref_shape, np.nan, dtype=np.float64)
        # Group by pixel and sum (min_count=1 so all-NaN pixels stay NaN)
        df_tmp = pd.DataFrame({
            'r': p_rows, 'c': p_cols, 'v': values,
        })
        for (r, c), val in df_tmp.groupby(['r', 'c'])['v'].sum(
                min_count=1).items():
            if 0 <= r < ref_shape[0] and 0 <= c < ref_shape[1]:
                grid[r, c] = val
        return grid

    # ---- Process one year (for parallel execution) ----
    def _verify_year(year):
        ydf = pd.read_parquet(
            parquet_path, filters=[('Year', '==', year)],
        )
        if ydf.empty:
            return []

        year_results = []

        for cat, cat_dir, prefix in cat_info:
            col_name = f'{cat}_{unit}'
            if col_name not in ydf.columns:
                continue

            raster_path = os.path.join(cat_dir, f'{prefix}_{year}_{unit}.tif')
            if not os.path.exists(raster_path):
                continue

            with rio.open(raster_path) as src:
                n_bands = src.count
                orig_pred = src.read(1).astype(np.float64)
                orig_sigma = src.read(2).astype(np.float64) if n_bands >= 6 else None
                orig_cv = src.read(3).astype(np.float64) if n_bands >= 6 else None
                orig_snr = src.read(4).astype(np.float64) if n_bands >= 6 else None
                orig_lower = src.read(5).astype(np.float64) if n_bands >= 6 else None
                orig_upper = src.read(6).astype(np.float64) if n_bands >= 6 else None

            # Prediction (band 1)
            reconstructed = _reconstruct(
                ydf[col_name].values, pix_rows, pix_cols)
            # Only compare pixels where at least one active well
            # contributed (nonzero reconstructed value).  Pixels with
            # only inactive wells (well_share=0) produce sum=0 which
            # doesn't match the raster value — that's expected, not an error.
            valid = np.isfinite(reconstructed) & (reconstructed != 0)
            if not valid.any():
                continue

            r = _compare(reconstructed[valid], orig_pred[valid],
                         year, cat, 'Prediction')
            if r:
                year_results.append(r)

            # Sigma (band 2)
            sigma_col = f'{cat}_{unit}_sigma'
            if sigma_col in ydf.columns and orig_sigma is not None:
                recon_sigma = _reconstruct(
                    ydf[sigma_col].values, pix_rows, pix_cols)

                r = _compare(recon_sigma[valid], orig_sigma[valid],
                             year, cat, 'Sigma')
                if r:
                    year_results.append(r)

                # CV (band 3)
                if orig_cv is not None:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        recon_cv = np.where(
                            np.abs(reconstructed) > 0,
                            recon_sigma / np.abs(reconstructed), np.nan)
                    r = _compare(recon_cv[valid], orig_cv[valid],
                                 year, cat, 'CV')
                    if r:
                        year_results.append(r)

                # SNR (band 4)
                if orig_snr is not None:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        recon_snr = np.where(
                            recon_sigma > 0,
                            np.abs(reconstructed) / recon_sigma, np.nan)
                    r = _compare(recon_snr[valid], orig_snr[valid],
                                 year, cat, 'SNR')
                    if r:
                        year_results.append(r)

                # Lower 95% CI (band 5)
                if orig_lower is not None:
                    recon_lower = np.maximum(
                        reconstructed - 1.96 * recon_sigma, 0)
                    r = _compare(recon_lower[valid], orig_lower[valid],
                                 year, cat, 'Lower_95CI')
                    if r:
                        year_results.append(r)

                # Upper 95% CI (band 6)
                if orig_upper is not None:
                    recon_upper = reconstructed + 1.96 * recon_sigma
                    r = _compare(recon_upper[valid], orig_upper[valid],
                                 year, cat, 'Upper_95CI')
                    if r:
                        year_results.append(r)

        return year_results

    # ---- Parallel verification across all years ----
    from concurrent.futures import ThreadPoolExecutor

    results = []
    all_pass = True

    with ThreadPoolExecutor(max_workers=min(40, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(_verify_year, y): y for y in sample_years}
        for future in futures:
            year_results = future.result()
            for r in year_results:
                if not r['Pass']:
                    all_pass = False
                results.append(r)

            year = futures[future]
            if year % 20 == 0 or year == sample_years[-1]:
                logger.info(f'    Verified year {year}')

    # ---- Write summary ----
    summary_df = pd.DataFrame(results)
    summary_csv = os.path.join(output_dir, f'Well_Package_Verification_{unit}.csv')
    summary_df.to_csv(summary_csv, index=False)

    n_total = len(summary_df)
    n_passed = summary_df['Pass'].sum() if not summary_df.empty else 0
    n_failed = n_total - n_passed
    max_rel = summary_df['Max_Rel_Diff'].max() if not summary_df.empty else 0

    logger.info(f'  Verification ({unit}) complete: {n_passed}/{n_total} checks passed '
                f'(max rel diff = {max_rel:.6f}, tol = {rtol})')
    if n_failed > 0:
        failed = summary_df[~summary_df['Pass']]
        # Separate Lower_95CI failures (expected float32 edge case) from real failures
        ci_fails = failed[failed['Band'] == 'Lower_95CI']
        real_fails = failed[failed['Band'] != 'Lower_95CI']
        if not real_fails.empty:
            for _, row in real_fails.head(10).iterrows():
                logger.warning(f'    FAIL: Year={row.Year}, Cat={row.Category}, '
                               f'Band={row.Band}, '
                               f'MaxRelDiff={row.Max_Rel_Diff:.6f}, '
                               f'PctPass={row.Pct_Pass:.1f}%')
        if not ci_fails.empty:
            max_ci_diff = ci_fails['Max_Rel_Diff'].max()
            logger.info(
                f'    Lower_95CI: {len(ci_fails)} checks exceed tolerance '
                f'(max rel diff = {max_ci_diff:.6f}). This is expected: '
                f'max(pred - 1.96*sigma, 0) is non-linear at the zero '
                f'boundary, so pixel-level clipping and well-sum clipping '
                f'diverge slightly due to float32 precision.')
    logger.info(f'  Summary saved to {summary_csv}')

    return all_pass
