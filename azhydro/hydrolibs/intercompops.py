"""
Intercomparison of irrigation groundwater and surface-water withdrawals
across Arizona groundwater basins.

Compares three datasets:
    1. **ML-based 2 km pumping estimates** — Predicted annual rasters from the
       XGBoost pipeline (``Predicted_Rasters/Postprocessed/``), partitioned
       into Irrigation_GW and Irrigation_SW via ``partitionops``.
    2. **USGS NHM Withdrawals** — HUC12-scale monthly data (2000-2020),
       converted to mean annual rasters using AZ HUC12 polygons.
    3. **USGS Reitz Irrigation** — County-scale 800 m annual rasters
       (``GW_irr_YYYY.tif`` / ``SW_irr_YYYY.tif``, 1980-2018, in meters).

Because the three products live at different native resolutions (2 km,
HUC12 polygons, 800 m), the intercomparison aggregates each dataset to
Arizona groundwater basin totals (volume in acre-feet) and computes
mean-annual values over the common year range for each GW/SW category.

Metrics reported:
    * Root Mean Square Difference (RMSD)
    * Mean Absolute Difference (MAD)
    * Percent Difference (%)

Also compares non-irrigation predictions with:
    4. **USGS Public Supply Reanalysis** — HUC12-scale monthly PS GW/SW
       withdrawals (2000-2020), from Alzraiee et al. (2024, WRR).

References
----------
NHM metadata: IR_metadata.xml — withdrawals in million gallons per day.
Reitz metadata: HistoricalET_metadata.xml — irrigation in meters/year.
PS metadata: PS_WU_reanalysis_v2.xml — public supply in Mgal/d.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, mapping

from hydrolibs.rasterops import read_raster_as_arr
from hydrolibs.sysops import makedirs
from hydrolibs.visualops import (
    _overlay_boundaries,
    add_ama_ina_legend,
    get_ama_ina_basin_names,
    plot_intercomp_scatter,
    plot_intercomp_stacked_bars,
    plot_intercomp_taylor,
    plot_intercomp_time_series,
    plot_temporal_box_violin,
    plot_temporal_heatmap,
    plot_temporal_r_vs_nse,
)

logger = logging.getLogger(__name__)

# ── Unit-conversion constants ────────────────────────────────────────────────
MGAL_TO_M3 = 3785.41178                # 1 Mgal → m³
M3_TO_AF = 1 / 1233.48184              # m³ → acre-feet
M_TO_MM = 1000.0                        # meters → millimeters
MM_TO_FT = 1.0 / 304.8                 # millimeters → feet
# 1 mgd × 365.25 days × 1e6 gal ÷ 325,851 gal/AF ≈ 1,120.34 AF/yr
MGD_TO_AF_PER_YEAR = 365.25 * 1e6 / 325_851.0

# NHM sentinel values (no irrigated area or null ET)
NHM_SENTINEL = {999, 888}


_filter_huc12_cache: dict[str, set[str]] | None = None


def _filter_huc12_within_az(
    huc_gdf: gpd.GeoDataFrame,
    basin_gdf: gpd.GeoDataFrame,
    threshold: float = 0.10,
) -> gpd.GeoDataFrame:
    """Keep HUC12 polygons with at least ``threshold`` of their area in AZ.

    A HUC12 is kept if at least ``threshold`` (default 10 %) of its
    area overlaps the union of AZ basin polygons.  The default was
    relaxed from 95 % → 10 % to retain Yuma- and Mexico-border HUC12s
    that have substantive AZ-interior cropland.  The earlier tight
    threshold dropped ~270 HUC12s with 10–95 % AZ-interior fraction,
    biasing the NHM-derived per-basin withdrawal, IE, and CU at Yuma
    / Lower Gila / Western Mexican Drainage / Sacramento Valley.

    Mass conservation at cross-border HUC12s is handled by the
    downstream ``_get_huc_basin_overlay`` which weights every HUC12's
    value by ``area_frac = overlap_area / huc_area`` — a HUC12 that is
    60 % in AZ contributes 60 % of its reported NHM value to the AZ
    basin, never 100 %.  The 10 % threshold only drops polygons where
    the AZ sliver is too small (<10 %) to be a physically meaningful
    contribution and where the ``area_frac`` weighting would be
    dominated by boundary geometry noise.

    The result is cached at module level keyed by the number of input
    HUC12s so the expensive ``union_all`` + ``intersection`` geometry
    computation runs only once per session regardless of how many times
    the function is called.
    """
    global _filter_huc12_cache
    cache_key = f'{len(huc_gdf)}_{threshold:.4f}'
    if _filter_huc12_cache is not None and cache_key in _filter_huc12_cache:
        kept_ids = _filter_huc12_cache[cache_key]
        huc_ids = huc_gdf['huc12'].astype(str)
        result = huc_gdf[huc_ids.isin(kept_ids)].copy()
        logger.info(
            f'  HUC12 filter (cached): {len(result)} AZ-interior HUC12s'
        )
        return result

    target_crs = basin_gdf.crs
    huc_proj = (
        huc_gdf.to_crs(target_crs) if huc_gdf.crs != target_crs
        else huc_gdf
    )
    az_union = basin_gdf.geometry.union_all()
    frac = huc_proj.geometry.intersection(az_union).area / huc_proj.geometry.area
    keep = frac >= threshold
    n_dropped = int((~keep).sum())
    n_kept = int(keep.sum())
    if n_dropped > 0:
        logger.info(
            f'  Dropped {n_dropped} cross-border HUC12s '
            f'(< {threshold * 100:.0f}% within AZ basins); '
            f'{n_kept} HUC12s retained'
        )
    else:
        logger.info(
            f'  All {n_kept} HUC12s within AZ basins (no cross-border drops)'
        )

    kept_ids = set(huc_gdf.loc[keep.values, 'huc12'].astype(str))
    if _filter_huc12_cache is None:
        _filter_huc12_cache = {}
    _filter_huc12_cache[cache_key] = kept_ids

    return huc_gdf[keep.values].copy()


def _get_huc_basin_overlay(
    huc_reproj: gpd.GeoDataFrame,
    basin_reproj: gpd.GeoDataFrame,
    basin_col: str,
    cache_dir: str | None = None,
) -> gpd.GeoDataFrame:
    """Compute (or load cached) HUC12 → basin spatial overlay.

    The overlay is the geometric intersection of every HUC12 polygon
    with every basin polygon, with ``overlap_area`` and ``area_frac``
    columns pre-computed. It is the most expensive operation in the
    NHM aggregation pipeline (~2 min for 3,645 HUC12s × 52 basins),
    and the result is deterministic for a given HUC12 + basin pair, so
    caching it to a GeoParquet file avoids recomputing on every Step 4
    rerun.

    The cache is keyed by ``{cache_dir}/huc_basin_overlay_n{N}.parquet``
    where ``N`` is the count of input HUC12s — this prevents stale
    caches from being reused if the upstream HUC12 filter threshold
    changes (different threshold → different N → different cache file).
    If the file exists it is loaded; otherwise the overlay is computed
    and saved.  Legacy caches at the old unversioned path
    (``huc_basin_overlay.parquet``) are ignored to force a rebuild
    after the 2026-04-22 threshold change (0.95 → 0.10).
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(
            cache_dir,
            f'huc_basin_overlay_n{len(huc_reproj)}.parquet',
        )
        if os.path.isfile(cache_path):
            logger.info(f'  Loading cached HUC12→basin overlay from {cache_path}')
            return gpd.read_parquet(cache_path)

    logger.info('  Computing HUC12→basin spatial overlay (this is slow; '
                'result will be cached for future runs)...')
    overlay = gpd.overlay(
        huc_reproj[['huc12', 'area_m2', 'geometry']],
        basin_reproj[[basin_col, 'geometry']],
        how='intersection',
    )
    overlay['overlap_area'] = overlay.geometry.area
    overlay['area_frac'] = overlay['overlap_area'] / overlay['area_m2']

    if cache_path:
        makedirs(os.path.dirname(cache_path))
        overlay.to_parquet(cache_path)
        logger.info(f'  Cached HUC12→basin overlay to {cache_path}')

    return overlay


# ═════════════════════════════════════════════════════════════════════════════
# Helper: aggregate a raster to basin volumes (AF)
# ═════════════════════════════════════════════════════════════════════════════
def _raster_basin_volumes(
    raster_path: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    pixel_area_m2: float,
    depth_unit: str = 'mm',
) -> dict[str, float]:
    """Sum raster depth values within each basin polygon and return
    volumes in acre-feet.

    Args:
        raster_path (str): Path to a single-band depth raster (mm or m).
        basin_gdf (gpd.GeoDataFrame): Basin polygons in the same CRS as *raster_path*.
        basin_col (str): Column in *basin_gdf* identifying basins.
        pixel_area_m2 (float): Pixel area in square meters.
        depth_unit (str): ``'mm'`` or ``'m'`` — unit of pixel values.

    Returns:
        dict[str, float]: ``{basin_name: volume_AF}``.
    """
    depth_to_m = 1.0 / M_TO_MM if depth_unit == 'mm' else 1.0
    volumes = {}
    with rio.open(raster_path) as src:
        raster_bounds = src.bounds
        for _, row in basin_gdf.iterrows():
            basin_name = row[basin_col]
            geom = [mapping(row.geometry)]
            try:
                if not row.geometry.intersects(
                    box(*raster_bounds)
                ):
                    logger.warning('Basin %s does not intersect raster extent, setting volume to 0', basin_name)
                    volumes[basin_name] = 0.0
                    continue
                clipped, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = clipped[0].astype(np.float64)
                arr[np.isnan(arr)] = 0.0
                arr[arr < 0] = 0.0
                vol_m3 = float(np.nansum(arr)) * depth_to_m * pixel_area_m2
                volumes[basin_name] = vol_m3 * M3_TO_AF
            except (ValueError, rio.errors.WindowError) as exc:
                logger.warning('Clipping failed for basin %s: %s. Setting volume to 0', basin_name, exc)
                volumes[basin_name] = 0.0
    return volumes


def _raster_basin_means(
    raster_path: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
) -> dict[str, float]:
    """Compute the area-weighted mean of valid (non-NaN, non-zero) pixels
    within each basin polygon.  Used for dimensionless ratio rasters
    (e.g. irrigation efficiency).

    Returns:
        dict[str, float]: ``{basin_name: mean_value}``.
    """
    means = {}
    with rio.open(raster_path) as src:
        raster_bounds = src.bounds
        for _, row in basin_gdf.iterrows():
            basin_name = row[basin_col]
            geom = [mapping(row.geometry)]
            try:
                if not row.geometry.intersects(
                    box(*raster_bounds)
                ):
                    logger.warning('Basin %s does not intersect raster extent, setting mean to NaN', basin_name)
                    means[basin_name] = np.nan
                    continue
                clipped, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = clipped[0].astype(np.float64)
                valid = arr[np.isfinite(arr) & (arr > 0)]
                means[basin_name] = float(np.mean(valid)) if valid.size > 0 else np.nan
            except (ValueError, rio.errors.WindowError) as exc:
                logger.warning('Clipping failed for basin %s: %s. Setting mean to NaN', basin_name, exc)
                means[basin_name] = np.nan
    return means


def _compute_huc12_zonal_stats(
    raster_path: str,
    huc_gdf: gpd.GeoDataFrame,
    pixel_area_m2: float,
    depth_unit: str = 'mm',
) -> dict[str, dict[str, float]]:
    """Compute per-HUC12 zonal statistics from a depth raster.

    For each HUC12 polygon in *huc_gdf*, clips the raster, sums the
    pixel depths to get volume, and computes the area-weighted mean
    depth.

    Args:
        raster_path: Path to a single-band depth raster.
        huc_gdf: HUC12 polygons (must already be in the raster CRS).
        pixel_area_m2: Pixel area in m².
        depth_unit: ``'mm'`` or ``'m'``.

    Returns:
        ``{huc12_id: {'depth_mm': ..., 'volume_m3': ..., 'volume_AF': ...}}``
    """
    depth_to_m = 1.0 / M_TO_MM if depth_unit == 'mm' else 1.0
    result: dict[str, dict[str, float]] = {}
    with rio.open(raster_path) as src:
        raster_bounds = src.bounds
        for _, row in huc_gdf.iterrows():
            huc_id = str(row['huc12'])
            geom = [mapping(row.geometry)]
            try:
                if not row.geometry.intersects(box(*raster_bounds)):
                    result[huc_id] = {
                        'depth_mm': 0.0, 'volume_m3': 0.0, 'volume_AF': 0.0,
                    }
                    continue
                clipped, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = clipped[0].astype(np.float64)
                valid = arr[np.isfinite(arr) & (arr > 0)]
                if valid.size > 0:
                    mean_depth_mm = float(np.mean(valid)) * (
                        1.0 if depth_unit == 'mm' else M_TO_MM
                    )
                    vol_m3 = float(np.sum(valid)) * depth_to_m * pixel_area_m2
                else:
                    mean_depth_mm = 0.0
                    vol_m3 = 0.0
                result[huc_id] = {
                    'depth_mm': mean_depth_mm,
                    'volume_m3': vol_m3,
                    'volume_AF': vol_m3 * M3_TO_AF,
                }
            except (ValueError, rio.errors.WindowError):
                result[huc_id] = {
                    'depth_mm': 0.0, 'volume_m3': 0.0, 'volume_AF': 0.0,
                }
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load USGS NHM HUC12 data → mean-annual basin volumes
# ═════════════════════════════════════════════════════════════════════════════
def load_nhm_basin_volumes(
    nhm_dir: str,
    huc12_geojson: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    ref_raster: str,
    year_range: tuple[int, int],
    output_dir: str,
    predictor_dir: str | None = None,
    irr_fraction_band: int = 14,
) -> dict[str, dict]:
    """
    Read the NHM monthly GW/SW CSVs, compute annual totals per AZ HUC12,
    rasterize onto the ML prediction grid, then aggregate to basin volumes.

    The NHM values are in **million gallons per day** (Mgal/d).  Monthly
    values are converted to total volume (m³) by multiplying by the number
    of days in each month, then summing to annual.  The annual totals are
    joined to the AZ HUC12 polygons and rasterized onto the reference grid.

    Because NHM withdrawals apply only to irrigated areas, the volume-to-
    depth conversion uses the *irrigated* area within each HUC12 rather
    than the full polygon area.  The irrigated fraction is obtained from
    the ``annual_irr_fraction`` band of the ``Predictor_YYYY.tif`` rasters
    (mean over the year range), so that:

        depth_mm = volume_m³ / (polygon_area_m² × mean_irr_fraction) × 1000

    If *predictor_dir* is not provided, the full HUC12 area is used with
    a warning.

    Args:
        nhm_dir (str): Directory containing the NHM CSV files.
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_gdf (gpd.GeoDataFrame): Basin polygons (target CRS = reference raster CRS).
        basin_col (str): Column in *basin_gdf* naming basins.
        ref_raster (str): Reference raster for grid/CRS information.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.
        output_dir (str): Directory for intermediate rasters.
        predictor_dir (str or None): Directory with ``Predictor_YYYY.tif`` multi-band rasters containing
            ``annual_irr_fraction`` (band *irr_fraction_band*).  When provided,
            the mean irrigated fraction per HUC12 is used to scale the area.
        irr_fraction_band (int): 1-indexed band number for ``annual_irr_fraction`` in the predictor
            rasters (default 14).

    Returns:
        dict[str, dict[str, float]]: ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
    """
    makedirs(output_dir)
    start_yr, end_yr = year_range
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    result = {}
    for category, csv_name in [
        ('GW', 'IR_HUC12_GW_WD_monthly_2000_2020.csv'),
        ('SW', 'IR_HUC12_SW_WD_monthly_2000_2020.csv'),
        ('Total', 'IR_HUC12_Tot_WD_monthly_2000_2020.csv'),
    ]:
        csv_path = os.path.join(nhm_dir, csv_name)
        logger.info(f'Reading NHM {category}: {csv_path}')

        # Wide-format CSV: Year, Month, <HUC12_code_1>, ..., <HUC12_code_N>
        df = pd.read_csv(csv_path, dtype={'Year': int, 'Month': int})
        huc_cols = [c for c in df.columns if c not in ('Year', 'Month')]

        # Load AZ HUC12 polygons, drop cross-border HUC12s that extend
        # outside AZ basin boundaries, and determine the column set.
        huc_gdf = gpd.read_file(huc12_geojson)
        huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
        az_huc12_set = set(huc_gdf['huc12'].astype(str).values)

        # Filter CSV columns to AZ-interior HUC12s only
        az_cols = [c for c in huc_cols if c in az_huc12_set]
        logger.info(f'  {len(az_cols)} AZ-interior HUC12 regions found in NHM data')

        if not az_cols:
            logger.warning(f'  No AZ HUC12 matches for {category}')
            result[category] = {
                'mean': {b: 0.0 for b in basin_gdf[basin_col]},
                'yearly': {},
            }
            continue

        # Subset data
        df_az = df[['Year', 'Month'] + az_cols].copy()
        df_az = df_az[(df_az.Year >= start_yr) & (df_az.Year <= end_yr)]

        # Replace sentinel values with 0
        for col in az_cols:
            df_az[col] = pd.to_numeric(df_az[col], errors='coerce')
            df_az.loc[df_az[col].isin(NHM_SENTINEL), col] = 0.0
            df_az[col] = df_az[col].fillna(0.0)

        # Convert Mgal/d → m³ per month, then sum to annual per HUC12
        annual_records = []
        for year in range(start_yr, end_yr + 1):
            yr_df = df_az[df_az.Year == year]
            annual_vol = np.zeros(len(az_cols))
            for _, row in yr_df.iterrows():
                month = int(row['Month'])
                ndays = days_in_month[month - 1]
                # Adjust for leap year
                if month == 2 and year % 4 == 0 and (
                        year % 100 != 0 or year % 400 == 0):
                    ndays = 29
                vals = row[az_cols].values.astype(np.float64)
                # Mgal/d × days → Mgal; × MGAL_TO_M3 → m³
                annual_vol += vals * ndays * MGAL_TO_M3
            for i, huc_id in enumerate(az_cols):
                annual_records.append({
                    'huc12': huc_id,
                    'year': year,
                    'volume_m3': annual_vol[i],
                })

        ann_df = pd.DataFrame(annual_records)

        # Mean annual volume per HUC12 (m³/yr)
        mean_annual = ann_df.groupby('huc12')['volume_m3'].mean().reset_index()
        mean_annual.columns = ['huc12', 'mean_vol_m3']

        # Join to HUC12 polygons
        huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
        huc_merged = huc_gdf.merge(mean_annual, on='huc12', how='left')
        huc_merged['mean_vol_m3'] = huc_merged['mean_vol_m3'].fillna(0.0)

        # Reproject HUC12 to match reference raster CRS
        with rio.open(ref_raster) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_shape = (ref_src.height, ref_src.width)
            pixel_area_m2 = abs(ref_transform.a * ref_transform.e)

        huc_reproj = huc_merged.to_crs(ref_crs)

        # Compute mean irrigated fraction per HUC12 from predictor rasters
        huc_reproj['area_m2'] = huc_reproj.geometry.area
        huc_reproj['irr_fraction'] = 1.0  # fallback: full area

        if predictor_dir is not None:
            logger.info('  Computing mean irrigated fraction per HUC12...')
            irr_counts = np.zeros(len(huc_reproj))
            irr_n_years = 0
            for yr in range(start_yr, end_yr + 1):
                pred_file = os.path.join(predictor_dir, f'Predictor_{yr}.tif')
                if not os.path.isfile(pred_file):
                    continue
                # Open raster once per year; reuse handle for all HUC12 clips
                with rio.open(pred_file) as src:
                    for idx, row in huc_reproj.iterrows():
                        geom = [mapping(row.geometry)]
                        try:
                            clipped, _ = rio_mask(src, geom, crop=True,
                                                  all_touched=True,
                                                  indexes=[irr_fraction_band],
                                                  nodata=np.nan)
                            vals = clipped[0].astype(np.float64)
                            vals = vals[~np.isnan(vals)]
                            vals = np.clip(vals, 0, 1)
                            if vals.size > 0:
                                irr_counts[huc_reproj.index.get_loc(idx)] += np.mean(vals)
                        except (ValueError, rio.errors.WindowError):
                            logger.debug('Irr fraction clipping failed for HUC at index %s', idx)
                irr_n_years += 1

            if irr_n_years > 0:
                huc_reproj['irr_fraction'] = np.clip(
                    irr_counts / irr_n_years, 0, 1,
                )
            logger.info('  Mean irr_fraction range: %.4f – %.4f',
                        huc_reproj['irr_fraction'].min(),
                        huc_reproj['irr_fraction'].max())
        else:
            logger.warning(
                '  predictor_dir not provided; using full HUC12 area '
                'for NHM depth conversion (overestimates area).'
            )

        # Convert volume to depth (mm) for rasterisation
        # depth_mm = volume_m³ / (polygon_area_m² × irr_fraction) × 1000
        irr_area = huc_reproj['area_m2'] * huc_reproj['irr_fraction']
        huc_reproj['depth_mm'] = np.where(
            irr_area > 0,
            huc_reproj['mean_vol_m3'] / irr_area * M_TO_MM,
            0.0,
        )

        # Rasterize depth onto the reference grid (mean value per pixel)
        shapes = [
            (geom, val) for geom, val in
            zip(huc_reproj.geometry, huc_reproj['depth_mm'])
            if val > 0
        ]
        if shapes:
            nhm_raster = rasterize(
                shapes,
                out_shape=ref_shape,
                transform=ref_transform,
                fill=0.0,
                dtype='float64',
                merge_alg=rio.enums.MergeAlg.replace,
            )
        else:
            nhm_raster = np.zeros(ref_shape, dtype=np.float64)

        # Write intermediate raster
        out_tif = os.path.join(output_dir, f'NHM_mean_annual_{category}_mm.tif')
        with rio.open(ref_raster) as ref_src:
            profile = ref_src.profile.copy()
        profile.update(dtype='float64', nodata=np.nan, count=1)
        nhm_raster[nhm_raster == 0] = np.nan
        with rio.open(out_tif, 'w', **profile) as dst:
            dst.write(nhm_raster, 1)
        logger.info(f'  Wrote NHM raster: {out_tif}')

        # Aggregate to basin volumes
        basin_reproj = basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
        basin_vols = _raster_basin_volumes(
            out_tif, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
        )

        # Per-year basin volumes via spatial overlay of HUC12 → basins
        yearly_vols = {}
        overlay = _get_huc_basin_overlay(
            huc_reproj, basin_reproj, basin_col,
            cache_dir=output_dir,
        )
        for year in range(start_yr, end_yr + 1):
            yr_vols = ann_df[ann_df.year == year].set_index('huc12')['volume_m3']
            merged = overlay.merge(
                yr_vols, left_on='huc12', right_index=True, how='left',
            ).fillna(0.0)
            merged['weighted_vol'] = merged['volume_m3'] * merged['area_frac']
            basin_sums = merged.groupby(basin_col)['weighted_vol'].sum()
            yearly_vols[year] = {
                b: basin_sums.get(b, 0.0) * M3_TO_AF
                for b in basin_reproj[basin_col]
            }

        # Compute mean-annual basin volumes from the overlay-based
        # yearly_vols (which does the correct mass-conserving
        # volume × area_frac aggregation) rather than from the
        # rasterize-then-sum path via _raster_basin_volumes, which
        # over-counts volume because the HUC12 depth values get
        # replicated across all pixels inside the basin polygon.
        mean_basin_vols: dict[str, float] = {}
        for b in basin_reproj[basin_col]:
            yr_vals = [
                yearly_vols[yr].get(b, 0.0)
                for yr in yearly_vols
            ]
            mean_basin_vols[b] = float(np.mean(yr_vals)) if yr_vals else 0.0
        result[category] = {'mean': mean_basin_vols, 'yearly': yearly_vols}

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2. Load USGS Reitz rasters → mean-annual basin volumes
# ═════════════════════════════════════════════════════════════════════════════
def _reproject_reitz_to_ref(
    reitz_path: str,
    ref_raster: str,
    out_path: str,
) -> str:
    """Reproject a Reitz 800 m geographic raster to the ML prediction grid."""
    with rio.open(reitz_path) as src:
        with rio.open(ref_raster) as ref_src:
            dst_crs = ref_src.crs
            dst_transform = ref_src.transform

            profile = ref_src.profile.copy()
            profile.update(dtype='float64', nodata=np.nan, count=1)

            with rio.open(out_path, 'w', **profile) as dst:
                reproject(
                    source=rio.band(src, 1),
                    destination=rio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.average,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
    return out_path


def load_reitz_basin_volumes(
    reitz_base_dir: str,
    ref_raster: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
    output_dir: str,
) -> dict[str, dict]:
    """
    Read annual Reitz GW/SW rasters, reproject to the ML grid, compute
    mean-annual depth, and aggregate to basin volumes (AF).

    The Reitz rasters are in **meters/year** at ~800 m geographic resolution.

    Args:
        reitz_base_dir (str): Parent directory containing ``Irrigation_groundwater_1980-2018/``
            and ``Irrigation_surfacewater_1980-2018/``.
        ref_raster (str): Reference ML prediction raster for CRS/grid alignment.
        basin_gdf (gpd.GeoDataFrame): Basin polygons.
        basin_col (str): Basin name column.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.
        output_dir (str): Directory for reprojected/intermediate rasters.

    Returns:
        dict[str, dict[str, float]]: ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
    """
    makedirs(output_dir)
    start_yr, end_yr = year_range

    with rio.open(ref_raster) as ref_src:
        ref_crs = ref_src.crs
        pixel_area_m2 = abs(ref_src.transform.a * ref_src.transform.e)

    result = {}
    for category, subdir, prefix in [
        ('GW', 'Irrigation_groundwater_1980-2018', 'GW_irr'),
        ('SW', 'Irrigation_surfacewater_1980-2018', 'SW_irr'),
    ]:
        logger.info(f'Processing Reitz {category} rasters...')
        mean_depth = None
        n_years = 0

        for year in range(start_yr, end_yr + 1):
            tif_name = f'{prefix}_{year}.tif'
            tif_path = os.path.join(reitz_base_dir, subdir, tif_name)
            if not os.path.isfile(tif_path):
                logger.warning(f'  Missing Reitz raster: {tif_path}')
                continue

            # Reproject to reference grid
            reproj_path = os.path.join(output_dir, f'Reitz_{category}_{year}_reproj.tif')
            _reproject_reitz_to_ref(tif_path, ref_raster, reproj_path)

            arr = read_raster_as_arr(reproj_path, get_file=False).astype(np.float64)
            arr[np.isnan(arr)] = 0.0
            arr[arr < 0] = 0.0
            # Convert meters → mm
            arr *= M_TO_MM

            if mean_depth is None:
                mean_depth = arr.copy()
            else:
                mean_depth += arr
            n_years += 1

        if n_years > 0:
            mean_depth /= n_years
        else:
            logger.warning(f'  No Reitz {category} rasters in year range')
            result[category] = {
                'mean': {b: 0.0 for b in basin_gdf[basin_col]},
                'yearly': {},
            }
            continue

        # Write mean-annual raster
        out_tif = os.path.join(output_dir, f'Reitz_mean_annual_{category}_mm.tif')
        with rio.open(ref_raster) as ref_src:
            profile = ref_src.profile.copy()
        profile.update(dtype='float64', nodata=np.nan, count=1)
        mean_depth[mean_depth == 0] = np.nan
        with rio.open(out_tif, 'w', **profile) as dst:
            dst.write(mean_depth, 1)
        logger.info(f'  Wrote Reitz mean-annual raster: {out_tif}')

        # Aggregate to basin volumes
        basin_reproj = basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
        basin_vols = _raster_basin_volumes(
            out_tif, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
        )

        # Per-year basin volumes from reprojected rasters (on-disk units are meters)
        yearly_vols = {}
        for year in range(start_yr, end_yr + 1):
            reproj_path = os.path.join(
                output_dir, f'Reitz_{category}_{year}_reproj.tif',
            )
            if os.path.isfile(reproj_path):
                yearly_vols[year] = _raster_basin_volumes(
                    reproj_path, basin_reproj, basin_col,
                    pixel_area_m2, depth_unit='m',
                )

        result[category] = {'mean': basin_vols, 'yearly': yearly_vols}

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. Load ML prediction rasters → mean-annual basin volumes
# ═════════════════════════════════════════════════════════════════════════════
def load_ml_basin_volumes(
    pred_raster_dir: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
    irr_gw_dir: str | None = None,
    irr_sw_dir: str | None = None,
) -> dict[str, dict]:
    """
    Read ML prediction rasters (total pumping in mm) and, when available,
    category-specific Irrigation_GW / Irrigation_SW rasters.  Compute
    mean-annual basin volumes.

    If *irr_gw_dir* / *irr_sw_dir* are not provided or their rasters are
    absent, the total prediction rasters from *pred_raster_dir* are used
    as the **total** irrigation withdrawal estimate.  This is a simplification
    because the total predictions include non-irrigation pumping; a warning
    is emitted.

    Args:
        pred_raster_dir (str): Directory with ``pred_YYYY.tif`` total pumping rasters (mm).
        basin_gdf (gpd.GeoDataFrame): Basin polygons.
        basin_col (str): Basin name column.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.
        irr_gw_dir (str or None): Directory with ``Irrigation_GW_YYYY_mm.tif`` rasters.
        irr_sw_dir (str or None): Directory with ``Irrigation_SW_YYYY_mm.tif`` rasters.

    Returns:
        dict[str, dict[str, float]]: ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
    """
    start_yr, end_yr = year_range

    # Determine available category raster paths
    cat_dirs = {'GW': irr_gw_dir, 'SW': irr_sw_dir}
    cat_patterns = {
        'GW': 'Irrigation_GW_{year}_mm.tif',
        'SW': 'Irrigation_SW_{year}_mm.tif',
    }
    use_total_fallback = {}

    for cat in ('GW', 'SW'):
        d = cat_dirs[cat]
        if d and os.path.isdir(d):
            # Verify at least one file exists
            sample = cat_patterns[cat].format(year=start_yr)
            if os.path.isfile(os.path.join(d, sample)):
                use_total_fallback[cat] = False
            else:
                use_total_fallback[cat] = True
        else:
            use_total_fallback[cat] = True

    if any(use_total_fallback.values()):
        logger.warning(
            'Category-specific ML rasters not found for %s; '
            'using total prediction rasters as fallback (includes all categories).',
            [k for k, v in use_total_fallback.items() if v],
        )

    result = {}
    for cat in ('GW', 'SW'):
        ref_raster = None
        mean_depth = None
        n_years = 0

        for year in range(start_yr, end_yr + 1):
            if use_total_fallback[cat]:
                raster_path = os.path.join(pred_raster_dir, f'pred_{year}.tif')
            else:
                raster_path = os.path.join(
                    cat_dirs[cat], cat_patterns[cat].format(year=year),
                )
            if not os.path.isfile(raster_path):
                continue

            if ref_raster is None:
                ref_raster = raster_path

            arr = read_raster_as_arr(raster_path, get_file=False).astype(np.float64)
            arr[np.isnan(arr)] = 0.0
            arr[arr < 0] = 0.0

            if mean_depth is None:
                mean_depth = arr.copy()
            else:
                mean_depth += arr
            n_years += 1

        if n_years > 0 and ref_raster is not None:
            mean_depth /= n_years
        else:
            logger.warning(f'No ML {cat} rasters in year range')
            result[cat] = {
                'mean': {b: 0.0 for b in basin_gdf[basin_col]},
                'yearly': {},
            }
            continue

        with rio.open(ref_raster) as src:
            ref_crs = src.crs
            pixel_area_m2 = abs(src.transform.a * src.transform.e)

        basin_reproj = basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
        # Write mean-annual raster
        cat_out = os.path.join(
            os.path.dirname(pred_raster_dir.rstrip('/')),
            f'ML_mean_annual_{cat}_mm.tif',
        )
        with rio.open(ref_raster) as ref_src:
            profile = ref_src.profile.copy()
        profile.update(dtype='float64', nodata=np.nan, count=1)
        tmp = mean_depth.copy()
        tmp[tmp == 0] = np.nan
        with rio.open(cat_out, 'w', **profile) as dst:
            dst.write(tmp, 1)
        logger.info(f'Wrote ML mean-annual raster: {cat_out}')

        basin_vols = _raster_basin_volumes(
            cat_out, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
        )

        # Per-year basin volumes
        yearly_vols = {}
        for year in range(start_yr, end_yr + 1):
            if use_total_fallback[cat]:
                raster_path = os.path.join(pred_raster_dir, f'pred_{year}.tif')
            else:
                raster_path = os.path.join(
                    cat_dirs[cat], cat_patterns[cat].format(year=year),
                )
            if os.path.isfile(raster_path):
                yearly_vols[year] = _raster_basin_volumes(
                    raster_path, basin_reproj, basin_col,
                    pixel_area_m2, depth_unit='mm',
                )

        result[cat] = {'mean': basin_vols, 'yearly': yearly_vols}

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. Intercomparison metrics
# ═════════════════════════════════════════════════════════════════════════════
def _compute_metrics(
    basin_names: list[str],
    data_a: dict[str, float],
    data_b: dict[str, float],
    label_a: str,
    label_b: str,
    basin_areas_m2: dict[str, float] | None = None,
) -> dict:
    """Compute RMSD, MAD, and Percent Difference between two datasets across
    basins in AF, m\u00b3, and optionally mm (basin-average depth).

    Percent Difference is defined as:
        100 \u00d7 mean(|a - b|) / mean(a, b)
    where the mean in the denominator is taken across all basins.
    """
    vals_a = np.array([data_a.get(b, 0.0) for b in basin_names], dtype=np.float64)
    vals_b = np.array([data_b.get(b, 0.0) for b in basin_names], dtype=np.float64)

    diff = vals_a - vals_b
    rmsd_af = float(np.sqrt(np.mean(diff ** 2)))
    mad_af = float(np.mean(np.abs(diff)))
    denom = (np.sum(vals_a) + np.sum(vals_b)) / 2.0
    pct_diff = float(np.sum(np.abs(diff)) / denom * 100) if denom > 0 else np.nan

    af_to_m3 = 1.0 / M3_TO_AF
    vals_a_m3 = vals_a * af_to_m3
    vals_b_m3 = vals_b * af_to_m3
    diff_m3 = vals_a_m3 - vals_b_m3
    rmsd_m3 = float(np.sqrt(np.mean(diff_m3 ** 2)))
    mad_m3 = float(np.mean(np.abs(diff_m3)))

    result = {
        'Pair': f'{label_a} vs {label_b}',
        'RMSD_AF': round(rmsd_af, 2),
        'RMSD_m3': round(rmsd_m3, 2),
        'MAD_AF': round(mad_af, 2),
        'MAD_m3': round(mad_m3, 2),
        'Pct_Diff': round(pct_diff, 2),
        f'Total_{label_a}_AF': round(float(np.sum(vals_a)), 2),
        f'Total_{label_b}_AF': round(float(np.sum(vals_b)), 2),
        f'Total_{label_a}_m3': round(float(np.sum(vals_a_m3)), 2),
        f'Total_{label_b}_m3': round(float(np.sum(vals_b_m3)), 2),
    }

    if basin_areas_m2 is not None:
        areas = np.array(
            [basin_areas_m2.get(b, 1.0) for b in basin_names], dtype=np.float64,
        )
        areas = np.where(areas > 0, areas, 1.0)
        vals_a_mm = vals_a_m3 / areas * M_TO_MM
        vals_b_mm = vals_b_m3 / areas * M_TO_MM
        diff_mm = vals_a_mm - vals_b_mm
        result['RMSD_mm'] = round(float(np.sqrt(np.mean(diff_mm ** 2))), 4)
        result['MAD_mm'] = round(float(np.mean(np.abs(diff_mm))), 4)

        vals_a_ft = vals_a_mm * MM_TO_FT
        vals_b_ft = vals_b_mm * MM_TO_FT
        diff_ft = vals_a_ft - vals_b_ft
        result['RMSD_ft'] = round(float(np.sqrt(np.mean(diff_ft ** 2))), 6)
        result['MAD_ft'] = round(float(np.mean(np.abs(diff_ft))), 6)

    return result


def _huc12_temporal_diagnostics(
    huc12_yearly_sources: dict[str, dict[int, dict[str, float]]],
    pairs: list[tuple[str, str]],
    huc12_ids: list[str],
    category: str,
    output_dir: str,
    huc_areas: dict[str, float] | None = None,
) -> None:
    """Compute temporal agreement at HUC12 level and render diagnostics.

    Produces:
    - ``huc12_temporal_agreement.csv`` — per-HUC12 Pearson r and NSE
    - ``Temporal_Agreement/BoxViolin_*.png`` — distribution plots
    - ``Taylor/Taylor_HUC12_*.png`` — Taylor diagrams

    Args:
        huc12_yearly_sources: ``{source_label: {year: {huc12: AF}}}``
        pairs: List of ``(src_a, src_b)`` — second is treated as
            reference for NSE.
        huc12_ids: List of HUC12 IDs to evaluate.
        category: Category label (e.g. ``'Irrigation_GW'``).
        output_dir: Root HUC12 comparison directory.
        huc_areas: Optional ``{huc12: area_m2}`` for Taylor diagram.
    """
    from hydrolibs.visualops import (
        plot_temporal_box_violin,
        plot_intercomp_taylor,
    )

    # Temporal metrics
    all_per_huc = []
    all_summary = []
    for label_a, label_b in pairs:
        yearly_a = huc12_yearly_sources.get(label_a, {})
        yearly_b = huc12_yearly_sources.get(label_b, {})
        tm = _compute_temporal_metrics(
            huc12_ids, yearly_a, yearly_b, label_a, label_b,
        )
        summary = {
            'Category': category,
            'Level': 'HUC12',
            'Pair': tm['Pair'],
            'Pearson_r_mean': tm['Pearson_r_mean'],
            'Pearson_r_median': tm['Pearson_r_median'],
            'NSE_mean': tm['NSE_mean'],
            'NSE_median': tm['NSE_median'],
            'n_common_years': tm['n_common_years'],
            'n_zones_with_data': tm['n_basins_with_data'],
        }
        all_summary.append(summary)
        for pb in tm.get('per_basin', []):
            pb['Category'] = category
            pb['Pair'] = tm['Pair']
            all_per_huc.append(pb)
        logger.info(
            f'    {tm["Pair"]} (HUC12): r_mean={tm["Pearson_r_mean"]}, '
            f'NSE_mean={tm["NSE_mean"]}, n={tm["n_basins_with_data"]}'
        )

    if all_summary:
        pd.DataFrame(all_summary).to_csv(
            os.path.join(output_dir, 'huc12_temporal_agreement.csv'),
            index=False,
        )
    if all_per_huc:
        per_huc_df = pd.DataFrame(all_per_huc)
        per_huc_df.to_csv(
            os.path.join(output_dir, 'huc12_temporal_per_zone.csv'),
            index=False,
        )
        # Box-violin
        ta_dir = os.path.join(output_dir, 'Temporal_Agreement/')
        plot_temporal_box_violin(per_huc_df, ta_dir)

    # Taylor diagrams removed (model-vs-model intercomparison has no
    # "true" reference; bundling correlation/std/RMSD onto one panel
    # was hard to interpret).  Box-violin + r-vs-NSE plots remain.
    logger.info(f'  HUC12 temporal diagnostics saved to {output_dir}')


def _compute_temporal_metrics(
    basin_names: list[str],
    yearly_a: dict[int, dict[str, float]],
    yearly_b: dict[int, dict[str, float]],
    label_a: str,
    label_b: str,
) -> dict:
    """Compute interannual agreement metrics between two datasets.

    For each basin, extracts the overlapping yearly time series from both
    datasets and computes Pearson correlation (r) and Nash-Sutcliffe
    Efficiency (NSE).  Returns basin-mean, basin-median, and per-basin
    values.

    Args:
        basin_names (list[str]): Basin identifiers.
        yearly_a, yearly_b (dict[int, dict[str, float]]): ``{year: {basin: volume_AF}}`` for each dataset.
        label_a, label_b (str): Dataset labels.

    Returns:
        dict: Summary with Pearson_r_mean, Pearson_r_median, NSE_mean, NSE_median,
            n_basins_with_data, and per_basin detail list.
    """
    common_years = sorted(set(yearly_a.keys()) & set(yearly_b.keys()))
    if not common_years:
        return {
            'Pair': f'{label_a} vs {label_b}',
            'Pearson_r_mean': np.nan,
            'Pearson_r_median': np.nan,
            'NSE_mean': np.nan,
            'NSE_median': np.nan,
            'n_common_years': 0,
            'n_basins_with_data': 0,
        }

    pearson_rs = []
    nses = []
    per_basin = []
    for basin in basin_names:
        ts_a = np.array([yearly_a[yr].get(basin, 0.0) for yr in common_years])
        ts_b = np.array([yearly_b[yr].get(basin, 0.0) for yr in common_years])
        # Skip basins where either series is all-zero or has non-finite values
        if np.all(ts_a == 0) or np.all(ts_b == 0):
            continue
        if not (np.all(np.isfinite(ts_a)) and np.all(np.isfinite(ts_b))):
            continue
        if len(ts_a) < 2:
            continue
        # Pearson r
        if np.std(ts_a) > 0 and np.std(ts_b) > 0:
            r = float(np.corrcoef(ts_a, ts_b)[0, 1])
        else:
            r = np.nan
        # NSE: 1 - SS_res / SS_tot  (treating ts_b as "observed")
        ss_res = float(np.sum((ts_b - ts_a) ** 2))
        ss_tot = float(np.sum((ts_b - np.mean(ts_b)) ** 2))
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        pearson_rs.append(r)
        nses.append(nse)
        per_basin.append({
            'Basin': basin, 'Pearson_r': round(r, 4),
            'NSE': round(nse, 4),
        })

    valid_rs = [v for v in pearson_rs if np.isfinite(v)]
    valid_nses = [v for v in nses if np.isfinite(v)]
    return {
        'Pair': f'{label_a} vs {label_b}',
        'Pearson_r_mean': round(float(np.mean(valid_rs)), 4) if valid_rs else np.nan,
        'Pearson_r_median': round(float(np.median(valid_rs)), 4) if valid_rs else np.nan,
        'NSE_mean': round(float(np.mean(valid_nses)), 4) if valid_nses else np.nan,
        'NSE_median': round(float(np.median(valid_nses)), 4) if valid_nses else np.nan,
        'n_common_years': len(common_years),
        'n_basins_with_data': len(per_basin),
        'per_basin': per_basin,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4c. Temporal agreement visualizations


# 5. Time series plotting


# 6. Scatter plots — per-basin volumes (pairwise)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Spatial difference maps — mean-annual depth
# ═════════════════════════════════════════════════════════════════════════════
def _plot_basin_diff_panels(
    panels: list[dict],
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    title: str,
    out_path: str,
    *,
    cmap: str = 'RdBu_r',
    shared_colorbar: bool = True,
) -> None:
    """Render a 1×N basin-aggregated Δ Volume choropleth figure.

    Each panel shows one (A − B) comparison in 10⁶ m³; basins are
    colored by the difference.  No per-basin pct / CI annotations
    (colorbar carries the magnitude).  AMA / INA / GW basin legend
    appears once on the leftmost panel (outside-right).

    Args:
        panels: List of panel descriptors.  Each dict requires
            ``basin_a_vols``, ``basin_b_vols`` (``{basin: AF}``),
            and ``panel_title``.  Optional ``label_a`` / ``label_b``
            for the colorbar label.
        basin_gdf: Basin polygons in the target CRS.
        basin_col: Column naming each basin.
        title: Figure suptitle.
        out_path: Output PNG path.
        cmap: Diverging colormap.
        shared_colorbar: If True (default), all panels share a single
            horizontal colorbar with vmax computed across all panels.
            If False, each panel gets its own colorbar (use when
            magnitudes differ substantially across panels).
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.ticker as mticker
    af_to_m3 = 1.0 / M3_TO_AF
    name_col = (
        basin_col if basin_col in basin_gdf.columns
        else basin_gdf.columns[0]
    )

    n = len(panels)
    if n == 0:
        return

    # Compute per-panel diff series + global vmax (for shared colorbar)
    panel_diffs = []
    global_vmax = 1e-6
    for p in panels:
        a_vols = p['basin_a_vols']
        b_vols = p['basin_b_vols']
        diff_m3: dict[str, float] = {}
        for _, row in basin_gdf.iterrows():
            b = row[name_col]
            a_af = a_vols.get(b, np.nan)
            b_af = b_vols.get(b, np.nan)
            if not (np.isfinite(a_af) and np.isfinite(b_af)):
                continue
            diff_m3[b] = (a_af - b_af) * af_to_m3
        panel_diffs.append(diff_m3)
        if diff_m3:
            d_arr = np.array(list(diff_m3.values()))
            d_arr = d_arr[np.abs(d_arr) > 1e-3]
            if d_arr.size:
                global_vmax = max(
                    global_vmax,
                    abs(np.nanpercentile(d_arr, 2)),
                    abs(np.nanpercentile(d_arr, 98)),
                )

    fig, axes = plt.subplots(
        1, n, figsize=(7 * n, 8), constrained_layout=True,
    )
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold')

    last_im = None
    for i, (panel, diff_m3) in enumerate(zip(panels, panel_diffs)):
        ax = axes[i]
        if shared_colorbar:
            vmax = global_vmax
        else:
            d_arr = np.array(list(diff_m3.values())) if diff_m3 else np.array([])
            d_arr = d_arr[np.abs(d_arr) > 1e-3]
            vmax = (
                max(abs(np.nanpercentile(d_arr, 2)),
                    abs(np.nanpercentile(d_arr, 98)), 1e-6)
                if d_arr.size else 1.0
            )
        plot_gdf = basin_gdf.set_index(name_col).copy()
        plot_gdf['diff'] = plot_gdf.index.map(
            lambda b: diff_m3.get(b, np.nan),
        )
        plot_gdf.loc[plot_gdf['diff'].abs() < 1e-3, 'diff'] = np.nan
        plot_gdf.plot(
            ax=ax, column='diff', cmap=cmap,
            vmin=-vmax, vmax=vmax,
            edgecolor='#666666', linewidth=0.5,
            legend=False, missing_kwds={'color': '#EEEEEE'},
        )
        _overlay_boundaries(
            ax, basin_gdf, get_ama_ina_basin_names(), name_col,
            label_fontsize=5.0, label_all=False,
        )
        ax.set_title(panel['panel_title'], fontsize=12, fontweight='bold')
        if not shared_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=Normalize(-vmax, vmax))
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=ax, shrink=0.5, pad=0.04,
                orientation='horizontal', aspect=30, extend='both',
            )
            cbar.formatter = mticker.FuncFormatter(
                lambda x, _: f'{x / 1e6:g}',
            )
            cbar.update_ticks()
            cbar.set_label(
                rf'$\Delta$ Volume ($\times$10$^{{6}}$ m$^3$, '
                f'{panel.get("label_a", "A")} − '
                f'{panel.get("label_b", "B")})',
                fontsize=9, fontweight='bold',
            )
            cbar.ax.tick_params(labelsize=9)
            secax = cbar.ax.secondary_xaxis(
                'top',
                functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
            )
            secax.set_xlabel(
                '\u0394 Volume (AF)', fontsize=9, fontweight='bold',
            )
            secax.tick_params(labelsize=9)
        else:
            last_im = ScalarMappable(
                cmap=cmap, norm=Normalize(-global_vmax, global_vmax),
            )
            last_im.set_array([])
    if shared_colorbar and last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=list(axes), shrink=0.5, pad=0.04,
            orientation='horizontal', aspect=40, extend='both',
        )
        cbar.formatter = mticker.FuncFormatter(
            lambda x, _: f'{x / 1e6:g}',
        )
        cbar.update_ticks()
        cbar.set_label(
            r'$\Delta$ Volume ($\times$10$^{6}$ m$^3$, A − B per panel)',
            fontsize=10, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=10)
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(lambda x: x * M3_TO_AF, lambda x: x / M3_TO_AF),
        )
        secax.set_xlabel(
            '\u0394 Volume (AF)', fontsize=10, fontweight='bold',
        )
        secax.tick_params(labelsize=10)
    add_ama_ina_legend(axes[0])
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def _plot_spatial_diff_maps(
    mean_raster_paths: dict[str, dict[str, str]],
    ref_raster: str,
    output_dir: str,
    basin_shp: str | None = None,
    basin_col: str = 'BASIN_NAME',
) -> None:
    """Create spatial maps of pairwise mean-annual depth differences.

    For each GW/SW category, produces three difference maps:
        ML − NHM, ML − Reitz, NHM − Reitz
    using a diverging color map centered on zero, with groundwater
    basin boundaries and AMA/INA labels overlaid.

    Args:
        mean_raster_paths (dict): ``{source: {cat: path}}`` where source ∈ {ML, NHM, Reitz} and
            cat ∈ {GW, SW}.  Paths to mean-annual depth rasters (mm).
        ref_raster (str): Reference raster for extent / CRS.
        output_dir (str): Directory for saved plots.
        basin_shp (str or None): Path to GW basin boundary shapefile.
            When provided, basin boundaries and AMA/INA labels are
            overlaid on every panel.
        basin_col (str): Column in *basin_shp* naming each basin.

    Returns:
        None
    """
    from hydrolibs.visualops import (
        _overlay_boundaries, get_ama_ina_basin_names, apply_journal_style,
    )

    apply_journal_style()
    makedirs(output_dir)

    with rio.open(ref_raster) as src:
        extent = [
            src.bounds.left, src.bounds.right,
            src.bounds.bottom, src.bounds.top,
        ]
        raster_crs = src.crs

    # Load basin boundaries if provided
    basins_gdf = None
    ama_ina = None
    name_col = basin_col
    if basin_shp and os.path.isfile(basin_shp):
        basins_gdf = gpd.read_file(basin_shp)
        if basins_gdf.crs != raster_crs:
            basins_gdf = basins_gdf.to_crs(raster_crs)
        name_col = (
            basin_col if basin_col in basins_gdf.columns
            else basins_gdf.columns[0]
        )
        ama_ina = get_ama_ina_basin_names()

    pairs = [('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')]

    # Pixel area for depth → volume conversion
    with rio.open(ref_raster) as src:
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
    mm_to_m3 = pixel_area_m2 / 1000.0  # 1 mm depth × pixel_area → m³
    _mm_to_ft = 1.0 / 304.8
    _m3_to_af = 1.0 / 1233.48184

    # Volume-only diff maps.  Depth (mm) panels were dropped because
    # at the pixel level depth and volume differ only by a constant
    # pixel-area factor — they convey the same spatial pattern but
    # the volume axis is the unit reviewers want to see.
    unit_configs = [
        {
            'label': r'$\Delta$ Volume ($\times$10$^{6}$ m$^3$)',
            'secondary_label': '\u0394 Volume (AF)',
            'scale': mm_to_m3,
            'secondary_factor': _m3_to_af,
            'suffix': '_Volume',
            'tick_div': 1e6,
        },
    ]

    for cat in ('GW', 'SW'):
        # Build AZ domain mask from the ML raster
        ml_path = mean_raster_paths.get('ML', {}).get(cat)
        az_mask = None
        if ml_path and os.path.isfile(ml_path):
            ml_arr = read_raster_as_arr(ml_path, get_file=False).astype(np.float64)
            az_mask = np.isfinite(ml_arr)

        # Load and clip all raster arrays once
        raster_arrays: dict[str, np.ndarray] = {}
        for source in ('ML', 'NHM', 'Reitz'):
            path = mean_raster_paths.get(source, {}).get(cat)
            if path and os.path.isfile(path):
                arr = read_raster_as_arr(path, get_file=False).astype(np.float64)
                arr[np.isnan(arr)] = 0.0
                if az_mask is not None:
                    arr[~az_mask] = 0.0
                raster_arrays[source] = arr

        for ucfg in unit_configs:
            scale = ucfg['scale']
            fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                                     constrained_layout=True)
            title_unit = 'Depth' if scale == 1.0 else 'Volume'
            fig.suptitle(
                f'Irrigation {cat} \u2014 Mean-Annual {title_unit} Difference',
                fontsize=14, fontweight='bold',
            )

            # Compute shared vmax across all three pairs
            global_vmax = 1e-6
            for src_a, src_b in pairs:
                if src_a in raster_arrays and src_b in raster_arrays:
                    d = (raster_arrays[src_a] - raster_arrays[src_b]) * scale
                    m = (raster_arrays[src_a] == 0) & (raster_arrays[src_b] == 0)
                    d_valid = d[~m]
                    if d_valid.size > 0:
                        global_vmax = max(
                            global_vmax,
                            abs(np.nanpercentile(d_valid, 2)),
                            abs(np.nanpercentile(d_valid, 98)),
                        )

            last_im = None
            for col_i, (src_a, src_b) in enumerate(pairs):
                ax = axes[col_i]
                ax.set_facecolor('#D5D5D5')
                if src_a not in raster_arrays or src_b not in raster_arrays:
                    ax.set_title(
                        f'{src_a} \u2212 {src_b}  (data unavailable)',
                    )
                    ax.axis('off')
                    continue

                diff = (raster_arrays[src_a] - raster_arrays[src_b]) * scale
                mask = (
                    (raster_arrays[src_a] == 0)
                    & (raster_arrays[src_b] == 0)
                )
                diff_masked = np.ma.masked_where(mask, diff)

                last_im = ax.imshow(
                    diff_masked, extent=extent, origin='upper',
                    cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax,
                    interpolation='nearest',
                )
                if basins_gdf is not None:
                    _overlay_boundaries(
                        ax, basins_gdf, ama_ina, name_col,
                        label_fontsize=5.0, label_all=True,
                    )
                ax.set_title(f'{src_a} \u2212 {src_b}', fontweight='bold')

            if last_im is not None:
                import matplotlib.ticker as mticker
                cbar = fig.colorbar(
                    last_im, ax=list(axes), shrink=0.5, pad=0.06,
                    orientation='horizontal', aspect=40, extend='both',
                )
                # For volume maps, scale tick labels by 1e6
                tick_div = ucfg.get('tick_div')
                if tick_div:
                    cbar.formatter = mticker.FuncFormatter(
                        lambda x, _: f'{x / tick_div:g}',
                    )
                    cbar.update_ticks()
                cbar.set_label(
                    ucfg['label'], fontsize=10, fontweight='bold',
                )
                cbar.ax.tick_params(labelsize=10)
                # Secondary unit axis
                sec_factor = ucfg['secondary_factor']
                if tick_div:
                    secax = cbar.ax.secondary_xaxis(
                        'top',
                        functions=(
                            lambda x: x * sec_factor,
                            lambda x: x / sec_factor,
                        ),
                    )
                else:
                    secax = cbar.ax.secondary_xaxis(
                        'top',
                        functions=(
                            lambda x: x * sec_factor,
                            lambda x: x / sec_factor,
                        ),
                    )
                secax.set_xlabel(
                    ucfg['secondary_label'],
                    fontsize=10, fontweight='bold',
                )
                secax.tick_params(labelsize=10)

            # Single AMA / INA / GW basin legend, outside-right of the
            # leftmost panel (one per figure rather than one per axes).
            if basins_gdf is not None:
                add_ama_ina_legend(axes[0])

            suffix = ucfg['suffix']
            out_path = os.path.join(
                output_dir, f'Spatial_Diff_{cat}{suffix}.png',
            )
            fig.savefig(out_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

    logger.info(f'Spatial difference maps saved to {output_dir}')


# ═════════════════════════════════════════════════════════════════════════════
# CU/IE: Load NHM HUC12 annual CSV → basin aggregates
# ═════════════════════════════════════════════════════════════════════════════
def _load_nhm_annual_csv_to_basins(
    csv_path: str,
    huc12_geojson: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    ref_raster: str,
    year_range: tuple[int, int],
    output_dir: str,
    mode: str = 'volume',
    predictor_dir: str | None = None,
    irr_fraction_band: int = 14,
    raster_label: str = 'CU',
) -> dict:
    """
    Generic loader for NHM annual HUC12 CSVs (CU or IE).

    For ``mode='volume'`` (CU): values are rates in Mgal/day — converted to
    annual volume (m³), then to depth (mm), rasterised, and aggregated to
    basin volumes in AF.  Returns
    ``{'mean': {basin: AF}, 'yearly': {year: {basin: AF}}}``.

    For ``mode='ratio'`` (IE): values are dimensionless ratios — the mean
    annual value is rasterised and aggregated to basin area-weighted means.
    Returns ``{'mean': {basin: ratio}, 'yearly': {year: {basin: ratio}}}``.

    Args:
        csv_path (str): Path to the NHM annual CSV (Year, <HUC12_code>, ...).
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_gdf (gpd.GeoDataFrame): Basin polygons (target CRS = reference raster CRS).
        basin_col (str): Column in *basin_gdf* naming each basin.
        ref_raster (str): Reference raster for grid/CRS information.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.
        output_dir (str): Directory for intermediate rasters.
        mode (str): ``'volume'`` for CU (Mgal -> AF) or ``'ratio'`` for IE.
        predictor_dir (str or None): Directory with ``Predictor_YYYY.tif`` rasters for irrigated-area
            scaling (used only when ``mode='volume'``).
        irr_fraction_band (int): Band number in predictor rasters for irrigated fraction
            (used only when ``mode='volume'``).
        raster_label (str): Label used in intermediate raster filenames (default ``'CU'``).

    Returns:
        dict: See description above.
    """
    makedirs(output_dir)
    start_yr, end_yr = year_range

    logger.info(f'Reading NHM annual CSV: {csv_path}')
    df = pd.read_csv(csv_path, dtype={'Year': int})
    huc_cols = [c for c in df.columns if c != 'Year']

    # AZ HUC12 polygons — drop cross-border HUC12s
    huc_gdf = gpd.read_file(huc12_geojson)
    huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    az_huc12_set = set(huc_gdf['huc12'].astype(str).values)
    az_cols = [c for c in huc_cols if c in az_huc12_set]
    logger.info(f'  {len(az_cols)} AZ-interior HUC12 regions found')

    if not az_cols:
        logger.warning('  No AZ HUC12 matches in CSV')
        fallback = np.nan if mode == 'ratio' else 0.0
        return {
            'mean': {b: fallback for b in basin_gdf[basin_col]},
            'yearly': {},
        }

    df_az = df[['Year'] + az_cols].copy()
    df_az = df_az[(df_az.Year >= start_yr) & (df_az.Year <= end_yr)]

    # Replace sentinels
    for col in az_cols:
        df_az[col] = pd.to_numeric(df_az[col], errors='coerce')
        df_az.loc[df_az[col].isin(NHM_SENTINEL), col] = np.nan
        if mode == 'volume':
            df_az[col] = df_az[col].fillna(0.0)

    # Reference raster properties
    with rio.open(ref_raster) as ref_src:
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_shape = (ref_src.height, ref_src.width)
        pixel_area_m2 = abs(ref_transform.a * ref_transform.e)

    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    huc_reproj = huc_gdf.to_crs(ref_crs)
    huc_reproj['area_m2'] = huc_reproj.geometry.area
    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    if mode == 'volume':
        return _nhm_cu_volume_path(
            df_az, az_cols, huc_reproj, basin_reproj, basin_col,
            ref_raster, ref_transform, ref_shape,
            pixel_area_m2, start_yr, end_yr, output_dir,
            predictor_dir, irr_fraction_band,
            raster_label=raster_label,
        )
    else:
        return _nhm_ie_ratio_path(
            df_az, az_cols, huc_reproj, basin_reproj, basin_col,
            ref_raster, ref_transform, ref_shape,
            start_yr, end_yr, output_dir,
        )


def _nhm_cu_volume_path(
    df_az, az_cols, huc_reproj, basin_reproj, basin_col,
    ref_raster, ref_transform, ref_shape,
    pixel_area_m2, start_yr, end_yr, output_dir,
    predictor_dir, irr_fraction_band,
    raster_label='CU',
) -> dict:
    """Process NHM CU CSV into basin volumes (AF)."""
    # Annual CSV values are rates in Mgal/day; convert to annual m³
    annual_records = []
    for _, row in df_az.iterrows():
        year = int(row['Year'])
        ndays = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        for huc_id in az_cols:
            val = row[huc_id]
            # Mgal/d × days → Mgal; × 3785.41178 → m³
            annual_records.append({
                'huc12': huc_id,
                'year': year,
                'volume_m3': val * ndays * MGAL_TO_M3,
            })

    ann_df = pd.DataFrame(annual_records)
    mean_annual = ann_df.groupby('huc12')['volume_m3'].mean().reset_index()
    mean_annual.columns = ['huc12', 'mean_vol_m3']

    huc_merged = huc_reproj.merge(mean_annual, on='huc12', how='left')
    huc_merged['mean_vol_m3'] = huc_merged['mean_vol_m3'].fillna(0.0)

    # Irrigated fraction (same logic as withdrawal NHM loader)
    huc_merged['irr_fraction'] = 1.0
    if predictor_dir is not None:
        logger.info('  Computing mean irrigated fraction per HUC12...')
        irr_counts = np.zeros(len(huc_merged))
        irr_n_years = 0
        for yr in range(start_yr, end_yr + 1):
            pred_file = os.path.join(predictor_dir, f'Predictor_{yr}.tif')
            if not os.path.isfile(pred_file):
                continue
            # Open raster once per year; reuse handle for all HUC12 clips
            with rio.open(pred_file) as src:
                for idx, row in huc_merged.iterrows():
                    geom = [mapping(row.geometry)]
                    try:
                        clipped, _ = rio_mask(src, geom, crop=True,
                                              all_touched=True,
                                              indexes=[irr_fraction_band],
                                              nodata=np.nan)
                        vals = clipped[0].astype(np.float64)
                        vals = vals[~np.isnan(vals)]
                        vals = np.clip(vals, 0, 1)
                        if vals.size > 0:
                            irr_counts[huc_merged.index.get_loc(idx)] += np.mean(vals)
                    except (ValueError, rio.errors.WindowError):
                        logger.debug('Irr fraction clipping failed for HUC at index %s', idx)
            irr_n_years += 1
        if irr_n_years > 0:
            huc_merged['irr_fraction'] = np.clip(
                irr_counts / irr_n_years, 0, 1,
            )

    # Volume → depth (mm)
    irr_area = huc_merged['area_m2'] * huc_merged['irr_fraction']
    huc_merged['depth_mm'] = np.where(
        irr_area > 0,
        huc_merged['mean_vol_m3'] / irr_area * M_TO_MM,
        0.0,
    )

    # Rasterize
    shapes = [
        (geom, val) for geom, val in
        zip(huc_merged.geometry, huc_merged['depth_mm'])
        if val > 0
    ]
    if shapes:
        nhm_raster = rasterize(
            shapes, out_shape=ref_shape, transform=ref_transform,
            fill=0.0, dtype='float64',
            merge_alg=rio.enums.MergeAlg.replace,
        )
    else:
        nhm_raster = np.zeros(ref_shape, dtype=np.float64)

    out_tif = os.path.join(output_dir, f'NHM_mean_annual_{raster_label}_mm.tif')
    with rio.open(ref_raster) as ref_src:
        profile = ref_src.profile.copy()
    profile.update(dtype='float64', nodata=np.nan, count=1)
    nhm_raster[nhm_raster == 0] = np.nan
    with rio.open(out_tif, 'w', **profile) as dst:
        dst.write(nhm_raster, 1)
    logger.info(f'  Wrote NHM {raster_label} raster: {out_tif}')

    # Per-year basin volumes via spatial overlay (correct mass-conserving
    # aggregation: volume × area_frac).  The rasterize-then-sum path
    # via _raster_basin_volumes over-counts for HUC12-level polygon
    # rasters, so we derive 'mean' from yearly_vols instead.
    yearly_vols = {}
    overlay = _get_huc_basin_overlay(
        huc_reproj, basin_reproj, basin_col,
        cache_dir=output_dir,
    )
    for year in range(start_yr, end_yr + 1):
        yr_vols = ann_df[ann_df.year == year].set_index('huc12')['volume_m3']
        merged = overlay.merge(
            yr_vols, left_on='huc12', right_index=True, how='left',
        ).fillna(0.0)
        merged['weighted_vol'] = merged['volume_m3'] * merged['area_frac']
        basin_sums = merged.groupby(basin_col)['weighted_vol'].sum()
        yearly_vols[year] = {
            b: basin_sums.get(b, 0.0) * M3_TO_AF
            for b in basin_reproj[basin_col]
        }

    mean_basin_vols: dict[str, float] = {}
    for b in basin_reproj[basin_col]:
        yr_vals = [yearly_vols[yr].get(b, 0.0) for yr in yearly_vols]
        mean_basin_vols[b] = float(np.mean(yr_vals)) if yr_vals else 0.0
    return {'mean': mean_basin_vols, 'yearly': yearly_vols}


def _nhm_ie_ratio_path(
    df_az, az_cols, huc_reproj, basin_reproj, basin_col,
    ref_raster, ref_transform, ref_shape,
    start_yr, end_yr, output_dir,
) -> dict:
    """Process NHM IE CSV into basin-mean efficiency ratios."""
    # Mean annual IE per HUC12
    mean_ie = df_az[az_cols].mean(axis=0)
    mean_ie_df = pd.DataFrame({
        'huc12': mean_ie.index.astype(str),
        'mean_ie': mean_ie.values,
    })

    out_tif = os.path.join(output_dir, 'NHM_mean_annual_IE.tif')

    if os.path.isfile(out_tif):
        logger.info(f'  Reusing existing NHM IE raster: {out_tif}')
    else:
        huc_merged = huc_reproj.merge(mean_ie_df, on='huc12', how='left')
        # NaN stays NaN (no data)

        # Rasterize
        shapes = [
            (geom, val) for geom, val in
            zip(huc_merged.geometry, huc_merged['mean_ie'])
            if np.isfinite(val) and val > 0
        ]
        if shapes:
            nhm_raster = rasterize(
                shapes, out_shape=ref_shape, transform=ref_transform,
                fill=np.nan, dtype='float64',
                merge_alg=rio.enums.MergeAlg.replace,
            )
        else:
            nhm_raster = np.full(ref_shape, np.nan, dtype=np.float64)

        with rio.open(ref_raster) as ref_src:
            profile = ref_src.profile.copy()
        profile.update(dtype='float64', nodata=np.nan, count=1)
        with rio.open(out_tif, 'w', **profile) as dst:
            dst.write(nhm_raster, 1)
        logger.info(f'  Wrote NHM IE raster: {out_tif}')

    # Basin-mean IE
    basin_means = _raster_basin_means(out_tif, basin_reproj, basin_col)

    # Per-year basin means via spatial overlay
    yearly_means = {}
    overlay = _get_huc_basin_overlay(
        huc_reproj, basin_reproj, basin_col,
        cache_dir=output_dir,
    )
    for year in range(start_yr, end_yr + 1):
        yr_row = df_az[df_az.Year == year]
        if yr_row.empty:
            continue
        yr_vals = yr_row[az_cols].iloc[0]
        yr_ie_df = pd.DataFrame({
            'huc12': yr_vals.index.astype(str),
            'ie': yr_vals.values,
        })
        merged = overlay.merge(yr_ie_df, on='huc12', how='left')
        valid = merged.dropna(subset=['ie'])
        valid = valid[valid['ie'].between(0, 1, inclusive='both')]
        if valid.empty:
            continue
        # Area-weighted mean per basin
        valid = valid.copy()
        valid['weighted_ie'] = valid['ie'] * valid['overlap_area']
        basin_wt = valid.groupby(basin_col).agg(
            wt_sum=('weighted_ie', 'sum'),
            area_sum=('overlap_area', 'sum'),
        )
        yearly_means[year] = {
            b: (basin_wt.loc[b, 'wt_sum'] / basin_wt.loc[b, 'area_sum']
                if b in basin_wt.index and basin_wt.loc[b, 'area_sum'] > 0
                else np.nan)
            for b in basin_reproj[basin_col]
        }

    return {'mean': basin_means, 'yearly': yearly_means}


def load_nhm_basin_ie(
    nhm_ie_csv: str,
    huc12_geojson: str,
    basin_shp: str,
    basin_col: str,
    ref_raster: str,
    output_dir: str,
    year_range: tuple[int, int] = (2000, 2020),
) -> dict:
    """
    Load USGS NHM irrigation efficiencies and aggregate to basin means.

    Returns per-year basin-level IEs (for years within the NHM range) and
    a long-term mean + std for each basin (used outside the NHM range).

    Args:
        nhm_ie_csv (str): Path to NHM IE CSV
            (``IR_HUC12_Eff_annual_2000_2020.csv``).
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_shp (str): Shapefile or GeoJSON for Arizona groundwater basins.
        basin_col (str): Column in *basin_shp* identifying each basin.
        ref_raster (str): Reference raster for CRS/grid information.
        output_dir (str): Directory for intermediate rasters.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.

    Returns:
        dict: ``{'per_year': {year: {basin: ie}},
               'mean': {basin: mean_ie},
               'std': {basin: std_ie},
               'overall_mean': float}``
    """
    makedirs(output_dir)
    cache_csv = os.path.join(output_dir, 'NHM_basin_IE_cache.csv')

    if os.path.isfile(cache_csv):
        logger.info(f'Loading cached NHM basin IE from {cache_csv}')
        return _load_nhm_ie_from_cache(cache_csv)

    basin_gdf = gpd.read_file(basin_shp)
    result = _load_nhm_annual_csv_to_basins(
        csv_path=nhm_ie_csv,
        huc12_geojson=huc12_geojson,
        basin_gdf=basin_gdf,
        basin_col=basin_col,
        ref_raster=ref_raster,
        year_range=year_range,
        output_dir=output_dir,
        mode='ratio',
    )

    # result = {'mean': {basin: ratio}, 'yearly': {year: {basin: ratio}}}
    yearly = result.get('yearly', {})
    basin_names = sorted(basin_gdf[basin_col].unique().tolist())

    # Compute per-basin std of IE across NHM years
    basin_std = {}
    for basin in basin_names:
        vals = [
            yearly[yr][basin]
            for yr in yearly
            if basin in yearly[yr] and np.isfinite(yearly[yr][basin])
        ]
        basin_std[basin] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Overall mean IE (across all basins)
    mean_vals = [v for v in result['mean'].values() if np.isfinite(v)]
    overall_mean = float(np.nanmean(mean_vals)) if mean_vals else 0.5

    logger.info(f'NHM basin IE: overall mean = {overall_mean:.3f}, '
                f'{len(yearly)} years, {len(basin_names)} basins')

    ie_dict = {
        'per_year': yearly,
        'mean': result['mean'],
        'std': basin_std,
        'overall_mean': overall_mean,
    }

    # Cache to CSV for fast reload on subsequent runs
    _save_nhm_ie_to_cache(ie_dict, cache_csv)
    return ie_dict


def _save_nhm_ie_to_cache(ie_dict: dict, cache_csv: str) -> None:
    """Serialize NHM basin IE dict to a CSV."""
    rows = []
    overall_mean = ie_dict['overall_mean']
    for basin, mean_ie in ie_dict['mean'].items():
        rows.append({
            'basin': basin,
            'year': 'mean',
            'ie': mean_ie,
            'std_ie': ie_dict['std'].get(basin, 0.0),
            'overall_mean': overall_mean,
        })
    for year, basin_ie in ie_dict['per_year'].items():
        for basin, ie_val in basin_ie.items():
            rows.append({
                'basin': basin,
                'year': year,
                'ie': ie_val,
                'std_ie': np.nan,
                'overall_mean': np.nan,
            })
    pd.DataFrame(rows).to_csv(cache_csv, index=False)
    logger.info(f'  Cached NHM basin IE to {cache_csv}')


def _load_nhm_ie_from_cache(cache_csv: str) -> dict:
    """Deserialize NHM basin IE dict from a cached CSV."""
    df = pd.read_csv(cache_csv)

    # Mean and std rows (year == 'mean')
    mean_rows = df[df['year'] == 'mean']
    basin_mean = dict(zip(mean_rows['basin'], mean_rows['ie']))
    basin_std = dict(zip(mean_rows['basin'], mean_rows['std_ie'].fillna(0.0)))
    overall_mean = float(mean_rows['overall_mean'].iloc[0])

    # Per-year rows
    yearly_rows = df[df['year'] != 'mean'].copy()
    yearly_rows['year'] = yearly_rows['year'].astype(int)
    per_year = {}
    for year, grp in yearly_rows.groupby('year'):
        per_year[int(year)] = dict(zip(grp['basin'], grp['ie']))

    logger.info(f'  Loaded {len(per_year)} years, {len(basin_mean)} basins, '
                f'overall mean IE = {overall_mean:.3f}')
    return {
        'per_year': per_year,
        'mean': basin_mean,
        'std': basin_std,
        'overall_mean': overall_mean,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CU/IE: Load ML rasters → basin aggregates
# ═════════════════════════════════════════════════════════════════════════════
def _load_ml_rasters_to_basins(
    raster_dir: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
    file_pattern: str,
    mode: str = 'volume',
) -> dict:
    """
    Load ML-produced rasters and aggregate to basin-scale.

    For ``mode='volume'`` (CU): compute mean-annual depth (mm) then basin
    total volumes (AF).   Returns ``{'mean': {basin: AF}, 'yearly': …}``.

    For ``mode='ratio'`` (IE): compute mean-annual ratio then basin
    area-weighted means.  Returns ``{'mean': {basin: ratio}, 'yearly': …}``.

    Args:
        raster_dir (str): Directory containing annual rasters.
        basin_gdf (gpd.GeoDataFrame): Basin polygons.
        basin_col (str): Basin name column.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive.
        file_pattern (str): Python format string with ``{year}`` placeholder,
            e.g. ``'Irrigation_CU_{year}_mm.tif'``.
        mode (str): ``'volume'`` for depth rasters or ``'ratio'`` for dimensionless.

    Returns:
        dict: Basin aggregates with ``'mean'`` and ``'yearly'`` keys.
    """
    start_yr, end_yr = year_range
    ref_raster = None
    mean_arr = None
    n_years = 0

    for year in range(start_yr, end_yr + 1):
        raster_path = os.path.join(raster_dir, file_pattern.format(year=year))
        if not os.path.isfile(raster_path):
            continue
        if ref_raster is None:
            ref_raster = raster_path
        arr = read_raster_as_arr(raster_path, get_file=False).astype(np.float64)
        if mode == 'volume':
            arr[np.isnan(arr)] = 0.0
            arr[arr < 0] = 0.0
        if mean_arr is None:
            mean_arr = arr.copy()
        else:
            if mode == 'ratio':
                # Nanmean accumulation
                mean_arr = np.nansum(
                    np.stack([mean_arr * n_years, arr]), axis=0,
                ) / (n_years + 1)
                n_years += 1
                continue
            mean_arr += arr
        n_years += 1

    if n_years == 0 or ref_raster is None:
        logger.warning(f'No ML rasters found in {raster_dir}')
        fallback = np.nan if mode == 'ratio' else 0.0
        return {
            'mean': {b: fallback for b in basin_gdf[basin_col]},
            'yearly': {},
        }

    if mode == 'volume':
        mean_arr /= n_years

    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    # Write mean-annual raster
    suffix = 'CU_mm' if mode == 'volume' else 'IE'
    out_tif = os.path.join(
        os.path.dirname(raster_dir.rstrip('/')),
        f'ML_mean_annual_{suffix}.tif',
    )
    with rio.open(ref_raster) as ref_src:
        profile = ref_src.profile.copy()
    profile.update(dtype='float64', nodata=np.nan, count=1)
    tmp = mean_arr.copy()
    if mode == 'volume':
        tmp[tmp == 0] = np.nan
    with rio.open(out_tif, 'w', **profile) as dst:
        dst.write(tmp, 1)
    logger.info(f'Wrote ML mean-annual raster: {out_tif}')

    if mode == 'volume':
        basin_agg = _raster_basin_volumes(
            out_tif, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
        )
    else:
        basin_agg = _raster_basin_means(out_tif, basin_reproj, basin_col)

    # Per-year
    yearly = {}
    for year in range(start_yr, end_yr + 1):
        raster_path = os.path.join(raster_dir, file_pattern.format(year=year))
        if not os.path.isfile(raster_path):
            continue
        if mode == 'volume':
            yearly[year] = _raster_basin_volumes(
                raster_path, basin_reproj, basin_col,
                pixel_area_m2, depth_unit='mm',
            )
        else:
            yearly[year] = _raster_basin_means(
                raster_path, basin_reproj, basin_col,
            )

    return {'mean': basin_agg, 'yearly': yearly}


# ═════════════════════════════════════════════════════════════════════════════
# CU/IE metrics (ratio variant)
# ═════════════════════════════════════════════════════════════════════════════
def _compute_ratio_metrics(
    basin_names: list[str],
    data_a: dict[str, float],
    data_b: dict[str, float],
    label_a: str,
    label_b: str,
) -> dict:
    """Compute RMSD, MAD, Percent Difference for dimensionless ratio data
    (e.g. irrigation efficiency).  Only basins where both values are finite
    are included.
    """
    vals_a, vals_b = [], []
    for b in basin_names:
        a = data_a.get(b, np.nan)
        bv = data_b.get(b, np.nan)
        if np.isfinite(a) and np.isfinite(bv):
            vals_a.append(a)
            vals_b.append(bv)

    if not vals_a:
        return {
            'Pair': f'{label_a} vs {label_b}',
            'RMSD': np.nan, 'MAD': np.nan, 'Pct_Diff': np.nan,
            f'Mean_{label_a}': np.nan, f'Mean_{label_b}': np.nan,
            'N_basins': 0,
        }

    a = np.array(vals_a)
    b = np.array(vals_b)
    diff = a - b
    rmsd = float(np.sqrt(np.mean(diff ** 2)))
    mad = float(np.mean(np.abs(diff)))
    denom = (np.mean(a) + np.mean(b)) / 2.0
    pct_diff = float(mad / denom * 100) if denom > 0 else np.nan

    return {
        'Pair': f'{label_a} vs {label_b}',
        'RMSD': round(rmsd, 4),
        'MAD': round(mad, 4),
        'Pct_Diff': round(pct_diff, 2),
        f'Mean_{label_a}': round(float(np.mean(a)), 4),
        f'Mean_{label_b}': round(float(np.mean(b)), 4),
        'N_basins': len(a),
    }


# CU/IE: Time series & scatter plotting


def run_intercomparison(
    ml_pred_dir: str,
    nhm_dir: str,
    reitz_base_dir: str,
    huc12_geojson: str,
    basin_shp: str,
    basin_col: str,
    output_dir: str,
    ref_raster: str | None = None,
    irr_gw_dir: str | None = None,
    irr_sw_dir: str | None = None,
    predictor_dir: str | None = None,
    ml_year_range: tuple[int, int] = (1980, 2020),
    nhm_year_range: tuple[int, int] = (2000, 2020),
    reitz_year_range: tuple[int, int] = (1980, 2018),
) -> pd.DataFrame:
    """
    Run the full three-way intercomparison for Irrigation GW, Irrigation
    SW, and Irrigation Total withdrawals across Arizona groundwater basins.

    Year ranges default to each dataset's actual coverage:
        ML: 1980-2020 (predictions available 1896-2099; 1980 chosen to
            give full historical context in time-series plots while
            keeping mean computations bounded)
        NHM: 2000-2020 (CSV native coverage)
        Reitz: 1980-2018 (Reitz native coverage)
    Per-basin means use **pairwise common-year windows** (intersection
    of each pair's coverage), so different-coverage datasets are
    averaged apples-to-apples in the metrics CSV and Stacked_Bar_Mean.
    Time-series plots use each source's full native range so the
    visual context is preserved (gaps appear where a dataset has no
    data for a given year).  Previously defaulted to 1980-2020 across
    all three, which diluted NHM means by including 20 zero-padded
    years 1980-1999.

    Args:
        ml_pred_dir (str): Directory with ``pred_YYYY.tif`` (or use *irr_gw_dir*/*irr_sw_dir*).
        nhm_dir (str): Directory containing the NHM CSV files.
        reitz_base_dir (str): Parent directory with Reitz sub-folders.
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_shp (str): Shapefile or GeoJSON for Arizona groundwater basins.
        basin_col (str): Column in *basin_shp* identifying each basin.
        output_dir (str): Root output directory for all results.
        ref_raster (str or None): Reference raster for CRS/grid.  Defaults to the first ML prediction.
        irr_gw_dir, irr_sw_dir (str or None): Optional category-specific ML raster directories.
        predictor_dir (str or None): Directory with ``Predictor_YYYY.tif`` rasters containing
            ``annual_irr_fraction``.  Passed to :func:`load_nhm_basin_volumes`
            so NHM volumes are converted to depth using irrigated area.
        ml_year_range, nhm_year_range, reitz_year_range (tuple[int, int]): Per-dataset year ranges (inclusive).

    Returns:
        pd.DataFrame: Summary metrics table for every pairwise comparison x category.
    """
    makedirs(output_dir)
    logger.info('='*60)
    logger.info('Irrigation Withdrawal Intercomparison')
    logger.info('='*60)

    # ── Load basin polygons ──────────────────────────────────────────────
    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Determine reference raster ───────────────────────────────────────
    if ref_raster is None:
        search_patterns = [
            (ml_pred_dir, 'pred_{yr}.tif'),
            (ml_pred_dir, 'Total_Predicted_{yr}_mm.tif'),
        ]
        if irr_gw_dir:
            search_patterns.append((irr_gw_dir, 'Irrigation_GW_{yr}_mm.tif'))
        if irr_sw_dir:
            search_patterns.append((irr_sw_dir, 'Irrigation_SW_{yr}_mm.tif'))
        for d, pattern in search_patterns:
            if ref_raster is not None:
                break
            for yr in range(ml_year_range[0], ml_year_range[1] + 1):
                candidate = os.path.join(d, pattern.format(yr=yr))
                if os.path.isfile(candidate):
                    ref_raster = candidate
                    break
    if ref_raster is None:
        raise FileNotFoundError(
            f'No ML prediction rasters found in {ml_pred_dir}'
        )
    logger.info(f'Reference raster: {ref_raster}')
    with rio.open(ref_raster) as _src:
        ref_crs = _src.crs

    # ── 1. ML predictions ────────────────────────────────────────────────
    logger.info('--- Loading ML predictions ---')
    ml_vols = load_ml_basin_volumes(
        ml_pred_dir, basin_gdf, basin_col, ml_year_range,
        irr_gw_dir=irr_gw_dir, irr_sw_dir=irr_sw_dir,
    )

    # ── 2. USGS NHM ──────────────────────────────────────────────────────
    logger.info('--- Loading USGS NHM data ---')
    nhm_out = os.path.join(output_dir, 'NHM_Rasters/')
    nhm_vols = load_nhm_basin_volumes(
        nhm_dir, huc12_geojson, basin_gdf, basin_col,
        ref_raster, nhm_year_range, nhm_out,
        predictor_dir=predictor_dir,
    )

    # ── 3. USGS Reitz ────────────────────────────────────────────────────
    logger.info('--- Loading USGS Reitz data ---')
    reitz_out = os.path.join(output_dir, 'Reitz_Rasters/')
    reitz_vols = load_reitz_basin_volumes(
        reitz_base_dir, ref_raster, basin_gdf, basin_col,
        reitz_year_range, reitz_out,
    )

    # ── 4. Compute metrics ───────────────────────────────────────────────
    basin_names = sorted(basin_gdf[basin_col].unique().tolist())
    all_metrics = []

    # Basin areas for mm conversion
    basin_reproj = (
        basin_gdf.to_crs(ref_crs)
        if basin_gdf.crs != ref_crs else basin_gdf
    )
    basin_areas_m2 = {
        row[basin_col]: row.geometry.area
        for _, row in basin_reproj.iterrows()
    }
    af_to_m3 = 1.0 / M3_TO_AF

    # Synthesize a 'Total' irrigation category per source = GW + SW
    # (per-basin and per-year-per-basin) so the intercomparison can
    # report a Total_Irrigation row alongside the GW / SW rows.  This
    # is useful because the per-basin GW caps at CO-direct basins
    # (Parker / Yuma / Lake Mohave) reshuffle volume between GW and
    # SW relative to off-the-shelf attribution products like Reitz —
    # the Total_Irrigation comparison cancels that reshuffle and
    # surfaces the underlying agreement on irrigation volume per basin.
    def _add_total_irrigation(vols: dict) -> None:
        # If the loader already produced a 'Total' entry (e.g. NHM
        # explicit monthly-Total CSV), keep it as the authoritative
        # source rather than overwriting with GW + SW synthesis.
        if 'Total' in vols and vols['Total'].get('mean'):
            return
        gw_mean = vols.get('GW', {}).get('mean', {})
        sw_mean = vols.get('SW', {}).get('mean', {})
        gw_yearly = vols.get('GW', {}).get('yearly', {})
        sw_yearly = vols.get('SW', {}).get('yearly', {})
        total_mean = {
            b: gw_mean.get(b, 0.0) + sw_mean.get(b, 0.0)
            for b in basin_names
        }
        years_union = set(gw_yearly.keys()) | set(sw_yearly.keys())
        total_yearly = {
            yr: {
                b: (gw_yearly.get(yr, {}).get(b, 0.0)
                    + sw_yearly.get(yr, {}).get(b, 0.0))
                for b in basin_names
            }
            for yr in years_union
        }
        vols['Total'] = {'mean': total_mean, 'yearly': total_yearly}

    for vols in (ml_vols, nhm_vols, reitz_vols):
        _add_total_irrigation(vols)

    # Pairwise common-year ranges (intersect each pair's native coverage):
    #   ML vs NHM   = (2000, 2020)
    #   ML vs Reitz = (1980, 2018)
    #   NHM vs Reitz = (2000, 2018)
    # Plus a three-way "Common" intersection (2000, 2018) so all three
    # are compared on the same axis for ribbon / overlap plots.
    def _intersect_yr_range(r_a, r_b):
        return (max(r_a[0], r_b[0]), min(r_a[1], r_b[1]))

    pair_yr_ranges = {
        ('ML', 'NHM'): _intersect_yr_range(ml_year_range, nhm_year_range),
        ('ML', 'Reitz'): _intersect_yr_range(ml_year_range, reitz_year_range),
        ('NHM', 'Reitz'): _intersect_yr_range(nhm_year_range, reitz_year_range),
    }
    common_yr_range = (
        max(ml_year_range[0], nhm_year_range[0], reitz_year_range[0]),
        min(ml_year_range[1], nhm_year_range[1], reitz_year_range[1]),
    )
    logger.info(
        f'Pairwise year ranges: ML-NHM={pair_yr_ranges[("ML", "NHM")]}, '
        f'ML-Reitz={pair_yr_ranges[("ML", "Reitz")]}, '
        f'NHM-Reitz={pair_yr_ranges[("NHM", "Reitz")]}; '
        f'Common (3-way)={common_yr_range}'
    )

    def _mean_over_years(vols_cat: dict, basin_names_local, yr_range) -> dict:
        """Recompute per-basin mean restricted to the given year range."""
        yearly = vols_cat.get('yearly', {})
        if not yearly:
            return vols_cat.get('mean', {})
        years_in_range = [y for y in yearly if yr_range[0] <= y <= yr_range[1]]
        if not years_in_range:
            return {b: 0.0 for b in basin_names_local}
        return {
            b: float(np.mean([
                yearly[y].get(b, 0.0) for y in years_in_range
            ])) for b in basin_names_local
        }

    for cat in ('GW', 'SW', 'Total'):
        logger.info(f'--- Irrigation {cat} metrics ---')
        pairs = [
            ('ML', 'NHM', ml_vols[cat], nhm_vols[cat]),
            ('ML', 'Reitz', ml_vols[cat], reitz_vols[cat]),
            ('NHM', 'Reitz', nhm_vols[cat], reitz_vols[cat]),
        ]
        for label_a, label_b, vols_a, vols_b in pairs:
            yr_range = pair_yr_ranges[(label_a, label_b)]
            data_a = _mean_over_years(vols_a, basin_names, yr_range)
            data_b = _mean_over_years(vols_b, basin_names, yr_range)
            m = _compute_metrics(
                basin_names, data_a, data_b, label_a, label_b,
                basin_areas_m2=basin_areas_m2,
            )
            m['Category'] = f'Irrigation_{cat}'
            m['Year_Range'] = f'{yr_range[0]}-{yr_range[1]}'
            all_metrics.append(m)
            logger.info(
                f'  {m["Pair"]} ({m["Year_Range"]}): '
                f'RMSD={m["RMSD_AF"]:.2f} AF '
                f'({m["RMSD_m3"]:.2f} m³), '
                f'MAD={m["MAD_AF"]:.2f} AF, PctDiff={m["Pct_Diff"]:.2f}%'
            )

        # Common 3-way intersection — same year range for all three datasets
        if common_yr_range[0] <= common_yr_range[1]:
            data_ml = _mean_over_years(ml_vols[cat], basin_names, common_yr_range)
            data_nhm = _mean_over_years(nhm_vols[cat], basin_names, common_yr_range)
            data_reitz = _mean_over_years(reitz_vols[cat], basin_names, common_yr_range)
            for la, lb, da, db in [
                ('ML', 'NHM', data_ml, data_nhm),
                ('ML', 'Reitz', data_ml, data_reitz),
                ('NHM', 'Reitz', data_nhm, data_reitz),
            ]:
                m = _compute_metrics(
                    basin_names, da, db, la, lb,
                    basin_areas_m2=basin_areas_m2,
                )
                m['Category'] = f'Irrigation_{cat}'
                m['Year_Range'] = f'Common_{common_yr_range[0]}-{common_yr_range[1]}'
                all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    col_order = [
        'Category', 'Pair',
        'RMSD_mm', 'NRMSD_mm', 'RMSD_ft', 'NRMSD_ft',
        'RMSD_m3', 'NRMSD_m3', 'RMSD_AF', 'NRMSD_AF',
        'MAD_mm', 'NMAD_mm', 'MAD_ft', 'NMAD_ft',
        'MAD_m3', 'NMAD_m3', 'MAD_AF', 'NMAD_AF',
        'Pct_Diff',
    ]
    extra_cols = [c for c in metrics_df.columns if c not in col_order]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns] + extra_cols]

    metrics_csv = os.path.join(output_dir, 'intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'Metrics saved to {metrics_csv}')

    # ── 4b. Temporal agreement metrics (Pearson r, NSE) ─────────────────
    logger.info('--- Interannual variability metrics ---')
    temporal_metrics = []
    temporal_per_basin_rows = []
    for cat in ('GW', 'SW', 'Total'):
        pairs = [
            ('ML', 'NHM', ml_vols[cat].get('yearly', {}),
             nhm_vols[cat].get('yearly', {})),
            ('ML', 'Reitz', ml_vols[cat].get('yearly', {}),
             reitz_vols[cat].get('yearly', {})),
            ('NHM', 'Reitz', nhm_vols[cat].get('yearly', {}),
             reitz_vols[cat].get('yearly', {})),
        ]
        for label_a, label_b, yearly_a, yearly_b in pairs:
            tm = _compute_temporal_metrics(
                basin_names, yearly_a, yearly_b, label_a, label_b,
            )
            summary = {
                'Category': f'Irrigation_{cat}',
                'Pair': tm['Pair'],
                'Pearson_r_mean': tm['Pearson_r_mean'],
                'Pearson_r_median': tm['Pearson_r_median'],
                'NSE_mean': tm['NSE_mean'],
                'NSE_median': tm['NSE_median'],
                'n_common_years': tm['n_common_years'],
                'n_basins_with_data': tm['n_basins_with_data'],
            }
            temporal_metrics.append(summary)
            logger.info(
                f'  Irrigation {cat} {tm["Pair"]}: '
                f'Pearson r (mean={tm["Pearson_r_mean"]}, '
                f'median={tm["Pearson_r_median"]}), '
                f'NSE (mean={tm["NSE_mean"]}, '
                f'median={tm["NSE_median"]}), '
                f'{tm["n_common_years"]} common years, '
                f'{tm["n_basins_with_data"]} basins'
            )
            for pb in tm.get('per_basin', []):
                temporal_per_basin_rows.append({
                    'Category': f'Irrigation_{cat}',
                    'Pair': tm['Pair'],
                    **pb,
                })

    temporal_df = pd.DataFrame(temporal_metrics)
    temporal_csv = os.path.join(output_dir, 'temporal_agreement_metrics.csv')
    temporal_df.to_csv(temporal_csv, index=False)
    logger.info(f'Temporal agreement metrics saved to {temporal_csv}')

    if temporal_per_basin_rows:
        temporal_basin_df = pd.DataFrame(temporal_per_basin_rows)
        temporal_basin_csv = os.path.join(
            output_dir, 'temporal_agreement_per_basin.csv',
        )
        temporal_basin_df.to_csv(temporal_basin_csv, index=False)
        logger.info(f'Per-basin temporal metrics saved to {temporal_basin_csv}')

    # ── 4c. Temporal agreement visualizations ────────────────────────────
    all_sources = {'ML': ml_vols, 'NHM': nhm_vols, 'Reitz': reitz_vols}
    if temporal_per_basin_rows:
        temporal_basin_df = pd.DataFrame(temporal_per_basin_rows)
        temporal_plot_dir = os.path.join(output_dir, 'Temporal_Agreement/')
        plot_temporal_heatmap(temporal_basin_df, temporal_plot_dir)
        plot_temporal_box_violin(temporal_basin_df, temporal_plot_dir)
        pair_colors_4a = {
            'ML vs NHM': '#2C3E50', 'ML vs Reitz': '#E67E22',
            'NHM vs Reitz': '#27AE60',
        }
        # Taylor diagrams removed — they bundle correlation, normalised
        # std, and centred RMSD onto a single panel that's hard to
        # interpret for model-vs-model intercomparisons (no "true"
        # reference exists; all three sources are estimates).
        plot_temporal_r_vs_nse(
            temporal_basin_df, temporal_plot_dir,
            pair_colors=pair_colors_4a,
        )

    # ── 5. Per-basin comparison table (mm, m³, AF) ──────────────────────
    # Native columns (ML_AF / NHM_AF / Reitz_AF) report each dataset
    # over its OWN year range.  Common columns (suffix _Common_AF)
    # report each dataset over the 3-way intersection year range so
    # ML / NHM / Reitz can be compared apples-to-apples on the same
    # axis.  Pairwise common-range deltas are in intercomparison_metrics.csv.
    rows = []
    for cat in ('GW', 'SW', 'Total'):
        ml_common = _mean_over_years(ml_vols[cat], basin_names, common_yr_range)
        nhm_common = _mean_over_years(nhm_vols[cat], basin_names, common_yr_range)
        reitz_common = _mean_over_years(
            reitz_vols[cat], basin_names, common_yr_range,
        )
        for basin in basin_names:
            ml_af = ml_vols[cat]['mean'].get(basin, 0.0)
            nhm_af = nhm_vols[cat]['mean'].get(basin, 0.0)
            reitz_af = reitz_vols[cat]['mean'].get(basin, 0.0)
            ml_c_af = ml_common.get(basin, 0.0)
            nhm_c_af = nhm_common.get(basin, 0.0)
            reitz_c_af = reitz_common.get(basin, 0.0)
            area = basin_areas_m2.get(basin, 1.0)

            rows.append({
                'Category': f'Irrigation_{cat}',
                'Basin': basin,
                'ML_mm': round(ml_af * af_to_m3 / area * M_TO_MM, 4),
                'ML_ft': round(ml_af * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                'ML_m3': round(ml_af * af_to_m3, 2),
                'ML_AF': round(ml_af, 2),
                'NHM_mm': round(nhm_af * af_to_m3 / area * M_TO_MM, 4),
                'NHM_ft': round(nhm_af * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                'NHM_m3': round(nhm_af * af_to_m3, 2),
                'NHM_AF': round(nhm_af, 2),
                'Reitz_mm': round(reitz_af * af_to_m3 / area * M_TO_MM, 4),
                'Reitz_ft': round(reitz_af * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                'Reitz_m3': round(reitz_af * af_to_m3, 2),
                'Reitz_AF': round(reitz_af, 2),
                'ML_Common_AF': round(ml_c_af, 2),
                'NHM_Common_AF': round(nhm_c_af, 2),
                'Reitz_Common_AF': round(reitz_c_af, 2),
            })
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'per_basin_volumes.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(
        f'Per-basin volumes saved to {basin_csv} '
        f'(native + Common {common_yr_range[0]}-{common_yr_range[1]})'
    )

    # ── 6. Time series CSV ───────────────────────────────────────────────
    ts_rows = []
    for cat in ('GW', 'SW', 'Total'):
        for source_name, src_data in all_sources.items():
            yearly = src_data[cat].get('yearly', {})
            for year in sorted(yearly.keys()):
                for basin in basin_names:
                    af_val = yearly[year].get(basin, 0.0)
                    area = basin_areas_m2.get(basin, 1.0)
                    ts_rows.append({
                        'Category': f'Irrigation_{cat}',
                        'Source': source_name,
                        'Year': year,
                        'Basin': basin,
                        'Volume_mm': round(af_val * af_to_m3 / area * M_TO_MM, 4),
                        'Volume_ft': round(af_val * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                        'Volume_m3': round(af_val * af_to_m3, 2),
                        'Volume_AF': round(af_val, 2),
                    })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'time_series_volumes.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'Time series saved to {ts_csv}')

    # ── 7. Time series plots ─────────────────────────────────────────────
    _ts_colors = {'ML': '#2C3E50', 'NHM': '#27AE60', 'Reitz': '#E67E22'}
    _ts_markers = {'ML': 'o', 'NHM': 's', 'Reitz': '^'}
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    plot_intercomp_time_series(
        all_sources, categories=['GW', 'SW', 'Total'],
        basin_names=basin_names, basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_ts_colors, markers=_ts_markers,
        title_prefix='Irrigation ', file_prefix='TS',
    )

    # ── 8. Scatter plots ────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    for cat in ('GW', 'SW', 'Total'):
        scatter_pairs = [
            (sa, sb, all_sources[sa][cat]['mean'], all_sources[sb][cat]['mean'])
            for sa, sb in [('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')]
        ]
        plot_intercomp_scatter(
            scatter_pairs, basin_names, basin_areas_m2, scatter_dir,
            title=f'Irrigation {cat} — Per-Basin Scatter Comparison',
            filename=f'Scatter_{cat}.png',
        )

    # ── 8b. Statewide stacked bar plots ────────────────────────────────
    # Mean-annual summary bar uses the 3-way Common year range so
    # ML / NHM / Reitz are averaged over the same 19-year window
    # (otherwise Reitz's 1980-2018 range would inflate its mean
    # relative to ML / NHM averaged over 2000-2020).
    bar_dir = os.path.join(output_dir, 'Stacked_Bar/')
    plot_intercomp_stacked_bars(
        all_sources, source_order=['ML', 'NHM', 'Reitz'],
        output_dir=bar_dir,
        stack_cats=['GW', 'SW'],
        stack_labels={'GW': 'Groundwater', 'SW': 'Surface Water'},
        stack_colors={'GW': '#2C3E50', 'SW': '#3498DB'},
        title_prefix='Irrigation Withdrawal — ',
        mean_year_range=common_yr_range,
    )

    # ── 9. Spatial difference maps ───────────────────────────────────────
    ml_parent = os.path.dirname(ml_pred_dir.rstrip('/'))
    mean_raster_paths = {
        'ML':    {cat: os.path.join(ml_parent, f'ML_mean_annual_{cat}_mm.tif')
                  for cat in ('GW', 'SW')},
        'NHM':   {cat: os.path.join(nhm_out, f'NHM_mean_annual_{cat}_mm.tif')
                  for cat in ('GW', 'SW')},
        'Reitz': {cat: os.path.join(reitz_out, f'Reitz_mean_annual_{cat}_mm.tif')
                  for cat in ('GW', 'SW')},
    }
    diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
    _plot_spatial_diff_maps(
        mean_raster_paths, ref_raster, diff_dir,
        basin_shp=basin_shp, basin_col=basin_col,
    )

    # ── 9b. Basin-aggregated Δ volume choropleths with pct + 95% CI ───
    # Per-basin pct difference annotated at each basin centroid; ML σ
    # loaded from Sigma_Total/Rasters/ when available (others treated as
    # deterministic).  Produces ML−NHM, ML−Reitz, NHM−Reitz pairs for
    # each category (GW, SW, Total_Irrigation).
    sigma_raster_dir_irr = os.path.join(
        ml_parent, 'Uncertainty', 'Sigma_Total', 'Rasters',
    )
    _ml_cat_to_sigma_prefix_irr = {
        'GW': 'Irrigation_GW',
        'SW': 'Irrigation_SW',
        'Total': 'Irrigation',
    }

    def _ml_sigma_mean_irr(cat_key: str) -> dict[str, float]:
        prefix = _ml_cat_to_sigma_prefix_irr.get(cat_key)
        if not prefix:
            return {}
        per_yr = _load_basin_sigma_yearly(
            sigma_raster_dir_irr, prefix, basin_reproj, basin_col,
            (nhm_year_range[0], nhm_year_range[1]),
        )
        out: dict[str, float] = {}
        for b, yr_dict in per_yr.items():
            vals = [v for v in yr_dict.values() if np.isfinite(v) and v > 0]
            if vals:
                out[b] = float(np.mean(vals))
        return out

    pairs_basin = [
        ('ML', 'NHM', ml_vols, nhm_vols),
        ('ML', 'Reitz', ml_vols, reitz_vols),
        ('NHM', 'Reitz', nhm_vols, reitz_vols),
    ]
    cat_keys_basin = [k for k in ('GW', 'SW', 'Total')
                      if k in ml_vols and k in nhm_vols and k in reitz_vols]
    # One figure per category — three pair panels (ML−NHM, ML−Reitz,
    # NHM−Reitz) side-by-side with a shared colorbar.
    for cat in cat_keys_basin:
        cat_label = (
            'Total_Irrigation' if cat == 'Total'
            else f'Irrigation_{cat}'
        )
        panels = []
        for label_a, label_b, vols_a, vols_b in pairs_basin:
            a_mean = vols_a.get(cat, {}).get('mean', {})
            b_mean = vols_b.get(cat, {}).get('mean', {})
            if not (a_mean and b_mean):
                continue
            panels.append({
                'basin_a_vols': a_mean,
                'basin_b_vols': b_mean,
                'panel_title': f'{label_a} \u2212 {label_b}',
                'label_a': label_a,
                'label_b': label_b,
            })
        if not panels:
            continue
        _plot_basin_diff_panels(
            panels=panels,
            basin_gdf=basin_reproj,
            basin_col=basin_col,
            title=(
                f'{cat_label.replace("_", " ")} \u2014 '
                f'Basin-Level Volume Diff'
            ),
            out_path=os.path.join(
                diff_dir, f'Spatial_Diff_Basin_{cat_label}.png',
            ),
            shared_colorbar=True,
        )
    logger.info(
        'Basin-level Δ volume multi-panel maps saved to %s', diff_dir,
    )

    # ── 10. HUC12-level comparison (ML/Reitz aggregated to NHM's native unit)
    # NHM reports at HUC12 resolution; aggregating ML/Reitz to HUC12 via
    # zonal statistics gives an apples-to-apples comparison without the
    # lossy polygon→pixel→basin round-trip and without the cross-basin
    # area-weighted split that can misattribute irrigated volume from a
    # mixed-use HUC12 (e.g. Phoenix/Harquahala boundary).
    logger.info('--- HUC12-level comparison (ML/Reitz → NHM native unit) ---')
    huc12_dir = os.path.join(output_dir, 'HUC12_Comparison/')
    makedirs(huc12_dir)

    huc_gdf = gpd.read_file(huc12_geojson)
    huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    with rio.open(ref_raster) as _src:
        pixel_area_m2 = abs(_src.transform.a * _src.transform.e)
        huc_reproj = huc_gdf.to_crs(_src.crs)

    az_huc12_ids = sorted(huc_reproj['huc12'].unique())
    huc12_metrics = []

    # NHM raw volumes per HUC12 (already computed during load_nhm_basin_volumes
    # as ann_df, but not exposed; recompute from CSV for cleanliness)
    nhm_cats = {
        'GW': os.path.join(nhm_dir, 'IR_HUC12_GW_WD_monthly_2000_2020.csv'),
        'SW': os.path.join(nhm_dir, 'IR_HUC12_SW_WD_monthly_2000_2020.csv'),
        'Total': os.path.join(nhm_dir, 'IR_HUC12_Tot_WD_monthly_2000_2020.csv'),
    }
    # Per-cat ML raster directory and prefix (Total uses the
    # Irrigation_*_mm.tif rasters; Reitz uses the All_irr_YYYY.tif
    # rasters in the Irrigation_all_1980-2018 subdir).
    irr_total_dir = (
        os.path.join(os.path.dirname(irr_gw_dir.rstrip('/')),
                     'Irrigation_Rasters', 'Depth_mm')
        if irr_gw_dir else None
    )
    ml_cat_dirs = {
        'GW': (irr_gw_dir, 'Irrigation_GW'),
        'SW': (irr_sw_dir, 'Irrigation_SW'),
        'Total': (irr_total_dir, 'Irrigation'),
    }
    reitz_cat_specs = {
        'GW': ('Irrigation_groundwater_1980-2018', 'GW_irr'),
        'SW': ('Irrigation_surfacewater_1980-2018', 'SW_irr'),
        'Total': ('Irrigation_all_1980-2018', 'All_irr'),
    }
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    nhm_start, nhm_end = nhm_year_range

    for cat in ('GW', 'SW', 'Total'):
        logger.info(f'  HUC12-level comparison for Irrigation {cat}...')

        # ── NHM per-HUC12 mean-annual volume (AF) ──
        nhm_csv = nhm_cats[cat]
        if not os.path.isfile(nhm_csv):
            logger.warning(f'  NHM CSV not found: {nhm_csv}')
            continue
        nhm_df = pd.read_csv(nhm_csv)
        az_cols = [c for c in nhm_df.columns
                   if c not in ('Year', 'Month') and c in set(az_huc12_ids)]
        if not az_cols:
            continue
        nhm_sub = nhm_df[['Year', 'Month'] + az_cols].copy()
        nhm_sub = nhm_sub[
            (nhm_sub.Year >= nhm_start) & (nhm_sub.Year <= nhm_end)
        ]
        for col in az_cols:
            nhm_sub[col] = pd.to_numeric(nhm_sub[col], errors='coerce')
            nhm_sub.loc[nhm_sub[col].isin(NHM_SENTINEL), col] = 0.0
            nhm_sub[col] = nhm_sub[col].fillna(0.0)
        # Annual volumes per HUC12
        nhm_huc_annual: dict[str, list[float]] = {h: [] for h in az_cols}
        for year in range(nhm_start, nhm_end + 1):
            yr_df = nhm_sub[nhm_sub.Year == year]
            annual_vol = np.zeros(len(az_cols))
            for _, row in yr_df.iterrows():
                month = int(row['Month'])
                ndays = days_in_month[month - 1]
                if month == 2 and year % 4 == 0 and (
                        year % 100 != 0 or year % 400 == 0):
                    ndays = 29
                vals = row[az_cols].values.astype(np.float64)
                annual_vol += vals * ndays * MGAL_TO_M3
            for i, huc_id in enumerate(az_cols):
                nhm_huc_annual[huc_id].append(annual_vol[i] * M3_TO_AF)
        nhm_huc_mean = {
            h: float(np.mean(v)) if v else 0.0
            for h, v in nhm_huc_annual.items()
        }

        # ── ML per-HUC12 mean-annual volume (AF) via zonal stats ──
        ml_cat_dir, cat_prefix = ml_cat_dirs[cat]
        if ml_cat_dir is None:
            logger.warning(f'  ML {cat} raster dir not provided')
            continue
        ml_huc_accum: dict[str, list[float]] = {h: [] for h in az_huc12_ids}
        ml_start, ml_end = ml_year_range
        for year in range(max(ml_start, nhm_start), min(ml_end, nhm_end) + 1):
            raster_path = os.path.join(
                ml_cat_dir, f'{cat_prefix}_{year}_mm.tif',
            )
            if not os.path.isfile(raster_path):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                raster_path, huc_reproj, pixel_area_m2, depth_unit='mm',
            )
            for huc_id in az_huc12_ids:
                s = yr_stats.get(huc_id)
                if s:
                    ml_huc_accum[huc_id].append(s['volume_AF'])
        ml_huc_mean = {
            h: float(np.mean(v)) if v else 0.0
            for h, v in ml_huc_accum.items()
        }

        # ── Reitz per-HUC12 mean-annual volume (AF) via zonal stats ──
        reitz_huc_mean: dict[str, float] = {}
        reitz_cat_subdir, reitz_prefix = reitz_cat_specs[cat]
        reitz_huc_accum: dict[str, list[float]] = {
            h: [] for h in az_huc12_ids
        }
        reitz_start, reitz_end = reitz_year_range
        for year in range(
                max(reitz_start, nhm_start), min(reitz_end, nhm_end) + 1):
            reitz_raw = os.path.join(
                reitz_base_dir, reitz_cat_subdir,
                f'{reitz_prefix}_{year}.tif',
            )
            reproj_path = os.path.join(
                reitz_out, f'Reitz_{cat}_{year}_reproj.tif',
            )
            if os.path.isfile(reproj_path):
                yr_stats = _compute_huc12_zonal_stats(
                    reproj_path, huc_reproj, pixel_area_m2, depth_unit='m',
                )
            elif os.path.isfile(reitz_raw):
                _reproject_reitz_to_ref(reitz_raw, ref_raster, reproj_path)
                yr_stats = _compute_huc12_zonal_stats(
                    reproj_path, huc_reproj, pixel_area_m2, depth_unit='m',
                )
            else:
                continue
            for huc_id in az_huc12_ids:
                s = yr_stats.get(huc_id)
                if s:
                    reitz_huc_accum[huc_id].append(s['volume_AF'])
        reitz_huc_mean = {
            h: float(np.mean(v)) if v else 0.0
            for h, v in reitz_huc_accum.items()
        }

        # ── Compute HUC12-level metrics ──
        common_hucs = [h for h in az_huc12_ids
                       if h in nhm_huc_mean and h in ml_huc_mean]
        if common_hucs:
            huc_areas = {
                str(row['huc12']): row.geometry.area
                for _, row in huc_reproj.iterrows()
            }
            for pair_label, dict_a, dict_b, name_a, name_b in [
                ('ML_vs_NHM', ml_huc_mean, nhm_huc_mean, 'ML', 'NHM'),
                ('Reitz_vs_NHM', reitz_huc_mean, nhm_huc_mean, 'Reitz', 'NHM'),
                ('ML_vs_Reitz', ml_huc_mean, reitz_huc_mean, 'ML', 'Reitz'),
            ]:
                m = _compute_metrics(
                    common_hucs, dict_a, dict_b, name_a, name_b,
                    basin_areas_m2=huc_areas,
                )
                m['Category'] = f'Irrigation_{cat}'
                m['Level'] = 'HUC12'
                huc12_metrics.append(m)
                logger.info(
                    f'    {pair_label} (HUC12): RMSD={m["RMSD_AF"]:.1f} AF, '
                    f'PctDiff={m["Pct_Diff"]:.1f}%'
                )

        # ── HUC12-level scatter ──
        huc12_scatter_dir = os.path.join(huc12_dir, 'Scatter/')
        makedirs(huc12_scatter_dir)
        huc12_scatter_pairs = [
            ('ML', 'NHM', ml_huc_mean, nhm_huc_mean),
            ('ML', 'Reitz', ml_huc_mean, reitz_huc_mean),
            ('NHM', 'Reitz', nhm_huc_mean, reitz_huc_mean),
        ]
        plot_intercomp_scatter(
            huc12_scatter_pairs, common_hucs, huc_areas,
            huc12_scatter_dir,
            title=f'Irrigation {cat} — HUC12-Level Scatter',
            filename=f'Scatter_HUC12_{cat}.png',
        )

        # ── HUC12-level per-unit CSV ──
        huc12_rows = []
        for huc_id in az_huc12_ids:
            area = huc_areas.get(huc_id, 1.0)
            af_to_m3_local = 1.0 / M3_TO_AF
            huc12_rows.append({
                'HUC12': huc_id,
                'Category': f'Irrigation_{cat}',
                'ML_AF': round(ml_huc_mean.get(huc_id, 0.0), 2),
                'NHM_AF': round(nhm_huc_mean.get(huc_id, 0.0), 2),
                'Reitz_AF': round(reitz_huc_mean.get(huc_id, 0.0), 2),
                'ML_mm': round(
                    ml_huc_mean.get(huc_id, 0.0) * af_to_m3_local
                    / area * M_TO_MM, 4,
                ) if area > 0 else 0.0,
                'NHM_mm': round(
                    nhm_huc_mean.get(huc_id, 0.0) * af_to_m3_local
                    / area * M_TO_MM, 4,
                ) if area > 0 else 0.0,
                'Reitz_mm': round(
                    reitz_huc_mean.get(huc_id, 0.0) * af_to_m3_local
                    / area * M_TO_MM, 4,
                ) if area > 0 else 0.0,
            })
        pd.DataFrame(huc12_rows).to_csv(
            os.path.join(huc12_dir, f'per_huc12_{cat}.csv'), index=False,
        )

        # ── HUC12-level spatial diff choropleth maps ──
        # Color each HUC12 polygon by the pairwise depth or volume
        # difference, with GW basin boundaries and AMA/INA labels
        # overlaid for geographic context.
        logger.info(f'  Rendering HUC12-level spatial diff maps for {cat}...')
        huc12_diff_dir = os.path.join(huc12_dir, 'Spatial_Diff/')
        makedirs(huc12_diff_dir)

        basin_reproj = (
            basin_gdf.to_crs(huc_reproj.crs)
            if basin_gdf.crs != huc_reproj.crs else basin_gdf
        )
        ama_ina_names = get_ama_ina_basin_names()
        b_name_col = (
            basin_col if basin_col in basin_reproj.columns
            else basin_reproj.columns[0]
        )

        af_to_m3_local = 1.0 / M3_TO_AF
        _mm_to_ft = 1.0 / 304.8
        _m3_to_af_local = M3_TO_AF

        nhm_pairs = [
            ('ML', 'NHM', ml_huc_mean, nhm_huc_mean),
            ('Reitz', 'NHM', reitz_huc_mean, nhm_huc_mean),
            ('ML', 'Reitz', ml_huc_mean, reitz_huc_mean),
        ]

        # Volume-only HUC12 diff (depth-mode dropped — at this scale the
        # depth signal is dominated by polygon-area variation rather
        # than the source comparison we want to visualize).
        for unit_mode, unit_label, sec_label, sec_factor, scale_fn, tick_div in [
            (
                'volume',
                r'$\Delta$ Volume ($\times$10$^{6}$ m$^3$)',
                '\u0394 Volume (AF)',
                _m3_to_af_local,
                lambda af, area: af * af_to_m3_local,
                1e6,
            ),
        ]:
            fig, axes = plt.subplots(
                1, 3, figsize=(20, 7), constrained_layout=True,
            )
            title_unit = 'Volume'
            fig.suptitle(
                f'Irrigation {cat} \u2014 HUC12-Level {title_unit} Difference',
                fontsize=14, fontweight='bold',
            )

            # Compute shared vmax
            global_vmax = 1e-6
            for _, _, dict_a, dict_b in nhm_pairs:
                diffs = []
                for h in common_hucs:
                    area = huc_areas.get(h, 1.0)
                    va = scale_fn(dict_a.get(h, 0.0), area)
                    vb = scale_fn(dict_b.get(h, 0.0), area)
                    if va != 0 or vb != 0:
                        diffs.append(va - vb)
                if diffs:
                    d_arr = np.array(diffs)
                    global_vmax = max(
                        global_vmax,
                        abs(np.nanpercentile(d_arr, 2)),
                        abs(np.nanpercentile(d_arr, 98)),
                    )

            last_im = None
            for col_i, (name_a, name_b, dict_a, dict_b) in enumerate(
                    nhm_pairs):
                ax = axes[col_i]
                ax.set_facecolor('#D5D5D5')

                # Compute per-HUC12 diff and assign to geometry
                diff_vals = []
                for h in common_hucs:
                    area = huc_areas.get(h, 1.0)
                    va = scale_fn(dict_a.get(h, 0.0), area)
                    vb = scale_fn(dict_b.get(h, 0.0), area)
                    diff_vals.append(va - vb)

                plot_gdf = huc_reproj[
                    huc_reproj['huc12'].isin(common_hucs)
                ].copy()
                plot_gdf = plot_gdf.set_index('huc12').loc[common_hucs]
                plot_gdf['diff'] = diff_vals
                # Mask HUC12s where both sources are zero
                zero_mask = plot_gdf['diff'].abs() < 1e-10
                plot_gdf.loc[zero_mask, 'diff'] = np.nan

                last_im = plot_gdf.plot(
                    ax=ax, column='diff', cmap='RdBu_r',
                    vmin=-global_vmax, vmax=global_vmax,
                    edgecolor='#AAAAAA', linewidth=0.3,
                    legend=False, missing_kwds={'color': '#EEEEEE'},
                )
                # Overlay GW basin boundaries + AMA/INA labels
                _overlay_boundaries(
                    ax, basin_reproj, ama_ina_names, b_name_col,
                    label_fontsize=5.0, label_all=True,
                )
                ax.set_title(
                    f'{name_a} \u2212 {name_b}', fontweight='bold',
                )

            # Shared colorbar with dual units
            import matplotlib.ticker as mticker
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(
                cmap='RdBu_r',
                norm=Normalize(vmin=-global_vmax, vmax=global_vmax),
            )
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=list(axes), shrink=0.5, pad=0.06,
                orientation='horizontal', aspect=40, extend='both',
            )
            if tick_div:
                cbar.formatter = mticker.FuncFormatter(
                    lambda x, _: f'{x / tick_div:g}',
                )
                cbar.update_ticks()
            cbar.set_label(unit_label, fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            secax = cbar.ax.secondary_xaxis(
                'top',
                functions=(
                    lambda x: x * sec_factor,
                    lambda x: x / sec_factor,
                ),
            )
            secax.set_xlabel(
                sec_label, fontsize=10, fontweight='bold',
            )
            secax.tick_params(labelsize=10)
            add_ama_ina_legend(axes[0])

            suffix = '' if unit_mode == 'depth' else '_Volume'
            out_path = os.path.join(
                huc12_diff_dir,
                f'Spatial_Diff_HUC12_{cat}{suffix}.png',
            )
            fig.savefig(out_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

        logger.info(f'  HUC12-level diff maps saved to {huc12_diff_dir}')

        # ── HUC12-level temporal diagnostics (box-violin + Taylor) ──
        # Rebuild per-year {year: {huc12: AF}} dicts from the raw CSVs
        # and zonal stats for temporal agreement computation.
        logger.info(f'  Computing HUC12-level temporal diagnostics for {cat}...')

        # NHM yearly: {year: {huc12: AF}}
        nhm_yearly_huc: dict[int, dict[str, float]] = {}
        for year in range(nhm_start, nhm_end + 1):
            yr_df = nhm_sub[nhm_sub.Year == year]
            annual_vol = np.zeros(len(az_cols))
            for _, row in yr_df.iterrows():
                month = int(row['Month'])
                ndays = days_in_month[month - 1]
                if month == 2 and year % 4 == 0 and (
                        year % 100 != 0 or year % 400 == 0):
                    ndays = 29
                vals = row[az_cols].values.astype(np.float64)
                annual_vol += vals * ndays * MGAL_TO_M3
            nhm_yearly_huc[year] = {
                az_cols[i]: annual_vol[i] * M3_TO_AF
                for i in range(len(az_cols))
            }

        # ML yearly: {year: {huc12: AF}}
        ml_yearly_huc: dict[int, dict[str, float]] = {}
        for year in range(max(ml_start, nhm_start), min(ml_end, nhm_end) + 1):
            raster_path = os.path.join(
                ml_cat_dir, f'{cat_prefix}_{year}_mm.tif',
            )
            if not os.path.isfile(raster_path):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                raster_path, huc_reproj, pixel_area_m2, depth_unit='mm',
            )
            ml_yearly_huc[year] = {
                h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                for h in az_huc12_ids
            }

        # Reitz yearly: {year: {huc12: AF}}
        reitz_yearly_huc: dict[int, dict[str, float]] = {}
        for year in range(
                max(reitz_start, nhm_start), min(reitz_end, nhm_end) + 1):
            reproj_path = os.path.join(
                reitz_out, f'Reitz_{cat}_{year}_reproj.tif',
            )
            if not os.path.isfile(reproj_path):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                reproj_path, huc_reproj, pixel_area_m2, depth_unit='m',
            )
            reitz_yearly_huc[year] = {
                h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                for h in az_huc12_ids
            }

        _huc12_temporal_diagnostics(
            huc12_yearly_sources={
                'ML': ml_yearly_huc,
                'NHM': nhm_yearly_huc,
                'Reitz': reitz_yearly_huc,
            },
            pairs=[('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')],
            huc12_ids=common_hucs,
            category=f'Irrigation_{cat}',
            output_dir=huc12_dir,
            huc_areas=huc_areas,
        )

    if huc12_metrics:
        huc12_metrics_df = pd.DataFrame(huc12_metrics)
        huc12_metrics_df.to_csv(
            os.path.join(huc12_dir, 'huc12_intercomparison_metrics.csv'),
            index=False,
        )
        logger.info(f'\nHUC12-level metrics:\n{huc12_metrics_df.to_string(index=False)}')

    # ── 11. Summary print ────────────────────────────────────────────────
    logger.info('\n' + '='*60)
    logger.info('Intercomparison Summary')
    logger.info('='*60)
    logger.info(f'\nBasin-level metrics:\n{metrics_df.to_string(index=False)}')
    if huc12_metrics:
        logger.info(f'\nHUC12-level metrics:\n{huc12_metrics_df.to_string(index=False)}')
    logger.info(f'\nML year range: {ml_year_range}')
    logger.info(f'NHM year range: {nhm_year_range}')
    logger.info(f'Reitz year range: {reitz_year_range}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# CU Intercomparison (ML vs NHM)
# ═════════════════════════════════════════════════════════════════════════════
def run_cu_intercomparison(
    irr_cu_dir: str,
    nhm_cu_csv: str,
    huc12_geojson: str,
    basin_shp: str,
    basin_col: str,
    output_dir: str,
    ref_raster: str | None = None,
    predictor_dir: str | None = None,
    ml_year_range: tuple[int, int] = (2000, 2020),
    nhm_year_range: tuple[int, int] = (2000, 2020),
) -> pd.DataFrame:
    """
    Run basin-scale intercomparison of Irrigation Consumptive Use (CU)
    between ML predictions and USGS NHM data.

    CU comparison follows the same volume-based framework as withdrawals
    (metrics in AF, m³, mm).

    Args:
        irr_cu_dir (str): Directory with ``Irrigation_CU_{year}_mm.tif`` rasters.
        nhm_cu_csv (str): Path to NHM CU CSV
            (``Irr_CU_HUC12_Tot_annual_2000_2020.csv``).
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_shp (str): Shapefile or GeoJSON for Arizona groundwater basins.
        basin_col (str): Column in *basin_shp* identifying each basin.
        output_dir (str): Root output directory for CU results.
        ref_raster (str or None): Reference raster for CRS/grid.  Defaults to the first ML CU raster.
        predictor_dir (str or None): Directory with ``Predictor_YYYY.tif`` rasters for irrigated-area
            scaling in NHM CU conversion.
        ml_year_range, nhm_year_range (tuple[int, int]): Per-dataset year ranges (inclusive).

    Returns:
        pd.DataFrame: Summary metrics table.
    """
    makedirs(output_dir)
    logger.info('=' * 60)
    logger.info('Irrigation CU Intercomparison')
    logger.info('=' * 60)

    # ── Load basins ──────────────────────────────────────────────────────
    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Determine reference raster ───────────────────────────────────────
    if ref_raster is None:
        for yr in range(ml_year_range[0], ml_year_range[1] + 1):
            candidate = os.path.join(
                irr_cu_dir, f'Irrigation_CU_{yr}_mm.tif',
            )
            if os.path.isfile(candidate):
                ref_raster = candidate
                break
    if ref_raster is None:
        raise FileNotFoundError(
            f'No Irrigation_CU rasters found in {irr_cu_dir}'
        )
    logger.info(f'Reference raster: {ref_raster}')
    with rio.open(ref_raster) as _src:
        ref_crs = _src.crs

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )
    basin_names = sorted(basin_gdf[basin_col].unique().tolist())
    basin_areas_m2 = {
        row[basin_col]: row.geometry.area
        for _, row in basin_reproj.iterrows()
    }
    af_to_m3 = 1.0 / M3_TO_AF
    all_metrics = []

    # ═════════════════════════════════════════════════════════════════════
    # CU comparison
    # ═════════════════════════════════════════════════════════════════════
    logger.info('--- Loading ML CU rasters ---')
    ml_cu = _load_ml_rasters_to_basins(
        irr_cu_dir, basin_gdf, basin_col, ml_year_range,
        file_pattern='Irrigation_CU_{year}_mm.tif',
        mode='volume',
    )

    logger.info('--- Loading NHM CU data ---')
    nhm_cu_out = os.path.join(output_dir, 'NHM_CU_Rasters/')
    nhm_cu = _load_nhm_annual_csv_to_basins(
        nhm_cu_csv, huc12_geojson, basin_gdf, basin_col,
        ref_raster, nhm_year_range, nhm_cu_out,
        mode='volume',
        predictor_dir=predictor_dir,
    )

    logger.info('--- CU metrics ---')
    m_cu = _compute_metrics(
        basin_names, ml_cu['mean'], nhm_cu['mean'], 'ML', 'NHM',
        basin_areas_m2=basin_areas_m2,
    )
    m_cu['Category'] = 'Irrigation_CU'
    all_metrics.append(m_cu)
    logger.info(
        f'  ML vs NHM CU: RMSD={m_cu["RMSD_AF"]:.2f} AF '
        f'({m_cu["RMSD_m3"]:.2f} m³), '
        f'MAD={m_cu["MAD_AF"]:.2f} AF, PctDiff={m_cu["Pct_Diff"]:.2f}%'
    )

    # ── Metrics CSV ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, 'cu_intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'CU metrics saved to {metrics_csv}')

    # ── Per-basin comparison tables ──────────────────────────────────────
    rows = []
    for basin in basin_names:
        area = basin_areas_m2.get(basin, 1.0)
        ml_af = ml_cu['mean'].get(basin, 0.0)
        nhm_af = nhm_cu['mean'].get(basin, 0.0)
        rows.append({
            'Category': 'Irrigation_CU',
            'Basin': basin,
            'ML_mm': round(ml_af * af_to_m3 / area * M_TO_MM, 4),
            'ML_AF': round(ml_af, 2),
            'NHM_mm': round(nhm_af * af_to_m3 / area * M_TO_MM, 4),
            'NHM_AF': round(nhm_af, 2),
        })
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'cu_per_basin.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(f'Per-basin CU saved to {basin_csv}')

    # ── Time series CSV ──────────────────────────────────────────────────
    ts_rows = []
    for source_name, cu_src in [('ML', ml_cu), ('NHM', nhm_cu)]:
        cu_yearly = cu_src.get('yearly', {})
        for year in sorted(cu_yearly.keys()):
            for basin in basin_names:
                af_val = cu_yearly[year].get(basin, 0.0)
                area = basin_areas_m2.get(basin, 1.0)
                ts_rows.append({
                    'Category': 'Irrigation_CU',
                    'Source': source_name,
                    'Year': year,
                    'Basin': basin,
                    'Value_mm': round(af_val * af_to_m3 / area * M_TO_MM, 4),
                    'Value_AF': round(af_val, 2),
                })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'cu_time_series.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'CU time series saved to {ts_csv}')

    # ── Plots ────────────────────────────────────────────────────────────
    _cu_colors = {'ML': '#2C3E50', 'NHM': '#27AE60'}
    _cu_markers = {'ML': 'o', 'NHM': 's'}
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    cu_sources = {'ML': {'CU': ml_cu}, 'NHM': {'CU': nhm_cu}}
    plot_intercomp_time_series(
        cu_sources, categories=['CU'],
        basin_names=basin_names, basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_cu_colors, markers=_cu_markers,
        title_prefix='Irrigation ', file_prefix='TS_CU',
    )

    scatter_dir = os.path.join(output_dir, 'Scatter/')
    plot_intercomp_scatter(
        [('ML', 'NHM', ml_cu['mean'], nhm_cu['mean'])],
        basin_names, basin_areas_m2, scatter_dir,
        title='Irrigation CU — Per-Basin Scatter (ML vs NHM)',
        filename='Scatter_CU.png',
    )

    # Statewide stacked bar (CU is a single category, no GW/SW split).
    # Use distinct per-source colours (ML blue, NHM black) since the
    # default cat-color + alpha-shift fallback is hard to read with a
    # single category.
    bar_dir = os.path.join(output_dir, 'Stacked_Bar/')
    plot_intercomp_stacked_bars(
        {'ML': {'CU': ml_cu}, 'NHM': {'CU': nhm_cu}},
        source_order=['ML', 'NHM'],
        output_dir=bar_dir,
        stack_cats=['CU'],
        stack_labels={'CU': 'Irrigation CU'},
        stack_colors={'CU': '#27AE60'},
        title_prefix='Irrigation CU — ',
        source_colors={'ML': '#1F77B4', 'NHM': '#000000'},
    )

    # ── HUC12-level comparison (ML aggregated to NHM's native unit) ────
    logger.info('--- HUC12-level CU comparison ---')
    huc12_dir = os.path.join(output_dir, 'HUC12_Comparison/')
    makedirs(huc12_dir)

    huc_gdf = gpd.read_file(huc12_geojson)
    huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    with rio.open(ref_raster) as _src:
        pixel_area_m2 = abs(_src.transform.a * _src.transform.e)
        huc_reproj = huc_gdf.to_crs(_src.crs)

    az_huc12_ids = sorted(huc_reproj['huc12'].unique())
    huc_areas = {
        str(row['huc12']): row.geometry.area
        for _, row in huc_reproj.iterrows()
    }

    # NHM CU per HUC12 (annual CSV, already in Mgal/d → AF)
    nhm_df_raw = pd.read_csv(nhm_cu_csv, dtype={'Year': int})
    az_cols = [c for c in nhm_df_raw.columns
               if c != 'Year' and c in set(az_huc12_ids)]
    nhm_huc_mean: dict[str, float] = {}
    if az_cols:
        nhm_sub = nhm_df_raw[['Year'] + az_cols].copy()
        nhm_sub = nhm_sub[
            (nhm_sub.Year >= nhm_year_range[0])
            & (nhm_sub.Year <= nhm_year_range[1])
        ]
        for col in az_cols:
            nhm_sub[col] = pd.to_numeric(nhm_sub[col], errors='coerce')
            nhm_sub.loc[nhm_sub[col].isin(NHM_SENTINEL), col] = 0.0
            nhm_sub[col] = nhm_sub[col].fillna(0.0)
        # NHM CU CSV is annual Mgal/d — convert to AF/yr
        # annual_vol = rate_Mgal_d × 365.25 × MGAL_TO_M3 × M3_TO_AF
        days_per_year = 365.25
        for h in az_cols:
            vals = nhm_sub[h].values * days_per_year * MGAL_TO_M3 * M3_TO_AF
            nhm_huc_mean[h] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    # ML CU per HUC12 via zonal stats
    ml_start, ml_end = ml_year_range
    nhm_start, nhm_end = nhm_year_range
    ml_huc_accum: dict[str, list[float]] = {h: [] for h in az_huc12_ids}
    for year in range(max(ml_start, nhm_start), min(ml_end, nhm_end) + 1):
        rpath = os.path.join(irr_cu_dir, f'Irrigation_CU_{year}_mm.tif')
        if not os.path.isfile(rpath):
            continue
        yr_stats = _compute_huc12_zonal_stats(
            rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
        )
        for h in az_huc12_ids:
            s = yr_stats.get(h)
            if s:
                ml_huc_accum[h].append(s['volume_AF'])
    ml_huc_mean = {
        h: float(np.mean(v)) if v else 0.0
        for h, v in ml_huc_accum.items()
    }

    common_hucs = [h for h in az_huc12_ids
                   if h in nhm_huc_mean and h in ml_huc_mean]
    if common_hucs:
        m_cu_huc = _compute_metrics(
            common_hucs, ml_huc_mean, nhm_huc_mean, 'ML', 'NHM',
            basin_areas_m2=huc_areas,
        )
        m_cu_huc['Category'] = 'Irrigation_CU'
        m_cu_huc['Level'] = 'HUC12'
        pd.DataFrame([m_cu_huc]).to_csv(
            os.path.join(huc12_dir, 'huc12_cu_metrics.csv'), index=False,
        )
        logger.info(
            f'  ML vs NHM CU (HUC12): RMSD={m_cu_huc["RMSD_AF"]:.1f} AF, '
            f'PctDiff={m_cu_huc["Pct_Diff"]:.1f}%'
        )

        # HUC12-level scatter
        huc12_scatter_dir = os.path.join(huc12_dir, 'Scatter/')
        plot_intercomp_scatter(
            [('ML', 'NHM', ml_huc_mean, nhm_huc_mean)],
            common_hucs, huc_areas, huc12_scatter_dir,
            title='Irrigation CU — HUC12-Level Scatter',
            filename='Scatter_HUC12_CU.png',
        )

        # Combined HUC12 + Pixel + Basin Δ depth (3 panels, shared
        # colorbar).  Δ Depth (mm) is the only physically meaningful
        # unit shared across pixel / HUC12 / basin aggregation levels —
        # Δ Volume per polygon scales by ~10^4 between pixel and basin
        # and would visually saturate the basin panel while zeroing the
        # pixel panel under one colorbar.  The single-mode standalone
        # depth and volume HUC12 maps were dropped in favour of this
        # unified 3-scale figure.
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        _af_to_m3_local = 1.0 / M3_TO_AF
        _mm_to_ft = 1.0 / 304.8
        spatial_diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
        makedirs(spatial_diff_dir)
        b_reproj = (
            basin_gdf.to_crs(huc_reproj.crs)
            if basin_gdf.crs != huc_reproj.crs else basin_gdf
        )
        b_name = (
            basin_col if basin_col in b_reproj.columns
            else b_reproj.columns[0]
        )
        ama_ina_names = get_ama_ina_basin_names()

        # Panel 1: HUC12 Δ depth (AF / m² → mm)
        huc12_diff_vals: dict[str, float] = {}
        for h in common_hucs:
            area = huc_areas.get(h, 0.0)
            if area > 0:
                huc12_diff_vals[h] = (
                    (ml_huc_mean.get(h, 0.0) - nhm_huc_mean.get(h, 0.0))
                    * _af_to_m3_local / area * M_TO_MM
                )

        # Panel 2: Pixel-level Δ depth (read mean-annual rasters)
        ml_mean_raster = os.path.join(
            os.path.dirname(irr_cu_dir.rstrip('/')),
            'ML_mean_annual_CU_mm.tif',
        )
        nhm_mean_raster = os.path.join(
            nhm_cu_out, 'NHM_mean_annual_CU_mm.tif',
        )
        pixel_diff_arr = None
        pixel_extent = None
        if os.path.isfile(ml_mean_raster) and os.path.isfile(nhm_mean_raster):
            with rio.open(ml_mean_raster) as src:
                ml_arr = src.read(1).astype(np.float64)
                pixel_extent = [
                    src.bounds.left, src.bounds.right,
                    src.bounds.bottom, src.bounds.top,
                ]
            with rio.open(nhm_mean_raster) as src:
                nhm_arr = src.read(1).astype(np.float64)
            ml_arr[np.isnan(ml_arr)] = 0.0
            nhm_arr[np.isnan(nhm_arr)] = 0.0
            pixel_diff_arr = ml_arr - nhm_arr

        # Panel 3: Basin Δ depth (AF / m² → mm)
        basin_areas_lookup = {
            row[basin_col]: row.geometry.area
            for _, row in b_reproj.iterrows()
        }
        basin_diff_vals: dict[str, float] = {}
        for b in basin_names:
            area = basin_areas_lookup.get(b, 0.0)
            if area > 0:
                basin_diff_vals[b] = (
                    (ml_cu['mean'].get(b, 0.0) - nhm_cu['mean'].get(b, 0.0))
                    * _af_to_m3_local / area * M_TO_MM
                )

        # Shared vmax across the three panels (2nd/98th percentile)
        vmax_candidates: list[float] = []
        if huc12_diff_vals:
            vals = np.array(
                [v for v in huc12_diff_vals.values() if abs(v) > 1e-6],
            )
            if vals.size:
                vmax_candidates.extend([
                    abs(np.nanpercentile(vals, 2)),
                    abs(np.nanpercentile(vals, 98)),
                ])
        if pixel_diff_arr is not None:
            pix_vals = pixel_diff_arr[np.abs(pixel_diff_arr) > 1e-6]
            if pix_vals.size:
                vmax_candidates.extend([
                    abs(np.nanpercentile(pix_vals, 2)),
                    abs(np.nanpercentile(pix_vals, 98)),
                ])
        if basin_diff_vals:
            vals = np.array(
                [v for v in basin_diff_vals.values() if abs(v) > 1e-6],
            )
            if vals.size:
                vmax_candidates.extend([
                    abs(np.nanpercentile(vals, 2)),
                    abs(np.nanpercentile(vals, 98)),
                ])
        vmax = max(vmax_candidates) if vmax_candidates else 1.0
        vmax = max(vmax, 1e-6)

        fig, axes = plt.subplots(
            1, 3, figsize=(20, 7), constrained_layout=True,
        )
        fig.suptitle(
            'Irrigation CU \u2014 Mean-Annual Depth Difference '
            '(ML \u2212 NHM)',
            fontsize=14, fontweight='bold',
        )

        # Panel 1: HUC12
        ax_huc = axes[0]
        ax_huc.set_facecolor('#D5D5D5')
        plot_gdf = huc_reproj[huc_reproj['huc12'].isin(common_hucs)].copy()
        plot_gdf = plot_gdf.set_index('huc12').loc[common_hucs]
        plot_gdf['diff'] = [
            huc12_diff_vals.get(h, np.nan) for h in common_hucs
        ]
        plot_gdf.loc[plot_gdf['diff'].abs() < 1e-6, 'diff'] = np.nan
        plot_gdf.plot(
            ax=ax_huc, column='diff', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            edgecolor='#AAAAAA', linewidth=0.3,
            legend=False, missing_kwds={'color': '#EEEEEE'},
        )
        _overlay_boundaries(
            ax_huc, b_reproj, ama_ina_names, b_name,
            label_fontsize=5.0, label_all=False,
        )
        ax_huc.set_title('HUC12-Level', fontweight='bold')

        # Panel 2: Pixel-level
        ax_pix = axes[1]
        ax_pix.set_facecolor('#D5D5D5')
        if pixel_diff_arr is not None:
            pix_mask = np.abs(pixel_diff_arr) < 1e-6
            pix_masked = np.ma.masked_where(pix_mask, pixel_diff_arr)
            ax_pix.imshow(
                pix_masked, extent=pixel_extent, origin='upper',
                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                interpolation='nearest',
            )
            _overlay_boundaries(
                ax_pix, b_reproj, ama_ina_names, b_name,
                label_fontsize=5.0, label_all=False,
            )
        else:
            ax_pix.text(
                0.5, 0.5, 'Pixel-level rasters unavailable',
                ha='center', va='center', transform=ax_pix.transAxes,
            )
            ax_pix.axis('off')
        ax_pix.set_title('Pixel-Level', fontweight='bold')

        # Panel 3: Basin
        ax_bas = axes[2]
        ax_bas.set_facecolor('#D5D5D5')
        plot_b = b_reproj.set_index(basin_col).copy()
        plot_b['diff'] = plot_b.index.map(
            lambda b: basin_diff_vals.get(b, np.nan),
        )
        plot_b.loc[plot_b['diff'].abs() < 1e-6, 'diff'] = np.nan
        plot_b.plot(
            ax=ax_bas, column='diff', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            edgecolor='#666666', linewidth=0.5,
            legend=False, missing_kwds={'color': '#EEEEEE'},
        )
        _overlay_boundaries(
            ax_bas, b_reproj, ama_ina_names, b_name,
            label_fontsize=5.0, label_all=False,
        )
        ax_bas.set_title('Basin-Level', fontweight='bold')

        # Single shared colorbar (Δ Depth mm, secondary axis ft)
        sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(-vmax, vmax))
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=list(axes), shrink=0.5, pad=0.06,
            orientation='horizontal', aspect=40, extend='both',
        )
        cbar.set_label(
            '\u0394 Depth (mm)', fontsize=10, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=10)
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(
                lambda x: x * _mm_to_ft,
                lambda x: x / _mm_to_ft,
            ),
        )
        secax.set_xlabel(
            '\u0394 Depth (ft)', fontsize=10, fontweight='bold',
        )
        secax.tick_params(labelsize=10)

        add_ama_ina_legend(axes[0])
        fig.savefig(
            os.path.join(spatial_diff_dir, 'Spatial_Diff_CU.png'),
            dpi=600, bbox_inches='tight',
        )
        plt.close(fig)
        logger.info(
            f'  CU 3-panel diff figure saved to {spatial_diff_dir}'
        )

        # ── HUC12-level temporal diagnostics ──
        logger.info('  Computing HUC12-level CU temporal diagnostics...')
        # NHM yearly {year: {huc12: AF}}
        nhm_yearly_huc_cu: dict[int, dict[str, float]] = {}
        days_per_year = 365.25
        for year in range(nhm_year_range[0], nhm_year_range[1] + 1):
            yr_row = nhm_sub[nhm_sub.Year == year]
            if yr_row.empty:
                continue
            nhm_yearly_huc_cu[year] = {
                h: float(yr_row[h].values[0]) * days_per_year * MGAL_TO_M3 * M3_TO_AF
                if h in yr_row.columns else 0.0
                for h in az_cols
            }
        # ML yearly {year: {huc12: AF}}
        ml_yearly_huc_cu: dict[int, dict[str, float]] = {}
        for year in range(max(ml_start, nhm_start), min(ml_end, nhm_end) + 1):
            rpath = os.path.join(irr_cu_dir, f'Irrigation_CU_{year}_mm.tif')
            if not os.path.isfile(rpath):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
            )
            ml_yearly_huc_cu[year] = {
                h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                for h in az_huc12_ids
            }
        _huc12_temporal_diagnostics(
            huc12_yearly_sources={'ML': ml_yearly_huc_cu, 'NHM': nhm_yearly_huc_cu},
            pairs=[('ML', 'NHM')],
            huc12_ids=common_hucs,
            category='Irrigation_CU',
            output_dir=huc12_dir,
            huc_areas=huc_areas,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('CU Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
    logger.info(f'\nML year range: {ml_year_range}')
    logger.info(f'NHM year range: {nhm_year_range}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# Effective Precipitation Intercomparison (USDA-SCS Peff vs ML Peff PCML vs NHM)


def _load_ml_peff_to_basins(
    predictor_dir: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
    output_dir: str,
    peff_band: int,
    irr_fraction_band: int = 14,
    label: str = 'Peff',
) -> dict:
    """Extract Peff from predictor rasters, scale by ``irr_fraction``,
    and aggregate to basin volumes (AF).

    For each year the irrigated-area effective precipitation depth is
    ``depth = Peff_mm × irr_fraction``.  The scaled depth is written as
    a single-band raster, then aggregated to basin volumes.

    Returns ``{'mean': {basin: AF}, 'yearly': {year: {basin: AF}}}``.
    """
    makedirs(output_dir)
    start_yr, end_yr = year_range
    ref_raster = None
    mean_depth = None
    n_years = 0

    for year in range(start_yr, end_yr + 1):
        pred_file = os.path.join(predictor_dir, f'Predictor_{year}.tif')
        if not os.path.isfile(pred_file):
            continue

        with rio.open(pred_file) as src:
            peff_arr = src.read(peff_band).astype(np.float64)
            irr_arr = src.read(irr_fraction_band).astype(np.float64)
            if ref_raster is None:
                ref_raster = pred_file

        peff_arr[np.isnan(peff_arr)] = 0.0
        peff_arr[peff_arr < 0] = 0.0
        irr_arr[np.isnan(irr_arr)] = 0.0
        irr_arr = np.clip(irr_arr, 0, 1)

        scaled = peff_arr * irr_arr

        # Write per-year raster
        out_tif = os.path.join(output_dir, f'ML_{label}_{year}_mm.tif')
        with rio.open(ref_raster) as ref_src:
            yr_profile = ref_src.profile.copy()
        yr_profile.update(dtype='float64', nodata=np.nan, count=1)
        tmp = scaled.copy()
        tmp[tmp == 0] = np.nan
        with rio.open(out_tif, 'w', **yr_profile) as dst:
            dst.write(tmp, 1)

        if mean_depth is None:
            mean_depth = scaled.copy()
        else:
            mean_depth += scaled
        n_years += 1

    if n_years == 0 or ref_raster is None:
        logger.warning(f'No predictor rasters found for {label}')
        return {
            'mean': {b: 0.0 for b in basin_gdf[basin_col]},
            'yearly': {},
        }

    mean_depth /= n_years

    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    # Write mean-annual raster
    out_mean = os.path.join(output_dir, f'ML_mean_annual_{label}_mm.tif')
    with rio.open(ref_raster) as ref_src:
        profile = ref_src.profile.copy()
    profile.update(dtype='float64', nodata=np.nan, count=1)
    tmp = mean_depth.copy()
    tmp[tmp == 0] = np.nan
    with rio.open(out_mean, 'w', **profile) as dst:
        dst.write(tmp, 1)
    logger.info(f'Wrote ML mean-annual {label} raster: {out_mean}')

    basin_vols = _raster_basin_volumes(
        out_mean, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
    )

    # Per-year basin volumes
    yearly_vols = {}
    for year in range(start_yr, end_yr + 1):
        raster_path = os.path.join(output_dir, f'ML_{label}_{year}_mm.tif')
        if os.path.isfile(raster_path):
            yearly_vols[year] = _raster_basin_volumes(
                raster_path, basin_reproj, basin_col,
                pixel_area_m2, depth_unit='mm',
            )

    return {'mean': basin_vols, 'yearly': yearly_vols}


def run_peff_intercomparison(
    predictor_dir: str,
    nhm_peff_csv: str,
    huc12_geojson: str,
    basin_shp: str,
    basin_col: str,
    output_dir: str,
    ref_raster: str | None = None,
    peff_band: int = 4,
    peff_pcml_band: int = 5,
    irr_fraction_band: int = 14,
    ml_year_range: tuple[int, int] = (2000, 2024),
    ml_pcml_year_range: tuple[int, int] = (2000, 2023),
    nhm_year_range: tuple[int, int] = (2000, 2020),
) -> pd.DataFrame:
    """
    Compare irrigated effective precipitation across three sources:

        1. **USDA-SCS Peff** — SCS formula-based (predictor band 4 × irr_fraction)
        2. **ML Peff PCML** — observation-based (predictor band 5 × irr_fraction)
        3. **NHM Peff** — USGS NHM PPTeff HUC12 data (Mgal/day → basin AF)

    All three datasets are scaled by ``annual_irr_fraction`` so that
    volumes represent only the irrigated-area contribution.

    Args:
        predictor_dir (str): Directory with ``Predictor_YYYY.tif`` multi-band rasters.
        nhm_peff_csv (str): Path to NHM PPTeff CSV
            (``PPTeff_HUC12_Tot_annual_2000_2020.csv``).
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_shp (str): Shapefile or GeoJSON for Arizona groundwater basins.
        basin_col (str): Column in *basin_shp* identifying each basin.
        output_dir (str): Root output directory.
        ref_raster (str or None): Reference raster for CRS/grid.  Defaults to the first predictor.
        peff_band (int): Band index for ``annual_peff_mm`` (default 4).
        peff_pcml_band (int): Band index for ``annual_peff_pcml_mm`` (default 5).
        irr_fraction_band (int): Band index for ``annual_irr_fraction`` (default 14).
        ml_year_range (tuple[int, int]): Year range for ML Peff (default 2000-2024).
        ml_pcml_year_range (tuple[int, int]): Year range for ML Peff PCML (default 2000-2023).
        nhm_year_range (tuple[int, int]): Year range for NHM PPTeff (default 2000-2020).

    Returns:
        pd.DataFrame: Summary metrics table.
    """
    makedirs(output_dir)
    logger.info('=' * 60)
    logger.info('Effective Precipitation Intercomparison')
    logger.info('=' * 60)

    # ── Load basins ──────────────────────────────────────────────────────
    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Determine reference raster ───────────────────────────────────────
    if ref_raster is None:
        for yr in range(ml_year_range[0], ml_year_range[1] + 1):
            candidate = os.path.join(predictor_dir, f'Predictor_{yr}.tif')
            if os.path.isfile(candidate):
                ref_raster = candidate
                break
    if ref_raster is None:
        raise FileNotFoundError(
            f'No Predictor rasters found in {predictor_dir}'
        )
    logger.info(f'Reference raster: {ref_raster}')
    with rio.open(ref_raster) as _src:
        ref_crs = _src.crs

    basin_reproj = (
        basin_gdf.to_crs(ref_crs)
        if basin_gdf.crs != ref_crs else basin_gdf
    )
    basin_names = sorted(basin_gdf[basin_col].unique().tolist())
    basin_areas_m2 = {
        row[basin_col]: row.geometry.area
        for _, row in basin_reproj.iterrows()
    }
    af_to_m3 = 1.0 / M3_TO_AF

    # ── 1. USDA-SCS Peff ──────────────────────────────────────────────
    logger.info('--- Loading USDA-SCS Peff (SCS formula) ---')
    ml_peff_out = os.path.join(output_dir, 'ML_Peff_Rasters/')
    ml_peff = _load_ml_peff_to_basins(
        predictor_dir, basin_gdf, basin_col, ml_year_range,
        ml_peff_out, peff_band=peff_band,
        irr_fraction_band=irr_fraction_band, label='Peff',
    )

    # ── 2. ML Peff PCML ─────────────────────────────────────────────────
    logger.info('--- Loading ML Peff PCML (observation-based) ---')
    ml_pcml_out = os.path.join(output_dir, 'ML_Peff_PCML_Rasters/')
    ml_peff_pcml = _load_ml_peff_to_basins(
        predictor_dir, basin_gdf, basin_col, ml_pcml_year_range,
        ml_pcml_out, peff_band=peff_pcml_band,
        irr_fraction_band=irr_fraction_band, label='Peff_PCML',
    )

    # ── 3. NHM PPTeff ────────────────────────────────────────────────────
    logger.info('--- Loading USGS NHM PPTeff ---')
    nhm_peff_out = os.path.join(output_dir, 'NHM_Peff_Rasters/')
    nhm_peff = _load_nhm_annual_csv_to_basins(
        nhm_peff_csv, huc12_geojson, basin_gdf, basin_col,
        ref_raster, nhm_year_range, nhm_peff_out,
        mode='volume',
        predictor_dir=predictor_dir,
        irr_fraction_band=irr_fraction_band,
        raster_label='Peff',
    )

    # ── 4. Compute metrics ───────────────────────────────────────────────
    all_sources = {
        'ML_Peff': ml_peff,
        'ML_Peff_PCML': ml_peff_pcml,
        'NHM_Peff': nhm_peff,
    }
    _peff_display_name = {
        'ML_Peff': 'USDA_SCS_Peff',
        'ML_Peff_PCML': 'ML_Peff_PCML',
        'NHM_Peff': 'NHM_Peff',
    }
    src_yr_ranges = {
        'ML_Peff': ml_year_range,
        'ML_Peff_PCML': ml_pcml_year_range,
        'NHM_Peff': nhm_year_range,
    }

    def _intersect_yr_range(r_a, r_b):
        return (max(r_a[0], r_b[0]), min(r_a[1], r_b[1]))

    def _peff_mean_over_years(src_data, basins, yr_range):
        yearly = src_data.get('yearly', {})
        if not yearly:
            return src_data.get('mean', {})
        years_in = [y for y in yearly if yr_range[0] <= y <= yr_range[1]]
        if not years_in:
            return {b: 0.0 for b in basins}
        return {
            b: float(np.mean([yearly[y].get(b, 0.0) for y in years_in]))
            for b in basins
        }

    common_yr = (
        max(src_yr_ranges[k][0] for k in src_yr_ranges),
        min(src_yr_ranges[k][1] for k in src_yr_ranges),
    )
    logger.info(
        f'  Pairwise + Common year ranges: '
        f'ML_Peff={ml_year_range}, ML_Peff_PCML={ml_pcml_year_range}, '
        f'NHM_Peff={nhm_year_range}, Common={common_yr}'
    )

    all_metrics = []
    pairs = [
        ('ML_Peff', 'NHM_Peff'),
        ('ML_Peff_PCML', 'NHM_Peff'),
        ('ML_Peff', 'ML_Peff_PCML'),
    ]
    for label_a, label_b in pairs:
        yr_range = _intersect_yr_range(
            src_yr_ranges[label_a], src_yr_ranges[label_b],
        )
        data_a = _peff_mean_over_years(
            all_sources[label_a], basin_names, yr_range,
        )
        data_b = _peff_mean_over_years(
            all_sources[label_b], basin_names, yr_range,
        )
        m = _compute_metrics(
            basin_names, data_a, data_b,
            _peff_display_name[label_a], _peff_display_name[label_b],
            basin_areas_m2=basin_areas_m2,
        )
        m['Category'] = 'Effective_Precipitation'
        m['Year_Range'] = f'{yr_range[0]}-{yr_range[1]}'
        all_metrics.append(m)
        logger.info(
            f'  {m["Pair"]} ({m["Year_Range"]}): '
            f'RMSD={m["RMSD_AF"]:.2f} AF '
            f'({m["RMSD_m3"]:.2f} m³), '
            f'MAD={m["MAD_AF"]:.2f} AF, PctDiff={m["Pct_Diff"]:.2f}%'
        )

    # Common 3-way intersection
    if common_yr[0] <= common_yr[1]:
        common_means = {
            k: _peff_mean_over_years(all_sources[k], basin_names, common_yr)
            for k in all_sources
        }
        for label_a, label_b in pairs:
            m = _compute_metrics(
                basin_names, common_means[label_a], common_means[label_b],
                _peff_display_name[label_a], _peff_display_name[label_b],
                basin_areas_m2=basin_areas_m2,
            )
            m['Category'] = 'Effective_Precipitation'
            m['Year_Range'] = f'Common_{common_yr[0]}-{common_yr[1]}'
            all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, 'peff_intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'Peff metrics saved to {metrics_csv}')

    # ── 5. Per-basin comparison table ────────────────────────────────────
    # Native columns use each source's own year range; Common columns
    # use the 3-way intersection year range so sources are directly
    # comparable on the same axis.
    rows = []
    common_means_for_csv = (
        common_means if common_yr[0] <= common_yr[1]
        else {k: {b: 0.0 for b in basin_names} for k in all_sources}
    )
    for basin in basin_names:
        area = basin_areas_m2.get(basin, 1.0)
        row = {'Basin': basin}
        for src_key in ('ML_Peff', 'ML_Peff_PCML', 'NHM_Peff'):
            display = _peff_display_name[src_key]
            af_val = all_sources[src_key]['mean'].get(basin, 0.0)
            row[f'{display}_mm'] = round(af_val * af_to_m3 / area * M_TO_MM, 4)
            row[f'{display}_AF'] = round(af_val, 2)
            af_common = common_means_for_csv[src_key].get(basin, 0.0)
            row[f'{display}_Common_AF'] = round(af_common, 2)
        rows.append(row)
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'peff_per_basin.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(
        f'Per-basin Peff saved to {basin_csv} '
        f'(native + Common {common_yr[0]}-{common_yr[1]})'
    )

    # ── 6. Time series CSV ───────────────────────────────────────────────
    ts_rows = []
    for src_key, src_data in all_sources.items():
        yearly = src_data.get('yearly', {})
        for year in sorted(yearly.keys()):
            for basin in basin_names:
                af_val = yearly[year].get(basin, 0.0)
                area = basin_areas_m2.get(basin, 1.0)
                ts_rows.append({
                    'Source': _peff_display_name.get(src_key, src_key),
                    'Year': year,
                    'Basin': basin,
                    'Volume_mm': round(
                        af_val * af_to_m3 / area * M_TO_MM, 4,
                    ),
                    'Volume_AF': round(af_val, 2),
                })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'peff_time_series.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'Peff time series saved to {ts_csv}')

    # ── 7. Time series plots ─────────────────────────────────────────────
    _peff_colors = {
        'ML_Peff': '#2C3E50', 'ML_Peff_PCML': '#8E44AD', 'NHM_Peff': '#27AE60',
    }
    _peff_markers = {'ML_Peff': 'o', 'ML_Peff_PCML': 'D', 'NHM_Peff': 's'}
    peff_ts_sources = {k: {'Peff': v} for k, v in all_sources.items()}
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    plot_intercomp_time_series(
        peff_ts_sources, categories=['Peff'],
        basin_names=basin_names, basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_peff_colors, markers=_peff_markers,
        labels={
            'ML_Peff': 'USDA-SCS Peff',
            'ML_Peff_PCML': 'ML Peff PCML',
            'NHM_Peff': 'NHM Peff',
        },
        title_prefix='Effective Precipitation — ', file_prefix='TS_Peff',
    )

    # ── 8. Scatter plots ─────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    _peff_display = {
        'ML_Peff': 'USDA-SCS Peff',
        'ML_Peff_PCML': 'ML Peff PCML',
        'NHM_Peff': 'NHM Peff',
    }
    source_keys = list(all_sources.keys())
    peff_scatter_pairs = [
        (_peff_display.get(source_keys[i], source_keys[i]),
         _peff_display.get(source_keys[j], source_keys[j]),
         all_sources[source_keys[i]]['mean'], all_sources[source_keys[j]]['mean'])
        for i in range(len(source_keys))
        for j in range(i + 1, len(source_keys))
    ]
    plot_intercomp_scatter(
        peff_scatter_pairs, basin_names, basin_areas_m2, scatter_dir,
        title='Effective Precipitation — Per-Basin Scatter',
        filename='Scatter_Peff.png',
    )

    # ── 8b. Basin-aggregated Δ Volume choropleth (3 panels) ───────────
    # Mirrors the NHM-withdrawal / PS basin-diff multi-panel pattern:
    # one figure with three panels (USDA-SCS−NHM, ML PCML−NHM,
    # USDA-SCS−ML PCML).  Shared colorbar — all three panels are
    # in the same Peff units (10⁶ m³) so magnitudes are comparable.
    basin_diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
    makedirs(basin_diff_dir)
    peff_basin_pairs = [
        ('USDA-SCS Peff', 'NHM Peff', 'ML_Peff', 'NHM_Peff'),
        ('ML Peff PCML', 'NHM Peff', 'ML_Peff_PCML', 'NHM_Peff'),
        ('USDA-SCS Peff', 'ML Peff PCML', 'ML_Peff', 'ML_Peff_PCML'),
    ]
    panels_peff = []
    for label_a, label_b, key_a, key_b in peff_basin_pairs:
        a_mean = all_sources[key_a]['mean']
        b_mean = all_sources[key_b]['mean']
        if not (a_mean and b_mean):
            continue
        panels_peff.append({
            'basin_a_vols': a_mean,
            'basin_b_vols': b_mean,
            'panel_title': f'{label_a} \u2212 {label_b}',
            'label_a': label_a,
            'label_b': label_b,
        })
    if panels_peff:
        _plot_basin_diff_panels(
            panels=panels_peff,
            basin_gdf=basin_reproj,
            basin_col=basin_col,
            title=(
                'Effective Precipitation \u2014 Basin-Level Volume Diff'
            ),
            out_path=os.path.join(
                basin_diff_dir, 'Spatial_Diff_Basin_Peff.png',
            ),
            shared_colorbar=True,
        )
        logger.info(
            f'Basin-level Peff Δ volume figure saved to {basin_diff_dir}'
        )

    # ── HUC12-level Peff comparison ─────────────────────────────────────
    logger.info('--- HUC12-level Peff comparison ---')
    huc12_dir = os.path.join(output_dir, 'HUC12_Comparison/')
    makedirs(huc12_dir)

    huc_gdf = gpd.read_file(huc12_geojson)
    huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    with rio.open(ref_raster) as _src:
        pixel_area_m2 = abs(_src.transform.a * _src.transform.e)
        huc_reproj = huc_gdf.to_crs(_src.crs)

    az_huc12_ids = sorted(huc_reproj['huc12'].unique())
    huc_areas = {
        str(row['huc12']): row.geometry.area
        for _, row in huc_reproj.iterrows()
    }

    # NHM Peff per HUC12 (annual CSV, Mgal/d → AF)
    nhm_df_raw = pd.read_csv(nhm_peff_csv, dtype={'Year': int})
    az_cols = [c for c in nhm_df_raw.columns
               if c != 'Year' and c in set(az_huc12_ids)]
    nhm_huc_mean: dict[str, float] = {}
    if az_cols:
        nhm_sub = nhm_df_raw[['Year'] + az_cols].copy()
        nhm_sub = nhm_sub[
            (nhm_sub.Year >= nhm_year_range[0])
            & (nhm_sub.Year <= nhm_year_range[1])
        ]
        for col in az_cols:
            nhm_sub[col] = pd.to_numeric(nhm_sub[col], errors='coerce')
            nhm_sub.loc[nhm_sub[col].isin(NHM_SENTINEL), col] = 0.0
            nhm_sub[col] = nhm_sub[col].fillna(0.0)
        days_per_year = 365.25
        for h in az_cols:
            vals = nhm_sub[h].values * days_per_year * MGAL_TO_M3 * M3_TO_AF
            nhm_huc_mean[h] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    # ML Peff (USDA-SCS) per HUC12 via zonal stats
    ml_peff_raster_dir = os.path.join(output_dir, 'ML_Peff_Rasters/')
    ml_huc_accum: dict[str, list[float]] = {h: [] for h in az_huc12_ids}
    common_start = max(ml_year_range[0], nhm_year_range[0])
    common_end = min(ml_year_range[1], nhm_year_range[1])
    for year in range(common_start, common_end + 1):
        rpath = os.path.join(ml_peff_raster_dir, f'ML_Peff_{year}_mm.tif')
        if not os.path.isfile(rpath):
            continue
        yr_stats = _compute_huc12_zonal_stats(
            rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
        )
        for h in az_huc12_ids:
            s = yr_stats.get(h)
            if s:
                ml_huc_accum[h].append(s['volume_AF'])
    ml_huc_mean = {
        h: float(np.mean(v)) if v else 0.0
        for h, v in ml_huc_accum.items()
    }

    # ML Peff PCML per HUC12 via zonal stats
    ml_pcml_raster_dir = os.path.join(output_dir, 'ML_Peff_PCML_Rasters/')
    pcml_huc_accum: dict[str, list[float]] = {h: [] for h in az_huc12_ids}
    pcml_common_end = min(ml_pcml_year_range[1], nhm_year_range[1])
    for year in range(common_start, pcml_common_end + 1):
        rpath = os.path.join(ml_pcml_raster_dir, f'ML_Peff_PCML_{year}_mm.tif')
        if not os.path.isfile(rpath):
            continue
        yr_stats = _compute_huc12_zonal_stats(
            rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
        )
        for h in az_huc12_ids:
            s = yr_stats.get(h)
            if s:
                pcml_huc_accum[h].append(s['volume_AF'])
    pcml_huc_mean = {
        h: float(np.mean(v)) if v else 0.0
        for h, v in pcml_huc_accum.items()
    }

    common_hucs = [h for h in az_huc12_ids
                   if h in nhm_huc_mean and h in ml_huc_mean]
    _peff_huc_display = {
        'ML_Peff': 'USDA_SCS_Peff',
        'ML_Peff_PCML': 'ML_Peff_PCML',
        'NHM_Peff': 'NHM_Peff',
    }
    if common_hucs:
        huc12_peff_metrics = []
        peff_huc_dicts = {
            'ML_Peff': ml_huc_mean,
            'ML_Peff_PCML': pcml_huc_mean,
            'NHM_Peff': nhm_huc_mean,
        }
        peff_huc_pairs = [
            ('ML_Peff', 'NHM_Peff'),
            ('ML_Peff_PCML', 'NHM_Peff'),
            ('ML_Peff', 'ML_Peff_PCML'),
        ]
        for key_a, key_b in peff_huc_pairs:
            m = _compute_metrics(
                common_hucs,
                peff_huc_dicts[key_a], peff_huc_dicts[key_b],
                _peff_huc_display[key_a], _peff_huc_display[key_b],
                basin_areas_m2=huc_areas,
            )
            m['Category'] = 'Effective_Precipitation'
            m['Level'] = 'HUC12'
            huc12_peff_metrics.append(m)
            logger.info(
                f'    {_peff_huc_display[key_a]} vs {_peff_huc_display[key_b]} '
                f'(HUC12): RMSD={m["RMSD_AF"]:.1f} AF, '
                f'PctDiff={m["Pct_Diff"]:.1f}%'
            )
        pd.DataFrame(huc12_peff_metrics).to_csv(
            os.path.join(huc12_dir, 'huc12_peff_metrics.csv'), index=False,
        )

        # HUC12-level scatter (3 pairs)
        huc12_scatter_dir = os.path.join(huc12_dir, 'Scatter/')
        plot_intercomp_scatter(
            [
                (_peff_huc_display['ML_Peff'], _peff_huc_display['NHM_Peff'],
                 ml_huc_mean, nhm_huc_mean),
                (_peff_huc_display['ML_Peff_PCML'], _peff_huc_display['NHM_Peff'],
                 pcml_huc_mean, nhm_huc_mean),
                (_peff_huc_display['ML_Peff'], _peff_huc_display['ML_Peff_PCML'],
                 ml_huc_mean, pcml_huc_mean),
            ],
            common_hucs, huc_areas, huc12_scatter_dir,
            title='Effective Precipitation — HUC12-Level Scatter',
            filename='Scatter_HUC12_Peff.png',
        )

        # HUC12-level spatial diff choropleth (NHM pairs only)
        _af_to_m3_local = 1.0 / M3_TO_AF
        _mm_to_ft = 1.0 / 304.8
        _m3_to_af_local = M3_TO_AF
        huc12_diff_dir = os.path.join(huc12_dir, 'Spatial_Diff/')
        makedirs(huc12_diff_dir)
        b_reproj = (
            basin_gdf.to_crs(huc_reproj.crs)
            if basin_gdf.crs != huc_reproj.crs else basin_gdf
        )
        b_name = basin_col if basin_col in b_reproj.columns else b_reproj.columns[0]
        ama_ina_names = get_ama_ina_basin_names()

        nhm_peff_pairs = [
            ('USDA-SCS Peff', 'NHM Peff', ml_huc_mean, nhm_huc_mean),
            ('ML Peff PCML', 'NHM Peff', pcml_huc_mean, nhm_huc_mean),
            ('USDA-SCS Peff', 'ML Peff PCML', ml_huc_mean, pcml_huc_mean),
        ]

        # Volume-only diff for Peff HUC12 (depth-mode dropped: per-HUC12
        # depth differences are dominated by polygon size variation
        # rather than the precipitation signal we want to compare).
        for unit_mode, unit_label, sec_label, sec_factor, scale_fn, tick_div in [
            ('volume', r'$\Delta$ Volume ($\times$10$^{6}$ m$^3$)', '\u0394 Volume (AF)',
             _m3_to_af_local,
             lambda af, area: af * _af_to_m3_local, 1e6),
        ]:
            fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
            title_u = 'Volume'
            fig.suptitle(
                f'Effective Precipitation \u2014 HUC12-Level {title_u} Difference',
                fontsize=14, fontweight='bold',
            )
            global_vmax = 1e-6
            for _, _, d_a, d_b in nhm_peff_pairs:
                diffs = []
                for h in common_hucs:
                    area = huc_areas.get(h, 1.0)
                    va = scale_fn(d_a.get(h, 0.0), area)
                    vb = scale_fn(d_b.get(h, 0.0), area)
                    if va != 0 or vb != 0:
                        diffs.append(va - vb)
                if diffs:
                    d_arr = np.array(diffs)
                    global_vmax = max(
                        global_vmax,
                        abs(np.nanpercentile(d_arr, 2)),
                        abs(np.nanpercentile(d_arr, 98)),
                    )
            last_im = None
            for col_i, (name_a, name_b, d_a, d_b) in enumerate(nhm_peff_pairs):
                ax = axes[col_i]
                ax.set_facecolor('#D5D5D5')
                diff_vals = []
                for h in common_hucs:
                    area = huc_areas.get(h, 1.0)
                    diff_vals.append(
                        scale_fn(d_a.get(h, 0.0), area)
                        - scale_fn(d_b.get(h, 0.0), area)
                    )
                plot_gdf = huc_reproj[huc_reproj['huc12'].isin(common_hucs)].copy()
                plot_gdf = plot_gdf.set_index('huc12').loc[common_hucs]
                plot_gdf['diff'] = diff_vals
                plot_gdf.loc[plot_gdf['diff'].abs() < 1e-10, 'diff'] = np.nan
                plot_gdf.plot(
                    ax=ax, column='diff', cmap='RdBu_r',
                    vmin=-global_vmax, vmax=global_vmax,
                    edgecolor='#AAAAAA', linewidth=0.3,
                    legend=False, missing_kwds={'color': '#EEEEEE'},
                )
                _overlay_boundaries(ax, b_reproj, ama_ina_names, b_name,
                                    label_fontsize=5.0, label_all=True)
                ax.set_title(f'{name_a} \u2212 {name_b}', fontweight='bold')

            import matplotlib.ticker as mticker
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(-global_vmax, global_vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=list(axes), shrink=0.5, pad=0.06,
                                orientation='horizontal', aspect=40, extend='both')
            if tick_div:
                cbar.formatter = mticker.FuncFormatter(lambda x, _: f'{x/tick_div:g}')
                cbar.update_ticks()
            cbar.set_label(unit_label, fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            secax = cbar.ax.secondary_xaxis(
                'top', functions=(lambda x: x*sec_factor, lambda x: x/sec_factor))
            secax.set_xlabel(sec_label, fontsize=10, fontweight='bold')
            secax.tick_params(labelsize=10)
            add_ama_ina_legend(axes[0])

            suffix = '' if unit_mode == 'depth' else '_Volume'
            fig.savefig(
                os.path.join(huc12_diff_dir, f'Spatial_Diff_HUC12_Peff{suffix}.png'),
                dpi=600, bbox_inches='tight',
            )
            plt.close(fig)
        logger.info(f'  HUC12-level Peff diff maps saved to {huc12_diff_dir}')

        # ── Pixel-level Peff diff (3 pairs side by side) ──────────────
        # Mirrors the HUC12 figure structure (3 panels = 3 pairs) but
        # at native raster resolution.  Uses Δ Depth (mm) since every
        # pixel has identical area and depth is the most direct
        # comparison unit at pixel scale.
        pixel_diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
        makedirs(pixel_diff_dir)
        peff_mean_rasters = {
            'ML_Peff': os.path.join(
                ml_peff_out, 'ML_mean_annual_Peff_mm.tif',
            ),
            'ML_Peff_PCML': os.path.join(
                ml_pcml_out, 'ML_mean_annual_Peff_PCML_mm.tif',
            ),
            'NHM_Peff': os.path.join(
                nhm_peff_out, 'NHM_mean_annual_Peff_mm.tif',
            ),
        }
        peff_pixel_arrays: dict[str, np.ndarray] = {}
        pixel_extent_pf = None
        for src_key, rpath in peff_mean_rasters.items():
            if os.path.isfile(rpath):
                with rio.open(rpath) as src:
                    arr = src.read(1).astype(np.float64)
                    if pixel_extent_pf is None:
                        pixel_extent_pf = [
                            src.bounds.left, src.bounds.right,
                            src.bounds.bottom, src.bounds.top,
                        ]
                arr[np.isnan(arr)] = 0.0
                peff_pixel_arrays[src_key] = arr

        peff_pixel_pairs = [
            ('USDA-SCS Peff', 'NHM Peff', 'ML_Peff', 'NHM_Peff'),
            ('ML Peff PCML', 'NHM Peff', 'ML_Peff_PCML', 'NHM_Peff'),
            ('USDA-SCS Peff', 'ML Peff PCML', 'ML_Peff', 'ML_Peff_PCML'),
        ]
        # Compute shared vmax across the 3 pair panels
        global_vmax_pix = 1e-6
        for _, _, key_a, key_b in peff_pixel_pairs:
            if key_a in peff_pixel_arrays and key_b in peff_pixel_arrays:
                d = peff_pixel_arrays[key_a] - peff_pixel_arrays[key_b]
                m_zero = (
                    (peff_pixel_arrays[key_a] == 0)
                    & (peff_pixel_arrays[key_b] == 0)
                )
                d_valid = d[~m_zero]
                if d_valid.size:
                    global_vmax_pix = max(
                        global_vmax_pix,
                        abs(np.nanpercentile(d_valid, 2)),
                        abs(np.nanpercentile(d_valid, 98)),
                    )

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        fig, axes = plt.subplots(
            1, 3, figsize=(20, 7), constrained_layout=True,
        )
        fig.suptitle(
            'Effective Precipitation \u2014 Pixel-Level Depth '
            'Difference',
            fontsize=14, fontweight='bold',
        )
        for col_i, (name_a, name_b, key_a, key_b) in enumerate(
            peff_pixel_pairs,
        ):
            ax = axes[col_i]
            ax.set_facecolor('#D5D5D5')
            if (
                key_a in peff_pixel_arrays
                and key_b in peff_pixel_arrays
                and pixel_extent_pf is not None
            ):
                diff = (
                    peff_pixel_arrays[key_a] - peff_pixel_arrays[key_b]
                )
                mask = (
                    (peff_pixel_arrays[key_a] == 0)
                    & (peff_pixel_arrays[key_b] == 0)
                )
                diff_masked = np.ma.masked_where(mask, diff)
                ax.imshow(
                    diff_masked, extent=pixel_extent_pf,
                    origin='upper', cmap='RdBu_r',
                    vmin=-global_vmax_pix, vmax=global_vmax_pix,
                    interpolation='nearest',
                )
                _overlay_boundaries(
                    ax, b_reproj, ama_ina_names, b_name,
                    label_fontsize=5.0, label_all=False,
                )
            else:
                ax.text(
                    0.5, 0.5,
                    f'Pixel rasters unavailable\n({key_a} or {key_b})',
                    ha='center', va='center', transform=ax.transAxes,
                )
                ax.axis('off')
            ax.set_title(f'{name_a} \u2212 {name_b}', fontweight='bold')

        sm = ScalarMappable(
            cmap='RdBu_r',
            norm=Normalize(-global_vmax_pix, global_vmax_pix),
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=list(axes), shrink=0.5, pad=0.06,
            orientation='horizontal', aspect=40, extend='both',
        )
        cbar.set_label(
            '\u0394 Depth (mm)', fontsize=10, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=10)
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(
                lambda x: x * _mm_to_ft,
                lambda x: x / _mm_to_ft,
            ),
        )
        secax.set_xlabel(
            '\u0394 Depth (ft)', fontsize=10, fontweight='bold',
        )
        secax.tick_params(labelsize=10)
        add_ama_ina_legend(axes[0])
        fig.savefig(
            os.path.join(pixel_diff_dir, 'Spatial_Diff_Pixel_Peff.png'),
            dpi=600, bbox_inches='tight',
        )
        plt.close(fig)
        logger.info(
            f'  Pixel-level Peff diff figure saved to {pixel_diff_dir}'
        )

        # ── HUC12-level Peff temporal diagnostics ──
        logger.info('  Computing HUC12-level Peff temporal diagnostics...')
        # NHM yearly {year: {huc12: AF}}
        nhm_yearly_huc_peff: dict[int, dict[str, float]] = {}
        days_per_year = 365.25
        for year in range(nhm_year_range[0], nhm_year_range[1] + 1):
            yr_row = nhm_sub[nhm_sub.Year == year]
            if yr_row.empty:
                continue
            nhm_yearly_huc_peff[year] = {
                h: float(yr_row[h].values[0]) * days_per_year * MGAL_TO_M3 * M3_TO_AF
                if h in yr_row.columns else 0.0
                for h in az_cols
            }
        # ML Peff (SCS) yearly {year: {huc12: AF}}
        ml_yearly_huc_peff: dict[int, dict[str, float]] = {}
        for year in range(common_start, common_end + 1):
            rpath = os.path.join(ml_peff_raster_dir, f'ML_Peff_{year}_mm.tif')
            if not os.path.isfile(rpath):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
            )
            ml_yearly_huc_peff[year] = {
                h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                for h in az_huc12_ids
            }
        # ML Peff PCML yearly {year: {huc12: AF}}
        pcml_yearly_huc_peff: dict[int, dict[str, float]] = {}
        for year in range(common_start, pcml_common_end + 1):
            rpath = os.path.join(ml_pcml_raster_dir, f'ML_Peff_PCML_{year}_mm.tif')
            if not os.path.isfile(rpath):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                rpath, huc_reproj, pixel_area_m2, depth_unit='mm',
            )
            pcml_yearly_huc_peff[year] = {
                h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                for h in az_huc12_ids
            }
        _huc12_temporal_diagnostics(
            huc12_yearly_sources={
                'USDA_SCS_Peff': ml_yearly_huc_peff,
                'ML_Peff_PCML': pcml_yearly_huc_peff,
                'NHM_Peff': nhm_yearly_huc_peff,
            },
            pairs=[
                ('USDA_SCS_Peff', 'NHM_Peff'),
                ('ML_Peff_PCML', 'NHM_Peff'),
                ('USDA_SCS_Peff', 'ML_Peff_PCML'),
            ],
            huc12_ids=common_hucs,
            category='Effective_Precipitation',
            output_dir=huc12_dir,
            huc_areas=huc_areas,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('Peff Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
    logger.info(f'\nUSDA-SCS Peff year range: {ml_year_range}')
    logger.info(f'ML Peff PCML year range: {ml_pcml_year_range}')
    logger.info(f'NHM Peff year range: {nhm_year_range}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# CAP/SRP Surface Water Validation
# ═════════════════════════════════════════════════════════════════════════════

# Mapping from CAP Excel AMA names to ADWR GW basin names.
# Parker is intentionally excluded — CAP delivers ~4 AF/yr to Parker
# (essentially zero), and the basin's actual SW supply is dominated
# by CRIT senior Priority-1 mainstem rights (~720 kAF/yr) which the
# CAP delivery file does not track.  Validating ML Total_SW against
# CAP-only data at Parker compares apples-to-CRIT-oranges and produces
# a misleading "ML grossly overestimates" signal that is really just
# the model correctly capturing CRIT deliveries the CAP file omits.
_CAP_AMA_TO_BASIN = {
    'Phoenix AMA':     'PHOENIX AMA',
    'Tucson AMA':      'TUCSON AMA',
    'Pinal AMA':       'PINAL AMA',
    'Harquahala INA':  'HARQUAHALA INA',
    # Ranegras Plain excluded from validation: direct-use CAP
    # delivery is incidental pipeline pass-through (~1 kAF/yr peak,
    # essentially zero post-2015), and the basin is documented to
    # stay ~100 % GW (handled by the 0.95 floor in partitionops, not
    # the CAP cap mechanism).  Including it added a near-origin
    # outlier point to the per-basin scatter without contributing
    # meaningful validation signal.
}


# CAP customer → sector classification ('mi' = municipal & industrial,
# 'ag' = irrigation/agricultural, 'tribal_ag' = tribal communities
# treated as ag for SW-validation purposes).  Used to split the CAP
# observation into M&I and Ag components so the model's NonIrr_SW can
# be compared against CAP M&I (apples-to-apples) and Irr_SW against
# CAP Ag.  Hardcoded based on the top customers (covering >95 % of
# total CAP delivery); unknown customers fall through to 'mi' as a
# conservative default (most small CAP customers are M&I providers).
_CAP_CUSTOMER_AG_PATTERNS = (
    'MSIDD', 'CAIDD', 'HVID', 'NMIDD', 'HIDD', 'QCID',
    'Maricopa Stanfield', 'Maricopa-Stanfield',
    'Central Arizona Irrigation',
    'Harquahala Valley',
    'New Magma', 'Hohokam', 'Queen Creek',
    'Cortaro-Marana', 'Cortaro Marana',
    'Roosevelt', 'San Tan',
    'Irrigation', 'IDD', 'ID ',  # ID followed by space (district)
)
_CAP_CUSTOMER_TRIBAL_AG_PATTERNS = (
    'Ak-Chin', 'Ak Chin',
    'GRIC', 'Gila River Indian',
    'San Carlos Apache', 'San Carlos',
    'Tohono', 'Pascua', 'Yavapai-Prescott',
    'Indian Community', 'Tribe',
)
_CAP_CUSTOMER_MI_PATTERNS = (
    'Phoenix', 'Tucson', 'Mesa', 'Scottsdale', 'Gilbert',
    'Chandler', 'Glendale', 'Peoria', 'Tempe', 'Goodyear',
    'Avondale', 'Buckeye', 'Surprise', 'Litchfield',
    'Cave Creek', 'Carefree', 'Fountain Hills', 'Sun City',
    'Marana', 'Oro Valley', 'Sahuarita',
    'AWBA', 'CAGRD', 'CAWCD',
    'Water Co', 'Water Company', 'Water Utility',
    'City of', 'Town of', 'Metro Water',
)


def _classify_cap_customer(customer: str) -> str:
    """Return 'ag' or 'mi' for a CAP customer name."""
    if not isinstance(customer, str) or not customer:
        return 'mi'
    s = customer.strip()
    for pat in _CAP_CUSTOMER_TRIBAL_AG_PATTERNS:
        if pat in s:
            return 'ag'
    for pat in _CAP_CUSTOMER_AG_PATTERNS:
        if pat in s:
            return 'ag'
    for pat in _CAP_CUSTOMER_MI_PATTERNS:
        if pat in s:
            return 'mi'
    return 'mi'  # default (most unknown small customers are M&I)


def _compute_ml_cap_approximation(
        ml_basin_yearly: dict[str, dict[int, float]],
        cap_xlsx: str,
        baseline_period: tuple[int, int] = (2000, 2009),
) -> dict[str, dict[int, float]]:
    """Approximate the CAP contribution to model Total_SW per basin per year.

    Returns ``{basin: {year: AF}}`` where each entry is::

        approx_CAP[year] = ML_Total_SW[year] × delivery_ratio[year]
        delivery_ratio[year] = direct_CAP_delivery[year] / baseline_period_mean

    Multiplies the model's per-basin Total_SW by the basin's CAP
    delivery ratio for that year.  Pre-arrival years yield 0 (no CAP
    delivery record → ratio = 0).  Designed as a coarse visualisation
    only — at SRP-dominant Phoenix the formula over-attributes some
    SRP-derived SW to "CAP" (since the model can't separate SRP from
    CAP cleanly), but it tracks the observation curve shape at all
    four CAP basins.

    A pre-CAP-baseline subtraction was tried earlier but failed at
    Phoenix and Pinal because the model's Total_SW DROPS with CAP
    arrival there (the dynamic NonIrr cap shifts SW → GW at CAP-served
    pixels), making ``max(0, current − baseline_1985)`` clip to zero
    at all years.  The simpler ``Total_SW × ratio`` formula avoids
    that pitfall.

    Visualisation only — not used by the partition itself.  Returns
    ``{}`` if the CAP Excel cannot be loaded.
    """
    try:
        from hydrolibs import partitionops as _partops
        lookup = _partops.load_cap_basin_delivery(
            cap_xlsx, baseline_period=baseline_period,
        )
    except Exception:
        return {}
    if not lookup:
        return {}
    out: dict[str, dict[int, float]] = {}
    for basin, ml_yearly in ml_basin_yearly.items():
        if basin not in lookup or not ml_yearly:
            continue
        basin_data = lookup[basin]
        baseline_af = basin_data.get('baseline_af', 0.0)
        if baseline_af <= 0:
            continue
        delivery = basin_data.get('yearly_af', {})
        approx = {}
        for yr, ml_total in ml_yearly.items():
            d = delivery.get(yr)
            if d is None:
                approx[yr] = 0.0  # no CAP delivery → no CAP contribution
                continue
            ratio = max(0.0, min(d / baseline_af, 1.0))
            approx[yr] = float(ml_total) * ratio
        if approx:
            out[basin] = approx
    return out


def load_cap_srp_annual_sw(
    cap_xlsx: str,
    srp_xlsx: str | None = None,
    include_spill_water: bool = False,
    include_recharge: bool = True,
    sector: str | None = None,
) -> dict[str, dict[int, float]]:
    """Load CAP (and optionally SRP) delivery data and return annual
    total surface-water deliveries (AF) per basin.

    CAP: by default includes ALL CAP deliveries (direct use + recharge
    facility deliveries: USF, GSF, ASR) for the "full CAP utilization
    footprint" view at each basin.  Recharge volumes are eventually
    recovered as GW (Tucson Water CAVSARP/SAVSARP, Phoenix CAGRD,
    Pinal in-lieu credits), so they are part of the basin's CAP
    consumptive use even though the partition's Total_SW excludes
    recovered recharge by construction.  Set ``include_recharge=False``
    to restrict to direct-use deliveries only (more apples-to-apples
    with model Total_SW but understates CAP footprint at recharge-
    heavy basins).
    Rows with ``AMA == 'Multiple'`` or ``NaN`` are excluded because they
    cannot be assigned to a single basin (25 records / ~15,600 AF total;
    16 NaN-AMA records / ~86,300 AF total).

    SRP: only loaded when ``srp_xlsx`` is provided (and not None).  If
    omitted, validation uses CAP-only data — recommended because the
    SRP service area boundary is not publicly mapped, so attributing
    SRP deliveries to a single GW basin (currently Phoenix AMA) is
    ambiguous.  When loaded, keeps rows where
    ``Parent Water Type == 'SURFACE WATER'``; when *include_spill_water*
    is True, ``SPILL WATER`` records are also included as a sensitivity
    test (spill water ranges from ~19 AF/yr at 2016 to ~366,000 AF/yr
    at 1993 in Phoenix AMA).

    Both datasets use calendar-year columns (CAP ``Year``; SRP
    ``Water Move Year``).

    Args:
        cap_xlsx (str): Path to CAP delivery Excel file.
        srp_xlsx (str or None): Path to SRP delivery Excel file.  If
            None (default), SRP is skipped — CAP-only validation.
        include_spill_water (bool): If True, include SRP ``SPILL WATER`` records in addition to
            ``SURFACE WATER``.  Default False (baseline).  Ignored when
            ``srp_xlsx`` is None.
        include_recharge (bool): If True (default), include recharge
            facility deliveries (USF, GSF, ASR) — the full CAP supply
            footprint per basin.  If False, restrict to direct-use
            deliveries only.
        sector (str or None): Filter CAP deliveries by customer sector.
            ``'mi'`` keeps only municipal & industrial customers
            (Phoenix, Tucson, AWBA, CAGRD, etc.) — apples-to-apples
            comparison with model NonIrr_SW.  ``'ag'`` keeps only
            irrigation/tribal-ag customers (MSIDD, CAIDD, GRIC, etc.)
            — comparison with model Irr_SW.  ``None`` (default) keeps
            all customers.  SRP is added unfiltered when ``srp_xlsx``
            is provided AND sector is not 'mi' (SRP deliveries to
            Phoenix AMA are predominantly ag-side).

    Returns:
        dict[str, dict[int, float]]: ``{basin_name: {year: delivery_AF}}``.
    """
    # ── CAP ──────────────────────────────────────────────────────────────
    cap_df = pd.read_excel(cap_xlsx)
    if not include_recharge:
        # Restrict to direct-use deliveries (null recharge facility).
        cap_df = cap_df[cap_df['Recharge Facility'].isna()].copy()
    else:
        cap_df = cap_df.copy()
    if sector in ('mi', 'ag'):
        cap_df['_sector'] = cap_df['Customer'].apply(_classify_cap_customer)
        cap_df = cap_df[cap_df['_sector'] == sector].copy()
    # Log excluded volume from unmappable AMA records
    mappable_mask = cap_df['AMA'].isin(_CAP_AMA_TO_BASIN)
    excluded = cap_df[~mappable_mask]
    if not excluded.empty:
        n_mult = int((excluded['AMA'] == 'Multiple').sum())
        n_nan = int(excluded['AMA'].isna().sum())
        n_other = len(excluded) - n_mult - n_nan
        vol_mult = excluded.loc[
            excluded['AMA'] == 'Multiple', 'Delivery AF'
        ].sum()
        vol_nan = excluded.loc[
            excluded['AMA'].isna(), 'Delivery AF'
        ].sum()
        vol_total = excluded['Delivery AF'].sum()
        logger.warning(
            'CAP validation: excluded %d records (%.0f AF) with '
            'unmappable AMA — %d "Multiple" (%.0f AF), %d NaN '
            '(%.0f AF)%s. These cannot be assigned to a single '
            'basin and represent a systematic undercount in '
            'per-basin comparisons.',
            len(excluded), vol_total,
            n_mult, vol_mult,
            n_nan, vol_nan,
            f', {n_other} other ({vol_total - vol_mult - vol_nan:.0f} AF)'
            if n_other else '',
        )
    cap_df = cap_df[mappable_mask].copy()
    cap_df['Basin'] = cap_df['AMA'].map(_CAP_AMA_TO_BASIN)
    cap_annual = (
        cap_df.groupby(['Basin', 'Year'])['Delivery AF']
        .sum()
        .reset_index()
        .rename(columns={'Delivery AF': 'CAP_AF'})
    )

    # ── Build CAP-only basin/year dict ───────────────────────────────────
    basin_year = {}
    for _, row in cap_annual.iterrows():
        basin_year.setdefault(row['Basin'], {})[int(row['Year'])] = float(row['CAP_AF'])

    # ── SRP (optional) ───────────────────────────────────────────────────
    # SRP service area boundary is not publicly mapped, so attributing
    # SRP deliveries to a single AZ GW basin (Phoenix AMA) is ambiguous.
    # Skip unless caller explicitly provides srp_xlsx.
    if srp_xlsx is None:
        logger.info(
            'CAP/SRP loader: srp_xlsx not provided — using CAP-only data '
            '(SRP service area is not unambiguously mappable to AZ basins).'
        )
        return basin_year
    # Skip SRP entirely when filtering for M&I — SRP deliveries are
    # predominantly ag-side (irrigation districts within Salt River
    # Valley); adding them would inflate the M&I observation.
    if sector == 'mi':
        return basin_year

    srp_df = pd.read_excel(srp_xlsx)
    srp_types = ['SURFACE WATER']
    if include_spill_water:
        srp_types.append('SPILL WATER')
    srp_df = srp_df[srp_df['Parent Water Type'].isin(srp_types)].copy()
    srp_annual = (
        srp_df.groupby('Water Move Year')['SUM_WATER_QTY']
        .sum()
        .reset_index()
        .rename(columns={'Water Move Year': 'Year', 'SUM_WATER_QTY': 'SRP_AF'})
    )

    # Add SRP surface water to Phoenix AMA
    phx = basin_year.setdefault('PHOENIX AMA', {})
    for _, row in srp_annual.iterrows():
        year = int(row['Year'])
        phx[year] = phx.get(year, 0.0) + float(row['SRP_AF'])

    return basin_year


def _load_basin_sigma_yearly(
        sigma_raster_dir: str,
        cat_file_prefix: str,
        basin_reproj: gpd.GeoDataFrame,
        basin_col: str,
        year_range: tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Per-basin per-year σ volume (AF) from category σ rasters.

    Reads ``Sigma_Total_{cat_file_prefix}_mm_{year}.tif`` from
    *sigma_raster_dir* and aggregates per-pixel σ_mm to per-basin σ
    via spatial quadrature: ``σ_basin = sqrt(Σ σ_pixel²)``.  Treats
    per-pixel σ as approximately independent — matches the convention
    used by ``_load_az_sigma_total_for_category`` for AZ-wide rollups.

    Returns ``{basin_name: {year: sigma_AF}}``.  Returns an empty dict
    silently when the σ raster directory or files don't exist.
    """
    out: dict[str, dict[int, float]] = {}
    if not sigma_raster_dir or not os.path.isdir(sigma_raster_dir):
        return out
    start_yr, end_yr = year_range
    # Load basin geometries once
    basin_geoms = {
        row[basin_col]: row.geometry
        for _, row in basin_reproj.iterrows()
    }
    for year in range(start_yr, end_yr + 1):
        path = os.path.join(
            sigma_raster_dir,
            f'Sigma_Total_{cat_file_prefix}_mm_{year}.tif',
        )
        if not os.path.isfile(path):
            continue
        try:
            with rio.open(path) as src:
                pixel_area_m2_ = abs(src.transform.a * src.transform.e)
                mm_to_m3 = pixel_area_m2_ / 1000.0
                sig_arr = src.read(1).astype(np.float64)
                transform = src.transform
                shape = sig_arr.shape
            sig_arr = np.where(np.isfinite(sig_arr), sig_arr, 0.0)
            sig_vol_m3 = sig_arr * mm_to_m3
            for basin_name, geom in basin_geoms.items():
                from rasterio.features import geometry_mask
                mask = geometry_mask(
                    [geom], transform=transform, invert=True,
                    out_shape=shape,
                )
                # Spatial quadrature within the basin
                sigma_m3 = float(np.sqrt(np.sum(sig_vol_m3[mask] ** 2)))
                sigma_af = sigma_m3 * M3_TO_AF
                out.setdefault(basin_name, {})[year] = sigma_af
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Failed to load σ raster %s: %s', path, exc,
            )
            continue
    return out


def load_ml_total_sw_basin_volumes(
    total_sw_dir: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
    file_prefix: str = 'Total_SW',
) -> dict[str, dict[int, float]]:
    """Aggregate ML ``{file_prefix}_YYYY_mm.tif`` rasters to annual
    basin volumes (AF).

    Default ``file_prefix='Total_SW'`` reads ``Total_SW_YYYY_mm.tif``.
    Pass ``file_prefix='Irrigation_SW'`` or ``'Non_Irrigation_SW'`` to
    read the per-category SW depth rasters from the corresponding
    ``Irrigation_SW_Rasters/Depth_mm/`` or
    ``Non_Irrigation_SW_Rasters/Depth_mm/`` directory.

    Returns:
        dict[str, dict[int, float]]: ``{basin_name: {year: volume_AF}}``.
    """
    start_yr, end_yr = year_range
    ref_raster = None
    for yr in range(start_yr, end_yr + 1):
        candidate = os.path.join(total_sw_dir, f'{file_prefix}_{yr}_mm.tif')
        if os.path.isfile(candidate):
            ref_raster = candidate
            break
    if ref_raster is None:
        logger.warning(f'No {file_prefix} rasters found in {total_sw_dir}')
        return {}

    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    result = {}  # basin → {year → AF}
    for yr in range(start_yr, end_yr + 1):
        raster_path = os.path.join(total_sw_dir, f'{file_prefix}_{yr}_mm.tif')
        if not os.path.isfile(raster_path):
            continue
        yearly_vols = _raster_basin_volumes(
            raster_path, basin_reproj, basin_col,
            pixel_area_m2, depth_unit='mm',
        )
        for basin, vol_af in yearly_vols.items():
            result.setdefault(basin, {})[yr] = vol_af

    return result


def _compute_cap_srp_metrics(
    ml_basin_yearly: dict[str, dict[int, float]],
    obs_basin_yearly: dict[str, dict[int, float]],
    basin_areas_m2: dict[str, float],
) -> pd.DataFrame:
    """Compute per-basin and AZ-total statistics over the common year range
    between ML Total SW predictions and CAP+SRP observed deliveries.

    Returns a DataFrame with one row per basin (+ AZ Total) containing:
        RMSD, MAD, Percent Difference, Pearson R, mean ML, mean observed
        (reported in AF, m³, and mm).
    """
    af_to_m3 = 1.0 / M3_TO_AF
    rows = []

    obs_basins = sorted(obs_basin_yearly.keys())
    targets = obs_basins + ['AZ_Total']

    for basin in targets:
        if basin == 'AZ_Total':
            # Gather all years present in both ML and obs for any basin
            all_obs_years = set()
            all_ml_years = set()
            for b in obs_basins:
                all_obs_years.update(obs_basin_yearly.get(b, {}).keys())
                all_ml_years.update(ml_basin_yearly.get(b, {}).keys())
            common_years = sorted(all_obs_years & all_ml_years)
            if not common_years:
                continue
            ml_vals = np.array([
                sum(ml_basin_yearly.get(b, {}).get(yr, 0.0) for b in obs_basins)
                for yr in common_years
            ])
            obs_vals = np.array([
                sum(obs_basin_yearly.get(b, {}).get(yr, 0.0) for b in obs_basins)
                for yr in common_years
            ])
            area = sum(basin_areas_m2.get(b, 0.0) for b in obs_basins)
        else:
            ml_years = set(ml_basin_yearly.get(basin, {}).keys())
            obs_years = set(obs_basin_yearly.get(basin, {}).keys())
            common_years = sorted(ml_years & obs_years)
            if not common_years:
                continue
            ml_vals = np.array([ml_basin_yearly[basin][yr] for yr in common_years])
            obs_vals = np.array([obs_basin_yearly[basin][yr] for yr in common_years])
            area = basin_areas_m2.get(basin, 1.0)

        diff = ml_vals - obs_vals
        rmsd_af = float(np.sqrt(np.mean(diff ** 2)))
        mad_af = float(np.mean(np.abs(diff)))
        denom = (np.mean(ml_vals) + np.mean(obs_vals)) / 2.0
        pct_diff = float(mad_af / denom * 100) if denom > 0 else np.nan

        # Temporal R² / NSE (sklearn r2_score with observed as y_true and
        # predicted as y_pred, computed per basin across years). This is
        # distinct from the spatial R² shown on the scatter plot, which
        # computes r2_score across basins for a single mean value per
        # basin.
        if len(ml_vals) > 1 and np.std(obs_vals) > 0:
            from sklearn.metrics import r2_score as _r2
            r2 = float(_r2(obs_vals, ml_vals))
        else:
            r2 = np.nan

        # Normalized error metrics (same as the scatter plot annotations)
        from hydrolibs.mlops import (
            normalized_rmse, normalized_mae, normalized_mbe,
        )
        rmse_pct = normalized_rmse(obs_vals, ml_vals)
        mae_pct = normalized_mae(obs_vals, ml_vals)
        mbe_pct = normalized_mbe(obs_vals, ml_vals)

        ml_m3 = ml_vals * af_to_m3
        obs_m3 = obs_vals * af_to_m3
        diff_m3 = diff * af_to_m3

        mbe_af = float(np.mean(diff))
        mbe_m3 = float(np.mean(diff_m3))

        row = {
            'Basin': basin,
            'N_Years': len(common_years),
            'Year_Range': f'{common_years[0]}-{common_years[-1]}',
            'Temporal_R2_NSE': round(r2, 4),
            'RMSE_AF': round(rmsd_af, 2),
            'RMSE_m3': round(float(np.sqrt(np.mean(diff_m3 ** 2))), 2),
            'MAE_AF': round(mad_af, 2),
            'MAE_m3': round(float(np.mean(np.abs(diff_m3))), 2),
            'MBE_AF': round(mbe_af, 2),
            'MBE_m3': round(mbe_m3, 2),
            'RMSE_pct': round(rmse_pct, 2),
            'MAE_pct': round(mae_pct, 2),
            'MBE_pct': round(mbe_pct, 2),
            'Pct_Diff': round(pct_diff, 2),
            'Mean_ML_AF': round(float(np.mean(ml_vals)), 2),
            'Mean_Obs_AF': round(float(np.mean(obs_vals)), 2),
            'Mean_ML_m3': round(float(np.mean(ml_m3)), 2),
            'Mean_Obs_m3': round(float(np.mean(obs_m3)), 2),
        }
        if area > 0:
            ml_mm = ml_m3 / area * M_TO_MM
            obs_mm = obs_m3 / area * M_TO_MM
            diff_mm = diff * af_to_m3 / area * M_TO_MM
            row['RMSE_mm'] = round(float(np.sqrt(np.mean(diff_mm ** 2))), 4)
            row['MAE_mm'] = round(float(np.mean(np.abs(diff_mm))), 4)
            row['MBE_mm'] = round(float(np.mean(diff_mm)), 4)
            row['Mean_ML_mm'] = round(float(np.mean(ml_mm)), 4)
            row['Mean_Obs_mm'] = round(float(np.mean(obs_mm)), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def run_cap_srp_validation(
    cap_xlsx: str,
    srp_xlsx: str | None = None,
    total_sw_dir: str = '',
    basin_shp: str = '',
    basin_col: str = '',
    output_dir: str = '',
    year_range: tuple[int, int] = (1985, 2023),
) -> pd.DataFrame:
    """
    Validate ML Total_SW predictions against observed CAP (and optionally
    SRP) surface-water delivery records.

    CAP deliveries include direct-use AND recharge-facility records by
    default (USF / GSF / ASR — the full CAP supply footprint per basin)
    because recharge volumes are part of consumptive use at the basin
    (later recovered as GW pumping by Tucson Water CAVSARP/SAVSARP,
    Phoenix CAGRD credits, Pinal in-lieu accounts).  SRP deliveries
    are skipped by default because the SRP service-area boundary is
    not publicly mapped — attributing its deliveries to a single GW
    basin (currently Phoenix AMA) is ambiguous.  When ``srp_xlsx`` is
    provided, SRP ``SURFACE WATER`` rows are summed into Phoenix AMA
    alongside CAP; otherwise the validation uses CAP data only
    (recommended default).

    Produces per-basin time series plots, a statistics CSV, and a time
    series CSV.

    Args:
        cap_xlsx (str): Path to the CAP delivery Excel file.
        srp_xlsx (str): Path to the SRP delivery Excel file.
        total_sw_dir (str): Directory with ``Total_SW_YYYY_mm.tif`` rasters.
        basin_shp (str): Shapefile or GeoJSON for Arizona groundwater basins.
        basin_col (str): Column in *basin_shp* identifying each basin.
        output_dir (str): Root output directory for validation results.
        year_range (tuple[int, int]): ``(start_year, end_year)`` inclusive for ML rasters.

    Returns:
        pd.DataFrame: Per-basin statistics (RMSD, MAD, Pct Diff, Pearson R).
    """
    makedirs(output_dir)
    logger.info('=' * 60)
    if srp_xlsx is None:
        logger.info('CAP Total Surface Water Validation (CAP-only)')
    else:
        logger.info('CAP/SRP Total Surface Water Validation')
    logger.info('=' * 60)

    # ── Load basin polygons ──────────────────────────────────────────────
    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Load observed CAP (+ optionally SRP) deliveries ──────────────────
    # Three observation series, all using the same CAP delivery file:
    #   - obs_basin_yearly_total: full CAP+SRP (compared to model Total_SW)
    #   - obs_basin_yearly_mi:    CAP M&I customers only
    #                             (compared to model Non_Irrigation_SW)
    #   - obs_basin_yearly_ag:    CAP irrigation/tribal-ag customers + SRP
    #                             (compared to model Irrigation_SW)
    # Sector classifier in ``_classify_cap_customer`` distinguishes
    # cities/AWBA/CAGRD (M&I) from MSIDD/CAIDD/HVID/GRIC/Ak-Chin (ag).
    logger.info('Loading CAP delivery data (per-sector splits)...')
    obs_basin_yearly = load_cap_srp_annual_sw(cap_xlsx, srp_xlsx)
    obs_basin_yearly_mi = load_cap_srp_annual_sw(
        cap_xlsx, srp_xlsx, sector='mi',
    )
    obs_basin_yearly_ag = load_cap_srp_annual_sw(
        cap_xlsx, srp_xlsx, sector='ag',
    )
    # Spill-water sensitivity test is only meaningful when SRP is loaded.
    obs_spill_basin_yearly = (
        load_cap_srp_annual_sw(cap_xlsx, srp_xlsx, include_spill_water=True)
        if srp_xlsx is not None else obs_basin_yearly
    )
    obs_basins = sorted(obs_basin_yearly.keys())
    logger.info(f'  Basins with observed SW data: {obs_basins}')
    for b in obs_basins:
        yrs = sorted(obs_basin_yearly[b].keys())
        logger.info(f'    {b}: {yrs[0]}-{yrs[-1]} ({len(yrs)} years)')

    # ── Load ML SW rasters per category → basin volumes ─────────────────
    # Total_SW + Irrigation_SW + Non_Irrigation_SW each compared
    # separately against the same observed CAP+SRP delivery.  The
    # per-category dirs are derived from total_sw_dir's prediction-
    # root parent (...Full_Prediction_XGBRF/<cat>_Rasters/Depth_mm/).
    _prediction_root = os.path.dirname(os.path.dirname(total_sw_dir))
    ml_category_dirs = {
        'Total_SW': total_sw_dir,
        'Irrigation_SW': os.path.join(
            _prediction_root, 'Irrigation_SW_Rasters', 'Depth_mm',
        ),
        'Non_Irrigation_SW': os.path.join(
            _prediction_root, 'Non_Irrigation_SW_Rasters', 'Depth_mm',
        ),
    }
    ml_basin_yearly_by_cat: dict[str, dict] = {}
    for _cat, _cat_dir in ml_category_dirs.items():
        logger.info(f'Loading ML {_cat} rasters from {_cat_dir}...')
        _basin_yearly = load_ml_total_sw_basin_volumes(
            _cat_dir, basin_gdf, basin_col, year_range,
            file_prefix=_cat,
        )
        if _basin_yearly:
            ml_basin_yearly_by_cat[_cat] = _basin_yearly
        else:
            logger.warning(f'No ML {_cat} rasters found in {_cat_dir}')

    # ── Per-category σ rasters → per-basin σ volumes (AF) ───────────────
    # Loaded from {prediction_root}/Uncertainty/Sigma_Total/Rasters/
    # via spatial quadrature (sqrt of sum of squared per-pixel σ).
    # Returns empty dict when the σ raster directory is absent —
    # downstream plots silently skip CI bands in that case.
    sigma_raster_dir = os.path.join(
        _prediction_root, 'Uncertainty', 'Sigma_Total', 'Rasters',
    )
    ml_basin_sigma_by_cat: dict[str, dict] = {}
    basin_reproj_for_sigma = (
        basin_gdf.to_crs(ref_crs)
        if ref_crs and basin_gdf.crs != ref_crs else basin_gdf
    ) if False else None  # placeholder; ref_crs is set below — defer load
    # NOTE: σ load happens after ref_crs is established (next block).
    if 'Total_SW' not in ml_basin_yearly_by_cat:
        logger.warning('No ML Total_SW rasters found; skipping validation.')
        return pd.DataFrame()
    ml_basin_yearly = ml_basin_yearly_by_cat['Total_SW']  # backward compat

    # ── Basin areas for depth conversion ─────────────────────────────────
    # Find ref CRS from first available raster
    ref_crs = None
    for yr in range(year_range[0], year_range[1] + 1):
        candidate = os.path.join(total_sw_dir, f'Total_SW_{yr}_mm.tif')
        if os.path.isfile(candidate):
            with rio.open(candidate) as src:
                ref_crs = src.crs
            break
    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if ref_crs and basin_gdf.crs != ref_crs
        else basin_gdf
    )
    basin_areas_m2 = {
        row[basin_col]: row.geometry.area
        for _, row in basin_reproj.iterrows()
    }

    # Now that basin_reproj is set, populate σ load
    for _cat in ml_category_dirs:
        ml_basin_sigma_by_cat[_cat] = _load_basin_sigma_yearly(
            sigma_raster_dir, _cat, basin_reproj, basin_col, year_range,
        )
    if any(ml_basin_sigma_by_cat.values()):
        logger.info(
            'Loaded σ rasters for CI bands from %s', sigma_raster_dir,
        )
    else:
        logger.info(
            'No σ rasters found at %s — CI bands skipped.',
            sigma_raster_dir,
        )

    # ── Per-category metrics + CSV + time-series plots + scatter ────────
    # Each ML category (Total_SW, Irrigation_SW, Non_Irrigation_SW) is
    # compared against the same observed CAP+SRP delivery.  Total_SW
    # additionally gets the ML approximate CAP contribution
    # (Total_SW × delivery_ratio) shown alongside.
    af_to_m3 = 1.0 / M3_TO_AF
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    scatter_dir = os.path.join(output_dir, 'Scatter/')

    # Transpose {basin: {year: AF}} → {year: {basin: AF}} for the
    # time-series plotter which expects the year-keyed format.
    def _transpose_basin_yearly(d):
        out = {}
        for basin, yr_dict in d.items():
            for yr, val in yr_dict.items():
                out.setdefault(yr, {})[basin] = val
        return out

    _cap_colors = {
        'ML': '#2C3E50',
        'ML_CAP_approx': '#9B59B6',
        'CAP_SRP': '#E74C3C',
        'CAP_SRP_spill': '#3498DB',
    }
    _cap_markers = {
        'ML': 'o', 'ML_CAP_approx': 'x', 'CAP_SRP': 's', 'CAP_SRP_spill': '^',
    }

    # Per-category labels for plot legends, file names, and titles.
    # ``plot_intercomp_time_series`` appends ``_{cat}_{basin}`` to the
    # file_prefix.  With cat='SW', file_prefix='TS_Total' produces
    # ``TS_Total_SW_<basin>.png`` (single trailing SW segment).
    _cat_meta = {
        'Total_SW':         ('Total SW',       'TS_Total',
                             'Scatter_ML_Total_SW_vs_CAP_SRP.png',
                             'cap_srp_total_sw'),
        'Irrigation_SW':    ('Irrigation SW',  'TS_Irr',
                             'Scatter_ML_Irr_SW_vs_CAP_SRP.png',
                             'cap_srp_irr_sw'),
        'Non_Irrigation_SW': ('NonIrr SW',      'TS_NIr',
                              'Scatter_ML_NIr_SW_vs_CAP_SRP.png',
                              'cap_srp_nonirr_sw'),
    }

    # Per-category observation pairing:
    #   Total_SW         ↔  CAP+SRP total (all customers + SRP)
    #   Irrigation_SW    ↔  CAP ag (MSIDD/CAIDD/HVID/tribal) + SRP
    #   Non_Irrigation_SW ↔ CAP M&I (cities/AWBA/CAGRD), SRP excluded
    _cat_obs = {
        'Total_SW': obs_basin_yearly,
        'Irrigation_SW': obs_basin_yearly_ag,
        'Non_Irrigation_SW': obs_basin_yearly_mi,
    }
    _cat_obs_label = {
        'Total_SW': 'CAP + SRP (all)',
        'Irrigation_SW': 'CAP Ag + SRP',
        'Non_Irrigation_SW': 'CAP M&I',
    }

    metrics_dfs: dict[str, pd.DataFrame] = {}
    for cat_name, ml_cat_yearly in ml_basin_yearly_by_cat.items():
        ml_label, plot_prefix, scatter_fname, csv_prefix = _cat_meta[cat_name]
        cat_obs_yearly = _cat_obs[cat_name]
        cat_obs_label = _cat_obs_label[cat_name]
        logger.info(
            f'  Processing category: {ml_label} (vs {cat_obs_label})'
        )

        # Per-basin metrics
        cat_metrics = _compute_cap_srp_metrics(
            ml_cat_yearly, cat_obs_yearly, basin_areas_m2,
        )
        cat_metrics_csv = os.path.join(
            output_dir, f'{csv_prefix}_validation_metrics.csv',
        )
        cat_metrics.to_csv(cat_metrics_csv, index=False)
        metrics_dfs[cat_name] = cat_metrics

        # ML approximate-CAP-contribution series for THIS category:
        #   approx[year] = ML_{cat}_SW[year] × delivery_ratio[year]
        # Total_SW × ratio  → approx total CAP contribution
        # Irr_SW × ratio    → approx ag CAP-NIA contribution
        # NIr_SW × ratio    → approx M&I CAP contribution
        # Each tracks the basin's CAP delivery curve relative to the
        # corresponding model SW component.
        approx_yearly = _compute_ml_cap_approximation(
            ml_cat_yearly, cap_xlsx,
        )

        # Per-category time series CSV
        ml_col = f'ML_{cat_name}'
        obs_col = f'CAP_SRP_{cat_name}'
        ts_rows = []
        for basin in obs_basins:
            all_years = sorted(
                set(ml_cat_yearly.get(basin, {}).keys())
                | set(cat_obs_yearly.get(basin, {}).keys())
            )
            for yr in all_years:
                ml_af = ml_cat_yearly.get(basin, {}).get(yr, np.nan)
                obs_af = cat_obs_yearly.get(basin, {}).get(yr, np.nan)
                approx_af = approx_yearly.get(basin, {}).get(yr, np.nan)
                area = basin_areas_m2.get(basin, 1.0)
                approx_col = f'ML_{cat_name}_CAP_approx'
                row = {
                    'Basin': basin,
                    'Year': yr,
                    f'{ml_col}_AF': round(ml_af, 2) if np.isfinite(ml_af) else np.nan,
                    f'{approx_col}_AF': round(approx_af, 2) if np.isfinite(approx_af) else np.nan,
                    f'{obs_col}_AF': round(obs_af, 2) if np.isfinite(obs_af) else np.nan,
                    f'{ml_col}_m3': round(ml_af * af_to_m3, 2) if np.isfinite(ml_af) else np.nan,
                    f'{approx_col}_m3': round(approx_af * af_to_m3, 2) if np.isfinite(approx_af) else np.nan,
                    f'{obs_col}_m3': round(obs_af * af_to_m3, 2) if np.isfinite(obs_af) else np.nan,
                    f'{ml_col}_mm': round(ml_af * af_to_m3 / area * M_TO_MM, 4) if np.isfinite(ml_af) and area > 0 else np.nan,
                    f'{approx_col}_mm': round(approx_af * af_to_m3 / area * M_TO_MM, 4) if np.isfinite(approx_af) and area > 0 else np.nan,
                    f'{obs_col}_mm': round(obs_af * af_to_m3 / area * M_TO_MM, 4) if np.isfinite(obs_af) and area > 0 else np.nan,
                }
                ts_rows.append(row)
        ts_df = pd.DataFrame(ts_rows)
        ts_csv = os.path.join(output_dir, f'{csv_prefix}_time_series.csv')
        ts_df.to_csv(ts_csv, index=False)

        # Per-category time-series plots — include σ for ML and approx
        # CAP (approx σ scales identically to approx mean since
        # approx = ML × ratio).
        ml_sigma_yearly = ml_basin_sigma_by_cat.get(cat_name, {})
        ml_src_dict: dict = {
            'yearly': _transpose_basin_yearly(ml_cat_yearly),
        }
        if ml_sigma_yearly:
            ml_src_dict['yearly_sigma'] = _transpose_basin_yearly(
                ml_sigma_yearly,
            )
        cat_ts_sources = {
            'ML': {'SW': ml_src_dict},
            'CAP_SRP': {'SW': {
                'yearly': _transpose_basin_yearly(cat_obs_yearly),
            }},
        }
        if approx_yearly:
            approx_src_dict: dict = {
                'yearly': _transpose_basin_yearly(approx_yearly),
            }
            # Compute approx σ per basin per year:
            # σ_approx = σ_ML × delivery_ratio, where ratio = approx / ML
            if ml_sigma_yearly:
                approx_sigma: dict[str, dict[int, float]] = {}
                for basin, yr_dict in approx_yearly.items():
                    for yr, approx_val in yr_dict.items():
                        ml_val = ml_cat_yearly.get(basin, {}).get(yr)
                        if (ml_val is None or not np.isfinite(ml_val)
                                or ml_val <= 0):
                            continue
                        ratio = approx_val / ml_val
                        sigma_ml = (
                            ml_sigma_yearly.get(basin, {}).get(yr)
                        )
                        if sigma_ml is None or not np.isfinite(sigma_ml):
                            continue
                        approx_sigma.setdefault(basin, {})[yr] = (
                            sigma_ml * ratio
                        )
                if approx_sigma:
                    approx_src_dict['yearly_sigma'] = (
                        _transpose_basin_yearly(approx_sigma)
                    )
            cat_ts_sources['ML_CAP_approx'] = {'SW': approx_src_dict}
        # CAP+SRP+Spill series only for Total_SW (full-CAP context).
        # Per-sector observations don't have a separate "+ Spill"
        # variant — spill water is a CAP delivery anomaly, not a
        # customer-sector attribute.
        if cat_name == 'Total_SW' and obs_spill_basin_yearly:
            cat_ts_sources['CAP_SRP_spill'] = {
                'SW': {'yearly': _transpose_basin_yearly(obs_spill_basin_yearly)},
            }
        cat_labels = {
            'ML': f'ML ({ml_label})',
            'ML_CAP_approx': f'ML approx CAP × {ml_label}',
            'CAP_SRP': cat_obs_label,
            'CAP_SRP_spill': 'CAP + SRP (+ Spill)',
        }
        plot_intercomp_time_series(
            cat_ts_sources, categories=['SW'],
            basin_names=sorted(cat_obs_yearly.keys()),
            basin_areas_m2=basin_areas_m2,
            output_dir=plot_dir,
            colors=_cap_colors, markers=_cap_markers, labels=cat_labels,
            title_prefix=f'{ml_label} — ', file_prefix=plot_prefix,
        )

        # Per-category scatter (mean over common years)
        ml_mean_vals, obs_mean_vals = {}, {}
        for basin in obs_basins:
            common_years = sorted(
                set(ml_cat_yearly.get(basin, {}).keys())
                & set(cat_obs_yearly.get(basin, {}).keys())
            )
            if common_years:
                ml_mean_vals[basin] = float(np.mean([
                    ml_cat_yearly[basin][yr] for yr in common_years
                ]))
                obs_mean_vals[basin] = float(np.mean([
                    cat_obs_yearly[basin][yr] for yr in common_years
                ]))
        if ml_mean_vals:
            plot_intercomp_scatter(
                [(f'ML {ml_label}', f'Observed {cat_obs_label}',
                  ml_mean_vals, obs_mean_vals)],
                list(ml_mean_vals.keys()), basin_areas_m2, scatter_dir,
                title=f'ML {ml_label} vs {cat_obs_label} — Per Basin',
                filename=scatter_fname,
                is_validation=True,
                annotate_basins=True,
                log_scale=True,
                af_divisor=1000.0,
                af_unit_label='1000 AF',
            )

        # ML approx CAP scatter — one per category
        if approx_yearly:
            approx_mean_vals = {}
            for basin in obs_basins:
                common_years = sorted(
                    set(approx_yearly.get(basin, {}).keys())
                    & set(cat_obs_yearly.get(basin, {}).keys())
                )
                vals = [approx_yearly[basin][yr] for yr in common_years]
                if vals:
                    approx_mean_vals[basin] = float(np.mean(vals))
            if approx_mean_vals:
                # Filename mirrors the per-category scatter:
                # Scatter_ML_<cat>_CAP_approx_vs_CAP_SRP.png
                approx_fname = scatter_fname.replace(
                    '_vs_CAP_SRP.png', '_CAP_approx_vs_CAP_SRP.png',
                )
                plot_intercomp_scatter(
                    [(f'ML approx CAP × {ml_label}',
                      f'Observed {cat_obs_label}',
                      approx_mean_vals, obs_mean_vals)],
                    list(approx_mean_vals.keys()),
                    basin_areas_m2, scatter_dir,
                    title=f'ML approx CAP × {ml_label} vs {cat_obs_label} — Per Basin',
                    filename=approx_fname,
                    is_validation=True,
                    annotate_basins=True,
                    log_scale=True,
                    af_divisor=1000.0,
                    af_unit_label='1000 AF',
                )

    metrics_df = metrics_dfs.get('Total_SW', pd.DataFrame())

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('CAP/SRP Validation Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# PS (Public Supply) Intercomparison — Non-Irrigation vs USGS PS
# ═════════════════════════════════════════════════════════════════════════════


def _load_ps_huc12_annual(
    csv_path: str,
    huc12_geojson: str,
    year_range: tuple[int, int],
    basin_gdf: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Load a USGS PS HUC12 CSV, filter to AZ-interior HUC12s, aggregate
    monthly to annual totals.

    The raw data is in million gallons per day (Mgal/d).  For each
    HUC12 × year we compute the annual total volume in acre-feet:

        annual_vol_AF = Σ_months (rate_Mgal_d × days_in_month) × Mgal_to_m3 × m3_to_AF

    Returns a long-form DataFrame with columns: ``huc12, year, volume_AF``.
    """
    import calendar

    logger.info(f'Loading PS data: {csv_path}')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str)

    # Identify AZ-interior HUC12 columns (drop cross-border HUC12s)
    huc_gdf = gpd.read_file(huc12_geojson)
    if basin_gdf is not None:
        huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    az_huc12_set = set(huc_gdf['huc12'].astype(str).values)
    all_huc_cols = [c for c in df.columns if c not in ('Year', 'Month')]
    az_cols = [c for c in all_huc_cols if c in az_huc12_set]
    logger.info(f'  {len(az_cols)}/{len(all_huc_cols)} AZ-interior HUC12s in PS')

    if not az_cols:
        logger.warning('  No AZ HUC12 matches in PS CSV')
        return pd.DataFrame(columns=['huc12', 'year', 'volume_AF'])

    start_yr, end_yr = year_range
    df = df[(df['Year'] >= start_yr) & (df['Year'] <= end_yr)].copy()

    # Vectorised approach: melt to long form, then aggregate
    id_vars = ['Year', 'Month']
    melted = df[id_vars + az_cols].melt(
        id_vars=id_vars, var_name='huc12', value_name='rate_mgald',
    )
    melted['rate_mgald'] = pd.to_numeric(melted['rate_mgald'], errors='coerce')
    melted = melted[melted['rate_mgald'] > 0].copy()

    # Days per month
    melted['ndays'] = melted.apply(
        lambda r: calendar.monthrange(int(r['Year']), int(r['Month']))[1],
        axis=1,
    )
    # Mgal/d × days → Mgal; × MGAL_TO_M3 → m³; × M3_TO_AF → AF
    melted['volume_AF'] = (
        melted['rate_mgald'] * melted['ndays'] * MGAL_TO_M3 * M3_TO_AF
    )

    # Sum months → annual per HUC12
    annual = (
        melted.groupby(['huc12', 'Year'])['volume_AF']
        .sum()
        .reset_index()
        .rename(columns={'Year': 'year'})
    )
    return annual


def _ps_annual_to_basin_volumes(
    annual_df: pd.DataFrame,
    huc12_geojson: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    ref_crs,
) -> dict:
    """Aggregate PS annual HUC12 volumes to basin totals.

    Returns ``{'mean': {basin: AF}, 'yearly': {year: {basin: AF}}}``.
    """
    if annual_df.empty:
        basins = basin_gdf[basin_col].unique()
        return {
            'mean': {b: 0.0 for b in basins},
            'yearly': {},
        }

    huc_gdf = gpd.read_file(huc12_geojson)
    huc_gdf = _filter_huc12_within_az(huc_gdf, basin_gdf)
    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    huc_reproj = huc_gdf.to_crs(ref_crs)
    huc_reproj['area_m2'] = huc_reproj.geometry.area

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    # Spatial overlay: HUC12 → basin fractional membership
    overlay = _get_huc_basin_overlay(
        huc_reproj, basin_reproj, basin_col,
    )

    years = sorted(annual_df['year'].unique())
    yearly_vols = {}
    for year in years:
        yr_data = annual_df[annual_df.year == year].set_index('huc12')['volume_AF']
        merged = overlay.merge(
            yr_data, left_on='huc12', right_index=True, how='left',
        ).fillna(0.0)
        merged['weighted_vol'] = merged['volume_AF'] * merged['area_frac']
        basin_sums = merged.groupby(basin_col)['weighted_vol'].sum()
        yearly_vols[year] = {
            b: basin_sums.get(b, 0.0)
            for b in basin_reproj[basin_col]
        }

    # Mean annual
    basin_names = basin_reproj[basin_col].unique()
    mean_vols = {}
    for b in basin_names:
        vals = [yearly_vols[y].get(b, 0.0) for y in years]
        mean_vols[b] = float(np.mean(vals)) if vals else 0.0

    return {'mean': mean_vols, 'yearly': yearly_vols}


def run_ps_intercomparison(
    nonirr_dir: str,
    nonirr_gw_dir: str,
    nonirr_sw_dir: str,
    ps_data_dir: str,
    huc12_geojson: str,
    basin_shp: str,
    basin_col: str,
    output_dir: str,
    year_range: tuple[int, int] = (2000, 2020),
) -> pd.DataFrame:
    """Compare ML non-irrigation withdrawal predictions with USGS Public
    Supply (PS) reanalysis data (Alzraiee et al. 2024, WRR).

    Public supply is a **subset** of non-irrigation water use.  ML
    Non_Irrigation predictions should be >= PS estimates in most basins.
    This comparison quantifies how much of the non-irrigation sector is
    attributable to public supply, and validates the GW/SW source
    partitioning independently.

    Categories compared:
        * Non_Irrigation (total) vs PS Total
        * Non_Irrigation_GW vs PS GW
        * Non_Irrigation_SW vs PS SW

    Args:
        nonirr_dir (str): Directory with ``Non_Irrigation_YYYY_mm.tif`` rasters.
        nonirr_gw_dir (str): Directory with ``Non_Irrigation_GW_YYYY_mm.tif`` rasters.
        nonirr_sw_dir (str): Directory with ``Non_Irrigation_SW_YYYY_mm.tif`` rasters.
        ps_data_dir (str): Directory with PS HUC12 CSVs (Tot, GW, SW).
        huc12_geojson (str): Path to ``AZ_HUC12.geojson``.
        basin_shp (str): Basin boundary shapefile.
        basin_col (str): Column in *basin_shp* naming each basin.
        output_dir (str): Root output directory.
        year_range (tuple[int, int]): Year range (default 2000-2020 to match PS data availability).

    Returns:
        pd.DataFrame: Summary metrics table.
    """
    makedirs(output_dir)
    logger.info('=' * 60)
    logger.info('Non-Irrigation vs USGS Public Supply Intercomparison')
    logger.info('=' * 60)

    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Find reference raster for CRS ─────────────────────────────────
    ref_raster = None
    start_yr, end_yr = year_range
    for yr in range(start_yr, end_yr + 1):
        candidate = os.path.join(
            nonirr_dir, f'Non_Irrigation_{yr}_mm.tif',
        )
        if os.path.isfile(candidate):
            ref_raster = candidate
            break
    if ref_raster is None:
        raise FileNotFoundError(
            f'No Non_Irrigation rasters found in {nonirr_dir}'
        )
    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basin_reproj = (
        basin_gdf.to_crs(ref_crs)
        if basin_gdf.crs != ref_crs else basin_gdf
    )
    basin_areas_m2 = {
        row[basin_col]: row.geometry.area
        for _, row in basin_reproj.iterrows()
    }

    # ── 1. Load ML non-irrigation basin volumes ──────────────────────
    logger.info('--- Loading ML non-irrigation predictions ---')
    ml_cats = {
        'Total': (nonirr_dir, 'Non_Irrigation_{year}_mm.tif'),
        'GW': (nonirr_gw_dir, 'Non_Irrigation_GW_{year}_mm.tif'),
        'SW': (nonirr_sw_dir, 'Non_Irrigation_SW_{year}_mm.tif'),
    }
    ml_vols = {}
    for cat_label, (cat_dir, pattern) in ml_cats.items():
        mean_depth = None
        n_years = 0
        yearly_vols = {}
        for year in range(start_yr, end_yr + 1):
            raster_path = os.path.join(cat_dir, pattern.format(year=year))
            if not os.path.isfile(raster_path):
                continue
            yearly_vols[year] = _raster_basin_volumes(
                raster_path, basin_reproj, basin_col,
                pixel_area_m2, depth_unit='mm',
            )
            arr = read_raster_as_arr(raster_path, get_file=False).astype(np.float64)
            arr[np.isnan(arr)] = 0.0
            arr[arr < 0] = 0.0
            if mean_depth is None:
                mean_depth = arr.copy()
            else:
                mean_depth += arr
            n_years += 1

        if n_years > 0:
            mean_depth /= n_years
            # Write mean-annual raster
            out_tif = os.path.join(
                output_dir, f'ML_mean_annual_NonIrr_{cat_label}_mm.tif',
            )
            with rio.open(ref_raster) as ref_src:
                profile = ref_src.profile.copy()
            profile.update(dtype='float64', nodata=np.nan, count=1)
            tmp = mean_depth.copy()
            tmp[tmp == 0] = np.nan
            with rio.open(out_tif, 'w', **profile) as dst:
                dst.write(tmp, 1)
            logger.info(f'  Wrote ML mean-annual raster: {out_tif}')

            basin_vols = _raster_basin_volumes(
                out_tif, basin_reproj, basin_col,
                pixel_area_m2, depth_unit='mm',
            )
        else:
            logger.warning(f'No ML Non_Irrigation {cat_label} rasters found')
            basin_vols = {b: 0.0 for b in basin_gdf[basin_col]}

        ml_vols[cat_label] = {'mean': basin_vols, 'yearly': yearly_vols}

    # ── 2. Load USGS PS data ─────────────────────────────────────────
    logger.info('--- Loading USGS Public Supply data ---')
    ps_files = {
        'Total': os.path.join(ps_data_dir, 'PS_HUC12_Tot_2000_2020.csv'),
        'GW': os.path.join(ps_data_dir, 'PS_HUC12_GW_2000_2020.csv'),
        'SW': os.path.join(ps_data_dir, 'PS_HUC12_SW_2000_2020.csv'),
    }
    ps_vols = {}
    for cat_label, csv_path in ps_files.items():
        if not os.path.isfile(csv_path):
            logger.warning(f'PS file not found: {csv_path}')
            ps_vols[cat_label] = {
                'mean': {b: 0.0 for b in basin_gdf[basin_col]},
                'yearly': {},
            }
            continue
        annual_df = _load_ps_huc12_annual(
            csv_path, huc12_geojson, year_range, basin_gdf=basin_gdf,
        )
        ps_vols[cat_label] = _ps_annual_to_basin_volumes(
            annual_df, huc12_geojson, basin_gdf, basin_col, ref_crs,
        )

    # ── 3. Compute metrics ───────────────────────────────────────────
    basin_names = sorted(basin_gdf[basin_col].unique().tolist())
    all_metrics = []

    cat_labels = {
        'Total': 'Non_Irrigation',
        'GW': 'Non_Irrigation_GW',
        'SW': 'Non_Irrigation_SW',
    }
    for cat_key, cat_name in cat_labels.items():
        m = _compute_metrics(
            basin_names,
            ml_vols[cat_key]['mean'],
            ps_vols[cat_key]['mean'],
            'ML', 'PS',
            basin_areas_m2=basin_areas_m2,
        )
        m['Category'] = cat_name
        all_metrics.append(m)
        logger.info(
            f'  {cat_name}: RMSD={m["RMSD_AF"]:.2f} AF, '
            f'MAD={m["MAD_AF"]:.2f} AF, PctDiff={m["Pct_Diff"]:.2f}%'
        )

        # Check: ML should be >= PS (PS is a subset of non-irrigation)
        ml_total = sum(ml_vols[cat_key]['mean'].get(b, 0.0) for b in basin_names)
        ps_total = sum(ps_vols[cat_key]['mean'].get(b, 0.0) for b in basin_names)
        ps_fraction = ps_total / ml_total * 100 if ml_total > 0 else np.nan
        logger.info(
            f'    PS / ML ratio: {ps_fraction:.1f}% '
            f'(PS={ps_total:,.0f} AF, ML={ml_total:,.0f} AF)'
        )
        if ml_total > 0 and ps_total > ml_total:
            logger.warning(
                f'    PS exceeds ML for {cat_name} — expected PS <= ML '
                f'since public supply is a subset of non-irrigation use'
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, 'ps_intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'Metrics saved to {metrics_csv}')

    # ── 4. Temporal agreement ─────────────────────────────────────────
    logger.info('--- Temporal agreement metrics ---')
    temporal_metrics = []
    temporal_per_basin_rows = []
    for cat_key, cat_name in cat_labels.items():
        tm = _compute_temporal_metrics(
            basin_names,
            ml_vols[cat_key].get('yearly', {}),
            ps_vols[cat_key].get('yearly', {}),
            'ML', 'PS',
        )
        summary = {
            'Category': cat_name,
            'Pair': tm['Pair'],
            'Pearson_r_mean': tm['Pearson_r_mean'],
            'Pearson_r_median': tm['Pearson_r_median'],
            'NSE_mean': tm['NSE_mean'],
            'NSE_median': tm['NSE_median'],
            'n_common_years': tm['n_common_years'],
            'n_basins_with_data': tm['n_basins_with_data'],
        }
        temporal_metrics.append(summary)
        logger.info(
            f'  {cat_name}: r (mean={tm["Pearson_r_mean"]}, '
            f'median={tm["Pearson_r_median"]}), '
            f'NSE (mean={tm["NSE_mean"]}, median={tm["NSE_median"]})'
        )
        for pb in tm.get('per_basin', []):
            temporal_per_basin_rows.append({
                'Category': cat_name,
                'Pair': tm['Pair'],
                **pb,
            })

    temporal_df = pd.DataFrame(temporal_metrics)
    temporal_csv = os.path.join(output_dir, 'ps_temporal_agreement.csv')
    temporal_df.to_csv(temporal_csv, index=False)

    if temporal_per_basin_rows:
        tb_df = pd.DataFrame(temporal_per_basin_rows)
        tb_csv = os.path.join(output_dir, 'ps_temporal_per_basin.csv')
        tb_df.to_csv(tb_csv, index=False)

        # Temporal agreement plots
        temporal_plot_dir = os.path.join(output_dir, 'Temporal_Agreement/')
        plot_temporal_heatmap(tb_df, temporal_plot_dir)
        plot_temporal_box_violin(tb_df, temporal_plot_dir)
        plot_temporal_r_vs_nse(tb_df, temporal_plot_dir)

        # Taylor diagram removed — see docstring at withdrawal
        # intercomparison for the rationale (model-vs-model has no
        # "true" reference, hard to interpret).

    # ── 5. Per-basin comparison table ─────────────────────────────────
    af_to_m3 = 1.0 / M3_TO_AF
    rows = []
    for cat_key, cat_name in cat_labels.items():
        for basin in basin_names:
            ml_af = ml_vols[cat_key]['mean'].get(basin, 0.0)
            ps_af = ps_vols[cat_key]['mean'].get(basin, 0.0)
            area = basin_areas_m2.get(basin, 1.0)
            ps_frac = ps_af / ml_af * 100 if ml_af > 0 else np.nan
            rows.append({
                'Category': cat_name,
                'Basin': basin,
                'ML_mm': round(ml_af * af_to_m3 / area * M_TO_MM, 4),
                'ML_ft': round(ml_af * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                'ML_m3': round(ml_af * af_to_m3, 2),
                'ML_AF': round(ml_af, 2),
                'PS_mm': round(ps_af * af_to_m3 / area * M_TO_MM, 4),
                'PS_ft': round(ps_af * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                'PS_m3': round(ps_af * af_to_m3, 2),
                'PS_AF': round(ps_af, 2),
                'PS_pct_of_ML': round(ps_frac, 1) if np.isfinite(ps_frac) else np.nan,
            })
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'ps_per_basin_volumes.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(f'Per-basin volumes saved to {basin_csv}')

    # ── 6. Time series CSV ────────────────────────────────────────────
    ts_rows = []
    all_sources = {'ML': ml_vols, 'PS': ps_vols}
    for cat_key, cat_name in cat_labels.items():
        for source_name, src_data in all_sources.items():
            yearly = src_data[cat_key].get('yearly', {})
            for year in sorted(yearly.keys()):
                for basin in basin_names:
                    af_val = yearly[year].get(basin, 0.0)
                    area = basin_areas_m2.get(basin, 1.0)
                    ts_rows.append({
                        'Category': cat_name,
                        'Source': source_name,
                        'Year': year,
                        'Basin': basin,
                        'Volume_mm': round(af_val * af_to_m3 / area * M_TO_MM, 4),
                        'Volume_ft': round(af_val * af_to_m3 / area * M_TO_MM * MM_TO_FT, 6),
                        'Volume_m3': round(af_val * af_to_m3, 2),
                        'Volume_AF': round(af_val, 2),
                    })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'ps_time_series_volumes.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'Time series saved to {ts_csv}')

    # ── 7. Time series plots ──────────────────────────────────────────
    _ps_colors = {'ML': '#2C3E50', 'PS': '#E74C3C'}
    _ps_markers = {'ML': 'o', 'PS': 's'}
    _ps_labels = {'ML': 'ML Non-Irrigation', 'PS': 'USGS Public Supply'}
    ps_ts_sources = {}
    for cat_key, cat_name in cat_labels.items():
        ps_ts_sources.setdefault('ML', {})[cat_name] = ml_vols[cat_key]
        ps_ts_sources.setdefault('PS', {})[cat_name] = ps_vols[cat_key]
    cat_name_list_ts = list(cat_labels.values())
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    plot_intercomp_time_series(
        ps_ts_sources, categories=cat_name_list_ts,
        basin_names=basin_names, basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_ps_colors, markers=_ps_markers, labels=_ps_labels,
        file_prefix='PS',
    )

    # ── 8. Scatter plots ──────────────────────────────────────────────
    # Log-scale axes for PS basin scatter — public-supply volumes span
    # ~3 decades across AZ basins (a few hundred AF in rural basins to
    # >100 kAF in Phoenix AMA), so a linear axis collapses the small-
    # basin signal at the origin.
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    for cat_key, cat_name in cat_labels.items():
        plot_intercomp_scatter(
            [('ML Non-Irrigation', 'USGS Public Supply',
              ml_vols[cat_key]['mean'], ps_vols[cat_key]['mean'])],
            basin_names, basin_areas_m2, scatter_dir,
            title=f'{cat_name} — ML vs USGS Public Supply',
            filename=f'Scatter_{cat_name}.png',
            log_scale=True,
        )

    # ── 8b. Statewide stacked bar plots ──────────────────────────────
    ps_bar_sources = {
        'ML': {'GW': ml_vols['GW'], 'SW': ml_vols['SW']},
        'PS': {'GW': ps_vols['GW'], 'SW': ps_vols['SW']},
    }
    bar_dir = os.path.join(output_dir, 'Stacked_Bar/')
    plot_intercomp_stacked_bars(
        ps_bar_sources, source_order=['ML', 'PS'],
        output_dir=bar_dir,
        stack_cats=['GW', 'SW'],
        stack_labels={'GW': 'Groundwater', 'SW': 'Surface Water'},
        stack_colors={'GW': '#2C3E50', 'SW': '#3498DB'},
        title_prefix='Non-Irrigation (ML) vs Public Supply (PS) — ',
    )

    # ── 9. HUC12-level comparison ────────────────────────────────────
    logger.info('--- HUC12-level PS comparison ---')
    huc12_dir = os.path.join(output_dir, 'HUC12_Comparison/')
    makedirs(huc12_dir)

    huc_gdf_ps = gpd.read_file(huc12_geojson)
    huc_gdf_ps = _filter_huc12_within_az(huc_gdf_ps, basin_gdf)
    huc_gdf_ps['huc12'] = huc_gdf_ps['huc12'].astype(str)
    with rio.open(ref_raster) as _src:
        ps_pixel_area = abs(_src.transform.a * _src.transform.e)
        huc_reproj_ps = huc_gdf_ps.to_crs(_src.crs)

    az_huc12_ids_ps = sorted(huc_reproj_ps['huc12'].unique())
    huc_areas_ps = {
        str(row['huc12']): row.geometry.area
        for _, row in huc_reproj_ps.iterrows()
    }

    # PS is already at HUC12 level via _load_ps_huc12_annual
    # ML needs zonal stats aggregation to HUC12
    huc12_ps_metrics = []
    for cat_key, cat_name in cat_labels.items():
        logger.info(f'  HUC12-level comparison for {cat_name}...')

        # PS per-HUC12 mean-annual (from the annual_df already loaded)
        ps_csv = ps_files[cat_key]
        if not os.path.isfile(ps_csv):
            continue
        ps_annual = _load_ps_huc12_annual(
            ps_csv, huc12_geojson, year_range, basin_gdf=basin_gdf,
        )
        ps_huc_mean: dict[str, float] = {}
        if not ps_annual.empty:
            for h, grp in ps_annual.groupby('huc12'):
                ps_huc_mean[str(h)] = float(grp['volume_AF'].mean())

        # ML per-HUC12 via zonal stats
        cat_dir, pattern = ml_cats[cat_key]
        ml_huc_accum_ps: dict[str, list[float]] = {
            h: [] for h in az_huc12_ids_ps
        }
        for year in range(start_yr, end_yr + 1):
            rpath = os.path.join(cat_dir, pattern.format(year=year))
            if not os.path.isfile(rpath):
                continue
            yr_stats = _compute_huc12_zonal_stats(
                rpath, huc_reproj_ps, ps_pixel_area, depth_unit='mm',
            )
            for h in az_huc12_ids_ps:
                s = yr_stats.get(h)
                if s:
                    ml_huc_accum_ps[h].append(s['volume_AF'])
        ml_huc_mean_ps = {
            h: float(np.mean(v)) if v else 0.0
            for h, v in ml_huc_accum_ps.items()
        }

        common_hucs_ps = [
            h for h in az_huc12_ids_ps
            if h in ps_huc_mean and h in ml_huc_mean_ps
        ]
        if common_hucs_ps:
            m = _compute_metrics(
                common_hucs_ps, ml_huc_mean_ps, ps_huc_mean,
                'ML', 'PS', basin_areas_m2=huc_areas_ps,
            )
            m['Category'] = cat_name
            m['Level'] = 'HUC12'
            huc12_ps_metrics.append(m)
            logger.info(
                f'    ML vs PS (HUC12): RMSD={m["RMSD_AF"]:.1f} AF, '
                f'PctDiff={m["Pct_Diff"]:.1f}%'
            )

            # HUC12 scatter (log scale — HUC12 PS volumes span 4-5
            # decades, dominated by a long tail of small rural HUC12s)
            huc12_scatter_dir = os.path.join(huc12_dir, 'Scatter/')
            plot_intercomp_scatter(
                [('ML Non-Irrigation', 'USGS PS', ml_huc_mean_ps, ps_huc_mean)],
                common_hucs_ps, huc_areas_ps, huc12_scatter_dir,
                title=f'{cat_name} — HUC12-Level Scatter',
                filename=f'Scatter_HUC12_{cat_name}.png',
                log_scale=True,
            )

            # Combined HUC12 + Pixel + Basin Δ depth (3 panels, shared
            # colorbar in mm).  Same rationale as the CU 3-panel figure:
            # Δ Depth is the only physically meaningful unit shared
            # across pixel / HUC12 / basin scales.  PS data is
            # natively HUC12-level — rasterising it produces a
            # piecewise-uniform raster, so the pixel diff (ML pixel −
            # PS HUC12-uniform) shows where ML's pixel-level estimate
            # deviates from the HUC12 average.
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            _af_to_m3_local = 1.0 / M3_TO_AF
            _mm_to_ft = 1.0 / 304.8
            spatial_diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
            makedirs(spatial_diff_dir)
            b_reproj_ps = (
                basin_gdf.to_crs(huc_reproj_ps.crs)
                if basin_gdf.crs != huc_reproj_ps.crs else basin_gdf
            )
            b_name_ps = (
                basin_col if basin_col in b_reproj_ps.columns
                else b_reproj_ps.columns[0]
            )
            ama_ina_names_ps = get_ama_ina_basin_names()

            # Panel 1: HUC12 Δ depth (mm)
            huc12_diff_vals: dict[str, float] = {}
            for h in common_hucs_ps:
                area = huc_areas_ps.get(h, 0.0)
                if area > 0:
                    huc12_diff_vals[h] = (
                        (ml_huc_mean_ps.get(h, 0.0)
                         - ps_huc_mean.get(h, 0.0))
                        * _af_to_m3_local / area * M_TO_MM
                    )

            # Panel 2: Pixel-level Δ depth — ML mean raster minus
            # PS HUC12 depth rasterised onto the ML grid.
            ml_mean_raster_ps = os.path.join(
                output_dir,
                f'ML_mean_annual_NonIrr_{cat_key}_mm.tif',
            )
            pixel_diff_arr_ps = None
            pixel_extent_ps = None
            if os.path.isfile(ml_mean_raster_ps):
                with rio.open(ml_mean_raster_ps) as src:
                    ml_arr_ps = src.read(1).astype(np.float64)
                    pixel_extent_ps = [
                        src.bounds.left, src.bounds.right,
                        src.bounds.bottom, src.bounds.top,
                    ]
                    ps_ref_transform = src.transform
                    ps_ref_shape = (src.height, src.width)
                ml_arr_ps[np.isnan(ml_arr_ps)] = 0.0

                # Rasterise PS HUC12 depths onto the ML grid
                ps_depth_shapes = []
                for _, hrow in huc_reproj_ps.iterrows():
                    h_id = str(hrow['huc12'])
                    area = huc_areas_ps.get(h_id, 0.0)
                    if area <= 0:
                        continue
                    ps_af = ps_huc_mean.get(h_id, 0.0)
                    if ps_af == 0:
                        continue
                    depth_mm = ps_af * _af_to_m3_local / area * M_TO_MM
                    ps_depth_shapes.append((hrow.geometry, depth_mm))
                if ps_depth_shapes:
                    ps_arr = rasterize(
                        ps_depth_shapes, out_shape=ps_ref_shape,
                        transform=ps_ref_transform, fill=0.0,
                        dtype='float64',
                        merge_alg=rio.enums.MergeAlg.replace,
                    )
                else:
                    ps_arr = np.zeros(ps_ref_shape, dtype=np.float64)
                pixel_diff_arr_ps = ml_arr_ps - ps_arr

            # Panel 3: Basin Δ depth (mm)
            basin_areas_lookup_ps = {
                row[basin_col]: row.geometry.area
                for _, row in b_reproj_ps.iterrows()
            }
            basin_diff_vals: dict[str, float] = {}
            for b in basin_names:
                area = basin_areas_lookup_ps.get(b, 0.0)
                if area > 0:
                    basin_diff_vals[b] = (
                        (ml_vols[cat_key]['mean'].get(b, 0.0)
                         - ps_vols[cat_key]['mean'].get(b, 0.0))
                        * _af_to_m3_local / area * M_TO_MM
                    )

            # Shared vmax across the three panels
            vmax_candidates: list[float] = []
            if huc12_diff_vals:
                vals = np.array(
                    [v for v in huc12_diff_vals.values() if abs(v) > 1e-6],
                )
                if vals.size:
                    vmax_candidates.extend([
                        abs(np.nanpercentile(vals, 2)),
                        abs(np.nanpercentile(vals, 98)),
                    ])
            if pixel_diff_arr_ps is not None:
                pix_vals = pixel_diff_arr_ps[
                    np.abs(pixel_diff_arr_ps) > 1e-6
                ]
                if pix_vals.size:
                    vmax_candidates.extend([
                        abs(np.nanpercentile(pix_vals, 2)),
                        abs(np.nanpercentile(pix_vals, 98)),
                    ])
            if basin_diff_vals:
                vals = np.array(
                    [v for v in basin_diff_vals.values() if abs(v) > 1e-6],
                )
                if vals.size:
                    vmax_candidates.extend([
                        abs(np.nanpercentile(vals, 2)),
                        abs(np.nanpercentile(vals, 98)),
                    ])
            vmax = max(vmax_candidates) if vmax_candidates else 1.0
            vmax = max(vmax, 1e-6)

            fig, axes = plt.subplots(
                1, 3, figsize=(20, 7), constrained_layout=True,
            )
            fig.suptitle(
                f'{cat_name} \u2014 Mean-Annual Depth Difference '
                f'(ML \u2212 PS)',
                fontsize=14, fontweight='bold',
            )

            # Panel 1: HUC12
            ax_huc = axes[0]
            ax_huc.set_facecolor('#D5D5D5')
            plot_gdf = huc_reproj_ps[
                huc_reproj_ps['huc12'].isin(common_hucs_ps)
            ].copy()
            plot_gdf = plot_gdf.set_index('huc12').loc[common_hucs_ps]
            plot_gdf['diff'] = [
                huc12_diff_vals.get(h, np.nan) for h in common_hucs_ps
            ]
            plot_gdf.loc[plot_gdf['diff'].abs() < 1e-6, 'diff'] = np.nan
            plot_gdf.plot(
                ax=ax_huc, column='diff', cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                edgecolor='#AAAAAA', linewidth=0.3,
                legend=False, missing_kwds={'color': '#EEEEEE'},
            )
            _overlay_boundaries(
                ax_huc, b_reproj_ps, ama_ina_names_ps, b_name_ps,
                label_fontsize=5.0, label_all=False,
            )
            ax_huc.set_title('HUC12-Level', fontweight='bold')

            # Panel 2: Pixel-level
            ax_pix = axes[1]
            ax_pix.set_facecolor('#D5D5D5')
            if pixel_diff_arr_ps is not None:
                pix_mask = np.abs(pixel_diff_arr_ps) < 1e-6
                pix_masked = np.ma.masked_where(pix_mask, pixel_diff_arr_ps)
                ax_pix.imshow(
                    pix_masked, extent=pixel_extent_ps, origin='upper',
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    interpolation='nearest',
                )
                _overlay_boundaries(
                    ax_pix, b_reproj_ps, ama_ina_names_ps, b_name_ps,
                    label_fontsize=5.0, label_all=False,
                )
            else:
                ax_pix.text(
                    0.5, 0.5, 'Pixel-level rasters unavailable',
                    ha='center', va='center', transform=ax_pix.transAxes,
                )
                ax_pix.axis('off')
            ax_pix.set_title('Pixel-Level', fontweight='bold')

            # Panel 3: Basin
            ax_bas = axes[2]
            ax_bas.set_facecolor('#D5D5D5')
            plot_b = b_reproj_ps.set_index(basin_col).copy()
            plot_b['diff'] = plot_b.index.map(
                lambda b: basin_diff_vals.get(b, np.nan),
            )
            plot_b.loc[plot_b['diff'].abs() < 1e-6, 'diff'] = np.nan
            plot_b.plot(
                ax=ax_bas, column='diff', cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                edgecolor='#666666', linewidth=0.5,
                legend=False, missing_kwds={'color': '#EEEEEE'},
            )
            _overlay_boundaries(
                ax_bas, b_reproj_ps, ama_ina_names_ps, b_name_ps,
                label_fontsize=5.0, label_all=False,
            )
            ax_bas.set_title('Basin-Level', fontweight='bold')

            # Shared colorbar (Δ Depth in mm, secondary axis ft)
            sm = ScalarMappable(cmap='RdBu_r', norm=Normalize(-vmax, vmax))
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=list(axes), shrink=0.5, pad=0.06,
                orientation='horizontal', aspect=40, extend='both',
            )
            cbar.set_label(
                '\u0394 Depth (mm)', fontsize=10, fontweight='bold',
            )
            cbar.ax.tick_params(labelsize=10)
            secax = cbar.ax.secondary_xaxis(
                'top',
                functions=(
                    lambda x: x * _mm_to_ft,
                    lambda x: x / _mm_to_ft,
                ),
            )
            secax.set_xlabel(
                '\u0394 Depth (ft)', fontsize=10, fontweight='bold',
            )
            secax.tick_params(labelsize=10)

            add_ama_ina_legend(axes[0])
            fig.savefig(
                os.path.join(
                    spatial_diff_dir, f'Spatial_Diff_PS_{cat_name}.png',
                ),
                dpi=600, bbox_inches='tight',
            )
            plt.close(fig)
            logger.info(
                f'  PS 3-panel diff figure for {cat_name} saved to '
                f'{spatial_diff_dir}'
            )

            # HUC12 temporal diagnostics
            # Build per-year {year: {huc12: AF}} for ML
            ml_yearly_huc_ps: dict[int, dict[str, float]] = {}
            for year in range(start_yr, end_yr + 1):
                rpath = os.path.join(cat_dir, pattern.format(year=year))
                if not os.path.isfile(rpath):
                    continue
                yr_stats = _compute_huc12_zonal_stats(
                    rpath, huc_reproj_ps, ps_pixel_area, depth_unit='mm',
                )
                ml_yearly_huc_ps[year] = {
                    h: yr_stats.get(h, {}).get('volume_AF', 0.0)
                    for h in az_huc12_ids_ps
                }
            # PS per-year {year: {huc12: AF}}
            ps_yearly_huc: dict[int, dict[str, float]] = {}
            if not ps_annual.empty:
                for yr, grp in ps_annual.groupby('year'):
                    ps_yearly_huc[int(yr)] = {
                        str(row['huc12']): row['volume_AF']
                        for _, row in grp.iterrows()
                    }
            _huc12_temporal_diagnostics(
                huc12_yearly_sources={
                    'ML': ml_yearly_huc_ps, 'PS': ps_yearly_huc,
                },
                pairs=[('ML', 'PS')],
                huc12_ids=common_hucs_ps,
                category=cat_name,
                output_dir=huc12_dir,
                huc_areas=huc_areas_ps,
            )

    if huc12_ps_metrics:
        pd.DataFrame(huc12_ps_metrics).to_csv(
            os.path.join(huc12_dir, 'huc12_ps_metrics.csv'), index=False,
        )

    # ── 9b. Basin-aggregated Δ volume choropleths (ML − PS) ───────────
    # Per-basin mean-annual Δ Volume + per-basin pct difference + 95 %
    # CI annotation derived from the ML σ rasters (PS treated as
    # deterministic — no σ from the reanalysis).  Volume only.
    basin_diff_dir = os.path.join(output_dir, 'Spatial_Diff/')
    makedirs(basin_diff_dir)
    basin_b_reproj = (
        basin_gdf.to_crs(huc_reproj_ps.crs)
        if basin_gdf.crs != huc_reproj_ps.crs else basin_gdf
    )
    # Per-category basin σ rasters (from PS prediction root)
    sigma_raster_dir_ps = os.path.join(
        os.path.dirname(os.path.dirname(nonirr_dir)),
        'Uncertainty', 'Sigma_Total', 'Rasters',
    )
    cat_to_sigma_prefix = {
        'Total': 'Non_Irrigation',
        'GW': 'Non_Irrigation_GW',
        'SW': 'Non_Irrigation_SW',
    }
    # Mean σ over the year_range = quadrature of per-year σ / N_years
    def _sigma_mean_for_cat(cat_key: str) -> dict[str, float]:
        prefix = cat_to_sigma_prefix.get(cat_key)
        if not prefix:
            return {}
        # Build one synthetic year_range covering the analysis window
        per_yr = _load_basin_sigma_yearly(
            sigma_raster_dir_ps, prefix, basin_b_reproj, basin_col,
            (start_yr, end_yr),
        )
        # Spatial quadrature already done per-year; compute mean over
        # available years per basin (treating per-year σ as sample
        # spread on the time series).
        out: dict[str, float] = {}
        for b, yr_dict in per_yr.items():
            vals = [v for v in yr_dict.values() if np.isfinite(v) and v > 0]
            if vals:
                out[b] = float(np.mean(vals))
        return out
    panels_ps = []
    for cat_key, cat_name in cat_labels.items():
        a_mean = ml_vols[cat_key]['mean']
        b_mean = ps_vols[cat_key]['mean']
        if not (a_mean and b_mean):
            continue
        panels_ps.append({
            'basin_a_vols': a_mean,
            'basin_b_vols': b_mean,
            'panel_title': cat_name.replace('_', ' '),
            'label_a': 'ML',
            'label_b': 'PS',
        })
    if panels_ps:
        # Per-panel colorbars (Total / GW / SW magnitudes can differ
        # by 2-3× — a shared colorbar would compress the GW / SW
        # signal next to the larger Total).
        _plot_basin_diff_panels(
            panels=panels_ps,
            basin_gdf=basin_b_reproj,
            basin_col=basin_col,
            title='Non-Irrigation \u2014 Basin-Level Volume Diff (ML \u2212 PS)',
            out_path=os.path.join(
                basin_diff_dir, 'Spatial_Diff_PS.png',
            ),
            shared_colorbar=False,
        )
    logger.info(f'Basin-level Δ volume map saved to {basin_diff_dir}')

    # ── Summary ───────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('PS Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# USGS AZ statewide calibration overview
# Mirrors USGS OFR 94-476 (Anning & Duet 1994) Figure 1 + post-1950 USGS
# Circular anchors as a direct visual calibration check on the model's
# AZ-wide annual Total_GW (and Total_SW) time series.
# ═════════════════════════════════════════════════════════════════════════════

def _load_usgs_az_anchors(usgs_csv: str) -> pd.DataFrame:
    """Load USGS AZ statewide annual water-use anchors as a tidy DataFrame.

    Reads ``USGS_AZ_Water_Use_1950_1980.csv`` and returns rows whose
    ``Category`` contains ``Total`` (e.g. ``Total Offstream``,
    ``Total (excl power)``, ``Total (GW only)``).  All values are
    normalized to thousand acre-feet (kAF), preferring the explicit
    ``*_1000AF`` columns and falling back to ``*_mgd × 1.12034`` when
    only the daily-rate columns are populated.

    Returned columns: ``Year, Source, GW_kAF, SW_kAF, Total_kAF``.
    """
    # The USGS calibration CSV has quoted Notes with embedded commas
    # AND a few post-1950 rows carry an extra (empty) field, so the
    # default parser truncates or skips them.  Read manually via the
    # csv module to reliably extract the first 11 columns per row.
    import csv
    cols = [
        'Year', 'Source', 'Category',
        'GW_mgd', 'SW_mgd', 'Total_mgd',
        'GW_1000AF', 'SW_1000AF', 'Total_1000AF',
        'CU_1000AF', 'Notes',
    ]
    rows = []
    with open(usgs_csv, newline='') as fh:
        reader = csv.reader(fh, quotechar='"')
        next(reader, None)  # skip header
        for raw in reader:
            if not raw:
                continue
            # Pad short rows; take first 11 from longer rows.
            fields = (raw + [''] * (11 - len(raw)))[:11]
            rows.append(fields)
    df = pd.DataFrame(rows, columns=cols)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    df = df.loc[
        df['Category'].astype(str).str.contains('Total', case=False, na=False)
    ].copy()

    def _kaf(row, col_kaf, col_mgd):
        v = pd.to_numeric(row.get(col_kaf), errors='coerce')
        if pd.notna(v):
            return float(v)
        v = pd.to_numeric(row.get(col_mgd), errors='coerce')
        if pd.notna(v):
            return float(v) * MGD_TO_AF_PER_YEAR / 1000.0
        return np.nan

    df['GW_kAF'] = df.apply(lambda r: _kaf(r, 'GW_1000AF', 'GW_mgd'), axis=1)
    df['SW_kAF'] = df.apply(lambda r: _kaf(r, 'SW_1000AF', 'SW_mgd'), axis=1)
    df['Total_kAF'] = df.apply(
        lambda r: _kaf(r, 'Total_1000AF', 'Total_mgd'), axis=1,
    )
    return (
        df[['Year', 'Source', 'GW_kAF', 'SW_kAF', 'Total_kAF']]
        .sort_values('Year').reset_index(drop=True)
    )


def _load_az_sigma_total_for_category(
        sigma_rasters_dir: str,
        cat_name: str,
        years: list[int],
        pixel_area_m2: float,
) -> dict[int, float]:
    """AZ-wide σ (AF/yr) per year for one category, by spatial quadrature.

    Reads ``Sigma_Total_{cat_name}_mm_{year}.tif`` from *sigma_rasters_dir*
    and aggregates the per-pixel σ to an AZ-wide σ via
    ``sqrt(sum(σ_pixel²))`` (treats per-pixel σ as approximately
    independent for the spatial sum — matches what `_plot_basin_sigma`
    does at the AZ level).
    """
    out: dict[int, float] = {}
    if not os.path.isdir(sigma_rasters_dir):
        return out
    mm_to_m3 = pixel_area_m2 / 1000.0
    for yr in years:
        f = os.path.join(
            sigma_rasters_dir, f'Sigma_Total_{cat_name}_mm_{yr}.tif',
        )
        if not os.path.isfile(f):
            continue
        arr = read_raster_as_arr(f, get_file=False)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        sigma_m3 = arr * mm_to_m3
        sigma_total_m3 = float(np.sqrt(np.sum(sigma_m3 ** 2)))
        out[yr] = sigma_total_m3 * M3_TO_AF
    return out


def _plot_calibration_bars(
        years: np.ndarray,
        model: np.ndarray,
        sigma: np.ndarray,
        usgs: np.ndarray,
        title: str,
        out_path: str,
        bar_color: str,
        anchor_color: str,
        ylabel: str = 'Withdrawal (MAF)',
) -> None:
    """Annual model bars (MAF) with 95% CI caps + USGS anchor overlay bars.

    At each anchor year, an outlined transparent USGS bar is drawn
    over the model bar with a Δ% annotation above the pair so the
    calibration gap is visible at a glance.

    Args:
        years: 1-D int array of years (every annual bar).
        model: 1-D model values aligned with ``years`` (MAF).
        sigma: 1-D model 1σ values aligned with ``years`` (MAF).
            Use NaN at years without σ.  Drawn as ±1.96 × σ (95% CI).
        usgs: 1-D USGS anchor values aligned with ``years`` (MAF).
            Use NaN at non-anchor years.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        years, model, color=bar_color, alpha=0.85,
        edgecolor='#1B4F72', linewidth=0.4,
        label='Model annual',
    )
    valid_sig = ~np.isnan(sigma) & ~np.isnan(model)
    if np.any(valid_sig):
        ci_95 = 1.96 * sigma[valid_sig]
        ax.errorbar(
            years[valid_sig], model[valid_sig],
            yerr=ci_95,
            fmt='none', ecolor='#1B2631', elinewidth=1.4, capsize=4,
            capthick=1.4, alpha=0.95, zorder=6,
            label='Model 95 % CI',
        )
    valid_u = ~np.isnan(usgs)
    if np.any(valid_u):
        ax.bar(
            years[valid_u], usgs[valid_u],
            color='none', edgecolor=anchor_color, linewidth=1.8,
            alpha=1.0, zorder=4,
            label='USGS anchor',
        )
        for yr, m_v, u_v in zip(years[valid_u], model[valid_u], usgs[valid_u]):
            if not (np.isfinite(m_v) and np.isfinite(u_v) and u_v > 0):
                continue
            d = 100.0 * (m_v - u_v) / u_v
            top = max(m_v, u_v)
            mask = years == yr
            if mask.any():
                s = sigma[mask][0]
                if np.isfinite(s):
                    top = max(top, m_v + 1.96 * s)
            ax.annotate(
                f'{d:+.1f}%', xy=(yr, top), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=7, color='#2C3E50', fontweight='bold',
                rotation=90,
            )
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_usgs_calibration_bars(
        df: pd.DataFrame,
        column: str,        # 'GW' or 'SW'
        title: str,
        out_path: str,
        bar_color: str,
        anchor_color: str,
) -> None:
    """Backward-compatible wrapper around ``_plot_calibration_bars``.

    Reads the legacy column convention ``Model_{column}_kAF``,
    ``Sigma_Model_{column}_kAF``, ``USGS_{column}_kAF``.
    """
    years = df['Year'].astype(int).values
    model = df[f'Model_{column}_kAF'].values / 1000.0  # → MAF
    sigma = df[f'Sigma_Model_{column}_kAF'].values / 1000.0  # → MAF
    usgs = df[f'USGS_{column}_kAF'].values / 1000.0  # → MAF
    _plot_calibration_bars(
        years=years, model=model, sigma=sigma, usgs=usgs,
        title=title, out_path=out_path,
        bar_color=bar_color, anchor_color=anchor_color,
        ylabel=f'Total {column} Withdrawal (MAF)',
    )


def _load_anchors_from_summary(usgs_summary_csv: str) -> pd.DataFrame:
    """Load USGS+ADWR anchor totals from AZ_Annual_WU_Summary.csv.

    Returns columns: Year, Source, GW_kAF, SW_kAF, Total_kAF, plus
    per-category kAF (Irr, NIR, IrrGW, IrrSW, NIGW, NISW).  Pulls
    every USGS row.  ADWR rows are excluded here (they appear in the
    category comparison table but do not constitute a USGS anchor for
    the bar charts).
    """
    df = pd.read_csv(usgs_summary_csv)
    df = df[df['Source'].astype(str).str.startswith('USGS', na=False)].copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['Year']).reset_index(drop=True)

    def _val(col):
        return df[col].apply(
            lambda v: float(v) * 1000.0 if pd.notna(v) else float('nan'),
        )

    return pd.DataFrame({
        'Year':       df['Year'].astype(int),
        'Source':     df['Source'].astype(str),
        'Total_kAF':  _val('Total_MAF'),
        'GW_kAF':     _val('Total_GW_MAF'),
        'SW_kAF':     _val('Total_SW_MAF'),
        'Irr_kAF':    _val('Total_Irr_MAF'),
        'NIR_kAF':    _val('Total_NonIrr_MAF'),
        'IrrGW_kAF':  _val('Irr_GW_MAF'),
        'IrrSW_kAF':  _val('Irr_SW_MAF'),
        'NIGW_kAF':   _val('NonIrr_GW_MAF'),
        'NISW_kAF':   _val('NonIrr_SW_MAF'),
    })


def run_usgs_az_calibration_overview(
        annual_summaries_dir: str,
        usgs_csv: str,
        sigma_rasters_dir: str,
        output_dir: str,
        pixel_area_m2: float = 4_000_000,
        start_year: int = 1915,
        end_year: int = 2017,
        usgs_summary_csv: str | None = None,
) -> pd.DataFrame:
    """AZ-wide annual Total GW & SW bars with ±1σ caps + USGS anchors.

    Mirrors USGS OFR 94-476 (Anning & Duet 1994) Figure 1 in bar form
    and overlays the per-source Circular and OFR 94-476 anchors as a
    direct visual calibration check.  The bar plot shows annual model
    statewide volumes (in MAF) with ±1σ error caps; reference markers
    are USGS Circular and OFR 94-476 reported totals at the same year.

    Args:
        annual_summaries_dir: Directory containing ``Total_GW.csv`` and
            ``Total_SW.csv`` (columns ``Year, Volume_AF``).
        usgs_csv: Path to ``USGS_AZ_Water_Use_1950_1980.csv``.
        sigma_rasters_dir: Path to
            ``Uncertainty/Sigma_Total/Rasters/`` containing
            per-category σ rasters.  When absent, bars render without
            error caps.
        output_dir: Output directory for the bar PNGs and side-by-side
            CSV.
        pixel_area_m2: Pixel area in m² (default 4e6 = 2 km grid).
        start_year, end_year: Plot range.  Default 1915–2017 matches
            USGS coverage.

    Outputs:
        - ``USGS_AZ_Calibration_Bars.csv`` — side-by-side
          model + σ + USGS anchor table.
        - ``USGS_AZ_Total_GW_Bars.png`` — bar comparison for Total_GW.
        - ``USGS_AZ_Total_SW_Bars.png`` — bar comparison for Total_SW
          (1950+ only; pre-1950 USGS reports no SW separately).

    Returns:
        DataFrame written to the side-by-side CSV.
    """
    from hydrolibs.visualops import apply_journal_style
    apply_journal_style()
    makedirs(output_dir)

    gw_csv = os.path.join(annual_summaries_dir, 'Total_GW.csv')
    sw_csv = os.path.join(annual_summaries_dir, 'Total_SW.csv')
    if not (os.path.isfile(gw_csv) and os.path.isfile(sw_csv)):
        logger.warning(
            'Annual summaries Total_GW/SW not found at %s; '
            'skipping USGS calibration overview',
            annual_summaries_dir,
        )
        return pd.DataFrame()

    gw_df = pd.read_csv(gw_csv)[['Year', 'Volume_AF']].rename(
        columns={'Volume_AF': 'Model_GW_AF'},
    )
    sw_df = pd.read_csv(sw_csv)[['Year', 'Volume_AF']].rename(
        columns={'Volume_AF': 'Model_SW_AF'},
    )
    model_df = (
        gw_df.merge(sw_df, on='Year', how='outer')
        .sort_values('Year').reset_index(drop=True)
    )

    yrs_in_range = [
        int(y) for y in model_df['Year'].astype(int).values
        if start_year <= int(y) <= end_year
    ]
    sigma_gw = _load_az_sigma_total_for_category(
        sigma_rasters_dir, 'Total_GW', yrs_in_range, pixel_area_m2,
    )
    sigma_sw = _load_az_sigma_total_for_category(
        sigma_rasters_dir, 'Total_SW', yrs_in_range, pixel_area_m2,
    )

    # Prefer the curated AZ_Annual_WU_Summary.csv when available — it
    # has clean per-Year totals for every USGS anchor (including 1950
    # and 1975 which the legacy USGS_AZ_Water_Use_1950_1980.csv mis-
    # parses due to column shifts).  Fall back to the legacy CSV.
    if usgs_summary_csv and os.path.isfile(usgs_summary_csv):
        usgs_df = _load_anchors_from_summary(usgs_summary_csv)
    elif not os.path.isfile(usgs_csv):
        logger.warning(
            'USGS calibration CSV not found at %s; '
            'skipping anchor overlay',
            usgs_csv,
        )
        usgs_df = pd.DataFrame(
            columns=['Year', 'Source', 'GW_kAF', 'SW_kAF', 'Total_kAF'],
        )
    else:
        usgs_df = _load_usgs_az_anchors(usgs_csv)

    out_rows = []
    for yr in yrs_in_range:
        m_row = model_df[model_df['Year'] == yr]
        u_row = usgs_df[usgs_df['Year'] == yr]
        out_rows.append({
            'Year': yr,
            'Model_GW_kAF': (
                float(m_row['Model_GW_AF'].iloc[0]) / 1000.0
                if not m_row.empty else np.nan
            ),
            'Model_SW_kAF': (
                float(m_row['Model_SW_AF'].iloc[0]) / 1000.0
                if not m_row.empty else np.nan
            ),
            'Sigma_Model_GW_kAF': (
                sigma_gw[yr] / 1000.0 if yr in sigma_gw else np.nan
            ),
            'Sigma_Model_SW_kAF': (
                sigma_sw[yr] / 1000.0 if yr in sigma_sw else np.nan
            ),
            'USGS_GW_kAF': (
                float(u_row['GW_kAF'].iloc[0])
                if not u_row.empty
                and pd.notna(u_row['GW_kAF'].iloc[0])
                else np.nan
            ),
            'USGS_SW_kAF': (
                float(u_row['SW_kAF'].iloc[0])
                if not u_row.empty
                and pd.notna(u_row['SW_kAF'].iloc[0])
                else np.nan
            ),
            'USGS_Source': (
                u_row['Source'].iloc[0] if not u_row.empty else ''
            ),
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(
        os.path.join(output_dir, 'USGS_AZ_Calibration_Bars.csv'),
        index=False,
    )

    _plot_usgs_calibration_bars(
        out_df,
        column='GW',
        title=(
            'Arizona — Annual Total Groundwater Withdrawal\n'
            '(Model bars with 95 % CI vs USGS anchors)'
        ),
        out_path=os.path.join(output_dir, 'USGS_AZ_Total_GW_Bars.png'),
        bar_color='#3498DB',
        anchor_color='#E74C3C',
    )

    sw_df_plot = out_df[out_df['Year'] >= 1950].reset_index(drop=True)
    if not sw_df_plot.empty:
        _plot_usgs_calibration_bars(
            sw_df_plot,
            column='SW',
            title=(
                'Arizona — Annual Total Surface-Water Withdrawal\n'
                '(Model bars with 95 % CI vs USGS anchors)'
            ),
            out_path=os.path.join(output_dir, 'USGS_AZ_Total_SW_Bars.png'),
            bar_color='#16A085',
            anchor_color='#E74C3C',
        )

    # Per-category bar plots (Total / Irr / NIR + IrrGW / IrrSW / NIGW /
    # NISW).  Same chart format: solid model bars + transparent USGS
    # overlay + Δ% annotation + 95 % CI caps where σ rasters exist.
    # USGS anchor data come from AZ_Annual_WU_Summary.csv (only
    # populated when ``usgs_summary_csv`` is passed); pre-1950 SW
    # categories will simply have no anchors (USGS untracked SW).
    if usgs_summary_csv and os.path.isfile(usgs_summary_csv):
        anchors_full = _load_anchors_from_summary(usgs_summary_csv)
        category_specs = [
            ('Total', 'Total_Predicted', 'Total_kAF',
             'Annual Total Withdrawal', '#7F8C8D'),
            ('Irrigation', 'Irrigation', 'Irr_kAF',
             'Annual Irrigation Withdrawal', '#F39C12'),
            ('Non_Irrigation', 'Non_Irrigation', 'NIR_kAF',
             'Annual Non-Irrigation Withdrawal', '#9B59B6'),
            ('Irrigation_GW', 'Irrigation_GW', 'IrrGW_kAF',
             'Annual Irrigation Groundwater', '#3498DB'),
            ('Irrigation_SW', 'Irrigation_SW', 'IrrSW_kAF',
             'Annual Irrigation Surface Water', '#16A085'),
            ('Non_Irrigation_GW', 'Non_Irrigation_GW', 'NIGW_kAF',
             'Annual Non-Irrigation Groundwater', '#3498DB'),
            ('Non_Irrigation_SW', 'Non_Irrigation_SW', 'NISW_kAF',
             'Annual Non-Irrigation Surface Water', '#16A085'),
        ]
        for cat_label, ml_name, anchor_col, subtitle, color in category_specs:
            ml_csv = os.path.join(annual_summaries_dir, f'{ml_name}.csv')
            if not os.path.isfile(ml_csv):
                continue
            cat_model = (
                pd.read_csv(ml_csv)[['Year', 'Volume_AF']]
                .rename(columns={'Volume_AF': 'Model_AF'})
            )
            cat_yrs = [
                int(y) for y in cat_model['Year'].astype(int).values
                if start_year <= int(y) <= end_year
            ]
            cat_sigma_dict = _load_az_sigma_total_for_category(
                sigma_rasters_dir, ml_name, cat_yrs, pixel_area_m2,
            )
            yrs_arr = np.array(cat_yrs)
            model_arr = np.array([
                float(cat_model.loc[cat_model['Year'] == y, 'Model_AF']
                      .iloc[0]) / 1000.0 / 1000.0  # AF → kAF → MAF
                for y in cat_yrs
            ])
            sigma_arr = np.array([
                cat_sigma_dict.get(y, np.nan) / 1000.0 / 1000.0
                if cat_sigma_dict.get(y) is not None else np.nan
                for y in cat_yrs
            ])
            usgs_arr = np.full(len(cat_yrs), np.nan)
            for i, y in enumerate(cat_yrs):
                row = anchors_full[anchors_full['Year'] == y]
                if not row.empty and pd.notna(row[anchor_col].iloc[0]):
                    usgs_arr[i] = float(row[anchor_col].iloc[0]) / 1000.0
            _plot_calibration_bars(
                years=yrs_arr,
                model=model_arr,
                sigma=sigma_arr,
                usgs=usgs_arr,
                title=(
                    f'Arizona — {subtitle}\n'
                    '(Model bars with 95 % CI vs USGS anchors)'
                ),
                out_path=os.path.join(
                    output_dir, f'USGS_AZ_{cat_label}_Bars.png',
                ),
                bar_color=color,
                anchor_color='#E74C3C',
                ylabel=f'{subtitle.replace("Annual ", "")} (MAF)',
            )

    logger.info(
        'USGS calibration overview written to %s', output_dir,
    )
    return out_df


def run_usgs_az_category_comparison(
        annual_summaries_dir: str,
        usgs_summary_csv: str,
        output_dir: str,
        gw_only_cutoff: int = 1950,
) -> pd.DataFrame:
    """Per-category AZ-wide ML vs USGS comparison at anchor years.

    Loads ``AZ_Annual_WU_Summary.csv`` (the curated USGS/ADWR rollup
    with per-category MAF columns) and the model's
    ``Annual_Summaries/`` CSVs, then for each USGS-anchor year produces:

      - Pre-``gw_only_cutoff`` (default 1950): GW Δ%  =
        (ML_TotGW − USGS_GW) / USGS_GW × 100.  USGS pre-1945 reports
        only GW (no SW data — not zero, just untracked), so a
        share-of-Total comparison is not meaningful; the volume-%
        diff against USGS_GW is the right calibration check.
      - ``gw_only_cutoff`` onward: Δ pp for every category share
        (Irr%, NIR%, IrrGW, IrrSW, NIGW, NISW, GW%, SW%) computed as
        ``ML_share − USGS_share``.

    Logs the per-year table and the MAE summary.  Writes
    ``USGS_AZ_Category_Comparison.csv`` (per-year diffs) and
    ``USGS_AZ_Category_MAE.csv`` (per-category MAE / MAPE) to
    *output_dir*.

    Args:
        annual_summaries_dir: Directory containing the model's per-
            category ``Annual_Summaries/*.csv``.
        usgs_summary_csv: Path to ``AZ_Annual_WU_Summary.csv``.
        output_dir: Output directory for the comparison CSVs.
        gw_only_cutoff: Year boundary between pre-1945 GW-only USGS
            era and the full-category USGS era (default 1950).

    Returns:
        Per-year diff DataFrame (also written to CSV).
    """
    makedirs(output_dir)
    if not os.path.isfile(usgs_summary_csv):
        logger.warning(
            'USGS summary CSV not found at %s; skipping category '
            'comparison',
            usgs_summary_csv,
        )
        return pd.DataFrame()

    usgs_df = pd.read_csv(usgs_summary_csv)
    # Keep all USGS + ADWR rows.  USGS dominates anchors at 1915/1950+,
    # ADWR provides additional anchors at 1957, 1980, 1990, 2000, 2010,
    # 2014, 2017 (Total only) and 2019 (shares only).  Multiple rows can
    # exist for a year (e.g. USGS + ADWR 1990) — both are scored.
    usgs_df = usgs_df[
        usgs_df['Source'].astype(str).str.match(r'^(USGS|ADWR)', na=False)
    ].copy()

    ml_files = {
        'Total_Predicted': 'Total_Predicted.csv',
        'Irrigation': 'Irrigation.csv',
        'Non_Irrigation': 'Non_Irrigation.csv',
        'Irrigation_GW': 'Irrigation_GW.csv',
        'Irrigation_SW': 'Irrigation_SW.csv',
        'Non_Irrigation_GW': 'Non_Irrigation_GW.csv',
        'Non_Irrigation_SW': 'Non_Irrigation_SW.csv',
        'Total_GW': 'Total_GW.csv',
        'Total_SW': 'Total_SW.csv',
    }
    ml_data = {}
    for key, fname in ml_files.items():
        path = os.path.join(annual_summaries_dir, fname)
        if not os.path.isfile(path):
            logger.warning(
                'Model CSV %s missing — skipping category comparison',
                path,
            )
            return pd.DataFrame()
        ml_data[key] = (
            pd.read_csv(path)
            .set_index('Year')['Volume_AF']
            / 1000.0  # → kAF
        )

    keymap = {
        'Irr%':  'Total_Irr_MAF',
        'NIR%':  'Total_NonIrr_MAF',
        'IrrGW': 'Irr_GW_MAF',
        'IrrSW': 'Irr_SW_MAF',
        'NIGW':  'NonIrr_GW_MAF',
        'NISW':  'NonIrr_SW_MAF',
        'GW%':   'Total_GW_MAF',
        'SW%':   'Total_SW_MAF',
    }
    ml_keymap = {
        'Irr%':  'Irrigation',
        'NIR%':  'Non_Irrigation',
        'IrrGW': 'Irrigation_GW',
        'IrrSW': 'Irrigation_SW',
        'NIGW':  'Non_Irrigation_GW',
        'NISW':  'Non_Irrigation_SW',
        'GW%':   'Total_GW',
        'SW%':   'Total_SW',
    }
    share_cols = list(keymap.keys())

    pct_cols = [
        'GW_pct', 'SW_pct', 'Reclaimed_pct', 'Irr_pct', 'NonIrr_pct',
    ]
    rows = []
    pre_gw_pcts: list[float] = []
    total_pcts: list[float] = []
    share_diffs: dict[str, list[float]] = {k: [] for k in share_cols}
    # Iterate over each USGS / ADWR row (multiple per year possible).
    for _, u_row in usgs_df.sort_values(['Year', 'Source']).iterrows():
        yr = int(u_row['Year'])
        if yr not in ml_data['Total_Predicted'].index:
            continue
        record: dict = {'Year': yr, 'Source': u_row['Source']}
        mt = float(ml_data['Total_Predicted'].loc[yr])
        mg = float(ml_data['Total_GW'].loc[yr])
        # Total volume diff (when source has Total_MAF)
        ut = (
            float(u_row['Total_MAF']) * 1000.0
            if pd.notna(u_row.get('Total_MAF')) else np.nan
        )
        if np.isfinite(ut) and ut > 0:
            tot_d = 100.0 * (mt - ut) / ut
            record['Total_VolDiff_pct'] = tot_d
            total_pcts.append(abs(tot_d))
        else:
            record['Total_VolDiff_pct'] = np.nan
        # Pre-1950 GW volume diff (USGS GW-only era; SW untracked, not zero)
        if yr < gw_only_cutoff:
            ug = (
                float(u_row['Total_GW_MAF']) * 1000.0
                if pd.notna(u_row.get('Total_GW_MAF')) else np.nan
            )
            if np.isfinite(ug) and ug > 0:
                d = 100.0 * (mg - ug) / ug
                record['GW_VolDiff_pct'] = d
                pre_gw_pcts.append(abs(d))
            else:
                record['GW_VolDiff_pct'] = np.nan
        else:
            record['GW_VolDiff_pct'] = np.nan
        # Per-category share pp diffs (>= cutoff): two paths
        #   (a) MAF columns populated → compute USGS share = USGS_MAF/Total
        #   (b) ADWR _pct columns populated (e.g. ADWR 2019 shares) →
        #       use directly as the USGS share
        if yr >= gw_only_cutoff:
            for k in share_cols:
                p_us = np.nan
                if np.isfinite(ut) and ut > 0 and pd.notna(u_row.get(keymap[k])):
                    p_us = 100.0 * (float(u_row[keymap[k]]) * 1000.0) / ut
                else:
                    # ADWR-style _pct fallback for the four broad shares
                    pct_lookup = {
                        'GW%': 'GW_pct', 'SW%': 'SW_pct',
                        'Irr%': 'Irr_pct', 'NIR%': 'NonIrr_pct',
                    }
                    pct_col = pct_lookup.get(k)
                    if pct_col and pd.notna(u_row.get(pct_col)):
                        p_us = float(u_row[pct_col])
                if pd.notna(p_us):
                    p_ml = 100.0 * float(ml_data[ml_keymap[k]].loc[yr]) / mt
                    d = p_ml - p_us
                    record[f'{k}_pp_diff'] = d
                    share_diffs[k].append(abs(d))
                else:
                    record[f'{k}_pp_diff'] = np.nan
        else:
            for k in share_cols:
                record[f'{k}_pp_diff'] = np.nan
        rows.append(record)
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(
        os.path.join(output_dir, 'USGS_AZ_Category_Comparison.csv'),
        index=False,
    )

    mae_rows = []
    if total_pcts:
        mae_rows.append({
            'Metric': 'Total MAPE (all anchors, volume %)',
            'N': len(total_pcts),
            'Value': sum(total_pcts) / len(total_pcts),
            'Unit': '%',
        })
    if pre_gw_pcts:
        mae_rows.append({
            'Metric': f'GW MAPE (pre-{gw_only_cutoff}, volume %)',
            'N': len(pre_gw_pcts),
            'Value': sum(pre_gw_pcts) / len(pre_gw_pcts),
            'Unit': '%',
        })
    for k in share_cols:
        if share_diffs[k]:
            mae_rows.append({
                'Metric': f'{k} MAE (>= {gw_only_cutoff}, share)',
                'N': len(share_diffs[k]),
                'Value': sum(share_diffs[k]) / len(share_diffs[k]),
                'Unit': 'pp',
            })
    mae_df = pd.DataFrame(mae_rows)
    mae_df.to_csv(
        os.path.join(output_dir, 'USGS_AZ_Category_MAE.csv'),
        index=False,
    )

    # Log the per-year diff table + MAE summary
    logger.info(
        'USGS / ADWR category comparison: Total/GW Δ%% (volume) and '
        'category Δ pp (share of Total) at every anchor row.'
    )
    header = (
        f'{"Yr":>4} {"Src":>4} | {"Tot Δ%":>6} {"GW Δ%":>6} | '
        f'{"Irr%":>6} {"NIR%":>6} | '
        f'{"IrrGW":>6} {"IrrSW":>6} {"NIGW":>6} {"NISW":>6} | '
        f'{"GW%":>6} {"SW%":>6}'
    )
    logger.info(header)
    logger.info('-' * len(header))

    def fmt(v):
        return f'{v:>+6.1f}' if pd.notna(v) else '   --'

    for _, r in diff_df.iterrows():
        src_short = (str(r.Source).split()[0])[:4]
        logger.info(
            f'{int(r.Year):>4} {src_short:>4} | '
            f'{fmt(r.Total_VolDiff_pct)} {fmt(r.GW_VolDiff_pct)} | '
            f'{fmt(r["Irr%_pp_diff"])} {fmt(r["NIR%_pp_diff"])} | '
            f'{fmt(r.IrrGW_pp_diff)} {fmt(r.IrrSW_pp_diff)} '
            f'{fmt(r.NIGW_pp_diff)} {fmt(r.NISW_pp_diff)} | '
            f'{fmt(r["GW%_pp_diff"])} {fmt(r["SW%_pp_diff"])}'
        )
    for _, r in mae_df.iterrows():
        logger.info('  %s = %.2f %s (n=%d)', r.Metric, r.Value, r.Unit, r.N)
    logger.info(
        'Wrote USGS_AZ_Category_Comparison.csv + USGS_AZ_Category_MAE.csv to %s',
        output_dir,
    )
    return diff_df
