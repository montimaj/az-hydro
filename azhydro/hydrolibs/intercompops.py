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
       (``GW_irr_YYYY.tif`` / ``SW_irr_YYYY.tif``, 1980-2018, in metres).

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
Reitz metadata: HistoricalET_metadata.xml — irrigation in metres/year.
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
    plot_intercomp_scatter,
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
M_TO_MM = 1000.0                        # metres → millimetres
MM_TO_FT = 1.0 / 304.8                 # millimetres → feet

# NHM sentinel values (no irrigated area or null ET)
NHM_SENTINEL = {999, 888}


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
        pixel_area_m2 (float): Pixel area in square metres.
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
    ]:
        csv_path = os.path.join(nhm_dir, csv_name)
        logger.info(f'Reading NHM {category}: {csv_path}')

        # Wide-format CSV: Year, Month, <HUC12_code_1>, ..., <HUC12_code_N>
        df = pd.read_csv(csv_path, dtype={'Year': int, 'Month': int})
        huc_cols = [c for c in df.columns if c not in ('Year', 'Month')]

        # Load AZ HUC12 polygons and determine which HUC12s are in AZ
        huc_gdf = gpd.read_file(huc12_geojson)
        az_huc12_set = set(huc_gdf['huc12'].astype(str).values)

        # Filter CSV columns to AZ HUC12s only
        az_cols = [c for c in huc_cols if c in az_huc12_set]
        logger.info(f'  {len(az_cols)} AZ HUC12 regions found in NHM data')

        if not az_cols:
            logger.warning(f'  No AZ HUC12 matches for {category}')
            result[category] = {b: 0.0 for b in basin_gdf[basin_col]}
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
        overlay = gpd.overlay(
            huc_reproj[['huc12', 'area_m2', 'geometry']],
            basin_reproj[[basin_col, 'geometry']],
            how='intersection',
        )
        overlay['overlap_area'] = overlay.geometry.area
        overlay['area_frac'] = overlay['overlap_area'] / overlay['area_m2']
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

        result[category] = {'mean': basin_vols, 'yearly': yearly_vols}

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

    The Reitz rasters are in **metres/year** at ~800 m geographic resolution.

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
            # Convert metres → mm
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
            result[category] = {b: 0.0 for b in basin_gdf[basin_col]}
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

        # Per-year basin volumes from reprojected rasters (on-disk units are metres)
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
            result[cat] = {b: 0.0 for b in basin_gdf[basin_col]}
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
def _plot_spatial_diff_maps(
    mean_raster_paths: dict[str, dict[str, str]],
    ref_raster: str,
    output_dir: str,
) -> None:
    """Create spatial maps of pairwise mean-annual depth differences.

    For each GW/SW category, produces three difference maps:
        ML − NHM, ML − Reitz, NHM − Reitz
    using a diverging colour map centred on zero.

    Args:
        mean_raster_paths (dict): ``{source: {cat: path}}`` where source ∈ {ML, NHM, Reitz} and
            cat ∈ {GW, SW}.  Paths to mean-annual depth rasters (mm).
        ref_raster (str): Reference raster for extent / CRS.
        output_dir (str): Directory for saved plots.

    Returns:
        None
    """
    makedirs(output_dir)

    with rio.open(ref_raster) as src:
        extent = [
            src.bounds.left, src.bounds.right,
            src.bounds.bottom, src.bounds.top,
        ]

    pairs = [('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')]

    for cat in ('GW', 'SW'):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        fig.suptitle(f'Irrigation {cat} — Mean-Annual Depth Difference (mm)',
                     fontsize=14, fontweight='bold')

        for col_i, (src_a, src_b) in enumerate(pairs):
            ax = axes[col_i]
            path_a = mean_raster_paths.get(src_a, {}).get(cat)
            path_b = mean_raster_paths.get(src_b, {}).get(cat)

            if path_a is None or path_b is None or \
               not os.path.isfile(path_a) or not os.path.isfile(path_b):
                ax.set_title(f'{src_a} − {src_b}  (data unavailable)')
                ax.axis('off')
                continue

            arr_a = read_raster_as_arr(path_a, get_file=False).astype(np.float64)
            arr_b = read_raster_as_arr(path_b, get_file=False).astype(np.float64)
            arr_a[np.isnan(arr_a)] = 0.0
            arr_b[np.isnan(arr_b)] = 0.0

            diff = arr_a - arr_b
            # Mask where both are zero (outside domain)
            mask = (arr_a == 0) & (arr_b == 0)
            diff_masked = np.ma.masked_where(mask, diff)

            vmax = max(abs(np.nanmin(diff_masked)), abs(np.nanmax(diff_masked)), 1e-6)
            im = ax.imshow(
                diff_masked, extent=extent, origin='upper',
                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                interpolation='nearest',
            )
            ax.set_title(f'{src_a} − {src_b}', fontweight='bold')
            ax.set_xlabel('Easting (m)')
            if col_i == 0:
                ax.set_ylabel('Northing (m)')
            fig.colorbar(im, ax=ax, shrink=0.8, label='Δ Depth (mm)')

        out_path = os.path.join(output_dir, f'Spatial_Diff_{cat}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
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

    # AZ HUC12 polygons
    huc_gdf = gpd.read_file(huc12_geojson)
    az_huc12_set = set(huc_gdf['huc12'].astype(str).values)
    az_cols = [c for c in huc_cols if c in az_huc12_set]
    logger.info(f'  {len(az_cols)} AZ HUC12 regions found')

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

    # Basin volumes
    basin_vols = _raster_basin_volumes(
        out_tif, basin_reproj, basin_col, pixel_area_m2, depth_unit='mm',
    )

    # Per-year basin volumes via spatial overlay
    yearly_vols = {}
    overlay = gpd.overlay(
        huc_reproj[['huc12', 'area_m2', 'geometry']],
        basin_reproj[[basin_col, 'geometry']],
        how='intersection',
    )
    overlay['overlap_area'] = overlay.geometry.area
    overlay['area_frac'] = overlay['overlap_area'] / overlay['area_m2']
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

    return {'mean': basin_vols, 'yearly': yearly_vols}


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
    overlay = gpd.overlay(
        huc_reproj[['huc12', 'area_m2', 'geometry']],
        basin_reproj[[basin_col, 'geometry']],
        how='intersection',
    )
    overlay['overlap_area'] = overlay.geometry.area
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

    return {
        'per_year': yearly,
        'mean': result['mean'],
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
    nhm_year_range: tuple[int, int] = (1980, 2020),
    reitz_year_range: tuple[int, int] = (1980, 2020),
) -> pd.DataFrame:
    """
    Run the full three-way intercomparison for Irrigation GW and Irrigation
    SW withdrawals across Arizona groundwater basins.

    Year ranges default to 1980-2020 to cover the full span of all three
    datasets (ML: 2002-2020, NHM: 2000-2020, Reitz: 1980-2018).  Years
    without data for a given USGS dataset will appear as blank/zero.

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
            (ml_pred_dir, 'Predicted_GW_{yr}_mm.tif'),
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

    for cat in ('GW', 'SW'):
        logger.info(f'--- Irrigation {cat} metrics ---')
        pairs = [
            ('ML', 'NHM', ml_vols[cat]['mean'], nhm_vols[cat]['mean']),
            ('ML', 'Reitz', ml_vols[cat]['mean'], reitz_vols[cat]['mean']),
            ('NHM', 'Reitz', nhm_vols[cat]['mean'], reitz_vols[cat]['mean']),
        ]
        for label_a, label_b, data_a, data_b in pairs:
            m = _compute_metrics(
                basin_names, data_a, data_b, label_a, label_b,
                basin_areas_m2=basin_areas_m2,
            )
            m['Category'] = f'Irrigation_{cat}'
            all_metrics.append(m)
            logger.info(
                f'  {m["Pair"]}: RMSD={m["RMSD_AF"]:.2f} AF '
                f'({m["RMSD_m3"]:.2f} m³), '
                f'MAD={m["MAD_AF"]:.2f} AF, PctDiff={m["Pct_Diff"]:.2f}%'
            )

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
    for cat in ('GW', 'SW'):
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
        plot_intercomp_taylor(
            all_sources,
            pairs=[('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')],
            categories=['GW', 'SW'],
            basin_names=basin_names,
            output_dir=temporal_plot_dir,
            pair_colors=pair_colors_4a,
            title_prefix='Irrigation ',
        )
        plot_temporal_r_vs_nse(
            temporal_basin_df, temporal_plot_dir,
            pair_colors=pair_colors_4a,
        )

    # ── 5. Per-basin comparison table (mm, m³, AF) ──────────────────────
    rows = []
    for cat in ('GW', 'SW'):
        for basin in basin_names:
            ml_af = ml_vols[cat]['mean'].get(basin, 0.0)
            nhm_af = nhm_vols[cat]['mean'].get(basin, 0.0)
            reitz_af = reitz_vols[cat]['mean'].get(basin, 0.0)
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
            })
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'per_basin_volumes.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(f'Per-basin volumes saved to {basin_csv}')

    # ── 6. Time series CSV ───────────────────────────────────────────────
    ts_rows = []
    for cat in ('GW', 'SW'):
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
        all_sources, categories=['GW', 'SW'],
        basin_names=basin_names, basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_ts_colors, markers=_ts_markers,
        title_prefix='Irrigation ', file_prefix='TS',
    )

    # ── 8. Scatter plots ────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    for cat in ('GW', 'SW'):
        scatter_pairs = [
            (sa, sb, all_sources[sa][cat]['mean'], all_sources[sb][cat]['mean'])
            for sa, sb in [('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')]
        ]
        plot_intercomp_scatter(
            scatter_pairs, basin_names, basin_areas_m2, scatter_dir,
            title=f'Irrigation {cat} — Per-Basin Scatter Comparison',
            filename=f'Scatter_{cat}.png',
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
    _plot_spatial_diff_maps(mean_raster_paths, ref_raster, diff_dir)

    # ── 10. Summary print ────────────────────────────────────────────────
    logger.info('\n' + '='*60)
    logger.info('Intercomparison Summary')
    logger.info('='*60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
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

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('CU Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
    logger.info(f'\nML year range: {ml_year_range}')
    logger.info(f'NHM year range: {nhm_year_range}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# Effective Precipitation Intercomparison (ML Peff vs ML Peff PCML vs NHM)


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

        1. **ML Peff** — SCS formula-based (predictor band 4 × irr_fraction)
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

    # ── 1. ML Peff (SCS) ────────────────────────────────────────────────
    logger.info('--- Loading ML Peff (SCS formula) ---')
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
    all_metrics = []
    pairs = [
        ('ML_Peff', 'NHM_Peff'),
        ('ML_Peff_PCML', 'NHM_Peff'),
        ('ML_Peff', 'ML_Peff_PCML'),
    ]
    for label_a, label_b in pairs:
        m = _compute_metrics(
            basin_names,
            all_sources[label_a]['mean'],
            all_sources[label_b]['mean'],
            label_a, label_b,
            basin_areas_m2=basin_areas_m2,
        )
        m['Category'] = 'Effective_Precipitation'
        all_metrics.append(m)
        logger.info(
            f'  {m["Pair"]}: RMSD={m["RMSD_AF"]:.2f} AF '
            f'({m["RMSD_m3"]:.2f} m³), '
            f'MAD={m["MAD_AF"]:.2f} AF, PctDiff={m["Pct_Diff"]:.2f}%'
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, 'peff_intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'Peff metrics saved to {metrics_csv}')

    # ── 5. Per-basin comparison table ────────────────────────────────────
    rows = []
    for basin in basin_names:
        area = basin_areas_m2.get(basin, 1.0)
        row = {'Basin': basin}
        for src_key in ('ML_Peff', 'ML_Peff_PCML', 'NHM_Peff'):
            af_val = all_sources[src_key]['mean'].get(basin, 0.0)
            row[f'{src_key}_mm'] = round(af_val * af_to_m3 / area * M_TO_MM, 4)
            row[f'{src_key}_AF'] = round(af_val, 2)
        rows.append(row)
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'peff_per_basin.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(f'Per-basin Peff saved to {basin_csv}')

    # ── 6. Time series CSV ───────────────────────────────────────────────
    ts_rows = []
    for src_key, src_data in all_sources.items():
        yearly = src_data.get('yearly', {})
        for year in sorted(yearly.keys()):
            for basin in basin_names:
                af_val = yearly[year].get(basin, 0.0)
                area = basin_areas_m2.get(basin, 1.0)
                ts_rows.append({
                    'Source': src_key,
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
        labels={k: k.replace('_', ' ') for k in all_sources},
        title_prefix='Effective Precipitation — ', file_prefix='TS_Peff',
    )

    # ── 8. Scatter plots ─────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    source_keys = list(all_sources.keys())
    peff_scatter_pairs = [
        (source_keys[i].replace('_', ' '), source_keys[j].replace('_', ' '),
         all_sources[source_keys[i]]['mean'], all_sources[source_keys[j]]['mean'])
        for i in range(len(source_keys))
        for j in range(i + 1, len(source_keys))
    ]
    plot_intercomp_scatter(
        peff_scatter_pairs, basin_names, basin_areas_m2, scatter_dir,
        title='Effective Precipitation — Per-Basin Scatter',
        filename='Scatter_Peff.png',
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('Peff Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
    logger.info(f'\nML Peff year range: {ml_year_range}')
    logger.info(f'ML Peff PCML year range: {ml_pcml_year_range}')
    logger.info(f'NHM Peff year range: {nhm_year_range}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# CAP/SRP Surface Water Validation
# ═════════════════════════════════════════════════════════════════════════════

# Mapping from CAP Excel AMA names to ADWR GW basin names
_CAP_AMA_TO_BASIN = {
    'Phoenix AMA':     'PHOENIX AMA',
    'Tucson AMA':      'TUCSON AMA',
    'Pinal AMA':       'PINAL AMA',
    'Harquahala INA':  'HARQUAHALA INA',
    'Ranegras Plain':  'RANEGRAS PLAIN',
    'Parker':          'PARKER',
}


def _load_cap_srp_annual_sw(
    cap_xlsx: str,
    srp_xlsx: str,
    include_spill_water: bool = False,
) -> dict[str, dict[int, float]]:
    """Load CAP and SRP delivery data and return annual total surface-water
    deliveries (AF) per basin.

    CAP: keeps only rows where ``Recharge Facility`` is null (direct use).
    Rows with ``AMA == 'Multiple'`` or ``NaN`` are excluded because they
    cannot be assigned to a single basin (25 records / ~15,600 AF total;
    16 NaN-AMA records / ~86,300 AF total).

    SRP: keeps rows where ``Parent Water Type == 'SURFACE WATER'``.
    When *include_spill_water* is True, ``SPILL WATER`` records are also
    included as a sensitivity test; spill water ranges from ~19 AF/yr
    (2016) to ~366,000 AF/yr (1993) in Phoenix AMA.

    For Phoenix AMA the two sources are summed.  Both datasets use
    calendar-year columns (CAP ``Year``; SRP ``Water Move Year``).

    Args:
        cap_xlsx (str): Path to CAP delivery Excel file.
        srp_xlsx (str): Path to SRP delivery Excel file.
        include_spill_water (bool): If True, include SRP ``SPILL WATER`` records in addition to
            ``SURFACE WATER``.  Default False (baseline).

    Returns:
        dict[str, dict[int, float]]: ``{basin_name: {year: delivery_AF}}``.
    """
    # ── CAP ──────────────────────────────────────────────────────────────
    cap_df = pd.read_excel(cap_xlsx)
    # Keep only direct-use deliveries (null recharge facility)
    cap_df = cap_df[cap_df['Recharge Facility'].isna()].copy()
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

    # ── SRP ──────────────────────────────────────────────────────────────
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
    srp_annual['Basin'] = 'PHOENIX AMA'

    # ── Merge ────────────────────────────────────────────────────────────
    # Start with CAP totals per basin per year
    basin_year = {}
    for _, row in cap_annual.iterrows():
        basin_year.setdefault(row['Basin'], {})[int(row['Year'])] = float(row['CAP_AF'])

    # Add SRP surface water to Phoenix AMA
    phx = basin_year.setdefault('PHOENIX AMA', {})
    for _, row in srp_annual.iterrows():
        year = int(row['Year'])
        phx[year] = phx.get(year, 0.0) + float(row['SRP_AF'])

    return basin_year


def _load_ml_total_sw_basin_volumes(
    total_sw_dir: str,
    basin_gdf: gpd.GeoDataFrame,
    basin_col: str,
    year_range: tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Aggregate ML ``Total_SW_YYYY_mm.tif`` rasters to annual basin
    volumes (AF).

    Returns:
        dict[str, dict[int, float]]: ``{basin_name: {year: volume_AF}}``.
    """
    start_yr, end_yr = year_range
    ref_raster = None
    for yr in range(start_yr, end_yr + 1):
        candidate = os.path.join(total_sw_dir, f'Total_SW_{yr}_mm.tif')
        if os.path.isfile(candidate):
            ref_raster = candidate
            break
    if ref_raster is None:
        logger.warning(f'No Total_SW rasters found in {total_sw_dir}')
        return {}

    with rio.open(ref_raster) as src:
        ref_crs = src.crs
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    result = {}  # basin → {year → AF}
    for yr in range(start_yr, end_yr + 1):
        raster_path = os.path.join(total_sw_dir, f'Total_SW_{yr}_mm.tif')
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

        # Pearson correlation
        if len(ml_vals) > 1 and np.std(ml_vals) > 0 and np.std(obs_vals) > 0:
            pearson_r = float(np.corrcoef(ml_vals, obs_vals)[0, 1])
        else:
            pearson_r = np.nan

        ml_m3 = ml_vals * af_to_m3
        obs_m3 = obs_vals * af_to_m3
        diff_m3 = diff * af_to_m3

        row = {
            'Basin': basin,
            'N_Years': len(common_years),
            'Year_Range': f'{common_years[0]}-{common_years[-1]}',
            'Pearson_R': round(pearson_r, 4),
            'RMSD_AF': round(rmsd_af, 2),
            'RMSD_m3': round(float(np.sqrt(np.mean(diff_m3 ** 2))), 2),
            'MAD_AF': round(mad_af, 2),
            'MAD_m3': round(float(np.mean(np.abs(diff_m3))), 2),
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
            row['RMSD_mm'] = round(float(np.sqrt(np.mean(diff_mm ** 2))), 4)
            row['MAD_mm'] = round(float(np.mean(np.abs(diff_mm))), 4)
            row['Mean_ML_mm'] = round(float(np.mean(ml_mm)), 4)
            row['Mean_Obs_mm'] = round(float(np.mean(obs_mm)), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def run_cap_srp_validation(
    cap_xlsx: str,
    srp_xlsx: str,
    total_sw_dir: str,
    basin_shp: str,
    basin_col: str,
    output_dir: str,
    year_range: tuple[int, int] = (1985, 2023),
) -> pd.DataFrame:
    """
    Validate ML Total_SW predictions against observed CAP and SRP
    surface-water delivery records.

    CAP deliveries are filtered to exclude recharge-facility records (keeping
    only direct-use deliveries).  SRP deliveries are filtered to ``Parent
    Water Type == 'SURFACE WATER'`` only.  For Phoenix AMA the two sources
    are summed; all other AMAs use CAP data only.

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
    logger.info('CAP/SRP Total Surface Water Validation')
    logger.info('=' * 60)

    # ── Load basin polygons ──────────────────────────────────────────────
    basin_gdf = gpd.read_file(basin_shp)
    logger.info(f'Loaded {len(basin_gdf)} basins from {basin_shp}')

    # ── Load observed CAP + SRP deliveries ───────────────────────────────
    logger.info('Loading CAP/SRP delivery data...')
    obs_basin_yearly = _load_cap_srp_annual_sw(cap_xlsx, srp_xlsx)
    obs_spill_basin_yearly = _load_cap_srp_annual_sw(
        cap_xlsx, srp_xlsx, include_spill_water=True,
    )
    obs_basins = sorted(obs_basin_yearly.keys())
    logger.info(f'  Basins with observed SW data: {obs_basins}')
    for b in obs_basins:
        yrs = sorted(obs_basin_yearly[b].keys())
        logger.info(f'    {b}: {yrs[0]}-{yrs[-1]} ({len(yrs)} years)')

    # ── Load ML Total_SW rasters → basin volumes ────────────────────────
    logger.info('Loading ML Total_SW rasters...')
    ml_basin_yearly = _load_ml_total_sw_basin_volumes(
        total_sw_dir, basin_gdf, basin_col, year_range,
    )
    if not ml_basin_yearly:
        logger.warning('No ML Total_SW rasters found; skipping validation.')
        return pd.DataFrame()

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

    # ── Compute statistics ───────────────────────────────────────────────
    logger.info('Computing per-basin statistics...')
    metrics_df = _compute_cap_srp_metrics(
        ml_basin_yearly, obs_basin_yearly, basin_areas_m2,
    )
    metrics_csv = os.path.join(output_dir, 'cap_srp_sw_validation_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'Metrics saved to {metrics_csv}')

    # ── Time series CSV ──────────────────────────────────────────────────
    af_to_m3 = 1.0 / M3_TO_AF
    ts_rows = []
    for basin in obs_basins:
        all_years = sorted(
            set(ml_basin_yearly.get(basin, {}).keys())
            | set(obs_basin_yearly.get(basin, {}).keys())
        )
        for yr in all_years:
            ml_af = ml_basin_yearly.get(basin, {}).get(yr, np.nan)
            obs_af = obs_basin_yearly.get(basin, {}).get(yr, np.nan)
            area = basin_areas_m2.get(basin, 1.0)
            ts_rows.append({
                'Basin': basin,
                'Year': yr,
                'ML_Total_SW_AF': round(ml_af, 2) if np.isfinite(ml_af) else np.nan,
                'CAP_SRP_AF': round(obs_af, 2) if np.isfinite(obs_af) else np.nan,
                'ML_Total_SW_m3': round(ml_af * af_to_m3, 2) if np.isfinite(ml_af) else np.nan,
                'CAP_SRP_m3': round(obs_af * af_to_m3, 2) if np.isfinite(obs_af) else np.nan,
                'ML_Total_SW_mm': round(ml_af * af_to_m3 / area * M_TO_MM, 4) if np.isfinite(ml_af) and area > 0 else np.nan,
                'CAP_SRP_mm': round(obs_af * af_to_m3 / area * M_TO_MM, 4) if np.isfinite(obs_af) and area > 0 else np.nan,
            })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'cap_srp_sw_time_series.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'Time series saved to {ts_csv}')

    # ── Time series plots ────────────────────────────────────────────────
    _cap_colors = {
        'ML': '#2C3E50', 'CAP_SRP': '#E74C3C', 'CAP_SRP_spill': '#3498DB',
    }
    _cap_markers = {'ML': 'o', 'CAP_SRP': 's', 'CAP_SRP_spill': '^'}
    _cap_labels = {
        'ML': 'ML (Total SW)', 'CAP_SRP': 'CAP + SRP',
        'CAP_SRP_spill': 'CAP + SRP (+ Spill)',
    }
    cap_ts_sources = {
        'ML': {'SW': {'yearly': ml_basin_yearly}},
        'CAP_SRP': {'SW': {'yearly': obs_basin_yearly}},
    }
    if obs_spill_basin_yearly:
        cap_ts_sources['CAP_SRP_spill'] = {
            'SW': {'yearly': obs_spill_basin_yearly},
        }
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    plot_intercomp_time_series(
        cap_ts_sources, categories=['SW'],
        basin_names=sorted(obs_basin_yearly.keys()),
        basin_areas_m2=basin_areas_m2,
        output_dir=plot_dir,
        colors=_cap_colors, markers=_cap_markers, labels=_cap_labels,
        title_prefix='Total Surface Water — ', file_prefix='TS_Total_SW',
    )

    # ── Scatter plot ─────────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    # Build per-basin-year scatter (mean values from yearly overlap)
    obs_basins = sorted(obs_basin_yearly.keys())
    ml_mean_vals, obs_mean_vals = {}, {}
    for basin in obs_basins:
        common_years = sorted(
            set(ml_basin_yearly.get(basin, {}).keys())
            & set(obs_basin_yearly[basin].keys())
        )
        if common_years:
            ml_mean_vals[basin] = float(np.mean([
                ml_basin_yearly[basin][yr] for yr in common_years
            ]))
            obs_mean_vals[basin] = float(np.mean([
                obs_basin_yearly[basin][yr] for yr in common_years
            ]))
    plot_intercomp_scatter(
        [('Observed CAP + SRP', 'ML Total SW', obs_mean_vals, ml_mean_vals)],
        list(ml_mean_vals.keys()), basin_areas_m2, scatter_dir,
        title='ML Total SW vs CAP + SRP — Per Basin',
        filename='Scatter_ML_vs_CAP_SRP.png',
    )

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
) -> pd.DataFrame:
    """Load a USGS PS HUC12 CSV, filter to AZ HUC12s, aggregate monthly
    to annual totals.

    The raw data is in million gallons per day (Mgal/d).  For each
    HUC12 × year we compute the annual total volume in acre-feet:

        annual_vol_AF = Σ_months (rate_Mgal_d × days_in_month) × Mgal_to_m3 × m3_to_AF

    Returns a long-form DataFrame with columns: ``huc12, year, volume_AF``.
    """
    import calendar

    logger.info(f'Loading PS data: {csv_path}')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str)

    # Identify AZ HUC12 columns
    huc_gdf = gpd.read_file(huc12_geojson)
    az_huc12_set = set(huc_gdf['huc12'].astype(str).values)
    all_huc_cols = [c for c in df.columns if c not in ('Year', 'Month')]
    az_cols = [c for c in all_huc_cols if c in az_huc12_set]
    logger.info(f'  {len(az_cols)}/{len(all_huc_cols)} HUC12s in AZ')

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
    huc_gdf['huc12'] = huc_gdf['huc12'].astype(str)
    huc_reproj = huc_gdf.to_crs(ref_crs)
    huc_reproj['area_m2'] = huc_reproj.geometry.area

    basin_reproj = (
        basin_gdf.to_crs(ref_crs) if basin_gdf.crs != ref_crs else basin_gdf
    )

    # Spatial overlay: HUC12 → basin fractional membership
    overlay = gpd.overlay(
        huc_reproj[['huc12', 'area_m2', 'geometry']],
        basin_reproj[[basin_col, 'geometry']],
        how='intersection',
    )
    overlay['overlap_area'] = overlay.geometry.area
    overlay['area_frac'] = overlay['overlap_area'] / overlay['area_m2']

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
        annual_df = _load_ps_huc12_annual(csv_path, huc12_geojson, year_range)
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

        # Taylor diagram
        taylor_sources = {}
        cat_name_list = []
        for cat_key, cat_name in cat_labels.items():
            taylor_sources.setdefault('ML', {})[cat_name] = ml_vols[cat_key]
            taylor_sources.setdefault('PS', {})[cat_name] = ps_vols[cat_key]
            cat_name_list.append(cat_name)
        plot_intercomp_taylor(
            taylor_sources,
            pairs=[('ML', 'PS')],
            categories=cat_name_list,
            basin_names=basin_names,
            output_dir=temporal_plot_dir,
            pair_colors={'ML vs PS': '#E74C3C'},
        )

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
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    for cat_key, cat_name in cat_labels.items():
        plot_intercomp_scatter(
            [('ML Non-Irrigation', 'USGS Public Supply',
              ml_vols[cat_key]['mean'], ps_vols[cat_key]['mean'])],
            basin_names, basin_areas_m2, scatter_dir,
            title=f'{cat_name} — ML vs USGS Public Supply',
            filename=f'Scatter_{cat_name}.png',
        )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('PS Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')

    return metrics_df
