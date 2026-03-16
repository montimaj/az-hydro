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

References
----------
NHM metadata: IR_metadata.xml — withdrawals in million gallons per day.
Reitz metadata: HistoricalET_metadata.xml — irrigation in metres/year.
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
from shapely.geometry import mapping

from hydrolibs.rasterops import read_raster_as_arr
from hydrolibs.sysops import makedirs

logger = logging.getLogger(__name__)

# ── Unit-conversion constants ────────────────────────────────────────────────
MGAL_D_TO_M3_YR = 3785.41178 * 365.25  # 1 Mgal/d → m³/yr
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

    Parameters
    ----------
    raster_path : str
        Path to a single-band depth raster (mm or m).
    basin_gdf : gpd.GeoDataFrame
        Basin polygons in the same CRS as *raster_path*.
    basin_col : str
        Column in *basin_gdf* identifying basins.
    pixel_area_m2 : float
        Pixel area in square metres.
    depth_unit : str
        ``'mm'`` or ``'m'`` — unit of pixel values.

    Returns
    -------
    dict[str, float]
        ``{basin_name: volume_AF}``.
    """
    depth_to_m = 1.0 / M_TO_MM if depth_unit == 'mm' else 1.0
    volumes = {}
    with rio.open(raster_path) as src:
        for _, row in basin_gdf.iterrows():
            basin_name = row[basin_col]
            geom = [mapping(row.geometry)]
            try:
                clipped, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = clipped[0].astype(np.float64)
                arr[np.isnan(arr)] = 0.0
                arr[arr < 0] = 0.0
                vol_m3 = float(np.nansum(arr)) * depth_to_m * pixel_area_m2
                volumes[basin_name] = vol_m3 * M3_TO_AF
            except Exception:
                logger.debug('Clipping failed for basin %s, setting volume to 0', basin_name)
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

    Returns
    -------
    dict[str, float]
        ``{basin_name: mean_value}``.
    """
    means = {}
    with rio.open(raster_path) as src:
        for _, row in basin_gdf.iterrows():
            basin_name = row[basin_col]
            geom = [mapping(row.geometry)]
            try:
                clipped, _ = rio_mask(src, geom, crop=True, nodata=np.nan)
                arr = clipped[0].astype(np.float64)
                valid = arr[np.isfinite(arr) & (arr > 0)]
                means[basin_name] = float(np.mean(valid)) if valid.size > 0 else np.nan
            except Exception:
                logger.debug('Clipping failed for basin %s, setting mean to NaN', basin_name)
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

    Parameters
    ----------
    nhm_dir : str
        Directory containing the NHM CSV files.
    huc12_geojson : str
        Path to ``AZ_HUC12.geojson``.
    basin_gdf : gpd.GeoDataFrame
        Basin polygons (target CRS = reference raster CRS).
    basin_col : str
        Column in *basin_gdf* naming basins.
    ref_raster : str
        Reference raster for grid/CRS information.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive.
    output_dir : str
        Directory for intermediate rasters.
    predictor_dir : str or None
        Directory with ``Predictor_YYYY.tif`` multi-band rasters containing
        ``annual_irr_fraction`` (band *irr_fraction_band*).  When provided,
        the mean irrigated fraction per HUC12 is used to scale the area.
    irr_fraction_band : int
        1-indexed band number for ``annual_irr_fraction`` in the predictor
        rasters (default 14).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
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
                # Mgal/d × days → Mgal; × 3785.41178 → m³
                annual_vol += vals * ndays * 3785.41178
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
                with rio.open(pred_file) as src:
                    irr_arr = src.read(irr_fraction_band).astype(np.float64)
                    irr_arr[np.isnan(irr_arr)] = 0.0
                    irr_arr = np.clip(irr_arr, 0, 1)
                # Zonal mean per HUC12
                for idx, row in huc_reproj.iterrows():
                    geom = [mapping(row.geometry)]
                    try:
                        with rio.open(pred_file) as src:
                            clipped, _ = rio_mask(src, geom, crop=True,
                                                  all_touched=True,
                                                  indexes=[irr_fraction_band],
                                                  nodata=np.nan)
                        vals = clipped[0].astype(np.float64)
                        vals = vals[~np.isnan(vals)]
                        vals = np.clip(vals, 0, 1)
                        if vals.size > 0:
                            irr_counts[huc_reproj.index.get_loc(idx)] += np.mean(vals)
                    except Exception:
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

    Parameters
    ----------
    reitz_base_dir : str
        Parent directory containing ``Irrigation_groundwater_1980-2018/``
        and ``Irrigation_surfacewater_1980-2018/``.
    ref_raster : str
        Reference ML prediction raster for CRS/grid alignment.
    basin_gdf : gpd.GeoDataFrame
        Basin polygons.
    basin_col : str
        Basin name column.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive.
    output_dir : str
        Directory for reprojected/intermediate rasters.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
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

        # Per-year basin volumes from reprojected rasters
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

    Parameters
    ----------
    pred_raster_dir : str
        Directory with ``pred_YYYY.tif`` total pumping rasters (mm).
    basin_gdf : gpd.GeoDataFrame
        Basin polygons.
    basin_col : str
        Basin name column.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive.
    irr_gw_dir : str or None
        Directory with ``Irrigation_GW_YYYY_mm.tif`` rasters.
    irr_sw_dir : str or None
        Directory with ``Irrigation_SW_YYYY_mm.tif`` rasters.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{'GW': {basin: AF}, 'SW': {basin: AF}}``.
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

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. Time series plotting
# ═════════════════════════════════════════════════════════════════════════════
_TS_COLORS = {'ML': '#2C3E50', 'NHM': '#27AE60', 'Reitz': '#E67E22'}
_TS_MARKERS = {'ML': 'o', 'NHM': 's', 'Reitz': '^'}


def _plot_basin_time_series(
    all_sources: dict,
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
) -> None:
    """Create per-basin time series plots with all three datasets.

    Each basin gets one figure with two columns (Irrigation GW, Irrigation SW)
    and two rows:
        Row 0: Depth — mm (left axis) / ft (right twin axis)
        Row 1: Volume — m³ (left axis) / AF (right twin axis)
    An additional 'AZ Total' figure sums across all basins.

    Parameters
    ----------
    all_sources : dict
        ``{'ML': ml_vols, 'NHM': nhm_vols, 'Reitz': reitz_vols}`` where
        each ``*_vols`` has ``{cat: {'mean': ..., 'yearly': ...}}``.
    basin_names : list[str]
        Sorted list of basin names.
    basin_areas_m2 : dict[str, float]
        Basin areas for depth conversion.
    output_dir : str
        Directory for saved plots.
    """
    makedirs(output_dir)
    af_to_m3 = 1.0 / M3_TO_AF

    targets = list(basin_names) + ['AZ_Total']

    for basin in targets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        title = basin.replace('_', ' ')
        fig.suptitle(title, fontsize=14, fontweight='bold')

        for col_i, cat in enumerate(('GW', 'SW')):
            # ─── Row 0: Depth (mm / ft) ───
            ax_mm = axes[0, col_i]
            ax_ft = ax_mm.twinx()
            ax_mm.set_title(f'Irrigation {cat}')
            ax_mm.set_ylabel('Depth (mm)')
            ax_ft.set_ylabel('Depth (ft)')

            # ─── Row 1: Volume (m³ / AF) ───
            ax_m3 = axes[1, col_i]
            ax_af = ax_m3.twinx()
            ax_m3.set_ylabel('Volume (m³)')
            ax_af.set_ylabel('Volume (AF)')
            ax_m3.set_xlabel('Year')

            for source in ('ML', 'NHM', 'Reitz'):
                yearly = all_sources[source][cat].get('yearly', {})
                years = sorted(yearly.keys())
                if not years:
                    continue

                if basin == 'AZ_Total':
                    af_vals = np.array([
                        sum(yearly[yr].values()) for yr in years
                    ])
                    total_area = sum(basin_areas_m2.values())
                else:
                    af_vals = np.array([
                        yearly[yr].get(basin, 0.0) for yr in years
                    ])
                    total_area = basin_areas_m2.get(basin, 1.0)

                m3_vals = af_vals * af_to_m3
                mm_vals = m3_vals / total_area * M_TO_MM
                ft_vals = mm_vals * MM_TO_FT

                # Depth row — plot on mm axis (ft axis is linked)
                ax_mm.plot(
                    years, mm_vals,
                    label=source,
                    color=_TS_COLORS[source],
                    marker=_TS_MARKERS[source],
                    markersize=3, linewidth=1.2,
                )
                ax_ft.plot(
                    years, ft_vals,
                    color=_TS_COLORS[source],
                    linestyle='none',  # hidden; twinx shares scale
                )

                # Volume row — plot on m³ axis (AF axis is linked)
                ax_m3.plot(
                    years, m3_vals,
                    label=source,
                    color=_TS_COLORS[source],
                    marker=_TS_MARKERS[source],
                    markersize=3, linewidth=1.2,
                )
                ax_af.plot(
                    years, af_vals,
                    color=_TS_COLORS[source],
                    linestyle='none',
                )

            # Sync twin axis limits
            mm_lo, mm_hi = ax_mm.get_ylim()
            ax_ft.set_ylim(mm_lo * MM_TO_FT, mm_hi * MM_TO_FT)
            m3_lo, m3_hi = ax_m3.get_ylim()
            ax_af.set_ylim(m3_lo * M3_TO_AF, m3_hi * M3_TO_AF)

            ax_mm.legend(fontsize=9)
            ax_m3.legend(fontsize=9)
            for ax in (ax_mm, ax_m3):
                ax.grid(True, alpha=0.3, linestyle='--')

        clean_name = basin.replace(' ', '_').replace('/', '_')
        out_path = os.path.join(output_dir, f'TS_{clean_name}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Time series plots saved to {output_dir}')


# ═════════════════════════════════════════════════════════════════════════════
# 6. Scatter plots — per-basin volumes (pairwise)
# ═════════════════════════════════════════════════════════════════════════════
def _plot_scatter(
    all_sources: dict,
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
) -> None:
    """Create pairwise scatter plots of mean-annual per-basin volumes.

    One figure per GW/SW category with three subplots:
        ML vs NHM, ML vs Reitz, NHM vs Reitz.
    Each subplot shows two rows (AF top, mm bottom) with a 1:1 line and
    linear fit.

    Parameters
    ----------
    all_sources : dict
        ``{'ML': ml_vols, 'NHM': nhm_vols, 'Reitz': reitz_vols}``.
    basin_names : list[str]
        Sorted basin names.
    basin_areas_m2 : dict[str, float]
        Basin areas for depth conversion.
    output_dir : str
        Directory for saved plots.
    """
    makedirs(output_dir)
    af_to_m3 = 1.0 / M3_TO_AF

    pairs = [('ML', 'NHM'), ('ML', 'Reitz'), ('NHM', 'Reitz')]

    for cat in ('GW', 'SW'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        fig.suptitle(f'Irrigation {cat} — Per-Basin Scatter Comparison',
                     fontsize=14, fontweight='bold')

        for col_i, (src_a, src_b) in enumerate(pairs):
            mean_a = all_sources[src_a][cat]['mean']
            mean_b = all_sources[src_b][cat]['mean']

            af_a = np.array([mean_a.get(b, 0.0) for b in basin_names])
            af_b = np.array([mean_b.get(b, 0.0) for b in basin_names])
            areas = np.array([basin_areas_m2.get(b, 1.0) for b in basin_names])
            mm_a = af_a * af_to_m3 / areas * M_TO_MM
            mm_b = af_b * af_to_m3 / areas * M_TO_MM

            for row_i, (vals_x, vals_y, unit) in enumerate([
                (af_a, af_b, 'AF'),
                (mm_a, mm_b, 'mm'),
            ]):
                ax = axes[row_i, col_i]

                ax.scatter(vals_x, vals_y, s=30, alpha=0.7,
                           edgecolors='white', linewidths=0.5)

                # 1:1 line
                lo = min(vals_x.min(), vals_y.min(), 0)
                hi = max(vals_x.max(), vals_y.max()) * 1.05
                ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')

                # Linear fit
                if len(vals_x) > 1 and np.std(vals_x) > 0:
                    z = np.polyfit(vals_x, vals_y, 1)
                    x_fit = np.linspace(lo, hi, 100)
                    ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                            label=f'y={z[0]:.2f}x+{z[1]:.1f}')

                    ss_res = np.sum((vals_y - np.polyval(z, vals_x)) ** 2)
                    ss_tot = np.sum((vals_y - np.mean(vals_y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    ax.set_title(f'{src_a} vs {src_b}  (R²={r2:.3f})',
                                 fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f'{src_a} vs {src_b}', fontsize=11)

                ax.set_xlabel(f'{src_a} ({unit})')
                ax.set_ylabel(f'{src_b} ({unit})')
                ax.legend(fontsize=8, loc='upper left')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)

        out_path = os.path.join(output_dir, f'Scatter_{cat}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Scatter plots saved to {output_dir}')


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

    Parameters
    ----------
    mean_raster_paths : dict
        ``{source: {cat: path}}`` where source ∈ {ML, NHM, Reitz} and
        cat ∈ {GW, SW}.  Paths to mean-annual depth rasters (mm).
    ref_raster : str
        Reference raster for extent / CRS.
    output_dir : str
        Directory for saved plots.
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
) -> dict:
    """
    Generic loader for NHM annual HUC12 CSVs (CU or IE).

    For ``mode='volume'`` (CU): values are total annual Mgal — converted to
    depth (mm), rasterised, and aggregated to basin volumes in AF.  Returns
    ``{'mean': {basin: AF}, 'yearly': {year: {basin: AF}}}``.

    For ``mode='ratio'`` (IE): values are dimensionless ratios — the mean
    annual value is rasterised and aggregated to basin area-weighted means.
    Returns ``{'mean': {basin: ratio}, 'yearly': {year: {basin: ratio}}}``.

    Parameters
    ----------
    csv_path : str
        Path to the NHM annual CSV (Year, <HUC12_code>, …).
    huc12_geojson : str
        Path to ``AZ_HUC12.geojson``.
    basin_gdf : gpd.GeoDataFrame
        Basin polygons (target CRS = reference raster CRS).
    basin_col : str
        Column in *basin_gdf* naming each basin.
    ref_raster : str
        Reference raster for grid/CRS information.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive.
    output_dir : str
        Directory for intermediate rasters.
    mode : str
        ``'volume'`` for CU (Mgal → AF) or ``'ratio'`` for IE.
    predictor_dir : str or None
        Directory with ``Predictor_YYYY.tif`` rasters for irrigated-area
        scaling (used only when ``mode='volume'``).
    irr_fraction_band : int
        Band number in predictor rasters for irrigated fraction
        (used only when ``mode='volume'``).

    Returns
    -------
    dict
        See description above.
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
            ref_raster, ref_crs, ref_transform, ref_shape,
            pixel_area_m2, start_yr, end_yr, output_dir,
            predictor_dir, irr_fraction_band,
        )
    else:
        return _nhm_ie_ratio_path(
            df_az, az_cols, huc_reproj, basin_reproj, basin_col,
            ref_raster, ref_crs, ref_transform, ref_shape,
            start_yr, end_yr, output_dir,
        )


def _nhm_cu_volume_path(
    df_az, az_cols, huc_reproj, basin_reproj, basin_col,
    ref_raster, ref_crs, ref_transform, ref_shape,
    pixel_area_m2, start_yr, end_yr, output_dir,
    predictor_dir, irr_fraction_band,
) -> dict:
    """Process NHM CU CSV into basin volumes (AF)."""
    # Annual total per HUC12: values are in Mgal/yr → m³
    annual_records = []
    for _, row in df_az.iterrows():
        year = int(row['Year'])
        for huc_id in az_cols:
            val = row[huc_id]
            # Values are in Mgal/d; convert to m³/yr
            annual_records.append({
                'huc12': huc_id,
                'year': year,
                'volume_m3': val * MGAL_D_TO_M3_YR,
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
            for idx, row in huc_merged.iterrows():
                geom = [mapping(row.geometry)]
                try:
                    with rio.open(pred_file) as src:
                        clipped, _ = rio_mask(src, geom, crop=True,
                                              all_touched=True,
                                              indexes=[irr_fraction_band],
                                              nodata=np.nan)
                    vals = clipped[0].astype(np.float64)
                    vals = vals[~np.isnan(vals)]
                    vals = np.clip(vals, 0, 1)
                    if vals.size > 0:
                        irr_counts[huc_merged.index.get_loc(idx)] += np.mean(vals)
                except Exception:
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

    out_tif = os.path.join(output_dir, 'NHM_mean_annual_CU_mm.tif')
    with rio.open(ref_raster) as ref_src:
        profile = ref_src.profile.copy()
    profile.update(dtype='float64', nodata=np.nan, count=1)
    nhm_raster[nhm_raster == 0] = np.nan
    with rio.open(out_tif, 'w', **profile) as dst:
        dst.write(nhm_raster, 1)
    logger.info(f'  Wrote NHM CU raster: {out_tif}')

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
    ref_raster, ref_crs, ref_transform, ref_shape,
    start_yr, end_yr, output_dir,
) -> dict:
    """Process NHM IE CSV into basin-mean efficiency ratios."""
    # Mean annual IE per HUC12
    mean_ie = df_az[az_cols].mean(axis=0)
    mean_ie_df = pd.DataFrame({
        'huc12': mean_ie.index.astype(str),
        'mean_ie': mean_ie.values,
    })

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

    out_tif = os.path.join(output_dir, 'NHM_mean_annual_IE.tif')
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

    Parameters
    ----------
    raster_dir : str
        Directory containing annual rasters.
    basin_gdf : gpd.GeoDataFrame
        Basin polygons.
    basin_col : str
        Basin name column.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive.
    file_pattern : str
        Python format string with ``{year}`` placeholder,
        e.g. ``'Irrigation_CU_{year}_mm.tif'``.
    mode : str
        ``'volume'`` for depth rasters or ``'ratio'`` for dimensionless.
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


# ═════════════════════════════════════════════════════════════════════════════
# CU/IE: Time series & scatter plotting
# ═════════════════════════════════════════════════════════════════════════════
_CUIE_COLORS = {'ML': '#2C3E50', 'NHM': '#27AE60'}
_CUIE_MARKERS = {'ML': 'o', 'NHM': 's'}


def _plot_cu_ie_time_series(
    ml_data: dict,
    nhm_data: dict,
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
    variable: str,
) -> None:
    """Per-basin time series plots for CU or IE.

    For CU (``variable='CU'``): two rows — depth (mm/ft) and volume (m³/AF).
    For IE (``variable='IE'``): single row — efficiency ratio.
    """
    makedirs(output_dir)
    af_to_m3 = 1.0 / M3_TO_AF
    targets = list(basin_names) + ['AZ_Total']

    for basin in targets:
        if variable == 'CU':
            fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                                     constrained_layout=True)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 4),
                                     constrained_layout=True)
            axes = [axes]

        title = basin.replace('_', ' ')
        fig.suptitle(f'{title} — Irrigation {variable}',
                     fontsize=14, fontweight='bold')

        for source_name, src_data in [('ML', ml_data), ('NHM', nhm_data)]:
            yearly = src_data.get('yearly', {})
            years = sorted(yearly.keys())
            if not years:
                continue

            if variable == 'CU':
                if basin == 'AZ_Total':
                    af_vals = np.array([
                        sum(yearly[yr].values()) for yr in years
                    ])
                    total_area = sum(basin_areas_m2.values())
                else:
                    af_vals = np.array([
                        yearly[yr].get(basin, 0.0) for yr in years
                    ])
                    total_area = basin_areas_m2.get(basin, 1.0)

                m3_vals = af_vals * af_to_m3
                mm_vals = m3_vals / total_area * M_TO_MM

                # Row 0: depth
                ax_mm = axes[0]
                ax_mm.plot(years, mm_vals, label=source_name,
                           color=_CUIE_COLORS[source_name],
                           marker=_CUIE_MARKERS[source_name],
                           markersize=3, linewidth=1.2)
                ax_mm.set_ylabel('Depth (mm)')
                ax_mm.grid(True, alpha=0.3, linestyle='--')
                ax_mm.legend(fontsize=9)

                # Row 1: volume
                ax_m3 = axes[1]
                ax_m3.plot(years, af_vals, label=source_name,
                           color=_CUIE_COLORS[source_name],
                           marker=_CUIE_MARKERS[source_name],
                           markersize=3, linewidth=1.2)
                ax_m3.set_ylabel('Volume (AF)')
                ax_m3.set_xlabel('Year')
                ax_m3.grid(True, alpha=0.3, linestyle='--')
                ax_m3.legend(fontsize=9)
            else:
                # IE — ratio
                if basin == 'AZ_Total':
                    # Area-weighted mean across basins
                    ie_vals = []
                    for yr in years:
                        yr_d = yearly[yr]
                        vals = [yr_d.get(b, np.nan) for b in basin_names]
                        areas = [basin_areas_m2.get(b, 0) for b in basin_names]
                        finite = [(v, a) for v, a in zip(vals, areas)
                                  if np.isfinite(v)]
                        if finite:
                            v_arr = np.array([x[0] for x in finite])
                            a_arr = np.array([x[1] for x in finite])
                            ie_vals.append(float(np.average(v_arr, weights=a_arr)))
                        else:
                            ie_vals.append(np.nan)
                    ie_vals = np.array(ie_vals)
                else:
                    ie_vals = np.array([
                        yearly[yr].get(basin, np.nan) for yr in years
                    ])

                ax = axes[0]
                ax.plot(years, ie_vals, label=source_name,
                        color=_CUIE_COLORS[source_name],
                        marker=_CUIE_MARKERS[source_name],
                        markersize=3, linewidth=1.2)
                ax.set_ylabel('Irrigation Efficiency')
                ax.set_xlabel('Year')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(fontsize=9)

        # ── Twin y-axes: mm↔ft and AF↔m³ ──
        if variable == 'CU':
            ax_ft = axes[0].twinx()
            ax_ft.set_ylabel('(ft)', fontweight='bold')
            lo, hi = axes[0].get_ylim()
            ax_ft.set_ylim(lo * MM_TO_FT, hi * MM_TO_FT)

            ax_m3_tw = axes[1].twinx()
            ax_m3_tw.set_ylabel('(m³)', fontweight='bold')
            lo, hi = axes[1].get_ylim()
            ax_m3_tw.set_ylim(lo / M3_TO_AF, hi / M3_TO_AF)

        clean_name = basin.replace(' ', '_').replace('/', '_')
        out_path = os.path.join(output_dir, f'TS_{variable}_{clean_name}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'{variable} time series plots saved to {output_dir}')


def _plot_cu_ie_scatter(
    ml_data: dict,
    nhm_data: dict,
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
    variable: str,
) -> None:
    """Scatter plot of ML vs NHM per-basin mean-annual values."""
    makedirs(output_dir)

    if variable == 'CU':
        af_to_m3 = 1.0 / M3_TO_AF
        af_ml = np.array([ml_data['mean'].get(b, 0.0) for b in basin_names])
        af_nhm = np.array([nhm_data['mean'].get(b, 0.0) for b in basin_names])
        areas = np.array([basin_areas_m2.get(b, 1.0) for b in basin_names])
        mm_ml = af_ml * af_to_m3 / areas * M_TO_MM
        mm_nhm = af_nhm * af_to_m3 / areas * M_TO_MM

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle('Irrigation CU — Per-Basin Scatter (ML vs NHM)',
                     fontsize=14, fontweight='bold')
        for col_i, (vals_x, vals_y, unit) in enumerate([
            (af_ml, af_nhm, 'AF'), (mm_ml, mm_nhm, 'mm'),
        ]):
            ax = axes[col_i]
            ax.scatter(vals_x, vals_y, s=30, alpha=0.7,
                       edgecolors='white', linewidths=0.5)
            lo = min(vals_x.min(), vals_y.min(), 0)
            hi = max(vals_x.max(), vals_y.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
            if len(vals_x) > 1 and np.std(vals_x) > 0:
                z = np.polyfit(vals_x, vals_y, 1)
                x_fit = np.linspace(lo, hi, 100)
                ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                        label=f'y={z[0]:.2f}x+{z[1]:.1f}')
                ss_res = np.sum((vals_y - np.polyval(z, vals_x)) ** 2)
                ss_tot = np.sum((vals_y - np.mean(vals_y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                ax.set_title(f'ML vs NHM  (R²={r2:.3f})',
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title('ML vs NHM', fontsize=11)
            ax.set_xlabel(f'ML ({unit})')
            ax.set_ylabel(f'NHM ({unit})')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        out_path = os.path.join(output_dir, 'Scatter_CU.png')
    else:
        # IE scatter
        ml_vals_raw = [ml_data['mean'].get(b, np.nan) for b in basin_names]
        nhm_vals_raw = [nhm_data['mean'].get(b, np.nan) for b in basin_names]
        valid_mask = [
            np.isfinite(a) and np.isfinite(b)
            for a, b in zip(ml_vals_raw, nhm_vals_raw)
        ]
        vals_x = np.array([v for v, m in zip(ml_vals_raw, valid_mask) if m])
        vals_y = np.array([v for v, m in zip(nhm_vals_raw, valid_mask) if m])

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        fig.suptitle('Irrigation Efficiency — Per-Basin Scatter (ML vs NHM)',
                     fontsize=14, fontweight='bold')
        if vals_x.size > 0:
            ax.scatter(vals_x, vals_y, s=30, alpha=0.7,
                       edgecolors='white', linewidths=0.5)
            lo = min(vals_x.min(), vals_y.min()) * 0.9
            hi = max(vals_x.max(), vals_y.max()) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
            if len(vals_x) > 1 and np.std(vals_x) > 0:
                z = np.polyfit(vals_x, vals_y, 1)
                x_fit = np.linspace(lo, hi, 100)
                ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                        label=f'y={z[0]:.2f}x+{z[1]:.2f}')
                ss_res = np.sum((vals_y - np.polyval(z, vals_x)) ** 2)
                ss_tot = np.sum((vals_y - np.mean(vals_y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                ax.set_title(f'ML vs NHM  (R²={r2:.3f})',
                             fontsize=11, fontweight='bold')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel('ML (efficiency)')
        ax.set_ylabel('NHM (efficiency)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        out_path = os.path.join(output_dir, 'Scatter_IE.png')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'{variable} scatter plot saved to {out_path}')


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

    Parameters
    ----------
    ml_pred_dir : str
        Directory with ``pred_YYYY.tif`` (or use *irr_gw_dir*/*irr_sw_dir*).
    nhm_dir : str
        Directory containing the NHM CSV files.
    reitz_base_dir : str
        Parent directory with Reitz sub-folders.
    huc12_geojson : str
        Path to ``AZ_HUC12.geojson``.
    basin_shp : str
        Shapefile or GeoJSON for Arizona groundwater basins.
    basin_col : str
        Column in *basin_shp* identifying each basin.
    output_dir : str
        Root output directory for all results.
    ref_raster : str or None
        Reference raster for CRS/grid.  Defaults to the first ML prediction.
    irr_gw_dir, irr_sw_dir : str or None
        Optional category-specific ML raster directories.
    predictor_dir : str or None
        Directory with ``Predictor_YYYY.tif`` rasters containing
        ``annual_irr_fraction``.  Passed to :func:`load_nhm_basin_volumes`
        so NHM volumes are converted to depth using irrigated area.
    ml_year_range, nhm_year_range, reitz_year_range : tuple[int, int]
        Per-dataset year ranges (inclusive).

    Returns
    -------
    pd.DataFrame
        Summary metrics table for every pairwise comparison × category.
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
    all_sources = {'ML': ml_vols, 'NHM': nhm_vols, 'Reitz': reitz_vols}
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
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    _plot_basin_time_series(
        all_sources, basin_names, basin_areas_m2, plot_dir,
    )

    # ── 8. Scatter plots ────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    _plot_scatter(all_sources, basin_names, basin_areas_m2, scatter_dir)

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
# CU/IE Intercomparison (ML vs NHM)
# ═════════════════════════════════════════════════════════════════════════════
def run_cu_ie_intercomparison(
    irr_cu_dir: str,
    irr_ie_dir: str,
    nhm_cu_csv: str,
    nhm_ie_csv: str,
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
    Run basin-scale intercomparisons of Irrigation Consumptive Use (CU) and
    Irrigation Efficiency (IE) between ML predictions and USGS NHM data.

    CU comparison follows the same volume-based framework as withdrawals
    (metrics in AF, m³, mm).  IE comparison uses dimensionless ratios
    (area-weighted basin means).

    Parameters
    ----------
    irr_cu_dir : str
        Directory with ``Irrigation_CU_{year}_mm.tif`` rasters.
    irr_ie_dir : str
        Directory with ``Irrigation_Efficiency_{year}.tif`` rasters.
    nhm_cu_csv : str
        Path to NHM CU CSV
        (``Irr_CU_HUC12_Tot_annual_2000_2020.csv``).
    nhm_ie_csv : str
        Path to NHM IE CSV
        (``IR_HUC12_Eff_annual_2000_2020.csv``).
    huc12_geojson : str
        Path to ``AZ_HUC12.geojson``.
    basin_shp : str
        Shapefile or GeoJSON for Arizona groundwater basins.
    basin_col : str
        Column in *basin_shp* identifying each basin.
    output_dir : str
        Root output directory for all CU/IE results.
    ref_raster : str or None
        Reference raster for CRS/grid.  Defaults to the first ML CU raster.
    predictor_dir : str or None
        Directory with ``Predictor_YYYY.tif`` rasters for irrigated-area
        scaling in NHM CU conversion.
    ml_year_range, nhm_year_range : tuple[int, int]
        Per-dataset year ranges (inclusive).

    Returns
    -------
    pd.DataFrame
        Summary metrics table.
    """
    makedirs(output_dir)
    logger.info('=' * 60)
    logger.info('Irrigation CU / IE Intercomparison')
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

    # ═════════════════════════════════════════════════════════════════════
    # IE comparison
    # ═════════════════════════════════════════════════════════════════════
    logger.info('--- Loading ML IE rasters ---')
    ml_ie = _load_ml_rasters_to_basins(
        irr_ie_dir, basin_gdf, basin_col, ml_year_range,
        file_pattern='Irrigation_Efficiency_{year}.tif',
        mode='ratio',
    )

    logger.info('--- Loading NHM IE data ---')
    nhm_ie_out = os.path.join(output_dir, 'NHM_IE_Rasters/')
    nhm_ie = _load_nhm_annual_csv_to_basins(
        nhm_ie_csv, huc12_geojson, basin_gdf, basin_col,
        ref_raster, nhm_year_range, nhm_ie_out,
        mode='ratio',
    )

    logger.info('--- IE metrics ---')
    m_ie = _compute_ratio_metrics(
        basin_names, ml_ie['mean'], nhm_ie['mean'], 'ML', 'NHM',
    )
    m_ie['Category'] = 'Irrigation_Efficiency'
    all_metrics.append(m_ie)
    logger.info(
        f'  ML vs NHM IE: RMSD={m_ie["RMSD"]}, '
        f'MAD={m_ie["MAD"]}, PctDiff={m_ie["Pct_Diff"]}%'
    )

    # ── Metrics CSV ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(output_dir, 'cu_ie_intercomparison_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f'CU/IE metrics saved to {metrics_csv}')

    # ── Per-basin comparison tables ──────────────────────────────────────
    rows = []
    for basin in basin_names:
        area = basin_areas_m2.get(basin, 1.0)
        # CU
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
        # IE
        ml_ie_val = ml_ie['mean'].get(basin, np.nan)
        nhm_ie_val = nhm_ie['mean'].get(basin, np.nan)
        rows.append({
            'Category': 'Irrigation_Efficiency',
            'Basin': basin,
            'ML_IE': round(ml_ie_val, 4) if np.isfinite(ml_ie_val) else np.nan,
            'NHM_IE': round(nhm_ie_val, 4) if np.isfinite(nhm_ie_val) else np.nan,
        })
    basin_df = pd.DataFrame(rows)
    basin_csv = os.path.join(output_dir, 'cu_ie_per_basin.csv')
    basin_df.to_csv(basin_csv, index=False)
    logger.info(f'Per-basin CU/IE saved to {basin_csv}')

    # ── Time series CSV ──────────────────────────────────────────────────
    ts_rows = []
    for source_name, cu_src, ie_src in [
        ('ML', ml_cu, ml_ie), ('NHM', nhm_cu, nhm_ie),
    ]:
        # CU time series
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
        # IE time series
        ie_yearly = ie_src.get('yearly', {})
        for year in sorted(ie_yearly.keys()):
            for basin in basin_names:
                ie_val = ie_yearly[year].get(basin, np.nan)
                ts_rows.append({
                    'Category': 'Irrigation_Efficiency',
                    'Source': source_name,
                    'Year': year,
                    'Basin': basin,
                    'Value_IE': round(ie_val, 4) if np.isfinite(ie_val) else np.nan,
                })
    ts_df = pd.DataFrame(ts_rows)
    ts_csv = os.path.join(output_dir, 'cu_ie_time_series.csv')
    ts_df.to_csv(ts_csv, index=False)
    logger.info(f'CU/IE time series saved to {ts_csv}')

    # ── Plots ────────────────────────────────────────────────────────────
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    _plot_cu_ie_time_series(
        ml_cu, nhm_cu, basin_names, basin_areas_m2, plot_dir, 'CU',
    )
    _plot_cu_ie_time_series(
        ml_ie, nhm_ie, basin_names, basin_areas_m2, plot_dir, 'IE',
    )

    scatter_dir = os.path.join(output_dir, 'Scatter/')
    _plot_cu_ie_scatter(
        ml_cu, nhm_cu, basin_names, basin_areas_m2, scatter_dir, 'CU',
    )
    _plot_cu_ie_scatter(
        ml_ie, nhm_ie, basin_names, basin_areas_m2, scatter_dir, 'IE',
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('CU/IE Intercomparison Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')
    logger.info(f'\nML year range: {ml_year_range}')
    logger.info(f'NHM year range: {nhm_year_range}')

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

_CAP_SRP_COLORS = {'ML': '#2C3E50', 'CAP_SRP': '#E74C3C', 'CAP_SRP_spill': '#3498DB'}
_CAP_SRP_MARKERS = {'ML': 'o', 'CAP_SRP': 's', 'CAP_SRP_spill': '^'}


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

    Parameters
    ----------
    cap_xlsx : str
        Path to CAP delivery Excel file.
    srp_xlsx : str
        Path to SRP delivery Excel file.
    include_spill_water : bool
        If True, include SRP ``SPILL WATER`` records in addition to
        ``SURFACE WATER``.  Default False (baseline).

    Returns
    -------
    dict[str, dict[int, float]]
        ``{basin_name: {year: delivery_AF}}``.
    """
    # ── CAP ──────────────────────────────────────────────────────────────
    cap_df = pd.read_excel(cap_xlsx)
    # Keep only direct-use deliveries (null recharge facility)
    cap_df = cap_df[cap_df['Recharge Facility'].isna()].copy()
    # Drop rows with unmappable AMA (e.g. 'Multiple', NaN)
    cap_df = cap_df[cap_df['AMA'].isin(_CAP_AMA_TO_BASIN)]
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

    Returns
    -------
    dict[str, dict[int, float]]
        ``{basin_name: {year: volume_AF}}``.
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


def _plot_cap_srp_time_series(
    ml_basin_yearly: dict[str, dict[int, float]],
    obs_basin_yearly: dict[str, dict[int, float]],
    basin_areas_m2: dict[str, float],
    output_dir: str,
    obs_spill_basin_yearly: dict[str, dict[int, float]] | None = None,
) -> None:
    """Per-basin time series plots: ML Total SW vs CAP+SRP observed
    deliveries.

    Each basin with observed data gets one figure with two rows:
        Row 0: Depth (mm / ft)
        Row 1: Volume (AF / m³)
    An additional 'AZ Total' figure sums across all basins with data.

    When *obs_spill_basin_yearly* is provided, a third 'CAP + SRP (+ Spill)'
    series is drawn as a sensitivity band.
    """
    makedirs(output_dir)
    af_to_m3 = 1.0 / M3_TO_AF

    # Only plot basins that have observed data
    obs_basins = sorted(obs_basin_yearly.keys())
    targets = obs_basins + ['AZ_Total']

    series_list = [
        ('ML', ml_basin_yearly, _CAP_SRP_COLORS['ML'], _CAP_SRP_MARKERS['ML'], 'ML (Total SW)'),
        ('CAP_SRP', obs_basin_yearly, _CAP_SRP_COLORS['CAP_SRP'], _CAP_SRP_MARKERS['CAP_SRP'], 'CAP + SRP'),
    ]
    if obs_spill_basin_yearly:
        series_list.append(
            ('CAP_SRP_spill', obs_spill_basin_yearly,
             _CAP_SRP_COLORS['CAP_SRP_spill'], _CAP_SRP_MARKERS['CAP_SRP_spill'],
             'CAP + SRP (+ Spill)')
        )

    for basin in targets:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        title = basin.replace('_', ' ')
        fig.suptitle(f'{title} — Total Surface Water Withdrawals',
                     fontsize=14, fontweight='bold')

        for source, yearly, color, marker, label in series_list:
            if basin == 'AZ_Total':
                all_years = set()
                for b in obs_basins:
                    all_years.update(yearly.get(b, {}).keys())
                years = sorted(all_years)
                af_vals = np.array([
                    sum(yearly.get(b, {}).get(yr, 0.0) for b in obs_basins)
                    for yr in years
                ])
                total_area = sum(
                    basin_areas_m2.get(b, 0.0) for b in obs_basins
                )
            else:
                bdata = yearly.get(basin, {})
                years = sorted(bdata.keys())
                af_vals = np.array([bdata[yr] for yr in years])
                total_area = basin_areas_m2.get(basin, 1.0)

            if not years:
                continue

            m3_vals = af_vals * af_to_m3
            mm_vals = m3_vals / total_area * M_TO_MM if total_area > 0 else m3_vals * 0

            # Row 0: depth
            ax_mm = axes[0]
            ax_mm.plot(years, mm_vals, label=label, color=color,
                       marker=marker, markersize=3, linewidth=1.2)

            # Row 1: volume
            ax_af = axes[1]
            ax_af.plot(years, af_vals, label=label, color=color,
                       marker=marker, markersize=3, linewidth=1.2)

        axes[0].set_ylabel('Depth (mm)')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=9)
        # Add twin ft axis
        ax_ft = axes[0].twinx()
        mm_lo, mm_hi = axes[0].get_ylim()
        ax_ft.set_ylim(mm_lo * MM_TO_FT, mm_hi * MM_TO_FT)
        ax_ft.set_ylabel('Depth (ft)')

        axes[1].set_ylabel('Volume (AF)')
        axes[1].set_xlabel('Year')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=9)
        # Add twin m³ axis
        ax_m3 = axes[1].twinx()
        af_lo, af_hi = axes[1].get_ylim()
        ax_m3.set_ylim(af_lo * af_to_m3, af_hi * af_to_m3)
        ax_m3.set_ylabel('Volume (m³)')

        clean_name = basin.replace(' ', '_').replace('/', '_')
        out_path = os.path.join(output_dir, f'TS_Total_SW_{clean_name}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'CAP/SRP time series plots saved to {output_dir}')


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


def _plot_cap_srp_scatter(
    ml_basin_yearly: dict[str, dict[int, float]],
    obs_basin_yearly: dict[str, dict[int, float]],
    output_dir: str,
) -> None:
    """Scatter plot of ML vs observed (CAP+SRP) annual basin volumes.

    Plots one point per basin-year pair (AF), with a 1:1 line, linear fit,
    and R² annotation.  An additional panel shows the same data in mm.
    """
    makedirs(output_dir)

    obs_basins = sorted(obs_basin_yearly.keys())
    ml_vals_list, obs_vals_list, labels = [], [], []
    for basin in obs_basins:
        common_years = sorted(
            set(ml_basin_yearly.get(basin, {}).keys())
            & set(obs_basin_yearly[basin].keys())
        )
        for yr in common_years:
            ml_vals_list.append(ml_basin_yearly[basin][yr])
            obs_vals_list.append(obs_basin_yearly[basin][yr])
            labels.append(basin)

    if not ml_vals_list:
        return

    ml_af = np.array(ml_vals_list)
    obs_af = np.array(obs_vals_list)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig.suptitle('ML Total SW vs CAP + SRP — Per Basin-Year',
                 fontsize=14, fontweight='bold')

    ax.scatter(obs_af, ml_af, s=30, alpha=0.7,
               edgecolors='white', linewidths=0.5)

    # 1:1 line
    lo = min(obs_af.min(), ml_af.min(), 0)
    hi = max(obs_af.max(), ml_af.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')

    # Linear fit
    if len(obs_af) > 1 and np.std(obs_af) > 0:
        z = np.polyfit(obs_af, ml_af, 1)
        x_fit = np.linspace(lo, hi, 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                label=f'y={z[0]:.2f}x+{z[1]:.1f}')
        ss_res = np.sum((ml_af - np.polyval(z, obs_af)) ** 2)
        ss_tot = np.sum((ml_af - np.mean(ml_af)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.set_title(f'R² = {r2:.3f}', fontsize=12, fontweight='bold')

    ax.set_xlabel('Observed CAP + SRP (AF)')
    ax.set_ylabel('ML Total SW (AF)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    out_path = os.path.join(output_dir, 'Scatter_ML_vs_CAP_SRP.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'CAP/SRP scatter plot saved to {out_path}')


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

    Parameters
    ----------
    cap_xlsx : str
        Path to the CAP delivery Excel file.
    srp_xlsx : str
        Path to the SRP delivery Excel file.
    total_sw_dir : str
        Directory with ``Total_SW_YYYY_mm.tif`` rasters.
    basin_shp : str
        Shapefile or GeoJSON for Arizona groundwater basins.
    basin_col : str
        Column in *basin_shp* identifying each basin.
    output_dir : str
        Root output directory for validation results.
    year_range : tuple[int, int]
        ``(start_year, end_year)`` inclusive for ML rasters.

    Returns
    -------
    pd.DataFrame
        Per-basin statistics (RMSD, MAD, Pct Diff, Pearson R).
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
    plot_dir = os.path.join(output_dir, 'Time_Series/')
    _plot_cap_srp_time_series(
        ml_basin_yearly, obs_basin_yearly, basin_areas_m2, plot_dir,
        obs_spill_basin_yearly=obs_spill_basin_yearly,
    )

    # ── Scatter plot ─────────────────────────────────────────────────────
    scatter_dir = os.path.join(output_dir, 'Scatter/')
    _plot_cap_srp_scatter(
        ml_basin_yearly, obs_basin_yearly, scatter_dir,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('\n' + '=' * 60)
    logger.info('CAP/SRP Validation Summary')
    logger.info('=' * 60)
    logger.info(f'\n{metrics_df.to_string(index=False)}')

    return metrics_df


# ═════════════════════════════════════════════════════════════════════════════
# Recommendations
# ═════════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS = """
Intercomparison Recommendations
================================
1. **Common overlap period**: The default range is 1980-2020, covering the
   full span of all three datasets (ML 2002-2020, NHM 2000-2020,
   Reitz 1980-2018).  Years without data for a given dataset will
   appear as blank/zero.

2. **Spatial aggregation**: Because the three datasets have fundamentally
   different native resolutions (2 km ML, ~800 m Reitz, HUC12 polygons NHM),
   aggregating to groundwater basin totals (in AF) removes artefacts
   from pixel-level misalignment and allows direct volumetric comparison.

3. **Caution with NHM rasterisation**: NHM withdrawals are tabular per
   HUC12; rasterising them assumes uniform depth within each polygon.
   Comparison is most meaningful at the basin or state total level.

4. **ML category rasters**: If the pipeline has produced separate
   ``Irrigation_GW_YYYY_mm.tif`` and ``Irrigation_SW_YYYY_mm.tif`` rasters,
   use those directly (set *irr_gw_dir* / *irr_sw_dir*).  Otherwise the
   total ``pred_YYYY.tif`` rasters are used as a rough upper bound.

5. **Additional diagnostics** *(implemented)*: Scatter plots of per-basin
   mean-annual volumes (ML vs NHM, ML vs Reitz, NHM vs Reitz) with 1:1 lines
   and linear fits are produced via ``_plot_scatter``.  Spatial maps of
   pixel-wise mean-annual depth differences (diverging colourmap centred on
   zero) are produced via ``_plot_spatial_diff_maps``.

6. **Unit consistency**: All rasters are internally converted to mm depth
   before aggregation; final basin totals are reported in mm, ft, m³,
   and AF across all output CSVs and plots.
"""
