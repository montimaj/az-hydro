"""
Contains codes for downloading and processing streamflow data from USGS and USBR
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import dataretrieval.nwis as nwis
import logging

from glob import glob
from sysops import makedirs, az_nodata
from rasterops import write_raster
from vectorops import shp2raster

logger = logging.getLogger(__name__)


def _load_usbr_data(usbr_dir, usbr_id):
    """Load and average USBR ensemble data for a given USBR site ID."""
    usbr_files = glob(os.path.join(usbr_dir, f'streamflow_*_{usbr_id}.csv'))
    if not usbr_files:
        return None
    usbr_parts = []
    for usbr_file in usbr_files:
        udf = pd.read_csv(usbr_file)
        model_cols = [c for c in udf.columns if c not in ('Year', 'Month')]
        udf['discharge_cfs'] = udf[model_cols].mean(axis=1)
        udf['date'] = pd.to_datetime(
            udf['Year'].astype(str) + '-' + udf['Month'].astype(str).str.zfill(2) + '-01'
        )
        usbr_parts.append(udf[['date', 'discharge_cfs']].set_index('date'))
    usbr_monthly = pd.concat(usbr_parts)
    return usbr_monthly.groupby(level=0).mean()


def _download_usgs_monthly(site_id, param_cd, stat_cd, start_year, end_year):
    """Download USGS daily values and resample to monthly mean."""
    df, _ = nwis.get_dv(
        sites=site_id,
        parameterCd=param_cd,
        statCd=stat_cd,
        start=f'{start_year}-01-01',
        end=f'{end_year}-12-31'
    )
    if df.empty:
        return None
    flow_col = [c for c in df.columns if param_cd in c and 'cd' not in c.lower()]
    if not flow_col:
        return None
    df = df[[flow_col[0]]].rename(columns={flow_col[0]: 'discharge_cfs'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.resample('MS').mean().dropna()


def _find_nearest_usbr_site(site_id, usbr_sites, sites_df):
    """Find the nearest USBR-gauged site based on USGS site coordinates."""
    try:
        info, _ = nwis.get_info(sites=[site_id] + usbr_sites)
        if info.empty:
            return usbr_sites[0]
        target = info[info['site_no'] == site_id]
        if target.empty:
            return usbr_sites[0]
        t_lon = float(target.iloc[0]['dec_long_va'])
        t_lat = float(target.iloc[0]['dec_lat_va'])
        best_sid, best_dist = usbr_sites[0], float('inf')
        for ref_sid in usbr_sites:
            ref = info[info['site_no'] == ref_sid]
            if ref.empty:
                continue
            r_lon = float(ref.iloc[0]['dec_long_va'])
            r_lat = float(ref.iloc[0]['dec_lat_va'])
            dist = (t_lon - r_lon) ** 2 + (t_lat - r_lat) ** 2
            if dist < best_dist:
                best_dist = dist
                best_sid = ref_sid
        return best_sid
    except Exception:
        return usbr_sites[0]


def download_streamflow(
        sites_csv: str,
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2025,
        param_cd: str = '00060',
        stat_cd: str = '00003',
        already_downloaded: bool = False,
        verbose: bool = True
) -> str:
    """
    Download USGS streamflow data for all sites in a CSV file, merge with USBR
    modeled projections where available, and use a historical ratio method to
    generate 2026-2099 projections for sites without USBR data.

    For USBR-gauged sites: USGS observations take priority; USBR fills months
    outside the USGS record. For ungauged sites: monthly ratios between the site
    and the nearest USBR-gauged site are computed over their overlapping USGS
    period, then applied to the reference site's USBR projections.

    Args:
        sites_csv (str): Path to a CSV with 'USGS_SITE_ID', 'USBR_SITE_ID', and 'SITE_NAME' columns.
            USBR_SITE_ID can be empty for sites without USBR data.
        output_dir (str): Output directory for the downloaded data.
        start_year (int): Start year for the data download. Defaults to 1896.
        end_year (int): End year for the data download. Defaults to 2025.
        param_cd (str): USGS parameter code. Defaults to '00060' (discharge in cfs).
        stat_cd (str): USGS statistic code. Defaults to '00003' (mean).
        already_downloaded (bool): If True, skip download and return existing output path.
        verbose (bool): If True, print progress messages.

    Returns:
        str: Path to the output directory containing the downloaded CSV files.
    """

    makedirs(output_dir)
    if already_downloaded:
        logger.info('Streamflow data already downloaded, skipping...')
        return output_dir

    sites_df = pd.read_csv(sites_csv, dtype={'USGS_SITE_ID': str, 'USBR_SITE_ID': str})
    site_ids = sites_df['USGS_SITE_ID'].tolist()
    site_names = dict(zip(sites_df['USGS_SITE_ID'], sites_df['SITE_NAME']))
    usbr_ids = dict(zip(sites_df['USGS_SITE_ID'], sites_df['USBR_SITE_ID']))
    usbr_dir = os.path.dirname(sites_csv)

    # Identify which sites have USBR data
    usbr_sites = [sid for sid in site_ids
                  if pd.notna(usbr_ids.get(sid)) and usbr_ids.get(sid, '').strip()]
    no_usbr_sites = [sid for sid in site_ids if sid not in usbr_sites]

    # --- Pass 1: Process sites with USBR data ---
    usgs_data = {}  # Store USGS monthly data for ratio computation
    usbr_data = {}  # Store USBR monthly data for ratio computation
    all_monthly = []

    for site_id in usbr_sites:
        site_name = site_names.get(site_id, site_id)
        usbr_id = usbr_ids[site_id]

        if verbose:
            logger.info(f'Downloading USGS streamflow for site {site_id} ({site_name})...')
        usgs_monthly = None
        try:
            usgs_monthly = _download_usgs_monthly(site_id, param_cd, stat_cd, start_year, end_year)
        except Exception as e:
            logger.warning(f'Failed to download USGS data for site {site_id}: {e}')

        usbr_monthly = _load_usbr_data(usbr_dir, usbr_id)
        if usbr_monthly is not None and verbose:
            logger.info(f'Loaded USBR data for site {site_id}')

        # Store for ratio computation
        if usgs_monthly is not None:
            usgs_data[site_id] = usgs_monthly
        if usbr_monthly is not None:
            usbr_data[site_id] = usbr_monthly

        # Merge: USGS takes priority, USBR fills the rest
        if usgs_monthly is not None and usbr_monthly is not None:
            usgs_start = usgs_monthly.index.min()
            usgs_end = usgs_monthly.index.max()
            usbr_fill = usbr_monthly[
                (usbr_monthly.index < usgs_start) | (usbr_monthly.index > usgs_end)
            ]
            monthly = pd.concat([usgs_monthly, usbr_fill]).sort_index()
            monthly = monthly[~monthly.index.duplicated(keep='first')]
        elif usgs_monthly is not None:
            monthly = usgs_monthly
        elif usbr_monthly is not None:
            monthly = usbr_monthly
        else:
            logger.warning(f'No data available for site {site_id}, skipping')
            continue

        # Reindex and gap-fill with monthly climatology
        full_range = pd.date_range(start='1896-01-01', end='2099-12-01', freq='MS')
        monthly = monthly.reindex(full_range)
        month_clim = monthly.groupby(monthly.index.month)['discharge_cfs'].transform('mean')
        monthly['discharge_cfs'] = monthly['discharge_cfs'].fillna(month_clim)
        monthly.index.name = 'date'
        monthly['site_id'] = site_id
        monthly['site_name'] = site_name

        site_file = os.path.join(output_dir, f'streamflow_{site_id}.csv')
        monthly.to_csv(site_file, date_format='%Y-%m-%d')
        if verbose:
            logger.info(f'Saved {len(monthly)} monthly records for site {site_id}')
        all_monthly.append(monthly)

    # --- Pass 2: Process sites without USBR data (historical ratio method) ---
    for site_id in no_usbr_sites:
        site_name = site_names.get(site_id, site_id)

        if verbose:
            logger.info(f'Downloading USGS streamflow for site {site_id} ({site_name})...')
        usgs_monthly = None
        try:
            usgs_monthly = _download_usgs_monthly(site_id, param_cd, stat_cd, start_year, end_year)
        except Exception as e:
            logger.warning(f'Failed to download USGS data for site {site_id}: {e}')

        if usgs_monthly is None or usgs_monthly.empty:
            logger.warning(f'No USGS data for site {site_id}, skipping')
            continue

        # Find the nearest USBR-gauged reference site
        ref_site = _find_nearest_usbr_site(site_id, usbr_sites, sites_df)
        ref_usgs = usgs_data.get(ref_site)
        ref_usbr = usbr_data.get(ref_site)

        if verbose:
            logger.info(f'Using reference site {ref_site} for historical ratio method')

        synthetic_usbr = None
        if ref_usgs is not None and ref_usbr is not None:
            # Compute monthly ratios during overlapping USGS period
            overlap_start = max(usgs_monthly.index.min(), ref_usgs.index.min())
            overlap_end = min(usgs_monthly.index.max(), ref_usgs.index.max())

            if overlap_start < overlap_end:
                site_overlap = usgs_monthly.loc[overlap_start:overlap_end, 'discharge_cfs']
                ref_overlap = ref_usgs.loc[overlap_start:overlap_end, 'discharge_cfs']

                # Align on common dates before computing ratios
                common_idx = site_overlap.index.intersection(ref_overlap.index)
                site_overlap = site_overlap.loc[common_idx]
                ref_overlap = ref_overlap.loc[common_idx]

                # Compute ratio per calendar month (avoid division by zero)
                ratio_df = pd.DataFrame({
                    'site': site_overlap,
                    'ref': ref_overlap,
                    'month': site_overlap.index.month
                }).dropna()
                ratio_df = ratio_df[ratio_df['ref'] > 0]
                monthly_ratios = ratio_df.groupby('month').apply(
                    lambda g: g['site'].mean() / g['ref'].mean() if g['ref'].mean() > 0 else 0
                )

                # Apply ratios to reference USBR projections
                usgs_end = usgs_monthly.index.max()
                future_usbr = ref_usbr[ref_usbr.index > usgs_end].copy()
                if not future_usbr.empty:
                    future_usbr['ratio'] = future_usbr.index.month.map(monthly_ratios)
                    future_usbr['discharge_cfs'] = future_usbr['discharge_cfs'] * future_usbr['ratio']
                    synthetic_usbr = future_usbr[['discharge_cfs']]

                # Also fill pre-USGS period using USBR * ratio
                pre_usgs_start = usgs_monthly.index.min()
                past_usbr = ref_usbr[ref_usbr.index < pre_usgs_start].copy()
                if not past_usbr.empty:
                    past_usbr['ratio'] = past_usbr.index.month.map(monthly_ratios)
                    past_usbr['discharge_cfs'] = past_usbr['discharge_cfs'] * past_usbr['ratio']
                    past_synthetic = past_usbr[['discharge_cfs']]
                    if synthetic_usbr is not None:
                        synthetic_usbr = pd.concat([past_synthetic, synthetic_usbr])
                    else:
                        synthetic_usbr = past_synthetic

                if verbose:
                    logger.info(f'Computed monthly ratios from {overlap_start.strftime("%Y-%m")} '
                                f'to {overlap_end.strftime("%Y-%m")} '
                                f'({len(ratio_df)} overlapping months)')

        # Merge USGS with synthetic projections
        if synthetic_usbr is not None:
            monthly = pd.concat([usgs_monthly, synthetic_usbr]).sort_index()
            monthly = monthly[~monthly.index.duplicated(keep='first')]
        else:
            monthly = usgs_monthly

        # Reindex and gap-fill with monthly climatology
        full_range = pd.date_range(start='1896-01-01', end='2099-12-01', freq='MS')
        monthly = monthly.reindex(full_range)
        month_clim = monthly.groupby(monthly.index.month)['discharge_cfs'].transform('mean')
        monthly['discharge_cfs'] = monthly['discharge_cfs'].fillna(month_clim)
        monthly.index.name = 'date'
        monthly['site_id'] = site_id
        monthly['site_name'] = site_name

        site_file = os.path.join(output_dir, f'streamflow_{site_id}.csv')
        monthly.to_csv(site_file, date_format='%Y-%m-%d')
        if verbose:
            logger.info(f'Saved {len(monthly)} monthly records for site {site_id}')
        all_monthly.append(monthly)

    if all_monthly:
        combined = pd.concat(all_monthly).reset_index()
        combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
        combined_file = os.path.join(output_dir, 'streamflow_all_sites.csv')
        combined.to_csv(combined_file, index=False)
        if verbose:
            logger.info(f'Saved combined streamflow data to {combined_file}')

    return output_dir


def _get_site_watershed_map(sites_csv, watershed_geojson):
    """Map each USGS site to its surface watershed via spatial join."""
    sites_df = pd.read_csv(sites_csv, dtype=str)
    site_ids = sites_df['USGS_SITE_ID'].tolist()
    info, _ = nwis.get_info(sites=site_ids)
    info = info[['site_no', 'dec_lat_va', 'dec_long_va']].drop_duplicates('site_no')
    site_gdf = gpd.GeoDataFrame(
        info,
        geometry=gpd.points_from_xy(
            info['dec_long_va'].astype(float),
            info['dec_lat_va'].astype(float)
        ),
        crs='EPSG:4326'
    )
    ws = gpd.read_file(watershed_geojson)
    joined = gpd.sjoin(site_gdf, ws[['OBJECTID', 'WATERSHED', 'geometry']],
                       how='left', predicate='within')
    # Map: watershed OBJECTID -> list of site_no in that watershed
    ws_sites = {}
    for _, row in joined.dropna(subset=['OBJECTID']).iterrows():
        oid = int(row['OBJECTID'])
        ws_sites.setdefault(oid, []).append(row['site_no'])
    return ws_sites


def _compute_annual_streamflow(streamflow_dir, site_ids, start_year, end_year):
    """Compute annual mean streamflow in m3/s for a list of sites, averaged across sites."""
    cfs_to_m3s = 0.028316846592
    all_annual = []
    for site_id in site_ids:
        site_file = os.path.join(streamflow_dir, f'streamflow_{site_id}.csv')
        if not os.path.exists(site_file):
            continue
        df = pd.read_csv(site_file, parse_dates=['date'], index_col='date')
        df = df[['discharge_cfs']].copy()
        df['year'] = df.index.year
        annual = df.groupby('year')['discharge_cfs'].mean() * cfs_to_m3s
        all_annual.append(annual)
    if not all_annual:
        return pd.Series(dtype=float)
    # Average across gauges for each year
    combined = pd.concat(all_annual, axis=1).mean(axis=1)
    combined = combined.loc[start_year:end_year]
    return combined


def create_streamflow_rasters(
        watershed_geojson: str,
        cap_service_area_geojson: str,
        sites_csv: str,
        output_dir: str,
        xres: float = 2000,
        yres: float = 2000,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False
) -> None:
    """
    Create annual streamflow rasters (mm/yr) for Arizona surface watersheds.

    Each pixel is assigned the area-normalized annual streamflow of its
    watershed, computed by averaging all USGS gauges within that watershed
    and dividing by the watershed area. Pixels within the CAP service area
    additionally receive Colorado River streamflow normalized by the CAP
    service area. Units are mm/yr, consistent with ET and precipitation bands.

    Args:
        watershed_geojson (str): Path to Surface_Watershed.geojson.
        cap_service_area_geojson (str): Path to CAP_Service_Area.geojson.
        sites_csv (str): Path to sites.csv with USGS_SITE_ID column.
        output_dir (str): Output directory for the streamflow rasters.
        xres (float): X-resolution in geographic units. Defaults to 2000.
        yres (float): Y-resolution in geographic units. Defaults to 2000.
        start_year (int): Start year. Defaults to 1896.
        end_year (int): End year. Defaults to 2099.
        already_created (bool): If True, skip creation.

    Returns:
        None
    """
    if already_created:
        logger.info('Streamflow rasters already created, skipping...')
        return

    streamflow_dir = f'{output_dir}Streamflow/'
    makedirs(streamflow_dir)
    download_streamflow(
        sites_csv=sites_csv,
        output_dir=streamflow_dir,
        start_year=start_year,
        end_year=end_year,
    )
    makedirs(streamflow_dir)

    # Step 1: Rasterize watersheds using OBJECTID
    ws_raster = os.path.join(streamflow_dir, 'watershed_template.tif')
    shp2raster(
        watershed_geojson, ws_raster,
        xres=xres, yres=yres,
        value_field='OBJECTID',
        add_value=False
    )

    # Step 2: Rasterize CAP service area as binary mask
    cap_raster = os.path.join(streamflow_dir, 'cap_template.tif')
    shp2raster(
        cap_service_area_geojson, cap_raster,
        xres=xres, yres=yres,
        burn_value=1.0
    )

    # Read template rasters
    _, ws_file = rio.open(ws_raster), None
    ws_file = rio.open(ws_raster)
    ws_arr = ws_file.read(1).astype(np.float32)
    ws_transform = ws_file.transform
    ws_shape = ws_arr.shape

    # Read and align CAP raster to watershed grid
    cap_file = rio.open(cap_raster)
    cap_arr_native = cap_file.read(1).astype(np.float32)

    # Reproject CAP to match watershed raster grid if needed
    if cap_file.shape != ws_shape or cap_file.transform != ws_transform:
        from rasterio.warp import reproject, Resampling
        cap_arr = np.zeros(ws_shape, dtype=np.float32)
        reproject(
            source=cap_arr_native,
            destination=cap_arr,
            src_transform=cap_file.transform,
            src_crs=cap_file.crs,
            dst_transform=ws_transform,
            dst_crs=ws_file.crs,
            resampling=Resampling.nearest
        )
    else:
        cap_arr = cap_arr_native
    cap_file.close()

    # Step 3: Map gauges to watersheds and compute watershed areas (m²)
    ws_sites = _get_site_watershed_map(sites_csv, watershed_geojson)
    unique_oids = [int(v) for v in np.unique(ws_arr) if v > 0 and not np.isnan(v)]

    ws_gdf = gpd.read_file(watershed_geojson)
    ws_gdf_proj = ws_gdf.to_crs(ws_gdf.estimate_utm_crs())
    ws_areas = {}  # OBJECTID -> area in m²
    for _, row in ws_gdf_proj.iterrows():
        ws_areas[int(row['OBJECTID'])] = row.geometry.area

    # CAP service area in m²
    cap_gdf = gpd.read_file(cap_service_area_geojson)
    cap_gdf_proj = cap_gdf.to_crs(cap_gdf.estimate_utm_crs())
    cap_area_m2 = cap_gdf_proj.geometry.area.sum()

    # Conversion factor: m³/s -> mm/yr = (Q * 86400 * 365.25 / A) * 1000
    # = Q * 31_557_600_000 / A
    m3s_to_mm_yr = 31_557_600_000.0  # seconds/year * 1000 mm/m

    # Colorado River site IDs for CAP overlay (Lees Ferry + CAP Canal at Havasu)
    co_river_sites = ['09380000', '09426650']

    # Step 4: Compute annual streamflow per watershed
    logger.info('Computing annual streamflow per watershed...')
    ws_annual = {}
    for oid in unique_oids:
        site_ids = ws_sites.get(oid, [])
        if site_ids:
            annual_m3s = _compute_annual_streamflow(
                streamflow_dir, site_ids, start_year, end_year
            )
            # Normalize by watershed area to mm/yr
            area = ws_areas.get(oid, 1.0)
            ws_annual[oid] = annual_m3s * m3s_to_mm_yr / area

    # Colorado River annual streamflow for CAP pixels (normalized by CAP service area)
    co_annual_m3s = _compute_annual_streamflow(
        streamflow_dir, co_river_sites, start_year, end_year
    )
    co_annual = co_annual_m3s * m3s_to_mm_yr / cap_area_m2

    # Step 5: Create annual rasters (mm/yr)
    no_data = az_nodata()
    for year in range(start_year, end_year + 1):
        out_arr = np.full(ws_shape, 0.0, dtype=np.float32)

        # Assign watershed streamflow (mm/yr)
        for oid in unique_oids:
            if oid in ws_annual and year in ws_annual[oid].index:
                flow_val = ws_annual[oid].loc[year]
            else:
                # Use climatological mean if year is missing
                flow_val = ws_annual[oid].mean() if oid in ws_annual else 0.0
            out_arr[ws_arr == oid] = flow_val

        # Overlay CAP service area with Colorado River streamflow (mm/yr)
        if year in co_annual.index:
            co_flow = co_annual.loc[year]
        else:
            co_flow = co_annual.mean() if not co_annual.empty else 0.0
        cap_mask = cap_arr > 0
        out_arr[cap_mask] += co_flow

        out_arr[np.isnan(out_arr)] = 0.0
        out_file = os.path.join(output_dir, f'Streamflow_{year}.tif')
        write_raster(
            out_arr, ws_file,
            transform_=ws_transform,
            outfile_path=out_file,
            no_data_value=no_data
        )

    ws_file.close()
    logger.info(f'Created {end_year - start_year + 1} annual streamflow rasters in {output_dir}')
