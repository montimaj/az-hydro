"""
Contains codes for handling Google Earth Engine datasets
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import os
import pickle
import shutil
import subprocess
import time

import ee
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import sklearn.utils as sk
import swifter  # noqa: F401 — imported for .swifter.apply() side-effect
from osgeo import gdal

# Suppress GDAL warnings (keep errors only)
os.environ['CPL_LOG'] = '/dev/null'
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('pyogrio').setLevel(logging.ERROR)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('googleapiclient').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.http').setLevel(logging.ERROR)
logging.getLogger('google_auth_httplib2').setLevel(logging.ERROR)
from glob import glob
from http.client import RemoteDisconnected

from dask import compute, delayed
from dask.distributed import Client, LocalCluster
from google.cloud import storage
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from hydrolibs.gwops import create_land_use_data
from hydrolibs.rasterops import (
    clamp_and_rewrite_raster,
    get_xy_grids_from_raster,
    read_raster_as_arr,
    reproject_raster_gdal,
)
from hydrolibs.sysops import makedirs
from hydrolibs.visualops import get_ama_ina_basin_names

for _dask_logger in ['dask', 'distributed', 'distributed.worker', 'distributed.worker.state_machine',
                     'distributed.scheduler', 'distributed.nanny', 'dask_jobqueue',
                     'tornado', 'tornado.application']:
    logging.getLogger(_dask_logger).setLevel(logging.CRITICAL)


def get_maca_scenarios_models_bands():
    """
    Get MACA scenarios, models, and common bands to use for ETo and actual ET calculations for 2026-2099. 
    """
    maca_scenarios = ['rcp45', 'rcp85']
    maca_models = [
        'bcc-csm1-1',
        'bcc-csm1-1-m',
        'BNU-ESM',
        'CanESM2',
        'CCSM4',
        'CNRM-CM5',
        'CSIRO-Mk3-6-0',
        'GFDL-ESM2G',
        'GFDL-ESM2M',
        'HadGEM2-CC365',
        'HadGEM2-ES365',
        'inmcm4',
        'IPSL-CM5A-MR',
        'IPSL-CM5A-LR',
        'IPSL-CM5B-LR',
        'MIROC5',
        'MIROC-ESM',
        'MIROC-ESM-CHEM',
        'MRI-CGCM3',
        'NorESM1-M'
    ]
    maca_bands = ['tasmax', 'tasmin', 'pr', 'rsds', 'uas', 'vas', 'huss']
    return maca_scenarios, maca_models, maca_bands


def create_fishnet(
        input_gdf: gpd.GeoDataFrame,
        fishnet_unit: str = 'm',
        fishnet_size: int = 1000,
        fishnet_crs: str = 'EPSG:26712',
        output_dir: str = '../../Data/Inputs/GW_Data/'
) -> gpd.GeoDataFrame:
    """
    Create 1 km or 1 mi fishnet from an input geodataframe.

    Args:
    input_gdf (gpd.GeoDataFrame): Input geodataframe.
    fishnet_unit (str): Whether to create polygon grids in m or mi.
    fishnet_size (str): Defaults to 1000 m. If fishnet_unit = 'mi', then 1609.34 m grid will be used.
    fishnet_crs (str): Defaults to the Arizona UTM Zone 12.
    output_dir (str): Output directory to store the fishnets.

    Returns:
        The fishnet as a gpd.GeoDataFrame.
    """

    fishnet_file = os.path.join(output_dir, f'AZ_Polygons_{fishnet_size}{fishnet_unit}.geojson')
    if os.path.exists(fishnet_file):
        return gpd.read_file(fishnet_file)
    gdf = input_gdf.to_crs(fishnet_crs)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    if fishnet_unit == 'mi':
        fishnet_size *= 1609.34 / 1000
    length = fishnet_size
    wide = fishnet_size
    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + wide, y), (x + wide, y + length), (x, y + length)]))
    fishnet = gpd.GeoDataFrame({'geometry': polygons}, crs=fishnet_crs).clip(gdf).reset_index(drop=True)
    fishnet['FID'] = fishnet.index + 1
    fishnet.to_file(fishnet_file)
    return fishnet


def generate_chunks(input_list: list, num_chunks: int):
    """
    Partition a list into equally sized chunks.

    Args:
        input_list (List): List of objects.
        num_chunks (int): Number of chunks.

    Returns:
        Generator object
    """

    num_chunks = max(1, num_chunks)
    for idx in range(0, len(input_list), num_chunks):
        yield input_list[idx: idx + num_chunks]


def get_eto(
        year: int,
        prism_eto_ic: ee.ImageCollection | None = None,
        maca_eto_ic: ee.ImageCollection | None = None,
        return_annual: bool = False
) -> ee.ImageCollection:
    """
    Process daily collection to calculate annual or monthly ETo.
    Author: Peter ReVelle (peter.revelle@dri.edu)
    Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

    Args:
    year (int): Year in YYYY format for which the daily image belongs to. This is required to determine the 
    climate data source for ETo calculation. Note that since ETo calculation requires daily data, for 1895-1978, we 
    take the monthly PRISM data and assign the same ETo value for all days in that month. For 1979-2025, 
    gridMET daily ETo is directly used. For 2026-2099, we use MACA daily data for ETo calculation.
    prism_eto_ic (ee.ImageCollection or None): Monthly PRISM ETo ImageCollection to use for ETo calculation for 
    1895-1978. Only required if year is between 1895 and 1978. Should have a band named 'eto' in mm.
    maca_eto_ic (ee.ImageCollection or None): Monthly MACA ETo ImageCollection to use for ETo calculation for 2026-2099. 
    Only required if year > 2025. Should have a band named 'eto' in mm.
    return_annual (bool): If True, return annual ETo instead of monthly.
    Returns:
    ee.ImageCollection: Monthly ETo ImageCollection for the input year with band 'eto' (in mm) and properties 
    'system:time_start' and 'system:time_end'.
    """

    start_year = f'{year}-01-01'
    end_year = f'{year + 1}-01-01'
    if year > 2025:
        # Use MACA for future scenarios, applying gridMET bias correction factors to it
        monthly_eto = maca_eto_ic
    elif 1895 <= year <= 1978:
        # Use PRISM for 1895-1978, calculate Hargreaves ETo, and apply Hargreaves/gridMET ratio to it
        monthly_eto = prism_eto_ic
    else:
        # Use gridMET for 1979-2025 (already monthly)
        monthly_eto = ee.ImageCollection('projects/openet/assets/reference_et/conus/gridmet/monthly/v1').select('eto')
    monthly_eto = monthly_eto.filterDate(start_year, end_year)
    # Ensure consistent band name 'eto' and add 'month' property for joining
    monthly_eto = monthly_eto.map(
        lambda img: img.rename('eto')
            .set('month', ee.Date(img.get('system:time_start')).get('month'))
            .copyProperties(img, ['system:time_start', 'system:time_end'])
    )
    return monthly_eto if not return_annual else monthly_eto.sum().rename('eto')


def get_annual_peff_pcml(year: int) -> ee.Image:
    """
    Calculate annual effective precipitation using the USDA-SCS method.
    Author: Christopher Pearson (chris.pearson@dri.edu)
    Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

    Args:
    year (int): Year in YYYY format for which the annual effective precipitation is calculated
    Returns:
    ee.Image: Annual effective precipitation using the PCML method.
    """

    peff_pcml_img = ee.Image('projects/ee-peff-westus-unmasked/assets/effective_precip_monthly_unmasked')
    peff_pcml_projection = peff_pcml_img.projection()
    if 2000 <= year <= 2023:
        # Select monthly bands bYYYY_1 through bYYYY_12 and sum to get annual total
        peff_pcml_bands = [f'b{year}_{m}' for m in range(1, 13)]
        annual_peff = peff_pcml_img.select(peff_pcml_bands).reduce(ee.Reducer.sum())
    elif year == 2024:
        # 2024 only has months 1-9; use mean monthly (2000-2023) for months 10-12
        actual_sum = peff_pcml_img.select([f'b2024_{m}' for m in range(1, 10)]) \
            .reduce(ee.Reducer.sum())
        # For fill months, select all 24 year-bands at once and reduce (stays as single-image op)
        fill_sum = ee.Image.constant(0).setDefaultProjection(peff_pcml_projection)
        for m in range(10, 13):
            fill_sum = fill_sum.add(
                peff_pcml_img.select([f'b{y}_{m}' for y in range(2000, 2024)])
                    .reduce(ee.Reducer.mean())
            )
        annual_peff = actual_sum.add(fill_sum)
    else:
        # Use mean annual peff_pcml across 2000-2023 for years outside that range
        # Sum all 288 bands (24 years x 12 months) and divide by 24 — avoids ImageCollection
        all_bands = [f'b{y}_{m}' for y in range(2000, 2024) for m in range(1, 13)]
        annual_peff = peff_pcml_img.select(all_bands).reduce(ee.Reducer.sum()).divide(24)
    annual_peff = annual_peff.rename('annual_peff_mm').set('year', year) \
        .setDefaultProjection(peff_pcml_projection)
    return annual_peff


def download_gee_tif(
        data_bands: list[ee.Image],
        data_band_name_list: list[str],
        local_file_name: str,
        ee_geom: ee.Geometry,
        gee_scale: float,
        crs: str,
        verbose: bool = False
):
    """
    Download multiple GEE data bands as a single GeoTIFF file. 
    This function handles the download of multiple bands, applies appropriate reducers for categorical and 
    continuous data, and retries downloads in case of failures.

    Args:
    data_bands (list[ee.Image]): List of GEE Image bands to download.
    data_band_name_list (list[str]): List of band names corresponding to the data_bands.
    local_file_name (str): Local file name (including path) to save the downloaded GeoTIFF.
    ee_geom (ee.Geometry): Earth Engine geometry defining the region to download.
    gee_scale (float): Scale in meters for the GEE data download.
    crs (str): Coordinate reference system for the downloaded data.
    verbose (bool): Set True to see extra details on file downloads.

    Returns:
        None. The function saves the downloaded data as a GeoTIFF file at the specified local_file_name.
    """

    year = int(local_file_name.split('.tif')[0].split('_')[-1])
    band_scale_dict = {
        'annual_et_ensemble_mm': 800 if year < 2000 else 30 if year <= 2025 else 4638.3,
        'annual_eto_mm': 4638.3,
        'annual_precip_mm': 4638.3,
        'annual_peff_mm': 4638.3,
        'annual_peff_pcml_mm': 2200,
        'annual_tmmx_K': 4638.3,
        'annual_tmmn_K': 4638.3,
        'lulc': 30 if 1985 <= year <= 2025 else 250,
        'soil_depth_mm': 800,
        'awc_mm': 90,
        'ksat_mean_micromps': 800,
        'annual_gw_fraction': 60,
        'annual_crop_fraction': 30 if 1985 <= year <= 2025 else 250,
        'annual_irr_fraction': 30
    }
    data_img = data_bands[0].rename(data_band_name_list[0])
    band_scale = band_scale_dict[data_band_name_list[0]]
    max_pixels = max(64, np.ceil((gee_scale / band_scale) ** 2))
    data_img = data_img.setDefaultProjection(
        crs=crs,
        scale=band_scale
    ).reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=max_pixels
    ).reproject(crs, scale=gee_scale)

    for band, band_name in zip(data_bands, data_band_name_list):
        band = band.rename(band_name)
        band_scale = band_scale_dict[band_name]
        if gee_scale > 30:
            if band_name == 'lulc':
                reducer = ee.Reducer.mode(maxRaw=max_pixels)
            elif band_name in ['annual_irr_fraction', 'annual_crop_fraction']:
                reducer = ee.Reducer.count()
            elif band_scale < 1000:
                reducer = ee.Reducer.mean()
            else:
                reducer = None
            if reducer:
                max_pixels = max(64, np.ceil((gee_scale / band_scale) ** 2))
                band = band.setDefaultProjection(
                    crs=crs,
                    scale=band_scale
                ).reduceResolution(
                    reducer=reducer,
                    maxPixels=max_pixels
                ).reproject(crs, scale=gee_scale)
            else:
                band = band.setDefaultProjection(
                    crs=crs,
                    scale=band_scale
                ).reproject(crs, scale=gee_scale)
        if band_name in ['annual_irr_fraction', 'annual_crop_fraction']:
            band = band.multiply(band_scale ** 2).divide(gee_scale ** 2)
            band = band.where(band.gt(1), 1)
        if 'precip' in band_name:
            precip_band = band
        if 'eto' in band_name:
            eto_band = band
        if 'peff' in band_name:
            # make sure these are clamped using precip and eto
            band = band.where(band.gt(precip_band), precip_band) \
                .where(band.gt(eto_band), eto_band) \
                .clamp(0, 10000)
        data_img = data_img.addBands(band, overwrite=True)
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                logger.info('Downloading %s (attempt %d/%d)...', local_file_name, attempt, max_retries)
            gee_url = data_img.getDownloadUrl({
                'scale': gee_scale,
                'region': ee_geom,
                'format': 'GEO_TIFF',
                'crs': crs
            })
            r = requests.get(gee_url,
                allow_redirects=True,
                timeout=300
            )
            if r.status_code != 200:
                error_body = r.text[:1000]
                if verbose:
                    logger.warning(
                        'GEE returned HTTP %d for %s: %s',
                        r.status_code, local_file_name, error_body
                    )
            r.raise_for_status()
            # Check response content-type and TIFF magic bytes
            content_type = r.headers.get('Content-Type', '')
            if 'tiff' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                error_msg = r.content[:500].decode('utf-8', errors='replace')
                if verbose:
                    logger.warning(
                        'GEE returned non-TIFF response for %s (Content-Type: %s): %s',
                        local_file_name, content_type, error_msg
                    )
                raise ValueError(f'Invalid Content-Type: {content_type}')
            # Verify TIFF magic bytes (II for little-endian or MM for big-endian)
            if len(r.content) < 4 or r.content[:2] not in (b'II', b'MM'):
                error_msg = r.content[:500].decode('utf-8', errors='replace')
                if verbose:
                    logger.warning(
                        'Downloaded content is not a valid TIFF for %s (first bytes: %s): %s',
                        local_file_name, r.content[:4], error_msg
                    )
                raise ValueError('Downloaded content does not have valid TIFF header')
            with open(local_file_name, 'wb') as fd:
                fd.write(r.content)
            # Validate the downloaded file
            tile_arr, tile_rio = read_raster_as_arr(local_file_name, change_dtype=False)
            if tile_arr.size == 0:
                if verbose:
                    logger.warning('Downloaded file is empty. Deleting %s', local_file_name)
                tile_rio.close()
                os.remove(local_file_name)
                raise rio.errors.RasterioIOError
            # Detect all-zero tiles (GEE returned valid TIFF with no data)
            if np.all(tile_arr == 0):
                if verbose:
                    logger.warning('Downloaded file has all-zero data. Deleting %s', local_file_name)
                tile_rio.close()
                os.remove(local_file_name)
                raise ValueError('All-zero tile data')
            tile_rio.close()
            clamp_and_rewrite_raster(
                local_file_name,
                min_val=0,
                band_descriptions=data_band_name_list
            )
            break  # Success
        except (
                ee.EEException, requests.exceptions.RequestException,
                requests.exceptions.ConnectionError, RemoteDisconnected,
                rio.errors.RasterioIOError, ValueError, Exception
        ) as e:
            if os.path.exists(local_file_name):
                os.remove(local_file_name)
            if attempt == max_retries:
                if verbose:
                    logger.warning('Failed to download %s after %d attempts: %s', local_file_name, max_retries, e)
                return
            if verbose:
                logger.warning('Error %s during %s download (attempt %d/%d). Retrying...',
                               e, local_file_name, attempt, max_retries)
        time.sleep(0.01 * attempt)


def _get_lulc_image(year, crs, lulc_scenario, usgs_lulc_historical_ic,
                    lulc_projection_raw_ic, lulc_projection_ensemble_ic, nlcd_ic):
    """Select the appropriate LULC image for a given year.

    Returns an ee.Image with a single 'lulc' band recoded to
    1=AGRI, 2=URBAN, 3=SW (0 = masked out).
    """
    if year < 1985 or year > 2025:
        if year < 1985:
            usgs_lulc_ic = usgs_lulc_historical_ic
        elif lulc_scenario is not None:
            usgs_lulc_ic = lulc_projection_raw_ic.filterMetadata(
                'scenario', 'equals', lulc_scenario
            )
        else:
            usgs_lulc_ic = lulc_projection_ensemble_ic
        usgs_lulc_year = year if year >= 1938 else 1938
        usgs_lulc = usgs_lulc_ic \
            .filterDate(f'{usgs_lulc_year}-01-01', f'{usgs_lulc_year + 1}-01-01') \
            .first() \
            .setDefaultProjection(crs=crs, scale=250)
        usgs_mask = usgs_lulc.eq(13).Or(usgs_lulc.eq(2)).Or(
            usgs_lulc.eq(6)).Or(usgs_lulc.eq(1))
        return usgs_lulc.updateMask(usgs_mask).remap(
            [13, 2, 6, 1],
            [1, 2, 2, 3]  # 1: AGRI, 2: URBAN, 3: SW
        ).rename('lulc')
    else:
        nlcd_year = year if year <= 2024 else 2024
        nlcd_img = nlcd_ic.filterDate(
            f'{nlcd_year}-01-01', f'{nlcd_year + 1}-01-01'
        ).first()
        nlcd_mask = nlcd_img.eq(82).Or(
            nlcd_img.gte(21).And(nlcd_img.lte(24))
        ).Or(nlcd_img.eq(11))
        return nlcd_img.updateMask(nlcd_mask).remap(
            [82, 21, 22, 23, 24, 11],
            [1, 2, 2, 2, 2, 3]  # 1: AGRI, 2: URBAN, 3: SW
        ).rename('lulc')


def _get_climate_images(year, gcm, prism_ic, prism_hargreaves_eto_ic,
                        maca_monthly_eto_ic, openet_ic_v2, openet_ic_v2_1,
                        usgs_adjusted_et_ic, maca_monthly_et_ic, _ASSET_PREFIX):
    """Return (actual_et, annual_eto, precip, tmmx, tmmn) for a given year."""
    start_year_gee = f'{year}-01-01'
    end_year_gee = f'{year + 1}-01-01'

    # --- ET ---
    openet_ic = openet_ic_v2 if 2000 <= year <= 2024 else openet_ic_v2_1 if year == 2025 else None
    if 2000 <= year <= 2025:
        actual_et = openet_ic \
            .filterDate(start_year_gee, end_year_gee) \
            .select('et_ensemble_mad').sum()
    elif year < 2000:
        actual_et = usgs_adjusted_et_ic \
            .filterDate(start_year_gee, end_year_gee) \
            .select('actual_et').sum()
    else:
        if gcm is not None:
            actual_et = ee.Image(
                f'{_ASSET_PREFIX}/maca_gcm_annual_et/{gcm}_{year}'
            ).select('actual_et')
        else:
            actual_et = maca_monthly_et_ic \
                .filterDate(start_year_gee, end_year_gee) \
                .select('actual_et').sum()

    # --- ETo ---
    if year < 2026:
        prism_eto_ic = prism_hargreaves_eto_ic.filterDate(start_year_gee, end_year_gee)
        annual_eto = get_eto(year, prism_eto_ic=prism_eto_ic, return_annual=True)
    else:
        if gcm is not None:
            annual_eto = ee.Image(
                f'{_ASSET_PREFIX}/maca_gcm_annual_eto/{gcm}_{year}'
            ).select('eto').rename('annual_eto_mm')
        else:
            annual_eto = maca_monthly_eto_ic \
                .filterDate(start_year_gee, end_year_gee) \
                .select('eto').sum().rename('annual_eto_mm')

    # --- Precip & Temp ---
    if year < 2026:
        precip = prism_ic.select('ppt') \
            .filterDate(start_year_gee, end_year_gee) \
            .sum()
        tmmx = prism_ic.select('tmax') \
            .filterDate(start_year_gee, end_year_gee) \
            .mean() \
            .add(273.15)
        tmmn = prism_ic.select('tmin') \
            .filterDate(start_year_gee, end_year_gee) \
            .mean() \
            .add(273.15)
    else:
        maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
            .filterDate(start_year_gee, end_year_gee)
        if gcm is not None:
            maca_ic = maca_ic.filter(ee.Filter.eq('model', gcm))
            n_members = 2
        else:
            maca_scenarios, maca_models, _ = get_maca_scenarios_models_bands()
            n_members = len(maca_models) * len(maca_scenarios)
        precip = maca_ic.select('pr').sum().divide(n_members)
        tmmx = maca_ic.select('tasmax').mean()
        tmmn = maca_ic.select('tasmin').mean()

    return actual_et, annual_eto, precip, tmmx, tmmn


def download_gee_tile(
        tile_values: tuple[int, float, float, float],
        download_dir: str,
        year_list: list,
        data_band_names: list[str],
        gcloud_project: str = 'azhydro',
        gee_scale: float = 30,
        verbose: bool = False,
        crs: str = 'EPSG:4326',
        gcm: str | None = None,
        lulc_scenario: str | None = None,
):
    """
    Download GEE tile through dask.

    Args:
    tile_values (tuple (int, float, float, float)): Tile values as a tuple of (FID, xmin, ymin, xmax, ymax)
    download_dir (str): Download directory path.
    year_list (list): List of years in YYYY format.
    data_band_names (list (str, ...)): List of data bands as strings.
    gcloud_project (str): GCloud project name.
    gee_scale (float): GEE data download scale in m.
    verbose (bool): Set True to see extra details on file downloads.
    crs (str): Coordinate reference system for the downloaded data. Defaults to 'EPSG:4326'.
    gcm (str | None): When provided, use per-GCM climate data instead of
        the 40-member ensemble mean for future years (2026-2099).  Must be
        one of the 5 representative GCMs.
    lulc_scenario (str | None): When provided, use a single USGS LULC
        projection scenario (e.g. 'B1', 'B2', 'A1B', 'A2') instead of
        the pixel-wise mode ensemble for future years (2026-2099).

    Returns:
        None.
    """

    if min(year_list) < 1896 or max(year_list) > 2099:
        raise ValueError('Year list should be between 1896 and 2099')
    retry_ee_init = True
    while retry_ee_init:
        try:
            ee.Initialize(
                project=gcloud_project,
                opt_url='https://earthengine-highvolume.googleapis.com'
            )
            retry_ee_init = False
        except (ee.EEException, requests.exceptions.RequestException,
                requests.exceptions.ConnectionError, Exception
                ) as e:
            if verbose:
                logger.warning('Initialization exception: %s', e)
            retry_ee_init = True
            time.sleep(1)

    openet_ic_v2 = ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0')
    openet_ic_v2_1 = ee.ImageCollection('projects/openet/assets/ensemble/conus/gridmet/monthly/v2_1')
    prism_ic = ee.ImageCollection('OREGONSTATE/PRISM/ANm') # using 4 km to match MACA future scenarios
    irrmapper_ic = ee.ImageCollection('UMT/Climate/IrrMapper_RF/v1_2')
    # Pre-exported GEE asset collections
    _ASSET_PREFIX = 'projects/azhydro/assets'
    prism_hargreaves_eto_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/prism_hargreaves_eto')
    usgs_adjusted_et_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/usgs_adjusted_et')
    maca_monthly_eto_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_eto_v2')
    maca_monthly_et_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_et_v2')
    lulc_projection_ensemble_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/lulc_projection_ensemble')
    lulc_projection_raw_ic = ee.ImageCollection('projects/nwi-usgs/assets/USGS-LULC-CONUS')
    monthly_peff_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/monthly_peff_v2')
    gw_fraction_img_dict = {
        '2000': ee.Image("projects/fwhung/assets/CroplandNew60m/Crop_IrrSource_2000"),
        '2005': ee.Image("projects/fwhung/assets/CroplandNew60m/Crop_IrrSource_2005"),
        '2010': ee.Image("projects/fwhung/assets/CroplandNew60m/Crop_IrrSource_2010"),
        '2015': ee.Image("projects/fwhung/assets/CroplandNew60m/Crop_IrrSource_2015")
    }
    nlcd_ic = ee.ImageCollection('projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER')
    usgs_lulc_historical_ic = ee.ImageCollection('projects/nwi-usgs/assets/USGS-LULC-CONUS').filterMetadata(
        'scenario', 'equals', 'Historical'
    )
    soil_depth = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/soil_depth'
    ).multiply(10).rename('soil_depth_mm')
    awc = ee.Image(
        'projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite'
    ).multiply(25.4).rename('awc_mm')
    ksat_mean = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/physical/ksat_mean'
    ).rename('ksat_mean_micromps')
    ee_geom = ee.Geometry.Rectangle(tile_values[1:])
    peff_pcml_1896 = None
    for year in year_list:
        local_file_name = os.path.join(download_dir, f'Tile_{tile_values[0]}_{year}.tif')
        if os.path.exists(local_file_name):
            try:
                tile_arr, tile_rio = read_raster_as_arr(local_file_name)
                if tile_arr.size == 0:
                    if verbose:
                        logger.warning('Downloaded file is empty. Deleting %s', local_file_name)
                    tile_rio.close()
                    os.remove(local_file_name)
                    raise rio.errors.RasterioIOError
                if np.all(tile_arr == 0):
                    if verbose:
                        logger.warning('Existing file has all-zero data. Deleting %s', local_file_name)
                    tile_rio.close()
                    os.remove(local_file_name)
                    raise rio.errors.RasterioIOError
                tile_rio.close()
                continue
            except rio.errors.RasterioIOError:
                if os.path.exists(local_file_name):
                    os.remove(local_file_name)
                if verbose:
                    logger.warning('Existing file %s is corrupted. Deleted and retrying download...', local_file_name)
        irrmapper_year = year if 1985 <= year <= 2025 else 1985 if year < 1985 else 2025
        irr = irrmapper_ic.filterDate(f'{irrmapper_year}-01-01', f'{irrmapper_year + 1}-01-01') \
            .select('classification') \
            .max()
        mask = irr.eq(0)
        irr_mask = irr.updateMask(mask).remap([0], [1])
        start_year_gee = f'{year}-01-01'
        end_year_gee = f'{year + 1}-01-01'
        actual_et, annual_eto, precip, tmmx, tmmn = _get_climate_images(
            year, gcm, prism_ic, prism_hargreaves_eto_ic,
            maca_monthly_eto_ic, openet_ic_v2, openet_ic_v2_1,
            usgs_adjusted_et_ic, maca_monthly_et_ic, _ASSET_PREFIX,
        )
        if gcm is not None and year > 2025:
            peff = ee.Image(
                f'{_ASSET_PREFIX}/maca_gcm_annual_peff/{gcm}_{year}'
            ).select('peff').rename('annual_peff_mm')
        else:
            peff = monthly_peff_ic.filterDate(start_year_gee, end_year_gee) \
                    .select('peff').sum().rename('annual_peff_mm')
        if year == 1896:
            # get mean annual 2000-2023 if year is 1896, and save it for use in other years outside 2000-2024
            # since PCML data only starts in 2000
            peff_pcml = get_annual_peff_pcml(1896)
            peff_pcml_1896 = peff_pcml
        elif 2000 <= year <= 2024:
            peff_pcml = get_annual_peff_pcml(year)
        else:
            peff_pcml = peff_pcml_1896

        lulc = _get_lulc_image(
            year, crs, lulc_scenario,
            usgs_lulc_historical_ic, lulc_projection_raw_ic,
            lulc_projection_ensemble_ic, nlcd_ic,
        )
        if year < 2005:
            gw_fraction = gw_fraction_img_dict['2000']
        elif 2005 <= year < 2010:
            gw_fraction = gw_fraction_img_dict['2005']
        elif 2010 <= year < 2015:
            gw_fraction = gw_fraction_img_dict['2010']
        else:
            gw_fraction = gw_fraction_img_dict['2015']
        gw_fraction = gw_fraction.select('GWIrrFraction').rename('annual_gw_fraction')
        # calculate crop fraction from lulc
        crop_mask = lulc.eq(1)
        crop_mask = lulc.updateMask(crop_mask).rename('annual_crop_fraction')
        data_bands = [
            actual_et,
            annual_eto,
            precip,
            peff,
            peff_pcml,
            tmmx,
            tmmn,
            lulc,
            soil_depth,
            awc,
            ksat_mean,
            gw_fraction,
            crop_mask,
            irr_mask
        ]
        download_gee_tif(
            data_bands=data_bands,
            data_band_name_list=data_band_names,
            local_file_name=local_file_name,
            ee_geom=ee_geom,
            gee_scale=gee_scale,
            crs=crs,
            verbose=verbose
        )
        if not os.path.exists(local_file_name):
            logger.warning('Tile %s was not created after all retries', local_file_name)


def download_gee_data(
        geom_file: str,
        gcloud_project: str = 'azhydro',
        gcloud_bucket: str = 'azhydro',
        download_dir: str = '../../Data/Inputs/GEE_Data/',
        start_year: int = 1985,
        end_year: int = 2023,
        skip_download: bool = False,
        tile_size: int = 10000,
        num_workers: int = 32,
        worker_memory='0.5G',
        gee_scale: float = 2000,
        verbose: bool = False,
        gcm: str | None = None,
        lulc_scenario: str | None = None,
) -> tuple[str, list[str]]:
    """
    Download multiple GEE datasets as rasters at 100 m spatial resolution.

    Args:
    geom_file (str): Area of interest shapefile or geojson path. E.g., the AZ state geojson or shapefile
    gcloud_project (str): Name of the Google Cloud Project. This should have the GEE API service enabled.
    gcloud_bucket (str): Name of the GCloud bucket.
    download_dir (str): Download directory
    start_year (int): Start year in YYYY format
    end_year (int): End year in YYYY format
    skip_download (bool): Set True to skip downloading and return existing CSV files.
    tile_size (int): Tile size for downloading GEE Data. Default is 10000 m.
    num_workers (int): Number of tiles to parallely download. Note that both tile_size and num_workers have quotas for
    free GEE users.
    gee_scale (float): GEE data download scale in m.
    verbose (bool): Set True to see extra details on file downloads.
    gcm (str | None): When provided, download per-GCM climate data for
        future years (2026-2099) instead of the ensemble mean.  Tiles are
        stored in a GCM-specific subdirectory.
    lulc_scenario (str | None): When provided, download per-scenario LULC
        projection data for future years (2026-2099) instead of the
        pixel-wise mode ensemble.  Tiles are stored in a scenario-specific
        subdirectory.

    Returns:
        tuple (str, list (str, ...)): Tuple containing the directory path containing all the downloaded GEE tiles and
        the ordered list of band names for each tile.
    """

    gcm_suffix = f'_{gcm}' if gcm is not None else ''
    lulc_suffix = f'_LULC_{lulc_scenario}' if lulc_scenario is not None else ''
    data_dir = os.path.join(download_dir, f'GEE_Data/GEE_Tiles_{int(gee_scale)}m{gcm_suffix}{lulc_suffix}')
    data_band_names = [
        'annual_et_ensemble_mm',
        'annual_eto_mm',
        'annual_precip_mm',
        'annual_peff_mm',
        'annual_peff_pcml_mm',
        'annual_tmmx_K',
        'annual_tmmn_K',
        'lulc',
        'soil_depth_mm',
        'awc_mm',
        'ksat_mean_micromps',
        'annual_gw_fraction',
        'annual_crop_fraction',
        'annual_irr_fraction'
    ]
    if not skip_download:
        makedirs(data_dir)
        logger.info('Downloading GEE data...')
        ee.Initialize(
            project=gcloud_project,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
        gee_crs = 'EPSG:4326'
        area_gdf = gpd.read_file(geom_file)
        ee_geom = ee.Geometry.Rectangle(area_gdf.to_crs(gee_crs).total_bounds.tolist())
        huc12_path = 'AZ_HUC12.geojson'
        huc12_local_path = os.path.join(download_dir, 'GEE_Data', f'{huc12_path}')
        if not os.path.exists(huc12_local_path):
            huc12_fc = ee.FeatureCollection('USGS/WBD/2017/HUC12').filterBounds(ee_geom)
            task = ee.batch.Export.table.toCloudStorage(
                collection=huc12_fc,
                description='AZ HUC12',
                bucket=gcloud_bucket,
                fileNamePrefix=huc12_path,
                fileFormat='GeoJSON'
            )
            task.start()
            logger.info('Waiting to download the HUC12 polygons over the study area from GCloud...')
            while task.active():
                continue
            storage_client = storage.Client()
            storage_bucket = storage_client.bucket(gcloud_bucket)
            gcloud_blob = storage.Blob(bucket=storage_bucket, name=huc12_path)
            logger.info(f'Downloading {huc12_path}')
            gcloud_blob.download_to_filename(huc12_local_path)
        year_list = list(range(start_year, end_year + 1))
        logger.info(f'Creating Fishnet with {tile_size} m tile size...')
        fishnet_gdf = create_fishnet(
            area_gdf,
            fishnet_size=tile_size,
            output_dir=os.path.join(download_dir, 'GW_Data')
        ).to_crs(gee_crs)
        tile_val_list = []
        for _, tile in fishnet_gdf.iterrows():
            tile_gdf = gpd.GeoDataFrame(data={
                'geometry': [tile.geometry],
                'FID': [tile.FID]
            })
            xmin, ymin, xmax, ymax = tile_gdf.total_bounds.tolist()
            fid = int(tile_gdf.iloc[0]['FID'])
            tile_val_list.append((fid, xmin, ymin, xmax, ymax))
        if verbose:
            logger.info('Each tile has the following bands in order:')
            for band_idx, band_name in enumerate(data_band_names):
                logger.info(f'{band_idx + 1} {band_name}')
        tile_chunks = generate_chunks(tile_val_list, num_workers)
        num_chunks = int(np.ceil(fishnet_gdf.shape[0] / num_workers))
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit=worker_memory)
        dask_cluster.scale(num_workers)
        dask_client = Client(dask_cluster)
        dask_client.wait_for_workers(1)
        logger.info(f'Using {num_workers} local workers...')
        tile_start = 1
        for chunk_idx, tile_chunk in enumerate(tqdm(tile_chunks, desc='Processing tile chunks', total=num_chunks), 1):
            tile_end = tile_start + len(tile_chunk) - 1
            logger.info(f'Processing Tiles {tile_start}-{tile_end} in Chunk {chunk_idx}')
            tile_start = tile_end + 1
            compute(
                delayed(download_gee_tile)(
                    tile_vals, data_dir, year_list,
                    data_band_names, gcloud_project,
                    gee_scale,
                    verbose=verbose,
                    gcm=gcm,
                    lulc_scenario=lulc_scenario,
                )
                for tile_vals in tile_chunk
            )
        try:
            dask_client.close()
            dask_cluster.close()
        except (TimeoutError, OSError):
            logger.warning('Dask cluster cleanup timed out (all tasks completed successfully)')
    return data_dir, data_band_names


def mosaic_tiles(
        input_tile_dir: str,
        output_dir: str,
        start_year: int = 1985,
        end_year: int = 2023,
        output_prefix: str = 'Predictor',
        already_mosaicked: bool = False,
        fishnet_file: str | None = None,
) -> None:
    """
    Mosaic all tiles based on the start and end years.

    Args:
        input_tile_dir (str): Input tile directory. The naming convention is Tile_<tile_number>_<year>.tif.
        output_dir (str): Output directory.
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        output_prefix (str): Output prefix name to append to output files.
        already_mosaicked (bool): Set True to skip mosaicking.
        fishnet_file (str | None): Path to the fishnet GeoJSON used to
            determine the expected tile count.  When provided, a warning
            is logged for any year whose tile count is lower than
            expected.

    Returns:
        None.
    """

    def _mosaic_year(year):
        tile_list = glob(os.path.join(input_tile_dir, f'*{year}.tif'))
        # Warn about missing tiles by comparing to the full set from the
        # first year (all years should have the same tile count).
        if expected_tile_count and len(tile_list) < expected_tile_count:
            logger.warning(
                'Year %d: only %d of %d tiles found — mosaic will have zero-filled gaps',
                year, len(tile_list), expected_tile_count,
            )
        merged_tif = os.path.join(output_dir, f'{output_prefix}_{year}.tif')
        if os.path.exists(merged_tif):
            os.remove(merged_tif)
        subprocess.call(
            [gdal_merge_path, '-o', merged_tif, '-of', 'GTiff', '-init', '0'] + tile_list,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Preserve band descriptions from source tile
        if tile_list:
            with rio.open(tile_list[0]) as src:
                descriptions = src.descriptions
            if any(descriptions):
                with rio.open(merged_tif, 'r+') as dst:
                    for idx, desc in enumerate(descriptions, start=1):
                        if desc:
                            dst.set_band_description(idx, desc)

    if not already_mosaicked:
        makedirs(output_dir)
        logger.info('Mosaicking tiles...')
        gdal_merge_path = shutil.which('gdal_merge.py')
        if gdal_merge_path is None:
            raise FileNotFoundError('gdal_merge.py not found on PATH')
        # Determine expected tile count from the fishnet
        expected_tile_count = 0
        if fishnet_file and os.path.exists(fishnet_file):
            expected_tile_count = len(gpd.read_file(fishnet_file))
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_mosaic_year)(year)
            for year in range(start_year, end_year + 1)
        )


def reproject_gee_mosaics(
        gee_mosaic_dir: str,
        output_dir: str,
        gw_data_dir: str,
        already_reprojected: bool = False
) -> None:
    """
    Reproject all predictor data mosaics using groundwater withdrawal rasters.

    Args:
        gee_mosaic_dir (str): Path to the GEE mosaics.
        output_dir (str): Output directory to store the reprojected rasters.
        gw_data_dir (str): Groundwater depth raster directory.
        already_reprojected (bool): Set True to skip reprojecting the mosaics.

    Returns:
        None.
    """

    if not already_reprojected:
        makedirs(output_dir)
        gee_mosaics = glob(os.path.join(gee_mosaic_dir, '*.tif'))
        gw_ref = glob(os.path.join(gw_data_dir, '*.tif'))[0]
        for gee_mosaic in gee_mosaics:
            gee_mosaic_reproj = os.path.join(output_dir, f'{gee_mosaic[gee_mosaic.rfind(os.sep) + 1:]}')
            reproject_raster_gdal(
                gee_mosaic,
                gee_mosaic_reproj,
                from_raster=gw_ref
            )


def create_az_data_parquet(
        input_file_dir: str,
        gw_data_dir: str,
        output_dir: str,
        data_band_names: list[str],
        gw_basin_vector: str,
        start_year: int = 1985,
        end_year: int = 2023,
        exclude_years: list[int] | None = None,
        lu_smoothing: int = 3,
        load_parquet: bool = False,
        subbasin_vector: str | None = None,
) -> pd.DataFrame:
    """
    Create Arizona predictor data Parquet.

    Args:
        input_file_dir (str): Input directory where the file names follow <Variable>_<Year>, e.g, Predictor_2015.tif.
        gw_data_dir (str): Path to the GW pumping depth rasters with GW_<Year>.tif as the file name format.
        output_dir (str): Output directory.
        data_band_names (list (str, ...)): List of data bands as strings.
        gw_basin_vector (str): Path to the Arizona GW basin shapefile or geojson.
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        exclude_years (list(int, ...) or None): Exclude these years from the dataframe.
        lu_smoothing (int): Gaussian smoothing window size for land use rasters obtained from CDL.
        load_parquet (bool): Set True to load existing Parquet.
        subbasin_vector (str | None): Path to the ADWR Groundwater Sub-basin
            shapefile.  When provided, ``GW_Subbasin`` OBJECTID values are
            mapped to ``SUBBASIN_N`` names (AMA/INA only; others get
            ``'NO_SUBBASIN'``).

    Returns:
        pd.DataFrame: Output dataframe.
    """

    data_parquet = os.path.join(output_dir, 'AZ_Data.parquet')
    if not load_parquet:
        makedirs(output_dir)
        if exclude_years is None:
            exclude_years = []
        data_df_parts = []
        var_names = [
            'Predictor',
            'GW_Basin',
            'GW_Subbasin',
            'Streamflow',
            'Canal_Weighted_Streamflow',
            'Canal_Density',
            'Well_Density'
        ]
        for year in range(start_year, end_year + 1):
            df = pd.DataFrame()
            if year not in exclude_years:
                for var_name in var_names:
                    raster_file = os.path.join(input_file_dir, f'{var_name}_{year}.tif')
                    if var_name == 'Predictor':
                        for band_num, band_name in enumerate(data_band_names):
                            try:
                                raster_arr = read_raster_as_arr(
                                    raster_file,
                                    band=band_num + 1,
                                    get_file=False
                                )
                                if band_name == 'lulc':
                                    df = create_land_use_data(
                                        df, raster_arr,
                                        smoothing=lu_smoothing
                                    )
                                else:
                                    df[band_name] = raster_arr.ravel()
                            except IndexError:
                                logger.warning('IndexError for %s', raster_file)
                                df[band_name] = 0
                    else:
                        raster_arr = read_raster_as_arr(raster_file, get_file=False).ravel()
                        if var_name == 'Streamflow':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['streamflow_mm'] = raster_arr
                        elif var_name == 'Canal_Weighted_Streamflow':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['canal_weighted_streamflow_mm'] = raster_arr
                        elif var_name == 'Canal_Density':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['canal_density'] = raster_arr
                        elif var_name == 'Well_Density':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['well_density'] = raster_arr
                        elif var_name == 'GW':
                            raster_arr[np.isnan(raster_arr)] = 0
                        else:
                            df[var_name] = raster_arr
                # We will later predict GW pumping for years outside 1984-2024
                gw_file = os.path.join(gw_data_dir, f'GW_{year}.tif') if 1984 <= year <= 2024 else sorted(
                    glob(os.path.join(gw_data_dir, 'GW_*.tif')))[0]
                df['gw_pumping_mm'] = read_raster_as_arr(gw_file, get_file=False).ravel()
                lon_grid, lat_grid = get_xy_grids_from_raster(gw_file)
                df['easting_m'] = lon_grid.ravel()
                df['northing_m'] = lat_grid.ravel()
                df['Year'] = year
                data_df_parts.append(df)
        data_df = pd.concat(data_df_parts, ignore_index=True) if data_df_parts else pd.DataFrame()
        # Set annual_irr_fraction to 0 where annual_crop_fraction is 0
        data_df.loc[data_df.annual_crop_fraction == 0, 'annual_irr_fraction'] = 0.0
        # Fit a linear regression of annual_irr_fraction on annual_crop_fraction using
        # 1985-2025 IrrMapper data (non-zero crop fraction only), then predict for years
        # outside that range. Where crop fraction is zero, irr fraction is set to zero.
        irr_mask = data_df.Year.between(1985, 2025)
        train = data_df.loc[irr_mask].dropna(subset=['annual_irr_fraction', 'annual_crop_fraction'])
        train = train[train.annual_crop_fraction > 0]
        irr_reg = LinearRegression().fit(
            train[['annual_crop_fraction']].values,
            train['annual_irr_fraction'].values
        )
        for mask in [data_df.Year < 1985, data_df.Year > 2025]:
            data_df.loc[mask, 'annual_irr_fraction'] = 0.0
            nonzero = mask & (data_df.annual_crop_fraction > 0)
            pred = irr_reg.predict(data_df.loc[nonzero, ['annual_crop_fraction']].values)
            data_df.loc[nonzero, 'annual_irr_fraction'] = np.clip(pred, 0, 1)

        gw_basin_gdf = gpd.read_file(gw_basin_vector)
        gw_basin_dict = {}
        ama_ina_basins = get_ama_ina_basin_names()
        for gw_basin in gw_basin_gdf.OBJECTID:
            gw_basin_name = gw_basin_gdf[gw_basin_gdf.OBJECTID == gw_basin].BASIN_NAME.values[0]
            gw_basin_dict[gw_basin] = gw_basin_name
        nan_str = 'OUTSIDE AZ'
        gw_basin_dict[0] = nan_str
        data_df = data_df.reset_index(drop=True)
        data_df.GW_Basin = data_df.GW_Basin.swifter.apply(
            lambda x: gw_basin_dict[x] if not np.isnan(x) else nan_str)
        data_df = data_df[data_df.GW_Basin != nan_str]
        ama_basins = [b for b in ama_ina_basins if 'AMA' in b]
        ina_basins = [b for b in ama_ina_basins if b not in ama_basins]
        data_df = data_df.reset_index(drop=True)
        data_df['GW_Basin_Type'] = data_df.GW_Basin.swifter.apply(
            lambda x: 0 if x in ama_basins else 1 if x in ina_basins else 2
        ).reset_index(drop=True)

        # Map GW_Subbasin OBJECTID → SUBBASIN_N
        if subbasin_vector is not None:
            sub_gdf = gpd.read_file(subbasin_vector)
            oid_to_name = dict(zip(sub_gdf['OBJECTID'], sub_gdf['SUBBASIN_N']))
            data_df['GW_Subbasin'] = data_df['GW_Subbasin'].swifter.apply(
                lambda x: oid_to_name.get(x, 'NO_SUBBASIN')
                if not np.isnan(x) else 'NO_SUBBASIN'
            )
            logger.debug('GW_Subbasin unique values: %s',
                        sorted(data_df.GW_Subbasin.unique()))

        data_df.to_parquet(data_parquet, index=False)
    else:
        data_df = pd.read_parquet(data_parquet)
    return data_df


def split_data_train_test(
        input_df: pd.DataFrame,
        pred_attr: str = 'gw_pumping_mm',
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = 0.2,
        test_year: tuple[int, ...] | bool = True,
        test_gw_basins: tuple[str, ...] = (),
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        split_strategy: int = 1
) -> tuple[pd.DataFrame, ...]:
    """Split data yearly, randomly, or based on train-test percentage based on year or crop. For the last option,
    by default test_size amount of data is kept from each year for testing.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        shuffle (bool): Default True for shuffling.
        random_state (int): Random state used during train test split.
        test_size (float): Test data size (0<=test_size<=1).
        test_year (tuple(int, ...) or bool): If split_strategy = 1, then this needs to be a tuple of years in YYYY
                                             format, i.e., (2014, 2015), and the test data is created from these years.
                                             For split_strategy=2, set this to True if the test data needs to be created
                                             based on year_col.
        test_gw_basins (tuple (str, ...)): Build test data from only these tuple of groundwater basins.
        year_col (str): Name of the year column.
        gw_basin_col (str): Name of the GW basin column.
        split_strategy (int): If 1, Split train test data based on year_col. If 2, then test_size amount of data from
                              year_col are kept for testing and rest for training;
                              for this option, test-year should have a tuple of integers or a True value. If 3, then
                              test_gw_basins are used for spatial holdouts. For any other value of split-strategy,
                              the data are randomly split.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
    """
    if split_strategy == 1:
        x_train, x_test, y_train, y_test = split_data_yearly(
            input_df, pred_attr=pred_attr, year_col=year_col,
            test_years=test_year, shuffle=shuffle,
            random_state=random_state
        )
    elif split_strategy == 2:
        x_train, x_test, y_train, y_test = split_data_train_test_ratio(
            input_df, pred_attr=pred_attr,
            test_size=test_size, random_state=random_state,
            shuffle=shuffle, test_year=test_year,
            year_col=year_col
        )
    elif split_strategy == 3:
        x_train, x_test, y_train, y_test = split_spatial(
            input_df, pred_attr=pred_attr, gw_basin_col=gw_basin_col,
            test_gw_basins=test_gw_basins, shuffle=shuffle,
            random_state=random_state
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            input_df, input_df[pred_attr].to_frame(),
            shuffle=shuffle, random_state=random_state,
            test_size=test_size
        )
    return x_train, x_test, y_train, y_test


def split_data_train_test_ratio(
        input_df: pd.DataFrame,
        pred_attr: str = 'gw_pumping_mm',
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = 0.2,
        test_year: bool = True,
        year_col: str = 'Year',
        crop_col: str | None = None
) -> tuple[pd.DataFrame, ...]:
    """Split data based on train-test percentage based on year or crop. By default test_size amount of data is kept from
    each year for testing.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        shuffle (bool): Default True for shuffling.
        random_state (int): Random state used during train test split.
        test_size (float): Test data size (0<=test_size<=1).
        test_year (bool): If True, build test data from the year_col. Otherwise, use crop_col.
        year_col (str): Name of the year column.
        crop_col (str or None): Name of the crop column. By default it's None.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
    """
    selection_var = input_df[year_col].unique()
    selection_label = year_col
    if not test_year:
        selection_var = input_df[crop_col].unique()
        selection_label = crop_col
    x_train_parts, x_test_parts = [], []
    y_train_parts, y_test_parts = [], []
    for svar in selection_var:
        selected_data = input_df.loc[input_df[selection_label] == svar]
        y = selected_data[pred_attr].to_frame()
        x_train, x_test, y_train, y_test = train_test_split(
            selected_data, y, shuffle=shuffle,
            random_state=random_state, test_size=test_size
        )
        x_train_parts.append(x_train)
        x_test_parts.append(x_test)
        y_train_parts.append(y_train)
        y_test_parts.append(y_test)
    return (pd.concat(x_train_parts), pd.concat(x_test_parts),
            pd.concat(y_train_parts), pd.concat(y_test_parts))


def split_data_yearly(
        input_df: pd.DataFrame,
        pred_attr: str = 'gw_pumping_mm',
        test_years: tuple[int, ...] = (2016,),
        year_col: str = 'Year',
        shuffle: bool = True,
        random_state: int = 0
) -> tuple[pd.DataFrame, ...]:
    """Split data based on a particular year.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        test_years (tuple (int, ...)): Build test data from only these tuple of years, i.e., (2014, 2015).
        year_col (str): Name of the year column.
        shuffle (bool): Set False to stop data shuffling.
        random_state (int): Random state used during train test split.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
    """
    years = input_df[year_col].unique()
    x_train_parts, x_test_parts = [], []
    for year in years:
        selected_data = input_df.loc[input_df[year_col] == year]
        if year not in test_years:
            x_train_parts.append(selected_data)
        else:
            x_test_parts.append(selected_data)
    x_train_df = pd.concat(x_train_parts) if x_train_parts else pd.DataFrame()
    x_test_df = pd.concat(x_test_parts) if x_test_parts else pd.DataFrame()
    y_train_df = x_train_df[pred_attr].to_frame()
    y_test_df = x_test_df[pred_attr].to_frame()
    if shuffle:
        x_train_df, y_train_df = sk.shuffle(x_train_df, y_train_df, random_state=random_state)
        x_test_df, y_test_df = sk.shuffle(x_test_df, y_test_df, random_state=random_state)
    return x_train_df, x_test_df, y_train_df, y_test_df


def split_spatial(
        input_df: pd.DataFrame,
        pred_attr: str = 'gw_pumping_mm',
        test_gw_basins: tuple[str, ...] = ('HARQUAHALA INA',),
        gw_basin_col: str = 'GW_Basin',
        shuffle: bool = True,
        random_state: int = 0
) -> tuple[pd.DataFrame, ...]:
    """Split data spatially by holding out entire groundwater basins.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        pred_attr (str): Prediction attribute name.
        test_gw_basins (tuple (str, ...)): Build test data from only these tuple of groundwater basins.
        gw_basin_col (str): Name of the GW basin column.
        shuffle (bool): Set False to stop data shuffling.
        random_state (int): Random state used during train test split.

    Returns:
        tuple[pd.DataFrame, ...]: A tuple of X_train, X_test, y_train, y_test data frames.
    """
    gw_basins = input_df[gw_basin_col].unique()
    x_train_parts, x_test_parts = [], []
    for gw_basin in gw_basins:
        selected_data = input_df.loc[input_df[gw_basin_col] == gw_basin]
        if gw_basin not in test_gw_basins:
            x_train_parts.append(selected_data)
        else:
            x_test_parts.append(selected_data)
    x_train_df = pd.concat(x_train_parts) if x_train_parts else pd.DataFrame()
    x_test_df = pd.concat(x_test_parts) if x_test_parts else pd.DataFrame()
    y_train_df = x_train_df[pred_attr].to_frame()
    y_test_df = x_test_df[pred_attr].to_frame()
    if shuffle:
        x_train_df, y_train_df = sk.shuffle(x_train_df, y_train_df, random_state=random_state)
        x_test_df, y_test_df = sk.shuffle(x_test_df, y_test_df, random_state=random_state)
    return x_train_df, x_test_df, y_train_df, y_test_df


def reindex_df(
        df: pd.DataFrame,
        column_names: tuple[str, ...] | None,
        ordering: bool = False
) -> pd.DataFrame:
    """Reindex dataframe columns.

    Args:
        df (pd.DataFrame): Input pandas DataFrame object.
        column_names (tuple (str, ...)): Data frame column names, these must be df headers.
        ordering (bool): Set True to sort df by column_names.

    Returns:
        pd.DataFrame: Reindexed pandas DataFrame object.
    """
    if column_names is None:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)


def process_outliers(
        input_df: pd.DataFrame,
        target_attr: str,
        year_col: str,
        gw_basin_col: str,
        operation: int = 2,
        min_gw_pumping: float = 0,
        max_gw_pumping: float = np.inf,
) -> pd.DataFrame:
    """Remove outliers from a dataframe based on target_attr.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame object.
        target_attr (str): Target attribute based on which outlier removal will occur.
        year_col (str): Name of the year column.
        gw_basin_col (str): Name of the GW basin column.
        operation (int): Outlier operation to perform. Set to 1 for removing outlier directly or 2 for removing outliers
                         by each basin and each year. Set to 3 to remove all values less than min_gw_pumping and greater than
                            max_gw_pumping.
        min_gw_pumping (float): Minimum gw pumping value in mm. Default is 0.
        max_gw_pumping (float): Maximum gw pumping value in mm. Default is np.inf.

    Returns:
        pd.DataFrame: Outlier removed input_df.
    """
    input_df = input_df.copy(deep=True)
    init_rows = input_df.shape[0]
    num_outliers = 0
    if operation == 1:
        target_vals = input_df[target_attr].to_numpy().ravel()
        q3, q1 = np.percentile(target_vals, [75, 25])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        max_data = input_df[target_attr].max()
        min_data = input_df[target_attr].min()
        upper_limit = max_data if upper_limit > max_data else upper_limit
        lower_limit = min_data if lower_limit < min_data else lower_limit
        invalid_idx = (input_df[target_attr] >= upper_limit) | (input_df[target_attr] <= lower_limit)
        num_outliers = invalid_idx.sum()
        input_df.loc[invalid_idx, target_attr] = np.nan
    elif operation == 2:
        year_list = input_df[year_col].unique()
        gw_basins = input_df[gw_basin_col].unique()
        for gw_basin in gw_basins:
            for year in year_list:
                selection = (input_df[gw_basin_col] == gw_basin) & (input_df[year_col] == year)
                selected_data = input_df[selection]
                target_vals = selected_data[target_attr].to_numpy().ravel()
                if target_vals.size == 0:
                    continue
                q3, q1 = np.percentile(target_vals, [75, 25])
                iqr = q3 - q1
                upper_limit = q3 + 1.5 * iqr
                lower_limit = q1 - 1.5 * iqr
                max_data = selected_data[target_attr].max()
                min_data = selected_data[target_attr].min()
                upper_limit = max_data if upper_limit > max_data else upper_limit
                lower_limit = min_data if lower_limit < min_data else lower_limit
                invalid_idx = (selected_data[target_attr] >= upper_limit) | (selected_data[target_attr] <= lower_limit)
                outliers = invalid_idx.sum()
                logger.info(f'{gw_basin} {year} outliers: {outliers}')
                num_outliers += outliers
                input_df.loc[selection, 'Outlier'] = invalid_idx
        input_df = input_df[input_df['Outlier'] == False]
        input_df = input_df.drop(columns='Outlier')
    elif operation == 3:
        invalid_idx = (input_df[target_attr] < min_gw_pumping) | (input_df[target_attr] > max_gw_pumping)
        input_df.loc[invalid_idx, target_attr] = np.nan
        num_outliers = invalid_idx.sum()
    input_df = input_df.dropna()
    logger.info(f'Old DF rows = {init_rows}, New DF rows = {input_df.shape[0]}')
    logger.info(f'{num_outliers} outliers removed...')
    return input_df


def create_train_test_data(
        input_df: pd.DataFrame,
        output_dir: str,
        pred_attr: str = 'gw_pumping_mm',
        drop_attr: tuple[str, ...] = ('Year',),
        test_size: float = 0.2,
        test_year: tuple[int, ...] | bool = True,
        test_gw_basins: tuple[str, ...] = (),
        year_col: str = 'Year',
        random_state: int = 42,
        already_created: bool = False,
        scaling: bool = False,
        year_list: list[int] = (1985,),
        gw_basin_col: str = 'GW_Basin',
        split_strategy: int = 1,
        outlier_op: int | None = 2,
        min_gw_pumping: float = 0,
        max_gw_pumping: float = np.inf,
        shuffle: bool = True,
        use_ama_ina: bool = False,
        drop_gw_basins: tuple[str, ...] = ('JOSEPH CITY INA', 'WILLCOX AMA', 'HUALAPAI VALLEY INA'),
        water_use: str = 'IRRIGATION'
) -> tuple:
    """Create train and test data.

    Args:
        input_df (pd.DataFrame): Input pandas DataFrame.
        output_dir (str): Output directory.
        pred_attr (str): Attribute to be predicted.
        drop_attr (tuple (str, ...)): Tuple of attributes to drop from model training.
        test_size (float): Test size between (0, 1).
        test_year (tuple (int, ...) or bool): If split_strategy = 1, then this needs to be a tuple of years in YYYY
                                             format, i.e., (2014, 2015), and the test data is created from these years.
                                             For split_strategy=2, set this to True if the test data needs to be created
                                             based on year_col.
        test_gw_basins (tuple (str, ...)): Build test data from only these tuple of groundwater basins.
        year_col (str): Name of the year column.
        random_state (int): Random state used during train test split.
        already_created (bool): Set True to load existing train and test data.
        scaling (bool): Set True to perform minmax scaling.
        year_list (list (int,...)): List of years in YYYY format, i.e., (1985, ..., 2024) to build the data set.
        gw_basin_col (str): Name of the GW basin column.
        split_strategy (int): If 1, Split train test data based on year_col. If 2, then test_size amount of data from
                              year_col are kept for testing and rest for training;
                              for this option, test-year should have a tuple of integers or a True value. If 3, then
                              test_gw_basins are used for spatial holdouts. For any other value of split-strategy,
                              the data are randomly split.
        outlier_op (int): Outlier operation to perform. Set to 1 for removing outlier directly or 2 for removing
                          outliers by each basin and each year. Set to 3 to remove all values less than min_gw_pumping 
                          and greater than max_gw_pumping.
        min_gw_pumping (float): Minimum gw pumping value in mm. Default is 0.
        max_gw_pumping (float): Maximum gw pumping value in mm. Default is np.inf.
        shuffle (bool): Set False to stop data shuffling.
        use_ama_ina (bool): Set True to use AMA-INA basins.
        drop_gw_basins (tuple (str, ...)): Tuple of GW basins to drop from the data set. Default basins are: 
        ('JOSEPH CITY INA', 'WILLCOX AMA', 'HUALAPAI VALLEY INA'). These basins have very less data.
        water_use (str): Type of water use to consider. Default is 'IRRIGATION'. Other option is 'All'.

    Returns:
        tuple: A tuple containing X_train, X_test as pandas data frames, y_train, y_test as numpy arrays.
        If scaling=True, then x_scaler and y_scaler are also returned. Year_train, Year_test, GW_Basin_Train, and
        GW_Basin_Test are returned as well
        for future analyses.
    """
    makedirs(output_dir)
    x_train_file = os.path.join(output_dir, 'X_train.parquet')
    x_test_file = os.path.join(output_dir, 'X_test.parquet')
    y_train_file = os.path.join(output_dir, 'y_train.parquet')
    y_test_file = os.path.join(output_dir, 'y_test.parquet')
    year_train_file = os.path.join(output_dir, 'Year_train.parquet')
    year_test_file = os.path.join(output_dir, 'Year_test.parquet')
    gw_basin_train_file = os.path.join(output_dir, 'GW_Basin_train.parquet')
    gw_basin_test_file = os.path.join(output_dir, 'GW_Basin_test.parquet')
    x_scaler_file, x_scaler, y_scaler_file, y_scaler = [None] * 4
    if scaling:
        x_scaler_file = os.path.join(output_dir, 'x_scaler')
        y_scaler_file = os.path.join(output_dir, 'y_scaler')
    if not already_created:
        if use_ama_ina:
            ama_ina_basins = get_ama_ina_basin_names()
            ama_ina_basins = [b for b in ama_ina_basins if b not in drop_gw_basins]
            input_df = input_df[input_df[gw_basin_col].isin(ama_ina_basins)]
        drop_attr = [attr for attr in drop_attr]
        if year_list and year_col in input_df.columns:
            input_df = input_df[input_df[year_col].isin(year_list)]
        cols_before = set(input_df.columns)
        input_df = input_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        dropped_cols = sorted(cols_before - set(input_df.columns))
        if dropped_cols:
            logger.info(f'Dropped {len(dropped_cols)} all-NaN columns after year filter: {dropped_cols}')
        n_before = len(input_df)
        input_df = input_df.dropna()
        if len(input_df) < n_before:
            logger.info(f'Dropped {n_before - len(input_df)} rows with NaN values')
        input_df = input_df[input_df[pred_attr] > 0] if water_use == 'IRRIGATION' else input_df
        if outlier_op is not None:
            input_df = process_outliers(
                input_df, pred_attr,
                year_col, gw_basin_col,
                outlier_op, min_gw_pumping,
                max_gw_pumping
            )
        if year_col in drop_attr:
            drop_attr.remove(year_col)
        if gw_basin_col in drop_attr:
            drop_attr.remove(gw_basin_col)
        drop_attr = list(set(drop_attr).intersection(input_df.columns.tolist()))
        input_df = input_df.drop(columns=drop_attr)
        input_df.to_parquet(os.path.join(output_dir, 'Cleaned_AZ_GW_Data.parquet'), index=False)
        x_train, x_test, y_train, y_test = split_data_train_test(
            input_df, pred_attr=pred_attr,
            test_size=test_size,
            random_state=random_state, shuffle=shuffle,
            test_year=test_year, gw_basin_col=gw_basin_col,
            test_gw_basins=test_gw_basins,
            split_strategy=split_strategy, year_col=year_col
        )
        year_train = x_train[year_col].copy().to_frame()
        year_test = x_test[year_col].copy().to_frame()
        gw_basin_train = x_train[gw_basin_col].copy().to_frame()
        gw_basin_test = x_test[gw_basin_col].copy().to_frame()
        x_train = x_train.drop(columns=[year_col, gw_basin_col, pred_attr])
        x_test = x_test.drop(columns=[year_col, gw_basin_col, pred_attr])
        x_train = reindex_df(x_train, column_names=None)
        x_test = reindex_df(x_test, column_names=None)
        if scaling:
            x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
            x_train = pd.DataFrame(x_scaler.fit_transform(x_train), columns=x_train.columns)
            x_test = pd.DataFrame(x_scaler.transform(x_test), columns=x_test.columns)
            y_train = pd.DataFrame(y_scaler.fit_transform(y_train), columns=y_train.columns)
            y_test = pd.DataFrame(y_scaler.transform(y_test), columns=y_test.columns)
        x_train.to_parquet(x_train_file, index=False)
        x_test.to_parquet(x_test_file, index=False)
        y_train.to_parquet(y_train_file, index=False)
        y_test.to_parquet(y_test_file, index=False)
        year_train.to_parquet(year_train_file, index=False)
        year_test.to_parquet(year_test_file, index=False)
        gw_basin_train.to_parquet(gw_basin_train_file, index=False)
        gw_basin_test.to_parquet(gw_basin_test_file, index=False)
        if scaling:
            with open(x_scaler_file, 'wb') as f:
                pickle.dump(x_scaler, f)
            with open(y_scaler_file, 'wb') as f:
                pickle.dump(y_scaler, f)
    else:
        x_train = pd.read_parquet(x_train_file)
        x_test = pd.read_parquet(x_test_file)
        y_train = pd.read_parquet(y_train_file)
        y_test = pd.read_parquet(y_test_file)
        year_train = pd.read_parquet(year_train_file)
        year_test = pd.read_parquet(year_test_file)
        gw_basin_train = pd.read_parquet(gw_basin_train_file)
        gw_basin_test = pd.read_parquet(gw_basin_test_file)
        if scaling:
            with open(x_scaler_file, 'rb') as f:
                x_scaler = pickle.load(f)
            with open(y_scaler_file, 'rb') as f:
                y_scaler = pickle.load(f)
    ret_vals = (
        x_train, x_test, y_train.to_numpy().ravel(), y_test.to_numpy().ravel(),
        x_scaler, y_scaler, year_train, year_test, gw_basin_train, gw_basin_test
    )

    return ret_vals
