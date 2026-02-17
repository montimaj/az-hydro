"""
Contains codes for handling Google Earth Engine datasets
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import ee
import os
import time
import pandas as pd
import requests
import subprocess
import numpy as np
import geopandas as gpd
import rasterio as rio
import warnings
import swifter
import joblib
import pickle
import sklearn.utils as sk

from osgeo import gdal
# Suppress GDAL TIFFReadDirectory warnings
os.environ['CPL_LOG'] = '/dev/null'
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)
logging.getLogger("distributed").setLevel(logging.ERROR)
from openet.refetgee import Daily, calcs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sysops import makedirs
from gwops import create_land_use_data, get_ama_ina_basin_names
from rasterops import reproject_raster_gdal, read_raster_as_arr, get_xy_grids_from_raster, \
    clamp_and_rewrite_raster
from google.cloud import storage
from shapely.geometry import Polygon
from glob import glob
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from http.client import RemoteDisconnected
from tqdm import tqdm


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

    fishnet_file = f'{output_dir}AZ_Polygons_{fishnet_size}{fishnet_unit}.geojson'
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


def generate_chunks(input_list: list[...], num_chunks: int):
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


def calculate_eto(
        year: int, 
        maca_scenario: str | None = None,
        maca_model: str | None = None,
        return_monthly: bool = False
) -> ee.Image | ee.ImageCollection:
    """
    Process daily collection to calculate annual or monthly ETo.
    Author: Peter ReVelle (peter.revelle@dri.edu)
    Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

    Args:
    year (int): Year in YYYY format for which the daily image belongs to. This is required to determine the 
    climate data source for ETo calculation. Note that since ETo calculation requires daily data, for 1895-1978, we 
    take the monthly PRISM data and assign the same ETo value for all days in that month. For 1979-2025, 
    gridMET daily ETo is directly used. For 2026-2100, we use MACA daily data for ETo calculation.
    maca_scenario (str or None): MACA scenario to use for ETo calculation for 2026-2100. Required only if year > 2025.
    maca_model (str or None): MACA model to use for ETo calculation for 2026-2100. Required only if year > 2025.
    return_monthly (bool): If True, return monthly ETo as an ImageCollection. If False (default), return annual ETo sum.

    Returns:
    ee.Image | ee.ImageCollection: Annual ETo image (if return_monthly=False) or monthly ETo ImageCollection 
    (if return_monthly=True) for the input year. Band name is 'eto' (in mm).
    """
    
    start_year = f'{year}-01-01'
    end_year = f'{year + 1}-01-01'
    lat_img = ee.Image.pixelLonLat().select('latitude')
    months = ee.List.sequence(1, 12)
    
    if year > 2025:
        # Use MACA for future scenarios
        if maca_scenario is None or maca_model is None:
            raise ValueError('maca_scenario and maca_model are required for years > 2025')
        
        maca_daily = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
            .filterDate(start_year, end_year) \
            .filterMetadata('model', 'equals', maca_model) \
            .filterMetadata('scenario', 'equals', maca_scenario)
        
        def calc_maca_daily_eto(daily_img):
            eto = Daily.maca(input_img=daily_img, lat=lat_img).eto
            return eto.rename('eto').set('system:time_start', daily_img.date().millis())
        
        daily_eto = maca_daily.map(calc_maca_daily_eto)
        monthly_eto = ee.ImageCollection(months.map(lambda m:
            daily_eto.filter(ee.Filter.calendarRange(m, m, 'month')).sum()
                .set('system:time_start', ee.Date.fromYMD(year, m, 1).millis())
                .set('system:time_end', ee.Date.fromYMD(year, ee.Number(m).add(1).mod(12).add(1), 1).millis())
        ))
        
    elif 1895 <= year <= 1978:
        # Use PRISM for ETo (monthly data converted to monthly ETo using Hargreaves)
        prism_monthly = ee.ImageCollection('projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY') \
            .filterDate(start_year, end_year)
        
        def calc_prism_monthly_eto(monthly_img):
            month = ee.Date(monthly_img.get('system:time_start')).get('month')
            ndays = ee.Number(ee.Date.fromYMD(year, month, 1).advance(1, 'month')
                .difference(ee.Date.fromYMD(year, month, 1), 'day'))
            # Get day of year for middle of month
            doy = ee.Date(monthly_img.get('system:time_start')).getRelative('day', 'year').add(ndays.divide(2).floor())
            # Get temperature data from PRISM (in Celsius)
            tmax = monthly_img.select('tmax')
            tmin = monthly_img.select('tmin')
            # Use Daily class with minimal inputs for Hargreaves PET
            # Create dummy images for required parameters (not used in Hargreaves)
            daily_obj = Daily(
                tmax=tmax,
                tmin=tmin,
                ea=ee.Image.constant(1),  # dummy, not used in Hargreaves
                rs=ee.Image.constant(1),  # dummy, not used in Hargreaves
                uz=ee.Image.constant(1),  # dummy, not used in Hargreaves
                zw=ee.Number(2),
                elev=ee.Image.constant(0),
                lat=lat_img,
                doy=doy
            )
            eto = daily_obj.pet_hargreaves.multiply(ndays)
            return eto.rename('eto') \
                .set('system:time_start', monthly_img.get('system:time_start')) \
                .set('system:time_end', monthly_img.get('system:time_end'))
        
        monthly_eto = prism_monthly.map(calc_prism_monthly_eto)
        
    else:
        # Use gridMET for 1979-2025 (already monthly)
        monthly_eto = ee.ImageCollection('projects/openet/assets/reference_et/conus/gridmet/monthly/v1') \
            .filterDate(start_year, end_year) \
            .select('eto')
    
    if return_monthly:
        return monthly_eto
    else:
        return monthly_eto.sum().rename('annual_eto_mm')


def calculate_annual_peff(
        year: int, 
        rz_depth_m: list[float] | None = None,
        mad_factor: float = 0.5,
        maca_scenario: str | None = None,
        maca_model: str | None = None
) -> ee.Image:
    """
    Calculate annual effective precipitation using the USDA-SCS method.
    Author: Christopher Pearson (chris.pearson@dri.edu)
    Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

    Args:
    year (int): Year in YYYY format for which the annual effective precipitation is calculated. This is required to 
    determine the climate data source for precipitation and ETo. PRISM precipitation is used for 1895-2025.
    - For 1895-1978: PRISM Hargreaves ETo, PRISM precipitation
    - For 1979-2025: gridMET ETo, PRISM precipitation
    - For 2026-2100: MACA ETo, MACA precipitation
    rz_depth_m (list[float] or None): Monthly root zone depth in meters. Defaults to [1] * 12
    (1 meter for all months). Should be a list of 12 values for each month.
    mad_factor (float): Management Allowed Depletion factor. Defaults to 0.5.
    maca_scenario (str or None): MACA scenario to use for precipitation calculation for 2026-2100. 
    Required only if year > 2025.
    maca_model (str or None): MACA model to use for precipitation calculation for 2026-2100. 
    Required only if year > 2025.

    Returns:
    ee.Image: Annual effective precipitation image with bands 'annual_peff_mm' (effective precipitation in mm).
    """

    # Default root zone depth: 1 meter for all months
    if rz_depth_m is None:
        rz_depth_m = [1] * 12
    
    # Convert root zone depth from meters to inches (1 meter = 39.37 inches)
    rz_inches = ee.List([rz * 39.37 for rz in rz_depth_m])
    
    # AWC (Available Water Capacity) from OpenET soil dataset (already in inches)
    awc = ee.Image('projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')
    
    # Define date range for the year
    start_year = f'{year}-01-01'
    end_year = f'{year + 1}-01-01'
    
    # Get monthly ETo using calculate_eto
    eto_monthly = calculate_eto(year, maca_scenario, maca_model, return_monthly=True)
    # Ensure consistent band name 'eto' and add 'month' property for joining
    eto_monthly = eto_monthly.map(
        lambda img: img.rename('eto')
            .set('month', ee.Date(img.get('system:time_start')).get('month'))
            .copyProperties(img, ['system:time_start', 'system:time_end'])
    )
    
    # Get monthly precipitation
    if year > 2025:
        # Use MACA precipitation for future scenarios
        if maca_scenario is None or maca_model is None:
            raise ValueError('maca_scenario and maca_model are required for years >= 2025')
        pr_daily = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
            .filterDate(start_year, end_year) \
            .filterMetadata('model', 'equals', maca_model) \
            .filterMetadata('scenario', 'equals', maca_scenario) \
            .select('pr') \
            .map(lambda img: img.set('month', ee.Date(img.get('system:time_start')).get('month')))
        # Aggregate daily MACA precipitation to monthly
        months = ee.List.sequence(1, 12)
        pr_monthly = ee.ImageCollection(months.map(lambda m: 
            pr_daily.filter(ee.Filter.eq('month', m)) \
                .sum() \
                .rename('pr') \
                .set('month', m) \
                .set('system:time_start', ee.Date.fromYMD(year, m, 1).millis()) \
                .set('system:time_end', ee.Date.fromYMD(year, ee.Number(m).add(1).mod(12).add(1), 1).millis())
        ))
    else:
        # Use PRISM for 1895-2025, rename 'ppt' to 'pr' for consistency
        pr_monthly = ee.ImageCollection('projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY') \
            .filterDate(start_year, end_year) \
            .select(['ppt'], ['pr']) \
            .map(lambda img: img.set('month', ee.Date(img.get('system:time_start')).get('month')))
    
    # Join ETo and precipitation collections on month (avoids timestamp mismatch between different collections)
    def join_collections(coll_1, coll_2):
        filter_month_eq = ee.Filter.equals(
            leftField='month',
            rightField='month'
        )
        joined = ee.Join.inner().apply(coll_1, coll_2, filter_month_eq)
        return joined.map(lambda feature: 
            ee.Image.cat(feature.get('primary'), feature.get('secondary'))
                .copyProperties(ee.Image(feature.get('primary')), ['system:time_start', 'system:time_end', 'month'])
        )
    
    joined = ee.ImageCollection(join_collections(eto_monthly, pr_monthly))
    
    # SCS effective precipitation function
    def calculate_ep(img):
        # Convert from mm to inches (bands are consistently named 'eto' and 'pr')
        pr = img.select('pr').divide(25.4).rename('pr')  # mm to inches
        eto = img.select('eto').divide(25.4).rename('eto')  # mm to inches
        
        # Get month value for root zone depth lookup (1-indexed)
        month = ee.Number.parse(ee.Date(img.get('system:time_start')).format('MM'))
        
        # d term for soil storage factor (eq. 2-85)
        # d = 50% of AWC * root zone depth in inches
        d = awc.multiply(mad_factor).multiply(ee.Number(rz_inches.get(month.subtract(1))))
        
        # Soil storage factor (eq. 2-85)
        # sf = 0.531747 + 0.295164*d - 0.057697*d^2 + 0.003804*d^3
        sf = d.multiply(0.295164) \
            .add(0.531747) \
            .subtract(d.pow(2).multiply(0.057697)) \
            .add(d.pow(3).multiply(0.003804)) \
            .rename('sf')
        
        # SCS effective precipitation (eq. 2-84)
        # ep = sf * (0.70917 * pr^0.82416 - 0.11556) * 10^(0.02426 * eto)
        ep = sf.multiply(
            pr.pow(0.82416).multiply(0.70917).subtract(0.11556)
        ).multiply(
            ee.Image.constant(10).pow(eto.multiply(0.02426))
        ).rename('ep')
        
        # Limit ep: ep <= pr, ep <= eto, ep >= 0
        ep_cleaned = ep.where(ep.gte(pr), pr) \
            .where(ep.gt(eto), eto) \
            .clamp(0, 10000)
        
        return ee.Image(ep_cleaned) \
            .setDefaultProjection(crs='EPSG:4326', scale=4000) \
            .copyProperties(img, ['system:time_start', 'system:time_end'])
    
    # Build monthly collection with effective precipitation
    monthly_peff = joined.map(calculate_ep)
    
    # Sum monthly values to get annual totals and convert back to mm
    annual_peff = monthly_peff \
        .sum().multiply(25.4).set('year', year) \
        .rename('annual_peff_mm') \
        .set('system:time_start', ee.Date.fromYMD(year, 1, 1).millis())
    
    return annual_peff


def download_gee_tif(
        data_bands: list[ee.Image],
        data_band_name_list: list[str],
        local_file_name: str,
        ee_geom: ee.Geometry,
        gee_scale: float,
        crs: str,
        categorical_bands: list[str],
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
    categorical_bands (list[str]): List of band names that are categorical and require mode reduction.
    verbose (bool): Set True to see extra details on file downloads.

    Returns:
        None. The function saves the downloaded data as a GeoTIFF file at the specified local_file_name.
    """
    
    data_img = data_bands[0].rename(data_band_name_list[0])
    band_scale = data_img.projection().nominalScale().getInfo()
    max_pixels = max(64, np.ceil((gee_scale / band_scale) ** 2))
    time.sleep(0.01)
    data_img = data_img.setDefaultProjection(
        crs=crs,
        scale=band_scale
    ).reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True,
        maxPixels=max_pixels
    ).reproject(crs, scale=gee_scale)
    for band, band_name in zip(data_bands, data_band_name_list):
        band = band.rename(band_name)
        try:
            band_scale = band.projection().nominalScale().getInfo()
            time.sleep(0.01)
        except ee.EEException as e:
            print('Error getting band scale for', band_name, ':', e)
            return
        if gee_scale > 30:
            if band_name in categorical_bands:
                reducer = ee.Reducer.mode(maxRaw=max_pixels)
            elif band_name == 'annual_irrmapper_fraction':
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
                    bestEffort=True,
                    maxPixels=max_pixels
                ).reproject(crs, scale=gee_scale)
            else:
                band = band.setDefaultProjection(
                    crs=crs,
                    scale=band_scale
                ).reproject(crs, scale=gee_scale)
        if band_name == 'irrigation_status':
            band = band.where(band.gt(0), 1)
        data_img = data_img.addBands(band, overwrite=True)
    retry_download = True
    while retry_download:
        try:
            if verbose:
                print('Downloading', local_file_name, '...')
            gee_url = data_img.getDownloadUrl({
                'scale': gee_scale,
                'region': ee_geom,
                'format': 'GEO_TIFF',
                'crs': crs
            })
            r = requests.get(
                gee_url,
                allow_redirects=True,
                timeout=None,
                stream=True
            )
            with open(local_file_name, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=1024):
                    fd.write(chunk)
            # Validate the downloaded file
            tile_arr, tile_rio = read_raster_as_arr(local_file_name)
            if tile_arr.size == 0:
                if verbose:
                    logger.warning('Downloaded file is empty. Deleting %s', local_file_name)
                tile_rio.close()
                os.remove(local_file_name)
                raise rio.errors.RasterioIOError
            tile_rio.close()
            clamp_and_rewrite_raster(
                local_file_name,
                min_val=0,
                band_descriptions=data_band_name_list
            )                
            retry_download = False
        except (
                ee.EEException, requests.exceptions.RequestException,
                requests.exceptions.ConnectionError, RemoteDisconnected,
                rio.errors.RasterioIOError, Exception
        ) as e:
            if verbose:
                logger.warning('Error %s during %s download! Retrying...', e, local_file_name)
            retry_download = True
        time.sleep(0.01)


def download_gee_tile(
        tile_values: tuple[int, float, float, float],
        download_dir: str,
        year_list: list,
        data_band_names: list[str],
        gcloud_project: str = 'azhydro',
        gee_scale: float = 30,
        verbose: bool = False,
        crs: str = 'EPSG:4326'
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

    Returns:
        None.
    """

    if min(year_list) < 1896 or max(year_list) > 2100:
        raise ValueError('Year list should be between 1896 and 2100')
    scenario_years = list(range(2026, 2101))
    historical_years = [year for year in year_list if year not in scenario_years]
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
    openet_ic = ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0")
    usgs_ensemble_et_ic = ee.ImageCollection("users/montimajumdar/USGS-Reitz-Ensemble-ET") 
    prism_ic = ee.ImageCollection('projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_MONTHLY') 
    irrmapper_ic = ee.ImageCollection('UMT/Climate/IrrMapper_RF/v1_2')
    nlcd_ic = ee.ImageCollection('projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER')
    cdl_ic = ee.ImageCollection('USDA/NASS/CDL')
    usgs_lulc_ic = ee.ImageCollection('users/montimajumdar/USGS-LULC-CONUS')
    soil_depth = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/soil_depth'
    ).rename('soil_depth_cm')
    awc = ee.Image(
        'projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite'
    ).rename('awc_in')
    ksat_mean = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/physical/ksat_mean'
    ).rename('ksat_mean_micromps')
    ee_geom = ee.Geometry.Rectangle(tile_values[1:])
    categorical_bands = get_categorical_bands()
    usgs_lulc_scenarios = ['B1', 'B2', 'A1B', 'A2']
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
    historical_download_dir = f'{download_dir}historical/'
    scenario_download_dir = f'{download_dir}scenarios/'
    makedirs([historical_download_dir, scenario_download_dir])
    for year in historical_years:
        local_file_name = f'{historical_download_dir}Tile_{tile_values[0]}_{year}.tif'
        if os.path.exists(local_file_name):
            try:
                tile_arr, tile_rio = read_raster_as_arr(local_file_name)
                if tile_arr.size == 0:
                    if verbose:
                        logger.warning('Downloaded file is empty. Deleting %s', local_file_name)
                    tile_rio.close()
                    os.remove(local_file_name)
                    raise rio.errors.RasterioIOError
                tile_rio.close()
                continue
            except rio.errors.RasterioIOError:
                os.remove(local_file_name)
                if verbose:
                    logger.warning('Existing file %s is corrupted. Deleted and retrying download...', local_file_name)
        irrmapper_year = year if year >= 1985 else 1985
        irr = irrmapper_ic.filterDate(f'{irrmapper_year}-01-01', f'{irrmapper_year + 1}-01-01') \
            .select('classification') \
            .max()
        mask = irr.eq(0)
        irr_mask = irr.updateMask(mask).remap([0], [1])
        start_year_gee = f'{year}-01-01'
        end_year_gee = f'{year + 1}-01-01'
        if year >= 2000:
            actual_et = openet_ic.select('et_ensemble_mad') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
        else:
            actual_et = usgs_ensemble_et_ic.filterDate(start_year_gee, end_year_gee) \
                .sum() \
                .multiply(30) \
                .setDefaultProjection(crs=crs, scale=800)
        eto = calculate_eto(year)
        precip = prism_ic.select('ppt') \
            .filterDate(start_year_gee, end_year_gee) \
            .sum()
        peff = calculate_annual_peff(year, mad_factor=1, rz_depth_m=[2] * 12) # consistent with UCRB comparisons
        tmmx = prism_ic.select('tmax') \
            .filterDate(start_year_gee, end_year_gee) \
            .mean() \
            .add(273.15)
        tmmn = prism_ic.select('tmin') \
            .filterDate(start_year_gee, end_year_gee) \
            .mean()\
            .add(273.15)
        if year < 1985:
            # Use historical USGS LULC data from 1938-1984.
            usgs_lulc_year = year if year >= 1938 else 1938
            usgs_lulc = usgs_lulc_ic.filterMetadata('scenario', 'equals', 'Historical') \
                .filterDate(f'{usgs_lulc_year}-01-01', f'{usgs_lulc_year + 1}-01-01') \
                .first() \
                .setDefaultProjection(crs=crs, scale=250)
            usgs_mask_1 = usgs_lulc.eq(13) # cropland
            usgs_mask_2 = usgs_lulc.eq(2) # developed
            usgs_mask_3 = usgs_lulc.eq(6) # mining
            usgs_mask_4 = usgs_lulc.eq(1) # open water
            usgs_mask = usgs_mask_1.Or(usgs_mask_2).Or(usgs_mask_3).Or(usgs_mask_4)
            lulc = usgs_lulc.updateMask(usgs_mask).remap(
                [13, 2, 6, 1],
                [1, 124, 121, 111]
            ).rename('lulc')
        elif 1985 <= year <= 2007:
            # Switch to NLCD for 1985-2007
            nlcd_year = nlcd_ic.filterDate(start_year_gee, end_year_gee).first()
            nlcd_mask_1 = nlcd_year.eq(82) # cropland
            nlcd_mask_2 = nlcd_year.gte(21).And(nlcd_year.lte(24)) # developed
            nlcd_mask_3 = nlcd_year.eq(11) # open water
            nlcd_mask = nlcd_mask_1.Or(nlcd_mask_2).Or(nlcd_mask_3)
            lulc = nlcd_year.updateMask(nlcd_mask).remap(
                [82, 21, 22, 23, 24, 11],
                [1, 121, 122, 123, 124, 111]
            ).rename('lulc')
        else:
            cdl_year = year if year <= 2024 else 2024
            # Switch to CDL for 2008-2024
            lulc = cdl_ic.filterDate(f'{cdl_year}-01-01', f'{cdl_year + 1}-01-01') \
                .select('cropland') \
                .first() \
                .rename('lulc')
        data_bands = [
            actual_et,
            eto,
            precip,
            peff,
            tmmx,
            tmmn,
            lulc,
            soil_depth,
            awc,
            ksat_mean,
            irr_mask
        ]
        download_gee_tif(
            data_bands=data_bands,
            data_band_name_list=data_band_names,
            local_file_name=local_file_name,
            ee_geom=ee_geom,
            gee_scale=gee_scale,
            crs=crs,
            categorical_bands=categorical_bands,
            verbose=verbose
        )
        
    scenario_data_list = ['LULC', 'MACA']
    # We download LULC and MACA scenario data separately.
    for scenario_data in scenario_data_list:
        for scenario in (usgs_lulc_scenarios if scenario_data == 'LULC' else maca_scenarios):
            for model in (maca_models if scenario_data == 'MACA' else [None]):
                scenario_suffix = f'{scenario_data}_{scenario}_{model}' if model else f'{scenario_data}_{scenario}'
                for year in scenario_years:
                    local_file_name = f'{scenario_download_dir}Tile_{tile_values[0]}_{year}_{scenario_suffix}.tif'
                    if os.path.exists(local_file_name):
                        try:
                            tile_arr, tile_rio = read_raster_as_arr(local_file_name)
                            if tile_arr.size == 0:
                                if verbose:
                                    logger.warning('Downloaded file is empty. Deleting %s', local_file_name)
                                tile_rio.close()
                                os.remove(local_file_name)
                                raise rio.errors.RasterioIOError
                            tile_rio.close()
                            continue
                        except rio.errors.RasterioIOError:
                            os.remove(local_file_name)
                            if verbose:
                                logger.warning('%s corrupted. Downloading again...', local_file_name)
                    if scenario_data == 'LULC':
                        lulc = usgs_lulc_ic.filterMetadata('scenario', 'equals', scenario) \
                            .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                            .first() \
                            .setDefaultProjection(crs=crs, scale=250)
                        lulc_mask_1 = lulc.eq(13) # cropland
                        lulc_mask_2 = lulc.eq(2) # developed
                        lulc_mask_3 = lulc.eq(6) # mining
                        lulc_mask_4 = lulc.eq(1) # open water
                        lulc_mask = lulc_mask_1.Or(lulc_mask_2).Or(lulc_mask_3).Or(lulc_mask_4)
                        lulc = lulc.updateMask(lulc_mask).remap(
                            [13, 2, 6, 1],
                            [1, 124, 121, 111]
                        ).rename('lulc')
                        data_bands = [lulc]
                        data_band_name_list = ['lulc']
                    else:
                        maca_daily = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA') \
                            .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                            .filterMetadata('model', 'equals', model) \
                            .filterMetadata('scenario', 'equals', scenario)
                        maca_pr = maca_daily.select('pr').sum().rename('annual_precip_mm')
                        maca_tmmx = maca_daily.select('tasmax').mean().rename('annual_tmmx_K')
                        maca_tmmn = maca_daily.select('tasmin').mean().rename('annual_tmmn_K')
                        maca_eto = calculate_eto(
                            year, maca_scenario=scenario, 
                            maca_model=model, return_monthly=False
                        )
                        maca_peff = calculate_annual_peff(
                            year, maca_scenario=scenario,
                            maca_model=model
                        )
                        data_bands = [maca_eto, maca_pr, maca_peff, maca_tmmx, maca_tmmn]
                        data_band_name_list = [
                            'annual_eto_mm', 'annual_precip_mm', 
                            'annual_peff_mm', 'annual_tmmx_K', 'annual_tmmn_K'
                        ]
                    download_gee_tif(
                        data_bands=data_bands,
                        data_band_name_list=data_band_name_list,
                        local_file_name=local_file_name,
                        ee_geom=ee_geom,
                        gee_scale=gee_scale,
                        crs=crs,
                        categorical_bands=categorical_bands,
                        verbose=verbose
                    )


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
        gee_scale: float = 30,
        verbose: bool = False
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

    Returns:
        tuple (str, list (str, ...)): Tuple containing the directory path containing all the downloaded GEE tiles and
        the ordered list of band names for each tile.
    """

    data_dir = f'{download_dir}GEE_Data/GEE_Tiles_{int(gee_scale)}m/'
    data_band_names = [
        'annual_et_ensemble_mm',
        'annual_eto_mm',
        'annual_precip_mm',
        'annual_peff_mm',
        'annual_tmmx_K',
        'annual_tmmn_K',
        'lulc',
        'soil_depth_cm',
        'awc_in',
        'ksat_mean_micromps',
        'irrigation_status'
    ]
    if not skip_download:
        makedirs(data_dir)
        print('Downloading GEE data...')
        ee.Initialize(
            project=gcloud_project,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )
        gee_crs = 'EPSG:4326'
        area_gdf = gpd.read_file(geom_file)
        ee_geom = ee.Geometry.Rectangle(area_gdf.to_crs(gee_crs).total_bounds.tolist())
        huc12_path = 'AZ_HUC12.geojson'
        huc12_local_path = f'{download_dir}GEE_Data/{huc12_path}'
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
            print(f'Waiting to download the HUC12 polygons over the study area from GCloud...')
            while task.active():
                continue
            storage_client = storage.Client()
            storage_bucket = storage_client.bucket(gcloud_bucket)
            gcloud_blob = storage.Blob(bucket=storage_bucket, name=huc12_path)
            print('Downloading', huc12_path)
            gcloud_blob.download_to_filename(huc12_local_path)
        year_list = list(range(start_year, end_year + 1))
        print(f'Creating Fishnet with {tile_size} m tile size...')
        fishnet_gdf = create_fishnet(
            area_gdf,
            fishnet_size=tile_size,
            output_dir=f'{download_dir}/GW_Data/'
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
        print('Each tile has the following bands in order:')
        for band_idx, band_name in enumerate(data_band_names):
            print(band_idx + 1, band_name)
        print('\n')
        tile_chunks = generate_chunks(tile_val_list, num_workers)
        num_chunks = int(np.ceil(fishnet_gdf.shape[0] / num_workers))
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit=worker_memory)
        dask_cluster.scale(num_workers)
        dask_client = Client(dask_cluster)
        dask_client.wait_for_workers(1)
        print(f'Using {num_workers} local workers...')
        for tile_chunk in tqdm(tile_chunks, desc='Processing tile chunks', total=num_chunks):
            compute(
                delayed(download_gee_tile)(
                    tile_vals, data_dir, year_list,
                    data_band_names, gcloud_project,
                    gee_scale,
                    verbose=verbose
                )
                for tile_vals in tile_chunk
            )
        dask_client.shutdown()
    return data_dir, data_band_names


def get_categorical_bands() -> list[str]:
    """
    Get the list of categorical bands.

    Returns:
        List of categorical band names.
    """
    return ['lulc', 'HSG']


def resample_gee_rasters(
        gee_raster_dir: str,
        data_band_names: list[str],
        output_dir: str,
        original_raster_res: float = 30,
        target_raster_res: float = 1000,
        num_workers: int = 32,
        already_resampled: bool = False,
        use_tile_format: bool = True,
        irr_data_only: bool = False
) -> None:
    """
    Resample 30-m GEE tiles to a higher scale.

    Args:
        gee_raster_dir (str): GEE raster directory containing the 30-m multi-band rasters.
                              These can be tiles or mosaics. If these are tiles, then use_tile_format must be True
        data_band_names (str): List of data band names.
        output_dir (str): Output directory.
        original_raster_res (float): Original raster resolution in m.
        target_raster_res (float): Target raster resolution in m.
        num_workers (int): Number of dask workers to use to resample GEE rasters.
        already_resampled (bool): Set True to skip resampling.
        use_tile_format (bool): Set False to resample GEE mosaics instead of tiles.
        irr_data_only (bool): Set True to resample only the irrigated area raster.
                              Works only if use_tile_format is False.

    Returns:
        None.
    """

    if not already_resampled:
        makedirs(output_dir)
        data_band_dict = {}
        categorical_bands = get_categorical_bands()
        for band_num, data_band_name in enumerate(data_band_names):
            if data_band_name not in categorical_bands:
                val = (band_num + 1, 'average', 'float32')
            else:
                gdal_dtype = 'int16'
                if data_band_name in ['lulc', 'HSG']:
                    gdal_dtype = 'byte'
                val = (band_num + 1, 'mode', gdal_dtype)
            data_band_dict[data_band_name] = val
        gee_rasters = sorted(glob(f'{gee_raster_dir}*.tif'))
        resampling_factor = target_raster_res / original_raster_res
        if use_tile_format:
            itr = 1
            gee_tiles = gee_rasters[(itr - 1) * num_workers:]
            num_chunks = int(np.ceil(len(gee_tiles) / num_workers))
            tile_chunks = generate_chunks(gee_tiles, num_workers)
            dask_cluster = LocalCluster(n_workers=num_workers, memory_limit='1.5G')
            dask_cluster.scale(num_workers)
            dask_client = Client(dask_cluster)
            dask_client.wait_for_workers(1)
            print(f'Using {num_workers} local workers...')
            for tile_chunk in tile_chunks:
                print(f'Working on tile chunk {itr} / {num_chunks} ...')
                compute(
                    delayed(reproject_raster_gdal)(
                        tile, None, resampling_factor,
                        'average', True,
                        None, None, None,
                        'float32', data_band_dict,
                        output_dir
                    )
                    for tile in tile_chunk
                )
                itr += 1
            dask_client.shutdown()
        else:
            for gee_raster in gee_rasters:
                gee_file = gee_raster[gee_raster.rfind(os.sep) + 1:]
                gee_resampled_raster = f'{output_dir}{gee_file}'
                if 'IRR' in gee_file:
                    reproject_raster_gdal(
                        gee_raster,
                        gee_resampled_raster,
                        resampling_factor=resampling_factor,
                        output_dtype='int16',
                        resampling_func='sum'
                    )
                elif 'Predictor' in gee_file and not irr_data_only:
                    reproject_raster_gdal(
                        gee_raster,
                        gee_resampled_raster,
                        resampling_factor=resampling_factor,
                        src_band_dict=data_band_dict,
                        dst_raster_dir=output_dir
                    )

        print('All rasters/tiles resampled...')
    else:
        print('GEE tiles already resampled...')





def mosaic_tiles_parallel(
        input_tile_dir: str,
        output_dir: str,
        year: int,
        output_prefix: str = 'Predictor',
        gdal_merge_path: str = '/usr/bin/gdal_merge.py'
) -> None:
    """
    Mosaic all tiles based on the start and end years.

    Args:
        input_tile_dir (str): Input tile directory. The naming convention is Tile_<tile_number>_<year>.tif.
        output_dir (str): Output directory.
        year (int): Year in YYYY.
        output_prefix (str): Output prefix name to append to output files.
        gdal_merge_path (str): Path to the gdal_merge.py script.

    Returns:
        None.
    """

    tiles = ' '.join(glob(f'{input_tile_dir}*{year}.tif'))
    merged_tif = f'{output_dir}{output_prefix}_{year}.tif'
    if os.path.exists(merged_tif):
        os.remove(merged_tif)
    gdal_sys_call = f'{gdal_merge_path} -o {merged_tif} -of GTiff -init 0 {tiles}'
    subprocess.call(
        gdal_sys_call,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def mosaic_tiles(
        input_tile_dir: str,
        output_dir: str,
        start_year: int = 1985,
        end_year: int = 2023,
        output_prefix: str = 'Predictor',
        already_mosaicked: bool = False,
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

    Returns:
        None.
    """

    if not already_mosaicked:
        makedirs(output_dir)
        print('Mosaicking tiles...')
        gdal_merge_path = f'{os.environ["CONDA_PREFIX"]}/bin/gdal_merge.py'
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(mosaic_tiles_parallel)(
                input_tile_dir,
                output_dir,
                year,
                output_prefix,
                gdal_merge_path
            ) for year in range(start_year, end_year + 1)
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
        gee_mosaics = glob(gee_mosaic_dir + '*.tif')
        gw_ref = glob(gw_data_dir + '*.tif')[0]
        for gee_mosaic in gee_mosaics:
            gee_mosaic_reproj = f'{output_dir}{gee_mosaic[gee_mosaic.rfind(os.sep) + 1:]}'
            reproject_raster_gdal(
                gee_mosaic,
                gee_mosaic_reproj,
                from_raster=gw_ref
            )


def create_az_data_csv(
        input_file_dir: str,
        gw_data_dir: str,
        output_dir: str,
        data_band_names: list[str],
        gw_basin_vector: str,
        start_year: int = 1985,
        end_year: int = 2023,
        exclude_years: list[int] | None = None,
        lu_smoothing: int = 3,
        load_csv: bool = False,
) -> pd.DataFrame:
    """
    Create Arizona predictor data CSV.

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
        load_csv (bool): Set True to load existing CSV.

    Returns:
        pd.DataFrame: Output dataframe.
    """

    data_parquet = f'{output_dir}AZ_Data.parquet'
    if not load_csv:
        makedirs(output_dir)
        if exclude_years is None:
            exclude_years = []
        data_df = pd.DataFrame()
        var_names = [
            'Predictor',
            'GW_Basin',
            'Streamflow',
            'GW_Basin_CAP_SRP_Total',
            'Peff'
        ]
        for year in range(start_year, end_year + 1):
            df = pd.DataFrame()
            if year not in exclude_years:
                for var_name in var_names:
                    raster_file = f'{input_file_dir}{var_name}_{year}.tif'
                    if var_name == 'Predictor':
                        for band_num, band_name in enumerate(data_band_names):
                            try:
                                raster_arr = read_raster_as_arr(
                                    raster_file,
                                    band=band_num + 1,
                                    get_file=False
                                )
                                if band_name == 'crop_cdl':
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
                            df['streamflow_m3s'] = raster_arr
                        elif var_name == 'GW_Basin_CAP_SRP_Total':
                            df['cap_srp_delivery_km3'] = raster_arr * 1.23348e-6 # Convert acre-feet to km3
                        elif var_name == 'GW':
                            raster_arr[np.isnan(raster_arr)] = 0
                        elif var_name == 'Peff':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['annual_peff_mm'] = raster_arr
                        else:
                            df[var_name] = raster_arr
                gw_file = f'{gw_data_dir}GW_{year}.tif'
                df['gw_pumping_mm'] = read_raster_as_arr(gw_file, get_file=False).ravel()
                lon_grid, lat_grid = get_xy_grids_from_raster(gw_file)
                df['easting_m'] = lon_grid.ravel()
                df['northing_m'] = lat_grid.ravel()
                df['awc_mm'] = df.awc_cm * 10
                df['Year'] = year
                df = df.drop(columns=['awc_cm'])
                data_df = pd.concat([data_df, df])
        data_df = data_df[~np.isnan(data_df.gw_pumping_mm)].reset_index(drop=True)
        gw_basin_gdf = gpd.read_file(gw_basin_vector)
        gw_basin_dict = {}
        ama_ina_basins = get_ama_ina_basin_names()
        for gw_basin in gw_basin_gdf.OBJECTID:
            gw_basin_name = gw_basin_gdf[gw_basin_gdf.OBJECTID == gw_basin].BASIN_NAME.values[0]
            gw_basin_dict[gw_basin] = gw_basin_name
        nan_str = 'OUTSIDE AZ'
        gw_basin_dict[0] = nan_str
        data_df.GW_Basin = data_df.GW_Basin.swifter.apply(
            lambda x: gw_basin_dict[x] if not np.isnan(x) else nan_str)
        data_df = data_df[data_df.GW_Basin != nan_str]

        ama_basins = [b for b in ama_ina_basins if 'AMA' in b]
        ina_basins = [b for b in ama_ina_basins if b not in ama_basins]
        data_df = data_df.reset_index(drop=True)
        data_df['GW_Basin_Type'] = data_df.GW_Basin.swifter.apply(
            lambda x: 0 if x in ama_basins else 1 if x in ina_basins else 2
        ).reset_index(drop=True)
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
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    for svar in selection_var:
        selected_data = input_df.loc[input_df[selection_label] == svar]
        y = selected_data[pred_attr].to_frame()
        x_train, x_test, y_train, y_test = train_test_split(
            selected_data, y, shuffle=shuffle,
            random_state=random_state, test_size=test_size
        )
        x_train_df = pd.concat([x_train_df, x_train])
        x_test_df = pd.concat([x_test_df, x_test])
        y_train_df = pd.concat([y_train_df, y_train])
        y_test_df = pd.concat([y_test_df, y_test])
    return x_train_df, x_test_df, y_train_df, y_test_df


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
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    for year in years:
        selected_data = input_df.loc[input_df[year_col] == year]
        x_t = selected_data
        if year not in test_years:
            x_train_df = pd.concat([x_train_df, x_t])
        else:
            x_test_df = pd.concat([x_test_df, x_t])
    y_train_df = x_train_df[pred_attr].to_frame()
    y_test_df = x_test_df[pred_attr].to_frame()
    if shuffle:
        x_train_df = sk.shuffle(x_train_df, random_state=random_state)
        y_train_df = sk.shuffle(y_train_df, random_state=random_state)
        x_test_df = sk.shuffle(x_test_df, random_state=random_state)
        y_test_df = sk.shuffle(y_test_df, random_state=random_state)
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
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    for gw_basin in gw_basins:
        selected_data = input_df.loc[input_df[gw_basin_col] == gw_basin]
        x_t = selected_data
        if gw_basin not in test_gw_basins:
            x_train_df = pd.concat([x_train_df, x_t])
        else:
            x_test_df = pd.concat([x_test_df, x_t])
    y_train_df = x_train_df[pred_attr].to_frame()
    y_test_df = x_test_df[pred_attr].to_frame()
    if shuffle:
        x_train_df = sk.shuffle(x_train_df, random_state=random_state)
        y_train_df = sk.shuffle(y_train_df, random_state=random_state)
        x_test_df = sk.shuffle(x_test_df, random_state=random_state)
        y_test_df = sk.shuffle(y_test_df, random_state=random_state)
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
                print(f'{gw_basin} {year} outliers: {outliers}')
                num_outliers += outliers
                input_df.loc[selection, 'Outlier'] = invalid_idx
        input_df = input_df[input_df['Outlier'] == False]
        input_df = input_df.drop(columns='Outlier')
    elif operation == 3:
        invalid_idx = (input_df[target_attr] < min_gw_pumping) | (input_df[target_attr] > max_gw_pumping)
        input_df.loc[invalid_idx, target_attr] = np.nan
        num_outliers = invalid_idx.sum()
    input_df = input_df.dropna()
    print('Old DF rows = {}, New DF rows = {}'.format(init_rows, input_df.shape[0]))
    print(f'{num_outliers} outliers removed...')
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
    x_train_file = output_dir + 'X_train.parquet'
    x_test_file = output_dir + 'X_test.parquet'
    y_train_file = output_dir + 'y_train.parquet'
    y_test_file = output_dir + 'y_test.parquet'
    year_train_file = output_dir + 'Year_train.parquet'
    year_test_file = output_dir + 'Year_test.parquet'
    gw_basin_train_file = output_dir + 'GW_Basin_train.parquet'
    gw_basin_test_file = output_dir + 'GW_Basin_test.parquet'
    x_scaler_file, x_scaler, y_scaler_file, y_scaler = [None] * 4
    if scaling:
        x_scaler_file = output_dir + 'x_scaler'
        y_scaler_file = output_dir + 'y_scaler'
    if not already_created:
        if use_ama_ina:
            ama_ina_basins = get_ama_ina_basin_names()
            ama_ina_basins = [b for b in ama_ina_basins if b not in drop_gw_basins]
            input_df = input_df[input_df[gw_basin_col].isin(ama_ina_basins)]
        drop_attr = [attr for attr in drop_attr]
        input_df = input_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        input_df = input_df[input_df[pred_attr] > 0] if water_use == 'IRRIGATION' else input_df
        if year_list and year_col in input_df.columns:
            input_df = input_df[input_df[year_col].isin(year_list)]
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
        input_df.to_parquet(output_dir + 'Cleaned_AZ_GW_Data.parquet', index=False)
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
            pickle.dump(x_scaler, open(x_scaler_file, mode='wb'))
            pickle.dump(y_scaler, open(y_scaler_file, mode='wb'))
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
            x_scaler = pickle.load(open(x_scaler_file, mode='rb'))
            y_scaler = pickle.load(open(y_scaler_file, mode='rb'))
    ret_vals = (
        x_train, x_test, y_train.to_numpy().ravel(), y_test.to_numpy().ravel(),
        x_scaler, y_scaler, year_train, year_test, gw_basin_train, gw_basin_test
    )

    return ret_vals
