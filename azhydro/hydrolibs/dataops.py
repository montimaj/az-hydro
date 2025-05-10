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
from osgeo import gdal
import rasterio as rio
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
import swifter
import joblib
import pickle
import sklearn.utils as sk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sysops import makedirs
from gwops import create_land_use_data, get_ama_ina_basin_names
from rasterops import reproject_raster_gdal, read_raster_as_arr, write_raster, get_xy_grids_from_raster
from google.cloud import storage
from shapely.geometry import Polygon
from glob import glob
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from http.client import RemoteDisconnected


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


def download_gee_tile(
        tile_values: tuple[int, float, float, float],
        download_dir: str,
        year_list: list,
        data_band_names: list[str],
        gcloud_project: str = 'azhydro',
        gee_scale: int = 30,
        irrigated_tiles: bool = True,
        multiply_irr_mask: bool = True,
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
    gee_scale (int): GEE data download scale in m.
    irrigated_tiles (bool): Set False to download all tiles regardless of irrigation.
    multiply_irr_mask (bool): Set False to skip multiplying image bands by the irrigation mask.
    verbose (bool): Set True to see extra details on file downloads.
    crs (str): Coordinate reference system for the downloaded data. Defaults to 'EPSG:4326'.

    Returns:
        None.
    """

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
            print('Initialization exception', e)
            retry_ee_init = True
            time.sleep(1)
    openet_ic = [
        ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/ensemble/conus/gridmet/monthly/provisional")
    ]
    ssebop_ic = [
        ee.ImageCollection("OpenET/SSEBOP/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/ssebop/conus/gridmet/monthly/provisional")
    ]
    eemetric_ic = [
        ee.ImageCollection("OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/eemetric/conus/gridmet/monthly/provisional")
    ]
    sims_ic = [
        ee.ImageCollection("OpenET/SIMS/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/sims/conus/gridmet/monthly/provisional")
    ]
    pt_jpl_ic = [
        ee.ImageCollection("OpenET/PTJPL/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/ptjpl/conus/gridmet/monthly/provisional")
    ]
    geesebal_ic = [
        ee.ImageCollection("OpenET/GEESEBAL/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/geesebal/conus/gridmet/monthly/provisional")
    ]
    disalexi_ic = [
        ee.ImageCollection("OpenET/DISALEXI/CONUS/GRIDMET/MONTHLY/v2_0"),
        ee.ImageCollection("projects/openet/assets/disalexi/conus/gridmet/monthly/provisional")
    ]
    gridmet_ic = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    gridmet_drought_ic = ee.ImageCollection("GRIDMET/DROUGHT")
    prism_ic = ee.ImageCollection('OREGONSTATE/PRISM/AN81m')
    daymet_ic = ee.ImageCollection('NASA/ORNL/DAYMET_V4')
    terraclimate_ic = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    era5land_ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    npp_ic = [
        ee.ImageCollection('UMT/NTSG/v2/LANDSAT/NPP'),
        ee.ImageCollection('MODIS/061/MOD17A3HGF')
    ]
    irrmapper_ic = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp')
    cdl_ic = ee.ImageCollection("USDA/NASS/CDL")
    nlcd_ic = ee.ImageCollection('projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER')
    hsg = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/'
        'hydrologic_group'
    ).rename('HSG')
    soil_depth = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/land_use/soil_depth'
    ).rename('soil_depth_mm')
    ksat_mean = ee.Image(
        'projects/earthengine-legacy/assets/projects/sat-io/open-datasets/CSRL_soil_properties/physical/ksat_mean'
    ).rename('ksat_mean_micromps')
    nasa_dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').rename('nasadem_elevation_m')
    nasa_dem_slope = ee.Terrain.slope(nasa_dem).rename('nasadem_slope')
    bd_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/bd_mean').first().rename('bulk_density_gcm3')
    clay_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').first().rename('clay_percent')
    polaris_ksat_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/ksat_mean'
    ).first().rename('ksat_log10cmhr1')
    n_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/n_mean').first().rename('pore_size_dist')
    om_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/om_mean'
    ).first().rename('organic_matter_log10percent')
    ph_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/ph_mean').first().rename('soil_ph')
    sand_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').first().rename('sand_percent')
    silt_mean = ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').first().rename('silt_percent')
    theta_r_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/theta_r_mean'
    ).first().rename('residual_swc')
    theta_s_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/theta_s_mean'
    ).first().rename('saturated_swc')
    lambda_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/lambda_mean'
    ).first().rename('pore_size_index')
    hb_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/hb_mean'
    ).first().rename('soil_bubbling_pressure_log10kPa')
    alpha_mean = ee.ImageCollection(
        'projects/sat-io/open-datasets/polaris/alpha_mean'
    ).first().rename('polaris_scale_log10kPa1')
    ee_geom = ee.Geometry.Rectangle(tile_values[1:])
    categorical_bands = get_categorical_bands()
    max_pixels = np.ceil((gee_scale / 30) ** 2)
    for year in year_list:
        local_file_name = f'{download_dir}Tile_{tile_values[0]}_{year}.tif'
        if os.path.exists(local_file_name):
            try:
                tile_arr, tile_rio = read_raster_as_arr(local_file_name)
                if tile_arr.size == 0:
                    print('Downloaded file is empty. Deleting', local_file_name)
                    tile_rio.close()
                    os.remove(local_file_name)
                    raise rio.errors.RasterioIOError
                tile_rio.close()
                continue
            except rio.errors.RasterioIOError:
                os.remove(local_file_name)
                print(local_file_name, 'corrupted. Downloading again...')
        irr = irrmapper_ic.filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
            .select('classification') \
            .max()
        mask = irr.eq(0)
        irr_mask = irr.updateMask(mask).remap([0], [1])
        valid_tile = True
        if irrigated_tiles:
            tile_irr_sum = irr_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                scale=30,
                geometry=ee_geom
            )
            retry_irr_mask = True
            while retry_irr_mask:
                try:
                    valid_tile = tile_irr_sum.getInfo()['remapped'] > 0
                    retry_irr_mask = False
                except (
                        ee.EEException, requests.exceptions.RequestException,
                        requests.exceptions.ConnectionError, RemoteDisconnected
                ) as e:
                    print('Error:', e, '.Retrying...')
                    retry_irr_mask = True
                    time.sleep(5)
        if valid_tile:
            openet_idx = 1 if year < 2000 else 0
            start_year_gee = f'{year}-01-01'
            end_year_gee = f'{year + 1}-01-01'
            openet_ensemble = openet_ic[openet_idx].select('et_ensemble_mad') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            ssebop_et = ssebop_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            eemetric_et = eemetric_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            sims_et = sims_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            pt_jpl_et = pt_jpl_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            geesebal_et = geesebal_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            disalexi_et = disalexi_ic[openet_idx].select('et') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            gridmet_precip = gridmet_ic.select('pr') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            gridmet_tmmx = gridmet_ic.select('tmmx') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_tmmn = gridmet_ic.select('tmmn') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_eto = gridmet_ic.select('eto') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            gridmet_etr = gridmet_ic.select('etr') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            gridmet_vpd = gridmet_ic.select('vpd') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            gridmet_vs = gridmet_ic.select('vs') \
                .filterDate(start_year_gee, end_year_gee) \
                .mean()
            gridmet_rmax = gridmet_ic.select('rmax') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_rmin = gridmet_ic.select('rmin') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_spi1y = gridmet_drought_ic.select('spi1y') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_eddi1y = gridmet_drought_ic.select('eddi1y') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_spei1y = gridmet_drought_ic.select('spei1y') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            gridmet_pdsi = gridmet_drought_ic.select('pdsi') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            prism_precip = prism_ic.select('ppt') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            prism_tmmx = prism_ic.select('tmax') \
                .filterDate(start_year_gee, end_year_gee) \
                .median() \
                .add(273.15)
            prism_tmmn = prism_ic.select('tmin') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()\
                .add(273.15)
            if year < 2024:
                daymet_precip = daymet_ic.select('prcp') \
                    .filterDate(start_year_gee, end_year_gee) \
                    .sum()
                daymet_tmmx = daymet_ic.select('tmax') \
                    .filterDate(start_year_gee, end_year_gee) \
                    .median() \
                    .add(273.15)
                daymet_tmmn = daymet_ic.select('tmin') \
                    .filterDate(start_year_gee, end_year_gee) \
                    .median() \
                    .add(273.15)
            else:
                daymet_precip = gridmet_precip
                daymet_tmmx = gridmet_tmmx
                daymet_tmmn = gridmet_tmmn
            terraclimate_sm_first = terraclimate_ic.select('soil') \
                .filterDate(start_year_gee, end_year_gee) \
                .first()
            terraclimate_sm = terraclimate_ic.select('soil') \
                .filterDate(start_year_gee, end_year_gee) \
                .sort('system:time_start', False) \
                .first() \
                .subtract(terraclimate_sm_first) \
                .multiply(0.1)
            terraclimate_ro = terraclimate_ic.select('ro') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_snowfall = era5land_ic.select('snowfall_sum') \
                .filterDate(f'{year - 1}-10-01', f'{year}-10-01') \
                .sum()
            era5land_swe = era5land_ic.select('snow_depth_water_equivalent') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_snowmelt = era5land_ic.select('snowmelt_sum') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_runoff = era5land_ic.select('runoff_sum') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_sub_surface_runoff = era5land_ic.select('sub_surface_runoff_sum') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_surface_runoff = era5land_ic.select('surface_runoff_sum') \
                .filterDate(start_year_gee, end_year_gee) \
                .sum()
            era5land_volumetric_soil_water_layer_1 = era5land_ic.select('volumetric_soil_water_layer_1') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            era5land_volumetric_soil_water_layer_2 = era5land_ic.select('volumetric_soil_water_layer_2') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            era5land_volumetric_soil_water_layer_3 = era5land_ic.select('volumetric_soil_water_layer_3') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            era5land_volumetric_soil_water_layer_4 = era5land_ic.select('volumetric_soil_water_layer_4') \
                .filterDate(start_year_gee, end_year_gee) \
                .median()
            npp_idx = 0 if year < 2021 else 1
            npp_band = 'annualNPP' if npp_idx == 0 else 'Npp'
            npp_start_year = start_year_gee if year >= 1986 else '1986-01-01'
            npp_end_year = end_year_gee if year >= 1986 else '1987-01-01'
            npp = npp_ic[npp_idx].select(npp_band) \
                .filterDate(npp_start_year, npp_end_year) \
                .sum() \
                .rename('annual_npp_mm4')
            if year < 2008:
                # Switch to NLCD for 1985-2007
                nlcd_year = nlcd_ic.filterDate(f'{year}-01-01', f'{year + 1}-01-01').first()
                nlcd_mask_1 = nlcd_year.eq(82) # cropland
                nlcd_mask_2 = nlcd_year.gte(21).And(nlcd_year.lte(24)) # developed
                nlcd_mask_3 = nlcd_year.eq(11) # open water
                nlcd_mask = nlcd_mask_1.Or(nlcd_mask_2).Or(nlcd_mask_3)
                cdl = nlcd_year.updateMask(nlcd_mask).remap(
                    [82, 21, 22, 23, 24, 11],
                    [1, 121, 122, 123, 124, 111]
                ).rename('cropland')
            else:
                cdl = cdl_ic.filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .select('cropland') \
                    .first()
            data_bands = [
                openet_ensemble,
                ssebop_et,
                eemetric_et,
                sims_et,
                pt_jpl_et,
                geesebal_et,
                disalexi_et,
                gridmet_precip,
                gridmet_tmmx,
                gridmet_tmmn,
                gridmet_eto,
                gridmet_etr,
                gridmet_vpd,
                gridmet_vs,
                gridmet_rmax,
                gridmet_rmin,
                gridmet_spi1y,
                gridmet_eddi1y,
                gridmet_spei1y,
                gridmet_pdsi,
                prism_precip,
                prism_tmmx,
                prism_tmmn,
                daymet_precip,
                daymet_tmmx,
                daymet_tmmn,
                terraclimate_sm,
                terraclimate_ro,
                era5land_snowfall,
                era5land_swe,
                era5land_snowmelt,
                era5land_runoff,
                era5land_sub_surface_runoff,
                era5land_surface_runoff,
                era5land_volumetric_soil_water_layer_1,
                era5land_volumetric_soil_water_layer_2,
                era5land_volumetric_soil_water_layer_3,
                era5land_volumetric_soil_water_layer_4,
                npp,
                cdl,
                hsg,
                soil_depth,
                ksat_mean,
                nasa_dem,
                nasa_dem_slope,
                bd_mean,
                clay_mean,
                polaris_ksat_mean,
                n_mean,
                om_mean,
                ph_mean,
                sand_mean,
                silt_mean,
                theta_r_mean,
                theta_s_mean,
                lambda_mean,
                hb_mean,
                alpha_mean,
                irr_mask
            ]
            data_img = openet_ensemble.rename(data_band_names[0])
            data_img = data_img.setDefaultProjection(
                crs=crs,
                scale=30
            ).reduceResolution(
                reducer=ee.Reducer.mean(),
                bestEffort=True,
                maxPixels=max_pixels
            ).reproject(crs, scale=gee_scale)
            for band, band_name in zip(data_bands, data_band_names):
                if band_name == 'annual_et_disalexi_mm' and year < 2001:
                    band = ee.ImageCollection([
                        ssebop_et,
                        eemetric_et,
                        sims_et,
                        pt_jpl_et,
                        geesebal_et,
                    ]).mean()
                if multiply_irr_mask:
                    band = band.multiply(irr_mask)
                band = band.rename(band_name)
                band_scale = band.projection().nominalScale().getInfo()
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
                        band = band.setDefaultProjection(
                            crs=crs,
                            scale=30
                        ).reduceResolution(
                            reducer=reducer,
                            bestEffort=True,
                            maxPixels=max_pixels
                        ).reproject(crs, scale=gee_scale)
                    else:
                        band = band.setDefaultProjection(
                            crs=crs,
                            scale=30
                        ).reproject(crs, scale=gee_scale)
                if band_name == 'annual_irrmapper_fraction':
                    band = band.multiply(900).divide(gee_scale ** 2)
                    band = band.where(band.gt(1), 1)
                data_img = data_img.addBands(band, overwrite=True)
            retry_download = True
            while retry_download:
                try:
                    gee_url = data_img.getDownloadUrl({
                        'scale': gee_scale,
                        'region': ee_geom,
                        'format': 'GEO_TIFF',
                        'crs': crs
                    })
                    if verbose:
                        print('Dowloading', local_file_name, '...')
                    r = requests.get(
                        gee_url,
                        allow_redirects=True,
                        timeout=None,
                        stream=True
                    )
                    with open(local_file_name, 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=1024):
                            fd.write(chunk)
                    tile_arr, tile_rio = read_raster_as_arr(local_file_name)
                    if tile_arr.size == 0:
                        print('Downloaded file is empty. Deleting', local_file_name)
                        tile_rio.close()
                        os.remove(local_file_name)
                        raise rio.errors.RasterioIOError
                    tile_rio.close()
                    retry_download = False
                except (
                        ee.EEException, requests.exceptions.RequestException,
                        requests.exceptions.ConnectionError, RemoteDisconnected,
                        rio.errors.RasterioIOError
                ) as e:
                    print('Error', e, 'during', local_file_name, 'download!')
                    print('Retrying download...')
                    retry_download = True
                time.sleep(0.001)
    time.sleep(0.001)


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
        gee_scale: int = 30,
        irrigated_tiles: bool = True,
        multiply_irr_mask: bool = True
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
    gee_scale (int): GEE data download scale in m.
    irrigated_tiles (bool): Set False to download all tiles regardless of irrigation.
    multiply_irr_mask (bool): Set False to skip multiplying image bands by the irrigation mask.

    Returns:
        tuple (str, list (str, ...)): Tuple containing the directory path containing all the downloaded GEE tiles and
        the ordered list of band names for each tile.
    """

    data_dir = f'{download_dir}GEE_Data/GEE_Tiles_{gee_scale}m_IrrMaskMul_{multiply_irr_mask}/'
    data_band_names = [
        'annual_et_ensemble_mm',
        'annual_et_ssebop_mm',
        'annual_et_eemetric_mm',
        'annual_et_sims_mm',
        'annual_et_pt_jpl_mm',
        'annual_et_geesebal_mm',
        'annual_et_disalexi_mm',
        'annual_gridmet_precip_mm',
        'annual_gridmet_tmmx_K',
        'annual_gridmet_tmmn_K',
        'annual_gridmet_eto_mm',
        'annual_gridmet_etr_mm',
        'annual_gridmet_vpd_kPa',
        'annual_gridmet_vs_mps',
        'annual_gridmet_rmax',
        'annual_gridmet_rmin',
        'annual_gridmet_spi1y',
        'annual_gridmet_eddi1y',
        'annual_gridmet_spei1y',
        'annual_gridmet_pdsi',
        'annual_prism_precip_mm',
        'annual_prism_tmmx_K',
        'annual_prism_tmmn_K',
        'annual_daymet_precip_mm',
        'annual_daymet_tmmx_K',
        'annual_daymet_tmmn_K',
        'annual_terraclimate_sm_change_mm',
        'annual_terraclimate_ro_mm',
        'wy_era5land_snowfall_m',
        'annual_era5land_swe_m',
        'annual_era5land_snowmelt_m',
        'annual_era5land_runoff_m',
        'annual_era5land_sub_surface_runoff_m',
        'annual_era5land_surface_runoff_m',
        'annual_era5land_volumetric_soil_water_layer_1',
        'annual_era5land_volumetric_soil_water_layer_2',
        'annual_era5land_volumetric_soil_water_layer_3',
        'annual_era5land_volumetric_soil_water_layer_4',
        'annual_npp_mm4',
        'crop_cdl',
        'HSG',
        'soil_depth_mm',
        'ksat_mean_micromps',
        'elevation_m',
        'slope',
        'bulk_density_gcm3',
        'clay_percent',
        'ksat_log10cmhr1',
        'pore_size_dist',
        'organic_matter_log10percent',
        'soil_ph',
        'sand_percent',
        'silt_percent',
        'residual_swc',
        'saturated_swc',
        'pore_size_index',
        'soil_bubbling_pressure_log10kPa',
        'polaris_scale_log10kPa1',
        'annual_irrmapper_fraction'
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
        itr = 1
        num_chunks = int(np.ceil(fishnet_gdf.shape[0] / num_workers))
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit=worker_memory)
        dask_cluster.scale(num_workers)
        dask_client = Client(dask_cluster)
        dask_client.wait_for_workers(1)
        print(f'Using {num_workers} local workers...')
        for tile_chunk in tile_chunks:
            print(f'Working on tile chunk {itr} / {num_chunks} ...')
            compute(
                delayed(download_gee_tile)(
                    tile_vals, data_dir, year_list,
                    data_band_names, gcloud_project,
                    gee_scale, irrigated_tiles,
                    multiply_irr_mask,
                    verbose=False
                )
                for tile_vals in tile_chunk
            )
            itr += 1
        dask_client.shutdown()
    return data_dir, data_band_names


def get_categorical_bands() -> list[str]:
    """
    Get the list of categorical bands.

    Returns:
        List of categorical band names.
    """
    return ['crop_cdl', 'HSG']


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
                if data_band_name in ['crop_cdl', 'HSG']:
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


def create_irrigation_tiles_parallel(
        year: int,
        gee_tile_dir: str,
        irr_30m_dir: str,
        output_prefix: str,
        irr_resampled_dir: str,
        resampling_factor: float
) -> None:
    """
    Parallely create yearly irrigated area tiles at 30 m and resampled resolutions.

    Args:
        year (int): Year in YYYY format.
        gee_tile_dir (str): Directory containing the yearly tiles.
        irr_30m_dir (str): 30-m irrigated area raster directory.
        output_prefix (str): Output prefix for irrigated area files.
        irr_resampled_dir (str): Resampled irrigated area raster directory.
        resampling_factor (float): Tile resampling factor.

    Returns:
        None.
    """
    print('Creating irrigation rasters for', year, '...')
    gee_yearly_tiles = glob(f'{gee_tile_dir}*{year}.tif')
    for tile in gee_yearly_tiles:
        tile_file = tile[tile.rfind(os.sep) + 1:]
        try:
            tile_arr, tile_obj = read_raster_as_arr(tile)
            tile_arr[tile_arr > 0] = 1
            irr_raster_file_30m = f'{irr_30m_dir}{output_prefix}_{tile_file}'
            write_raster(
                tile_arr,
                tile_obj,
                transform_=tile_obj.transform,
                outfile_path=irr_raster_file_30m,
                no_data_value=0,
                num_bands=1
            )
            irr_raster_file_resampled = f'{irr_resampled_dir}{output_prefix}_{tile_file}'
            reproject_raster_gdal(
                irr_raster_file_30m,
                irr_raster_file_resampled,
                resampling_factor=resampling_factor,
                resampling_func='sum',
                output_dtype='int32'
            )
        except Exception as e:
            print('Error occured while processing', tile, '\n', e)
            continue


def create_irrigation_tiles(
        gee_tile_dir: str,
        output_dir: str,
        start_year: int = 1985,
        end_year: int = 2024,
        raster_res: float = 1000,
        output_prefix: str = 'IRR',
        already_created: bool = False
) -> str:
    """
    Create irrigation tiles by aggregating 30-m GEE tiles which are already masked using IrrMapper.

    Args:
        gee_tile_dir: GEE tile directory containing the 30-m multi-band rasters.
        output_dir (str): Output directory.
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        raster_res (float): Raster resolution in m.
        output_prefix (str): Prefix for the output file names.
        already_created (bool): Set True to skip creating irrigated area rasters.

    Returns:
        str: Directory path to the resampled irrigation tiles. Each <raster_res> size pixel contains the number of
             irrigated pixels at the 30-m scale.
    """

    irr_resampled_dir = f'{output_dir}Irrigation_{raster_res}m/'
    if not already_created:
        irr_30m_dir = f'{output_dir}Irrigation_30m/'
        makedirs((irr_30m_dir, irr_resampled_dir))
        resampling_factor = raster_res / 30
        joblib.Parallel(n_jobs=-1)(joblib.delayed(create_irrigation_tiles_parallel)(
            year, gee_tile_dir, irr_30m_dir, output_prefix,
            irr_resampled_dir, resampling_factor
        ) for year in range(start_year, end_year + 1))
    return irr_resampled_dir


def mosaic_tiles_parallel(
        input_tile_dir: str,
        output_dir: str,
        year: int,
        output_prefix: str = 'Predictor'
) -> None:
    """
    Mosaic all tiles based on the start and end years.

    Args:
        input_tile_dir (str): Input tile directory. The naming convention is Tile_<tile_number>_<year>.tif.
        output_dir (str): Output directory.
        year (int): Year in YYYY.
        output_prefix (str): Output prefix name to append to output files.

    Returns:
        None.
    """

    tiles = ' '.join(glob(f'{input_tile_dir}*{year}.tif'))
    merged_tif = f'{output_dir}{output_prefix}_{year}.tif'
    if os.path.exists(merged_tif):
        os.remove(merged_tif)
    gdal_sys_call = f'{os.environ["CONDA_PREFIX"]}/bin/gdal_merge.py -o {merged_tif} -of GTiff -init 0 {tiles}'
    subprocess.call(
        gdal_sys_call,
        shell=True,
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
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(mosaic_tiles_parallel)(
                input_tile_dir,
                output_dir,
                year,
                output_prefix
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
        nan_str = 'OUTSIDE AZ'
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
                                print('IndexError for', raster_file)
                                df[band_name] = 0
                    else:
                        raster_arr = read_raster_as_arr(raster_file, get_file=False).ravel()
                        if var_name == 'Streamflow':
                            raster_arr[np.isnan(raster_arr)] = 0
                            df['streamflow_m3s'] = raster_arr
                        elif var_name == 'GW_Basin_CAP_SRP_Total':
                            df['cap_srp_delivery_m3'] = raster_arr * 1233.48
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
                df['lon_deg'] = lon_grid.ravel()
                df['lat_deg'] = lat_grid.ravel()
                df['Year'] = year
                data_df = pd.concat([data_df, df])
        data_df = data_df[~np.isnan(data_df.gw_pumping_mm)].reset_index(drop=True)
        gw_basin_gdf = gpd.read_file(gw_basin_vector)
        gw_basin_dict = {}
        ama_ina_basins = get_ama_ina_basin_names()
        # find Douglas basin name in ama_ina_basins
        douglas_basin_name = [b for b in ama_ina_basins if 'DOUGLAS' in b][0]
        for gw_basin in gw_basin_gdf.OBJECTID:
            gw_basin_name = gw_basin_gdf[gw_basin_gdf.OBJECTID == gw_basin].BASIN_NAME.values[0]
            if 'DOUGLAS' in gw_basin_name:
                gw_basin_name = douglas_basin_name
            gw_basin_dict[gw_basin] = gw_basin_name
        gw_basin_dict[0] = nan_str
        data_df.GW_Basin = data_df.GW_Basin.swifter.apply(
            lambda x: gw_basin_dict[x] if not np.isnan(x) else nan_str)
        data_df = data_df[data_df.GW_Basin != nan_str]

        ama_basins = ama_ina_basins[:7]
        ina_basins = ama_ina_basins[7:]
        data_df = data_df.reset_index(drop=True)
        data_df['GW_Basin_Type'] = data_df.GW_Basin.swifter.apply(
            lambda x: 0 if x in ama_basins else 1 if x in ina_basins else 2
        ).reset_index(drop=True)
        data_df['GW_Basin_Type'] = data_df.swifter.apply(
            lambda x: 0 if (x.Year < 2023) & (x.GW_Basin == douglas_basin_name)
            else 1 if x.GW_Basin == douglas_basin_name else x.GW_Basin_Type,
            axis=1
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
        irrigation_col: str = 'annual_irrmapper_fraction',
        split_strategy: int = 1,
        outlier_op: int | None = 2,
        min_gw_pumping: float = 0,
        max_gw_pumping: float = np.inf,
        shuffle: bool = True,
        use_ama_ina: bool = False,
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
        irrigation_col (str): Name of the irrigation column.
        split_strategy (int): If 1, Split train test data based on year_col. If 2, then test_size amount of data from
                              year_col are kept for testing and rest for training;
                              for this option, test-year should have a tuple of integers or a True value. If 3, then
                              test_gw_basins are used for spatial holdouts. For any other value of split-strategy,
                              the data are randomly split.
        outlier_op (int): Outlier operation to perform. Set to 1 for removing outlier directly or 2 for removing
                          outliers by each basin and each year. Set to 3 to remove all values less than min_gw_pumping and greater than
                            max_gw_pumping.
        min_gw_pumping (float): Minimum gw pumping value in mm. Default is 0.
        max_gw_pumping (float): Maximum gw pumping value in mm. Default is np.inf.
        shuffle (bool): Set False to stop data shuffling.
        use_ama_ina (bool): Set True to use AMA-INA basins.

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
            input_df = input_df[input_df[gw_basin_col].isin(ama_ina_basins)]
        drop_attr = [attr for attr in drop_attr]
        input_df = input_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        input_df = input_df[input_df[pred_attr] > 0]
        # input_df = input_df[input_df[irrigation_col] > 0]
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
