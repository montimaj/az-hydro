"""
Contains codes for handling Google Earth Engine datasets
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import ee
import os
import time
import requests
import subprocess
import numpy as np
import geopandas as gpd
import rasterio as rio

from sysops import makedirs
from rasterops import reproject_raster_gdal, read_raster_as_arr, write_raster
from google.cloud import storage
from shapely.geometry import Polygon
from glob import glob
from dask import delayed, compute
from dask.distributed import Client, LocalCluster


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
        data_band_names: list,
        gcloud_project: str = 'azhydro',
        verbose: bool = False
):
    """
    Download GEE tile through dask.

    Args:
    tile_values (tuple (int, float, float, float)): Tile values as a tuple of (FID, xmin, ymin, xmax, ymax)
    download_dir (str): Download directory path.
    year_list (list): List of years in YYYY format.
    data_band_names (list): List of data bands as strings.
    gcloud_project (str): GCloud project name.
    verbose (bool): Set True to see extra details on file downloads.

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
    gridmet_bc_ic = ee.ImageCollection('projects/openet/reference_et/gridmet/monthly')
    prism_ic = ee.ImageCollection('OREGONSTATE/PRISM/AN81m')
    daymet_ic = ee.ImageCollection('NASA/ORNL/DAYMET_V4')
    conus404_ic = ee.ImageCollection('projects/openet/meteorology/conus/conus404/daily')
    terraclimate_ic = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
    irrmapper_ic = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp')
    cdl_ic = ee.ImageCollection("USDA/NASS/CDL")
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
    nasa_dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').rename('elevation_m')
    nasa_dem_slope = ee.Terrain.slope(nasa_dem).rename('slope')
    ee_geom = ee.Geometry.Rectangle(tile_values[1:])
    for year in year_list:
        local_file_name = f'{download_dir}Tile_{tile_values[0]}_{year}.tif'
        if os.path.exists(local_file_name):
            try:
                tile_rio = rio.open(local_file_name, mode='r')
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
        tile_irr_sum = irr_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            scale=30,
            geometry=ee_geom
        )
        retry_irr_mask = True
        valid_tile = True
        while retry_irr_mask:
            try:
                valid_tile = tile_irr_sum.getInfo()['remapped'] > 0
                retry_irr_mask = False
            except (ee.EEException, requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
                print('Error:', e, '.Retrying...')
                retry_irr_mask = True
                time.sleep(5)
        if valid_tile:
            openet_idx = 0
            if year < 2016:
                openet_idx = 1
            openet_ensemble = openet_ic[openet_idx].select('et_ensemble_mad') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            ssebop_et = ssebop_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            eemetric_et = eemetric_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            sims_et = sims_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            pt_jpl_et = pt_jpl_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            geesebal_et = geesebal_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            disalexi_et = disalexi_ic[openet_idx].select('et') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_precip = gridmet_ic.select('pr') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_tmmx = gridmet_ic.select('tmmx') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_tmmn = gridmet_ic.select('tmmn') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_eto = gridmet_ic.select('eto') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_etr = gridmet_ic.select('etr') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_bc_eto = gridmet_bc_ic.select('eto') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_bc_etr = gridmet_bc_ic.select('etr') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_vpd = gridmet_ic.select('vpd') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            gridmet_vs = gridmet_ic.select('vs') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .mean()
            gridmet_rmax = gridmet_ic.select('rmax') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_rmin = gridmet_ic.select('rmin') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_spi1y = gridmet_drought_ic.select('spi1y') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_eddi1y = gridmet_drought_ic.select('eddi1y') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_spei1y = gridmet_drought_ic.select('spei1y') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            gridmet_pdsi = gridmet_drought_ic.select('pdsi') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()
            prism_precip = prism_ic.select('ppt') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            prism_tmmx = prism_ic.select('tmax') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median() \
                .add(273.15)
            prism_tmmn = prism_ic.select('tmin') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .median()\
                .add(273.15)
            terraclimate_sm_first = terraclimate_ic.select('soil') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .first()
            terraclimate_sm = terraclimate_ic.select('soil') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sort('system:time_start', False) \
                .first() \
                .subtract(terraclimate_sm_first) \
                .multiply(0.1)
            terraclimate_ro = terraclimate_ic.select('ro') \
                .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                .sum()
            if year < 2023:
                daymet_precip = daymet_ic.select('prcp') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .sum()
                daymet_tmmx = daymet_ic.select('tmax') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .median() \
                    .add(273.15)
                daymet_tmmn = daymet_ic.select('tmin') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .median() \
                    .add(273.15)
                conus404_precip = conus404_ic.select('PREC_ACC_NC') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .sum()
                conus404_tmmx = conus404_ic.select('T2_MAX') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .median() \
                    .add(273.15)
                conus404_tmmn = conus404_ic.select('T2_MIN') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .median() \
                    .add(273.15)
                conus404_eto = conus404_ic.select('ETO_ASCE') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .sum()
                conus404_etr = conus404_ic.select('ETR_ASCE') \
                    .filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
                    .sum()
            else:
                daymet_precip = gridmet_precip
                daymet_tmmx = gridmet_tmmx
                daymet_tmmn = gridmet_tmmn
                conus404_precip = gridmet_precip
                conus404_tmmx = gridmet_tmmx
                conus404_tmmn = gridmet_tmmn
                conus404_eto = gridmet_bc_eto
                conus404_etr = gridmet_bc_etr
            cdl_start_year = year
            cdl_end_year = year + 1
            if year < 2008:
                cdl_start_year = 2008
                cdl_end_year = year_list[-1]
                cdl = cdl_ic.filterDate(f'{cdl_start_year}-01-01', f'{cdl_end_year + 1}-01-01') \
                    .select('cropland') \
                    .mode()
            else:
                cdl = cdl_ic.filterDate(f'{cdl_start_year}-01-01', f'{cdl_end_year + 1}-01-01') \
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
                gridmet_bc_eto,
                gridmet_bc_etr,
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
                conus404_precip,
                conus404_tmmx,
                conus404_tmmn,
                conus404_eto,
                conus404_etr,
                terraclimate_sm,
                terraclimate_ro,
                cdl,
                hsg,
                soil_depth,
                ksat_mean,
                nasa_dem,
                nasa_dem_slope
            ]
            data_img = openet_ensemble.rename(data_band_names[0])
            for band, band_name in zip(data_bands, data_band_names):
                if band_name == 'annual_et_disalexi_mm' and year < 2001:
                    band = ee.ImageCollection([
                        ssebop_et,
                        eemetric_et,
                        sims_et,
                        pt_jpl_et,
                        geesebal_et,
                    ]).mean()
                band = band.rename(band_name)
                data_img = data_img.addBands(band, overwrite=True)
            data_img = data_img.multiply(irr_mask)
            retry_download = True
            while retry_download:
                try:
                    gee_url = data_img.getDownloadUrl({
                        'scale': 30,
                        'region': ee_geom,
                        'format': 'GEO_TIFF',
                        'crs': 'EPSG:4326'
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
                    retry_download = False
                except (ee.EEException, requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
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
        num_workers: int = 32
) -> tuple[str, list[str, ...]]:
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

    Returns:
        tuple (str, list (str, ...)): Tuple containing the directory path containing all the downloaded GEE tiles and
        the ordered list of band names for each tile.
    """

    data_dir = f'{download_dir}GEE_Data/GEE_Tiles/'
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
        'annual_gridmet_bc_eto_mm',
        'annual_gridmet_bc_etr_mm',
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
        'annual_conus404_precip_mm',
        'annual_conus404_tmmx_K',
        'annual_conus404_tmmn_K',
        'annual_conus404_eto_mm',
        'annual_conus404_etr_mm',
        'annual_terraclimate_sm_change_mm',
        'annual_terraclimate_ro_mm',
        'crop_cdl',
        'HSG',
        'soil_depth_mm',
        'ksat_mean_micromps',
        'elevation_m',
        'slope'
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
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit='0.5G')
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
                    verbose=False
                )
                for tile_vals in tile_chunk
            )
            itr += 1
        dask_client.close()
    return data_dir, data_band_names


def resample_gee_tiles(
        gee_tile_dir: str,
        data_band_names: list[str, ...],
        output_dir: str,
        raster_res: float = 1000,
        num_workers: int = 32,
        already_resampled: bool = False
) -> None:
    """
    Resample 30-m GEE tiles to a higher scale.

    Args:
        gee_tile_dir (str): GEE tile directory containing the 30-m multi-band rasters.
        data_band_names (str): List of data band names.
        output_dir (str): Output directory.
        raster_res (float): Raster resolution in m.
        num_workers (int): Number of dask workers to use to resample GEE tiles.
        already_resampled (bool): Set True to skip resampling.

    Returns:
        None.
    """

    if not already_resampled:
        data_band_dict = {}
        categorical_bands = [
            'annual_gridmet_spi1y',
            'annual_gridmet_eddi1y',
            'annual_gridmet_spei1y',
            'annual_gridmet_pdsi',
            'crop_cdl',
            'HSG',
        ]
        for band_num, data_band_name in enumerate(data_band_names):
            if data_band_name not in categorical_bands:
                val = (band_num + 1, 'average', 'float32')
            else:
                gdal_dtype = 'int16'
                if data_band_name in ['crop_cdl', 'HSG']:
                    gdal_dtype = 'byte'
                val = (band_num + 1, 'mode', gdal_dtype)
            data_band_dict[data_band_name] = val
        gee_tiles = sorted(glob(gee_tile_dir + '*.tif'))
        itr = 1
        gee_tiles = gee_tiles[(itr - 1) * num_workers:]
        num_chunks = int(np.ceil(len(gee_tiles) / num_workers))
        tile_chunks = generate_chunks(gee_tiles, num_workers)
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit='1.5G')
        dask_cluster.scale(num_workers)
        dask_client = Client(dask_cluster)
        dask_client.wait_for_workers(1)
        print(f'Using {num_workers} local workers...')
        resampling_factor = raster_res / 30
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
        dask_client.close()
        print('All tiles resampled...')
    else:
        print('GEE tiles already resampled...')


def create_irrigation_tiles(
        gee_tile_dir: str,
        output_dir: str,
        start_year: int = 1985,
        end_year: int = 2023,
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
        for year in range(start_year, end_year + 1):
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
    return irr_resampled_dir


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
        for year in range(start_year, end_year + 1):
            tiles = ' '.join(glob(f'{input_tile_dir}*{year}.tif'))
            merged_tif = f'{output_dir}{output_prefix}_{year}.tif'
            gdal_sys_call = f'{os.environ["CONDA_PREFIX"]}/bin/gdal_merge.py -o {merged_tif} -of GTiff -init 0 {tiles}'
            subprocess.call(
                gdal_sys_call,
                shell=True,
                stdout=subprocess.DEVNULL
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
