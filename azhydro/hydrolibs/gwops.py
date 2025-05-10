"""
Handle groundwater withdrawal processing codes.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import rasterops as rops
import vectorops as vops
import os
import shutil
import numpy as np
import geopandas as gpd
import pandas as pd
import dataretrieval.nwis as nwis
import scipy.ndimage.filters as flt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import multiprocessing
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from typing import Any
from joblib import Parallel, delayed
from sysops import makedirs, copy_file
from glob import glob
from copy import deepcopy


def reproject_vectors(
        input_dir: str,
        output_dir: str,
        ref_file: str,
        pattern: str = '*.geojson',
        already_reprojected: bool = False
) -> None:
    """
    Reproject various AZ vector files.

    Args:
        input_dir (str): Input directory containing vector files.
        output_dir (str): Output directory.
        ref_file (str): Reference vector file.
        pattern (str): Vector file path pattern.
        already_reprojected (bool): Set True to disable reprojection.

    Returns:
        tuple (str, ...): Sorted list of reprojected vector file paths.
    """

    if not already_reprojected:
        vector_files = glob(f'{input_dir}{pattern}')
        makedirs(output_dir)
        for f in vector_files:
            output_f = f"{output_dir}{f[f.rfind(os.sep) + 1:]}"
            vops.reproject_vector(
                f,
                outfile_path=output_f,
                ref_file=ref_file,
                raster=False
            )
    else:
        print('Boundary/State/AMA_INA shapefiles are already reprojected')


def crop_gw_rasters(
        input_gw_dir: str,
        output_gw_dir: str,
        az_state_file: str,
        already_cropped: bool = False
) -> str:
    """
    Crop GW rasters based on a mask, should be called after GW rasters have been created.

    Args:
        input_gw_dir (str): Input GW pumping raster directory.
        output_gw_dir (str): Output directory.
        az_state_file (str): Arizona state geojson or shapefile path.
        already_cropped (bool): Set True to disable cropping.

    Returns:
        str: Final cropped raster directory path.
    """

    cropped_dir = f'{output_gw_dir}GW_Cropped/'
    if not already_cropped:
        makedirs(cropped_dir)
        rops.crop_rasters(
            input_gw_dir, outdir=cropped_dir,
            input_mask_file=az_state_file,
            ext_mask=True
        )
    else:
        print('GW rasters already cropped')
    return cropped_dir


def preprocess_gw_csv(
        well_registry_file: str,
        input_gw_csv_dir: str,
        output_dir: str,
        fill_attr: str = 'AF Pumped',
        filter_attr: str | None = None,
        filter_attr_value: str = 'OUTSIDE OF AMA OR INA',
        use_only_ama_ina: bool = False,
        already_preprocessed: bool = False,
        **kwargs: dict[str, str]
) -> str:
    """
    Preprocess the well registry file to add GW pumping from each CSV file. That is, add an attribute present in the
    GW csv file to the Well Registry shape files (yearwise) based on matching ids given in kwargs.
    By default, the GW withdrawal is added. The csv ids must include: csv_well_id, csv_mov_id, csv_water_id,
    movement_type, water_type, The shp id must include shp_well_id. For the Arizona datasets, csv_well_id='Well Id',
    csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL', water_type='GROUNDWATER', and
    shp_well_id='REGISTRY_I' by default. For changing, pass appropriate kwargs.

    Args:
        well_registry_file (str): ADWR well registry geojson or shapefile.
        input_gw_csv_dir (str): Input GW csv directory.
        output_dir (str): Output directory.
        fill_attr (str): Attribute present in the CSV file to add to Well Registry.
        filter_attr (str or None): Remove specific wells based on this attribute. Set None to disable filtering.
        filter_attr_value (str): Value for filter_attr
        use_only_ama_ina (bool): Set True to use only AMA/INA for model training.
        already_preprocessed (bool): Set True to disable preprocessing.

    Returns:
        str: File path of the first GW pumping shapefile or geojson.
    """

    if not already_preprocessed:
        makedirs(output_dir)
        vops.add_attribute_well_reg_multiple(
            input_well_reg_file=well_registry_file,
            input_gw_csv_dir=input_gw_csv_dir, out_gw_shp_dir=output_dir,
            fill_attr=fill_attr, filter_attr=filter_attr,
            filter_attr_value=filter_attr_value,
            use_only_ama_ina=use_only_ama_ina,
            **kwargs
        )
    ref_file = glob(f'{output_dir}*.shp')[0]
    return ref_file


def fix_gw_raster_values(
        input_raster_dir: str,
        outdir: str,
        max_threshold: float = 1e+5,
        min_threshold: float = 0,
        fix_only_negative: bool = False,
        gw_pattern: str = 'GW*.tif'
) -> None:
    """
    Fix unusually large values introduced by gdal_rasterize sometimes or remove negative pumpings indicating
    no well data.

    Args:
        input_raster_dir (str): Input raster directory.
        outdir (str): Output directory.
        max_threshold (float): Max value beyond which values will be set to no data value, default unit is acrefeet.
        min_threshold (float): Min value below which values will be set to no data value, default unit is acre-feet.
        fix_only_negative (bool): Set True to fix only negative values.
        gw_pattern (str): File extension for the GW withdrawal rasters.

    Returns:
        None.
    """

    for raster_file in glob(input_raster_dir + gw_pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_file = rops.read_raster_as_arr(raster_file)
        no_data = rops.az_nodata()
        raster_arr[np.isnan(raster_arr)] = no_data
        raster_arr[np.logical_and(raster_arr > 0, raster_arr < 1e-8)] = 0.
        if not fix_only_negative:
            raster_arr[raster_arr > max_threshold] = no_data
            raster_arr[raster_arr < min_threshold] = no_data
        raster_arr[raster_arr < 0] = no_data
        rops.write_raster(
            raster_arr, raster_file, transform_=raster_file.transform,
            outfile_path=out_raster, no_data_value=no_data
        )


def create_gw_volume_rasters(
        input_gw_dir: str,
        output_gw_dir: str,
        xres: float = 1000.,
        yres: float = 1000.,
        value_field: str | None = None,
        value_field_pos: int = 0,
        already_created: bool = True,
        max_gw: float | None = None,
        min_gw: float | None = None
) -> None:
    """
    Create GW pumping volume rasters from shapefiles.

    Args:
        input_gw_dir (str): Input directory containing preprocessed GW files from #preprocess_gw_csv.
        output_gw_dir (str): Output directory.
        xres (float): X-Resolution (map unit).
        yres (float): Y-Resolution (map unit).
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing).
        already_created (bool): Set False to re-compute GW pumping volume rasters.
        max_gw (float or None): Maximum groundwater pumping depth in mm. Values beyond this will be ignored.
        min_gw (float or None): Minimum GW pumping depth in mm. Values below this will be set to no data value.

    Returns:
        str: Output raster directory path containing the GW pumping volume rasters (acreft).
    """

    if not already_created:
        print('Creating GW withdrawal volume (acreft) rasters...')
        gw_volume_dir_uncorrected = f'{output_gw_dir}Uncorrected_GW_Volumes/'
        makedirs(gw_volume_dir_uncorrected)
        vops.shps2rasters(
            input_gw_dir,
            gw_volume_dir_uncorrected,
            xres=xres, yres=yres,
            value_field=value_field,
            value_field_pos=value_field_pos
        )
        makedirs(output_gw_dir)
        if max_gw:
            max_gw *= xres * yres / 1.233e+6
        else:
            max_gw = np.inf
        if min_gw:
            min_gw *= xres * yres / 1.233e+6
        else:
            min_gw = 0
        fix_gw_raster_values(
            gw_volume_dir_uncorrected,
            output_gw_dir,
            fix_only_negative=False,
            max_threshold=max_gw,
            min_threshold=min_gw,
        )
        shutil.rmtree(gw_volume_dir_uncorrected, ignore_errors=True)
    else:
        print('GW  pumping volume rasters already created')


def create_gw_depth_rasters(
        gw_volume_dir: str,
        output_gw_dir: str,
        gw_pattern: str = '*.tif',
        already_created: bool = False
) -> None:
    """
    Create GW withdrawal depth rasters.

    Args:
        gw_volume_dir (str): Input GW volume (acreft) directory.
        output_gw_dir (str): Output raster directory containing the GW depth rasters (mm).
        gw_pattern (str): File extension for the GW withdrawal rasters.
        already_created (bool): Set False to re-compute GW pumping depth rasters.

    Returns:
        None.
    """

    if not already_created:
        makedirs(output_gw_dir)
        nodata = rops.az_nodata()
        for gw_volume_file in glob(gw_volume_dir + gw_pattern):
            gw_depth_file = f'{output_gw_dir}{gw_volume_file[gw_volume_file.rfind(os.sep) + 1:]}'
            gw_vol_arr, gw_vol_ref = rops.read_raster_as_arr(gw_volume_file)
            xres, yres = gw_vol_ref.res
            gw_depth_arr = gw_vol_arr * 1.233 / abs(xres * yres * 1e-6)
            gw_depth_arr[np.isnan(gw_depth_arr)] = nodata
            rops.write_raster(
                gw_depth_arr, gw_vol_ref, transform_=gw_vol_ref.transform,
                outfile_path=gw_depth_file, no_data_value=nodata
            )
    else:
        print('GW pumping depth rasters already created...')


def create_gw_basin_streamflow_rasters(
        gw_basin_vector: str,
        canal_vector: str,
        output_dir: str,
        xres: float = 500,
        yres: float = 500,
        start_year: int = 1985,
        end_year: int = 2023,
        water_year_agg: bool = False,
        already_created: bool = False
) -> None:
    """
    Create GW basin and Colorado river streamflow rasters for Arizona.

    Args:
        gw_basin_vector (str): GW Basin shapefile or geojson for Arizona.
        canal_vector (str): Canal shapefile or geojson for Arizona.
        output_dir (str): Output directory.
        xres (float): X-Resolution (m).
        yres (float): Y-Resolution (m).
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        water_year_agg (bool): Set True to aggregate by water year.
        already_created (bool): Set True if raster already exists.

    Returns:
        None
    """

    if not already_created:
        year_list = range(start_year + 1, end_year + 1)
        makedirs(output_dir)
        az_gw_basin_sy_tif = f'{output_dir}GW_Basin_{start_year}.tif'
        vops.shp2raster(
            gw_basin_vector,
            az_gw_basin_sy_tif,
            xres=xres, yres=yres,
            value_field='OBJECTID',
            add_value=False
        )
        for year in year_list:
            az_gw_basin_tif = f'{output_dir}GW_Basin_{year}.tif'
            copy_file(
                az_gw_basin_sy_tif,
                az_gw_basin_tif
            )
        canal_dir = f'{output_dir}Canal/'
        makedirs(canal_dir)
        canal_gdf_az = gpd.read_file(canal_vector)
        co_attr = 'ColoRiver'
        co_river_flt = canal_gdf_az[co_attr] == 1
        canal_gdf_az.loc[~co_river_flt, co_attr] = 0
        canal_az_shp = f'{canal_dir}Canal_AZ.shp'
        canal_gdf_az.to_file(canal_az_shp)
        az_canal_tif = f'{canal_dir}Canal_AZ.tif'
        vops.shp2raster(
            canal_az_shp, az_canal_tif,
            xres=xres, yres=yres,
            value_field=co_attr,
            add_value=False
        )
        year_list = range(start_year, end_year + 1)
        canal_raster_arr, canal_raster_file = rops.read_raster_as_arr(az_canal_tif)
        flow_attr = 'dv_ft3_sec'
        dv_start_year = f'{start_year}-01-01'
        dv_end_year = f'{end_year}-12-31'
        if water_year_agg:
            dv_start_year = f'{start_year - 1}-10-01'
            dv_end_year = f'{end_year}-09-30'
        usgs_site_daily = nwis.get_record(
            sites='09427520', # COLORADO RIVER BELOW PARKER DAM, AZ-CA
            service='dv', # daily discharge values in ft3/s
            start=dv_start_year, # water year
            end=dv_end_year,
            parameterCd='00060'
        ).rename(columns={'00060_Mean': flow_attr}).reset_index()
        usgs_site_daily.datetime = pd.to_datetime(usgs_site_daily.datetime)
        usgs_site_monthly = usgs_site_daily.groupby(
            pd.Grouper(key='datetime', freq='ME')
        ).agg({flow_attr: 'mean'}).reset_index()
        if water_year_agg:
            usgs_site_monthly['Year'] = usgs_site_monthly.datetime.dt.year.where(
                usgs_site_monthly.datetime.dt.month < 10,
                usgs_site_monthly.datetime.dt.year + 1
            )
        else:
            usgs_site_monthly['Year'] = usgs_site_monthly.datetime.dt.year
        usgs_site_annual = usgs_site_monthly.groupby('Year').agg({flow_attr: 'mean'}).reset_index()
        usgs_site_annual[flow_attr] *= 0.0283168
        flow_attr = 'dv_m3_sec'
        usgs_site_annual.columns = ['Year', flow_attr]
        for year in year_list:
            streamflow_tif = f'{output_dir}Streamflow_{year}.tif'
            canal_arr = deepcopy(canal_raster_arr)
            canal_arr[canal_arr == 1] *= usgs_site_annual[usgs_site_annual.Year == year][flow_attr].values[0]
            canal_arr[np.isnan(canal_arr)] = 0
            rops.write_raster(
                canal_arr,
                canal_raster_file,
                transform_=canal_raster_file.transform,
                outfile_path=streamflow_tif,
                no_data_value=0
            )

    print('GW Basin and Streamflow rasters created...')


def create_annual_eff_precip_rasters(
        monthly_eff_precip_dir: str,
        output_dir: str,
        start_year: int = 1985,
        end_year: int = 2023,
        start_month: int = 1,
        end_month: int = 12,
        already_created: bool = False
) -> None:
    """
    Create annual effective precipitation rasters from monthly effective precipitation rasters.

    Args:
        monthly_eff_precip_dir (str): Monthly effective precipitation raster directory.
        output_dir (str): Output directory.
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        start_month (int): Start month in MM.
        end_month (int): End month in MM.
        already_created (bool): Set True if raster already exists.

    Returns:
        None
    """

    if not already_created:
        year_list = range(start_year, end_year + 1)
        makedirs(output_dir)
        # copy months 1 to 9 from second year to first year
        for month in range(1, 10):
            annual_eff_precip_file = f'{monthly_eff_precip_dir}effective_precip_{start_year}_{month}.tif'
            if not os.path.exists(annual_eff_precip_file):
                monthly_eff_precip_file = f'{monthly_eff_precip_dir}effective_precip_{start_year + 1}_{month}.tif'
                if os.path.exists(monthly_eff_precip_file):
                    copy_file(
                        monthly_eff_precip_file,
                        annual_eff_precip_file
                    )
        monthly_eff_precip_rio = None
        for year in year_list:
            annual_eff_precip_arr = None
            for month in range(start_month, end_month + 1):
                monthly_eff_precip_file = f'{monthly_eff_precip_dir}effective_precip_{year}_{month}.tif'
                if os.path.exists(monthly_eff_precip_file):
                    monthly_eff_precip_arr, monthly_eff_precip_rio = rops.read_raster_as_arr(monthly_eff_precip_file)
                    if annual_eff_precip_arr is None:
                        annual_eff_precip_arr = monthly_eff_precip_arr
                    else:
                        annual_eff_precip_arr += monthly_eff_precip_arr
            if annual_eff_precip_arr is not None:
                annual_eff_precip_file = f'{output_dir}Peff_{year}.tif'
                rops.write_raster(
                    annual_eff_precip_arr,
                    monthly_eff_precip_rio,
                    transform_=monthly_eff_precip_rio.transform,
                    outfile_path=annual_eff_precip_file,
                    no_data_value=monthly_eff_precip_rio.nodata
                )
    print('Annual effective precipitation rasters created...')



def create_gw_basin_sw_delivery_rasters(
        gw_basin_vector: str,
        cap_delivery_xls: str,
        srp_delivery_xls: str,
        output_dir: str,
        xres: float = 500,
        yres: float = 500,
        start_year: int = 1985,
        end_year: int = 2023,
        already_created: bool = False
) -> None:
    """
    Create CAP and SRP surface water delivery data for Arizona.

    Args:
        gw_basin_vector (str): GW Basin shapefile or geojson for Arizona.
        cap_delivery_xls (str): XLS filepath for the CAP annual surface water delivery data in acre-feet.
        srp_delivery_xls (str): XLS filepath for the SRP annual surface water delivery data in acre-feet.
        output_dir (str): Output directory.
        xres (float): X-Resolution (m).
        yres (float): Y-Resolution (m).
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        already_created (bool): Set True if raster already exists.

    Returns:
        None
    """

    if not already_created:
        year_list = range(start_year, end_year + 1)
        makedirs(output_dir)
        gw_basin_gdf = gpd.read_file(gw_basin_vector)
        cap_df = pd.read_excel(cap_delivery_xls)[['Year', 'AMA', 'Delivery AF']]
        cap_df = cap_df[cap_df.Year.isin(year_list)]
        year_col = 'Year'
        basin_col = 'BASIN_NAME'
        delivery_col = 'AF'
        cap_df.columns = [year_col, basin_col, delivery_col]
        cap_df[basin_col] = cap_df[basin_col].str.upper()
        srp_df = pd.read_excel(srp_delivery_xls)[['Water Move Year', 'AMA', 'SUM_WATER_QTY', 'Water Type', 'Water Use']]
        srp_df.columns = [year_col, basin_col, delivery_col, 'WT', 'WU']
        srp_df = srp_df[
            (srp_df.Year.isin(year_list)) &
            (srp_df.WT.str.contains('SURFACE WATER')) &
            (srp_df.WU.str.contains('EXEMPT IRRIGATION DELIVERY'))
        ].drop(columns=['WT', 'WU']).groupby([year_col, basin_col]).sum().reset_index()
        srp_missing_data = pd.DataFrame({
            year_col: [2023],
            basin_col: ['Phoenix AMA'],
            delivery_col: [
                srp_df[srp_df.Year == 2022][delivery_col].values[0]
            ]
        })
        srp_df = pd.concat([srp_df, srp_missing_data])
        srp_df[basin_col] = srp_df[basin_col].str.upper()
        for year in year_list:
            select_rows = (cap_df[basin_col] == 'PHOENIX AMA') & (cap_df[year_col] == year)
            cap_srp_delivery = cap_df[select_rows][delivery_col] + srp_df[srp_df[year_col] == year][delivery_col]
            cap_df.loc[select_rows, delivery_col] = cap_srp_delivery.copy()
        gw_basin_cap_srp_gdf = gw_basin_gdf.merge(cap_df, on=basin_col)[[year_col, basin_col, delivery_col, 'geometry']]
        for year in year_list:
            basin_gdf = gw_basin_gdf.copy(deep=True)[[basin_col, 'geometry']]
            gw_basin_shp = f'{output_dir}GW_Basin_CAP_SRP_Total_{year}.shp'
            for gw_basin in basin_gdf[basin_col]:
                delivery_vals = gw_basin_cap_srp_gdf[
                    (gw_basin_cap_srp_gdf[basin_col] == gw_basin) &
                    (gw_basin_cap_srp_gdf[year_col] == year) &
                    (gw_basin_cap_srp_gdf[delivery_col] > 0)
                ][delivery_col].values
                if not delivery_vals.size:
                    delivery_data = 0
                else:
                    delivery_data = delivery_vals[0]
                basin_gdf.loc[basin_gdf[basin_col] == gw_basin, delivery_col] = delivery_data
            basin_gdf.to_file(gw_basin_shp)
            gw_basin_tif = f'{output_dir}GW_Basin_CAP_SRP_Total_{year}.tif'
            vops.shp2raster(
                gw_basin_shp,
                gw_basin_tif,
                xres=xres, yres=yres,
                value_field=delivery_col,
                add_value=False
            )
    print('GW Basin surface water delivery rasters created...')


def create_land_use_data(
        input_df: pd.DataFrame,
        cdl_arr: np.array,
        smoothing: int = 3
) -> pd.DataFrame:
    """
    Create Gaussian-filtered land use array.

    Args:
        input_df (pd.DataFrame): Dataframe used to store the Guassian-filtered reclassified CDL arrays, where
                                 1 = Agriculture, 2 = Surface Water, and 3 = Urban.
        cdl_arr (np.array): CDL array.
        smoothing (int): Smoothing window size for the Gaussian filter.

    Returns:
        pd.DataFrame: input_df updated with the Gaussian-filtered reclassified CDL arrays.
    """

    cdl_reclass_dict = {
        (0, 59.5): 1,
        (66.5, 77.5): 1,
        (203.5, 255): 1,
        (110.5, 111.5): 2,
        (111.5, 112.5): 0,
        (120.5, 124.5): 3,
        (59.5, 61.5): 0,
        (130.5, 195.5): 0
    }
    cdl_labels = ('AGRI', 'SW', 'URBAN')
    for key in cdl_reclass_dict.keys():
        cdl_arr[np.logical_and(cdl_arr > key[0], cdl_arr <= key[1])] = cdl_reclass_dict[key]
    cdl_arr = cdl_arr.astype(np.float32)
    for idx, cdl_label in enumerate(cdl_labels):
        lu_arr = np.full_like(cdl_arr, fill_value=0.)
        lu_arr[cdl_arr == idx + 1] = 1
        gaussian_lu_arr = flt.gaussian_filter(lu_arr, sigma=smoothing, order=0)
        gaussian_lu_arr = np.abs(gaussian_lu_arr)
        gaussian_lu_arr -= np.min(gaussian_lu_arr)
        gaussian_lu_arr /= np.ptp(gaussian_lu_arr)
        input_df[cdl_label] = gaussian_lu_arr.ravel()
    return input_df


def get_ama_ina_basin_names() -> list[str]:
    """
    Get the names of AMA and INA basins.

    Returns:
        list: List of AMA and INA basin names.
    """

    ama_ina_basins = [
        'SANTA CRUZ AMA',
        'PRESCOTT AMA',
        'TUCSON AMA',
        'PINAL AMA',
        'PHOENIX AMA',
        'DOUGLAS AMA_INA',
        'JOSEPH CITY INA',
        'HARQUAHALA INA',
        'HUALAPAI VALLEY'
    ]
    return ama_ina_basins


def parallel_make_time_series_plots(
        idx: int,
        gw_basin: str,
        input_df: pd.DataFrame,
        output_dir: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str,
        actual_gw_col: str,
        gw_basin_col: str,
        split_strategy: int,
        test_gw_basins: tuple[str],
        raster_res: float = 2000
) -> None:
    """
    Create time series plots for individual groundwater basins.

    Args:
        idx (int): Index of the groundwater basin.
        gw_basin (str): Name of the groundwater basin.
        input_df (pd.DataFrame): Input dataframe containing the ML-predicted and actual pumping data.
        output_dir (str): Output directory.
        test_year_limits (tuple[tuple[int, int], ...]): Tuple of tuples containing the start and end years for testing.
        year_col (str): Name of the year column.
        actual_gw_col (str): Name of the actual groundwater pumping column.
        gw_basin_col (str): Name of the groundwater basin column.
        split_strategy (int): Split strategy used for the model. 1 = Temporal, 2 = Random Stratified, 3 = Spatial.
        test_gw_basins (tuple (str)): Test GW basin names.
        raster_res (float): Raster resolution in meters.

    Returns:
        None.
    """

    ama_ina_basins = get_ama_ina_basin_names()
    if idx == 0:
        gw_basin_name = 'AMA_INA'
        basin_df = input_df[input_df[gw_basin_col].isin(gw_basin)].copy()
        gw_basin_type = 'AMA/INA'
    else:
        gw_basin_name = gw_basin
        basin_df = input_df[input_df[gw_basin_col] == gw_basin].copy()
        gw_basin_type = 'AMA/INA' if gw_basin in ama_ina_basins else 'Other'
    plot_dir = f'{output_dir}{gw_basin_name}/'
    makedirs(plot_dir)
    area = raster_res ** 2
    # m2 to acre-ft and mm to ft and then 1000s of acre-ft
    basin_df['Actual_GW_af'] = basin_df['Actual_GW_mm'] * area / (4047 * 304.8 * 1000)
    basin_df['Pred_GW_af'] = basin_df['Pred_GW_mm'] * area / (4047 * 304.8 * 1000)
    # mm to m and then 1e6 m3
    basin_df['Actual_GW_m3'] = basin_df['Actual_GW_mm'] * area * 1e-9
    basin_df['Pred_GW_m3'] = basin_df['Pred_GW_mm'] * area * 1e-9

    basin_df['Actual_GW_ft'] = basin_df['Actual_GW_mm'] / 304.8
    basin_df['Pred_GW_ft'] = basin_df['Pred_GW_mm'] / 304.8
    min_yr = basin_df[year_col].min()
    max_yr = basin_df[year_col].max()
    if gw_basin_type != 'Other':
        replace_df = basin_df[basin_df[actual_gw_col] > 0]
        if replace_df.shape[0] > 0:
            basin_df = replace_df
        else:
            gw_basin_type = 'Other'
    for estimator in ['mean', 'sum']:
        for unit in ['af', 'm3', 'ft', 'mm']:
            plt.figure(figsize=(20, 10))
            plt.rcParams.update({'font.size': 20})
            ax = sns.lineplot(
                basin_df,
                x=year_col,
                y=f'Pred_GW_{unit}',
                estimator=estimator,
                errorbar='ci',
                color='black',
                marker='o',
                err_style='bars'
            )
            if gw_basin_type != 'Other':
                sns.lineplot(
                    basin_df,
                    x=year_col,
                    y=f'Actual_GW_{unit}',
                    estimator=estimator,
                    errorbar='ci',
                    color='orange',
                    marker='o',
                    err_style='bars',
                    ax=ax
                )
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))  # Format ticks as integers
            plt.legend(
                ['Predicted', 'Metered', '95% CI', '95% CI'] if gw_basin_type != 'Other'
                else ['Predicted', '95% CI'], ncol=2
            )
            if gw_basin_type == 'Other':
                ax.axvspan(min_yr, max_yr, color='lightblue', alpha=0.3)
            elif split_strategy < 3:
                for test_year in test_year_limits:
                    ax.axvspan(test_year[0], test_year[1], color='lightblue', alpha=0.3)
            elif split_strategy == 3 and gw_basin in test_gw_basins:
                ax.axvspan(min_yr, max_yr, color='lightblue', alpha=0.3)

            unit_str = f'(1000s of acre-ft)' if unit == 'af' else f'(1e6 m$^3$)' if unit == 'm3' else f'({unit})'
            ylabel_prefix = 'Total' if estimator == 'sum' else 'Mean'
            plt.ylabel(f'{ylabel_prefix} Agricultural Groundwater Withdrawals {unit_str}')
            plt.tight_layout()
            plt.xticks(range(min_yr, max_yr + 1, 3))
            plt.savefig(f'{plot_dir}TS_{ylabel_prefix}_{unit}.png', dpi=300)
            plt.close()


def make_time_series_plots(
        input_df: pd.DataFrame,
        model: Any,
        features: list[str],
        output_dir: str,
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        test_year_limits: tuple[tuple[int, int], ...] = ((2000, 2010),),
        pred_attr: str = 'gw_pumping_mm',
        split_strategy: int = 1,
        test_gw_basins: tuple[str, ...] = ('HARQUAHALA INA',),
        raster_res: float = 2000
) -> None:
    """
    Make time series plots for individual groundwater basins.

    Args:
        input_df (pd.DataFrame): Input dataframe containing the ML-predicted and actual pumping data.
        model (Any): ML model used for predictions.
        features (list[str]): List of features used in the model.
        output_dir (str): Output directory.
        year_col (str): Name of the year column.
        gw_basin_col (str): Name of the GW basin column.
        test_year_limits (tuple[tuple[int, int], ...]): Tuple of tuples containing the start and end years for testing.
        pred_attr (str): Name of the actual GW pumping attribute.
        split_strategy (int): Split strategy used for the model. 1 = Temporal, 2 = Random Stratified, 3 = Spatial,
                            4 = Random.
        test_gw_basins (tuple[str]): Tuple of GW basins to be used for testing. Only used if split_strategy is 3.
        raster_res (float): Resolution of the raster in meters.

    Returns:
        None.
    """

    print('Creating time series plots...')
    gw_basins = [get_ama_ina_basin_names()] + input_df[gw_basin_col].unique().tolist()
    actual_gw_col = 'Actual_GW_mm'
    pred_gw_col = 'Pred_GW_mm'
    input_df = input_df.rename(columns={pred_attr: actual_gw_col})
    input_df[pred_gw_col] = np.abs(model.predict(input_df[features]))
    num_cores = multiprocessing.cpu_count() - 1
    Parallel(n_jobs=num_cores - 1)(delayed(parallel_make_time_series_plots)(
        idx,
        gw_basin,
        input_df,
        output_dir,
        test_year_limits,
        year_col,
        actual_gw_col,
        gw_basin_col,
        split_strategy,
        test_gw_basins,
        raster_res
    ) for idx, gw_basin in enumerate(gw_basins))
