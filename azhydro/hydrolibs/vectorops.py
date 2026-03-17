"""
Handle various vector operations
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import multiprocessing
import os
import warnings

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal

gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
from glob import glob

from joblib import Parallel, delayed

from hydrolibs.sysops import az_nodata

logger = logging.getLogger(__name__)
from shapely.geometry import Point

# Sentinel values for GDAL additive rasterization (initValues=0).
# After rasterize, exact-zero pixels have no wells and are set to NaN.
# These tiny values keep well-occupied pixels distinguishable from zero.
UNMETERED_WELL_SENTINEL = -1e-16   # well exists but no meter data (outside AMA/INA)
ZERO_PUMPING_SENTINEL = 1e-10      # metered well reporting zero withdrawal


def reproject_vector(
        input_vector_file: str,
        outfile_path: str,
        ref_file: str | None = None,
        crs: str = 'epsg:4326',
        crs_from_file: bool = True,
        raster: bool = True,
        verbose: bool = True
) -> None:
    """
    Reproject a vector file.

    Args:
        input_vector_file (str): Input vector file path.
        outfile_path (str): Output vector file path.
        crs (str): Target CRS.
        ref_file (str or None): Reference file (raster or vector) for obtaining target CRS.
        crs_from_file (bool): If true (default) read CRS from file (raster or vector).
        raster (bool): If true (default) read CRS from raster else vector.
        verbose (bool): Set True to print messages.

    Returns:
        None.
    """

    if verbose:
        logger.info(f'Reprojecting {input_vector_file}')
    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    output_vector_file = input_vector_file.to_crs(crs)
    drop_cols = [c for c in output_vector_file.columns
                 if c.startswith('Shape__') or c.startswith('Shape_')]
    if drop_cols:
        output_vector_file = output_vector_file.drop(columns=drop_cols)
    for col in output_vector_file.select_dtypes(include=['datetime', 'datetimetz']).columns:
        output_vector_file[col] = output_vector_file[col].astype(str)
    output_vector_file.to_file(outfile_path)


def csv2shp(
        input_csv_file: str,
        outfile_path: str,
        delim: str = ',',
        source_crs: str = 'epsg:4326',
        target_crs: str = 'epsg:4326',
        long_lat_pos: tuple[int, int] = (7, 8)
) -> None:
    """
    Convert CSV to Shapefile.

    Args:
        input_csv_file (str): Input CSV file path.
        outfile_path (str): Output file path.
        delim (str): CSV file delimiter.
        source_crs (str): CRS of the source file.
        target_crs (str): Target CRS.
        long_lat_pos (tuple (int, int)): Tuple containing positions of longitude and latitude columns,
                                         respectively (zero indexing).

    Returns:
        None.
    """

    input_df = pd.read_csv(input_csv_file, delimiter=delim)
    input_df = input_df.dropna(axis=1)
    long, lat = input_df.columns[long_lat_pos[0]], input_df.columns[long_lat_pos[1]]
    geometry = [Point(xy) for xy in zip(input_df[long], input_df[lat])]
    gdf2shp(input_df, geometry, source_crs, target_crs, outfile_path)


def csvs2shps(
        input_dir: str,
        output_dir: str,
        pattern: str = '*.csv',
        target_crs: str = 'EPSG:4326',
        delim: str = ',',
        long_lat_pos: tuple[int, int] = (7, 8)
) -> None:
    """
    Convert all CSV files present in a folder to corresponding Shapefiles.

    Args:
        input_dir (str): Input directory containing csv files which are named as <Layer_Name>_<Year>.[csv|txt].
        output_dir (str): Output directory.
        pattern (str): CSV  file pattern.
        target_crs (str): Target CRS.
        delim (str): CSV file delimiter.
        long_lat_pos (tuple (int, int)): Tuple containing positions of longitude and latitude columns,
                                         respectively (zero indexing).

    Returns:
        None.
    """

    for file in glob(os.path.join(input_dir, pattern)):
        basename = file[file.rfind(os.sep) + 1: file.rfind('.')]
        outfile_path = os.path.join(output_dir, f'{basename}.shp')
        csv2shp(
            file,
            outfile_path=outfile_path,
            delim=delim,
            target_crs=target_crs,
            long_lat_pos=long_lat_pos
        )


def add_attribute_well_reg(
        input_well_reg_file: str,
        input_gw_csv_file: str,
        out_gw_shp_file: str,
        fill_attr: str = 'AF Pumped',
        filter_attr: str = 'AMA INA',
        filter_attr_value: str = 'NOT WITHIN ANY AMA OR INA',
        use_only_ama_ina: bool = False,
        af_max_threshold: float = 5000.,
        **kwargs: dict[str, str]
) -> None:
    """
    Add an attribute present in the GW csv file to the Well Registry shape file based on matching ids given in kwargs.
    By default, the GW withdrawal is added. The csv ids must include: csv_well_id, csv_mov_id, csv_water_id,
    movement_type, water_type, The shp id must include shp_well_id. For the Arizona datasets, csv_well_id='Well Id',
    csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL', water_type='GROUNDWATER', and
    shp_well_id='REGISTRY_I' by default. For changing, pass appropriate kwargs.

    Args:
        input_well_reg_file (str): Input Well Registry shapefile or geojson path.
        input_gw_csv_file (str): Input GW csv file.
        out_gw_shp_file (str): Output GWSI shapefile having GW withdrawal data.
        fill_attr (str): Attribute present in the CSV file to add to Well Registry
        filter_attr (str): Remove specific wells based on this attribute. Set None to disable filtering.
        filter_attr_value (str): Value for filter_attr
        use_only_ama_ina (bool): Set True to use only AMA/INA for model training
        af_max_threshold (float): Maximum per-well AF Pumped value; rows exceeding this are dropped.
        kwargs (dict (str, str)): Additional variables, which include csv_well_id='Well Id',
                                  csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL',
                                  water_type='GROUNDWATER', csv_right_type='Right Type', and shp_well_id='REGISTRY_I'
                                  as defaults. if water_use is set to 'All', then all water uses will be considered. 
                                  Default is 'IRRIGATION'.

    Returns:
        None.
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    well_reg_gdf = gpd.read_file(input_well_reg_file)
    gw_df = pd.read_csv(input_gw_csv_file)
    csv_well_id = 'Well Id'
    shp_water_use_id = 'WATER_USE'
    csv_mov_id = 'Movement Type'
    csv_water_id = 'Water Type'
    csv_right_type = 'Right Type'
    movement_type = 'WITHDRAWAL'
    water_type = 'GROUNDWATER'
    shp_well_id = 'REGISTRY_I'
    right_type = 'IRRIGATION'
    water_use = 'IRRIGATION'
    well_reg_gdf[shp_well_id] = well_reg_gdf[shp_well_id].astype(str).apply(
        lambda x: x.zfill(6) if len(x) == 5 else x
    )
    gw_df[csv_well_id] = gw_df[csv_well_id].astype(str).apply(
        lambda x: x.zfill(6) if len(x) == 5 else x
    )
    af_values = pd.to_numeric(gw_df[fill_attr], errors='coerce')
    n_dropped = (af_values > af_max_threshold).sum()
    if n_dropped:
        logger.info(f'{input_gw_csv_file}: Dropping {n_dropped} row(s) with {fill_attr} > {af_max_threshold:,.0f}')
        gw_df = gw_df[af_values <= af_max_threshold]
    if kwargs:
        csv_well_id = kwargs.get('csv_well_id', csv_well_id)
        movement_type = kwargs.get('movement_type', movement_type)
        water_type = kwargs.get('water_type', water_type)
        shp_well_id = kwargs.get('shp_well_id', shp_well_id)
        csv_right_type = kwargs.get('csv_right_type', csv_right_type)
        water_use = kwargs.get('water_use', water_use)
        water_use = water_use if water_use in ['IRRIGATION', 'All'] else 'IRRIGATION'
    if water_use == 'IRRIGATION':
        well_reg_gdf = well_reg_gdf[well_reg_gdf[shp_water_use_id] == water_use]
    if filter_attr:
        well_reg_gdf = well_reg_gdf[well_reg_gdf[filter_attr] != filter_attr_value]
    well_id_selected = []
    for well_id in gw_df[csv_well_id].unique():
        selection_cond = (gw_df[csv_well_id] == well_id) & \
                         (gw_df[csv_mov_id] == movement_type) & \
                         (gw_df[csv_water_id] == water_type)
        if water_use == 'IRRIGATION':
            selection_cond = selection_cond & (gw_df[csv_right_type].str.startswith(right_type))
        sub_gw_df = gw_df[selection_cond]
        if not sub_gw_df.empty:
            well_id_selected.append(well_id)
            fill_value = list(sub_gw_df[fill_attr])
            if len(fill_value) > 1:
                fill_value = np.sum(fill_value)
            else:
                fill_value = fill_value[0]
            well_reg_gdf.loc[well_reg_gdf[shp_well_id] == well_id, fill_attr] = fill_value
    if not use_only_ama_ina:
        well_reg_schema = fiona.open(input_well_reg_file).schema
        well_reg_schema['properties'][fill_attr] = 'float:24.20'
        well_reg_gdf.loc[well_reg_gdf['AMA'] == filter_attr_value, fill_attr] = UNMETERED_WELL_SENTINEL
        well_reg_gdf.loc[well_reg_gdf[fill_attr] == 0., fill_attr] = ZERO_PUMPING_SENTINEL
        well_reg_gdf.to_file(out_gw_shp_file, schema=well_reg_schema, engine='fiona')
    else:
        well_reg_gdf.to_file(out_gw_shp_file)
    matched_wells = len(set(well_id_selected).intersection(
        set(well_reg_gdf[shp_well_id].tolist())
    ))
    logger.info(f'{input_gw_csv_file}: Matched wells: {matched_wells}')


def gdf2shp(
        input_df: pd.DataFrame,
        geometry: list[Point],
        source_crs: str,
        target_crs: str,
        outfile_path: str
) -> None:
    """
    Convert pandas dataframe containing location information to SHP.

    Args:
        input_df (pd.DataFrame): Input pandas dataframe.
        geometry (list (Point)): Geometry (Point) list.
        source_crs (str): CRS of the source file.
        target_crs (str): Target CRS.
        outfile_path (str): Output file path.

    Returns:
        None.
    """

    gdf = gpd.GeoDataFrame(input_df, crs=source_crs, geometry=geometry)
    gdf.to_file(outfile_path)
    if target_crs != source_crs:
        reproject_vector(
            outfile_path,
            outfile_path=outfile_path,
            crs=target_crs,
            crs_from_file=False,
            ref_file=None
        )


def shp2raster(
        input_shp_file: str,
        outfile_path: str,
        value_field: str | None = None,
        value_field_pos: int = 0,
        xres: float = 1000.,
        yres: float = 1000.,
        add_value: bool = True,
        burn_value: float | None = None
) -> None:
    """Convert Shapefile to Raster TIFF file using GDAL rasterize.

    Args:
        input_shp_file (str): Input shapefile path.
        outfile_path (str): Output TIFF file path.
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing). Only used if value_field is None.
        xres (float): Pixel width in geographic units.
        yres (float): Pixel height in geographic units.
        add_value (bool): Set False to disable adding value to existing raster cell.
        burn_value (float or None): Set burn value. If not None, then add_value, value_field, and value_field_pos
                                    arguments are ignored.

    Returns:
        None
    """
    ext_pos = input_shp_file.rfind('.')
    sep_pos = input_shp_file.rfind(os.sep)
    if sep_pos == -1:
        sep_pos = input_shp_file.rfind('/')
    layer_name = input_shp_file[sep_pos + 1: ext_pos]
    shp_file = gpd.read_file(input_shp_file)
    output_crs = shp_file.crs
    if value_field is None:
        value_field = shp_file.columns[value_field_pos]
    minx, miny, maxx, maxy = shp_file.geometry.total_bounds
    no_data_value = az_nodata()
    if burn_value is None:
        rasterize_options = gdal.RasterizeOptions(
            format='GTiff', outputType=gdal.GDT_Float32,
            outputSRS=output_crs,
            outputBounds=[minx, miny, maxx, maxy],
            xRes=xres, yRes=yres, noData=no_data_value,
            initValues=0., layers=[layer_name],
            add=add_value, attribute=value_field
        )
    else:
        rasterize_options = gdal.RasterizeOptions(
            format='GTiff', outputType=gdal.GDT_Float32,
            outputSRS=output_crs,
            outputBounds=[minx, miny, maxx, maxy],
            xRes=xres, yRes=yres, noData=no_data_value,
            initValues=0., layers=[layer_name],
            burnValues=[burn_value], allTouched=True
        )
    gdal.UseExceptions()
    gdal.Rasterize(
        outfile_path,
        input_shp_file,
        options=rasterize_options
    )


def shps2rasters(
        input_dir: str,
        output_dir: str,
        burn_value: float | None = None,
        value_field: str | None = None,
        value_field_pos: int = 0,
        xres: float = 1000,
        yres: float = 1000,
        add_value: bool = True,
) -> None:
    """
    Convert all shapefiles to corresponding TIFF files.

    Args:
    input_dir (str): Input directory containing Shapefiles which are named as <Layer_Name>_<Year>.shp.
    output_dir (str): Output directory.
    burn_value (float or None): Set burn value. If not None, then add_value, value_field, and value_field_pos
                                arguments are ignored.
    value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
    value_field_pos (int): Value field position (zero indexing). Only used if value_field is None.
    xres (float): Pixel width in geographic units.
    yres (float): Pixel height in geographic units.
    add_value (bool): Set False to disable adding value to existing raster cell.

    Returns:
        None.
    """

    def _convert(shp_file):
        outfile_path = os.path.join(output_dir, os.path.splitext(os.path.basename(shp_file))[0] + '.tif')
        shp2raster(
            shp_file,
            outfile_path=outfile_path,
            value_field=value_field,
            value_field_pos=value_field_pos,
            xres=xres,
            yres=yres,
            burn_value=burn_value,
            add_value=add_value
        )

    num_cores = multiprocessing.cpu_count() - 2
    Parallel(n_jobs=num_cores)(delayed(_convert)(f) for f in glob(os.path.join(input_dir, '*.shp')))
