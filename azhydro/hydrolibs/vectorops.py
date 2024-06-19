"""
Handle various vector operations
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import fiona
import os
import multiprocessing

from osgeo import gdal
from joblib import Parallel, delayed
from glob import glob
from sysops import az_nodata
from shapely.geometry import Point


def reproject_vector(
        input_vector_file: str,
        outfile_path: str,
        ref_file: str,
        crs: str = 'epsg:4326',
        crs_from_file: bool = True,
        raster: bool = True
) -> None:
    """
    Reproject a vector file.

    Args:
        input_vector_file (str): Input vector file path.
        outfile_path (str): Output vector file path.
        crs (str): Target CRS.
        ref_file (str): Reference file (raster or vector) for obtaining target CRS.
        crs_from_file (bool): If true (default) read CRS from file (raster or vector).
        raster (bool): If true (default) read CRS from raster else vector.

    Returns:
        None.
    """

    print('Reprojecting', input_vector_file)
    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    else:
        crs = {'init': crs}
    output_vector_file = input_vector_file.to_crs(crs)
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

    for file in glob(input_dir + pattern):
        outfile_path = output_dir + '{}.shp'.format(
            file[file.rfind(os.sep) + 1: file.rfind('.')]
        )
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
        filter_attr: str = 'AMA',
        filter_attr_value: str = 'OUTSIDE OF AMA OR INA',
        use_only_ama_ina: bool = False,
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
        kwargs (dict (str, str)): Additional variables, which include csv_well_id='Well Id',
                                  csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL',
                                  water_type='GROUNDWATER', and shp_well_id='REGISTRY_I' as defaults.

    Returns:
        None.
    """

    well_reg_gdf = gpd.read_file(input_well_reg_file)
    gw_df = pd.read_csv(input_gw_csv_file)
    csv_well_id = 'Well Id'
    csv_mov_id = 'Movement Type'
    csv_water_id = 'Water Type'
    movement_type = 'WITHDRAWAL'
    water_type = 'GROUNDWATER'
    shp_well_id = 'REGISTRY_I'
    if kwargs:
        csv_well_id = kwargs['csv_well_id']
        movement_type = kwargs['movement_type']
        water_type = kwargs['water_type']
        shp_well_id = kwargs['shp_well_id']
    if filter_attr:
        well_reg_gdf = well_reg_gdf[well_reg_gdf[filter_attr] != filter_attr_value]
    for csv_id in set(gw_df[csv_well_id]):
        sub_gw_df = gw_df[(gw_df[csv_well_id] == csv_id) & (gw_df[csv_mov_id] == movement_type) &
                          (gw_df[csv_water_id] == water_type)]
        if not sub_gw_df.empty:
            fill_value = list(sub_gw_df[fill_attr])
            if len(fill_value) > 1:
                fill_value = np.sum(fill_value)
            else:
                fill_value = fill_value[0]
            csv_id_modified = str(csv_id)
            if len(csv_id_modified) == 5:
                csv_id_modified = '0' + csv_id_modified
            well_reg_gdf.loc[well_reg_gdf[shp_well_id] == csv_id_modified, fill_attr] = fill_value
    if not use_only_ama_ina:
        well_reg_schema = fiona.open(input_well_reg_file).schema
        well_reg_schema['properties'][fill_attr] = 'float:24.20'
        well_reg_gdf.loc[well_reg_gdf['AMA'] == filter_attr_value, fill_attr] = -1e-16
        well_reg_gdf.loc[well_reg_gdf[fill_attr] == 0., fill_attr] = 1e-10
        well_reg_gdf.to_file(out_gw_shp_file, schema=well_reg_schema)
    else:
        well_reg_gdf.to_file(out_gw_shp_file)
    print(input_gw_csv_file, ': Matched wells:', well_reg_gdf.count()[shp_well_id])


def add_attribute_well_reg_multiple(
        input_well_reg_file,
        input_gw_csv_dir,
        out_gw_shp_dir,
        fill_attr='AF Pumped',
        filter_attr='AMA',
        filter_attr_value='OUTSIDE OF AMA OR INA',
        use_only_ama_ina=False,
        **kwargs
) -> None:
    """
    Parallilzation based on multiple groundwater pumping CSV files.
    Add an attribute present in the GW csv file to the Well Registry shape file based on matching ids given in kwargs.
    By default, the GW withdrawal is added. The csv ids must include: csv_well_id, csv_mov_id, csv_water_id,
    movement_type, water_type, The shp id must include shp_well_id. For the Arizona datasets, csv_well_id='Well Id',
    csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL', water_type='GROUNDWATER', and
    shp_well_id='REGISTRY_I' by default. For changing, pass appropriate kwargs.

    Args:
        input_well_reg_file (str): Input Well Registry shapefile or geojson path.
        input_gw_csv_dir (str): Input GW csv directory containing yearly withdrawal CSVs.
        out_gw_shp_dir (str): Output directory to store the GW withdrawal data.
        fill_attr (str): Attribute present in the CSV file to add to Well Registry
        filter_attr (str): Remove specific wells based on this attribute. Set None to disable filtering.
        filter_attr_value (str): Value for filter_attr
        use_only_ama_ina (bool): Set True to use only AMA/INA for model training
        kwargs (dict (str, str)): Additional variables, which include csv_well_id='Well Id',
                                  csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL',
                                  water_type='GROUNDWATER', and shp_well_id='REGISTRY_I' as defaults.

    Returns:
        None.
    """

    num_cores = multiprocessing.cpu_count() - 1
    print('Updating Well Registry shapefiles...\n')
    Parallel(n_jobs=num_cores - 1)(delayed(parallel_add_attribute_well_reg)(
        input_well_reg_file,
        input_gw_csv_file,
        out_gw_shp_dir,
        fill_attr,
        filter_attr,
        filter_attr_value,
        use_only_ama_ina,
        **kwargs
    ) for input_gw_csv_file in glob(input_gw_csv_dir + '*.csv'))


def parallel_add_attribute_well_reg(input_well_reg_file, input_gw_csv_file, out_gw_shp_dir, fill_attr='AF Pumped',
                                    filter_attr='AMA', filter_attr_value='OUTSIDE OF AMA OR INA',
                                    use_only_ama_ina=False, **kwargs):
    """
    Add an attribute present in the GW csv file to the Well Registry shape files (yearwise) based on matching ids given
    in kwargs. By default, the GW withdrawal is added. The csv ids must include: csv_well_id, csv_mov_id, csv_water_id,
    movement_type, water_type, The shp id must include shp_well_id. For the Arizona datasets, csv_well_id='Well Id',
    csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL', water_type='GROUNDWATER', and
    shp_well_id='REGISTRY_I' by default. For changing, pass appropriate kwargs. This function should be called from
    #add_attribute_well_reg_multiple(...)
    :param input_well_reg_file: Input well registry shapefile
    :param input_gw_csv_file: Input GW csv file
    :param out_gw_shp_dir: Output GW geojson directory having GW withdrawal data
    :param fill_attr: Attribute present in the CSV file to add to Well Registry
    :param filter_attr: Remove specific wells based on this attribute. Set None to disable filtering.
    :param use_only_ama_ina: Set True to use only AMA/INA for model training
    :param filter_attr_value: Value for filter_attr
    :return: None
    """

    out_well_reg_file = out_gw_shp_dir + input_gw_csv_file[
                                         input_gw_csv_file.rfind(os.sep) + 1: input_gw_csv_file.rfind('.')
                                         ] + '.shp'
    add_attribute_well_reg(
        input_well_reg_file,
        input_gw_csv_file,
        out_well_reg_file,
        fill_attr,
        filter_attr,
        filter_attr_value,
        use_only_ama_ina,
        **kwargs
    )


def gdf2shp(input_df, geometry, source_crs, target_crs, outfile_path):
    """
    Convert Geodatafarme to SHP
    :param input_df: Input geodataframe
    :param geometry: Geometry (Point) list
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param outfile_path: Output file path
    :return:
    """

    crs = {'init': source_crs}
    gdf = gpd.GeoDataFrame(input_df, crs=crs, geometry=geometry)
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

    num_cores = multiprocessing.cpu_count() - 2
    Parallel(n_jobs=num_cores)(delayed(parallel_shp2raster)(
        shp_file,
        output_dir=output_dir,
        burn_value=burn_value,
        value_field=value_field,
        value_field_pos=value_field_pos,
        xres=xres, yres=yres,
        add_value=add_value
    ) for shp_file in glob(input_dir + '*.shp'))


def parallel_shp2raster(
        shp_file: str,
        output_dir: str,
        burn_value: float | None = None,
        value_field: str | None = None,
        value_field_pos: int = 0,
        xres: float = 1000,
        yres: float = 1000,
        add_value: bool = True,
) -> None:
    """
    Use this from #shp2rasters to parallelize raster creation.
    Args:
        shp_file (str): Input shapefile path.
        output_dir (str): Output TIFF directory.
        burn_value (float or None): Set burn value. If not None, then add_value, value_field, and value_field_pos
                                    arguments are ignored.
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing). Only used if value_field is None.
        xres (float): Pixel width in geographic units.
        yres (float): Pixel height in geographic units.
        add_value (bool): Set False to disable adding value to existing raster cell.

    Returns:
        None
    """

    outfile_path = output_dir + shp_file[shp_file.rfind(os.sep) + 1: shp_file.rfind('.') + 1] + 'tif'
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
