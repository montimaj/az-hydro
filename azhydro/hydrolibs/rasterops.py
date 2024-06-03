# Author: Sayantan Majumdar
# Email: sayantan.majumdar@colostate.edu

import matplotlib.pyplot as plt
import rasterio as rio
import geopandas as gpd
import numpy as np
import os
import multiprocessing

from osgeo import gdal
from joblib import Parallel, delayed
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry import Point
from glob import glob
from fiona import transform
from sysops import az_nodata


def read_raster_as_arr(
        raster_file: str | rio.DatasetReader,
        band: int = 1,
        get_file: bool = True,
        change_dtype: bool = True
) -> tuple[np.ndarray, rio.DatasetReader] | np.ndarray:
    """Read a raster band as a numpy array.

    Args:
        raster_file (str or rio.DatasetReader): Input raster file path or rasterio DatasetReader object.
        band (int): Selected band to read (Default 1).
        get_file (bool): Get rasterio DatasetReader object file if set to True.
        change_dtype (bool): Change raster data type to float if True. Also, if a no data value exists, then it is set
                             to np.nan if change_dtype is True.

    Returns:
        np.ndarray: Raster numpy array (if, get_file is False).
        tuple (np.ndarray, rio.DatasetReader): A tuple of raster numpy array and rasterio object file (if,
                                               get_file is True and raster_file is a raster file path).
    """
    rasterio_obj = isinstance(raster_file, rio.DatasetReader)
    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata is not None:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
    if get_file:
        return raster_arr, raster_file
    return raster_arr


def write_raster(
        raster_data: np.ndarray,
        raster_file: rio.DatasetReader,
        transform_: rio.transform,
        outfile_path: str,
        no_data_value: float,
        ref_file: str | None = None,
        out_crs: str | None = None
) -> None:
    """Write raster file in GeoTIFF format.

    Args:
        raster_data (np.ndarray): Raster data (numpy array) to be written.
        raster_file (rio.DatasetReader): Original rasterio raster DatasetReader object containing geo-coordinates.
        transform_ (rio.transform): Affine transformation matrix.
        outfile_path (str): Outfile file path.
        no_data_value (float): No data value for raster (default float32 type is considered).
        ref_file (str): Write output raster considering parameters from reference raster file path
        out_crs (str): Output crs.

    Returns:
        None
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform_ = raster_file.transform
    crs = raster_file.crs
    if out_crs:
        crs = out_crs
    with rio.open(
            outfile_path,
            'w',
            driver='GTiff',
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            dtype=raster_data.dtype,
            crs=crs,
            transform=transform_,
            count=raster_file.count,
            nodata=no_data_value
    ) as dst:
        dst.write(raster_data, raster_file.count)


def crop_raster(
        input_raster_file: str,
        input_mask_path: str,
        outfile_path: str,
        plot_fig: bool = False,
        plot_title: str = "",
        ext_mask: bool = True,
        multi_poly: bool = False,
) -> None:
    """
    Crop raster data based on given shapefile.

    Args:
        input_raster_file (str): Input raster dataset path.
        input_mask_path (str): Shapefile path.
        outfile_path (str): Output file path (only tiff file).
        plot_fig (bool): If true, then cropped raster data is plotted.
        plot_title (str): Plot title to display.
        ext_mask (bool): Set true to extract raster by mask file.
        multi_poly (bool): Set True if input_mask_file has multiple polygons/features.

    Returns:
        None.
    """

    if multi_poly:
        mask_shp_file = gpd.read_file(input_mask_path)
        raster_arr, raster_file = read_raster_as_arr(input_raster_file)
        for idx, value in np.ndenumerate(raster_arr):
            gx, gy = raster_file.xy(idx[0], idx[1])
            gp = Point(gx, gy)
            check_flag = False
            for poly in mask_shp_file['geometry']:
                if poly.contains(gp):
                    check_flag = True
                    break
            if not check_flag:
                raster_arr[idx] = np.nan
        no_data_value = az_nodata()
        raster_arr[np.isnan(raster_arr)] = no_data_value
        write_raster(
            raster_arr, raster_file, transform_=raster_file.transform,
            outfile_path=outfile_path, no_data_value=no_data_value
        )
    else:
        if ext_mask:
            src_raster_file = gdal.Open(input_raster_file)
            src_band = src_raster_file.GetRasterBand(1)
            transform_ = src_raster_file.GetGeoTransform()
            xres, yres = transform_[1], transform_[5]
            no_data = src_band.GetNoDataValue()
            os_sep = input_mask_path.rfind(os.sep)
            if os_sep == -1:
                os_sep = input_mask_path.rfind('/')
            layer_name = input_mask_path[os_sep + 1: input_mask_path.rfind('.')]
            warp_options = gdal.WarpOptions(
                dstNodata=no_data,
                cutlineDSName=input_mask_path,
                cutlineLayer=layer_name,
                cropToCutline=True,
                targetAlignedPixels=True,
                xRes=xres, yRes=yres,
                outputType=gdal.GDT_Float32,
                format='GTiff',
                options=['-overwrite']
            )
            gdal.Warp(outfile_path, input_raster_file, options=warp_options)
        else:
            shape_file = gpd.read_file(input_mask_path)
            shape_file_geom = mapping(shape_file['geometry'][0])
            raster_file = rio.open(input_raster_file)
            raster_crop, raster_transform = mask(raster_file, [shape_file_geom], crop=True)
            shape_extent = plotting_extent(raster_crop[0], raster_transform)
            raster_crop = np.squeeze(raster_crop)
            write_raster(raster_crop, raster_file, transform_=raster_transform, outfile_path=outfile_path,
                         no_data_value=raster_file.nodata)
            if plot_fig:
                fig, ax = plt.subplots(figsize=(10, 8))
                raster_plot = ax.imshow(raster_crop[0], extent=shape_extent)
                ax.set_title(plot_title)
                ax.set_axis_off()
                fig.colorbar(raster_plot)
                plt.show()


def reproject_coords(
        src_crs: str,
        dst_crs: str,
        coords: tuple[tuple[float, float], ...]
) -> list[tuple[float, float]]:
    """Reproject coordinates. Copied from https://bit.ly/3mBtowB.

    Author: user2856 (StackExchange user).

    Args:
        src_crs (str): Source CRS.
        dst_crs (str): Destination CRS.
        coords (tuple( tuple(float, float), ...): Coordinates as a tuple of long, lat pairs as tuples.

    Returns:
        list (tuple (float, float)): Transformed coordinates as a list of long, lat pairs as tuples.
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = transform.transform(src_crs, dst_crs, xs, ys)
    return [(x, y) for x, y in zip(xs, ys)]


def get_raster_extent(
        input_raster: str | rio.DatasetReader,
        new_crs: str | None = None
) -> tuple[float, float, float, float]:
    """Get raster extents using rasterio.

    Args:
        input_raster (str or rio.DatasetReader): Input raster file path or rasterio DatasetReader object.
        new_crs (str): Specify a new crs to convert original extents.

    Returns:
        tuple (float, float, float, float): A tuple containing raster extents as (left, bottom, right, top).
    """
    is_rio_obj = isinstance(input_raster, rio.DatasetReader)
    if not is_rio_obj:
        input_raster = rio.open(input_raster)
    raster_extent = input_raster.bounds
    left = raster_extent.left
    bottom = raster_extent.bottom
    right = raster_extent.right
    top = raster_extent.top
    if new_crs:
        raster_crs = input_raster.crs.to_string()
        if raster_crs != new_crs:
            new_coords = reproject_coords(
                raster_crs,
                new_crs,
                ((left, bottom), (right, top))
            )
            left, bottom = new_coords[0]
            right, top = new_coords[1]
    return left, bottom, right, top


def reproject_raster_gdal(
        input_raster_file: str,
        outfile_path: str,
        resampling_factor: int | None = 1,
        resampling_func: str = 'near',
        downsampling: bool = True,
        from_raster: str | rio.DatasetReader | None = None,
        keep_original: bool = False,
        dst_xres: float | None = None,
        dst_yres: float | None = None,
        output_dtype: str = 'float32'
) -> None:
    """Reproject raster using GDALWarp Python API.

    Args:
        input_raster_file (str): Input raster file.
        outfile_path (str): Output file path.
        resampling_factor (int or None): Resampling factor (default 1).
        resampling_func (str): Resampling function. Valid names include 'near', 'bilinear', 'cubic', 'cubicspline',
                               'lanczos', 'sum', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'.
        downsampling (bool): Downsample raster (default True).
        from_raster (str or rio.DatasetReader or None): Reproject input raster considering another raster
                                                        (either raster path or rasterio object).
        keep_original (bool): Set True to only use the new projection system from 'from_raster'.
                              The original raster extent is not changed.
        dst_xres (float or None): Target xres in input_raster_file units. Set resampling_factor to None.
        dst_yres (float or None): Target yres in input_raster_file units. Set resampling factor to None.
        output_dtype (str):  Output data type. Valid data types include 'byte', 'int16', 'int32', 'float32'.

    Returns:
        None
    """
    src_raster_file = rio.open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        if isinstance(from_raster, str):
            rfile = rio.open(from_raster)
        else:
            rfile = from_raster
        resampling_factor = 1
    xres, yres = rfile.res
    extent = get_raster_extent(rfile)
    dst_proj = rfile.crs.to_string()
    no_data = src_raster_file.nodata
    if dst_xres and dst_yres:
        xres, yres = dst_xres, dst_yres
    elif resampling_factor:
        if not downsampling:
            resampling_factor = 1 / resampling_factor
        xres, yres = xres * resampling_factor, yres * resampling_factor
    resampling_dict = {
        'near': gdal.GRA_NearestNeighbour, 'bilinear': gdal.GRA_Bilinear, 'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline, 'lanczos': gdal.GRA_Lanczos,  'sum': gdal.GRA_Sum,
        'average': gdal.GRA_Average, 'mode': gdal.GRA_Mode, 'max': gdal.GRA_Max,
        'min': gdal.GRA_Min, 'med': gdal.GRA_Med, 'q1': gdal.GRA_Q1, 'q3': gdal.GRA_Q3,
    }
    output_dtype_dict = {
        'byte': gdal.GDT_Byte,
        'int16': gdal.GDT_Int16,
        'int32': gdal.GDT_Int32,
        'float32': gdal.GDT_Float32
    }
    warp_options = gdal.WarpOptions(
        outputBounds=extent,
        dstNodata=no_data,
        dstSRS=dst_proj,
        resampleAlg=resampling_dict[resampling_func],
        xRes=xres, yRes=yres,
        outputType=output_dtype_dict[output_dtype],
        multithread=True,
        format='GTiff',
        options=['-overwrite']
    )
    gdal.Warp(outfile_path, input_raster_file, options=warp_options)


def crop_rasters(
        input_raster_dir: str,
        input_mask_file: str,
        outdir: str,
        pattern: str = '*.tif',
        ext_mask: bool = True,
        multi_poly: bool = False,
        verbose: bool = False
) -> None:
    """
    Crop multiple rasters in a directory.

    Args:
    input_raster_dir (str): Directory containing raster files which are named as *_<Year>.*
    input_mask_file (str): Mask file (shapefile) used for cropping.
    outdir (str): Output directory for storing masked rasters.
    pattern (str): Raster extension.
    ext_mask (str): Set False to extract by geometry only.
    multi_poly (bool): Set True if input_mask_file has multiple polygons/features.
    verbose (bool): Set True to print system call info.

    Returns:
        None.
    """

    num_cores = multiprocessing.cpu_count() - 1
    Parallel(n_jobs=num_cores)(delayed(parallel_crop_rasters)(raster_file, input_mask_file, outdir, ext_mask,
                                                              multi_poly, verbose)
                               for raster_file in glob(input_raster_dir + pattern))


def parallel_crop_rasters(
        input_raster_file,
        input_mask_file,
        outdir, ext_mask=True,
        multi_poly=False, verbose=False
):
    """
    Parallely crop rasters, should be called from #crop_rasters(...)
    :param input_raster_file: Input raster file
    :param input_mask_file: Mask file (shapefile) used for cropping
    :param outdir: Output directory for storing masked rasters
    :param ext_mask: Set False to extract by geometry only
    :param multi_poly: Set True if input_mask_file has multiple polygons/features
    :param verbose: Set True to print system call info
    :return: None
    """

    out_raster = outdir + input_raster_file[input_raster_file.rfind(os.sep) + 1:]
    if verbose:
        print('Cropping', input_raster_file, '...')
    crop_raster(
        input_raster_file, input_mask_file,
        out_raster, ext_mask=ext_mask,
        multi_poly=multi_poly
    )


def convert_gw_data(
        input_raster_dir: str,
        outdir: str,
        pattern: str = '*.tif'
) -> None:
    """
    Convert groundwater data (in acreft) to mm.

    Args:
        input_raster_dir (str): Input raster directory.
        outdir (str): Output raster directory.
        pattern (str): Raster extension.

    Returns:
        None.
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_ref = read_raster_as_arr(raster_file)
        transform = raster_ref.get_transform()
        xres, yres = transform[1] / 1000., transform[5] / 1000.
        raster_arr[~np.isnan(raster_arr)] *= 1.233 / (np.abs(xres * yres))
        no_data = az_nodata()
        raster_arr[np.isnan(raster_arr)] = no_data
        write_raster(
            raster_arr, raster_ref, transform_=raster_ref.transform,
            outfile_path=out_raster, no_data_value=no_data
        )


def fix_gw_raster_values(
        input_raster_dir: str,
        outdir: str,
        max_threshold: float = 1e+5,
        fix_only_negative: bool = False,
        pattern: str = 'GW*.tif'
) -> None:
    """
    Fix unusually large values introduced by gdal_rasterize sometimes or remove negative pumpings indicating
    no well data.

    Args:
        input_raster_dir (str): Input raster directory.
        outdir (str): Output directory.
        max_threshold (float): Max value beyond which values will be set to no data value, default unit is acrefeet.
        fix_only_negative (bool): Set True to fix only negative values.
        pattern (str): File pattern.

    Returns:
        None.
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_file = read_raster_as_arr(raster_file)
        no_data = az_nodata()
        raster_arr[np.isnan(raster_arr)] = no_data
        raster_arr[np.logical_and(raster_arr > 0, raster_arr < 1e-8)] = 0.
        if fix_only_negative:
            raster_arr[raster_arr < 0] = no_data
        else:
            raster_arr[raster_arr >= max_threshold] = no_data
        write_raster(
            raster_arr, raster_file, transform_=raster_file.transform,
            outfile_path=out_raster, no_data_value=no_data
        )
