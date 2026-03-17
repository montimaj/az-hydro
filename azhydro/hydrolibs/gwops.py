"""
Handle groundwater withdrawal processing codes.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import multiprocessing
import os
import shutil
import warnings
from glob import glob
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as flt
from joblib import Parallel, delayed

import hydrolibs.rasterops as rops
import hydrolibs.vectorops as vops
from hydrolibs.sysops import copy_file, makedirs

logger = logging.getLogger(__name__)


def reproject_vectors(
        input_dir: str,
        output_dir: str,
        ref_file: str,
        already_reprojected: bool = False
) -> dict[str, str]:
    """
    Reproject all shapefiles and geojsons found recursively under input_dir.

    Args:
        input_dir (str): Input directory containing vector files (searched recursively).
        output_dir (str): Output directory for reprojected files (flat structure).
        ref_file (str): Reference vector file for target CRS.
        already_reprojected (bool): Set True to disable reprojection.

    Returns:
        dict[str, str]: Mapping of stem filename (without extension) to reprojected file path.
    """

    def _find_vectors(search_dir):
        shps = glob(os.path.join(search_dir, '**', '*.shp'), recursive=True)
        geojsons = glob(os.path.join(search_dir, '**', '*.geojson'), recursive=True)
        return sorted(shps + geojsons)

    if not already_reprojected:
        makedirs(output_dir)
        vector_files = _find_vectors(input_dir)

        def _reproject(f):
            basename = os.path.basename(f)
            output_f = os.path.join(output_dir, basename)
            vops.reproject_vector(
                f,
                outfile_path=output_f,
                ref_file=ref_file,
                raster=False
            )

        num_cores = min(multiprocessing.cpu_count() - 1, len(vector_files))
        Parallel(n_jobs=num_cores)(delayed(_reproject)(f) for f in vector_files)
    else:
        logger.info('Vector files are already reprojected')

    reprojected = _find_vectors(output_dir)
    result = {}
    for f in reprojected:
        stem = os.path.splitext(os.path.basename(f))[0]
        result[stem] = f
    return result


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
        logger.info('GW rasters already cropped')
    return cropped_dir


def preprocess_gw_csv(
        well_registry_file: str,
        input_gw_csv_dir: str,
        output_dir: str,
        fill_attr: str = 'AF Pumped',
        filter_attr: str | None = None,
        filter_attr_value: str = 'NOT WITHIN ANY AMA OR INA',
        use_only_ama_ina: bool = False,
        already_preprocessed: bool = False,
        af_max_threshold: float = 5000.,
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
        af_max_threshold (float): Maximum per-well AF Pumped value; rows exceeding this are dropped.
        kwargs (dict (str, str)): Additional variables, which include csv_well_id='Well Id',
                                  csv_mov_id='Movement Type', csv_water_id='Water Type', movement_type='WITHDRAWAL',
                                  water_type='GROUNDWATER', water_use = 'IRRIGATION', and shp_well_id='REGISTRY_I'. If
                                  water_use is set to 'All', then all water uses will be considered. 
                                  Default is 'IRRIGATION'.

    Returns:
        str: File path of the first GW pumping shapefile or geojson.
    """

    def _process_csv(csv_file):
        out_shp = os.path.join(output_dir, f'{csv_file[csv_file.rfind(os.sep) + 1:csv_file.rfind(".")]}.shp')
        vops.add_attribute_well_reg(
            well_registry_file, csv_file, out_shp,
            fill_attr, filter_attr, filter_attr_value,
            use_only_ama_ina, af_max_threshold, **kwargs
        )

    if not already_preprocessed:
        makedirs(output_dir)
        csv_files = glob(f'{input_gw_csv_dir}*.csv')
        num_cores = multiprocessing.cpu_count() - 2
        logger.info('Updating Well Registry shapefiles...')
        Parallel(n_jobs=num_cores)(delayed(_process_csv)(f) for f in csv_files)

    ref_file = glob(os.path.join(output_dir, '*.shp'))[0]
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

    for raster_file in glob(os.path.join(input_raster_dir, gw_pattern)):
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
        logger.info('Creating GW withdrawal volume (acreft) rasters...')
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
            max_gw = 10_000 * xres * yres / 1.233e+6  # 10,000 mm default cap
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
        logger.info('GW  pumping volume rasters already created')


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

        def _convert_volume_to_depth(gw_volume_file):
            gw_depth_file = f'{output_gw_dir}{gw_volume_file[gw_volume_file.rfind(os.sep) + 1:]}'
            gw_vol_arr, gw_vol_ref = rops.read_raster_as_arr(gw_volume_file)
            xres, yres = gw_vol_ref.res
            gw_depth_arr = gw_vol_arr * 1.233 / abs(xres * yres * 1e-6)
            gw_depth_arr[np.isnan(gw_depth_arr)] = nodata
            rops.write_raster(
                gw_depth_arr, gw_vol_ref, transform_=gw_vol_ref.transform,
                outfile_path=gw_depth_file, no_data_value=nodata
            )

        volume_files = glob(os.path.join(gw_volume_dir, gw_pattern))
        num_cores = min(multiprocessing.cpu_count() - 1, len(volume_files))
        Parallel(n_jobs=num_cores)(
            delayed(_convert_volume_to_depth)(f) for f in volume_files
        )
    else:
        logger.info('GW pumping depth rasters already created...')


def create_gw_basin_rasters(
        gw_basin_vector: str,
        output_dir: str,
        xres: float = 500,
        yres: float = 500,
        start_year: int = 1985,
        end_year: int = 2023,
        already_created: bool = False,
        verbose: bool = False,
        subbasin_vector: str | None = None,
) -> None:
    """
    Create GW basin (and optionally sub-basin) rasters for Arizona.

    Args:
        gw_basin_vector (str): GW Basin shapefile or geojson for Arizona.
        output_dir (str): Output directory.
        xres (float): X-Resolution (m).
        yres (float): Y-Resolution (m).
        start_year (int): Start year in YYYY.
        end_year (int): End year in YYYY.
        already_created (bool): Set True if raster already exists.
        verbose (bool): Set True to print additional information.
        subbasin_vector (str | None): ADWR Groundwater Sub-basin shapefile.
            When provided, ``GW_Subbasin_{year}.tif`` rasters are created
            alongside the basin rasters.

    Returns:
        None
    """

    if not already_created:
        year_list = range(start_year + 1, end_year + 1)
        makedirs(output_dir)
        az_gw_basin_sy_tif = os.path.join(output_dir, f'GW_Basin_{start_year}.tif')
        vops.shp2raster(
            gw_basin_vector,
            az_gw_basin_sy_tif,
            xres=xres, yres=yres,
            value_field='OBJECTID',
            add_value=False
        )

        if subbasin_vector is not None:
            az_gw_subbasin_sy_tif = os.path.join(output_dir, f'GW_Subbasin_{start_year}.tif')
            vops.shp2raster(
                subbasin_vector,
                az_gw_subbasin_sy_tif,
                xres=xres, yres=yres,
                value_field='OBJECTID',
                add_value=False
            )

        for year in year_list:
            az_gw_basin_tif = os.path.join(output_dir, f'GW_Basin_{year}.tif')
            copy_file(
                az_gw_basin_sy_tif,
                az_gw_basin_tif,
                verbose=verbose
            )
            if subbasin_vector is not None:
                az_gw_subbasin_tif = os.path.join(output_dir, f'GW_Subbasin_{year}.tif')
                copy_file(
                    az_gw_subbasin_sy_tif,
                    az_gw_subbasin_tif,
                    verbose=verbose
                )

    logger.info('GW Basin rasters created...')


def create_land_use_data(
        input_df: pd.DataFrame,
        cdl_arr: np.array,
        smoothing: int = 3
) -> pd.DataFrame:
    """
    Create Gaussian-filtered land use array.

    Args:
        input_df (pd.DataFrame): Dataframe used to store the Gaussian-filtered LULC arrays, where
                                 1 = Agriculture, 2 = Surface Water, and 3 = Urban.
        cdl_arr (np.array): CDL array.
        smoothing (int): Smoothing window size for the Gaussian filter.

    Returns:
        pd.DataFrame: input_df updated with the Gaussian-filtered reclassified CDL arrays.
    """

    cdl_labels = ('AGRI', 'SW', 'URBAN')
    cdl_arr = cdl_arr.astype(np.float32)
    for idx, cdl_label in enumerate(cdl_labels):
        lu_arr = np.full_like(cdl_arr, fill_value=0.)
        lu_arr[cdl_arr == idx + 1] = 1
        gaussian_lu_arr = flt.gaussian_filter(lu_arr, sigma=smoothing, order=0)
        gaussian_lu_arr = np.abs(gaussian_lu_arr)
        gaussian_lu_arr -= np.min(gaussian_lu_arr)
        ptp = np.ptp(gaussian_lu_arr)
        if ptp > 0:
            gaussian_lu_arr /= ptp
        input_df[cdl_label] = gaussian_lu_arr.ravel()
    return input_df


def get_ama_ina_basin_names() -> list[str]:
    """
    Get the names of AMA and INA basins.
    
    .. deprecated::
        Import from ``hydrolibs.visualops`` instead.

    Returns:
        list: List of AMA and INA basin names.
    """
    warnings.warn(
        "gwops.get_ama_ina_basin_names() is deprecated; "
        "import from hydrolibs.visualops instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import hydrolibs.visualops as vizops
    return vizops.get_ama_ina_basin_names()


def parallel_make_time_series_plots(
        idx: int,
        gw_basin: str,
        input_df: pd.DataFrame,
        output_dir: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str,
        actual_gw_col: str,
        pred_gw_col: str,
        gw_basin_col: str,
        split_strategy: int,
        test_gw_basins: tuple[str],
        raster_res: float = 2000
) -> None:
    """
    Create time series plots for individual groundwater basins.
    
    Note: This function now uses journal-quality plotting from visualops.
    For advanced plotting options, use visualops.create_basin_time_series_plot() directly.

    Args:
        idx (int): Index of the groundwater basin.
        gw_basin (str): Name of the groundwater basin.
        input_df (pd.DataFrame): Input dataframe containing the ML-predicted and actual pumping data.
        output_dir (str): Output directory.
        test_year_limits (tuple[tuple[int, int], ...]): Tuple of tuples containing the start and end years for testing.
        year_col (str): Name of the year column.
        actual_gw_col (str): Name of the actual groundwater pumping column.
        pred_gw_col (str): Name of the predicted groundwater pumping column.
        gw_basin_col (str): Name of the groundwater basin column.
        split_strategy (int): Split strategy used for the model. 1 = Temporal, 2 = Random Stratified, 3 = Spatial.
        test_gw_basins (tuple (str)): Test GW basin names.
        raster_res (float): Raster resolution in meters.

    Returns:
        None.
    """
    # Use the new visualops module for journal-quality plots
    import hydrolibs.visualops as vizops
    vizops.parallel_make_time_series_plots(
        idx, gw_basin, input_df, output_dir, test_year_limits,
        year_col, actual_gw_col, pred_gw_col, gw_basin_col,
        split_strategy, test_gw_basins, raster_res
    )


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
        raster_res: float = 2000,
        x_scaler: Any = None,
        y_scaler: Any = None
) -> None:
    """
    Make time series plots for individual groundwater basins.
    
    Note: This function now uses journal-quality plotting from visualops.
    For advanced plotting options, use visualops.create_complete_model_visualization() directly.

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
        x_scaler (Any): Scaler used to scale the features. If None, the model's predict method is used directly.
        y_scaler (Any): Scaler used to scale the target variable. If None, the model's predict method is used directly.

    Returns:
        None.
    """
    # Use the new visualops module for journal-quality plots
    import hydrolibs.visualops as vizops
    vizops.make_time_series_plots(
        input_df, model, features, output_dir,
        year_col, gw_basin_col, test_year_limits,
        pred_attr, split_strategy, test_gw_basins,
        raster_res, x_scaler, y_scaler
    )


def create_well_density_raster(
        well_registry_file: str,
        output_dir: str,
        water_use: str | None = None,
        xres: float = 2000,
        yres: float = 2000,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False
) -> str:
    """
    Create well density rasters (well count per pixel) from the ADWR Well Registry.

    A single raster is computed from the well registry and then replicated for
    each year to maintain consistency with the per-year predictor file pattern.

    Args:
        well_registry_file (str): Path to the ADWR Well Registry shapefile.
        output_dir (str): Output directory for the well density rasters.
        water_use (str or None): Filter wells by WATER_USE attribute.
            E.g. 'IRRIGATION'. Set None to include all wells.
        xres (float): X-resolution in map units. Defaults to 2000.
        yres (float): Y-resolution in map units. Defaults to 2000.
        start_year (int): Start year. Defaults to 1896.
        end_year (int): End year. Defaults to 2099.
        already_created (bool): If True, skip creation.

    Returns:
        str: Path to the base well density raster.
    """

    base_raster = os.path.join(output_dir, f'Well_Density_{start_year}.tif')
    if already_created:
        logger.info('Well density rasters already created, skipping...')
        return base_raster

    makedirs(output_dir)

    # Load well registry
    gdf = gpd.read_file(well_registry_file)
    if water_use:
        gdf = gdf[gdf['WATER_USE'] == water_use]
    logger.info(f'Using {len(gdf)} wells for density rasterization')

    # Add count attribute
    gdf = gdf.copy()
    gdf['_count'] = 1.0

    # Save as temp shapefile
    tmp_dir = os.path.join(output_dir, '_well_temp')
    makedirs(tmp_dir)
    tmp_shp = f'{tmp_dir}wells.shp'
    gdf[['_count', 'geometry']].to_file(tmp_shp)

    # Rasterize: well count per pixel
    vops.shp2raster(
        tmp_shp, base_raster,
        value_field='_count',
        xres=xres, yres=yres,
        add_value=True
    )

    # Copy for each year
    for year in range(start_year + 1, end_year + 1):
        out_file = os.path.join(output_dir, f'Well_Density_{year}.tif')
        copy_file(base_raster, out_file, verbose=False)

    # Clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f'Created well density rasters ({start_year}-{end_year}) in {output_dir}')
    return base_raster
