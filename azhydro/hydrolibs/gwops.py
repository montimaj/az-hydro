"""
Handle groundwater withdrawal processing codes.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import multiprocessing
import os
import shutil
from glob import glob
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as flt
from joblib import Parallel, delayed

import hydrolibs.rasterops as rops
import hydrolibs.vectorops as vops
from hydrolibs.sysops import (NON_CONSUMPTIVE_USES, copy_file,
                              derive_well_start_year, makedirs)

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

    cropped_dir = os.path.join(output_gw_dir, 'GW_Cropped')
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
        csv_files = glob(os.path.join(input_gw_csv_dir, '*.csv'))
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
        out_raster = os.path.join(outdir, os.path.basename(raster_file))
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
        gw_volume_dir_uncorrected = os.path.join(output_gw_dir, 'Uncorrected_GW_Volumes')
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
            gw_depth_file = os.path.join(output_gw_dir, gw_volume_file[gw_volume_file.rfind(os.sep) + 1:])
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
                                 1 = Agriculture, 2 = Urban, and 3 = Surface Water.
        cdl_arr (np.array): CDL array.
        smoothing (int): Smoothing window size for the Gaussian filter.

    Returns:
        pd.DataFrame: input_df updated with the Gaussian-filtered reclassified CDL arrays.
    """

    cdl_labels = ('AGRI', 'URBAN', 'SW')  # matches GEE recode: 1=AGRI, 2=URBAN, 3=SW
    # SW uses a tighter kernel (sigma=1) to avoid spreading narrow
    # water-body signals across basin boundaries (e.g., into Willcox).
    sw_smoothing = max(smoothing // 3, 1)
    label_sigma = {'AGRI': smoothing, 'URBAN': smoothing, 'SW': sw_smoothing}
    cdl_arr = cdl_arr.astype(np.float32)
    for idx, cdl_label in enumerate(cdl_labels):
        lu_arr = np.full_like(cdl_arr, fill_value=0.)
        lu_arr[cdl_arr == idx + 1] = 1
        gaussian_lu_arr = flt.gaussian_filter(lu_arr, sigma=label_sigma[cdl_label], order=0)
        gaussian_lu_arr = np.abs(gaussian_lu_arr)
        gaussian_lu_arr -= np.min(gaussian_lu_arr)
        ptp = np.ptp(gaussian_lu_arr)
        if ptp > 0:
            gaussian_lu_arr /= ptp
        if cdl_label == 'SW':
            gaussian_lu_arr = np.round(gaussian_lu_arr, 2)
        input_df[cdl_label] = gaussian_lu_arr.ravel()
    return input_df


def generate_basin_data_summary(
        az_df: pd.DataFrame,
        output_dir: str,
        year_list: list[int],
        pred_attr: str = 'gw_pumping_mm',
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
) -> pd.DataFrame:
    """Generate per-basin data availability summary for AMA/INA basins.

    Computes pixel count, non-zero row count, metered year coverage, and
    basic pumping statistics for each AMA/INA basin during the metered period.

    Args:
        az_df: Full Arizona predictor DataFrame.
        output_dir: Directory to save the CSV.
        year_list: Metered years to include (e.g. 1984-2024).
        pred_attr: Target column name.
        year_col: Year column name.
        gw_basin_col: Basin column name.

    Returns:
        pd.DataFrame: Summary table (one row per basin, sorted by non-zero count).
    """
    import hydrolibs.visualops as vizops

    makedirs(output_dir)
    ama_ina = vizops.get_ama_ina_basin_names()
    metered = az_df[
        az_df[year_col].isin(year_list) &
        az_df[gw_basin_col].isin(ama_ina)
    ]
    n_years = len(year_list)

    rows = []
    for basin in sorted(ama_ina):
        sub = metered[metered[gw_basin_col] == basin]
        if sub.empty:
            continue
        vals = sub[pred_attr]
        pos = sub[sub[pred_attr] > 0]
        pixels_per_year = len(sub) // max(n_years, 1)
        metered_years = sorted(pos[year_col].unique().tolist()) if not pos.empty else []
        rows.append({
            'Basin': basin,
            'Pixels_Per_Year': pixels_per_year,
            'Total_Rows': len(sub),
            'Rows_GT_0': len(pos),
            'Rows_EQ_0': int((vals == 0).sum()),
            'Rows_NaN': int(vals.isna().sum()),
            'Pct_NonZero': round(100 * len(pos) / max(len(sub), 1), 2),
            'Metered_Years': len(metered_years),
            'First_Year': metered_years[0] if metered_years else None,
            'Last_Year': metered_years[-1] if metered_years else None,
            'Mean_GT_0_mm': round(pos[pred_attr].mean(), 1) if not pos.empty else None,
            'Max_mm': round(pos[pred_attr].max(), 1) if not pos.empty else None,
        })

    summary_df = pd.DataFrame(rows).sort_values('Rows_GT_0', ascending=False)
    csv_path = os.path.join(output_dir, 'Basin_Data_Summary.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.info(f'Basin data summary ({len(summary_df)} basins) saved to {csv_path}')
    return summary_df


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
        water_use_exclude: bool = False,
        output_prefix: str = 'Well_Density',
        xres: float = 2000,
        yres: float = 2000,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False
) -> str:
    """
    Create per-year well density rasters (well count per pixel) from the ADWR
    Well Registry, filtered by installation date.

    For each year, only wells installed by that year are included, producing
    temporally varying density rasters that reflect the actual build-out of
    well infrastructure over time.

    Args:
        well_registry_file (str): Path to the ADWR Well Registry shapefile.
        output_dir (str): Output directory for the well density rasters.
        water_use (str or None): Filter wells by WATER_USE attribute using
            substring matching (``str.contains``).  E.g. ``'IRRIGATION'``
            matches ``'IRRIGATION'``, ``'IRRIGATION, DOMESTIC'``, etc.
            Set None to include all consumptive wells.
        water_use_exclude (bool): If True, **exclude** wells matching
            *water_use* instead of keeping them.  E.g.
            ``water_use='IRRIGATION', water_use_exclude=True`` keeps
            only non-irrigation consumptive wells.
        output_prefix (str): Filename prefix for the output rasters
            (default ``'Well_Density'`` → ``Well_Density_YYYY.tif``).
        xres (float): X-resolution in map units. Defaults to 2000.
        yres (float): Y-resolution in map units. Defaults to 2000.
        start_year (int): Start year. Defaults to 1896.
        end_year (int): End year. Defaults to 2099.
        already_created (bool): If True, skip creation.

    Returns:
        str: Path to the base well density raster.
    """

    base_raster = os.path.join(output_dir, f'{output_prefix}_{start_year}.tif')
    if already_created:
        logger.info(f'{output_prefix} rasters already created, skipping...')
        return base_raster

    makedirs(output_dir)

    # Load well registry and drop non-consumptive wells
    gdf = gpd.read_file(well_registry_file)
    pattern = '|'.join(NON_CONSUMPTIVE_USES)
    has_use = gdf['WATER_USE'].notna()
    gdf = gdf[has_use & ~gdf['WATER_USE'].str.contains(pattern, na=False)]
    if water_use:
        mask = gdf['WATER_USE'].str.contains(water_use, na=False)
        gdf = gdf[~mask] if water_use_exclude else gdf[mask]
    gdf = gdf.copy()

    well_start_year = derive_well_start_year(gdf, default_year=start_year)
    gdf['_count'] = 1.0

    tmp_dir = os.path.join(output_dir, f'_{output_prefix}_temp')
    makedirs(tmp_dir)
    tmp_shp = os.path.join(tmp_dir, 'wells.shp')

    logger.info(f'Creating {output_prefix} rasters ({start_year}-{end_year}) '
                f'from {len(gdf)} wells...')

    prev_raster = None
    prev_mask = np.zeros(len(gdf), dtype=bool)
    for year in range(start_year, end_year + 1):
        out_file = os.path.join(output_dir, f'{output_prefix}_{year}.tif')
        active = well_start_year <= year

        # Skip rasterization if the active set hasn't changed
        if np.array_equal(active, prev_mask) and prev_raster is not None:
            copy_file(prev_raster, out_file, verbose=False)
        else:
            gdf.loc[~active, '_count'] = 0.0
            gdf.loc[active, '_count'] = 1.0
            gdf[['_count', 'geometry']].to_file(tmp_shp)
            vops.shp2raster(
                tmp_shp, out_file,
                value_field='_count',
                xres=xres, yres=yres,
                add_value=True,
            )
            prev_mask = active.copy()
            prev_raster = out_file

    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f'Created {output_prefix} rasters ({start_year}-{end_year}) in {output_dir}')
    return base_raster


def create_irr_capacity_fraction_raster(
        well_registry_file: str,
        output_dir: str,
        xres: float = 2000,
        yres: float = 2000,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False,
) -> str:
    """Create per-year irrigation pump-capacity fraction rasters.

    For each 2 km pixel and each year, computes the fraction of total pump
    capacity (PUMPRATE, gal/min) that belongs to wells with 'IRRIGATION' in
    their ``WATER_USE`` attribute, considering only wells installed by that
    year.  The complement (1 − fraction) is the non-irrigation (M&I) share.

    This replaces the area-based ``annual_irr_fraction`` for the
    irr / non-irr volume split in ``partitionops.partition_predictions``,
    giving a physically meaningful withdrawal fraction (AZ is ~72 % ag by
    pump capacity, matching ADWR 2017 statewide estimates).

    Args:
        well_registry_file: Path to the ADWR Well Registry shapefile.
        output_dir: Output directory for the fraction rasters.
        xres: X-resolution in map units (default 2000 m = 2 km).
        yres: Y-resolution in map units (default 2000 m = 2 km).
        start_year: First year (default 1896).
        end_year: Last year (default 2099).
        already_created: If True, skip creation.

    Returns:
        str: Path to the base irrigation capacity fraction raster.
    """
    base_raster = os.path.join(output_dir, f'Irr_Capacity_Fraction_{start_year}.tif')
    if already_created:
        logger.info('Irr capacity fraction rasters already created, skipping...')
        return base_raster

    makedirs(output_dir)

    gdf = gpd.read_file(well_registry_file)

    # Filter to consumptive-use wells. Drop wells whose WATER_USE is
    # missing or contains any non-consumptive keyword.  Using keyword
    # matching instead of exact strings handles the combinatorial
    # comma-separated labels in the ADWR registry.
    pattern = '|'.join(NON_CONSUMPTIVE_USES)
    has_use = gdf['WATER_USE'].notna()
    is_non_consumptive = gdf['WATER_USE'].str.contains(pattern, na=False)
    gdf = gdf[has_use & ~is_non_consumptive].copy()

    # Impute missing PUMPRATE using per-WATER_USE median.
    # Only ~42 % of active wells have a reported pump rate; dropping
    # the rest would bias toward irrigation wells (62 % coverage vs 39 %
    # for domestic). Imputing by category preserves the physical reality
    # that ag pumps are larger (~900 gal/min median) while domestic are
    # smaller (~15 gal/min median).
    gdf['PUMPRATE'] = pd.to_numeric(gdf['PUMPRATE'], errors='coerce')
    has_rate = gdf['PUMPRATE'].notna() & (gdf['PUMPRATE'] > 0)
    n_with_rate = has_rate.sum()
    medians = gdf.loc[has_rate].groupby('WATER_USE')['PUMPRATE'].median()
    for use, med in medians.items():
        missing = (gdf['WATER_USE'] == use) & ~has_rate
        gdf.loc[missing, 'PUMPRATE'] = med
    # Fallback: categories with no reported rates → overall median
    overall_median = gdf.loc[has_rate, 'PUMPRATE'].median()
    still_missing = gdf['PUMPRATE'].isna() | (gdf['PUMPRATE'] <= 0)
    gdf.loc[still_missing, 'PUMPRATE'] = overall_median
    logger.info(
        f'Using {len(gdf)} active wells for irrigation capacity fraction '
        f'({n_with_rate} with reported PUMPRATE, '
        f'{len(gdf) - n_with_rate} imputed by WATER_USE median)'
    )

    well_start_year = derive_well_start_year(gdf, default_year=start_year)

    is_irr = gdf['WATER_USE'].str.contains('IRRIGATION', na=False)
    gdf['_irr_cap'] = 0.0
    gdf.loc[is_irr, '_irr_cap'] = gdf.loc[is_irr, 'PUMPRATE'].astype(float)
    gdf['_total_cap'] = gdf['PUMPRATE'].astype(float)

    tmp_dir = os.path.join(output_dir, '_irr_cap_temp')
    makedirs(tmp_dir)
    irr_shp = os.path.join(tmp_dir, 'irr_cap.shp')
    total_shp = os.path.join(tmp_dir, 'total_cap.shp')
    irr_raster = os.path.join(tmp_dir, 'irr_cap.tif')
    total_raster = os.path.join(tmp_dir, 'total_cap.tif')

    import rasterio as rio
    profile = None
    prev_mask = np.zeros(len(gdf), dtype=bool)
    prev_raster = None

    logger.info(f'Creating irr capacity fraction rasters ({start_year}-{end_year})...')

    for year in range(start_year, end_year + 1):
        out_file = os.path.join(output_dir, f'Irr_Capacity_Fraction_{year}.tif')
        active = well_start_year <= year

        # Skip rasterization if the active set hasn't changed
        if np.array_equal(active, prev_mask) and prev_raster is not None:
            copy_file(prev_raster, out_file, verbose=False)
            continue

        # Zero out inactive wells' contributions
        gdf.loc[~active, '_irr_cap'] = 0.0
        gdf.loc[~active, '_total_cap'] = 0.0
        active_irr = active & is_irr
        gdf.loc[active_irr, '_irr_cap'] = gdf.loc[active_irr, 'PUMPRATE'].astype(float)
        gdf.loc[active & ~is_irr, '_irr_cap'] = 0.0
        gdf.loc[active, '_total_cap'] = gdf.loc[active, 'PUMPRATE'].astype(float)

        # Rasterize irrigation and total capacity
        gdf[['_irr_cap', 'geometry']].to_file(irr_shp)
        vops.shp2raster(
            irr_shp, irr_raster,
            value_field='_irr_cap',
            xres=xres, yres=yres,
            add_value=True,
        )
        gdf[['_total_cap', 'geometry']].to_file(total_shp)
        vops.shp2raster(
            total_shp, total_raster,
            value_field='_total_cap',
            xres=xres, yres=yres,
            add_value=True,
        )

        with rio.open(irr_raster) as irr_src, rio.open(total_raster) as tot_src:
            irr_arr = irr_src.read(1)
            tot_arr = tot_src.read(1)
            if profile is None:
                profile = irr_src.profile.copy()
                profile.update(dtype='float32', nodata=np.nan)

        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(tot_arr > 0, irr_arr / tot_arr, 0.0)
        frac = np.clip(frac, 0.0, 1.0).astype(np.float32)

        with rio.open(out_file, 'w', **profile) as dst:
            dst.write(frac, 1)

        prev_mask = active.copy()
        prev_raster = out_file

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Log statewide summary for final year
    if profile is not None:
        with rio.open(prev_raster) as src:
            final_frac = src.read(1)
        valid = final_frac > 0
        if valid.sum() > 0:
            mean_frac = float(np.mean(final_frac[valid]))
            logger.info(
                f'Created irr capacity fraction rasters ({start_year}-{end_year}). '
                f'Final-year mean irr fraction = {mean_frac:.3f} '
                f'({valid.sum()} pixels with wells)'
            )
    return base_raster


# ---------------------------------------------------------------------------
# HarDWR water-rights raster functions
# ---------------------------------------------------------------------------

_POD_SHAPEFILE_COLS = ['primarySe0', 'source', 'priorityD0', 'AF', 'geometry']

# Non-irrigation sectors for the non-irr SW fraction (exclude environmental
# in-stream flow rights which represent legal river protections, not actual
# M&I surface water diversions).
_NONIRR_SW_SECTORS = {'domestic', 'industrial', 'livestock', 'other'}


def _load_pod_points(
        pod_shapefile: str,
        ref_raster: str,
        source_filter: str | None = None,
        sector_filter: str | set[str] | None = None,
        require_af: bool = False,
) -> tuple:
    """Load and project HarDWR PODs, returning arrays for rasterization.

    Args:
        pod_shapefile: Path to arizonaStatePOD.shp.
        ref_raster: Reference raster for CRS, transform, shape.
        source_filter: 'Surface Water' or 'Groundwater' (None = all).
        sector_filter: Single sector string or set of sectors to include.
        require_af: If True, drop records with AF <= 0.

    Returns:
        (rows, cols, priority_years, af_values, raster_shape, transform, profile)
        All arrays are for valid (in-bounds) PODs only.
    """
    import rasterio as rio

    gdf = gpd.read_file(pod_shapefile, include_fields=_POD_SHAPEFILE_COLS)
    if source_filter:
        gdf = gdf[gdf['source'] == source_filter]
    if sector_filter:
        if isinstance(sector_filter, str):
            gdf = gdf[gdf['primarySe0'] == sector_filter]
        else:
            gdf = gdf[gdf['primarySe0'].isin(sector_filter)]
    if require_af:
        gdf['AF'] = pd.to_numeric(gdf['AF'], errors='coerce')
        gdf = gdf[gdf['AF'].notna() & (gdf['AF'] > 0)]

    with rio.open(ref_raster) as src:
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
        profile = src.profile.copy()

    gdf = gdf.to_crs(crs)
    gdf['_year'] = pd.to_datetime(gdf['priorityD0'], errors='coerce').dt.year
    gdf = gdf[gdf['_year'].notna()].copy()
    gdf['_year'] = gdf['_year'].astype(int)

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    cols, rows = ~transform * (xs, ys)
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])

    af_vals = pd.to_numeric(gdf['AF'], errors='coerce').fillna(0).values
    return (rows[valid], cols[valid], gdf['_year'].values[valid],
            af_vals[valid], shape, transform, profile)


def create_sw_access_year_raster(
        pod_shapefile: str,
        output_dir: str,
        ref_raster: str,
        already_created: bool = False,
) -> str:
    """Create a raster of the earliest irrigation SW priority year per pixel.

    For each 2 km pixel, stores the minimum priority date year across all
    irrigation surface-water PODs in that pixel.  Used to temporally adjust
    the Hung et al. ``annual_gw_fraction`` — pixels whose SW access had
    not yet been established are set to ``gw_frac = 1.0`` at partition time.

    Args:
        pod_shapefile: Path to HarDWR arizonaStatePOD.shp.
        output_dir: Output directory for the raster.
        ref_raster: Reference raster for CRS/grid alignment.
        already_created: Skip if True.

    Returns:
        Path to SW_Access_Year.tif.
    """
    import rasterio as rio

    out_file = os.path.join(output_dir, 'SW_Access_Year.tif')
    if already_created:
        logger.info('SW access year raster already created, skipping...')
        return out_file

    makedirs(output_dir)
    rows, cols, years, _, shape, transform, profile = _load_pod_points(
        pod_shapefile, ref_raster,
        source_filter='Surface Water',
        sector_filter='irrigation',
    )
    logger.info(f'Creating SW access year raster from {len(rows)} '
                f'irrigation SW PODs...')

    # Compute min year per pixel
    grid = np.full(shape, np.nan, dtype=np.float32)
    for r, c, y in zip(rows, cols, years):
        if np.isnan(grid[r, c]) or y < grid[r, c]:
            grid[r, c] = y

    n_pixels = int(np.isfinite(grid).sum())
    profile.update(dtype='float32', count=1, nodata=np.nan)
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(grid, 1)

    logger.info(f'Created SW_Access_Year.tif: {n_pixels} pixels with '
                f'irrigation SW rights')
    return out_file


def create_irr_sw_rights_density_raster(
        pod_shapefile: str,
        output_dir: str,
        ref_raster: str,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False,
) -> str:
    """Create per-year cumulative irrigation SW rights count rasters.

    For each year and pixel, counts the number of irrigation surface-water
    PODs with ``priority_year ≤ year``.  This is an ML predictor capturing
    irrigation surface-water availability build-out over time.

    Args:
        pod_shapefile: Path to HarDWR arizonaStatePOD.shp.
        output_dir: Output directory.
        ref_raster: Reference raster for CRS/grid.
        start_year: First year (default 1896).
        end_year: Last year (default 2099).
        already_created: Skip if True.

    Returns:
        Path to the base raster (first year).
    """
    import rasterio as rio

    base_raster = os.path.join(output_dir,
                               f'Irr_SW_Rights_Density_{start_year}.tif')
    if already_created:
        logger.info('Irr SW rights density rasters already created, skipping...')
        return base_raster

    makedirs(output_dir)
    rows, cols, years, _, shape, transform, profile = _load_pod_points(
        pod_shapefile, ref_raster,
        source_filter='Surface Water',
        sector_filter='irrigation',
    )
    logger.info(f'Creating Irr SW rights density rasters ({start_year}-{end_year}) '
                f'from {len(rows)} irrigation SW PODs...')

    profile.update(dtype='float32', count=1, nodata=np.nan)
    pixel_keys = rows * shape[1] + cols

    prev_raster = None
    prev_count = -1
    for year in range(start_year, end_year + 1):
        out_file = os.path.join(output_dir,
                                f'Irr_SW_Rights_Density_{year}.tif')
        active = years <= year
        n_active = int(active.sum())

        if n_active == prev_count and prev_raster is not None:
            copy_file(prev_raster, out_file, verbose=False)
        else:
            grid = np.zeros(shape, dtype=np.float32)
            np.add.at(grid.ravel(), pixel_keys[active], 1)
            with rio.open(out_file, 'w', **profile) as dst:
                dst.write(grid, 1)
            prev_raster = out_file
            prev_count = n_active

    logger.info(f'Created Irr SW rights density rasters ({start_year}-{end_year})')
    return base_raster


def create_gw_allocation_density_raster(
        pod_shapefile: str,
        output_dir: str,
        ref_raster: str,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False,
) -> str:
    """Create per-year cumulative irrigation GW allocation depth rasters.

    For each year and pixel, sums AF allocations of irrigation groundwater
    rights with ``priority_year ≤ year`` and converts to mm depth using the
    pixel area (consistent with the pumping depth rasters).  This is an ML
    predictor representing legally permitted groundwater demand.

    Conversion: ``mm = AF × 1233.48 / pixel_area_m² × 1000``

    Args:
        pod_shapefile: Path to HarDWR arizonaStatePOD.shp.
        output_dir: Output directory.
        ref_raster: Reference raster for CRS/grid.
        start_year: First year (default 1896).
        end_year: Last year (default 2099).
        already_created: Skip if True.

    Returns:
        Path to the base raster (first year).
    """
    import rasterio as rio

    base_raster = os.path.join(output_dir,
                               f'GW_Allocation_Density_{start_year}.tif')
    if already_created:
        logger.info('GW allocation density rasters already created, skipping...')
        return base_raster

    makedirs(output_dir)
    rows, cols, years, af_vals, shape, transform, profile = _load_pod_points(
        pod_shapefile, ref_raster,
        source_filter='Groundwater',
        sector_filter='irrigation',
        require_af=True,
    )

    # Convert AF to mm: AF × 1233.48 m³/AF / pixel_area_m² × 1000 mm/m
    pixel_area_m2 = abs(transform.a * transform.e)
    af_to_mm = 1233.48 / pixel_area_m2 * 1000
    af_vals_mm = af_vals * af_to_mm

    logger.info(f'Creating GW allocation density rasters ({start_year}-{end_year}) '
                f'from {len(rows)} irrigation GW rights with AF > 0 '
                f'(total {af_vals.sum():,.0f} AF = '
                f'{af_vals_mm.sum():,.0f} mm·pixels)...')

    profile.update(dtype='float32', count=1, nodata=np.nan)
    pixel_keys = rows * shape[1] + cols

    prev_raster = None
    prev_count = -1
    for year in range(start_year, end_year + 1):
        out_file = os.path.join(output_dir,
                                f'GW_Allocation_Density_{year}.tif')
        active = years <= year
        n_active = int(active.sum())

        if n_active == prev_count and prev_raster is not None:
            copy_file(prev_raster, out_file, verbose=False)
        else:
            grid = np.zeros(shape, dtype=np.float32)
            np.add.at(grid.ravel(), pixel_keys[active], af_vals_mm[active])
            with rio.open(out_file, 'w', **profile) as dst:
                dst.write(grid, 1)
            prev_raster = out_file
            prev_count = n_active

    logger.info(f'Created GW allocation density rasters ({start_year}-{end_year})')
    return base_raster


def create_nonirr_sw_rights_density_raster(
        pod_shapefile: str,
        output_dir: str,
        ref_raster: str,
        start_year: int = 1896,
        end_year: int = 2099,
        already_created: bool = False,
) -> str:
    """Create per-year cumulative non-irrigation SW rights count rasters.

    For each year and pixel, counts non-irrigation surface-water PODs
    (domestic, industrial, livestock, other — excluding environmental
    in-stream flow rights) with ``priority_year ≤ year``.  Used in
    ``partitionops`` as a temporally varying proxy for the non-irrigation
    GW/SW split.

    Args:
        pod_shapefile: Path to HarDWR arizonaStatePOD.shp.
        output_dir: Output directory.
        ref_raster: Reference raster for CRS/grid.
        start_year: First year (default 1896).
        end_year: Last year (default 2099).
        already_created: Skip if True.

    Returns:
        Path to the base raster (first year).
    """
    import rasterio as rio

    base_raster = os.path.join(output_dir,
                               f'NonIrr_Irr_SW_Rights_Density_{start_year}.tif')
    if already_created:
        logger.info('Non-irr Irr SW rights density rasters already created, '
                    'skipping...')
        return base_raster

    makedirs(output_dir)
    rows, cols, years, _, shape, transform, profile = _load_pod_points(
        pod_shapefile, ref_raster,
        source_filter='Surface Water',
        sector_filter=_NONIRR_SW_SECTORS,
    )
    logger.info(f'Creating non-irr Irr SW rights density rasters '
                f'({start_year}-{end_year}) from {len(rows)} PODs '
                f'(domestic/industrial/livestock/other)...')

    profile.update(dtype='float32', count=1, nodata=np.nan)
    pixel_keys = rows * shape[1] + cols

    prev_raster = None
    prev_count = -1
    for year in range(start_year, end_year + 1):
        out_file = os.path.join(output_dir,
                                f'NonIrr_Irr_SW_Rights_Density_{year}.tif')
        active = years <= year
        n_active = int(active.sum())

        if n_active == prev_count and prev_raster is not None:
            copy_file(prev_raster, out_file, verbose=False)
        else:
            grid = np.zeros(shape, dtype=np.float32)
            np.add.at(grid.ravel(), pixel_keys[active], 1)
            with rio.open(out_file, 'w', **profile) as dst:
                dst.write(grid, 1)
            prev_raster = out_file
            prev_count = n_active

    logger.info(f'Created non-irr Irr SW rights density rasters '
                f'({start_year}-{end_year})')
    return base_raster


def create_wtd_raster(
        wtd_dir: str,
        az_boundary_file: str,
        output_dir: str,
        ref_raster: str,
        states: tuple[str, ...] = ('arizona', 'nevada', 'california'),
        already_created: bool = False,
) -> str:
    """Create a 2 km water table depth (WTD) raster for Arizona.

    Mosaics state-level WTD rasters from Ma et al. (2026), reprojects
    from Lambert Conformal Conic to match the reference raster grid
    (EPSG:26912, 2 km), and resamples using mean aggregation.  The
    reference raster's bounding box defines the spatial extent; the
    downstream reprojection step handles final alignment and masking.

    Args:
        wtd_dir: Directory containing ``wtd_{state}.tif`` files.
        az_boundary_file: Path to the reprojected AZ boundary file.
        output_dir: Output directory for the resampled WTD raster.
        ref_raster: Reference raster for target CRS, transform, and shape.
        states: Tuple of state names to mosaic (default: AZ, NV, CA
            to cover the full AZ bounding box at borders).
        already_created: If True, skip creation.

    Returns:
        str: Path to the output WTD raster.

    Reference:
        Ma, Y., Condon, L. E., Koch, J., Bennett, A., Defnet, A.,
        Tijerina-Kreuzer, D., Melchior, P., & Maxwell, R. M. (2026).
        High resolution US water table depth estimates reveal quantity
        of accessible groundwater. Communications Earth & Environment,
        7(1), 45. https://doi.org/10.1038/s43247-025-03094-3
    """
    import rasterio as rio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject

    out_file = os.path.join(output_dir, 'WTD.tif')
    if already_created and os.path.exists(out_file):
        logger.info('WTD raster already created, skipping...')
        return out_file

    makedirs(output_dir)

    # ---- Compute AZ bounding box in source CRS ----
    # Clip NV/CA rasters to AZ extent to avoid loading full state tiles.
    src_files = []
    for state in states:
        f = os.path.join(wtd_dir, f'wtd_{state}.tif')
        if os.path.exists(f):
            src_files.append(rio.open(f))
        else:
            logger.warning(f'WTD file not found: {f}')
    if not src_files:
        logger.warning('No WTD files found — skipping.')
        return out_file

    mosaic_crs = src_files[0].crs

    # Get AZ bounding box from the reference raster and transform
    # to the source CRS for spatial clipping during merge.
    from rasterio.warp import transform_bounds
    with rio.open(ref_raster) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
    az_bounds_src = transform_bounds(ref_crs, mosaic_crs,
                                     *ref_bounds)

    # Use nodata=np.nan to convert the source nodata sentinel
    # (float32 max ≈ 3.4e38) to NaN during merge, avoiding
    # float64→float32 precision issues that produce all-zero output.
    # bounds= clips to AZ bounding box, reducing memory.
    mosaic, mosaic_transform = merge(src_files, method='first',
                                     nodata=np.nan,
                                     bounds=az_bounds_src)
    for f in src_files:
        f.close()

    mosaic = mosaic[0].astype(np.float32)

    logger.info(f'Mosaicked {len(states)} WTD state rasters '
                f'(clipped to AZ bbox): shape={mosaic.shape}, '
                f'valid pixels={np.isfinite(mosaic).sum():,}')

    # ---- Reproject and resample to match reference raster ----
    with rio.open(ref_raster) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    dst_arr = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=mosaic,
        destination=dst_arr,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,  # mean aggregation
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    del mosaic

    # ---- Gap-fill NaN pixels at 2 km (nearest neighbor) ----
    # The Ma et al. WTD dataset covers only the US, so pixels near the
    # Mexico border have no data.  Fill from the nearest valid pixel.
    from scipy.ndimage import distance_transform_edt
    nan_mask = ~np.isfinite(dst_arr)
    n_valid = np.isfinite(dst_arr).sum()
    n_gaps = nan_mask.sum()
    if n_gaps > 0 and n_valid > 0:
        _, nearest_idx = distance_transform_edt(
            nan_mask, return_distances=True, return_indices=True,
        )
        dst_arr[nan_mask] = dst_arr[
            nearest_idx[0][nan_mask],
            nearest_idx[1][nan_mask],
        ]
        logger.info(f'  Gap-filled {n_gaps:,} NaN pixels at 2 km '
                    f'using nearest-neighbor interpolation')

    valid = np.isfinite(dst_arr)
    logger.info(f'WTD resampled to 2 km: shape={dst_shape}, '
                f'valid={valid.sum():,} pixels, '
                f'range=[{np.nanmin(dst_arr[valid]):.1f}, '
                f'{np.nanmax(dst_arr[valid]):.1f}] m')

    # ---- Write output ----
    with rio.open(ref_raster) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype='float32', nodata=np.nan)
    with rio.open(out_file, 'w', **profile) as dst:
        dst.write(dst_arr, 1)
        dst.set_band_description(1, 'water_table_depth_m')

    logger.info(f'WTD raster saved to {out_file}')
    return out_file
