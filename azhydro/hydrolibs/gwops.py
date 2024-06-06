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

from sysops import makedirs
from glob import glob


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


def create_gw_depth_rasters(
        gw_volume_dir: str,
        irrigated_area_dir: str,
        outdir: str,
        gw_pattern: str = '*.tif',
        irr_prefix: str = 'IRR'
) -> None:
    """
    Create GW withdrawal depth rasters.

    Args:
        gw_volume_dir (str): Input GW volume (acreft) directory.
        irrigated_area_dir (str): The mosaicked annual irrigated area raster (km2) directory path.
        outdir (str): Output raster directory containing the GW depth rasters (mm).
        gw_pattern (str): File extension for the GW withdrawal rasters.
        irr_prefix (str): Prefix for the irrigation raster files.

    Returns:
        None.
    """

    irr_reproj_dir = f'{irrigated_area_dir}Irr_Reproj/'
    makedirs(irr_reproj_dir)
    for gw_volume_file in glob(gw_volume_dir + gw_pattern):
        year = gw_volume_file[gw_volume_file.rfind('_') + 1: gw_volume_file.rfind('.')]
        irr_file = glob(f'{irrigated_area_dir}{irr_prefix}*{year}.tif')[0]
        irr_reproj_file = f'{irr_reproj_dir}{irr_file[irr_file.rfind(os.sep) + 1:]}'
        rops.reproject_raster_gdal(
            irr_file,
            irr_reproj_file,
            from_raster=gw_volume_file
        )
        gw_depth_file = f'{outdir}{gw_volume_file[gw_volume_file.rfind(os.sep) + 1:]}'
        gw_vol_arr, gw_vol_ref = rops.read_raster_as_arr(gw_volume_file)
        irr_area_arr, _ = rops.read_raster_as_arr(irr_reproj_file)
        irr_area_arr[np.isnan(gw_vol_arr)] = np.nan
        gw_vol_arr[~np.isnan(gw_vol_arr)] *= 1.233 / irr_area_arr
        no_data = rops.az_nodata()
        gw_vol_arr[np.isnan(gw_vol_arr)] = no_data
        rops.write_raster(
            gw_vol_arr, gw_vol_ref, transform_=gw_vol_ref.transform,
            outfile_path=gw_depth_file, no_data_value=no_data
        )


def fix_gw_raster_values(
        input_raster_dir: str,
        outdir: str,
        max_threshold: float = 1e+5,
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
        if fix_only_negative:
            raster_arr[raster_arr < 0] = no_data
        else:
            raster_arr[raster_arr >= max_threshold] = no_data
        rops.write_raster(
            raster_arr, raster_file, transform_=raster_file.transform,
            outfile_path=out_raster, no_data_value=no_data
        )


def create_gw_rasters(
        input_gw_dir: str,
        output_gw_dir: str,
        irrigated_area_dir: str,
        xres: float = 1000.,
        yres: float = 1000.,
        value_field: str | None = None,
        value_field_pos: int = 0,
        irr_prefix: str = 'IRR',
        already_created: bool = True
) -> str:
    """
    Create GW rasters from shapefiles.

    Args:
        input_gw_dir (str): Input directory containing preprocessed GW files from #preprocess_gw_csv.
        output_gw_dir (str): Output directory.
        irrigated_area_dir (str): The mosaicked annual irrigated area raster directory path.
        xres (float): X-Resolution (map unit).
        yres (float): Y-Resolution (map unit).
        max_gw (float): Maximum GW pumping in mm. Any value higher than this will be set to no data.
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing).
        irr_prefix (str): Prefix for the irrigation raster files.
        already_created (bool): Set False to re-compute GW pumping rasters.

    Returns:
        str: Output raster directory path containing the GW withdrawal depth rasters (mm).
    """

    gw_depth_dir = f'{output_gw_dir}GW_Depths/'
    if not already_created:
        print('Creating GW withdrawal volume (acreft) rasters...')
        gw_volume_dir_uncorrected = f'{output_gw_dir}Uncorrected_GW_Volumes/'
        makedirs((gw_volume_dir_uncorrected, gw_depth_dir))
        vops.shps2rasters(
            input_gw_dir,
            gw_volume_dir_uncorrected,
            xres=xres, yres=yres,
            value_field=value_field,
            value_field_pos=value_field_pos
        )
        gw_volume_dir = f'{output_gw_dir}GW_Volumes/'
        fix_gw_raster_values(
            gw_volume_dir_uncorrected,
            gw_volume_dir,
            fix_only_negative=True
        )
        shutil.rmtree(gw_volume_dir_uncorrected, ignore_errors=True)
        print('Creating GW withdrawal depth (mm) rasters...')
        create_gw_depth_rasters(
            gw_volume_dir,
            gw_depth_dir,
            irrigated_area_dir,
            irr_prefix=irr_prefix
        )
    else:
        print('GW  pumping rasters already created')
    return gw_depth_dir




