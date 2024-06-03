"""
Handle groundwater withdrawal processing codes.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import rasterops as rops
import vectorops as vops
import os

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
    output_gw_dir = f'{cropped_dir}Fixed/'
    if not already_cropped:
        makedirs(cropped_dir)
        rops.crop_rasters(
            input_gw_dir, outdir=cropped_dir,
            input_mask_file=az_state_file,
            ext_mask=True
        )
        makedirs(output_gw_dir)
        rops.fix_gw_raster_values(
            cropped_dir,
            outdir=output_gw_dir,
            fix_only_negative=True
        )
    else:
        print('GW rasters already cropped')
    return output_gw_dir


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


def create_gw_rasters(
        input_gw_dir: str,
        output_gw_dir: str,
        xres: float = 1000.,
        yres: float = 1000.,
        max_gw: float = 3000.,
        value_field: str | None = None,
        value_field_pos: int = 0,
        convert_units: bool = True,
        already_created: bool = True
) -> str:
    """
    Create GW rasters from shapefiles.

    Args:
        input_gw_dir (str): Input directory containing preprocessed GW files from #preprocess_gw_csv.
        output_gw_dir (str): Output directory.
        xres (float): X-Resolution (map unit).
        yres (float): Y-Resolution (map unit).
        max_gw (float): Maximum GW pumping in mm. Any value higher than this will be set to no data.
        value_field (str or None): Name of the value attribute. Set None to use value_field_pos.
        value_field_pos (int): Value field position (zero indexing).
        convert_units (bool): If true, converts GW pumping values in acreft to mm.
        already_created (bool): Set False to re-compute GW pumping rasters.

    Returns:
        str: Output raster directory path.
    """

    fixed_dir = f'{output_gw_dir}Fixed/'
    converted_dir = f'{output_gw_dir}Converted/'
    if convert_units:
        final_gw_dir = converted_dir
    else:
        final_gw_dir = fixed_dir
    if not already_created:
        print('Converting SHP to TIF...')
        makedirs(fixed_dir)
        vops.shps2rasters(
            input_gw_dir,
            output_gw_dir,
            xres=xres, yres=yres,
            value_field=value_field,
            value_field_pos=value_field_pos
        )
        if convert_units:
            max_gw *= xres * yres / 1.233e+6
        rops.fix_gw_raster_values(
            output_gw_dir,
            max_threshold=max_gw,
            outdir=fixed_dir
        )
        final_gw_dir = fixed_dir
        if convert_units:
            print('Changing GW units from acreft to mm')
            makedirs(converted_dir)
            rops.convert_gw_data(fixed_dir, converted_dir)
            final_gw_dir = converted_dir
    else:
        print('GW  pumping rasters already created')
    return final_gw_dir
