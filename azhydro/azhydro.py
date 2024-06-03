"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import hydrolibs.gwops as gwops
from hydrolibs.dataops import download_gee_data

if __name__ == '__main__':
    input_dir = '../Data/Inputs/'
    output_dir = '../Data/Outputs/'
    output_gw_vector_dir = f'{output_dir}GW/Vectors/'
    output_gw_raster_dir = f'{output_dir}GW/Rasters/'
    vector_dir = f'{input_dir}GW_Data/'
    vector_reproj_dir = f'{output_dir}GW_Data/Vector_Reproj/'
    az_state = f'{vector_dir}/AZ.geojson'
    well_reg_file = f'{vector_dir}Well_Registry/Well_Registry.shp'
    az_gw_basin = f'{vector_dir}/Groundwater_Basin.geojson'
    ama_ina_file = f'{vector_dir}AMA_and_INA.geojson'
    az_canal = f'{vector_dir}Canals_NHD_Buffer.geojson'
    gw_csv_dir = f'{vector_dir}Meter Data/'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1985
    end_year = 2023
    skip_download = True
    tile_size = 10000
    num_workers = 32
    xres = 1000
    yres = 1000
    fill_attr = 'AF Pumped'
    load_files = False

    gee_data_dir = download_gee_data(
        az_state,
        gcloud_project,
        gcloud_bucket,
        input_dir,
        start_year,
        end_year,
        skip_download,
        tile_size,
        num_workers
    )

    ref_gw_file = gwops.preprocess_gw_csv(
        well_reg_file,
        gw_csv_dir,
        output_gw_vector_dir,
        fill_attr=fill_attr,
        use_only_ama_ina=False,
        already_preprocessed=load_files
    )
    gwops.reproject_vectors(
        vector_dir,
        vector_reproj_dir,
        ref_file=ref_gw_file,
        already_reprojected=load_files
    )

    gw_raster_dir = gwops.create_gw_rasters(
        output_gw_vector_dir,
        output_gw_raster_dir,
        already_created=load_files,
        value_field=fill_attr,
        xres=xres,
        yres=yres,
        max_gw=3000
    )
    output_gw_raster_dir = gwops.crop_gw_rasters(
        output_gw_raster_dir,
        output_gw_raster_dir,
        az_state_file=f'{vector_reproj_dir}AZ.geojson',
        already_cropped=load_files
    )