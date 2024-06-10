"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import hydrolibs.gwops as gwops
import hydrolibs.dataops as dataops

if __name__ == '__main__':
    input_dir = '../Data/Inputs/'
    output_dir = '../Data/Outputs/'
    output_gw_vector_dir = f'{output_dir}GW/Vectors/'
    output_gw_volume_raster_dir = f'{output_dir}GW/Rasters/GW_Volumes/'
    output_gw_depth_raster_dir = f'{output_dir}GW/Rasters/GW_Depths/'
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
    xres = 500
    yres = 500
    fill_attr = 'AF Pumped'
    resampled_gee_mosaic_dir = f'{output_dir}GEE_Mosaics_{xres}m/'
    resampled_tile_dir = f'{output_dir}GEE_Tiles_{xres}m/'
    gee_mosaic_data_dir = f'{output_dir}GEE_Mosaics/'
    pred_data_dir = f'{output_dir}Predictor_Data/'
    irr_output_prefix = 'IRR'
    load_files = True

    gee_data_dir, data_band_names = dataops.download_gee_data(
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
    load_files = False
    dataops.resample_gee_tiles(
        gee_data_dir,
        data_band_names,
        resampled_tile_dir,
        xres,
        num_workers,
        load_files
    )
    dataops.mosaic_tiles(
        resampled_tile_dir,
        gee_mosaic_data_dir,
        start_year,
        end_year,
        already_mosaicked=load_files
    )
    irr_tile_dir = dataops.create_irrigation_tiles(
        gee_data_dir,
        output_dir,
        start_year,
        end_year,
        xres,
        output_prefix=irr_output_prefix,
        already_created=load_files
    )
    dataops.mosaic_tiles(
        irr_tile_dir,
        gee_mosaic_data_dir,
        start_year,
        end_year,
        output_prefix=irr_output_prefix,
        already_mosaicked=load_files
    )
    gwops.create_gw_volume_rasters(
        output_gw_vector_dir,
        output_gw_volume_raster_dir,
        value_field=fill_attr,
        xres=xres,
        yres=yres,
        already_created=load_files
    )
    gwops.create_gw_depth_rasters(
        output_gw_volume_raster_dir,
        output_gw_depth_raster_dir,
        already_created=load_files
    )
    gw_cropped_raster_dir = gwops.crop_gw_rasters(
        output_gw_depth_raster_dir,
        output_gw_depth_raster_dir,
        az_state_file=f'{vector_reproj_dir}AZ.geojson',
        already_cropped=load_files
    )
    dataops.reproject_gee_mosaics(
        gee_mosaic_data_dir,
        pred_data_dir,
        gw_cropped_raster_dir,
        already_reprojected=load_files
    )
