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
    az_canal = f'{vector_dir}Canals/canals_az.shp'
    gw_csv_dir = f'{vector_dir}Meter Data/'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1985
    end_year = 2023
    skip_download = True
    tile_size = 10000
    num_workers = 32
    tile_raster_res = 500
    fill_attr = 'AF Pumped'
    resampled_tile_dir = f'{output_dir}GEE_Tiles_{tile_raster_res}m/'
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
    dataops.resample_gee_rasters(
        gee_data_dir,
        data_band_names,
        resampled_tile_dir,
        target_raster_res=tile_raster_res,
        num_workers=num_workers,
        already_resampled=load_files
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
        tile_raster_res,
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
    mosaic_raster_res = 1000
    resampled_gee_mosaic_dir = f'{output_dir}GEE_Mosaics_{mosaic_raster_res}m/'
    dataops.resample_gee_rasters(
        gee_mosaic_data_dir,
        data_band_names,
        resampled_gee_mosaic_dir,
        original_raster_res=tile_raster_res,
        target_raster_res=mosaic_raster_res,
        already_resampled=load_files,
        use_tile_format=False
    )
    gwops.create_gw_volume_rasters(
        output_gw_vector_dir,
        output_gw_volume_raster_dir,
        value_field=fill_attr,
        xres=mosaic_raster_res,
        yres=mosaic_raster_res,
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
    load_files = True
    gwops.reproject_vectors(
        f'{vector_dir}Canals/',
        vector_reproj_dir,
        ref_file=ref_gw_file,
        pattern='*.shp',
        already_reprojected=load_files
    )
    gwops.reproject_vectors(
        vector_dir,
        vector_reproj_dir,
        ref_file=ref_gw_file,
        already_reprojected=load_files
    )
    gw_basin_proj = f'{vector_reproj_dir}Groundwater_Basin.geojson'
    az_canal_proj = f'{vector_reproj_dir}canals_az.shp'
    gwops.create_gw_basin_streamflow_rasters(
        gw_basin_proj,
        az_canal_proj,
        resampled_gee_mosaic_dir,
        mosaic_raster_res,
        mosaic_raster_res,
        start_year,
        end_year,
        load_files
    )
    dataops.reproject_gee_mosaics(
        resampled_gee_mosaic_dir,
        pred_data_dir,
        gw_cropped_raster_dir,
        already_reprojected=load_files
    )
    load_files = False
    az_csv = dataops.create_az_data_csv(
        pred_data_dir,
        gw_cropped_raster_dir,
        output_dir,
        data_band_names,
        gw_basin_proj,
        start_year,
        end_year,
        load_csv=load_files
    )
