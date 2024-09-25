"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import hydrolibs.gwops as gwops
import hydrolibs.dataops as dataops
import hydrolibs.mlops as mlops

if __name__ == '__main__':
    input_dir = '../Data/Inputs/'
    output_dir = '../Data/Outputs/'
    output_gw_vector_dir = f'{output_dir}GW/Vectors/'
    vector_dir = f'{input_dir}GW_Data/'
    vector_reproj_dir = f'{output_dir}GW_Data/Vector_Reproj/'
    az_state = f'{vector_dir}/AZ.geojson'
    well_reg_file = f'{vector_dir}Well_Registry/Well_Registry.shp'
    az_gw_basin = f'{vector_dir}/Groundwater_Basin.geojson'
    ama_ina_file = f'{vector_dir}AMA_and_INA.geojson'
    az_canal = f'{vector_dir}Canals/canals_az.shp'
    gw_csv_dir = f'{vector_dir}Meter Data/'
    cap_delivery_xls = f'{vector_dir}CAP/CAP Delivery Data DRI Request.xlsx'
    srp_delivery_xls = f'{vector_dir}SRP/SRP WATER DELVS HISTORY.xlsx'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1985
    end_year = 2023
    skip_download = True
    tile_size = 10000
    tile_raster_res = 30
    fill_attr = 'AF Pumped'
    mosaic_raster_res = 2000
    gee_mosaic_data_dir = f'{output_dir}GEE_Mosaics_{mosaic_raster_res}m/'
    gee_resampled_tile_dir = f'{output_dir}GEE_Tiles_{mosaic_raster_res}m/'
    output_gw_volume_raster_dir = f'{output_dir}GW/Rasters/GW_Volumes_{mosaic_raster_res}m/'
    output_gw_depth_raster_dir = f'{output_dir}GW/Rasters/GW_Depths_{mosaic_raster_res}m/'
    pred_data_dir = f'{output_dir}Predictor_Data_{mosaic_raster_res}m/'
    irr_output_prefix = 'IRR'
    load_files = True

    gee_data_dir_30m, data_band_names = dataops.download_gee_data(
        az_state,
        gcloud_project,
        gcloud_bucket,
        input_dir,
        start_year,
        end_year,
        skip_download,
        tile_size,
        num_workers=32,
        gee_scale=30,
        irrigated_tiles=True
    )

    dataops.resample_gee_rasters(
        gee_data_dir_30m,
        data_band_names,
        gee_resampled_tile_dir,
        original_raster_res=tile_raster_res,
        target_raster_res=mosaic_raster_res,
        already_resampled=load_files,
        use_tile_format=True,
        irr_data_only=False
    )
    dataops.mosaic_tiles(
        gee_resampled_tile_dir,
        gee_mosaic_data_dir,
        start_year,
        end_year,
        already_mosaicked=load_files
    )
    irr_tile_dir = dataops.create_irrigation_tiles(
        gee_data_dir_30m,
        output_dir,
        start_year,
        end_year,
        mosaic_raster_res,
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
    ref_gw_file = gwops.preprocess_gw_csv(
        well_reg_file,
        gw_csv_dir,
        output_gw_vector_dir,
        fill_attr=fill_attr,
        use_only_ama_ina=False,
        already_preprocessed=load_files
    )
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
    gw_basin_proj = f'{vector_reproj_dir}Groundwater_Basin.geojson'
    az_canal_proj = f'{vector_reproj_dir}canals_az.shp'
    gwops.create_gw_basin_streamflow_rasters(
        gw_basin_proj,
        az_canal_proj,
        gee_mosaic_data_dir,
        mosaic_raster_res,
        mosaic_raster_res,
        start_year,
        end_year,
        load_files
    )
    gwops.create_gw_basin_sw_delivery_rasters(
        gw_basin_proj,
        cap_delivery_xls,
        srp_delivery_xls,
        gee_mosaic_data_dir,
        mosaic_raster_res,
        mosaic_raster_res,
        start_year,
        end_year,
        load_files
    )
    dataops.reproject_gee_mosaics(
        gee_mosaic_data_dir,
        pred_data_dir,
        gw_cropped_raster_dir,
        already_reprojected=load_files
    )
    az_df = dataops.create_az_data_csv(
        pred_data_dir,
        gw_cropped_raster_dir,
        output_dir,
        data_band_names,
        gw_basin_proj,
        start_year,
        end_year,
        load_csv=load_files,
        lu_smoothing=3,
    )
    year_list = list(range(1985, 2024))
    test_years = tuple(range(2014, 2024))
    # test_years = (1990,)
    model_dir = f'{output_dir}ML_Model/'
    ml_model = 'RF'
    random_state = 42
    load_model = False
    fold_count = 5
    repeats = 1
    split_strategy = 3 # 1: temporal, 2: random stratified based on year_col, 3: spatial, 4: random
    randomized_search = False
    load_files = False

    drop_attrs = (
        'Year',
        'GW_Basin',
        # 'AGRI',
        # 'annual_et_ensemble_mm',
        'annual_et_ssebop_mm',
        'annual_et_sims_mm',
        'annual_et_pt_jpl_mm',
        'annual_et_eemetric_mm',
        'annual_gridmet_precip_mm',
        'annual_gridmet_tmmx_K',
        'annual_et_geesebal_mm',
        'annual_et_disalexi_mm',
        'annual_gridmet_tmmn_K',
        'annual_gridmet_eto_mm',
        'annual_gridmet_etr_mm',
        'annual_gridmet_bc_eto_mm',
        'annual_gridmet_bc_etr_mm',
        'annual_gridmet_vpd_kPa',
        'annual_gridmet_vs_mps',
        'annual_gridmet_rmax',
        'annual_gridmet_rmin',
        'annual_gridmet_spi1y',
        'annual_gridmet_eddi1y',
        'annual_gridmet_spei1y',
        'annual_gridmet_pdsi',
        #'annual_prism_precip_mm',
        'annual_prism_tmmx_K',
        'annual_prism_tmmn_K',
        'annual_conus404_precip_mm',
        'annual_conus404_tmmx_K',
        'annual_conus404_tmmn_K',
        'annual_conus404_eto_mm',
        'annual_conus404_etr_mm',
        'annual_terraclimate_sm_change_mm',
        'annual_terraclimate_ro_mm',
        'HSG',
        'annual_daymet_precip_mm',
        'annual_daymet_tmmn_K',
        'annual_daymet_tmmx_K',
        # 'soil_depth_mm',
        # 'ksat_mean_micromps',
        # 'elevation_m',
        # 'slope'
        # 'cap_srp_delivery_m3'
    )
    ret_vals = dataops.create_train_test_data(
        az_df, output_dir,
        drop_attr=drop_attrs,
        random_state=random_state,
        scaling=False, already_created=load_files,
        year_list=year_list, split_strategy=split_strategy,
        test_year=test_years, outlier_op=None,
        test_gw_basins=('HARQUAHALA INA',)
    )
    x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test, basin_train, basin_test = ret_vals
    model = mlops.build_ml_model(
        x_train, y_train, model_dir,
        ml_model, random_state,
        load_model, fold_count,
        repeats, y_scaler,
        randomized_search
    )
    pred_df = mlops.get_prediction_results(
        model, x_train, x_test,
        y_train, y_test, x_scaler,
        y_scaler, year_train,
        year_test, basin_train,
        basin_test, model_dir,
        ml_model
    )
    mlops.calc_train_test_metrics(
        pred_df, use_ama_ina=True
    )
    plot_dir_dict = {
        1: f'{output_dir}Plots/Temporal/',
        2: f'{output_dir}Plots/Random_Stratified/',
        3: f'{output_dir}Plots/Spatial/',
        4: f'{output_dir}Plots/Random/'
    }
    plot_dir = plot_dir_dict[split_strategy]
    gwops.make_time_series_plots(pred_df, plot_dir)
