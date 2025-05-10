"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import warnings
warnings.filterwarnings("ignore")
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
    monthly_eff_precip_dir = f'{vector_dir}Effective_precip_prediction_WestUS/v19_monthly_scaled/'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1985
    end_year = 2023
    skip_download = True
    tile_raster_res = 2000
    tile_size = 10000 if tile_raster_res == 30 else 80000
    fill_attr = 'AF Pumped'
    mosaic_raster_res = tile_raster_res
    gee_mosaic_data_dir = f'{output_dir}GEE_Mosaics_{mosaic_raster_res}m/'
    gee_resampled_tile_dir = f'{output_dir}GEE_Tiles_{mosaic_raster_res}m/'
    output_gw_volume_raster_dir = f'{output_dir}GW/Rasters/GW_Volumes_{mosaic_raster_res}m/'
    output_gw_depth_raster_dir = f'{output_dir}GW/Rasters/GW_Depths_{mosaic_raster_res}m/'
    pred_data_dir = f'{output_dir}Predictor_Data_{mosaic_raster_res}m/'
    load_files = True
    multiply_irr_mask = False
    gee_data_dir, data_band_names = dataops.download_gee_data(
        az_state,
        gcloud_project,
        gcloud_bucket,
        input_dir,
        start_year,
        end_year,
        skip_download,
        tile_size,
        num_workers=32,
        worker_memory='1.5G',
        gee_scale=tile_raster_res,
        irrigated_tiles=True,
        multiply_irr_mask=multiply_irr_mask
    )
    dataops.mosaic_tiles(
        gee_data_dir,
        gee_mosaic_data_dir,
        start_year,
        end_year,
        already_mosaicked=load_files
    )
    load_files = True
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
    load_files = True
    gwops.create_gw_volume_rasters(
        output_gw_vector_dir,
        output_gw_volume_raster_dir,
        value_field=fill_attr,
        xres=mosaic_raster_res,
        yres=mosaic_raster_res,
        already_created=load_files,
        max_gw=3000, # capped at ~10 ft like Majumdar et al. 2022
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
        water_year_agg=False,
        already_created=load_files
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
    gwops.create_annual_eff_precip_rasters(
        monthly_eff_precip_dir,
        gee_mosaic_data_dir,
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
    # test_year_limits = ((1990, 1992), (2005, 2007), (2021, 2023))
    test_year_limits = ((2014, 2023),)
    # test_year_limits = ((1985, 1989), (2019, 2023))
    test_years = []
    for test_year_limit in test_year_limits:
        test_years.extend(list(range(test_year_limit[0], test_year_limit[1] + 1)))
    test_years = tuple(test_years)
    model_dir = f'{output_dir}ML_Model_{mosaic_raster_res}m/'
    ml_model = 'XGB'
    random_state = 42
    load_model = False
    fold_count = 5
    repeats = 1
    split_strategy = 3 # 1: temporal, 2: random stratified based on year_col, 3: spatial, 4: random
    randomized_search = True
    load_files = False
    perm_imp = True

    drop_attrs = (
        'Year',
        'GW_Basin',
        # 'AGRI',
        # 'URBAN',
        'lon_deg',
        'lat_deg',
        # 'SW',
        'GW_Basin_Type',
        # 'annual_peff_mm',
        # 'annual_et_ensemble_mm',
        # 'annual_irrmapper_fraction',
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
        'annual_gridmet_vpd_kPa',
        'annual_gridmet_vs_mps',
        'annual_gridmet_rmax',
        'annual_gridmet_rmin',
        'annual_gridmet_spi1y',
        'annual_gridmet_eddi1y',
        'annual_gridmet_spei1y',
        'annual_gridmet_pdsi',
        # 'annual_prism_precip_mm',
        'annual_prism_tmmx_K',
        'annual_prism_tmmn_K',
        'annual_terraclimate_sm_change_mm',
        'annual_terraclimate_ro_mm',
        'HSG',
        'annual_daymet_precip_mm',
        'annual_daymet_tmmn_K',
        'annual_daymet_tmmx_K',
        'wy_era5land_snowfall_m',
        'annual_era5land_swe_m',
        'annual_era5land_snowmelt_m',
        'annual_era5land_runoff_m',
        'annual_era5land_sub_surface_runoff_m',
        'annual_era5land_surface_runoff_m',
        # 'annual_era5land_volumetric_soil_water_layer_1',
        'annual_era5land_volumetric_soil_water_layer_2',
        'annual_era5land_volumetric_soil_water_layer_3',
        'annual_era5land_volumetric_soil_water_layer_4',
        'annual_npp_mm4',
        # 'streamflow_m3s',
        'soil_depth_mm',
        'ksat_mean_micromps',
        # 'elevation_m',
        # 'slope',
        # 'cap_srp_delivery_m3',
        'bulk_density_gcm3',
        'clay_percent',
        'ksat_log10cmhr1',
        'pore_size_dist',
        'organic_matter_log10percent',
        'soil_ph',
        'sand_percent',
        'silt_percent',
        'residual_swc',
        'saturated_swc',
        'pore_size_index',
        'soil_bubbling_pressure_log10kPa',
        'polaris_scale_log10kPa1',
    )
    use_ama_ina = True
    test_gw_basins = (
        'HARQUAHALA INA',
        'SANTA CRUZ AMA',
        # 'DOUGLAS AMA_INA',
        'JOSEPH CITY INA',
        # 'PINAL AMA',
        # 'TUCSON AMA',
        # 'PRESCOTT AMA',
    )
    outlier_op = None
    ret_vals = dataops.create_train_test_data(
        az_df, output_dir,
        drop_attr=drop_attrs,
        random_state=random_state,
        scaling=False, already_created=load_files,
        year_list=year_list, split_strategy=split_strategy,
        test_year=test_years, outlier_op=outlier_op,
        test_gw_basins=test_gw_basins,
        use_ama_ina=use_ama_ina
    )
    x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test, basin_train, basin_test = ret_vals
    model = mlops.build_ml_model(
        x_train, y_train, model_dir,
        ml_model, random_state,
        load_model, fold_count,
        repeats, y_scaler,
        randomized_search,
        tune_params=True
    )
    bias_corr_type = 1 if split_strategy == 3 else 2
    pred_df = mlops.get_prediction_results(
        model, x_train, x_test,
        y_train, y_test, x_scaler,
        y_scaler, year_train,
        year_test, basin_train,
        basin_test, model_dir,
        ml_model,
        apply_bias_correction=bias_corr_type
    )
    mlops.calc_train_test_metrics(
        pred_df,
        use_ama_ina=use_ama_ina
    )
    if perm_imp:
        mlops.compute_perm_imp(
            model_name=ml_model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            model=model,
            y_scaler=y_scaler,
            output_dir=model_dir,
            scoring_metric='normalized_rmse',
            random_state=random_state,
            create_plots=True
        )
    plot_dir_path = f'{output_dir}Plots_{mosaic_raster_res}m/'
    plot_dir_dict = {
        1: f'{plot_dir_path}Temporal/',
        2: f'{plot_dir_path}Random_Stratified/',
        3: f'{plot_dir_path}Spatial/',
        4: f'{plot_dir_path}Random/'
    }
    plot_dir = plot_dir_dict[split_strategy]
    az_df = az_df[az_df.Year.isin(year_list)]
    gwops.make_time_series_plots(
        az_df, model,
        x_train.columns.tolist(),
        plot_dir,
        test_year_limits=test_year_limits,
        split_strategy=split_strategy,
        test_gw_basins=test_gw_basins,
        raster_res=mosaic_raster_res
    )
