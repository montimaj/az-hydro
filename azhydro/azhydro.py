"""
Main driver file for running the project.
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import hydrolibs.gwops as gwops
import hydrolibs.dataops as dataops
import hydrolibs.mlops as mlops
import hydrolibs.visualops as vizops  # Journal-quality visualization module


if __name__ == '__main__':
    input_dir = '../Data/Inputs/'
    output_dir = '../Data/Outputs/'
    water_use = 'IRRIGATION'  # 'IRRIGATION' or 'All'. 'All' was used in Majumdar et al. (2022)
    wname = 'All_Wells' if water_use == 'All' else 'Irr_Wells'
    output_gw_vector_dir = f'{output_dir}GW/Vectors/{wname}/'
    vector_dir = f'{input_dir}GW_Data/'
    vector_reproj_dir = f'{output_dir}GW_Data/Vector_Reproj/'
    az_state = f'{vector_dir}/AZ.geojson'
    well_reg_file = f'{vector_dir}Well_Registry_2024/Well_Registry.shp'
    az_gw_basin = f'{vector_dir}/Groundwater_Basin/Groundwater_Basin.shp'
    az_gw_subbasin = f'{vector_dir}/ADWR_Groundwater_Subbasin/ADWR_Groundwater_Subbasin.shp'
    ama_ina_file = f'{vector_dir}AMA_and_INA.geojson'
    az_canal = f'{vector_dir}Canals/canals_az.shp'
    gw_csv_dir = f'{vector_dir}Meter Data/'
    az_vectors = [
        well_reg_file, az_gw_basin, 
        ama_ina_file, az_canal,
        az_gw_subbasin, vector_dir
    ]
    cap_delivery_xls = f'{vector_dir}CAP/CAP Delivery Data DRI Request.xlsx'
    srp_delivery_xls = f'{vector_dir}SRP/SRP WATER DELVS HISTORY.xlsx'
    gcloud_project = 'azhydro'
    gcloud_bucket = 'azhydro'
    start_year = 1896
    end_year = 2099
    skip_download = False
    tile_raster_res = 2000 # this is the same as Majumdar et al. (2022)
    tile_size = 10000 if tile_raster_res == 30 else 80000
    fill_attr = 'AF Pumped'
    mosaic_raster_res = tile_raster_res
    gee_mosaic_data_dir = f'{output_dir}GEE_Mosaics_{int(mosaic_raster_res)}m/'
    gee_resampled_tile_dir = f'{output_dir}GEE_Tiles_{int(mosaic_raster_res)}m/'
    output_gw_volume_raster_dir = f'{output_dir}GW/Rasters/GW_Volumes_{wname}_{int(mosaic_raster_res)}m/'
    output_gw_depth_raster_dir = f'{output_dir}GW/Rasters/GW_Depths_{wname}_{int(mosaic_raster_res)}m/'
    pred_data_dir = f'{output_dir}Predictor_Data_{wname}_{int(mosaic_raster_res)}m/'
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
        num_workers=40,
        worker_memory='1G',
        gee_scale=tile_raster_res,
        verbose=False
    )
    
    # dataops.mosaic_tiles(
    #     gee_data_dir,
    #     gee_mosaic_data_dir,
    #     start_year,
    #     end_year,
    #     already_mosaicked=load_files
    # )
    # ref_gw_file = gwops.preprocess_gw_csv(
    #     well_reg_file,
    #     gw_csv_dir,
    #     output_gw_vector_dir,
    #     fill_attr=fill_attr,
    #     use_only_ama_ina=False,
    #     already_preprocessed=load_files,
    #     water_use=water_use
    # )
    # for az_vector in az_vectors:
    #     gwops.reproject_vectors(
    #         az_vector,
    #         vector_reproj_dir,
    #         ref_file=ref_gw_file,
    #         already_reprojected=load_files
    #     )
    # max_gw = 3000 if water_use == 'All' else None # 3000 mm (~10 ft) was used in Majumdar et al. (2022)
    # gwops.create_gw_volume_rasters(
    #     output_gw_vector_dir,
    #     output_gw_volume_raster_dir,
    #     value_field=fill_attr,
    #     xres=mosaic_raster_res,
    #     yres=mosaic_raster_res,
    #     already_created=load_files,
    #     max_gw = max_gw 
    # )
    # gwops.create_gw_depth_rasters(
    #     output_gw_volume_raster_dir,
    #     output_gw_depth_raster_dir,
    #     already_created=load_files
    # )
    # gw_cropped_raster_dir = gwops.crop_gw_rasters(
    #     output_gw_depth_raster_dir,
    #     output_gw_depth_raster_dir,
    #     az_state_file=f'{vector_reproj_dir}AZ.geojson',
    #     already_cropped=load_files
    # )
    # gw_basin_proj = f'{vector_reproj_dir}Groundwater_Basin.shp'
    # az_canal_proj = f'{vector_reproj_dir}canals_az.shp'
    # gwops.create_gw_basin_streamflow_rasters(
    #     gw_basin_proj,
    #     az_canal_proj,
    #     gee_mosaic_data_dir,
    #     mosaic_raster_res,
    #     mosaic_raster_res,
    #     start_year,
    #     end_year,
    #     water_year_agg=False,
    #     already_created=load_files
    # )
    # dataops.reproject_gee_mosaics(
    #     gee_mosaic_data_dir,
    #     pred_data_dir,
    #     gw_cropped_raster_dir,
    #     already_reprojected=load_files
    # )
    # model_dir = f'{output_dir}ML_Model_{wname}_{int(mosaic_raster_res)}m/'
    # # load_files = False
    # az_df = dataops.create_az_data_csv(
    #     pred_data_dir,
    #     gw_cropped_raster_dir,
    #     model_dir,
    #     data_band_names,
    #     gw_basin_proj,
    #     start_year,
    #     end_year,
    #     load_csv=load_files,
    #     lu_smoothing=3,
    # )
    # model_eval_dict = {
    #     'T1': ((2015, 2024),),
    #     # 'T2': ((1990, 1992), (2005, 2007), (2022, 2024)),
    #     # 'T3': ((2007, 2010),),
    #     # 'T4': ((1985, 1989), (2020, 2024)),
    #     # 'T5': ((2024, 2024),),
    #     # 'T6': ((2010, 2020),) # Test data in Majumdar et al. (2022)
    # }
    # drop_attrs = (
    #     'Year',
    #     'GW_Basin',
    #     # 'AGRI',
    #     # 'URBAN',
    #     # 'northing_m',
    #     'easting_m',
    #     'SW',
    #     'GW_Basin_Type',
    #     # 'annual_peff_mm',
    #     # 'annual_et_ensemble_mm',
    #     # 'annual_irrmapper_fraction',
    #     # 'annual_prism_precip_mm',
    #     'annual_prism_tmmx_K',
    #     'annual_prism_tmmn_K',
    #     'annual_era5land_swe_m',
    #     'annual_era5land_runoff_m',
    #     # 'annual_era5land_volumetric_soil_water_layer_1',
    #     'annual_era5land_volumetric_soil_water_layer_2',
    #     'annual_era5land_volumetric_soil_water_layer_3',
    #     'annual_era5land_volumetric_soil_water_layer_4',
    #     # 'streamflow_m3s',
    #     # 'soil_depth_cm',
    #     # 'awc_mm',
    #     # 'ksat_mean_micromps',
    #     # 'cap_srp_delivery_km3'
    # )
    # use_ama_ina = True
    # test_gw_basins = (
    #     'HARQUAHALA INA',
    #     # 'SANTA CRUZ AMA',
    #     # 'DOUGLAS AMA',
    #     # 'PINAL AMA',
    #     # 'TUCSON AMA',
    #     # 'PRESCOTT AMA',
    # )
    # outlier_op = None
    # random_state = 42
    # load_model = False
    # fold_count = 5
    # repeats = 1
    # split_strategy = 1 # 1: temporal, 2: random stratified based on year_col, 3: spatial, 4: random
    # randomized_search = True
    # load_files = False
    # perm_imp = False
    # show_ale = True
    # # there are only a few pixels in these basins for irrigation water use
    # drop_gw_basins = ('WILLCOX AMA', 'JOSEPH CITY INA', 'HUALAPAI VALLEY INA') if water_use == 'IRRIGATION' else \
    #     ('WILLCOX AMA', 'HUALAPAI VALLEY INA')
    # use_optuna = True
    # use_dask = True  # Enable Dask parallelization for Optuna
    # n_dask_workers = 10  # Number of Dask workers
    # use_model_comparison = True  # Run full model comparison
    # include_all_models = False  # Include additional models (GBR, AdaBoost, Bagging, CatBoost)
    # generate_visualizations = True  # Generate journal-quality visualizations
    # create_basin_plots = True  # Create individual basin-level plots (slower but more comprehensive)
    
    # # Get available models
    # ml_models = mlops.get_model_param_dict(get_model_names_only=True, include_all_models=include_all_models)
    # alpha = 1
    # year_list = list(range(1984, 2025))
    # n_trials = 100  # Number of Optuna trials
    
    # for test_case in model_eval_dict.keys():
    #     print(f'\n{"="*60}')
    #     print(f'Running test case {test_case}...')
    #     print(f'{"="*60}')
        
    #     test_year_limits = model_eval_dict[test_case]
    #     test_years = []
    #     for test_year_limit in test_year_limits:
    #         test_years.extend(list(range(test_year_limit[0], test_year_limit[1] + 1)))
    #     test_years = tuple(test_years)
        
    #     model_eval_base_dir = f'{model_dir}Model_Evaluation/{test_case}/'
        
    #     # Create train/test data once per test case
    #     model_eval_dir = f'{model_eval_base_dir}data/'
    #     ret_vals = dataops.create_train_test_data(
    #         az_df, model_eval_dir,
    #         drop_attr=drop_attrs,
    #         random_state=random_state,
    #         scaling=False, already_created=load_files,
    #         year_list=year_list, split_strategy=split_strategy,
    #         test_year=test_years, outlier_op=outlier_op,
    #         test_gw_basins=test_gw_basins,
    #         use_ama_ina=use_ama_ina,
    #         drop_gw_basins=drop_gw_basins,
    #         water_use=water_use
    #     )
    #     x_train, x_test, y_train, y_test, x_scaler, y_scaler, year_train, year_test, basin_train, basin_test = ret_vals
        
    #     # Option 1: Run full model comparison (includes all models and ensembles)
    #     if use_model_comparison:
    #         comparison_dir = f'{model_eval_base_dir}Model_Comparison/'
    #         bias_corr_type = 1 if split_strategy == 3 else 2
    #         comparison_df = mlops.compare_all_models(
    #             x_train, x_test, y_train, y_test,
    #             comparison_dir,
    #             model_names=ml_models,
    #             random_state=random_state,
    #             use_optuna=use_optuna,
    #             n_trials=n_trials,
    #             n_dask_workers=n_dask_workers,
    #             use_dask=use_dask,
    #             year_train=year_train,
    #             year_test=year_test,
    #             basin_train=basin_train,
    #             basin_test=basin_test,
    #             x_scaler=x_scaler,
    #             y_scaler=y_scaler,
    #             use_ama_ina=use_ama_ina,
    #             apply_bias_correction=bias_corr_type
    #         )
    #         print(f'\nModel Comparison Results for {test_case}:')
    #         print(comparison_df.to_string(index=False))
        
    #     # Option 2: Train individual models as before
    #     else:
    #         for ml_model in ml_models:
    #             print(f'\nRunning test case {test_case} with model {ml_model}...')
    #             model_eval_dir = f'{model_eval_base_dir}{ml_model}/'
                
    #             if not use_optuna:
    #                 model, cv_metric_df = mlops.build_ml_model(
    #                     x_train, y_train, model_eval_dir,
    #                     ml_model, random_state,
    #                     load_model, fold_count,
    #                     repeats, randomized_search,
    #                     tune_params=True
    #                 )
    #             else:
    #                 # Use enhanced Optuna with Dask parallelization
    #                 model, cv_metric_df = mlops.build_ml_model_optuna_dask(
    #                     x_train, y_train, model_eval_dir,
    #                     ml_model, random_state,
    #                     load_model, fold_count, repeats,
    #                     n_trials=n_trials, alpha=alpha,
    #                     n_dask_workers=n_dask_workers,
    #                     use_dask=use_dask,
    #                     pruning=True
    #                 )
                
    #             bias_corr_type = 1 if split_strategy == 3 else 2
    #             pred_df = mlops.get_prediction_results(
    #                 model, x_train, x_test,
    #                 y_train, y_test, x_scaler,
    #                 y_scaler, year_train,
    #                 year_test, basin_train,
    #                 basin_test, model_eval_dir,
    #                 ml_model,
    #                 apply_bias_correction=bias_corr_type,
    #             )
    #             mlops.calc_train_test_metrics(
    #                 pred_df,
    #                 cv_metric_df,
    #                 model_eval_dir,
    #                 use_ama_ina=use_ama_ina,
    #                 model_name=ml_model,
    #             )
    #             if perm_imp:
    #                 mlops.compute_perm_imp(
    #                     model_name=ml_model,
    #                     x_train=x_train,
    #                     x_test=x_test,
    #                     y_train=y_train,
    #                     y_test=y_test,
    #                     model=model,
    #                     output_dir=model_eval_dir,
    #                     scoring_metric='scaled_rmse',
    #                     random_state=random_state,
    #                     create_plots=True
    #                 )
    #             if show_ale:
    #                 mlops.compute_ale_plots(
    #                     model_name=ml_model,
    #                     x_train=x_train,
    #                     x_test=x_test,
    #                     y_train=y_train,
    #                     y_test=y_test,
    #                     model=model,
    #                     output_dir=f'{model_eval_dir}/ALE/'
    #                 )
                
    #             # Generate journal-quality visualizations
    #             if generate_visualizations:
    #                 viz_dir = f'{model_eval_dir}Visualizations/'
    #                 mlops.generate_model_visualizations(
    #                     pred_df=pred_df,
    #                     output_dir=viz_dir,
    #                     model_name=ml_model,
    #                     test_case=test_case,
    #                     test_year_limits=test_year_limits,
    #                     raster_res=mosaic_raster_res,
    #                     use_ama_ina=use_ama_ina,
    #                     create_basin_plots=create_basin_plots
    #                 )
    
    # # Optional: Time series plots with journal-quality formatting
    # # Uncomment below to generate additional aggregate visualizations
    # # 
    # # Example: Create visualizations from saved prediction parquet files
    # # import pandas as pd
    # # pred_df = pd.read_parquet(f'{model_eval_base_dir}{ml_model}/Predictions_{ml_model}.parquet')
    # # viz_dir = f'{model_eval_base_dir}{ml_model}/Journal_Plots/'
    # # mlops.generate_model_visualizations(
    # #     pred_df=pred_df,
    # #     output_dir=viz_dir,
    # #     model_name=ml_model,
    # #     test_case=test_case,
    # #     test_year_limits=test_year_limits,
    # #     raster_res=mosaic_raster_res,
    # #     use_ama_ina=use_ama_ina,
    # #     create_basin_plots=True
    # # )
    # #
    # # Example: Compare multiple models in a single plot
    # # model_predictions = {
    # #     'XGB': pd.read_parquet(f'{model_eval_base_dir}XGB/Predictions_XGB.parquet'),
    # #     'LGBM': pd.read_parquet(f'{model_eval_base_dir}LGBM/Predictions_LGBM.parquet'),
    # #     'RF': pd.read_parquet(f'{model_eval_base_dir}RF/Predictions_RF.parquet')
    # # }
    # # mlops.generate_model_comparison_visualization(
    # #     model_predictions=model_predictions,
    # #     output_dir=f'{model_eval_base_dir}Comparison_Plots/',
    # #     test_case=test_case,
    # #     test_year_limits=test_year_limits,
    # #     raster_res=mosaic_raster_res,
    # #     use_ama_ina=use_ama_ina,
    # #     unit='af'
    # # )
    
    # print('\n' + '='*60)
    # print('Processing complete!')
    # print('='*60)

