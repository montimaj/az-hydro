"""
Provides methods for machine learning (ML) operations required.
"""
import os

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu


import pandas as pd
import numpy as np
import pickle
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import skexplain

from typing import Any
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGCV
from dask_ml.model_selection import RandomizedSearchCV as DaskRCV
from dask_jobqueue import SLURMCluster
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, make_scorer
from sklearn.inspection import permutation_importance
from sysops import makedirs, make_proper_dir_name
from gwops import get_ama_ina_basin_names


def get_model_param_dict(
        random_state: int = 0,
        use_dask: bool = False,
        get_model_names_only: bool = False
) -> list[str] | tuple[dict[str, Any], dict[str, dict[str, list]]]:
    """Get model object dictionaries and parameter dictionary for different models.

    Args:
        random_state (int): Random state (seed) for some ML algorithms.
        use_dask (bool): Set True if using Dask in a distributed computing environment.
        get_model_names_only (bool): Set True to return only model names.

    Returns:
        Either a list of model names (if get_model_names_only is True) or a tuple of
        dict (str, Any) : Dictionary of the model objects.
        dict (str, dict (str, list)): Dictionary of models containing dictionary of the corresponding
                                      hyperparameters.
    """
    n_jobs = -2
    if use_dask:
        n_jobs = 1
    model_dict = {
        'XGB': XGBRegressor(
            n_jobs=-2,
            seed=random_state,
        ),
        'XGBRF': XGBRFRegressor(
            n_jobs=-2,
            seed=random_state,
        ),
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1, n_estimators=300, max_depth=16, num_leaves=31
        ), 
        'RF': RandomForestRegressor(
            n_jobs=-2, oob_score=False,
            n_estimators=300, max_features=None,
            random_state=random_state, max_depth=None
        ),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs, bootstrap=True),
        'HGBR': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1,
            max_depth=None, random_state=random_state,
            
        )
    }
    if get_model_names_only:
        return list(model_dict.keys())

    param_dict = {'XGB': {
        'eta': [0.01],
        'max_depth': [0],
        'grow_policy': ['depthwise', 'lossguide'],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'colsample_bynode': [0.8, 0.9, 1],
        'colsample_bylevel': [0.8, 0.9, 1],
        'reg_lambda': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'gamma': [0, 0.1, 0.5, 1],
        'num_parallel_tree': [1],
        'min_child_weight': [80, 90, 100],
        'n_estimators': [300, 400, 500],
    }, 'XGBRF': {
        'eta': [0.01],
        'max_depth': [0],
        'grow_policy': ['depthwise', 'lossguide'],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'colsample_bynode': [0.8, 0.9, 1],
        'colsample_bylevel': [0.8, 0.9, 1],
        'reg_lambda': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'gamma': [0, 0.1, 0.5, 1],
        'num_parallel_tree': [300, 400, 500],
        'min_child_weight': [30, 40]
    }, 'LGBM': {
        'n_estimators': [300, 400, 500],
        'max_depth': [16, 20, -1],
        'learning_rate': [0.01, 0.05],
        'subsample': [1, 0.9],
        'colsample_bytree': [1, 0.9],
        'colsample_bynode': [1, 0.9],
        'path_smooth': [0.1, 0.2],
        'num_leaves': [31, 32],
        'min_child_samples': [30, 40]
    }, 'RF': {
        'n_estimators': [300, 400, 500],
        'max_features': [None, 10, 8],
        'max_depth': [None],
        'max_leaf_nodes': [None],
        'max_samples': [None],
        'min_samples_leaf': [1, 2, 3]
    }, 'ETR': {
        'n_estimators': [300, 400, 500],
        'max_features': [None, 10, 8],
        'max_depth': [None],
        'max_leaf_nodes': [None],
        'max_samples': [None],
        'min_samples_leaf': [1, 2, 3]
    }, 'HGBR': {
        'max_iter': [300, 400, 500],
        'max_depth': [None, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_leaf_nodes': [31, 63, 127],
        'max_bins': [127, 255],
        'l2_regularization': [0.0, 0.1, 0.5],
    }
    }
    return model_dict, param_dict


def adjusted_r2(y: np.array, y_pred: np.array, p: int) -> float:
  """
  Calculates the adjusted R-squared.

  Args:
    y (np.array): Actual values.
    y_pred (np.array): Predicted values.
    p: Number of predictors.

  Returns:
    Adjusted R-squared value.
  """
  r2 = r2_score(y, y_pred)
  n = y.size
  if n == p + 1:
      return np.nan
  adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
  return adj_r2


def normalized_rmse(
        y: np.array,
        y_pred: np.array
) -> float:
    """
    Normalized RMSE using mean.

    Args:
        y (np.array): Actual values.
        y_pred (np.array): Predicted values.

    Returns:
        float: Normalized RMSE using mean.
    """

    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    nrmse = root_mean_squared_error(y, y_pred) * 100 / mean_y
    return nrmse


def normalized_mae(
        y: np.array,
        y_pred: np.array
) -> float:
    """
    Normalized MAE using mean.

    Args:
        y (np.array): Actual values.
        y_pred (np.array): Predicted values.

    Returns:
        float: Normalized MAE using standard mean.
    """
    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    nmae = mean_absolute_error(y, y_pred) * 100 / mean_y
    return nmae


def normalized_mbe(
        y: np.array,
        y_pred: np.array
) -> float:
    """
    Normalized MBE using mean.

    Args:
        y (np.array): Actual values.
        y_pred (np.array): Predicted values.

    Returns:
        float: Normalized MAE using standard mean.
    """
    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    nmbe = np.mean(y - y_pred) * 100 / mean_y
    return nmbe


def round_to_n_nonzero(
        value: float,
        n: int = 2
) -> float:
    """
    Round a value to n significant decimal places after the first non-zero digit.
    
    For values >= 1, rounds to n decimal places.
    For values < 1, finds the first non-zero decimal digit and rounds to show
    n significant digits from that position.

    Args:
        value (float): The value to round.
        n (int): Number of significant digits to keep after the first non-zero decimal.

    Returns:
        float: The rounded value.
    """
    if value == 0 or np.isnan(value) or np.isinf(value):
        return value
    
    abs_val = abs(value)
    
    if abs_val >= 1:
        return round(value, n)
    
    # Find the position of first non-zero digit after decimal
    decimal_pos = -int(np.floor(np.log10(abs_val)))
    # Round to n digits starting from first non-zero
    precision = decimal_pos + n - 1
    return round(value, precision)


def get_feature_dict(get_units: bool = False) -> dict[str, str] | tuple[dict[str, str], dict[str, str]]:
    """
    Get feature name dictionary for better visualization.

    Returns:
        dict (str, str): Dictionary mapping original feature names to descriptive names if get_units is False.
        tuple (dict (str, str), dict (str, str)): Tuple containing two dictionaries, the first mapping original feature 
        names to descriptive names and the second mapping original feature names to their units if get_units is True.
    """
    feature_dict = {
        'Year': 'Year',
        'GW_Basin': 'Groundwater Basin',
        'AGRI': 'Agricultural Density',
        'URBAN': 'Urban Density',
        'easting_m': 'Easting',
        'northing_m': 'Northing',
        'SW': 'Surface Water Density',
        'GW_Basin_Type': 'Groundwater Basin Type',
        'annual_peff_mm': 'Effective Precipitation',
        'annual_et_ensemble_mm': 'Actual ET',
        'annual_irrmapper_fraction': 'Irrigation Fraction',
        'annual_prism_precip_mm': 'Precipitation',
        'annual_prism_tmmx_K': '$T_{max}$',
        'annual_prism_tmmn_K': '$T_{min}$',
        'annual_era5land_swe_m': 'SWE',
        'annual_era5land_runoff_m': 'Runoff',
        'annual_era5land_volumetric_soil_water_layer_1': 'Volumetric Soil Water',
        'annual_era5land_volumetric_soil_water_layer_2': 'Volumetric Soil Water L2',
        'annual_era5land_volumetric_soil_water_layer_3': 'Volumetric Soil Water L3',
        'annual_era5land_volumetric_soil_water_layer_4': 'Volumetric Soil Water L4',
        'streamflow_m3s': 'Streamflow',
        'soil_depth_cm': 'Soil Depth',
        'awc_mm': 'Available Water Capacity',
        'ksat_mean_micromps': '$K_{sat}$',
        'cap_srp_delivery_km3': 'Surface Water Delivery'
    }

    feature_dict_units = {
        'easting_m': 'm',
        'northing_m': 'm',
        'annual_peff_mm': 'mm',
        'annual_et_ensemble_mm': 'mm',
        'annual_prism_precip_mm': 'mm',
        'annual_prism_tmmx_K': 'K',
        'annual_prism_tmmn_K': 'K',
        'annual_era5land_swe_m': 'm',
        'annual_era5land_runoff_m': 'm',
        'annual_era5land_volumetric_soil_water_layer_1': 'm$^3$/m$^3$',
        'annual_era5land_volumetric_soil_water_layer_2': 'm$^3$/m$^3$',
        'annual_era5land_volumetric_soil_water_layer_3': 'm$^3$/m$^3$',
        'annual_era5land_volumetric_soil_water_layer_4': 'm$^3$/m$^3$',
        'streamflow_m3s': 'm$^3$/s',
        'soil_depth_cm': 'cm',
        'awc_mm': 'mm',
        'ksat_mean_micromps': '$\mu$m/s',
        'cap_srp_delivery_km3': 'km$^3$'
    }
    return feature_dict if not get_units else (feature_dict, feature_dict_units)


def compute_perm_imp(
        model_name: str,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: np.array,
        y_test: np.array,
        model: Any,
        output_dir: str,
        scoring_metric: str,
        random_state: int,
        create_plots: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Compute permutation importances.

    Args:
        model_name (str): Name of the ML model. Has to be one of 'XGB', 'XGBRF', 'RF', 'ETR', 'LGBM', or 'HGBR.'
        x_train (pd.DataFrame): Training dataframe containing the predictor data.
        x_test (pd.DataFrame): Test dataframe containing the predictor data.
        y_train (np.array): Training labels containing the observed streamflow.
        y_test (np.array): Test labels containing the observed streamflow.
        model (Any): Fitted model object.
        output_dir (str): Output directory.
        scoring_metric (str): Name of the scoring metric. Has to be one of 'r2', 'normalized_rmse', 'normalized_mae.'
        random_state (int): Random seed.
        create_plots (bool): Set True to create permutation importance plots.

    Returns:
        Tuple of training and test importance dataframes or
        None if model_name is not one of 'RF', 'ETR', 'LGBM', or 'DRF.'
    """
    if model_name in ['RF', 'ETR', 'LGBM', 'XGB', 'XGBRF', 'HGBR']:
        print('Computing permutation importance...')
        scoring_metrics = {
            'r2': 'r2',
            'scaled_rmse': make_scorer(normalized_rmse, greater_is_better=False),
            'scaled_mae': make_scorer(normalized_mae, greater_is_better=False),
            'scaled_mbe': make_scorer(normalized_mbe, greater_is_better=False)
        }
        feature_dict = get_feature_dict()
        if create_plots and model_name != 'HGBR':
            imp_dict = {'Features': list(x_train.columns)}
            f_imp = np.array(model.feature_importances_).astype(float)
            if model_name == 'LGBM':
                f_imp /= np.sum(f_imp)
            imp_dict['F_IMP'] = np.round(f_imp, 5)
            imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
            imp_df['Features'] = imp_df['Features'].replace(feature_dict)
            plt.rcParams.update({'font.size': 30})
            plt.figure(figsize=(30, 15))
            sns.barplot(
                data=imp_df,
                y='Features',
                x='F_IMP'
            )
            plt.xlabel(f'{model_name} Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{output_dir}F_IMP_{model_name}.png', dpi=300)
            imp_df.to_csv(f'{output_dir}F_IMP_{model_name}.csv', index=False)
        perm_scorer = scoring_metrics[scoring_metric]
        train_result = permutation_importance(
            model, x_train.to_numpy(), y_train, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=perm_scorer
        )
        test_results = permutation_importance(
            model, x_test.to_numpy(), y_test, n_repeats=10, random_state=random_state, n_jobs=-1, scoring=perm_scorer
        )
        sorted_importances_idx = train_result.importances_mean.argsort()
        train_importances = pd.DataFrame(
            train_result.importances[sorted_importances_idx].T,
            columns=x_train.columns[sorted_importances_idx],
        )
        sorted_importances_idx = test_results.importances_mean.argsort()
        test_importances = pd.DataFrame(
            test_results.importances[sorted_importances_idx].T,
            columns=x_train.columns[sorted_importances_idx],
        )
        train_importances = train_importances.rename(columns=feature_dict)
        test_importances = test_importances.rename(columns=feature_dict)
        if create_plots:
            for name, importances in zip(["train", "test"], [train_importances, test_importances]):
                plt.figure(figsize=(10, 6))
                plt.rcParams.update({'font.size': 12})
                ax = importances.plot.box(vert=False, whis=10)
                ax.set_xlabel(f"Increase in {scoring_metric.split('_')[1].upper()} (%)")
                ax.axvline(x=0, color="k", linestyle="--")
                ax.figure.tight_layout()
                plt.savefig(f'{output_dir}{model_name}_{name}_PI.png', dpi=300)
                plt.clf()
        return train_importances, test_importances
    return None


def compute_ale_plots(
        model_name: str,
        model: Any,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        output_dir: str
) -> None:
    """
    Create accumulated local effects (ALE) plots.

    Args:
        model_name (str): Name of the ML model. Has to be one of 'XGB', 'XGBRF', 'RF', 'ETR', 'LGBM', or 'HGBR.'
        model (Any): Fitted model object.
        x_train (pd.DataFrame): Training dataframe containing the predictor data.
        y_train (np.ndarray): Training array containing the target data.
        x_test (pd.DataFrame): Test dataframe containing the predictor data.
        y_test (np.ndarray): Test array containing the target data.
        output_dir (str): Output directory.
        random_state (int): Random state for reproducibility.

    Returns:
        None
    """

    makedirs(output_dir)
    feature_dict, feature_dict_units = get_feature_dict(get_units=True)
    feature_names = x_train.columns.tolist()
    feature_2d_names = [
        ('annual_irrmapper_fraction', 'annual_peff_mm'),
        ('annual_prism_precip_mm', 'annual_et_ensemble_mm'),
        ('annual_irrmapper_fraction', 'soil_depth_cm'),
        ('annual_irrmapper_fraction', 'awc_cm'),
        ('annual_prism_precip_mm', 'soil_depth_cm'),
        ('annual_prism_precip_mm', 'awc_cm'),
        ('annual_et_ensemble_mm', 'soil_depth_cm'),
        ('annual_irrmapper_fraction', 'ksat_mean_micromps'),
        ('annual_irrmapper_fraction', 'cap_srp_delivery_km3'),
        ('annual_irrmapper_fraction', 'streamflow_m3s'),
        ('annual_irrmapper_fraction', 'annual_era5land_volumetric_soil_water_layer_1'),
        ('annual_era5land_volumetric_soil_water_layer_1', 'soil_depth_cm'),
        ('AGRI', 'annual_peff_mm'),
        ('URBAN', 'annual_peff_mm'),
        ('AGRI', 'streamflow_m3s'),
        ('URBAN', 'streamflow_m3s'),
        ('AGRI', 'cap_srp_delivery_km3'),
        ('URBAN', 'cap_srp_delivery_km3')
    ]
    data_dict = {
        'train': (x_train, y_train),
        'test': (x_test, y_test)
    }
    for data_type, (x_data, y_data) in data_dict.items():
        print(f'Creating ALE plots for {data_type} data...')        
        explainer = skexplain.ExplainToolkit(
            (model_name, model), 
            X=x_data, y=y_data, 
            feature_names=feature_names
        )
        ale_1d_reg = explainer.ale(
            features=feature_names,
            n_bootstrap=10,
            subsample=10000,
            n_jobs=len(feature_names),
            n_bins=30
        )
        _, ax = explainer.plot_ale(
            ale_1d_reg,
            display_feature_names=feature_dict,
            display_units=feature_dict_units
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}{model_name}_ALE_{data_type}.png', dpi=600)
        plt.clf()
        # ale_var_1d_reg = explainer.ale_variance(ale_1d_reg)
        # explainer.plot_importance(
        #     data=ale_var_1d_reg,
        #     panels=[('ale_variance', model_name)],
        #     num_vars_to_plot=len(feature_names),
        #     display_feature_names=feature_dict,
        #     plot_correlated_features=True
        # )
        # plt.tight_layout()
        # plt.savefig(f'{output_dir}{model_name}_ALE_Var_{data_type}.png', dpi=600)
        # plt.clf()

        # ale_2d_ds = explainer.ale(
        #     features=feature_2d_names,
        #     n_bootstrap=10,
        #     subsample=1.0,
        #     n_jobs=len(feature_2d_names),
        #     n_bins=30
        # )
        # explainer.plot_ale(
        #     ale=ale_2d_ds,
        #     display_feature_names=feature_dict,
        #     display_units=feature_dict_units
        # )
        # plt.tight_layout()
        # plt.savefig(f'{output_dir}{model_name}_ALE_2D_{data_type}.png', dpi=600)
        # plt.clf()



def build_ml_model(
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.array,
        model_dir: str,
        model_name: str = 'LGBM',
        random_state: int = 42,
        load_model: bool = False,
        fold_count: int = 5,
        repeats: int = 3,
        randomized_search: bool = False,
        stratified_kfold: bool = False,
        use_dask: bool = False,
        tune_params: bool = True,
        **kwargs: Any
) -> tuple[Any, pd.DataFrame]:
    """Build an ML model.

    Args:
        x_train (np.ndarray or pd.DataFrame): X_train numpy array or pandas dataframe.
        y_train (np.array): y_train numpy array.
        model_dir (str): Model directory to store/load model.
        model_name (str): ML model name as per the model_dict keys. 
        Has to be one of 'XGB', 'XGBRF', 'RF', 'ETR', 'LGBM', or 'HGBR'. Default is 'LGBM'.
        random_state (int): Random state (seed) for some ML algorithms.
        load_model (bool): Set model name to load existing model.
        fold_count (int): Number of folds for KFold.
        repeats (int): Number of repeats for KFold.
        randomized_search (bool): Set True to use the more computationally efficient RandomizedSearchCV.
        stratified_kfold (bool): Set True to use RepeatedStratifiedKFold based on the crop type.
        use_dask (bool): Flag for using dask.
        tune_params (bool): Set True to tune hyperparameters.
        kwargs (dict (str, str)): Pass the 'year_train' Pandas dataframe if stratified_kfold is True.

    Returns:
        tuple[Any, pd.DataFrame]: Trained model object and dataframe containing CV stats.
    """
    model_file = model_dir + model_name
    metric_csv = f'{model_dir}CV_Metrics_{model_name}.csv'
    if not load_model:
        dask_client = None
        cv_lib = 'sklearn'
        if use_dask:
            cluster = SLURMCluster(
                cores=32,
                processes=1,
                memory="10G",
                walltime="00:30:00",
                env_extra=['#SBATCH --out=Foundry-Dask-%j.out']
            )
            cluster.adapt(
                minimum=10, maximum=50,
                minimum_jobs=10, maximum_jobs=50,
                minimum_memory='8G', maximum_memory='10G'
            )
            dask_client = Client(cluster)
            print('Waiting for dask workers...')
            dask_client.wait_for_workers(1)
            cv_lib = 'dask_ml'
        model_dict, param_dict = get_model_param_dict(random_state, use_dask)
        if not tune_params:
            param_dict = {model_name: {}}
        model = model_dict[model_name]
        cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        if stratified_kfold:
            stratify_labels = kwargs['stratify_labels'].to_numpy().ravel()
            cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
            cv = cv.split(x_train, stratify_labels)
        makedirs(make_proper_dir_name(model_dir))
        print('\nSearching best params for {}...'.format(model_name))
        scoring_metrics = {
            'r2': 'r2',
            'adjusted_r2': make_scorer(adjusted_r2, p=x_train.shape[1], greater_is_better=True),
            'normalized_rmse': make_scorer(normalized_rmse, greater_is_better=False),
            'normalized_mae': make_scorer(normalized_mae, greater_is_better=False),
            'normalized_mbe': make_scorer(normalized_mbe, greater_is_better=False)
        }
        main_scorer = 'normalized_rmse'
        cv_func_dict = {
            'dask_ml': {1: DaskRCV, 0: DaskGCV},
            'sklearn': {1: RandomizedSearchCV, 0: GridSearchCV}
        }
        cv_func = cv_func_dict[cv_lib][int(randomized_search)]
        if randomized_search:
            model_grid = cv_func(
                estimator=model, param_distributions=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=main_scorer,
                return_train_score=True, random_state=random_state
            )
        else:
            model_grid = cv_func(
                estimator=model, param_grid=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=main_scorer,
                return_train_score=True
            )
        model_grid.fit(x_train, y_train)
        metric_df = get_grid_search_stats(model_grid, metric_csv)
        print('Best params: ', model_grid.best_params_)
        model = model_dict[model_name]
        model.set_params(**model_grid.best_params_)
        model.fit(x_train, y_train)
        pickle.dump(model, open(model_file, mode='wb+'))
        if dask_client:
            dask_client.close()
    else:
        model = pickle.load(open(model_file, mode='rb'))
        metric_df = pd.read_csv(metric_csv)
    return model, metric_df


def objective_with_cv(
        trial: Any, 
        x_train: np.ndarray | pd.DataFrame, 
        y_train: np.ndarray,
        model_name: str,
        cv: Any,
        scoring_metrics: dict[str, Any],
        alpha: float = 0.1,
        random_state: int = 42
) -> float: 
    """
    Objective function for Optuna hyperparameter tuning with cross-validation.
    
    Args:
        trial (Any): Optuna trial object.
        x_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.
        model_name (str): Name of the ML model. Has to be one of 'XGB', 'XGBRF', 'RF', 'ETR', 'LGBM', or 'HGBR.'
        cv (Any): Cross-validation strategy.
        scoring_metrics (dict (str, Any)): Scoring metrics for cross-validation.
        alpha (float): Weighting factor for combining training and validation scores. Default is 0.1.
        random_state (int): Random state (seed) for some ML algorithms. Default is 42.

    Returns:
        float: Mean negative normalized RMSE across cross-validation folds.
    """
    model = get_model_param_dict(random_state)[0][model_name]
    if model_name == 'XGB':
        params = {
            'eta': trial.suggest_float('eta', 0.01, 0.1),
            'max_depth': trial.suggest_categorical('max_depth', [0, 16, 20]),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100, step=10),
            'n_estimators': trial.suggest_int('n_estimators', 300, 600, step=100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
    elif model_name == 'XGBRF':
        params = {
            'eta': trial.suggest_float('eta', 0.01, 0.1),
            'max_depth': trial.suggest_categorical('max_depth', [0, 16, 20, 32, 64]),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100, step=10),
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 300, 600, step=100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
    elif model_name == 'LGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 600),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 16, 20, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'path_smooth': trial.suggest_float('path_smooth', 0.1, 0.5),
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 32, 63, 127]),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, step=10)
        }
    elif model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 600),
            'max_features': trial.suggest_categorical('max_features', [None, 10, 8, 15]),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20, 32, 64]),
            'max_samples': trial.suggest_categorical('max_samples', [None, 0.8, 0.9, 1.0]),
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 31, 63, 128]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
    elif model_name == 'ETR':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 600),
            'max_features': trial.suggest_categorical('max_features', [None, 10, 8, 15]),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20, 32, 64]),
            'max_samples': trial.suggest_categorical('max_samples', [None, 0.8, 0.9, 1.0]),
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 31, 63, 128]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
    elif model_name == 'HGBR':
        params = {
            'max_iter': trial.suggest_int('max_iter', 300, 600),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20, 32, 64]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 128),
            'max_bins': trial.suggest_categorical('max_bins', [31, 63, 127, 255]),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.set_params(**params)    
    cv_results = cross_validate(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_metrics,
        return_train_score=True
    )
    for data in ['train', 'test']:
        r2_key = f'{data}_r2'
        adj_r2_key = f'{data}_adjusted_r2'
        neg_rmse_key = f'{data}_normalized_rmse'
        neg_mae_key = f'{data}_normalized_mae'
        mbe_key = f'{data}_normalized_mbe'
        r2 = cv_results[r2_key].mean()
        adj_r2 = cv_results[adj_r2_key].mean()
        rmse = -cv_results[neg_rmse_key].mean()
        if data == 'test':
            test_mean_rmse = rmse
            test_std_rmse = abs(cv_results[neg_rmse_key].std())
        if data == 'train':
            train_mean_rmse = rmse
        mae = -cv_results[neg_mae_key].mean()
        mbe = cv_results[mbe_key].mean()
        trial.set_user_attr(r2_key, r2)
        trial.set_user_attr(adj_r2_key, adj_r2)
        trial.set_user_attr(neg_rmse_key, rmse)
        trial.set_user_attr(neg_mae_key, mae)
        trial.set_user_attr(mbe_key, mbe)
    beta = 0.5 * alpha
    trial_obj = test_mean_rmse + alpha * abs(train_mean_rmse - test_mean_rmse) + beta * test_std_rmse
    return trial_obj


def build_ml_model_optuna(
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.array,
        model_dir: str,
        model_name: str = 'LGBM',
        random_state: int = 42,
        load_model: bool = False,
        fold_count: int = 5,
        repeats: int = 3,
        stratified_kfold: bool = False,
        n_trials: int = 100,
        alpha: float = 0.1,
        **kwargs: Any
) -> tuple[Any, pd.DataFrame]:
    """
    Build an ML model using Optuna for hyperparameter tuning.

    Args:
        x_train (np.ndarray or pd.DataFrame): X_train numpy array or pandas dataframe.
        y_train (np.array): y_train numpy array.
        model_dir (str): Model directory to store/load model.
        model_name (str): ML model name as per the model_dict keys. 
        Has to be one of 'XGB', 'XGBRF', 'RF', 'ETR', 'LGBM', 'HGBR'. Default is 'LGBM'.
        random_state (int): Random state (seed) for some ML algorithms.
        load_model (bool): Set model name to load existing model.
        fold_count (int): Number of folds for KFold.
        repeats (int): Number of repeats for KFold.
        stratified_kfold (bool): Set True to use RepeatedStratifiedKFold based on the crop type.
        n_trials (int): Number of Optuna trials for hyperparameter tuning. Default is 100.
        alpha (float): Weighting factor for combining training and validation scores. Default is 0.1.
        kwargs (dict (str, str)): Pass the 'year_train' Pandas dataframe if stratified_kfold is True.

    Returns:
        tuple[Any, pd.DataFrame]: Trained model object and dataframe containing CV stats.
    """

    if not load_model:
        scoring_metrics = {
            'r2': 'r2',
            'adjusted_r2': make_scorer(adjusted_r2, p=x_train.shape[1], greater_is_better=True),
            'normalized_rmse': make_scorer(normalized_rmse, greater_is_better=False),
            'normalized_mae': make_scorer(normalized_mae, greater_is_better=False),
            'normalized_mbe': make_scorer(normalized_mbe, greater_is_better=False)
        }
        cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        if stratified_kfold:
            stratify_labels = kwargs['stratify_labels'].to_numpy().ravel()
            cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
            cv = cv.split(x_train, stratify_labels)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_storage = f'{model_dir}optuna_study_{model_name}.db'
        if os.path.isfile(optuna_storage):
            study = optuna.load_study(
                study_name=f'Optuna_{model_name}',
                storage=f'sqlite:///{optuna_storage}'
            )
        else:
            study = optuna.create_study(
                direction='minimize',
                storage=f'sqlite:///{optuna_storage}',
                study_name=f'Optuna_{model_name}',
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=random_state)
            )
            study.set_metric_names(['NRMSE_with_Overfitting_Penalty'])
            study.optimize(
                lambda trial: objective_with_cv(
                    trial, x_train, y_train, 
                    model_name, cv, scoring_metrics, 
                    alpha, random_state
                ),
                n_trials=n_trials,
                show_progress_bar=True
            )
        best_params = study.best_params
        print('Best params: ', best_params)
        model_dict, _ = get_model_param_dict(random_state)
        model = model_dict[model_name]
        model.set_params(**best_params)
        model.fit(x_train, y_train)
        model_file = model_dir + model_name
        pickle.dump(model, open(model_file, mode='wb+'))
        metric_csv = f'{model_dir}CV_Metrics_{model_name}.csv'
        metric_df = get_grid_search_stats(study, metric_csv, search_type='optuna')
    else:
        model_file = model_dir + model_name
        model = pickle.load(open(model_file, mode='rb'))
        metric_csv = f'{model_dir}CV_Metrics_{model_name}.csv'
        metric_df = pd.read_csv(metric_csv)
    return model, metric_df



def calc_train_test_metrics(
        pred_df: pd.DataFrame,
        cv_metric_df: pd.DataFrame,
        output_dir: str,
        use_ama_ina: bool = True,
        gw_basin_col: str = 'GW_Basin',
        year_col: str = 'Year',
        model_name: str = 'LGBM',
        precision: int = 2
) -> None:
    """Calculate train and test metrics from the prediction data frames.

    Args:
        pred_df (pd.DataFrame): Prediction data frame.
        cv_metric_df (pd.DataFrame): Cross-validation metric data frame obtained during model training. This
        will be appended to the final metrics CSV.
        output_dir (str): Output directory.
        use_ama_ina (bool): Set False to calculate error metrics over entire AZ.
        gw_basin_col (str): Name of the GW basin column.
        year_col (str): Name of the year column.
        model_name (str): Name of the model.
        precision (int): Floating point precision to use.

    Returns
        None
    """
    if use_ama_ina:
        ama_ina_basins = get_ama_ina_basin_names()
        pred_df = pred_df[pred_df[gw_basin_col].isin(ama_ina_basins)]
    year_df_list = [pred_df]
    year_df_names = ['ALL']
    for yr in pred_df[year_col].unique():
        yr_df = pred_df[pred_df[year_col] == yr].copy(deep=True)
        year_df_list.append(yr_df)
        year_df_names.append(str(yr))
    metric_df = pd.DataFrame()
    for year_df, year_name in zip(year_df_list, year_df_names):
        for data_type in year_df.DATA.unique():
            data_df = year_df[year_df.DATA == data_type]
            data_actual = data_df.Actual_GW_mm.to_numpy().ravel()
            data_pred = data_df.Pred_GW_mm.to_numpy().ravel()            
            r2 = r2_score(data_actual, data_pred)
            adj_r2 = adjusted_r2(data_actual, data_pred, data_df.shape[1])
            mae = normalized_mae(data_actual, data_pred)
            rmse = normalized_rmse(data_actual, data_pred)
            mbe = normalized_mbe(data_actual, data_pred)
            temp_dict = {
                'Year': year_name,
                'Data': data_type,
                'R2': r2,
                'Adjusted_R2': adj_r2,
                'RMSE (%)': rmse,
                'MAE (%)': mae,
                'MBE (%)': mbe
            }
            temp_df = pd.DataFrame(data=[temp_dict])
            metric_df = pd.concat([metric_df, temp_df], ignore_index=True)
    metric_df = pd.concat([metric_df, cv_metric_df], ignore_index=True)
    metric_df = metric_df.sort_values(by=['Year', 'Data'])
    for col in ['R2', 'Adjusted_R2', 'RMSE (%)', 'MAE (%)', 'MBE (%)']:
        metric_df[col] = metric_df[col].apply(lambda x: round_to_n_nonzero(x, precision))
    metric_df.to_csv(f'{output_dir}Error_Metrics_{model_name}.csv', index=False)
    

def get_grid_search_stats(
        gs_model: Any,
        metric_csv: str,
        precision: int = 2,
        search_type: str = 'grid_search'
) -> pd.DataFrame:
    """Get GridSearchCV stats.

    Args:
        gs_model: Fitted GridSearchCV/RandomizedSearchCV (can be Dask variants also) object.
        metric_csv (str): Output CSV file to save the metrics.
        precision (int): Floating point precision to use.
        search_type (str): Type of the model tuning method ('grid_search' or 'optuna').
    Returns:
        pd.DataFrame: Dataframe containing the CV stats.
    """
    if search_type == 'grid_search':
        scores = gs_model.cv_results_
    else:
        # get scores for the best trial
        best_trial = gs_model.best_trial
        scores = best_trial.user_attrs
    metric_df = pd.DataFrame()
    for data in ['train', 'test']:
        if search_type == 'grid_search':
            r2 = scores[f'mean_{data}_r2'].mean()
            adj_r2 = scores[f'mean_{data}_adjusted_r2'].mean()
            rmse = -scores[f'mean_{data}_normalized_rmse'].mean()
            mae = -scores[f'mean_{data}_normalized_mae'].mean()
            mbe = scores[f'mean_{data}_normalized_mbe'].mean()
        else:
            r2 = scores[f'{data}_r2']
            adj_r2 = scores[f'{data}_adjusted_r2']
            rmse = scores[f'{data}_normalized_rmse']
            mae = scores[f'{data}_normalized_mae']
            mbe = scores[f'{data}_normalized_mbe']
        data_name = 'VALIDATION' if data == 'test' else 'TRAIN'
        temp_dict = {
            'Year': 'CV',
            'Data': data_name,
            'R2': r2,
            'Adjusted_R2': adj_r2,
            'RMSE (%)': rmse,
            'MAE (%)': mae,
            'MBE (%)': mbe
        }
        temp_df = pd.DataFrame(data=[temp_dict])
        metric_df = pd.concat([metric_df, temp_df], ignore_index=True)
    for col in ['R2', 'Adjusted_R2', 'RMSE (%)', 'MAE (%)', 'MBE (%)']:
        metric_df[col] = metric_df[col].apply(lambda x: round_to_n_nonzero(x, precision))
    print(metric_df)
    metric_df.to_csv(metric_csv, index=False)
    return metric_df


def perform_bias_correction(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str,
        output_dir: str,
        error_gw_col: str = 'Error_GW_mm',
) -> tuple[float, float] | tuple[np.array, np.array]:
    """
    Apply bias correction to the model predictions.

    Args:
        train_df (pd.DataFrame): Training dataframe containing the model predictions and actual values.
        test_df (pd.DataFrame): Test dataframe containing the model predictions and actual values.
        model_name (str): Name of the ML model.
        output_dir (str): Output directory to save the files.
        error_gw_col (str): Name of the column containing the error in groundwater predictions.

    Returns:
        Tuple of floats representing the slope and bias of the regression line if linear regression-based
        bias correction is better than that of ML. Else, the predicted training and test residuals.
    """

    train_data_ecdf = train_df.copy(deep=True).drop(columns=[error_gw_col])
    test_data_ecdf = test_df.copy(deep=True).drop(columns=[error_gw_col])
    m_roe = np.cov(
        train_data_ecdf.Pred_GW_mm, train_data_ecdf.Actual_GW_mm
    )[0, 1] / np.var(train_data_ecdf.Pred_GW_mm)
    b_roe = np.mean(train_data_ecdf.Actual_GW_mm) - m_roe * np.mean(train_data_ecdf.Pred_GW_mm)
    train_data_ecdf['BC_GW_mm'] = np.abs(m_roe * train_data_ecdf.Pred_GW_mm + b_roe)
    train_r2_ecdf = r2_score(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm)
    train_adj_r2_ecdf = adjusted_r2(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm, train_data_ecdf.shape[1])
    train_r2 = r2_score(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
    train_adj_r2 = adjusted_r2(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm, train_data_ecdf.shape[1])
    train_rmse_ecdf = normalized_rmse(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm)
    train_rmse = normalized_rmse(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
    train_mae_ecdf = normalized_mae(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm)
    train_mae = normalized_mae(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
    train_mbe_ecdf = normalized_mbe(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm)
    train_mbe = normalized_mbe(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
    plt.rcParams.update({'font.size': 16})
    plot_ecdf_train_df = train_data_ecdf.filter(like="_GW", axis="columns")
    hue_order = plot_ecdf_train_df.columns
    sns.ecdfplot(data=plot_ecdf_train_df, hue_order=hue_order)
    plt.legend(labels=[f'BC{model_name}', model_name, 'Metered'], loc='lower right')
    plt.ylim(0, 1.1)
    plt.ylabel('ECDF')
    plt.xlabel('Annual Agricultural Groundwater Pumping (mm)')
    plt.tight_layout()
    plt.savefig(output_dir + 'ECDF_Train.png', dpi=300)
    plt.clf()
    test_data_ecdf['BC_GW_mm'] = np.abs(m_roe * test_data_ecdf.Pred_GW_mm + b_roe)
    plot_ecdf_test_df = test_data_ecdf.filter(like="_GW", axis="columns")
    sns.ecdfplot(data=plot_ecdf_test_df, hue_order=hue_order)
    plt.legend(labels=[f'BC{model_name}', model_name, 'Metered'], loc='lower right')
    plt.ylabel('ECDF')
    plt.xlabel('Annual Agricultural Groundwater Pumping (mm)')
    plt.ylim(0, 1.1)
    plt.savefig(output_dir + 'ECDF_Test.png', dpi=300)
    plt.close()
    test_r2_ecdf = r2_score(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm)
    test_adj_r2_ecdf = adjusted_r2(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm, test_data_ecdf.shape[1])
    test_r2 = r2_score(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
    test_adj_r2 = adjusted_r2(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm, test_data_ecdf.shape[1])
    test_rmse_ecdf = normalized_rmse(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm)
    test_rmse = normalized_rmse(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
    test_mae_ecdf = normalized_mae(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm)
    test_mae = normalized_mae(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
    test_mbe_ecdf = normalized_mbe(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm)
    test_mbe = normalized_mbe(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
    metrics_df_train_linear = pd.DataFrame(
        data={
            'Model': [model_name, f'BC{model_name}'],
            'Train R2': [train_r2, train_r2_ecdf],
            'Train Adjusted R2': [train_adj_r2, train_adj_r2_ecdf],
            'Train RMSE (%)': [train_rmse, train_rmse_ecdf],
            'Train MAE (%)': [train_mae, train_mae_ecdf],
            'Train MBE (%)': [train_mbe, train_mbe_ecdf],
        }
    ).round(2)
    metrics_df_train_linear.to_csv(output_dir + 'Train_Metrics_Linear.csv', index=False)
    metrics_df_test_linear = pd.DataFrame(
        data={
            'Model': [model_name, f'BC{model_name}'],
            'Test R2': [test_r2, test_r2_ecdf],
            'Test Adjusted R2': [test_adj_r2, test_adj_r2_ecdf],
            'Test RMSE (%)': [test_rmse, test_rmse_ecdf],
            'Test MAE (%)': [test_mae, test_mae_ecdf],
            'Test MBE (%)': [test_mbe, test_mbe_ecdf],
        }
    ).round(2)
    metrics_df_test_linear.to_csv(output_dir + 'Test_Metrics_Linear.csv', index=False)
    model_dict, _ = get_model_param_dict(random_state=42, use_dask=False)
    model = model_dict[model_name]
    drop_cols = ['Year', 'DATA', 'Actual_GW_mm', 'Pred_GW_mm'] + [error_gw_col]
    train_data_ecdf_ml = train_df.copy(deep=True).drop(columns=[error_gw_col])
    test_data_ecdf_ml = test_df.copy(deep=True).drop(columns=[error_gw_col])
    x_train_data = train_df.drop(columns=drop_cols)
    x_test_data = test_df.drop(columns=drop_cols)
    y_train_data = train_df[error_gw_col].to_numpy().ravel()
    model.fit(x_train_data, y_train_data)
    residuals_pred_train = model.predict(x_train_data)
    residuals_pred_test = model.predict(x_test_data)
    train_data_ecdf_ml['BC_GW_mm'] = np.abs(train_data_ecdf_ml.Pred_GW_mm + residuals_pred_train)
    test_data_ecdf_ml['BC_GW_mm'] = np.abs(test_data_ecdf_ml.Pred_GW_mm + residuals_pred_test)
    train_r2_ml = r2_score(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    train_adj_r2_ml = adjusted_r2(
        train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm, train_data_ecdf_ml.shape[1]
    )
    train_rmse_ml = normalized_rmse(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    train_mae_ml = normalized_mae(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    train_mbe_ml = normalized_mbe(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    test_r2_ml = r2_score(test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm)
    test_adj_r2_ml = adjusted_r2(
        test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm, test_data_ecdf_ml.shape[1]
    )
    test_rmse_ml = normalized_rmse(test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm)
    test_mae_ml = normalized_mae(test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm)
    test_mbe_ml = normalized_mbe(test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm)

    plt.rcParams.update({'font.size': 16})
    plot_ecdf_train_df_ml = train_data_ecdf_ml.filter(like="_GW", axis="columns")
    sns.ecdfplot(data=plot_ecdf_train_df_ml, hue_order=hue_order)
    plt.legend(labels=[f'BC{model_name}', model_name, 'Metered'], loc='lower right')
    plt.ylim(0, 1.1)
    plt.ylabel('ECDF')
    plt.xlabel('Annual Agricultural Groundwater Pumping (mm)')
    plt.tight_layout()
    plt.savefig(output_dir + 'ECDF_Train_ML.png', dpi=300)
    plt.clf()
    plot_ecdf_test_df_ml = test_data_ecdf_ml.filter(like="_GW", axis="columns")
    sns.ecdfplot(data=plot_ecdf_test_df_ml, hue_order=hue_order)
    plt.legend(labels=[f'BC{model_name}', model_name, 'Metered'], loc='lower right')
    plt.ylabel('ECDF')
    plt.xlabel('Annual Agricultural Groundwater Pumping (mm)')
    plt.ylim(0, 1.1)
    plt.savefig(output_dir + 'ECDF_Test_ML.png', dpi=300)
    plt.clf()

    metric_df_train_ml = pd.DataFrame(
        data={
            'Model': [model_name, f'BC{model_name}'],
            'Train R2': [train_r2, train_r2_ml],
            'Train Adjusted R2': [train_adj_r2, train_adj_r2_ml],
            'Train RMSE (%)': [train_rmse, train_rmse_ml],
            'Train MAE (%)': [train_mae, train_mae_ml],
            'Train MBE (%)': [train_mbe, train_mbe_ml],
        }
    ).round(2)
    metric_df_train_ml.to_csv(output_dir + 'Train_Metrics_ML.csv', index=False)
    metric_df_test_ml = pd.DataFrame(
        data={
            'Model': [model_name, f'BC{model_name}'],
            'Test R2': [test_r2, test_r2_ml],
            'Test Adjusted R2': [test_adj_r2, test_adj_r2_ml],
            'Test RMSE (%)': [test_rmse, test_rmse_ml],
            'Test MAE (%)': [test_mae, test_mae_ml],
            'Test MBE (%)': [test_mbe, test_mbe_ml],
        }
    ).round(2)
    metric_df_test_ml.to_csv(output_dir + 'Test_Metrics_ML.csv', index=False)
    if test_rmse < test_rmse_ecdf and test_rmse < test_rmse_ml:
        return 1, 0
    if test_rmse_ecdf < test_rmse_ml:
        return m_roe, b_roe
    else:
        return residuals_pred_train, residuals_pred_test


def get_prediction_results(
        model: Any,
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.array,
        y_test: np.array,
        x_scaler: MinMaxScaler | None,
        y_scaler: MinMaxScaler | None,
        year_train: pd.DataFrame,
        year_test: pd.DataFrame,
        gw_basin_train: pd.DataFrame,
        gw_basin_test: pd.DataFrame,
        model_dir: str,
        model_name: str = 'LGBM',
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        apply_bias_correction: int = 0
) -> pd.DataFrame:
    """Get model prediction results.

    Args:
        model (Any): Trained model object.
        x_train (np.ndarray or pd.DataFrame): X_train numpy array or pandas dataframe.
        x_test (np.ndarray or pd.DataFrame): X_test numpy array or pandas dataframe.
        y_train (np.array): y_train numpy array.
        y_test (np.array): y_test numpy array.
        x_scaler (MinMaxScaler or None): X scaler object.
        y_scaler (MinMaxScaler or None): y scaler object.
        year_train (pd.DataFrame): Year train data frame to append to train data.
        year_test (pd.DataFrame): Year test data frame to append to test data.
        gw_basin_train (pd.DataFrame): GW basin train data frame to append to train data.
        gw_basin_test (pd.DataFrame): GW basin test data frame to append to test data.
        model_dir (str): Model directory to store/load results.
        model_name (str): Model name. Default is 'LGBM'.
        year_col (str): Name of the year column.
        gw_basin_col (str): Name of the GW basin column.
        apply_bias_correction (int): Type of bias correction to apply. 0 for no bias correction, 1 for global
        training data-based correction, 2 for basin-wise correction.

    Returns:
        pd.DataFrame: Modified prediction data frame.
    """

    y_pred_train = np.abs(model.predict(x_train))
    y_pred_test = np.abs(model.predict(x_test))
    if x_scaler and y_scaler:
        x_train = pd.DataFrame(x_scaler.inverse_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(x_scaler.inverse_transform(x_test), columns=x_test.columns)
        y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    train_df = x_train.copy()
    train_df[year_col] = year_train[year_col].to_numpy().ravel()
    train_df[gw_basin_col] = gw_basin_train[gw_basin_col].to_numpy().ravel()
    test_df = x_test.copy()
    test_df[year_col] = year_test[year_col].to_numpy().ravel()
    test_df[gw_basin_col] = gw_basin_test[gw_basin_col].to_numpy().ravel()
    train_df['DATA'] = ['TRAIN'] * train_df.shape[0]
    train_df['Pred_GW_mm'] = y_pred_train
    train_df['Actual_GW_mm'] = y_train
    train_df['Error_GW_mm'] = train_df['Actual_GW_mm'] - train_df['Pred_GW_mm']
    test_df['DATA'] = ['TEST'] * test_df.shape[0]
    test_df['Pred_GW_mm'] = y_pred_test
    test_df['Actual_GW_mm'] = y_test
    test_df['Error_GW_mm'] = test_df['Actual_GW_mm'] - test_df['Pred_GW_mm']
    pred_df = pd.concat([train_df, test_df])
    pred_df.to_parquet(f'{model_dir}Predictions_{model_name}.parquet', index=False)
    if apply_bias_correction == 0:
        return pred_df
    elif model_name not in ['LGBM', 'DRF', 'ETR', 'RF', 'XGB', 'XGBRF', 'HGBR']:
        print(f'No bias correction for {model_name} model.')
        return pred_df
    elif apply_bias_correction == 1:
        print('Applying global bias correction...')
        output_dir = f'{model_dir}Global_Bias_Correction_{model_name}/'
        makedirs(make_proper_dir_name(output_dir))
        train_df = train_df.drop(columns=[gw_basin_col])
        test_df = test_df.drop(columns=[gw_basin_col])
        val1, val2 = perform_bias_correction(
            train_df, test_df, model_name, output_dir
        )
        if isinstance(val1, float):
            pred_df.Pred_GW_mm = np.abs(val1 * pred_df.Pred_GW_mm + val2)
        else:
            pred_df.loc[pred_df.DATA == 'TRAIN', 'Pred_GW_mm'] += val1
            pred_df.loc[pred_df.DATA == 'TEST', 'Pred_GW_mm'] += val2
        pred_df.Error_GW_mm = pred_df.Actual_GW_mm - pred_df.Pred_GW_mm
    else:
        print('Applying basin-wise bias correction...')
        gw_pred_df = pd.DataFrame()
        output_dir = f'{model_dir}Basin_Bias_Correction_{model_name}/'
        makedirs(output_dir)
        for gw_basin in pred_df[gw_basin_col].unique():
            bias_dir = f'{output_dir}{gw_basin}/'
            makedirs(bias_dir)
            basin_df = pred_df[pred_df[gw_basin_col] == gw_basin].copy(deep=True)
            basin_df_train = basin_df[basin_df.DATA == 'TRAIN'].copy(deep=True).dropna()
            basin_df_test = basin_df[basin_df.DATA == 'TEST'].copy(deep=True).dropna()
            if basin_df_train.shape[0] == 0 or basin_df_test.shape[0] == 0:
                print(f'No data for {gw_basin} in train/test data. Skipping bias correction.')
                val1 = 1
                val2 = 0
            else:
                val1, val2 = perform_bias_correction(
                    basin_df_train.drop(columns=[gw_basin_col]),
                    basin_df_test.drop(columns=[gw_basin_col]),
                    model_name,
                    bias_dir
                )
            if isinstance(val1, float):
                basin_df.Pred_GW_mm = np.abs(val1 * basin_df.Pred_GW_mm + val2)
            else:
                basin_df.loc[basin_df.DATA == 'TRAIN', 'Pred_GW_mm'] += val1
                basin_df.loc[basin_df.DATA == 'TEST', 'Pred_GW_mm'] += val2
            basin_df['Pred_GW_mm'] = np.abs(basin_df['Pred_GW_mm'])
            basin_df.Error_GW_mm = basin_df.Actual_GW_mm - basin_df.Pred_GW_mm
            gw_pred_df = pd.concat([gw_pred_df, basin_df])
        pred_df = gw_pred_df.copy()
    pred_df.to_parquet(f'{model_dir}Predictions_{model_name}_BC.parquet', index=False)
    return pred_df
