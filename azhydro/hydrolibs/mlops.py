"""
Provides methods for machine learning (ML) operations required.
"""
import os

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGCV
from dask_ml.model_selection import RandomizedSearchCV as DaskRCV
from dask_jobqueue import SLURMCluster
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, make_scorer
from sklearn.inspection import PartialDependenceDisplay as PDisp
from sklearn.inspection import permutation_importance
from sysops import makedirs, make_proper_dir_name
from gwops import get_ama_ina_basin_names


def get_model_param_dict(
        random_state: int = 0,
        use_dask: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, list]]]:
    """Get model object dictionaries and parameter dictionary for different models.

    Args:
        random_state (int): Random state (seed) for some ML algorithms.
        use_dask (bool): Set True if using Dask in a distributed computing environment.

    Returns:
        A tuple of
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
        'XGBRF': XGBRegressor(
            n_jobs=-2,
            seed=random_state,
        ),
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1, n_estimators=300, max_depth=16, num_leaves=63
        ),
        'DRF': LGBMRegressor(
            boosting_type='rf',
            random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1
        ),
        'RF': RandomForestRegressor(
            n_jobs=-2, oob_score=False,
            n_estimators=100, max_features=5,
            random_state=random_state, max_depth=None,
            # max_samples=None, min_samples_leaf=1,
            # min_samples_split=2, max_leaf_nodes=None,
            # min_impurity_decrease=0., min_weight_fraction_leaf=0.,
            # ccp_alpha=0.
        ),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs, bootstrap=True),
        'DT': DecisionTreeRegressor(random_state=random_state),
        'BT': BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
        'ABR': AdaBoostRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(n_jobs=n_jobs),
        'SVR': LinearSVR(random_state=random_state),
        'LR': LinearRegression(n_jobs=n_jobs)
    }

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
        'min_child_weight': [30, 40],
        'n_estimators': [300],
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
        'num_parallel_tree': [100, 200, 300],
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
    }, 'DRF': {
        'n_estimators': [400, 500, 600],
        'max_depth': [16, 20, 32, -1],
        'learning_rate': [1e-4],
        'subsample': [0.8, 0.5],
        'colsample_bytree': [0.8, 0.9],
        'reg_lambda': [0, 0.1],
        'path_smooth': [0, 0.1],
        'num_leaves': [100, 150, 200],
        'min_child_samples': [25, 28, 30],
    }, 'RF': {
        'n_estimators': [300, 400, 500],
        'max_features': [None, 10, 30],
        'max_depth': [None],
        'max_leaf_nodes': [None],
        'max_samples': [None],
        'min_samples_leaf': [1, 2]
    }, 'ETR': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, None],
        'max_samples': [None, 0.9, 0.8, 0.7],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'DT': {
        'max_features': [5, 6, 7],
        'max_depth': [6, 10, 20, None],
        'min_samples_leaf': [1, 5e-4, 1e-5]
    }, 'BT': {
        'n_estimators': [300, 400, 500],
        'max_features': [5, 6, 7],
        'max_samples': [1, 0.9, 0.8]
    }, 'ABR': {
        'n_estimators': [300, 400, 500, 600, 700],
        'learning_rate': [0.005, 0.0098, 0.01, 0.05],
        'loss': ['linear', 'square']
    }, 'KNN': {
        'n_neighbors': [5, 8, 10],
        'weights': ['uniform', 'distance'],
        'leaf_size': [30, 50, 20],
        'p': [1, 2, 3, 5],
    }, 'SVR': {
        'C': [1, 1.5, 2],
        'max_iter': [1000, 2000],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    }, 'LR': {
    }}
    return model_dict, param_dict

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


def compute_perm_imp(
        model_name: str,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: np.array,
        y_test: np.array,
        model: Any,
        y_scaler: MinMaxScaler | None,
        output_dir: str,
        scoring_metric: str,
        random_state: int,
        create_plots: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Compute permutation importances.

    Args:
        model_name (str): Name of the ML model. Has to be one of 'RF', 'ETR', 'LGBM', or 'DRF.'
        x_train (pd.DataFrame): Training dataframe containing the predictor data.
        x_test (pd.DataFrame): Test dataframe containing the predictor data.
        y_train (np.array): Training labels containing the observed streamflow.
        y_test (np.array): Test labels containing the observed streamflow.
        model (Any): Fitted model object.
        y_scaler (MinMaxScaler or None): y scaler object.
        output_dir (str): Output directory.
        scoring_metric (str): Name of the scoring metric. Has to one of 'r2', 'normalized_rmse', or 'normalized_mae.'
        random_state (int): Random seed.
        create_plots (bool): Set True to create permutation importance plots.

    Returns:
        Tuple of training and test importance dataframes or
        None if model_name is not one of 'RF', 'ETR', 'LGBM', or 'DRF.'
    """
    if model_name in ['RF', 'ETR', 'LGBM', 'DRF', 'XGB', 'XGBRF']:
        print('Computing permutation importance...')
        scoring_metrics = {
            'r2': 'r2',
            'normalized_rmse': make_scorer(normalized_rmse, greater_is_better=False),
            'normalized_mae': make_scorer(normalized_mae, greater_is_better=False)
        }
        if create_plots:
            imp_dict = {'Features': list(x_train.columns)}
            f_imp = np.array(model.feature_importances_).astype(float)
            if model_name in ['LGBM', 'DRF']:
                f_imp /= np.sum(f_imp)
            imp_dict['F_IMP'] = np.round(f_imp, 5)
            imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
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
        test_importances = pd.DataFrame(
            test_results.importances[sorted_importances_idx].T,
            columns=x_train.columns[sorted_importances_idx],
        )
        if y_scaler:
            train_importances = pd.DataFrame(
                y_scaler.inverse_transform(train_importances.to_numpy()),
                columns=train_importances.columns
            )
            test_importances = pd.DataFrame(
                y_scaler.inverse_transform(test_importances.to_numpy()),
                columns=test_importances.columns
            )
        if create_plots:
            for name, importances in zip(["train", "test"], [train_importances, test_importances]):
                plt.figure(figsize=(10, 6))
                plt.rcParams.update({'font.size': 12})
                ax = importances.plot.box(vert=False, whis=10)
                ax.set_xlabel("Increase in RMSE (%)")
                ax.axvline(x=0, color="k", linestyle="--")
                ax.figure.tight_layout()
                plt.savefig(f'{output_dir}{model_name}_{name}_PI.png', dpi=300)
                plt.clf()
        return train_importances, test_importances
    return None


def build_ml_model(
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.array,
        model_dir: str,
        model_name: str = 'DRF',
        random_state: int = 43,
        load_model: bool = False,
        fold_count: int = 5,
        repeats: int = 3,
        y_scaler: MinMaxScaler | None = None,
        randomized_search: bool = False,
        stratified_kfold: bool = False,
        use_dask: bool = False,
        tune_params: bool = True,
        **kwargs: Any
) -> Any:
    """Build an ML model.

    Args:
        x_train (np.ndarray or pd.DataFrame): X_train numpy array or pandas dataframe.
        y_train (np.array): y_train numpy array.
        model_dir (str): Model directory to store/load model.
        model_name (str): ML model name as per the model_dict keys.
        random_state (int): Random state (seed) for some ML algorithms.
        load_model (bool): Set model name to load existing model.
        fold_count (int): Number of folds for KFold.
        repeats (int): Number of repeats for KFold.
        y_scaler (MinMaxScaler or None): y scaler object.
        randomized_search (bool): Set True to use the more computationally efficient RandomizedSearchCV.
        stratified_kfold (bool): Set True to use RepeatedStratifiedKFold based on the crop type.
        use_dask (bool): Flag for using dask.
        tune_params (bool): Set True to tune hyperparameters.
        kwargs (dict (str, str)): Pass the 'year_train' Pandas dataframe if stratified_kfold is True.

    Returns:
        Trained model object.
    """
    model_file = model_dir + model_name
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
        get_grid_search_stats(model_grid, y_scaler)
        print('Best params: ', model_grid.best_params_)
        model = model_grid.best_estimator_
        pickle.dump(model, open(model_file, mode='wb+'))
        if dask_client:
            dask_client.close()
    else:
        model = pickle.load(open(model_file, mode='rb'))
    return model


def calc_train_test_metrics(
        pred_df: pd.DataFrame,
        use_ama_ina: bool = True,
        gw_basin_col: str = 'GW_Basin',
        precision: int = 2
) -> None:
    """Calculate train and test metrics from the prediction data frames.

    Args:
        pred_df (pd.DataFrame): Prediction data frame.
        use_ama_ina (bool): Set False to calculate error metrics over entire AZ.
        gw_basin_col (str): Name of the GW basin column.
        precision (int): Floating point precision to use.

    Returns
        None
    """
    if use_ama_ina:
        ama_ina_basins = get_ama_ina_basin_names()
        pred_df = pred_df[pred_df[gw_basin_col].isin(ama_ina_basins)]
    print('Calculating train and test metrics for:')
    print(pred_df[gw_basin_col].unique())
    train_data = pred_df[pred_df.DATA == 'TRAIN']
    test_data = pred_df[pred_df.DATA == 'TEST']
    train_actual = train_data.Actual_GW_mm.to_numpy().ravel()
    train_pred = train_data.Pred_GW_mm.to_numpy().ravel()
    test_actual = test_data.Actual_GW_mm.to_numpy().ravel()
    test_pred = test_data.Pred_GW_mm.to_numpy().ravel()
    print('\n***Overall stats***\n')
    print('Train + Validation results...')
    r2 = np.round(r2_score(train_actual, train_pred), precision)
    mae = np.round(normalized_mae(train_actual, train_pred), precision)
    rmse = np.round(normalized_rmse(train_actual, train_pred), precision)
    mbe = np.round(normalized_mbe(train_actual, train_pred), precision)
    print('R2:', r2, 'RMSE (%):', rmse, 'MAE (%):', mae, 'MBE (%):', mbe)
    print('\nTest results...')
    r2 = np.round(r2_score(test_actual, test_pred), precision)
    mae = np.round(normalized_mae(test_actual, test_pred), precision)
    rmse = np.round(normalized_rmse(test_actual, test_pred), precision)
    mbe = np.round(normalized_mbe(test_actual, test_pred), precision)
    print('R2:', r2, 'RMSE (%):', rmse, 'MAE (%):', mae, 'MBE (%):', mbe)



def get_grid_search_stats(
        gs_model: Any,
        y_scaler: MinMaxScaler | None = None,
        precision: int = 2
) -> None:
    """Get GridSearchCV stats.

    Args:
        gs_model: Fitted GridSearchCV/RandomizedSearchCV (can be Dask variants also) object.
        y_scaler (MinMaxScaler or None):y scaler object.
        precision (int): Floating point precision to use.

    Returns:
        None
    """
    scores = gs_model.cv_results_
    print('Annual CV train Results...')
    r2 = np.round(scores['mean_train_r2'].mean(), precision)
    rmse = -np.round(scores['mean_train_normalized_rmse'].mean(), precision)
    mae = -np.round(scores['mean_train_normalized_mae'].mean(), precision)
    mbe = np.round(scores['mean_train_normalized_mbe'].mean(), precision)
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
        mbe = y_scaler.inverse_transform(np.array([mbe]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE (%):', rmse, 'MAE (%):', mae, 'MBE (%):', mbe)
    print('Annual CV Validation Results...')
    r2 = np.round(scores['mean_test_r2'].mean(), precision)
    rmse = -np.round(scores['mean_test_normalized_rmse'].mean(), precision)
    mae = -np.round(scores['mean_test_normalized_mae'].mean(), precision)
    mbe = np.round(scores['mean_test_normalized_mbe'].mean(), precision)
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
        mbe = y_scaler.inverse_transform(np.array([mbe]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE (%):', rmse, 'MAE (%):', mae, 'MBE (%):', mbe)


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
    train_r2 = r2_score(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
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
    test_r2 = r2_score(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
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
    train_rmse_ml = normalized_rmse(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    train_mae_ml = normalized_mae(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    train_mbe_ml = normalized_mbe(train_data_ecdf_ml.Actual_GW_mm, train_data_ecdf_ml.BC_GW_mm)
    test_r2_ml = r2_score(test_data_ecdf_ml.Actual_GW_mm, test_data_ecdf_ml.BC_GW_mm)
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
        model_name: str = 'DRF',
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
        model_name (str): Model name.
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
    elif apply_bias_correction == 1:
        print('Applying global bias correction...')
        output_dir = f'{model_dir}Global_Bias_Correction/'
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
        output_dir = f'{model_dir}Basin_Bias_Correction/'
        makedirs(output_dir)
        for gw_basin in pred_df[gw_basin_col].unique():
            bias_dir = f'{output_dir}{gw_basin}/'
            makedirs(bias_dir)
            basin_df = pred_df[pred_df[gw_basin_col] == gw_basin].copy(deep=True)
            basin_df_train = basin_df[basin_df.DATA == 'TRAIN'].copy(deep=True)
            basin_df_test = basin_df[basin_df.DATA == 'TEST'].copy(deep=True)
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


def create_pdplots(
        x_train: pd.DataFrame,
        model: Any,
        feature_names: tuple[str, ...],
        outdir: str,
        scaling: bool = True,
        random_state: int = 0
) -> None:
    """Create partial dependence plots for ensemble tree-based algorithms (DRF, RF, LGBM, ETR).

    Args:
        x_train (pd.DataFrame): Training set.
        model (Any): Fitted model object.
        feature_names (tuple (str, ...)): Feature names for which PDP will be generated. Set 'All' to use all the
                                          features used for model training.
        outdir (str): Output directory for storing partial dependence plot.
        scaling (bool): Set False if scaling is not used for the model.
        random_state (int): Random state for PDP.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('Plotting PDP...')
    matplotlib.rcParams.update({'font.size': 16})
    feature_dict = {
        'ppt': 'PRISM Precipitation',
        'PPT': 'PRISM Precipitation',
        'SSEBop': 'SSEBop ET',
        'Relative_SSEBop': 'Relative SSEBop ET',
        'SM_IDAHO': 'Soil Moisture Change',
        'SWB_IRR': 'Irrigation Demand',
        'HSG_INF': 'Infiltration Rate',
        'tmax': r'PRISM Max Temperature',
        'TMAX': r'PRISM Max Temperature',
        'tmin': r'PRISM Min Temperature',
        'TMIN': r'PRISM Min Temperature',
        'tmean': r'PRISM Mean Temperature',
        'TMEAN': r'PRISM Mean Temperature',
        'RO':  'Surface Runoff',
        'Latitude': r'Latitude',
        'Longitude': r'Longitude',
        'GW': 'Groundwater Use',
        'Crop(s)_Corn': 'Corn',
        'Crop(s)_Soybeans': 'Soybeans',
        'Crop(s)_Cotton': 'Cotton',
        'Crop(s)_Fish Culture': 'Aquaculture',
        'Crop(s)_Rice': 'Rice'
    }
    pdp_feature_dict = {}
    if 'All' in feature_names:
        feature_names = x_train.columns
    for feature in feature_dict.keys():
        if not scaling:
            if feature in ['tmax', 'tmin', 'tmean']:
                unit = r'$^\circ$C'
            elif feature in ['Latitude', 'Longitude']:
                unit = r'$^\circ$'
            elif feature == 'HSG_INF':
                unit = 'mm/hr'
            elif feature.startswith('Relative') or feature.startswith('Crop(s)'):
                unit = 'Unitless'
            else:
                unit = 'mm'
        else:
            unit = 'Normalized'
        feature_dict[feature] += f' ({unit})'
        if feature in feature_names:
            pdp_feature_dict[feature] = feature_dict[feature]
    x_train = x_train.rename(columns=pdp_feature_dict)
    feature_names = sorted(list(pdp_feature_dict.values()))
    pdisp = PDisp.from_estimator(
        model, X=x_train, features=feature_names,
        n_jobs=-1, random_state=random_state,
        subsample=0.8
    )
    for row_idx in range(pdisp.axes_.shape[0]):
        pdisp.axes_[row_idx][0].set_ylabel(feature_dict['GW'])
    pdisp.figure_.set_size_inches(30, 15)
    plt.savefig(outdir + 'PDP.png', dpi=600)
