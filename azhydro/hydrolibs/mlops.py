"""
Provides methods for machine learning (ML) operations required.
"""

# Author: Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu


import pandas as pd
import numpy as np
import pickle
from typing import Any
from lightgbm import LGBMRegressor
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay as PDisp
from sysops import makedirs, make_proper_dir_name


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
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1
        ),
        'DRF': LGBMRegressor(
            boosting_type='rf',
            random_state=random_state,
            deterministic=True, force_row_wise=True,
            verbosity=-1
        ),
        'RF': RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
        'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs, bootstrap=True),
        'DT': DecisionTreeRegressor(random_state=random_state),
        'BT': BaggingRegressor(random_state=random_state, n_jobs=n_jobs),
        'ABR': AdaBoostRegressor(random_state=random_state),
        'KNN': KNeighborsRegressor(n_jobs=n_jobs),
        'SVR': LinearSVR(random_state=random_state),
        'LR': LinearRegression(n_jobs=n_jobs)
    }

    param_dict = {'LGBM': {
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
        kwargs (dict (str, str)): Pass the 'crop_train' or 'year_train' Pandas dataframe if stratified_kfold is True.

    Returns:
        Trained model object.
    """
    model_file = model_dir + model_name
    if model_name == 'DRF':
        model_file = f'{model_dir}{model_name}_AIWUM2'
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
        cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        if stratified_kfold:
            stratify_labels = kwargs['stratify_labels'].to_numpy().ravel()
            cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
            cv = cv.split(x_train, stratify_labels)
        makedirs(make_proper_dir_name(model_dir))
        print('\nSearching best params for {}...'.format(model_name))
        if '_' in model_name:
            model_name = model_name[: model_name.find('_')]
        model = model_dict[model_name]
        scoring_metrics = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
        cv_func_dict = {
            'dask_ml': {1: DaskRCV, 0: DaskGCV},
            'sklearn': {1: RandomizedSearchCV, 0: GridSearchCV}
        }
        cv_func = cv_func_dict[cv_lib][int(randomized_search)]
        if randomized_search:
            model_grid = cv_func(
                estimator=model, param_distributions=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True, random_state=random_state
            )
        else:
            model_grid = cv_func(
                estimator=model, param_grid=param_dict[model_name],
                scoring=scoring_metrics, n_jobs=-1, cv=cv, refit=scoring_metrics[1],
                return_train_score=True
            )
        model.fit(x_train, y_train)
        print('Train score:', model.score(x_train, y_train))
        # get_grid_search_stats(model_grid, y_scaler)
        # model = model_grid.best_estimator_
        # print('Best params: ', model_grid.best_params_)
        pickle.dump(model, open(model_file, mode='wb+'))
        if dask_client:
            dask_client.close()
    else:
        model = pickle.load(open(model_file, mode='rb'))
    if model_name in ['RF', 'ETR', 'LGBM', 'DRF']:
        imp_dict = {'Features': list(x_train.columns)}
        f_imp = np.array(model.feature_importances_).astype(float)
        if model_name in ['LGBM', 'DRF']:
            f_imp /= np.sum(f_imp)
        imp_dict['F_IMP'] = np.round(f_imp, 5)
        imp_df = pd.DataFrame(data=imp_dict).sort_values(by='F_IMP', ascending=False)
        print(imp_df)
        imp_df.to_csv(model_dir + 'F_IMP.csv', index=False)
    return model


def calc_train_test_metrics(
        pred_df: pd.DataFrame,
        crop_col: str = 'crop_cdl',
        year_col: str = 'Year'
) -> None:
    """Calculate train and test metrics from the prediction data frames.

    Args:
        pred_df (pd.DataFrame): Prediction data frame.
        crop_col (str): Name of the crop column.
        year_col (str): Name of the year column.

    Returns
        None
    """
    train_data = pred_df[pred_df.DATA == 'TRAIN']
    test_data = pred_df[pred_df.DATA == 'TEST']
    train_actual = train_data.Actual_GW.to_numpy().ravel()
    train_pred = train_data.Pred_GW.to_numpy().ravel()
    test_actual = test_data.Actual_GW.to_numpy().ravel()
    test_pred = test_data.Pred_GW.to_numpy().ravel()
    print('***Overall stats***\n')
    print('Train + Validation results...')
    r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
    print('R2:', r2, 'RMSE:', rmse, '% MAE:', mae, '%')
    print('\nTest results...')
    r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
    print('R2:', r2, 'RMSE:', rmse, '% MAE:', mae, '%')
    # cols = [col for col in pred_df.columns if col.startswith(crop_col)] + [year_col]
    # for col in cols:
    #     print('\n***{} specific stats***\n'.format(col))
    #     if col != year_col:
    #         col_val_list = [1]
    #     else:
    #         col_val_list = np.unique(np.append(train_data[col].unique(), test_data[col].unique()))
    #     for val in sorted(col_val_list):
    #         print('\n{} type: {}'.format(col, val))
    #         train_actual = train_data[train_data[col] == val]['Actual_GW'].to_numpy().ravel()
    #         train_pred = train_data[train_data[col] == val]['Pred_GW'].to_numpy().ravel()
    #         test_actual = test_data[test_data[col] == val]['Actual_GW'].to_numpy().ravel()
    #         test_pred = test_data[test_data[col] == val]['Pred_GW'].to_numpy().ravel()
    #         print('Train + Validation results...')
    #         r2, mae, rmse = get_prediction_stats(train_actual, train_pred)
    #         print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    #         print('\nTest results...')
    #         r2, mae, rmse = get_prediction_stats(test_actual, test_pred)
    #         print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_grid_search_stats(gs_model: Any, y_scaler: MinMaxScaler | None = None) -> None:
    """Get GridSearchCV stats.

    Args:
        gs_model: Fitted GridSearchCV/RandomizedSearchCV (can be Dask variants also) object.
        y_scaler (MinMaxScaler or None):y scaler object.

    Returns:
        None
    """
    scores = gs_model.cv_results_
    print('Train Results...')
    r2 = scores['mean_train_r2'].mean()
    rmse = -scores['mean_train_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_train_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)
    print('Validation Results...')
    r2 = scores['mean_test_r2'].mean()
    rmse = -scores['mean_test_neg_root_mean_squared_error'].mean()
    mae = -scores['mean_test_neg_mean_absolute_error'].mean()
    if y_scaler:
        rmse = y_scaler.inverse_transform(np.array([rmse]).reshape(1, -1)).ravel()[0]
        mae = y_scaler.inverse_transform(np.array([mae]).reshape(1, -1)).ravel()[0]
    print('R2:', r2, 'RMSE:', rmse, 'MAE:', mae)


def get_prediction_stats(
        actual_values: np.array,
        pred_values: np.array,
        precision: int = 3
) -> tuple[float, float, float]:
    """Get prediction statistics R^2, MAE, RMSE.

    Args:
        actual_values (np.array): Numpy array of actual values.
        pred_values (np.array): Numpy array of predicted values.
        precision (int): Floating point precision to use.

    Returns:
        A tuple of R^2, MAE, and RMSE.
    """
    r2, mae, rmse = (np.nan,) * 3
    if actual_values.size and pred_values.size:
        r2 = np.round(r2_score(actual_values, pred_values), precision)
        mae = np.round(mean_absolute_error(actual_values, pred_values) * 100 / np.std(actual_values), precision)
        rmse = np.round(mean_squared_error(actual_values, pred_values, squared=False) * 100 / np.std(actual_values), precision)
    return r2, mae, rmse


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
        model_dir: str,
        model_name: str = 'DRF',
        year_col: str = 'Year',
        crop_col: str = 'crop_cdl',
        crop_train: pd.DataFrame | None = None,
        crop_test: pd.DataFrame | None = None
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
        model_dir (str): Model directory to store/load results.
        model_name (str): Model name.
        year_col (str): Name of the year column.
        crop_col (str): Name of the crop column.
        crop_train (pd.DataFrame or None): Crop train data frame to append to train data.
        crop_test (pd.DataFrame or None): Crop test data frame to append to test data.

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
    test_df = x_test.copy()
    test_df[year_col] = year_test[year_col].to_numpy().ravel()
    crop_models = '_' in model_name
    if (not crop_models) and (crop_col not in train_df.columns):
        train_df[crop_col] = crop_train[crop_col].to_numpy().ravel()
        test_df[crop_col] = crop_test[crop_col].to_numpy().ravel()
    train_df['DATA'] = ['TRAIN'] * train_df.shape[0]
    train_df['Pred_GW'] = y_pred_train
    train_df['Actual_GW'] = y_train
    test_df['DATA'] = ['TEST'] * test_df.shape[0]
    test_df['Pred_GW'] = y_pred_test
    test_df['Actual_GW'] = y_test
    pred_df = pd.concat([train_df, test_df])
    pred_df['Error_GW'] = pred_df['Actual_GW'] - pred_df['Pred_GW']
    if crop_models:
        crop = model_name[model_name.find('_') + 1:]
        pred_df[crop_col] = [crop] * pred_df['Error_GW'].size
    pred_df.to_csv(f'{model_dir}Predictions_{model_name}.csv', index=False)
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
