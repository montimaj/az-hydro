"""
Handle ML Model Building Operations.

This module provides comprehensive ML model building, hyperparameter optimization,
and evaluation capabilities for groundwater pumping prediction.

Features:
- Baseline linear models (LR, Ridge, Lasso) and ensemble tree algorithms (XGB, LGBM, RF, ETR, HGBR, CatBoost, GBR, AdaBoost)
- Optuna-based hyperparameter optimization with Dask parallelization
- Model comparison and evaluation utilities
"""
import logging
import os
import pickle
import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import skexplain
from dask.distributed import Client, LocalCluster
from scipy.linalg import LinAlgWarning
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor

import hydrolibs.visualops as vizops  # Journal-quality visualization module
from hydrolibs.sysops import make_proper_dir_name, makedirs, round_to_n_nonzero
from hydrolibs.visualops import get_ama_ina_basin_names

logger = logging.getLogger(__name__)
from catboost import CatBoostRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
# Physics-Informed ML: constraints, custom objectives, and wrapper estimators
# ═════════════════════════════════════════════════════════════════════════════

# Monotone constraint map: +1 = increasing, -1 = decreasing, 0 = unconstrained
MONOTONE_CONSTRAINT_MAP: dict[str, int] = {
    'annual_et_ensemble_mm': 0,
    'well_density': 0,
    'AGRI': 0,
    'URBAN': 0,
    'annual_peff_mm': 0,
    'canal_weighted_streamflow_mm': 0,
}

# Interaction constraint groups (overlapping allowed in XGBoost)
INTERACTION_GROUPS: list[list[str]] = [
    # Water-balance core
    ['annual_et_ensemble_mm', 'annual_peff_mm', 'annual_irr_fraction', 'annual_gw_fraction'],
    # Climate drivers
    ['annual_et_ensemble_mm', 'annual_eto_mm', 'annual_tmmx_K', 'annual_tmmn_K',
     'annual_peff_mm', 'annual_precip_mm'],
    # Pumping infrastructure
    ['well_density', 'AGRI', 'URBAN', 'annual_crop_fraction'],
    # Water source
    ['streamflow_mm', 'canal_weighted_streamflow_mm', 'canal_density', 'annual_gw_fraction'],
    # Soil properties
    ['soil_depth_mm', 'awc_mm', 'ksat_mean_micromps'],
]

# Models that receive physics-informed treatment
_PIML_MODELS = {'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF'}

# Optional models (require include_all_models=True)
_OPTIONAL_MODELS = {'ETR', 'HGBR', 'GBR', 'ADA', 'BAG', 'CAT', 'LR', 'RIDGE', 'LASSO'}


def compute_irrigation_demand_floor(
        x: pd.DataFrame,
        basin_series: pd.Series,
        nhm_basin_ie: dict,
        et_col: str = 'annual_et_ensemble_mm',
        peff_col: str = 'annual_peff_mm',
        irr_frac_col: str = 'annual_irr_fraction',
        gw_frac_col: str = 'annual_gw_fraction',
) -> np.ndarray:
    """Compute irrigation demand floor for each training sample.

    The water balance ``(ET - Peff) × irr_frac × gw_frac / IE`` estimates
    only the irrigation component of GW pumping.  Since the model predicts
    total GW pumping (irrigation + municipal/industrial), this serves as a
    **lower bound**.

    Args:
        x: Feature DataFrame (must contain ET, Peff, irr_fraction, gw_fraction).
        basin_series: Per-sample basin name (aligned with x).
        nhm_basin_ie: Dict from ``load_nhm_basin_ie()`` with key ``'mean'``
            mapping basin names to mean irrigation efficiency, and
            ``'overall_mean'`` as fallback.
        et_col: Column name for annual ET.
        peff_col: Column name for annual effective precipitation.
        irr_frac_col: Column name for irrigated fraction.
        gw_frac_col: Column name for groundwater fraction.

    Returns:
        1-D array of irrigation demand floor values (mm), clipped to ≥ 0.
    """
    et = x[et_col].values
    peff = x[peff_col].values
    irr_frac = x[irr_frac_col].values
    gw_frac = x[gw_frac_col].values

    ie_mean = nhm_basin_ie['mean']
    overall_ie = nhm_basin_ie.get('overall_mean', 0.5)
    ie_arr = np.array([ie_mean.get(b, overall_ie) for b in basin_series.values])
    # Avoid division by zero
    ie_arr = np.where(ie_arr > 0, ie_arr, overall_ie)

    y_floor = (et - peff) * irr_frac * gw_frac / ie_arr
    return np.clip(y_floor, 0, None)


def _build_monotone_constraints(feature_names: list[str]) -> tuple[int, ...]:
    """Build monotone constraint tuple aligned with feature column order."""
    return tuple(MONOTONE_CONSTRAINT_MAP.get(f, 0) for f in feature_names)


def _build_interaction_constraints(
        feature_names: list[str],
        use_interaction_constraints: bool = True,
) -> list[list[str]] | None:
    """Build interaction constraint name lists aligned with feature columns.

    Returns feature **name** lists (not integer indices) so that constraints
    remain valid after the PIML wrapper strips ``__y_phys_log__`` from the
    DataFrame before calling the underlying estimator's ``fit()``.

    Returns None when disabled or when no group maps to the feature set.
    """
    if not use_interaction_constraints:
        return None
    feature_set = set(feature_names)
    constraints = []
    for group in INTERACTION_GROUPS:
        name_list = [f for f in group if f in feature_set]
        if len(name_list) >= 2:
            constraints.append(name_list)
    return constraints if constraints else None


class PhysicsFloorObjective:
    """Pickle-safe custom objective with irrigation-demand floor penalty.

    The penalty is **one-sided**: only active when the prediction falls below
    the physics-derived irrigation demand floor (in log1p space).

    Gradients are normalized to match ``reg:squarederror`` scale (grad = ŷ−y,
    hess = 1 when no floor violation) so that XGBoost's regularization
    parameters (``reg_lambda``, ``min_child_weight``, ``gamma``) retain their
    usual effective strength.

    Attributes:
        y_phys_log: 1-D array of ``log1p(irrigation_demand_floor)`` per sample.
        physics_lambda: Strength of the floor penalty (tuned by Optuna).
        mask: Binary mask — 1 where irrigation demand > 0.
    """

    def __init__(self, y_phys_log: np.ndarray, physics_lambda: float):
        self.y_phys_log = y_phys_log
        self.physics_lambda = physics_lambda
        self.mask = (y_phys_log > 0).astype(np.float64)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # XGBoost 3.0+ passes (y_true, y_pred); older versions passed
        # (y_pred, DMatrix).  Handle both by checking for DMatrix.
        if hasattr(y_true, 'get_label'):
            # Legacy XGBoost: first arg is predictions, second is DMatrix
            y_pred, y_true = y_true, y_pred.get_label()
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        residual = y_pred - y_true                     # MSE component
        violation = np.clip(self.y_phys_log - y_pred, 0, None)  # floor violation

        # Normalized to reg:squarederror scale (base grad = ŷ−y, base hess = 1)
        grad = residual - self.physics_lambda * self.mask * violation
        hess = 1.0 + self.physics_lambda * self.mask * (violation > 0).astype(np.float64)
        return grad, hess


def make_physics_floor_objective(y_phys_log: np.ndarray, physics_lambda: float):
    """Factory for XGBoost/LightGBM custom objective with irrigation-demand floor.

    Returns a pickle-safe ``PhysicsFloorObjective`` callable suitable for
    ``XGBRegressor(objective=...)`` or ``LGBMRegressor(objective=...)``.
    """
    return PhysicsFloorObjective(y_phys_log, physics_lambda)


# ── Physics wrapper estimators ───────────────────────────────────────────────
# These wrap XGBRegressor / LGBMRegressor / XGBRFRegressor and smuggle the
# physics floor (y_phys_log) through sklearn's cross_validate by appending it
# as an extra column ``__y_phys_log__`` in X.  fit() strips it, builds the
# custom objective, and delegates to the underlying estimator.

_PHYS_COL = '__y_phys_log__'


def _real_n_features(x: pd.DataFrame | np.ndarray) -> int:
    """Return the number of real features (excluding the physics column)."""
    n = x.shape[1]
    if isinstance(x, pd.DataFrame) and _PHYS_COL in x.columns:
        n -= 1
    return n


class _PhysicsXGBMixin:
    """Shared logic for physics-informed XGBoost-family estimators."""

    def _extract_phys(self, X):
        """Strip the physics column from X and return (X_clean, y_phys_log)."""
        if isinstance(X, pd.DataFrame) and _PHYS_COL in X.columns:
            y_phys = X[_PHYS_COL].values
            X_clean = X.drop(columns=[_PHYS_COL])
            return X_clean, y_phys
        elif isinstance(X, np.ndarray) and hasattr(self, '_phys_col_idx'):
            y_phys = X[:, self._phys_col_idx]
            X_clean = np.delete(X, self._phys_col_idx, axis=1)
            return X_clean, y_phys
        return X, None

    def fit(self, X, y, **kwargs):
        X_clean, y_phys = self._extract_phys(X)
        if y_phys is not None and self.physics_lambda > 0:
            self.set_params(
                objective=make_physics_floor_objective(y_phys, self.physics_lambda)
            )
        else:
            # Pure MSE fallback (λ=0 or no physics column)
            self.set_params(objective='reg:squarederror')
        return super().fit(X_clean, y, **kwargs)

    def predict(self, X, **kwargs):
        X_clean, _ = self._extract_phys(X)
        return super().predict(X_clean, **kwargs)

    def score(self, X, y, **kwargs):
        X_clean, _ = self._extract_phys(X)
        return super().score(X_clean, y, **kwargs)

    def get_xgb_params(self):
        """Exclude physics_lambda from XGBoost's internal parameter dict."""
        params = super().get_xgb_params()
        params.pop('physics_lambda', None)
        return params

    def get_n_features(self, X):
        """Return the number of real features (excluding physics column)."""
        if isinstance(X, pd.DataFrame) and _PHYS_COL in X.columns:
            return X.shape[1] - 1
        return X.shape[1]


class _PhysicsLGBMMixin:
    """Shared logic for physics-informed LightGBM estimators."""

    def _extract_phys(self, X):
        if isinstance(X, pd.DataFrame) and _PHYS_COL in X.columns:
            y_phys = X[_PHYS_COL].values
            X_clean = X.drop(columns=[_PHYS_COL])
            return X_clean, y_phys
        elif isinstance(X, np.ndarray) and hasattr(self, '_phys_col_idx'):
            y_phys = X[:, self._phys_col_idx]
            X_clean = np.delete(X, self._phys_col_idx, axis=1)
            return X_clean, y_phys
        return X, None

    def fit(self, X, y, **kwargs):
        X_clean, y_phys = self._extract_phys(X)
        if y_phys is not None and self.physics_lambda > 0:
            self.set_params(
                objective=make_physics_floor_objective(y_phys, self.physics_lambda)
            )
        else:
            self.set_params(objective='regression')
        return super().fit(X_clean, y, **kwargs)

    def predict(self, X, **kwargs):
        X_clean, _ = self._extract_phys(X)
        return super().predict(X_clean, **kwargs)

    def score(self, X, y, **kwargs):
        X_clean, _ = self._extract_phys(X)
        return super().score(X_clean, y, **kwargs)

    def get_n_features(self, X):
        if isinstance(X, pd.DataFrame) and _PHYS_COL in X.columns:
            return X.shape[1] - 1
        return X.shape[1]


class PhysicsXGBRegressor(_PhysicsXGBMixin, XGBRegressor):
    """XGBRegressor with physics-informed floor constraint objective."""

    def __init__(self, physics_lambda: float = 0.0, **kwargs):
        self.physics_lambda = physics_lambda
        super().__init__(**kwargs)


class PhysicsXGBRFRegressor(_PhysicsXGBMixin, XGBRFRegressor):
    """XGBRFRegressor with physics-informed floor constraint objective."""

    def __init__(self, physics_lambda: float = 0.0, **kwargs):
        self.physics_lambda = physics_lambda
        super().__init__(**kwargs)


class PhysicsLGBMRegressor(_PhysicsLGBMMixin, LGBMRegressor):
    """LGBMRegressor with physics-informed floor constraint objective."""

    def __init__(self, physics_lambda: float = 0.0, **kwargs):
        self.physics_lambda = physics_lambda
        super().__init__(**kwargs)


def get_model_dict(
        random_state: int = 0,
        use_dask: bool = False,
        get_model_names_only: bool = False,
        include_all_models: bool = False,
        feature_names: list[str] | None = None,
        use_interaction_constraints: bool = True,
) -> list[str] | dict[str, Any]:
    """Get model object dictionary for different models.

    Args:
        random_state (int): Random state (seed) for some ML algorithms.
        use_dask (bool): Set True if using Dask in a distributed computing environment.
        get_model_names_only (bool): Set True to return only model names.
        include_all_models (bool): Set True to include additional ensemble models
                                   (ETR, HGBR, GBR, AdaBoost, Bagging, CatBoost,
                                   LR, Ridge, Lasso).
                                   Core models (XGB, LGBM, RF, XGBRF, PIML_XGB,
                                   PIML_LGBM, PIML_XGBRF) are always included.
        feature_names (list[str] | None): Feature column names for building
            monotone/interaction constraints on PIML models.  Required when
            ``get_model_names_only`` is False.
        use_interaction_constraints (bool): Whether to apply Tier 2 interaction
            constraints to PIML models.

    Returns:
        Either a list of model names (if get_model_names_only is True) or a
        dict (str, Any) of model objects.
    """
    n_jobs = -2
    if use_dask:
        n_jobs = 1

    # Build physics constraints from feature names
    mono = None
    interaction = None
    if feature_names is not None:
        mono = _build_monotone_constraints(feature_names)
        interaction = _build_interaction_constraints(feature_names, use_interaction_constraints)

    # Core models (always available — 7 models)
    model_dict = {
        # Standard baselines (4)
        'XGB': XGBRegressor(
            n_jobs=n_jobs,
            seed=random_state,
        ),
        'LGBM': LGBMRegressor(
            tree_learner='feature', random_state=random_state,
            verbosity=-1, n_estimators=300, max_depth=16, num_leaves=31,
            n_jobs=n_jobs
        ),
        'RF': RandomForestRegressor(
            n_jobs=n_jobs, oob_score=False,
            n_estimators=300, max_features=None,
            random_state=random_state, max_depth=None
        ),
        'XGBRF': XGBRFRegressor(
            n_jobs=n_jobs,
            seed=random_state,
        ),
    }

    # Physics-informed models (3) — add constraints if feature_names provided
    piml_xgb_kwargs = dict(n_jobs=n_jobs, seed=random_state)
    piml_lgbm_kwargs = dict(
        tree_learner='feature', random_state=random_state,
        verbosity=-1, n_estimators=300, max_depth=16, num_leaves=31,
        n_jobs=n_jobs,
    )
    piml_xgbrf_kwargs = dict(n_jobs=n_jobs, seed=random_state)

    if mono is not None:
        piml_xgb_kwargs['monotone_constraints'] = mono
        piml_xgbrf_kwargs['monotone_constraints'] = mono
        # LGBM uses a string format for monotone constraints
        piml_lgbm_kwargs['monotone_constraints'] = list(mono)

    if interaction is not None:
        piml_xgb_kwargs['interaction_constraints'] = interaction
        piml_xgbrf_kwargs['interaction_constraints'] = interaction
        # LGBM uses interaction_constraints as list of lists
        piml_lgbm_kwargs['interaction_constraints'] = interaction

    model_dict['PIML_XGB'] = PhysicsXGBRegressor(**piml_xgb_kwargs)
    model_dict['PIML_LGBM'] = PhysicsLGBMRegressor(**piml_lgbm_kwargs)
    model_dict['PIML_XGBRF'] = PhysicsXGBRFRegressor(**piml_xgbrf_kwargs)

    # Additional ensemble models (optional)
    # Note: GBR and ADA don't support n_jobs (sequential only)
    # CatBoost uses thread_count instead of n_jobs
    thread_count = -1 if n_jobs == -2 else n_jobs
    if include_all_models:
        model_dict.update({
            'ETR': ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs, bootstrap=True),
            'HGBR': HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.1,
                max_depth=None, random_state=random_state
            ),
            'GBR': GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.1,
                max_depth=5, random_state=random_state,
                subsample=0.8
            ),
            'ADA': AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=10, random_state=random_state),
                n_estimators=100, random_state=random_state,
                learning_rate=0.1
            ),
            'BAG': BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=None, random_state=random_state),
                n_estimators=100, random_state=random_state,
                n_jobs=n_jobs, max_samples=0.8, max_features=0.8
            ),
            'CAT': CatBoostRegressor(
                iterations=300, learning_rate=0.1,
                depth=8, random_state=random_state,
                verbose=False, thread_count=thread_count
            ),
            'LR': LinearRegression(n_jobs=n_jobs),
            'RIDGE': Ridge(random_state=random_state),
            'LASSO': Lasso(random_state=random_state, max_iter=10000),
        })

    if get_model_names_only:
        return list(model_dict.keys())

    return model_dict


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
        float: Normalized RMSE (%) using mean of observed values.
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
        float: Normalized MAE (%) using mean of observed values.
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
        float: Normalized MBE using mean.
    """
    mean_y = np.mean(y)
    if mean_y == 0:
        return np.nan
    nmbe = np.mean(y - y_pred) * 100 / mean_y
    return nmbe


# --- Abs-wrapped scoring functions for CV/Optuna --------------------------
# Production code applies ``np.abs()`` after ``model.predict()`` to ensure
# non-negative pumping values.  These wrappers apply the same transform
# inside cross-validation scorers so that hyperparameter tuning optimizes
# the same quantity that is reported at evaluation time.

def _abs_r2_score(y: np.array, y_pred: np.array) -> float:
    return r2_score(y, np.abs(y_pred))


def _abs_adjusted_r2(y: np.array, y_pred: np.array, p: int) -> float:
    return adjusted_r2(y, np.abs(y_pred), p)


def _abs_normalized_rmse(y: np.array, y_pred: np.array) -> float:
    return normalized_rmse(y, np.abs(y_pred))


def _abs_normalized_mae(y: np.array, y_pred: np.array) -> float:
    return normalized_mae(y, np.abs(y_pred))


def _abs_normalized_mbe(y: np.array, y_pred: np.array) -> float:
    return normalized_mbe(y, np.abs(y_pred))


# --- Log-target scoring: expm1 both y and y_pred before computing ----------
def _log_r2_score(y: np.array, y_pred: np.array) -> float:
    return r2_score(np.expm1(y), np.abs(np.expm1(y_pred)))


def _log_adjusted_r2(y: np.array, y_pred: np.array, p: int) -> float:
    return adjusted_r2(np.expm1(y), np.abs(np.expm1(y_pred)), p)


def _log_normalized_rmse(y: np.array, y_pred: np.array) -> float:
    return normalized_rmse(np.expm1(y), np.abs(np.expm1(y_pred)))


def _log_normalized_mae(y: np.array, y_pred: np.array) -> float:
    return normalized_mae(np.expm1(y), np.abs(np.expm1(y_pred)))


def _log_normalized_mbe(y: np.array, y_pred: np.array) -> float:
    return normalized_mbe(np.expm1(y), np.abs(np.expm1(y_pred)))


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
        'GW_Subbasin': 'Groundwater Subbasin',
        'AGRI': 'Agricultural Density',
        'URBAN': 'Urban Density',
        'easting_m': 'Easting',
        'northing_m': 'Northing',
        'SW': 'Surface Water Density',
        'GW_Basin_Type': 'Groundwater Basin Type',
        'annual_et_ensemble_mm': 'Actual ET',
        'annual_eto_mm': 'Reference ET',
        'annual_precip_mm': 'Precipitation',
        'annual_peff_mm': 'Effective Precipitation',
        'annual_peff_pcml_mm': 'Effective Precipitation (PCML)',
        'annual_tmmx_K': '$T_{max}$',
        'annual_tmmn_K': '$T_{min}$',
        'annual_gw_fraction': 'GW Irrigation Fraction',
        'annual_crop_fraction': 'Crop Fraction',
        'annual_irr_fraction': 'Irrigation Fraction',
        'streamflow_mm': 'Streamflow',
        'canal_weighted_streamflow_mm': 'Canal-Weighted Streamflow',
        'canal_density': 'Canal Density',
        'well_density': 'Well Density',
        'soil_depth_mm': 'Soil Depth',
        'awc_mm': 'Available Water Capacity',
        'ksat_mean_micromps': '$K_{sat}$',
        'gw_pumping_mm': 'Groundwater Withdrawals',
    }

    feature_dict_units = {
        'easting_m': 'm',
        'northing_m': 'm',
        'annual_et_ensemble_mm': 'mm',
        'annual_eto_mm': 'mm',
        'annual_precip_mm': 'mm',
        'annual_peff_mm': 'mm',
        'annual_peff_pcml_mm': 'mm',
        'annual_tmmx_K': 'K',
        'annual_tmmn_K': 'K',
        'streamflow_mm': 'mm',
        'canal_weighted_streamflow_mm': 'mm',
        'canal_density': 'segments/pixel',
        'well_density': 'count/pixel',
        'soil_depth_mm': 'mm',
        'awc_mm': 'mm',
        'ksat_mean_micromps': r'$\mu$m/s',
        'gw_pumping_mm': 'mm',
    }
    return feature_dict if not get_units else (feature_dict, feature_dict_units)


class _OrigScaleModelWrapper:
    """Thin wrapper that applies expm1 to predictions of a log1p-trained model."""

    def __init__(self, model):
        self._model = model

    def predict(self, X):
        return np.abs(np.expm1(self._model.predict(X)))

    def __getattr__(self, name):
        # Guard against infinite recursion during unpickling: _model may not
        # exist yet when __getattr__ is called before __init__ completes.
        if name == '_model':
            raise AttributeError(name)
        return getattr(self._model, name)


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
        create_plots: bool = False,
        log_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Compute permutation importances.

    Args:
        model_name (str): Name of the ML model. Supported: 'XGB', 'XGBRF', 'RF',
            'ETR', 'LGBM', 'HGBR', 'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF'.
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
        None if model_name is not in the supported set.
    """
    _perm_imp_models = {
        'RF', 'ETR', 'LGBM', 'XGB', 'XGBRF', 'HGBR',
        'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF',
    }
    if model_name in _perm_imp_models:
        logger.info('Computing permutation importance...')
        if log_target:
            model = _OrigScaleModelWrapper(model)
            y_train = np.expm1(y_train)
            y_test = np.expm1(y_test)
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
            fig, ax = plt.subplots(figsize=(14, max(8, len(imp_df) * 0.7)))
            sns.barplot(data=imp_df, y='Features', x='F_IMP', ax=ax)
            ax.set_xlabel(f'{model_name} Feature Importance', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'F_IMP_{model_name}.png'), dpi=600)
            plt.close(fig)
            imp_df.to_csv(os.path.join(output_dir, f'F_IMP_{model_name}.csv'), index=False)
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
                fig, ax = plt.subplots(figsize=(12, max(6, len(importances.columns) * 0.5)))
                importances.plot.box(vert=False, whis=10, ax=ax)
                ax.set_xlabel(f"Increase in {scoring_metric.split('_')[1].upper()} (%)", fontsize=14)
                ax.tick_params(axis='both', labelsize=12)
                ax.axvline(x=0, color="k", linestyle="--")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f'{model_name}_{name}_PI.png'), dpi=600)
                plt.close(fig)
        return train_importances, test_importances
    return None


def compute_ale_plots(
        model_name: str,
        model: Any,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        output_dir: str,
        log_target: bool = False,
) -> None:
    """
    Create accumulated local effects (ALE) plots.

    Args:
        model_name (str): Name of the ML model. Supported: 'XGB', 'XGBRF', 'RF',
            'ETR', 'LGBM', 'HGBR', 'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF'.
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
    if log_target:
        model = _OrigScaleModelWrapper(model)
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
    feature_dict, feature_dict_units = get_feature_dict(get_units=True)
    feature_names = x_train.columns.tolist()
    data_dict = {
        'train': (x_train, y_train),
        'test': (x_test, y_test)
    }
    for data_type, (x_data, y_data) in data_dict.items():
        logger.info(f'Creating ALE plots for {data_type} data...')
        explainer = skexplain.ExplainToolkit(
            (model_name, model),
            X=x_data, y=y_data,
            feature_names=feature_names
        )
        ale_1d_reg = explainer.ale(
            features=feature_names,
            n_bootstrap=10,
            subsample=10000,
            n_jobs=min(len(feature_names), os.cpu_count() or 1),
            n_bins=30
        )
        explainer.plot_ale(
            ale_1d_reg,
            display_feature_names=feature_dict,
            display_units=feature_dict_units
        )
        # Override y-axis label (skexplain hardcodes it internally)
        ale_label = 'Centered ALE on Groundwater Withdrawals (mm)'
        fig = plt.gcf()
        for ax in fig.get_axes():
            if 'ALE' in ax.get_ylabel():
                ax.set_ylabel(ale_label)
            # Use scientific notation for large-value x-axes (e.g. Easting)
            xlim = ax.get_xlim()
            if abs(xlim[1]) >= 1e5:
                ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
                ax.xaxis.get_offset_text().set_fontsize(8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_ALE_{data_type}.png'), dpi=600)
        plt.clf()
        plt.close()


def compute_shap_plots(
        model_name: str,
        model: Any,
        x_data: pd.DataFrame,
        output_dir: str,
        max_display: int = 20,
        n_dependence: int = 10,
        subsample: int = 5000,
        data_label: str = '',
) -> None:
    """
    Create SHAP beeswarm, bar, dependence, and waterfall plots.

    Args:
        model_name (str): Name of the ML model (e.g. 'XGB', 'LGBM').
        model (Any): Fitted model object.
        x_data (pd.DataFrame): Feature matrix used for SHAP value computation.
        output_dir (str): Output directory for saved plots.
        max_display (int): Maximum number of features to display (default 20).
        n_dependence (int): Number of top features for dependence plots (default 10).
        subsample (int): Maximum number of samples for SHAP computation.
            Set to 0 to use all samples (may be slow for large datasets).
        data_label (str): Optional label appended to filenames (e.g. 'train', 'test').

    Returns:
        None
    """
    import shap

    makedirs(output_dir)
    feature_dict, feature_units = get_feature_dict(get_units=True)
    # Build display_name → unit mapping for axis labeling
    _display_units = {}
    for raw_name, unit in feature_units.items():
        display_name = feature_dict.get(raw_name, raw_name)
        _display_units[display_name] = unit
    suffix = f'_{data_label}' if data_label else ''

    # Subsample if necessary
    if 0 < subsample < len(x_data):
        x_sub = x_data.sample(n=subsample, random_state=42)
    else:
        x_sub = x_data

    # Rename columns to display names
    x_display = x_sub.rename(columns=feature_dict)

    logger.info(f'Computing SHAP values for {model_name} {data_label} ({len(x_sub)} samples)...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sub)

    # 1. Beeswarm summary plot
    shap.summary_plot(
        shap_values, x_display,
        max_display=max_display, show=False,
    )
    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_SHAP_Summary{suffix}.png'), dpi=600)
    plt.close('all')

    # 2. Bar plot (mean |SHAP|)
    shap.summary_plot(
        shap_values, x_display,
        plot_type='bar', max_display=max_display, show=False,
    )
    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_SHAP_Bar{suffix}.png'), dpi=600)
    plt.close('all')

    # 3. Dependence plots for top N features
    dep_dir = os.path.join(output_dir, f'Dependence{suffix}')
    makedirs(dep_dir)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:n_dependence]
    display_cols = list(x_display.columns)

    for idx in top_indices:
        feat_name = display_cols[idx]
        shap.dependence_plot(
            idx, shap_values, x_display,
            show=False,
        )
        # Append units to x-axis and colorbar (y-axis is SHAP value, unitless)
        fig = plt.gcf()
        ax = plt.gca()
        unit = _display_units.get(feat_name)
        if unit:
            xlabel = ax.get_xlabel()
            if unit not in xlabel:
                ax.set_xlabel(f'{xlabel} ({unit})')
        # Colorbar label (interaction feature on the secondary axis)
        for cb_ax in fig.get_axes():
            if cb_ax is ax:
                continue
            cb_label = cb_ax.get_ylabel()
            if cb_label:
                cb_unit = _display_units.get(cb_label)
                if cb_unit and cb_unit not in cb_label:
                    cb_ax.set_ylabel(f'{cb_label} ({cb_unit})')
        plt.gcf().set_size_inches(10, 6)
        plt.tight_layout()
        _safe_names = {'$T_{max}$': 'Tmax', '$T_{min}$': 'Tmin', '$K_{sat}$': 'Ksat'}
        safe_name = _safe_names.get(feat_name, feat_name).replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(dep_dir, f'{model_name}_SHAP_Dep_{safe_name}.png'), dpi=600)
        plt.close('all')

    logger.info(f'  {n_dependence} dependence plots saved to {dep_dir}')

    # 4. Waterfall plot for a representative sample (median prediction)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(x_sub), explainer.expected_value),
        data=x_display.values,
        feature_names=display_cols,
    )
    pred_vals = shap_values.sum(axis=1) + explainer.expected_value
    median_idx = np.argmin(np.abs(pred_vals - np.median(pred_vals)))

    shap.waterfall_plot(explanation[median_idx], max_display=max_display, show=False)
    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_SHAP_Waterfall{suffix}.png'), dpi=600)
    plt.close('all')

    logger.info(f'SHAP plots saved to {output_dir}')



def get_optuna_params_for_model(
        trial: Any,
        model_name: str
) -> dict[str, Any]:
    """
    Get Optuna hyperparameter suggestions for a specific model.
    
    Args:
        trial (Any): Optuna trial object.
        model_name (str): Name of the ML model.
        
    Returns:
        dict[str, Any]: Dictionary of suggested hyperparameters.
    """
    if model_name in ('XGB', 'PIML_XGB'):
        params = {
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 300, 600, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        }
        if model_name == 'PIML_XGB':
            params['physics_lambda'] = trial.suggest_float('physics_lambda', 0.0, 0.5)
        return params
    elif model_name in ('XGBRF', 'PIML_XGBRF'):
        params = {
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_categorical('max_depth', [0, 10, 16]),
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 300, 500, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        }
        if model_name == 'PIML_XGBRF':
            params['physics_lambda'] = trial.suggest_float('physics_lambda', 0.0, 0.5)
        return params
    elif model_name in ('LGBM', 'PIML_LGBM'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 600, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 63, 127]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 10, 20]),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, step=10)
        }
        if model_name == 'PIML_LGBM':
            params['physics_lambda'] = trial.suggest_float('physics_lambda', 0.0, 0.5)
        return params
    elif model_name == 'RF':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 300, 500, step=100),
            'max_features': trial.suggest_categorical('max_features', [None, 8, 10]),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20]),
            'max_samples': trial.suggest_categorical('max_samples', [None, 0.8, 0.9]),
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 63, 128]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3)
        }
    elif model_name == 'ETR':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 300, 500, step=100),
            'max_features': trial.suggest_categorical('max_features', [None, 8, 10]),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20]),
            'max_samples': trial.suggest_categorical('max_samples', [None, 0.8, 0.9]),
            'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 63, 128]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3)
        }
    elif model_name == 'HGBR':
        return {
            'max_iter': trial.suggest_int('max_iter', 300, 500, step=100),
            'max_depth': trial.suggest_categorical('max_depth', [None, 16, 20]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 128),
            'max_bins': trial.suggest_categorical('max_bins', [63, 127, 255]),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        }
    elif model_name == 'GBR':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 400, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
    elif model_name == 'ADA':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'estimator__max_depth': trial.suggest_int('estimator__max_depth', 3, 10),
            'estimator__min_samples_split': trial.suggest_int('estimator__min_samples_split', 2, 10),
            'estimator__min_samples_leaf': trial.suggest_int('estimator__min_samples_leaf', 1, 5)
        }
    elif model_name == 'BAG':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
            'max_samples': trial.suggest_float('max_samples', 0.6, 1.0),
            'max_features': trial.suggest_float('max_features', 0.6, 1.0),
            'estimator__max_depth': trial.suggest_categorical('estimator__max_depth', [None, 10, 20]),
            'estimator__min_samples_split': trial.suggest_int('estimator__min_samples_split', 2, 10),
            'estimator__min_samples_leaf': trial.suggest_int('estimator__min_samples_leaf', 1, 5)
        }
    elif model_name == 'CAT':
        return {
            'iterations': trial.suggest_int('iterations', 300, 600, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
            'border_count': trial.suggest_categorical('border_count', [128, 255]),
        }
    elif model_name == 'LR':
        # No hyperparameters — single trial is sufficient
        return {}
    elif model_name == 'RIDGE':
        return {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
        }
    elif model_name == 'LASSO':
        return {
            'alpha': trial.suggest_float('alpha', 1e-5, 1e1, log=True),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _build_scoring_metrics(x_train, log_target: bool) -> dict:
    """Build the CV scoring dict used by Optuna and best-fold selection."""
    if log_target:
        return {
            'r2': make_scorer(_abs_r2_score),
            'adjusted_r2': make_scorer(_abs_adjusted_r2, p=_real_n_features(x_train),
                                       greater_is_better=True),
            'log_nrmse': make_scorer(_abs_normalized_rmse, greater_is_better=False),
            'normalized_rmse': make_scorer(_log_normalized_rmse, greater_is_better=False),
            'normalized_mae': make_scorer(_log_normalized_mae, greater_is_better=False),
            'normalized_mbe': make_scorer(_log_normalized_mbe, greater_is_better=False),
        }
    return {
        'r2': make_scorer(_abs_r2_score),
        'adjusted_r2': make_scorer(_abs_adjusted_r2, p=_real_n_features(x_train),
                                   greater_is_better=True),
        'normalized_rmse': make_scorer(_abs_normalized_rmse, greater_is_better=False),
        'normalized_mae': make_scorer(_abs_normalized_mae, greater_is_better=False),
        'normalized_mbe': make_scorer(_abs_normalized_mbe, greater_is_better=False),
    }


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
        n_dask_workers: int = 4,
        use_dask: bool = True,
        pruning: bool = True,
        cv_groups: np.ndarray | pd.Series | None = None,
        log_target: bool = False,
        tuning_dir: str | None = None,
        use_interaction_constraints: bool = True,
        **kwargs: Any
) -> tuple[Any, pd.DataFrame]:
    """
    Build an ML model using Optuna with Dask parallelization for hyperparameter tuning.

    Args:
        x_train (np.ndarray or pd.DataFrame): X_train numpy array or pandas dataframe.
        y_train (np.array): y_train numpy array.
        model_dir (str): Model directory to store/load model.
        model_name (str): ML model name. Default is 'LGBM'.
        random_state (int): Random state (seed). Default is 42.
        load_model (bool): Set True to load existing model.
        fold_count (int): Number of folds for KFold. Default is 5.
        repeats (int): Number of repeats for KFold. Default is 3.
        stratified_kfold (bool): Set True to use RepeatedStratifiedKFold.
        n_trials (int): Number of Optuna trials. Default is 100.
        alpha (float): Weighting factor for overfitting penalty. Default is 0.1.
        n_dask_workers (int): Number of Dask workers. Default is 4.
        use_dask (bool): Set True to use Dask parallelization. Default is True.
        pruning (bool): Set True to enable Optuna pruning. Default is True.
        cv_groups (np.ndarray, pd.Series, or None): Group labels for GroupKFold CV.
            When provided, uses GroupKFold instead of RepeatedKFold so that the
            inner CV matches the outer evaluation strategy (e.g. pixel, temporal,
            or spatial holdout). Default is None (random RepeatedKFold).
        tuning_dir (str or None): Path to a model directory that already
            contains a completed Optuna study DB. When provided, the best
            hyperparameters and CV metrics are loaded from that study and
            Optuna tuning is skipped. Default is None (run Optuna).
        kwargs: Additional arguments.

    Returns:
        tuple[Any, pd.DataFrame]: Trained model object and dataframe containing CV stats.
    """
    makedirs(make_proper_dir_name(model_dir))

    if load_model:
        model_file = os.path.join(model_dir, model_name)
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        metric_csv = os.path.join(model_dir, f'CV_Metrics_{model_name}.csv')
        metric_df = pd.read_csv(metric_csv)
        return model, metric_df

    # Load best params + CV metrics from an existing Optuna study
    if tuning_dir is not None:
        tuning_db = os.path.join(tuning_dir, f'optuna_study_{model_name}.db')
        logger.info(f'Loading tuned params for {model_name} from {tuning_db}')
        study = optuna.create_study(
            direction='minimize',
            storage=f'sqlite:///{tuning_db}',
            study_name=f'Optuna_{model_name}',
            load_if_exists=True,
        )
        best_params = study.best_params
        logger.info(f'Best params for {model_name}: {best_params}')

        include_all = model_name in _OPTIONAL_MODELS
        feature_names = list(x_train.columns) if isinstance(x_train, pd.DataFrame) else None
        if feature_names and _PHYS_COL in feature_names:
            feature_names = [f for f in feature_names if f != _PHYS_COL]
        model_dict = get_model_dict(random_state, include_all_models=include_all,
                                    feature_names=feature_names,
                                    use_interaction_constraints=use_interaction_constraints)
        model = model_dict[model_name]
        fit_params = dict(best_params)
        if model_name in _PIML_MODELS:
            if 'physics_lambda' in fit_params:
                model.physics_lambda = fit_params.pop('physics_lambda')
            if 'underpred_alpha' in fit_params:
                model.underpred_alpha = fit_params.pop('underpred_alpha')
        model.set_params(**fit_params)
        x_fit = x_train
        if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
            x_fit = x_train.drop(columns=[_PHYS_COL])
        model.fit(x_fit, y_train)

        model_file = os.path.join(model_dir, model_name)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        metric_csv = os.path.join(model_dir, f'CV_Metrics_{model_name}.csv')
        metric_df = get_grid_search_stats(study, metric_csv, search_type='optuna')
        metric_df.to_csv(metric_csv, index=False)
        return model, metric_df

    # --- Optuna hyperparameter tuning ---
    scoring_metrics = _build_scoring_metrics(x_train, log_target)
    if cv_groups is not None:
        n_unique_groups = len(np.unique(cv_groups))
        effective_folds = min(fold_count, n_unique_groups)
        cv = GroupKFold(n_splits=effective_folds)
        logger.info(f'Using GroupKFold with {effective_folds} splits '
                    f'({n_unique_groups} unique groups)')
    elif stratified_kfold:
        stratify_labels = kwargs['stratify_labels'].to_numpy().ravel()
        cv = RepeatedStratifiedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)
        cv = cv.split(x_train, stratify_labels)
    else:
        cv = RepeatedKFold(n_splits=fold_count, n_repeats=repeats, random_state=random_state)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_storage = os.path.join(model_dir, f'optuna_study_{model_name}.db')

    # Setup Dask cluster if needed
    dask_client = None
    if use_dask and n_dask_workers > 1:
        dask_cluster = LocalCluster(
            n_workers=n_dask_workers,
            threads_per_worker=2,
            memory_limit='2GB'
        )
        dask_client = Client(dask_cluster)
        logger.info(f'Dask cluster started with {n_dask_workers} workers')

    # Create or load study (load_if_exists preserves completed trials)
    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        multivariate=True,
        group=True
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    ) if pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction='minimize',
        storage=f'sqlite:///{optuna_storage}',
        study_name=f'Optuna_{model_name}',
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    study.set_metric_names(['Log_NRMSE_with_Overfitting_Penalty'])

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    attempted = len([t for t in study.trials
                     if t.state in (optuna.trial.TrialState.COMPLETE,
                                    optuna.trial.TrialState.PRUNED)])
    logger.info(f'Study has {completed} completed, {attempted} attempted trials')

    # For parameter-free models (LR), a single trial suffices.
    # Scikit-learn tree ensembles (RF, ETR, GBR, ADA, BAG) are much
    # slower per trial than histogram-based models, so cap at 5 trials.
    # RIDGE/LASSO are simple models and don't need capping.
    _FAST_MODELS = {'XGB', 'XGBRF', 'LGBM', 'CAT', 'HGBR', 'ETR',
                     'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF'}
    _SIMPLE_MODELS = {'LR', 'RIDGE', 'LASSO'}
    if model_name == 'LR':
        effective_n_trials = 1
    elif model_name in _SIMPLE_MODELS:
        effective_n_trials = n_trials
    elif model_name not in _FAST_MODELS:
        effective_n_trials = min(n_trials, 10)
    else:
        effective_n_trials = n_trials
    remaining = max(0, effective_n_trials - attempted)

    t0 = time.perf_counter()
    if remaining > 0:
        logger.info(f'Running {remaining} remaining trials '
                    f'(target={effective_n_trials}, completed={completed})')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Ill-conditioned matrix',
                                    category=LinAlgWarning)
            study.optimize(
                lambda trial: objective_with_cv_enhanced(
                    trial, x_train, y_train,
                    model_name, cv, scoring_metrics,
                    alpha, random_state, pruning,
                    cv_groups=cv_groups,
                    use_interaction_constraints=use_interaction_constraints,
                ),
                n_trials=remaining,
                n_jobs=1,
                show_progress_bar=True,
                gc_after_trial=True
            )
    else:
        logger.info(f'Study already has {completed} completed trials '
                    f'(target={effective_n_trials}) — skipping optimization')
    tuning_runtime_sec = round(time.perf_counter() - t0, 2)
    logger.info(f'Tuning runtime for {model_name}: {tuning_runtime_sec:.2f}s')

    # Cleanup Dask
    if dask_client:
        dask_client.close()

    best_params = study.best_params
    logger.info(f'Best params for {model_name}: {best_params}')
    logger.info(f'Best value: {study.best_value:.4f}')

    # Refit on full training data with best parameters
    include_all = model_name in _OPTIONAL_MODELS
    feature_names = list(x_train.columns) if isinstance(x_train, pd.DataFrame) else None
    if feature_names and _PHYS_COL in feature_names:
        feature_names = [f for f in feature_names if f != _PHYS_COL]
    model_dict = get_model_dict(random_state, include_all_models=include_all,
                                feature_names=feature_names,
                                use_interaction_constraints=use_interaction_constraints)
    model = model_dict[model_name]
    fit_params = dict(best_params)
    if model_name in _PIML_MODELS:
        if 'physics_lambda' in fit_params:
            model.physics_lambda = fit_params.pop('physics_lambda')
        if 'underpred_alpha' in fit_params:
            model.underpred_alpha = fit_params.pop('underpred_alpha')
    model.set_params(**fit_params)
    # PIML wrappers strip _PHYS_COL internally; standard models need it removed
    x_fit = x_train
    if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
        x_fit = x_train.drop(columns=[_PHYS_COL])
    model.fit(x_fit, y_train)

    model_file = os.path.join(model_dir, model_name)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    metric_csv = os.path.join(model_dir, f'CV_Metrics_{model_name}.csv')
    metric_df = get_grid_search_stats(study, metric_csv, search_type='optuna')
    metric_df.to_csv(metric_csv, index=False)

    return model, metric_df


def objective_with_cv_enhanced(
        trial: Any,
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        model_name: str,
        cv: Any,
        scoring_metrics: dict[str, Any],
        alpha: float = 0.1,
        random_state: int = 42,
        pruning: bool = True,
        cv_groups: np.ndarray | pd.Series | None = None,
        use_interaction_constraints: bool = True,
) -> float:
    """
    Enhanced objective function for Optuna with pruning support.

    Args:
        trial (Any): Optuna trial object.
        x_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.
        model_name (str): Name of the ML model.
        cv (Any): Cross-validation strategy.
        scoring_metrics (dict): Scoring metrics for cross-validation.
        alpha (float): Weighting factor for overfitting penalty.
        random_state (int): Random state.
        pruning (bool): Whether to enable pruning.
        cv_groups (np.ndarray, pd.Series, or None): Group labels for GroupKFold.

    Returns:
        float: Objective value (lower is better — log-space NRMSE with penalties).
    """
    # Get hyperparameters for the model
    params = get_optuna_params_for_model(trial, model_name)

    # Get model
    include_all = model_name in _OPTIONAL_MODELS
    feature_names = list(x_train.columns) if isinstance(x_train, pd.DataFrame) else None
    if feature_names and _PHYS_COL in feature_names:
        feature_names = [f for f in feature_names if f != _PHYS_COL]
    model = get_model_dict(random_state, include_all_models=include_all,
                           feature_names=feature_names,
                           use_interaction_constraints=use_interaction_constraints)[model_name]
    # Extract physics_lambda for PIML wrappers before set_params
    fit_params = dict(params)
    if model_name in _PIML_MODELS and 'physics_lambda' in fit_params:
        model.physics_lambda = fit_params.pop('physics_lambda')
    model.set_params(**fit_params)
    # Single-threaded model training — cross_validate parallelises across folds
    if hasattr(model, 'n_jobs'):
        model.set_params(n_jobs=1)
    elif hasattr(model, 'thread_count'):
        model.set_params(thread_count=1)

    # PIML wrappers strip _PHYS_COL internally; standard models need it removed
    x_cv = x_train
    if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
        x_cv = x_train.drop(columns=[_PHYS_COL])

    # Cross-validation with early stopping for pruning
    cv_results = cross_validate(
        estimator=model,
        X=x_cv,
        y=y_train,
        cv=cv,
        groups=cv_groups,
        n_jobs=-1,
        scoring=scoring_metrics,
        return_train_score=True
    )

    # Use log-space NRMSE as objective when available (log_target),
    # otherwise fall back to original-scale NRMSE.
    nrmse_key = 'log_nrmse' if 'test_log_nrmse' in cv_results else 'normalized_rmse'
    test_mean_nrmse = -cv_results[f'test_{nrmse_key}'].mean()
    test_std_nrmse = abs(cv_results[f'test_{nrmse_key}'].std())
    train_mean_nrmse = -cv_results[f'train_{nrmse_key}'].mean()

    # Store user attributes
    for data in ['train', 'test']:
        r2_key = f'{data}_r2'
        adj_r2_key = f'{data}_adjusted_r2'
        neg_rmse_key = f'{data}_normalized_rmse'
        neg_mae_key = f'{data}_normalized_mae'
        mbe_key = f'{data}_normalized_mbe'

        trial.set_user_attr(r2_key, cv_results[r2_key].mean())
        trial.set_user_attr(adj_r2_key, cv_results[adj_r2_key].mean())
        trial.set_user_attr(neg_rmse_key, -cv_results[neg_rmse_key].mean())
        trial.set_user_attr(neg_mae_key, -cv_results[neg_mae_key].mean())
        trial.set_user_attr(mbe_key, -cv_results[mbe_key].mean())

    # Calculate objective: minimize NRMSE with overfitting ratio and variance penalties.
    # The ratio test/train measures proportional overfitting — a 20× gap is penalized
    # much harder than a 2× gap, unlike the absolute difference which treats them similarly.
    beta = 0.5 * alpha
    overfit_ratio = test_mean_nrmse / max(train_mean_nrmse, 1e-6)
    trial_obj = test_mean_nrmse * (1 + alpha * max(overfit_ratio - 1, 0)) + beta * test_std_nrmse

    # Report per-fold metrics at incremental steps for pruning
    if pruning:
        fold_nrmses = -cv_results[f'test_{nrmse_key}']
        for step, fold_nrmse in enumerate(fold_nrmses):
            trial.report(fold_nrmse, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return trial_obj


def _plot_physics_floor_diagnostic(
        model: Any,
        model_name: str,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        output_dir: str,
        log_target: bool = False,
) -> None:
    """Scatter predicted vs physics floor, colored by constraint activity.

    Only generated for PIML models.  Silently returns for standard models.
    """
    if model_name not in _PIML_MODELS:
        return
    if not isinstance(x_train, pd.DataFrame) or _PHYS_COL not in x_train.columns:
        return

    y_floor_log = x_train[_PHYS_COL].values
    x_clean = x_train.drop(columns=[_PHYS_COL])
    y_pred_log = model.predict(x_clean)

    # Convert to original scale for plotting
    if log_target:
        y_floor = np.expm1(y_floor_log)
        y_pred = np.abs(np.expm1(y_pred_log))
        y_obs = np.expm1(y_train)
    else:
        y_floor = y_floor_log
        y_pred = np.abs(y_pred_log)
        y_obs = y_train

    active = y_pred <= y_floor
    n_active = int(active.sum())
    n_total = len(active)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Panel 1: predicted vs floor
    ax = axes[0]
    ax.scatter(y_floor[~active], y_pred[~active], alpha=0.3, s=8,
               color='steelblue', label=f'Above floor ({n_total - n_active:,})')
    ax.scatter(y_floor[active], y_pred[active], alpha=0.5, s=12,
               color='firebrick', label=f'At/below floor ({n_active:,})')
    fmax = max(np.nanmax(y_floor), np.nanmax(y_pred))
    ax.plot([0, fmax], [0, fmax], 'k--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('Physics Floor — (ET−Peff)×irr×gw/IE  [mm]')
    ax.set_ylabel('Predicted GW Pumping [mm]')
    ax.set_title(f'{model_name}: Predicted vs Irrigation Demand Floor')
    ax.legend(fontsize=12)

    # Panel 2: residual (pred - observed) split by constraint activity
    ax = axes[1]
    resid = y_pred - y_obs
    resid_active = resid[active]
    resid_inactive = resid[~active]
    bins = np.linspace(np.nanpercentile(resid, 1), np.nanpercentile(resid, 99), 60)
    ax.hist(resid_inactive, bins=bins, alpha=0.6, color='steelblue',
            label=f'Floor inactive ({n_total - n_active:,})')
    ax.hist(resid_active, bins=bins, alpha=0.6, color='firebrick',
            label=f'Floor active ({n_active:,})')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual (Predicted − Observed) [mm]')
    ax.set_ylabel('Count')
    ax.set_title(f'{model_name}: Residual Distribution by Floor Activity')
    ax.legend(fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{model_name}_Physics_Floor_Diagnostic.png')
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Physics floor diagnostic: {n_active}/{n_total} '
                f'({100*n_active/n_total:.1f}%) training samples at/below floor')


def generate_interp_plots(
        model: Any,
        model_name: str,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        output_dir: str,
        random_state: int,
        log_target: bool = False,
) -> None:
    """Generate permutation importance, ALE, SHAP, and floor diagnostic plots.

    Strips the physics column (if present) before passing data to the
    interpretability functions.  For PIML models, also generates a physics
    floor diagnostic plot showing predicted vs irrigation demand floor.
    """
    interp_dir = os.path.join(output_dir, 'Interpretability')
    makedirs(interp_dir)

    # Physics floor diagnostic (before stripping _PHYS_COL)
    _plot_physics_floor_diagnostic(
        model, model_name, x_train, y_train,
        interp_dir, log_target=log_target,
    )

    # Strip physics column — interp functions expect real features only
    x_tr = x_train.drop(columns=[_PHYS_COL], errors='ignore')
    x_te = x_test.drop(columns=[_PHYS_COL], errors='ignore')

    # Permutation importance (train + test)
    compute_perm_imp(
        model_name, x_tr, x_te, y_train, y_test,
        model, interp_dir, scoring_metric='scaled_rmse',
        random_state=random_state, create_plots=True,
        log_target=log_target,
    )

    # ALE plots (train + test)
    compute_ale_plots(
        model_name, model,
        x_tr, y_train, x_te, y_test,
        interp_dir, log_target=log_target,
    )

    # SHAP plots (train + test, separate labels)
    compute_shap_plots(
        model_name, model, x_tr, interp_dir,
        data_label='train',
    )
    compute_shap_plots(
        model_name, model, x_te, interp_dir,
        data_label='test',
    )


def compare_all_models(
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.array,
        y_test: np.array,
        model_dir: str,
        model_names: list[str] = None,
        random_state: int = 42,
        use_optuna: bool = True,
        fold_count: int = 5,
        repeats: int = 3,
        n_trials: int = 50,
        n_dask_workers: int = 4,
        use_dask: bool = True,
        year_train: pd.DataFrame = None,
        year_test: pd.DataFrame = None,
        basin_train: pd.DataFrame = None,
        basin_test: pd.DataFrame = None,
        x_scaler: Any = None,
        y_scaler: Any = None,
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        use_ama_ina: bool = True,
        apply_bias_correction: bool = False,
        cv_groups: np.ndarray | pd.Series | None = None,
        easting_train: pd.DataFrame | None = None,
        easting_test: pd.DataFrame | None = None,
        northing_train: pd.DataFrame | None = None,
        northing_test: pd.DataFrame | None = None,
        test_case: str = '',
        test_year_limits: tuple[tuple[int, int], ...] = (),
        raster_res: float = 2000,
        create_basin_plots: bool = True,
        log_target: bool = False,
        tuning_model_dir: str | None = None,
        create_interp_plots: bool = False,
        use_interaction_constraints: bool = True,
) -> pd.DataFrame:
    """
    Compare all available models with full evaluation.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        model_dir: Directory to save results.
        model_names: List of model names to compare.
        random_state: Random state.
        use_optuna: Whether to use Optuna for tuning.
        fold_count: Number of folds for KFold. Default is 5.
        repeats: Number of repeats for RepeatedKFold. Default is 3.
        n_trials: Number of Optuna trials.
        n_dask_workers: Number of Dask workers for parallel Optuna trials.
        use_dask: Whether to use Dask parallelization.
        year_train: DataFrame with year info for training data.
        year_test: DataFrame with year info for test data.
        basin_train: DataFrame with basin info for training data.
        basin_test: DataFrame with basin info for test data.
        x_scaler: Feature scaler (for inverse transform).
        y_scaler: Target scaler (for inverse transform).
        year_col: Name of year column.
        gw_basin_col: Name of basin column.
        use_ama_ina: Whether to filter for AMA/INA basins in metrics.
        apply_bias_correction: Whether to apply global linear bias correction.
        cv_groups: Group labels for GroupKFold CV. When provided, inner CV
            uses GroupKFold to match the outer evaluation strategy.
        test_case: Test case identifier for visualizations (e.g. 'Random',
            'Pixel_Holdout'). When non-empty, per-model visualizations are
            generated immediately after each model is trained.
        test_year_limits: Test period definitions for visualization shading.
        raster_res: Raster resolution in meters for visualization.
        create_basin_plots: Whether to create individual basin plots.
        tuning_model_dir: Path to the Model_Comparison directory from a
            previous run that contains per-model Optuna study DBs. When
            provided, best params and CV metrics are loaded from those DBs
            instead of running Optuna.

    Returns:
        DataFrame with comparison results.
    """
    makedirs(make_proper_dir_name(model_dir))
    results = []
    trained_models = {}  # Store trained models for later use

    # Check if we have full data for detailed evaluation
    has_full_data = all([
        year_train is not None, year_test is not None,
        basin_train is not None, basin_test is not None
    ])

    # Default models
    if model_names is None:
        model_names = ['XGB', 'LGBM', 'RF', 'XGBRF', 'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF']

    logger.info('='*60)
    logger.info('Model Comparison')
    logger.info('='*60)

    # Train individual models
    for model_name in model_names:
        model_subdir = os.path.join(model_dir, model_name)

        # ── Fast path: reuse existing BC predictions if available ────
        bc_parquet = os.path.join(model_subdir, f'Predictions_{model_name}_BC.parquet')
        plain_parquet = os.path.join(model_subdir, f'Predictions_{model_name}.parquet')
        cached_parquet = bc_parquet if os.path.isfile(bc_parquet) else (
            plain_parquet if os.path.isfile(plain_parquet) else None)

        if cached_parquet is not None:
            logger.info(f'Loading cached predictions for {model_name} from {cached_parquet}')
            pred_df = pd.read_parquet(cached_parquet)

            # Load CV metrics from disk if available
            cv_metric_csv = os.path.join(model_subdir, f'CV_Metrics_{model_name}.csv')
            cv_metric_df = (pd.read_csv(cv_metric_csv)
                            if os.path.isfile(cv_metric_csv) else pd.DataFrame())

            cv_val_r2 = cv_val_rmse = cv_val_mae = cv_train_r2 = np.nan
            if not cv_metric_df.empty and 'Data' in cv_metric_df.columns:
                val_row = cv_metric_df[cv_metric_df['Data'] == 'VALIDATION']
                train_row = cv_metric_df[cv_metric_df['Data'] == 'TRAIN']
                if not val_row.empty:
                    cv_val_r2 = val_row['R2'].values[0]
                    cv_val_rmse = val_row['RMSE (%)'].values[0]
                    cv_val_mae = (val_row['MAE (%)'].values[0]
                                  if 'MAE (%)' in val_row.columns else np.nan)
                if not train_row.empty:
                    cv_train_r2 = train_row['R2'].values[0]

            # Recalculate metrics + regenerate visualizations
            calc_train_test_metrics(
                pred_df, cv_metric_df, model_subdir,
                use_ama_ina=use_ama_ina,
                gw_basin_col=gw_basin_col,
                year_col=year_col,
                model_name=model_name,
                n_features=_real_n_features(x_train),
            )
            if test_case:
                generate_model_visualizations(
                    pred_df=pred_df,
                    output_dir=os.path.join(model_subdir, 'Visualizations'),
                    model_name=model_name,
                    test_case=test_case,
                    test_year_limits=test_year_limits,
                    raster_res=raster_res,
                    use_ama_ina=use_ama_ina,
                    create_basin_plots=create_basin_plots,
                )

            # Run interpretability if dir is empty/missing (cached predictions
            # but no prior interp run).  Load the saved model from disk.
            if create_interp_plots:
                interp_dir = os.path.join(model_subdir, 'Interpretability')
                interp_exists = (
                    os.path.isdir(interp_dir)
                    and len(os.listdir(interp_dir)) > 0
                )
                if not interp_exists:
                    model_file = os.path.join(model_subdir, model_name)
                    if os.path.isfile(model_file):
                        logger.info(f'Interpretability missing for cached {model_name} — '
                                    f'loading model and generating plots')
                        with open(model_file, 'rb') as f:
                            cached_model = pickle.load(f)
                        generate_interp_plots(
                            cached_model, model_name, x_train, x_test,
                            y_train, y_test, model_subdir,
                            random_state, log_target,
                        )
                    else:
                        logger.warning(f'Cannot generate interpretability for {model_name}: '
                                       f'no saved model at {model_file}')

            train_pred = pred_df[pred_df['DATA'] == 'TRAIN']
            test_pred = pred_df[pred_df['DATA'] == 'TEST']
            train_r2 = r2_score(train_pred['Actual_GW_mm'], train_pred['Pred_GW_mm'])
            test_r2 = r2_score(test_pred['Actual_GW_mm'], test_pred['Pred_GW_mm'])
            train_rmse = normalized_rmse(train_pred['Actual_GW_mm'].values, train_pred['Pred_GW_mm'].values)
            test_rmse = normalized_rmse(test_pred['Actual_GW_mm'].values, test_pred['Pred_GW_mm'].values)
            train_mae = normalized_mae(train_pred['Actual_GW_mm'].values, train_pred['Pred_GW_mm'].values)
            test_mae = normalized_mae(test_pred['Actual_GW_mm'].values, test_pred['Pred_GW_mm'].values)

        else:
            # ── Normal path: train model and generate predictions ────
            logger.info(f'Training {model_name}...')

            if use_optuna:
                td = os.path.join(tuning_model_dir, model_name) if tuning_model_dir else None
                model, cv_metric_df = build_ml_model_optuna(
                    x_train, y_train, model_subdir, model_name,
                    random_state=random_state, fold_count=fold_count,
                    repeats=repeats, n_trials=n_trials,
                    n_dask_workers=n_dask_workers, use_dask=use_dask,
                    cv_groups=cv_groups, log_target=log_target,
                    tuning_dir=td,
                    use_interaction_constraints=use_interaction_constraints,
                )
            else:
                feature_names = list(x_train.columns) if isinstance(x_train, pd.DataFrame) else None
                if feature_names and _PHYS_COL in feature_names:
                    feature_names = [f for f in feature_names if f != _PHYS_COL]
                model_dict = get_model_dict(
                    random_state, include_all_models=True,
                    feature_names=feature_names,
                    use_interaction_constraints=use_interaction_constraints,
                )
                model = model_dict[model_name]
                # PIML wrappers strip _PHYS_COL internally; standard models need it removed
                x_fit = x_train
                if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
                    x_fit = x_train.drop(columns=[_PHYS_COL])
                model.fit(x_fit, y_train)
                cv_metric_df = pd.DataFrame()

            trained_models[model_name] = model

            # Extract CV validation metrics
            cv_val_r2 = np.nan
            cv_val_rmse = np.nan
            cv_val_mae = np.nan
            cv_train_r2 = np.nan

            if not cv_metric_df.empty and 'Data' in cv_metric_df.columns:
                val_row = cv_metric_df[cv_metric_df['Data'] == 'VALIDATION']
                train_row = cv_metric_df[cv_metric_df['Data'] == 'TRAIN']
                if not val_row.empty:
                    cv_val_r2 = val_row['R2'].values[0]
                    cv_val_rmse = val_row['RMSE (%)'].values[0]
                    cv_val_mae = val_row['MAE (%)'].values[0] if 'MAE (%)' in val_row.columns else np.nan
                if not train_row.empty:
                    cv_train_r2 = train_row['R2'].values[0]

            # Generate full prediction results if we have the data
            if has_full_data:
                pred_df = get_prediction_results(
                    model, x_train, x_test,
                    y_train, y_test, x_scaler,
                    y_scaler, year_train,
                    year_test, basin_train,
                    basin_test, model_subdir,
                    model_name,
                    apply_bias_correction=apply_bias_correction,
                    year_col=year_col,
                    gw_basin_col=gw_basin_col,
                    easting_train=easting_train,
                    easting_test=easting_test,
                    northing_train=northing_train,
                    northing_test=northing_test,
                    log_target=log_target,
                )

                # Calculate detailed metrics
                calc_train_test_metrics(
                    pred_df, cv_metric_df, model_subdir,
                    use_ama_ina=use_ama_ina,
                    gw_basin_col=gw_basin_col,
                    year_col=year_col,
                    model_name=model_name,
                    n_features=_real_n_features(x_train)
                )

                # Per-model visualizations (immediately after training)
                if test_case:
                    generate_model_visualizations(
                        pred_df=pred_df,
                        output_dir=os.path.join(model_subdir, 'Visualizations'),
                        model_name=model_name,
                        test_case=test_case,
                        test_year_limits=test_year_limits,
                        raster_res=raster_res,
                        use_ama_ina=use_ama_ina,
                        create_basin_plots=create_basin_plots,
                    )

                # Interpretability plots (perm importance, ALE, SHAP)
                if create_interp_plots:
                    generate_interp_plots(
                        model, model_name, x_train, x_test,
                        y_train, y_test, model_subdir,
                        random_state, log_target,
                    )

                # Use metrics from pred_df for comparison
                train_pred = pred_df[pred_df['DATA'] == 'TRAIN']
                test_pred = pred_df[pred_df['DATA'] == 'TEST']

                train_r2 = r2_score(train_pred['Actual_GW_mm'], train_pred['Pred_GW_mm'])
                test_r2 = r2_score(test_pred['Actual_GW_mm'], test_pred['Pred_GW_mm'])
                train_rmse = normalized_rmse(train_pred['Actual_GW_mm'].values, train_pred['Pred_GW_mm'].values)
                test_rmse = normalized_rmse(test_pred['Actual_GW_mm'].values, test_pred['Pred_GW_mm'].values)
                train_mae = normalized_mae(train_pred['Actual_GW_mm'].values, train_pred['Pred_GW_mm'].values)
                test_mae = normalized_mae(test_pred['Actual_GW_mm'].values, test_pred['Pred_GW_mm'].values)
            else:
                # Evaluate directly on raw data
                if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
                    x_tr_pred = x_train.drop(columns=[_PHYS_COL])
                    x_te_pred = x_test.drop(columns=[_PHYS_COL])
                else:
                    x_tr_pred = x_train
                    x_te_pred = x_test
                y_pred_train = model.predict(x_tr_pred)
                y_pred_test = model.predict(x_te_pred)

                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = normalized_rmse(y_train, y_pred_train)
                test_rmse = normalized_rmse(y_test, y_pred_test)
                train_mae = normalized_mae(y_train, y_pred_train)
                test_mae = normalized_mae(y_test, y_pred_test)

        # Calculate overfitting indicators
        overfit_r2 = train_r2 - test_r2
        overfit_rmse = test_rmse - train_rmse
        overfit_val_r2 = cv_train_r2 - cv_val_r2 if not np.isnan(cv_val_r2) else np.nan

        results.append({
            'Model': model_name,
            'Train_R2': train_r2,
            'Val_R2': cv_val_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Val_RMSE': cv_val_rmse,
            'Test_RMSE': test_rmse,
            'Train_MAE': train_mae,
            'Val_MAE': cv_val_mae,
            'Test_MAE': test_mae,
            'Overfit_R2': overfit_r2,
            'Overfit_RMSE': overfit_rmse,
            'Overfit_Val_R2': overfit_val_r2,
        })

        logger.info(f'  Train R2: {train_r2:.4f}, Val R2: {cv_val_r2:.4f}, Test R2: {test_r2:.4f}')
        logger.info(f'  Train RMSE: {train_rmse:.2f}%, Val RMSE: {cv_val_rmse:.2f}%, Test RMSE: {test_rmse:.2f}%')
        logger.info(f'  Overfitting (R2 gap): {overfit_r2:.4f}')

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_RMSE')
    results_df.to_csv(os.path.join(model_dir, 'Model_Comparison.csv'), index=False)

    # Create comprehensive comparison plots with overfitting analysis
    _create_model_comparison_plots(results_df, model_dir)

    logger.info('='*60)
    logger.info(f'Best Model: {results_df.iloc[0]["Model"]}')
    logger.info(f"Test R2: {results_df.iloc[0]['Test_R2']:.4f}")
    logger.info(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}%")
    logger.info(f"Overfitting (R2 gap): {results_df.iloc[0]['Overfit_R2']:.4f}")
    logger.info('='*60)

    return results_df, model_dir


def _create_model_comparison_plots(results_df: pd.DataFrame, model_dir: str) -> None:
    """
    Create comprehensive model comparison plots with overfitting analysis.
    
    Args:
        results_df: DataFrame with model comparison results.
        model_dir: Directory to save plots.
    """
    # Apply journal-quality settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })

    n_models = len(results_df)
    x_pos = np.arange(n_models)
    bar_width = 0.25

    # =====================================================================
    # Figure 1: Train/Validation/Test R² Comparison with Overfitting
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: R² Score Comparison (Train, Val, Test)
    ax = axes[0, 0]
    bars1 = ax.bar(x_pos - bar_width, results_df['Train_R2'], bar_width,
                   label='Train', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, results_df['Val_R2'], bar_width,
                   label='Validation (CV)', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + bar_width, results_df['Test_R2'], bar_width,
                   label='Test', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score: Train vs Validation vs Test', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Good fit threshold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Plot 2: NRMSE Comparison (Train, Val, Test)
    ax = axes[0, 1]
    bars1 = ax.bar(x_pos - bar_width, results_df['Train_RMSE'], bar_width,
                   label='Train', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, results_df['Val_RMSE'], bar_width,
                   label='Validation (CV)', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + bar_width, results_df['Test_RMSE'], bar_width,
                   label='Test', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('NRMSE (%)')
    ax.set_title('NRMSE: Train vs Validation vs Test', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Overfitting Analysis - R² Gap (Train - Test)
    # Thresholds adjusted for groundwater pumping (high uncertainty domain)
    ax = axes[1, 0]
    colors = ['#e74c3c' if x > 0.15 else '#f39c12' if x > 0.08 else '#2ecc71'
              for x in results_df['Overfit_R2']]
    bars = ax.bar(x_pos, results_df['Overfit_R2'], 0.6, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('R² Gap (Train - Test)')
    ax.set_title('Overfitting Analysis: R² Gap\n(Higher = More Overfitting)', fontweight='bold')
    ax.axhline(y=0.08, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate (0.08)')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Severe (0.15)')
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Acceptable')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, results_df['Overfit_R2']):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Plot 4: Overfitting Analysis - RMSE Gap (Test - Train)
    # Thresholds adjusted for groundwater pumping (high uncertainty domain)
    ax = axes[1, 1]
    colors = ['#e74c3c' if x > 15 else '#f39c12' if x > 8 else '#2ecc71'
              for x in results_df['Overfit_RMSE']]
    bars = ax.bar(x_pos, results_df['Overfit_RMSE'], 0.6, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('NRMSE Gap (Test - Train) %')
    ax.set_title('Overfitting Analysis: NRMSE Gap\n(Higher = More Overfitting)', fontweight='bold')
    ax.axhline(y=8, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate (8%)')
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Severe (15%)')
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Acceptable')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, results_df['Overfit_RMSE']):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Model_Comparison_TrainValTest.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # Figure 2: Heatmap of Overfitting Severity
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for heatmap
    heatmap_data = results_df[['Model', 'Train_R2', 'Val_R2', 'Test_R2',
                                'Overfit_R2', 'Train_RMSE', 'Val_RMSE', 'Test_RMSE', 'Overfit_RMSE']].copy()
    heatmap_data = heatmap_data.set_index('Model')

    # Normalize for visualization (different scales)
    # Thresholds adjusted for groundwater pumping (high uncertainty domain)
    norm_data = heatmap_data.copy()
    # For R2: higher is better, scale 0-1
    for col in ['Train_R2', 'Val_R2', 'Test_R2']:
        norm_data[col] = heatmap_data[col]  # Already 0-1
    # For Overfit_R2: lower is better (closer to 0), max 0.25 for GW
    norm_data['Overfit_R2'] = 1 - np.clip(heatmap_data['Overfit_R2'] / 0.25, 0, 1)
    # For RMSE: lower is better
    rmse_max = max(heatmap_data[['Train_RMSE', 'Val_RMSE', 'Test_RMSE']].max().max(), 50)
    for col in ['Train_RMSE', 'Val_RMSE', 'Test_RMSE']:
        norm_data[col] = 1 - (heatmap_data[col] / rmse_max)
    # For Overfit_RMSE: max 25% for GW uncertainty
    norm_data['Overfit_RMSE'] = 1 - np.clip(heatmap_data['Overfit_RMSE'] / 25, 0, 1)

    # Plot heatmap with original values as annotations
    sns.heatmap(norm_data.T, annot=heatmap_data.T, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Performance (normalized)'})

    ax.set_title('Model Performance Heatmap\n(Green = Good, Red = Poor/Overfitting)', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Model_Comparison_Heatmap.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # Figure 3: Line plot showing Train-Val-Test progression
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R2 progression
    ax = axes[0]
    for idx, row in results_df.iterrows():
        model = row['Model']
        values = [row['Train_R2'], row['Val_R2'], row['Test_R2']]
        x_vals = ['Train', 'Validation', 'Test']

        # Color based on overfitting severity (adjusted for groundwater uncertainty)
        if row['Overfit_R2'] > 0.15:
            color = '#e74c3c'
            linestyle = '--'
        elif row['Overfit_R2'] > 0.08:
            color = '#f39c12'
            linestyle = '-.'
        else:
            color = '#2ecc71'
            linestyle = '-'

        ax.plot(x_vals, values, marker='o', label=model, linewidth=2,
                markersize=8, linestyle=linestyle, alpha=0.8)

    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Progression: Train → Validation → Test\n(Declining trend = Overfitting)', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # RMSE progression
    ax = axes[1]
    for idx, row in results_df.iterrows():
        model = row['Model']
        values = [row['Train_RMSE'], row['Val_RMSE'], row['Test_RMSE']]
        x_vals = ['Train', 'Validation', 'Test']

        # Color based on overfitting severity (adjusted for groundwater uncertainty)
        if row['Overfit_RMSE'] > 15:
            color = '#e74c3c'
            linestyle = '--'
        elif row['Overfit_RMSE'] > 8:
            color = '#f39c12'
            linestyle = '-.'
        else:
            color = '#2ecc71'
            linestyle = '-'

        ax.plot(x_vals, values, marker='s', label=model, linewidth=2,
                markersize=8, linestyle=linestyle, alpha=0.8)

    ax.set_ylabel('NRMSE (%)')
    ax.set_title('NRMSE Progression: Train → Validation → Test\n(Increasing trend = Overfitting)', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Model_Comparison_Progression.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # =====================================================================
    # Figure 4: Summary ranking plot
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create composite score (lower is better)
    # Penalize both poor test performance and overfitting
    # Weights adjusted for groundwater pumping (tolerating higher uncertainty)
    results_df_sorted = results_df.copy()
    results_df_sorted['Composite_Score'] = (
        (1 - results_df_sorted['Test_R2']) * 50 +  # Weight test R2 (primary)
        results_df_sorted['Test_RMSE'] * 0.4 +     # Weight test RMSE
        np.abs(results_df_sorted['Overfit_R2']) * 60 +  # Penalize R2 overfitting (reduced for GW)
        np.abs(results_df_sorted['Overfit_RMSE']) * 1.2  # Penalize RMSE gap (reduced for GW)
    )
    results_df_sorted = results_df_sorted.sort_values('Composite_Score')

    # Create horizontal bar chart with ranking
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_df_sorted)))
    y_pos = np.arange(len(results_df_sorted))

    bars = ax.barh(y_pos, results_df_sorted['Composite_Score'], color=colors,
                   edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i+1} {m}" for i, m in enumerate(results_df_sorted['Model'])])
    ax.set_xlabel('Composite Score (Lower = Better)')
    ax.set_title('Model Ranking: Composite Score\n(Balancing Test Performance & Overfitting)', fontweight='bold')
    ax.invert_yaxis()

    # Add annotations
    for i, (bar, row) in enumerate(zip(bars, results_df_sorted.itertuples())):
        ax.annotate(f'Test R²: {row.Test_R2:.3f}, R² Gap: {row.Overfit_R2:.3f}, RMSE Gap: {row.Overfit_RMSE:.1f}%',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=8)

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Model_Ranking.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # ===============================
    # Figure 5: Combined Overfitting Analysis (R² vs RMSE)
    # ===============================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 5a: Scatter plot of R² Gap vs RMSE Gap
    ax = axes[0]

    # Color points based on combined overfitting status
    for i, row in results_df.iterrows():
        r2_gap = row['Overfit_R2']
        rmse_gap = row['Overfit_RMSE']

        # Classify based on both metrics (groundwater-adjusted thresholds)
        if r2_gap > 0.15 or rmse_gap > 15:
            color = '#e74c3c'
            status = 'Severe'
            marker = 'X'
            size = 200
        elif r2_gap > 0.08 or rmse_gap > 8:
            color = '#f39c12'
            status = 'Moderate'
            marker = 's'
            size = 150
        else:
            color = '#2ecc71'
            status = 'Good'
            marker = 'o'
            size = 120

        ax.scatter(r2_gap, rmse_gap, c=color, marker=marker, s=size,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.annotate(row['Model'], (r2_gap, rmse_gap), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')

    # Add threshold regions
    ax.axvline(x=0.08, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    ax.axvline(x=0.15, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax.axhline(y=8, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.6, linewidth=2)

    # Shade regions
    ax.axvspan(0, 0.08, alpha=0.1, color='green', label='Acceptable R²')
    ax.axhspan(0, 8, alpha=0.1, color='green', label='Acceptable RMSE')

    ax.set_xlabel('R² Gap (Train - Test)')
    ax.set_ylabel('NRMSE Gap (Test - Train) %')
    ax.set_title('Combined Overfitting Analysis: R² vs RMSE Gap\n(Groundwater Domain Thresholds)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=12, markeredgecolor='black', label='Good'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#f39c12',
               markersize=12, markeredgecolor='black', label='Moderate'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c',
               markersize=12, markeredgecolor='black', label='Severe')
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Overfitting Status')

    # Plot 5b: Dual bar chart comparing R² and RMSE overfitting
    ax = axes[1]

    x = np.arange(len(results_df))
    width = 0.35

    # Normalize both metrics to similar scale for comparison
    r2_normalized = results_df['Overfit_R2'] * 100  # Convert to percentage-like scale
    rmse_normalized = results_df['Overfit_RMSE']  # Already in percentage

    bars1 = ax.bar(x - width/2, r2_normalized, width, label='R² Gap × 100',
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, rmse_normalized, width, label='NRMSE Gap (%)',  # noqa: F841
                   color='#e67e22', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Overfitting Gap (normalized)')
    ax.set_title('R² vs NRMSE Overfitting Comparison\n(Side-by-side)', fontweight='bold')

    # Add threshold lines
    ax.axhline(y=8, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate threshold')
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Severe threshold')

    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Combined_Overfitting_Analysis.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # ===============================
    # Figure 6: Overfitting Summary Table
    # ===============================
    fig, ax = plt.subplots(figsize=(16, len(results_df) * 0.5 + 2))
    ax.axis('off')

    # Prepare summary data
    summary_data = []
    for _, row in results_df.iterrows():
        r2_gap = row['Overfit_R2']
        rmse_gap = row['Overfit_RMSE']

        # Status determination with groundwater thresholds
        if r2_gap > 0.15 or rmse_gap > 15:
            status = 'SEVERE'
        elif r2_gap > 0.08 or rmse_gap > 8:
            status = 'MODERATE'
        else:
            status = 'GOOD'

        summary_data.append([
            row['Model'],
            f"{row['Train_R2']:.3f}",
            f"{row['Val_R2']:.3f}",
            f"{row['Test_R2']:.3f}",
            f"{r2_gap:.3f}",
            f"{row['Train_RMSE']:.2f}",
            f"{row['Val_RMSE']:.2f}",
            f"{row['Test_RMSE']:.2f}",
            f"{rmse_gap:.1f}%",
            status
        ])

    columns = ['Model', 'Train R²', 'Val R²', 'Test R²', 'R² Gap',
               'Train NRMSE(%)', 'Val NRMSE(%)', 'Test NRMSE(%)', 'NRMSE Gap(%)', 'Status']

    table = ax.table(
        cellText=summary_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color cells based on status
    for i, row_data in enumerate(summary_data):
        status = row_data[-1]
        if status == 'SEVERE':
            color = '#ffcccc'
        elif status == 'MODERATE':
            color = '#fff3cd'
        else:
            color = '#d4edda'

        for j in range(len(columns)):
            table[(i + 1, j)].set_facecolor(color)

    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Model Overfitting Summary Table\n(Groundwater-adjusted thresholds: R² Gap - Moderate >0.08, Severe >0.15; RMSE Gap - Moderate >8%, Severe >15%)',
                fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'Overfitting_Summary_Table.png'), dpi=600, bbox_inches='tight')
    plt.close()

    logger.info(f'Visualization plots saved to {model_dir}')


def calc_train_test_metrics(
        pred_df: pd.DataFrame,
        cv_metric_df: pd.DataFrame,
        output_dir: str,
        use_ama_ina: bool = True,
        gw_basin_col: str = 'GW_Basin',
        year_col: str = 'Year',
        model_name: str = 'LGBM',
        precision: int = 2,
        n_features: int | None = None
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
        n_features (int or None): Number of model features for adjusted R². Falls back to column count if None.

    Returns:
        None.
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
    metric_parts = []
    for year_df, year_name in zip(year_df_list, year_df_names):
        for data_type in year_df.DATA.unique():
            data_df = year_df[year_df.DATA == data_type]
            data_actual = data_df.Actual_GW_mm.to_numpy().ravel()
            data_pred = data_df.Pred_GW_mm.to_numpy().ravel()
            r2 = r2_score(data_actual, data_pred)
            p = n_features if n_features is not None else data_df.shape[1]
            adj_r2 = adjusted_r2(data_actual, data_pred, p)
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
            metric_parts.append(temp_dict)
    metric_df = pd.DataFrame(metric_parts)
    metric_df = pd.concat([metric_df, cv_metric_df], ignore_index=True)
    metric_df = metric_df.sort_values(by=['Year', 'Data'])
    for col in ['R2', 'Adjusted_R2', 'RMSE (%)', 'MAE (%)', 'MBE (%)']:
        metric_df[col] = metric_df[col].apply(lambda x: round_to_n_nonzero(x, precision))
    metric_df.to_csv(os.path.join(output_dir, f'Error_Metrics_{model_name}.csv'), index=False)


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
    metric_parts = []
    for data in ['train', 'test']:
        if search_type == 'grid_search':
            r2 = scores[f'mean_{data}_r2'].mean()
            adj_r2 = scores[f'mean_{data}_adjusted_r2'].mean()
            rmse = -scores[f'mean_{data}_normalized_rmse'].mean()
            mae = -scores[f'mean_{data}_normalized_mae'].mean()
            mbe = -scores[f'mean_{data}_normalized_mbe'].mean()
        else:
            r2 = scores[f'{data}_r2']
            adj_r2 = scores[f'{data}_adjusted_r2']
            rmse = scores[f'{data}_normalized_rmse']
            mae = scores[f'{data}_normalized_mae']
            mbe = scores[f'{data}_normalized_mbe']
        data_name = 'VALIDATION' if data == 'test' else 'TRAIN'
        metric_parts.append({
            'Year': 'CV',
            'Data': data_name,
            'R2': r2,
            'Adjusted_R2': adj_r2,
            'RMSE (%)': rmse,
            'MAE (%)': mae,
            'MBE (%)': mbe
        })
    metric_df = pd.DataFrame(metric_parts)
    for col in ['R2', 'Adjusted_R2', 'RMSE (%)', 'MAE (%)', 'MBE (%)']:
        metric_df[col] = metric_df[col].apply(lambda x: round_to_n_nonzero(x, precision))
    logger.info(f'\n{metric_df}')
    metric_df.to_csv(metric_csv, index=False)
    return metric_df


def fit_linear_bc(
        y_pred: np.ndarray,
        y_actual: np.ndarray,
) -> tuple[float, float]:
    """Fit a linear bias correction: y_actual ≈ m * y_pred + b.

    Args:
        y_pred: Predicted values (original scale).
        y_actual: Actual/observed values (original scale).

    Returns:
        tuple[float, float]: (slope, intercept). Returns (1.0, 0.0) if
            prediction variance is zero.
    """
    pv = np.var(y_pred)
    if pv == 0:
        logger.warning(
            'Zero prediction variance in bias correction — '
            'all predictions are constant; returning identity.'
        )
        return 1.0, 0.0
    m = np.cov(y_pred, y_actual)[0, 1] / pv
    b = float(np.mean(y_actual) - m * np.mean(y_pred))
    logger.info(f'Linear BC: slope={m:.4f}, intercept={b:.4f}')
    return float(m), b


def apply_linear_bc(
        predictions: np.ndarray,
        m: float,
        b: float,
) -> np.ndarray:
    """Apply linear bias correction: corrected = |m * predictions + b|.

    Args:
        predictions: Raw predictions.
        m: Slope from fit_linear_bc.
        b: Intercept from fit_linear_bc.

    Returns:
        np.ndarray: Bias-corrected (non-negative) predictions.
    """
    return np.abs(m * predictions + b)


def perform_bias_correction(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str,
        output_dir: str,
        error_gw_col: str = 'Error_GW_mm',
        n_features: int | None = None,
) -> tuple[float, float]:
    """
    Apply linear bias correction to the model predictions.

    Fits a linear regression (y_actual = m * y_pred + b) on training data
    and returns the slope and intercept for correction.

    Args:
        train_df (pd.DataFrame): Training dataframe containing the model predictions and actual values.
        test_df (pd.DataFrame): Test dataframe containing the model predictions and actual values.
        model_name (str): Name of the ML model.
        output_dir (str): Output directory to save the files.
        error_gw_col (str): Name of the column containing the error in groundwater predictions.
        n_features (int or None): Number of model features for adjusted R².

    Returns:
        tuple[float, float, bool]: Slope, intercept, and whether BC improved all train metrics.
    """

    train_data_ecdf = train_df.copy(deep=True).drop(columns=[error_gw_col])
    test_data_ecdf = test_df.copy(deep=True).drop(columns=[error_gw_col])
    m_roe, b_roe = fit_linear_bc(
        train_data_ecdf.Pred_GW_mm.values, train_data_ecdf.Actual_GW_mm.values
    )
    train_data_ecdf['BC_GW_mm'] = apply_linear_bc(train_data_ecdf.Pred_GW_mm.values, m_roe, b_roe)
    train_r2_ecdf = r2_score(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm)
    _p = n_features if n_features is not None else train_data_ecdf.shape[1]
    train_adj_r2_ecdf = adjusted_r2(train_data_ecdf.Actual_GW_mm, train_data_ecdf.BC_GW_mm, _p)
    train_r2 = r2_score(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm)
    train_adj_r2 = adjusted_r2(train_data_ecdf.Actual_GW_mm, train_data_ecdf.Pred_GW_mm, _p)
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
    plt.xlabel('Annual Withdrawals (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ECDF_Train.png'), dpi=600)
    plt.clf()
    plt.close()
    test_data_ecdf['BC_GW_mm'] = apply_linear_bc(test_data_ecdf.Pred_GW_mm.values, m_roe, b_roe)
    plot_ecdf_test_df = test_data_ecdf.filter(like="_GW", axis="columns")
    sns.ecdfplot(data=plot_ecdf_test_df, hue_order=hue_order)
    plt.legend(labels=[f'BC{model_name}', model_name, 'Metered'], loc='lower right')
    plt.ylabel('ECDF')
    plt.xlabel('Annual Withdrawals (mm)')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'ECDF_Test.png'), dpi=600)
    plt.close()
    test_r2_ecdf = r2_score(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm)
    test_adj_r2_ecdf = adjusted_r2(test_data_ecdf.Actual_GW_mm, test_data_ecdf.BC_GW_mm, _p)
    test_r2 = r2_score(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm)
    test_adj_r2 = adjusted_r2(test_data_ecdf.Actual_GW_mm, test_data_ecdf.Pred_GW_mm, _p)
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
    metrics_df_train_linear.to_csv(os.path.join(output_dir, 'Train_Metrics_Linear.csv'), index=False)
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
    metrics_df_test_linear.to_csv(os.path.join(output_dir, 'Test_Metrics_Linear.csv'), index=False)
    # Check if BC improved all key train metrics
    bc_improved = (
        train_r2_ecdf >= train_r2
        and train_rmse_ecdf <= train_rmse
        and train_mae_ecdf <= train_mae
        and abs(train_mbe_ecdf) <= abs(train_mbe)
    )
    if not bc_improved:
        logger.warning(
            f'Bias correction did NOT improve all train metrics for {model_name}. '
            f'R2: {train_r2:.3f}→{train_r2_ecdf:.3f}, '
            f'RMSE: {train_rmse:.2f}→{train_rmse_ecdf:.2f}, '
            f'MAE: {train_mae:.2f}→{train_mae_ecdf:.2f}, '
            f'MBE: {train_mbe:.2f}→{train_mbe_ecdf:.2f}. '
            f'Skipping BC application.'
        )
    return m_roe, b_roe, bc_improved


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
        apply_bias_correction: bool = False,
        easting_train: pd.DataFrame | None = None,
        easting_test: pd.DataFrame | None = None,
        northing_train: pd.DataFrame | None = None,
        northing_test: pd.DataFrame | None = None,
        log_target: bool = False,
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
        apply_bias_correction (bool): Whether to apply global linear bias correction.
        log_target (bool): If True, inverse-transform (expm1) predictions and y values back to
            original scale before computing metrics.

    Returns:
        pd.DataFrame: Modified prediction data frame.
    """

    # PIML wrappers strip _PHYS_COL internally during predict();
    # standard models were trained without it, so strip before predict.
    if model_name not in _PIML_MODELS and isinstance(x_train, pd.DataFrame) and _PHYS_COL in x_train.columns:
        x_tr_pred = x_train.drop(columns=[_PHYS_COL])
        x_te_pred = x_test.drop(columns=[_PHYS_COL])
    else:
        x_tr_pred = x_train
        x_te_pred = x_test

    y_pred_train = model.predict(x_tr_pred)
    y_pred_test = model.predict(x_te_pred)
    if log_target:
        y_pred_train = np.expm1(y_pred_train)
        y_pred_test = np.expm1(y_pred_test)
        y_train = np.expm1(y_train)
        y_test = np.expm1(y_test)
    y_pred_train = np.abs(y_pred_train)
    y_pred_test = np.abs(y_pred_test)

    # Output DataFrames always exclude the physics column
    x_tr_out = x_train.drop(columns=[_PHYS_COL], errors='ignore') if isinstance(x_train, pd.DataFrame) else x_train
    x_te_out = x_test.drop(columns=[_PHYS_COL], errors='ignore') if isinstance(x_test, pd.DataFrame) else x_test
    if x_scaler and y_scaler:
        x_tr_out = pd.DataFrame(x_scaler.inverse_transform(x_tr_out), columns=x_tr_out.columns)
        x_te_out = pd.DataFrame(x_scaler.inverse_transform(x_te_out), columns=x_te_out.columns)
        y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_train = y_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    train_df = x_tr_out.copy() if isinstance(x_tr_out, pd.DataFrame) else pd.DataFrame(x_tr_out)
    test_df = x_te_out.copy() if isinstance(x_te_out, pd.DataFrame) else pd.DataFrame(x_te_out)
    train_df[year_col] = year_train[year_col].to_numpy().ravel()
    train_df[gw_basin_col] = gw_basin_train[gw_basin_col].to_numpy().ravel()
    if easting_train is not None:
        train_df['easting_m'] = easting_train['easting_m'].to_numpy().ravel()
    if northing_train is not None:
        train_df['northing_m'] = northing_train['northing_m'].to_numpy().ravel()
    test_df[year_col] = year_test[year_col].to_numpy().ravel()
    test_df[gw_basin_col] = gw_basin_test[gw_basin_col].to_numpy().ravel()
    if easting_test is not None:
        test_df['easting_m'] = easting_test['easting_m'].to_numpy().ravel()
    if northing_test is not None:
        test_df['northing_m'] = northing_test['northing_m'].to_numpy().ravel()
    train_df['DATA'] = ['TRAIN'] * train_df.shape[0]
    train_df['Pred_GW_mm'] = y_pred_train
    train_df['Actual_GW_mm'] = y_train
    train_df['Error_GW_mm'] = train_df['Actual_GW_mm'] - train_df['Pred_GW_mm']
    test_df['DATA'] = ['TEST'] * test_df.shape[0]
    test_df['Pred_GW_mm'] = y_pred_test
    test_df['Actual_GW_mm'] = y_test
    test_df['Error_GW_mm'] = test_df['Actual_GW_mm'] - test_df['Pred_GW_mm']
    pred_df = pd.concat([train_df, test_df])
    pred_df.to_parquet(os.path.join(model_dir, f'Predictions_{model_name}.parquet'), index=False)
    if not apply_bias_correction:
        return pred_df
    elif model_name not in ['LGBM', 'DRF', 'ETR', 'RF', 'XGB', 'XGBRF', 'HGBR',
                            'PIML_XGB', 'PIML_LGBM', 'PIML_XGBRF']:
        logger.info(f'No bias correction for {model_name} model.')
        return pred_df
    else:
        logger.info('Applying global linear bias correction...')
        output_dir = os.path.join(model_dir, f'Global_Bias_Correction_{model_name}')
        makedirs(make_proper_dir_name(output_dir))
        train_df = train_df.drop(columns=[gw_basin_col])
        test_df = test_df.drop(columns=[gw_basin_col])
        m, b, bc_improved = perform_bias_correction(
            train_df, test_df, model_name, output_dir,
            n_features=_real_n_features(x_train)
        )
        if bc_improved:
            pred_df.Pred_GW_mm = apply_linear_bc(pred_df.Pred_GW_mm.values, m, b)
            pred_df.Error_GW_mm = pred_df.Actual_GW_mm - pred_df.Pred_GW_mm
            pred_df.to_parquet(os.path.join(model_dir, f'Predictions_{model_name}_BC.parquet'), index=False)
        else:
            logger.info(f'BC skipped for {model_name} — returning original predictions.')
    return pred_df


def generate_model_visualizations(
        pred_df: pd.DataFrame,
        output_dir: str,
        model_name: str,
        test_case: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str = 'Year',
        actual_col: str = 'Actual_GW_mm',
        pred_col: str = 'Pred_GW_mm',
        gw_basin_col: str = 'GW_Basin',
        raster_res: float = 2000,
        use_ama_ina: bool = True,
        create_basin_plots: bool = True,
        n_jobs: int = -1
) -> None:
    """
    Generate journal-quality visualizations for a trained model.
    
    Creates comprehensive visualization suite including:
    - Time series plots with training/test period highlighting
    - Scatter plots of actual vs predicted (train/test)
    - Residual analysis plots
    - Individual basin-level time series
    
    All figures are saved in both PNG (600 dpi) and PDF formats for publication.
    
    Args:
        pred_df: DataFrame with predictions (must have columns: Year, GW_Basin, 
                 Actual_GW_mm, Pred_GW_mm, DATA).
        output_dir: Output directory for visualizations.
        model_name: Name of the ML model (e.g., 'XGB', 'LGBM').
        test_case: Test case identifier (e.g., 'T1', 'T2').
        test_year_limits: Tuple of tuples defining test periods, e.g., ((2019, 2024),).
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution in meters.
        use_ama_ina: Whether to filter for AMA/INA basins only.
        create_basin_plots: Whether to create individual basin plots (can be slow).
        n_jobs: Number of parallel jobs for basin plots (-1 for all cores).
        
    Returns:
        None. Plots are saved to output_dir.
    
    Example:
        >>> generate_model_visualizations(
        ...     pred_df, 
        ...     output_dir='./Plots/',
        ...     model_name='XGB',
        ...     test_case='T1',
        ...     test_year_limits=((2019, 2024),)
        ... )
    """
    vizops.create_complete_model_visualization(
        pred_df=pred_df,
        output_dir=output_dir,
        model_name=model_name,
        test_case=test_case,
        test_year_limits=test_year_limits,
        year_col=year_col,
        actual_col=actual_col,
        pred_col=pred_col,
        gw_basin_col=gw_basin_col,
        raster_res=raster_res,
        use_ama_ina=use_ama_ina,
        create_basin_plots=create_basin_plots,
        n_jobs=n_jobs
    )


def generate_model_comparison_visualization(
        model_predictions: dict[str, pd.DataFrame],
        output_dir: str,
        test_case: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str = 'Year',
        actual_col: str = 'Actual_GW_mm',
        pred_col: str = 'Pred_GW_mm',
        gw_basin_col: str = 'GW_Basin',
        raster_res: float = 2000,
        use_ama_ina: bool = True,
        unit: str = 'af'
) -> None:
    """
    Generate model comparison time series visualization.
    
    Creates a single plot comparing multiple models' predictions against observed values.
    
    Args:
        model_predictions: Dictionary mapping model names to their prediction DataFrames.
        output_dir: Output directory.
        test_case: Test case identifier.
        test_year_limits: Test period definitions.
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution.
        use_ama_ina: Filter for AMA/INA basins.
        unit: Unit for plotting ('mm', 'af', 'm3', 'ft').
        
    Returns:
        None. Plot saved to output_dir.
        
    Example:
        >>> predictions = {
        ...     'XGB': xgb_pred_df,
        ...     'LGBM': lgbm_pred_df,
        ...     'RF': rf_pred_df
        ... }
        >>> generate_model_comparison_visualization(
        ...     predictions,
        ...     output_dir='./Comparison/',
        ...     test_case='T1',
        ...     test_year_limits=((2019, 2024),)
        ... )
    """
    vizops.create_model_comparison_time_series(
        model_predictions=model_predictions,
        output_dir=output_dir,
        test_case=test_case,
        test_year_limits=test_year_limits,
        year_col=year_col,
        actual_col=actual_col,
        pred_col=pred_col,
        gw_basin_col=gw_basin_col,
        raster_res=raster_res,
        use_ama_ina=use_ama_ina,
        unit=unit
    )


# ═════════════════════════════════════════════════════════════════════════════
# Out-of-Distribution Detection
# ═════════════════════════════════════════════════════════════════════════════

class OODDetector:
    """Mahalanobis distance-based out-of-distribution detector.

    Fits on training features and flags prediction-time pixels whose feature
    vectors fall outside the training distribution.  The Mahalanobis distance
    is computed using the training covariance matrix (regularized for numerical
    stability).

    Attributes:
        mean_ (np.ndarray): Training feature means (n_features,).
        cov_inv_ (np.ndarray): Inverse of the regularized training covariance matrix.
        threshold_ (float): Chi-squared threshold at the chosen significance level.
        n_features_ (int): Number of features.
    """

    def __init__(self, alpha: float = 0.01, reg: float = 1e-6):
        """
        Args:
            alpha (float): Significance level for the chi-squared threshold.
                Pixels with Mahalanobis distance exceeding the (1 - alpha)
                quantile of chi-squared(n_features) are flagged as OOD.
            reg (float): Tikhonov regularisation added to the covariance
                diagonal for numerical stability.
        """
        self.alpha = alpha
        self.reg = reg
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None
        self.n_features_ = None

    def fit(self, x_train: np.ndarray | pd.DataFrame) -> 'OODDetector':
        """Fit the detector on training features.

        Args:
            x_train: Training feature matrix of shape (n_samples, n_features).

        Returns:
            OODDetector: self.
        """
        from scipy.stats import chi2

        x = np.asarray(x_train, dtype=np.float64)
        n_nan = np.isnan(x).sum()
        if n_nan:
            logger.warning(
                'OODDetector.fit(): %d NaN values in x_train (%s); '
                'dropping rows with NaNs before covariance estimation.',
                n_nan, x.shape,
            )
            x = x[~np.isnan(x).any(axis=1)]
        self.n_features_ = x.shape[1]
        self.mean_ = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False)
        # Regularise for numerical stability
        cov += np.eye(self.n_features_) * self.reg
        self.cov_inv_ = np.linalg.inv(cov)
        self.threshold_ = chi2.ppf(1 - self.alpha, df=self.n_features_)
        logger.info(
            'OOD detector fitted on %d samples, %d features.  '
            'Chi-squared threshold (alpha=%.3f): %.2f',
            x.shape[0], self.n_features_, self.alpha, self.threshold_,
        )
        return self

    def mahalanobis(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute Mahalanobis distances for new data.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Squared Mahalanobis distances of shape (n_samples,).
        """
        x = np.asarray(x, dtype=np.float64)
        diff = x - self.mean_
        # Efficient batch computation: d² = diag(diff @ cov_inv @ diff.T)
        left = diff @ self.cov_inv_
        return np.einsum('ij,ij->i', left, diff)

    def is_ood(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Flag OOD samples.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Boolean array where True indicates OOD samples.
        """
        return self.mahalanobis(x) > self.threshold_

    def score_and_summarise(
        self,
        x: np.ndarray | pd.DataFrame,
        year: int | None = None,
    ) -> dict:
        """Compute OOD statistics for a batch of prediction features.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            year (int | None): Year label for logging.

        Returns:
            dict: Keys: 'n_total', 'n_ood', 'pct_ood', 'mean_d2', 'max_d2'.
        """
        d2 = self.mahalanobis(x)
        ood_mask = d2 > self.threshold_
        n_total = len(d2)
        n_ood = int(ood_mask.sum())
        pct_ood = n_ood / n_total * 100 if n_total > 0 else 0.0
        stats = {
            'n_total': n_total,
            'n_ood': n_ood,
            'pct_ood': round(pct_ood, 2),
            'mean_d2': round(float(np.mean(d2)), 2),
            'max_d2': round(float(np.max(d2)), 2),
            'threshold': round(self.threshold_, 2),
        }
        label = f'Year {year}' if year is not None else 'Batch'
        if pct_ood > 10:
            logger.warning(
                '%s: %.1f%% of pixels (%d/%d) are out-of-distribution '
                '(Mahalanobis d² > %.1f)',
                label, pct_ood, n_ood, n_total, self.threshold_,
            )
        elif pct_ood > 0:
            logger.info(
                '%s: %.1f%% of pixels (%d/%d) are out-of-distribution',
                label, pct_ood, n_ood, n_total,
            )
        return stats
