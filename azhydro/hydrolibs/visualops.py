"""
Visualization operations for journal-quality figures.

This module provides comprehensive visualization functions for:
- Time series plots with training/test period highlighting
- Model comparison plots
- Confidence interval visualizations
- Basin-level and aggregate analysis plots
"""

# Author: Dr. Sayantan Majumdar
# Email: sayantan.majumdar@dri.edu

import logging
import multiprocessing
import os
from typing import Any

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats

from hydrolibs.sysops import makedirs

logger = logging.getLogger(__name__)


# Journal-quality plot settings
JOURNAL_SETTINGS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# Color palette for consistent styling
COLORS = {
    'actual': '#E74C3C',       # Red for actual/observed
    'predicted': '#2C3E50',    # Dark blue for predicted
    'train_shade': '#3498DB',  # Light blue for training period
    'test_shade': '#F39C12',   # Orange for test period
    'ci_actual': '#FADBD8',    # Light red for CI
    'ci_predicted': '#D5DBDB', # Light gray for CI
}

# ── Unit-conversion constants for twinx axes ─────────────────────────────
_MM_TO_FT = 1.0 / 304.8
_AF_TO_M3 = 1233.48   # 1 AF = 1233.48 m³


def _add_ft_twinx(ax):
    """Add a ft twinx to a mm depth axis. Call after all plotting."""
    ax_ft = ax.twinx()
    ax_ft.set_ylabel('(ft)', fontweight='bold')
    lo, hi = ax.get_ylim()
    ax_ft.set_ylim(lo * _MM_TO_FT, hi * _MM_TO_FT)
    return ax_ft


def _add_m3_twinx(ax):
    """Add a m³ twinx to an AF volume axis. Call after all plotting."""
    ax_m3 = ax.twinx()
    ax_m3.set_ylabel('(m³)', fontweight='bold')
    lo, hi = ax.get_ylim()
    ax_m3.set_ylim(lo * _AF_TO_M3, hi * _AF_TO_M3)
    return ax_m3


def get_ama_ina_basin_names() -> list[str]:
    """
    Get the names of AMA and INA basins.

    Returns:
        list: List of AMA and INA basin names.
    """
    ama_ina_basins = [
        'SANTA CRUZ AMA',
        'PRESCOTT AMA',
        'TUCSON AMA',
        'PINAL AMA',
        'PHOENIX AMA',
        'DOUGLAS AMA',
        'JOSEPH CITY INA',
        'HARQUAHALA INA',
        'HUALAPAI VALLEY INA',
        'WILLCOX AMA'
    ]
    return ama_ina_basins


def apply_journal_style():
    """Apply journal-quality matplotlib settings."""
    plt.rcParams.update(JOURNAL_SETTINGS)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)


def compute_confidence_interval(
        data: pd.Series | np.ndarray,
        confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute confidence interval for data.
    
    Args:
        data: Input data array or series.
        confidence: Confidence level (default 0.95 for 95% CI).
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci).
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    se = stats.sem(data) if len(data) > 1 else 0
    ci = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1) if len(data) > 1 else 0
    return mean, mean - ci, mean + ci


def _basin_block_bootstrap(
        values: np.ndarray,
        basin_labels: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42
) -> tuple[float, float]:
    """
    Basin-level block bootstrap CI for the sum statistic.

    Resamples entire basins with replacement to respect within-basin
    spatial autocorrelation. Each bootstrap replicate draws n_basins
    basins (with replacement), sums their per-basin totals, and the
    percentile interval is computed from the bootstrap distribution.

    Args:
        values: Per-pixel values for one year.
        basin_labels: Corresponding basin label for each pixel.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_ci, upper_ci).
    """
    unique_basins = np.unique(basin_labels)
    n_basins = len(unique_basins)
    basin_sums = np.array([np.sum(values[basin_labels == b]) for b in unique_basins])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_basins, size=(n_bootstrap, n_basins))
    bootstrap_totals = basin_sums[idx].sum(axis=1)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_totals, alpha / 2 * 100)
    upper = np.percentile(bootstrap_totals, (1 - alpha / 2) * 100)
    return lower, upper


def aggregate_yearly_data(
        df: pd.DataFrame,
        year_col: str,
        value_col: str,
        aggregation: str = 'sum',
        confidence: float = 0.95,
        basin_col: str | None = None
) -> pd.DataFrame:
    """
    Aggregate data by year with confidence intervals.

    When ``basin_col`` is provided and aggregation is ``'sum'``, a
    basin-level block bootstrap is used instead of pixel-level i.i.d.
    resampling.  This accounts for within-basin spatial autocorrelation
    and produces more conservative (wider) CIs suitable for publication.

    Args:
        df: Input dataframe.
        year_col: Name of year column.
        value_col: Name of value column to aggregate.
        aggregation: Aggregation method ('sum', 'mean').
        confidence: Confidence level for CI.
        basin_col: Optional basin column for block bootstrap.
            When provided with aggregation='sum', basins are resampled
            as spatial blocks instead of individual pixels.

    Returns:
        DataFrame with year, mean, lower_ci, upper_ci columns.
    """
    years = sorted(df[year_col].unique())
    results = []
    rng = np.random.default_rng(42)

    for year in years:
        year_mask = df[year_col] == year
        year_data = df.loc[year_mask, value_col].values

        if aggregation == 'sum':
            total = np.sum(year_data)

            if basin_col is not None:
                basin_labels = df.loc[year_mask, basin_col].values
                lower, upper = _basin_block_bootstrap(
                    year_data, basin_labels, confidence=confidence
                )
            else:
                # Pixel-level bootstrap (fallback for per-basin plots)
                n_bootstrap = 1000
                idx = rng.integers(0, len(year_data), size=(n_bootstrap, len(year_data)))
                bootstrap_sums = year_data[idx].sum(axis=1)
                lower = np.percentile(bootstrap_sums, (1 - confidence) / 2 * 100)
                upper = np.percentile(bootstrap_sums, (1 + confidence) / 2 * 100)

            results.append({
                'year': year,
                'value': total,
                'lower_ci': lower,
                'upper_ci': upper
            })
        else:
            mean, lower, upper = compute_confidence_interval(year_data, confidence)
            results.append({
                'year': year,
                'value': mean,
                'lower_ci': lower,
                'upper_ci': upper
            })

    return pd.DataFrame(results)


def create_time_series_plot_journal(
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
        aggregation: str = 'sum',
        confidence: float = 0.95,
        units: list[str] = None,
        figsize: tuple[float, float] = (12, 6),
        split_strategy: int = 1
) -> None:
    """
    Create journal-quality time series plot with training/test period highlighting.
    
    Args:
        pred_df: DataFrame with predictions containing year, actual, and predicted columns.
        output_dir: Output directory for saving plots.
        model_name: Name of the ML model.
        test_case: Test case identifier (e.g., 'T1', 'T2').
        test_year_limits: Tuple of tuples defining test periods.
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution in meters.
        use_ama_ina: Whether to filter for AMA/INA basins only.
        aggregation: Aggregation method ('sum' or 'mean').
        confidence: Confidence level for CI (default 0.95).
        units: List of units to plot ['mm', 'af', 'm3'].
        figsize: Figure size.
        split_strategy: Split strategy (1=temporal, 2=random stratified, 3=spatial).
    """
    makedirs(output_dir)
    apply_journal_style()

    if units is None:
        units = ['mm', 'af', 'm3', 'ft']

    # Filter for AMA/INA if requested
    if use_ama_ina:
        ama_ina_basins = get_ama_ina_basin_names()
        pred_df = pred_df[pred_df[gw_basin_col].isin(ama_ina_basins)].copy()

    # Get all test years
    test_years = set()
    for start, end in test_year_limits:
        test_years.update(range(start, end + 1))

    # Add data type column
    pred_df['DataType'] = pred_df[year_col].apply(
        lambda x: 'Test' if x in test_years else 'Train'
    )

    # Pixel area for unit conversions
    area = raster_res ** 2

    # Unit conversion factors and labels
    unit_info = {
        'mm': {
            'factor': 1.0,
            'label': 'Groundwater Withdrawals (mm)',
            'agg_label': 'Total Groundwater Withdrawals (mm)' if aggregation == 'sum'
                        else 'Mean Groundwater Withdrawals (mm)'
        },
        'af': {
            'factor': area / (4047 * 304.8 * 1000),  # mm to 1000s acre-ft
            'label': 'Groundwater Withdrawals (1000s acre-ft)',
            'agg_label': 'Total Groundwater Withdrawals (1000s acre-ft)' if aggregation == 'sum'
                        else 'Mean Groundwater Withdrawals (1000s acre-ft)'
        },
        'm3': {
            'factor': area * 1e-9,  # mm to 1e6 m³
            'label': 'Groundwater Withdrawals (10⁶ m³)',
            'agg_label': 'Total Groundwater Withdrawals (10⁶ m³)' if aggregation == 'sum'
                        else 'Mean Groundwater Withdrawals (10⁶ m³)'
        },
        'ft': {
            'factor': 1 / 304.8,  # mm to ft
            'label': 'Groundwater Withdrawals (ft)',
            'agg_label': 'Total Groundwater Withdrawals (ft)' if aggregation == 'sum'
                        else 'Mean Groundwater Withdrawals (ft)'
        }
    }

    for unit in units:
        if unit not in unit_info:
            continue

        factor = unit_info[unit]['factor']
        ylabel = unit_info[unit]['agg_label']

        # Convert units
        df_plot = pred_df.copy()
        df_plot[f'Actual_{unit}'] = df_plot[actual_col] * factor
        df_plot[f'Pred_{unit}'] = df_plot[pred_col] * factor

        # Aggregate by year (basin block bootstrap for AZ-wide sums)
        actual_agg = aggregate_yearly_data(
            df_plot, year_col, f'Actual_{unit}', aggregation, confidence,
            basin_col=gw_basin_col
        )
        pred_agg = aggregate_yearly_data(
            df_plot, year_col, f'Pred_{unit}', aggregation, confidence,
            basin_col=gw_basin_col
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        years = actual_agg['year'].values
        min_year, max_year = years.min(), years.max()

        # Shade test periods
        for start, end in test_year_limits:
            ax.axvspan(start - 0.5, end + 0.5,
                      alpha=0.2, color=COLORS['test_shade'],
                      label='Test Period' if start == test_year_limits[0][0] else '')

        # Plot confidence intervals as shaded regions
        ax.fill_between(
            actual_agg['year'],
            actual_agg['lower_ci'],
            actual_agg['upper_ci'],
            alpha=0.3,
            color=COLORS['actual'],
            label=f'{int(confidence*100)}% CI (Observed)'
        )

        ax.fill_between(
            pred_agg['year'],
            pred_agg['lower_ci'],
            pred_agg['upper_ci'],
            alpha=0.3,
            color=COLORS['predicted'],
            label=f'{int(confidence*100)}% CI (Predicted)'
        )

        # Plot lines
        ax.plot(
            actual_agg['year'],
            actual_agg['value'],
            color=COLORS['actual'],
            marker='o',
            markersize=6,
            linewidth=2,
            label='Observed'
        )

        ax.plot(
            pred_agg['year'],
            pred_agg['value'],
            color=COLORS['predicted'],
            marker='s',
            markersize=6,
            linewidth=2,
            label='Predicted'
        )

        # Formatting
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{model_name} - {test_case}: Groundwater Withdrawals Time Series',
                    fontweight='bold', fontsize=14)

        # Set x-axis ticks
        year_range = max_year - min_year
        if year_range > 20:
            tick_interval = 5
        elif year_range > 10:
            tick_interval = 3
        else:
            tick_interval = 2

        ax.set_xticks(np.arange(min_year, max_year + 1, tick_interval))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        # Reorder legend
        order = [labels.index('Observed'), labels.index('Predicted'),
                 labels.index(f'{int(confidence*100)}% CI (Observed)'),
                 labels.index(f'{int(confidence*100)}% CI (Predicted)'),
                 labels.index('Test Period') if 'Test Period' in labels else -1]
        order = [o for o in order if o >= 0]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                 loc='upper left', framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(min_year - 0.5, max_year + 0.5)

        # Save figure
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'TS_{model_name}_{test_case}_{unit}_{aggregation}.png'),
                   dpi=600, bbox_inches='tight')
        plt.close()


def create_basin_time_series_plot(
        pred_df: pd.DataFrame,
        output_dir: str,
        model_name: str,
        test_case: str,
        basin_name: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str = 'Year',
        actual_col: str = 'Actual_GW_mm',
        pred_col: str = 'Pred_GW_mm',
        gw_basin_col: str = 'GW_Basin',
        raster_res: float = 2000,
        aggregation: str = 'sum',
        confidence: float = 0.95,
        unit: str = 'af',
        figsize: tuple[float, float] = (12, 6)
) -> None:
    """
    Create time series plot for a specific groundwater basin.
    
    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        basin_name: Name of the groundwater basin.
        test_year_limits: Test period definitions.
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution in meters.
        aggregation: Aggregation method.
        confidence: Confidence level.
        unit: Unit for plotting ('mm', 'af', 'm3', 'ft').
        figsize: Figure size.
    """
    makedirs(output_dir)
    apply_journal_style()

    # Filter for basin
    basin_df = pred_df[pred_df[gw_basin_col] == basin_name].copy()

    if basin_df.empty:
        logger.info(f'No data for basin: {basin_name}')
        return

    # Pixel area for unit conversions
    area = raster_res ** 2

    # Unit conversion
    unit_factors = {
        'mm': (1.0, 'mm'),
        'af': (area / (4047 * 304.8 * 1000), '1000s acre-ft'),
        'm3': (area * 1e-9, '10⁶ m³'),
        'ft': (1 / 304.8, 'ft')
    }

    factor, unit_label = unit_factors.get(unit, (1.0, 'mm'))

    basin_df[f'Actual_{unit}'] = basin_df[actual_col] * factor
    basin_df[f'Pred_{unit}'] = basin_df[pred_col] * factor

    # Aggregate by year
    actual_agg = aggregate_yearly_data(
        basin_df, year_col, f'Actual_{unit}', aggregation, confidence
    )
    pred_agg = aggregate_yearly_data(
        basin_df, year_col, f'Pred_{unit}', aggregation, confidence
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Shade test periods
    for start, end in test_year_limits:
        ax.axvspan(start - 0.5, end + 0.5,
                  alpha=0.2, color=COLORS['test_shade'],
                  label='Test Period' if start == test_year_limits[0][0] else '')

    # Plot confidence intervals
    ax.fill_between(
        actual_agg['year'],
        actual_agg['lower_ci'],
        actual_agg['upper_ci'],
        alpha=0.3,
        color=COLORS['actual']
    )

    ax.fill_between(
        pred_agg['year'],
        pred_agg['lower_ci'],
        pred_agg['upper_ci'],
        alpha=0.3,
        color=COLORS['predicted']
    )

    # Plot lines
    ax.plot(
        actual_agg['year'],
        actual_agg['value'],
        color=COLORS['actual'],
        marker='o',
        markersize=6,
        linewidth=2,
        label='Observed'
    )

    ax.plot(
        pred_agg['year'],
        pred_agg['value'],
        color=COLORS['predicted'],
        marker='s',
        markersize=6,
        linewidth=2,
        label='Predicted'
    )

    # Formatting
    agg_label = 'Total' if aggregation == 'sum' else 'Mean'
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel(f'{agg_label} Groundwater Withdrawals ({unit_label})', fontweight='bold')
    ax.set_title(f'{basin_name}\n{model_name} - {test_case}', fontweight='bold')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Clean basin name for filename
    basin_clean = basin_name.replace(' ', '_').replace('/', '_')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'TS_{model_name}_{test_case}_{basin_clean}_{unit}.png'),
               dpi=600, bbox_inches='tight')
    plt.close()


def create_all_basin_plots_parallel(
        pred_df: pd.DataFrame,
        output_dir: str,
        model_name: str,
        test_case: str,
        test_year_limits: tuple[tuple[int, int], ...],
        gw_basin_col: str = 'GW_Basin',
        raster_res: float = 2000,
        unit: str = 'af',
        n_jobs: int = -1
) -> None:
    """
    Create time series plots for all basins in parallel.
    
    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        test_year_limits: Test period definitions.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution.
        unit: Unit for plotting.
        n_jobs: Number of parallel jobs (-1 for all cores).
    """
    basins = pred_df[gw_basin_col].unique().tolist()
    basin_dir = os.path.join(output_dir, 'Basins')
    makedirs(basin_dir)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1

    Parallel(n_jobs=n_jobs)(delayed(create_basin_time_series_plot)(
        pred_df, basin_dir, model_name, test_case, basin,
        test_year_limits, gw_basin_col=gw_basin_col,
        raster_res=raster_res, unit=unit
    ) for basin in basins)


def create_model_comparison_time_series(
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
        unit: str = 'af',
        figsize: tuple[float, float] = (14, 8)
) -> None:
    """
    Create comparison time series plot showing multiple models.
    
    Args:
        model_predictions: Dictionary mapping model names to prediction DataFrames.
        output_dir: Output directory.
        test_case: Test case identifier.
        test_year_limits: Test period definitions.
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution.
        use_ama_ina: Filter for AMA/INA basins.
        unit: Unit for plotting.
        figsize: Figure size.
    """
    makedirs(output_dir)
    apply_journal_style()

    # Unit conversion
    area = raster_res ** 2
    unit_factors = {
        'mm': (1.0, 'mm'),
        'af': (area / (4047 * 304.8 * 1000), '1000s acre-ft'),
        'm3': (area * 1e-9, '10⁶ m³'),
        'ft': (1 / 304.8, 'ft')
    }
    factor, unit_label = unit_factors.get(unit, (1.0, 'mm'))

    # Color palette for models
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual (from first model's data)
    first_model = list(model_predictions.keys())[0]
    first_df = model_predictions[first_model].copy()

    if use_ama_ina:
        ama_ina_basins = get_ama_ina_basin_names()
        first_df = first_df[first_df[gw_basin_col].isin(ama_ina_basins)]

    first_df[f'Actual_{unit}'] = first_df[actual_col] * factor
    actual_agg = aggregate_yearly_data(
        first_df, year_col, f'Actual_{unit}', 'sum', 0.95,
        basin_col=gw_basin_col
    )

    years = actual_agg['year'].values
    min_year, max_year = years.min(), years.max()

    # Shade test periods
    for start, end in test_year_limits:
        ax.axvspan(start - 0.5, end + 0.5,
                  alpha=0.15, color=COLORS['test_shade'],
                  label='Test Period' if start == test_year_limits[0][0] else '')

    # Plot actual values
    ax.plot(
        actual_agg['year'],
        actual_agg['value'],
        color=COLORS['actual'],
        marker='o',
        markersize=8,
        linewidth=2.5,
        label='Observed',
        zorder=10
    )

    # Plot each model's predictions
    for idx, (model_name, pred_df) in enumerate(model_predictions.items()):
        df = pred_df.copy()
        if use_ama_ina:
            df = df[df[gw_basin_col].isin(ama_ina_basins)]

        df[f'Pred_{unit}'] = df[pred_col] * factor
        pred_agg = aggregate_yearly_data(
            df, year_col, f'Pred_{unit}', 'sum', 0.95,
            basin_col=gw_basin_col
        )

        ax.plot(
            pred_agg['year'],
            pred_agg['value'],
            color=model_colors[idx],
            marker='s',
            markersize=5,
            linewidth=1.5,
            alpha=0.8,
            label=model_name
        )

    # Formatting
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel(f'Total Groundwater Withdrawals ({unit_label})', fontweight='bold')
    ax.set_title(f'{test_case}: Model Comparison - Groundwater Withdrawals',
                fontweight='bold', fontsize=14)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(min_year - 0.5, max_year + 0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'TS_Model_Comparison_{test_case}_{unit}.png'),
               dpi=600, bbox_inches='tight')
    plt.close()


def create_train_test_scatter(
        pred_df: pd.DataFrame,
        output_dir: str,
        model_name: str,
        test_case: str,
        actual_col: str = 'Actual_GW_mm',
        pred_col: str = 'Pred_GW_mm',
        data_col: str = 'DATA',
        figsize: tuple[float, float] = (12, 5)
) -> None:
    """
    Create scatter plots comparing actual vs predicted for train and test data.
    
    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        data_col: Name of data type column (TRAIN/TEST).
        figsize: Figure size.
    """
    makedirs(output_dir)
    apply_journal_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, (data_type, color) in enumerate([('TRAIN', COLORS['train_shade']),
                                               ('TEST', COLORS['test_shade'])]):
        ax = axes[idx]
        df = pred_df[pred_df[data_col] == data_type]

        if df.empty:
            continue

        ax.scatter(
            df[actual_col],
            df[pred_col],
            alpha=0.5,
            s=20,
            c=color,
            edgecolors='white',
            linewidths=0.5
        )

        # 1:1 line
        max_val = max(df[actual_col].max(), df[pred_col].max())
        min_val = min(df[actual_col].min(), df[pred_col].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 Line')

        # Regression line
        z = np.polyfit(df[actual_col], df[pred_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), color='red', linewidth=1.5, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(df[actual_col], df[pred_col])

        ax.set_xlabel('Observed (mm)', fontweight='bold')
        ax.set_ylabel('Predicted (mm)', fontweight='bold')
        ax.set_title(f'{data_type} Data (R² = {r2:.3f})', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(f'{model_name} - {test_case}', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, f'Scatter_{model_name}_{test_case}.png'),
               dpi=600, bbox_inches='tight')
    plt.close()


def create_residual_plot(
        pred_df: pd.DataFrame,
        output_dir: str,
        model_name: str,
        test_case: str,
        actual_col: str = 'Actual_GW_mm',
        pred_col: str = 'Pred_GW_mm',
        year_col: str = 'Year',
        figsize: tuple[float, float] = (12, 5)
) -> None:
    """
    Create residual analysis plots.
    
    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        year_col: Name of year column.
        figsize: Figure size.
    """
    makedirs(output_dir)
    apply_journal_style()

    pred_df = pred_df.copy()
    pred_df['Residual'] = pred_df[actual_col] - pred_df[pred_col]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residual vs Predicted
    ax = axes[0]
    ax.scatter(
        pred_df[pred_col],
        pred_df['Residual'],
        alpha=0.5,
        s=20,
        c=COLORS['predicted'],
        edgecolors='white'
    )
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Predicted (mm)', fontweight='bold')
    ax.set_ylabel('Residual (mm)', fontweight='bold')
    ax.set_title('Residuals vs Predicted', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax = axes[1]
    ax.hist(
        pred_df['Residual'],
        bins=50,
        color=COLORS['predicted'],
        edgecolor='white',
        alpha=0.7
    )
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Residual (mm)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} - {test_case}', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, f'Residuals_{model_name}_{test_case}.png'),
               dpi=600, bbox_inches='tight')
    plt.close()


def create_complete_model_visualization(
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
    Create complete visualization suite for a model.
    
    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        test_year_limits: Test period definitions.
        year_col: Name of year column.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution.
        use_ama_ina: Filter for AMA/INA basins.
        create_basin_plots: Whether to create individual basin plots.
        n_jobs: Number of parallel jobs.
    """
    logger.info(f'Creating visualizations for {model_name} - {test_case}...')
    makedirs(output_dir)

    # 1. Aggregate time series plot
    logger.info('  Creating time series plots...')
    create_time_series_plot_journal(
        pred_df, output_dir, model_name, test_case, test_year_limits,
        year_col=year_col, actual_col=actual_col, pred_col=pred_col,
        gw_basin_col=gw_basin_col, raster_res=raster_res,
        use_ama_ina=use_ama_ina, units=['af', 'm3', 'mm']
    )

    # 2. Scatter plots
    logger.info('  Creating scatter plots...')
    create_train_test_scatter(
        pred_df, output_dir, model_name, test_case,
        actual_col=actual_col, pred_col=pred_col
    )

    # 3. Residual plots
    logger.info('  Creating residual plots...')
    create_residual_plot(
        pred_df, output_dir, model_name, test_case,
        actual_col=actual_col, pred_col=pred_col, year_col=year_col
    )

    # 4. Basin-level plots (parallel)
    if create_basin_plots:
        logger.info('  Creating basin-level plots...')
        create_all_basin_plots_parallel(
            pred_df, output_dir, model_name, test_case, test_year_limits,
            gw_basin_col=gw_basin_col, raster_res=raster_res, n_jobs=n_jobs
        )

    logger.info(f'  Visualizations saved to: {output_dir}')


# Legacy function wrappers for backward compatibility
def parallel_make_time_series_plots(
        idx: int,
        gw_basin: str,
        input_df: pd.DataFrame,
        output_dir: str,
        test_year_limits: tuple[tuple[int, int], ...],
        year_col: str,
        actual_gw_col: str,
        pred_gw_col: str,
        gw_basin_col: str,
        split_strategy: int,
        test_gw_basins: tuple[str],
        raster_res: float = 2000
) -> None:
    """Legacy wrapper for backward compatibility."""
    if idx == 0:
        gw_basin_name = 'AMA_INA'
        basin_df = input_df[input_df[gw_basin_col].isin(gw_basin)].copy()
    else:
        gw_basin_name = gw_basin
        basin_df = input_df[input_df[gw_basin_col] == gw_basin].copy()

    # Create prediction dataframe format
    pred_df = basin_df.rename(columns={
        actual_gw_col: 'Actual_GW_mm',
        pred_gw_col: 'Pred_GW_mm'
    })
    pred_df['DATA'] = 'TRAIN'  # Will be updated based on test years

    test_years = set()
    for start, end in test_year_limits:
        test_years.update(range(start, end + 1))
    pred_df.loc[pred_df[year_col].isin(test_years), 'DATA'] = 'TEST'

    create_basin_time_series_plot(
        pred_df, output_dir, 'Model', 'Analysis',
        gw_basin_name, test_year_limits,
        year_col=year_col, gw_basin_col=gw_basin_col,
        raster_res=raster_res, unit='af'
    )


def make_time_series_plots(
        input_df: pd.DataFrame,
        model: Any,
        features: list[str],
        output_dir: str,
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        test_year_limits: tuple[tuple[int, int], ...] = ((2000, 2010),),
        pred_attr: str = 'gw_pumping_mm',
        split_strategy: int = 1,
        test_gw_basins: tuple[str, ...] = ('HARQUAHALA INA',),
        raster_res: float = 2000,
        x_scaler: Any = None,
        y_scaler: Any = None
) -> None:
    """Legacy wrapper for backward compatibility."""
    logger.info('Creating time series plots...')
    makedirs(output_dir)

    actual_gw_col = 'Actual_GW_mm'
    pred_gw_col = 'Pred_GW_mm'

    df = input_df.copy()
    df = df.rename(columns={pred_attr: actual_gw_col})

    if x_scaler:
        df[features] = x_scaler.transform(df[features])

    model_predictions = model.predict(df[features])

    if y_scaler:
        df[pred_gw_col] = np.abs(y_scaler.inverse_transform(
            model_predictions.reshape(-1, 1)).ravel())
    else:
        df[pred_gw_col] = np.abs(model_predictions)

    # Add DATA column
    test_years = set()
    for start, end in test_year_limits:
        test_years.update(range(start, end + 1))
    df['DATA'] = df[year_col].apply(lambda x: 'TEST' if x in test_years else 'TRAIN')

    create_time_series_plot_journal(
        df, output_dir, 'Model', 'Analysis', test_year_limits,
        year_col=year_col, actual_col=actual_gw_col, pred_col=pred_gw_col,
        gw_basin_col=gw_basin_col, raster_res=raster_res, use_ama_ina=True,
        units=['af', 'm3', 'mm']
    )


def get_variable_name_dict() -> dict[str, str]:
    """
    Returns a dictionary mapping variable names to more descriptive labels for plotting.
    """

    var_name_dict = {
        'gw_pumping_mm': 'Annual Groundwater Withdrawals (mm)',
        'annual_et_ensemble_mm': 'Annual ET (mm)',
        'annual_eto_mm': 'Annual ETo (mm)',
        'annual_precip_mm': 'Annual Precipitation (mm)',
        'annual_peff_mm': 'Annual USDA-SCS Effective Precipitation (mm)',
        'annual_peff_pcml_mm': 'Annual PCML Effective Precipitation (mm)',
        'annual_tmmx_K': 'Annual Maximum Air Temperature (K)',
        'annual_tmmn_K': 'Annual Minimum Air Temperature (K)',
        'AGRI': 'Agricultural Density',
        'URBAN': 'Urban Density',
        'SW': 'Surface Water Density',
        'streamflow_mm': 'Streamflow (mm)',
        'gw_basin_type': 'Groundwater Basin Type',
        'GW_Basin': 'Groundwater Basin',
        'soil_depth_mm': 'Soil Depth (mm)',
        'awc_mm': 'Available Water Capacity (mm)',
        'ksat_mean_micromps': 'Mean Saturated Hydraulic Conductivity (µm/s)',
        'annual_gw_fraction': 'Annual Groundwater Irrigation Fraction',
        'annual_crop_fraction': 'Annual Crop Fraction',
        'annual_irr_fraction': 'Annual Irrigated Fraction',
    }
    return var_name_dict


# ─── Exploratory data analysis ───────────────────────────────────────────────

# Period definitions for era-based coloring/shading
ERA_PERIODS = {
    'Hindcast':    (1896, 1983),
    'Historical':  (1984, 2024),
    'Forecast':    (2025, 2025),
    'Projection':  (2026, 2099),
}

ERA_COLORS = {
    'Hindcast':   '#8E44AD',   # Purple
    'Historical': '#2980B9',   # Blue
    'Forecast':   '#E67E22',   # Orange
    'Projection': '#27AE60',   # Green
}


def _assign_era(year: int) -> str:
    """Map a year to its era label."""
    for era, (start, end) in ERA_PERIODS.items():
        if start <= year <= end:
            return era
    return 'Other'


def explore_az_data(
        az_df: pd.DataFrame,
        output_dir: str,
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        basin_type_col: str = 'GW_Basin_Type',
        skip_cols: tuple[str, ...] = ('easting_m', 'northing_m'),
        figsize_ts: tuple[float, float] = (14, 5),
        figsize_box: tuple[float, float] = (14, 6),
) -> None:
    """
    Exploratory visualizations (boxplots, violin plots, time series) for all
    numeric columns in *az_df*, grouped by GW_Basin_Type and GW_Basin.

    Four eras are highlighted:
        Hindcast 1896-1983 | Historical 1985-2024 | Forecast 2025 | Projection 2026-2099

    Args:
        az_df: Arizona predictor dataframe produced by ``create_az_data_parquet``.
        output_dir: Directory where plots are saved.
        year_col: Name of the year column.
        gw_basin_col: Name of the groundwater basin column.
        basin_type_col: Name of the basin type column (0=AMA, 1=INA, 2=Other).
        skip_cols: Columns to skip in the visualizations.
        figsize_ts: Figure size for time series plots.
        figsize_box: Figure size for box/violin plots.
    """
    makedirs(output_dir)
    apply_journal_style()

    df = az_df.copy()
    df['Era'] = df[year_col].apply(_assign_era)

    basin_type_map = {0: 'AMA', 1: 'INA', 2: 'Other'}
    df['Basin_Type_Label'] = df[basin_type_col].map(basin_type_map)

    numeric_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in (year_col, basin_type_col) and c not in skip_cols
    ]

    era_order = list(ERA_PERIODS.keys())
    era_palette = [ERA_COLORS[e] for e in era_order]

    logger.info(f'Generating exploratory plots for {len(numeric_cols)} columns …')

    ama_ina_basins = get_ama_ina_basin_names()
    var_name_dict = get_variable_name_dict()

    for col in numeric_cols:
        safe = col.replace('/', '_')
        label = var_name_dict.get(col, col)

        # For gw_pumping_mm restrict to 1984-2024 (metered years) and AMA/INA
        if col == 'gw_pumping_mm':
            col_df = df[(df[year_col].between(1984, 2024)) &
                        (df[gw_basin_col].isin(ama_ina_basins))].copy()
        else:
            col_df = df

        # Exclude zero values before plotting
        col_df = col_df[col_df[col] > 0]

        # ── 1. Time series (mean ± std per year), shaded by era ──────────
        yearly = col_df.groupby(year_col)[col].agg(['mean', 'std']).reset_index()
        yearly['Era'] = yearly[year_col].apply(_assign_era)

        fig, ax = plt.subplots(figsize=figsize_ts)
        for era in era_order:
            mask = yearly['Era'] == era
            if not mask.any():
                continue
            sub = yearly[mask]
            ax.plot(sub[year_col], sub['mean'], color=ERA_COLORS[era], lw=1.5)
            ax.fill_between(
                sub[year_col],
                sub['mean'] - sub['std'],
                sub['mean'] + sub['std'],
                color=ERA_COLORS[era], alpha=0.18,
            )
        # shade era backgrounds
        for era, (s, e) in ERA_PERIODS.items():
            ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.06)
        ax.set_xlabel('Year')
        ax.set_ylabel(label)
        ax.set_title(f'{label} — Annual Mean ± Std')
        handles = [mpatches.Patch(color=ERA_COLORS[e], label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
                   for e in era_order]
        ax.legend(handles=handles, loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_timeseries.png'))
        plt.close()

        # ── 2. Time series grouped by Basin Type ─────────────────────────
        fig, ax = plt.subplots(figsize=figsize_ts)
        for bt_label, bt_df in col_df.groupby('Basin_Type_Label'):
            yt = bt_df.groupby(year_col)[col].mean().reset_index()
            ax.plot(yt[year_col], yt[col], label=bt_label, lw=1.3)
        for era, (s, e) in ERA_PERIODS.items():
            ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.06)
        ax.set_xlabel('Year')
        ax.set_ylabel(label)
        ax.set_title(f'{label} — Annual Mean by Basin Type')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_timeseries_by_basin_type.png'))
        plt.close()

        # ── 3. Boxplot by Era ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize_box)
        era_df = col_df[col_df['Era'].isin(era_order)]
        sns.boxplot(
            data=era_df, x='Era', y=col, order=era_order,
            palette=era_palette, ax=ax, fliersize=2,
        )
        ax.set_title(f'{label} — Distribution by Era')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_boxplot_era.png'))
        plt.close()

        # ── 4. Violin plot by Era ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize_box)
        sns.violinplot(
            data=era_df, x='Era', y=col, order=era_order,
            palette=era_palette, ax=ax, inner='quartile', cut=0,
        )
        ax.set_title(f'{label} — Violin by Era')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_violin_era.png'))
        plt.close()

        # ── 5. Boxplot by GW_Basin_Type ──────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize_box)
        sns.boxplot(
            data=col_df, x='Basin_Type_Label', y=col,
            hue='Era', hue_order=era_order,
            palette=ERA_COLORS, ax=ax, fliersize=2,
        )
        ax.set_title(f'{label} — by Basin Type & Era')
        ax.set_xlabel('Basin Type')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_boxplot_basin_type.png'))
        plt.close()

        # ── 6. Boxplot by GW_Basin ───────────────────────────────────────
        if col == 'gw_pumping_mm':
            # Show all AMA/INA basins for pumping
            basin_df = col_df[col_df[gw_basin_col].isin(ama_ina_basins)]
            basin_list = sorted(basin_df[gw_basin_col].unique())
        else:
            top_basins = col_df[gw_basin_col].value_counts().head(10).index.tolist()
            basin_df = col_df[col_df[gw_basin_col].isin(top_basins)]
            basin_list = top_basins
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.boxplot(
            data=basin_df, x=gw_basin_col, y=col,
            order=basin_list,
            hue='Era', hue_order=era_order,
            palette=ERA_COLORS, ax=ax, fliersize=1,
        )
        title_suffix = 'AMA/INA Basins' if col == 'gw_pumping_mm' else 'top 10'
        ax.set_title(f'{label} — by GW Basin ({title_suffix}) & Era')
        ax.set_xlabel('GW Basin')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_boxplot_gw_basin.png'))
        plt.close()

    logger.info(f'Exploratory plots saved to {output_dir}')


# ─── ML Pipeline Visualization Helpers ───────────────────────────────────────


def plot_loo_heatmap(
        metrics_df: pd.DataFrame,
        fold_col: str,
        output_dir: str,
        title: str = 'LOO Heatmap',
) -> None:
    """Heatmap of Test R² (fold × model)."""
    apply_journal_style()
    pivot = metrics_df.pivot(index=fold_col, columns='Model', values='Test_R2')
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(4, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'LOO_Heatmap_R2.png'), dpi=600)
    plt.close()


def plot_loo_bar(
        metrics_df: pd.DataFrame,
        fold_col: str,
        output_dir: str,
) -> None:
    """Grouped bar chart of averaged Test RMSE and R² per model."""
    apply_journal_style()
    avg = metrics_df.groupby('Model').agg(
        R2_mean=('Test_R2', 'mean'),
        R2_std=('Test_R2', 'std'),
        RMSE_mean=('Test_RMSE', 'mean'),
        RMSE_std=('Test_RMSE', 'std'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(avg.Model, avg.R2_mean, yerr=avg.R2_std, capsize=4,
                color='#2980B9', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Mean Test R²', fontweight='bold')
    axes[0].set_title(f'LOO Averaged Test R² ({fold_col})', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(avg.Model, avg.RMSE_mean, yerr=avg.RMSE_std, capsize=4,
                color='#E74C3C', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Mean Test RMSE (%)', fontweight='bold')
    axes[1].set_title(f'LOO Averaged Test RMSE ({fold_col})', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'LOO_Averaged_Metrics.png'), dpi=600)
    plt.close()


def create_cross_strategy_summary(all_results: dict, output_dir: str) -> None:
    """Create a summary table and figure comparing all strategies."""
    makedirs(output_dir)
    rows = []
    for strategy_name, res in all_results.items():
        if 'comparison_df' in res:
            for _, row in res['comparison_df'].iterrows():
                rows.append({
                    'Strategy': strategy_name,
                    'Model': row['Model'],
                    'Test_R2': row['Test_R2'],
                    'Test_RMSE': row['Test_RMSE'],
                    'Overfit_R2': row['Overfit_R2'],
                })
        elif 'avg_df' in res:
            for _, row in res['avg_df'].iterrows():
                rows.append({
                    'Strategy': strategy_name,
                    'Model': row['Model'],
                    'Test_R2': row['Mean_Test_R2'],
                    'Test_RMSE': row['Mean_Test_RMSE'],
                    'Overfit_R2': row['Mean_Overfit_R2'],
                })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, 'Cross_Strategy_Summary.csv'), index=False)

    apply_journal_style()
    strategies = summary_df.Strategy.unique()
    models = summary_df.Model.unique()
    n_strategies = len(strategies)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['Test_R2', 'Test_RMSE', 'Overfit_R2']
    ylabels = ['Test R²', 'Test RMSE (%)', 'Overfitting (R² gap)']
    width = 0.8 / n_strategies
    x = np.arange(len(models))

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for i, strategy in enumerate(strategies):
            sub = summary_df[summary_df.Strategy == strategy]
            vals = [sub[sub.Model == m][metric].values[0]
                    if len(sub[sub.Model == m]) > 0 else 0
                    for m in models]
            ax.bar(x + i * width, vals, width, label=strategy, alpha=0.8)
        ax.set_xticks(x + width * (n_strategies - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontweight='bold')
    axes[0].legend()
    plt.suptitle('Cross-Strategy Model Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Cross_Strategy_Comparison.png'), dpi=600)
    plt.close()
    logger.info(f'Cross-strategy summary saved to {output_dir}')


def create_full_period_time_series(
        yearly_predictions: dict,
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2099,
        actual_data: dict | None = None,
        title_prefix: str = '',
        sigma_data: dict | None = None,
) -> None:
    """Time series of predicted AMA/INA pumping with era shading."""
    apply_journal_style()
    makedirs(output_dir)
    label = f'{title_prefix} ' if title_prefix else ''

    years = sorted(yearly_predictions.keys())
    vol_af = [yearly_predictions[y]['Volume_AF'] for y in years]
    depth_mm = [yearly_predictions[y]['Mean_Depth_mm'] for y in years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    for era, (s, e) in ERA_PERIODS.items():
        ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
        ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

    ax1.plot(years, depth_mm, color=COLORS['predicted'], linewidth=1.5, marker='.',
             markersize=3, label='Predicted')
    ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
    ax1.set_title(f'XGBoost {label}AMA/INA Groundwater Pumping (1896–2099)',
                  fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(years, vol_af, color=COLORS['predicted'], linewidth=1.5, marker='.',
             markersize=3, label='Predicted')
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Total Volume (acre-ft)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 95% CI from model uncertainty
    if sigma_data:
        ci_years = [y for y in years if y in sigma_data]
        ci_depth = np.array([yearly_predictions[y]['Mean_Depth_mm']
                             for y in ci_years])
        ci_vol = np.array([yearly_predictions[y]['Volume_AF']
                           for y in ci_years])
        s_depth = np.array([sigma_data[y]['Mean_Depth_mm']
                            for y in ci_years])
        s_vol = np.array([sigma_data[y]['Volume_AF']
                          for y in ci_years])
        ax1.fill_between(ci_years, ci_depth - 1.96 * s_depth,
                         ci_depth + 1.96 * s_depth,
                         alpha=0.2, color=COLORS['ci_predicted'],
                         label='95% CI (model σ)', zorder=1)
        ax2.fill_between(ci_years, ci_vol - 1.96 * s_vol,
                         ci_vol + 1.96 * s_vol,
                         alpha=0.2, color=COLORS['ci_predicted'],
                         label='95% CI (model σ)', zorder=1)

    # Overlay actual meter data for available years
    if actual_data:
        act_years = sorted(actual_data.keys())
        act_depth = [actual_data[y]['Mean_Depth_mm'] for y in act_years]
        act_vol = [actual_data[y]['Volume_AF'] for y in act_years]
        ax1.plot(act_years, act_depth, color=COLORS['actual'], linewidth=1.5,
                 marker='o', markersize=4, label='Observed (ADWR Meter)', zorder=5)
        ax2.plot(act_years, act_vol, color=COLORS['actual'], linewidth=1.5,
                 marker='o', markersize=4, label='Observed (ADWR Meter)', zorder=5)

    handles = [
        mpatches.Patch(color=ERA_COLORS[e], alpha=0.4,
                       label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
        for e in ERA_PERIODS
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + handles,
               loc='upper left', framealpha=0.9)
    ax2.set_xlim(start_year - 1, end_year + 1)

    _add_ft_twinx(ax1)
    _add_m3_twinx(ax2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Full_Period_Time_Series.png'), dpi=600, bbox_inches='tight')
    plt.close()

    ts_df = pd.DataFrame({
        'Year': years,
        'Mean_Depth_mm': depth_mm,
        'Mean_Depth_ft': [yearly_predictions[y]['Mean_Depth_ft'] for y in years],
        'Volume_m3': [yearly_predictions[y]['Volume_m3'] for y in years],
        'Volume_AF': vol_af,
    })
    # Merge actual data into the CSV if available
    if actual_data:
        act_df = pd.DataFrame([
            {'Year': y, 'Actual_Depth_mm': actual_data[y]['Mean_Depth_mm'],
             'Actual_Volume_AF': actual_data[y]['Volume_AF']}
            for y in sorted(actual_data.keys())
        ])
        ts_df = ts_df.merge(act_df, on='Year', how='left')
    ts_df['Era'] = ts_df.Year.apply(lambda y: next(
        (e for e, (s, end) in ERA_PERIODS.items() if s <= y <= end), 'Other'))
    if sigma_data:
        ts_df['Sigma_Depth_mm'] = ts_df['Year'].map(
            lambda y: sigma_data.get(y, {}).get('Mean_Depth_mm', np.nan))
        ts_df['Sigma_Depth_ft'] = ts_df['Sigma_Depth_mm'] * _MM_TO_FT
        ts_df['Sigma_Volume_AF'] = ts_df['Year'].map(
            lambda y: sigma_data.get(y, {}).get('Volume_AF', np.nan))
        ts_df['Sigma_Volume_m3'] = ts_df['Sigma_Volume_AF'] * _AF_TO_M3
    ts_df.to_csv(os.path.join(output_dir, 'Full_Period_Time_Series.csv'), index=False)
    logger.info('Full-period time series saved.')


def create_era_summary_maps(
        yearly_predictions: dict,
        output_dir: str,
        title_prefix: str = '',
) -> None:
    """Bar chart of mean annual pumping per era."""
    apply_journal_style()
    makedirs(output_dir)

    label = f'{title_prefix} ' if title_prefix else ''
    era_means = {}
    for era, (s, e) in ERA_PERIODS.items():
        era_vals = [yearly_predictions[y]['Volume_AF']
                    for y in range(s, e + 1) if y in yearly_predictions]
        era_means[era] = np.mean(era_vals) if era_vals else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        era_means.keys(),
        era_means.values(),
        color=[ERA_COLORS[e] for e in era_means],
        edgecolor='black',
        linewidth=0.8,
    )
    for bar, val in zip(bars, era_means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Mean Annual Pumping\n(acre-ft)', fontweight='bold')
    ax.set_title(f'Mean Annual {label}GW Pumping by Era', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    _add_m3_twinx(ax)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Era_Summary_Bar.png'), dpi=600, bbox_inches='tight')
    plt.close()
    logger.info('Era summary bar chart saved.')


def build_annual_df(
        yearly_dict: dict[int, dict[str, dict]],
        name_col: str,
) -> pd.DataFrame:
    """Pivot *yearly_dict* {year: {name: metrics_dict}} → long-form DataFrame."""
    rows = []
    for year, totals in sorted(yearly_dict.items()):
        for name, metrics in totals.items():
            row = {'Year': year, name_col: name}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df['Era'] = df.Year.apply(
        lambda y: next(
            (e for e, (s, end) in ERA_PERIODS.items() if s <= y <= end), 'Other'
        )
    )
    return df


def era_shaded_ts(
        ax: plt.Axes,
        years: np.ndarray,
        mean_vals: np.ndarray,
        std_vals: np.ndarray,
        color: str = '#2C3E50',
        label: str | None = None,
) -> None:
    """Plot a line with ±1 σ shading and era background colours."""
    for era, (s, e) in ERA_PERIODS.items():
        ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.08)
    ax.plot(years, mean_vals, color=color, linewidth=1.4, marker='.', markersize=3,
            label=label)
    ax.fill_between(years, mean_vals - std_vals, mean_vals + std_vals,
                    color=color, alpha=0.18)


def create_basin_time_series(
        basin_yearly: dict[int, dict[str, dict]],
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2099,
        title_prefix: str = '',
        actual_basin_yearly: dict[int, dict[str, dict]] | None = None,
        sigma_basin_yearly: dict[int, dict[str, dict]] | None = None,
) -> None:
    """Per-basin annual GW withdrawal time series with era shading + uncertainty."""
    apply_journal_style()
    label = f'{title_prefix} ' if title_prefix else ''
    ts_dir = os.path.join(output_dir, 'Basin_Time_Series')
    makedirs(ts_dir)

    df = build_annual_df(basin_yearly, 'Basin')

    # Build actual DataFrame if meter data is provided
    actual_df = None
    if actual_basin_yearly:
        actual_df = build_annual_df(actual_basin_yearly, 'Basin')
        actual_df = actual_df.rename(columns={
            'Mean_Depth_mm': 'Actual_Depth_mm',
            'Volume_AF': 'Actual_Volume_AF',
        })
        merged = df.merge(
            actual_df[['Year', 'Basin', 'Actual_Depth_mm', 'Actual_Volume_AF']],
            on=['Year', 'Basin'], how='left',
        )
        merged.to_csv(f'{ts_dir}Basin_Annual_GW.csv', index=False)
    else:
        df.to_csv(f'{ts_dir}Basin_Annual_GW.csv', index=False)

    basins = sorted(df.Basin.unique())
    n_basins = len(basins)
    palette = sns.color_palette('tab10', n_basins)

    # --- Combined plot (all basins, volume in acre-ft) ----------------------
    fig, ax = plt.subplots(figsize=(16, 7))
    for i, basin in enumerate(basins):
        bdf = df[df.Basin == basin].sort_values('Year')
        ax.plot(bdf.Year, bdf.Volume_AF, linewidth=1.3, label=basin,
                color=palette[i])
        # Overlay actual data for this basin
        if actual_df is not None:
            abdf = actual_df[actual_df.Basin == basin].sort_values('Year')
            if not abdf.empty:
                ax.plot(abdf.Year, abdf.Actual_Volume_AF, linewidth=1.3,
                        color=palette[i], linestyle='--', alpha=0.7,
                        marker='o', markersize=3)
    for era, (s, e) in ERA_PERIODS.items():
        ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.08)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('GW Withdrawal (acre-ft)', fontweight='bold')
    ax.set_title(f'Annual {label}GW Withdrawal by Basin (1896–2099)', fontweight='bold',
                 fontsize=14)
    # Add a single dashed-line entry for 'Observed' in the legend
    if actual_df is not None:
        from matplotlib.lines import Line2D
        obs_handle = Line2D([], [], color='gray', linestyle='--', marker='o',
                            markersize=3, label='Observed (ADWR Meter)')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(obs_handle)
        ax.legend(handles=handles, fontsize=8, ncol=2, loc='upper left', framealpha=0.9)
    else:
        ax.legend(fontsize=8, ncol=2, loc='upper left', framealpha=0.9)
    ax.set_xlim(start_year - 1, end_year + 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    _add_m3_twinx(ax)
    plt.tight_layout()
    fig.savefig(f'{ts_dir}All_Basins_Time_Series.png', dpi=600, bbox_inches='tight')
    plt.close()

    # --- Individual basin plots (2-panel: depth + volume) -------------------
    for i, basin in enumerate(basins):
        bdf = df[df.Basin == basin].sort_values('Year')
        years = bdf.Year.values
        depth_mm = bdf.Mean_Depth_mm.values
        vol_af = bdf.Volume_AF.values

        std_depth = pd.Series(depth_mm).rolling(5, min_periods=1, center=True).std().fillna(0).values
        std_vol = pd.Series(vol_af).rolling(5, min_periods=1, center=True).std().fillna(0).values
        sigma_label = '±1σ (5-yr rolling)'

        if sigma_basin_yearly:
            cv_arr = np.array([
                sigma_basin_yearly.get(y, {}).get(basin, {}).get('CV', 0)
                for y in years
            ])
            sigma_af = np.array([
                sigma_basin_yearly.get(y, {}).get(basin, {}).get(
                    'Sigma_Volume_AF', 0)
                for y in years
            ])
            if np.any(sigma_af > 0):
                std_depth = 1.96 * cv_arr * np.abs(depth_mm)
                std_vol = 1.96 * sigma_af
                sigma_label = '95% CI (model σ)'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        era_shaded_ts(ax1, years, depth_mm, std_depth, color=palette[i])
        ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
        ax1.set_title(f'{basin} — Annual {label}GW Withdrawal (1896–2099)',
                      fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')

        era_shaded_ts(ax2, years, vol_af, std_vol, color=palette[i])
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Volume (acre-ft)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Overlay actual data for this basin
        if actual_df is not None:
            abdf = actual_df[actual_df.Basin == basin].sort_values('Year')
            if not abdf.empty:
                ax1.plot(abdf.Year, abdf.Actual_Depth_mm, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)
                ax2.plot(abdf.Year, abdf.Actual_Volume_AF, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)

        handles = [
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ]
        handles.append(mpatches.Patch(color=palette[i], alpha=0.25,
                                       label=sigma_label))
        if actual_df is not None:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], color=COLORS['actual'], marker='o',
                                  markersize=4, label='Observed (ADWR Meter)'))
        ax1.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.9)
        ax2.set_xlim(start_year - 1, end_year + 1)
        _add_ft_twinx(ax1)
        _add_m3_twinx(ax2)
        plt.tight_layout()
        safe = basin.replace(' ', '_')
        fig.savefig(f'{ts_dir}{safe}_Time_Series.png', dpi=600, bbox_inches='tight')
        plt.close()

        bdf.to_csv(f'{ts_dir}{safe}_Annual_GW.csv', index=False)

    logger.info(f'Basin time series saved to {ts_dir}')


def create_subbasin_time_series(
        subbasin_yearly: dict[int, dict[str, dict]],
        output_dir: str,
        subbasin_shp: str,
        ama_code_map: dict[str, str],
        start_year: int = 1896,
        end_year: int = 2099,
        title_prefix: str = '',
        actual_subbasin_yearly: dict[int, dict[str, dict]] | None = None,
        sigma_subbasin_yearly: dict[int, dict[str, dict]] | None = None,
) -> None:
    """Per-sub-basin annual GW withdrawal time series with era shading + uncertainty."""
    apply_journal_style()
    label = f'{title_prefix} ' if title_prefix else ''
    ts_dir = os.path.join(output_dir, 'Subbasin_Time_Series')
    makedirs(ts_dir)

    df = build_annual_df(subbasin_yearly, 'Subbasin')

    # Build actual DataFrame if meter data is provided
    actual_df = None
    if actual_subbasin_yearly:
        actual_df = build_annual_df(actual_subbasin_yearly, 'Subbasin')
        actual_df = actual_df.rename(columns={
            'Mean_Depth_mm': 'Actual_Depth_mm',
            'Volume_AF': 'Actual_Volume_AF',
        })
        merged = df.merge(
            actual_df[['Year', 'Subbasin', 'Actual_Depth_mm', 'Actual_Volume_AF']],
            on=['Year', 'Subbasin'], how='left',
        )
        merged.to_csv(f'{ts_dir}Subbasin_Annual_GW.csv', index=False)
    else:
        df.to_csv(f'{ts_dir}Subbasin_Annual_GW.csv', index=False)

    # Map sub-basins → parent AMA/INA for grouped plots
    sub_gdf = gpd.read_file(subbasin_shp)
    sub_to_parent = {}
    for _, row in sub_gdf.iterrows():
        code = row.get('AMA_CODE', '')
        if code in ama_code_map:
            sub_to_parent[row['SUBBASIN_N']] = ama_code_map[code]
    df['Parent_Basin'] = df.Subbasin.map(sub_to_parent).fillna('Other')

    subbasins = sorted(df.Subbasin.unique())
    parent_basins = sorted(df[df.Parent_Basin != 'Other'].Parent_Basin.unique())

    # --- Grouped plot: one subplot per parent AMA/INA -----------------------
    n_parents = len(parent_basins)
    n_cols = 2
    n_rows = int(np.ceil(n_parents / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows),
                             sharex=True)
    axes_flat = axes.ravel() if n_parents > 1 else [axes]

    for idx, parent in enumerate(parent_basins):
        ax = axes_flat[idx]
        children = sorted(df[df.Parent_Basin == parent].Subbasin.unique())
        palette_p = sns.color_palette('Set2', len(children))
        for j, sb in enumerate(children):
            sdf = df[df.Subbasin == sb].sort_values('Year')
            ax.plot(sdf.Year, sdf.Volume_AF, linewidth=1.2, label=sb,
                    color=palette_p[j])
            # Overlay actual data for this sub-basin
            if actual_df is not None:
                asdf = actual_df[actual_df.Subbasin == sb].sort_values('Year')
                if not asdf.empty:
                    ax.plot(asdf.Year, asdf.Actual_Volume_AF, linewidth=1.2,
                            color=palette_p[j], linestyle='--', alpha=0.7,
                            marker='o', markersize=2)
        for era, (s, e) in ERA_PERIODS.items():
            ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.07)
        ax.set_title(parent, fontweight='bold', fontsize=11)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.set_xlim(start_year - 1, end_year + 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Year', fontweight='bold')
        if idx % n_cols == 0:
            ax.set_ylabel('GW Withdrawal\n(acre-ft)', fontweight='bold')
        ax_m3 = ax.twinx()
        af_lo, af_hi = ax.get_ylim()
        ax_m3.set_ylim(af_lo * _AF_TO_M3, af_hi * _AF_TO_M3)
        if (idx + 1) % n_cols == 0 or idx == n_parents - 1:
            ax_m3.set_ylabel('(m³)', fontweight='bold', fontsize=10)

    for k in range(n_parents, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.suptitle(f'Annual {label}GW Withdrawal by Sub-basin (1896–2099)',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f'{ts_dir}Subbasin_Grouped_Time_Series.png', dpi=600,
                bbox_inches='tight')
    plt.close()

    # --- Individual sub-basin plots (2-panel: depth + volume) ---------------
    n_sb = len(subbasins)
    palette_all = sns.color_palette('husl', n_sb)
    for i, sb in enumerate(subbasins):
        sdf = df[df.Subbasin == sb].sort_values('Year')
        years = sdf.Year.values
        depth_mm = sdf.Mean_Depth_mm.values
        vol_af = sdf.Volume_AF.values
        std_depth = pd.Series(depth_mm).rolling(5, min_periods=1, center=True).std().fillna(0).values
        std_vol = pd.Series(vol_af).rolling(5, min_periods=1, center=True).std().fillna(0).values
        sigma_label = '±1σ (5-yr rolling)'

        if sigma_subbasin_yearly:
            cv_arr = np.array([
                sigma_subbasin_yearly.get(y, {}).get(sb, {}).get('CV', 0)
                for y in years
            ])
            sigma_af = np.array([
                sigma_subbasin_yearly.get(y, {}).get(sb, {}).get(
                    'Sigma_Volume_AF', 0)
                for y in years
            ])
            if np.any(sigma_af > 0):
                std_depth = 1.96 * cv_arr * np.abs(depth_mm)
                std_vol = 1.96 * sigma_af
                sigma_label = '95% CI (model σ)'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        parent = sub_to_parent.get(sb, '')
        era_shaded_ts(ax1, years, depth_mm, std_depth, color=palette_all[i])
        ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
        title_suffix = f' ({parent})' if parent else ''
        ax1.set_title(f'{sb}{title_suffix} — Annual {label}GW Withdrawal (1896–2099)',
                      fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')

        era_shaded_ts(ax2, years, vol_af, std_vol, color=palette_all[i])
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Volume (acre-ft)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Overlay actual data for this sub-basin
        if actual_df is not None:
            asdf = actual_df[actual_df.Subbasin == sb].sort_values('Year')
            if not asdf.empty:
                ax1.plot(asdf.Year, asdf.Actual_Depth_mm, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)
                ax2.plot(asdf.Year, asdf.Actual_Volume_AF, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)

        handles = [
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ]
        handles.append(mpatches.Patch(color=palette_all[i], alpha=0.25,
                                       label=sigma_label))
        if actual_df is not None:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], color=COLORS['actual'], marker='o',
                                  markersize=4, label='Observed (ADWR Meter)'))
        ax1.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.9)
        ax2.set_xlim(start_year - 1, end_year + 1)
        _add_ft_twinx(ax1)
        _add_m3_twinx(ax2)
        plt.tight_layout()
        safe = sb.replace(' ', '_').replace('.', '')
        fig.savefig(f'{ts_dir}{safe}_Time_Series.png', dpi=600, bbox_inches='tight')
        plt.close()

        sdf.to_csv(f'{ts_dir}{safe}_Annual_GW.csv', index=False)

    logger.info(f'Sub-basin time series saved to {ts_dir}')
