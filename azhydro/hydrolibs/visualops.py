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
    'ci_predicted': '#2980B9', # Blue for CI (matching prediction tone)
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
    lo, hi = ax.get_ylim()
    ax_m3 = ax.twinx()
    ax_m3.set_ylabel(r'(m$^3$)', fontweight='bold')
    ax_m3.set_ylim(lo * _AF_TO_M3, hi * _AF_TO_M3)
    return ax_m3


def _add_af_twinx(ax):
    """Add an AF twinx to a m³ volume axis. Call after all plotting."""
    lo, hi = ax.get_ylim()
    ax_af = ax.twinx()
    ax_af.set_ylabel('(acre-ft)', fontweight='bold')
    ax_af.set_ylim(lo / _AF_TO_M3, hi / _AF_TO_M3)
    return ax_af


def _add_dual_volume_axes(ax, label: str = ''):
    """Set up m³ (left) and acre-ft (right) on a volume axis plotted in AF.

    Call after all plotting is done on *ax*.  Reformats the left axis to
    show ×10⁶ m³ and adds a right twin showing ×1000 acre-ft.

    Args:
        ax: Primary matplotlib axis (plotted in AF).
        label (str): Descriptive prefix for both y-axis labels
            (e.g. 'Total Annual Withdrawal').
    """
    lo, hi = ax.get_ylim()
    prefix = f'{label}\n' if label else ''
    # Left axis: m³ (×10⁶)
    ax_m3 = ax.twinx()
    ax_m3.set_ylim(lo * _AF_TO_M3, hi * _AF_TO_M3)
    ax_m3.yaxis.set_label_position('left')
    ax_m3.yaxis.tick_left()
    ax_m3.set_ylabel(prefix + r'(×10$^6$ m$^3$)', fontweight='bold')
    ax_m3.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x / 1e6:,.1f}'))
    # Right axis: AF (×1000)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_ylabel(prefix + '(×1000 acre-ft)', fontweight='bold')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x / 1e3:,.0f}'))
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
    """Apply journal-quality matplotlib settings.

    Returns:
        None.
    """
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

    result_df = pd.DataFrame(results)
    result_df['lower_ci'] = result_df['lower_ci'].clip(lower=0)
    return result_df


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
    Create journal-quality time series plots with dual-axis unit pairs.

    Produces two twinx figures:
    - **Volume plot** (AF left / m³ right): annual **sum** across all pixels.
    - **Depth plot** (mm left / ft right): annual **mean** per pixel.

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
        aggregation: Ignored (kept for backward compatibility). Volume units
            always use sum; depth units always use mean.
        confidence: Confidence level for CI (default 0.95).
        units: Ignored (kept for backward compatibility). Both unit pairs are
            always generated.
        figsize: Figure size.
        split_strategy: Split strategy (1=temporal, 2=random stratified, 3=spatial).
    """
    makedirs(output_dir)
    apply_journal_style()

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

    # Unit conversion factors (from mm)
    unit_info = {
        'mm': {'factor': 1.0, 'label': r'Annual Withdrawals (mm)'},
        'ft': {'factor': 1 / 304.8, 'label': r'Annual Withdrawals (ft)'},
        'af': {'factor': area / (4047 * 304.8 * 1000),
               'label': r'Annual Withdrawals ($10^3$ acre-ft)'},
        'm3': {'factor': area * 1e-9,
               'label': r'Annual Withdrawals ($10^6$ m$^3$)'},
    }

    # Two twinx pairs: (left_unit, right_unit, aggregation) — metric left, imperial right
    unit_pairs = [
        ('m3', 'af', 'sum'),
        ('mm', 'ft', 'mean'),
    ]

    for left_unit, right_unit, agg in unit_pairs:
        left_factor = unit_info[left_unit]['factor']
        right_factor = unit_info[right_unit]['factor']
        # Ratio to convert left-axis values to right-axis values
        twin_ratio = right_factor / left_factor

        agg_label = 'Total' if agg == 'sum' else 'Mean'

        # Convert to primary (left) unit
        df_plot = pred_df.copy()
        df_plot[f'Actual_{left_unit}'] = df_plot[actual_col] * left_factor
        df_plot[f'Pred_{left_unit}'] = df_plot[pred_col] * left_factor

        # Aggregate by year
        actual_agg = aggregate_yearly_data(
            df_plot, year_col, f'Actual_{left_unit}', agg, confidence,
            basin_col=gw_basin_col
        )
        pred_agg = aggregate_yearly_data(
            df_plot, year_col, f'Pred_{left_unit}', agg, confidence,
            basin_col=gw_basin_col
        )

        # Create figure with twinx
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

        # Primary axis formatting
        left_label = f'{agg_label} {unit_info[left_unit]["label"]}'
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel(left_label, fontweight='bold')
        ax.set_title(f'{model_name} - {test_case}: Annual Withdrawals Time Series',
                     fontweight='bold', fontsize=14)

        # Secondary (twinx) axis — same data, converted scale
        ax2 = ax.twinx()
        right_label = f'{agg_label} {unit_info[right_unit]["label"]}'
        ax2.set_ylabel(right_label, fontweight='bold')

        # Synchronise right-axis limits with left-axis via the conversion ratio
        def _sync_twin(ax_src, _ax_dst=ax2, _ratio=twin_ratio):
            lo, hi = ax_src.get_ylim()
            _ax_dst.set_ylim(lo * _ratio, hi * _ratio)
        ax.callbacks.connect('ylim_changed', _sync_twin)
        _sync_twin(ax)  # initial sync

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

        # Legend (from primary axis only)
        handles, labels = ax.get_legend_handles_labels()
        desired = ['Observed', 'Predicted',
                   f'{int(confidence*100)}% CI (Observed)',
                   f'{int(confidence*100)}% CI (Predicted)',
                   'Test Period']
        order = [labels.index(lbl) for lbl in desired if lbl in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc='upper left', framealpha=0.7)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(min_year - 0.5, max_year + 0.5)

        # Save figure
        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir,
                         f'TS_{model_name}_{test_case}_{left_unit}_{right_unit}_{agg}.png'),
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
        confidence: float = 0.95,
        figsize: tuple[float, float] = (12, 6),
        **_kwargs,
) -> None:
    """Create dual-twinx time series plots for a specific groundwater basin.

    Produces two figures per basin:
    - **Volume plot** (AF left / m³ right): annual **sum** across basin pixels.
    - **Depth plot** (mm left / ft right): annual **mean** per pixel.

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
        confidence: Confidence level.
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

    unit_info = {
        'mm': {'factor': 1.0, 'label': r'Annual Withdrawals (mm)'},
        'ft': {'factor': 1 / 304.8, 'label': r'Annual Withdrawals (ft)'},
        'af': {'factor': area / (4047 * 304.8 * 1000),
               'label': r'Annual Withdrawals ($10^3$ acre-ft)'},
        'm3': {'factor': area * 1e-9,
               'label': r'Annual Withdrawals ($10^6$ m$^3$)'},
    }

    unit_pairs = [
        ('m3', 'af', 'sum'),
        ('mm', 'ft', 'mean'),
    ]

    basin_clean = basin_name.replace(' ', '_').replace('/', '_')

    for left_unit, right_unit, agg in unit_pairs:
        left_factor = unit_info[left_unit]['factor']
        right_factor = unit_info[right_unit]['factor']
        twin_ratio = right_factor / left_factor

        agg_label = 'Total' if agg == 'sum' else 'Mean'

        df_plot = basin_df.copy()
        df_plot[f'Actual_{left_unit}'] = df_plot[actual_col] * left_factor
        df_plot[f'Pred_{left_unit}'] = df_plot[pred_col] * left_factor

        actual_agg = aggregate_yearly_data(
            df_plot, year_col, f'Actual_{left_unit}', agg, confidence
        )
        pred_agg = aggregate_yearly_data(
            df_plot, year_col, f'Pred_{left_unit}', agg, confidence
        )

        fig, ax = plt.subplots(figsize=figsize)

        years = actual_agg['year'].values
        min_year, max_year = years.min(), years.max()

        # Shade test periods
        for start, end in test_year_limits:
            ax.axvspan(start - 0.5, end + 0.5,
                       alpha=0.2, color=COLORS['test_shade'],
                       label='Test Period' if start == test_year_limits[0][0] else '')

        # Confidence intervals
        ax.fill_between(actual_agg['year'], actual_agg['lower_ci'],
                        actual_agg['upper_ci'], alpha=0.3, color=COLORS['actual'],
                        label=f'{int(confidence*100)}% CI (Observed)')
        ax.fill_between(pred_agg['year'], pred_agg['lower_ci'],
                        pred_agg['upper_ci'], alpha=0.3, color=COLORS['predicted'],
                        label=f'{int(confidence*100)}% CI (Predicted)')

        # Lines
        ax.plot(actual_agg['year'], actual_agg['value'],
                color=COLORS['actual'], marker='o', markersize=6,
                linewidth=2, label='Observed')
        ax.plot(pred_agg['year'], pred_agg['value'],
                color=COLORS['predicted'], marker='s', markersize=6,
                linewidth=2, label='Predicted')

        # Primary axis
        left_label = f'{agg_label} {unit_info[left_unit]["label"]}'
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel(left_label, fontweight='bold')
        ax.set_title(f'{basin_name}\n{model_name} - {test_case}',
                     fontweight='bold')

        # Twinx
        ax2 = ax.twinx()
        right_label = f'{agg_label} {unit_info[right_unit]["label"]}'
        ax2.set_ylabel(right_label, fontweight='bold')

        def _sync_twin(ax_src, _ax_dst=ax2, _ratio=twin_ratio):
            lo, hi = ax_src.get_ylim()
            _ax_dst.set_ylim(lo * _ratio, hi * _ratio)
        ax.callbacks.connect('ylim_changed', _sync_twin)
        _sync_twin(ax)

        # X-axis ticks
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
        desired = ['Observed', 'Predicted',
                   f'{int(confidence*100)}% CI (Observed)',
                   f'{int(confidence*100)}% CI (Predicted)',
                   'Test Period']
        order = [labels.index(lbl) for lbl in desired if lbl in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc='upper left', framealpha=0.7)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(min_year - 0.5, max_year + 0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(
            output_dir,
            f'TS_{model_name}_{test_case}_{basin_clean}_{left_unit}_{right_unit}_{agg}.png'),
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
        n_jobs: int = -1,
        **_kwargs,
) -> None:
    """Create dual-twinx time series plots for all basins in parallel.

    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        test_year_limits: Test period definitions.
        gw_basin_col: Name of basin column.
        raster_res: Raster resolution.
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
        raster_res=raster_res,
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
        'af': (area / (4047 * 304.8 * 1000), r'$10^3$ acre-ft'),
        'm3': (area * 1e-9, r'$10^6$ m$^3$'),
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
    ax.set_ylabel(f'Total Annual Withdrawals ({unit_label})', fontweight='bold')
    ax.set_title(f'{test_case}: Model Comparison - Annual Withdrawals',
                fontweight='bold', fontsize=14)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    ax.legend(loc='upper left', ncol=2, framealpha=0.7)
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
        figsize: tuple[float, float] = (12, 5),
        axis_max: float | None = None,
) -> None:
    """
    Create scatter plots comparing actual vs predicted for train and test data.

    Metrics (R², RMSE, MAE, MBE) are loaded from the pre-computed
    ``Error_Metrics_{model_name}.csv`` in the parent directory. If the CSV
    is not found, metrics are computed on the fly.

    Args:
        pred_df: DataFrame with predictions.
        output_dir: Output directory.
        model_name: Name of the ML model.
        test_case: Test case identifier.
        actual_col: Name of actual values column.
        pred_col: Name of predicted values column.
        data_col: Name of data type column (TRAIN/TEST).
        figsize: Figure size.
        axis_max: Optional upper bound for both axes. When set, the axis
            range is clipped to ``[global_min, axis_max * 1.05]``.
    """
    from sklearn.metrics import r2_score as _r2_score
    from hydrolibs.mlops import normalized_rmse, normalized_mae, normalized_mbe

    makedirs(output_dir)
    apply_journal_style()

    # Load pre-computed metrics from Error_Metrics CSV if available
    metric_csv = os.path.join(
        os.path.dirname(output_dir), f'Error_Metrics_{model_name}.csv')
    metric_lookup: dict[str, dict[str, float]] = {}
    if os.path.isfile(metric_csv):
        mdf = pd.read_csv(metric_csv)
        for _, row in mdf[mdf['Year'] == 'ALL'].iterrows():
            metric_lookup[row['Data']] = {
                'R2': row['R2'], 'RMSE': row['RMSE (%)'],
                'MAE': row['MAE (%)'], 'MBE': row['MBE (%)'],
            }

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Global axis limits so train and test panels share the same scale
    all_vals = pd.concat([pred_df[actual_col], pred_df[pred_col]], ignore_index=True)
    global_min = float(all_vals.min())
    global_max = float(all_vals.max())
    if axis_max is not None:
        padding = axis_max * 0.10
        global_min = global_min - padding
        global_max = axis_max + padding

    for idx, (data_type, color) in enumerate([('TRAIN', COLORS['train_shade']),
                                               ('TEST', COLORS['test_shade'])]):
        ax = axes[idx]
        df = pred_df[pred_df[data_col] == data_type]

        if df.empty:
            continue

        ax.scatter(
            df[pred_col],
            df[actual_col],
            alpha=0.5,
            s=20,
            c=color,
            edgecolors='white',
            linewidths=0.5
        )

        # 1:1 line (use global limits so both panels match)
        ax.plot([global_min, global_max], [global_min, global_max],
                'k--', linewidth=1.5, label='1:1 Line')

        # Regression line (fit on panel data, draw over global range)
        z = np.polyfit(df[pred_col], df[actual_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(global_min, global_max, 100)
        sign = '\u2212' if z[1] < 0 else '+'
        ax.plot(x_line, p(x_line), color='red', linewidth=1.5,
                label=f'Fit: y={z[0]:.2f}x {sign} {abs(z[1]):.2f}')

        # Use pre-computed metrics; fall back to on-the-fly computation
        if data_type in metric_lookup:
            m = metric_lookup[data_type]
            r2, rmse, mae, mbe = m['R2'], m['RMSE'], m['MAE'], m['MBE']
        else:
            actual = df[actual_col].to_numpy()
            pred = df[pred_col].to_numpy()
            r2 = _r2_score(actual, pred)
            rmse = normalized_rmse(actual, pred)
            mae = normalized_mae(actual, pred)
            mbe = normalized_mbe(actual, pred)

        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_xlabel('Predicted (mm)', fontweight='bold')
        ax.set_ylabel('Observed (mm)', fontweight='bold')
        ax.set_title(f'{data_type} Data', fontweight='bold')
        mbe_sign = '\u2212' if mbe < 0 else ''
        metrics_text = (f'R²={r2:.3f}\n'
                        f'RMSE={rmse:.1f}%\n'
                        f'MAE={mae:.1f}%\n'
                        f'MBE={mbe_sign}{abs(mbe):.1f}%')
        ax.text(0.97, 0.03, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='gray'))
        ax.legend(loc='upper left', framealpha=0.7)
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
        skip_aggregate_ts: bool = False,
        n_jobs: int = -1,
        scatter_axis_max: float | None = None,
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
        skip_aggregate_ts: Skip the statewide aggregate time series plot
            (useful for spatial LOO where the training set dominates).
        n_jobs: Number of parallel jobs.
        scatter_axis_max: Optional upper bound for scatter plot axes.
    """
    logger.info(f'Creating visualizations for {model_name} - {test_case}...')
    makedirs(output_dir)

    # 1. Aggregate time series plot
    if not skip_aggregate_ts:
        logger.info('  Creating time series plots...')
        create_time_series_plot_journal(
            pred_df, output_dir, model_name, test_case, test_year_limits,
            year_col=year_col, actual_col=actual_col, pred_col=pred_col,
            gw_basin_col=gw_basin_col, raster_res=raster_res,
            use_ama_ina=use_ama_ina, units=['af', 'm3', 'mm']
        )

    # 2. Scatter plots (BC predictions if available, otherwise raw)
    logger.info('  Creating scatter plots...')
    create_train_test_scatter(
        pred_df, output_dir, model_name, test_case,
        actual_col=actual_col, pred_col=pred_col,
        axis_max=scatter_axis_max,
    )
    # 2b. Raw (non-BC) scatter plots
    if 'Pred_GW_mm_raw' in pred_df.columns:
        create_train_test_scatter(
            pred_df, output_dir, f'{model_name}_Raw', test_case,
            actual_col=actual_col, pred_col='Pred_GW_mm_raw',
            axis_max=scatter_axis_max,
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
    """Legacy wrapper for backward compatibility.

    Args:
        input_df (pd.DataFrame): Input DataFrame with features and target.
        model: Trained ML model.
        features (list[str]): Feature column names.
        output_dir (str): Output directory for plots.
        year_col (str): Column name for year.
        gw_basin_col (str): Column name for GW basin.
        test_year_limits (tuple): Tuple of (start, end) year ranges for test set.
        pred_attr (str): Target column name.
        split_strategy (int): Splitting strategy identifier.
        test_gw_basins (tuple[str, ...]): GW basin names for spatial holdout.
        raster_res (float): Raster resolution in metres.
        x_scaler: Feature scaler (or None).
        y_scaler: Target scaler (or None).

    Returns:
        None.
    """
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
    Return a dictionary mapping variable names to descriptive labels for plotting.

    Returns:
        dict[str, str]: Mapping of variable names to display labels.
    """

    var_name_dict = {
        'gw_pumping_mm': 'Annual Withdrawals (mm)',
        'annual_et_ensemble_mm': 'Annual ET (mm)',
        'annual_eto_mm': 'Annual ETo (mm)',
        'annual_precip_mm': 'Annual Precipitation (mm)',
        'annual_peff_mm': 'Annual USDA-SCS Effective Precipitation (mm)',
        'annual_peff_pcml_mm': 'Annual PCML Effective Precipitation (mm)',
        'annual_netet_mm': 'Annual Net ET (mm) [ET − Peff]',
        'annual_tmmx_K': 'Annual Maximum Air Temperature (K)',
        'annual_tmmn_K': 'Annual Minimum Air Temperature (K)',
        'AGRI': 'Agricultural Density',
        'URBAN': 'Urban Density',
        'SW': 'Surface Water Density',
        'streamflow_mm': 'Streamflow (mm)',
        'canal_weighted_streamflow_mm': 'Canal-Weighted Streamflow (mm)',
        'gw_basin_type': 'Groundwater Basin Type',
        'GW_Basin': 'Groundwater Basin',
        'soil_depth_mm': 'Soil Depth (mm)',
        'awc_mm': 'Available Water Capacity (mm)',
        'ksat_mean_micromps': 'Mean Saturated Hydraulic Conductivity (µm/s)',
        'annual_gw_fraction': 'Annual Groundwater Irrigation Fraction',
        'annual_crop_fraction': 'Annual Crop Fraction',
        'annual_urban_fraction': 'Annual Urban Fraction',
        'annual_irr_fraction': 'Annual Irrigated Fraction',
        'well_density': 'Well Density (count/pixel)',
        'canal_density': 'Canal Density (segments/pixel)',
        'sw_rights_density': 'SW Rights Density (count/pixel)',
        'irr_sw_rights_density': 'Irrigation SW Rights Density (count/pixel)',
        'nonirr_sw_rights_density': 'Non-Irrigation SW Rights Density (count/pixel)',
        'lulc': 'Land Use / Land Cover Class',
    }
    return var_name_dict


def _clean_col_label(col: str) -> str:
    """Auto-clean a column name into a human-readable label.

    Replaces underscores with spaces and applies title case, preserving
    known unit suffixes in parentheses.
    """
    unit_map = {
        '_mm': ' (mm)', '_m': ' (m)', '_ft': ' (ft)',
        '_K': ' (K)', '_micromps': ' (µm/s)',
    }
    suffix = ''
    for key, unit in unit_map.items():
        if col.endswith(key):
            col = col[:-len(key)]
            suffix = unit
            break
    return col.replace('_', ' ').title() + suffix


# ─── Exploratory data analysis ───────────────────────────────────────────────

# Period definitions for era-based coloring/shading
ERA_PERIODS = {
    'Hindcast':    (1896, 1983),
    'Historical':  (1984, 2025),
    'Projection':  (2026, 2099),
}

ERA_COLORS = {
    'Hindcast':   '#8E44AD',   # Purple
    'Historical': '#2980B9',   # Blue
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
        static_cols: tuple[str, ...] = (
            'soil_depth_mm', 'awc_mm', 'ksat_mean_micromps',
        ),
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
        skip_cols: Columns to skip entirely in the visualizations.
        static_cols: Time-invariant columns for which time series plots are
            skipped (boxplots and violin plots are still generated).
        figsize_ts: Figure size for time series plots.
        figsize_box: Figure size for box/violin plots.
    """
    makedirs(output_dir)
    apply_journal_style()

    df = az_df.copy()
    df['Era'] = df[year_col].apply(_assign_era)

    basin_type_map = {0: 'AMA', 1: 'INA', 2: 'Other'}
    df['Basin_Type_Label'] = df[basin_type_col].map(basin_type_map)

    # Derived EDA column: Net ET = max(ET − Peff, 0)
    if 'annual_et_ensemble_mm' in df.columns and 'annual_peff_mm' in df.columns:
        df['annual_netet_mm'] = (df['annual_et_ensemble_mm'] - df['annual_peff_mm']).clip(lower=0)

    numeric_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in (year_col, basin_type_col) and c not in skip_cols
    ]

    era_order = list(ERA_PERIODS.keys())

    logger.info(f'Generating exploratory plots for {len(numeric_cols)} columns …')

    ama_ina_basins = get_ama_ina_basin_names()
    var_name_dict = get_variable_name_dict()

    def _plot_column(col):
        """Generate all EDA plots for a single column."""
        import logging as _logging
        import matplotlib
        matplotlib.use('Agg')
        _logging.getLogger('matplotlib.category').setLevel(_logging.WARNING)
        import matplotlib.pyplot as _plt
        import matplotlib.patches as _mpatches
        import seaborn as _sns
        apply_journal_style()

        safe = col.replace('/', '_')
        label = var_name_dict.get(col, _clean_col_label(col))

        # For gw_pumping_mm restrict to 1984-2024 (metered years) and AMA/INA
        if col == 'gw_pumping_mm':
            col_df = df[(df[year_col].between(1984, 2024)) &
                        (df[gw_basin_col].isin(ama_ina_basins))].copy()
        else:
            col_df = df

        # Exclude zero values before plotting
        col_df = col_df[col_df[col] > 0]

        skip_era = col in static_cols or col == 'gw_pumping_mm'

        # ── 1 & 2. Time series (skip for static/time-invariant variables) ─
        if col not in static_cols:
            # ── 1. Time series (mean ± std per year), shaded by era ──────
            yearly = col_df.groupby(year_col)[col].agg(['mean', 'std']).reset_index()
            yearly['Era'] = yearly[year_col].apply(_assign_era)

            fig, ax = _plt.subplots(figsize=figsize_ts)
            yearly = yearly.sort_values(year_col)
            ax.plot(yearly[year_col], yearly['mean'], color='#2C3E50', lw=1.5)
            ax.fill_between(
                yearly[year_col],
                np.maximum(yearly['mean'] - yearly['std'], 0),
                yearly['mean'] + yearly['std'],
                color='#2C3E50', alpha=0.15,
            )
            ax.set_xlabel('Year')
            ax.set_ylabel(label)
            ax.set_title(f'{label} — Annual Mean ± Std')
            if not skip_era:
                for era, (s, e) in ERA_PERIODS.items():
                    ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
                handles = [_mpatches.Patch(color=ERA_COLORS[e], label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
                           for e in era_order]
                ax.legend(handles=handles, loc='best', fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_timeseries.png'))
            _plt.close(fig)

            # ── 2. Time series grouped by Basin Type ─────────────────────
            fig, ax = _plt.subplots(figsize=figsize_ts)
            for bt_label, bt_df in col_df.groupby('Basin_Type_Label'):
                yt = bt_df.groupby(year_col)[col].mean().reset_index()
                ax.plot(yt[year_col], yt[col], label=bt_label, lw=1.3)
            ax.set_xlabel('Year')
            ax.set_ylabel(label)
            ax.set_title(f'{label} — Annual Mean by Basin Type')
            bt_handles, _ = ax.get_legend_handles_labels()
            if not skip_era:
                for era, (s, e) in ERA_PERIODS.items():
                    ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.06)
                bt_handles.extend([
                    _mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                                    label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
                    for e in era_order
                ])
            ax.legend(handles=bt_handles, loc='best', fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_timeseries_by_basin_type.png'))
            _plt.close(fig)

            # ── 2b. Boxplot by Year with era shading ────────────────────
            fig, ax = _plt.subplots(figsize=(14, 6))
            years_present = sorted(col_df[year_col].unique())
            year_to_idx = {y: i for i, y in enumerate(years_present)}
            _sns.boxplot(
                data=col_df, x=year_col, y=col,
                order=years_present, ax=ax, fliersize=1,
                color='#AED6F1',
            )
            if not skip_era:
                for era, (s, e) in ERA_PERIODS.items():
                    idx_s = year_to_idx.get(s)
                    idx_e = year_to_idx.get(e)
                    if idx_s is not None and idx_e is not None:
                        ax.axvspan(idx_s - 0.5, idx_e + 0.5,
                                   color=ERA_COLORS[era], alpha=0.10)
                handles = [
                    _mpatches.Patch(color=ERA_COLORS[e],
                                    label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
                    for e in era_order
                ]
                ax.legend(handles=handles, loc='best', fontsize=9)
            ax.set_xlabel('Year')
            ax.set_ylabel(label)
            ax.set_title(f'{label} — Distribution by Year')
            # Thin out x-tick labels when many years are present
            if len(years_present) > 30:
                for i, lbl in enumerate(ax.get_xticklabels()):
                    if i % 5 != 0:
                        lbl.set_visible(False)
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_year.png'))
            _plt.close(fig)

        # ── Histogram + KDE (clipped to P1–P99) ──────────────────────────
        clip_lower = col_df[col].quantile(0.01)
        clip_upper = col_df[col].quantile(0.99)
        clip_mask = col_df[col].between(clip_lower, clip_upper)
        clip_vals = col_df.loc[clip_mask, col]

        # H1. Overall histogram + KDE
        fig, ax = _plt.subplots(figsize=figsize_box)
        _sns.histplot(clip_vals, kde=True, ax=ax, color='#2C3E50', edgecolor='white', stat='count')
        ax.set_xlabel(label)
        ax.set_title(f'{label} — Histogram + KDE (P1–P99)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{safe}_hist_kde.png'))
        _plt.close(fig)

        clip_col_df = col_df[clip_mask]

        if not skip_era:
            # H2. Histogram + KDE by Era
            era_df_h = clip_col_df[clip_col_df['Era'].isin(era_order)]
            present_eras_h = [e for e in era_order if e in era_df_h['Era'].unique()]
            present_palette_h = {e: ERA_COLORS[e] for e in present_eras_h}

            fig, ax = _plt.subplots(figsize=figsize_box)
            _sns.histplot(
                data=era_df_h, x=col, hue='Era', hue_order=present_eras_h,
                palette=present_palette_h, kde=True, stat='count',
                edgecolor='white', alpha=0.35, ax=ax,
            )
            ax.set_xlabel(label)
            ax.set_title(f'{label} — Histogram + KDE by Era (P1–P99)')
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title('Era')
                legend.set_loc('upper right')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_kde_era.png'))
            _plt.close(fig)

        # H3. Histogram + KDE by Basin Type
        fig, ax = _plt.subplots(figsize=figsize_box)
        _sns.histplot(
            data=clip_col_df, x=col, hue='Basin_Type_Label',
            kde=True, stat='count', edgecolor='white', alpha=0.35, ax=ax,
        )
        ax.set_xlabel(label)
        ax.set_title(f'{label} — Histogram + KDE by Basin Type (P1–P99)')
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('Basin Type')
            legend.set_loc('upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{safe}_kde_basin_type.png'))
        _plt.close(fig)

        # H4. Histogram + KDE by GW Basin (AMA/INA)
        basin_df_kde = clip_col_df[clip_col_df[gw_basin_col].isin(ama_ina_basins)]
        basin_list_kde = sorted(basin_df_kde[gw_basin_col].unique())
        if basin_list_kde:
            fig, ax = _plt.subplots(figsize=(16, 7))
            _sns.histplot(
                data=basin_df_kde, x=col, hue=gw_basin_col,
                hue_order=basin_list_kde,
                kde=True, stat='count', edgecolor='white', alpha=0.25, ax=ax,
            )
            ax.set_xlabel(label)
            ax.set_title(f'{label} — Histogram + KDE by GW Basin (AMA/INA) (P1–P99)')
            legend = ax.get_legend()
            if legend:
                legend.set_title('GW Basin')
                legend.set_bbox_to_anchor((1.02, 1))
                for text in legend.get_texts():
                    text.set_fontsize(7)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_kde_gw_basin.png'))
            _plt.close(fig)

        if not skip_era:
            # ── 3. Boxplot by Era ────────────────────────────────────────
            fig, ax = _plt.subplots(figsize=figsize_box)
            era_df = col_df[col_df['Era'].isin(era_order)]
            present_eras = [e for e in era_order if e in era_df['Era'].unique()]
            present_palette = {e: ERA_COLORS[e] for e in present_eras}
            _sns.boxplot(
                data=era_df, x='Era', y=col, hue='Era', order=present_eras,
                palette=present_palette, ax=ax, fliersize=2, legend=False,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — Distribution by Era')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_era.png'))
            _plt.close(fig)

            # ── 4. Violin plot by Era ────────────────────────────────────
            fig, ax = _plt.subplots(figsize=figsize_box)
            _sns.violinplot(
                data=era_df, x='Era', y=col, hue='Era', order=present_eras,
                palette=present_palette, ax=ax, inner='quartile', cut=0, legend=False,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — Violin by Era')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_violin_era.png'))
            _plt.close(fig)

            # ── 5. Boxplot by GW_Basin_Type ──────────────────────────────
            fig, ax = _plt.subplots(figsize=figsize_box)
            _sns.boxplot(
                data=col_df, x='Basin_Type_Label', y=col,
                hue='Era', hue_order=era_order,
                palette=ERA_COLORS, ax=ax, fliersize=2,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — by Basin Type & Era')
            ax.set_xlabel('Basin Type')
            ax.legend(loc='best', fontsize=9, title='Era')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_basin_type.png'))
            _plt.close(fig)

            # ── 6. Boxplot by GW_Basin ───────────────────────────────────
            basin_df = col_df[col_df[gw_basin_col].isin(ama_ina_basins)]
            basin_list = sorted(basin_df[gw_basin_col].unique())
            fig, ax = _plt.subplots(figsize=(16, 7))
            _sns.boxplot(
                data=basin_df, x=gw_basin_col, y=col,
                order=basin_list,
                hue='Era', hue_order=era_order,
                palette=ERA_COLORS, ax=ax, fliersize=1,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — by GW Basin (AMA/INA) & Era')
            ax.set_xlabel('GW Basin')
            ax.tick_params(axis='x', rotation=35)
            ax.legend(loc='best', fontsize=8, title='Era')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_gw_basin.png'))
            _plt.close(fig)
        else:
            # Static / single-era variables: spatial distribution only (no era split)
            # ── 3s. Boxplot by Basin Type ────────────────────────────────
            fig, ax = _plt.subplots(figsize=figsize_box)
            _sns.boxplot(
                data=col_df, x='Basin_Type_Label', y=col,
                ax=ax, fliersize=2,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — by Basin Type')
            ax.set_xlabel('Basin Type')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_basin_type.png'))
            _plt.close(fig)

            # ── 4s. Boxplot by GW_Basin ──────────────────────────────────
            basin_df = col_df[col_df[gw_basin_col].isin(ama_ina_basins)]
            basin_list = sorted(basin_df[gw_basin_col].unique())
            fig, ax = _plt.subplots(figsize=(16, 7))
            _sns.boxplot(
                data=basin_df, x=gw_basin_col, y=col,
                order=basin_list, ax=ax, fliersize=1,
            )
            ax.set_ylabel(label)
            ax.set_title(f'{label} — by GW Basin (AMA/INA)')
            ax.set_xlabel('GW Basin')
            ax.tick_params(axis='x', rotation=35)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{safe}_boxplot_gw_basin.png'))
            _plt.close(fig)

    n_workers = max(1, multiprocessing.cpu_count() - 2)
    logger.info(f'Processing EDA with {n_workers} threads …')
    Parallel(n_jobs=n_workers, backend='threading')(
        delayed(_plot_column)(col) for col in numeric_cols
    )
    logger.info(f'Exploratory plots saved to {output_dir}')


def analyze_et_by_land_use(
        az_df: pd.DataFrame,
        output_dir: str,
        et_col: str = 'annual_et_ensemble_mm',
        eto_col: str = 'annual_eto_mm',
        gw_col: str = 'gw_pumping_mm',
        kc_max: float = 2.0,
) -> None:
    """
    Boxplot comparison of ET, ETo, and GW pumping grouped by dominant land use.

    Pixels are classified by the land-use density columns (AGRI, URBAN, SW)
    into the category with the highest density, or 'Other' when all three are
    zero.  Three figures are saved:

    1. Side-by-side ET vs ETo boxplots by land-use category.
    2. ET/ETo ratio boxplot by land-use category with Kc_max reference line.
    3. Groundwater pumping depth boxplot by land-use category (positive
       pumping only).

    Args:
        az_df: Full Arizona predictor DataFrame.
        output_dir: Directory to save plots.
        et_col: Actual ET column name.
        eto_col: Reference ET column name.
        gw_col: Groundwater pumping column name.
        kc_max: Maximum Kc threshold shown as reference line.
    """
    if et_col not in az_df.columns or eto_col not in az_df.columns:
        logger.warning(f'{et_col} or {eto_col} not in DataFrame — skipping ET land-use analysis.')
        return

    makedirs(output_dir)
    apply_journal_style()

    df = az_df[[et_col, eto_col]].copy()
    lu_cols = ['AGRI', 'URBAN', 'SW']
    available = [c for c in lu_cols if c in az_df.columns]
    if not available:
        logger.warning('No land-use density columns found — skipping ET land-use analysis.')
        return

    for c in available:
        df[c] = az_df[c]

    # Classify each pixel by dominant land use (highest density) or 'Other'
    lu_df = df[available]
    max_density = lu_df.max(axis=1)
    dominant = lu_df.idxmax(axis=1)
    df['Land_Use'] = np.where(max_density > 0, dominant, 'Other')

    # Drop rows with missing ET/ETo
    df = df.dropna(subset=[et_col, eto_col])
    df = df[df[eto_col] > 0]  # avoid division by zero for ratio

    lu_order = [c for c in ['AGRI', 'URBAN', 'SW', 'Other'] if c in df['Land_Use'].values]
    lu_palette = {'AGRI': '#2ecc71', 'URBAN': '#e74c3c', 'SW': '#3498db', 'Other': '#95a5a6'}

    # ── 1. ET, ETo, and GW pumping boxplots by land use ─────────────────
    value_vars = [et_col, eto_col]
    var_labels = {et_col: 'Actual ET', eto_col: 'Reference ETo'}
    var_palette = {'Actual ET': '#e67e22', 'Reference ETo': '#2980b9'}
    if gw_col in az_df.columns:
        df[gw_col] = az_df.loc[df.index, gw_col]
        # Only include positive pumping; set rest to NaN so they drop from melt
        df.loc[df[gw_col] <= 0, gw_col] = np.nan
        value_vars.append(gw_col)
        var_labels[gw_col] = 'Withdrawals'
        var_palette['Withdrawals'] = '#2ecc71'

    melted = df.melt(
        id_vars=['Land_Use'],
        value_vars=value_vars,
        var_name='Variable',
        value_name='mm',
    ).dropna(subset=['mm'])
    melted['Variable'] = melted['Variable'].map(var_labels)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=melted, x='Land_Use', y='mm', hue='Variable',
        order=lu_order, ax=ax, fliersize=1,
        palette=var_palette,
    )
    ax.set_xlabel('Dominant Land Use', fontweight='bold')
    ax.set_ylabel('Depth (mm)', fontweight='bold')
    ax.set_title('ET, ETo & Withdrawals by Land Use', fontweight='bold')
    ax.legend(title='Variable', framealpha=0.7)
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ET_ETo_GW_by_LandUse.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # ── 2. ET/ETo ratio boxplot by land use ──────────────────────────────
    df['ET_ETo_ratio'] = df[et_col] / df[eto_col]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x='Land_Use', y='ET_ETo_ratio',
        order=lu_order, ax=ax, fliersize=1,
        palette=lu_palette,
    )
    ax.axhline(1.0, ls='--', color='black', lw=1.2, label='Kc = 1.0 (ET = ETo)')
    ax.axhline(kc_max, ls='--', color='red', lw=1.2, label=f'Kc_max = {kc_max}')
    ax.set_xlabel('Dominant Land Use', fontweight='bold')
    ax.set_ylabel('ET / ETo Ratio', fontweight='bold')
    ax.set_title('ET/ETo Ratio by Land Use', fontweight='bold')
    ax.legend(framealpha=0.7)
    ax.grid(True, axis='y', alpha=0.3, ls='--')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ET_ETo_Ratio_by_LandUse.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # Log exceedance stats by land use
    logger.info('\n--- ET > ETo exceedance by land use ---')
    for lu in lu_order:
        sub = df[df['Land_Use'] == lu]
        n_exceed = (sub[et_col] > sub[eto_col]).sum()
        n_exceed_kc = (sub['ET_ETo_ratio'] > kc_max).sum()
        logger.info(f'  {lu:6s}: {len(sub):>8,} pixels, '
                    f'{n_exceed:>7,} ET>ETo ({100 * n_exceed / len(sub):.1f}%), '
                    f'{n_exceed_kc:>7,} ET>{kc_max}×ETo ({100 * n_exceed_kc / len(sub):.1f}%)')

    logger.info(f'ET vs ETo land-use plots saved to {output_dir}')


def analyze_pumping_distribution(
        az_df: pd.DataFrame,
        output_dir: str,
        year_list: list[int],
        max_gw: float | None = None,
        pred_attr: str = 'gw_pumping_mm',
        year_col: str = 'Year',
        gw_basin_col: str = 'GW_Basin',
        thresholds: list[int] | None = None,
) -> None:
    """
    Analyze groundwater pumping depth distribution.

    Logs summary statistics (percentiles, threshold counts) and generates an
    empirical CDF plot with separate curves for each threshold and no threshold.

    Args:
        az_df: Full Arizona predictor DataFrame.
        output_dir: Directory to save plots.
        year_list: Metered years to include (e.g. 1984-2024).
        max_gw: Optional MAX_GW threshold (mm). Shown as vertical line.
        pred_attr: Target column name.
        year_col: Year column name.
        gw_basin_col: Basin column name.
        thresholds: Depth thresholds (mm) for pixel-count analysis.
    """
    makedirs(output_dir)
    apply_journal_style()

    if thresholds is None:
        thresholds = [1000, 2000, 3000, 4000, 5000]

    # Filter to metered years, positive pumping, AMA/INA basins
    ama_ina = get_ama_ina_basin_names()
    metered = az_df[
        az_df[year_col].isin(year_list) &
        az_df[gw_basin_col].isin(ama_ina)
    ].copy()
    pos = metered[metered[pred_attr] > 0][pred_attr].dropna()

    if pos.empty:
        logger.warning('No positive pumping values found — skipping distribution analysis.')
        return

    # ── Log summary statistics ───────────────────────────────────────────
    pcts = [90, 95, 99, 99.5, 99.9, 100]
    pct_vals = np.percentile(pos, pcts)
    logger.info(f'\n--- Pumping depth distribution ({pred_attr}, metered years) ---')
    logger.info(f'  Count: {len(pos):,}  Mean: {pos.mean():.1f} mm  Std: {pos.std():.1f} mm')
    for p, v in zip(pcts, pct_vals):
        logger.info(f'  P{p:>5.1f}: {v:,.1f} mm')

    # ── Statistical outlier thresholds ───────────────────────────────────
    q1, q3 = np.percentile(pos, [25, 75])
    iqr = q3 - q1
    tukey_mild = q3 + 1.5 * iqr      # Tukey's inner fence
    tukey_extreme = q3 + 3.0 * iqr    # Tukey's outer fence
    p99 = np.percentile(pos, 99)
    p995 = np.percentile(pos, 99.5)

    logger.info(f'\n  Statistical outlier benchmarks:')
    logger.info(f'    Q1: {q1:,.1f} mm  Q3: {q3:,.1f} mm  IQR: {iqr:,.1f} mm')
    logger.info(f'    Tukey mild   (Q3 + 1.5×IQR): {tukey_mild:,.1f} mm')
    logger.info(f'    Tukey extreme (Q3 + 3×IQR):  {tukey_extreme:,.1f} mm')
    logger.info(f'    P99:  {p99:,.1f} mm')
    logger.info(f'    P99.5: {p995:,.1f} mm')
    if max_gw is not None:
        logger.info(f'    MAX_GW (configured): {max_gw:,.1f} mm')
        # Determine where MAX_GW falls relative to benchmarks
        pct_rank = 100.0 * (pos <= max_gw).mean()
        logger.info(f'    MAX_GW percentile rank: P{pct_rank:.2f}')

    logger.info(f'\n  Pixels above threshold (of {len(pos):,} positive):')
    for t in thresholds:
        n = (pos > t).sum()
        logger.info(f'    > {t:,} mm: {n:,} ({100 * n / len(pos):.3f}%)')

    if max_gw is not None:
        n_drop = (pos > max_gw).sum()
        logger.info(f'  MAX_GW = {max_gw:.0f} mm would remove {n_drop:,} pixels '
                    f'({100 * n_drop / len(pos):.3f}%)')

    # ── Empirical CDFs — one per threshold + no threshold ──────────────
    cdf_scenarios = [('No threshold', None)] + [
        (f'≤ {t:,} mm', t) for t in thresholds
    ]
    cmap = plt.cm.get_cmap('viridis', len(cdf_scenarios))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (label, cutoff) in enumerate(cdf_scenarios):
        subset = pos if cutoff is None else pos[pos <= cutoff]
        if subset.empty:
            continue
        sorted_vals = np.sort(subset.values)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, ecdf, lw=2, color=cmap(idx),
                label=f'{label} (n={len(subset):,})')

    # Statistical benchmark lines
    ax.axvline(p99, ls=':', color='purple', lw=1.3,
               label=f'P99 = {p99:,.0f} mm')
    ax.axvline(tukey_mild, ls=':', color='orange', lw=1.3,
               label=f'Tukey mild (Q3+1.5×IQR) = {tukey_mild:,.0f} mm')
    ax.axvline(tukey_extreme, ls=':', color='darkorange', lw=1.3,
               label=f'Tukey extreme (Q3+3×IQR) = {tukey_extreme:,.0f} mm')
    if max_gw is not None:
        pct_rank = 100.0 * (pos <= max_gw).mean()
        n_removed = (pos > max_gw).sum()
        ax.axvline(max_gw, ls='--', color='red', lw=1.5,
                   label=f'Upper bound = {max_gw:,.0f} mm '
                         f'(P{pct_rank:.1f}, removes {n_removed:,} pixels)')

    ax.set_xlabel('Annual Withdrawal Depth (mm)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_title('Empirical CDF of Withdrawal Depth (Metered Years, AMA/INA)',
                 fontweight='bold')
    ax.legend(framealpha=0.9, fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3, ls='--')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Pumping_ECDF.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    logger.info(f'Pumping distribution plots saved to {output_dir}')


# ─── ML Pipeline Visualization Helpers ───────────────────────────────────────


def plot_grid_heatmap(
        avg_csv: str,
        output_dir: str,
        strategy_label: str = 'Random',
) -> None:
    """Heatmap of mean Test R² (test-size × model).

    Reads ``Model_Comparison_Averaged.csv`` and pivots ``Test_R2_mean``
    into a test-size (rows) × model (columns) heatmap.

    Args:
        avg_csv (str): Path to ``Model_Comparison_Averaged.csv``.
        output_dir (str): Directory for saved figures.
        strategy_label (str): Label used in figure title.
    """
    apply_journal_style()
    makedirs(output_dir)
    df = pd.read_csv(avg_csv)
    df['test_size_label'] = df['test_size'].map(lambda x: f'{x:.0%}')
    pivot = df.pivot(index='test_size_label', columns='Model', values='Test_R2_mean')
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: float(s.strip('%')) / 100)]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(f'{strategy_label}: Mean Test R² per Test Size',
                 fontweight='bold')
    ax.set_ylabel('Test Size')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Heatmap_R2.png'), dpi=600)
    plt.close()
    logger.info(f'{strategy_label} heatmap saved to {output_dir}')


def plot_loo_heatmap(
        metrics_df: pd.DataFrame,
        fold_col: str,
        output_dir: str,
        title: str = 'LOO Heatmap',
) -> None:
    """Heatmap of Test R² (fold × model).

    Args:
        metrics_df (pd.DataFrame): DataFrame with columns Model, Test_R2,
            and one row per fold × model.
        fold_col (str): Column name identifying folds (e.g. 'Test_Year', 'Basin').
        output_dir (str): Output directory for the plot.
        title (str): Figure title.

    Returns:
        None.
    """
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
    """Grouped bar chart of averaged Test RMSE and R² per model.

    Args:
        metrics_df (pd.DataFrame): DataFrame with columns Model, Test_R2,
            and Test_RMSE.
        fold_col (str): Column name identifying folds (used in titles).
        output_dir (str): Output directory for the plot.

    Returns:
        None.
    """
    apply_journal_style()
    avg = metrics_df.groupby('Model').agg(
        R2_mean=('Test_R2', 'mean'),
        R2_std=('Test_R2', 'std'),
        RMSE_mean=('Test_RMSE', 'mean'),
        RMSE_std=('Test_RMSE', 'std'),
    ).reset_index().sort_values('RMSE_mean')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(avg.Model, avg.R2_mean, yerr=avg.R2_std, capsize=4,
                color='#2980B9', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Mean Test R²', fontweight='bold')
    axes[0].set_title(f'LOO Averaged Test R² ({fold_col})', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

    axes[1].bar(avg.Model, avg.RMSE_mean, yerr=avg.RMSE_std, capsize=4,
                color='#E74C3C', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Mean Test RMSE (%)', fontweight='bold')
    axes[1].set_title(f'LOO Averaged Test RMSE ({fold_col})', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'LOO_Averaged_Metrics.png'), dpi=600)
    plt.close()

    # Overfitting bar charts
    overfit_cols = [c for c in ['Overfit_R2', 'Overfit_RMSE']
                    if c in metrics_df.columns]
    if overfit_cols:
        agg_dict = {}
        for c in overfit_cols:
            agg_dict[f'{c}_mean'] = (c, 'mean')
            agg_dict[f'{c}_std'] = (c, 'std')
        avg_of = metrics_df.groupby('Model').agg(**agg_dict).reset_index()
        avg_of = avg_of.loc[avg.index]  # keep same model order

        fig, axes = plt.subplots(1, len(overfit_cols), figsize=(7 * len(overfit_cols), 5))
        if len(overfit_cols) == 1:
            axes = [axes]

        if 'Overfit_R2' in overfit_cols:
            ax = axes[overfit_cols.index('Overfit_R2')]
            ax.bar(avg_of.Model, avg_of['Overfit_R2_mean'],
                   yerr=avg_of['Overfit_R2_std'], capsize=4,
                   color='#F39C12', edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Mean Overfit R² (Train − Test)', fontweight='bold')
            ax.set_title(f'LOO Averaged Overfit R² ({fold_col})', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        if 'Overfit_RMSE' in overfit_cols:
            ax = axes[overfit_cols.index('Overfit_RMSE')]
            ax.bar(avg_of.Model, avg_of['Overfit_RMSE_mean'],
                   yerr=avg_of['Overfit_RMSE_std'], capsize=4,
                   color='#8E44AD', edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Mean Overfit RMSE (%) (Train − Test)', fontweight='bold')
            ax.set_title(f'LOO Averaged Overfit RMSE ({fold_col})', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'LOO_Averaged_Overfitting.png'), dpi=600)
        plt.close()


def plot_grid_bar_charts(
        all_runs_csv: str,
        output_dir: str,
        strategy_label: str = 'Random',
) -> None:
    """Bar charts of metrics averaged across all splits (test-size × seed).

    Reads ``All_Runs.csv`` and aggregates across every split to produce two
    figures, each with a single pair of bar charts (one bar per model):

    1. **Metrics** — Mean Test R² and Mean Test RMSE (± std).
    2. **Overfitting** — Mean Overfit R² and Mean Overfit RMSE (± std).

    Args:
        all_runs_csv (str): Path to ``All_Runs.csv``.
        output_dir (str): Directory for saved figures.
        strategy_label (str): Label used in figure titles.
    """
    apply_journal_style()
    makedirs(output_dir)
    df = pd.read_csv(all_runs_csv)

    avg = df.groupby('Model').agg(
        Test_R2_mean=('Test_R2', 'mean'), Test_R2_std=('Test_R2', 'std'),
        Test_RMSE_mean=('Test_RMSE', 'mean'), Test_RMSE_std=('Test_RMSE', 'std'),
        Overfit_R2_mean=('Overfit_R2', 'mean'), Overfit_R2_std=('Overfit_R2', 'std'),
        Overfit_RMSE_mean=('Overfit_RMSE', 'mean'), Overfit_RMSE_std=('Overfit_RMSE', 'std'),
    ).reset_index().sort_values('Test_RMSE_mean')
    models = avg['Model'].values

    # --- 1. Error-metric bar charts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(models, avg['Test_R2_mean'], yerr=avg['Test_R2_std'],
                capsize=4, color='#2980B9', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Mean Test R²', fontweight='bold')
    axes[0].set_title(f'{strategy_label}: Averaged Test R²', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

    axes[1].bar(models, avg['Test_RMSE_mean'], yerr=avg['Test_RMSE_std'],
                capsize=4, color='#E74C3C', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Mean Test RMSE (%)', fontweight='bold')
    axes[1].set_title(f'{strategy_label}: Averaged Test RMSE', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Averaged_Metrics.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Overfitting bar charts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(models, avg['Overfit_R2_mean'], yerr=avg['Overfit_R2_std'],
                capsize=4, color='#F39C12', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Mean Overfit R² (Train − Test)', fontweight='bold')
    axes[0].set_title(f'{strategy_label}: Averaged Overfit R²', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

    axes[1].bar(models, avg['Overfit_RMSE_mean'], yerr=avg['Overfit_RMSE_std'],
                capsize=4, color='#8E44AD', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Mean Overfit RMSE (%) (Train − Test)', fontweight='bold')
    axes[1].set_title(f'{strategy_label}: Averaged Overfit RMSE', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Averaged_Overfitting.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)

    logger.info(f'{strategy_label} averaged bar charts saved to {output_dir}')


def create_cross_strategy_summary(all_results: dict, output_dir: str) -> None:
    """Create a publication-ready summary table and figure comparing all strategies.

    Produces:
    - ``Cross_Strategy_Summary.csv`` — full table with R², RMSE, MAE, MBE, Overfit_R²
    - ``Cross_Strategy_Summary.tex`` — LaTeX table for direct inclusion in manuscripts
    - ``Cross_Strategy_Comparison.png`` — grouped bar chart (R², RMSE, MAE, MBE)
    """
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
                    'Test_MAE': row.get('Test_MAE', np.nan),
                    'Overfit_R2': row['Overfit_R2'],
                })
        elif 'avg_df' in res:
            for _, row in res['avg_df'].iterrows():
                rows.append({
                    'Strategy': strategy_name,
                    'Model': row['Model'],
                    'Test_R2': row['Mean_Test_R2'],
                    'Test_RMSE': row['Mean_Test_RMSE'],
                    'Test_MAE': row.get('Mean_Test_MAE', np.nan),
                    'Test_MBE': row.get('Mean_Test_MBE', np.nan),
                    'Overfit_R2': row['Mean_Overfit_R2'],
                })
    summary_df = pd.DataFrame(rows)
    # Ensure all metric columns exist even if not provided by every strategy
    for col in ('Test_MAE', 'Test_MBE'):
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df.to_csv(os.path.join(output_dir, 'Cross_Strategy_Summary.csv'), index=False)

    # ---- LaTeX table (publication-ready) ----
    _export_cross_strategy_latex(summary_df, output_dir)

    # ---- Grouped bar chart ----
    apply_journal_style()
    strategies = summary_df.Strategy.unique()
    # Order models by mean Test_RMSE across strategies (ascending)
    model_rmse = (summary_df.groupby('Model')['Test_RMSE']
                  .mean().sort_values())
    models = model_rmse.index.tolist()
    n_strategies = len(strategies)
    # Human-readable legend labels (replace underscores with spaces)
    strategy_labels = {s: s.replace('_', ' ') for s in strategies}

    metrics = ['Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfit_R2']
    ylabels = ['Test R²', 'Test RMSE (%)', 'Test MAE (%)', 'Overfitting (R² gap)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    width = 0.8 / n_strategies
    x = np.arange(len(models))

    for ax, metric, ylabel in zip(axes.flat, metrics, ylabels):
        for i, strategy in enumerate(strategies):
            sub = summary_df[summary_df.Strategy == strategy]
            vals = [sub[sub.Model == m][metric].values[0]
                    if len(sub[sub.Model == m]) > 0 else 0
                    for m in models]
            ax.bar(x + i * width, vals, width,
                   label=strategy_labels[strategy], alpha=0.8)
        ax.set_xticks(x + width * (n_strategies - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontweight='bold')
    axes.flat[0].legend()
    plt.suptitle('Cross-Strategy Model Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Cross_Strategy_Comparison.png'), dpi=600)
    plt.close()
    logger.info(f'Cross-strategy summary saved to {output_dir}')


def _export_cross_strategy_latex(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Export a compact LaTeX validation table (models × strategies).

    Pivots the summary so each row is a model and columns are grouped by
    evaluation strategy, suitable for direct inclusion in journal manuscripts.
    """
    metric_cols = ['Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfit_R2']
    metric_labels = {'Test_R2': 'R²', 'Test_RMSE': 'RMSE (%)',
                     'Test_MAE': 'MAE (%)', 'Overfit_R2': 'Overfit R²'}

    strategies = summary_df.Strategy.unique()
    models = summary_df.Model.unique()

    # Build pivot: one row per model
    header_parts = ['Model']
    for s in strategies:
        for m in metric_cols:
            header_parts.append(f'{s} {metric_labels[m]}')

    tex_rows = []
    for model in models:
        row_parts = [model]
        for s in strategies:
            sub = summary_df[(summary_df.Strategy == s) & (summary_df.Model == model)]
            for m in metric_cols:
                val = sub[m].values[0] if len(sub) > 0 else np.nan
                if np.isnan(val):
                    row_parts.append('--')
                elif m == 'Test_R2' or m == 'Overfit_R2':
                    row_parts.append(f'{val:.4f}')
                else:
                    row_parts.append(f'{val:.2f}')
        tex_rows.append(' & '.join(row_parts) + r' \\')

    n_metrics = len(metric_cols)
    # Column spec: l + (n_metrics * n_strategies) centered columns
    col_spec = 'l' + 'c' * (n_metrics * len(strategies))

    # Build multicolumn header
    mc_parts = [r'\textbf{Model}']
    for s in strategies:
        mc_parts.append(
            rf'\multicolumn{{{n_metrics}}}{{c}}{{\textbf{{{s.replace("_", " ")}}}}}'
        )
    mc_header = ' & '.join(mc_parts) + r' \\'

    # Sub-header row with metric names
    sub_parts = ['']
    for _s in strategies:
        for m in metric_cols:
            sub_parts.append(metric_labels[m])
    sub_header = ' & '.join(sub_parts) + r' \\'

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Cross-strategy model validation summary.}',
        r'\label{tab:cross_strategy}',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        mc_header,
        r'\cmidrule(lr){2-' + str(1 + n_metrics) + '}'
        + ''.join(
            r'\cmidrule(lr){' + str(1 + n_metrics * i + n_metrics) + '-'
            + str(n_metrics * (i + 2)) + '}'
            for i in range(len(strategies) - 1)
        ) if len(strategies) > 1 else '',
        sub_header,
        r'\midrule',
        *tex_rows,
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    tex_path = os.path.join(output_dir, 'Cross_Strategy_Summary.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f'LaTeX validation table saved to {tex_path}')


def create_full_period_time_series(
        yearly_predictions: dict,
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2099,
        actual_data: dict | None = None,
        title_prefix: str = '',
        sigma_data: dict | None = None,
) -> None:
    """Time series of predicted AMA/INA pumping with era shading.

    Args:
        yearly_predictions (dict): {year: {'Volume_AF': ..., 'Mean_Depth_mm': ...}}.
        output_dir (str): Output directory for plots.
        start_year (int): First year on the x-axis.
        end_year (int): Last year on the x-axis.
        actual_data (dict or None): Observed meter data in same format as
            yearly_predictions.
        title_prefix (str): Prefix for figure title.
        sigma_data (dict or None): Per-year uncertainty
            {year: {'Mean_Depth_mm': σ, 'Volume_AF': σ}}.

    Returns:
        None.
    """
    apply_journal_style()
    makedirs(output_dir)
    label = f'{title_prefix} ' if title_prefix else ''

    years = sorted(yearly_predictions.keys())
    vol_m3 = [yearly_predictions[y]['Volume_m3'] for y in years]
    depth_mm = [yearly_predictions[y]['Mean_Depth_mm'] for y in years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    for era, (s, e) in ERA_PERIODS.items():
        ax1.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)
        ax2.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

    ax1.plot(years, depth_mm, color=COLORS['predicted'], linewidth=1.5, marker='.',
             markersize=3, label='Predicted')
    ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
    ax1.set_title(f'{label}Annual Withdrawals (1896–2099)',
                  fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.plot(years, vol_m3, color=COLORS['predicted'], linewidth=1.5, marker='.',
             markersize=3, label='Predicted')
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel(r'Total Volume (m$^3$)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 95% CI from model uncertainty
    if sigma_data:
        ci_years = [y for y in years if y in sigma_data]
        ci_depth = np.array([yearly_predictions[y]['Mean_Depth_mm']
                             for y in ci_years])
        ci_vol_m3 = np.array([yearly_predictions[y]['Volume_m3']
                              for y in ci_years])
        s_depth = np.array([sigma_data[y]['Mean_Depth_mm']
                            for y in ci_years])
        s_vol_m3 = np.array([sigma_data[y].get('Volume_m3',
                             sigma_data[y]['Volume_AF'] * _AF_TO_M3)
                             for y in ci_years])
        ax1.fill_between(ci_years,
                         np.maximum(ci_depth - 1.96 * s_depth, 0),
                         ci_depth + 1.96 * s_depth,
                         alpha=0.35, color=COLORS['ci_predicted'],
                         label='95% CI', zorder=1)
        ax2.fill_between(ci_years,
                         np.maximum(ci_vol_m3 - 1.96 * s_vol_m3, 0),
                         ci_vol_m3 + 1.96 * s_vol_m3,
                         alpha=0.35, color=COLORS['ci_predicted'],
                         label='95% CI', zorder=1)

    # Overlay actual meter data for available years
    if actual_data:
        act_years = sorted(actual_data.keys())
        act_depth = [actual_data[y]['Mean_Depth_mm'] for y in act_years]
        act_vol_m3 = [actual_data[y].get('Volume_m3',
                      actual_data[y]['Volume_AF'] * _AF_TO_M3)
                      for y in act_years]
        ax1.plot(act_years, act_depth, color=COLORS['actual'], linewidth=1.5,
                 marker='o', markersize=4, label='Observed (ADWR Meter)', zorder=5)
        ax2.plot(act_years, act_vol_m3, color=COLORS['actual'], linewidth=1.5,
                 marker='o', markersize=4, label='Observed (ADWR Meter)', zorder=5)

    handles = [
        mpatches.Patch(color=ERA_COLORS[e], alpha=0.4,
                       label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
        for e in ERA_PERIODS
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + handles,
               loc='upper left', framealpha=0.7)
    ax2.set_xlim(start_year - 1, end_year + 1)

    _add_ft_twinx(ax1)
    _add_af_twinx(ax2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Full_Period_Time_Series.png'), dpi=600, bbox_inches='tight')
    plt.close()

    ts_df = pd.DataFrame({
        'Year': years,
        'Mean_Depth_mm': depth_mm,
        'Mean_Depth_ft': [yearly_predictions[y]['Mean_Depth_ft'] for y in years],
        'Volume_m3': [yearly_predictions[y]['Volume_m3'] for y in years],
        'Volume_AF': [yearly_predictions[y]['Volume_AF'] for y in years],
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
        scenario_volumes: dict | None = None,
) -> None:
    """Bar chart of mean annual pumping per era with 95% confidence bars.

    Args:
        yearly_predictions (dict): {year: {'Volume_AF': ...}}.
        output_dir (str): Output directory for plots.
        title_prefix (str): Prefix for figure title.
        scenario_volumes (dict or None):
            ``{scenario_name: [{Year, Volume_AF, ...}, ...]}`` — per-scenario
            volume time series for the projection era.  When provided, the
            Projection bar shows the scenario min–max range as a hatched band.

    Returns:
        None.
    """
    apply_journal_style()
    makedirs(output_dir)

    label = f'{title_prefix} ' if title_prefix else ''
    is_cu = 'Consumptive Use' in title_prefix or 'CU' in title_prefix
    ylabel_term = 'Consumptive Use' if is_cu else 'Withdrawal'

    era_means = {}
    era_stds = {}
    for era, (s, e) in ERA_PERIODS.items():
        era_vals = [yearly_predictions[y]['Volume_AF']
                    for y in range(s, e + 1) if y in yearly_predictions]
        era_means[era] = np.mean(era_vals) if era_vals else 0
        era_stds[era] = 1.96 * np.std(era_vals) / np.sqrt(len(era_vals)) if len(era_vals) > 1 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        list(era_means.keys()),
        list(era_means.values()),
        yerr=list(era_stds.values()),
        capsize=5,
        color=[ERA_COLORS[e] for e in era_means],
        edgecolor='black',
        linewidth=0.8,
        error_kw={'linewidth': 1.5, 'color': 'black'},
    )

    # Scenario range on the Projection bar
    if scenario_volumes and 'Projection' in era_means:
        proj_s, proj_e = ERA_PERIODS['Projection']
        sc_means = []
        for sc_rows in scenario_volumes.values():
            sc_vals = [r['Volume_AF'] for r in sc_rows
                       if proj_s <= r['Year'] <= proj_e]
            if sc_vals:
                sc_means.append(np.mean(sc_vals))
        if len(sc_means) > 1:
            sc_lo, sc_hi = min(sc_means), max(sc_means)
            proj_idx = list(era_means.keys()).index('Projection')
            proj_bar = bars[proj_idx]
            bx = proj_bar.get_x()
            bw = proj_bar.get_width()
            ax.fill_between(
                [bx, bx + bw], sc_lo, sc_hi,
                alpha=0.25, color='gray', hatch='///',
                label=f'Scenario range ({len(sc_means)} LULC)',
            )
            ax.legend(fontsize=9, loc='upper right')

    for bar, val in zip(bars, era_means.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{val / 1e3:,.0f}k', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')

    ax.set_title(f'Mean Annual {label}{ylabel_term} by Era', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    _add_dual_volume_axes(ax, label=f'Mean Annual {ylabel_term}')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Era_Summary_Bar.png'), dpi=600, bbox_inches='tight')
    plt.close()
    logger.info('Era summary bar chart saved.')


def build_annual_df(
        yearly_dict: dict[int, dict[str, dict]],
        name_col: str,
) -> pd.DataFrame:
    """Pivot *yearly_dict* {year: {name: metrics_dict}} → long-form DataFrame.

    Args:
        yearly_dict (dict[int, dict[str, dict]]): Nested dict keyed by year
            then zone name, with metric dicts as values.
        name_col (str): Column name for the zone identifier (e.g. 'Basin').

    Returns:
        pd.DataFrame: Long-form DataFrame with Year, *name_col*, Era, and
            all metric columns.
    """
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
        std_vals: np.ndarray | None,
        color: str = '#2C3E50',
        label: str | None = None,
) -> None:
    """Plot a line with ±1 σ shading and era background colours.

    Args:
        ax (plt.Axes): Matplotlib axes to plot on.
        years (np.ndarray): Array of years.
        mean_vals (np.ndarray): Mean values per year.
        std_vals (np.ndarray or None): Standard deviation per year (or None).
        color (str): Line and shading color.
        label (str or None): Legend label.

    Returns:
        None.
    """
    for era, (s, e) in ERA_PERIODS.items():
        ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.08)
    ax.plot(years, mean_vals, color=color, linewidth=1.4, marker='.', markersize=3,
            label=label)
    if std_vals is not None:
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
    """Per-basin annual GW withdrawal time series with era shading + uncertainty.

    Args:
        basin_yearly (dict[int, dict[str, dict]]): {year: {basin: metrics}}.
        output_dir (str): Output directory for plots.
        start_year (int): First year on the x-axis.
        end_year (int): Last year on the x-axis.
        title_prefix (str): Prefix for figure titles.
        actual_basin_yearly (dict or None): Observed meter data in same format.
        sigma_basin_yearly (dict or None): Per-basin uncertainty in same format.

    Returns:
        None.
    """
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
        merged.to_csv(os.path.join(ts_dir, 'Basin_Annual_GW.csv'), index=False)
    else:
        df.to_csv(os.path.join(ts_dir, 'Basin_Annual_GW.csv'), index=False)

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
    ax.set_ylabel('Annual Withdrawal (acre-ft)', fontweight='bold')
    ax.set_title(f'Annual {label}Withdrawal by Basin (1896–2099)', fontweight='bold',
                 fontsize=14)
    # Build legend: basin lines + era patches + optional observed
    handles, labels = ax.get_legend_handles_labels()
    era_handles = [
        mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                       label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
        for e in ERA_PERIODS
    ]
    handles.extend(era_handles)
    if actual_df is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color='gray', linestyle='--', marker='o',
                               markersize=3, label='Observed (ADWR Meter)'))
    ax.legend(handles=handles, fontsize=8, ncol=2, loc='upper left', framealpha=0.7)
    ax.set_xlim(start_year - 1, end_year + 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    _add_m3_twinx(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(ts_dir, 'All_Basins_Time_Series.png'), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Individual basin plots (2-panel: depth + volume) -------------------
    single_color = '#2C3E50'
    for i, basin in enumerate(basins):
        bdf = df[df.Basin == basin].sort_values('Year')
        years = bdf.Year.values
        depth_mm = bdf.Mean_Depth_mm.values
        vol_m3 = bdf.Volume_m3.values if 'Volume_m3' in bdf.columns else bdf.Volume_AF.values * _AF_TO_M3

        std_depth = None
        std_vol = None
        sigma_label = None

        if sigma_basin_yearly:
            cv_arr = np.array([
                sigma_basin_yearly.get(y, {}).get(basin, {}).get('CV', 0)
                for y in years
            ])
            sigma_m3 = np.array([
                sigma_basin_yearly.get(y, {}).get(basin, {}).get(
                    'Sigma_Volume_m3',
                    sigma_basin_yearly.get(y, {}).get(basin, {}).get(
                        'Sigma_Volume_AF', 0) * _AF_TO_M3)
                for y in years
            ])
            if np.any(sigma_m3 > 0):
                std_depth = 1.96 * cv_arr * np.abs(depth_mm)
                std_vol = 1.96 * sigma_m3
                sigma_label = '95% CI'
        else:
            logger.warning('No σ data for basin %s — CI band omitted', basin)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        era_shaded_ts(ax1, years, depth_mm, std_depth, color=single_color)
        ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
        ax1.set_title(f'{basin} — Annual {label}Withdrawal (1896–2099)',
                      fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')

        era_shaded_ts(ax2, years, vol_m3, std_vol, color=single_color)
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel(r'Volume (m$^3$)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Overlay actual data for this basin
        if actual_df is not None:
            abdf = actual_df[actual_df.Basin == basin].sort_values('Year')
            if not abdf.empty:
                ax1.plot(abdf.Year, abdf.Actual_Depth_mm, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)
                act_vol_m3 = abdf.Actual_Volume_AF.values * _AF_TO_M3
                ax2.plot(abdf.Year, act_vol_m3, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)

        handles = [
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ]
        if sigma_label is not None:
            handles.append(mpatches.Patch(color=single_color, alpha=0.25,
                                           label=sigma_label))
        if actual_df is not None:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], color=COLORS['actual'], marker='o',
                                  markersize=4, label='Observed (ADWR Meter)'))
        ax1.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.7)
        ax2.set_xlim(start_year - 1, end_year + 1)
        _add_ft_twinx(ax1)
        _add_af_twinx(ax2)
        plt.tight_layout()
        safe = basin.replace(' ', '_')
        fig.savefig(os.path.join(ts_dir, f'{safe}_Time_Series.png'), dpi=600, bbox_inches='tight')
        plt.close()

        bdf.to_csv(os.path.join(ts_dir, f'{safe}_Annual_GW.csv'), index=False)

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
    """Per-sub-basin annual GW withdrawal time series with era shading + uncertainty.

    Args:
        subbasin_yearly (dict[int, dict[str, dict]]): {year: {subbasin: metrics}}.
        output_dir (str): Output directory for plots.
        subbasin_shp (str): Path to ADWR sub-basin shapefile.
        ama_code_map (dict[str, str]): Mapping of AMA/INA codes to names.
        start_year (int): First year on the x-axis.
        end_year (int): Last year on the x-axis.
        title_prefix (str): Prefix for figure titles.
        actual_subbasin_yearly (dict or None): Observed meter data in same format.
        sigma_subbasin_yearly (dict or None): Per-sub-basin uncertainty in same format.

    Returns:
        None.
    """
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
        merged.to_csv(os.path.join(ts_dir, 'Subbasin_Annual_GW.csv'), index=False)
    else:
        df.to_csv(os.path.join(ts_dir, 'Subbasin_Annual_GW.csv'), index=False)

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
        sb_handles, _ = ax.get_legend_handles_labels()
        sb_handles.extend([
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ])
        ax.legend(handles=sb_handles, fontsize=7, loc='upper left', framealpha=0.7)
        ax.set_xlim(start_year - 1, end_year + 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Year', fontweight='bold')
        if idx % n_cols == 0:
            ax.set_ylabel('Annual Withdrawal\n(acre-ft)', fontweight='bold')
        ax_m3 = ax.twinx()
        af_lo, af_hi = ax.get_ylim()
        ax_m3.set_ylim(af_lo * _AF_TO_M3, af_hi * _AF_TO_M3)
        if (idx + 1) % n_cols == 0 or idx == n_parents - 1:
            ax_m3.set_ylabel(r'(m$^3$)', fontweight='bold', fontsize=10)

    for k in range(n_parents, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.suptitle(f'Annual {label}Withdrawal by Sub-basin (1896–2099)',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(ts_dir, 'Subbasin_Grouped_Time_Series.png'), dpi=600,
                bbox_inches='tight')
    plt.close()

    # --- Individual sub-basin plots (2-panel: depth + volume) ---------------
    single_color = '#2C3E50'
    for i, sb in enumerate(subbasins):
        sdf = df[df.Subbasin == sb].sort_values('Year')
        years = sdf.Year.values
        depth_mm = sdf.Mean_Depth_mm.values
        vol_m3 = sdf.Volume_m3.values if 'Volume_m3' in sdf.columns else sdf.Volume_AF.values * _AF_TO_M3
        std_depth = None
        std_vol = None
        sigma_label = None

        if sigma_subbasin_yearly:
            cv_arr = np.array([
                sigma_subbasin_yearly.get(y, {}).get(sb, {}).get('CV', 0)
                for y in years
            ])
            sigma_m3 = np.array([
                sigma_subbasin_yearly.get(y, {}).get(sb, {}).get(
                    'Sigma_Volume_m3',
                    sigma_subbasin_yearly.get(y, {}).get(sb, {}).get(
                        'Sigma_Volume_AF', 0) * _AF_TO_M3)
                for y in years
            ])
            if np.any(sigma_m3 > 0):
                std_depth = 1.96 * cv_arr * np.abs(depth_mm)
                std_vol = 1.96 * sigma_m3
                sigma_label = '95% CI'
        else:
            logger.warning('No σ data for sub-basin %s — CI band omitted', sb)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        parent = sub_to_parent.get(sb, '')
        era_shaded_ts(ax1, years, depth_mm, std_depth, color=single_color)
        ax1.set_ylabel('Mean Depth (mm)', fontweight='bold')
        title_suffix = f' ({parent})' if parent else ''
        ax1.set_title(f'{sb}{title_suffix} — Annual {label}Withdrawal (1896–2099)',
                      fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')

        era_shaded_ts(ax2, years, vol_m3, std_vol, color=single_color)
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel(r'Volume (m$^3$)', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Overlay actual data for this sub-basin
        if actual_df is not None:
            asdf = actual_df[actual_df.Subbasin == sb].sort_values('Year')
            if not asdf.empty:
                ax1.plot(asdf.Year, asdf.Actual_Depth_mm, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)
                act_vol_m3 = asdf.Actual_Volume_AF.values * _AF_TO_M3
                ax2.plot(asdf.Year, act_vol_m3, color=COLORS['actual'],
                         linewidth=1.5, marker='o', markersize=4,
                         label='Observed (ADWR Meter)', zorder=5)

        handles = [
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.35,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ]
        if sigma_label is not None:
            handles.append(mpatches.Patch(color=single_color, alpha=0.25,
                                           label=sigma_label))
        if actual_df is not None:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], color=COLORS['actual'], marker='o',
                                  markersize=4, label='Observed (ADWR Meter)'))
        ax1.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.7)
        ax2.set_xlim(start_year - 1, end_year + 1)
        _add_ft_twinx(ax1)
        _add_af_twinx(ax2)
        plt.tight_layout()
        safe = sb.replace(' ', '_').replace('.', '')
        fig.savefig(os.path.join(ts_dir, f'{safe}_Time_Series.png'), dpi=600, bbox_inches='tight')
        plt.close()

        sdf.to_csv(os.path.join(ts_dir, f'{safe}_Annual_GW.csv'), index=False)

    logger.info(f'Sub-basin time series saved to {ts_dir}')


# =============================================================================
# Graphical abstract / Figure 1 — spatial overview map
# =============================================================================


def create_graphical_abstract(
        raster_dir: str,
        basin_shp: str,
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2099,
        ref_raster: str | None = None,
        yearly_predictions: dict | None = None,
        basin_yearly: dict | None = None,
        sigma_yearly: dict | None = None,
) -> None:
    """Create a publication-quality Figure 1 / graphical abstract.

    Layout (3-panel figure):
      - **Left**: Spatial map of mean-annual predicted withdrawal depth (mm)
        with GW basin boundaries overlaid.
      - **Top-right**: Time series of total annual withdrawals (acre-ft)
        with era shading and ±1σ UQ confidence band.
      - **Bottom-right**: Era mean bar chart with 95% CI error bars.

    Args:
        raster_dir (str): Directory containing ``Total_Predicted_<year>.tif``
            depth rasters (mm).
        basin_shp (str): Path to GW basin boundary shapefile.
        output_dir (str): Where to save the figure.
        start_year (int): First year of the raster stack.
        end_year (int): Last year of the raster stack.
        ref_raster (str or None): A reference raster for CRS/extent.  If
            None, the first raster in *raster_dir* is used.
        yearly_predictions (dict or None):
            ``{year: {'Volume_AF': ..., 'Mean_Depth_mm': ...}}`` for the
            time-series panel.  If None, only the spatial panel is produced.
        basin_yearly (dict or None):
            ``{year: {basin_name: {'Volume_AF': ...}}}`` for computing
            inter-basin spatial std on the time series (fallback if
            *sigma_yearly* is not provided).
        sigma_yearly (dict or None):
            ``{year: {'Volume_AF': ...}}`` — per-year UQ-derived σ_total
            in Volume_AF.  When provided, used for the ±1σ confidence
            band instead of inter-basin variability.

    Returns:
        None.
    """
    import rasterio as rio

    apply_journal_style()
    makedirs(output_dir)

    # ---- Compute mean-annual depth from raster stack ----
    raster_files = sorted(
        f for f in os.listdir(raster_dir)
        if f.startswith('Total_Predicted_') and f.endswith('.tif')
    )
    if not raster_files:
        logger.warning('No rasters found in %s — skipping graphical abstract.', raster_dir)
        return

    # Use first raster as template
    template_path = os.path.join(raster_dir, raster_files[0])
    if ref_raster is None:
        ref_raster = template_path

    with rio.open(ref_raster) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        crs = src.crs
        raster_shape = src.shape

    # Accumulate mean depth across all years
    depth_sum = np.zeros(raster_shape, dtype=np.float64)
    count = 0
    for fname in raster_files:
        with rio.open(os.path.join(raster_dir, fname)) as src:
            arr = src.read(1).astype(np.float64)
            arr[np.isnan(arr)] = 0.0
            depth_sum += arr
            count += 1

    mean_depth = depth_sum / max(count, 1)
    mean_depth_masked = np.ma.masked_where(mean_depth == 0, mean_depth)

    # Save mean annual depth as a standalone GeoTIFF
    mean_tif = os.path.join(output_dir, 'Mean_Annual_Predicted_mm.tif')
    with rio.open(ref_raster) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype='float32', nodata=np.nan)
    mean_out = mean_depth.astype(np.float32)
    mean_out[mean_depth == 0] = np.nan
    with rio.open(mean_tif, 'w', **profile) as dst:
        dst.write(mean_out, 1)
    logger.info(f'Mean annual depth raster saved to {mean_tif}')

    # ---- Read basin boundaries ----
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)

    # ---- Determine layout ----
    has_ts = yearly_predictions is not None and len(yearly_predictions) > 0

    if has_ts:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 9))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2],
                      height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
        ax_map = fig.add_subplot(gs[:, 0])  # map spans both rows
        ax_ts = fig.add_subplot(gs[0, 1])   # time series top-right
        ax_bar = fig.add_subplot(gs[1, 1])  # bar chart bottom-right
    else:
        fig, ax_map = plt.subplots(1, 1, figsize=(9, 9))

    # ---- Panel A: Spatial map ----

    # 2%-98% percentile colorbar range
    valid_vals = mean_depth_masked.compressed()
    if len(valid_vals) > 0:
        vmin = np.percentile(valid_vals, 2)
        vmax = np.percentile(valid_vals, 98)
    else:
        vmin, vmax = 0, 1

    im = ax_map.imshow(
        mean_depth_masked, extent=extent, origin='upper',
        cmap='Spectral_r', interpolation='nearest',
        vmin=vmin, vmax=vmax,
    )

    # Draw all basin boundaries
    basins_gdf.boundary.plot(ax=ax_map, color='#2C3E50', linewidth=0.5)

    # Highlight AMA/INA basins with thicker boundary and semi-transparent fill
    ama_ina = get_ama_ina_basin_names()
    name_col = 'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns else basins_gdf.columns[0]
    ama_ina_gdf = basins_gdf[basins_gdf[name_col].isin(ama_ina)]
    ama_ina_gdf.boundary.plot(ax=ax_map, color='black', linewidth=1.5)

    # Label AMA/INA basins
    for _, row in ama_ina_gdf.iterrows():
        bname = row[name_col]
        centroid = row.geometry.centroid
        short_name = bname.replace(' AMA', '').replace(' INA', '')
        ax_map.annotate(
            short_name, (centroid.x, centroid.y),
            fontsize=6, fontweight='bold', ha='center', va='center',
            color='black',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, lw=0),
        )

    cbar = fig.colorbar(im, ax=ax_map, shrink=0.75, pad=0.02, extend='both')
    cbar.set_label('Mean Annual Withdrawal Depth (mm)', fontweight='bold')
    ax_map.axis('off')
    ax_map.set_title(
        f'(a) Mean Annual Predicted Withdrawal ({start_year}–{end_year})',
        fontweight='bold', fontsize=13,
    )
    # ---- Panel B: Time series (top-right) ----
    if has_ts:
        years = sorted(yearly_predictions.keys())
        vol_af = np.array([yearly_predictions[y]['Volume_AF'] for y in years])

        # Compute ±1σ band: prefer UQ-derived σ, fall back to inter-basin std
        vol_std = np.zeros(len(years))
        std_label = None
        if sigma_yearly:
            for i, y in enumerate(years):
                if y in sigma_yearly:
                    val = sigma_yearly[y].get('Volume_AF', 0)
                    vol_std[i] = val if np.isfinite(val) else 0
            std_label = '±1σ (UQ)'
        elif basin_yearly:
            for i, y in enumerate(years):
                if y in basin_yearly:
                    basin_vols = [v['Volume_AF'] for v in basin_yearly[y].values()
                                  if np.isfinite(v.get('Volume_AF', np.nan))]
                    if len(basin_vols) > 1:
                        vol_std[i] = np.std(basin_vols)
            std_label = '±1σ (inter-basin)'

        for era, (s, e) in ERA_PERIODS.items():
            ax_ts.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

        # ±1σ band
        if vol_std.any():
            ax_ts.fill_between(years, vol_af - vol_std, vol_af + vol_std,
                               alpha=0.2, color=COLORS['predicted'], label=std_label)

        ax_ts.plot(years, vol_af, color=COLORS['predicted'],
                   linewidth=1.2, marker='.', markersize=2)
        ax_ts.set_xlabel('Year', fontweight='bold')
        ax_ts.set_title(
            '(b) Arizona Annual Withdrawal',
            fontweight='bold', fontsize=13,
        )
        ax_ts.set_xlim(start_year - 2, end_year + 2)

        handles = [
            mpatches.Patch(color=ERA_COLORS[e], alpha=0.4,
                           label=f'{e} ({ERA_PERIODS[e][0]}–{ERA_PERIODS[e][1]})')
            for e in ERA_PERIODS
        ]
        ax_ts.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.7)

        _add_dual_volume_axes(ax_ts, label='Total Annual Withdrawal')

        # ---- Panel C: Era mean bar chart (bottom-right) ----
        era_means = {}
        era_stds = {}
        for era, (s, e) in ERA_PERIODS.items():
            era_vals = [yearly_predictions[y]['Volume_AF']
                        for y in range(s, e + 1) if y in yearly_predictions]
            era_means[era] = np.mean(era_vals) if era_vals else 0
            era_stds[era] = 1.96 * np.std(era_vals) / np.sqrt(len(era_vals)) if len(era_vals) > 1 else 0
        era_names = list(era_means.keys())
        era_vals_list = list(era_means.values())
        era_errs = list(era_stds.values())
        x_pos = np.arange(len(era_names))
        bars = ax_bar.bar(
            x_pos, era_vals_list, yerr=era_errs, capsize=5,
            color=[ERA_COLORS[e] for e in era_names],
            edgecolor='black', linewidth=0.8,
            error_kw={'linewidth': 1.5, 'color': 'black'},
        )
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(era_names, fontsize=10)
        for bar, val in zip(bars, era_vals_list):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2, f'{val / 1e3:,.0f}k',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white',
            )
        ax_bar.set_title('(c) Era Averages', fontsize=13, fontweight='bold')
        ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
        _add_dual_volume_axes(ax_bar, label='Mean Annual Withdrawal')

    out_path = os.path.join(output_dir, 'Graphical_Abstract_Fig1.png')
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f'Graphical abstract saved to {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Raster map visualizations (era-mean panels, actual vs predicted)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_year(filename: str) -> int | None:
    """Extract the 4-digit year from a raster filename."""
    import re
    m = re.search(r'_(\d{4})[_.]', filename)
    return int(m.group(1)) if m else None


def _overlay_boundaries(
    ax,
    basins_gdf: gpd.GeoDataFrame,
    ama_ina_names: list[str],
    name_col: str,
    *,
    label_fontsize: float = 5.5,
) -> None:
    """Draw basin boundaries and highlight AMA/INA basins on a map axis."""
    basins_gdf.boundary.plot(ax=ax, color='#555555', linewidth=0.4)
    ama_ina_gdf = basins_gdf[basins_gdf[name_col].isin(ama_ina_names)]
    ama_ina_gdf.boundary.plot(ax=ax, color='black', linewidth=1.2)
    for _, row in ama_ina_gdf.iterrows():
        centroid = row.geometry.centroid
        short = row[name_col].replace(' AMA', '').replace(' INA', '')
        ax.annotate(
            short, (centroid.x, centroid.y),
            fontsize=label_fontsize, fontweight='bold',
            ha='center', va='center', color='black',
            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                      alpha=0.8, lw=0),
        )
    ax.axis('off')


def _compute_era_means(
    raster_dir: str,
    raster_shape: tuple[int, int],
    band: int = 1,
    mask_nan_only: bool = False,
) -> dict[str, np.ma.MaskedArray]:
    """Load all .tif files in *raster_dir*, group by era, return era means.

    Args:
        raster_dir (str): Directory containing ``*.tif`` rasters.
        raster_shape (tuple): (rows, cols) expected raster shape.
        band (int): Band number to read (1-based).  Default 1.
        mask_nan_only (bool): If True, only mask NaN pixels (keep zeros
            visible).  Useful for ratio bands like CV where zero is valid.

    Returns:
        dict: ``{era_name: masked_array}`` where masked pixels are zero/NaN.
    """
    import rasterio as rio

    tif_files = sorted(f for f in os.listdir(raster_dir) if f.endswith('.tif'))
    era_sums: dict[str, np.ndarray] = {}
    era_counts: dict[str, np.ndarray] = {}

    for fname in tif_files:
        year = _extract_year(fname)
        if year is None:
            continue
        era = _assign_era(year)
        if era == 'Other':
            continue
        fpath = os.path.join(raster_dir, fname)
        with rio.open(fpath) as src:
            if band > src.count:
                continue
            arr = src.read(band).astype(np.float64)
        finite = np.isfinite(arr)
        if era not in era_sums:
            era_sums[era] = np.zeros(raster_shape, dtype=np.float64)
            era_counts[era] = np.zeros(raster_shape, dtype=np.float64)
        era_sums[era][finite] += arr[finite]
        era_counts[era][finite] += 1

    era_means = {}
    for era in ERA_PERIODS:
        if era in era_sums and era_counts[era].max() > 0:
            with np.errstate(invalid='ignore'):
                mean_arr = np.where(
                    era_counts[era] > 0,
                    era_sums[era] / era_counts[era],
                    0.0,
                )
            if mask_nan_only:
                era_means[era] = np.ma.masked_where(
                    era_counts[era] == 0, mean_arr,
                )
            else:
                era_means[era] = np.ma.masked_where(
                    (mean_arr == 0) | (era_counts[era] == 0), mean_arr,
                )
        else:
            era_means[era] = np.ma.masked_all(raster_shape)
    return era_means


def create_era_raster_maps(
    raster_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str = 'Predicted Annual Withdrawal',
    unit_label: str = 'Depth (mm)',
    cmap: str = 'Spectral_r',
    out_filename: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    band: int = 1,
    mask_nan_only: bool = False,
) -> None:
    """Create a 2×2 panel figure of era-mean raster maps.

    Each panel shows the mean raster value for one era (Hindcast,
    Historical, Projection) with groundwater basin boundaries
    and AMA/INA labels overlaid.  Designed for Scientific Data
    publication quality.

    Args:
        raster_dir (str): Directory containing ``*.tif`` rasters with years
            in filenames.
        basin_shp (str): Path to GW basin boundary shapefile.
        output_dir (str): Where to save the figure.
        title (str): Figure super-title (category name).
        unit_label (str): Colorbar label (e.g. 'Depth (mm)').
        cmap (str): Matplotlib colormap name.
        out_filename (str or None): Output PNG filename.  Defaults to
            ``Era_Maps_{title_slug}.png``.
        vmin (float or None): Explicit colorbar minimum.
        vmax (float or None): Explicit colorbar maximum.
        symmetric (bool): If True, center colorbar on zero.
        band (int): Band number to read from each raster (1-based).
        mask_nan_only (bool): If True, only mask NaN pixels (keep zeros
            visible).

    Returns:
        None.
    """
    import rasterio as rio

    apply_journal_style()
    makedirs(output_dir)

    # ---- Detect rasters and get spatial metadata ----
    tif_files = sorted(f for f in os.listdir(raster_dir) if f.endswith('.tif'))
    if not tif_files:
        logger.warning('No .tif files in %s — skipping era maps.', raster_dir)
        return

    template = os.path.join(raster_dir, tif_files[0])
    with rio.open(template) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        crs = src.crs
        raster_shape = src.shape

    # ---- Compute era means ----
    era_means = _compute_era_means(raster_dir, raster_shape, band=band,
                                    mask_nan_only=mask_nan_only)

    # ---- Load basin boundaries ----
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)
    name_col = 'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns else basins_gdf.columns[0]
    ama_ina = get_ama_ina_basin_names()

    # ---- Determine color limits ----
    if vmin is None or vmax is None:
        all_vals = np.concatenate([
            em.compressed() for em in era_means.values()
            if em is not None and em.count() > 0
        ])
        if len(all_vals) == 0:
            logger.warning('All era means are empty for %s — skipping.', title)
            return
        if vmin is None:
            vmin = float(np.nanpercentile(all_vals, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(all_vals, 98))
    if symmetric:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # ---- Create figure ----
    n_eras = len(ERA_PERIODS)
    fig, axes = plt.subplots(
        1, n_eras, figsize=(6 * n_eras, 7), constrained_layout=True,
    )
    fig.suptitle(f'{title} — Era Mean', fontsize=16, fontweight='bold')
    if n_eras == 1:
        axes = [axes]

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    axes_flat = list(axes) if isinstance(axes, np.ndarray) else axes
    for idx, (era, (y1, y2)) in enumerate(ERA_PERIODS.items()):
        ax = axes_flat[idx]
        ax.set_facecolor('#D5D5D5')  # gray for no-data

        era_arr = era_means.get(era)
        if era_arr is not None and era_arr.count() > 0:
            im = ax.imshow(
                era_arr, extent=extent, origin='upper',
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation='nearest',
            )
        else:
            # Blank panel
            blank = np.ma.masked_all(raster_shape)
            im = ax.imshow(
                blank, extent=extent, origin='upper',
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation='nearest',
            )

        _overlay_boundaries(ax, basins_gdf, ama_ina, name_col)
        ax.set_title(
            f'{panel_labels[idx]} {era} ({y1}–{y2})',
            fontsize=12, fontweight='bold',
        )

    # Shared colorbar
    cbar = fig.colorbar(
        im, ax=axes_flat, shrink=0.6, pad=0.02,
        orientation='horizontal', aspect=40, extend='both',
    )
    cbar.set_label(unit_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    if out_filename is None:
        slug = title.replace(' ', '_').replace('/', '_')
        out_filename = f'Era_Maps_{slug}.png'
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Era raster maps saved to {out_path}')


def create_actual_vs_predicted_maps(
    actual_dir: str,
    predicted_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str = 'Annual Withdrawal',
    unit_label: str = 'Depth (mm)',
    cmap: str = 'Spectral_r',
    diff_cmap: str = 'RdBu_r',
    start_year: int = 1984,
    end_year: int = 2024,
    out_filename: str = 'Actual_vs_Predicted.png',
) -> None:
    """Create a 3-panel figure comparing actual and predicted raster means.

    Layout: Actual (mean) | Predicted (mean) | Difference (Predicted − Actual).
    Actual no-data regions (unmetered areas outside AMA/INA) are shown in gray.
    Groundwater basin boundaries and AMA/INA labels are overlaid on all panels.

    Args:
        actual_dir (str): Directory containing actual meter-based rasters
            (``GW_<year>.tif``).
        predicted_dir (str): Directory containing predicted rasters
            (``*_<year>_*.tif``).
        basin_shp (str): Path to GW basin boundary shapefile.
        output_dir (str): Where to save the figure.
        title (str): Figure super-title.
        unit_label (str): Colorbar label.
        cmap (str): Colormap for actual and predicted panels.
        diff_cmap (str): Diverging colormap for the difference panel.
        start_year (int): First year of the comparison period.
        end_year (int): Last year of the comparison period.
        out_filename (str): Output PNG filename.

    Returns:
        None.
    """
    import rasterio as rio

    apply_journal_style()
    makedirs(output_dir)

    # ---- Discover raster files ----
    actual_files = sorted(f for f in os.listdir(actual_dir) if f.endswith('.tif'))
    predicted_files = sorted(f for f in os.listdir(predicted_dir) if f.endswith('.tif'))
    if not actual_files or not predicted_files:
        logger.warning('Missing actual or predicted rasters — skipping comparison.')
        return

    # ---- Get spatial metadata from first actual raster ----
    template = os.path.join(actual_dir, actual_files[0])
    with rio.open(template) as src:
        extent_actual = [src.bounds.left, src.bounds.right,
                         src.bounds.bottom, src.bounds.top]
        crs_actual = src.crs
        shape_actual = src.shape

    # ---- Get predicted spatial metadata ----
    template_pred = os.path.join(predicted_dir, predicted_files[0])
    with rio.open(template_pred) as src:
        extent_pred = [src.bounds.left, src.bounds.right,
                       src.bounds.bottom, src.bounds.top]
        shape_pred = src.shape

    # ---- Accumulate means over common year range ----
    def _accumulate(directory, files, shape):
        total = np.zeros(shape, dtype=np.float64)
        valid_count = np.zeros(shape, dtype=np.float64)
        for fname in files:
            year = _extract_year(fname)
            if year is None or year < start_year or year > end_year:
                continue
            with rio.open(os.path.join(directory, fname)) as src:
                arr = src.read(1).astype(np.float64)
            mask = ~np.isnan(arr) & (arr != 0)
            total[mask] += arr[mask]
            valid_count[mask] += 1
        with np.errstate(invalid='ignore'):
            mean = np.where(valid_count > 0, total / valid_count, 0.0)
        return mean, valid_count

    actual_mean, actual_count = _accumulate(actual_dir, actual_files, shape_actual)
    pred_mean, pred_count = _accumulate(predicted_dir, predicted_files, shape_pred)

    # Mask: actual pixels with no data across ALL years → gray
    actual_masked = np.ma.masked_where(actual_count == 0, actual_mean)
    pred_masked = np.ma.masked_where(pred_count == 0, pred_mean)

    # Difference: only where actual has data
    # Resample predicted to actual grid if shapes differ
    if shape_actual != shape_pred:
        from scipy.ndimage import zoom
        zoom_factors = (shape_actual[0] / shape_pred[0],
                        shape_actual[1] / shape_pred[1])
        pred_for_diff = zoom(pred_mean, zoom_factors, order=1)
    else:
        pred_for_diff = pred_mean

    diff = pred_for_diff - actual_mean
    diff_masked = np.ma.masked_where(actual_count == 0, diff)

    # ---- Load basin boundaries ----
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs_actual:
        basins_gdf = basins_gdf.to_crs(crs_actual)
    name_col = 'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns else basins_gdf.columns[0]
    ama_ina = get_ama_ina_basin_names()

    # ---- Color limits (shared for actual and predicted) ----
    all_vals = np.concatenate([
        actual_masked.compressed(), pred_masked.compressed(),
    ])
    if len(all_vals) == 0:
        logger.warning('No valid data for actual vs predicted — skipping.')
        return
    v_min = float(np.nanpercentile(all_vals, 2))
    v_max = float(np.nanpercentile(all_vals, 98))

    # Difference limits (symmetric)
    diff_vals = diff_masked.compressed()
    if len(diff_vals) > 0:
        d_abs = max(abs(np.nanpercentile(diff_vals, 2)),
                    abs(np.nanpercentile(diff_vals, 98)))
    else:
        d_abs = 1.0

    # ---- Create 1×3 figure ----
    fig, axes = plt.subplots(
        1, 3, figsize=(21, 7), constrained_layout=True,
    )
    fig.suptitle(
        f'{title} — Actual vs Predicted ({start_year}–{end_year} Mean)',
        fontsize=15, fontweight='bold',
    )

    panels = [
        ('(a) Actual (Metered)', actual_masked, extent_actual,
         cmap, v_min, v_max),
        ('(b) Predicted (ML)', pred_masked, extent_pred,
         cmap, v_min, v_max),
        ('(c) Difference (Predicted − Actual)', diff_masked, extent_actual,
         diff_cmap, -d_abs, d_abs),
    ]

    for ax, (panel_title, data, ext, cm, lo, hi) in zip(axes, panels):
        ax.set_facecolor('#D5D5D5')  # gray for no-data
        im = ax.imshow(
            data, extent=ext, origin='upper',
            cmap=cm, vmin=lo, vmax=hi,
            interpolation='nearest',
        )
        _overlay_boundaries(ax, basins_gdf, ama_ina, name_col)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')

        label = f'Δ {unit_label}' if 'Difference' in panel_title else unit_label
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label=label, extend='both')

    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Actual vs predicted maps saved to {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Trend analysis (Mann-Kendall + Sen's slope)
# ═════════════════════════════════════════════════════════════════════════════

def _mk_sen_pixel(values: np.ndarray, years: np.ndarray,
                   alpha: float = 0.05):
    """Mann-Kendall test + Sen's slope for a single pixel time series.

    Returns (slope, p_value, significant) or (NaN, NaN, False) when the
    time series has fewer than 4 finite observations.
    """
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return np.nan, np.nan, False
    y = values[finite]
    x = years[finite]
    # Sen's slope (Theil-Sen estimator)
    slope = stats.theilslopes(y, x, alpha=1 - alpha)[0]
    # Mann-Kendall via Kendall's tau correlation with time
    _, p_value = stats.kendalltau(x, y)
    return float(slope), float(p_value), p_value < alpha


def _rasterize_zones(
    gdf: gpd.GeoDataFrame,
    name_col: str,
    raster_shape: tuple[int, int],
    transform,
) -> tuple[np.ndarray, dict[int, str]]:
    """Rasterize vector polygons to a label grid matching the raster.

    Returns (label_grid, id_to_name) where label_grid has integer zone
    IDs (0 = no zone) and id_to_name maps IDs to zone names.
    """
    from rasterio.features import rasterize as rio_rasterize

    names = gdf[name_col].unique().tolist()
    name_to_id = {n: i + 1 for i, n in enumerate(names)}
    id_to_name = {v: k for k, v in name_to_id.items()}

    shapes = [
        (row.geometry, name_to_id[row[name_col]])
        for _, row in gdf.iterrows()
        if row[name_col] in name_to_id
    ]
    label_grid = rio_rasterize(
        shapes, out_shape=raster_shape, transform=transform,
        fill=0, dtype=np.int32,
    )
    return label_grid, id_to_name


def _compute_zonal_trend_stats(
    slope_map: np.ndarray,
    pval_map: np.ndarray,
    sig_map: np.ndarray,
    all_nan: np.ndarray,
    label_grid: np.ndarray,
    id_to_name: dict[int, str],
    alpha: float,
) -> pd.DataFrame:
    """Compute per-zone trend statistics from pixel-wise results.

    Returns a DataFrame with one row per zone and columns:
    Region, N_Pixels, Median_Slope, Mean_Slope, Mean_Slope_Sig,
    Pct_Sig_Increase, Pct_Sig_Decrease, Pct_Not_Sig,
    P10_Slope, P90_Slope, Median_P_Value.
    """
    records = []
    for zone_id, zone_name in sorted(id_to_name.items()):
        in_zone = (label_grid == zone_id) & ~all_nan
        n = int(in_zone.sum())
        if n == 0:
            records.append({
                'Region': zone_name, 'N_Pixels': 0,
                'Median_Slope': np.nan, 'Mean_Slope': np.nan,
                'Mean_Slope_Sig': np.nan,
                'Pct_Sig_Increase': 0.0, 'Pct_Sig_Decrease': 0.0,
                'Pct_Not_Sig': 100.0,
                'P10_Slope': np.nan, 'P90_Slope': np.nan,
                'Median_P_Value': np.nan,
            })
            continue

        slopes = slope_map[in_zone]
        pvals = pval_map[in_zone]
        sigs = sig_map[in_zone]

        finite = np.isfinite(slopes)
        slopes_f = slopes[finite]
        pvals_f = pvals[finite & np.isfinite(pvals)]
        sigs_f = sigs[finite]

        n_valid = int(finite.sum())
        if n_valid == 0:
            records.append({
                'Region': zone_name, 'N_Pixels': n,
                'Median_Slope': np.nan, 'Mean_Slope': np.nan,
                'Mean_Slope_Sig': np.nan,
                'Pct_Sig_Increase': 0.0, 'Pct_Sig_Decrease': 0.0,
                'Pct_Not_Sig': 100.0,
                'P10_Slope': np.nan, 'P90_Slope': np.nan,
                'Median_P_Value': np.nan,
            })
            continue

        sig_inc = sigs_f & (slopes_f > 0)
        sig_dec = sigs_f & (slopes_f < 0)
        sig_slopes = slopes_f[sigs_f]

        records.append({
            'Region': zone_name,
            'N_Pixels': n_valid,
            'Median_Slope': float(np.median(slopes_f)),
            'Mean_Slope': float(np.mean(slopes_f)),
            'Mean_Slope_Sig': float(np.mean(sig_slopes)) if len(sig_slopes) > 0 else np.nan,
            'Pct_Sig_Increase': 100 * sig_inc.sum() / n_valid,
            'Pct_Sig_Decrease': 100 * sig_dec.sum() / n_valid,
            'Pct_Not_Sig': 100 * (~sigs_f).sum() / n_valid,
            'P10_Slope': float(np.percentile(slopes_f, 10)),
            'P90_Slope': float(np.percentile(slopes_f, 90)),
            'Median_P_Value': float(np.median(pvals_f)) if len(pvals_f) > 0 else np.nan,
        })

    return pd.DataFrame(records)


def create_trend_maps(
    raster_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str = 'Predicted Annual Withdrawal',
    unit_label: str = 'mm',
    periods: dict[str, tuple[int, int]] | None = None,
    alpha: float = 0.05,
    band: int = 1,
    subbasin_shp: str | None = None,
    basin_col: str = 'BASIN_NAME',
    subbasin_col: str = 'SUBBASIN_N',
) -> None:
    """Create Mann-Kendall / Sen's slope trend maps for each period.

    For every period a single figure is produced showing:

    - **Sen's slope** (unit/year) with a diverging colormap (blue =
      decreasing, red = increasing).
    - **Stippling** on pixels where the Mann-Kendall test is *not*
      significant at level *alpha*, so significant trends appear clean
      and non-significant areas are visually muted.
    - Groundwater basin boundaries and AMA/INA labels overlaid.

    Per-basin and (optionally) per-sub-basin trend statistics CSVs are
    also written alongside the maps.

    Args:
        raster_dir (str): Directory containing ``*.tif`` rasters with years
            in filenames.
        basin_shp (str): Path to GW basin boundary shapefile.
        output_dir (str): Where to save figures.
        title (str): Category name for the figure title.
        unit_label (str): Depth/volume unit (e.g. 'mm', 'AF').
        periods (dict or None): ``{period_name: (start, end)}`` year ranges
            to analyze.  Defaults to the four standard eras plus full period.
        alpha (float): Significance level for Mann-Kendall (default 0.05).
        band (int): Band number to read from each raster (1-based).
        subbasin_shp (str or None): Path to sub-basin shapefile for
            sub-basin statistics.
        basin_col (str): Column in basin shapefile identifying basins.
        subbasin_col (str): Column in sub-basin shapefile identifying
            sub-basins.

    Returns:
        None.
    """
    import rasterio as rio

    apply_journal_style()
    makedirs(output_dir)

    # ── Default periods: full + 3 eras ─
    if periods is None:
        periods = {
            'Full (1896–2099)': (1896, 2099),
            f'Hindcast ({ERA_PERIODS["Hindcast"][0]}–{ERA_PERIODS["Hindcast"][1]})': ERA_PERIODS['Hindcast'],
            f'Historical ({ERA_PERIODS["Historical"][0]}–{ERA_PERIODS["Historical"][1]})': ERA_PERIODS['Historical'],
            f'Projection ({ERA_PERIODS["Projection"][0]}–{ERA_PERIODS["Projection"][1]})': ERA_PERIODS['Projection'],
        }

    # ── Discover raster files and build {year: filepath} ────────────
    tif_files = sorted(f for f in os.listdir(raster_dir) if f.endswith('.tif'))
    if not tif_files:
        logger.warning('No .tif files in %s — skipping trend maps.', raster_dir)
        return

    year_paths: dict[int, str] = {}
    for fname in tif_files:
        yr = _extract_year(fname)
        if yr is not None:
            year_paths[yr] = os.path.join(raster_dir, fname)

    if not year_paths:
        return

    # ── Spatial metadata ────────────────────────────────────────────
    first_path = year_paths[min(year_paths)]
    with rio.open(first_path) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        crs = src.crs
        raster_shape = src.shape
        raster_transform = src.transform

    # ── Basin boundaries ────────────────────────────────────────────
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)
    name_col = (basin_col if basin_col in basins_gdf.columns
                else basins_gdf.columns[0])
    ama_ina = get_ama_ina_basin_names()

    # Rasterize basin zones for zonal statistics
    basin_labels, basin_id_map = _rasterize_zones(
        basins_gdf, name_col, raster_shape, raster_transform,
    )

    # Optionally rasterize sub-basin zones
    subbasin_labels = subbasin_id_map = None
    if subbasin_shp is not None and os.path.isfile(subbasin_shp):
        sub_gdf = gpd.read_file(subbasin_shp)
        if sub_gdf.crs != crs:
            sub_gdf = sub_gdf.to_crs(crs)
        sub_col = (subbasin_col if subbasin_col in sub_gdf.columns
                   else sub_gdf.columns[0])
        subbasin_labels, subbasin_id_map = _rasterize_zones(
            sub_gdf, sub_col, raster_shape, raster_transform,
        )

    n_pixels = raster_shape[0] * raster_shape[1]

    for period_name, (yr_start, yr_end) in periods.items():
        # ── Build year-sorted raster stack ──────────────────────────
        sel_years = sorted(y for y in year_paths if yr_start <= y <= yr_end)
        if len(sel_years) < 4:
            logger.info(f'  Skipping {period_name}: < 4 rasters')
            continue

        years_arr = np.array(sel_years)
        stack = np.full((len(sel_years), *raster_shape), np.nan,
                        dtype=np.float32)
        for i, yr in enumerate(sel_years):
            with rio.open(year_paths[yr]) as src:
                if band <= src.count:
                    stack[i] = src.read(band).astype(np.float32)

        # Flatten to (n_years, n_pixels) for parallel processing
        stack_2d = stack.reshape(len(sel_years), n_pixels)

        # ── Pixel-wise Mann-Kendall + Sen's slope (parallel) ───────
        n_cores = max(1, multiprocessing.cpu_count() - 2)
        results = Parallel(n_jobs=n_cores, prefer='threads')(
            delayed(_mk_sen_pixel)(stack_2d[:, j], years_arr, alpha)
            for j in range(n_pixels)
        )

        slope_flat = np.array([r[0] for r in results], dtype=np.float32)
        pval_flat = np.array([r[1] for r in results], dtype=np.float32)
        sig_flat = np.array([r[2] for r in results], dtype=bool)

        slope_map = slope_flat.reshape(raster_shape)
        pval_map = pval_flat.reshape(raster_shape)
        sig_map = sig_flat.reshape(raster_shape)

        # Mask pixels that were NaN in ALL years (outside domain)
        all_nan = np.all(np.isnan(stack), axis=0)
        slope_masked = np.ma.masked_where(all_nan, slope_map)

        # ── Zonal trend statistics (basin + sub-basin CSVs) ─────────
        slug = (f'{title}_{period_name}'
                .replace(' ', '_').replace('–', '-')
                .replace('(', '').replace(')', ''))

        basin_stats = _compute_zonal_trend_stats(
            slope_map, pval_map, sig_map, all_nan,
            basin_labels, basin_id_map, alpha,
        )
        basin_stats.insert(0, 'Category', title)
        basin_stats.insert(1, 'Period', period_name)
        basin_csv = os.path.join(output_dir, f'Basin_Trend_{slug}.csv')
        basin_stats.to_csv(basin_csv, index=False)

        if subbasin_labels is not None:
            sub_stats = _compute_zonal_trend_stats(
                slope_map, pval_map, sig_map, all_nan,
                subbasin_labels, subbasin_id_map, alpha,
            )
            sub_stats.insert(0, 'Category', title)
            sub_stats.insert(1, 'Period', period_name)
            sub_csv = os.path.join(output_dir, f'Subbasin_Trend_{slug}.csv')
            sub_stats.to_csv(sub_csv, index=False)

        # ── Plot ────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 9),
                               constrained_layout=True)
        ax.set_facecolor('#D5D5D5')

        # Symmetric color limits from data
        valid_slopes = slope_masked.compressed()
        if len(valid_slopes) == 0:
            plt.close(fig)
            continue
        abs_max = max(
            abs(np.nanpercentile(valid_slopes, 2)),
            abs(np.nanpercentile(valid_slopes, 98)),
            1e-6,
        )

        im = ax.imshow(
            slope_masked, extent=extent, origin='upper',
            cmap='RdBu_r', vmin=-abs_max, vmax=abs_max,
            interpolation='nearest',
        )

        # Stipple non-significant pixels (light gray dots)
        nonsig = ~sig_map & ~all_nan
        if nonsig.any():
            rows, cols = np.where(nonsig)
            # Sub-sample stipple points if too dense (max ~4000 dots)
            n_nonsig = len(rows)
            step = max(1, n_nonsig // 4000)
            row_coords = (extent[3]
                          - (rows[::step] + 0.5)
                          * (extent[3] - extent[2]) / raster_shape[0])
            col_coords = (extent[0]
                          + (cols[::step] + 0.5)
                          * (extent[1] - extent[0]) / raster_shape[1])
            ax.scatter(col_coords, row_coords, s=0.15, c='#888888',
                       alpha=0.4, marker='.', linewidths=0)

        _overlay_boundaries(ax, basins_gdf, ama_ina, name_col)

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, extend='both')
        cbar.set_label(f"Sen's slope ({unit_label}/year)", fontsize=12,
                       fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(f'{title} — Trend {period_name}\n'
                     f"(stipple = not significant at α = {alpha})",
                     fontsize=13, fontweight='bold')

        # ── Inset: fraction of significant pixels ───────────────────
        domain_pixels = (~all_nan).sum()
        if domain_pixels > 0:
            n_sig_inc = (sig_map & (slope_map > 0) & ~all_nan).sum()
            n_sig_dec = (sig_map & (slope_map < 0) & ~all_nan).sum()
            pct_inc = 100 * n_sig_inc / domain_pixels
            pct_dec = 100 * n_sig_dec / domain_pixels
            pct_ns = 100 - pct_inc - pct_dec
            summary = (f'Significant: ↑{pct_inc:.1f}%  ↓{pct_dec:.1f}%'
                       f'  n.s. {pct_ns:.1f}%')
            ax.text(
                0.02, 0.02, summary, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          alpha=0.85, lw=0.5),
            )

        out_path = os.path.join(output_dir, f'Trend_{slug}.png')
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Trend maps for {title} saved to {output_dir}')


# ═════════════════════════════════════════════════════════════════════════════
# Shared intercomparison plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def plot_intercomp_time_series(
    all_sources: dict[str, dict],
    categories: list[str],
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
    *,
    colors: dict[str, str],
    markers: dict[str, str],
    labels: dict[str, str] | None = None,
    title_prefix: str = '',
    file_prefix: str = 'TS',
    mode: str = 'volume',
    af_to_m3: float = 1233.48184,
    m_to_mm: float = 1000.0,
    mm_to_ft: float = 1.0 / 304.8,
    m3_to_af: float = 1 / 1233.48184,
) -> None:
    """Generic per-basin intercomparison time series.

    Args:
        all_sources (dict): ``{source_name: {cat_key: {'mean': ...,
            'yearly': {year: {basin: val}}}}}``  Values are in AF for
            ``mode='volume'`` or dimensionless for ``mode='ratio'``.
        categories (list[str]): Category keys to iterate over.
        basin_names (list[str]): Basin identifiers.
        basin_areas_m2 (dict[str, float]): Basin areas for depth conversion.
        output_dir (str): Directory for saved plots.
        colors (dict[str, str]): Per-source colours.
        markers (dict[str, str]): Per-source markers.
        labels (dict or None): Display labels per source.
        title_prefix (str): Prepended to plot titles.
        file_prefix (str): Prepended to filenames.
        mode (str): ``'volume'`` → 2-row (depth + volume).
            ``'ratio'`` → 1-row (dimensionless).
        af_to_m3 (float): Conversion factor AF to m³.
        m_to_mm (float): Conversion factor metres to mm.
        mm_to_ft (float): Conversion factor mm to ft.
        m3_to_af (float): Conversion factor m³ to AF.

    Returns:
        None.
    """
    makedirs(output_dir)
    if labels is None:
        labels = {k: k for k in all_sources}

    targets = list(basin_names) + ['AZ_Total']

    for cat in categories:
        for basin in targets:
            if mode == 'volume':
                fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                                         constrained_layout=True)
            else:
                fig, axes = plt.subplots(1, 1, figsize=(10, 4),
                                         constrained_layout=True)
                axes = [axes]

            basin_title = basin.replace('_', ' ')
            cat_title = f'{title_prefix}{cat}' if title_prefix else cat
            fig.suptitle(f'{cat_title} — {basin_title}',
                         fontsize=14, fontweight='bold')

            if basin == 'AZ_Total':
                total_area = sum(basin_areas_m2.values())
            else:
                total_area = basin_areas_m2.get(basin, 1.0)

            for source, src_data in all_sources.items():
                yearly = src_data.get(cat, {}).get('yearly', {})
                years = sorted(yearly.keys())
                if not years:
                    continue

                color = colors.get(source, '#333333')
                marker = markers.get(source, 'o')
                label = labels.get(source, source)

                if mode == 'volume':
                    if basin == 'AZ_Total':
                        af_vals = np.array([
                            sum(yearly[yr].values()) for yr in years
                        ])
                    else:
                        af_vals = np.array([
                            yearly[yr].get(basin, 0.0) for yr in years
                        ])

                    m3_vals = af_vals * af_to_m3
                    mm_vals = (m3_vals / total_area * m_to_mm
                               if total_area > 0 else m3_vals * 0)

                    axes[0].plot(years, mm_vals, label=label, color=color,
                                 marker=marker, markersize=3, linewidth=1.2)
                    axes[1].plot(years, m3_vals, label=label, color=color,
                                 marker=marker, markersize=3, linewidth=1.2)
                else:
                    # Ratio mode (e.g. IE)
                    if basin == 'AZ_Total':
                        ie_vals = []
                        for yr in years:
                            yr_d = yearly[yr]
                            vals = [yr_d.get(b, np.nan) for b in basin_names]
                            areas = [basin_areas_m2.get(b, 0) for b in basin_names]
                            finite = [(v, a) for v, a in zip(vals, areas)
                                      if np.isfinite(v)]
                            if finite:
                                v_arr = np.array([x[0] for x in finite])
                                a_arr = np.array([x[1] for x in finite])
                                ie_vals.append(float(np.average(v_arr, weights=a_arr)))
                            else:
                                ie_vals.append(np.nan)
                        plot_vals = np.array(ie_vals)
                    else:
                        plot_vals = np.array([
                            yearly[yr].get(basin, np.nan) for yr in years
                        ])
                    axes[0].plot(years, plot_vals, label=label, color=color,
                                 marker=marker, markersize=3, linewidth=1.2)

            # ── Axis formatting ──
            if mode == 'volume':
                axes[0].set_ylabel('Depth (mm)')
                axes[0].grid(True, alpha=0.3, linestyle='--')
                axes[0].legend(fontsize=9)
                ax_ft = axes[0].twinx()
                ax_ft.set_ylabel('Depth (ft)')
                lo, hi = axes[0].get_ylim()
                ax_ft.set_ylim(lo * mm_to_ft, hi * mm_to_ft)

                axes[1].set_ylabel(r'Volume (m$^3$)')
                axes[1].set_xlabel('Year')
                axes[1].grid(True, alpha=0.3, linestyle='--')
                axes[1].legend(fontsize=9)
                ax_af = axes[1].twinx()
                ax_af.set_ylabel('Volume (AF)')
                lo, hi = axes[1].get_ylim()
                ax_af.set_ylim(lo * m3_to_af, hi * m3_to_af)
            else:
                axes[0].set_ylabel(cat_title)
                axes[0].set_xlabel('Year')
                axes[0].grid(True, alpha=0.3, linestyle='--')
                axes[0].legend(fontsize=9)

            clean = basin.replace(' ', '_').replace('/', '_').replace('.', '')
            out_path = os.path.join(output_dir,
                                    f'{file_prefix}_{cat}_{clean}.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    logger.info(f'Time series plots saved to {output_dir}')


def plot_intercomp_scatter(
    pairs: list[tuple[str, str, dict, dict]],
    basin_names: list[str],
    basin_areas_m2: dict[str, float],
    output_dir: str,
    *,
    title: str = 'Per-Basin Scatter',
    filename: str = 'Scatter.png',
    mode: str = 'volume',
    af_to_m3: float = 1233.48184,
    m_to_mm: float = 1000.0,
) -> None:
    """Generic intercomparison scatter plots with 1:1 line and linear fit.

    Args:
        pairs (list): List of (label_a, label_b, mean_a, mean_b) tuples.
            ``mean_a``/``mean_b`` are ``{basin: value}`` dicts.
        basin_names (list[str]): Basin identifiers.
        basin_areas_m2 (dict[str, float]): Basin areas for mm conversion.
        output_dir (str): Directory for saved plots.
        title (str): Figure suptitle.
        filename (str): Output filename.
        mode (str): ``'volume'`` → 2 rows (AF, mm).
            ``'ratio'`` → 1 row.
        af_to_m3 (float): Conversion factor AF to m³.
        m_to_mm (float): Conversion factor metres to mm.

    Returns:
        None.
    """
    makedirs(output_dir)
    n_pairs = len(pairs)
    n_rows = 2 if mode == 'volume' else 1

    fig, axes = plt.subplots(
        n_rows, n_pairs, figsize=(7 * n_pairs, 5 * n_rows),
        constrained_layout=True, squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for col_i, (label_a, label_b, mean_a, mean_b) in enumerate(pairs):
        vals_a = np.array([mean_a.get(b, 0.0) for b in basin_names])
        vals_b = np.array([mean_b.get(b, 0.0) for b in basin_names])

        if mode == 'ratio':
            # Filter NaN for ratio mode
            valid = np.isfinite(vals_a) & np.isfinite(vals_b)
            vals_a = vals_a[valid]
            vals_b = vals_b[valid]

        if mode == 'volume':
            areas = np.array([basin_areas_m2.get(b, 1.0) for b in basin_names])
            mm_a = vals_a * af_to_m3 / areas * m_to_mm
            mm_b = vals_b * af_to_m3 / areas * m_to_mm
            row_data = [(vals_a, vals_b, 'AF'), (mm_a, mm_b, 'mm')]
        else:
            row_data = [(vals_a, vals_b, '')]

        for row_i, (vx, vy, unit) in enumerate(row_data):
            ax = axes[row_i, col_i]
            if vx.size == 0:
                ax.set_title(f'{label_a} vs {label_b}')
                continue

            ax.scatter(vx, vy, s=30, alpha=0.7,
                       edgecolors='white', linewidths=0.5)

            lo = min(vx.min(), vy.min(), 0) if mode == 'volume' else min(vx.min(), vy.min()) * 0.9
            hi = max(vx.max(), vy.max()) * 1.05
            if hi <= lo:
                hi = lo + 1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')

            if len(vx) > 1 and np.std(vx) > 0:
                from sklearn.metrics import r2_score as _r2_score
                from hydrolibs.mlops import (
                    normalized_rmse, normalized_mae, normalized_mbe,
                )
                z = np.polyfit(vx, vy, 1)
                x_fit = np.linspace(lo, hi, 100)
                sign = '\u2212' if z[1] < 0 else '+'
                ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                        label=f'y={z[0]:.2f}x {sign} {abs(z[1]):.1f}')
                r2 = _r2_score(vy, vx)
                rmse_pct = normalized_rmse(vy, vx)
                mae_pct = normalized_mae(vy, vx)
                mbe_pct = normalized_mbe(vy, vx)
                mbe_sign = '\u2212' if mbe_pct < 0 else ''
                metrics_text = (f'R²={r2:.3f}\n'
                                f'RMSE={rmse_pct:.1f}%\n'
                                f'MAE={mae_pct:.1f}%\n'
                                f'MBE={mbe_sign}{abs(mbe_pct):.1f}%')
                ax.text(0.97, 0.03, metrics_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='gray'))
                unit_label = f'  ({unit})' if unit else ''
                ax.set_title(f'{label_a} vs {label_b}{unit_label}',
                             fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{label_a} vs {label_b}', fontsize=11)

            unit_suffix = f' ({unit})' if unit else ''
            ax.set_xlabel(f'{label_a}{unit_suffix}')
            ax.set_ylabel(f'{label_b}{unit_suffix}')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Scatter plot saved to {out_path}')


def plot_intercomp_taylor(
    all_sources: dict[str, dict],
    pairs: list[tuple[str, str]],
    categories: list[str],
    basin_names: list[str],
    output_dir: str,
    *,
    pair_colors: dict[str, str] | None = None,
    title_prefix: str = '',
    file_prefix: str = 'Taylor',
) -> None:
    """Generic Taylor diagram for intercomparison datasets.

    Args:
        all_sources (dict): ``{source_name: {cat_key: {'yearly':
            {year: {basin: val}}}}}``.
        pairs (list): List of (src_a, src_b) tuples.  Second element is
            the reference.
        categories (list[str]): Category keys to iterate over.
        basin_names (list[str]): Basin identifiers.
        output_dir (str): Directory for saved plots.
        pair_colors (dict or None): ``{'src_a vs src_b': '#color'}``.
            Auto-generated if None.
        title_prefix (str): Prepended to diagram titles.
        file_prefix (str): Prepended to filenames.

    Returns:
        None.
    """
    makedirs(output_dir)
    default_palette = ['#2C3E50', '#E67E22', '#27AE60', '#E74C3C', '#8E44AD']

    if pair_colors is None:
        pair_colors = {}
        for i, (sa, sb) in enumerate(pairs):
            pair_colors[f'{sa} vs {sb}'] = default_palette[i % len(default_palette)]

    for cat in categories:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)

        # Reference arc (normalized std = 1)
        theta_ref = np.linspace(0, np.pi / 2, 100)
        ax.plot(theta_ref, np.ones_like(theta_ref), 'k--', lw=1, label='Reference')

        # Centered RMSD arcs
        for rmsd_val in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            theta_arc = np.linspace(0, np.pi / 2, 200)
            r_arc = []
            for t in theta_arc:
                disc = np.cos(t)**2 - (1 - rmsd_val**2)
                if disc >= 0:
                    r_arc.append(np.cos(t) + np.sqrt(disc))
                else:
                    r_arc.append(np.nan)
            ax.plot(theta_arc, r_arc, 'grey', lw=0.5, alpha=0.4)
            valid = [(t, r) for t, r in zip(theta_arc, r_arc) if np.isfinite(r)]
            if valid:
                t_mid, r_mid = valid[len(valid) // 2]
                ax.annotate(f'{rmsd_val:.2f}', (t_mid, r_mid), fontsize=6,
                            color='grey', alpha=0.7)

        for src_a, src_b in pairs:
            pair_key = f'{src_a} vs {src_b}'
            yearly_a_all = all_sources.get(src_a, {}).get(cat, {}).get('yearly', {})
            yearly_b_all = all_sources.get(src_b, {}).get(cat, {}).get('yearly', {})
            common_years = sorted(set(yearly_a_all.keys()) & set(yearly_b_all.keys()))
            if not common_years:
                continue

            for basin in basin_names:
                ts_a = np.array([yearly_a_all[yr].get(basin, 0.0)
                                 for yr in common_years])
                ts_b = np.array([yearly_b_all[yr].get(basin, 0.0)
                                 for yr in common_years])
                if np.all(ts_a == 0) or np.all(ts_b == 0):
                    continue
                std_a = np.std(ts_a)
                std_b = np.std(ts_b)
                if std_b == 0:
                    continue
                sigma_n = std_a / std_b
                r = float(np.corrcoef(ts_a, ts_b)[0, 1]) if std_a > 0 else 0.0
                r = max(r, 0)
                theta = np.arccos(r)
                ax.plot(theta, sigma_n, 'o',
                        color=pair_colors.get(pair_key, '#333333'),
                        markersize=4, alpha=0.5)

        # Legend entries
        for pair_key, color in pair_colors.items():
            ax.plot([], [], 'o', color=color, markersize=6, label=pair_key)

        ax.set_rlabel_position(0)
        ax.set_ylabel('Normalized Std Dev', labelpad=30)
        ax.set_xlabel('Correlation', labelpad=10)
        corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                      0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_thetagrids(
            [np.degrees(np.arccos(c)) for c in corr_ticks],
            labels=[f'{c:.2f}' for c in corr_ticks], fontsize=7,
        )
        cat_title = f'{title_prefix}{cat}' if title_prefix else cat
        ax.set_title(f'Taylor Diagram — {cat_title}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.05))

        out_path = os.path.join(output_dir, f'{file_prefix}_{cat}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Taylor diagrams saved to {output_dir}')


def plot_temporal_heatmap(
    temporal_basin_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Heatmap of per-basin Pearson r and NSE, one panel per category.

    Args:
        temporal_basin_df (pd.DataFrame): Must contain columns: Category,
            Pair, Basin, Pearson_r, NSE.
        output_dir (str): Directory for saved plots.

    Returns:
        None.
    """
    import matplotlib.colors as mcolors
    makedirs(output_dir)

    for metric, vmin, vmax, cmap in [
        ('Pearson_r', -1, 1, 'RdYlGn'),
        ('NSE', -1, 1, 'RdYlBu'),
    ]:
        categories = sorted(temporal_basin_df['Category'].unique())
        n_cats = len(categories)
        fig, axes = plt.subplots(
            1, n_cats,
            figsize=(7 * n_cats, max(6, 0.35 * temporal_basin_df['Basin'].nunique())),
            constrained_layout=True, squeeze=False,
        )
        fig.suptitle(f'Temporal Agreement — {metric}', fontsize=14, fontweight='bold')
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        for ax, cat in zip(axes[0], categories):
            sub = temporal_basin_df[temporal_basin_df['Category'] == cat]
            pivot = sub.pivot(index='Basin', columns='Pair', values=metric)
            pivot = pivot.sort_index()

            im = ax.imshow(
                pivot.values, aspect='auto', cmap=cmap, norm=norm,
                interpolation='nearest',
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title(cat, fontsize=12)

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color='black' if abs(val) < 0.6 else 'white')

        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label=metric)
        out_path = os.path.join(output_dir, f'Heatmap_{metric}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Temporal heatmaps saved to {output_dir}')


def plot_temporal_box_violin(
    temporal_basin_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Box + overlaid violin plots of per-basin r/NSE distributions.

    Args:
        temporal_basin_df (pd.DataFrame): Must contain columns: Category,
            Pair, Basin, Pearson_r, NSE.
        output_dir (str): Directory for saved plots.

    Returns:
        None.
    """
    makedirs(output_dir)

    for metric, ylabel in [('Pearson_r', 'Pearson r'), ('NSE', 'NSE')]:
        categories = sorted(temporal_basin_df['Category'].unique())
        n_cats = len(categories)
        fig, axes = plt.subplots(
            1, n_cats, figsize=(5 * n_cats, 6),
            constrained_layout=True, squeeze=False, sharey=True,
        )
        fig.suptitle(f'Distribution of Per-Basin {ylabel}', fontsize=14, fontweight='bold')

        for ax, cat in zip(axes[0], categories):
            sub = temporal_basin_df[temporal_basin_df['Category'] == cat].dropna(subset=[metric])
            pair_list = sorted(sub['Pair'].unique())
            data = [sub.loc[sub['Pair'] == p, metric].values for p in pair_list]

            vp = ax.violinplot(data, positions=range(len(pair_list)), showextrema=False)
            for body in vp['bodies']:
                body.set_alpha(0.25)
            bp = ax.boxplot(
                data, positions=range(len(pair_list)), widths=0.3,
                patch_artist=True, showfliers=True,
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
            )
            for patch in bp['boxes']:
                patch.set_facecolor('#AED6F1')
                patch.set_alpha(0.7)

            ax.set_xticks(range(len(pair_list)))
            ax.set_xticklabels(pair_list, rotation=30, ha='right', fontsize=9)
            ax.set_title(cat, fontsize=12)
            ax.set_ylabel(ylabel)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        out_path = os.path.join(output_dir, f'BoxViolin_{metric}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'Box/violin plots saved to {output_dir}')


def plot_temporal_r_vs_nse(
    temporal_basin_df: pd.DataFrame,
    output_dir: str,
    *,
    pair_colors: dict[str, str] | None = None,
) -> None:
    """Paired scatter of per-basin Pearson r vs NSE for each dataset pair.

    Args:
        temporal_basin_df (pd.DataFrame): Must contain columns: Category,
            Pair, Basin, Pearson_r, NSE.
        output_dir (str): Directory for saved plots.
        pair_colors (dict or None): ``{'pair_label': '#color'}``.  Falls
            back to grey if missing.

    Returns:
        None.
    """
    makedirs(output_dir)
    if pair_colors is None:
        pair_colors = {}

    for cat in sorted(temporal_basin_df['Category'].unique()):
        sub = temporal_basin_df[temporal_basin_df['Category'] == cat].dropna(
            subset=['Pearson_r', 'NSE'],
        )
        pair_list = sorted(sub['Pair'].unique())
        n_pairs = len(pair_list)
        fig, axes = plt.subplots(
            1, n_pairs, figsize=(6 * n_pairs, 5),
            constrained_layout=True, squeeze=False,
        )
        fig.suptitle(f'Pearson r vs NSE — {cat}', fontsize=14, fontweight='bold')

        for ax, pair in zip(axes[0], pair_list):
            psub = sub[sub['Pair'] == pair]
            color = pair_colors.get(pair, '#555555')
            ax.scatter(psub['Pearson_r'], psub['NSE'], s=35, alpha=0.7,
                       edgecolors='white', linewidths=0.5, color=color)

            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

            q1 = ((psub['Pearson_r'] > 0) & (psub['NSE'] > 0)).sum()
            q2 = ((psub['Pearson_r'] < 0) & (psub['NSE'] > 0)).sum()
            q3 = ((psub['Pearson_r'] < 0) & (psub['NSE'] < 0)).sum()
            q4 = ((psub['Pearson_r'] > 0) & (psub['NSE'] < 0)).sum()
            ax.text(0.95, 0.95, f'n={q1}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='green')
            ax.text(0.05, 0.95, f'n={q2}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color='orange')
            ax.text(0.05, 0.05, f'n={q3}', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=8, color='red')
            ax.text(0.95, 0.05, f'n={q4}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='blue')

            ax.set_xlabel('Pearson r')
            ax.set_ylabel('NSE')
            ax.set_title(pair, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-1.05, 1.05)

        out_path = os.path.join(output_dir, f'r_vs_NSE_{cat}.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'r vs NSE scatter plots saved to {output_dir}')


# ---------------------------------------------------------------------------
# Multi-seed / multi-test-size boxplot visualizations
# ---------------------------------------------------------------------------

_METRIC_LABELS: dict[str, str] = {
    'Test_R2': 'Test R²',
    'Test_RMSE': 'Test RMSE (%)',
    'Test_MAE': 'Test MAE (%)',
    'Overfit_R2': 'Overfit R² (Train − Test)',
    'Val_R2': 'CV Validation R²',
    'Val_RMSE': 'CV Validation RMSE (%)',
}


def plot_grid_boxplots(
        all_runs_csv: str,
        output_dir: str,
        strategy_label: str = 'Random',
        metrics: tuple[str, ...] = ('Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfit_R2'),
) -> None:
    """Boxplots of metric distributions across the seed × test-size grid.

    For each metric a figure is produced with one subplot per test_size,
    models on the x-axis.

    Args:
        all_runs_csv (str): Path to ``All_Runs.csv`` produced by
            ``evaluate_random`` or ``evaluate_pixel_holdout``.
        output_dir (str): Directory for saved figures.
        strategy_label (str): Label used in figure titles
            (e.g. ``'Random'``, ``'Pixel Holdout'``).
        metrics (tuple[str,...]): Metric column names to plot.
    """
    apply_journal_style()
    makedirs(output_dir)
    df = pd.read_csv(all_runs_csv)

    for metric in metrics:
        if metric not in df.columns:
            continue
        label = _METRIC_LABELS.get(metric, metric)
        test_sizes = sorted(df['test_size'].unique())
        n_ts = len(test_sizes)

        fig, axes = plt.subplots(
            1, n_ts, figsize=(5 * n_ts, 6),
            constrained_layout=True, squeeze=False, sharey=True,
        )
        fig.suptitle(f'{strategy_label}: {label} Distribution',
                     fontsize=14, fontweight='bold')

        for ax, ts in zip(axes[0], test_sizes):
            sub = df[df['test_size'] == ts]
            models = sorted(sub['Model'].unique())
            data = [sub.loc[sub['Model'] == m, metric].dropna().values
                    for m in models]

            bp = ax.boxplot(
                data, positions=range(len(models)), widths=0.45,
                patch_artist=True, showfliers=True,
                flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
            )
            palette = sns.color_palette('Set2', len(models))
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            # Overlay individual points
            for i, vals in enumerate(data):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full_like(vals, i) + jitter, vals,
                           color='black', s=15, alpha=0.5, zorder=3)

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'Test size = {ts:.0%}', fontsize=12)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        fname = f'Boxplot_{metric}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'{strategy_label} grid boxplots saved to {output_dir}')


def plot_loo_distribution(
        per_fold_csv: str,
        fold_col: str,
        output_dir: str,
        strategy_label: str = 'Temporal LOO',
        metrics: tuple[str, ...] = ('Test_R2', 'Test_RMSE', 'Test_MAE',
                                    'Test_MBE', 'Overfit_R2'),
) -> None:
    """Box + strip plots of per-fold metric distributions for LOO strategies.

    Produces one figure per metric with models on the x-axis and individual
    fold values overlaid as strip points.

    Args:
        per_fold_csv (str): Path to ``Per_Holdout_Metrics.csv`` or
            ``Per_Subbasin_Metrics.csv``.
        fold_col (str): Column identifying the fold (``'Holdout'`` or
            ``'Subbasin'``).
        output_dir (str): Directory for saved figures.
        strategy_label (str): Label for figure titles.
        metrics (tuple[str,...]): Metric column names to plot.
    """
    apply_journal_style()
    makedirs(output_dir)
    df = pd.read_csv(per_fold_csv)

    # Order models by median Test RMSE (ascending) for consistency with bar charts
    if 'Test_RMSE' in df.columns:
        model_order = (
            df.groupby('Model')['Test_RMSE'].median()
            .sort_values()
            .index.tolist()
        )
    else:
        model_order = sorted(df['Model'].unique())

    for metric in metrics:
        if metric not in df.columns:
            continue
        label = _METRIC_LABELS.get(metric, metric)
        models = model_order
        data = [df.loc[df['Model'] == m, metric].dropna().values
                for m in models]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 6),
                               constrained_layout=True)

        vp = ax.violinplot(data, positions=range(len(models)), showextrema=False)
        for body in vp['bodies']:
            body.set_alpha(0.2)

        bp = ax.boxplot(
            data, positions=range(len(models)), widths=0.35,
            patch_artist=True, showfliers=False,
            flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
        )
        palette = sns.color_palette('Set2', len(models))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual fold points
        for i, vals in enumerate(data):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color='black', s=20, alpha=0.6, zorder=3)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{strategy_label}: {label}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        fname = f'Distribution_{metric}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches='tight')
        plt.close(fig)

    logger.info(f'{strategy_label} distribution plots saved to {output_dir}')


def plot_stratified_metrics(
        strat_df: pd.DataFrame,
        output_dir: str,
        strategy_label: str = 'Spatial LOO',
        metrics: tuple[str, ...] = ('R2', 'RMSE_pct', 'MAE_pct', 'MBE_pct'),
) -> None:
    """Grouped bar charts of test metrics stratified by pumping category.

    For each metric, produces a bar chart with models on the x-axis and
    one group of bars per pumping category (Low / Medium / High).  Bars
    show the mean across all holdout folds; error bars show ± 1 std.

    Args:
        strat_df (pd.DataFrame): DataFrame with columns ``Category``,
            ``Model``, and one or more metric columns.
        output_dir (str): Directory for saved figures.
        strategy_label (str): Label for figure titles.
        metrics (tuple[str,...]): Metric column names to plot.
    """
    apply_journal_style()
    makedirs(output_dir)

    cat_order = [c for c in ['Low (<500 mm)', 'Medium (500-2000 mm)',
                              'High (>=500 mm)', 'High (>2000 mm)']
                 if c in strat_df['Category'].values]
    cat_colors = {'Low (<500 mm)': '#2ECC71',
                  'Medium (500-2000 mm)': '#F39C12',
                  'High (>=500 mm)': '#E74C3C',
                  'High (>2000 mm)': '#E74C3C'}

    metric_labels = {
        'R2': 'Test R²',
        'RMSE_pct': 'Test RMSE (%)',
        'MAE_pct': 'Test MAE (%)',
        'MBE_pct': 'Test MBE (%)',
    }

    # Order models by overall median RMSE for consistency
    if 'RMSE_pct' in strat_df.columns:
        model_order = (
            strat_df.groupby('Model')['RMSE_pct'].median()
            .sort_values()
            .index.tolist()
        )
    else:
        model_order = sorted(strat_df['Model'].unique())

    for metric in metrics:
        if metric not in strat_df.columns:
            continue
        label = metric_labels.get(metric, metric)

        # Aggregate: mean ± std per (Model, Category) across folds
        agg = (
            strat_df.groupby(['Model', 'Category'])[metric]
            .agg(['mean', 'std'])
            .reset_index()
        )

        n_models = len(model_order)
        n_cats = len(cat_order)
        bar_width = 0.8 / max(n_cats, 1)
        x = np.arange(n_models)

        fig, ax = plt.subplots(
            figsize=(max(8, n_models * 1.4), 6), constrained_layout=True)

        for j, cat in enumerate(cat_order):
            cat_data = agg[agg['Category'] == cat].set_index('Model')
            means = [cat_data.loc[m, 'mean'] if m in cat_data.index
                     else np.nan for m in model_order]
            stds = [cat_data.loc[m, 'std'] if m in cat_data.index
                    else 0 for m in model_order]
            offset = (j - (n_cats - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   label=cat, color=cat_colors.get(cat, '#999999'),
                   edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{strategy_label}: {label} by Pumping Category',
                     fontsize=14, fontweight='bold')
        ax.legend(title='Pumping Category', fontsize=9, title_fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        fname = f'Stratified_{metric}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=600,
                    bbox_inches='tight')
        plt.close(fig)

    # Summary table: mean N per category across folds
    if 'N' in strat_df.columns:
        n_summary = (
            strat_df.groupby(['Model', 'Category'])['N']
            .agg(['mean', 'sum'])
            .rename(columns={'mean': 'Mean_N', 'sum': 'Total_N'})
            .reset_index()
            .round(1)
        )
        n_summary.to_csv(
            os.path.join(output_dir, 'Stratified_Sample_Counts.csv'),
            index=False)

    logger.info(f'{strategy_label} stratified metric plots saved to '
                f'{output_dir}')
