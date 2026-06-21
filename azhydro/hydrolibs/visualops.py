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
    """Add a km³ twinx to an AF volume axis. Call after all plotting.

    Tick labels render in km³ (×10⁹ m³) to suppress matplotlib's offset
    notation and match the MAF order of magnitude on the primary axis.
    """
    lo, hi = ax.get_ylim()
    ax_m3 = ax.twinx()
    ax_m3.set_ylim(lo * _AF_TO_M3, hi * _AF_TO_M3)
    _format_volume_axis(ax_m3, unit='m3', label='')
    ax_m3.set_ylabel(r'(km$^3$)', fontweight='bold')
    return ax_m3


def _add_af_twinx(ax):
    """Add a MAF twinx to a m³ volume axis. Call after all plotting.

    Tick labels render in MAF to suppress matplotlib's offset notation
    and match the km³ order of magnitude on the primary m³ axis.
    """
    lo, hi = ax.get_ylim()
    ax_af = ax.twinx()
    ax_af.set_ylim(lo / _AF_TO_M3, hi / _AF_TO_M3)
    _format_volume_axis(ax_af, unit='AF', label='')
    ax_af.set_ylabel('(MAF)', fontweight='bold')
    return ax_af


def _format_volume_axis(ax, unit: str = 'AF', label: str = 'Volume') -> None:
    """Format an AF or m³ volume/sigma axis to a MAF-scale display.

    Suppresses matplotlib's offset notation (no more "1e7" hovering at
    the top of the axis) by converting tick labels to MAF (millions of
    acre-feet) for AF axes, or km³ (×10⁹ m³) for m³ axes.  Both units
    have the same order of magnitude so paired m³/AF twinx panels read
    cleanly side-by-side.

    Args:
        ax: matplotlib axis to restyle.
        unit: ``'AF'`` for acre-foot data, ``'m3'`` (or ``'m³'``) for
            cubic-meter data.
        label: y-axis label prefix (e.g. ``'Volume'``, ``'σ'``,
            ``'σ_total'``); the unit suffix is appended automatically.
    """
    if unit == 'AF':
        scale = 1e6
        ylabel = f'{label} (MAF)'
    elif unit in ('m3', 'm³'):
        scale = 1e9
        ylabel = rf'{label} (km$^3$)'
    else:
        return

    lo, hi = ax.get_ylim()
    disp_max = max(abs(lo), abs(hi)) / scale
    # Switch to scientific notation when fixed-point with 2 decimals
    # would round most ticks to 0.00 (small basins / small σ).  At
    # disp_max < 0.05 the auto-locator places ticks at ≤ 0.01 spacing,
    # which collapses to repeated "0.00"/"0.01" strings under ,.2f.
    if 0 < disp_max < 0.05:
        fmt = lambda x, _: '0' if x == 0 else f'{x / scale:.2e}'
    else:
        fmt = lambda x, _: f'{x / scale:,.2f}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.set_ylabel(ylabel, fontweight='bold')


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

    Includes Ranegras Plain, designated an AMA in January 2026 (ADWR);
    its underlying basin name in the GW basin shapefile remains
    ``'RANEGRAS PLAIN'`` (no ``AMA`` suffix), so callers that classify
    by substring matching (e.g. ``'AMA' in name``) should use
    :func:`get_ama_ina_classification` instead.

    Returns:
        list: List of AMA and INA basin names.
    """
    ama, ina = get_ama_ina_classification()
    return list(ama) + list(ina)


def get_ama_ina_classification() -> tuple[list[str], list[str]]:
    """Return ``(ama_basins, ina_basins)`` tuples by explicit classification.

    Avoids the ``'AMA' in name`` substring trap for basins whose ADWR
    designation differs from their shapefile basin name (e.g.
    ``'RANEGRAS PLAIN'`` was designated an AMA in January 2026 but
    keeps its bare basin name in the GW basin shapefile).

    Returns:
        tuple[list[str], list[str]]: ``(ama_basins, ina_basins)``.
    """
    ama_basins = [
        'SANTA CRUZ AMA',
        'PRESCOTT AMA',
        'TUCSON AMA',
        'PINAL AMA',
        'PHOENIX AMA',
        'DOUGLAS AMA',
        'WILLCOX AMA',
        'RANEGRAS PLAIN',  # designated AMA Jan 2026 (ADWR)
    ]
    ina_basins = [
        'JOSEPH CITY INA',
        'HARQUAHALA INA',
        'HUALAPAI VALLEY INA',
    ]
    return ama_basins, ina_basins


def apply_journal_style():
    """Apply journal-quality matplotlib settings.

    Also installs a targeted ``warnings`` filter that silences the
    ``"overflow encountered in multiply"`` ``RuntimeWarning`` emitted
    by matplotlib's colormap ``__call__`` (``colors.py:778``,
    ``xa *= self.N``).  That overflow fires when a ``MaskedArray`` is
    passed to ``imshow`` and matplotlib internally fills masked pixels
    with a sentinel that, post-Normalize + uint8 cast, wraps around
    the lookup-table size.  It's a long-standing matplotlib internal
    quirk for diverging-cmap renders of masked rasters — the figures
    still save correctly — so the warning is filtered at the message
    level rather than module-wide to avoid hiding genuinely
    informative numpy overflows from elsewhere.

    Returns:
        None.
    """
    import warnings as _warnings
    plt.rcParams.update(JOURNAL_SETTINGS)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    _warnings.filterwarnings(
        'ignore',
        message='overflow encountered in multiply',
        category=RuntimeWarning,
        module=r'matplotlib\.colors',
    )


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
        raster_res (float): Raster resolution in meters.
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
        # Disable KDE when too few unique values (e.g., static columns
        # like wtd_m) — gaussian_kde requires > 1 unique element.
        use_kde = clip_vals.nunique() > 2

        # H1. Overall histogram + KDE
        fig, ax = _plt.subplots(figsize=figsize_box)
        _sns.histplot(clip_vals, kde=use_kde, ax=ax, color='#2C3E50', edgecolor='white', stat='count')
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
                palette=present_palette_h, kde=use_kde, stat='count',
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
            kde=use_kde, stat='count', edgecolor='white', alpha=0.35, ax=ax,
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
                kde=use_kde, stat='count', edgecolor='white', alpha=0.25, ax=ax,
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
    _format_volume_axis(ax2, unit='m3', label='Total Volume')
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
    """Plot a line with ±1 σ shading and era background colors.

    The lower edge of the uncertainty band is clipped at 0 because all
    callers pass non-negative quantities (annual withdrawal depths or
    volumes), so a CI tail extending below 0 is physically meaningless.

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
        lower = np.maximum(mean_vals - std_vals, 0)
        upper = mean_vals + std_vals
        ax.fill_between(years, lower, upper, color=color, alpha=0.18)


def create_basin_time_series(
        basin_yearly: dict[int, dict[str, dict]],
        output_dir: str,
        start_year: int = 1896,
        end_year: int = 2099,
        title_prefix: str = '',
        actual_basin_yearly: dict[int, dict[str, dict]] | None = None,
        sigma_basin_yearly: dict[int, dict[str, dict]] | None = None,
) -> None:
    """Per-basin annual time series (any pool) with era shading + uncertainty.

    Despite the historical ``_GW``-suffixed docstring, this function is
    reused for **every** withdrawal, surface-water, and consumptive-use
    pool (Total_GW, Total_SW, Irrigation, Non_Irrigation, Irrigation_GW,
    Irrigation_SW, Non_Irrigation_GW, Non_Irrigation_SW, Irrigation_CU,
    Irrigation_GW_CU, Irrigation_SW_CU) by :func:`_process_group` in
    ``uncertaintyops._replot_from_augmented_rasters``. The pool identity
    is implied by the enclosing ``{pool}/Basin_Time_Series/`` output
    directory, so the on-disk CSVs are named ``Basin_Annual.csv`` and
    ``{basin}_Annual.csv`` without a pool-specific suffix.

    Args:
        basin_yearly (dict[int, dict[str, dict]]): {year: {basin: metrics}}.
        output_dir (str): Output directory for plots.
        start_year (int): First year on the x-axis.
        end_year (int): Last year on the x-axis.
        title_prefix (str): Prefix for figure titles (typically the
            pretty pool name, e.g. ``'Total SW '``).
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
        merged.to_csv(os.path.join(ts_dir, 'Basin_Annual.csv'), index=False)
    else:
        df.to_csv(os.path.join(ts_dir, 'Basin_Annual.csv'), index=False)

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
        _format_volume_axis(ax2, unit='m3', label='Volume')
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

        bdf.to_csv(os.path.join(ts_dir, f'{safe}_Annual.csv'), index=False)

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
    """Per-sub-basin annual time series (any pool) with era shading + uncertainty.

    Same sub-basin-level counterpart to :func:`create_basin_time_series`
    — used for every pool via ``_process_group``. The on-disk CSVs are
    named ``Subbasin_Annual.csv`` and ``{subbasin}_Annual.csv`` without
    a pool-specific suffix; the pool identity is implied by the
    enclosing ``{pool}/Subbasin_Time_Series/`` directory.

    Args:
        subbasin_yearly (dict[int, dict[str, dict]]): {year: {subbasin: metrics}}.
        output_dir (str): Output directory for plots.
        subbasin_shp (str): Path to ADWR sub-basin shapefile.
        ama_code_map (dict[str, str]): Mapping of AMA/INA codes to names.
        start_year (int): First year on the x-axis.
        end_year (int): Last year on the x-axis.
        title_prefix (str): Prefix for figure titles (typically the
            pretty pool name, e.g. ``'Total SW '``).
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
        merged.to_csv(os.path.join(ts_dir, 'Subbasin_Annual.csv'), index=False)
    else:
        df.to_csv(os.path.join(ts_dir, 'Subbasin_Annual.csv'), index=False)

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
        _format_volume_axis(ax2, unit='m3', label='Volume')
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

        sdf.to_csv(os.path.join(ts_dir, f'{safe}_Annual.csv'), index=False)

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
        **_kwargs,
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
        fig = plt.figure(figsize=(18, 10.5))
        gs = GridSpec(
            3, 2, figure=fig,
            width_ratios=[1, 1.2],
            height_ratios=[1.2, 1, 0.45],
            hspace=0.35, wspace=0.3,
        )
        ax_map = fig.add_subplot(gs[:2, 0])
        ax_ts = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[1, 1])
        ax_contrib = fig.add_subplot(gs[2, :])
    else:
        fig, ax_map = plt.subplots(1, 1, figsize=(9, 9))
        ax_contrib = None

    # ---- Panel A: Spatial map ----
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

    ama_ina = get_ama_ina_basin_names()
    name_col = 'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns else basins_gdf.columns[0]
    _overlay_boundaries(ax_map, basins_gdf, ama_ina, name_col,
                        label_fontsize=9, label_all=True)
    # SW-corner legend nudged down into the gap below the map, matching
    # the other era maps (no box).
    add_ama_ina_legend(ax_map, loc='lower left', bbox_to_anchor=(0.0, -0.12))

    cbar = fig.colorbar(im, ax=ax_map, shrink=0.45, pad=0.08, extend='both')
    cbar_fontsize = 10
    cbar.set_label('Depth (mm)', fontweight='bold', fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()
    cb_ax2 = cbar.ax.twinx()
    cb_lo, cb_hi = cbar.ax.get_ylim()
    cb_ax2.set_ylim(cb_lo * _MM_TO_FT, cb_hi * _MM_TO_FT)
    cb_ax2.set_ylabel('Depth (ft)', fontsize=cbar_fontsize, fontweight='bold')
    cb_ax2.tick_params(labelsize=cbar_fontsize)
    cb_ax2.yaxis.set_label_position('right')
    cb_ax2.yaxis.tick_right()

    ax_map.set_title(
        f'(a) Mean Annual Predicted Withdrawal ({start_year}\u2013{end_year})',
        fontweight='bold', fontsize=13,
    )

    # ---- Panel B: Time series (top-right) ----
    if has_ts:
        years = sorted(yearly_predictions.keys())
        vol_af = np.array([yearly_predictions[y]['Volume_AF'] for y in years])

        vol_std = np.zeros(len(years))
        std_label = None
        if sigma_yearly:
            for i, y in enumerate(years):
                if y in sigma_yearly:
                    val = sigma_yearly[y].get('Volume_AF', 0)
                    vol_std[i] = val if np.isfinite(val) else 0
            std_label = '\u00b11\u03c3 (UQ)'
        elif basin_yearly:
            for i, y in enumerate(years):
                if y in basin_yearly:
                    basin_vols = [v['Volume_AF'] for v in basin_yearly[y].values()
                                  if np.isfinite(v.get('Volume_AF', np.nan))]
                    if len(basin_vols) > 1:
                        vol_std[i] = np.std(basin_vols)
            std_label = '\u00b11\u03c3 (inter-basin)'

        for era, (s, e) in ERA_PERIODS.items():
            ax_ts.axvspan(s, e, color=ERA_COLORS[era], alpha=0.10)

        if vol_std.any():
            ax_ts.fill_between(
                years,
                np.maximum(vol_af - vol_std, 0),
                vol_af + vol_std,
                alpha=0.2, color=COLORS['predicted'], label=std_label,
            )

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
                           label=f'{e} ({ERA_PERIODS[e][0]}\u2013{ERA_PERIODS[e][1]})')
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

    # ---- Panel D: Key contributions text panel (bottom row) ----
    if has_ts and ax_contrib is not None:
        ax_contrib.set_xlim(0, 1)
        ax_contrib.set_ylim(0, 1)
        ax_contrib.axis('off')

        bg = mpatches.FancyBboxPatch(
            (0.005, 0.05), 0.99, 0.90,
            boxstyle='round,pad=0.01,rounding_size=0.015',
            facecolor='#F5F5F5', edgecolor='#888888', linewidth=0.6,
            transform=ax_contrib.transAxes,
            zorder=0,
        )
        ax_contrib.add_patch(bg)

        ax_contrib.text(
            0.015, 0.92, '(d) Key Contributions',
            transform=ax_contrib.transAxes,
            fontsize=12, fontweight='bold', va='top',
        )

        bullets = [
            '2 km \u00d7 204-yr coverage \u2014 first statewide annual '
            'withdrawals, irrigation CU, and SW capture index, 1896\u20132099, '
            'in one self-consistent framework.',

            'First Arizona-wide irrigation CU dataset at 2 km annual '
            'resolution \u2014 no public alternative at this resolution and '
            'time horizon.',

            'Out-of-distribution validation \u2014 trained only in 10 ADWR '
            'AMA/INAs, predicts statewide; matches ADWR, USGS, and WestWater '
            '2026 within model \u03c3 \u2014 no per-basin agency calibration.',

            'Statewide shares \u2014 GW/SW within \u00b12 pp of USGS, Irr/'
            'Non-Irr within \u00b12 pp of ADWR.',

            'Novel SW capture index \u2014 apportions GW pumping into '
            'stream-depletion vs. storage-mining at 2 km annual, work that '
            'normally requires per-basin MODFLOW\u2013SFR.',

            'Hybrid 6-component \u03c3_total UQ (\u03c3_MACA + \u03c3_Model + '
            '\u03c3_Irr + \u03c3_LULC + \u03c3_GW + \u03c3_USBR) with physics-'
            'based CU error propagation; 6-band augmented rasters per product.',
        ]

        bullet_top = 0.74
        bullet_bot = 0.05
        line_h = (bullet_top - bullet_bot) / max(len(bullets), 1)
        for i, text in enumerate(bullets):
            y = bullet_top - i * line_h
            ax_contrib.text(
                0.025, y, '\u2022',
                transform=ax_contrib.transAxes,
                fontsize=11, fontweight='bold', va='top',
            )
            ax_contrib.text(
                0.04, y, text,
                transform=ax_contrib.transAxes,
                fontsize=9.5, va='top',
            )

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


AMA_BORDER_COLOR = 'black'
INA_BORDER_COLOR = '#B71C1C'
BASIN_BORDER_COLOR = '#555555'

# Major surface-water corridor basins (Colorado River mainstem + lower
# Gila).  On the fully-labeled maps these get the same large font as the
# AMA/INA management areas (in a distinct blue) so the surface-water
# story is readable, rather than being relegated to the small minor-basin
# font.  Names match BASIN_NAME in the GW basin shapefile exactly.
MAJOR_SW_BASINS = frozenset({'YUMA', 'PARKER', 'LOWER GILA', 'GILA BEND'})
SW_BASIN_LABEL_COLOR = '#0b5394'


CAP_SERVICE_AREA_COLOR = '#1565C0'

CAP_COUNTY_COLORS: dict[str, str] = {
    'MARICOPA': '#1565C0',
    'PIMA': '#6A1B9A',
    'PINAL': '#00897B',
}

CAP_COUNTY_DISPLAY: dict[str, str] = {
    'MARICOPA': 'Maricopa',
    'PIMA': 'Pima',
    'PINAL': 'Pinal',
}


def add_ama_ina_legend(
    target,
    *,
    loc: str = 'lower left',
    bbox_to_anchor: tuple[float, float] | None = None,
    fontsize: int = 10,
    framealpha: float = 0.9,
    frameon: bool = False,
    include_cap: bool = False,
    ncol: int | None = None,
) -> None:
    """Add a single AMA / INA / GW basin legend to a Figure or Axes.

    Default placement is **outside the right edge of the target Axes**
    (anchor ``(1.02, 1.0)`` with ``loc='upper left'``) — pass the
    first map's Axes (e.g. ``axes[0]``) for multi-panel figures so the
    legend appears once next to the leftmost panel.

    Use this once per figure with multi-panel boundary overlays —
    pass ``show_legend=False`` to ``_overlay_boundaries`` on each
    subplot and call this helper at the figure level.

    Args:
        target: A matplotlib ``Figure`` or ``Axes``.  When passing an
            Axes, the default anchor places the legend just outside
            the right edge of that axes.
        loc: Matplotlib legend ``loc``.
        bbox_to_anchor: Anchor in target coordinates.  Pass ``None``
            to skip anchoring.
        fontsize: Legend font size.
        framealpha: Legend frame transparency.
        include_cap: If True, also include per-county CAP service-area
            entries (one line per CAP county, color-keyed to the
            ``overlay_cap_service_area`` per-county outlines).
        ncol: Number of legend columns.  When None, defaults to 2 if
            *include_cap* (so the 6-entry legend renders as a 3-row ×
            2-col grid) and 1 otherwise.
    """
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=BASIN_BORDER_COLOR, lw=0.8,
               label='GW basin'),
        Line2D([0], [0], color=AMA_BORDER_COLOR, lw=1.4, label='AMA'),
        Line2D([0], [0], color=INA_BORDER_COLOR, lw=1.4, label='INA'),
    ]
    if include_cap:
        for county_key, display in CAP_COUNTY_DISPLAY.items():
            handles.append(
                Line2D([0], [0],
                       color=CAP_COUNTY_COLORS[county_key], lw=1.6,
                       label=f'CAP — {display}'),
            )
    if ncol is None:
        ncol = 2 if include_cap else 1
    kwargs = dict(handles=handles, loc=loc, fontsize=fontsize,
                  framealpha=framealpha, frameon=frameon, ncol=ncol)
    if bbox_to_anchor is not None:
        kwargs['bbox_to_anchor'] = bbox_to_anchor
    target.legend(**kwargs)


def overlay_cap_service_area(
    ax,
    cap_service_area_geojson: 'str | None',
    target_crs=None,
    *,
    linewidth: float = 1.2,
    linestyle: str = '-',
    alpha: float = 0.95,
    name_col: str = 'NAME',
) -> bool:
    """Overlay the CAP service-area boundary onto a map axis.

    Each CAP county polygon is drawn in its own color (per
    :data:`CAP_COUNTY_COLORS`) so the three service-area sub-units
    (Maricopa / Pima / Pinal) are individually identifiable.  Returns
    True if the overlay was drawn so callers can decide whether to
    add the CAP legend entries.
    """
    if not cap_service_area_geojson or not os.path.isfile(
        cap_service_area_geojson,
    ):
        return False
    try:
        cap_gdf = gpd.read_file(cap_service_area_geojson)
    except Exception:  # noqa: BLE001
        return False
    if cap_gdf.empty:
        return False
    if target_crs is not None and cap_gdf.crs != target_crs:
        cap_gdf = cap_gdf.to_crs(target_crs)
    if name_col not in cap_gdf.columns:
        cap_gdf.boundary.plot(
            ax=ax, color=CAP_SERVICE_AREA_COLOR,
            linewidth=linewidth, linestyle=linestyle, alpha=alpha,
        )
        return True
    for _, row in cap_gdf.iterrows():
        county = str(row[name_col]).upper().strip()
        color = CAP_COUNTY_COLORS.get(county, CAP_SERVICE_AREA_COLOR)
        gpd.GeoSeries([row.geometry], crs=cap_gdf.crs).boundary.plot(
            ax=ax, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha,
        )
    return True


def _overlay_boundaries(
    ax,
    basins_gdf: gpd.GeoDataFrame,
    ama_ina_names: list[str],
    name_col: str,
    *,
    label_fontsize: float = 9.0,
    label_all: bool = False,
    show_legend: bool = False,
    show_labels: bool = True,
) -> None:
    """Draw basin boundaries and label basins on a map axis.

    AMAs and INAs are drawn with distinct colors (AMA = black,
    INA = dark red) and a small legend in the upper-right corner
    distinguishes the three boundary classes (Basin / AMA / INA).
    The ``ama_ina_names`` argument is kept for backward compatibility
    but the actual AMA/INA split is sourced from
    :func:`get_ama_ina_classification` so the colors stay correct
    even if callers pass the combined list.

    Args:
        label_all: If True, label all basins (small font for non-AMA/INA).
            If False (default), label only AMA/INA basins.
        show_legend: If True, attach the AMA / INA / GW-basin legend
            to **this axes** (outside-right placement).  Defaults to
            False — for multi-panel figures, leave this False on every
            subplot and call :func:`add_ama_ina_legend` once at the
            figure level (or on the first axes) so the legend appears
            exactly once per figure.
    """
    ama_basins, ina_basins = get_ama_ina_classification()

    basins_gdf.boundary.plot(
        ax=ax, color=BASIN_BORDER_COLOR, linewidth=0.4,
    )
    ama_gdf = basins_gdf[basins_gdf[name_col].isin(ama_basins)]
    ina_gdf = basins_gdf[basins_gdf[name_col].isin(ina_basins)]
    if not ama_gdf.empty:
        ama_gdf.boundary.plot(
            ax=ax, color=AMA_BORDER_COLOR, linewidth=1.2,
        )
    if not ina_gdf.empty:
        ina_gdf.boundary.plot(
            ax=ax, color=INA_BORDER_COLOR, linewidth=1.2,
        )
    if show_labels:
        # Major labels: the AMA/INA management areas always, plus the
        # major surface-water corridor basins when the map is fully
        # labeled (label_all).  All share the large font and are
        # decluttered together so none hides another.
        major_frames = [ama_gdf, ina_gdf]
        if label_all:
            sw_gdf = basins_gdf[basins_gdf[name_col].isin(MAJOR_SW_BASINS)]
            if not sw_gdf.empty:
                major_frames.append(sw_gdf)
        label_rows = pd.concat(major_frames, ignore_index=True)
        # Several of these areas are east–west neighbours at nearly the
        # same latitude (Phoenix/Harquahala, Tucson/Willcox, Santa
        # Cruz/Douglas), so their horizontal labels overprint each other
        # at page-width panel sizes.  Nudge horizontally-adjacent labels
        # onto separate vertical rows with a leader line back to the
        # basin centroid, so no name is hidden behind another.
        minx, miny, maxx, maxy = basins_gdf.total_bounds
        span_x, span_y = (maxx - minx), (maxy - miny)
        x_thr = 0.22 * span_x     # horizontal proximity that risks overlap
        # Wide labels overprint at a larger horizontal separation than the
        # vertical-clustering threshold catches — e.g. Santa Cruz / Douglas
        # sit 0.23·span_x apart yet "SANTA CRUZ" still runs into "DOUGLAS".
        # The horizontal push-apart therefore uses its own wider threshold.
        x_thr_h = 0.28 * span_x
        y_thr = 0.055 * span_y    # vertical gap that counts as the same row
        step = 0.055 * span_y     # per-bump vertical offset
        # Allow several steps so dense clusters (Parker / Harquahala /
        # Ranegras / Phoenix in the central-west) can spread out, but an
        # order-preservation rule forbids a label from crossing a
        # neighbour's row — so an edge basin like Douglas (far SE corner,
        # blocked from moving south) can never climb past Willcox and
        # invert the geography.  A label that still can't clear stays on
        # its centroid and accepts a minor overlap.
        max_k = 4
        items = []
        for _, row in label_rows.iterrows():
            c = row.geometry.centroid
            nm = row[name_col]
            if nm in ama_basins:
                color = AMA_BORDER_COLOR
            elif nm in ina_basins:
                color = INA_BORDER_COLOR
            else:
                color = SW_BASIN_LABEL_COLOR
            items.append({
                'x': c.x, 'y0': c.y, 'y': c.y,
                'text': nm.replace(' AMA', '').replace(' INA', ''),
                'color': color,
            })
        placed: list[dict] = []
        for it in sorted(items, key=lambda d: d['x']):
            chosen = it['y0']
            for k in range(max_k + 1):
                found = False
                for s in ([0] if k == 0 else [1, -1]):
                    y = it['y0'] + s * k * step
                    if not (miny <= y <= maxy):
                        continue
                    # No collision with an already-placed nearby label.
                    if not all(
                        abs(it['x'] - p['x']) >= x_thr
                        or abs(y - p['y']) >= y_thr
                        for p in placed
                    ):
                        continue
                    # Preserve vertical order: never cross a nearby placed
                    # label relative to the original centroid order — EXCEPT
                    # for near-same-latitude neighbours (|Δy0| < y_thr), where
                    # there is no real geographic order to preserve.  Without
                    # this exemption a label squeezed between two same-row
                    # neighbours (e.g. Harquahala between Parker and Phoenix)
                    # can never reach a clear row and overprints its neighbour.
                    if not all(
                        abs(it['x'] - p['x']) >= x_thr
                        or (it['y0'] < p['y0']) == (y < p['y'])
                        or abs(it['y0'] - p['y0']) < y_thr
                        for p in placed
                    ):
                        continue
                    chosen, found = y, True
                    break
                if found:
                    break
            it['y'] = chosen
            it['ha'] = 'center'
            placed.append(it)
        # Labels that still share a row after the vertical declutter
        # (e.g. Harquahala vs Phoenix in the dense central-west) get their
        # text pushed apart horizontally: the left basin anchors its text
        # to the right (growing left, away from its neighbour) and the
        # right basin anchors left, so long names stop overprinting.
        for _i in range(len(placed)):
            for _j in range(_i + 1, len(placed)):
                a, b = placed[_i], placed[_j]
                if (abs(a['x'] - b['x']) < x_thr_h
                        and abs(a['y'] - b['y']) < y_thr):
                    left, right = (a, b) if a['x'] < b['x'] else (b, a)
                    left['ha'] = 'right'
                    right['ha'] = 'left'
        for it in items:
            ax.annotate(
                it['text'], (it['x'], it['y']),
                fontsize=label_fontsize, fontweight='bold',
                ha=it.get('ha', 'center'), va='center',
                color=it['color'],
                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                          alpha=0.8, lw=0),
                zorder=5,
            )
    if show_labels and label_all:
        # Remaining basins (not AMA/INA and not a major SW corridor basin)
        # are labeled in a distinctly smaller, lighter font so the major
        # labels stay visually dominant.  These minor labels are
        # intentionally small — a reference aid when the figure is zoomed,
        # not meant to be legible at 100 %.
        minor_fontsize = max(label_fontsize * 0.55, 4.0)
        major_names = set(ama_ina_names) | set(MAJOR_SW_BASINS)
        other_gdf = basins_gdf[~basins_gdf[name_col].isin(major_names)]
        for _, row in other_gdf.iterrows():
            centroid = row.geometry.centroid
            short = row[name_col].title()
            ax.annotate(
                short, (centroid.x, centroid.y),
                fontsize=minor_fontsize, fontstyle='italic',
                ha='center', va='center', color='#555555',
                bbox=dict(boxstyle='round,pad=0.08', fc='white',
                          alpha=0.55, lw=0),
                zorder=3,
            )
    if show_legend:
        add_ama_ina_legend(ax)
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
    era_present: dict[str, np.ndarray] = {}  # pixel was finite in any year

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
        # Per-year per-year-raster contribution.  ``finite`` (numerator)
        # only adds non-NaN, positive values.  ``era_counts``
        # (denominator) increments by 1 every era year regardless of
        # the pixel's value, so pixels active in only a handful of
        # peak years (e.g. crop_edge_halo at 1951-57 / 1970-80) are
        # diluted by the surrounding inactive years and don't show as
        # the same brightness as continuously-pumping core pixels.
        # ``era_present`` tracks pixels that had at least one finite
        # observation in the era so we can mask the truly-empty pixels
        # (count > 0 always now, so we need this separate mask).
        finite_any = np.isfinite(arr)
        finite = finite_any & (arr > 0)
        if era not in era_sums:
            era_sums[era] = np.zeros(raster_shape, dtype=np.float64)
            era_counts[era] = np.zeros(raster_shape, dtype=np.float64)
            era_present[era] = np.zeros(raster_shape, dtype=bool)
        era_sums[era][finite] += arr[finite]
        era_counts[era] += 1
        # Pixel counted as "present" if it had any finite observation
        # (including a finite zero, which is meaningful for ratio bands
        # like CV where 0 is valid).
        era_present[era] |= finite_any

    era_means = {}
    for era in ERA_PERIODS:
        if era in era_sums and era_counts[era].max() > 0:
            with np.errstate(invalid='ignore'):
                mean_arr = np.where(
                    era_counts[era] > 0,
                    era_sums[era] / era_counts[era],
                    0.0,
                )
            # Mask out pixels that were never observed (NaN in every
            # era year).  Pixels observed in at least one year keep
            # their (possibly small) frequency-weighted mean.
            if mask_nan_only:
                era_means[era] = np.ma.masked_where(
                    ~era_present[era], mean_arr,
                )
            else:
                era_means[era] = np.ma.masked_where(
                    (mean_arr == 0) | ~era_present[era], mean_arr,
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
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    gamma: float | None = None,
    cbar_extend: str = 'both',
    dual_depth_volume: bool = False,
) -> None:
    """Create a 2×2 panel figure of era-mean raster maps.

    When ``dual_depth_volume`` is True and the base raster is a depth
    (mm) raster, the colorbar carries BOTH the depth scale (mm, bottom)
    and the equivalent per-pixel volume scale (×10⁶ m³, top) — depth and
    volume are the same field on a constant-area grid, so this shows one
    map in both units instead of two identical figures.

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
        vmin (float or None): Explicit colorbar minimum.  When None,
            derived from ``percentile_clip[0]`` over the pooled era
            values.
        vmax (float or None): Explicit colorbar maximum.  When None,
            derived from ``percentile_clip[1]`` over the pooled era
            values.
        symmetric (bool): If True, center colorbar on zero.
        band (int): Band number to read from each raster (1-based).
        mask_nan_only (bool): If True, only mask NaN pixels (keep zeros
            visible).
        percentile_clip (tuple[float, float]): (low, high) percentiles
            used to auto-derive vmin/vmax when they are not supplied
            explicitly.  Default is (2, 98).  For heavily right-skewed
            ratio bands (CV, SNR) pass a tighter tuple like (2, 95) or
            (5, 90) so that a long upper tail does not compress the
            bulk of the distribution into the bottom few percent of
            the colorbar.
        gamma (float or None): When set, uses
            ``matplotlib.colors.PowerNorm(gamma, vmin, vmax)`` instead
            of a linear normalization.  ``gamma < 1`` gives more
            visual range to small values (useful for CV maps where
            most pixels cluster near zero); ``gamma > 1`` does the
            opposite.  Ignored when ``symmetric=True`` (diverging
            colormaps use linear norm).
        cbar_extend (str): Colorbar ``extend`` direction — one of
            ``'neither'``, ``'min'``, ``'max'``, ``'both'``.  Default
            ``'both'`` is right for most products.  Use ``'max'`` for
            non-negative bounded-below quantities (CV, SNR, capture
            fraction) so the colorbar terminates cleanly at zero
            instead of rendering a spurious lower triangle.

    Returns:
        None.
    """
    import rasterio as rio
    from matplotlib.colors import PowerNorm

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
        pixel_area_m2 = abs(src.transform.a * src.transform.e)

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
        valid_arrays = [
            em.compressed() for em in era_means.values()
            if em is not None and em.count() > 0
        ]
        if not valid_arrays:
            logger.warning('All era means are empty for %s — skipping.', title)
            return
        all_vals = np.concatenate(valid_arrays)
        lo_pct, hi_pct = percentile_clip
        if vmin is None:
            vmin = float(np.nanpercentile(all_vals, lo_pct))
        if vmax is None:
            vmax = float(np.nanpercentile(all_vals, hi_pct))
    if symmetric:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # Power-law normalization for heavily skewed ratio bands (CV, SNR)
    # gives more visual range to the bulk of the distribution while
    # still allowing the tail to saturate at vmax.  Only applied when
    # vmin >= 0 (PowerNorm requires non-negative range) and the map
    # is not symmetric.
    norm = None
    if gamma is not None and not symmetric and vmin >= 0 and vmax > vmin:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # ---- Create figure ----
    # Single-column vertical stack: one era per row so every map spans
    # the full page width (~6.5 in) instead of being squeezed into a
    # ~2 in third of a horizontal strip.  At full width the figure is
    # authored at roughly its display size, so the page-fit shrink is
    # ~1x and the font literals below are ordinary publication point
    # sizes (no shrink-compensation inflation).  The trade-off is a
    # tall figure (~one full page per ~3 eras).
    n_eras = len(ERA_PERIODS)
    fig, axes = plt.subplots(
        n_eras, 1, figsize=(5.2, 4.3 * n_eras), constrained_layout=True,
    )
    fig.suptitle(f'{title} — Era Mean', fontsize=16, fontweight='bold')
    if n_eras == 1:
        axes = [axes]

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    axes_flat = list(axes) if isinstance(axes, np.ndarray) else axes
    # When a PowerNorm is in play, pass it in lieu of vmin/vmax so
    # imshow uses the non-linear mapping consistently across all panels.
    imshow_kwargs = (
        {'cmap': cmap, 'norm': norm}
        if norm is not None
        else {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    )
    for idx, (era, (y1, y2)) in enumerate(ERA_PERIODS.items()):
        ax = axes_flat[idx]
        ax.set_facecolor('#D5D5D5')  # gray for no-data

        era_arr = era_means.get(era)
        if era_arr is not None and era_arr.count() > 0:
            im = ax.imshow(
                era_arr, extent=extent, origin='upper',
                interpolation='nearest', **imshow_kwargs,
            )
        else:
            # Blank panel
            blank = np.ma.masked_all(raster_shape)
            im = ax.imshow(
                blank, extent=extent, origin='upper',
                interpolation='nearest', **imshow_kwargs,
            )

        # Label only the AMA/INA management areas (the basins the paper
        # discusses by name).  Labeling all 52 basins overlaps into an
        # unreadable mass once the font is large enough to read on a
        # page-width 3-panel strip.
        _overlay_boundaries(ax, basins_gdf, ama_ina, name_col,
                            label_all=True, label_fontsize=8)
        ax.set_title(
            f'{panel_labels[idx]} {era} ({y1}–{y2})',
            fontsize=14, fontweight='bold',
        )

    # GW basin / AMA / INA legend as a horizontal row centred just below
    # the bottom (last-era) panel, between the map and the colorbar.  The
    # tight per-panel height leaves no SW-corner slack, so anchoring the
    # legend by its top (loc='upper center') makes it grow DOWN into the
    # gap above the colorbar instead of up into the map.
    add_ama_ina_legend(
        axes_flat[-1], loc='upper center', bbox_to_anchor=(0.5, -0.03),
        fontsize=10, ncol=3,
    )

    # Shared colorbar with dual units
    era_cbar_fontsize = 12
    cbar = fig.colorbar(
        im, ax=axes_flat, shrink=0.92, pad=0.02,
        orientation='horizontal', aspect=28, extend=cbar_extend,
    )

    # Detect volume maps and inline the 10⁶ scale factor into the
    # primary label + tick formatter.  Volume values in m³ run
    # into the 10⁶–10⁸ range for typical per-pixel annual
    # withdrawals, so matplotlib's default offset text renders as
    # "1e8" at the end of the colorbar axis — small, easy to miss,
    # and inconsistent with the rest of the framework which
    # reports volumes in ×10⁶ m³.  We bake the scale factor into
    # the label ("Volume (×10⁶ m³)") and divide tick values by
    # 1e6 via a FuncFormatter so the offset text goes away and
    # tick labels read as clean 1–100 range numbers.
    is_volume_m3 = 'm$^3$' in unit_label or 'm3' in unit_label.lower()
    if is_volume_m3:
        import matplotlib.ticker as mticker
        volume_label = r'Volume ($\times$10$^{6}$ m$^3$)'
        cbar.set_label(
            volume_label, fontsize=era_cbar_fontsize, fontweight='bold',
        )
        cbar.formatter = mticker.FuncFormatter(
            lambda x, _: f'{x / 1e6:g}',
        )
        cbar.update_ticks()
    else:
        cbar.set_label(
            unit_label, fontsize=era_cbar_fontsize, fontweight='bold',
        )
    cbar.ax.tick_params(labelsize=era_cbar_fontsize)

    # Add a secondary unit axis on top of the colorbar.
    #
    # We use ``cbar.ax.secondary_xaxis`` rather than ``cbar.ax.twiny()``
    # because ``twiny`` creates a second Axes that copies only the
    # *data* xlim from the colorbar and draws its own rectangular
    # spines on top of the colorbar axes bbox. When the colorbar has
    # ``extend='max'`` or ``extend='both'``, the arrowhead triangle
    # lives *outside* that data xlim, so the rectangular twin axis
    # ends up occluding the triangle and the colorbar renders with a
    # flat end. ``secondary_xaxis`` is explicitly designed to cooperate
    # with colorbar extensions: it shares the parent axis transform,
    # preserves the arrowhead region, and applies the unit conversion
    # through a paired (forward, inverse) function tuple.
    if 'mm' in unit_label.lower() and dual_depth_volume:
        # Same map, four scales: depth (mm, primary) + volume (×10⁶ m³)
        # on top, with the imperial equivalents — depth (ft) and volume
        # (AF) — stacked below the metric axes.  volume_m3 = depth_mm ×
        # pixel_area / 1000.
        _vol_factor = pixel_area_m2 / 1.0e9            # mm -> ×10⁶ m³
        _af_factor = pixel_area_m2 / 1000.0 / _AF_TO_M3  # mm -> AF
        # Top: volume (×10⁶ m³)
        sec_vol = cbar.ax.secondary_xaxis(
            'top',
            functions=(lambda mm: mm * _vol_factor,
                       lambda v: v / _vol_factor),
        )
        sec_vol.set_xlabel(
            r'Volume ($\times$10$^{6}$ m$^3$)',
            fontsize=era_cbar_fontsize, fontweight='bold',
        )
        sec_vol.tick_params(labelsize=era_cbar_fontsize)
        # Below the metric depth axis: depth (ft)
        sec_ft = cbar.ax.secondary_xaxis(
            -5.5,
            functions=(lambda mm: mm * _MM_TO_FT,
                       lambda ft: ft / _MM_TO_FT),
        )
        sec_ft.set_xlabel(
            'Depth (ft)', fontsize=era_cbar_fontsize, fontweight='bold',
            labelpad=2,
        )
        sec_ft.tick_params(labelsize=era_cbar_fontsize)
        # Below that: volume (AF)
        sec_af = cbar.ax.secondary_xaxis(
            -11.0,
            functions=(lambda mm: mm * _af_factor,
                       lambda af: af / _af_factor),
        )
        sec_af.set_xlabel(
            'Volume (AF)', fontsize=era_cbar_fontsize, fontweight='bold',
            labelpad=2,
        )
        sec_af.tick_params(labelsize=era_cbar_fontsize)
    elif 'mm' in unit_label.lower():
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(
                lambda mm: mm * _MM_TO_FT,
                lambda ft: ft / _MM_TO_FT,
            ),
        )
        secax.set_xlabel(
            'Depth (ft)', fontsize=era_cbar_fontsize, fontweight='bold',
        )
        secax.tick_params(labelsize=era_cbar_fontsize)
    elif is_volume_m3:
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(
                lambda m3: m3 / _AF_TO_M3,
                lambda af: af * _AF_TO_M3,
            ),
        )
        secax.set_xlabel(
            'Volume (AF)', fontsize=era_cbar_fontsize, fontweight='bold',
        )
        secax.tick_params(labelsize=era_cbar_fontsize)

    if out_filename is None:
        slug = title.replace(' ', '_').replace('/', '_')
        out_filename = f'Era_Maps_{slug}.png'
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Era raster maps saved to {out_path}')


def create_gw_sw_era_raster_maps(
    gw_raster_dir: str,
    sw_raster_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str,
    unit_label: str,
    col_labels: tuple[str, str] = ('Groundwater', 'Surface Water'),
    out_tag: str = 'GW_SW',
    cmap: str = 'Spectral_r',
    band: int = 1,
    mask_nan_only: bool = False,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_extend: str = 'both',
    out_filename: str | None = None,
) -> None:
    """Paired two-category era-mean maps in one figure.

    Columns are the two paired categories — labeled **(a)
    {col_labels[0]}** and **(b) {col_labels[1]}** — and rows are the
    eras (Hindcast / Historical / Projection), labeled down the left
    edge.  A single shared colorbar spans both columns (color limits
    pooled over both categories) so the two are directly comparable on
    one scale.  Used both for the groundwater/surface-water source split
    (default ``col_labels``) and the irrigation/non-irrigation use-type
    split.

    Args mirror :func:`create_era_raster_maps`, but take a separate
    ``gw_raster_dir`` (left/column-a) and ``sw_raster_dir``
    (right/column-b), each a directory of per-year ``*.tif`` rasters.
    Output defaults to ``Era_Maps_{title_slug}_{out_tag}.png``.
    """
    import rasterio as rio

    apply_journal_style()
    makedirs(output_dir)

    def _means_and_meta(rdir):
        tifs = sorted(f for f in os.listdir(rdir) if f.endswith('.tif'))
        if not tifs:
            return None, None
        with rio.open(os.path.join(rdir, tifs[0])) as src:
            meta = (
                [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top],
                src.crs, src.shape,
            )
        means = _compute_era_means(
            rdir, meta[2], band=band, mask_nan_only=mask_nan_only,
        )
        return means, meta

    gw_means, gw_meta = _means_and_meta(gw_raster_dir)
    sw_means, _ = _means_and_meta(sw_raster_dir)
    if gw_means is None or sw_means is None:
        logger.warning(
            'Missing GW or SW rasters for %s — skipping paired era maps.',
            title,
        )
        return
    extent, crs, raster_shape = gw_meta

    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)
    name_col = (
        'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    # Shared color limits pooled over BOTH pools so GW and SW share one
    # comparable scale under the single colorbar.
    valid = [
        em.compressed()
        for em in list(gw_means.values()) + list(sw_means.values())
        if em is not None and em.count() > 0
    ]
    if not valid:
        logger.warning('All era means empty for %s — skipping.', title)
        return
    all_vals = np.concatenate(valid)
    lo_pct, hi_pct = percentile_clip
    if vmin is None:
        vmin = float(np.nanpercentile(all_vals, lo_pct))
    if vmax is None:
        vmax = float(np.nanpercentile(all_vals, hi_pct))

    # Wider per-column canvas (~4.7 in/map) so the dense central-Arizona
    # labels (Phoenix vs Harquahala, etc.) have room and don't overprint.
    n_eras = len(ERA_PERIODS)
    fig, axes = plt.subplots(
        n_eras, 2, figsize=(9.6, 5.8 * n_eras), constrained_layout=True,
    )
    fig.suptitle(f'{title} — Era Mean', fontsize=16, fontweight='bold')
    axes = np.atleast_2d(axes)

    pools = [(col_labels[0], gw_means), (col_labels[1], sw_means)]
    col_tags = ['(a)', '(b)']
    im = None
    for ri, (era, (y1, y2)) in enumerate(ERA_PERIODS.items()):
        for ci, (pool_name, means) in enumerate(pools):
            ax = axes[ri, ci]
            ax.set_facecolor('#D5D5D5')
            arr = means.get(era)
            if arr is None or arr.count() == 0:
                arr = np.ma.masked_all(raster_shape)
            im = ax.imshow(
                arr, extent=extent, origin='upper',
                interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax,
            )
            _overlay_boundaries(ax, basins_gdf, ama_ina, name_col,
                                label_all=True, label_fontsize=8)
            # Column headers (pool name) only on the top row.
            if ri == 0:
                ax.set_title(
                    f'{col_tags[ci]} {pool_name}',
                    fontsize=14, fontweight='bold',
                )
        # Era label down the left edge of the row (no a/b/c prefix —
        # the a/b now denote the GW/SW columns).
        axes[ri, 0].text(
            -0.06, 0.5, f'{era}\n({y1}–{y2})',
            transform=axes[ri, 0].transAxes, rotation=90,
            va='center', ha='center', fontsize=12, fontweight='bold',
        )

    # GW basin / AMA / INA legend at the south-west corner of the
    # bottom-left (Groundwater / last-era) panel, nudged down slightly so
    # it sits in the gap between the bottom map and the colorbar —
    # Arizona's SW corner is the cut-off no-data triangle, so it stays
    # clear of the data.
    add_ama_ina_legend(
        axes[-1, 0], loc='lower left', bbox_to_anchor=(0.0, -0.12),
        fontsize=10, ncol=1, frameon=False,
    )

    # Single shared full-width colorbar with dual units across both
    # columns, nudged up close to the maps.
    era_cbar_fontsize = 12
    cbar = fig.colorbar(
        im, ax=axes, shrink=0.92, pad=0.02,
        orientation='horizontal', aspect=45, extend=cbar_extend,
    )
    is_volume_m3 = 'm$^3$' in unit_label or 'm3' in unit_label.lower()
    if is_volume_m3:
        import matplotlib.ticker as mticker
        cbar.set_label(
            r'Volume ($\times$10$^{6}$ m$^3$)',
            fontsize=era_cbar_fontsize, fontweight='bold',
        )
        cbar.formatter = mticker.FuncFormatter(lambda x, _: f'{x / 1e6:g}')
        cbar.update_ticks()
    else:
        cbar.set_label(
            unit_label, fontsize=era_cbar_fontsize, fontweight='bold',
        )
    cbar.ax.tick_params(labelsize=era_cbar_fontsize)
    if 'mm' in unit_label.lower():
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(lambda mm: mm * _MM_TO_FT, lambda ft: ft / _MM_TO_FT),
        )
        secax.set_xlabel(
            'Depth (ft)', fontsize=era_cbar_fontsize, fontweight='bold',
        )
        secax.tick_params(labelsize=era_cbar_fontsize)
    elif is_volume_m3:
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(lambda m3: m3 / _AF_TO_M3, lambda af: af * _AF_TO_M3),
        )
        secax.set_xlabel(
            'Volume (AF)', fontsize=era_cbar_fontsize, fontweight='bold',
        )
        secax.tick_params(labelsize=era_cbar_fontsize)

    if out_filename is None:
        slug = title.replace(' ', '_').replace('/', '_')
        out_filename = f'Era_Maps_{slug}_{out_tag}.png'
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info('Paired era maps saved to %s', out_path)


def create_dual_metric_era_raster_maps(
    basin_shp: str,
    output_dir: str,
    *,
    title: str,
    left: dict,
    right: dict,
    out_tag: str = 'dual',
    cbar_extend: str = 'both',
    out_filename: str | None = None,
) -> None:
    """Paired era-mean maps of two DIFFERENT metrics, one colorbar each.

    Unlike :func:`create_gw_sw_era_raster_maps` (single shared colorbar),
    the two columns carry different quantities on different scales
    (e.g. capture *fraction* vs capture *volume*), so each column gets
    its own colorbar with the appropriate unit treatment.  Columns are
    (a) ``left`` / (b) ``right``; rows are the eras.

    ``left`` and ``right`` are dicts with keys: ``raster_dir`` (required),
    ``label`` (column header), ``unit`` (colorbar label), ``cmap``
    (default ``'YlOrRd'``), ``band`` (1), ``vmin``/``vmax`` (None →
    percentile), ``percentile_clip`` ((2, 98)), ``mask_nan_only`` (False).
    """
    import rasterio as rio
    import matplotlib.ticker as mticker

    apply_journal_style()
    makedirs(output_dir)

    def _means_meta(rdir, band, mask_nan_only):
        tifs = sorted(f for f in os.listdir(rdir) if f.endswith('.tif'))
        if not tifs:
            return None, None
        with rio.open(os.path.join(rdir, tifs[0])) as src:
            meta = (
                [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top],
                src.crs, src.shape,
            )
        return _compute_era_means(
            rdir, meta[2], band=band, mask_nan_only=mask_nan_only,
        ), meta

    cols = []
    for spec in (left, right):
        means, meta = _means_meta(
            spec['raster_dir'], spec.get('band', 1),
            spec.get('mask_nan_only', False),
        )
        if means is None:
            logger.warning(
                'Missing rasters for %s — skipping dual-metric maps.', title,
            )
            return
        valid = [
            em.compressed() for em in means.values()
            if em is not None and em.count() > 0
        ]
        if not valid:
            logger.warning('All era means empty for %s — skipping.', title)
            return
        allv = np.concatenate(valid)
        lo, hi = spec.get('percentile_clip', (2.0, 98.0))
        vmin = spec.get('vmin')
        vmax = spec.get('vmax')
        if vmin is None:
            vmin = float(np.nanpercentile(allv, lo))
        if vmax is None:
            vmax = float(np.nanpercentile(allv, hi))
        cols.append({
            'means': means, 'meta': meta, 'vmin': vmin, 'vmax': vmax,
            'cmap': spec.get('cmap', 'YlOrRd'),
            'label': spec['label'], 'unit': spec['unit'],
        })

    extent, crs, raster_shape = cols[0]['meta']
    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)
    name_col = (
        'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    n_eras = len(ERA_PERIODS)
    fig, axes = plt.subplots(
        n_eras, 2, figsize=(9.6, 5.8 * n_eras), constrained_layout=True,
    )
    fig.suptitle(f'{title} — Era Mean', fontsize=16, fontweight='bold')
    axes = np.atleast_2d(axes)
    col_tags = ['(a)', '(b)']
    ims = [None, None]
    for ri, (era, (y1, y2)) in enumerate(ERA_PERIODS.items()):
        for ci, col in enumerate(cols):
            ax = axes[ri, ci]
            ax.set_facecolor('#D5D5D5')
            arr = col['means'].get(era)
            if arr is None or arr.count() == 0:
                arr = np.ma.masked_all(raster_shape)
            ims[ci] = ax.imshow(
                arr, extent=extent, origin='upper',
                interpolation='nearest', cmap=col['cmap'],
                vmin=col['vmin'], vmax=col['vmax'],
            )
            _overlay_boundaries(ax, basins_gdf, ama_ina, name_col,
                                label_all=True, label_fontsize=8)
            if ri == 0:
                ax.set_title(
                    f"{col_tags[ci]} {col['label']}",
                    fontsize=14, fontweight='bold',
                )
        axes[ri, 0].text(
            -0.06, 0.5, f'{era}\n({y1}–{y2})',
            transform=axes[ri, 0].transAxes, rotation=90,
            va='center', ha='center', fontsize=12, fontweight='bold',
        )

    add_ama_ina_legend(
        axes[-1, 0], loc='lower left', bbox_to_anchor=(0.0, -0.12),
        fontsize=10, ncol=1, frameon=False,
    )

    def _apply_units(cbar, unit_label):
        is_vol = 'm$^3$' in unit_label or 'm3' in unit_label.lower()
        if is_vol:
            cbar.set_label(
                r'Volume ($\times$10$^{6}$ m$^3$)',
                fontsize=11, fontweight='bold',
            )
            cbar.formatter = mticker.FuncFormatter(lambda x, _: f'{x / 1e6:g}')
            cbar.update_ticks()
        else:
            cbar.set_label(unit_label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)
        if 'mm' in unit_label.lower():
            sx = cbar.ax.secondary_xaxis(
                'top',
                functions=(lambda mm: mm * _MM_TO_FT,
                           lambda ft: ft / _MM_TO_FT),
            )
            sx.set_xlabel('Depth (ft)', fontsize=11, fontweight='bold')
            sx.tick_params(labelsize=11)
        elif is_vol:
            sx = cbar.ax.secondary_xaxis(
                'top',
                functions=(lambda m3: m3 / _AF_TO_M3,
                           lambda af: af * _AF_TO_M3),
            )
            sx.set_xlabel('Volume (AF)', fontsize=11, fontweight='bold')
            sx.tick_params(labelsize=11)
        elif 'fraction' in unit_label.lower():
            # Percent twin on top: a useful second reading of the capture
            # fraction AND it gives this colorbar the same height as the
            # dual-unit volume colorbar, so the two colorbar bars stay
            # vertically aligned under constrained_layout.
            sx = cbar.ax.secondary_xaxis(
                'top',
                functions=(lambda f: f * 100.0, lambda p: p / 100.0),
            )
            sx.set_xlabel('Capture (%)', fontsize=11, fontweight='bold')
            sx.tick_params(labelsize=11)

    # One colorbar per column, placed under its own column.
    cbar_a = fig.colorbar(
        ims[0], ax=list(axes[:, 0]), orientation='horizontal',
        shrink=0.92, pad=0.02, aspect=30, extend=cbar_extend,
    )
    _apply_units(cbar_a, cols[0]['unit'])
    cbar_b = fig.colorbar(
        ims[1], ax=list(axes[:, 1]), orientation='horizontal',
        shrink=0.92, pad=0.02, aspect=30, extend=cbar_extend,
    )
    _apply_units(cbar_b, cols[1]['unit'])

    if out_filename is None:
        slug = title.replace(' ', '_').replace('/', '_')
        out_filename = f'Era_Maps_{slug}_{out_tag}.png'
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info('Dual-metric paired era maps saved to %s', out_path)


def create_ood_era_raster_maps(
    raster_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str = 'Out-of-Distribution Probability',
    cmap: str | None = None,
    fully_ood_threshold: float = 0.999,
    fully_ood_color: str = '#888888',
    out_filename: str | None = None,
    withdrawal_mask_raster_dir: str | None = None,
) -> None:
    """Era-mean OOD probability maps with two-class rendering.

    OOD probability is a per-pixel (mean over years) fraction of
    prediction samples flagged as out-of-distribution. In practice, the
    majority of Arizona pixels saturate to 1.0 across every era, which
    compresses the visually interesting <1.0 range into a sliver at the
    bottom of a continuous colorbar (most of the map renders as the top
    color of the palette). This function splits the rendering into two
    disjoint classes so the <1 variation is legible:

    1. **Partial OOD** (``0 <= mean OOD < fully_ood_threshold``) is
       rendered on a continuous ``cmap`` colorbar with ``vmin=0`` and
       ``vmax=fully_ood_threshold``. This gives the sub-saturation range
       the full dynamic range of the colormap.
    2. **Fully OOD** (``mean OOD >= fully_ood_threshold``) is rendered
       as a uniform gray fill (``fully_ood_color``) and labeled via a
       separate legend patch that sits next to the continuous colorbar.

    ``fully_ood_threshold`` defaults to 0.999 to absorb floating-point
    error from the era-mean arithmetic; treat any era-mean within
    ``1e-3`` of 1.0 as fully OOD.

    Args:
        raster_dir: Directory containing OOD probability rasters named
            with years, readable by ``_compute_era_means``.
        basin_shp: GW basin shapefile path.
        output_dir: Output directory for the PNG.
        title: Figure suptitle.
        cmap: Matplotlib colormap for the partial-OOD range.
        fully_ood_threshold: Inclusive lower bound for the fully-OOD
            class. Defaults to 0.999.
        fully_ood_color: Hex color for the fully-OOD class in both the
            map and the legend patch.
        out_filename: Optional override for the output PNG filename.
            Defaults to ``Era_Maps_{title_slug}.png``.

    Returns:
        None. Writes the PNG to ``output_dir``.
    """
    import rasterio as rio
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    apply_journal_style()
    makedirs(output_dir)

    # Muted five-stop ramp; the high end is intentionally PURPLE
    # (not the original brick) so the INA boundary overlay
    # (``INA_BORDER_COLOR`` = dark red ``#B71C1C``) stays visible
    # against the high-OOD pixels.  Earlier versions used a
    # green→sage→sand→terracotta→brick ramp; the brick end
    # (``#a63a2a``) was close enough to the INA red that INA
    # boundaries effectively disappeared in projection-era panels
    # where most of the AMA region saturates near OOD ≈ 1.  Going
    # green→sage→sand→amber→purple keeps the warm-cool perceptual
    # gradient but removes the red collision.  Callers can override
    # via ``cmap=...``.
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'muted_ood_ramp',
            [
                '#2e6b3c',  # dark green     (low OOD probability)
                '#7ba86a',  # sage
                '#d8c98a',  # sand
                '#d38a5a',  # amber / terracotta
                '#5a3a8c',  # deep purple    (high OOD probability)
            ],
        )

    tif_files = sorted(f for f in os.listdir(raster_dir) if f.endswith('.tif'))
    if not tif_files:
        logger.warning('No .tif files in %s — skipping OOD era maps.', raster_dir)
        return

    template = os.path.join(raster_dir, tif_files[0])
    with rio.open(template) as src:
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        crs = src.crs
        raster_shape = src.shape

    era_means = _compute_era_means(raster_dir, raster_shape, band=1,
                                    mask_nan_only=True)

    # Optional: align OOD footprint with the withdrawal era maps.
    # OOD-prob rasters cover the entire predictor grid (every pixel
    # the model evaluated, including deserts with zero withdrawal),
    # while the withdrawal era maps mask out pixels where the era-mean
    # is zero.  Without alignment, OOD shows red/orange across vast
    # uninhabited areas the withdrawal map renders blank, suggesting
    # an OOD problem at pixels where the model isn't actually making
    # a meaningful prediction.  When ``withdrawal_mask_raster_dir`` is
    # provided, recompute the same era-mean with the
    # ``mask_nan_only=False`` rule (mask zeros) and apply it as an
    # additional mask to the OOD era-means.
    if withdrawal_mask_raster_dir and os.path.isdir(
        withdrawal_mask_raster_dir,
    ):
        wd_era_means = _compute_era_means(
            withdrawal_mask_raster_dir, raster_shape, band=1,
            mask_nan_only=False,
        )
        for era, wd in wd_era_means.items():
            ood = era_means.get(era)
            if ood is None:
                continue
            era_means[era] = np.ma.masked_where(
                np.ma.getmaskarray(wd), ood,
            )

    basins_gdf = gpd.read_file(basin_shp)
    if basins_gdf.crs != crs:
        basins_gdf = basins_gdf.to_crs(crs)
    name_col = (
        'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    ama_ina = get_ama_ina_basin_names()

    n_eras = len(ERA_PERIODS)
    fig, axes = plt.subplots(
        n_eras, 1, figsize=(5.2, 4.3 * n_eras), constrained_layout=True,
    )
    fig.suptitle(f'{title} — Era Mean', fontsize=16, fontweight='bold')
    if n_eras == 1:
        axes = [axes]
    axes_flat = list(axes) if isinstance(axes, np.ndarray) else axes

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    last_partial_im = None
    for idx, (era, (y1, y2)) in enumerate(ERA_PERIODS.items()):
        ax = axes_flat[idx]
        ax.set_facecolor('#D5D5D5')  # no-data gray background

        era_arr = era_means.get(era)
        if era_arr is None or era_arr.count() == 0:
            blank = np.ma.masked_all(raster_shape)
            ax.imshow(
                blank, extent=extent, origin='upper',
                interpolation='nearest', cmap=cmap,
            )
        else:
            # Split into partial vs fully-OOD. Both are masked arrays
            # that inherit era_arr.mask for no-data pixels.
            fully_mask = era_arr >= fully_ood_threshold
            partial_arr = np.ma.masked_where(
                era_arr.mask | fully_mask, era_arr,
            )
            fully_arr = np.ma.masked_where(
                era_arr.mask | ~fully_mask, era_arr,
            )

            # Layer 1: fully-OOD as uniform gray
            fully_cmap = ListedColormap([fully_ood_color])
            ax.imshow(
                fully_arr, extent=extent, origin='upper',
                interpolation='nearest', cmap=fully_cmap,
                vmin=0, vmax=1,
            )
            # Layer 2: partial-OOD on the continuous colormap
            last_partial_im = ax.imshow(
                partial_arr, extent=extent, origin='upper',
                interpolation='nearest', cmap=cmap,
                vmin=0.0, vmax=fully_ood_threshold,
            )

        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col, label_all=True,
        )
        ax.set_title(
            f'{panel_labels[idx]} {era} ({y1}–{y2})',
            fontsize=14, fontweight='bold',
        )

    # Single AMA / INA / GW basin legend, outside-right of first panel
    add_ama_ina_legend(axes_flat[-1], loc='upper center',
        bbox_to_anchor=(0.5, -0.03), ncol=3)

    era_cbar_fontsize = 10
    if last_partial_im is not None:
        # extend='both' gives the horizontal colorbar triangular
        # arrowheads on both ends, matching the shape of the σ volume
        # era-map colorbars. The left arrowhead is ornamental for the
        # OOD probability (the partial-OOD range starts at 0, so
        # nothing is actually clipped below vmin); the right arrowhead
        # points toward the separately-rendered fully-OOD class.
        cbar = fig.colorbar(
            last_partial_im, ax=axes_flat, shrink=0.92, pad=0.02,
            orientation='horizontal', aspect=45, extend='both',
        )
        cbar.set_label(
            'Mean OOD Probability (partial)',
            fontsize=era_cbar_fontsize, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=era_cbar_fontsize)
        # Append the fully-OOD class as a proxy Patch glued to the
        # right side of the colorbar's own axis so the swatch + label
        # always sit on the same baseline as the colorbar itself and
        # can't collide with the cbar label no matter how the figure
        # bbox is trimmed by bbox_inches='tight'.
        fully_handle = mpatches.Patch(
            facecolor=fully_ood_color, edgecolor='#333333', linewidth=0.6,
            label=(
                f'Fully OOD (\u2265 {fully_ood_threshold:.3g})'
            ),
        )
        cbar.ax.legend(
            handles=[fully_handle],
            loc='center left',
            bbox_to_anchor=(1.04, 0.5),
            frameon=False,
            fontsize=era_cbar_fontsize - 1,
            handlelength=1.4,
            handletextpad=0.5,
        )

    if out_filename is None:
        slug = title.replace(' ', '_').replace('/', '_')
        out_filename = f'Era_Maps_{slug}.png'
    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info('OOD era raster maps saved to %s', out_path)


# ═════════════════════════════════════════════════════════════════════════════
# σ attribution diagnostics
# -----------------------------------------------------------------------------
# Basin-level three-way variance decomposition of σ_total into:
#   * Management: σ_irr² + σ_LULC² + σ_GW²      (fixable by better data)
#   * Climate:    σ_MACA²                        (inherent to GCM spread)
#   * Model:      σ_Model²                       (training-procedure floor)
#
# Two complementary visual products share the same underlying per-basin
# DataFrame:
#   - Binary 5-bin discrete choropleth (NV-style) — readable at a glance
#     via a 2-way classification metric whose axis depends on the era.
#     Projection uses Climate↔Management; Hindcast/Historical use
#     Model↔Management (σ_clim is zero in those eras). Projection basins
#     where σ_Model is the single largest contributor are flagged with a
#     bold black polygon edge + an on-figure disclosure box.
#   - Ternary RGB-mixed continuous choropleth — an honest three-way
#     disclosure of Mgmt/Clim/Model share on one color axis in every era.
#
# CU attribution propagates the IE × Withdrawal uncertainty
# (σ_CU = √((IE·σ_wd)² + (wd·σ_IE)²)) into the same three-way share
# decomposition, using the cached NHM basin IE + std.
#
# All outputs (attribution maps, CSVs, timeseries grid, bubble chart)
# are written to the Raster_Maps/Sigma_Attribution/ subdirectory to keep
# them isolated from the main era-mean, trend, and SW capture outputs.
# ═════════════════════════════════════════════════════════════════════════════

# 5-bin discrete classification edges (matches NV reference figure)
_SIGMA_ATTR_CLASS_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# Projection palette (blue → gray → orange)
_SIGMA_ATTR_PALETTE_PROJECTION = (
    '#1f4fa5', '#5a9ad0', '#b0b0b0', '#e8a582', '#d04e1f',
)
# Hindcast/Historical palette (purple → gray → orange), distinct from the
# Projection palette so readers don't confuse Model-dominated basins with
# Climate-dominated basins.
_SIGMA_ATTR_PALETTE_MODEL_ERA = (
    '#5e4fa2', '#8c7eb8', '#b0b0b0', '#e8a582', '#d04e1f',
)

# Shared directional tags for the 5 bins, indexed by [left_label, right_label]
_SIGMA_ATTR_ERA_TAGS = {
    'Projection': ('Climate Dominated', 'Management Dominated'),
    'Historical': ('Model Dominated', 'Management Dominated'),
    'Hindcast':   ('Model Dominated', 'Management Dominated'),
}

_SIGMA_ATTR_SHARE_LABELS = (
    '0–25%', '25–50%', '50–50%', '50–75%', '75–100%',
)

# Headline basins for the per-year attribution timeseries grid
_SIGMA_ATTR_HEADLINE_BASINS: tuple[str, ...] = ()  # populated at first use

# Withdrawal → CU pool mapping for σ_CU propagation
_SIGMA_CU_POOL_MAP = {
    'Irrigation_CU':    'Irrigation',
    'Irrigation_GW_CU': 'Irrigation_GW',
    'Irrigation_SW_CU': 'Irrigation_SW',
}

# σ components in the three-way decomposition.
# - Mgmt: irrigation mapping (σ_Irr), LULC scenarios (σ_LULC),
#         well-density temporal sensitivity (σ_GW).
# - Clim: AZ-local downscaled climate (σ_MACA) +
#         Upper Colorado River Basin streamflow (σ_USBR).
#         Both are climate-driven but cover different geographic
#         basins — see README §3d climate-driver decomposition.
# - Model: XGBRF seed ensemble (σ_Model).
_SIGMA_MGMT_COMPONENTS = ('Irr', 'LULC', 'GW')
_SIGMA_CLIM_COMPONENTS = ('MACA', 'USBR')
_SIGMA_MODEL_COMPONENTS = ('Model',)
_ALL_SIGMA_COMPONENTS = (
    _SIGMA_MGMT_COMPONENTS + _SIGMA_CLIM_COMPONENTS + _SIGMA_MODEL_COMPONENTS
)

_SIGMA_COMPONENT_DIRS = {
    'MACA':  'Sigma_MACA',
    'Model': 'Sigma_Model',
    'Irr':   'Sigma_Irr',
    'LULC':  'Sigma_LULC',
    'GW':    'Sigma_GW',
    'USBR':  'Sigma_USBR',
}


def _load_nhm_basin_ie_cache(unc_dir: str) -> dict[str, tuple[float, float]] | None:
    """Load the NHM basin IE cache written by Step 3b.

    Returns ``{basin: (mean_ie, std_ie)}`` or ``None`` if the cache is
    missing. Used by the σ_CU attribution to propagate withdrawal-side
    σ through ``CU = IE × Withdrawal``.
    """
    cache_csv = os.path.join(
        unc_dir, 'Sigma_CU', 'NHM_IE', 'NHM_basin_IE_cache.csv',
    )
    if not os.path.isfile(cache_csv):
        logger.warning(
            'NHM basin IE cache not found at %s — σ_CU attribution disabled',
            cache_csv,
        )
        return None
    df = pd.read_csv(cache_csv)
    mean_rows = df[df['year'].astype(str) == 'mean']
    if mean_rows.empty:
        logger.warning(
            'NHM basin IE cache at %s has no "mean" rows — σ_CU disabled',
            cache_csv,
        )
        return None
    result: dict[str, tuple[float, float]] = {}
    for _, row in mean_rows.iterrows():
        basin = str(row['basin'])
        ie_mean = float(row['ie']) if np.isfinite(row['ie']) else np.nan
        ie_std = (
            float(row['std_ie'])
            if 'std_ie' in row and np.isfinite(row.get('std_ie', np.nan))
            else 0.0
        )
        result[basin] = (ie_mean, ie_std)
    return result


def _read_basin_sigma_category_csv(
    unc_dir: str,
    component: str,
    pool: str,
) -> pd.DataFrame | None:
    """Read ``Uncertainty/{component_dir}/Basin_Sigma_{comp}_{pool}.csv``.

    Returns ``None`` if the file is missing.
    """
    comp_dir = _SIGMA_COMPONENT_DIRS[component]
    csv_path = os.path.join(
        unc_dir, comp_dir, f'Basin_Sigma_{component}_{pool}.csv',
    )
    if not os.path.isfile(csv_path):
        return None
    return pd.read_csv(csv_path)


def _load_sigma_attribution_data(
    unc_dir: str,
    pool: str,
    era: str,
) -> pd.DataFrame | None:
    """Load per-basin σ components for a pool, filter by era, and compute
    the three-way variance-share decomposition.

    Returns a DataFrame indexed by basin name with columns:
    ``Region, Sigma_MACA_m3, Sigma_Irr_m3, Sigma_LULC_m3, Sigma_GW_m3,
    Sigma_Model_m3, Mean_Wd_m3, Sigma_Mgmt_m3, Sigma_Clim_m3,
    Sigma_Model_Total_m3, Sigma_TotalQ_m3, Mgmt_Share, Clim_Share,
    Model_Share``.

    Era filtering uses ``ERA_PERIODS``. Per-era σ per basin is the mean
    of ``Sigma_Volume_m3`` across all years in the era (matches the
    per-year σ that the framework treats as the typical annual
    uncertainty). Components with no rows in an era collapse to 0.

    Returns ``None`` if *every* per-category CSV is missing (which means
    Step 3b has not been rerun with the per-category extension).
    """
    yr_start, yr_end = ERA_PERIODS[era]
    per_basin: dict[str, dict[str, float]] = {}
    any_loaded = False
    mean_wd_per_basin: dict[str, float] = {}
    for comp in _ALL_SIGMA_COMPONENTS:
        df = _read_basin_sigma_category_csv(unc_dir, comp, pool)
        if df is None:
            continue
        any_loaded = True
        era_df = df[(df['Year'] >= yr_start) & (df['Year'] <= yr_end)]
        if era_df.empty:
            continue
        grp_sigma = era_df.groupby('Region')['Sigma_Volume_m3'].mean()
        grp_mean = era_df.groupby('Region')['Mean_Volume_m3'].mean()
        for basin, sigma in grp_sigma.items():
            per_basin.setdefault(basin, {})[f'Sigma_{comp}_m3'] = float(sigma)
        # Any component works as a source of mean withdrawal; σ_Model has
        # coverage in every era so it is the most reliable. Still, take
        # whichever loads first and leave it — the mean is the same
        # physical quantity.
        for basin, wd in grp_mean.items():
            mean_wd_per_basin.setdefault(basin, float(wd))
    if not any_loaded:
        return None
    if not per_basin:
        return pd.DataFrame(columns=['Region'])
    rows = []
    for basin, sigmas in per_basin.items():
        s_maca = sigmas.get('Sigma_MACA_m3', 0.0)
        s_usbr = sigmas.get('Sigma_USBR_m3', 0.0)
        s_irr = sigmas.get('Sigma_Irr_m3', 0.0)
        s_lulc = sigmas.get('Sigma_LULC_m3', 0.0)
        s_gw = sigmas.get('Sigma_GW_m3', 0.0)
        s_model = sigmas.get('Sigma_Model_m3', 0.0)
        sigma_mgmt_sq = s_irr ** 2 + s_lulc ** 2 + s_gw ** 2
        # Clim = AZ-local downscaled climate (MACA) + Upper Basin
        # Colorado River streamflow (USBR).  Both are climate-driven
        # but geographically decoupled — see README §3d.
        sigma_clim_sq = s_maca ** 2 + s_usbr ** 2
        sigma_model_sq = s_model ** 2
        sigma_total_sq = sigma_mgmt_sq + sigma_clim_sq + sigma_model_sq
        if sigma_total_sq > 0:
            mgmt_share = sigma_mgmt_sq / sigma_total_sq
            clim_share = sigma_clim_sq / sigma_total_sq
            model_share = sigma_model_sq / sigma_total_sq
        else:
            mgmt_share = clim_share = model_share = np.nan
        rows.append({
            'Region': basin,
            'Sigma_MACA_m3': s_maca,
            'Sigma_USBR_m3': s_usbr,
            'Sigma_Irr_m3': s_irr,
            'Sigma_LULC_m3': s_lulc,
            'Sigma_GW_m3': s_gw,
            'Sigma_Model_m3': s_model,
            'Mean_Wd_m3': mean_wd_per_basin.get(basin, np.nan),
            'Sigma_Mgmt_m3': np.sqrt(sigma_mgmt_sq),
            'Sigma_Clim_m3': np.sqrt(sigma_clim_sq),
            'Sigma_Model_Total_m3': np.sqrt(sigma_model_sq),
            'Sigma_TotalQ_m3': np.sqrt(sigma_total_sq),
            'Mgmt_Share': mgmt_share,
            'Clim_Share': clim_share,
            'Model_Share': model_share,
        })
    return pd.DataFrame(rows)


def _load_sigma_cu_attribution_data(
    unc_dir: str,
    pool: str,
    era: str,
) -> pd.DataFrame | None:
    """Load per-basin σ_CU three-way decomposition via the IE × Withdrawal
    error propagation.

    The CU pool is mapped to its parent withdrawal pool
    (``Irrigation_CU → Irrigation``, etc.). For each basin::

        σ_cu_mgmt²  = (IE × σ_wd_mgmt)² + (wd × σ_IE)²
        σ_cu_clim²  = (IE × σ_wd_clim)²
        σ_cu_model² = (IE × σ_wd_model)²

    σ_IE is absorbed into the management class because it is an
    irrigation-data-quality problem.

    Returns ``None`` if the parent withdrawal CSVs or the NHM IE cache
    are missing.
    """
    parent_pool = _SIGMA_CU_POOL_MAP.get(pool)
    if parent_pool is None:
        logger.warning(
            'σ_CU attribution: unknown pool %s (not in %s)',
            pool, sorted(_SIGMA_CU_POOL_MAP),
        )
        return None
    wd_df = _load_sigma_attribution_data(unc_dir, parent_pool, era)
    if wd_df is None or wd_df.empty:
        return None
    ie_map = _load_nhm_basin_ie_cache(unc_dir)
    if ie_map is None:
        return None

    rows = []
    for _, row in wd_df.iterrows():
        basin = row['Region']
        wd = row['Mean_Wd_m3'] if np.isfinite(row['Mean_Wd_m3']) else 0.0
        ie_mean, ie_std = ie_map.get(basin, (np.nan, np.nan))
        if not np.isfinite(ie_mean):
            continue
        s_wd_mgmt = row['Sigma_Mgmt_m3']
        s_wd_clim = row['Sigma_Clim_m3']
        s_wd_model = row['Sigma_Model_Total_m3']
        sigma_cu_mgmt_sq = (
            (ie_mean * s_wd_mgmt) ** 2 + (wd * ie_std) ** 2
        )
        sigma_cu_clim_sq = (ie_mean * s_wd_clim) ** 2
        sigma_cu_model_sq = (ie_mean * s_wd_model) ** 2
        sigma_cu_total_sq = (
            sigma_cu_mgmt_sq + sigma_cu_clim_sq + sigma_cu_model_sq
        )
        if sigma_cu_total_sq > 0:
            mgmt_share = sigma_cu_mgmt_sq / sigma_cu_total_sq
            clim_share = sigma_cu_clim_sq / sigma_cu_total_sq
            model_share = sigma_cu_model_sq / sigma_cu_total_sq
        else:
            mgmt_share = clim_share = model_share = np.nan
        rows.append({
            'Region': basin,
            'Sigma_MACA_m3': row['Sigma_MACA_m3'] * ie_mean,
            'Sigma_USBR_m3': row.get('Sigma_USBR_m3', 0.0) * ie_mean,
            'Sigma_Irr_m3':  row['Sigma_Irr_m3'] * ie_mean,
            'Sigma_LULC_m3': row['Sigma_LULC_m3'] * ie_mean,
            'Sigma_GW_m3':   row['Sigma_GW_m3'] * ie_mean,
            'Sigma_Model_m3': row['Sigma_Model_m3'] * ie_mean,
            'Mean_Wd_m3': wd,
            'IE_Mean': ie_mean,
            'IE_Std': ie_std,
            'Sigma_Mgmt_m3': np.sqrt(sigma_cu_mgmt_sq),
            'Sigma_Clim_m3': np.sqrt(sigma_cu_clim_sq),
            'Sigma_Model_Total_m3': np.sqrt(sigma_cu_model_sq),
            'Sigma_TotalQ_m3': np.sqrt(sigma_cu_total_sq),
            'Mgmt_Share': mgmt_share,
            'Clim_Share': clim_share,
            'Model_Share': model_share,
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _compute_binary_share(
    df: pd.DataFrame,
    era: str,
) -> np.ndarray:
    """Era-specific binary share metric used as the color axis of the
    NV-style 5-bin discrete map.

    * Projection: ``Mgmt / (Mgmt + Clim)`` — climate↔management trade-off.
    * Hindcast / Historical: ``Mgmt / (Mgmt + Model)`` — model floor vs
      management-driven data gaps. σ_clim is structurally zero in these
      eras, so it drops out of the classification.
    """
    mgmt = df['Mgmt_Share'].to_numpy(dtype=float)
    if era == 'Projection':
        other = df['Clim_Share'].to_numpy(dtype=float)
    else:
        other = df['Model_Share'].to_numpy(dtype=float)
    denom = mgmt + other
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(denom > 0, mgmt / denom, np.nan)
    return share


def _classify_share_bins(share: np.ndarray) -> np.ndarray:
    """Return integer bin indices in [0, 4] for a share array in [0, 1]."""
    bins = np.full(share.shape, -1, dtype=np.int8)
    finite = np.isfinite(share)
    if not finite.any():
        return bins
    cls = np.digitize(share[finite], _SIGMA_ATTR_CLASS_EDGES[1:-1])
    bins[finite] = np.clip(cls, 0, 4)
    return bins


def _palette_for_era(era: str) -> tuple[str, ...]:
    """Return the 5-bin palette for *era*."""
    if era == 'Projection':
        return _SIGMA_ATTR_PALETTE_PROJECTION
    return _SIGMA_ATTR_PALETTE_MODEL_ERA


def _draw_sigma_attribution_legend(
    fig,
    era: str,
) -> None:
    """Render the shared bottom 5-swatch legend with directional tags."""
    palette = _palette_for_era(era)
    left_tag, right_tag = _SIGMA_ATTR_ERA_TAGS[era]
    handles = [
        mpatches.Patch(facecolor=palette[i], edgecolor='#333333',
                       linewidth=0.6, label=_SIGMA_ATTR_SHARE_LABELS[i])
        for i in range(5)
    ]
    # σ_Model dominant edge handle (Projection era only — the
    # Mgmt-vs-Climate axis cannot represent Model-dominated variance,
    # so those basins get a thick lime polygon outline that overlays
    # without hiding the underlying Mgmt/Climate fill color).
    if era == 'Projection':
        model_dom_patch = mpatches.Patch(
            facecolor='white', edgecolor='#00C853',
            linewidth=2.0, label=r'$\sigma_\mathrm{model}$ dominated',
        )
        handles.append(model_dom_patch)
    na_patch = mpatches.Patch(
        facecolor='#E8E8E8', edgecolor='#999999',
        linewidth=0.6, hatch='///', label='N/A',
    )
    na_patch.set_edgecolor('#AAAAAA')
    handles.append(na_patch)
    # Wrap to at most 4 columns so the legend fits the single-column
    # (~5 in wide) vertical figure instead of forcing it wide.
    ncol = min(len(handles), 4)
    leg = fig.legend(
        handles=handles,
        loc='lower center',
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=0.9,
        title=(
            f'Management share of classifiable variance\n'
            f'({left_tag}  ←  Mixed (50 / 50)  →  {right_tag})'
        ),
        title_fontsize=9.5,
    )
    leg._legend_box.align = 'center'


def _draw_sigma_attribution_disclosure_box(
    fig,
    df: pd.DataFrame,
) -> None:
    """Render the Projection-era disclosure text box reporting σ_Model
    dominance, centered above the shared bottom legend of the figure.

    Also bumps the bottom subplots-adjust margin so the panels do not
    overlap the two-line disclosure, and lowers the legend's anchor
    slightly so the text sits cleanly between the panels and the
    5-swatch legend.
    """
    if df.empty:
        return
    model_share = df['Model_Share'].to_numpy(dtype=float)
    mgmt = df['Mgmt_Share'].to_numpy(dtype=float)
    clim = df['Clim_Share'].to_numpy(dtype=float)
    finite = np.isfinite(model_share) & np.isfinite(mgmt) & np.isfinite(clim)
    if not finite.any():
        return
    model_share_f = model_share[finite]
    mgmt_f = mgmt[finite]
    clim_f = clim[finite]
    n_model_dom = int(
        ((model_share_f > mgmt_f) & (model_share_f > clim_f)).sum()
    )
    n_total = int(finite.sum())
    median_model = float(np.median(model_share_f)) * 100.0
    # Hard-wrap each sentence so the disclosure stays within the narrow
    # (~5 in) single-column figure width — an unwrapped ~150-char line
    # would expand the tight bbox and blow the figure out sideways.
    import textwrap as _textwrap
    _line1 = (
        f'$\\sigma_\\mathrm{{model}}$ dominant in {n_model_dom}/{n_total} '
        f'basins ({n_model_dom / n_total * 100:.0f}%); median model share '
        f'{median_model:.0f}%.'
    )
    _line2 = (
        'Color axis classifies Management vs Climate within the '
        'remaining variance; basins outlined in lime green (see legend) '
        'are model-dominated overall.'
    )
    txt = '\n'.join(
        _textwrap.fill(s, width=48) for s in (_line1, _line2)
    )
    # Make room for the two-line disclosure between the map panels and
    # the bottom legend. The legend-only bottom margin is computed in
    # absolute inches by ``_setup_attr_figure`` so that 1-row and 2-row
    # layouts get the same visual whitespace below the maps; we grow
    # that margin by the same fixed amount (0.4 inches) regardless of
    # figure height so the disclosure box always has the same vertical
    # budget between the panels and the legend. Falls back to the old
    # 0.20 fraction if the figure was not set up via _setup_attr_figure.
    fig_height = getattr(fig, '_attr_fig_height', None)
    bottom_inches = getattr(fig, '_attr_bottom_inches', None)
    if fig_height and bottom_inches:
        # Reserve, in absolute inches: the legend strip (bottom_inches)
        # at the very bottom, then the disclosure text block above it,
        # then the map panels.  Growing the bottom margin by a fixed
        # amount keeps the same visual gap regardless of figure height
        # (11 in headline vs 21 in detailed). The text bottom sits a
        # little above the legend strip so it never collides with it.
        new_bottom_inches = bottom_inches + 1.8
        new_bottom = new_bottom_inches / fig_height
        text_y = (bottom_inches + 0.35) / fig_height
    else:
        new_bottom = 0.30
        text_y = 0.18
    fig.subplots_adjust(bottom=new_bottom)
    fig.text(
        0.5, text_y, txt,
        ha='center', va='bottom',
        fontsize=11,
        bbox=dict(
            boxstyle='round,pad=0.4', facecolor='white',
            edgecolor='#555555', linewidth=0.6, alpha=0.9,
        ),
        zorder=10,
    )


def _draw_sigma_attribution_ternary_legend(fig) -> None:
    """Render the small RGB-mixed ternary triangle inset at bottom-left.

    The inset is placed in the whitespace below the map panels, using
    the figure's ``subplotpars.bottom`` to avoid overlapping the
    bottom-left panel (c) on 2×2 detailed layouts.
    """
    from matplotlib.patches import Polygon
    # Place the inset inside the bottom margin that _setup_attr_figure
    # reserved. On a 1×2 headline figure the bottom margin is ~0.14
    # (1 inch / 7 inches); on a 2×2 detailed figure it is ~0.083
    # (1 inch / 12 inches). A small inset (~0.12 wide × 0.08 tall in
    # figure fraction) that sits at y = 0.005 stays fully below the
    # bottom row of panels in both layouts.
    bottom = fig.subplotpars.bottom
    inset_height = min(0.16, bottom * 0.85)
    inset_width = inset_height * (fig.get_figheight() / fig.get_figwidth())
    ax_inset = fig.add_axes([0.02, 0.005, inset_width, inset_height])
    ax_inset.set_aspect('equal')
    ax_inset.axis('off')
    # Paint the triangle interior by tiling a grid of small polygons
    # colored by their barycentric coordinates → RGB.
    tri_vertices = np.array([
        [0.5, np.sqrt(3) / 2],   # Top — Management (red)
        [0.0, 0.0],              # Bottom-left — Climate (blue)
        [1.0, 0.0],              # Bottom-right — Model (green)
    ])
    n_sub = 20
    for i in range(n_sub):
        for j in range(n_sub - i):
            # Barycentric tri-sub-cell at (i, j, k) with k = n_sub-i-j
            a0 = i / n_sub
            b0 = j / n_sub
            c0 = 1 - a0 - b0
            if c0 < 0:
                continue
            # Four corners of a small parallelogram tile; we render it as
            # two triangles for speed.
            def bary_to_xy(a, b):
                c = 1 - a - b
                return (
                    a * tri_vertices[0] + b * tri_vertices[1]
                    + c * tri_vertices[2]
                )
            a1 = a0 + 1 / n_sub
            b1 = b0 + 1 / n_sub
            p0 = bary_to_xy(a0, b0)
            p1 = bary_to_xy(a1, b0)
            p2 = bary_to_xy(a0, b1)
            # The color is given by the centroid's barycentric coords,
            # fed through the same muted mix as the basin choropleth
            # via _ternary_mix so the inset and the map colors are a
            # single source of truth.
            a_c = a0 + 1 / (3 * n_sub)
            b_c = b0 + 1 / (3 * n_sub)
            c_c = 1 - a_c - b_c
            color = _ternary_mix(a_c, c_c, b_c)
            poly = Polygon(
                np.stack([p0, p1, p2]), facecolor=color, edgecolor='none',
            )
            ax_inset.add_patch(poly)
            if a0 + b0 + 1 / n_sub < 1:
                p3 = bary_to_xy(a1, b1)
                a_c2 = a0 + 2 / (3 * n_sub)
                b_c2 = b0 + 2 / (3 * n_sub)
                c_c2 = 1 - a_c2 - b_c2
                color2 = _ternary_mix(a_c2, c_c2, b_c2)
                poly2 = Polygon(
                    np.stack([p1, p3, p2]), facecolor=color2, edgecolor='none',
                )
                ax_inset.add_patch(poly2)
    # Triangle outline + labels. Corner label hex colors are chosen to
    # match the muted corner of the new ternary mix — slightly darker
    # than the rendered corner tile so the label reads clearly against
    # the tile fill.
    outline = Polygon(
        tri_vertices, closed=True, fill=False,
        edgecolor='black', linewidth=0.8,
    )
    ax_inset.add_patch(outline)
    ax_inset.text(
        0.5, np.sqrt(3) / 2 + 0.05, 'Management',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='#6a2214',
    )
    ax_inset.text(
        -0.05, -0.02, 'Climate',
        ha='right', va='top', fontsize=12, fontweight='bold', color='#142a6a',
    )
    ax_inset.text(
        1.05, -0.02, 'Model',
        ha='left', va='top', fontsize=12, fontweight='bold', color='#1a4a22',
    )
    # N/A swatch below the triangle — hatched to match the map polygons
    na_rect = mpatches.FancyBboxPatch(
        (0.25, -0.28), 0.5, 0.12,
        boxstyle='round,pad=0.02',
        facecolor='#E8E8E8', edgecolor='#999999',
        linewidth=0.6, hatch='///',
    )
    na_rect.set_edgecolor('#AAAAAA')
    ax_inset.add_patch(na_rect)
    ax_inset.text(
        0.5, -0.22, 'N/A',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='#555555',
    )
    ax_inset.set_xlim(-0.35, 1.35)
    ax_inset.set_ylim(-0.35, np.sqrt(3) / 2 + 0.22)


def _ternary_mix(mgmt: float, model: float, clim: float) -> tuple[float, float, float]:
    """Map a single (Mgmt, Model, Clim) triple to a muted RGB tuple.

    The old formula used ``R = Mgmt × 0.9`` (etc.), which pushed pure
    corners of the simplex to saturated primaries — fully-management
    basins rendered as eye-piercing pure red, etc. That made the
    ternary maps uncomfortably bright next to the muted basin
    shapefile and basin labels.

    The replacement uses a bias-and-slope form that keeps the colors
    in a matte, journal-quality range:

        R = bias + Mgmt  × slope
        G = bias + Model × slope
        B = bias + Clim  × slope

    with ``bias = 0.22`` and ``slope = 0.50`` so that a pure-Mgmt
    corner renders as ``(0.72, 0.22, 0.22)`` (a muted dusty red rather
    than the saturated ``(0.9, 0, 0)`` of the old formula), a
    balanced 1/3-1/3-1/3 basin renders as ``(0.39, 0.39, 0.39)`` (a
    comfortable mid-gray rather than the old near-black
    ``(0.3, 0.3, 0.3)``), and every basin stays in the
    ``[bias, bias + slope] = [0.22, 0.72]`` brightness range.

    The shares are assumed to be finite; caller is responsible for
    substituting a no-data color when they are not.
    """
    bias = 0.22
    slope = 0.50
    return (
        bias + float(mgmt) * slope,
        bias + float(model) * slope,
        bias + float(clim) * slope,
    )


def _ternary_rgb_colors(df: pd.DataFrame) -> list[tuple[float, float, float]]:
    """Convert per-basin Mgmt/Clim/Model shares to muted RGB colors.

    Each row's three shares are fed through :func:`_ternary_mix` so
    that the inset-triangle painter and the basin choropleth share one
    source of truth for the color formula.

    Basins with NaN shares return a no-data gray RGB tuple (the literal
    ``#D5D5D5`` hex converted to ``(0.8353, 0.8353, 0.8353)``). Returning
    tuples uniformly — rather than mixing hex strings with RGB tuples —
    lets ``GeoSeries.plot(color=...)`` build a homogeneous numpy array
    from the list; otherwise geopandas crashes with
    ``setting an array element with a sequence``.
    """
    nodata_rgb = (0xD5 / 255.0, 0xD5 / 255.0, 0xD5 / 255.0)
    colors: list[tuple[float, float, float]] = []
    for _, row in df.iterrows():
        mgmt = row['Mgmt_Share']
        clim = row['Clim_Share']
        model = row['Model_Share']
        if not (np.isfinite(mgmt) and np.isfinite(clim) and np.isfinite(model)):
            colors.append(nodata_rgb)
            continue
        colors.append(_ternary_mix(mgmt, model, clim))
    return colors


def _merge_attr_df_to_gdf(
    basins_gdf: gpd.GeoDataFrame,
    attr_df: pd.DataFrame,
    basin_col: str = 'BASIN_NAME',
) -> gpd.GeoDataFrame:
    """Left-merge the attribution DataFrame onto the basin shapefile."""
    merged = basins_gdf.merge(
        attr_df, how='left', left_on=basin_col, right_on='Region',
    )
    return merged


def _save_attribution_csv(
    attr_df: pd.DataFrame,
    pool: str,
    era: str,
    output_dir: str,
    csv_basename: str,
) -> None:
    """Append/write the long-format per-basin attribution CSV.

    Each invocation writes a row per basin to ``{csv_basename}_{era}.csv``
    with a ``Pool`` column so that multiple pools (headline, detailed)
    end up in the same per-era file.
    """
    if attr_df.empty:
        return
    out_path = os.path.join(output_dir, f'{csv_basename}_{era}.csv')
    out_df = attr_df.copy()
    out_df.insert(0, 'Era', era)
    out_df.insert(1, 'Pool', pool)
    if os.path.isfile(out_path):
        existing = pd.read_csv(out_path)
        existing = existing[existing['Pool'] != pool]
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df.to_csv(out_path, index=False)


def _setup_attr_figure(
    n_panels: int,
    title: str,
) -> tuple:
    """Create an attribution map figure and return (fig, axes_flat).

    Layout rule:
      * n_panels == 1  → 1×1
      * n_panels in {2, 3} → 1×n (side-by-side headline or CU)
      * n_panels == 4  → 2×2 (detailed Irrigation/Non-Irrigation × GW/SW)
      * n_panels >= 5  → 2×ceil(n/2) fallback

    The ``bottom`` subplots-adjust margin is computed as an absolute
    inch value rather than a figure fraction so the bottom whitespace
    below the map grid is visually constant regardless of whether the
    figure is 7 inches tall (1-row) or 12 inches tall (2-row). Without
    this, the 2×2 detailed layout inherits the same 14% fraction as the
    1×2 headline and ends up with nearly twice as much dead space
    between the bottom row of maps and the shared legend.

    The figure records ``_attr_fig_height`` and ``_attr_bottom_inches``
    attributes so :func:`_draw_sigma_attribution_disclosure_box` can
    grow the bottom margin by the same absolute amount when the
    disclosure text has to fit between the maps and the legend.

    Axes are always returned as a flat list so callers can index by
    pool-order regardless of the underlying grid shape.
    """
    # Single-column vertical stack: one pool per row so every choropleth
    # spans the full page width instead of being squeezed into a narrow
    # column of a horizontal strip.  Authored near display width, so the
    # page-fit shrink is ~1x and the fonts are ordinary point sizes.
    n_rows, n_cols = n_panels, 1
    per_panel_h = 5.0
    bottom_inches = 1.0   # reserved strip for the shared bottom legend
    fig_height = per_panel_h * n_panels + bottom_inches
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2, fig_height),
        constrained_layout=False,
    )
    if n_panels == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.asarray(axes).ravel())
    # Hide any trailing axes that have no pool assigned to them.
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].axis('off')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

    bottom_frac = bottom_inches / fig_height
    plt.subplots_adjust(
        left=0.04, right=0.98, top=0.95, bottom=bottom_frac,
        hspace=0.12,
    )
    fig._attr_fig_height = fig_height       # type: ignore[attr-defined]
    fig._attr_bottom_inches = bottom_inches  # type: ignore[attr-defined]
    return fig, axes_flat


def _draw_attribution_basin_panel(
    ax,
    basins_gdf: gpd.GeoDataFrame,
    attr_df: pd.DataFrame,
    era: str,
    *,
    ternary: bool,
    panel_title: str,
    panel_label: str,
    basin_col: str = 'BASIN_NAME',
) -> None:
    """Draw a single basin-level attribution choropleth panel."""
    ax.set_facecolor('#EEEEEE')
    merged = _merge_attr_df_to_gdf(basins_gdf, attr_df, basin_col=basin_col)
    palette = _palette_for_era(era)
    # Default edge styling (thin gray solid)
    edge_colors = np.full(len(merged), '#666666', dtype=object)
    linewidths = np.full(len(merged), 0.5, dtype=float)

    # Identify N/A basins (NaN shares → no attribution data).
    na_mask = ~(
        np.isfinite(merged['Mgmt_Share'].to_numpy(dtype=float))
        & np.isfinite(merged['Clim_Share'].to_numpy(dtype=float))
        & np.isfinite(merged['Model_Share'].to_numpy(dtype=float))
    )
    nodata_rgb = (0xD5 / 255.0, 0xD5 / 255.0, 0xD5 / 255.0)

    face_colors: list
    if ternary:
        face_colors = _ternary_rgb_colors(merged)
    else:
        share = _compute_binary_share(merged, era)
        bins = _classify_share_bins(share)
        # Use hex string (not RGB tuple) so geopandas gets a
        # homogeneous list of strings — mixing tuples with strings
        # causes numpy to choke on inhomogeneous shapes.
        face_colors = [
            palette[b] if b >= 0 else '#D5D5D5' for b in bins
        ]
        if era == 'Projection':
            model_share = merged['Model_Share'].to_numpy(dtype=float)
            mgmt_share = merged['Mgmt_Share'].to_numpy(dtype=float)
            clim_share = merged['Clim_Share'].to_numpy(dtype=float)
            finite = ~na_mask
            dominant = np.zeros(len(merged), dtype=bool)
            dominant[finite] = (
                (model_share[finite] > mgmt_share[finite])
                & (model_share[finite] > clim_share[finite])
            )
        else:
            dominant = np.zeros(len(merged), dtype=bool)

    # Plot all basins with default thin gray edges
    merged.plot(
        ax=ax,
        color=face_colors,
        edgecolor=list(edge_colors),
        linewidth=list(linewidths),
    )

    # Highlight σ_Model-dominant basins (Projection only) with a thick
    # bright-lime polygon edge.  Earlier iterations tried a red edge
    # (which clashed with the INA dark-red border) and a lavender
    # cross-hatch fill (which COMPLETELY HID the underlying Mgmt-vs-
    # Climate color, defeating the figure's primary attribution axis).
    # A bright lime edge stands out against the cool/warm diverging
    # palette, doesn't conflict with AMA black or INA dark red, and
    # leaves the choropleth color visible so reviewers can read both
    # the Model-dominance flag AND the residual Mgmt/Climate balance
    # at the same time.
    if not ternary and era == 'Projection' and dominant.any():
        dom_gdf = merged[dominant]
        dom_gdf.boundary.plot(
            ax=ax,
            color='#00C853',
            linewidth=2.0,
        )

    # Overlay N/A basins with a diagonal-hatch pattern on a light
    # gray fill, matching the NV reference figure. The hatch is
    # rendered by plotting the N/A subset a second time with
    # ``hatch='///'`` — this draws diagonal lines across the polygon
    # interior that survive any subsequent boundary overlay from
    # ``_overlay_boundaries``. The earlier solid fill from the main
    # ``.plot()`` call is covered by this opaque gray + hatch layer.
    na_gdf = merged[na_mask]
    if not na_gdf.empty:
        na_gdf.plot(
            ax=ax,
            color='#E8E8E8',
            edgecolor='#AAAAAA',
            linewidth=0.6,
            hatch='///',
        )

    # Label only the AMA/INA management areas — labeling all 52 basins
    # overlaps into an unreadable mass once the font is large enough to
    # read on a page-width multi-panel figure.
    ama_ina = get_ama_ina_basin_names()
    _overlay_boundaries(
        ax, basins_gdf, ama_ina, basin_col,
        label_fontsize=8, label_all=True,
    )
    add_ama_ina_legend(ax, fontsize=9)
    ax.set_title(
        f'{panel_label} {panel_title}',
        fontsize=14, fontweight='bold',
    )


def create_sigma_attribution_map(
    unc_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    pools: tuple[str, ...] = ('Total_GW', 'Total_SW'),
    eras: tuple[str, ...] = ('Hindcast', 'Historical', 'Projection'),
    filename_tag: str | None = None,
    csv_basename: str = 'Sigma_Attribution',
    load_fn=None,
    title_prefix: str = 'σ Attribution',
) -> None:
    """Binary 5-bin discrete σ-attribution choropleth, one figure per era.

    Reads per-basin per-category σ CSVs written by Step 3b, computes the
    three-way variance-share decomposition, and renders a 1×len(pools)
    panel figure for each era in *eras*.

    The binary color axis metric varies with era:
      - Projection: Management / (Management + Climate)
      - Hindcast/Historical: Management / (Management + Model)

    In the Projection era, basins whose σ_Model is larger than both
    σ_mgmt and σ_clim (i.e. Model-dominant) are flagged with a bold
    black polygon edge and an on-figure disclosure box reporting the
    count and median Model share.

    Args:
        unc_dir: ``.../Uncertainty`` directory holding per-component
            per-category basin CSVs.
        basin_shp: Path to the AZ groundwater basin shapefile.
        output_dir: Output directory (typically
            ``Raster_Maps/Sigma_Attribution/``).
        pools: Withdrawal pools to render as side-by-side panels in each
            figure.
        eras: Eras to render. Each era produces one figure.
        filename_tag: Optional filename suffix (e.g. ``'Detailed'``) used
            to distinguish headline vs detailed vs CU invocations.
        csv_basename: Filename stem for the companion long-format CSV.
        load_fn: Internal override for the per-pool loader (used by the
            CU variant). Defaults to ``_load_sigma_attribution_data``.
        title_prefix: Figure suptitle prefix (e.g. ``σ Attribution`` or
            ``σ_CU Attribution``).

    Returns:
        None. Writes ``Era_Maps_Sigma_Attribution_{Era}[_Tag].png`` and a
        companion long-format CSV per era to *output_dir*.
    """
    apply_journal_style()
    makedirs(output_dir)
    basins_gdf = gpd.read_file(basin_shp)
    basin_col = (
        'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    loader = load_fn if load_fn is not None else _load_sigma_attribution_data

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for era in eras:
        # Load data for every pool first so we can bail early if none
        pool_dfs: dict[str, pd.DataFrame] = {}
        for pool in pools:
            df = loader(unc_dir, pool, era)
            if df is None or df.empty:
                logger.warning(
                    '  [σ attribution] no data for pool=%s era=%s — '
                    'skipping this panel', pool, era,
                )
                continue
            pool_dfs[pool] = df
        if not pool_dfs:
            logger.warning(
                '  [σ attribution] era=%s has no pool data — skipping figure',
                era,
            )
            continue

        n_panels = len(pools)
        fig, axes = _setup_attr_figure(
            n_panels=n_panels,
            title=f'{title_prefix} — {era} ({ERA_PERIODS[era][0]}–'
                  f'{ERA_PERIODS[era][1]})',
        )
        for idx, pool in enumerate(pools):
            ax = axes[idx]
            df = pool_dfs.get(pool)
            if df is None:
                ax.set_facecolor('#EEEEEE')
                ax.text(
                    0.5, 0.5, f'{pool.replace("_", " ")}\n(no data)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11,
                )
                ax.axis('off')
                continue
            n_class = int(
                np.isfinite(df['Mgmt_Share']).sum()
            )
            pretty = pool.replace('_', ' ')
            _draw_attribution_basin_panel(
                ax, basins_gdf, df, era,
                ternary=False,
                panel_title=f'{pretty} — n={n_class}',
                panel_label=panel_labels[idx],
                basin_col=basin_col,
            )
            _save_attribution_csv(df, pool, era, output_dir, csv_basename)

        if era == 'Projection':
            first_pool_df = next(
                (pool_dfs[p] for p in pools if p in pool_dfs), None,
            )
            if first_pool_df is not None:
                _draw_sigma_attribution_disclosure_box(fig, first_pool_df)
        _draw_sigma_attribution_legend(fig, era)

        # Build output filename
        base = 'Era_Maps_Sigma_Attribution'
        if csv_basename.endswith('CU_Attribution'):
            base = 'Era_Maps_Sigma_CU_Attribution'
        parts = [base, era]
        if filename_tag:
            parts.append(filename_tag)
        out_path = os.path.join(output_dir, '_'.join(parts) + '.png')
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info('  σ attribution map saved to %s', out_path)


def create_sigma_cu_attribution_map(
    unc_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    pools: tuple[str, ...] = (
        'Irrigation_CU', 'Irrigation_GW_CU', 'Irrigation_SW_CU',
    ),
    eras: tuple[str, ...] = ('Hindcast', 'Historical', 'Projection'),
) -> None:
    """σ_CU binary attribution map via the ``IE × Withdrawal`` propagation.

    Delegates to :func:`create_sigma_attribution_map` with the
    CU-specific loader so the same NV-style 5-bin classification, black
    edge flag, and disclosure box apply to the CU decomposition.
    """
    create_sigma_attribution_map(
        unc_dir=unc_dir,
        basin_shp=basin_shp,
        output_dir=output_dir,
        pools=pools,
        eras=eras,
        filename_tag=None,
        csv_basename='Sigma_CU_Attribution',
        load_fn=_load_sigma_cu_attribution_data,
        title_prefix='σ_CU Attribution',
    )


def create_sigma_attribution_ternary_map(
    unc_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    pools: tuple[str, ...] = ('Total_GW', 'Total_SW'),
    eras: tuple[str, ...] = ('Hindcast', 'Historical', 'Projection'),
    filename_tag: str | None = None,
    for_cu: bool = False,
) -> None:
    """Ternary (RGB-mixed three-way continuous) σ-attribution choropleth.

    Each basin's color is an RGB mix where R=Management, G=Model,
    B=Climate. A small equilateral-triangle inset shows the color
    gradient with the three corners labeled. Works identically in every
    era without an era-specific color axis swap; in Hindcast and
    Historical the Climate share is zero so basins fall on the
    red↔green edge of the triangle (a correct visual disclosure that
    climate is structurally absent in those eras).

    Args:
        for_cu: When True, load data via the CU propagation instead of
            the raw withdrawal CSVs.

    Returns:
        None. Writes ``Era_Maps_Sigma_Attribution_Ternary_{Era}[_Tag].png``
        (or the corresponding ``Sigma_CU_Attribution_Ternary`` variant)
        per era. The companion CSV is shared with the binary map family.
    """
    apply_journal_style()
    makedirs(output_dir)
    basins_gdf = gpd.read_file(basin_shp)
    basin_col = (
        'BASIN_NAME' if 'BASIN_NAME' in basins_gdf.columns
        else basins_gdf.columns[0]
    )
    loader = (
        _load_sigma_cu_attribution_data if for_cu
        else _load_sigma_attribution_data
    )
    csv_basename = (
        'Sigma_CU_Attribution' if for_cu else 'Sigma_Attribution'
    )
    title_prefix = (
        'σ_CU Attribution (Ternary)' if for_cu
        else 'σ Attribution (Ternary)'
    )
    file_base = (
        'Era_Maps_Sigma_CU_Attribution_Ternary' if for_cu
        else 'Era_Maps_Sigma_Attribution_Ternary'
    )

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for era in eras:
        pool_dfs: dict[str, pd.DataFrame] = {}
        for pool in pools:
            df = loader(unc_dir, pool, era)
            if df is None or df.empty:
                continue
            pool_dfs[pool] = df
        if not pool_dfs:
            logger.warning(
                '  [ternary] era=%s has no pool data — skipping figure', era,
            )
            continue

        fig, axes = _setup_attr_figure(
            n_panels=len(pools),
            title=f'{title_prefix} — {era} ({ERA_PERIODS[era][0]}–'
                  f'{ERA_PERIODS[era][1]})',
        )
        for idx, pool in enumerate(pools):
            ax = axes[idx]
            df = pool_dfs.get(pool)
            if df is None:
                ax.set_facecolor('#EEEEEE')
                ax.text(
                    0.5, 0.5, f'{pool.replace("_", " ")}\n(no data)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11,
                )
                ax.axis('off')
                continue
            pretty = pool.replace('_', ' ')
            _draw_attribution_basin_panel(
                ax, basins_gdf, df, era,
                ternary=True,
                panel_title=pretty,
                panel_label=panel_labels[idx],
                basin_col=basin_col,
            )
            # Ensure CSV is written (may already be if binary map ran,
            # but _save_attribution_csv is idempotent per Pool).
            _save_attribution_csv(df, pool, era, output_dir, csv_basename)

        _draw_sigma_attribution_ternary_legend(fig)
        parts = [file_base, era]
        if filename_tag:
            parts.append(filename_tag)
        out_path = os.path.join(output_dir, '_'.join(parts) + '.png')
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info('  σ attribution ternary map saved to %s', out_path)


def _load_year_resolved_attribution(
    unc_dir: str,
    pool: str,
    basin: str,
) -> pd.DataFrame | None:
    """Load per-year three-way variance shares for a single basin + pool.

    Used by the timeseries grid. Returns a DataFrame indexed by Year
    with columns ``Mgmt_Share, Clim_Share, Model_Share``. Rows for
    years where every σ component is zero (or missing) drop out.
    """
    merged: dict[int, dict[str, float]] = {}
    any_loaded = False
    for comp in _ALL_SIGMA_COMPONENTS:
        df = _read_basin_sigma_category_csv(unc_dir, comp, pool)
        if df is None:
            continue
        any_loaded = True
        sub = df[df['Region'] == basin]
        for _, row in sub.iterrows():
            year = int(row['Year'])
            merged.setdefault(year, {})[f'Sigma_{comp}_m3'] = (
                float(row['Sigma_Volume_m3'])
            )
    if not any_loaded or not merged:
        return None
    rows = []
    for year in sorted(merged):
        sigmas = merged[year]
        s_maca = sigmas.get('Sigma_MACA_m3', 0.0)
        s_usbr = sigmas.get('Sigma_USBR_m3', 0.0)
        s_irr = sigmas.get('Sigma_Irr_m3', 0.0)
        s_lulc = sigmas.get('Sigma_LULC_m3', 0.0)
        s_gw = sigmas.get('Sigma_GW_m3', 0.0)
        s_model = sigmas.get('Sigma_Model_m3', 0.0)
        sigma_mgmt_sq = s_irr ** 2 + s_lulc ** 2 + s_gw ** 2
        # Clim = MACA + USBR (geographically decoupled climate components)
        sigma_clim_sq = s_maca ** 2 + s_usbr ** 2
        sigma_model_sq = s_model ** 2
        sigma_total_sq = sigma_mgmt_sq + sigma_clim_sq + sigma_model_sq
        if sigma_total_sq <= 0:
            continue
        rows.append({
            'Year': year,
            'Mgmt_Share': sigma_mgmt_sq / sigma_total_sq,
            'Clim_Share': sigma_clim_sq / sigma_total_sq,
            'Model_Share': sigma_model_sq / sigma_total_sq,
        })
    if not rows:
        return None
    return pd.DataFrame(rows).set_index('Year')


def create_sigma_attribution_timeseries(
    unc_dir: str,
    output_dir: str,
    *,
    basins: tuple[str, ...] | None = None,
    pools: tuple[str, ...] = ('Total_GW', 'Total_SW'),
    era_years: tuple[int, int] = (1896, 2099),
) -> None:
    """Per-year stacked-area plot of the three-way variance decomposition
    for the AMA/INA basins × pools.

    Renders an N×len(pools) grid (default 10×2 = 10 AMA/INA basins × 2
    pools) where each panel stacks Management (red), Model (teal), and
    Climate (blue) share in [0, 1] across all years in ``era_years``.
    Era shading reproduces the Hindcast/Historical/Projection boundaries.

    Args:
        basins: Basins to include. Defaults to the 10 AMA/INA basins
            from ``get_ama_ina_basin_names()``.

    Also writes a long-format companion CSV
    ``Sigma_Attribution_Timeseries.csv`` with one row per basin × pool ×
    year.
    """
    apply_journal_style()
    makedirs(output_dir)
    if basins is None:
        basins = tuple(get_ama_ina_basin_names())

    csv_rows: list[dict] = []
    band_colors = {
        'Management': '#d04e1f',
        'Model': '#2a9d8f',
        'Climate': '#1f4fa5',
    }

    for pool in pools:
        # Pre-load data for every basin so we can drop basins with no
        # data (important for SW pools where many AMA/INAs have zero
        # surface-water withdrawal and therefore no attribution).
        basin_data: list[tuple[str, pd.DataFrame]] = []
        for basin in basins:
            df = _load_year_resolved_attribution(unc_dir, pool, basin)
            if df is not None and not df.empty:
                basin_data.append((basin, df))
        if not basin_data:
            logger.warning(
                '  [timeseries] pool=%s has no basin data — skipping', pool,
            )
            continue
        n_active = len(basin_data)

        # 2-column layout: rows = ceil(n_active / 2)
        n_cols = 2
        n_rows = int(np.ceil(n_active / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6.5 * n_cols, 2.3 * n_rows),
            sharex=True,
            constrained_layout=True,
        )
        axes_flat = list(np.asarray(axes).ravel())
        # Hide trailing axes if n_active is odd
        for k in range(n_active, len(axes_flat)):
            axes_flat[k].axis('off')

        pretty_pool = pool.replace('_', ' ')
        fig.suptitle(
            f'σ Attribution — {pretty_pool}',
            fontsize=14, fontweight='bold',
        )

        for ib, (basin, df) in enumerate(basin_data):
            ax = axes_flat[ib]
            ax.set_xlim(era_years[0], era_years[1])
            ax.set_ylim(0.0, 1.0)
            years = df.index.to_numpy()
            mgmt = df['Mgmt_Share'].to_numpy()
            model = df['Model_Share'].to_numpy()
            clim = df['Clim_Share'].to_numpy()
            ax.stackplot(
                years, mgmt, model, clim,
                colors=(
                    band_colors['Management'],
                    band_colors['Model'],
                    band_colors['Climate'],
                ),
                edgecolor='none',
            )
            for _, row in df.reset_index().iterrows():
                csv_rows.append({
                    'Basin': basin,
                    'Pool': pool,
                    'Year': int(row['Year']),
                    'Mgmt_Share': row['Mgmt_Share'],
                    'Clim_Share': row['Clim_Share'],
                    'Model_Share': row['Model_Share'],
                })
            for era, (s, e) in ERA_PERIODS.items():
                ax.axvspan(s, e, color=ERA_COLORS[era], alpha=0.05, zorder=0)
            ax.set_title(
                _format_basin_label(basin),
                fontsize=9, fontweight='bold',
            )
            if ib % n_cols == 0:
                ax.set_ylabel('Variance share', fontsize=9)
            if ib >= n_active - n_cols:
                ax.set_xlabel('Year', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2, linestyle='--')

        handles = [
            mpatches.Patch(color=band_colors['Management'], label='Management'),
            mpatches.Patch(color=band_colors['Model'], label='Model'),
            mpatches.Patch(color=band_colors['Climate'], label='Climate'),
        ]
        fig.legend(
            handles=handles,
            loc='lower center', ncol=3,
            bbox_to_anchor=(0.5, -0.015),
            frameon=False, fontsize=10,
        )

        pool_slug = pool.replace(' ', '_')
        out_path = os.path.join(
            output_dir, f'Sigma_Attribution_Timeseries_{pool_slug}.png',
        )
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info('  σ attribution timeseries saved to %s', out_path)

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(output_dir, 'Sigma_Attribution_Timeseries.csv'),
            index=False,
        )


def _format_basin_label(name: str) -> str:
    """Title-case a basin name while preserving AMA/INA as uppercase."""
    parts = name.title().split()
    return ' '.join(
        p.upper() if p.upper() in ('AMA', 'INA') else p for p in parts
    )


def create_sigma_attribution_bubble(
    unc_dir: str,
    output_dir: str,
    *,
    pools: tuple[str, ...] = ('Total_GW', 'Total_SW'),
    era: str = 'Projection',
) -> None:
    """Horizontal stacked-bar chart of the three-way variance
    decomposition per basin — one separate figure per pool.

    Each basin gets one horizontal bar whose total length is
    proportional to ``σ_total_quadrature`` (×10⁶ m³) and whose three
    stacked segments are colored by the Management / Model / Climate
    variance shares. Basins are sorted top-to-bottom by descending
    σ_total so the largest-uncertainty basins appear at the top.

    Basins whose σ_Model is the single largest variance contributor
    are marked with a red bar-edge rectangle (mirroring the red
    polygon edge on the binary attribution maps).
    """
    apply_journal_style()
    makedirs(output_dir)

    band_colors = {
        'Management': '#d04e1f',
        'Model': '#2a9d8f',
        'Climate': '#1f4fa5',
    }

    for pool in pools:
        df = _load_sigma_attribution_data(unc_dir, pool, era)
        if df is None or df.empty:
            logger.warning(
                '  [stacked bar] no data for pool=%s era=%s — skipping',
                pool, era,
            )
            continue

        valid = df.dropna(subset=['Mgmt_Share', 'Clim_Share', 'Model_Share'])
        if valid.empty:
            continue
        valid = valid.sort_values('Sigma_TotalQ_m3', ascending=True)

        n_basins = len(valid)
        basin_labels = [_format_basin_label(r) for r in valid['Region']]
        sigma_total_m6 = valid['Sigma_TotalQ_m3'].to_numpy() / 1e6
        mgmt_share = valid['Mgmt_Share'].to_numpy()
        model_share = valid['Model_Share'].to_numpy()
        clim_share = valid['Clim_Share'].to_numpy()
        model_dom = (
            (model_share > mgmt_share) & (model_share > clim_share)
        )

        pretty_pool = pool.replace('_', ' ')
        ama_ina_set = set(get_ama_ina_basin_names())
        handles = [
            mpatches.Patch(
                color=band_colors['Management'], label='Management',
            ),
            mpatches.Patch(
                color=band_colors['Model'], label='Model',
            ),
            mpatches.Patch(
                color=band_colors['Climate'], label='Climate',
            ),
            mpatches.Patch(
                fill=False, edgecolor='#d62728', linewidth=1.5,
                label='Model-dominated',
            ),
        ]

        # Split into two side-by-side panels when there are many
        # basins (> 30) so each panel is tall enough to read without
        # being absurdly long as a single column. The first panel
        # takes the top half (higher σ_total), the second panel takes
        # the bottom half. For ≤ 30 basins a single panel is fine.
        split_threshold = 30
        if n_basins > split_threshold:
            mid = n_basins // 2
            chunks = [
                (basin_labels[mid:], sigma_total_m6[mid:],
                 mgmt_share[mid:], model_share[mid:], clim_share[mid:],
                 model_dom[mid:]),
                (basin_labels[:mid], sigma_total_m6[:mid],
                 mgmt_share[:mid], model_share[:mid], clim_share[:mid],
                 model_dom[:mid]),
            ]
            n_cols = 2
        else:
            chunks = [
                (basin_labels, sigma_total_m6, mgmt_share,
                 model_share, clim_share, model_dom),
            ]
            n_cols = 1

        n_per_panel = max(len(c[0]) for c in chunks)
        fig_height = max(6, 0.35 * n_per_panel)
        fig, axes = plt.subplots(
            1, n_cols, figsize=(7 * n_cols, fig_height),
            constrained_layout=True,
        )
        if n_cols == 1:
            axes = [axes]
        else:
            axes = list(axes)
        fig.suptitle(
            f'σ Attribution — {pretty_pool}, {era}',
            fontsize=14, fontweight='bold',
        )

        x_max = float(sigma_total_m6.max()) * 1.05

        for col_idx, (lbl, stm6, ms, mds, cs, mdom) in enumerate(chunks):
            ax = axes[col_idx]
            n_chunk = len(lbl)
            y_pos = np.arange(n_chunk)
            bar_height = 0.7
            mw = stm6 * ms
            mdw = stm6 * mds
            cw = stm6 * cs

            ax.barh(y_pos, mw, height=bar_height,
                    color=band_colors['Management'])
            ax.barh(y_pos, mdw, height=bar_height, left=mw,
                    color=band_colors['Model'])
            ax.barh(y_pos, cw, height=bar_height, left=mw + mdw,
                    color=band_colors['Climate'])
            for i in range(n_chunk):
                if mdom[i]:
                    ax.barh(y_pos[i], stm6[i], height=bar_height,
                            fill=False, edgecolor='#d62728', linewidth=1.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(lbl), fontsize=7)
            ax.set_xlabel(
                r'σ$_{\mathrm{total}}$ ($\times10^{6}$ m$^3$)',
                fontsize=10, fontweight='bold',
            )
            ax.set_xlim(0, x_max)
            ax.grid(axis='x', alpha=0.25, linestyle='--')
            for tick_label in ax.get_yticklabels():
                if tick_label.get_text().upper() in ama_ina_set:
                    tick_label.set_fontweight('bold')
            if col_idx == n_cols - 1:
                ax.legend(
                    handles=handles,
                    loc='lower right', fontsize=8, framealpha=0.9,
                )

        pool_slug = pool.replace(' ', '_')
        out_path = os.path.join(
            output_dir, f'Sigma_Attribution_Stacked_Bar_{pool_slug}_{era}.png',
        )
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info('  σ attribution stacked bar saved to %s', out_path)


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

    Layout: Actual (mean) | Predicted (mean) | Difference (Actual − Predicted).
    The difference panel follows the same sign convention as ``normalized_mbe``
    in ``mlops.py`` (``np.mean(y - y_pred)``), so positive values indicate
    regions where the model under-predicts the metered record (actual > pred)
    and negative values indicate over-prediction.
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
                nd = src.nodata
            mask = ~np.isnan(arr) & (arr != 0)
            if nd is not None:
                mask &= arr != nd
            total[mask] += arr[mask]
            valid_count[mask] += 1
        with np.errstate(invalid='ignore'):
            mean = np.where(valid_count > 0, total / valid_count, 0.0)
        return mean, valid_count

    actual_mean, actual_count = _accumulate(actual_dir, actual_files, shape_actual)
    pred_mean, pred_count = _accumulate(predicted_dir, predicted_files, shape_pred)

    # Build AZ boundary mask from predicted rasters (any pixel with data is inside AZ)
    az_mask_pred = pred_count > 0
    # For actual grid, build AZ mask from the predicted grid (resample if needed)
    if shape_actual != shape_pred:
        from scipy.ndimage import zoom
        zoom_factors_az = (shape_actual[0] / shape_pred[0],
                           shape_actual[1] / shape_pred[1])
        az_mask_actual = zoom(az_mask_pred.astype(np.float64),
                              zoom_factors_az, order=0) > 0.5
    else:
        az_mask_actual = az_mask_pred

    # Mask: outside AZ = NaN (white background), inside AZ no-data = special gray
    # For actual: pixels inside AZ but with no meter data → set to a sentinel
    # that will be rendered as gray via a custom colormap treatment
    actual_display = actual_mean.copy()
    actual_display[~az_mask_actual] = np.nan  # outside AZ → white
    actual_masked = np.ma.masked_where(np.isnan(actual_display), actual_display)
    # Mark inside-AZ no-meter pixels as masked (will show as gray facecolor)
    actual_no_meter = az_mask_actual & (actual_count == 0)
    actual_masked = np.ma.masked_where(
        np.isnan(actual_display) | actual_no_meter, actual_display)

    pred_display = pred_mean.copy()
    pred_display[~az_mask_pred] = np.nan  # outside AZ → white
    pred_masked = np.ma.masked_where(np.isnan(pred_display), pred_display)

    # Difference: only where actual has meter data
    if shape_actual != shape_pred:
        from scipy.ndimage import zoom
        zoom_factors = (shape_actual[0] / shape_pred[0],
                        shape_actual[1] / shape_pred[1])
        pred_for_diff = zoom(pred_mean, zoom_factors, order=1)
    else:
        pred_for_diff = pred_mean

    # Sign convention matches normalized_mbe in mlops.py:
    # positive diff = actual > predicted (model under-predicts),
    # negative diff = actual < predicted (model over-predicts).
    diff = actual_mean - pred_for_diff
    diff_display = diff.copy()
    diff_display[~az_mask_actual] = np.nan
    diff_masked = np.ma.masked_where(
        np.isnan(diff_display) | (actual_count == 0), diff_display)

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

    # Build AZ-interior gray underlay images. Pixels inside AZ that lack
    # observed/predicted data render as a uniform light gray (#D1D1D1)
    # so the "Unmetered" legend swatch matches the on-map color exactly.
    from matplotlib.colors import ListedColormap as _ListedColormap
    _UNMETERED_CMAP = _ListedColormap(['#D1D1D1'])
    az_gray_actual = np.where(az_mask_actual, 0.5, np.nan)
    az_gray_pred = np.where(az_mask_pred, 0.5, np.nan)

    # ---- Helper: create one 1×3 figure with two shared colorbars ----
    # Layout via GridSpec:
    #   Row 0: [Actual]  [Predicted]  [Difference]   (map panels)
    #   Row 1: [──── shared cbar ────] [diff cbar ]   (colorbars)
    # The shared colorbar spans columns 0–1, the diff colorbar
    # sits under column 2. Both are horizontal and carry dual
    # units (primary label below, secondary label above via
    # secondary_xaxis).
    def _make_figure(data_sets, suptitle, primary_label, secondary_label,
                     secondary_factor, out_file, v_lo, v_hi, d_lo, d_hi):
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        # constrained_layout reliably aligns the three equal-aspect maps
        # in one row and manages the titles/colorbars/suptitle spacing
        # (the previous manual GridSpec left the difference panel's title
        # riding higher than the other two).
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 6.3), constrained_layout=True,
        )
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

        unmetered_handle = mpatches.Patch(
            facecolor='#D1D1D1', edgecolor='#555555', linewidth=0.4,
            label='Unmetered',
        )
        avp_fontsize = 10

        is_volume = 'm$^3$' in primary_label or 'm3' in primary_label.lower()
        vol_formatter = vol_label = dvol_label = None
        if is_volume:
            import matplotlib.ticker as mticker
            vol_formatter = mticker.FuncFormatter(lambda x, _: f'{x / 1e6:g}')
            vol_label = r'Volume ($\times$10$^{6}$ m$^3$)'
            dvol_label = r'$\Delta$ Volume ($\times$10$^{6}$ m$^3$)'

        im_shared = None
        im_diff = None
        for idx, (panel_title, data, ext, cm, lo, hi, az_gray) in enumerate(
                data_sets):
            ax = axes[idx]
            ax.set_facecolor('white')
            ax.imshow(
                az_gray, extent=ext, origin='upper',
                cmap=_UNMETERED_CMAP, vmin=0, vmax=1,
                interpolation='nearest', zorder=0,
            )
            im = ax.imshow(
                data, extent=ext, origin='upper', cmap=cm,
                vmin=lo, vmax=hi, interpolation='nearest', zorder=1,
            )
            _overlay_boundaries(ax, basins_gdf, ama_ina, name_col,
                                label_all=True)
            ax.set_title(panel_title, fontsize=14, fontweight='bold')
            if 'Difference' in panel_title:
                im_diff = im
            else:
                im_shared = im
            if idx == 0:
                # Two-column, no-box legend in the SW-corner no-data
                # triangle of the first map.
                handles = [
                    unmetered_handle,
                    Line2D([0], [0], color=BASIN_BORDER_COLOR, lw=0.8,
                           label='GW basin'),
                    Line2D([0], [0], color=AMA_BORDER_COLOR, lw=1.4,
                           label='AMA'),
                    Line2D([0], [0], color=INA_BORDER_COLOR, lw=1.4,
                           label='INA'),
                ]
                ax.legend(
                    handles=handles, loc='lower left',
                    bbox_to_anchor=(0.0, -0.06), ncol=2,
                    columnspacing=1.0, handlelength=1.2,
                    fontsize=avp_fontsize, frameon=False,
                )

        def _cbar_units(cbar, plabel, slabel, vlabel):
            if is_volume:
                cbar.set_label(vlabel, fontsize=avp_fontsize,
                               fontweight='bold')
                cbar.formatter = vol_formatter
                cbar.update_ticks()
            else:
                cbar.set_label(plabel, fontsize=avp_fontsize,
                               fontweight='bold')
            cbar.ax.tick_params(labelsize=avp_fontsize)
            secax = cbar.ax.secondary_xaxis(
                'top',
                functions=(lambda x: x * secondary_factor,
                           lambda x: x / secondary_factor),
            )
            secax.set_xlabel(slabel, fontsize=avp_fontsize, fontweight='bold')
            secax.tick_params(labelsize=avp_fontsize)

        # Shared colorbar under (a)+(b); a separate diff colorbar under (c).
        cb_shared = fig.colorbar(
            im_shared, ax=[axes[0], axes[1]], orientation='horizontal',
            location='bottom', shrink=0.8, pad=0.02, aspect=45,
            extend='both',
        )
        _cbar_units(cb_shared, primary_label, secondary_label, vol_label)
        cb_diff = fig.colorbar(
            im_diff, ax=axes[2], orientation='horizontal',
            location='bottom', shrink=0.9, pad=0.02, aspect=22,
            extend='both',
        )
        _cbar_units(cb_diff, f'Δ {primary_label}',
                    f'Δ {secondary_label}', dvol_label)

        out_path = os.path.join(output_dir, out_file)
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'Actual vs predicted maps saved to {out_path}')

    # ---- Depth maps (mm / ft) ----
    depth_panels = [
        ('(a) Actual (Metered)', actual_masked, extent_actual,
         cmap, v_min, v_max, az_gray_actual),
        ('(b) Predicted (ML)', pred_masked, extent_pred,
         cmap, v_min, v_max, az_gray_pred),
        ('(c) Difference (Actual \u2212 Predicted)', diff_masked, extent_actual,
         diff_cmap, -d_abs, d_abs, az_gray_actual),
    ]
    _make_figure(
        depth_panels,
        f'{title} \u2014 Actual vs Predicted Depth ({start_year}\u2013{end_year} Mean)',
        'Depth (mm)', 'Depth (ft)', _MM_TO_FT,
        out_filename,
        v_min, v_max, -d_abs, d_abs,
    )

    # ---- Volume maps (m³ / AF) ----
    # Convert depth (mm) to volume (m³) per pixel
    # Pixel area from the raster resolution
    with rio.open(os.path.join(predicted_dir, predicted_files[0])) as src:
        px = abs(src.transform.a)
        py = abs(src.transform.e)
    pixel_area_m2 = px * py
    mm_to_m3 = pixel_area_m2 / 1000
    m3_to_af = 1 / _AF_TO_M3

    actual_vol = np.ma.array(actual_masked * mm_to_m3,
                             mask=actual_masked.mask)
    pred_vol = np.ma.array(pred_masked * mm_to_m3,
                           mask=pred_masked.mask)
    diff_vol = np.ma.array(diff_masked * mm_to_m3,
                           mask=diff_masked.mask)

    vol_vals = np.concatenate([actual_vol.compressed(), pred_vol.compressed()])
    vol_min = float(np.nanpercentile(vol_vals, 2))
    vol_max = float(np.nanpercentile(vol_vals, 98))
    dvol_vals = diff_vol.compressed()
    dvol_abs = (max(abs(np.nanpercentile(dvol_vals, 2)),
                    abs(np.nanpercentile(dvol_vals, 98)))
                if len(dvol_vals) > 0 else 1.0)

    vol_panels = [
        ('(a) Actual (Metered)', actual_vol, extent_actual,
         cmap, vol_min, vol_max, az_gray_actual),
        ('(b) Predicted (ML)', pred_vol, extent_pred,
         cmap, vol_min, vol_max, az_gray_pred),
        ('(c) Difference (Actual \u2212 Predicted)', diff_vol, extent_actual,
         diff_cmap, -dvol_abs, dvol_abs, az_gray_actual),
    ]
    vol_filename = out_filename.replace('.png', '_Volume.png')
    _make_figure(
        vol_panels,
        f'{title} \u2014 Actual vs Predicted Volume ({start_year}\u2013{end_year} Mean)',
        r'Volume (m$^3$)', 'Volume (AF)', m3_to_af,
        vol_filename,
        vol_min, vol_max, -dvol_abs, dvol_abs,
    )


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
    units: str = '',
) -> pd.DataFrame:
    """Compute per-zone trend statistics from pixel-wise results.

    Args:
        units: Free-text unit label written into a ``Units`` column on
            every row (e.g. ``'mm/year'``, ``'m^3/year'``,
            ``'fraction/year'``). Lets a downstream reader tell whether
            the slope columns are in depth, volume, or dimensionless
            units without having to parse the filename or ``Category``
            column.

    Returns a DataFrame with one row per zone and columns:
    Region, Units, N_Pixels, Median_Slope, Mean_Slope, Mean_Slope_Sig,
    Pct_Sig_Increase, Pct_Sig_Decrease, Pct_Not_Sig,
    P10_Slope, P90_Slope, Median_P_Value.
    """
    records = []
    for zone_id, zone_name in sorted(id_to_name.items()):
        in_zone = (label_grid == zone_id) & ~all_nan
        n = int(in_zone.sum())
        if n == 0:
            records.append({
                'Region': zone_name,
                'Units': units,
                'N_Pixels': 0,
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
                'Region': zone_name,
                'Units': units,
                'N_Pixels': n,
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
            'Units': units,
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


def _slope_display_scale(max_abs: float) -> tuple[float, str]:
    """Pick a display multiplier for small Sen's slope values.

    Trend maps for capture fraction (~10⁻⁷/year) and capture volume
    (~10⁻³ mm/year/year) have slopes far below the ``:.2f`` precision
    floor used by basin labels and colorbar ticks, so every label
    collapses to ``0.00``.  Withdrawal/CU slopes (~0.1–6 mm/year/year),
    by contrast, are already in a readable range and need no scaling.

    Args:
        max_abs: Largest absolute slope across the data being plotted.
            Used to pick the multiplier; should be the same value that
            sets the colorbar limits so the displayed numbers and
            colorbar are mutually consistent.

    Returns:
        Tuple ``(scale, prefix)`` where ``scale`` is the multiplier to
        apply to slope values for display only (CSVs stay raw) and
        ``prefix`` is the LaTeX-formatted exponent text to insert into
        the colorbar label, e.g. ``r'$\\times 10^{-3}$ '``.
    """
    if max_abs <= 0 or not np.isfinite(max_abs):
        return 1.0, ''
    if max_abs >= 0.5:
        return 1.0, ''
    if max_abs >= 5e-4:
        return 1e3, r'$\times 10^{-3}$ '
    if max_abs >= 5e-7:
        return 1e6, r'$\times 10^{-6}$ '
    return 1e9, r'$\times 10^{-9}$ '


def create_trend_maps(
    raster_dir: str,
    basin_shp: str,
    output_dir: str,
    *,
    title: str = 'Predicted Annual Withdrawal',
    unit_label: str = 'mm',
    secondary_unit_label: str | None = None,
    secondary_unit_factor: float | None = None,
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
        unit_label (str): Primary depth/volume unit (e.g. 'mm', 'm³').
        secondary_unit_label (str or None): Optional secondary unit
            (e.g. 'ft', 'AF') to display on the right side of every
            colorbar via a twinx axis.  If ``None``, the colorbar shows
            only the primary unit.
        secondary_unit_factor (float or None): Multiplicative conversion
            factor from the primary unit to the secondary unit
            (e.g. ``1/304.8`` for mm → ft, ``1/1233.48`` for m³ → AF).
            Required if ``secondary_unit_label`` is set.
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

    # ── Pass 1: compute per-period results and write CSVs ─────────────
    period_results: dict[str, dict] = {}
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

        units_str = f'{unit_label}/year'
        basin_stats = _compute_zonal_trend_stats(
            slope_map, pval_map, sig_map, all_nan,
            basin_labels, basin_id_map, alpha,
            units=units_str,
        )
        basin_stats.insert(0, 'Category', title)
        basin_stats.insert(1, 'Period', period_name)
        basin_csv = os.path.join(output_dir, f'Basin_Trend_{slug}.csv')
        basin_stats.to_csv(basin_csv, index=False)

        if subbasin_labels is not None:
            sub_stats = _compute_zonal_trend_stats(
                slope_map, pval_map, sig_map, all_nan,
                subbasin_labels, subbasin_id_map, alpha,
                units=units_str,
            )
            sub_stats.insert(0, 'Category', title)
            sub_stats.insert(1, 'Period', period_name)
            sub_csv = os.path.join(output_dir, f'Subbasin_Trend_{slug}.csv')
            sub_stats.to_csv(sub_csv, index=False)

        valid_slopes = slope_masked.compressed()
        if len(valid_slopes) == 0:
            continue
        abs_max = max(
            abs(np.nanpercentile(valid_slopes, 2)),
            abs(np.nanpercentile(valid_slopes, 98)),
            1e-6,
        )

        # Significant-pixel summary for inset text
        domain_pixels = int((~all_nan).sum())
        if domain_pixels > 0:
            n_sig_inc = int((sig_map & (slope_map > 0) & ~all_nan).sum())
            n_sig_dec = int((sig_map & (slope_map < 0) & ~all_nan).sum())
            pct_inc = 100 * n_sig_inc / domain_pixels
            pct_dec = 100 * n_sig_dec / domain_pixels
            pct_ns = 100 - pct_inc - pct_dec
        else:
            pct_inc = pct_dec = pct_ns = 0.0

        period_results[period_name] = {
            'slope_map': slope_map,
            'sig_map': sig_map,
            'all_nan': all_nan,
            'slope_masked': slope_masked,
            'abs_max': abs_max,
            'basin_stats': basin_stats,
            'pct_inc': pct_inc,
            'pct_dec': pct_dec,
            'pct_ns': pct_ns,
            'slug': slug,
        }

    if not period_results:
        logger.warning('No valid periods to plot for %s', title)
        return

    # Helper: attach a secondary-unit twin axis to a colorbar so the
    # primary metric unit (e.g. mm, m³) sits on the LEFT side of the
    # colorbar and the secondary unit (e.g. ft, AF) sits on the RIGHT.
    # No-op if secondary_unit_label / secondary_unit_factor are not
    # set.  ``label_fontsize`` should match the primary axis label
    # size so the two sides read at the same weight; ``tick_fontsize``
    # should match the primary tick label size so the two tick rows
    # are visually aligned.
    def _add_secondary_unit_axis(cbar, slope_scale, unit_prefix,
                                 label_fontsize=12, tick_fontsize=10):
        if secondary_unit_label is None or secondary_unit_factor is None:
            return
        # Move the primary axis to the LEFT of the colorbar
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.tick_left()
        # Build the secondary twin axis on the RIGHT
        cb_ax2 = cbar.ax.twinx()
        cb_lo, cb_hi = cbar.ax.get_ylim()
        cb_ax2.set_ylim(cb_lo * secondary_unit_factor,
                        cb_hi * secondary_unit_factor)
        cb_ax2.set_ylabel(
            f"Sen's slope ({unit_prefix}{secondary_unit_label}/year)",
            fontsize=label_fontsize, fontweight='bold',
        )
        cb_ax2.tick_params(labelsize=tick_fontsize)
        cb_ax2.yaxis.set_label_position('right')
        cb_ax2.yaxis.tick_right()

    def _add_secondary_unit_axis_h(cbar, unit_prefix,
                                   label_fontsize=12, tick_fontsize=10):
        """Secondary-unit axis on TOP of a horizontal colorbar (e.g. ft,
        AF on top; primary mm, m³ on the bottom)."""
        if secondary_unit_label is None or secondary_unit_factor is None:
            return
        secax = cbar.ax.secondary_xaxis(
            'top',
            functions=(lambda x: x * secondary_unit_factor,
                       lambda x: x / secondary_unit_factor),
        )
        secax.set_xlabel(
            f"Sen's slope ({unit_prefix}{secondary_unit_label}/year)",
            fontsize=label_fontsize, fontweight='bold',
        )
        secax.tick_params(labelsize=tick_fontsize)

    # ── Rendering helpers ────────────────────────────────────────────
    def _draw_pixel_panel(ax, period_name, pr, color_abs, slope_scale=1.0,
                          show_title=True):
        ax.set_facecolor('#D5D5D5')
        im = ax.imshow(
            pr['slope_masked'] * slope_scale, extent=extent, origin='upper',
            cmap='RdBu_r',
            vmin=-color_abs * slope_scale, vmax=color_abs * slope_scale,
            interpolation='nearest',
        )
        nonsig = ~pr['sig_map'] & ~pr['all_nan']
        if nonsig.any():
            rows, cols = np.where(nonsig)
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
        _overlay_boundaries(ax, basins_gdf, ama_ina, name_col,
                            label_all=True)
        # Panel-level title only when there are multiple panels in the
        # figure (era 1×3 composites).  For single-axis Full-period
        # figures the period name is already carried by the suptitle,
        # so a panel title would be a duplicate.
        if show_title:
            ax.set_title(period_name, fontsize=18, fontweight='bold')
        summary = (f"Significant: \u2191{pr['pct_inc']:.1f}%  "
                   f"\u2193{pr['pct_dec']:.1f}%  n.s. {pr['pct_ns']:.1f}%")
        # Top-left corner (NW, away from the SW-corner legend) so the
        # summary box never overlaps the legend on the last panel.
        ax.text(
            0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=11, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      alpha=0.85, lw=0.5),
        )
        return im

    def _draw_basin_panel(ax, period_name, pr, color_abs,
                          slope_scale=1.0, font_scale=1.0, show_title=True):
        basin_stats = pr['basin_stats']
        basin_trend_gdf = basins_gdf.merge(
            basin_stats[['Region', 'Mean_Slope', 'Pct_Sig_Increase',
                         'Pct_Sig_Decrease']],
            left_on=name_col, right_on='Region', how='left',
        )
        basin_trend_gdf['Mean_Slope'] = basin_trend_gdf[
            'Mean_Slope'].fillna(0)
        # Display-only scaling: multiply slope into a scaled column for
        # both the choropleth fill and the basin labels.  CSV outputs
        # remain in raw units.
        basin_trend_gdf['Mean_Slope_Display'] = (
            basin_trend_gdf['Mean_Slope'] * slope_scale
        )
        basin_trend_gdf.plot(
            column='Mean_Slope_Display', ax=ax, cmap='RdBu_r',
            vmin=-color_abs * slope_scale, vmax=color_abs * slope_scale,
            edgecolor='#333333', linewidth=0.5,
            legend=False, missing_kwds={'color': '#D5D5D5'},
        )
        for _, row in basin_trend_gdf.iterrows():
            if row.geometry is None:
                continue
            centroid = row.geometry.centroid
            bname = row[name_col]
            slope_val = row['Mean_Slope_Display']
            short = bname.replace(' AMA', '').replace(' INA', '')
            arrow = ('\u2191' if slope_val > 0
                     else '\u2193' if slope_val < 0 else '\u2013')
            label = f'{short}\n{arrow}{abs(slope_val):.2f}'
            fontweight = 'bold' if bname in ama_ina else 'normal'
            ax.annotate(
                label, (centroid.x, centroid.y),
                fontsize=4.5 * font_scale, fontweight=fontweight,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                          alpha=0.7, lw=0),
            )
        # Overlay AMA / INA / GW basin boundaries (colored borders only —
        # labels are already added above with the per-basin trend value).
        _overlay_boundaries(
            ax, basins_gdf, ama_ina, name_col,
            label_all=True, show_labels=False,
        )
        if show_title:
            ax.set_title(period_name, fontsize=13, fontweight='bold')
        ax.axis('off')

    # ── Pass 2: render figures ───────────────────────────────────────
    # Identify the "Full" period (single-axis) vs the era periods
    # (combined into a 1×3 figure).  The Full period is the one whose
    # name starts with 'Full' (case-insensitive); everything else is
    # treated as an era.
    full_periods = [p for p in period_results
                    if p.lower().startswith('full')]
    era_periods = [p for p in period_results
                   if not p.lower().startswith('full')]

    # ── Standalone "Full" figures (1×1) ──────────────────────────────
    for period_name in full_periods:
        pr = period_results[period_name]
        pixel_scale, pixel_prefix = _slope_display_scale(pr['abs_max'])
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.8),
                               constrained_layout=True)
        # show_title=False suppresses the per-axis title because the
        # suptitle below already carries the period name.
        im = _draw_pixel_panel(ax, period_name, pr, pr['abs_max'],
                               slope_scale=pixel_scale,
                               show_title=False)
        add_ama_ina_legend(ax, bbox_to_anchor=(0.0, -0.12))
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02,
                            extend='both')
        cbar.set_label(
            f"Sen's slope ({pixel_prefix}{unit_label}/year)",
            fontsize=13, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=12)
        _add_secondary_unit_axis(cbar, pixel_scale, pixel_prefix,
                                 label_fontsize=12, tick_fontsize=10)
        fig.suptitle(
            f'{title} \u2014 Trend {period_name}\n'
            f"(stipple = not significant at \u03b1 = {alpha})",
            fontsize=14, fontweight='bold',
        )
        out_path = os.path.join(output_dir,
                                f"Trend_{pr['slug']}.png")
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

        if (not pr['basin_stats'].empty
                and 'Mean_Slope' in pr['basin_stats'].columns):
            slopes = pr['basin_stats']['Mean_Slope'].dropna().values
            b_abs = (max(abs(slopes.min()), abs(slopes.max()), 1e-6)
                     if len(slopes) > 0 else 1.0)
            basin_scale, basin_prefix = _slope_display_scale(b_abs)
            fig_b, ax_b = plt.subplots(1, 1, figsize=(7.5, 6.8),
                                       constrained_layout=True)
            _draw_basin_panel(ax_b, period_name, pr, b_abs,
                              slope_scale=basin_scale,
                              show_title=False)
            add_ama_ina_legend(ax_b, bbox_to_anchor=(0.0, -0.12))
            sm = plt.cm.ScalarMappable(
                cmap='RdBu_r',
                norm=plt.Normalize(
                    vmin=-b_abs * basin_scale,
                    vmax=b_abs * basin_scale,
                ),
            )
            sm.set_array([])
            cbar_b = fig_b.colorbar(sm, ax=ax_b, shrink=0.6, pad=0.02,
                                    extend='both')
            cbar_b.set_label(
                f"Mean Sen's slope ({basin_prefix}{unit_label}/year)",
                fontsize=12, fontweight='bold',
            )
            cbar_b.ax.tick_params(labelsize=12)
            _add_secondary_unit_axis(cbar_b, basin_scale, basin_prefix,
                                     label_fontsize=11, tick_fontsize=10)
            fig_b.suptitle(
                f'{title} \u2014 Basin Trend {period_name}',
                fontsize=14, fontweight='bold',
            )
            b_out = os.path.join(output_dir,
                                 f"Basin_Trend_{pr['slug']}.png")
            fig_b.savefig(b_out, dpi=600, bbox_inches='tight')
            plt.close(fig_b)

    # ── Combined era figure (1×3) for pixel trend maps ──────────────
    if era_periods:
        # Shared symmetric color limit across all era panels so the
        # single colorbar is comparable.
        eras_abs = max(
            (period_results[p]['abs_max'] for p in era_periods),
            default=1e-6,
        )
        eras_scale, eras_prefix = _slope_display_scale(eras_abs)
        n_eras = len(era_periods)
        fig, axes = plt.subplots(
            n_eras, 1, figsize=(5.2, 4.3 * n_eras),
            constrained_layout=True,
        )
        if n_eras == 1:
            axes = [axes]
        last_im = None
        for ax, period_name in zip(axes, era_periods):
            last_im = _draw_pixel_panel(
                ax, period_name, period_results[period_name], eras_abs,
                slope_scale=eras_scale,
            )
        add_ama_ina_legend(axes[-1], loc='upper center',
        bbox_to_anchor=(0.5, -0.03), ncol=3)
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.92, pad=0.02,
                            aspect=45, extend='both',
                            orientation='horizontal')
        cbar.set_label(
            f"Sen's slope ({eras_prefix}{unit_label}/year)",
            fontsize=12, fontweight='bold',
        )
        cbar.ax.tick_params(labelsize=11)
        _add_secondary_unit_axis_h(cbar, eras_prefix,
                                   label_fontsize=12, tick_fontsize=10)
        fig.suptitle(
            f'{title} \u2014 Trend by Era\n'
            f"(stipple = not significant at \u03b1 = {alpha})",
            fontsize=15, fontweight='bold',
        )
        eras_slug = (f'{title}_Eras'
                     .replace(' ', '_').replace('/', '_'))
        out_path = os.path.join(output_dir, f'Trend_{eras_slug}.png')
        fig.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

        # ── Combined era figure (1×3) for basin choropleth ──────────
        # Shared symmetric color limit across the 3 basin choropleths.
        all_basin_slopes = []
        for p in era_periods:
            bs = period_results[p]['basin_stats']
            if not bs.empty and 'Mean_Slope' in bs.columns:
                all_basin_slopes.append(
                    bs['Mean_Slope'].dropna().values,
                )
        if all_basin_slopes:
            cat_slopes = np.concatenate(all_basin_slopes)
            b_abs_eras = (max(abs(cat_slopes.min()),
                              abs(cat_slopes.max()), 1e-6)
                          if len(cat_slopes) > 0 else 1.0)
            basin_eras_scale, basin_eras_prefix = _slope_display_scale(
                b_abs_eras,
            )
            fig_b, axes_b = plt.subplots(
                n_eras, 1, figsize=(5.2, 4.3 * n_eras),
                constrained_layout=True,
            )
            if n_eras == 1:
                axes_b = [axes_b]
            for ax_b, period_name in zip(axes_b, era_periods):
                _draw_basin_panel(
                    ax_b, period_name, period_results[period_name],
                    b_abs_eras, slope_scale=basin_eras_scale,
                )
            add_ama_ina_legend(axes_b[-1], loc='upper center',
        bbox_to_anchor=(0.5, -0.03), ncol=3)
            sm = plt.cm.ScalarMappable(
                cmap='RdBu_r',
                norm=plt.Normalize(
                    vmin=-b_abs_eras * basin_eras_scale,
                    vmax=b_abs_eras * basin_eras_scale,
                ),
            )
            sm.set_array([])
            cbar_b = fig_b.colorbar(
                sm, ax=axes_b, shrink=0.92, pad=0.02,
                aspect=45, extend='both', orientation='horizontal',
            )
            cbar_b.set_label(
                f"Mean Sen's slope ({basin_eras_prefix}{unit_label}/year)",
                fontsize=12, fontweight='bold',
            )
            cbar_b.ax.tick_params(labelsize=11)
            _add_secondary_unit_axis_h(
                cbar_b, basin_eras_prefix,
                label_fontsize=11, tick_fontsize=10,
            )
            fig_b.suptitle(
                f'{title} \u2014 Basin Trend by Era',
                fontsize=22, fontweight='bold',
            )
            b_out = os.path.join(output_dir,
                                 f'Basin_Trend_{eras_slug}.png')
            fig_b.savefig(b_out, dpi=600, bbox_inches='tight')
            plt.close(fig_b)

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
    basin_label_overrides: dict[str, dict[str, str]] | None = None,
    basin_skip_sources: dict[str, set[str]] | None = None,
    basin_footnotes: dict[str, str] | None = None,
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
        colors (dict[str, str]): Per-source colors.
        markers (dict[str, str]): Per-source markers.
        labels (dict or None): Display labels per source.
        title_prefix (str): Prepended to plot titles.
        file_prefix (str): Prepended to filenames.
        mode (str): ``'volume'`` → 2-row (depth + volume).
            ``'ratio'`` → 1-row (dimensionless).
        af_to_m3 (float): Conversion factor AF to m³.
        m_to_mm (float): Conversion factor meters to mm.
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
            # Single-row layout for both volume and ratio modes.
            # Volume mode shows m³ (left primary) + AF (right twin)
            # only — the depth (mm) panel was removed because NHM /
            # Reitz / ML use different irrigated-area definitions, so
            # depth comparisons are not directly comparable.  Volume
            # space cancels the area mismatch and is the appropriate
            # axis for cross-product intercomparison.
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

            # Per-basin "skip this source" set (e.g. SRP-augmented
            # composite shouldn't appear at non-Phoenix basins where
            # the underlying value is CAP-only anyway — only the label
            # would be misleading).
            skip_set = (
                basin_skip_sources.get(basin, set())
                if basin_skip_sources else set()
            )

            has_any_data = False
            for source, src_data in all_sources.items():
                if source in skip_set:
                    continue
                yearly = src_data.get(cat, {}).get('yearly', {})
                # Filter to numeric year keys only — some data structures
                # (e.g. CAP/SRP) use {basin: {year: val}} and pass basin
                # names as the yearly dict keys at the wrong nesting level.
                numeric_keys = []
                for k in yearly.keys():
                    try:
                        numeric_keys.append(int(k))
                    except (ValueError, TypeError):
                        continue
                years = sorted(numeric_keys)
                if not years:
                    continue

                color = colors.get(source, '#333333')
                marker = markers.get(source, 'o')
                # Per-basin label override (e.g. relabel "CAP + SRP" as
                # "CAP" at non-Phoenix basins where SRP is not added).
                label = (
                    (basin_label_overrides or {})
                    .get(basin, {})
                    .get(source, labels.get(source, source))
                )

                # Build a lookup that handles both int and float year keys,
                # filtering to numeric keys only.
                yearly_int = {}
                for k, v in yearly.items():
                    try:
                        yearly_int[int(k)] = v
                    except (ValueError, TypeError):
                        continue
                if mode == 'volume':
                    if basin == 'AZ_Total':
                        # Sum across only the basins in ``basin_names``
                        # (the comparison set), not every basin present
                        # in ``yearly_int``.  Important when one source
                        # (e.g. ML) contains all AZ basins while the
                        # comparison source (e.g. CAP+SRP) only covers
                        # a subset — without this restriction the AZ_Total
                        # ML side double-counts basins outside the
                        # comparison footprint.
                        af_vals = np.array([
                            sum(yearly_int[yr].get(b, 0.0)
                                for b in basin_names
                                if np.isfinite(yearly_int[yr].get(b, np.nan)))
                            for yr in years
                        ])
                    else:
                        # Use NaN for missing basins so gaps appear
                        # instead of false zeros (e.g. NHM before 2000)
                        af_vals = np.array([
                            yearly_int[yr].get(basin, np.nan) for yr in years
                        ])

                    m3_vals = af_vals * af_to_m3

                    if np.any(np.isfinite(af_vals) & (af_vals != 0)):
                        has_any_data = True
                    # Plot m³ only (single volume panel); AF shown as
                    # twin right axis via _format_volume_axis below.
                    axes[0].plot(years, m3_vals, label=label, color=color,
                                 marker=marker, markersize=3, linewidth=1.2)
                    # 95 % CI band when source has 'yearly_sigma' (1σ)
                    sigma_yearly = src_data.get(cat, {}).get(
                        'yearly_sigma',
                    )
                    if sigma_yearly:
                        sigma_int = {}
                        for k, v in sigma_yearly.items():
                            try:
                                sigma_int[int(k)] = v
                            except (ValueError, TypeError):
                                continue
                        if basin == 'AZ_Total':
                            # Spatial quadrature across the basin set
                            sig_af = np.array([
                                np.sqrt(sum(
                                    (sigma_int.get(yr, {}).get(b, 0.0)) ** 2
                                    for b in basin_names
                                ))
                                for yr in years
                            ])
                        else:
                            sig_af = np.array([
                                sigma_int.get(yr, {}).get(basin, np.nan)
                                for yr in years
                            ])
                        sig_m3 = sig_af * af_to_m3
                        finite = np.isfinite(sig_m3) & (sig_m3 > 0)
                        if finite.any():
                            ci = 1.96 * sig_m3
                            lo = m3_vals - ci
                            hi = m3_vals + ci
                            axes[0].fill_between(
                                years, lo, hi,
                                where=finite,
                                color=color, alpha=0.18,
                                linewidth=0,
                                label=f'{label} 95 % CI',
                            )
                else:
                    # Ratio mode (e.g. IE)
                    if basin == 'AZ_Total':
                        ie_vals = []
                        for yr in years:
                            yr_d = yearly_int[yr]
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
                            yearly_int[yr].get(basin, np.nan) for yr in years
                        ])
                    if np.any(np.isfinite(plot_vals)):
                        has_any_data = True
                    axes[0].plot(years, plot_vals, label=label, color=color,
                                 marker=marker, markersize=3, linewidth=1.2)

            # Skip blank plots (no source had data for this basin/category)
            if not has_any_data:
                plt.close(fig)
                continue

            # ── Axis formatting ──
            # Force integer-only year ticks on every x-axis to prevent
            # matplotlib from rendering "2000.0" or cluttering the axis
            # with sub-year gridlines.  Tick labelsize fixed at 11 and
            # axis labels bold across mm/ft/m³/MAF panels for visual
            # consistency with the volume-axis formatter.
            from matplotlib.ticker import MaxNLocator
            tick_fontsize = 11
            if mode == 'volume':
                _format_volume_axis(axes[0], unit='m3', label='Volume')
                axes[0].set_xlabel('Year', fontweight='bold')
                axes[0].grid(True, alpha=0.3, linestyle='--')
                axes[0].legend(fontsize=9)
                axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[0].tick_params(axis='both', labelsize=tick_fontsize)
                ax_af = axes[0].twinx()
                lo, hi = axes[0].get_ylim()
                ax_af.set_ylim(lo * m3_to_af, hi * m3_to_af)
                _format_volume_axis(ax_af, unit='AF', label='Volume')
                ax_af.tick_params(axis='both', labelsize=tick_fontsize)
            else:
                axes[0].set_ylabel(cat_title, fontweight='bold')
                axes[0].set_xlabel('Year', fontweight='bold')
                axes[0].grid(True, alpha=0.3, linestyle='--')
                axes[0].legend(fontsize=9)
                axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                axes[0].tick_params(axis='both', labelsize=tick_fontsize)

            # Optional per-basin footnote (e.g. data-coverage caveat
            # like "CAP+SRP at 2024 is CAP-only — SRP records end 2023"
            # for Phoenix AMA).  Rendered as italic text below the plot.
            footnote = (basin_footnotes or {}).get(basin)
            if footnote:
                fig.text(
                    0.5, -0.02, footnote,
                    ha='center', va='top',
                    fontsize=9, style='italic', color='#444444',
                )

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
    is_validation: bool = False,
    af_divisor: float = 1.0,
    af_unit_label: str = 'AF',
    annotate_basins: bool = False,
    log_scale: bool = False,
    one_column: bool = False,
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
        m_to_mm (float): Conversion factor meters to mm.

    Returns:
        None.
    """
    makedirs(output_dir)
    n_pairs = len(pairs)
    # Single-row layout for both volume and ratio modes.  Volume mode
    # plots m³ data (left primary) with AF on the right twin axis —
    # the depth (mm) row was removed because NHM / Reitz / ML use
    # different irrigated-area definitions, so depth comparisons are
    # not directly comparable.  Volume space cancels the area
    # mismatch and is the appropriate axis for cross-product
    # intercomparison.
    n_units = 1
    # Panel figsize chosen to be roughly square — the inner data box
    # has to be square for set_aspect('equal', adjustable='datalim')
    # to leave xlim/ylim alone.  When the box is wider than tall
    # matplotlib stretches the data limits to satisfy equal aspect,
    # which overrides our set_xlim/set_ylim and emits the
    # "Ignoring fixed y limits to fulfill fixed data aspect" warning.
    if one_column:
        # Stack the pairs vertically: n_pairs rows × 1 column.
        fig, axes = plt.subplots(
            n_pairs, n_units, figsize=(6 * n_units, 6 * n_pairs),
            constrained_layout=True, squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            n_units, n_pairs, figsize=(6 * n_pairs, 6),
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
            # Plot in m³ on the primary axis; AF added as a right twin
            # later.  ``vals_a``/``vals_b`` are in AF; convert to m³.
            m3_a = vals_a * af_to_m3
            m3_b = vals_b * af_to_m3
            row_data = [(m3_a, m3_b, 'm³')]
        else:
            row_data = [(vals_a, vals_b, '')]

        for row_i, (vx, vy, unit) in enumerate(row_data):
            ax = axes[col_i, row_i] if one_column else axes[row_i, col_i]
            if vx.size == 0:
                ax.set_title(f'{label_a} vs {label_b}')
                continue

            scatter_size = 60 if annotate_basins else 30
            ax.scatter(vx, vy, s=scatter_size, alpha=0.75,
                       edgecolors='white', linewidths=0.5, zorder=3)

            if log_scale:
                # Log axes need strictly positive bounds; use the smallest
                # positive value as the floor (1 decade below) and the
                # largest as the ceiling (1 decade above).
                pos = np.concatenate([vx[vx > 0], vy[vy > 0]])
                lo = pos.min() / 3.0 if pos.size else 1.0
                hi = pos.max() * 3.0 if pos.size else 10.0
            else:
                lo = min(vx.min(), vy.min(), 0) if mode == 'volume' else min(vx.min(), vy.min()) * 0.9
                hi = max(vx.max(), vy.max()) * 1.05
            if hi <= lo:
                hi = lo + 1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')

            if annotate_basins:
                for bn, bx, by in zip(basin_names, vx, vy):
                    if not (np.isfinite(bx) and np.isfinite(by)):
                        continue
                    if log_scale and (bx <= 0 or by <= 0):
                        continue
                    ax.annotate(
                        bn.replace(' AMA', '').replace(' INA', '')
                          .replace(' PLAIN', '').title(),
                        xy=(bx, by), xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9, color='#222',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none'),
                    )

            if len(vx) > 1 and np.std(vx) > 0:
                from scipy.stats import pearsonr
                from sklearn.metrics import r2_score as _r2_score
                from hydrolibs.mlops import (
                    normalized_rmse, normalized_mae, normalized_mbe,
                )
                z = np.polyfit(vx, vy, 1)
                # Skip the linear fit line on log axes — a linear y = a·x+b
                # with negative intercept renders as a near-vertical drop
                # near the zero-crossing on log-scale axes, which is
                # visually confusing.  The fit equation in the legend
                # plus the metrics box still convey the regression.
                # Intercept reported in the SAME unit as the displayed
                # axis: km³ (= z[1] / 1e9) for volume mode, raw value
                # for ratio mode.  Without this conversion the equation
                # showed huge m³ values (e.g. "5,000,000.0") next to
                # km³ axis ticks (e.g. "0.005"), which was confusing.
                if not log_scale:
                    x_fit = np.linspace(lo, hi, 100)
                    if mode == 'volume':
                        intercept_disp = z[1] / 1e9
                        intercept_unit = ' km³'
                    else:
                        intercept_disp = z[1]
                        intercept_unit = ''
                    sign = '\u2212' if intercept_disp < 0 else '+'
                    ax.plot(
                        x_fit, np.polyval(z, x_fit), 'r-', lw=1.2,
                        label=(
                            f'y={z[0]:.2f}x {sign} '
                            f'{abs(intercept_disp):.3g}{intercept_unit}'
                        ),
                    )
                # Validation (is_validation=True) vs observed data:
                # R² (Nash-Sutcliffe form) is appropriate because a
                # single ground-truth reference exists and we want to
                # know how much of the observed variance the model
                # explains — including bias.
                # Intercomparison (is_validation=False) between
                # competing model estimates: Pearson's r isolates the
                # linear association without penalising a constant
                # bias, which is what we want when no source is a
                # "truth" reference.
                rmse_pct = normalized_rmse(vy, vx)
                mae_pct = normalized_mae(vy, vx)
                mbe_pct = normalized_mbe(vy, vx)
                mbe_sign = '\u2212' if mbe_pct < 0 else ''
                if is_validation:
                    r2 = _r2_score(vy, vx)
                    fit_label = 'R²'
                    fit_val = r2
                    e_labels = ('RMSE', 'MAE', 'MBE')
                else:
                    r_val, _ = pearsonr(vx, vy)
                    fit_label = 'r'
                    fit_val = r_val
                    e_labels = ('RMSD', 'MAD', 'MBD')
                metrics_text = (f'{fit_label}={fit_val:.3f}\n'
                                f'{e_labels[0]}={rmse_pct:.1f}%\n'
                                f'{e_labels[1]}={mae_pct:.1f}%\n'
                                f'{e_labels[2]}={mbe_sign}{abs(mbe_pct):.1f}%')
                ax.text(0.97, 0.03, metrics_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='gray'))
            # Per-axes title omitted — fig.suptitle already conveys
            # the comparison label.

            unit_suffix = f' ({unit})' if unit else ''
            scale_suffix = ', log scale' if log_scale else ''
            ax.set_xlabel(
                f'{label_a}{unit_suffix.rstrip(")")}{scale_suffix}{")" if unit_suffix else ""}',
                fontsize=12, fontweight='bold',
            )
            ax.set_ylabel(
                f'{label_b}{unit_suffix.rstrip(")")}{scale_suffix}{")" if unit_suffix else ""}',
                fontsize=12, fontweight='bold',
            )
            ax.tick_params(axis='both', labelsize=11)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--', which='both')
            # Order matters: set xlim/ylim → create twin → set twin
            # ylim → THEN set aspect.  set_aspect('equal',
            # adjustable='box') reapplies its constraint whenever the
            # bounding box would change, and the twin axis shares the
            # parent's bbox — so any set_ylim on the twin AFTER the
            # parent's aspect is locked emits the "Ignoring fixed y
            # limits to fulfill fixed data aspect" warning.  Doing
            # the twin setup first avoids that.
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            # Thin decade tick labels on log axes: an 8+-decade span
            # (e.g. PS volumes) crowds the x-axis with overlapping
            # labels, so show every Nth decade on both axes (and the
            # AF/MAF twin below) once the span is wide.
            _log_stride = 1
            if log_scale and hi > 0 and lo > 0:
                import math as _math
                from matplotlib import ticker as _tkx
                _dec = _math.log10(hi / lo)
                _log_stride = 1 if _dec <= 6 else (2 if _dec <= 12 else 3)
                if _log_stride > 1:
                    ax.xaxis.set_major_locator(
                        _tkx.LogLocator(base=10.0 ** _log_stride, numticks=20)
                    )
                    ax.yaxis.set_major_locator(
                        _tkx.LogLocator(base=10.0 ** _log_stride, numticks=20)
                    )
            # For volume mode, format primary axes as km³ and add an
            # AF twin on the right Y only so the same scatter can be
            # read in both unit systems.  Twiny on the top axis is
            # avoided because ``set_aspect('equal', adjustable='box')``
            # is incompatible with both axes shared simultaneously
            # (matplotlib RuntimeError).  AF reading on X requires
            # mental conversion (or read directly off Y twin since the
            # 1:1 line crosses both axes at the same volumes).
            if mode == 'volume' and unit == 'm³':
                from matplotlib import ticker as _tk

                # Adaptive formatter: small log-scale ticks (e.g.
                # 0.001 km³) collapsed to "0.00" with the previous
                # ``:,.2f`` format, hiding the value.  Switch to 3
                # significant figures for sub-unit values so they
                # render as "0.001" instead of "0.00".
                def _adaptive_fmt(divisor):
                    def _f(x, _):
                        if x == 0:
                            return '0'
                        v = x / divisor
                        a = abs(v)
                        if a >= 1:
                            return f'{v:,.2f}'
                        return f'{v:.3g}'
                    return _f

                ax.xaxis.set_major_formatter(
                    _tk.FuncFormatter(_adaptive_fmt(1e9)),
                )
                ax.yaxis.set_major_formatter(
                    _tk.FuncFormatter(_adaptive_fmt(1e9)),
                )
                _kmcubed_suffix = '(km³, log scale)' if log_scale else '(km³)'
                ax.set_xlabel(
                    f'{label_a} {_kmcubed_suffix}',
                    fontsize=12, fontweight='bold',
                )
                ax.set_ylabel(
                    f'{label_b} {_kmcubed_suffix}',
                    fontsize=12, fontweight='bold',
                )
                ax_right = ax.twinx()
                if log_scale:
                    ax_right.set_yscale('log')
                ax_right.set_ylim(lo / af_to_m3, hi / af_to_m3)
                ax_right.yaxis.set_major_formatter(
                    _tk.FuncFormatter(_adaptive_fmt(1e6)),
                )
                if log_scale and _log_stride > 1:
                    ax_right.yaxis.set_major_locator(
                        _tk.LogLocator(base=10.0 ** _log_stride, numticks=20)
                    )
                _maf_suffix = '(MAF, log scale)' if log_scale else '(MAF)'
                ax_right.set_ylabel(
                    f'{label_b} {_maf_suffix}',
                    fontsize=12, fontweight='bold',
                )
                ax_right.tick_params(axis='y', labelsize=11)
            # Intentionally NOT calling set_aspect('equal') —
            # constrained_layout + suptitle + twinx label shrink the
            # inner data box asymmetrically, so any equal-aspect
            # constraint forces matplotlib to either:
            #   - stretch data limits (datalim mode → emits the
            #     "Ignoring fixed y limits to fulfill fixed data
            #     aspect with adjustable data limits" warning), or
            #   - resize the box (box mode → RuntimeError on
            #     twinned axes).
            # Without equal aspect the 1:1 line still goes corner to
            # corner of the (lo, hi) × (lo, hi) data range and the
            # interpretation is identical; the visual angle is just
            # not exactly 45°.

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Scatter plot saved to {out_path}')


def plot_intercomp_stacked_bars(
    all_sources: dict[str, dict],
    source_order: list[str],
    output_dir: str,
    *,
    stack_cats: list[str] = ('GW', 'SW'),
    stack_labels: dict[str, str] | None = None,
    stack_colors: dict[str, str] | None = None,
    title_prefix: str = '',
    file_prefix: str = 'Stacked_Bar',
    af_divisor: float = 1e6,
    af_unit_label: str = 'MAF',
    mean_year_range: tuple[int, int] | None = None,
    source_colors: dict[str, str] | None = None,
) -> None:
    """Statewide stacked bar plots for intercomparison sources.

    Produces two figures:

    1. **Per-year grouped bars** — for each common year, one bar per
       source side by side, stacked by ``stack_cats`` (e.g. GW + SW).
    2. **Mean-annual summary** — one bar per source showing the
       mean-annual statewide total, stacked by category.

    Args:
        all_sources: ``{source: {cat: {'yearly': {year: {basin: AF}}}}}``
        source_order: Ordered list of source keys to plot.
        output_dir: Directory for saved plots.
        stack_cats: Categories to stack within each bar.
        stack_labels: Display labels per category. Defaults to cat names.
        stack_colors: Colors per category.
        title_prefix: Prepended to figure titles.
        file_prefix: Prepended to filenames.
        af_divisor: Divide AF values by this for axis labels.
        af_unit_label: Unit label for y-axis (e.g. 'MAF').
        mean_year_range: Optional ``(start, end)`` (inclusive) restricting
            the **mean-annual summary bar** (Figure 2) to a common
            period across all sources, so different-coverage datasets
            (e.g. ML 2000-2020 vs Reitz 1980-2018) are averaged
            apples-to-apples.  If ``None`` (default), each source's
            mean uses its own native year range.  Per-year grouped
            bars (Figure 1) always show every available year.
        source_colors: Optional ``{source: color}`` mapping that
            overrides both the per-source palette (``_SOURCE_PALETTES``)
            and the per-category ``stack_colors`` for the per-year
            grouped bars.  Useful for single-category stacks where
            distinct source-level colours are more legible than the
            cat-color + alpha-shift fallback (e.g. CU intercomparison
            with one bar per source, no GW/SW split).
    """
    makedirs(output_dir)
    if stack_labels is None:
        stack_labels = {c: c for c in stack_cats}
    if stack_colors is None:
        default_colors = ['#2C3E50', '#3498DB', '#E67E22', '#27AE60', '#E74C3C']
        stack_colors = {
            c: default_colors[i % len(default_colors)]
            for i, c in enumerate(stack_cats)
        }

    # Compute per-year statewide totals per source per category
    all_years: set[int] = set()
    src_cat_yearly: dict[str, dict[str, dict[int, float]]] = {}
    for src in source_order:
        src_cat_yearly[src] = {}
        for cat in stack_cats:
            yearly = all_sources.get(src, {}).get(cat, {}).get('yearly', {})
            cat_yearly: dict[int, float] = {}
            for yr, basins in yearly.items():
                yr_int = int(yr)
                cat_yearly[yr_int] = sum(basins.values())
                all_years.add(yr_int)
            src_cat_yearly[src][cat] = cat_yearly

    common_years = sorted(all_years)
    if not common_years:
        logger.warning('No common years for stacked bar plot')
        return

    n_sources = len(source_order)

    # ── Per-source color palette for the per-year grouped bars ──
    # Annual plot shows 3 sources × 2 categories side-by-side at every
    # year — using a single colour per category with alpha-shifts per
    # source makes the bars visually indistinguishable.  Use a 3-hue ×
    # 2-lightness palette (one hue family per source, dark = first cat,
    # light = second cat) so all source × cat combinations are clearly
    # separable.  Falls back to ``stack_colors`` (cat-only) for sources
    # beyond 3 or when stack_cats != 2.
    _SOURCE_PALETTES = {
        'ML':    ('#1B2631', '#5DADE2'),  # dark navy / light blue
        'NHM':   ('#1B4F2E', '#58D68D'),  # dark green / light green
        'Reitz': ('#7C3F00', '#F5B041'),  # dark orange / light orange
        'PS':    ('#641E16', '#EC7063'),  # dark red / light red
        'CAP_SRP': ('#4A235A', '#A569BD'), # dark purple / light purple
    }

    def _bar_color(src: str, cat_idx: int) -> tuple[str, float]:
        """Return (color, alpha) for a (source, category-index) bar."""
        if source_colors and src in source_colors:
            return source_colors[src], 1.0
        if src in _SOURCE_PALETTES and len(stack_cats) == 2:
            return _SOURCE_PALETTES[src][cat_idx], 1.0
        # Fallback to the legacy cat-color + alpha-shift scheme.
        return stack_colors[stack_cats[cat_idx]], 0.7 + 0.15 * source_order.index(src)

    # ── Figure 1: Per-year grouped stacked bars ──────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(common_years) * 0.6), 6),
                            constrained_layout=True)
    bar_width = 0.8 / n_sources
    x = np.arange(len(common_years))

    for si, src in enumerate(source_order):
        bottoms = np.zeros(len(common_years))
        for ci, cat in enumerate(stack_cats):
            vals = np.array([
                src_cat_yearly[src][cat].get(yr, 0.0) / af_divisor
                for yr in common_years
            ])
            color, alpha = _bar_color(src, ci)
            ax.bar(
                x + si * bar_width, vals, bar_width,
                bottom=bottoms,
                color=color,
                edgecolor='white', linewidth=0.3,
                alpha=alpha,
            )
            bottoms += vals

    ax.set_xticks(x + bar_width * (n_sources - 1) / 2)
    from matplotlib.ticker import MaxNLocator
    # Show every Nth year label to avoid clutter
    step = max(1, len(common_years) // 15)
    labels = [str(yr) if i % step == 0 else '' for i, yr in enumerate(common_years)]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'Statewide Total ({af_unit_label})', fontweight='bold')
    ax.set_title(f'{title_prefix}Statewide Annual Totals by Source',
                  fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # Build legend: one entry per source × category, ordered source-major
    # so each source's pair of colours appears together in the legend.
    handles = []
    for src in source_order:
        for ci, cat in enumerate(stack_cats):
            color, alpha = _bar_color(src, ci)
            handles.append(mpatches.Patch(
                facecolor=color,
                alpha=alpha,
                edgecolor='white',
                label=f'{src} — {stack_labels[cat]}',
            ))
    ax.legend(handles=handles, fontsize=8, loc='upper left', framealpha=0.7,
               ncol=n_sources)

    fig.savefig(
        os.path.join(output_dir, f'{file_prefix}_Annual.png'),
        dpi=300, bbox_inches='tight',
    )
    plt.close(fig)

    # ── Figure 2: Mean-annual summary bars ───────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_sources * 2.5), 5),
                            constrained_layout=True)
    x = np.arange(n_sources)
    bar_width = 0.6

    for src_i, src in enumerate(source_order):
        bottom = 0.0
        for cat in stack_cats:
            yr_to_val = src_cat_yearly[src][cat]
            if mean_year_range is not None:
                cat_vals = [
                    v for y, v in yr_to_val.items()
                    if mean_year_range[0] <= y <= mean_year_range[1]
                ]
            else:
                cat_vals = list(yr_to_val.values())
            mean_val = float(np.mean(cat_vals)) / af_divisor if cat_vals else 0.0
            # Per-source override takes precedence (single-cat case);
            # otherwise stack colour by category.
            if source_colors and src in source_colors:
                bar_color = source_colors[src]
            else:
                bar_color = stack_colors[cat]
            ax.bar(
                x[src_i], mean_val, bar_width,
                bottom=bottom,
                color=bar_color,
                edgecolor='black', linewidth=0.5,
                label=stack_labels[cat] if src_i == 0 else '',
            )
            bottom += mean_val
        # Label total on top
        ax.text(x[src_i], bottom + 0.01, f'{bottom:.2f}',
                 ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(source_order, fontsize=10)
    ax.set_ylabel(f'Mean Annual Total ({af_unit_label})', fontweight='bold')
    title_yr = (
        f' ({mean_year_range[0]}–{mean_year_range[1]})'
        if mean_year_range is not None else ''
    )
    ax.set_title(f'{title_prefix}Mean-Annual Statewide Totals{title_yr}',
                  fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    fig.savefig(
        os.path.join(output_dir, f'{file_prefix}_Mean.png'),
        dpi=300, bbox_inches='tight',
    )
    plt.close(fig)
    logger.info(f'Stacked bar plots saved to {output_dir}')


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
            ax.plot(theta_arc, r_arc, 'gray', lw=0.5, alpha=0.4)
            valid = [(t, r) for t, r in zip(theta_arc, r_arc) if np.isfinite(r)]
            if valid:
                t_mid, r_mid = valid[len(valid) // 2]
                ax.annotate(f'{rmsd_val:.2f}', (t_mid, r_mid), fontsize=6,
                            color='gray', alpha=0.7)

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
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
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
            back to gray if missing.

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

            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

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
