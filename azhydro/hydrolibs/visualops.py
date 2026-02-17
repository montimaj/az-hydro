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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import multiprocessing

from typing import Any
from joblib import Parallel, delayed
from sysops import makedirs
from scipy import stats


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


def aggregate_yearly_data(
        df: pd.DataFrame,
        year_col: str,
        value_col: str,
        aggregation: str = 'sum',
        confidence: float = 0.95
) -> pd.DataFrame:
    """
    Aggregate data by year with confidence intervals.
    
    Args:
        df: Input dataframe.
        year_col: Name of year column.
        value_col: Name of value column to aggregate.
        aggregation: Aggregation method ('sum', 'mean').
        confidence: Confidence level for CI.
        
    Returns:
        DataFrame with year, mean, lower_ci, upper_ci columns.
    """
    years = sorted(df[year_col].unique())
    results = []
    
    for year in years:
        year_data = df[df[year_col] == year][value_col].values
        
        if aggregation == 'sum':
            # For sum, we compute total and bootstrap CI
            total = np.sum(year_data)
            # Bootstrap for CI of sum
            n_bootstrap = 1000
            bootstrap_sums = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(year_data, size=len(year_data), replace=True)
                bootstrap_sums.append(np.sum(sample))
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
        units = ['mm', 'af', 'm3']
    
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
        
        # Aggregate by year
        actual_agg = aggregate_yearly_data(
            df_plot, year_col, f'Actual_{unit}', aggregation, confidence
        )
        pred_agg = aggregate_yearly_data(
            df_plot, year_col, f'Pred_{unit}', aggregation, confidence
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
        fig.savefig(f'{output_dir}TS_{model_name}_{test_case}_{unit}_{aggregation}.png',
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
        print(f'No data for basin: {basin_name}')
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
    
    years = actual_agg['year'].values
    min_year, max_year = years.min(), years.max()
    
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
    fig.savefig(f'{output_dir}TS_{model_name}_{test_case}_{basin_clean}_{unit}.png',
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
    basin_dir = f'{output_dir}Basins/'
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
    actual_agg = aggregate_yearly_data(first_df, year_col, f'Actual_{unit}', 'sum', 0.95)
    
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
        pred_agg = aggregate_yearly_data(df, year_col, f'Pred_{unit}', 'sum', 0.95)
        
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
    fig.savefig(f'{output_dir}TS_Model_Comparison_{test_case}_{unit}.png',
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
    
    fig.savefig(f'{output_dir}Scatter_{model_name}_{test_case}.png',
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
    
    fig.savefig(f'{output_dir}Residuals_{model_name}_{test_case}.png',
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
    print(f'\nCreating visualizations for {model_name} - {test_case}...')
    makedirs(output_dir)
    
    # 1. Aggregate time series plot
    print('  Creating time series plots...')
    create_time_series_plot_journal(
        pred_df, output_dir, model_name, test_case, test_year_limits,
        year_col=year_col, actual_col=actual_col, pred_col=pred_col,
        gw_basin_col=gw_basin_col, raster_res=raster_res,
        use_ama_ina=use_ama_ina, units=['af', 'm3', 'mm']
    )
    
    # 2. Scatter plots
    print('  Creating scatter plots...')
    create_train_test_scatter(
        pred_df, output_dir, model_name, test_case,
        actual_col=actual_col, pred_col=pred_col
    )
    
    # 3. Residual plots
    print('  Creating residual plots...')
    create_residual_plot(
        pred_df, output_dir, model_name, test_case,
        actual_col=actual_col, pred_col=pred_col, year_col=year_col
    )
    
    # 4. Basin-level plots (parallel)
    if create_basin_plots:
        print('  Creating basin-level plots...')
        create_all_basin_plots_parallel(
            pred_df, output_dir, model_name, test_case, test_year_limits,
            gw_basin_col=gw_basin_col, raster_res=raster_res, n_jobs=n_jobs
        )
    
    print(f'  Visualizations saved to: {output_dir}')


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
    ama_ina_basins = get_ama_ina_basin_names()
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
    print('Creating time series plots...')
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
