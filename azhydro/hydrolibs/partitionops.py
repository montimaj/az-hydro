"""
Partitioning operations for water-budget decomposition.

Splits total pumping predictions into irrigation / non-irrigation and
groundwater / surface-water components.

Categories produced by ``partition_predictions``:
    Irrigation, Non_Irrigation,
    Irrigation_GW, Irrigation_SW,
    Non_Irrigation_GW, Non_Irrigation_SW,
    Total_GW, Total_SW
"""

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

# Category keys (also used for raster sub-folder names)
CATEGORIES = (
    'Irrigation',
    'Non_Irrigation',
    'Irrigation_GW',
    'Irrigation_SW',
    'Non_Irrigation_GW',
    'Non_Irrigation_SW',
    'Total_GW',
    'Total_SW',
)


def focal_fill_irr_fraction(
        irr_frac: np.ndarray,
        well_dens: np.ndarray,
        raster_shape: tuple,
        valid_mask: np.ndarray,
        kernel_size: int = 5,
        min_irr_frac: float = 0.05,
) -> np.ndarray:
    """
    Gap-fill irrigation fraction for pixels that have wells but a
    negligibly small irrigated area (irr_fraction < *min_irr_frac*).
    These are typically edge pixels where a field barely overlaps the
    2 km cell.  Replace with the focal mean of neighbours whose
    irr_fraction >= *min_irr_frac* within a *kernel_size* × *kernel_size*
    window.

    If the focal neighbourhood also has no substantial irrigated pixels,
    the value stays unchanged (genuinely non-irrigation well).
    """
    filled = irr_frac.copy()
    needs_fill = (well_dens > 0) & ~np.isnan(well_dens) & (irr_frac < min_irr_frac)
    if not np.any(needs_fill):
        return filled

    irr_grid = np.full(valid_mask.shape[0], np.nan, dtype=np.float64)
    irr_grid[valid_mask] = irr_frac
    irr_grid = irr_grid.reshape(raster_shape)

    # Only count neighbours with substantial irrigation for the focal mean
    substantial = irr_grid.copy()
    substantial[np.isnan(substantial)] = 0
    substantial[substantial < min_irr_frac] = 0
    indicator = (substantial > 0).astype(np.float64)

    sum_grid = uniform_filter(substantial, size=kernel_size, mode='constant', cval=0)
    cnt_grid = uniform_filter(indicator, size=kernel_size, mode='constant', cval=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        focal_mean = np.where(cnt_grid > 0, sum_grid / cnt_grid, 0)

    focal_flat = focal_mean.ravel()[valid_mask]
    filled[needs_fill] = np.clip(focal_flat[needs_fill], 0, 1)
    return filled


def compute_sw_fraction(
        canal_dens: np.ndarray,
        raster_shape: tuple,
        valid_mask: np.ndarray,
        kernel_size: int = 5,
) -> np.ndarray:
    """
    Estimate the surface-water share of non-irrigation withdrawals
    using canal density as a proxy.

    The fraction is the pixel's canal density divided by the local
    maximum within a *kernel_size* × *kernel_size* neighbourhood,
    clipped to [0, 1].  Pixels with zero canal density get sw_frac = 0
    (100 % groundwater).
    """
    grid = np.full(valid_mask.shape[0], 0.0, dtype=np.float64)
    grid[valid_mask] = canal_dens
    grid = grid.reshape(raster_shape)

    local_max = maximum_filter(grid, size=kernel_size, mode='constant', cval=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sw_frac_grid = np.where(local_max > 0, grid / local_max, 0.0)
    sw_frac = sw_frac_grid.ravel()[valid_mask]
    return np.clip(sw_frac, 0, 1)


def partition_predictions(
        predictions: np.ndarray,
        year_df,
        raster_shape: tuple,
        valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Partition total pumping predictions into 8 withdrawal categories.

    Parameters
    ----------
    predictions : 1-D array
        Total pumping predictions (mm) for valid pixels.
    year_df : pd.DataFrame
        Feature DataFrame for a single year (must contain columns
        ``well_density``, ``annual_irr_fraction``,
        ``annual_gw_fraction``, ``canal_density`` when available).
    raster_shape : tuple
        (rows, cols) of the full raster grid.
    valid_mask : 1-D bool array
        Mask of valid (in-AZ) pixels in the flattened raster.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of category name → 1-D prediction array (same length
        as *predictions*).
    """
    # ---- Well density masking ----
    well_dens = year_df['well_density'].values if 'well_density' in year_df.columns else None
    if well_dens is not None:
        no_well = (well_dens == 0) | np.isnan(well_dens)
        predictions = predictions.copy()
        predictions[no_well] = np.nan

    # ---- Irrigation / Non-irrigation split ----
    irr_frac = year_df['annual_irr_fraction'].values if 'annual_irr_fraction' in year_df.columns else None
    if irr_frac is not None and well_dens is not None:
        irr_frac = focal_fill_irr_fraction(irr_frac, well_dens, raster_shape, valid_mask)

    irr = predictions * irr_frac if irr_frac is not None else predictions.copy()
    nonirr = predictions - irr

    # ---- GW / SW split of irrigation ----
    gw_frac = year_df['annual_gw_fraction'].values if 'annual_gw_fraction' in year_df.columns else None
    if gw_frac is not None:
        gw_frac = np.clip(np.nan_to_num(gw_frac, nan=1.0), 0, 1)
        irr_gw = irr * gw_frac
        irr_sw = irr - irr_gw
    else:
        irr_gw = irr.copy()
        irr_sw = np.zeros_like(irr)

    # ---- GW / SW split of non-irrigation (canal density proxy) ----
    canal_dens = year_df['canal_density'].values if 'canal_density' in year_df.columns else None
    if canal_dens is not None:
        sw_frac = compute_sw_fraction(canal_dens, raster_shape, valid_mask)
        nonirr_sw = nonirr * sw_frac
        nonirr_gw = nonirr - nonirr_sw
    else:
        nonirr_gw = nonirr.copy()
        nonirr_sw = np.zeros_like(nonirr)

    return {
        'Irrigation':         irr,
        'Non_Irrigation':     nonirr,
        'Irrigation_GW':      irr_gw,
        'Irrigation_SW':      irr_sw,
        'Non_Irrigation_GW':  nonirr_gw,
        'Non_Irrigation_SW':  nonirr_sw,
        'Total_GW':           irr_gw + nonirr_gw,
        'Total_SW':           irr_sw + nonirr_sw,
    }
