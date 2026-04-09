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

    Args:
        irr_frac (np.ndarray): 1-D irrigation fraction for valid pixels.
        well_dens (np.ndarray): 1-D well density for valid pixels.
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Boolean mask of valid pixels (ravelled).
        kernel_size (int): Focal neighbourhood size (default 5).
        min_irr_frac (float): Minimum irrigation fraction threshold.

    Returns:
        np.ndarray: Gap-filled irrigation fraction array.
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

    focal_raveled = focal_mean.ravel()
    if focal_raveled.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f'focal_mean ravel size ({focal_raveled.shape[0]}) != '
            f'valid_mask size ({valid_mask.shape[0]})'
        )
    focal_flat = focal_raveled[valid_mask]
    # Only overwrite when the focal neighbourhood has substantial irrigation;
    # otherwise preserve the original small value.
    fill_mask = needs_fill & (focal_flat > 0)
    filled[fill_mask] = np.clip(focal_flat[fill_mask], 0, 1)
    return filled


def adjust_gw_fraction_temporal(
        gw_frac: np.ndarray,
        sw_access_year: np.ndarray,
        year: int,
) -> np.ndarray:
    """Adjust Hung et al. GW fraction for pre-canal years.

    Pixels that had not yet gained surface-water access (based on the
    earliest irrigation SW water-right priority date from HarDWR) are
    set to ``gw_frac = 1.0`` (100 % groundwater).

    Args:
        gw_frac: Static GW fraction (1-D, valid pixels).
        sw_access_year: Per-pixel earliest irrigation SW priority year
            (NaN for pixels with no irrigation SW rights).
        year: Target prediction year.

    Returns:
        Adjusted GW fraction array (clipped to [0, 1]).
    """
    adjusted = gw_frac.copy()
    has_sw = np.isfinite(sw_access_year)
    pre_sw = has_sw & (year < sw_access_year)
    adjusted[pre_sw] = 1.0
    return adjusted


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

    Args:
        canal_dens (np.ndarray): 1-D canal density for valid pixels.
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Boolean mask of valid pixels (ravelled).
        kernel_size (int): Focal neighbourhood size (default 5).

    Returns:
        np.ndarray: Surface-water fraction array, clipped to [0, 1].
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

    Args:
        predictions (np.ndarray): Total pumping predictions (mm) for valid pixels (1-D array).
        year_df (pd.DataFrame): Feature DataFrame for a single year (must contain columns
            ``well_density``, ``annual_irr_fraction``,
            ``annual_gw_fraction``, ``canal_density`` when available).
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Mask of valid (in-AZ) pixels in the flattened raster (1-D bool array).

    Returns:
        dict[str, np.ndarray]: Mapping of category name to 1-D prediction array (same length
            as *predictions*).
    """
    # ---- Well density masking ----
    well_dens = year_df['well_density'].values if 'well_density' in year_df.columns else None
    if well_dens is not None:
        no_well = (well_dens == 0) | np.isnan(well_dens)
        predictions = predictions.copy()
        predictions[no_well] = np.nan

    # ---- Irrigation / Non-irrigation split ----
    # Use pump-capacity-weighted irrigation fraction when available:
    # irr_capacity_fraction = (sum of PUMPRATE for IRRIGATION wells) /
    #                         (sum of PUMPRATE for all active wells)
    # per 2 km pixel. This gives a physically meaningful withdrawal split
    # (~65% ag statewide after PUMPRATE imputation) unlike the area-based
    # irr_fraction which is geometrically small at 2 km resolution.
    #
    # The static well-registry fraction is made time-varying by scaling
    # each side by its respective area-fraction change relative to the
    # well-registry reference year (2024):
    #   irr_weight  = irr_cap_static  × (crop_frac / crop_frac_2024)
    #   mi_weight   = mi_cap_static   × (urban_frac / urban_frac_2024)
    #   irr_frac(y) = irr_weight / (irr_weight + mi_weight)
    # This captures the 1950s ag boom (more crop → higher irr share) and
    # future urbanisation (more urban → higher M&I share).
    #
    # Falls back to the area-based annual_irr_fraction for older parquets.
    irr_cap_frac = year_df['irr_capacity_fraction'].values \
        if 'irr_capacity_fraction' in year_df.columns else None
    if irr_cap_frac is not None:
        irr_cap_frac = np.clip(np.nan_to_num(irr_cap_frac, nan=0.0), 0, 1)
        mi_cap_frac = 1.0 - irr_cap_frac

        # Temporal scaling by crop/urban area changes
        crop_frac = year_df['annual_crop_fraction'].values \
            if 'annual_crop_fraction' in year_df.columns else None
        urban_frac = year_df['annual_urban_fraction'].values \
            if 'annual_urban_fraction' in year_df.columns else None
        crop_ref = year_df['crop_frac_ref'].values \
            if 'crop_frac_ref' in year_df.columns else None
        urban_ref = year_df['urban_frac_ref'].values \
            if 'urban_frac_ref' in year_df.columns else None

        if (crop_frac is not None and urban_frac is not None
                and crop_ref is not None and urban_ref is not None):
            with np.errstate(invalid='ignore', divide='ignore'):
                crop_ratio = np.where(crop_ref > 0, crop_frac / crop_ref, 1.0)
                urban_ratio = np.where(urban_ref > 0, urban_frac / urban_ref, 1.0)
            irr_weight = irr_cap_frac * crop_ratio
            mi_weight = mi_cap_frac * urban_ratio
            total_weight = irr_weight + mi_weight
            with np.errstate(invalid='ignore', divide='ignore'):
                irr_cap_frac = np.where(
                    total_weight > 0, irr_weight / total_weight, irr_cap_frac,
                )
            irr_cap_frac = np.clip(np.nan_to_num(irr_cap_frac, nan=0.0), 0, 1)

        irr = predictions * irr_cap_frac
        nonirr = predictions * (1 - irr_cap_frac)
    else:
        irr_frac = year_df['annual_irr_fraction'].values \
            if 'annual_irr_fraction' in year_df.columns else None
        if irr_frac is not None and well_dens is not None:
            irr_frac = focal_fill_irr_fraction(
                irr_frac, well_dens, raster_shape, valid_mask,
            )
        irr = predictions * irr_frac if irr_frac is not None else predictions.copy()
        nonirr = predictions - irr

    # ---- Urban-fraction weighting for non-irrigation outside AMA/INAs ----
    # AMA/INA basins (GW_Basin_Type 0 or 1) are managed groundwater areas
    # where M&I withdrawal is well-established — keep full nonirr value.
    # Outside AMA/INAs (GW_Basin_Type 2), non-irrigation withdrawals are
    # sparse and concentrated in small towns. Scale by the physical
    # urban-area fraction to zero out rural pixels with no real M&I demand.
    basin_type = year_df['GW_Basin_Type'].values \
        if 'GW_Basin_Type' in year_df.columns else None
    urban_frac_col = year_df['annual_urban_fraction'].values \
        if 'annual_urban_fraction' in year_df.columns else None
    if basin_type is not None and urban_frac_col is not None:
        is_other = basin_type == 2
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        nonirr[is_other] = nonirr[is_other] * uf[is_other]

    # ---- Zero-surface-water mask: pixels with no SW access ----
    # Step 1: Where regular streamflow is zero, force SW density to zero.
    # This removes Gaussian-smoothed bleed from adjacent rivers into
    # basins with no actual river flow (e.g., Willcox).
    # Step 2: Where both (corrected) SW density and canal-weighted
    # streamflow are zero, force all withdrawals to groundwater.
    sw_density = year_df['SW'].values.copy() if 'SW' in year_df.columns else None
    streamflow = year_df['streamflow_mm'].values \
        if 'streamflow_mm' in year_df.columns else None
    cw_streamflow = year_df['canal_weighted_streamflow_mm'].values \
        if 'canal_weighted_streamflow_mm' in year_df.columns else None

    # Zero out SW density where there is no river
    if sw_density is not None and streamflow is not None:
        sw_density[streamflow == 0] = 0.0

    zero_sw_mask = None
    if sw_density is not None and cw_streamflow is not None:
        zero_sw_mask = (sw_density == 0) & (cw_streamflow == 0)
        if not np.any(zero_sw_mask):
            zero_sw_mask = None
    elif sw_density is not None:
        zero_sw_mask = sw_density == 0
        if not np.any(zero_sw_mask):
            zero_sw_mask = None

    # ---- GW / SW split of irrigation ----
    gw_frac = year_df['annual_gw_fraction'].values if 'annual_gw_fraction' in year_df.columns else None
    if gw_frac is not None:
        gw_frac = np.clip(np.nan_to_num(gw_frac, nan=1.0), 0, 1)
        # Temporal adjustment: pre-canal pixels → 100 % GW
        sw_access_yr = year_df['sw_access_year'].values \
            if 'sw_access_year' in year_df.columns else None
        if sw_access_yr is not None:
            year_val = int(year_df['Year'].iloc[0])
            gw_frac = adjust_gw_fraction_temporal(
                gw_frac, sw_access_yr, year_val,
            )
        # Force 100% GW where streamflow is zero
        if zero_sw_mask is not None:
            gw_frac[zero_sw_mask] = 1.0
        irr_gw = irr * gw_frac
        irr_sw = irr - irr_gw
    else:
        irr_gw = irr.copy()
        irr_sw = np.zeros_like(irr)

    # ---- GW / SW split of non-irrigation ----
    # Prefer HarDWR non-irrigation SW rights density (temporally varying)
    # over static canal density when available.
    nonirr_sw_dens = year_df['nonirr_sw_rights_density'].values \
        if 'nonirr_sw_rights_density' in year_df.columns else None
    canal_dens = year_df['canal_density'].values \
        if 'canal_density' in year_df.columns else None
    if nonirr_sw_dens is not None:
        sw_frac = compute_sw_fraction(nonirr_sw_dens, raster_shape, valid_mask)
        # Force zero SW where streamflow is zero
        if zero_sw_mask is not None:
            sw_frac[zero_sw_mask] = 0.0
        nonirr_sw = nonirr * sw_frac
        nonirr_gw = nonirr - nonirr_sw
    elif canal_dens is not None:
        sw_frac = compute_sw_fraction(canal_dens, raster_shape, valid_mask)
        if zero_sw_mask is not None:
            sw_frac[zero_sw_mask] = 0.0
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
