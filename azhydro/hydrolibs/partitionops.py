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

import logging

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

logger = logging.getLogger(__name__)

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
    2 km cell.  Replace with the focal mean of neighbors whose
    irr_fraction >= *min_irr_frac* within a *kernel_size* × *kernel_size*
    window.

    If the focal neighborhood also has no substantial irrigated pixels,
    the value stays unchanged (genuinely non-irrigation well).

    Args:
        irr_frac (np.ndarray): 1-D irrigation fraction for valid pixels.
        well_dens (np.ndarray): 1-D well density for valid pixels.
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Boolean mask of valid pixels (ravelled).
        kernel_size (int): Focal neighborhood size (default 5).
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

    # Only count neighbors with substantial irrigation for the focal mean
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
    # Only overwrite when the focal neighborhood has substantial irrigation;
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
    maximum within a *kernel_size* × *kernel_size* neighborhood,
    clipped to [0, 1].  Pixels with zero canal density get sw_frac = 0
    (100 % groundwater).

    Args:
        canal_dens (np.ndarray): 1-D canal density for valid pixels.
        raster_shape (tuple): (rows, cols) of the full raster grid.
        valid_mask (np.ndarray): Boolean mask of valid pixels (ravelled).
        kernel_size (int): Focal neighborhood size (default 5).

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


RURAL_AMA_INA = frozenset({
    'WILLCOX AMA', 'HARQUAHALA INA', 'HUALAPAI VALLEY INA',
    'JOSEPH CITY INA', 'DOUGLAS AMA', 'PINAL AMA',
    'PHOENIX AMA', 'TUCSON AMA', 'PRESCOTT AMA', 'SANTA CRUZ AMA',
})


CANAL_PREDICTOR_START = 1938

# 1960 drought-recovery dip: USGS Circ 456 records a ~2.5 MAF drop
# from the 1955 peak (8.09) to 1960 (5.62). 1957-1959 were wet years
# (precip 323-373 mm vs 311 LT mean), SW deliveries recovered, and
# marginal ag pumping contracted. Phantom wells represent exactly this
# marginal-ag pumping — drop phantom K to a low value at 1958-1960 to
# capture the retreat, then restore by 1965. Year-specific wells and
# wd_1981 remain at full strength (those registered wells kept pumping).
PHANTOM_DROUGHT_DIP_START = 1957
PHANTOM_DROUGHT_DIP_MIN_YEAR = 1960
PHANTOM_DROUGHT_DIP_END = 1965
PHANTOM_DROUGHT_DIP_MIN = 0.20   # phantom K drops to 20% of plateau

# 2024-registry override active window: 1951–2020 with the 1958–1961
# drought-recovery trough skipped.  Magnitude ramps 1.0 → 0.2 over
# 1995–2005, holds at 0.2 through 2015 to preserve a residual registry
# signal post-CAP (where year-specific registration is complete but the
# 2024 snapshot still helps recover the 2000–2015 under-Irr bias), then
# ramps 0.2 → 0 over 2015–2020.
WD_2024_ACTIVE_START = 1951
WD_2024_ACTIVE_END = 2020
WD_2024_RAMP_DOWN_START = 1995
WD_2024_RAMP_DOWN_END = 2005
WD_2024_TAIL = 0.2
WD_2024_TAIL_END_START = 2015
WD_2024_TAIL_END_END = 2020
WD_2024_DIP_SKIP_START = 1958
WD_2024_DIP_SKIP_END = 1961


def _wd_2024_active(year: int) -> bool:
    """2024 registry override is active 1951–2020 but skipped during
    the 1958–1961 drought-recovery trough."""
    if not (WD_2024_ACTIVE_START <= year <= WD_2024_ACTIVE_END):
        return False
    if WD_2024_DIP_SKIP_START <= year <= WD_2024_DIP_SKIP_END:
        return False
    return True


def _wd_2024_scale(year: int) -> float:
    """Magnitude scale applied to the 2024 registry override.

    1951–1983: scaled by _phantom_k so the 1958–1960 drought-recovery
    trough reduces registered-well pumping too.  1984–1994: full
    strength (1.0).  1995–2005: linear ramp 1.0 → WD_2024_TAIL (0.2).
    2005–2015: flat tail at 0.2 (residual post-CAP registry signal).
    2015–2020: linear ramp 0.2 → 0.0."""
    if year <= 1983:
        return _phantom_k(year)
    if year <= WD_2024_RAMP_DOWN_START:
        return 1.0
    if year < WD_2024_RAMP_DOWN_END:
        return 1.0 - (1.0 - WD_2024_TAIL) * (
            (year - WD_2024_RAMP_DOWN_START)
            / float(WD_2024_RAMP_DOWN_END - WD_2024_RAMP_DOWN_START)
        )
    if year <= WD_2024_TAIL_END_START:
        return WD_2024_TAIL
    if year < WD_2024_TAIL_END_END:
        return WD_2024_TAIL * (
            1.0 - (year - WD_2024_TAIL_END_START)
            / float(WD_2024_TAIL_END_END - WD_2024_TAIL_END_START)
        )
    return 0.0

# Phantom-well override (replaces prior Option B / wd_1981 blend).
# Pre-GMA wells were registered only voluntarily; many ag wells abandoned
# before 1980 have no record in the 2024 registry.  Phantom wells fill
# ag-fringe pixels (gated by AGRI density) without importing modern
# industrial wells (Palo Verde, Intel, urban public supply).
PHANTOM_M_TOTAL = 12.0
# AGRI threshold drops sharply during the 1951–1955 drilling boom.
# Physically: ag-well drilling grew modestly 1938→1951 (threshold 0.60
# → 0.40), then exploded during the 1951–1955 peak (threshold crashes
# 0.40 → 0.15 as farmers drilled on any marginal ag land). Post-1955
# the effective threshold recovers to a 0.30 plateau by 1960.
PHANTOM_AGRI_THRESHOLD_EARLY = 0.60    # 1938
PHANTOM_AGRI_THRESHOLD_MID = 0.40      # 1951 (pre-boom)
PHANTOM_AGRI_THRESHOLD_PEAK = 0.08     # 1955 (drilling boom minimum — pushes activation wider)
PHANTOM_AGRI_THRESHOLD_LATE = 0.30     # 1960+ (plateau)
PHANTOM_AGRI_BOOM_START = 1951         # rapid drilling era begins

# Canal-gate relaxation — monotone continuous ramp aligned with the
# 2024-registry active window (1951–1983).
# Phantom at canal pixels activates the ML model's ag-extraction
# prediction at canal-served farms (Yuma, SRP, Pinal canal laterals);
# the GW/SW partition splitting then routes the volume to SW (canal
# deliveries) where SW-rights density dominates.
#
# 1951–1955: boom ramp 0 → PHANTOM_CANAL_BOOM_PEAK (0.30) — drought
#   drilling boom brings canal-fed supplemental pumping online.
# 1955–1975: gradual rise PHANTOM_CANAL_BOOM_PEAK → PHANTOM_CANAL_INCLUDE_MAX
#   (0.30 → 0.60) as conjunctive use matures through the CAP-planning era.
# 1975+: flat plateau at PHANTOM_CANAL_INCLUDE_MAX.
PHANTOM_CANAL_BOOM_START = 1951
PHANTOM_CANAL_BOOM_PEAK_YEAR = 1955
PHANTOM_CANAL_BOOM_PEAK = 0.30
PHANTOM_CANAL_RAMP_END = 1975
PHANTOM_CANAL_INCLUDE_MAX = 0.60


def _phantom_canal_include(year: int) -> float:
    """Fraction of phantom strength applied at canal-served pixels."""
    if year < PHANTOM_CANAL_BOOM_START:
        return 0.0
    if year < PHANTOM_CANAL_BOOM_PEAK_YEAR:
        frac = (year - PHANTOM_CANAL_BOOM_START) / float(
            PHANTOM_CANAL_BOOM_PEAK_YEAR - PHANTOM_CANAL_BOOM_START,
        )
        return PHANTOM_CANAL_BOOM_PEAK * frac
    if year < PHANTOM_CANAL_RAMP_END:
        frac = (year - PHANTOM_CANAL_BOOM_PEAK_YEAR) / float(
            PHANTOM_CANAL_RAMP_END - PHANTOM_CANAL_BOOM_PEAK_YEAR,
        )
        return (
            PHANTOM_CANAL_BOOM_PEAK
            + frac * (PHANTOM_CANAL_INCLUDE_MAX - PHANTOM_CANAL_BOOM_PEAK)
        )
    return PHANTOM_CANAL_INCLUDE_MAX


def _phantom_agri_threshold(year: int) -> float:
    """Three-segment V: 0.60 (1938) → 0.40 (1951) → 0.15 (1955) → 0.30 (1960+)."""
    if year < PHANTOM_K_RAMP_UP_START:
        return PHANTOM_AGRI_THRESHOLD_EARLY
    if year < PHANTOM_AGRI_BOOM_START:
        # 1938→1951: slow decline 0.60 → 0.40
        frac = (year - PHANTOM_K_RAMP_UP_START) / float(
            PHANTOM_AGRI_BOOM_START - PHANTOM_K_RAMP_UP_START,
        )
        return (
            PHANTOM_AGRI_THRESHOLD_EARLY
            + frac * (PHANTOM_AGRI_THRESHOLD_MID - PHANTOM_AGRI_THRESHOLD_EARLY)
        )
    if year < PHANTOM_K_RAMP_UP_END:
        # 1951→1955: sharp drop 0.40 → 0.15 (drilling boom)
        frac = (year - PHANTOM_AGRI_BOOM_START) / float(
            PHANTOM_K_RAMP_UP_END - PHANTOM_AGRI_BOOM_START,
        )
        return (
            PHANTOM_AGRI_THRESHOLD_MID
            + frac * (PHANTOM_AGRI_THRESHOLD_PEAK - PHANTOM_AGRI_THRESHOLD_MID)
        )
    post_peak_end = min(1960, PHANTOM_K_PLATEAU_END)
    if year < post_peak_end:
        # 1955→1960: recovery 0.15 → 0.30
        frac = (year - PHANTOM_K_RAMP_UP_END) / float(
            post_peak_end - PHANTOM_K_RAMP_UP_END,
        )
        return (
            PHANTOM_AGRI_THRESHOLD_PEAK
            + frac * (PHANTOM_AGRI_THRESHOLD_LATE - PHANTOM_AGRI_THRESHOLD_PEAK)
        )
    return PHANTOM_AGRI_THRESHOLD_LATE
PHANTOM_K_RAMP_UP_START = 1938   # GEE-LULC start; earliest year with real AGRI
PHANTOM_K_RAMP_UP_END = 1955     # ag-well drilling reaches maturity (USGS Circ 398 peak)
PHANTOM_K_PLATEAU_END = 1980     # GMA enacted 1980; registration mandatory afterwards
PHANTOM_K_RAMP_END = 1985        # enforcement maturity; unregistered stock absorbed


def _phantom_k(year: int) -> float:
    """Phantom-well temporal ramp with 1960 drought-recovery dip.

    0 pre-1938 (AGRI is a fixed 1938 snapshot — not representative of
    earlier eras). Linear up-ramp 1938→1955 mirrors the growth of ag
    well drilling. Plateau 1955–1980 with a V-shape dip 1957→1960→1965
    capturing the documented 1960 drought-recovery trough. Linear
    down-ramp 1980→1985 as mandatory GMA registration absorbs the
    unregistered-well stock.
    """
    if year < PHANTOM_K_RAMP_UP_START:
        return 0.0
    if year < PHANTOM_K_RAMP_UP_END:
        return (year - PHANTOM_K_RAMP_UP_START) / float(
            PHANTOM_K_RAMP_UP_END - PHANTOM_K_RAMP_UP_START,
        )
    # 1957→1960: ramp down 1.0 → PHANTOM_DROUGHT_DIP_MIN (drought-recovery retreat)
    if PHANTOM_DROUGHT_DIP_START < year < PHANTOM_DROUGHT_DIP_MIN_YEAR:
        frac = (year - PHANTOM_DROUGHT_DIP_START) / float(
            PHANTOM_DROUGHT_DIP_MIN_YEAR - PHANTOM_DROUGHT_DIP_START,
        )
        return 1.0 + frac * (PHANTOM_DROUGHT_DIP_MIN - 1.0)
    # 1960: full dip
    if year == PHANTOM_DROUGHT_DIP_MIN_YEAR:
        return PHANTOM_DROUGHT_DIP_MIN
    # 1960→1965: ramp up PHANTOM_DROUGHT_DIP_MIN → 1.0 (recovery)
    if PHANTOM_DROUGHT_DIP_MIN_YEAR < year < PHANTOM_DROUGHT_DIP_END:
        frac = (year - PHANTOM_DROUGHT_DIP_MIN_YEAR) / float(
            PHANTOM_DROUGHT_DIP_END - PHANTOM_DROUGHT_DIP_MIN_YEAR,
        )
        return PHANTOM_DROUGHT_DIP_MIN + frac * (1.0 - PHANTOM_DROUGHT_DIP_MIN)
    if year <= PHANTOM_K_PLATEAU_END:
        return 1.0
    if year < PHANTOM_K_RAMP_END:
        return (PHANTOM_K_RAMP_END - year) / float(
            PHANTOM_K_RAMP_END - PHANTOM_K_PLATEAU_END,
        )
    return 0.0

# Era-dependent GW/SW weighting for the density-ratio split.
# Pre-GMA (before 1981): GW dominated — SRP was the only major SW
# source; most irrigation was groundwater-fed.  USBR reports ~67% GW
# statewide for 1950-1970.
# Post-CAP (after 1993): CAP at full capacity, SW share rose sharply.
# USGS/ADWR report ~42% GW by 1990.
# The GW_WEIGHT multiplies the well-density side of the ratio:
#   gw_frac = (gw_weight × wd) / (gw_weight × wd + swd_smooth)
GMA_YEAR = 1981
CAP_FULL_YEAR = 1990
GW_WEIGHT_PRE_GMA = 1.0
GW_WEIGHT_POST_CAP = 0.1


def _era_gw_weight(year: int) -> float:
    """Return era-dependent GW weight for the density-ratio split.

    Pre-1938 (no GEE LULC): weight = 1.0 (default, GW already dominant
    from well-only pixel retention).
    1938–1980: boosted GW weight to match USBR ~67% GW statewide.
    1981–1993: linear ramp down as CAP comes online.
    1993+: reduced GW weight to match USGS/ADWR ~42% GW.
    """
    if year < CANAL_PREDICTOR_START:
        return 1.0
    if year < GMA_YEAR:
        return GW_WEIGHT_PRE_GMA
    if year >= CAP_FULL_YEAR:
        return GW_WEIGHT_POST_CAP
    frac = (year - GMA_YEAR) / (CAP_FULL_YEAR - GMA_YEAR)
    return GW_WEIGHT_PRE_GMA + frac * (GW_WEIGHT_POST_CAP - GW_WEIGHT_PRE_GMA)


def partition_predictions(
        predictions: np.ndarray,
        year_df,
        raster_shape: tuple,
        valid_mask: np.ndarray,
        sw_smooth_sigma: float = 4.0,
        year: int = 0,
        wd_1981: np.ndarray | None = None,
        irr_wd_1981: np.ndarray | None = None,
        nonirr_wd_1981: np.ndarray | None = None,
        irr_cap_1981: np.ndarray | None = None,
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
        sw_smooth_sigma (float): Gaussian smoothing kernel width (in pixels) used to
            spread SW-rights-density influence across canal service areas.
            Default 4.0 (~8 km radius at 2 km resolution).
        year (int): Prediction year. Canal-only pixels are kept only for
            years >= ``CANAL_PREDICTOR_START`` (1938) when GEE predictor
            data is available.

    Returns:
        dict[str, np.ndarray]: Mapping of category name to 1-D prediction array (same length
            as *predictions*).
    """
    # ---- Pixel retention masking ----
    # Keep predictions at pixels with wells, smoothed canal service
    # area, or crop/urban land use (from 1938 when GEE LULC starts).
    from scipy.ndimage import gaussian_filter as _gaussian_filter

    well_dens = year_df['well_density'].values if 'well_density' in year_df.columns else None
    canal_dens = year_df['canal_density'].values if 'canal_density' in year_df.columns else None
    cw_streamflow_raw = year_df['canal_weighted_streamflow_mm'].values \
        if 'canal_weighted_streamflow_mm' in year_df.columns else None
    crop_frac_col = year_df['annual_crop_fraction'].values \
        if 'annual_crop_fraction' in year_df.columns else None
    urban_dens_col = year_df['URBAN'].values \
        if 'URBAN' in year_df.columns else None

    # Smooth canal-weighted streamflow to identify canal service area
    has_smooth_canal = np.zeros(len(predictions), dtype=bool)
    cw_smooth_1d = np.zeros(len(predictions), dtype=np.float64)
    crop_frac_filter = np.zeros(len(predictions), dtype=np.float64)
    if cw_streamflow_raw is not None and year >= CANAL_PREDICTOR_START:
        cw_grid = np.zeros(raster_shape, dtype=np.float64)
        cw_grid.ravel()[valid_mask] = np.nan_to_num(cw_streamflow_raw, nan=0.0)
        cw_smoothed = _gaussian_filter(cw_grid, sigma=2.0)
        cw_smooth_1d = cw_smoothed.ravel()[valid_mask]
        crop_frac_filter = np.clip(
            np.nan_to_num(crop_frac_col, nan=0.0), 0, 1,
        ) if crop_frac_col is not None else np.zeros(len(predictions))
        has_smooth_canal = (cw_smooth_1d > 0) & (crop_frac_filter > 0)

    has_well = ~((well_dens == 0) | np.isnan(well_dens)) \
        if well_dens is not None else np.zeros(len(predictions), dtype=bool)

    # ---- Hindcast override ramps ----
    # LU-only retention ramp (applied only to LU-only pixel scaling at the
    # crop/urban retention step below): up 1945→1955, plateau 1955–1980,
    # down 1981→1983. Unchanged shape from prior iteration — LU-only
    # pixel retention ramps earlier than the phantom-well mechanism
    # because land-use-based retention reflects ag expansion, not
    # well-registration lag.
    _OV_RAMP_UP_START, _OV_RAMP_UP_END = 1945, 1960
    _RAMP_DN_START, _RAMP_DN_END = GMA_YEAR, 2010
    if year < 1945:
        _lu_ramp = 0.0
    elif year < 1955:
        _lu_ramp = (year - 1945) / (1955 - 1945)
    elif year <= 1980:
        _lu_ramp = 1.0
    elif year <= 1983:
        _lu_ramp = 1.0 - (year - 1980) / 3.0
    else:
        _lu_ramp = 0.0

    # Phantom-well temporal ramp — drives well_density override and
    # partition-time irr routing.
    _k_phantom = _phantom_k(year)
    # Note: no pre-1938 _wd_alpha floor — XGBoost treats any
    # well_density > 0 as significant, so even tiny floors inflate
    # predictions massively.  Pre-1938 stays well-only with the
    # year-specific registry (acknowledged limitation).

    if year < _OV_RAMP_UP_START:
        _override_alpha = 0.0
    elif year < _OV_RAMP_UP_END:
        _override_alpha = (year - _OV_RAMP_UP_START) / (_OV_RAMP_UP_END - _OV_RAMP_UP_START)
    elif year <= 1980:
        _override_alpha = 1.0
    elif year < _RAMP_DN_END:
        _override_alpha = 1.0 - (year - _RAMP_DN_START) / (_RAMP_DN_END - _RAMP_DN_START)
    else:
        _override_alpha = 0.0

    # Backward-compat alias for existing code paths
    _hindcast_alpha = _override_alpha

    # Phantom-well override: pre-GMA unregistered ag wells modeled as
    # AGRI(p, year) × PHANTOM_M_TOTAL × K(year), combined with the
    # year-specific registry via per-pixel max(). AGRI-gated so modern
    # industrial wells (Palo Verde, Intel, urban public supply) are not
    # projected back into historical eras.
    _gma_fill_mask = np.zeros(len(predictions), dtype=bool)
    _phantom_dominant = np.zeros(len(predictions), dtype=bool)
    _yr_wd_cached = np.nan_to_num(well_dens, nan=0.0) if well_dens is not None else None

    # 2024 registry override: active 1951–1983 (hard window). Gives the
    # ML model the spatial pattern of where wells actually ended up
    # post-GMA. No spatial filters on wd_1981 — urban-core areas that
    # were ag pre-GMA (Chandler, Gilbert, Mesa, east Tucson) have their
    # volume routed to irrigation via irr_cap_frac=1 downstream.
    # Magnitude is scaled by the phantom drought-K factor so the 1958–
    # 1960 drought-recovery trough applies to registered wells too
    # (irrigation capacity effectively reduced during wet years).
    _urban_wd_mask = np.zeros(len(predictions), dtype=bool)
    if _wd_2024_active(year) and wd_1981 is not None and _yr_wd_cached is not None:
        wd24 = np.nan_to_num(wd_1981, nan=0.0) * _wd_2024_scale(year)
        well_dens = np.maximum(_yr_wd_cached, wd24)
        # Track urban pixels where wd_1981 contributed — these get their
        # partitioned volume routed to irrigation (pre-GMA urban areas
        # were ag land).
        uf_retain_col = year_df['annual_urban_fraction'].values \
            if 'annual_urban_fraction' in year_df.columns else None
        if uf_retain_col is not None:
            uf = np.nan_to_num(uf_retain_col, nan=0.0)
            _urban_wd_mask = (uf > 0.2) & (wd24 > _yr_wd_cached)
        _yr_wd_cached = well_dens
        has_well = well_dens > 0
    # Gate phantom by GEE-LULC start: pre-1938 AGRI is a fixed 1938
    # snapshot, not representative of the unregistered-well era.
    if _k_phantom > 0 and year >= CANAL_PREDICTOR_START:
        agri_col = year_df['AGRI'].values if 'AGRI' in year_df.columns else None
        if agri_col is not None and _yr_wd_cached is not None:
            agri = np.clip(np.nan_to_num(agri_col, nan=0.0), 0.0, 1.0)
            # Year-dependent threshold: tighter early (0.60 at 1938),
            # drops sharply during 1951–1955 drilling boom, relaxes to
            # a 0.30 plateau by 1960.
            _agri_thr = _phantom_agri_threshold(year)
            ag_gate = agri >= _agri_thr
            # Canal-gate relaxation: canal-fed ag pixels get a fraction
            # of phantom strength (conjunctive groundwater use). Pre-1965:
            # canal weight = 0 (pure canal ag, no backup pumping). Post-
            # 1975: canal weight = PHANTOM_CANAL_INCLUDE_MAX.
            _canal_inc = _phantom_canal_include(year)
            canal_weight = np.where(has_smooth_canal, _canal_inc, 1.0)
            phantom_wd = (
                agri * PHANTOM_M_TOTAL * _k_phantom
                * ag_gate.astype(np.float64)
                * canal_weight
            )
            well_dens = np.maximum(_yr_wd_cached, phantom_wd)
            has_well = well_dens > 0
            _gma_fill_mask = has_well
            _phantom_dominant = phantom_wd > _yr_wd_cached

    _has_crop_any = (np.nan_to_num(crop_frac_col, nan=0.0) > 0) \
        if crop_frac_col is not None else np.zeros(len(predictions), dtype=bool)
    uf_retain_col = year_df['annual_urban_fraction'].values \
        if 'annual_urban_fraction' in year_df.columns else None
    _has_urban_any = (np.nan_to_num(uf_retain_col, nan=0.0) >= 0.2) \
        if uf_retain_col is not None else np.zeros(len(predictions), dtype=bool)

    # Pre-1938: only retain well pixels that overlap with crops or urban
    if year < CANAL_PREDICTOR_START:
        has_well = has_well & (_has_crop_any | _has_urban_any)

    # LU-only pixel retention from 1925+ (1938 LULC snapshot used as
    # proxy for 1925-1937 to capture unregistered well era).
    has_crop = _has_crop_any & (year >= 1925)
    has_urban = _has_urban_any & (year >= 1925)
    keep = has_well | has_smooth_canal | has_crop | has_urban
    predictions = predictions.copy()
    predictions[~keep] = np.nan

    # Identify LU-only pixels (crop/urban but no wells, no canal service)
    lu_only = ~has_well & ~has_smooth_canal & (has_crop | has_urban)

    # Scale LU-only pixel predictions:
    # 1925-1937: small ramp 0.05→0.15 to capture early ag boom era
    # 1938-1944: 0 (no override yet)
    # 1945-1955: ramp 0→1 (_lu_ramp up-ramp)
    # 1955-1980: 1.0 (full)
    # 1981-1983: ramp 1→0 (_lu_ramp down-ramp)
    # 1984+: 0 (year-specific registry only)
    if 1925 <= year < CANAL_PREDICTOR_START and lu_only.any():
        early_alpha = 0.10 + (year - 1925) * (0.30 - 0.10) / (1937 - 1925)
        predictions[lu_only] = predictions[lu_only] * early_alpha
    elif _lu_ramp < 1.0 and lu_only.any():
        predictions[lu_only] = predictions[lu_only] * _lu_ramp

    # Urban wd_1981 pixels (today's urban cores that were ag pre-GMA):
    # scale their volume by AGRI so only the ag fraction of the pixel
    # contributes (not the full 4 km² pixel).
    if _urban_wd_mask.any() and 'AGRI' in year_df.columns:
        _agri_scale = np.clip(
            np.nan_to_num(year_df['AGRI'].values, nan=0.0), 0.0, 1.0,
        )
        predictions[_urban_wd_mask] = (
            predictions[_urban_wd_mask] * _agri_scale[_urban_wd_mask]
        )

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
        # Pre-GMA: fill irr_capacity_fraction at crop pixels with
        # unregistered wells using the 1981 snapshot.
        if _gma_fill_mask.any() and irr_cap_1981 is not None:
            irr_cap_frac[_gma_fill_mask] = np.clip(
                np.nan_to_num(irr_cap_1981[_gma_fill_mask], nan=0.0), 0, 1,
            )
        # Phantom-dominant pixels (AGRI-driven phantom > year-specific wd)
        # are primarily irrigation, but may overlap urban encroachment —
        # route to Irr by (1 - urban_frac) so municipal NonIrr keeps its
        # share at urbanized ag fringe pixels.  Previously hard-set to
        # 1.0, which collapsed 1970–1980 NonIrr to ~0.
        _urban_frac_col = year_df['annual_urban_fraction'].values \
            if 'annual_urban_fraction' in year_df.columns else None
        _uf_route = np.clip(np.nan_to_num(
            _urban_frac_col if _urban_frac_col is not None
            else np.zeros(len(predictions)), nan=0.0,
        ), 0, 1)
        if _phantom_dominant.any():
            irr_cap_frac[_phantom_dominant] = 1.0 - _uf_route[_phantom_dominant]
        # Urban pixels where wd_1981 > year-specific: these were ag land
        # pre-1970 (Chandler/Gilbert/Mesa/east Tucson suburbs of today).
        # Gradually transition from "route to Irr by (1-uf)" (ag era) to
        # natural partition (municipal era) as Phoenix/Tucson urbanized
        # 1970-1980.
        if _urban_wd_mask.any():
            if year <= 1970:
                _urb_to_irr = 1.0
            elif year < 1980:
                _urb_to_irr = 1.0 - (year - 1970) / (1980 - 1970)
            else:
                _urb_to_irr = 0.0
            if _urb_to_irr > 0:
                natural = irr_cap_frac[_urban_wd_mask]
                ag_share = 1.0 - _uf_route[_urban_wd_mask]
                irr_cap_frac[_urban_wd_mask] = (
                    _urb_to_irr * ag_share + (1.0 - _urb_to_irr) * natural
                )
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

        # Hindcast irr_cap override using the ramp.
        # Ramp-up (1945-1969): boost only at crop pixels (partial alpha
        # — applying to all pixels inflates totals by preventing
        # nonirr from being urban-scaled away).
        # Plateau + ramp-down (1970-1984): boost at ALL pixels (USBR
        # reports ~91% ag).  The ramp-down alpha itself tapers the
        # boost smoothly toward the modern parquet values.
        if _override_alpha > 0 and urban_frac is not None:
            uf_override = np.clip(np.nan_to_num(urban_frac, nan=0.0), 0, 1)
            # Cap ag share at the era-observed USGS Irr% so NonIrr has
            # a realistic floor during the 1970–1984 plateau.  Without
            # the cap, (1 - urban_frac) ≈ 1 at rural 1970 pixels forces
            # irr_cap_frac=1 and collapses NonIrr to ~0 (observed
            # 91.4% ag in 1970, 93.9% in 1975, 89.1% in 1980 — midpoint
            # 0.92 leaves an ~8% NonIrr floor that matches the anchors).
            _ag_cap_max = 0.92
            ag_era_cap = np.minimum(_ag_cap_max, 1.0 - uf_override)
            blended = (
                irr_cap_frac * (1 - _override_alpha)
                + ag_era_cap * _override_alpha
            )
            if year >= 1970:
                irr_cap_frac = blended
            elif crop_frac is not None:
                has_crop_px = np.nan_to_num(crop_frac, nan=0.0) > 0
                irr_cap_frac = np.where(has_crop_px, blended, irr_cap_frac)
            irr_cap_frac = np.clip(irr_cap_frac, 0, 1)
        elif crop_frac is not None and year < CANAL_PREDICTOR_START:
            well_with_crop = has_well & (np.nan_to_num(crop_frac, nan=0.0) > 0)
            uf_pre = np.clip(np.nan_to_num(
                urban_frac[well_with_crop] if urban_frac is not None
                else np.zeros(well_with_crop.sum()), nan=0.0,
            ), 0, 1)
            irr_cap_frac[well_with_crop] = 1.0 - uf_pre

        # Post-CAP: light ag boost at crop pixels (partial blend toward
        # 1-urban_frac) to correct for 2024 registry undercounting
        # irrigation at urban-encroached ag areas.  USGS/ADWR show ag
        # remained ~72% of water use even in 2017.  Activated from 2000
        # (decoupled from _RAMP_DN_END) to recover the 2000–2015 under-
        # Irr bias that the 2024 registry ramp-down leaves behind.
        _POST_CAP_BLEND_START = 2000
        if crop_frac is not None and year >= _POST_CAP_BLEND_START:
            post_cap_blend = 0.75
            has_crop_px_pc = np.nan_to_num(crop_frac, nan=0.0) > 0
            uf_pc = np.clip(np.nan_to_num(
                urban_frac if urban_frac is not None
                else np.zeros(len(predictions)), nan=0.0,
            ), 0, 1)
            ag_cap_pc = 1.0 - uf_pc
            irr_cap_frac = np.where(
                has_crop_px_pc,
                irr_cap_frac * (1 - post_cap_blend) + ag_cap_pc * post_cap_blend,
                irr_cap_frac,
            )
            irr_cap_frac = np.clip(irr_cap_frac, 0, 1)

        irr = predictions * irr_cap_frac
        nonirr = predictions * (1 - irr_cap_frac)

        # LU-only pixels (no wells, no canal service area): split by
        # crop fraction (→ irrigation) and URBAN density (→ non-irr).
        # Canal-service-area pixels without wells: crop → irrigation only.
        has_direct_canal = (canal_dens > 0) if canal_dens is not None \
            else np.zeros(len(predictions), dtype=bool)
        canal_ag_no_wells = (has_smooth_canal | has_direct_canal) & ~has_well
        if lu_only.any() or canal_ag_no_wells.any():
            cf = np.clip(np.nan_to_num(
                crop_frac if crop_frac is not None
                else np.zeros(len(predictions)), nan=0.0,
            ), 0, 1)
            ud = np.clip(np.nan_to_num(
                urban_dens_col if urban_dens_col is not None
                else np.zeros(len(predictions)), nan=0.0,
            ), 0, 1)
            # LU-only: full prediction for irr (prediction rasters are
            # not masked to crops — every pixel has a value that reflects
            # local climate/LULC). NonIrr scaled by annual_urban_fraction.
            uf_col = year_df['annual_urban_fraction'].values \
                if 'annual_urban_fraction' in year_df.columns \
                else np.zeros(len(predictions))
            uf_vals = np.clip(np.nan_to_num(uf_col, nan=0.0), 0, 1)
            # Only assign irr at pixels with crop > 0
            lu_crop = lu_only & (cf > 0)
            lu_urban_only = lu_only & ~(cf > 0)
            irr[lu_crop] = predictions[lu_crop]
            nonirr[lu_crop] = predictions[lu_crop] * uf_vals[lu_crop]
            # Urban-only LU pixels: no irr, nonirr scaled by urban frac
            irr[lu_urban_only] = 0.0
            nonirr[lu_urban_only] = predictions[lu_urban_only] * uf_vals[lu_urban_only]
            # Canal ag without wells: full prediction for irr, nonirr = 0
            if canal_ag_no_wells.any():
                irr[canal_ag_no_wells] = predictions[canal_ag_no_wells]
                nonirr[canal_ag_no_wells] = 0.0
    else:
        irr_frac = year_df['annual_irr_fraction'].values \
            if 'annual_irr_fraction' in year_df.columns else None
        if irr_frac is not None and well_dens is not None:
            irr_frac = focal_fill_irr_fraction(
                irr_frac, well_dens, raster_shape, valid_mask,
            )
        irr = predictions * irr_frac if irr_frac is not None else predictions.copy()
        nonirr = predictions - irr

    # ---- Urban-fraction weighting for non-irrigation ----
    # Outside AMA/INAs (GW_Basin_Type 2) and in rural AMAs/INAs with
    # minimal urban footprint, non-irrigation is scaled by urban-area
    # fraction to zero out rural pixels with no real M&I demand.
    # Urban AMAs (Phoenix, Tucson, Pinal, Prescott, Santa Cruz) keep
    # full non-irrigation at all pixels.
    basin_type = year_df['GW_Basin_Type'].values \
        if 'GW_Basin_Type' in year_df.columns else None
    basin_names = year_df['GW_Basin'].values \
        if 'GW_Basin' in year_df.columns else None
    urban_frac_col = year_df['URBAN'].values \
        if 'URBAN' in year_df.columns else None
    if basin_type is not None and urban_frac_col is not None:
        is_other = basin_type == 2
        is_rural_ama = np.isin(basin_names, list(RURAL_AMA_INA)) \
            if basin_names is not None else np.zeros(len(predictions), dtype=bool)
        apply_uf = (is_other | is_rural_ama) & ~lu_only
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        # Urban scaling exemption: α=0 through 1969 (urban scaling active
        # → NonIrr at rural AMAs scaled down by urban_frac), then ramps
        # 1970→1980 (exemption switches on). Pushing the start from 1965
        # → 1970 suppresses the 1965→1970 NonIrr jump that was making
        # 1970 total over-predict by ~1.4 MAF. Urban development in
        # rural AMAs only materialized through the 1970s, justifying the
        # later exemption-activation window.
        if year < 1970:
            _urb_alpha = 0.0
        elif year < 1980:
            _urb_alpha = (year - 1970) / (1980 - 1970)
        elif year <= 1980:
            _urb_alpha = 1.0
        elif year < _RAMP_DN_END:
            _urb_alpha = 1.0 - (year - _RAMP_DN_START) / (_RAMP_DN_END - _RAMP_DN_START)
        else:
            _urb_alpha = 0.0

        if _urb_alpha > 0:
            scaled = nonirr[apply_uf] * uf[apply_uf]
            nonirr[apply_uf] = scaled * (1 - _urb_alpha) + nonirr[apply_uf] * _urb_alpha
        else:
            nonirr_before = nonirr[apply_uf].copy()
            nonirr[apply_uf] = nonirr[apply_uf] * uf[apply_uf]
            # Post-CAP: at rural pixels, recover the urban-scaled-away
            # nonirr volume as irrigation, scaled by crop_frac.  Desert
            # pixels (crop_frac=0) get nothing; partial-ag pixels get
            # proportional recovery.
            if year >= _RAMP_DN_END and crop_frac is not None:
                cf_apply = np.clip(np.nan_to_num(
                    crop_frac[apply_uf], nan=0.0,
                ), 0, 1)
                rural_residual = nonirr_before - nonirr[apply_uf]
                irr[apply_uf] = irr[apply_uf] + rural_residual * cf_apply

    # ---- Zero-surface-water mask: per-pixel canal-weighted streamflow ----
    # Where canal-weighted streamflow is zero, there is no canal-delivered
    # surface water at the pixel → force 100% GW.  Unlike regular
    # streamflow (which is uniform per-watershed and bleeds into desert
    # basins like Butler Valley), canal-weighted streamflow is precisely
    # located at canal-served pixels with no bleed, so a per-pixel check
    # is sufficient (no basin-median override needed).
    cw_streamflow = year_df['canal_weighted_streamflow_mm'].values \
        if 'canal_weighted_streamflow_mm' in year_df.columns else None
    zero_sw_mask = None
    if cw_streamflow is not None:
        ag_canal_service = has_smooth_canal & (crop_frac_filter > 0) \
            if crop_frac_filter is not None \
            else np.zeros(len(predictions), dtype=bool)
        # Post-GMA: also exempt pixels within the smoothed canal reach
        # from the zero-SW constraint.  CAP/SRP distribution networks
        # deliver SW to pixels without direct canal-weighted streamflow.
        in_canal_reach = (cw_smooth_1d > 0) & (crop_frac_filter > 0) \
            if year >= GMA_YEAR else np.zeros(len(predictions), dtype=bool)
        zero_sw_mask = (cw_streamflow == 0) & ~ag_canal_service & ~in_canal_reach
        if not np.any(zero_sw_mask):
            zero_sw_mask = None

    # ---- CW-streamflow-weighted SW POD smoothing ----
    # Weight SW rights density by canal-weighted streamflow (delivery
    # capacity), then Gaussian-smooth to spread influence over the canal
    # service area.  A POD at a major canal headgate (cw_sf = 1000 mm)
    # gets 1000× more influence than a POD at a dry wash.  This replaces
    # the separate cw_norm boost — canal delivery information is now
    # embedded in the smoothed SW density itself.
    #
    # σ ramp reflects canal-infrastructure maturation:
    #   1938–1970: σ = _SW_SIGMA_PRE (2.0, tighter — early SRP reach,
    #              pre-CAP canal laterals are short and local)
    #   1960–1985: linear ramp _SW_SIGMA_PRE → 16.0 (earlier start
    #              spreads SW earlier so 1985–1995 SW anchors get
    #              enough service-area coverage)
    #   1985+:     σ = 16 (post-CAP full service areas — matches the
    #              USGS SW anchors 1990+ better than σ=8)
    _SW_SIGMA_PRE = 2.0
    _SW_SIGMA_RAMP_START = 1960
    _SW_SIGMA_RAMP_END = 1985
    _SW_SIGMA_POST = 16.0
    if year <= _SW_SIGMA_RAMP_START:
        _sw_sigma = _SW_SIGMA_PRE
    elif year >= _SW_SIGMA_RAMP_END:
        _sw_sigma = _SW_SIGMA_POST
    else:
        _frac = (year - _SW_SIGMA_RAMP_START) / float(
            _SW_SIGMA_RAMP_END - _SW_SIGMA_RAMP_START,
        )
        _sw_sigma = _SW_SIGMA_PRE + _frac * (_SW_SIGMA_POST - _SW_SIGMA_PRE)

    def _smooth_sw_density(sw_dens, cw_sf):
        """Weight SW POD density by cw_streamflow and Gaussian-smooth."""
        sw_grid = np.zeros(raster_shape, dtype=np.float64)
        cw_grid = np.zeros(raster_shape, dtype=np.float64)
        sw_grid.ravel()[valid_mask] = np.nan_to_num(sw_dens, nan=0.0)
        cw_grid.ravel()[valid_mask] = np.nan_to_num(cw_sf, nan=0.0)
        weighted = sw_grid * cw_grid
        smoothed = _gaussian_filter(weighted, sigma=_sw_sigma)
        return smoothed.ravel()[valid_mask]

    cw_sf_vals = cw_streamflow if cw_streamflow is not None else np.zeros(len(predictions))

    # ---- GW / SW split of irrigation (CW-weighted density ratio) ----
    irr_wd = year_df['irr_well_density'].values \
        if 'irr_well_density' in year_df.columns else None
    if year < GMA_YEAR and irr_wd_1981 is not None:
        irr_wd = irr_wd_1981.copy()
    irr_swd = year_df['irr_sw_rights_density'].values \
        if 'irr_sw_rights_density' in year_df.columns else None

    gw_weight = _era_gw_weight(year)

    if irr_wd is not None and irr_swd is not None:
        irr_wd = np.nan_to_num(irr_wd, nan=0.0)
        irr_swd_smooth = _smooth_sw_density(irr_swd, cw_sf_vals)
        weighted_wd = gw_weight * irr_wd
        denom = weighted_wd + irr_swd_smooth
        with np.errstate(invalid='ignore', divide='ignore'):
            irr_gw_frac = np.where(denom > 0, weighted_wd / denom, 1.0)
        irr_gw_frac = np.clip(irr_gw_frac, 0, 1)
        # Force 100% GW where canal-weighted streamflow is zero
        if zero_sw_mask is not None:
            irr_gw_frac[zero_sw_mask] = 1.0
        irr_gw = irr * irr_gw_frac
        irr_sw = irr - irr_gw
    else:
        irr_gw = irr.copy()
        irr_sw = np.zeros_like(irr)

    # ---- GW / SW split of non-irrigation (CW-weighted density ratio) ----
    nonirr_wd = year_df['nonirr_well_density'].values \
        if 'nonirr_well_density' in year_df.columns else None
    if year < GMA_YEAR and nonirr_wd_1981 is not None:
        nonirr_wd = nonirr_wd_1981.copy()
    nonirr_swd = year_df['nonirr_sw_rights_density'].values \
        if 'nonirr_sw_rights_density' in year_df.columns else None

    if nonirr_wd is not None and nonirr_swd is not None:
        nonirr_wd = np.nan_to_num(nonirr_wd, nan=0.0)
        nonirr_swd_smooth = _smooth_sw_density(nonirr_swd, cw_sf_vals)
        weighted_nonirr_wd = gw_weight * nonirr_wd
        denom = weighted_nonirr_wd + nonirr_swd_smooth
        with np.errstate(invalid='ignore', divide='ignore'):
            nonirr_gw_frac = np.where(denom > 0, weighted_nonirr_wd / denom, 1.0)
        nonirr_gw_frac = np.clip(nonirr_gw_frac, 0, 1)
        # Force 100% GW where canal-weighted streamflow is zero
        if zero_sw_mask is not None:
            nonirr_gw_frac[zero_sw_mask] = 1.0
        nonirr_gw = nonirr * nonirr_gw_frac
        nonirr_sw = nonirr - nonirr_gw
    else:
        nonirr_gw = nonirr.copy()
        nonirr_sw = np.zeros_like(nonirr)

    # ---- LU-only GW/SW override ----
    # Pre-1981: LU-only pixels are always GW (pre-CAP era was
    # GW-dominated; smoothed canal signal overestimates SW).
    # Post-1980: allow SW at LU-only pixels near canals.
    if lu_only.any():
        if year <= 1980:
            irr_gw[lu_only] = irr[lu_only]
            irr_sw[lu_only] = 0.0
            nonirr_gw[lu_only] = nonirr[lu_only]
            nonirr_sw[lu_only] = 0.0
        else:
            canal_influence = cw_smooth_1d[lu_only] > 0
            irr_sw[lu_only] = np.where(canal_influence, irr[lu_only], 0.0)
            irr_gw[lu_only] = np.where(canal_influence, 0.0, irr[lu_only])
            nonirr_gw[lu_only] = nonirr[lu_only]
            nonirr_sw[lu_only] = 0.0

    # ---- Pre-GMA canal_ag_no_wells GW/SW override ----
    # Pre-1981: canal-served ag pixels without registered wells had
    # unregistered GW wells alongside canal SW deliveries.  The density
    # ratio assigns 100% SW (well_density = 0), but USBR data shows
    # ~67% GW statewide.  Override with a GW-dominant split.
    if canal_ag_no_wells.any() and year < GMA_YEAR:
        pre_gma_gw_frac = 0.67
        irr_gw[canal_ag_no_wells] = irr[canal_ag_no_wells] * pre_gma_gw_frac
        irr_sw[canal_ag_no_wells] = irr[canal_ag_no_wells] * (1 - pre_gma_gw_frac)

    # ---- SW delivery residual ----
    # After urban-density scaling, the partitioned total (irr + nonirr)
    # may be less than the ML-predicted total. The residual is recovered
    # at ag canal pixels. Pre-GMA: routed to Irr_GW (unregistered well
    # pumping). Post-GMA: routed to Irr_SW (canal deliveries).
    if year >= 1960:
        partitioned_total = irr + nonirr
        residual = np.maximum(
            np.nan_to_num(predictions, nan=0.0)
            - np.nan_to_num(partitioned_total, nan=0.0),
            0.0,
        )
        cf_residual = year_df['annual_crop_fraction'].values \
            if 'annual_crop_fraction' in year_df.columns \
            else np.zeros(len(predictions))
        cf_residual = np.nan_to_num(cf_residual, nan=0.0)
        has_well_residual = ~((well_dens == 0) | np.isnan(well_dens)) \
            if well_dens is not None else np.zeros(len(predictions), dtype=bool)
        delivery_residual = np.where(
            ((cw_smooth_1d >= 1) & (cf_residual > 0))
            | (has_well_residual & (cw_smooth_1d > 0) & (cf_residual > 0)),
            residual, 0.0,
        )
        if np.any(delivery_residual > 0):
            irr_sw = irr_sw + delivery_residual
            irr = irr_gw + irr_sw
            nonirr = nonirr_gw + nonirr_sw

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


# Default λ values for SW capture index (meters)
LAMBDA_LOWER = 5.0    # conservative: only very shallow connections
LAMBDA_CENTRAL = 10.0  # moderate: typical western US alluvial aquifers
LAMBDA_UPPER = 20.0    # liberal: deeper basin-fill connections


def compute_sw_capture_index(
        total_gw: np.ndarray,
        sigma_gw: np.ndarray | None,
        wtd_m: np.ndarray,
        cw_streamflow: np.ndarray,
        raster_shape: tuple,
        valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the Surface Water Capture Index per pixel.

    Estimates what fraction of GW pumping depletes surface water,
    based on hydraulic connectivity (exponential decay with water table
    depth) and surface water availability (focal-max normalized
    canal-weighted streamflow).

    ``capture_fraction = exp(-wtd / λ) × cw_norm``

    Three λ values (5, 10, 20 m) produce lower/central/upper bounds
    on the connectivity scale.  When ``sigma_gw`` is supplied (the
    per-pixel σ_total from the 5-component UQ framework, in mm),
    the volume bounds combine the λ envelope with σ_GW propagation
    via the asymmetric form

        ``gw_lower = max(gw − 1.96 σ, 0)``
        ``gw_upper = gw + 1.96 σ``
        ``vol_lower = gw_lower × cf_lower``  (narrow connectivity)
        ``vol_central = gw × cf_central``    (central)
        ``vol_upper = gw_upper × cf_upper``  (wide connectivity)

    so the Lower/Upper bounds encode the combined 95 % CI of the
    capture volume under both the pumping-side uncertainty and the
    connectivity-scale uncertainty.  The production pipeline always
    supplies ``sigma_gw`` via
    ``uncertaintyops.compute_sw_capture_with_sigma``; ``None`` is
    retained only as an escape hatch for callers that need a λ-only
    envelope.

    Args:
        total_gw: 1-D array of Total_GW withdrawal (mm) for valid pixels.
        sigma_gw: 1-D array of σ_total for Total_GW (mm), or None for
            a λ-only envelope without σ propagation.
        wtd_m: 1-D array of water table depth (m) for valid pixels.
        cw_streamflow: 1-D array of canal-weighted streamflow (mm) for
            valid pixels.
        raster_shape: (rows, cols) of the full raster grid.
        valid_mask: Boolean mask of valid pixels (ravelled).

    Returns:
        dict with keys:
            ``Capture_Fraction_Lower``, ``Capture_Fraction_Central``,
            ``Capture_Fraction_Upper`` — dimensionless [0, 1]
            ``Capture_Volume_Lower``, ``Capture_Volume_Central``,
            ``Capture_Volume_Upper`` — captured SW volume in mm
            (named ``Capture_Volume`` rather than ``SW_Capture`` so that
            downstream consumers prefixing the key with a category like
            ``Total_SW`` don't end up with a duplicated ``SW`` token)

    References:
        Condon & Maxwell (2019), de Graaf et al. (2019),
        Barlow & Leake (2012).
    """
    # Focal-max normalized canal-weighted streamflow [0, 1]
    cw_norm = compute_sw_fraction(cw_streamflow, raster_shape, valid_mask)

    # Water table depth — ensure non-negative
    wtd = np.clip(np.nan_to_num(wtd_m, nan=0.0), 0, None)

    # Connectivity at three λ values
    conn_lower = np.exp(-wtd / LAMBDA_LOWER)
    conn_central = np.exp(-wtd / LAMBDA_CENTRAL)
    conn_upper = np.exp(-wtd / LAMBDA_UPPER)

    # Capture fraction = connectivity × SW availability
    cf_lower = conn_lower * cw_norm
    cf_central = conn_central * cw_norm
    cf_upper = conn_upper * cw_norm

    # Volume of SW captured (mm)
    gw = np.maximum(np.nan_to_num(total_gw, nan=0.0), 0)

    if sigma_gw is not None:
        sigma = np.maximum(np.nan_to_num(sigma_gw, nan=0.0), 0)
        gw_lower = np.maximum(gw - 1.96 * sigma, 0)
        gw_upper = gw + 1.96 * sigma
    else:
        gw_lower = gw
        gw_upper = gw

    vol_lower = gw_lower * cf_lower
    vol_central = gw * cf_central
    vol_upper = gw_upper * cf_upper

    return {
        'Capture_Fraction_Lower': cf_lower,
        'Capture_Fraction_Central': cf_central,
        'Capture_Fraction_Upper': cf_upper,
        'Capture_Volume_Lower': vol_lower,
        'Capture_Volume_Central': vol_central,
        'Capture_Volume_Upper': vol_upper,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CAP/SRP delivery-calibrated GW/SW scaling
# ═════════════════════════════════════════════════════════════════════════════


def compute_basin_delivery_ratios(
    obs_basin_yearly: dict[str, dict[int, float]],
    ml_basin_yearly: dict[str, dict[int, float]],
    max_ratio: float = 10.0,
) -> dict[str, float]:
    """Compute time-averaged delivery ratios per basin.

    For each basin present in both dicts, finds the overlapping years
    and computes ``ratio = mean(observed) / mean(ml_total_sw)``.

    Args:
        obs_basin_yearly: ``{basin: {year: delivery_AF}}`` from
            CAP/SRP delivery records.
        ml_basin_yearly: ``{basin: {year: volume_AF}}`` from ML
            Total_SW predictions.
        max_ratio: Cap on the ratio to prevent extreme amplification.

    Returns:
        ``{basin: ratio}`` for basins with valid data.
    """
    ratios: dict[str, float] = {}
    for basin, obs_years in obs_basin_yearly.items():
        ml_years = ml_basin_yearly.get(basin, {})
        common = sorted(set(obs_years.keys()) & set(ml_years.keys()))
        if not common:
            continue
        mean_obs = float(np.mean([obs_years[yr] for yr in common]))
        mean_ml = float(np.mean([ml_years[yr] for yr in common]))
        if mean_ml <= 0:
            logger.warning(
                'Basin %s: ML Total_SW mean is zero over common years; '
                'skipping delivery scaling', basin,
            )
            continue
        ratio = mean_obs / mean_ml
        if ratio > max_ratio:
            logger.warning(
                'Basin %s: delivery ratio %.2f exceeds cap %.1f; '
                'clamping', basin, ratio, max_ratio,
            )
            ratio = max_ratio
        ratios[basin] = ratio
        logger.info(
            '  Delivery ratio %s: %.3f '
            '(obs=%.0f AF, ml=%.0f AF, n=%d common years)',
            basin, ratio, mean_obs, mean_ml, len(common),
        )
    return ratios


def apply_basin_sw_scaling(
    cat_predictions: dict[str, np.ndarray],
    pixel_basins: np.ndarray,
    basin_ratios: dict[str, float],
) -> dict[str, np.ndarray]:
    """Apply basin-level delivery-ratio scaling to the GW/SW partition.

    Modifies the 8-category dict **in-place** after
    ``partition_predictions()`` returns. For each basin with a delivery
    ratio, scales ``Total_SW *= ratio`` (clamped to ``<= Total``) and
    cascades through all sub-categories to maintain conservation:

    - ``Total_GW = Total - Total_SW``
    - ``Irrigation_SW *= sw_factor``; ``Non_Irrigation_SW *= sw_factor``
    - ``Irrigation_GW = Irrigation - Irrigation_SW``
    - ``Non_Irrigation_GW = Non_Irrigation - Non_Irrigation_SW``

    Pre-delivery temporal zeroing is handled **upstream** by the
    temporally-masked canal density and canal-weighted streamflow
    rasters (which produce zero canal features before a canal's
    first-delivery year, causing ``partition_predictions()`` to set
    ``gw_frac = 1.0`` at those pixels). This function only corrects
    the **magnitude** of SW via observed delivery ratios.

    Basins not in *basin_ratios* keep their unscaled density-ratio
    partitioning.

    Args:
        cat_predictions: Mutable dict of 8 category arrays from
            ``partition_predictions()``.
        pixel_basins: Per-pixel basin name array (same length as the
            category arrays).
        basin_ratios: ``{basin: ratio}`` from
            ``compute_basin_delivery_ratios()``.

    Returns:
        The same ``cat_predictions`` dict (mutated in-place).
    """
    total = cat_predictions['Irrigation'] + cat_predictions['Non_Irrigation']

    for basin, ratio in basin_ratios.items():
        if ratio == 1.0:
            continue
        bmask = pixel_basins == basin
        if not bmask.any():
            continue

        old_sw = cat_predictions['Total_SW'][bmask].copy()
        new_sw = np.minimum(old_sw * ratio, total[bmask])
        new_sw = np.maximum(new_sw, 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            sw_factor = np.where(old_sw > 0, new_sw / old_sw, 0.0)

        cat_predictions['Total_SW'][bmask] = new_sw
        cat_predictions['Total_GW'][bmask] = total[bmask] - new_sw

        cat_predictions['Irrigation_SW'][bmask] *= sw_factor
        cat_predictions['Non_Irrigation_SW'][bmask] *= sw_factor

        cat_predictions['Irrigation_GW'][bmask] = (
            cat_predictions['Irrigation'][bmask]
            - cat_predictions['Irrigation_SW'][bmask]
        )
        cat_predictions['Non_Irrigation_GW'][bmask] = (
            cat_predictions['Non_Irrigation'][bmask]
            - cat_predictions['Non_Irrigation_SW'][bmask]
        )

    return cat_predictions
