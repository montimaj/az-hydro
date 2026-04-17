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
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bare-baseline phantom-wells helper (random infill from 2024 registry,
# basin-gated).  Replaces the prior PHANTOM_K / WD_2024 / _override_alpha /
# ag_era_cap / post_cap_blend stack inside partition_predictions.  The
# legacy helper functions (_phantom_k, _wd_2024_active, _wd_2024_scale,
# _phantom_canal_include, _phantom_agri_threshold) are retained below
# because the ML-feature override in pipeline.py still calls them; they
# are no longer used by partition_predictions itself.
# ─────────────────────────────────────────────────────────────────────────────

PHANTOM_INFILL_START = 1938       # don't apply infill pre-1938 — earlier
                                  # years lack GEE LULC and the 2024 pump-
                                  # capacity registry doesn't reflect those
                                  # eras (mostly modern industrial wells)
PHANTOM_INFILL_END = 1983         # 1984+ is the model training era — the
                                  # year-specific registry is the same data
                                  # the ML model trained on, so injecting
                                  # random phantom wells there is a no-op
                                  # at best and a perturbation at worst
PHANTOM_LULC_THRESHOLD = 0.20     # ag/urban fraction above which a pixel
                                  # is considered "developed" and eligible
                                  # for random-infill from the 2024 registry
                                  # (was 0.05 — bumped to keep marginal
                                  # crop/urban pixels out of the infill)

# Year-dependent infill magnitude scale: pre-1955 the ag-well drilling
# era hadn't fully matured (USGS Circ 398 marks ~1955 as the drilling
# peak), so even at eligible pixels the phantom value should be small.
# Linear ramp 0.10 (1938) → 1.0 (1955+).  Applied as a multiplier on
# the sampled wd_2024 value inside _random_infill_phantom.
PHANTOM_INFILL_SCALE_RAMP_END = 1955
PHANTOM_INFILL_SCALE_MIN = 0.10


def _phantom_infill_scale(year: int) -> float:
    """Magnitude scale on infilled values: ramps 0.10 (1921) → 1.0 (1945+)."""
    if year < PHANTOM_INFILL_START:
        return 0.0
    if year >= PHANTOM_INFILL_SCALE_RAMP_END:
        return 1.0
    frac = (year - PHANTOM_INFILL_START) / float(
        PHANTOM_INFILL_SCALE_RAMP_END - PHANTOM_INFILL_START,
    )
    return PHANTOM_INFILL_SCALE_MIN + frac * (1.0 - PHANTOM_INFILL_SCALE_MIN)


def _random_infill_phantom(
        year: int,
        year_wd: np.ndarray,
        wd_2024: np.ndarray | None,
        crop_frac: np.ndarray | None,
        urban_frac: np.ndarray | None,
        basin_names: np.ndarray | None,
        purpose: str,
) -> np.ndarray:
    """Infill missing well-density pixels with random samples drawn from
    the non-zero ``wd_2024`` distribution **of the same GW basin**.

    Active for ``PHANTOM_INFILL_START <= year <= PHANTOM_INFILL_END``
    (1938–1983).  Pre-1938 there is no GEE LULC and the 2024 pump-
    capacity registry doesn't reflect that era; 1984+ is the model
    training era where the year-specific registry is the same data
    the ML trained against, so adding random phantoms there is at
    best a no-op and at worst a perturbation.  A pixel is eligible
    for infill when:
      - year-specific ``year_wd`` is 0 or NaN, AND
      - ``crop_frac >= PHANTOM_LULC_THRESHOLD`` OR
        ``urban_frac >= PHANTOM_LULC_THRESHOLD``.

    Sampling is bootstrap from ``wd_2024[wd_2024 > 0]`` within the same
    basin, with replacement, seeded by ``(year, basin, purpose)`` so the
    three category calls (purpose=``'total'`` / ``'irr'`` / ``'nonirr'``)
    don't pull correlated indices.

    Basins with no 2024 wells (e.g. undeveloped Coconino Plateau) get
    no infill — phantom values do not leak across basin boundaries.
    """
    out = np.where(np.isnan(year_wd), 0.0, year_wd).astype(np.float64, copy=True)
    # Infill disabled — return year-specific well density unchanged.
    # See git history for the previous Pass 1 (bootstrap) and Pass 2
    # (basin-median) implementations.  Infill was promoting LU-only
    # marginal crop pixels into the standard partition (post-infill
    # has_well = True), which caused 1960 totals to spike from 5.58
    # to 11.25 MAF as urban-only LU pixels gained pred × (1−uf) of
    # additional volume.  Keeping LU-only logic intact at the partition
    # level instead.
    return out
    if year < PHANTOM_INFILL_START:
        return out
    if wd_2024 is None or basin_names is None:
        return out
    crop = (
        np.nan_to_num(crop_frac, nan=0.0) if crop_frac is not None
        else np.zeros_like(out)
    )
    urban = (
        np.nan_to_num(urban_frac, nan=0.0) if urban_frac is not None
        else np.zeros_like(out)
    )
    scale = _phantom_infill_scale(year)
    if scale <= 0.0:
        return out
    wd_2024_filled = np.nan_to_num(wd_2024, nan=0.0)
    purpose_seed = hash(purpose) & 0xFFFFFFFF
    basin_series = pd.Series(basin_names)

    # Pass 1 (bootstrap): pixels with substantial crop or urban
    # (cf >= 0.20 OR uf >= 0.20) get a random sample from the basin's
    # non-zero wd_2024 distribution.  Active only within the original
    # PHANTOM_INFILL window (1938–1983) — post-1983 the year-specific
    # registry is the model training data and shouldn't be perturbed
    # by random samples.
    do_bootstrap = year <= PHANTOM_INFILL_END
    needs_bootstrap = (out == 0.0) & (
        (crop >= PHANTOM_LULC_THRESHOLD) | (urban >= PHANTOM_LULC_THRESHOLD)
    ) if do_bootstrap else None
    # Pass 2 (LU-only basin-max): remaining pixels with ANY crop/urban
    # (cf > 0 OR uf > 0) but no well get filled with the basin's MAX
    # wd_2024 value × scale.  Active for ALL years >= PHANTOM_INFILL_START
    # (including 1984+), giving LU-only pixels a deterministic
    # well-density signal so the partition treats them as has_well
    # rather than going through the LU-only branch.
    for b in basin_series.dropna().unique():
        b_mask = (basin_names == b)
        b_pool = wd_2024_filled[b_mask & (wd_2024_filled > 0)]
        if b_pool.size == 0:
            continue  # basin has no 2024 wells — leave year-specific as-is
        # Pass 1: bootstrap (1938–1983 only)
        if do_bootstrap:
            b_needs_boot = b_mask & needs_bootstrap
            if b_needs_boot.any():
                basin_seed = hash(str(b)) & 0xFFFFFFFF
                seed = ((year * 1_000_003 + basin_seed) ^ purpose_seed) & 0xFFFFFFFF
                rng = np.random.default_rng(seed=seed)
                out[b_needs_boot] = (
                    rng.choice(b_pool, size=int(b_needs_boot.sum()), replace=True)
                    * scale
                )
        # Pass 2: basin-median for LU-only marginal pixels (all years
        # >= PHANTOM_INFILL_START).  Basin median represents a "typical
        # active well pixel" — using max would inject the heaviest
        # well density everywhere (basin max ~100s of wells/pixel vs
        # median ~1–5).  Must come after Pass 1 so out[b_needs_boot]
        # is non-zero.
        b_needs_lu = b_mask & (out == 0.0) & (
            (crop > 0) | (urban > 0)
        )
        if b_needs_lu.any():
            basin_median = float(np.median(b_pool))
            out[b_needs_lu] = basin_median * scale
    return out


def _smooth_sw_density(
        sw_dens: np.ndarray,
        cw_sf: np.ndarray,
        raster_shape: tuple,
        valid_mask: np.ndarray,
        sigma: float,
) -> np.ndarray:
    """Weight an SW-rights-density array by canal-weighted streamflow
    (delivery capacity) and Gaussian-smooth the product on the raster
    grid, then return the smoothed values at valid pixels.

    Used by ``partition_predictions`` for both Irr and NonIrr GW/SW
    splits.  A POD at a major canal headgate (cw_sf large) gets much
    more influence than a POD at a dry wash, then the smoothing spreads
    that influence across the canal service area.
    """
    sw_grid = np.zeros(raster_shape, dtype=np.float64)
    cw_grid = np.zeros(raster_shape, dtype=np.float64)
    sw_grid.ravel()[valid_mask] = np.nan_to_num(sw_dens, nan=0.0)
    cw_grid.ravel()[valid_mask] = np.nan_to_num(cw_sf, nan=0.0)
    weighted = sw_grid * cw_grid
    smoothed = gaussian_filter(weighted, sigma=sigma)
    return smoothed.ravel()[valid_mask]

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

# All 7 AMAs + 3 INAs use the year-specific irr_capacity_fraction
# directly (no urban_frac or Option C override).  AMAs/INAs are
# mandatorily metered under GMA so the 2024 PUMPRATE-weighted ratio is
# reliable.  Only Outside-AMA basins use the urban-fraction-derived
# overrides (urban_frac / 1-urban_frac / Option C).
AMA_BASINS = frozenset({
    'PHOENIX AMA', 'TUCSON AMA', 'PINAL AMA', 'PRESCOTT AMA',
    'SANTA CRUZ AMA', 'DOUGLAS AMA', 'WILLCOX AMA',
    'HARQUAHALA INA', 'HUALAPAI VALLEY INA', 'JOSEPH CITY INA',
})

# Adaptive irr_frac floor: at every pixel, irrigation share is at
# least as large as the cropland share at that pixel.  Captures the
# physical intuition that mapped cropland implies at least that much
# irrigation pumping, and simultaneously fixes the outlier-over-
# cropland issue where the 2024 registry misclassifies old ag wells
# as M&I (those pixels still have high crop_frac → adaptive floor
# lifts irr_frac).  Replaces the per-basin RURAL_AMA_INA_FLOOR set.


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
GW_WEIGHT_PRE_GMA = 2.0
GW_WEIGHT_POST_CAP = 0.2


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
        sw_smooth_sigma: float = 2.0,
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
            Default 2.0 (~4 km radius at 2 km resolution).
        year (int): Prediction year. Canal-only pixels are kept only for
            years >= ``CANAL_PREDICTOR_START`` (1938) when GEE predictor
            data is available.

    Returns:
        dict[str, np.ndarray]: Mapping of category name to 1-D prediction array (same length
            as *predictions*).
    """
    # ---- Pull per-pixel inputs from year_df ----
    well_dens_yr = year_df['well_density'].values \
        if 'well_density' in year_df.columns else None
    canal_dens = year_df['canal_density'].values \
        if 'canal_density' in year_df.columns else None
    cw_streamflow_raw = year_df['canal_weighted_streamflow_mm'].values \
        if 'canal_weighted_streamflow_mm' in year_df.columns else None
    crop_frac_col = year_df['annual_crop_fraction'].values \
        if 'annual_crop_fraction' in year_df.columns else None
    urban_dens_col = year_df['URBAN'].values \
        if 'URBAN' in year_df.columns else None
    urban_frac_col = year_df['annual_urban_fraction'].values \
        if 'annual_urban_fraction' in year_df.columns else None
    basin_names = year_df['GW_Basin'].values \
        if 'GW_Basin' in year_df.columns else None
    irr_wd_yr = year_df['irr_well_density'].values \
        if 'irr_well_density' in year_df.columns else None
    nonirr_wd_yr = year_df['nonirr_well_density'].values \
        if 'nonirr_well_density' in year_df.columns else None
    irr_swd = year_df['irr_sw_rights_density'].values \
        if 'irr_sw_rights_density' in year_df.columns else None
    nonirr_swd = year_df['nonirr_sw_rights_density'].values \
        if 'nonirr_sw_rights_density' in year_df.columns else None

    # ---- Random-infill phantom wells (year >= PHANTOM_INFILL_START) ----
    # At ag/urban pixels missing from the year-specific registry, draw a
    # random well-density value from the same-basin 2024 distribution.
    # Pre-1925 the helper degenerates to the year-specific registry.
    if well_dens_yr is None:
        well_dens_yr = np.zeros(len(predictions), dtype=np.float64)
    if irr_wd_yr is None:
        irr_wd_yr = np.zeros(len(predictions), dtype=np.float64)
    if nonirr_wd_yr is None:
        nonirr_wd_yr = np.zeros(len(predictions), dtype=np.float64)

    well_dens = _random_infill_phantom(
        year, well_dens_yr, wd_1981,
        crop_frac_col, urban_frac_col, basin_names, purpose='total',
    )
    irr_wd = _random_infill_phantom(
        year, irr_wd_yr, irr_wd_1981,
        crop_frac_col, urban_frac_col, basin_names, purpose='irr',
    )
    nonirr_wd = _random_infill_phantom(
        year, nonirr_wd_yr, nonirr_wd_1981,
        crop_frac_col, urban_frac_col, basin_names, purpose='nonirr',
    )
    has_well = well_dens > 0

    # ---- Smooth canal-weighted streamflow to identify canal service area ----
    has_smooth_canal = np.zeros(len(predictions), dtype=bool)
    cw_smooth_1d = np.zeros(len(predictions), dtype=np.float64)
    crop_frac_filter = np.zeros(len(predictions), dtype=np.float64)
    if cw_streamflow_raw is not None and year >= CANAL_PREDICTOR_START:
        cw_grid = np.zeros(raster_shape, dtype=np.float64)
        cw_grid.ravel()[valid_mask] = np.nan_to_num(cw_streamflow_raw, nan=0.0)
        cw_smoothed = gaussian_filter(cw_grid, sigma=2.0)
        cw_smooth_1d = cw_smoothed.ravel()[valid_mask]
        crop_frac_filter = np.clip(
            np.nan_to_num(crop_frac_col, nan=0.0), 0, 1,
        ) if crop_frac_col is not None else np.zeros(len(predictions))
        has_smooth_canal = (cw_smooth_1d > 0) & (crop_frac_filter > 0)

    # ---- Pixel retention ----
    _has_crop_any = (np.nan_to_num(crop_frac_col, nan=0.0) > 0) \
        if crop_frac_col is not None else np.zeros(len(predictions), dtype=bool)
    _has_urban_any = (np.nan_to_num(urban_frac_col, nan=0.0) >= 0.2) \
        if urban_frac_col is not None else np.zeros(len(predictions), dtype=bool)
    # AGRI-extension retention at outside-AMA basins (peak years only).
    # AGRI extension makes sense when the well registry is incomplete
    # — peak USGS pumping years (1951–55, 1970–80).  Post-CAP (1985+)
    # the registry is more or less complete, so we don't extend via
    # AGRI; instead, use the explicit per-pixel LULC logic below.
    #   - 1951–55 and 1975–80 (peak USGS pumping):  AGRI > 0.01
    #   - 1970–74:                                   AGRI > 0.10
    #   - other years (incl. 1985+):                 no extension
    _is_loose_agri = (1951 <= year <= 1955 or 1975 <= year <= 1980)
    _is_std_agri   = (1970 <= year <= 1974)
    _agri_threshold_yr = 0.02 if _is_loose_agri else 0.10
    _agri_extension_active = _is_loose_agri or _is_std_agri
    if (_agri_extension_active and 'AGRI' in year_df.columns
            and basin_names is not None):
        _agri_retain = np.clip(np.nan_to_num(
            year_df['AGRI'].values, nan=0.0,
        ), 0, 1)
        _is_ama_retain = np.isin(basin_names, list(AMA_BASINS))
        _has_crop_any = _has_crop_any | (
            (_agri_retain > _agri_threshold_yr) & ~_is_ama_retain
        )
    # Pre-1945: retention uses the year-specific registry intersected
    # with the LULC mask (year-specific wells only, AND the pixel must
    # have nearby crop or urban).  LU-only branches are dropped entirely
    # and phantom infill is ignored so 1900–1944 totals only reflect
    # actually-registered wells co-located with developed land.  The
    # 1938–1944 cliff (driven by the LULC-frozen-at-1938 snapshot) is
    # gated out of retention by this rule.
    if year < 1945:
        has_well_for_retention = (well_dens_yr > 0) & (
            _has_crop_any | _has_urban_any
        )
        has_crop = np.zeros(len(predictions), dtype=bool)
        has_urban = np.zeros(len(predictions), dtype=bool)
    else:
        has_well_for_retention = has_well       # post-infill, includes phantoms
        has_crop = _has_crop_any
        has_urban = _has_urban_any
    keep = has_well_for_retention | has_smooth_canal | has_crop | has_urban
    predictions = predictions.copy()
    predictions[~keep] = np.nan

    # LU-only pixels: crop/urban-only (no wells, no canal service area)
    lu_only = ~has_well_for_retention & ~has_smooth_canal & (has_crop | has_urban)

    # ---- Irr / NonIrr split using pump-capacity-weighted irrigation
    # fraction.  irr_capacity_fraction = (sum of PUMPRATE for IRRIGATION
    # wells) / (sum of PUMPRATE for all active wells) per pixel — gives
    # a physically meaningful withdrawal split that accounts for wells
    # in pixel A irrigating fields in pixel B (the pump capacity is
    # at A; the IrrMapper area at B never enters the partition).  Falls
    # back to the area-based annual_irr_fraction only if the capacity
    # column is missing.
    irr_cap_col = year_df['irr_capacity_fraction'].values \
        if 'irr_capacity_fraction' in year_df.columns else None
    if irr_cap_col is None:
        irr_cap_col = year_df['annual_irr_fraction'].values \
            if 'annual_irr_fraction' in year_df.columns else None
    if irr_cap_col is not None:
        irr_frac = np.clip(np.nan_to_num(irr_cap_col, nan=0.0), 0, 1)
    else:
        irr_frac = np.zeros(len(predictions), dtype=np.float64)

    # Adaptive crop-fraction floor: irr_frac is at least crop_frac at
    # every pixel.  Pure crop pixel (cf=1) → irr_frac=1.  Pure desert
    # pixel (cf=0) → no boost.  Mixed pixel (cf=0.5) → irr_frac >= 0.5.
    # Naturally addresses the boxplot outliers where old ag wells got
    # misclassified as M&I in the 2024 registry — those pixels keep
    # high crop_frac and the floor lifts their irr_frac accordingly.
    if crop_frac_col is not None:
        cf_floor = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        irr_frac = np.maximum(irr_frac, cf_floor)

    # Irr-fraction overrides:
    #   (1) Pre-1970 ALL pixels: USGS shows ag was 91–97% of all AZ
    #       pumping through 1970 (1950=97%, 1960=94%, 1970=91%).  Force
    #       0.95 across all retained pixels for year < 1970, full volume.
    #   (2) 1970–1984 NON-AMA: irr_frac = 1 − urban_frac (clipped) so
    #       Irr% stays high at rural ag basins through the late-pre-GMA
    #       peak era; volume is conserved (no desert-residual drop).
    #   (3) 1985+ NON-AMA: Option C — area-weighted with desert residual
    #       dropped.  Addresses post-CAP over-prediction at desert-
    #       fringe pixels without affecting pre-CAP peak years.
    #   (4) 1970+ AMA (Phoenix, Tucson only): natural year-specific
    #       irr_capacity_fraction so the registry captures real
    #       municipal/industrial growth concentrated in those AMAs.
    IRR_OVERRIDE_PRE_1970 = 0.95
    IRR_OVERRIDE_FLOOR = 0.05
    if year < 1970:
        irr_frac = np.full_like(irr_frac, IRR_OVERRIDE_PRE_1970)
    elif (year < 1985 and basin_names is not None
            and urban_frac_col is not None):
        non_ama = ~np.isin(basin_names, list(AMA_BASINS))
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        irr_from_uf = np.clip(1.0 - uf, IRR_OVERRIDE_FLOOR, 1.0)
        irr_frac = irr_frac.copy()
        irr_frac[non_ama] = irr_from_uf[non_ama]
    irr = predictions * irr_frac
    nonirr = predictions * (1.0 - irr_frac)

    # 1985+ NON-AMA split: per-pixel LULC logic (registry mostly
    # complete post-CAP, so use raw cf/uf instead of AGRI smoothing).
    #   - only crop (cf > 0 AND uf <= 0.20):   irr = pred (full Irr)
    #   - both crop AND urban (cf > 0 AND uf > 0.20):
    #                                           irr = pred × (1-uf)
    #                                           nonirr = pred × uf
    #   - only urban (cf = 0 AND uf > 0.20):   default partition,
    #                                           BUT irr = 0 if AGRI < 0.5
    #                                           (no real ag activity →
    #                                           drop the Irr portion)
    #   - pure desert (cf = 0 AND uf <= 0.20):
    #       - WITH well: default partition (irr = pred × irr_capacity)
    #       - WITHOUT well: drop (irr = 0, nonirr = 0)
    URBAN_HIGH_THRESHOLD = 0.20
    ONLY_URBAN_AGRI_FLOOR = 0.50
    if (year >= 1985 and basin_names is not None
            and crop_frac_col is not None and urban_frac_col is not None):
        non_ama = ~np.isin(basin_names, list(AMA_BASINS))
        cf = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        only_crop = non_ama & (cf > 0) & ~(uf > URBAN_HIGH_THRESHOLD)
        both_lu = non_ama & (cf > 0) & (uf > URBAN_HIGH_THRESHOLD)
        only_urban = non_ama & ~(cf > 0) & (uf > URBAN_HIGH_THRESHOLD)
        pure_desert = non_ama & ~(cf > 0) & ~(uf > URBAN_HIGH_THRESHOLD)
        pure_desert_no_well = pure_desert & ~has_well
        irr[only_crop] = predictions[only_crop]
        nonirr[only_crop] = 0.0
        irr[both_lu] = predictions[both_lu] * (1.0 - uf[both_lu])
        nonirr[both_lu] = predictions[both_lu] * uf[both_lu]
        # Only-urban: zero out Irr where smoothed AGRI is low (no real
        # ag activity nearby).  NonIrr stays from default partition
        # (pred × (1−irr_capacity)).  This drops Irr volume at urban-
        # only pixels lacking ag context.
        if 'AGRI' in year_df.columns:
            _agri_only_urban = np.clip(np.nan_to_num(
                year_df['AGRI'].values, nan=0.0,
            ), 0, 1)
            only_urban_low_agri = only_urban & (_agri_only_urban < ONLY_URBAN_AGRI_FLOOR)
            irr[only_urban_low_agri] = 0.0
        irr[pure_desert_no_well] = 0.0
        nonirr[pure_desert_no_well] = 0.0

    # LU-only pixel split: at crop pixels, conserve volume by splitting
    # predictions between Irr (1 - urban_frac) and NonIrr (urban_frac);
    # at urban-only pixels, Irr=0 and NonIrr=pred×urban_frac.
    # Canal-ag pixels without wells: full Irr (real canal deliveries).
    #
    # The whole LU-only contribution is dampened by _phantom_infill_scale
    # so the 1938–1955 ag-well drilling era ramp dampens LU-only volume
    # too (1938: ×0.10, 1945: ×0.47, 1955+: ×1.0).  This addresses the
    # 1938–1954 over-prediction where LU-only crop pixels were dumping
    # full ML predictions into the partition before ag had matured.
    has_direct_canal = (canal_dens > 0) if canal_dens is not None \
        else np.zeros(len(predictions), dtype=bool)
    canal_ag_no_wells = (has_smooth_canal | has_direct_canal) & ~has_well
    if lu_only.any() or canal_ag_no_wells.any():
        cf_lu = np.clip(np.nan_to_num(
            crop_frac_col if crop_frac_col is not None
            else np.zeros(len(predictions)), nan=0.0,
        ), 0, 1)
        uf_lu = np.clip(np.nan_to_num(
            urban_frac_col if urban_frac_col is not None
            else np.zeros(len(predictions)), nan=0.0,
        ), 0, 1)
        lu_scale = _phantom_infill_scale(year)  # 0.10 (1938) → 1.0 (1955+)
        # crop threshold: pixels with cf > 0.05 qualify as lu_crop.
        # Peak years (1951–55, 1970–80) at outside-AMA also include
        # AGRI > 0.10 (smoothed ag halo) so the retention extension
        # routes through lu_crop (volume-conserving), not lu_urban_only
        # (which would zero them at uf=0 rural-ag-fringe pixels).
        LU_CROP_THRESHOLD = 0.05
        if (_agri_extension_active and 'AGRI' in year_df.columns
                and basin_names is not None):
            _agri_lu_branch = np.clip(np.nan_to_num(
                year_df['AGRI'].values, nan=0.0,
            ), 0, 1)
            _is_ama_lu = np.isin(basin_names, list(AMA_BASINS))
            _crop_gate = (cf_lu > LU_CROP_THRESHOLD) | (
                (_agri_lu_branch > _agri_threshold_yr) & ~_is_ama_lu
            )
        else:
            _crop_gate = (cf_lu > LU_CROP_THRESHOLD)
        lu_crop = lu_only & _crop_gate
        lu_urban_only = lu_only & ~_crop_gate
        # lu_crop: split by urban_frac (volume-conserving) × scale
        irr[lu_crop] = (
            predictions[lu_crop] * (1.0 - uf_lu[lu_crop]) * lu_scale
        )
        nonirr[lu_crop] = predictions[lu_crop] * uf_lu[lu_crop] * lu_scale
        # lu_urban_only: NonIrr = pred × uf × scale; Irr = 0
        irr[lu_urban_only] = 0.0
        nonirr[lu_urban_only] = (
            predictions[lu_urban_only] * uf_lu[lu_urban_only] * lu_scale
        )
        # canal_ag_no_wells: real canal deliveries — full Irr, no scale
        if canal_ag_no_wells.any():
            irr[canal_ag_no_wells] = predictions[canal_ag_no_wells]
            nonirr[canal_ag_no_wells] = 0.0

    # ---- GW / SW split via density-ratio with era-dependent GW weight ----
    # gw_share = (gw_weight * wd) / (gw_weight * wd + swd_smooth).
    # _era_gw_weight returns 1.0 pre-GMA (1981) — pre-CAP era was
    # GW-dominated — and ramps to 0.1 by CAP_FULL_YEAR (1990) to push
    # post-CAP shares toward SW (USGS reports 41–46% GW post-1990 vs the
    # pre-GMA 62–69% baseline).
    cw_sf_vals = cw_streamflow_raw if cw_streamflow_raw is not None \
        else np.zeros(len(predictions), dtype=np.float64)
    # Era-dependent SW smoothing sigma:
    #   pre-CAP (year < 1985): default sw_smooth_sigma (2.0) — tighter
    #     halo around POD locations matches the smaller pre-CAP canal
    #     network.
    #   post-CAP (year >= 1985): sigma = 8 — spreads CAP/SRP delivery
    #     influence across the broader modern service-area footprint.
    SW_SIGMA = 4.0 if year >= 1985 else sw_smooth_sigma
    gw_weight = _era_gw_weight(year)

    if irr_swd is not None:
        irr_swd_smooth = _smooth_sw_density(
            irr_swd, cw_sf_vals, raster_shape, valid_mask, SW_SIGMA,
        )
        weighted_irr_wd = gw_weight * irr_wd
        denom_irr = weighted_irr_wd + irr_swd_smooth
        with np.errstate(invalid='ignore', divide='ignore'):
            irr_gw_share = np.where(
                denom_irr > 0, weighted_irr_wd / denom_irr, 1.0,
            )
        irr_gw_share = np.clip(irr_gw_share, 0, 1)
        irr_gw = irr * irr_gw_share
        irr_sw = irr - irr_gw
    else:
        irr_gw = irr.copy()
        irr_sw = np.zeros_like(irr)

    if nonirr_swd is not None:
        nonirr_swd_smooth = _smooth_sw_density(
            nonirr_swd, cw_sf_vals, raster_shape, valid_mask, SW_SIGMA,
        )
        weighted_nonirr_wd = gw_weight * nonirr_wd
        denom_ni = weighted_nonirr_wd + nonirr_swd_smooth
        with np.errstate(invalid='ignore', divide='ignore'):
            nonirr_gw_share = np.where(
                denom_ni > 0, weighted_nonirr_wd / denom_ni, 1.0,
            )
        nonirr_gw_share = np.clip(nonirr_gw_share, 0, 1)
        nonirr_gw = nonirr * nonirr_gw_share
        nonirr_sw = nonirr - nonirr_gw
    else:
        nonirr_gw = nonirr.copy()
        nonirr_sw = np.zeros_like(nonirr)

    # Post-1985: scale NonIrr SW by urban_frac at every pixel.  The
    # urban share of SW stays as municipal/industrial (NonIrr); the
    # non-urban portion is DROPPED (post-CAP SW is already over-
    # predicted, so we don't reattribute the excess to Irr).  Reduces
    # both NonIrr_SW and Total_SW.
    if year >= 1985 and urban_frac_col is not None:
        uf_sw = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        nonirr_sw = nonirr_sw * uf_sw
        # Recompute NonIrr total (Irr unchanged)
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


# Note: the legacy phantom-K / wd_2024 / era-GW-weight helpers above
# (_phantom_k, _wd_2024_active, _wd_2024_scale, _phantom_agri_threshold,
# _phantom_canal_include, _era_gw_weight, plus the PHANTOM_*, WD_2024_*,
# GW_WEIGHT_* constants) are no longer used by partition_predictions.
# They are retained only because the ML-feature override in pipeline.py
# (lines ~1998–2027) still references them.  Cleanup is deferred to a
# future session per the plan.


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
