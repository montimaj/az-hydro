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

# URBAN_AMA_BASINS: AMAs where the default partition (nonirr =
# pred × (1 − irr_capacity)) is appropriate — large municipal
# footprints where most non-ag pumping is genuinely M&I.  Limited
# to Phoenix (~5M metro) and Tucson (~1M metro).  Pinal AMA is
# treated as rural despite Casa Grande / Florence urban patches
# because cotton / alfalfa volume dominates and the default
# partition was bleeding too much Pinal pred into NonIrr.  Other
# AMAs/INAs (Pinal, Prescott, Santa Cruz, Douglas, Willcox,
# Harquahala, Hualapai Valley, Joseph City) are routed through the
# same only_crop / both_lu / only_urban / pure_desert override as
# outside-AMA pixels at year >= 1985 to lift Irr% in ag basins.
URBAN_AMA_BASINS = frozenset({
    'PHOENIX AMA', 'TUCSON AMA',
})

# CANAL_HEAVY_BASINS: rural basins with substantial canal infrastructure
# or canal-weighted streamflow (canal_dens_mean > 0.15 OR
# cw_streamflow_mean > 100, sampled at year 2020).  In these basins,
# the post-2011 NonIrr_SW excess routing applies a 0.3 floor on the
# cf weight (irr_sw += excess × max(cf, 0.3)) so fragmented-ag pixels
# recover ~30% of their excess.  Canal-light basins (Douglas,
# Willcox, Santa Cruz, Joseph City, Lower Gila, Safford, etc.) keep
# the strict cf weighting — they have nominal SW rights on paper but
# no canal infrastructure to actually deliver SW, so excess at
# desert-fringe pixels there is genuinely municipal residual that
# should drop, not ag SW that should be recovered.
CANAL_HEAVY_BASINS = frozenset({
    'LAKE HAVASU', 'YUMA', 'PARKER', 'HARQUAHALA INA',
    'GILA BEND', 'HUALAPAI VALLEY INA', 'PRESCOTT AMA', 'PINAL AMA',
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

# 2024-registry override active window: 1951–2099 with the 1958–1961
# drought-recovery trough skipped.  Window extends through projection
# (2025–2099) so the basin-median LU-only fill applies to LULC-
# expansion pixels in the projection (pixels that become urban/crop
# after 2024 but have no 2024-registered wells).  Without this
# extension, projection LU-expansion pixels would have well_density=0
# in ML features → predicted Total under-shoots growing M&I demand.
#
# The per-pixel-max scale ramps 1.0 → 0.2 over 1995–2005, holds at
# 0.2 through 2015, and ramps 0.2 → 0 over 2015–2020 (i.e., the
# explicit per-pixel-max blend contributes nothing for year > 2020,
# which is correct: post-2020 the year-specific registry IS the 2024
# snapshot already, so the per-pixel max would add nothing).  The
# basin-median LU-only fill is what's active for projection years.
WD_2024_ACTIVE_START = 1951
WD_2024_ACTIVE_END = 2099
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


def apply_ml_well_density_override(
        pred_features: pd.DataFrame,
        year: int,
        year_df: pd.DataFrame,
        wd_2024: np.ndarray | None,
) -> pd.DataFrame:
    """Apply pre-1981 well_density override to ML feature matrix.

    Mirrors the partition-time well-density logic so that ML
    predictions use the same effective well density as the partition
    expects.  Single source of truth for both the central pipeline
    (Step 3 prediction loop) and the UQ ensemble members.

    Logic (lifted verbatim from pipeline.py 1995-2106):
      (1) Per-pixel max blend with the 2024 snapshot
          (``wd_2024``): full at peak years 1951-1955 / 1970-1980;
          scaled elsewhere via ``_wd_2024_scale(year)``.
      (2) AGRI-gated phantom_wd contribution
          (``_phantom_k(year) × PHANTOM_M_TOTAL × ag_gate × canal_weight``).
      (3) LU-only basin-median fill at ag-LULC pixels missing wells
          (``_phantom_infill_scale(year) × basin_median(wd_2024)``).

    Args:
        pred_features: ML feature DataFrame (will be copied if modified).
        year: Prediction year.
        year_df: Per-pixel-per-year DataFrame (must contain ``GW_Basin``
            for the basin-median fill step).
        wd_2024: 2024 well-density snapshot per pixel (1-D), or None
            to skip the override entirely.

    Returns:
        pred_features with ``well_density`` column updated (same object
        if no modification needed; a copy otherwise).
    """
    _k_phantom = _phantom_k(year)
    _wd2024_active = _wd_2024_active(year)
    if not (
        (_k_phantom > 0 or _wd2024_active)
        and year >= CANAL_PREDICTOR_START
        and 'well_density' in pred_features.columns
    ):
        return pred_features

    pred_features = pred_features.copy()
    year_wd = np.nan_to_num(pred_features['well_density'].values, nan=0.0)
    effective_wd = year_wd
    if _wd2024_active and wd_2024 is not None:
        # Peak USGS pumping years (1951–1955, 1970–1980): use the full
        # 2024 registry (no _wd_2024_scale dampening for drought dip /
        # ramp).  Outside those windows: use the standard scale-only
        # multiplier.
        wd24_raw = np.nan_to_num(wd_2024, nan=0.0)
        if 1951 <= year <= 1955 or 1970 <= year <= 1980:
            wd24 = wd24_raw
        else:
            wd24 = wd24_raw * _wd_2024_scale(year)
        effective_wd = np.maximum(effective_wd, wd24)
    if _k_phantom > 0 and 'AGRI' in pred_features.columns:
        agri = np.clip(np.nan_to_num(pred_features['AGRI'].values, nan=0.0), 0.0, 1.0)
        _agri_thr = _phantom_agri_threshold(year)
        ag_gate = (agri >= _agri_thr).astype(np.float64)
        _canal_inc = _phantom_canal_include(year)
        if 'canal_density' in pred_features.columns:
            canal_d = np.nan_to_num(pred_features['canal_density'].values, nan=0.0)
            canal_weight = np.where(canal_d > 0, _canal_inc, 1.0)
        else:
            canal_weight = 1.0
        phantom_wd = agri * PHANTOM_M_TOTAL * _k_phantom * ag_gate * canal_weight
        effective_wd = np.maximum(effective_wd, phantom_wd)

    # LU-only basin-median fill (ML features ONLY — does NOT affect
    # the partition's lu_only mask).  At pixels where effective_wd is
    # still 0 (no year_wd, no wd_2024) but the pixel has LULC signal,
    # fill with the basin's median wd_2024 value × scale.
    #
    # Gating logic:
    #   - AMA/INA pixels (any year): raw per-pixel fractions
    #     (annual_crop_fraction > 0 OR annual_urban_fraction > 0).
    #   - Outside-AMA pixels at peak years (1951-1955, 1970-1980):
    #     SMOOTHED ag (AGRI > 0) OR raw urban_fraction > 0.
    #   - Outside-AMA pixels at other years: raw per-pixel fractions.
    # AGRI-extension at outside-AMA: only at peak years (registry
    # incomplete then).  Post-CAP (1985+) uses raw cf/uf only.
    #   1951–55, 1975–80: AGRI > 0.02 (loose, peak years)
    #   1970–74:           AGRI > 0.10 (standard)
    #   other years:       no AGRI extension (raw cf/uf only)
    _is_loose_p = (1951 <= year <= 1955 or 1975 <= year <= 1980)
    _is_std_p   = (1970 <= year <= 1974)
    _agri_thr_p = 0.02 if _is_loose_p else 0.10
    _agri_active = _is_loose_p or _is_std_p
    if (year >= PHANTOM_INFILL_START
            and wd_2024 is not None
            and 'GW_Basin' in year_df.columns):
        _basins_lu = year_df['GW_Basin'].values
        _wd24_full = np.nan_to_num(wd_2024, nan=0.0)
        _cf_lu = np.clip(np.nan_to_num(
            pred_features['annual_crop_fraction'].values
            if 'annual_crop_fraction' in pred_features.columns
            else np.zeros(len(effective_wd)), nan=0.0,
        ), 0, 1)
        _uf_lu = np.clip(np.nan_to_num(
            pred_features['annual_urban_fraction'].values
            if 'annual_urban_fraction' in pred_features.columns
            else np.zeros(len(effective_wd)), nan=0.0,
        ), 0, 1)
        _raw_gate = (_cf_lu > 0) | (_uf_lu > 0)
        if _agri_active:
            _agri_lu = np.clip(np.nan_to_num(
                pred_features['AGRI'].values
                if 'AGRI' in pred_features.columns
                else np.zeros(len(effective_wd)), nan=0.0,
            ), 0, 1)
            _outside_gate = (_agri_lu > _agri_thr_p) | (_uf_lu > 0)
            _is_ama = np.isin(_basins_lu, list(AMA_BASINS))
            _lulc_gate = np.where(_is_ama, _raw_gate, _outside_gate)
        else:
            _lulc_gate = _raw_gate
        _lu_scale = _phantom_infill_scale(year)
        if _lu_scale > 0.0:
            _basin_series = pd.Series(_basins_lu)
            for _b in _basin_series.dropna().unique():
                _b_mask = (_basins_lu == _b)
                _b_pool = _wd24_full[_b_mask & (_wd24_full > 0)]
                if _b_pool.size == 0:
                    continue
                _b_lu = _b_mask & (effective_wd == 0.0) & _lulc_gate
                if _b_lu.any():
                    _basin_med = float(np.median(_b_pool))
                    effective_wd[_b_lu] = _basin_med * _lu_scale
    pred_features['well_density'] = effective_wd
    return pred_features


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
GW_WEIGHT_PRE_1945 = 5.0   # pre-1945: USGS shows ~100% GW (no/minimal SW
                           # except SRP); push very strongly toward GW
GW_WEIGHT_PRE_GMA = 2.0    # 1945–1980 GW-dominant era
GW_WEIGHT_POST_CAP = 0.2   # post-CAP SW-dominant era


GW_WEIGHT_MID_CAP = 0.5     # 1998–2007 mid-CAP bump to lift GW% at 2000/2005
GW_WEIGHT_1930_1935 = 10.0  # 1930–1935 bump — model 1935 Total ~ USGS
                            # Total_GW (1.38 vs 1.20) but GW share too
                            # low (0.91 vs 1.20) because SRP canal-pixel
                            # density-ratio attributed too much to SW.
                            # Higher gw_weight pushes canal-fringe SW
                            # toward GW, lifting GW from 0.91 → ~1.10.


def _era_gw_weight(year: int) -> float:
    """Return era-dependent GW weight for the density-ratio split.

    Pre-1945: weight = 5.0 (very GW-dominant, matches USGS pre-1945
        showing essentially all pumping was GW).
    1945–1985: weight = 2.0 (GW-dominant, USBR ~67% GW statewide).
        Extended through 1985 because the 1985 partition uses the
        1970–1984 irr_frac override (volume-conserving 1−uf split)
        and skips the post-1985 NonIrr_SW excess routing — without
        that routing lifting Irr_GW, the GW/SW split needs the full
        pre-CAP GW weight to keep 1985 GW% near the USGS anchor.
    1986–1990: linear ramp 2.0 → 0.2 as CAP comes online.
    1990–1997: weight = 0.2.
    1998–2007: weight = 0.4 (mid-CAP bump).  USGS shows GW% rebound
        in this window (1995=40.5, 2000=50.9, 2005=48.8) that the
        flat-0.2 weight under-shoots by 4–7 pp.  Stepping up to 0.4
        for these years lifts GW% without touching 1990/2010 (already
        in band).
    2008+: weight = 0.2 (SW-dominant, USGS/ADWR ~42% GW).
    """
    if 1930 <= year <= 1935:
        return GW_WEIGHT_1930_1935
    if year < 1945:
        return GW_WEIGHT_PRE_1945
    if year < 1985:
        return GW_WEIGHT_PRE_GMA
    if 1998 <= year <= 2007:
        return GW_WEIGHT_MID_CAP
    if year >= CAP_FULL_YEAR:
        return GW_WEIGHT_POST_CAP
    # 1985-1989 ramp 2.0 → 0.2 (year=1985 returns 2.0 at frac=0).
    frac = (year - 1985) / (CAP_FULL_YEAR - 1985)
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
    # 1956–1959 std_agri extension: 1955→1956 the loose AGRI gate
    # turned off (loose: AGRI > 0.02), dropping pixels retained via
    # AGRI extension and producing a 2.2 MAF Total cliff at 1956.
    # Re-extend with the std threshold (0.10) at 1956–1959 to bridge
    # the dip toward the ADWR 1957 (7.0 MAF) anchor without dropping
    # pixels that AGRI smoothing legitimately kept in service area.
    _is_std_agri   = (1970 <= year <= 1974 or 1956 <= year <= 1959)
    # 1964–1969 mid_agri extension: USGS 1965 = 7.04 (local peak
    # between 1960 = 5.62 and 1970 = 7.60).  Model was under-
    # predicting 1965 by 8.9%.  1970 had 0.10 threshold and matched
    # USGS; a slightly looser 0.05 threshold at 1964–1969 extends
    # AGRI retention in the pre-1970 ramp-up without going as wide
    # as the 0.02 loose window.
    _is_mid_agri   = (1964 <= year <= 1969)
    if _is_loose_agri:
        _agri_threshold_yr = 0.02
    elif _is_mid_agri:
        _agri_threshold_yr = 0.10
    else:
        _agri_threshold_yr = 0.10
    _agri_extension_active = _is_loose_agri or _is_std_agri or _is_mid_agri
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
    if year < 1948:
        # Pre-1922: keep the LULC intersection.  USGS shows water use
        # near-flat (1915: 0.10, 1920: 0.20 MAF), so the strict
        # well+LULC requirement is appropriate for that low-development
        # era.  1922 onward USGS jumps (1925: 0.45, 1930: 0.75, 1935:
        # 1.20 MAF) — drilling outpaced the GEE-LULC-frozen-at-1938
        # crop footprint, so the intersection drops too many real
        # year-specific well pixels.
        if year < 1922:
            has_well_for_retention = (well_dens_yr > 0) & (
                _has_crop_any | _has_urban_any
            )
        elif 1938 <= year <= 1944 and crop_frac_col is not None:
            # 1938–1944: ML predicts ~200+ mm/pixel pumping at 1938+
            # (vs ~129 at 1937), inflating Total when USGS shows
            # 1940 = 1.80 MAF.  Restrict retention to well pixels
            # co-located with high cf/uf to drop high-pred pixels at
            # marginal-LULC areas.  Threshold ramps tight (0.95 at
            # 1938) → loose (0.7 at 1942-1944) to smooth the
            # 1937→1938 cliff (which is also driven by an ML
            # mean-depth jump).
            #   1938: 0.95
            #   1939: 0.90
            #   1940: 0.80 (matches USGS 1940 = 1.80)
            #   1941: 0.75
            #   1942-1944: 0.70
            _ramp_intersect = {1938: 0.95, 1939: 0.90, 1940: 0.80, 1941: 0.75}
            _intersect_thresh = _ramp_intersect.get(year, 0.7)
            cf_intersect = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
            uf_intersect = (
                np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
                if urban_frac_col is not None
                else np.zeros(len(predictions))
            )
            has_well_for_retention = (well_dens_yr > 0) & (
                (cf_intersect > _intersect_thresh)
                | (uf_intersect > _intersect_thresh)
            )
        else:
            has_well_for_retention = (well_dens_yr > 0)
        # LU-only retention ramp at 1931–1937 to bridge the
        # 1932→1933 cliff (LU-only kicks in).  Threshold ramps high
        # → low so few pixels added at 1931–1932 (matching USGS
        # gradual climb), reaching standard 0.5 at 1935+.
        #   1931: 0.85 (very tight, few pixels)
        #   1932: 0.70
        #   1933: 0.60
        #   1934: 0.55
        #   1935-1937: 0.50
        # Skipped at 1930 (well-only matches USGS 0.75 MAF) and
        # 1938+ (intersection retention takes over).
        if 1930 <= year <= 1937 and crop_frac_col is not None:
            # Smaller-step ramp 1930-1937 to bridge the LU-only-on
            # transition smoothly:
            #   1930: 0.90 (very tight, few new pixels)
            #   1931: 0.80
            #   1932: 0.72
            #   1933: 0.65
            #   1934: 0.58
            #   1935: 0.53
            #   1936: 0.51
            #   1937: 0.50
            _lu_ramp = {1930: 0.90, 1931: 0.80, 1932: 0.72, 1933: 0.65,
                        1934: 0.58, 1935: 0.53, 1936: 0.51}
            _lu_thresh = _lu_ramp.get(year, 0.50)
            cf_lu_pre = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
            uf_lu_pre = (
                np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
                if urban_frac_col is not None
                else np.zeros(len(predictions))
            )
            has_crop = cf_lu_pre > _lu_thresh
            has_urban = uf_lu_pre > _lu_thresh
        elif 1945 <= year <= 1947:
            # 1945–1947: enable standard LU-only retention to recover
            # volume that the intersection-restricted well retention
            # drops.  Without LU-only here, 1945 Total fell to 2.6
            # MAF (USGS GW = 2.80).  Standard cf > 0 / uf > 0 brings
            # in LU-only contributions to bridge.
            has_crop = _has_crop_any
            has_urban = _has_urban_any
        else:
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
    #   (2) 1970–1985 NON-AMA: irr_frac = 1 − urban_frac (clipped) so
    #       Irr% stays high at rural ag basins through the late-pre-GMA
    #       peak era and the 1985 CAP-startup year; volume is conserved
    #       (no desert-residual drop).  Extended to include 1985 because
    #       USGS Irr% at 1985 was 85.5 vs the area-weighted partition's
    #       77 — the 1970–1984 logic produces ~89 Irr% at 1980 (matching
    #       USGS 89.1), so applying it at 1985 lands much closer.
    #   (3) 1986+ NON-AMA: Option C — area-weighted with desert residual
    #       dropped.  Addresses post-CAP over-prediction at desert-
    #       fringe pixels without affecting pre-CAP peak years.
    #   (4) 1970+ AMA (Phoenix, Tucson only): natural year-specific
    #       irr_capacity_fraction so the registry captures real
    #       municipal/industrial growth concentrated in those AMAs.
    IRR_OVERRIDE_PRE_1970 = 0.95
    IRR_OVERRIDE_FLOOR = 0.05
    if year < 1970:
        irr_frac = np.full_like(irr_frac, IRR_OVERRIDE_PRE_1970)
    elif (year < 1986 and basin_names is not None
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
    # URBAN_HIGH_THRESHOLD: pixel needs uf above this to be classified
    # as both_lu (loses uf×pred to NonIrr).  Below it, an ag pixel goes
    # to only_crop (full pred → Irr).  Raised from 0.20 → 0.30 to
    # capture suburban-fringe ag (uf 0.20–0.30 from spatially-smeared
    # urban) as Irr rather than NonIrr — addresses the persistent
    # ~5–8 pp Irr% under-attribution at 1985 and 2015+.
    URBAN_HIGH_THRESHOLD = 0.30
    ONLY_URBAN_AGRI_FLOOR = 0.50
    if (year >= 1986 and basin_names is not None
            and crop_frac_col is not None and urban_frac_col is not None):
        # Apply override outside URBAN_AMA_BASINS (Phoenix, Tucson).
        # The 8 rural AMAs/INAs (Pinal, Prescott, Santa Cruz, Douglas,
        # Willcox, Harquahala, Hualapai Valley, Joseph City) join
        # outside-AMA pixels in the only_crop / both_lu / only_urban /
        # pure_desert split — addresses the ~5–8 pp Irr% under-
        # attribution at 1986+ where rural AMA/INA ag pixels were
        # stuck on the default partition (nonirr = pred ×
        # (1 − irr_capacity)) and bleeding too much volume to NonIrr.
        # Year 1985 itself uses the 1970–1985 irr_frac override above.
        non_urban = ~np.isin(basin_names, list(URBAN_AMA_BASINS))
        cf = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        # AGRI-halo gate: pixels with cf = 0 but smoothed AGRI > 0.1
        # are in an ag halo — GEE LULC frequently misses patchy rural
        # ag (especially in Willcox/Harquahala/Douglas), but AGRI
        # smoothing captures the ag footprint.  Route these to the
        # only_crop branch instead of pure_desert so they keep their
        # full ML pred as Irr rather than falling into the default
        # partition's NonIrr default or the desert-well scaling.
        AG_HALO_AGRI = 0.10
        if 'AGRI' in year_df.columns:
            _agri_halo = np.clip(np.nan_to_num(
                year_df['AGRI'].values, nan=0.0,
            ), 0, 1)
            ag_halo = (cf <= 0) & (_agri_halo > AG_HALO_AGRI)
        else:
            ag_halo = np.zeros(len(predictions), dtype=bool)
        only_crop = non_urban & (
            (cf > 0) | ag_halo
        ) & ~(uf > URBAN_HIGH_THRESHOLD)
        both_lu = non_urban & (cf > 0) & (uf > URBAN_HIGH_THRESHOLD)
        only_urban = non_urban & ~(cf > 0) & ~ag_halo & (uf > URBAN_HIGH_THRESHOLD)
        pure_desert = non_urban & ~(cf > 0) & ~ag_halo & ~(uf > URBAN_HIGH_THRESHOLD)
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
        # 2003–2012: scale pure_desert_with_well volumes by 0.75.
        # USGS shows Total drops from 7.54 (2000) to 6.99 (2005) to
        # 6.82 (2010) as CAP ag retirement and drought reduced
        # statewide pumping.  The ML predictions hold near 7.60
        # through 2000–2010, over-predicting by 0.61–0.77 MAF at
        # 2005/2010.  True desert wells (cf = 0, uf < 0.30, AGRI
        # <= 0.1, has_well; the AGRI gate above re-routes ag-halo
        # desert pixels to only_crop) are sparse rural domestic /
        # stock / abandoned industrial wells that pump little water
        # in reality but get non-trivial ML prediction.  Scale 0.5
        # over-corrected (dropped Total UNDER by 0.7 and collapsed
        # GW% by 12 pp); 0.75 trims ~25% of true-desert volume,
        # landing ~0.6 MAF closer to USGS at 2005/2010.
        if 2003 <= year <= 2012:
            pure_desert_with_well = pure_desert & has_well
            irr[pure_desert_with_well] = irr[pure_desert_with_well] * 0.75
            nonirr[pure_desert_with_well] = nonirr[pure_desert_with_well] * 0.75

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
    #   early pre-GMA (< 1965): sigma = 1.0 (tight halo — only SRP with
    #                           small reach).
    #     1948–1955 override:   sigma = 3.0.  USGS shows GW% ~62–69 in
    #                           this window, but with sigma=1 the SW
    #                           halo doesn't reach enough pixels to
    #                           pull GW% off the wd-saturated ~71%
    #                           ceiling.  Wider halo here is justified
    #                           by SRP's mature service area covering
    #                           Salt River Valley — the spatial
    #                           footprint of SW delivery existed even
    #                           if it predates CAP.  gw_weight tuning
    #                           was futile (saturated by tiny smoothed
    #                           swd at sigma=1).
    #   1965–1984:              linear ramp 1.0 → 4.0 as SRP canal
    #                           network matured and CAP construction
    #                           progressed (first CAP delivery 1985).
    #     1973–1977 override:   sigma = 1.5.  USGS GW% jumps back to
    #                           ~62% at 1975 (model under-shot at 56
    #                           with the natural ramp value of 2.5).
    #                           Tighter halo at this window captures
    #                           the temporary mid-1970s drought / SW
    #                           shortage that pushed pumping back to
    #                           groundwater.
    #   1985–2007 (CAP rollout): sigma = 4.0 — CAP/SRP full service
    #                           area footprint.
    #   2008–2021 (mature CAP ramp): linear ramp 4.0 → 6.0.  Bridges
    #                           both sides of the real ML/IrrMapper
    #                           drop at 2010→2011.
    #   2022+:                   sigma = 6.0 — wider halo brings GW%
    #                           into the ADWR 41–42 anchor band.
    # Pre-1948 sigma schedule: piecewise-linear interpolation
    # between anchor points (year, sigma) — smooths the step
    # transitions that produced visible 5–20 pp GW% jumps in the
    # year-by-year time series.
    #
    # Anchor points (calibrated to USGS Total_GW anchors):
    #   1912: 0.0    pre-Yuma
    #   1915: 1.5    Yuma Project peak (USGS 1915 = 0.10 MAF, ~all SW)
    #   1917: 0.3    Pinal/SRP era (drilling era starts)
    #   1929: 0.3    end of Pinal era
    #   1935: 0.0    well-drilling boom (USGS 1935 = 1.20 mostly GW)
    #   1940: 0.3    Gila Project deliveries (USGS 1940 = 1.80)
    #   1945: 1.0    Gila dam era (USGS 1945 = 2.80)
    #   1948: 1.5    pre-GMA baseline (matches 1948–1955 era)
    if year < 1912:
        SW_SIGMA = 0.0
    elif year < 1948:
        _sigma_anchors = [
            (1912, 0.0), (1915, 1.5), (1917, 0.3), (1929, 0.3),
            (1935, 0.0), (1940, 0.3), (1945, 1.0), (1948, 1.5),
        ]
        SW_SIGMA = 0.0
        for i in range(len(_sigma_anchors) - 1):
            y0, s0 = _sigma_anchors[i]
            y1, s1 = _sigma_anchors[i + 1]
            if y0 <= year <= y1:
                if y1 == y0:
                    SW_SIGMA = s0
                else:
                    frac = (year - y0) / (y1 - y0)
                    SW_SIGMA = s0 + frac * (s1 - s0)
                break
    elif 1948 <= year <= 1955:
        SW_SIGMA = 1.5
    elif year < 1965:
        SW_SIGMA = 1.0
    elif year >= 2022:
        SW_SIGMA = 6.0
    elif 2003 <= year <= 2010:
        # CAP-era ag-retirement window.  Model has ~correct Total GW
        # at 2005 but excess SW; reducing sigma tightens the SW halo
        # so less volume gets attributed as SW via the density ratio.
        # Drops SW and lifts GW without changing total retained
        # volume (1985-2010 routing conserves excess via gw_share
        # split, so sigma only redistributes GW↔SW).
        SW_SIGMA = 3.0
    elif year >= 2008:
        # 2011-2021 ramp (after the 2003-2010 override).
        frac = (year - 2008) / (2022 - 2008)
        SW_SIGMA = 4.0 + frac * (6.0 - 4.0)
    elif year >= 1985:
        SW_SIGMA = 4.0
    elif 1973 <= year <= 1977:
        SW_SIGMA = 2.0
    else:
        # 1965-1984 ramp (20 years)
        frac = (year - 1965) / (1985 - 1965)
        SW_SIGMA = 1.0 + frac * (4.0 - 1.0)
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

    # Pre-1945: USGS shows ~100% GW statewide.  Restrict SW to pixels
    # that actually have direct access to surface water — canal-served
    # (smoothed or direct) or holding SW rights.  Elsewhere, collapse
    # SW back into GW so density-ratio leakage (Gaussian smoothing of
    # sw_rights_density into dry cells) does not produce phantom SW.
    if year < 1945:
        _irr_sw_ok = has_smooth_canal | has_direct_canal | (
            (irr_swd > 0) if irr_swd is not None
            else np.zeros(len(predictions), dtype=bool)
        )
        _ni_sw_ok = has_smooth_canal | has_direct_canal | (
            (nonirr_swd > 0) if nonirr_swd is not None
            else np.zeros(len(predictions), dtype=bool)
        )
        irr_gw = np.where(_irr_sw_ok, irr_gw, irr_gw + irr_sw)
        irr_sw = np.where(_irr_sw_ok, irr_sw, 0.0)
        nonirr_gw = np.where(_ni_sw_ok, nonirr_gw, nonirr_gw + nonirr_sw)
        nonirr_sw = np.where(_ni_sw_ok, nonirr_sw, 0.0)

    # Post-1985: scale NonIrr SW by urban_frac at every pixel.  The
    # urban share of SW stays as municipal/industrial (NonIrr); the
    # non-urban remainder is reattributed to Irrigation.
    #
    # Era-dependent treatment of the non-urban excess:
    #   1985–2010 (CAP rollout era): send the FULL excess back to Irr,
    #     splitting between Irr_GW and Irr_SW by the pixel's local
    #     irr_gw_share (the GW share already computed by the density
    #     ratio).  Sending all excess to Irr_SW alone (the prior
    #     attempt) recovered total volume but pushed GW% under by
    #     5–13 pp.  Splitting by local gw_share lifts GW% without
    #     losing the total recovery.
    #   2011+: basin-dependent.
    #     CANAL_HEAVY_BASINS (Lake Havasu, Yuma, Parker, Harquahala,
    #       Gila Bend, Hualapai Valley, Prescott, Pinal): excess ×
    #       max(cf, 0.3) to Irr_SW.  A 0.3 floor on the routing weight
    #       lets fragmented-ag pixels recover ~30% of their excess.
    #       Restricted to basins with real canal infrastructure or
    #       canal-weighted streamflow — the recovered volume is
    #       physically deliverable surface water.
    #     Everywhere else (canal-light rural + urban AMAs): excess ×
    #       cf to Irr_SW (strict cf, drops desert residual).  Includes
    #       Douglas/Willcox/Santa Cruz/Joseph City (no canals) and
    #       Phoenix/Tucson (municipal SW genuinely distinct from ag).
    #
    #   nonirr_sw_new   = nonirr_sw_old × uf
    #   excess          = nonirr_sw_old × (1 − uf)
    #   irr_gw_new      = irr_gw_old + excess × irr_gw_share          [1985–2010]
    #   irr_sw_new      = irr_sw_old + excess × (1 − irr_gw_share)    [1985–2010]
    #                   = irr_sw_old + excess × max(cf, 0.3)          [2011+ canal-heavy]
    #                   = irr_sw_old + excess × cf                    [2011+ everywhere else]
    if (year >= 1986 and urban_frac_col is not None):
        uf_sw = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        excess_sw = nonirr_sw * (1.0 - uf_sw)
        nonirr_sw = nonirr_sw * uf_sw
        if year <= 2010:
            if irr_swd is not None:
                excess_to_gw = excess_sw * irr_gw_share
                excess_to_sw = excess_sw - excess_to_gw
            else:
                excess_to_gw = np.zeros_like(excess_sw)
                excess_to_sw = excess_sw
            irr_gw = irr_gw + excess_to_gw
            irr_sw = irr_sw + excess_to_sw
        else:
            cf_sw = np.clip(np.nan_to_num(
                crop_frac_col if crop_frac_col is not None
                else np.zeros_like(uf_sw), nan=0.0,
            ), 0, 1)
            if basin_names is not None:
                _is_canal_heavy = np.isin(
                    basin_names, list(CANAL_HEAVY_BASINS),
                )
                # Canal-heavy: floor cf at 0.3 (recover deliverable SW).
                # Everywhere else: strict cf (drop desert residual).
                weight = np.where(
                    _is_canal_heavy, np.maximum(cf_sw, 0.3), cf_sw,
                )
            else:
                weight = cf_sw
            irr_sw = irr_sw + excess_sw * weight
        # Recompute Irr / NonIrr totals to reflect the reallocation
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
