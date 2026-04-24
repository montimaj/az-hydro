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

# Future AMA basins added to AMA_BASINS only for projection years.
# Ranegras Plain was designated an AMA in January 2026 (ADWR);
# pre-2026 the basin was unregulated and behaves identically to
# non-AMA in our partition.  From 2026+ the regulatory designation
# applies but the documented behavioural effect (CAP build-out +
# pumping caps) lags by years; we use the AMA-class flag for the
# partition path (1981+ AMA uses natural irr_capacity_fraction
# instead of the 1−uf override) without yet altering the GW/SW
# allocation otherwise.
FUTURE_AMA_BASINS = frozenset({
    'RANEGRAS PLAIN',
})
FUTURE_AMA_START_YEAR = 2026


def get_ama_basins(year: int) -> frozenset:
    """Return the AMA/INA basin set for *year*.

    For years ``< FUTURE_AMA_START_YEAR``, returns the historical
    ``AMA_BASINS``.  From ``FUTURE_AMA_START_YEAR`` onward, also
    includes ``FUTURE_AMA_BASINS`` (e.g. Ranegras Plain, AMA-
    designated 2026).
    """
    if year >= FUTURE_AMA_START_YEAR:
        return AMA_BASINS | FUTURE_AMA_BASINS
    return AMA_BASINS

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

# CO_RIVER_DIRECT_BASINS: basins with physical Colorado River
# mainstem access (direct riverside diversions — CRIT senior rights,
# Parker City, mainstem towns, Yuma Project, etc.).  Two effects:
#   1. Bypass basin-level canal-infra gate (their SW supply comes
#      from mainstem diversions, not canals in canal_density raster).
#   2. Cap GW share at MAX_GW_SHARE_CO_DIRECT so density-ratio can't
#      route everything to GW when irr_sw_rights_density is sparse
#      (CRIT allocations are essentially absent from sw_rights data).
# Excluded: SACRAMENTO VALLEY (tributary / Bullhead local),
# BILL WILLIAMS (tributary with local GW-dominant pumping).
CO_RIVER_DIRECT_BASINS = frozenset({
    'PARKER', 'YUMA', 'LAKE MOHAVE', 'LAKE HAVASU',
    'MEADVIEW', 'DETRITAL VALLEY',
})

# Maximum GW share at CO-river-direct basins.  Caps the density-
# ratio output when irr_sw_rights_density under-represents mainstem
# diversions (e.g. Parker has 4/1433 pixels with sw_rights — CRIT
# allocations missing from the rights raster).  Reality: all six
# CO-river-direct basins are physically 90-99 % SW (CRIT senior
# rights, Yuma Project + Cocopah/Quechan tribal deliveries, Bullhead
# City direct mainstem M&I, Lake Havasu City CAP intake, Detrital
# Valley + Meadview small mainstem M&I).  A uniform cap of 0.10
# enforces that the well-mediated GW share at any CO-direct pixel
# stays ≤ 10 %, with the partition's density-ratio + canal-smoothing
# free to land anywhere in [0, 0.10].
MAX_GW_SHARE_CO_DIRECT = 0.10

# Per-basin overrides (currently empty — all CO-direct basins use
# the default 0.10 cap).  Reserved for future basin-specific tuning
# if any CO-direct basin's physical SW share warrants a different cap.
CO_DIRECT_BASIN_GW_CAP: dict[str, float] = {}


def _co_direct_gw_cap_array(basin_names: np.ndarray) -> np.ndarray:
    """Return per-pixel GW-share cap for CO-river-direct basins.

    Default cap is ``MAX_GW_SHARE_CO_DIRECT`` (0.4) for every CO-direct
    basin not listed in ``CO_DIRECT_BASIN_GW_CAP``.  Per-basin overrides
    apply at Parker / Yuma / Lake Mohave (ag- or M&I-mainstem-dominant
    basins where physical SW share is 85-99 %).  Pixels in non-CO-direct
    basins receive a sentinel of ``np.inf`` (no cap) so callers can
    apply the cap unconditionally via ``np.minimum``.
    """
    cap = np.full(len(basin_names), np.inf, dtype=np.float64)
    is_co_direct = np.isin(basin_names, list(CO_RIVER_DIRECT_BASINS))
    cap = np.where(is_co_direct, MAX_GW_SHARE_CO_DIRECT, cap)
    for name, val in CO_DIRECT_BASIN_GW_CAP.items():
        cap = np.where(basin_names == name, val, cap)
    return cap


# Per-basin GW share FLOOR (mirror of the CO-direct GW cap).  The
# density-ratio + canal-smoothing partition structurally under-attributes
# GW at AMA/INAs that have dense SRP / CAP canal infrastructure but
# also significant residual GW pumping inside the canal service area
# (private municipal wells, dual-supply ag, etc.).  At those basins the
# physical GW share is anchored by ADWR mandatory metering, and applying
# a per-basin minimum on the density-ratio output recovers it without
# touching the global ``gw_weight`` schedule (which is load-bearing for
# the WestWater 2026 CAP-shortage tier-boost calibration).
#
# Currently floored (anchored to ADWR-norm + NHM/Reitz cross-check):
#   PHOENIX AMA   — 0.20.  ADWR Phoenix AMA Plan: 705 kAF GW /
#                   2.3 MAF total = ~30 % native + recharged GW.
#                   We use 0.20 (below the ADWR 30 %) because part of
#                   ADWR's "groundwater" share at Phoenix is recharged
#                   CAP/SRP storage rather than native aquifer mining,
#                   and the model already attributes some of that
#                   recharged volume to SW.
#   PINAL AMA     — 0.30.  CAP-served ag historically ~70 % CAP-SW +
#                   ~30 % GW; NHM 39 % / Reitz 42 % Irr_GW% confirm
#                   model 4 % Irr_GW is structurally too low.
#   TUCSON AMA    — 0.30.  Tucson Water (M&I) is GW-heavy + Avra
#                   Valley ag uses substantial GW; NHM 57 % / Reitz
#                   67 % Irr_GW% confirm model 2 % Irr_GW is too low.
#   HARQUAHALA INA — 0.30.  Higher GW use than the prior 10-20 %
#                   assumption; NHM 76 % / Reitz 72 % Irr_GW% support
#                   substantially higher GW share than the model's
#                   current 11 %.
#
# CO-direct basins (Yuma, Parker, Lake Mohave, Lake Havasu, Detrital
# Valley, Meadview) and Bill Williams have NO floor — the partition's
# density-ratio + canal-smoothing is allowed to land anywhere in
# [0, 0.10] (CO-direct cap) or [0, 1] (Bill Williams).  No lower-bound
# enforcement at those basins, since the cap is the only physical
# constraint we have verified ground for.
#
# Pending verified ADWR per-AMA tables (currently no floor):
#   PRESCOTT AMA          (~70-80 % GW per general literature)
#   HUALAPAI VALLEY INA   (~95 % GW per general literature)
BASIN_GW_FLOOR = {
    'PHOENIX AMA':    0.30,  # ADWR: 0.705 MAF GW / 2.3 MAF total ≈ 31 %.
                             # Raised from 0.20 alongside the new 1990+
                             # CAP_NONIRR_SW_BOOST that lowers modern
                             # Phoenix NIGW; the 0.30 floor anchors
                             # the basin to its documented 30 % GW
                             # share under the NIGW-reduced regime.
    'PINAL AMA':      0.50,  # CAP-fed cotton + continued GW ag pumping.
                             # Raised from 0.30 to ~50 %, the mid-
                             # point of the documented 50-60 % GW
                             # regime at Pinal AMA post-CAP.  Without
                             # this floor the NIGW boost pushes model
                             # Pinal GW% below the physical range.
    'TUCSON AMA':     0.60,  # Avra Valley pump-and-treat + continued
                             # municipal GW pumping give Tucson AMA a
                             # documented ~70 % GW share (CAP direct
                             # delivery ~40 kAF + recharge ~145 kAF
                             # = 184 kAF SW vs ~400 kAF GW) for the
                             # post-CAP era (1993+).  Raised from
                             # 0.30 to 0.60 (conservative mid).
                             # Pre-1993 Tucson had NO CAP delivery —
                             # see ``BASIN_GW_FLOOR_PRE_CAP`` below
                             # for the higher pre-CAP-Tucson floor.
    'HARQUAHALA INA': 0.70,  # Small CAP allocation (~78 kAF) vs large
                             # GW ag pumping; ADWR documents Harquahala
                             # as heavily GW-dependent with subsidence
                             # concerns.  Raised from 0.30 to 0.70 to
                             # anchor the physical regime.
    'PRESCOTT AMA':   0.64,  # ADWR Prescott AMA Total_GW share = 64 %
                             # (Irrigation = 8 % of total; mostly M&I
                             # pumping from Prescott Valley + Prescott
                             # city wells).  Earlier 0.40 anchor only
                             # captured native-GW pumping (9,466 AF /
                             # 24,713 AF = 38 %); the higher 0.64 anchor
                             # reflects the full ADWR-reported Total GW
                             # share.  Statewide impact negligible
                             # (Prescott total ~24 kAF/yr → +15 kAF GW).
    'RANEGRAS PLAIN': 0.95,  # Documented ~100 % GW basin.  CAP
                             # delivers ~1.2 kAF/yr (essentially zero)
                             # and is unlikely to expand even under
                             # the Jan-2026 AMA designation because
                             # CAP allocations are reserved for the
                             # three central counties (Maricopa, Pinal,
                             # Pima — i.e. Phoenix / Pinal / Tucson
                             # AMAs).  The basin's apparent ~4.5 %
                             # canal coverage is the CAP-NIA pipeline
                             # passing through to those AMAs, not
                             # local delivery.  Canal seepage from the
                             # pass-through pipeline can incidentally
                             # recharge the aquifer (small effect)
                             # which justifies a non-zero SW share in
                             # the partition; 0.95 floor leaves a 5 %
                             # SW slack for that seepage component +
                             # any smoothing leakage from neighbouring
                             # Harquahala SCIDD pixels.  Floor applies
                             # at all years (1896-2099).
}

# Subset of BASIN_GW_FLOOR keys that should NOT apply at projection
# years (>= FUTURE_AMA_START_YEAR).  Used by ``_basin_gw_floor_array``.
# Currently empty — Ranegras was a candidate but is documented to
# stay ~100 % GW even post-2026 AMA designation (CAP allocations
# reserved for the three central counties), so its 0.95 floor
# applies year-round.
BASIN_GW_FLOOR_HISTORICAL_ONLY: frozenset = frozenset()

# Pre-CAP-era (year < CAP arrival at the basin) per-basin GW floor
# overrides.  These are higher than the post-CAP floors in
# ``BASIN_GW_FLOOR`` because CAP delivery was zero before the basin's
# CAP reach completion year — the basin was effectively all-GW.
# Mapping: ``basin_name → (cap_arrival_year, pre_cap_floor)``.
# At ``year < cap_arrival_year`` the pre-CAP floor takes precedence
# over the post-CAP ``BASIN_GW_FLOOR`` value.
BASIN_GW_FLOOR_PRE_CAP: dict[str, tuple[int, float]] = {
    # Tucson reach completed 1993; pre-1993 Tucson AMA had no CAP
    # delivery, so the basin operated as essentially all-GW (~95 %).
    # The 0.85 pre-CAP floor reflects the small Avra Valley
    # reclaimed-water + minor riparian SW that did exist locally,
    # without the post-CAP 30-40 % CAP-derived SW share.
    'TUCSON AMA': (1993, 0.85),
}


def _basin_gw_floor_array(
        basin_names: np.ndarray,
        year: int = 0,
) -> np.ndarray:
    """Return per-pixel GW-share floor for floored basins.

    Default is ``0.0`` (no floor) at unlisted basins; per-basin overrides
    from ``BASIN_GW_FLOOR`` apply at the named basins.  Callers apply
    the floor unconditionally via ``np.maximum``.

    Basins in ``BASIN_GW_FLOOR_HISTORICAL_ONLY`` are floored only for
    years ``< FUTURE_AMA_START_YEAR`` (2026); from the projection era
    onward their floor is dropped to ``0.0``, letting the density-ratio
    decide the GW share naturally (e.g. Ranegras Plain is expected to
    tap into CAP after its 2026 AMA designation).

    Basins in ``BASIN_GW_FLOOR_PRE_CAP`` use a higher floor at years
    before the basin's CAP-reach completion (e.g. Tucson AMA pre-1993
    used 0.85 instead of the post-CAP 0.60).
    """
    floor = np.zeros(len(basin_names), dtype=np.float64)
    for name, val in BASIN_GW_FLOOR.items():
        if (year >= FUTURE_AMA_START_YEAR
                and name in BASIN_GW_FLOOR_HISTORICAL_ONLY):
            continue
        # Pre-CAP override: use the higher pre-CAP floor at the basin
        # if the year predates the basin's CAP-reach arrival year.
        if name in BASIN_GW_FLOOR_PRE_CAP:
            cap_arrival, pre_cap_floor = BASIN_GW_FLOOR_PRE_CAP[name]
            if year < cap_arrival:
                val = pre_cap_floor
        floor = np.where(basin_names == name, val, floor)
    return floor


# Per-basin GW share CAP for non-CO-direct canal-served basins where the
# density-ratio over-attributes to GW relative to NHM/Reitz cross-checks.
# Mirror of CO_DIRECT_BASIN_GW_CAP but applied at canal-served basins
# along the lower Gila River corridor that receive CAP / Yuma Project
# delivery infrastructure.  These basins pass the basin-canal-coverage
# gate (>= 2 % canal pixels) so the no-canal SW collapse doesn't fire,
# but the density-ratio still over-attributes to GW because the
# canal-weighted streamflow signal is weaker than at the metro AMAs.
#
# Currently capped:
#   LOWER GILA — 0.25.  Yuma Project + Gila River Indian Community
#                canals deliver substantial SW; cap holds the
#                density-ratio output to the documented SW-dominant
#                regime even when irr_swd density is sparse.
#
# NOT capped (despite high model GW% relative to NHM):
#   McMullen Valley — model 100 %, NHM 38 %.  Ground truth confirms
#                     McMullen is GW-dominant with documented aquifer
#                     depletion + subsidence; NHM is wrong here.
#   Big Sandy / Bill Williams / LCRP / Sacramento Valley — Irr is
#                     already 100 % GW from the canal-coverage gate
#                     firing on Irr_SW; NonIrr_SW preserved.  Capping
#                     here would un-collapse SW that physically doesn't
#                     exist for irrigation.
#   RANEGRAS PLAIN — previously capped at 0.25 against NHM Irr_GW% = 13.
#                    Removed: that NHM target was a year-range bug
#                    artifact (NHM was averaged over 1980-2020 with
#                    20 zero-padded years, under-counting by ~50 %).
#                    CAP delivers ~1 kAF/yr to Ranegras (essentially
#                    zero); the basin's apparent canal coverage is
#                    the CAP-NIA pipeline passing through en route to
#                    Harquahala/Phoenix, not local delivery
#                    infrastructure.  Density-ratio naturally gives
#                    ~70 % GW which matches the GW-dominant ag regime.
BASIN_GW_CAP: dict[str, float] = {
    'LOWER GILA': 0.25,
}


def _basin_gw_cap_array(basin_names: np.ndarray) -> np.ndarray:
    """Return per-pixel GW-share cap for non-CO-direct capped basins.

    Default is ``np.inf`` (no cap) at unlisted basins; per-basin overrides
    from ``BASIN_GW_CAP`` apply at the named basins.  Callers apply
    the cap via ``np.minimum``.  Combined with ``_co_direct_gw_cap_array``
    via element-wise ``np.minimum`` to give the effective per-pixel cap.
    """
    cap = np.full(len(basin_names), np.inf, dtype=np.float64)
    for name, val in BASIN_GW_CAP.items():
        cap = np.where(basin_names == name, val, cap)
    return cap

# Basins where NonIrr_SW should be preserved at the no-canal gate.
# These have physical NonIrr_SW supply from direct lake / river /
# tributary intakes that do NOT show up as canal_density pixels:
#   LCR — Page on Lake Powell intake
#   SACRAMENTO VALLEY — Bullhead City on Colorado mainstem
#   BIG SANDY — historical mining diversions
#   BILL WILLIAMS — Alamo Lake intake serving M&I
# At no-canal basins NOT on this list, the gate now collapses BOTH
# Irr_SW and NonIrr_SW (Willcox / Douglas / Joseph City — purely GW
# basins with no SW supply at all).
NONIRR_SW_PRESERVE_BASINS = frozenset({
    'LITTLE COLORADO RIVER PLATEAU',
    'SACRAMENTO VALLEY',
    'BIG SANDY',
    'BILL WILLIAMS',
})

# Basin canal-coverage threshold for the no-canal gate (fraction of
# valid pixels with canal_density > 0).  Set to 1 % alongside a second
# physical gate (mean basin canal_weighted_streamflow > 0): a basin
# must have BOTH meaningful canal coverage AND positive delivery
# capacity to retain SW.  The dual gate catches basins like Douglas
# (1.12 % coverage from spurious historical pixels with no actual
# delivery) without bumping the coverage threshold itself, which kept
# real-canal basins (Phoenix ≈ 40 %, Pinal ≈ 30 %, Yuma > 10 %)
# unaffected.  Willcox (0.16 %) is still caught by the coverage gate.
CANAL_COVERAGE_GATE_THRESHOLD = 0.01

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
        # Peak USGS pumping years (1951–1955, 1970–1980): pre-GMA
        # registry is incomplete at agricultural pixels in every
        # basin — undocumented ag drilling was widespread and
        # GMA-mandated registration (post-1980) was never
        # retroactively complete.  At peak years, use **basin-MAX
        # wd_2024** per pixel at cf > 0 OR uf > 0.2 pixels.  Gate
        # excludes urban-fringe halo pixels (0 < uf <= 0.2) which
        # are GEE-smoothing artifacts rather than real developed
        # pixels, but includes real urban pixels (uf > 0.2).  Real
        # ag pixels (cf > 0) always qualify.  This middle-ground
        # gate avoided both the `cf | uf` overshoot (1980 → 12.6
        # MAF at fringe inflation) and the `cf only` under-shoot
        # (1980 → 7.08 MAF, too restrictive).  Per-pixel max with
        # year_wd guarantees we don't lower any year-specific
        # registered pixel.  Outside those windows: use per-pixel
        # wd_2024 × _wd_2024_scale(year) (standard behavior).
        wd24_raw = np.nan_to_num(wd_2024, nan=0.0)
        # Two lifts:
        #   (A) Peak-year LU lift: basin-MAX at cf>0 | uf>0.2 in
        #       peak USGS pumping years (1951-55, 1970-80).
        #       Aggressive lift representing densest plausible peak-
        #       era well density at developed pixels.
        #   (B) All-pre-1981 ORPHAN lift: basin-p90 at orphan pixels
        #       (cf=0, uf=0, year_wd>0) across all pre-1981 years
        #       (1896-1980).  Pre-GMA registry attrition affects
        #       orphan / stock / rural domestic wells across the
        #       whole pre-GMA era, not just peak years.  Basin-p90
        #       (vs basin-max) gives a realistic "moderately dense
        #       cluster" target rather than the single densest
        #       outlier — orphan pixels are typically 1-2 wells, so
        #       basin-max would be 100-270× the natural value (way
        #       beyond ML training range, saturates).  has_well
        #       requirement excludes bare desert.
        wd24 = np.zeros_like(wd24_raw)
        if 'GW_Basin' in year_df.columns:
            _basins = year_df['GW_Basin'].values
            _cf_peak = np.clip(np.nan_to_num(
                pred_features['annual_crop_fraction'].values
                if 'annual_crop_fraction' in pred_features.columns
                else np.zeros(len(year_wd)), nan=0.0,
            ), 0, 1)
            _uf_peak = np.clip(np.nan_to_num(
                pred_features['annual_urban_fraction'].values
                if 'annual_urban_fraction' in pred_features.columns
                else np.zeros(len(year_wd)), nan=0.0,
            ), 0, 1)
            _is_peak = (1951 <= year <= 1957) or (1970 <= year <= 1980)
            _is_pre_1981 = year < 1981
            _lu_gate_peak = (_cf_peak > 0) | (_uf_peak > 0.2)
            _orphan_has_well = (
                (_cf_peak == 0) & (_uf_peak == 0) & (year_wd > 0)
            )
            _scale = _wd_2024_scale(year)
            _basin_series = pd.Series(_basins)
            for _b in _basin_series.dropna().unique():
                _b_mask = (_basins == _b)
                _b_pool = wd24_raw[_b_mask & (wd24_raw > 0)]
                if _b_pool.size == 0:
                    continue
                # (A) Peak-year LU pixels → basin-MAX
                if _is_peak:
                    wd24[_b_mask & _lu_gate_peak] = float(np.max(_b_pool))
                # (B) Pre-1981 orphan pixels → basin-p90 × scale
                if _is_pre_1981:
                    _b_p90 = float(np.quantile(_b_pool, 0.9))
                    _orphan_target = _b_p90 if _is_peak else _b_p90 * _scale
                    wd24[_b_mask & _orphan_has_well] = np.maximum(
                        wd24[_b_mask & _orphan_has_well],
                        _orphan_target,
                    )
            # For pixels not covered by either lift, fall back to
            # standard per-pixel scale (post-1980 era + LU pixels at
            # non-peak years).
            _no_lift = wd24 == 0
            wd24[_no_lift] = wd24_raw[_no_lift] * _scale
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
    # still 0 (no year_wd, no wd_2024, no basin-max lift for peak
    # years) but the pixel has LULC signal, fill with the basin's
    # median wd_2024 × scale.  Gate uses raw crop/urban fractions
    # (AGRI-smoothed version dropped to avoid hindcast halos).
    # At peak years this branch rarely fires because Step 1's
    # basin-max lift already populates all pixels in developed
    # basins; it remains useful at non-peak years and in basins
    # that have year-specific wells but no 2024 wells.
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
        _lulc_gate = (_cf_lu > 0) | (_uf_lu > 0)
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
                    effective_wd[_b_lu] = (
                        float(np.median(_b_pool)) * _lu_scale
                    )
    pred_features['well_density'] = effective_wd
    return pred_features


# CAP delivery factors — observed historical Tier shortage cuts
# (2022-2024) plus a sustained projection-era cut (2026-2099) driven
# by post-2026 Colorado River Compact renegotiation.  The model's
# `canal_weighted_streamflow_mm` is computed from raw Colorado River
# gauges and does NOT reflect Tier-allocation cuts; apply these
# factors at CAP-served pixels to scale the SW signal before the
# partition runs, so the GW/SW split correctly routes more pumping
# to GW (substituting for lost SW deliveries).
#
# Hindcast factors derived from ADWR's official Tier shortage
# declarations under the 2007 Interim Guidelines + 2019 DCP.  The
# implicit baseline is **CAP design capacity = 1500 kAF/yr** — AZ's
# Colorado River apportionment is 2.8 MAF total (1963 Arizona v.
# California + Colorado River Compact), of which CAP is designed to
# deliver ~1.5 MAF and the remaining ~1.3 MAF serves on-River users
# and tribal rights.  This 1500 kAF figure is consistent with the
# 2010-2021 observed CAP delivery plateau (mean ~1500 kAF/yr from
# CAP's DRI request data), the AWBA 2026 Plan of Operation, and
# WestWater Research 2026 ("Economic Impacts to Central Arizona of
# Reductions in CAP Deliveries", page 8: "delivering up to
# approximately 1.5 million acre-feet annually").
#
# Tier Year-of-Shortage assignments below are taken directly from
# WestWater 2026 Table 2 (page 10), which is the primary external
# source for the 2024-2026 Tier 1 attribution.
#
# ADWR's "Tier 1 = 30 % of CAP normal supply" is a rounded public-
# facing figure (actual: 512/1500 = 34.1 %), not a precise baseline
# definition.
#
#   Tier 0:   192 kAF mandatory cut / 1500 = 12.8 % → factor 0.872 ≈ 0.87
#   Tier 1:   512 kAF               / 1500 = 34.1 % → factor 0.659 ≈ 0.66
#   Tier 2a:  592 kAF               / 1500 = 39.5 % → factor 0.605 ≈ 0.61
#   Tier 2b:  640 kAF               / 1500 = 42.7 % → factor 0.573 ≈ 0.57
#   Tier 3:   720 kAF               / 1500 = 48.0 % → factor 0.520 ≈ 0.52
#
# Option A (mandatory cuts only).  Voluntary conservation contributions
# (e.g., AZ's 355 kAF extra in 2023) are discretionary water held back
# in Lake Mead, not a reduction in supply available to AZ — they are
# NOT added into these factors so the values are reproducible and
# externally citable.
#
# ADWR Tier declaration timeline:
#   2020, 2021: Tier 0
#   2022:       Tier 1
#   2023:       Tier 2a
#   2024:       Tier 1 (returns after 2023 Tier 2a; 512 kAF, "30 % of
#                       CAP normal supply" per ADWR)
#   2025:       Tier 1 (continued; 512 kAF cut per CAP/CAWCD)
#   2026:       Tier 1 (confirmed by USBR August 24-month study per
#                       AWBA 2026 Plan of Operation; 512 kAF = 320
#                       Interim Guidelines + 192 LBDCP; last year of
#                       the 2007 IG + 2019 DCP framework)
#
# Projection (2027-2099) factor of 0.66 represents **sustained Tier 1
# shortage conditions** — the dominant recent regime.  AZ has been in
# Tier 1 (or Tier 2a) every year 2022-2026, and USBR's 24-month
# projections suggest Lake Mead will remain in the 1050-1075 ft
# elevation band (Tier 1 trigger range) for the near-term under
# current Compact conditions.  Continuing factor 0.66 (= Tier 1's
# 512 kAF cut from 1500 kAF design capacity, ≈ 34 % reduction) past
# 2026 is the most defensible single central estimate; it preserves
# continuity at the 2026→2027 boundary (no boundary step) and
# matches the regime AZ stakeholders are actively planning around.
#
# This is a SEPARATE choice from WestWater Research (2026)'s "Basic
# Coordination" scenario.  Basic Coord assumes a maximum policy
# shortage that reduces deliveries to 237 kAF (= 663 kAF cut from
# 900 kAF baseline = 74 % reduction) — an upper-bound stress
# scenario, not a central estimate.  Basic Coordination, Extreme
# Shortage, and the DCP Tier 0-3 alternatives are all evaluated as
# alternative *what-if* trajectories in the CAP scenario step
# (`uncertaintyops.run_cap_scenario_analysis`), where the no-cut
# Baseline_900kAF row provides the counterfactual reference.
#
# Applying a single fixed factor across 74 years is a simplification
# — real Compact renegotiations happen every 15-20 years and
# climate/allocation conditions will continue to evolve.  The fixed
# 0.66 is best read as "the most likely sustained shortage condition
# we'd plan around today," not a year-specific forecast.  The CAP
# scenario sweep is the right place to look for policy-bounds
# uncertainty.
# First year CAP delivered Colorado River water to AZ.  Phoenix
# reach completed 1985; Tucson reach completed 1993.  Used to gate
# σ_USBR (Upper Basin streamflow uncertainty has no propagation
# pathway through the partition before CAP is operational because
# canal_weighted_streamflow at CAP-pixel locations contributes no
# Colorado River signal pre-1985).
CAP_OPERATIONAL_START = 1985

CAP_DELIVERY_FACTORS: dict[int, float] = {
    2020: 0.87,  # Tier 0
    2021: 0.87,  # Tier 0
    2022: 0.66,  # Tier 1
    2023: 0.61,  # Tier 2a
    2024: 0.66,  # Tier 1 (returns after 2023 Tier 2a)
    2025: 0.66,  # Tier 1
    2026: 0.66,  # Tier 1 (last year of 2007 IG + 2019 DCP framework)
    **{year: 0.66 for year in range(2027, 2100)},  # Sustained Tier 1
}

# Backwards-compat alias — older external callers may still import
# the hindcast-only name.  Will be removed after downstream code is
# updated.
CAP_HINDCAST_FACTORS = CAP_DELIVERY_FACTORS


# CAP-pixel GW weight boost during shortage years.
# The partition's density-ratio split is:
#   gw_share = (gw_weight × well_density) / (gw_weight × wd + smooth_swd)
# Post-CAP gw_weight = 0.2 (from `_era_gw_weight`), which keeps the
# partition SW-dominant even when the CAP overlay is perturbed.  During
# shortage years, AMA users shift more pumping to groundwater than the
# density-ratio at gw_weight = 0.2 can capture — regulatory Assured
# Water Supply accounting forces providers to draw on authorized GW
# allowances + LTSC reserves when CAP is curtailed.
#
# Multiplying `well_density` at CAP-served pixels by a boost factor k
# is mathematically equivalent to multiplying gw_weight by k at those
# pixels (both just scale the `gw_w × wd` numerator of the density
# ratio).  Because ML prediction runs BEFORE this perturbation, the
# total per-pixel pumping is preserved — only the GW/SW split shifts.
#
# Mapping from Tier severity to boost factor (k = target_gw_w / 0.2):
#   Tier 0   → 1.0 (no boost; just DCP contribution to Lake Mead,
#                   deliveries continue)
#   Tier 1   → 5.0 (target gw_w = 1.0, pre-CAP 1945-1980 era when
#                   AZ was GW-dominant at ~67% per USBR)
#   Tier 2a  → 7.5 (target gw_w = 1.5, pre-CAP peak 1948-1955)
#   Tier 2b  → 7.5 (same tier band as 2a)
#   Tier 3   → 10.0 (target gw_w = 2.0, approaching pre-1945 all-GW)
#
# Projection 2027-2099 inherits the Tier 1 boost (5.0) since the
# central CAP_DELIVERY_FACTORS for those years also represents
# sustained Tier 1 conditions.  This keeps the 2026→2027 boundary
# smooth and matches the regime AZ stakeholders plan around.
#
# Calibrated against WestWater 2026 ("Economic Impacts to Central
# Arizona...", Fig 4 + Section 3.6) which projects ~8.0 MAF cumulative
# GW + LTSC drawdown under Basic Coordination and ~8.7 MAF under
# Extreme Shortage over 2027-2060.  Our unboosted response was ~20 %
# of those magnitudes; the era-mapped boost brings AMA-scale behaviour
# into line with WestWater's projections while preserving the ML-
# predicted total pumping.
CAP_CUT_GW_BOOST_FACTORS: dict[int, float] = {
    2020: 1.0,   # Tier 0 — no boost
    2021: 1.0,   # Tier 0
    2022: 5.0,   # Tier 1
    2023: 7.5,   # Tier 2a
    2024: 5.0,   # Tier 1 (returns)
    2025: 5.0,   # Tier 1
    2026: 5.0,   # Tier 1
    **{year: 5.0 for year in range(2027, 2100)},  # Sustained Tier 1
}


# Persistent boost applied to nonirr_sw_rights_density at CAP-service-area
# pixels for ALL post-CAP years (1985+).  HarDWR's NonIrr SW rights
# catalog under-represents CAP M&I subcontractor allocations as
# point-of-diversion records — the regulatory water moves from CAP via
# subcontractor pipes to provider service areas without registering as
# a point-of-diversion, so the raw nonirr_sw_rights_density raster is
# sparse at CAP M&I pixels even though CAP delivers ~600-700 kAF/yr of
# M&I water.  Without this boost, the density-ratio routes most NonIrr
# volume at CAP M&I pixels to NonIrr_GW (because gw_weight × nonirr_wd
# >> smooth_nonirr_swd at sparse-NIRSWD pixels), inflating modern
# NonIrr_GW by ~0.6-1.0 MAF/yr (USGS 1990/1995/2010 GW% over by +7-8 pp).
# Boost factor 5.0 calibrated to bring 1990/2010 GW% from +7-8 pp down
# to ±2 pp.  Applied symmetrically to the CAP_CUT_GW_BOOST_FACTORS
# (which boost GW share during shortage); this constant boosts the SW
# share at CAP M&I service areas as a baseline correction.
CAP_NONIRR_SW_BOOST = 5.0
CAP_NONIRR_SW_BOOST_START_YEAR = 1985  # CAP Phoenix reach completion
                                        # (1985); boost active across
                                        # all post-CAP years for
                                        # consistency.  1985 itself is
                                        # already well-calibrated by
                                        # the partition (USGS GW% +1.8
                                        # pp) but the boost is symmetric
                                        # with CAP_OPERATIONAL_START so
                                        # pre/post-CAP behavior is
                                        # cleanly demarcated.


def apply_cap_delivery_perturbation(
        year_df: pd.DataFrame,
        year: int,
        cap_pixel_mask: np.ndarray | None,
) -> pd.DataFrame:
    """Scale CAP-pixel SW signal + boost GW weight during CAP cuts.

    Applies to both observed hindcast cuts (2020-2026 Tier 0/1/2a
    declarations per ADWR/USBR/AWBA 2026 Plan) and the projection-era
    central baseline (2027-2099 = WestWater "Basic Coordination"
    sustained cut).

    Two complementary effects on the density-ratio partition at CAP-
    served pixels:

    1. **SW-signal reduction** — scales BOTH
       ``canal_weighted_streamflow_mm`` AND the SW rights density
       columns (``irr_sw_rights_density`` / ``nonirr_sw_rights_density``
       / ``sw_rights_density``) by ``CAP_DELIVERY_FACTORS[year]``.
       Scaling both is physically defensible — Tier shortage cuts
       reduce both deliverable canal flow AND the effective SW rights
       honored.  The smoothed SW kernel
       ``smooth_swd = gaussian_filter(swd × cw_sf)`` then drops
       proportionally to the product (factor² at perturbed pixels).

    2. **GW-weight boost** — multiplies ``well_density`` columns
       (``well_density`` / ``irr_well_density`` /
       ``nonirr_well_density``) by ``CAP_CUT_GW_BOOST_FACTORS[year]``.
       Because the density ratio is
       ``gw_share = (gw_w × wd) / (gw_w × wd + smooth_swd)``, this is
       mathematically equivalent to boosting ``gw_weight`` at CAP
       pixels — it shifts the GW/SW allocation toward GW during
       shortage years without changing the ML-predicted total
       pumping (ML runs before this perturbation).  This captures
       the regulatory Assured Water Supply shift to groundwater that
       the post-CAP gw_weight = 0.2 schedule otherwise under-predicts.

    Both effects are applied in the same copy operation; if either
    dict lacks the year the corresponding effect is skipped.

    Args:
        year_df: Per-pixel-per-year DataFrame.
        year: Prediction year.
        cap_pixel_mask: Boolean mask of CAP-service-area pixels
            aligned with year_df row order, or None to skip
            perturbation entirely.

    Returns:
        year_df (perturbed copy if year is in CAP_DELIVERY_FACTORS
        or CAP_CUT_GW_BOOST_FACTORS, and cap_pixel_mask is not None;
        same object otherwise).
    """
    if cap_pixel_mask is None:
        return year_df
    sw_factor = CAP_DELIVERY_FACTORS.get(year)
    gw_boost = CAP_CUT_GW_BOOST_FACTORS.get(year)
    apply_nirswd_boost = year >= CAP_NONIRR_SW_BOOST_START_YEAR
    if (sw_factor is None and (gw_boost is None or gw_boost == 1.0)
            and not apply_nirswd_boost):
        return year_df

    sw_cols = [
        c for c in (
            'canal_weighted_streamflow_mm',
            'irr_sw_rights_density',
            'nonirr_sw_rights_density',
            'sw_rights_density',
        ) if c in year_df.columns
    ]
    wd_cols = [
        c for c in (
            'well_density',
            'irr_well_density',
            'nonirr_well_density',
        ) if c in year_df.columns
    ]
    if not sw_cols and not wd_cols:
        return year_df

    year_df_p = year_df.copy()
    idx = year_df_p.index[cap_pixel_mask]
    # Apply persistent NonIrr SW boost FIRST (1985+, all CAP-service
    # pixels, regardless of shortage) so any subsequent shortage
    # reduction (sw_factor) acts on the boosted value.  Order matters:
    # boosted_value × sw_factor preserves the relative shortage signal
    # at CAP M&I pixels.
    if (apply_nirswd_boost
            and 'nonirr_sw_rights_density' in year_df_p.columns):
        year_df_p.loc[idx, 'nonirr_sw_rights_density'] *= CAP_NONIRR_SW_BOOST
    if sw_factor is not None:
        for col in sw_cols:
            year_df_p.loc[idx, col] *= sw_factor
    if gw_boost is not None and gw_boost != 1.0:
        for col in wd_cols:
            year_df_p.loc[idx, col] *= gw_boost
    return year_df_p


# Backwards-compat alias for the old hindcast-only function name.
apply_cap_hindcast_perturbation = apply_cap_delivery_perturbation


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
GW_WEIGHT_PRE_1945 = 10.0  # pre-1945: USGS shows ~100% GW (no/minimal SW
                           # except SRP); push very strongly toward GW.
                           # Bumped from 5.0 → 10.0 alongside PRE_GMA
                           # extension (see GW_WEIGHT_1945_1980 below).
GW_WEIGHT_1945_1970 = 10.0 # 1945-1970 (pre-GMA, non-peak years): bumped
                           # from 2.0 → 10.0 to recover the 1955-1965
                           # GW% under-attribution.  Small canal basins
                           # (Verde, Upper San Pedro, Coconino) retain
                           # SW post-canal-gate change, so density-ratio
                           # needs heavier GW weight to push pre-CAP
                           # statewide GW share back toward USGS.
GW_WEIGHT_PEAK = 15.0      # Peak sub-era 1971-1979: extra GW push to
                           # close the IrrGW residual at 1975 (USGS
                           # 5.01 MAF, model under by ~0.9 MAF).
                           # 1980 dropped back to GW_WEIGHT_1945_1970
                           # (10.0) since +15 pushed GW% to +2.5 pp
                           # over USGS at 1980.
GW_WEIGHT_PEAK_HIGH = 30.0 # Sub-era 1950-1965: very strong GW push
                           # to close persistent GW% under-attribution
                           # (model -5 to -7 pp under USGS at 1950,
                           # 1955, 1960, 1965).  Bumped from 20 → 30 to
                           # cover the wider 1950-1965 window (was just
                           # 1951-1957).
GW_WEIGHT_PRE_GMA = 2.0    # 1981-1984 GMA transition: keep prior
                           # baseline (2.0) so the 1985-1989 ramp to
                           # POST_CAP starts smoothly without a
                           # discontinuity at 1985.
GW_WEIGHT_POST_CAP = 0.2   # post-CAP era.  This value is *load-bearing*
                           # for the WestWater 2026 cumulative-drawdown
                           # calibration of CAP_CUT_GW_BOOST_FACTORS.  The
                           # tier boost of 5.0× at 2022+ CAP pixels is
                           # calibrated to match WestWater Fig 4 with the
                           # baseline at 0.2 — so effective gw_weight at
                           # CAP pixels = 0.2 × 5.0 = 1.0.  Lifting the
                           # baseline (e.g. to 0.4) would also lift the
                           # tier-era effective to 2.0, doubling the
                           # post-2022 CAP-pixel GW substitution and
                           # over-shooting WestWater.  Per-basin Phoenix /
                           # Tucson / Prescott / Hualapai under-attribution
                           # (model GW share lower than ADWR) is a separate
                           # known limitation documented in the methods
                           # README; addressing it requires either per-basin
                           # GW floors anchored to verified ADWR per-AMA
                           # tables, or new feature engineering to
                           # distinguish on-canal vs off-canal wells inside
                           # AMA service areas.

GW_WEIGHT_EARLY_CAP = 0.1   # 1990-1997 early-CAP era.  Bumped down from
                            # 0.2 to lift SW share at 1990/1995 where
                            # USGS shows 56-57 % SW (40-41 % GW); the
                            # 0.2 baseline left model GW% +7 pp over USGS.
                            # Window stops at 1997 to preserve the 1998+
                            # MID_CAP rebound and POST_CAP=0.2 calibration
                            # anchor for the WestWater 2022+ CAP-cut tier.
GW_WEIGHT_MID_CAP = 0.4     # 1998–2007 mid-CAP bump to lift GW% at 2000/2005
                            # (USGS rebound: 1995=40.5, 2000=50.9, 2005=48.8).
                            # Slightly above POST_CAP baseline.
GW_WEIGHT_1930_1935 = 10.0  # 1930-1935 bump (now equal to PRE_1945/PRE_GMA
                            # since both are 10.0; kept for historical
                            # documentation of the 1935 calibration)
                            # — model 1935 Total ~ USGS
                            # Total_GW (1.38 vs 1.20) but GW share too
                            # low (0.91 vs 1.20) because SRP canal-pixel
                            # density-ratio attributed too much to SW.
                            # Higher gw_weight pushes canal-fringe SW
                            # toward GW, lifting GW from 0.91 → ~1.10.


def _era_gw_weight(year: int) -> float:
    """Return era-dependent GW weight for the density-ratio split.

    Pre-1945: weight = 10.0 (very GW-dominant, matches USGS pre-1945
        showing essentially all pumping was GW).
    1945–1980: weight = 10.0 (full pre-GMA bump).  Bumped from 2.0 to
        recover the 1955-1975 GW% under-attribution (model 56-58 % vs
        USGS 62-69 %) after small canal basins (Verde, Upper San Pedro,
        Coconino) regained their SW signal under the relaxed canal
        coverage gate (1 % + cw_streamflow > 0).
    1981–1984: weight = 2.0 (GMA transition).  Returns to the original
        baseline so the 1985-1989 ramp to POST_CAP starts smoothly
        without a discontinuity at 1985.
    1986–1989: linear ramp 2.0 → 0.1 as CAP comes online.
    1990–1997: weight = 0.1 (early-CAP, USGS shows 56-57 % SW share).
    1998–2007: weight = 0.4 (mid-CAP bump).  USGS shows GW% rebound
        in this window (1995=40.5, 2000=50.9, 2005=48.8) that the
        flat-0.2 weight under-shoots by 4–7 pp.  Stepping up to 0.4
        for these years lifts GW% without touching 1990/2010 (already
        in band).
    2008+: weight = 0.2 (post-CAP SW-dominant era).  This value is
        load-bearing for the WestWater-calibrated CAP-shortage tier
        boosts at 2022+ (CAP_CUT_GW_BOOST_FACTORS = 5.0× at CAP pixels
        gives effective gw_weight = 0.2 × 5.0 = 1.0, which matches
        WestWater Fig 4 cumulative drawdown).  Per-basin under-
        attribution at metro AMAs (Phoenix/Tucson/Prescott/Hualapai)
        is a separate known limitation requiring per-basin floors
        anchored to verified ADWR data — not addressable by tuning
        this baseline.
    """
    if 1930 <= year <= 1935:
        return GW_WEIGHT_1930_1935
    if year < 1945:
        return GW_WEIGHT_PRE_1945
    if 1950 <= year <= 1965:
        return GW_WEIGHT_PEAK_HIGH
    if 1971 <= year <= 1979:
        return GW_WEIGHT_PEAK
    if year <= 1970 or year == 1980:
        return GW_WEIGHT_1945_1970
    if year < 1985:
        return GW_WEIGHT_PRE_GMA
    if 1990 <= year <= 1997:
        return GW_WEIGHT_EARLY_CAP
    if 1998 <= year <= 2007:
        return GW_WEIGHT_MID_CAP
    if year >= CAP_FULL_YEAR:
        return GW_WEIGHT_POST_CAP
    # 1985-1989 ramp 2.0 → 0.1 (year=1985 returns 2.0 at frac=0).
    frac = (year - 1985) / (CAP_FULL_YEAR - 1985)
    return GW_WEIGHT_PRE_GMA + frac * (GW_WEIGHT_EARLY_CAP - GW_WEIGHT_PRE_GMA)


def era_sw_sigma(year: int) -> float:
    """Return the era-dependent SW-smoothing σ (Gaussian kernel width).

    Single source of truth for the SW canal-reach kernel σ used by
    ``partition_predictions``.  Production callers leave the
    ``sw_smooth_sigma`` argument None and get this value automatically;
    the σ-sensitivity diagnostic in ``uncertaintyops`` calls this
    helper to anchor a per-year sweep around the actual production σ
    instead of a fixed global pair.

    Schedule (calibrated against USGS Total_GW & GW/SW shares):

    | Era                | σ                                |
    |--------------------|-----------------------------------|
    | < 1912             | 0.0                               |
    | 1912–1948          | piecewise-linear through anchors  |
    | 1948–1955          | 1.5                               |
    | 1956–1964          | 1.0                               |
    | 1965–1984          | linear ramp 1.0 → 4.0             |
    | 1973–1977 override | 2.0 (drought)                     |
    | 1985–2002          | 4.0                               |
    | 2003–2010 override | 3.0 (CAP-era ag retirement)       |
    | 2008–2021          | linear ramp 4.0 → 6.0             |
    | 2022+              | 6.0                               |

    Anchor points for 1912–1948 (year, σ):
    (1912, 0.0), (1915, 1.5), (1917, 0.3), (1929, 0.3),
    (1935, 0.0), (1940, 0.3), (1945, 1.0), (1948, 1.5).
    """
    if year < 1912:
        return 0.0
    if year < 1948:
        anchors = [
            (1912, 0.0), (1915, 1.5), (1917, 0.3), (1929, 0.3),
            (1935, 0.0), (1940, 0.3), (1945, 1.0), (1948, 1.5),
        ]
        for i in range(len(anchors) - 1):
            y0, s0 = anchors[i]
            y1, s1 = anchors[i + 1]
            if y0 <= year <= y1:
                if y1 == y0:
                    return s0
                frac = (year - y0) / (y1 - y0)
                return s0 + frac * (s1 - s0)
        return 0.0
    if 1948 <= year <= 1955:
        return 1.5
    if year < 1965:
        return 1.0
    if year >= 2022:
        return 6.0
    if 2003 <= year <= 2010:
        return 3.0
    if year >= 2008:
        frac = (year - 2008) / (2022 - 2008)
        return 4.0 + frac * (6.0 - 4.0)
    if year >= 1985:
        return 4.0
    if 1973 <= year <= 1977:
        return 2.0
    # 1965-1984 ramp (20 years)
    frac = (year - 1965) / (1985 - 1965)
    return 1.0 + frac * (4.0 - 1.0)


def partition_predictions(
        predictions: np.ndarray,
        year_df,
        raster_shape: tuple,
        valid_mask: np.ndarray,
        sw_smooth_sigma: float | None = None,
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
        sw_smooth_sigma (float, optional): If passed, **overrides** the
            era-based piecewise SW smoothing schedule with this single
            value (used by the density-ratio sensitivity diagnostic to
            sweep σ at constant year).  Default None → use the
            calibrated era schedule (recommended for production runs).
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

    # Peak-year basin lift on partition-side well-density columns.
    # Mirrors the ML-feature basin-max lift in
    # apply_ml_well_density_override.  Without this, ML sees the
    # lifted well_density and predicts big peak-year volumes, but
    # the density-ratio split here would still use sparse year-
    # specific irr_wd / nonirr_wd → smooth_swd dominates → SW wins
    # → GW% under-predicted at peak years.
    #
    # Split aggregator:
    #   * well_dens → basin-MAX.  Used for has_well flipping; the
    #     max ensures all cf | uf>0.2 pixels have registered wells
    #     so lu_urban_only volume recovery reaches them.
    #   * irr_wd / nonirr_wd → basin-MEDIAN.  Used in the density-
    #     ratio GW/SW split: gw_share = (gw_weight × irr_wd) /
    #     (gw_weight × irr_wd + smooth_swd).  Basin-MAX here
    #     over-powered smooth_swd, forcing gw_share ≈ 1 and
    #     over-attributing GW at peak years (1970: 75 % vs USGS 62 %;
    #     1975: 72 % vs 62 %; 1980: 61 % vs 53 %).  Basin-median
    #     tempers the ratio so SW (SRP canal deliveries at peak
    #     pre-CAP years) can still compete with GW in the
    #     denominator.
    #
    # Two lifts (mirror ML-feature lifts in
    # apply_ml_well_density_override):
    #   (A) Peak-year LU lift (cf>0|uf>0.2): basin-MAX for
    #       well_dens, basin-MEDIAN for irr_wd/nonirr_wd, at peak
    #       years (1951-55, 1970-80).
    #   (B) All-pre-1981 ORPHAN lift (cf=0, uf=0, year_wd>0):
    #       basin-p90 for well_dens, basin-p75 for irr_wd/nonirr_wd
    #       (slightly softer to avoid GW-share over-routing),
    #       across all pre-1981 years.  Pre-GMA registry attrition
    #       affects orphans throughout, not just peak years.
    _is_peak_wd = (1951 <= year <= 1957) or (1970 <= year <= 1980)
    _is_pre_1981 = year < 1981

    # Peak-year crop-edge halo (1951-1955, 1970-1980): pixels NOT in
    # cf>0 but IMMEDIATELY ADJACENT (1-cell dilation) to a cf>0 pixel,
    # AND with AGRI > PEAK_HALO_AGRI.  CDL/IrrMapper at peak years
    # under-maps real ag at field-edge pixels — extending the
    # effective cropland mask there:
    #   (1) Retains those pixels' ML predictions (added to _has_crop_any
    #       in the retention block below) → pre-1980 IRR_OVERRIDE routes
    #       95 % to Irr.
    #   (2) Lifts their well_density to basin-MAX in the peak-year wd
    #       lift below, treating them as basin-typical ag wells (so the
    #       density-ratio split routes them GW-favored, matching the
    #       ag-well regime).
    # Urban-edge halo (same construction for uf>0.2 → URBAN>0.01
    # adjacency) captures field-fringe suburbia that GEE-LULC misses
    # at coarse resolution — these pixels route via the URBAN_REAL
    # 1-uf branch or the pre-1980 0.95 IRR_OVERRIDE.
    PEAK_HALO_AGRI = 0.10
    PEAK_HALO_URBAN = 0.30
    # Year-conditional crop core threshold:
    #   1951-1955 + 1971-1980: looser cf>0.1 core so more pixels seed
    #     the 2-cell dilation halo (these years are most under USGS).
    #   1956-1957: tighter cf>0.2 core to soften the 1957→1958 cliff
    #     (1957 ADWR overshoot was driven by combined 0.1 core + 20.0
    #     gw_weight + halo activation).
    #   1970: kept at 0.2 to preserve its ΔT +0.24 calibration.
    if (1951 <= year <= 1955) or (1971 <= year <= 1980):
        PEAK_HALO_CF_CORE = 0.10
    else:
        PEAK_HALO_CF_CORE = 0.20
    PEAK_HALO_UF_CORE = 0.30
    crop_edge_halo = np.zeros(len(predictions), dtype=bool)
    urban_edge_halo = np.zeros(len(predictions), dtype=bool)
    if _is_peak_wd:
        from scipy.ndimage import binary_dilation
        if crop_frac_col is not None and 'AGRI' in year_df.columns:
            _cf_arr = np.nan_to_num(crop_frac_col, nan=0.0)
            _agri_arr = np.nan_to_num(year_df['AGRI'].values, nan=0.0)
            _crop_grid = np.zeros(raster_shape, dtype=bool)
            _crop_grid.ravel()[valid_mask] = (_cf_arr > PEAK_HALO_CF_CORE)
            # 2-cell dilation (4 km reach at 2 km pixel) — real cotton /
            # alfalfa fields commonly span several pixels at coarse
            # resolution, so 2-cell adjacency is physical.
            _dilated = binary_dilation(_crop_grid, iterations=2)
            _halo_flat = (_dilated & ~_crop_grid).ravel()[valid_mask]
            crop_edge_halo = _halo_flat & (_agri_arr > PEAK_HALO_AGRI)
        if urban_frac_col is not None and 'URBAN' in year_df.columns:
            _uf_arr = np.nan_to_num(urban_frac_col, nan=0.0)
            _urb_arr = np.nan_to_num(year_df['URBAN'].values, nan=0.0)
            _urban_grid = np.zeros(raster_shape, dtype=bool)
            _urban_grid.ravel()[valid_mask] = (_uf_arr > PEAK_HALO_UF_CORE)
            _dilated_u = binary_dilation(_urban_grid, iterations=1)
            _halo_flat_u = (_dilated_u & ~_urban_grid).ravel()[valid_mask]
            urban_edge_halo = _halo_flat_u & (_urb_arr > PEAK_HALO_URBAN)

    if ((_is_peak_wd or _is_pre_1981) and basin_names is not None
            and crop_frac_col is not None
            and urban_frac_col is not None):
        cf_peak = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        uf_peak = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        # lu_gate_peak now also includes crop-edge and urban-edge halo
        # pixels so the peak-year basin-MAX wd lift extends to them.
        lu_gate_peak = (
            (cf_peak > 0) | (uf_peak > 0.2) | crop_edge_halo | urban_edge_halo
        )
        orphan_has_well = (
            (cf_peak == 0) & (uf_peak == 0) & (well_dens > 0)
        )
        _basin_series_wd = pd.Series(basin_names)
        _lift_specs = [
            # (array, snapshot, peak-LU agg, orphan agg)
            (well_dens, wd_1981, 'max', 'p90'),
            (irr_wd, irr_wd_1981, 'median', 'p75'),
            (nonirr_wd, nonirr_wd_1981, 'median', 'p75'),
        ]
        for _yr_arr, _snap, _peak_agg, _orph_agg in _lift_specs:
            if _snap is None:
                continue
            _snap_safe = np.nan_to_num(_snap, nan=0.0)
            for _b in _basin_series_wd.dropna().unique():
                _b_mask = (basin_names == _b)
                _b_pool = _snap_safe[_b_mask & (_snap_safe > 0)]
                if _b_pool.size == 0:
                    continue
                # (A) Peak-year LU lift.
                if _is_peak_wd:
                    _b_lu_val = (
                        float(np.max(_b_pool)) if _peak_agg == 'max'
                        else float(np.median(_b_pool))
                    )
                    _b_lu_lift = _b_mask & lu_gate_peak
                    if _b_lu_lift.any():
                        _yr_arr[_b_lu_lift] = np.maximum(
                            _yr_arr[_b_lu_lift], _b_lu_val,
                        )
                # (B) Pre-1981 orphan lift.
                if _is_pre_1981:
                    if _orph_agg == 'p90':
                        _b_orph_val = float(np.quantile(_b_pool, 0.9))
                    else:  # 'p75'
                        _b_orph_val = float(np.quantile(_b_pool, 0.75))
                    _b_orph_lift = _b_mask & orphan_has_well
                    if _b_orph_lift.any():
                        _yr_arr[_b_orph_lift] = np.maximum(
                            _yr_arr[_b_orph_lift], _b_orph_val,
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
    # Peak-year crop-edge and urban-edge halos: extend retention to
    # cropland-adjacent AGRI-rich pixels and urban-adjacent URBAN-rich
    # pixels (computed above before the peak-year wd lift).
    _has_crop_any = _has_crop_any | crop_edge_halo
    _has_urban_any = _has_urban_any | urban_edge_halo
    # Pre-1985 AGRI retention extension removed — the previous
    # outside-AMA AGRI > 0.02 / 0.10 gates relied on the smoothed AGRI
    # density to capture patchy rural ag that CDL's raw crop_frac
    # misses.  The smoothing created hindcast-era halos that show up
    # as ring artifacts in the predicted maps.  Reverting to raw
    # crop_frac-only retention eliminates the halos; expect the 1956
    # / 1965 / 1975-80 outside-AMA Irr totals to drop by whatever
    # volume the AGRI extension was propping up.
    _agri_extension_active = False
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

    # Phantom-icap gate (auto-detected low-cropland basins):
    # The 2024 ADWR Well Registry codes many wells "IRRIGATION" at
    # places with no actual cropland (hobby farms, vineyards,
    # residential ag, recreational use).  This inflates irr_frac at
    # basins with little / no real ag.
    #
    # Auto-rule: a basin is "phantom" if its total cropland is too
    # sparse to support meaningful irrigation —
    #   basin_cf_sum < PHANTOM_BASIN_CF_SUM_MAX  (5.0 cf-units)
    #   OR basin_n_cf_pos < PHANTOM_BASIN_N_CF_POS_MIN  (10 pixels)
    # At phantom basins, zero out irr_frac at pixels with neither
    # cropland (cf>0) nor moderate AGRI (AGRI > AGRI_THRESHOLD).
    # cf_floor below still preserves real cropland (cf>0 → irr_frac=cf).
    #
    # Catches both the demonstrated 9 (Prescott, Verde, Big Sandy,
    # Sacramento Valley, Bill Williams, Joseph City, Upper/Lower
    # San Pedro, Little CO Plateau) AND additional low-ag basins
    # (Detrital Valley, Salt River, Coconino Plateau, Aravaipa,
    # Cienega Creek, Upper Hassayampa, Agua Fria, Tonto Creek, etc.).
    # Real-ag basins (Phoenix, Pinal, Yuma, Willcox, Safford, Lower
    # Gila, Harquahala, Tucson, McMullen, Hualapai) all have
    # cf_sum >> 5 and n_cf_pos >> 10, so they pass the gate.
    PHANTOM_ICAP_AGRI_THRESHOLD = 0.05
    PHANTOM_BASIN_CF_SUM_MAX = 5.0
    PHANTOM_BASIN_N_CF_POS_MIN = 10
    if (irr_cap_col is not None and crop_frac_col is not None
            and basin_names is not None
            and 'AGRI' in year_df.columns):
        cf_arr = np.nan_to_num(crop_frac_col, nan=0.0)
        agri_arr = np.nan_to_num(year_df['AGRI'].values, nan=0.0)
        cf_gate = cf_arr > 0
        agri_gate = agri_arr > PHANTOM_ICAP_AGRI_THRESHOLD
        is_ag = cf_gate | agri_gate
        # Auto-detect phantom basins via per-basin cropland scarcity.
        cf_series = pd.Series(cf_arr)
        basin_cf_sum = cf_series.groupby(basin_names).transform('sum').values
        basin_n_cf_pos = pd.Series(cf_gate.astype(int)).groupby(
            basin_names).transform('sum').values
        in_phantom_basin = (
            (basin_cf_sum < PHANTOM_BASIN_CF_SUM_MAX)
            | (basin_n_cf_pos < PHANTOM_BASIN_N_CF_POS_MIN)
        )
        # Only mask icap at phantom basins; non-phantom basins unchanged.
        keep = is_ag | ~in_phantom_basin
        irr_frac = np.where(keep, irr_frac, 0.0)

    # Adaptive crop-fraction floor: irr_frac is at least crop_frac at
    # every pixel.  Pure crop pixel (cf=1) → irr_frac=1.  Pure desert
    # pixel (cf=0) → no boost.  Mixed pixel (cf=0.5) → irr_frac >= 0.5.
    # Naturally addresses the boxplot outliers where old ag wells got
    # misclassified as M&I in the 2024 registry — those pixels keep
    # high crop_frac and the floor lifts their irr_frac accordingly.
    if crop_frac_col is not None:
        cf_floor = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        irr_frac = np.maximum(irr_frac, cf_floor)

    # Peak-year basin-MAX irr_capacity lift: same registry-incompleteness
    # argument as the well_density basin-max lift — pre-GMA irrigation
    # wells were voluntarily registered, and the 2024 year-specific
    # irr_capacity_fraction is sparse at 1951-55 / 1970-80 (only ~3-6 %
    # of valid pixels have irr_cap > 0 vs ~9 % in 2020).  At peak years,
    # at any cf > 0 pixel in a basin with ANY registered irrigation,
    # lift irr_frac to the basin's MAX irr_cap_col value.  This routes
    # volume from NonIrr default back to Irr at halo ag pixels that
    # were under-attributed.
    #
    # Subsequent IRR_OVERRIDE branches still run: pre-1970 overwrites
    # everything with 0.95 (basin-max has no effect at 1951-55), and
    # 1970-1985 non-AMA overwrites with 1-uf (basin-max has no effect
    # at non-AMA in 1970-1985).  The lift is therefore only visible at
    # 1970-1985 AMA pixels (Phoenix/Pinal/Tucson/etc.) where the
    # natural irr_capacity_fraction is used — the exact basins where
    # we saw Irr% under-prediction at 1975/1980.
    _is_peak_wd = (1951 <= year <= 1957) or (1970 <= year <= 1980)
    if (_is_peak_wd and irr_cap_col is not None
            and basin_names is not None
            and crop_frac_col is not None):
        cf_peak = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        irr_cap_safe = np.clip(np.nan_to_num(irr_cap_col, nan=0.0), 0, 1)
        _basin_series_ic = pd.Series(basin_names)
        for _b in _basin_series_ic.dropna().unique():
            _b_mask = (basin_names == _b)
            _b_pool = irr_cap_safe[_b_mask & (irr_cap_safe > 0)]
            if _b_pool.size == 0:
                continue
            _b_lift = _b_mask & (cf_peak > 0)
            if _b_lift.any():
                irr_frac[_b_lift] = np.maximum(
                    irr_frac[_b_lift], float(np.max(_b_pool)),
                )

    # Irr-fraction overrides:
    #   (1) Through 1980 pixels with uf < 0.3 (rural + suburban
    #       fringe + LULC halo): force 0.95 flat.  USGS shows ag was
    #       89-97 % of AZ-total pumping through 1980; the flat 0.95
    #       is well-calibrated at rural / ag-dominated pixels and
    #       suburban-fringe pixels.  0.3 is the urban-core threshold
    #       — below it, pre-1980 M&I per-pixel was small enough to
    #       treat as ~5 % NonIrr; above it, the urban share is large
    #       enough to matter.
    #   (2) Through 1980 pixels with uf >= 0.3 (real urban cores,
    #       any basin): irr_frac = 1 − urban_frac (clipped
    #       [0.05, 1]).  Applies regardless of AMA status —
    #       Flagstaff, Lake Havasu City, Bullhead City, Page,
    #       Phoenix / Tucson urban cores.  Routes the uf share to
    #       NonIrr (municipal / industrial), the (1 - uf) share to
    #       Irr.  The 0.3 threshold (raised from 0.2) recovers the
    #       peak-year Irr% that over-correction at suburban-fringe
    #       pixels was dropping.
    #   (3) 1981–1985 NON-AMA: irr_frac = 1 − urban_frac.
    #   (4) 1986+ NON-AMA: Option C via post-1985 LU-aware branch.
    #   (5) 1981+ AMA: natural year-specific irr_capacity_fraction.
    IRR_OVERRIDE_PRE_1970 = 0.95
    IRR_OVERRIDE_FLOOR = 0.05
    URBAN_REAL_THRESHOLD = 0.3
    if year <= 1980:
        irr_frac = np.full_like(irr_frac, IRR_OVERRIDE_PRE_1970)
        # Real-urban pixels (uf >= 0.3) get 1 - uf routing regardless
        # of AMA status — M&I pumping at any urban core.
        if urban_frac_col is not None:
            uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
            urban_real = uf >= URBAN_REAL_THRESHOLD
            if urban_real.any():
                irr_frac[urban_real] = np.clip(
                    1.0 - uf[urban_real],
                    IRR_OVERRIDE_FLOOR, 1.0,
                )
    elif (year < 1986 and basin_names is not None
            and urban_frac_col is not None):
        non_ama = ~np.isin(basin_names, list(get_ama_basins(year)))
        uf = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        irr_from_uf = np.clip(1.0 - uf, IRR_OVERRIDE_FLOOR, 1.0)
        irr_frac = irr_frac.copy()
        irr_frac[non_ama] = irr_from_uf[non_ama]

    # Orphan-pixel Irr/NonIrr refinement — PRE-CAP ERA ONLY
    # (year <= 1985).  ~30 % of every year's registered wells land
    # on pixels with cf = 0 AND uf = 0 (no raw CDL crop or urban
    # class).  Pre-CAP: route AGRI-dominated orphans → 0.95 Irr
    # (CDL-miss ag) and URBAN-dominated orphans via a shallow
    # 1 − URBAN gradient.  POST-CAP (1986+): fall back to the
    # previous-commit default-partition behaviour — orphans at
    # 1-uf = 1 (uf=0) go to 100 % Irr.  Reverted because orphan
    # URBAN-gradient was shifting volume into NonIrr at 2020 that
    # then couldn't retain SW (raw-uf excess routing scaled it to 0)
    # → NonIrr_GW share inflated from ~60 % to 80 %.
    if (year <= 1985
            and crop_frac_col is not None and urban_frac_col is not None
            and 'AGRI' in year_df.columns
            and 'URBAN' in year_df.columns):
        cf_arr = np.clip(np.nan_to_num(crop_frac_col, nan=0.0), 0, 1)
        uf_arr = np.clip(np.nan_to_num(urban_frac_col, nan=0.0), 0, 1)
        orphan = (cf_arr == 0) & (uf_arr == 0) & has_well
        if orphan.any():
            agri_smooth = np.nan_to_num(
                year_df['AGRI'].values, nan=0.0,
            )
            urb_smooth = np.clip(np.nan_to_num(
                year_df['URBAN'].values, nan=0.0,
            ), 0, 1)
            orp_agri_dom = orphan & (agri_smooth > urb_smooth)
            orp_urb_dom = orphan & (urb_smooth >= agri_smooth) & (
                urb_smooth > 0
            )
            irr_frac = irr_frac.copy()
            # AGRI-dominated orphans → Irr (CDL-miss ag).
            irr_frac[orp_agri_dom] = IRR_OVERRIDE_PRE_1970  # 0.95
            # URBAN-dominated orphans → shallow gradient.
            irr_frac[orp_urb_dom] = np.clip(
                1.0 - urb_smooth[orp_urb_dom],
                IRR_OVERRIDE_FLOOR, 1.0,
            )
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
        # AGRI-halo gate (post-1985 ONLY): pixels with cf = 0 but
        # smoothed AGRI > 0.1 are in an ag halo.  CDL at 2000 m
        # under-maps patchy rural ag (Willcox, Harquahala, Douglas,
        # Pinal-fringe), but smoothed AGRI captures the footprint.
        # Route halo pixels into only_crop so they get full Irr —
        # otherwise they fall into pure_desert_with_well at AMA
        # basins where irr_cap = 0 → 100 % NonIrr → density-ratio
        # routes ~85 % to NonIrr_GW, inflating NonIrr by ~0.5-1.0 MAF
        # at every post-1985 anchor.  Restored at 1986+ only — the
        # pre-1985 hindcast halo issue was actually from the lu_only
        # AGRI extension + pre-1985 SW-kernel ramping (still
        # removed), not from this gate.
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
        # Previously the peak-year outside-AMA branch also included
        # AGRI > threshold — that gate is removed to eliminate the
        # smoothed-AGRI halo that showed up in hindcast maps.
        LU_CROP_THRESHOLD = 0.05
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
    SW_SIGMA = era_sw_sigma(year)
    # Caller-supplied σ override (used by the density-ratio
    # sensitivity diagnostic to sweep σ at constant year).  Production
    # callers leave this None and use the era schedule above.
    if sw_smooth_sigma is not None:
        SW_SIGMA = float(sw_smooth_sigma)
    gw_weight = _era_gw_weight(year)

    # Cap GW share at CO-river-direct basins so density-ratio can't
    # route everything to GW when irr_sw_rights_density is under-
    # represented in the raster (Parker / Lake Havasu have near-zero
    # sw_rights pixels despite real CRIT / mainstem SW access, so
    # density ratio naturally produces gw_share = 1 there).
    # Cap is per-pixel and per-basin: ag-dominant CRIT (Parker 0.15)
    # and Yuma Project (Yuma 0.10) basins use tighter caps than the
    # 0.40 default for M&I-dominant Lake Havasu / Detrital / Meadview.
    if basin_names is not None:
        co_gw_cap = _co_direct_gw_cap_array(basin_names)
        basin_cap = _basin_gw_cap_array(basin_names)
        # Combine CO-direct cap with non-CO basin cap via min (tighter wins).
        co_gw_cap = np.minimum(co_gw_cap, basin_cap)
        gw_floor = _basin_gw_floor_array(basin_names, year=year)
    else:
        co_gw_cap = np.full(len(predictions), np.inf, dtype=np.float64)
        gw_floor = np.zeros(len(predictions), dtype=np.float64)

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
        irr_gw_share = np.minimum(irr_gw_share, co_gw_cap)
        irr_gw_share = np.maximum(irr_gw_share, gw_floor)
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
        nonirr_gw_share = np.minimum(nonirr_gw_share, co_gw_cap)
        nonirr_gw_share = np.maximum(nonirr_gw_share, gw_floor)
        nonirr_gw = nonirr * nonirr_gw_share
        nonirr_sw = nonirr - nonirr_gw
    else:
        nonirr_gw = nonirr.copy()
        nonirr_sw = np.zeros_like(nonirr)

    # Basin-level canal-infra gate (all eras): collapse SW → GW in
    # basins that don't have meaningful canal infrastructure.
    # Gaussian smoothing of sw_rights_density + cw_streamflow bleeds
    # signals across basin boundaries, producing phantom SW in GW-
    # only basins like Willcox / Douglas / Joseph City / Lower Gila.
    #
    # Two-part gate:
    #   (a) Require BASIN CANAL COVERAGE >= CANAL_COVERAGE_GATE_THRESHOLD
    #       (2 % of basin pixels).  A 1 % threshold previously let
    #       Douglas (1.12 % coverage from 7-8 spurious historical
    #       pixels) leak SW; bumping to 2 % catches it without
    #       affecting any real-canal basin (Phoenix ≈ 40 %, Pinal ≈
    #       30 %, Yuma > 10 %).  Willcox (0.16 %) was already caught.
    #   (b) WHITELIST direct-CO-river basins (CO_RIVER_DIRECT_BASINS):
    #       bypass the gate regardless of canal coverage.  Parker,
    #       Yuma, Lake Mohave, Lake Havasu, etc. have physical
    #       mainstem river access via riverside diversions that
    #       don't register as canal_density pixels — without this
    #       bypass, pre-CAP Parker shows 100 % GW (wrong — CRIT
    #       senior rights deliver ~720 kAF/yr from Colorado direct).
    #
    # NonIrr_SW handling: collapse at no-canal basins UNLESS basin is
    # in NONIRR_SW_PRESERVE_BASINS (LCR / Sacramento Valley / Big
    # Sandy / Bill Williams — basins with direct lake/river intakes
    # serving M&I / mining demand that don't show up as canals).
    # Willcox / Douglas / Joseph City have no such intakes — they
    # are purely GW, so both Irr_SW and NonIrr_SW collapse to GW.
    if canal_dens is not None and basin_names is not None:
        canal_series = pd.Series(canal_dens)
        basin_canal_coverage = canal_series.groupby(basin_names).transform(
            lambda x: (x > 0).mean(),
        )
        # Second gate: mean basin canal_weighted_streamflow must be > 0.
        # Catches basins with spurious historical canal pixels but no
        # actual delivery capacity (Douglas 1.12 % coverage from 7-8
        # historical pixels carrying no streamflow).
        if cw_streamflow_raw is not None:
            cw_series = pd.Series(np.nan_to_num(cw_streamflow_raw, nan=0.0))
            basin_cw_mean = cw_series.groupby(basin_names).transform('mean')
            has_cw_capacity = basin_cw_mean.values > 0
        else:
            has_cw_capacity = np.ones(len(basin_canal_coverage), dtype=bool)
        is_co_direct = np.isin(basin_names, list(CO_RIVER_DIRECT_BASINS))
        no_canal_basin = (
            ((basin_canal_coverage.values < CANAL_COVERAGE_GATE_THRESHOLD)
             | ~has_cw_capacity)
            & ~is_co_direct
        )
        preserve_nonirr_sw = np.isin(
            basin_names, list(NONIRR_SW_PRESERVE_BASINS),
        )
        if no_canal_basin.any():
            irr_gw = np.where(
                no_canal_basin, irr_gw + irr_sw, irr_gw,
            )
            irr_sw = np.where(no_canal_basin, 0.0, irr_sw)
            # Collapse NonIrr_SW at no-canal basins not on the
            # preserve list.
            collapse_ni_sw = no_canal_basin & ~preserve_nonirr_sw
            if collapse_ni_sw.any():
                nonirr_gw = np.where(
                    collapse_ni_sw, nonirr_gw + nonirr_sw, nonirr_gw,
                )
                nonirr_sw = np.where(collapse_ni_sw, 0.0, nonirr_sw)

    # Pre-1945: USGS shows ~100% GW statewide.  Restrict SW to pixels
    # that actually have direct access to surface water — canal-served
    # (smoothed or direct) or holding SW rights.  Elsewhere, collapse
    # SW back into GW so density-ratio leakage (Gaussian smoothing of
    # sw_rights_density into dry cells) does not produce phantom SW.
    # This pixel-level gate is retained as a second line of defense
    # within canal-having basins — pixels in CAP service area that
    # are far from any actual canal still get phantom SW from
    # smoothing, and pre-1945 we want no SW without physical access.
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

    # Re-apply no-canal collapse AFTER the post-1985 excess routing.
    # The (1 − uf) excess pulled NonIrr_SW back into Irr_SW via the
    # density-ratio share, re-contaminating the Irr_SW that the
    # earlier gate had zeroed.  Re-running the same gate here
    # zeroes that re-contamination at no-canal basins.  Also re-
    # collapses NonIrr_SW at non-preserved no-canal basins (the
    # post-1985 routing leaves nonirr_sw = nonirr_sw_old × uf —
    # tiny but nonzero at urban cells).
    if canal_dens is not None and basin_names is not None:
        canal_series = pd.Series(canal_dens)
        basin_canal_coverage = canal_series.groupby(basin_names).transform(
            lambda x: (x > 0).mean(),
        )
        if cw_streamflow_raw is not None:
            cw_series = pd.Series(np.nan_to_num(cw_streamflow_raw, nan=0.0))
            basin_cw_mean = cw_series.groupby(basin_names).transform('mean')
            has_cw_capacity = basin_cw_mean.values > 0
        else:
            has_cw_capacity = np.ones(len(basin_canal_coverage), dtype=bool)
        is_co_direct = np.isin(basin_names, list(CO_RIVER_DIRECT_BASINS))
        no_canal_basin = (
            ((basin_canal_coverage.values < CANAL_COVERAGE_GATE_THRESHOLD)
             | ~has_cw_capacity)
            & ~is_co_direct
        )
        preserve_nonirr_sw = np.isin(
            basin_names, list(NONIRR_SW_PRESERVE_BASINS),
        )
        if no_canal_basin.any():
            irr_gw = np.where(
                no_canal_basin, irr_gw + irr_sw, irr_gw,
            )
            irr_sw = np.where(no_canal_basin, 0.0, irr_sw)
            collapse_ni_sw = no_canal_basin & ~preserve_nonirr_sw
            if collapse_ni_sw.any():
                nonirr_gw = np.where(
                    collapse_ni_sw, nonirr_gw + nonirr_sw, nonirr_gw,
                )
                nonirr_sw = np.where(collapse_ni_sw, 0.0, nonirr_sw)
            irr = irr_gw + irr_sw
            nonirr = nonirr_gw + nonirr_sw

    # Re-apply per-basin GW-share cap on the FINAL per-pixel result
    # at CO-river-direct basins.  The cap was applied earlier on the
    # density-ratio shares BEFORE the post-1985 excess routing — that
    # routing pushes NonIrr_SW back to Irr_SW (or, post-2010 at non-
    # canal-heavy basins, disappears excess into desert via cf
    # weighting), and the net effect at low-cf CO-direct basins
    # (Meadview / Detrital Valley) is to inflate the GW share back to
    # ~80-95 % at 2020.  Final cap re-enforces the per-basin floor:
    # Parker 0.15 GW / 0.85 SW, Yuma 0.10 / 0.90, Lake Mohave 0.10 /
    # 0.90, Lake Havasu / Detrital / Meadview 0.40 / 0.60.
    if basin_names is not None:
        co_gw_cap_final = _co_direct_gw_cap_array(basin_names)
        # Combine with non-CO-direct basin caps (Lower Gila / Ranegras Plain).
        co_gw_cap_final = np.minimum(
            co_gw_cap_final, _basin_gw_cap_array(basin_names),
        )
        is_capped = np.isfinite(co_gw_cap_final)
        if is_capped.any():
            # Replace the np.inf sentinel (non-CO-direct pixels) with a
            # benign finite cap (1.0 = no constraint).  Without this,
            # `irr * np.inf` produces inf/NaN at non-CO-direct pixels and
            # `cap_irr & (gw_share > inf)` is always False, but numpy
            # still evaluates the multiplication and emits a
            # RuntimeWarning when irr == 0 (0 * inf = NaN).  The masked
            # np.where below selects the original irr_gw at those pixels
            # regardless, so the safe-cap substitution is purely cosmetic.
            safe_cap = np.where(is_capped, co_gw_cap_final, 1.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                irr_gw_share_final = np.where(
                    irr > 0, irr_gw / irr, 0.0,
                )
                nonirr_gw_share_final = np.where(
                    nonirr > 0, nonirr_gw / nonirr, 0.0,
                )
            cap_irr = is_capped & (irr_gw_share_final > safe_cap)
            if cap_irr.any():
                new_irr_gw = np.where(
                    cap_irr, irr * safe_cap, irr_gw,
                )
                irr_sw = np.where(cap_irr, irr - new_irr_gw, irr_sw)
                irr_gw = new_irr_gw
            cap_ni = is_capped & (nonirr_gw_share_final > safe_cap)
            if cap_ni.any():
                new_ni_gw = np.where(
                    cap_ni, nonirr * safe_cap, nonirr_gw,
                )
                nonirr_sw = np.where(
                    cap_ni, nonirr - new_ni_gw, nonirr_sw,
                )
                nonirr_gw = new_ni_gw

        # Re-apply per-basin GW FLOOR on the FINAL per-pixel result
        # (mirror of the cap re-application above).  The floor was set
        # by the density-ratio step but the post-1985 NonIrr_SW excess
        # routing can dilute it (e.g. NonIrr at Phoenix gets routed
        # mostly back to Irr_SW via the density-ratio share, dropping
        # NonIrr_GW share below the floor).  Re-applying here ensures
        # the published per-basin shares match the verified ADWR
        # anchors at floored basins (currently Phoenix AMA at 0.30).
        gw_floor_final = _basin_gw_floor_array(basin_names, year=year)
        is_floored = gw_floor_final > 0
        if is_floored.any():
            with np.errstate(invalid='ignore', divide='ignore'):
                irr_gw_share_final = np.where(
                    irr > 0, irr_gw / irr, 0.0,
                )
                nonirr_gw_share_final = np.where(
                    nonirr > 0, nonirr_gw / nonirr, 0.0,
                )
            floor_irr = is_floored & (irr_gw_share_final < gw_floor_final)
            if floor_irr.any():
                new_irr_gw = np.where(
                    floor_irr, irr * gw_floor_final, irr_gw,
                )
                irr_sw = np.where(floor_irr, irr - new_irr_gw, irr_sw)
                irr_gw = new_irr_gw
            floor_ni = is_floored & (nonirr_gw_share_final < gw_floor_final)
            if floor_ni.any():
                new_ni_gw = np.where(
                    floor_ni, nonirr * gw_floor_final, nonirr_gw,
                )
                nonirr_sw = np.where(
                    floor_ni, nonirr - new_ni_gw, nonirr_sw,
                )
                nonirr_gw = new_ni_gw

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
    per-pixel σ_total from the 6-component UQ framework, in mm),
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
