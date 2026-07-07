# GEE Asset Export Pipeline: Harmonized Climate, ET, and LULC Data (1896–2099)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](https://earthengine.google.com/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19057935.svg)](https://doi.org/10.5281/zenodo.19057935)

Pre-exports computationally expensive [Google Earth Engine](https://earthengine.google.com/) (GEE; [Gorelick et al., 2017](https://doi.org/10.1016/j.rse.2017.06.031), [Roy et al., 2025](https://doi.org/10.5281/zenodo.17641528)) collections as asset ImageCollections under `projects/azhydro/assets/`. These assets are consumed by `dataops.py` at tile-download time via simple `ee.ImageCollection(...)` loads and `filterDate()` calls, eliminating repeated on-the-fly computation.

## Prerequisites

```bash
conda activate azhydro
```

The scripts use the GEE high-volume endpoint (`https://earthengine-highvolume.googleapis.com`) and require the `azhydro` GEE project.

## Asset Summary

| Asset Collection | Images | Years | Band | Scale (m) | Source Script |
|---|---|---|---|---|---|
| `gridmet_hargreaves_eto_ratio` | 12 | — (monthly climatology) | `ratio` | 4638.3 | `export_gridmet_hargreaves_ratio.py` |
| `openet_reitz_et_ratio` | 12 | — (monthly climatology) | `ratio` | 800 | `export_openet_reitz_ratio.py` |
| `monthly_etof` | 12 | — (monthly climatology) | `etof` | 4638.3 | `export_monthly_etof.py` |
| `prism_hargreaves_eto` | 996 | 1896–1978 | `eto` | 4638.3 | `export_prism_hargreaves_eto.py` |
| `usgs_adjusted_et` | 1,473 | 1896–2018 | `actual_et` | 800 | `export_usgs_adjusted_et.py` |
| `maca_monthly_eto_v2` | 888 | 2026–2099 | `eto` | 4638.3 | `export_maca_monthly_eto.py` |
| `maca_monthly_et_v2` | 888 | 2026–2099 | `actual_et` | 4638.3 | `export_maca_monthly_et.py` |
| `lulc_projection_ensemble` | 74 | 2026–2099 | `landcover` | 250 | `export_lulc_ensemble.py` |
| `monthly_peff_v2` | 2,448 | 1896–2099 | `peff` | 4638.3 | `export_monthly_peff.py` |
| `maca_gcm_annual_eto` | 370 | 2026–2099 | `eto` | 4638.3 | `export_maca_gcm_annual_eto.py` |
| `maca_gcm_annual_et` | 370 | 2026–2099 | `actual_et` | 4638.3 | `export_maca_gcm_annual_et.py` |
| `maca_gcm_annual_peff` | 370 | 2026–2099 | `peff` | 4638.3 | `export_maca_gcm_annual_peff.py` |

## Dependency Graph

Exports must run in dependency order. Use `run_all_exports.py` to orchestrate automatically, or run individual scripts manually following the levels below.

```
Level 1 (no dependencies — can run in parallel):
  ├── export_gridmet_hargreaves_ratio.py
  ├── export_openet_reitz_ratio.py
  ├── export_monthly_etof.py
  └── export_lulc_ensemble.py

Level 2 (depends on Level 1):
  ├── export_prism_hargreaves_eto.py    ← needs gridmet_hargreaves_eto_ratio
  ├── export_usgs_adjusted_et.py        ← needs openet_reitz_et_ratio
  └── export_maca_monthly_eto.py        ← no custom dep (uses OpenET gridMET ratios)

Level 3 (depends on Level 1 + Level 2):
  ├── export_maca_monthly_et.py         ← needs monthly_etof + maca_monthly_eto_v2
  └── export_monthly_peff.py            ← needs prism_hargreaves_eto + maca_monthly_eto_v2

Level 4 — per-GCM uncertainty (uses OpenET gridMET ratios + etof):
  ├── export_maca_gcm_annual_eto.py     ← 5 representative GCMs (370 images)
  ├── export_maca_gcm_annual_et.py      ← computes monthly ETo×EToF internally
  └── export_maca_gcm_annual_peff.py    ← computes monthly USDA SCS Peff internally
```

## Data Harmonization

The export pipeline harmonizes multiple heterogeneous data sources into temporally and methodologically consistent grids spanning 1896–2099. The core challenge is that no single dataset covers this entire period, so overlapping-period bias correction ratios are used to stitch together three distinct eras for both ET and ETo.

### ET Harmonization (3 eras)

```
         Reitz ET               OpenET                MACA ET
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   1896 – 1999   │   │   2000 – 2025   │   │   2026 – 2099   │
    │   (mm/day → mm) │   │ (et_ensemble_mad│   │ (EToF × ETo)    │
    │                 │   │   mm/month)     │   │                 │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             │  OpenET/Reitz ratio │  Monthly EToF       │
             │  (2000-2018 overlap)│  (2000-2025 overlap)│
             │         ┌───────────┘                     │
             ▼         ▼                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Harmonized monthly ET (band: actual_et)        │
    │                         1896 – 2099                         │
    └─────────────────────────────────────────────────────────────┘
```

- **1896–1999 (Reitz → OpenET scale):** USGS Reitz Ensemble ET ([Reitz et al., 2023](https://doi.org/10.1029/2022WR034012)) is available as mm/day. It is converted to mm/month, then multiplied by a monthly OpenET/Reitz ratio derived from the 2000–2018 overlap period (228 paired monthly images). The ratio for each calendar month is averaged across all overlapping years to produce 12 climatological ratio grids, ensuring the Reitz ET is scaled to match the magnitude and spatial patterns of OpenET.

- **2000–2025 (OpenET, native):** OpenET Ensemble v2.0/v2.1 ([Melton et al., 2022](https://doi.org/10.1111/1752-1688.12956); [Volk et al., 2024](https://doi.org/10.1038/s44221-023-00181-7)) (`et_ensemble_mad`) is used directly. Monthly ET is computed by summing all OpenET images within each calendar month.

- **2026–2099 (MACA → OpenET scale):** No observational ET exists for the future. Instead, monthly EToF (crop coefficient = ET/ETo) grids are derived from the 2000–2025 OpenET/OpenET gridMET overlap. Future monthly ET is then: $\text{ET}_{\text{future}} = \text{EToF} \times \text{MACA ETo}$, where MACA ETo is from [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312). This preserves the observed spatial crop-water-demand patterns while allowing the climate signal (ETo) to evolve under future scenarios.

### ETo Harmonization (3 eras)

```
      PRISM Hargreaves ETo    OpenET gridMET ETo        MACA ETo
    ┌─────────────────────┐   ┌───────────────┐   ┌─────────────────┐
    │    1896 – 1978      │   │  1979 – 2025  │   │   2026 – 2099   │
    │  (monthly tmax/tmin │   │  (native      │   │ (per-model/scen │
    │   → Hargreaves PET) │   │   monthly)    │   │  → RefET → ens) │
    └──────────┬──────────┘   └───────┬───────┘   └────────┬────────┘
               │                      │                    │
               │ Hargreaves/OpenET    │   OpenET gridMET   │
               │ gridMET ratio        │   bias correction  │
               │ overlap)             │   ratios           │
               ▼                      ▼                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Harmonized monthly ETo (band: eto)             │
    │                         1896 – 2099                         │
    └─────────────────────────────────────────────────────────────┘
```

- **1896–1978 (PRISM Hargreaves → OpenET gridMET scale):** Daily climate data is unavailable before 1979, so monthly ETo is estimated from [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) monthly tmax/tmin via the Hargreaves method (`openet.refetgee.Daily` with mid-month DOY). This systematically differs from Penman-Monteith-based [OpenET gridMET (Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026)](https://doi.org/10.5281/zenodo.18673484) ETo, so a per-pixel, per-month correction ratio is derived from the 1979–2025 overlap (564 paired monthly images) where both Hargreaves and OpenET gridMET are available. The corrected ETo is: $\text{ETo}_{\text{corrected}} = \text{Hargreaves PET} \times \overline{R}$, where $\overline{R} = \text{mean}\left(\frac{\text{OpenET gridMET ETo}}{\text{Hargreaves PET}}\right)$ over 1979–2025.

- **1979–2025 (OpenET gridMET, native):** [OpenET gridMET (Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026)](https://doi.org/10.5281/zenodo.18673484) monthly ETo (`projects/openet/assets/reference_et/conus/gridmet/monthly/v1`) is used directly. This is the reference standard to which all other eras are harmonized.

- **2026–2099 (MACA → OpenET gridMET scale):** To preserve the nonlinear response of the Penman-Monteith equation, ETo is computed for every individual [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) image (all 20 GCMs × 2 RCPs) using a flat pipeline. A single `map()` applies `openet.refetgee.Daily.maca()` to every MACA image in the year, then for each month the sum is divided by the number of members (40) to obtain the ensemble-mean monthly ETo. OpenET gridMET bias-correction ratios ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026](https://doi.org/10.5281/zenodo.18673484); `projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/`) are then applied: $\text{ETo}_{\text{corrected}} = \frac{1}{N}\sum_{i=1}^{N}\text{ETo}_i \times \text{OpenET gridMET ratio}$ to reduce high ETo bias, particularly over irrigated areas. The flat-pipeline approach (single map + reduce instead of 40 separate computation pipelines) keeps the GEE computation graph small enough to avoid out-of-memory errors.

### LULC Harmonization (2 eras)

```
     USGS Historical LULC        IrrMapper / NLCD      USGS Projected LULC
    ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
    │   1938 – 1984    │   │    1985 – 2025      │   │   2026 – 2099    │
    │  (Historical     │   │ (observational       │   │ (mode of B1, B2, │
    │   scenario)      │   │  satellite-based)    │   │  A1B, A2)        │
    └──────────────────┘   └─────────────────────┘   └──────────────────┘
```

- **1938–1984:** USGS Historical LULC ([Sohl et al., 2016](https://doi.org/10.1080/1747423X.2016.1147619); `projects/nwi-usgs/assets/USGS-LULC-CONUS`, scenario = "Historical") is used directly. Years before 1938 use the 1938 classification.
- **1985–2025:** IrrMapper for irrigation classification and [NLCD (USGS, 2024](https://doi.org/10.5066/P94UXNTS); [Fleckenstein et al., 2026)](https://doi.org/10.1016/j.rse.2026.115347) for urban/developed areas provide observational LULC.
- **2026–2099:** Four USGS LULC projection scenarios ([Sohl et al., 2014](https://doi.org/10.1890/13-1245.1); B1, B2, A1B, A2) are combined via pixel-wise mode to create a single consensus ensemble. Exported as integer (categorical) data.

For each year, two derived bands — `annual_crop_fraction` (LULC class 1 / ag) and `annual_urban_fraction` (LULC class 2 / urban) — are computed at the native LULC resolution and aggregated to the 2 km predictor grid via a count reducer. These give physical per-pixel area fractions (0–1), suitable for mass-conserving volume partitioning downstream.

**Basin-scale delta bias-correction (downstream):** Because NLCD (30 m) and USGS (250 m) produce systematically different basin-level urban/ag fractions after 2 km aggregation, the azhydro pipeline applies a multiplicative basin-scale delta correction to the four LULC-derived columns (`URBAN`, `AGRI`, `annual_crop_fraction`, `annual_urban_fraction`) for off-NLCD years (≤1984 and ≥2026). This is the LULC analog of the climate bias correction described above: use the non-NLCD source (USGS Historical or FORE-SCE) to provide *basin-scale relative change*, anchored to NLCD's *pixel-level spatial pattern* at the 1985/2025 training-period boundaries. See `azhydro/README.md` for details.

### Join Strategy

All ratio computations use `ee.Join.inner()` with paired filters on **both** `month` **and** `year` properties. This prevents many-to-many cross-joins that would occur if matching on `month` alone (e.g., January 2005 OpenET gridMET would incorrectly pair with January 2010 PRISM). The final climatological ratios are then produced by averaging across years within each calendar month.

### Scale Consistency

Each asset is exported at its native source resolution to avoid resampling artifacts during the ratio computation:

| Source | Native Scale |
|---|---|
| [PRISM](https://doi.org/10.1002/joc.1688) / [OpenET gridMET](https://doi.org/10.1002/joc.3413) ([Volk et al., 2026](https://doi.org/10.5281/zenodo.18673484)) / [MACA](https://doi.org/10.1002/joc.2312) | 4,638.3 m (~4 km) |
| [Reitz Ensemble ET](https://doi.org/10.5066/P9EZ3VAS) | 800 m |
| [OpenET Ensemble](https://doi.org/10.1111/1752-1688.12956) | 30 m |
| [USGS LULC](https://doi.org/10.1080/1747423X.2016.1147619) | 250 m |

GEE handles on-the-fly reprojection when these assets are combined at tile-download time in `dataops.py`.

### ScienceBase ingestion

Two external datasets — [Reitz Ensemble ET](https://doi.org/10.5066/P9EZ3VAS) and [USGS LULC](https://doi.org/10.5066/F7KK99RR) (historical + [projections](https://doi.org/10.5066/P95AK9HP)) — are distributed as GeoTIFFs on USGS ScienceBase. These were bulk-uploaded to GEE as ImageCollections (`projects/nwi-usgs/assets/USGS-Reitz-Ensemble-ET` and `projects/nwi-usgs/assets/USGS-LULC-CONUS`) using [geeup (Roy, 2025)](https://doi.org/10.5281/zenodo.18073520) prior to running the export scripts in this project.

## Detailed Export Logic

### 1. OpenET gridMET/Hargreaves ETo Ratio (`gridmet_hargreaves_eto_ratio`)

**Purpose:** Bias-correct [PRISM (Daly et al., 2008)](https://doi.org/10.1002/joc.1688) Hargreaves ETo (1896–1978) to be consistent with [OpenET gridMET (Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026)](https://doi.org/10.5281/zenodo.18673484) ETo (1979–2025).

**Method:**
1. Load OpenET gridMET monthly ETo for 1979–2025 (`projects/openet/assets/reference_et/conus/gridmet/monthly/v1`).
2. Load PRISM monthly data for 1979–2025 (`OREGONSTATE/PRISM/ANm`) and compute Hargreaves PET using `openet.refetgee.Daily` with tmax/tmin and day-of-year at mid-month.
3. Inner join on both `month` AND `year` to pair same-timestep images.
4. Compute per-pixel ratio: $\text{ratio} = \frac{\text{OpenET gridMET ETo}}{\text{Hargreaves ETo}}$.
5. Average ratios across all years for each month → 12 monthly ratio grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `ratio`.

> **Note:** A separate ETo ratio (`projects/azhydro/assets/prism_gridmet_ratio/monthly`) is also available across CONUS. That ratio is computed from **monthly** PRISM and OpenET gridMET ETo over **1981–2021**, whereas the `gridmet_hargreaves_eto_ratio` used in this project is calculated from **monthly** PRISM Hargreaves and OpenET gridMET ETo over **1979–2025**.

---

### 2. OpenET/Reitz ET Ratio (`openet_reitz_et_ratio`)

**Purpose:** Bias-correct USGS [Reitz Ensemble ET (Reitz et al., 2023)](https://doi.org/10.5066/P9EZ3VAS) (1896–1999) to be consistent with [OpenET (Melton et al., 2022)](https://doi.org/10.1111/1752-1688.12956) (2000–2025).

**Method:**
1. Build monthly OpenET ET for 2000–2018 by summing daily OpenET v2.0 images per month (band `et_ensemble_mad`).
2. Load USGS Reitz Ensemble ET (`projects/nwi-usgs/assets/USGS-Reitz-Ensemble-ET`) for 2000–2018. Convert from mm/day to mm/month by multiplying by the number of days in each month.
3. Inner join on both `month` AND `year`.
4. Compute per-pixel ratio: $\text{ratio} = \frac{\text{OpenET ET}}{\text{Reitz ET}}$.
5. Average ratios across all years for each month → 12 monthly ratio grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `ratio`.

---

### 3. Monthly EToF (`monthly_etof`)

**Purpose:** Derive crop coefficient (EToF = ET/ETo) grids for applying to future [MACA (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) ETo to estimate future ET.

**Method:**
1. Build monthly [OpenET](https://doi.org/10.1111/1752-1688.12956) ET for 2000–2025 (same as above).
2. Load [OpenET gridMET (Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026)](https://doi.org/10.5281/zenodo.18673484) monthly ETo for 2000–2025.
3. Inner join on both `month` AND `year`.
4. Compute per-pixel EToF: $\text{EToF} = \frac{\text{OpenET ET}}{\text{OpenET gridMET ETo}}$.
5. Average across all years for each month → 12 monthly EToF grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `etof`.

---

### 4. PRISM Hargreaves ETo (`prism_hargreaves_eto`)

**Dependency:** `gridmet_hargreaves_eto_ratio`

**Purpose:** Monthly ETo for the historical period 1896–1978, bias-corrected to match OpenET gridMET.

**Method:** For each year in 1896–1978:
1. Load PRISM monthly data and compute Hargreaves PET (same Hargreaves method as the ratio script).
2. Inner join with the 12 pre-exported ratio grids on `month`.
3. Multiply: $\text{ETo} = \text{Hargreaves PET} \times \text{ratio}$.
4. Export each of the 12 monthly images.

**Output:** 996 images (`{year}_{month:02d}`), each with band `eto` in mm/month.

---

### 5. USGS Adjusted ET (`usgs_adjusted_et`)

**Dependency:** `openet_reitz_et_ratio`

**Purpose:** Monthly bias-corrected ET spanning the full Reitz record (1896–2018). Only 1896–1999 is consumed at tile-download time — OpenET takes over from 2000 — but the 2000–2018 overlap years are exported as well so the ratio correction can be validated directly against OpenET.

**Method:** For each year in 1896–2018:
1. Load Reitz ET for that year, convert mm/day → mm/month.
2. Inner join with the 12 pre-exported OpenET/Reitz ratio grids on `month`.
3. Multiply: $\text{ET}_{\text{adj}} = \text{Reitz ET (mm/month)} \times \text{ratio}$ → band `actual_et`.
4. Export each of the 12 monthly images.

**Output:** 1,473 images (`{year}_{month:02d}`), each with band `actual_et` in mm/month.

---

### 6. MACA Monthly ETo (`maca_monthly_eto_v2`)

**Purpose:** Monthly ETo for future scenarios 2026–2099, bias-corrected with OpenET gridMET ratios ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413); [Volk et al., 2026](https://doi.org/10.5281/zenodo.18673484)). Computed per-image via a flat pipeline to preserve nonlinear ETo response while keeping the computation graph small.

**Method:** For each year in 2026–2099:
1. Load ALL [MACA v2 (Abatzoglou & Brown, 2012)](https://doi.org/10.1002/joc.2312) daily images for the year (20 models × 2 scenarios × 365 days ≈ 14,600 images).
2. Map `openet.refetgee.Daily.maca()` over the entire collection to compute daily ETo for each image (using [NASADEM](https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001) elevation and pixel latitude).
3. For each month, sum all daily ETo and divide by `N_MEMBERS` (40) to get the ensemble-mean monthly ETo.
4. Apply OpenET gridMET bias-correction ratios (`projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/{MonthName}`), joined on `month`.
5. Export each month.

The flat pipeline (one `.map()` + `.sum().divide(40)` per month) keeps the GEE computation graph as a simple map-reduce pattern, avoiding the out-of-memory errors that result from building 40 separate computation pipelines.

This approach still preserves per-image nonlinearity: ETo is computed from each individual MACA image's climate variables before any averaging occurs.

**Output:** 888 images (`{year}_{month:02d}`), each with band `eto` in mm/month.

---

### 7. MACA Monthly ET (`maca_monthly_et_v2`)

**Dependencies:** `monthly_etof` + `maca_monthly_eto_v2`

**Purpose:** Monthly ET for 2026–2099.

**Method:** For each year in 2026–2099:
1. Load pre-exported MACA monthly ETo for that year.
2. Load pre-exported monthly EToF (12 images).
3. Inner join on `month`.
4. Multiply: $\text{ET}_{\text{adj}} = \text{MACA ETo} \times \text{EToF}$ → band `actual_et`.
5. Export each month.

**Output:** 888 images (`{year}_{month:02d}`), each with band `actual_et` in mm/month.

---

### 8. LULC Projection Ensemble (`lulc_projection_ensemble`)

**Purpose:** Annual land use/land cover classification for 2026–2099.

**Method:** For each year:
1. Load USGS LULC projections ([Sohl et al., 2014](https://doi.org/10.1890/13-1245.1); `projects/nwi-usgs/assets/USGS-LULC-CONUS`) for all four scenarios: B1, B2, A1B, A2.
2. Compute pixel-wise mode across the 4 scenarios.
3. Export as integer (categorical data).

**Output:** 74 images (`{year}`), each with band `landcover` (integer class values).

---

### 9. Monthly USDA SCS Effective Precipitation (`monthly_peff_v2`)

**Dependencies:** `prism_hargreaves_eto` (for 1896–1978) + `maca_monthly_eto_v2` (for 2026–2099)

**Purpose:** Monthly effective precipitation for 1896–2099 using the [USDA SCS (1993)](https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf) method ([Muratoglu et al., 2023](https://doi.org/10.1016/j.watres.2023.120011); [Majumdar et al., 2026](https://doi.org/10.5281/zenodo.18706481)).

**Parameters:** `mad_factor = 1`, `rz_depth_m = 2 m` for all months (consistent with UCRB comparisons).

**Method:** For each year in 1896–2099:
1. **Get monthly ETo** from the appropriate source:
   - 1896–1978: Pre-exported `prism_hargreaves_eto` asset
   - 1979–2025: OpenET gridMET native monthly ETo
   - 2026–2099: Pre-exported `maca_monthly_eto_v2` asset
2. **Get monthly precipitation:**
   - 1896–2025: [PRISM](https://doi.org/10.1002/joc.1688) (`OREGONSTATE/PRISM/ANm`, band `ppt`)
   - 2026–2099: Flat [MACA](https://doi.org/10.1002/joc.2312) pipeline — sum all daily precip images per month, divide by `N_MEMBERS` (40) for ensemble mean
3. **Inner join** ETo and precipitation on `month`.
4. **Compute USDA SCS effective precipitation** (equations 2-84 and 2-85):
   - Convert ETo and precip from mm to inches.
   - Compute soil storage factor: $d = \text{MAD} \times \text{AWC} \times \text{RZ}_{\text{inches}}$
   - $sf = 0.531747 + 0.295164 \cdot d - 0.057697 \cdot d^2 + 0.003804 \cdot d^3$
   - $ep = sf \times (0.70917 \cdot P^{0.82416} - 0.11556) \times 10^{0.02426 \cdot \text{ETo}}$
   - Clamp: $ep \leq P$, $ep \leq \text{ETo}$, $ep \geq 0$
   - Convert back to mm.
5. **Export** each of the 12 monthly images.

**Soil data:** AWC from [SSURGO](https://websoilsurvey.nrcs.usda.gov/) (`projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite`) (inches).

**Output:** 2,448 images (`{year}_{month:02d}`), each with band `peff` in mm/month.

---

### 10–12. Per-GCM Uncertainty Assets (Level 4)

**Purpose:** Capture climate-model uncertainty (σ_MACA) by exporting per-GCM versions of the three MACA-driven variables (ETo, ET, Peff). Downstream, each GCM's complete chain is run through the pipeline, and the inter-GCM spread quantifies the uncertainty attributable to GCM selection.

**Representative GCMs:** Five GCMs spanning the Southwest US temperature × precipitation space (cf. Rupp et al., 2013):

| GCM | Climate Corner |
|---|---|
| CCSM4 | Central / median |
| CNRM-CM5 | Cool-wet |
| HadGEM2-ES365 | Hot-dry |
| MIROC-ESM-CHEM | Hot-wet |
| inmcm4 | Cool-dry (lowest climate sensitivity in CMIP5) |

For each GCM, both RCP scenarios (rcp45, rcp85) are averaged — yielding a 2-member per-GCM mean — using the same flat-pipeline approach as the 40-member ensemble version.

Since the downstream ML model operates on **annual** predictor variables, all per-GCM exports are annual (monthly computation happens server-side in GEE and is summed to annual before export). This reduces total assets from 13,320 (monthly) to 1,110 (annual).

#### 10. Per-GCM Annual ETo (`maca_gcm_annual_eto`)

Same flat pipeline as §6 (ensemble ETo), but filtered to one model (2 scenarios). Monthly ETo is `daily_eto.sum().divide(2)`, OpenET gridMET bias-corrected, then summed to annual. Used at tile-download time for the Peff clamp safeguard in `dataops.py`.

**Output:** 370 images (`{model}_{year}`), band `eto` (mm/year), properties `model` + `year`.

#### 11. Per-GCM Annual ET (`maca_gcm_annual_et`)

Computes bias-corrected monthly ETo internally (same as §10), multiplies by monthly EToF, sums to annual ET. No dependency on the per-GCM ETo asset — monthly ETo is recomputed internally for the EToF multiplication.

**Output:** 370 images (`{model}_{year}`), band `actual_et` (mm/year), properties `model` + `year`.

#### 12. Per-GCM Annual Peff (`maca_gcm_annual_peff`)

Computes bias-corrected monthly ETo + per-GCM precip internally, applies the nonlinear USDA SCS formula per month (same as §9), sums to annual. Only 2026–2099 because historical Peff uses PRISM/OpenET gridMET observations equally for all GCMs.

**Output:** 370 images (`{model}_{year}`), band `peff` (mm/year), properties `model` + `year`.

## Assets Used by the `azhydro` Pipeline

Not all exported assets are consumed at tile-download time. The intermediate ratio/climatology assets (Levels 1–2) exist only to produce the final harmonized collections. The table below lists the assets actually loaded by `dataops.py`:

| Asset Collection | Variable | Years | Used For |
|---|---|---|---|
| `prism_hargreaves_eto` | ETo | 1896–1978 | Historical reference ET |
| `usgs_adjusted_et` | ET | 1896–1999 | Historical actual ET |
| `maca_monthly_eto_v2` | ETo | 2026–2099 | Future reference ET (ensemble) |
| `maca_monthly_et_v2` | ET | 2026–2099 | Future actual ET (climatological EToF) |
| `lulc_projection_ensemble` | LULC | 2026–2099 | Future land-use classification |
| `monthly_peff_v2` | Peff | 1896–2099 | Effective precipitation (all eras) |
| `maca_gcm_annual_et` | ET | 2026–2099 | Per-GCM ET for σ_MACA uncertainty |
| `maca_gcm_annual_eto` | ETo | 2026–2099 | Per-GCM ETo for σ_MACA uncertainty |
| `maca_gcm_annual_peff` | Peff | 2026–2099 | Per-GCM Peff for σ_MACA uncertainty |

Assets **not** listed (e.g., `gridmet_hargreaves_eto_ratio`, `openet_reitz_et_ratio`, `monthly_etof`) are intermediate products consumed only by downstream export scripts.

## Usage

### Run everything in order

```bash
python run_all_exports.py
```

### Run a specific dependency level

```bash
python run_all_exports.py --level 1          # Level 1 only
python run_all_exports.py --level 2          # Level 2 only
python run_all_exports.py --no-wait          # Submit all and exit immediately
```

### Run individual scripts

```bash
# Ratio/climatology exports (Level 1)
python export_gridmet_hargreaves_ratio.py
python export_openet_reitz_ratio.py
python export_monthly_etof.py
python export_lulc_ensemble.py

# Year-range exports (Levels 2-3) with custom ranges
python export_prism_hargreaves_eto.py --start-year 1896 --end-year 1950
python export_usgs_adjusted_et.py --start-year 1950 --end-year 1999
python export_maca_monthly_eto.py --start-year 2026 --end-year 2050
python export_maca_monthly_et.py --start-year 2026 --end-year 2050
python export_monthly_peff.py --start-year 1896 --end-year 1978

# All scripts support --no-wait to submit tasks without blocking
python export_maca_monthly_eto.py --no-wait

# Per-GCM uncertainty exports (Level 4) — all 5 GCMs or a single GCM
python export_maca_gcm_annual_eto.py                          # all 5 GCMs
python export_maca_gcm_annual_eto.py --gcm HadGEM2-ES365      # single GCM
python export_maca_gcm_annual_et.py --gcm CCSM4
python export_maca_gcm_annual_peff.py --start-year 2050 --end-year 2099
```

### Visualize monthly ratio grids

```bash
python plot_monthly_ratios.py [--output-dir DIR]
```

Produces 12 PNG figures (default output: `Data/Outputs/GEE_Ratios/`), four per ratio:

| File | Description |
|---|---|
| `{ratio}.png` | 4x3 spatial maps (Jan-Dec) with basin overlays |
| `{ratio}_basin_avg.png` | 4x3 choropleth maps of basin-averaged ratios |
| `{ratio}_monthly_mean_ts.png` | AZ-wide monthly mean time series |
| `{ratio}_monthly_mean_ts_ama_ina.png` | AMA/INA-only area-weighted monthly mean time series |

Where `{ratio}` is one of: `gridmet_hargreaves_eto_ratio`, `monthly_etof`, `openet_reitz_et_ratio`.

All figures use a discrete diverging blue-white-red colormap centered on 1.0. Downloaded GeoTIFFs are cached in `{output-dir}/.cache/` so subsequent runs skip the GEE download.

### Resumability

All scripts check for existing assets before exporting. If a run is interrupted, simply re-run the same script — it will skip already-exported images and only submit remaining ones.

### Queue Throttling

GEE enforces a 3,000-task queue limit. `export_image()` automatically calls `_wait_for_queue_capacity()` before each submission. If there are already 2,900+ pending/running tasks, it polls every 60 seconds until the queue has room. This means large batch exports (e.g., 2,448 peff images) can be launched without manually splitting runs.

## Visualization (`plot_monthly_ratios.py`)

| Function | Purpose |
|---|---|
| `_download_ee_image(...)` | Download a single-band GEE image as a numpy array; caches GeoTIFFs locally |
| `_plot_ratio(...)` | 4x3 pixel-level spatial maps (Jan-Dec) with GW basin boundary overlays |
| `_plot_basin_avg(...)` | 4x3 choropleth maps of basin-averaged (zonal mean) ratios per month |
| `_plot_monthly_mean_ts(...)` | AZ-wide monthly mean time series (Jan-Dec) for all three ratios |
| `_plot_monthly_mean_ts_ama_ina(...)` | Area-weighted monthly mean time series for AMA/INA basins only |
| `_make_ratio_norm(...)` | Discrete `BoundaryNorm` centered on 1 for the diverging colormap |

## Shared Utilities (`config.py`)

| Function | Purpose |
|---|---|
| `init_ee()` | Initialize EE with high-volume endpoint |
| `get_az_geometry()` | Arizona state boundary from TIGER/Census |
| `create_ic_asset(asset_id)` | Create ImageCollection asset if missing |
| `list_existing_assets(collection_id)` | Batch-list all assets in a collection (paginated) |
| `asset_exists(asset_id)` | Check if a single asset exists |
| `export_image(...)` | Export clipped image to asset (supports `as_int` for categorical data). Auto-throttles near 3k queue limit |
| `wait_for_tasks(tasks)` | Poll until all tasks complete, report failures |
| `get_export_parser(description)` | Argparse with `--start-year`, `--end-year`, `--no-wait` |
| `calc_prism_monthly_eto(img)` | Hargreaves PET from PRISM tmax/tmin |
| `build_openet_monthly_et_ic()` | Monthly OpenET ET 2000–2025 |
| `build_daily_maca_ensemble(year)` | Daily MACA ensemble for one year (legacy, used by `dataops.py`) |
| `build_daily_maca_single(year, model, scenario)` | Daily MACA data for one GCM/scenario |
| `MACA_REPRESENTATIVE_GCMS` | 5 GCMs spanning hot-dry/hot-wet/cool-dry/cool-wet/median for uncertainty quantification |

## How `dataops.py` Consumes These Assets

After export, `dataops.py` loads the assets once before the tile-download loop:

```python
_ASSET_PREFIX = 'projects/azhydro/assets'
prism_hargreaves_eto_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/prism_hargreaves_eto')
usgs_adjusted_et_ic     = ee.ImageCollection(f'{_ASSET_PREFIX}/usgs_adjusted_et')
maca_monthly_eto_ic     = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_eto_v2')
maca_monthly_et_ic      = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_et_v2')
lulc_projection_ensemble_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/lulc_projection_ensemble')
```

Inside the per-year loop, each is filtered by date:

```python
prism_eto_ic = prism_hargreaves_eto_ic.filterDate(start_year_gee, end_year_gee)
usgs_ensemble_et_ic = usgs_adjusted_et_ic.filterDate(start_year_gee, end_year_gee)
maca_eto_ic = maca_monthly_eto_ic.filterDate(start_year_gee, end_year_gee)
maca_et_ic = maca_monthly_et_ic.filterDate(start_year_gee, end_year_gee)
```

This replaces the previous approach of building these collections from scratch (with expensive joins, Hargreaves calculations, and MACA ensemble averaging) on every tile × year combination.

For future-year (2026–2099) precipitation and temperature, `dataops.py` queries the raw MACA collection directly using the flat-pipeline approach:

```python
maca_ic = ee.ImageCollection('IDAHO_EPSCOR/MACAv2_METDATA').filterDate(start, end)
precip = maca_ic.select('pr').sum().divide(n_members)      # sum/40 for precip
tmmx   = maca_ic.select('tasmax').mean()                    # mean for temp
tmmn   = maca_ic.select('tasmin').mean()
```

Since every day has exactly one image per model/scenario pair, `.mean()` across all ~14,600 images/year gives the grand mean (equivalent to per-model averaging). For additive quantities (precipitation), `.sum().divide(40)` gives the ensemble-mean annual total.

## Output Visualization Pipeline

**Live app:** [AZ-Hydro Explorer](https://azhydro.projects.earthengine.app/view/azhydro-explorer) — published GEE App backed by the assets and the visualizer described below.

The scripts above export **inputs** consumed by the modeling pipeline.  The files below ingest the modeling pipeline's **outputs** (`Data/Outputs/.../Full_Prediction_XGBRF/`) into GEE and serve them through an interactive visualizer.  All assets land under `projects/azhydro/assets/az-wu/`.

The pipeline uses two distinct upload paths:
- **Rasters** → `geeup upload` (handles ImageCollections via `metadata.csv` files).
- **Tabular FeatureCollections (CSVs with `.geo` GeoJSON column)** → `gsutil cp` → `earthengine upload table`.  geeup tabup was tried first but rejected the CSVs in practice ("No tables to upload" both bare and zipped); the EE CLI accepts the `.geo` column natively.  CSVs are staged in `gs://azhydro/` (the same bucket already used by the pipeline for the HUC12 GeoJSON — see [`azhydro/hydrolibs/dataops.py`](../azhydro/hydrolibs/dataops.py)).

### Files

| File | Purpose |
|---|---|
| [`generate_geeup_metadata.py`](generate_geeup_metadata.py) | Walk `gee/Data/`, write per-leaf `metadata.csv` files (`id_no`, `system:time_start`, `year`, `unit`, `category`, `asset_collection`).  Detects unit-parent dirs (`Depth_mm`/`Depth_ft`/`Volume_m3`/`Volume_AF`) and consolidates the four unit conventions into a **single** ImageCollection per category, tagging each image with a `unit` property the visualizer filters on. |
| [`upload_to_gee.sh`](upload_to_gee.sh) | Batch raster upload.  Iterates every `metadata.csv`, runs `geeup upload --workers 10 --resume` on each.  Per-source logs in `gee/upload_logs/`.  Supports `--dry-run` and `--filter <pattern>`. |
| [`pivot_to_csv.py`](pivot_to_csv.py) | Pivot the Well_Package long-format GeoParquet (170,137 wells × 204 years × 15 categories, ~2.1 GB) into 15 per-category CSVs at `gee/Data/Well_Package/csv/Well_Package__<Cat>.csv`.  Each row carries a `.geo` column (GeoJSON Point), `REGISTRY_I`, `WATER_USE`, plus year-stamped property columns (`<Cat>_AF_<year>`, `<Cat>_AF_sigma_<year>`) — ~412 properties / row.  Paths anchored to the script location, so it runs from any cwd.  ~5–10 min per category on 64 GB RAM. |
| [`upload_well_package.sh`](upload_well_package.sh) | Loops the 15 CSVs: `gsutil cp` to `gs://azhydro/Well_Package__<Cat>.csv`, then `earthengine upload table --asset_id=projects/azhydro/assets/az-wu/Well_Package__<Cat>` to ingest each.  Per-stage log: `gee/upload_logs/Well_Package_upload.log`.  `--dry-run` and `--filter <pattern>` flags.  Cleanup hint at the end (`gsutil -m rm`). |
| [`pivot_cap_cumulative.py`](pivot_cap_cumulative.py) | Pivot `Data/Outputs/.../CAP_Scenario/CAP_Scenario_Cumulative.csv` (long format, ~27 k rows) into a wide-format CSV with `.geo` column: 364 features (52 basin polygons × 7 scenarios), `Cum_<year>` columns covering 2026–2099, basin centroid as the geometry. |
| [`upload_cap_cumulative.sh`](upload_cap_cumulative.sh) | Stage the CSV to `gs://azhydro/CAP_Scenario_Cumulative.csv` and ingest as `projects/azhydro/assets/az-wu/CAP_Scenario_Cumulative`.  The visualizer reads this asset on every CAP click to chart all 7 scenarios for the basin. |
| [`cap_service_area_to_csv.py`](cap_service_area_to_csv.py) | Convert the CAP service-area polygon GeoJSON (3 polygons, MARICOPA / PIMA / PINAL) into `gee/Data/CAP/CAP.csv` with a `.geo` GeoJSON-string column (polygon geometry preserved exactly). |
| [`upload_cap_service_area.sh`](upload_cap_service_area.sh) | Stage to `gs://azhydro/CAP.csv` and ingest as `projects/azhydro/assets/az-wu/CAP`.  Asset is consumed by the visualizer as the CAP-eligible county overlay layer. |
| [`azhydro-visualizer.js`](azhydro-visualizer.js) | GEE Apps interactive visualizer.  Year slider 1896–2099, category / unit / band dropdowns, manual color-stretch override (auto-fills with the 2nd / 98th percentile of the current image), click-driven pixel + basin + sub-basin time series with **prediction + 95 % CI envelope** on every chart, plus a **nearest-well** chart pulled from the Well_Package FeatureCollections (capacity-disaggregated per-well values; AF → selected unit converted client-side).  **Side-by-side comparison** via the Compare toggle: `ui.SplitPanel` with linked zoom/pan + draggable wipe divider; each map has its own category dropdown AND its own CAP scenario / window dropdowns so two CAP scenarios can be compared on the wipe.  Vector overlays include AMA / INA / regular basins in distinct colours, sub-basins, ADWR wells, and the **CAP-eligible counties** (Maricopa / Pima / Pinal as dashed outlines).  Special UX for OOD (single band, no unit), **SW Capture Fraction** (dimensionless 0–1, OOD-style palette), and **CAP scenarios** (scenario × window dropdowns + cumulative ΔGW readout + per-basin × 7-scenario time-series chart). |

### Workflow (one-time, after the modeling pipeline finishes)

```bash
# Pre-flight (one-time)
gcloud auth login
gcloud config set project azhydro
earthengine authenticate
pip install earthengine-api geeup

# 1. Rasters: generate metadata.csv files, batch-upload via geeup
python gee/generate_geeup_metadata.py
./gee/upload_to_gee.sh                    # all dirs (resumable)
./gee/upload_to_gee.sh --filter Total_GW  # one category
./gee/upload_to_gee.sh --dry-run          # preview only

# 2. Well_Package: pivot parquet → 15 CSVs, then GCS + earthengine
python gee/pivot_to_csv.py                # writes gee/Data/Well_Package/csv/
./gee/upload_well_package.sh              # gsutil cp + earthengine upload table

# 3. CAP cumulative basin time series (per-basin × 7-scenario chart)
python gee/pivot_cap_cumulative.py
./gee/upload_cap_cumulative.sh

# 4. CAP service-area polygons (county overlay)
python gee/cap_service_area_to_csv.py
./gee/upload_cap_service_area.sh

# 5. Open the visualizer
#    Live app:  https://azhydro.projects.earthengine.app/view/azhydro-explorer
#    Source:    paste gee/azhydro-visualizer.js into
#               https://code.earthengine.google.com
#               (or publish via Apps > Publish for your own shareable URL).
```

### Asset paths used by the visualizer

All assets live under `projects/azhydro/assets/az-wu/`.

#### Per-category augmented raster ImageCollections (12)

Each is a 6-band stack: prediction + σ + CV + SNR + lower 95 % CI + upper 95 % CI.  Four unit conventions (`Depth_mm`, `Depth_ft`, `Volume_m3`, `Volume_AF`) live in the **same** IC and are filterable via the `unit` property.  Year filterable via the `year` property.

| Asset | Category |
|---|---|
| `Predicted_Rasters` | Total annual prediction |
| `Total_GW_Rasters` | GW component of total |
| `Total_SW_Rasters` | SW component of total |
| `Irrigation_Rasters` | Irrigation total |
| `Irrigation_GW_Rasters` | Irrigation GW |
| `Irrigation_SW_Rasters` | Irrigation SW |
| `Non_Irrigation_Rasters` | Non-irrigation total |
| `Non_Irrigation_GW_Rasters` | Non-irrigation GW |
| `Non_Irrigation_SW_Rasters` | Non-irrigation SW |
| `Irrigation_CU_Rasters` | Irrigation consumptive use total |
| `Irrigation_GW_CU_Rasters` | Irrigation CU, GW component |
| `Irrigation_SW_CU_Rasters` | Irrigation CU, SW component |

#### SW Capture rasters (3 augmented + 3 fractions)

Augmented (`SW_Capture__*_Rasters`) follow the same 6-band convention and 4 unit conventions as above.  Fractions (`SW_Capture__*_Fraction`) are single-band, dimensionless 0–1.

| Asset | Category |
|---|---|
| `SW_Capture__Total_SW_Capture_Rasters` | Total SW captured by GW pumping (volume) |
| `SW_Capture__Irrigation_SW_Capture_Rasters` | Irrigation share of SW capture |
| `SW_Capture__Non_Irrigation_SW_Capture_Rasters` | Non-irrigation share of SW capture |
| `SW_Capture__Total_SW_Capture_Fraction` | Total SW capture fraction (0–1) |
| `SW_Capture__Irrigation_SW_Capture_Fraction` | Irrigation SW capture fraction |
| `SW_Capture__Non_Irrigation_SW_Capture_Fraction` | Non-irrigation SW capture fraction |

#### Other raster assets

| Asset | Category |
|---|---|
| `OOD_Rasters` | Annual out-of-distribution probability (single band, dimensionless 0–1) |
| `CAP_Scenario__Pixel_Rasters` | 14 cumulative ΔGW images = 7 shortfall scenarios × 2 windows (2027–2060, 2027–2099); single AF band per image, `system:index` = `CAP_Scenario_Pixel_<scenario>_cum_AF_<window>` |

#### Tabular FeatureCollections

| Asset | Contents |
|---|---|
| `Well_Package__<Cat>` (15) | One per category: ~170 k well features, each with `REGISTRY_I`, `WATER_USE`, `.geo` geometry, plus year-stamped `<Cat>_AF_<year>` and `<Cat>_AF_sigma_<year>` properties (1896–2099).  Capacity-disaggregated per-well values from the published parquet. |
| `CAP_Scenario_Cumulative` | 364 features (52 basin polygons × 7 scenarios), each with `Basin`, `Scenario`, `.geo` (basin centroid Point), and year-stamped `Cum_<year>` properties for 2026–2099.  Powers the per-basin × 7-scenario time-series chart in CAP click mode. |

#### Vector overlay layers

| Asset | Contents |
|---|---|
| `Groundwater_Basin` | 52 GW-basin polygons (`BASIN_NAME` property; spanning Arizona's 51 ADWR groundwater basins).  Visualizer filters into AMA / INA / other groups by exact name match for distinct styling. |
| `ADWR_Groundwater_Subbasin` | 82 sub-basin polygons (`SUBBASIN_N` property) |
| `Well_Registry_2024` | ADWR well-registry points; powers the nearest-well lookup on click |
| `CAP` | 3 CAP-eligible county polygons (Maricopa, Pima, Pinal) with `NAME` property; rendered as dashed outlines in the visualizer |

## Citations

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Freshwater withdrawals, irrigation consumptive use, and surface water capture for Arizona, 1896–2099. _In prep. for Nature Scientific Data_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). Where Arizona's Water Goes: Declining Agricultural Dominance and Rising Urban Demand Drive a Two-Century Shift in Withdrawal Patterns (1896–2099). _In prep. for AGU Earth's Future_.

Majumdar, S., Smith, R.G., ReVelle, P., Hasan, M.F., & Wogenstahl, C. (2026). AZ-Hydro — Historical and Projected Arizona Annual Water Use: Software, Input Data, Models, Raster and Well Package Predictions, and Validation at 2 km Resolution (1896–2099). _Zenodo_. https://doi.org/10.5281/zenodo.19057935.

### Data References

Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. _International Journal of Climatology_, _33_(1), 121–131. https://doi.org/10.1002/joc.3413

Abatzoglou, J. T., & Brown, T. J. (2012). A comparison of statistical downscaling methods suited for wildfire applications. _International Journal of Climatology_, _32_(5), 772–780. https://doi.org/10.1002/joc.2312

Daly, C., Halbleib, M., Smith, J. I., Gibson, W. P., Doggett, M. K., Taylor, G. H., Curtis, J., & Pasteris, P. P. (2008). Physiographically sensitive mapping of climatological temperature and precipitation across the conterminous United States. _International Journal of Climatology_, _28_(15), 2031–2064. https://doi.org/10.1002/joc.1688

Fleckenstein, R., Wellington, D., Jin, S., Tollerud, H., Brown, J. F., Dewitz, J., Pastick, N. J., Barber, C. P., O'Brien, A., & Spanier, M. (2026). A framework for integrating spatiotemporal deep learning methods with landsat for annual land cover and impervious surface mapping. _Remote Sensing of Environment_, _338_, 115347. https://doi.org/10.1016/j.rse.2026.115347

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. _Remote Sensing of Environment_, _202_, 18–27. https://doi.org/10.1016/j.rse.2017.06.031

Majumdar, S., ReVelle, P., Pearson, C., Nozari, S., Minor, B. A., Hasan, M. F., Huntington, J. L., & Smith, R. G. (2026). pyCropWat: A Python Package for Computing Effective Precipitation Using Google Earth Engine Climate Data (v1.2.1). _Zenodo_. https://doi.org/10.5281/zenodo.18706481.

Melton, F., Huntington, J., Grimm, R., Herring, J., Hall, M., Rollison, D., Erickson, T., Allen, R., Anderson, M., Fisher, J. B., Kilic, A., Senay, G. B., Volk, J., Hain, C., Johnson, L., Ruhoff, A., Blankenau, P., Bromley, M., Carrara, W., … Anderson, R. G. (2022). OpenET: Filling a Critical Data Gap in Water Management for the Western United States. _JAWRA Journal of the American Water Resources Association_. https://doi.org/10.1111/1752-1688.12956.

Muratoglu, A., Bilgen, G. K., Angin, I., & Kodal, S. (2023). Performance analyses of effective rainfall estimation methods for accurate quantification of agricultural water footprint. _Water Research_, _238_, 120011. https://doi.org/10.1016/j.watres.2023.120011.

NASA JPL. (2020). NASADEM Merged DEM Global 1 arc second V001 [Data set]. _NASA EOSDIS Land Processes DAAC_. https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001.

Reitz, M., Sanford, W. E., & Saxe, S. (2023). Ensemble Estimation of Historical Evapotranspiration for the Conterminous U.S. _Water Resources Research_, _59_(6). https://doi.org/10.1029/2022WR034012.

Reitz, M., Sanford, W. E., & Saxe, S. (2023). Historical evapotranspiration for the conterminous U.S. _U.S. Geological Survey Data Release_. https://doi.org/10.5066/P9EZ3VAS.

Roy, S., Majumdar, S., & Swetnam, T. (2025).  samapriya/awesome-gee-community-datasets: Community Catalog (3.9.0). _Zenodo_. https://doi.org/10.5281/zenodo.17641528.

Roy, S. (2025). samapriya/geeup: geeup: Simple CLI for Earth Engine Uploads (2.0.0). _Zenodo_. https://doi.org/10.5281/zenodo.18073520.

Sohl, T. L., Reker, R., Bouchard, M., Sayler, K., Dornbierer, J., Wika, S., Quenzer, R., & Friesz, A. (2016). Modeled historical land use and land cover for the conterminous United States. _Journal of Land Use Science_, _11_(4), 476–499. https://doi.org/10.1080/1747423X.2016.1147619.

Sohl, T. L., Reker, R., Bouchard, M., Sayler, K., Dornbierer, J., Wika, S., Quenzer, R., & Friesz, A. (2018). Modeled historical land use and land cover for the conterminous United States: 1938-1992. _U.S. Geological Survey data release_. https://doi.org/10.5066/F7KK99RR.

Sohl, T. L., Sayler, K. L., Bouchard, M. A., Reker, R. R., Friesz, A. M., Bennett, S. L., Sleeter, B. M., Sleeter, R. R., Wilson, T., Soulard, C., Knuppe, M., & van Hofwegen, T. (2014). Spatially explicit modeling of 1992–2100 land cover and forest stand age for the conterminous United States. _Ecological Applications_, _24_(5), 1015–1036. https://doi.org/10.1890/13-1245.1.

Sohl, T. L., Sayler, K. L., Bouchard, M. A., Reker, R. R., Friesz, A. M., Bennett, S. L., Sleeter, B. M., Sleeter, R. R., Wilson, T., Soulard, C., Knuppe, M., & van Hofwegen, T. (2018). Conterminous United States Land Cover Projections - 1992 to 2100. _U.S. Geological Survey data release_. https://doi.org/10.5066/P95AK9HP.

Soil Survey Staff, Natural Resources Conservation Service, United States Department of Agriculture. _Web Soil Survey_. Available online at https://websoilsurvey.nrcs.usda.gov/. 

USDA SCS. (1993). Chapter 2 Irrigation Water Requirements. In Part 623 National Engineering Handbook. _USDA Soil Conservation Service_. https://www.wcc.nrcs.usda.gov/ftpref/wntsc/waterMgt/irrigation/NEH15/ch2.pdf

USGS. (2024). Annual NLCD Collection 1 Science Products. _U.S. Geological Survey data release_. https://doi.org/10.5066/P94UXNTS.

Volk, J. M., Huntington, J. L., Melton, F. S., Allen, R., Anderson, M., Fisher, J. B., Kilic, A., Ruhoff, A., Senay, G. B., Minor, B., Morton, C., Ott, T., Johnson, L., de Andrade, B., Carrara, W., Doherty, C. T., Dunkerly, C., Friedrichs, M., Guzman, A., … Yang, Y. (2024). Assessing the accuracy of OpenET satellite-based evapotranspiration data to support water resource and land management applications. _Nature Water_, _2_(2), 193–205. https://doi.org/10.1038/s44221-023-00181-7.

Volk, J., Dunkerly, C., Majumdar, S., Huntington, J., Minor, B., Kim, Y., Morton, C., ReVelle, P., Kilic, A., Melton, F., Allen, R., Pearson, C., Purdy, A., & Caldwell, T. (2026). CONUS Gridded Reference Evapotranspiration Bias Correction: Inputs, Station Validation, and Outputs (gridMET/OpenET) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18673484.
