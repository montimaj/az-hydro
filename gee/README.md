# GEE Asset Export Pipeline

Pre-exports computationally expensive Google Earth Engine (GEE) collections as asset ImageCollections under `projects/azhydro/assets/`. These assets are consumed by `dataops.py` at tile-download time via simple `ee.ImageCollection(...)` loads and `filterDate()` calls, eliminating repeated on-the-fly computation.

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
| `usgs_adjusted_et` | 1,248 | 1896–1999 | `actual_et` | 800 | `export_usgs_adjusted_et.py` |
| `maca_monthly_eto` | 888 | 2026–2099 | `eto` | 4638.3 | `export_maca_monthly_eto.py` |
| `maca_monthly_et` | 888 | 2026–2099 | `actual_et` | 4638.3 | `export_maca_monthly_et.py` |
| `lulc_projection_ensemble` | 74 | 2026–2099 | `landcover` | 250 | `export_lulc_ensemble.py` |
| `monthly_peff` | 2,448 | 1896–2099 | `peff` | 4638.3 | `export_monthly_peff.py` |

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
  └── export_maca_monthly_eto.py        ← no custom dep (uses gridMET ratios)

Level 3 (depends on Level 2):
  ├── export_maca_monthly_et.py         ← needs monthly_etof + maca_monthly_eto
  └── export_monthly_peff.py            ← needs prism_hargreaves_eto + maca_monthly_eto
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

- **1896–1999 (Reitz → OpenET scale):** USGS Reitz Ensemble ET is available as mm/day. It is converted to mm/month, then multiplied by a monthly OpenET/Reitz ratio derived from the 2000–2018 overlap period (228 paired monthly images). The ratio for each calendar month is averaged across all overlapping years to produce 12 climatological ratio grids, ensuring the Reitz ET is scaled to match the magnitude and spatial patterns of OpenET.

- **2000–2025 (OpenET, native):** OpenET Ensemble v2.0/v2.1 (`et_ensemble_mad`) is used directly. Monthly ET is computed by summing all OpenET images within each calendar month.

- **2026–2099 (MACA → OpenET scale):** No observational ET exists for the future. Instead, monthly EToF (crop coefficient = ET/ETo) grids are derived from the 2000–2025 OpenET/gridMET overlap. Future monthly ET is then: $\text{ET}_{\text{future}} = \text{EToF} \times \text{MACA ETo}$. This preserves the observed spatial crop-water-demand patterns while allowing the climate signal (ETo) to evolve under future scenarios.

### ETo Harmonization (3 eras)

```
      PRISM Hargreaves ETo      gridMET ETo           MACA ETo
    ┌─────────────────────┐   ┌───────────────┐   ┌─────────────────┐
    │    1896 – 1978      │   │  1979 – 2025  │   │   2026 – 2099   │
    │  (monthly tmax/tmin │   │  (native      │   │ (daily ensemble │
    │   → Hargreaves PET) │   │   monthly)    │   │  → RefET → sum) │
    └──────────┬──────────┘   └───────┬───────┘   └────────┬────────┘
               │                      │                    │
               │ Hargreaves/gridMET   │   gridMET bias     │
               │ ratio (1979-2025     │   correction       │
               │ overlap)             │   ratios           │
               ▼                      ▼                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Harmonized monthly ETo (band: eto)             │
    │                         1896 – 2099                         │
    └─────────────────────────────────────────────────────────────┘
```

- **1896–1978 (PRISM Hargreaves → gridMET scale):** Daily climate data is unavailable before 1979, so monthly ETo is estimated from PRISM monthly tmax/tmin via the Hargreaves method (`openet.refetgee.Daily` with mid-month DOY). This systematically differs from Penman-Monteith-based gridMET ETo, so a per-pixel, per-month correction ratio is derived from the 1979–2025 overlap (564 paired monthly images) where both Hargreaves and gridMET are available. The corrected ETo is: $\text{ETo}_{\text{corrected}} = \text{Hargreaves PET} \times \overline{R}$, where $\overline{R} = \text{mean}\left(\frac{\text{gridMET ETo}}{\text{Hargreaves PET}}\right)$ over 1979–2025.

- **1979–2025 (gridMET, native):** gridMET monthly ETo (`projects/openet/assets/reference_et/conus/gridmet/monthly/v1`) is used directly. This is the reference standard to which all other eras are harmonized.

- **2026–2099 (MACA → gridMET scale):** A daily MACA ensemble (mean of 20 GCMs × 2 RCPs = 40 members) is constructed for each year. Daily ETo is computed via `openet.refetgee.Daily.maca()` (ASCE Penman-Monteith) and summed to monthly. Pre-existing gridMET bias-correction ratio grids (`projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/`) are then applied: $\text{ETo}_{\text{corrected}} = \text{raw MACA ETo} \times \text{gridMET ratio}$.

### LULC Harmonization (2 eras)

```
     USGS Historical LULC        IrrMapper / NLCD      USGS Projected LULC
    ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
    │   1938 – 1984    │   │    1985 – 2025      │   │   2026 – 2099    │
    │  (Historical     │   │ (observational       │   │ (mode of B1, B2, │
    │   scenario)      │   │  satellite-based)    │   │  A1B, A2)        │
    └──────────────────┘   └─────────────────────┘   └──────────────────┘
```

- **1938–1984:** USGS Historical LULC (`projects/nwi-usgs/assets/USGS-LULC-CONUS`, scenario = "Historical") is used directly. Years before 1938 use the 1938 classification.
- **1985–2025:** IrrMapper for irrigation classification and NLCD for urban/developed areas provide observational LULC.
- **2026–2099:** Four USGS LULC projection scenarios (B1, B2, A1B, A2) are combined via pixel-wise mode to create a single consensus ensemble. Exported as integer (categorical) data.

### Join Strategy

All ratio computations use `ee.Join.inner()` with paired filters on **both** `month` **and** `year` properties. This prevents many-to-many cross-joins that would occur if matching on `month` alone (e.g., January 2005 gridMET would incorrectly pair with January 2010 PRISM). The final climatological ratios are then produced by averaging across years within each calendar month.

### Scale Consistency

Each asset is exported at its native source resolution to avoid resampling artifacts during the ratio computation:

| Source | Native Scale |
|---|---|
| PRISM / gridMET / MACA | 4,638.3 m (~4 km) |
| Reitz Ensemble ET | 800 m |
| OpenET Ensemble | 30 m |
| USGS LULC | 250 m |

GEE handles on-the-fly reprojection when these assets are combined at tile-download time in `dataops.py`.

## Detailed Export Logic

### 1. gridMET/Hargreaves ETo Ratio (`gridmet_hargreaves_eto_ratio`)

**Purpose:** Bias-correct PRISM Hargreaves ETo (1896–1978) to be consistent with gridMET ETo (1979–2025).

**Method:**
1. Load gridMET monthly ETo for 1979–2025 (`projects/openet/assets/reference_et/conus/gridmet/monthly/v1`).
2. Load PRISM monthly data for 1979–2025 (`OREGONSTATE/PRISM/ANm`) and compute Hargreaves PET using `openet.refetgee.Daily` with tmax/tmin and day-of-year at mid-month.
3. Inner join on both `month` AND `year` to pair same-timestep images.
4. Compute per-pixel ratio: $\text{ratio} = \frac{\text{gridMET ETo}}{\text{Hargreaves ETo}}$.
5. Average ratios across all years for each month → 12 monthly ratio grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `ratio`.

---

### 2. OpenET/Reitz ET Ratio (`openet_reitz_et_ratio`)

**Purpose:** Bias-correct USGS Reitz Ensemble ET (1896–1999) to be consistent with OpenET (2000–2025).

**Method:**
1. Build monthly OpenET ET for 2000–2018 by summing daily OpenET v2.0 images per month (band `et_ensemble_mad`).
2. Load USGS Reitz Ensemble ET (`projects/nwi-usgs/assets/USGS-Reitz-Ensemble-ET`) for 2000–2018. Convert from mm/day to mm/month by multiplying by the number of days in each month.
3. Inner join on both `month` AND `year`.
4. Compute per-pixel ratio: $\text{ratio} = \frac{\text{OpenET ET}}{\text{Reitz ET}}$.
5. Average ratios across all years for each month → 12 monthly ratio grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `ratio`.

---

### 3. Monthly EToF (`monthly_etof`)

**Purpose:** Derive crop coefficient (EToF = ET/ETo) grids for applying to future MACA ETo to estimate future ET.

**Method:**
1. Build monthly OpenET ET for 2000–2025 (same as above).
2. Load gridMET monthly ETo for 2000–2025.
3. Inner join on both `month` AND `year`.
4. Compute per-pixel EToF: $\text{EToF} = \frac{\text{OpenET ET}}{\text{gridMET ETo}}$.
5. Average across all years for each month → 12 monthly EToF grids.

**Output:** 12 images (`month_01` through `month_12`), each with band `etof`.

---

### 4. PRISM Hargreaves ETo (`prism_hargreaves_eto`)

**Dependency:** `gridmet_hargreaves_eto_ratio`

**Purpose:** Monthly ETo for the historical period 1896–1978, bias-corrected to match gridMET.

**Method:** For each year in 1896–1978:
1. Load PRISM monthly data and compute Hargreaves PET (same Hargreaves method as the ratio script).
2. Inner join with the 12 pre-exported ratio grids on `month`.
3. Multiply: $\text{ETo} = \text{Hargreaves PET} \times \text{ratio}$.
4. Export each of the 12 monthly images.

**Output:** 996 images (`{year}_{month:02d}`), each with band `eto` in mm/month.

---

### 5. USGS Adjusted ET (`usgs_adjusted_et`)

**Dependency:** `openet_reitz_et_ratio`

**Purpose:** Monthly ET for 1896–1999, bias-corrected to match OpenET.

**Method:** For each year in 1896–1999:
1. Load Reitz ET for that year, convert mm/day → mm/month.
2. Inner join with the 12 pre-exported OpenET/Reitz ratio grids on `month`.
3. Multiply: $\text{ET}_{\text{adj}} = \text{Reitz ET (mm/month)} \times \text{ratio}$ → band `actual_et`.
4. Export each of the 12 monthly images.

**Output:** 1,248 images (`{year}_{month:02d}`), each with band `actual_et` in mm/month.

---

### 6. MACA Monthly ETo (`maca_monthly_eto`)

**Purpose:** Monthly ETo for future scenarios 2026–2099, bias-corrected with gridMET ratios.

**Method:** For each year in 2026–2099:
1. Build daily MACA ensemble by averaging all 20 GCMs × 2 scenarios (RCP4.5, RCP8.5) per day. Bands used: `tasmax`, `tasmin`, `pr`, `rsds`, `uas`, `vas`, `huss`.
2. Compute daily ETo using `openet.refetgee.Daily.maca()` with NASADEM elevation and pixel latitude.
3. Sum daily ETo to monthly (12 images per year).
4. Apply gridMET bias-correction ratios (`projects/openet/assets/reference_et/conus/gridmet/ratios/v1/monthly/eto/{MonthName}`), joined on `month`.
5. Multiply: $\text{ETo} = \text{raw MACA ETo} \times \text{gridMET ratio}$.
6. Export each month.

**Output:** 888 images (`{year}_{month:02d}`), each with band `eto` in mm/month.

---

### 7. MACA Monthly ET (`maca_monthly_et`)

**Dependencies:** `monthly_etof` + `maca_monthly_eto`

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
1. Load USGS LULC projections (`projects/nwi-usgs/assets/USGS-LULC-CONUS`) for all four scenarios: B1, B2, A1B, A2.
2. Compute pixel-wise mode across the 4 scenarios.
3. Export as integer (categorical data).

**Output:** 74 images (`{year}`), each with band `landcover` (integer class values).

---

### 9. Monthly USDA SCS Effective Precipitation (`monthly_peff`)

**Dependencies:** `prism_hargreaves_eto` (for 1896–1978) + `maca_monthly_eto` (for 2026–2099)

**Purpose:** Monthly effective precipitation for 1896–2099 using the USDA SCS method.

**Parameters:** `mad_factor = 1`, `rz_depth_m = 2 m` for all months (consistent with UCRB comparisons).

**Method:** For each year in 1896–2099:
1. **Get monthly ETo** from the appropriate source:
   - 1896–1978: Pre-exported `prism_hargreaves_eto` asset
   - 1979–2025: gridMET native monthly ETo
   - 2026–2099: Pre-exported `maca_monthly_eto` asset
2. **Get monthly precipitation:**
   - 1896–2025: PRISM (`OREGONSTATE/PRISM/ANm`, band `ppt`)
   - 2026–2099: MACA daily ensemble precipitation aggregated to monthly
3. **Inner join** ETo and precipitation on `month`.
4. **Compute USDA SCS effective precipitation** (equations 2-84 and 2-85):
   - Convert ETo and precip from mm to inches.
   - Compute soil storage factor: $d = \text{MAD} \times \text{AWC} \times \text{RZ}_{\text{inches}}$
   - $sf = 0.531747 + 0.295164 \cdot d - 0.057697 \cdot d^2 + 0.003804 \cdot d^3$
   - $ep = sf \times (0.70917 \cdot P^{0.82416} - 0.11556) \times 10^{0.02426 \cdot \text{ETo}}$
   - Clamp: $ep \leq P$, $ep \leq \text{ETo}$, $ep \geq 0$
   - Convert back to mm.
5. **Export** each of the 12 monthly images.

**Soil data:** AWC from `projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite` (inches).

**Output:** 2,448 images (`{year}_{month:02d}`), each with band `peff` in mm/month.

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
```

### Resumability

All scripts check for existing assets before exporting. If a run is interrupted, simply re-run the same script — it will skip already-exported images and only submit remaining ones.

### Queue Throttling

GEE enforces a 3,000-task queue limit. `export_image()` automatically calls `_wait_for_queue_capacity()` before each submission. If there are already 2,900+ pending/running tasks, it polls every 60 seconds until the queue has room. This means large batch exports (e.g., 2,448 peff images) can be launched without manually splitting runs.

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
| `build_daily_maca_ensemble(year)` | Daily MACA ensemble for one year |

## How `dataops.py` Consumes These Assets

After export, `dataops.py` loads the assets once before the tile-download loop:

```python
_ASSET_PREFIX = 'projects/azhydro/assets'
prism_hargreaves_eto_ic = ee.ImageCollection(f'{_ASSET_PREFIX}/prism_hargreaves_eto')
usgs_adjusted_et_ic     = ee.ImageCollection(f'{_ASSET_PREFIX}/usgs_adjusted_et')
maca_monthly_eto_ic     = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_eto')
maca_monthly_et_ic      = ee.ImageCollection(f'{_ASSET_PREFIX}/maca_monthly_et')
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
